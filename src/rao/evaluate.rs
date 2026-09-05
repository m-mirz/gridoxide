//! Evaluating a CRAC against a network: which CNECs are violated, in which
//! state, and by how much.
//!
//! This is the half of a remedial action optimization that does not optimize.
//! It answers "where does it hurt", and it is a deliverable on its own —
//! gridoxide could run a contingency analysis before this existed, but it could
//! not say whether the result was *acceptable*, because nothing told it what
//! the limits were or which elements anyone cared about.
//!
//! # Two steps, deliberately separated
//!
//! **Resolution** ([`Resolution`]) maps the CRAC's network-element ids onto
//! gridoxide's own branch indices. It needs a network, it is fallible in an
//! interesting way, and it gets its own report. An element that fails to
//! resolve is not a small problem: a CNEC that silently disappears is a
//! constraint the optimizer will never see, and a contingency that fails to
//! resolve is an outage it will never simulate. Either produces a confident
//! answer to the wrong question.
//!
//! **Evaluation** ([`evaluate`]) then computes flows and margins. It is
//! deliberately ignorant of where the indices came from, so it works the same
//! whether the network arrived as UCTE, IIDM or CGMES.
//!
//! # Why DC
//!
//! DC is exact for the linear model, and the whole post-contingency sweep comes
//! out of one factorization: [`DcSensitivity::multi_outage_flows`] answers an
//! N-k outage with a Woodbury update rather than a re-solve. On
//! `case9241pegase` that is 0.40 ms against 13.4 ms. A search tree evaluates
//! thousands of candidates, so the difference is the difference between a tool
//! that runs in a control room and one that does not.

use std::collections::HashMap;

use crate::linear::btheta::{dc_branches, dc_power_flow};
use crate::linear::DcOptions;
use crate::linear::sensitivity::DcSensitivity;
use crate::ratings::current_to_power_pu;
use crate::batch::{BatchSolver, Scenario};
use crate::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use crate::network::{build_ybus_with_outages, stamp_shunts};
use crate::outerloop::SlackDistribution;
use crate::solver::{
    IslandStatus, JacobianBackend,
};
use crate::types::{Bus, Line, Transformer};
use crate::network::ShuntAdm;

use super::crac::{Crac, FlowCnec, Side, State, Threshold, Unit};

/// The CRAC's element ids, resolved against a network.
#[derive(Clone, Debug, Default)]
pub struct Resolution {
    /// Element id → flat branch index (lines then transformers).
    pub branch_of: HashMap<String, usize>,
    /// Element id → bus index.
    ///
    /// A CNEC or a contingency names a *branch*; a redispatch names the
    /// generators and loads it shifts power between, which are buses. Those are
    /// different id spaces and resolving one against the other silently yields
    /// nothing — which is how an injection range action comes to be dropped
    /// from the optimization while every margin still looks right.
    pub bus_of: HashMap<String, usize>,
    /// Element ids the network does not contain. **Not** an error on its own —
    /// a CRAC written for a merged model legitimately names elements outside
    /// any single file — but every consumer needs to know, because what it
    /// costs is a constraint or an outage silently going missing.
    pub unresolved: Vec<String>,
}

impl Resolution {
    /// Resolve every element the CRAC names against a list of branch ids.
    ///
    /// Ids are matched exactly, then with surrounding whitespace trimmed on
    /// both sides. The second pass is not sloppiness: UCTE element ids are
    /// fixed-width and carry padding (`"BBE1AA1  BBE2AA1  1"`), and a CRAC
    /// written against the same network may or may not preserve it.
    pub fn new(crac: &Crac, branch_ids: &[String]) -> Self {
        Self::with_buses(crac, branch_ids, &[])
    }

    /// Resolve against branches *and* buses.
    ///
    /// Bus ids are matched exactly, then trimmed, then with a trailing
    /// `_generator` or `_load` removed: powsybl names a UCTE node's generator
    /// `<node>_generator`, and a CRAC written against that network uses the
    /// name powsybl gave it rather than the node code the file contains.
    pub fn with_buses(crac: &Crac, branch_ids: &[String], bus_ids: &[String]) -> Self {
        let mut exact: HashMap<&str, usize> = HashMap::with_capacity(branch_ids.len());
        let mut trimmed: HashMap<&str, usize> = HashMap::with_capacity(branch_ids.len());
        for (i, id) in branch_ids.iter().enumerate() {
            exact.entry(id.as_str()).or_insert(i);
            trimmed.entry(id.trim()).or_insert(i);
        }

        let mut bus_exact: HashMap<&str, usize> = HashMap::with_capacity(bus_ids.len());
        let mut bus_trimmed: HashMap<&str, usize> = HashMap::with_capacity(bus_ids.len());
        for (i, id) in bus_ids.iter().enumerate() {
            bus_exact.entry(id.as_str()).or_insert(i);
            bus_trimmed.entry(id.trim()).or_insert(i);
        }
        let find_bus = |element: &str| -> Option<usize> {
            if let Some(&i) = bus_exact.get(element).or_else(|| bus_trimmed.get(element.trim())) {
                return Some(i);
            }
            for suffix in ["_generator", "_load"] {
                if let Some(stem) = element.strip_suffix(suffix) {
                    if let Some(&i) =
                        bus_exact.get(stem).or_else(|| bus_trimmed.get(stem.trim()))
                    {
                        return Some(i);
                    }
                }
            }
            None
        };

        let mut branch_of = HashMap::new();
        let mut bus_of = HashMap::new();
        let mut unresolved = Vec::new();
        for element in crac.network_elements() {
            if let Some(&i) = exact.get(element).or_else(|| trimmed.get(element.trim())) {
                branch_of.insert(element.to_string(), i);
                continue;
            }
            if let Some(i) = find_bus(element) {
                bus_of.insert(element.to_string(), i);
                continue;
            }
            unresolved.push(element.to_string());
        }
        Self { branch_of, bus_of, unresolved }
    }

    pub fn branch(&self, element: &str) -> Option<usize> {
        self.branch_of.get(element).copied()
    }

    /// The bus an injection element sits on.
    pub fn bus(&self, element: &str) -> Option<usize> {
        self.bus_of.get(element).copied()
    }

    /// True when every element the CRAC names was found.
    pub fn is_complete(&self) -> bool {
        self.unresolved.is_empty()
    }
}

/// Which power-flow model the margins are measured with.
///
/// The optimizer is guided by DC sensitivities either way — they are cheap,
/// exact for the linear model, and the outer loop re-measures the truth after
/// every move, so an approximate gradient steers the search without deciding
/// the answer. What this chooses is what "the truth" means.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlowModel {
    /// Linear, lossless, and the whole post-contingency sweep from one
    /// factorization — 0.40 ms per outage against 13.4 ms for a re-solve on
    /// `case9241pegase`. A search tree evaluates thousands of candidates, so
    /// this is what makes one finish.
    #[default]
    Dc,
    /// A full Newton-Raphson per contingency, through
    /// [`BatchSolver::solve_contingencies`](crate::batch::BatchSolver), which
    /// holds its symbolic factorization across the sweep.
    ///
    /// Slower by orders of magnitude and the only model that sees reactive
    /// power, losses, and voltage. A plan chosen on DC margins can be rejected
    /// here, which is the entire point of re-checking it.
    Ac,
}

/// How one CNEC fared.
#[derive(Clone, Debug, PartialEq)]
pub struct CnecResult {
    /// Index into [`Crac::flow_cnecs`].
    pub cnec: usize,
    /// Flat branch index.
    pub branch: usize,
    /// Active power at the monitored terminal, MW. Positive means power
    /// entering the branch at its `from` end.
    pub flow_mw: f64,
    /// Distance to the nearest binding threshold, MW. **Negative means
    /// violated**, and its magnitude is the overload.
    pub margin_mw: f64,
    /// Upper bound on the **signed** flow, MW; `INFINITY` when the CNEC has
    /// none. Already tightened by the reliability margin.
    pub upper_mw: f64,
    /// Lower bound on the signed flow, MW; `NEG_INFINITY` when the CNEC has
    /// none.
    ///
    /// Kept separate from [`upper_mw`](Self::upper_mw) because a CRAC's
    /// thresholds are frequently **not** symmetric, and a margin computed from
    /// `|flow|` against one magnitude is then a constraint nobody wrote.
    pub lower_mw: f64,
    /// The tightest bound's magnitude, MW — `min(|upper|, |lower|)`.
    ///
    /// For a symmetric CNEC this is the limit in the ordinary sense and
    /// `margin_mw == limit_mw - |flow_mw|`. For a one-sided one that identity
    /// does **not** hold, and the margin is the quantity to trust; this is kept
    /// for display, where naming a number beats naming an infinity.
    pub limit_mw: f64,
    /// Distance to the nearest binding threshold in **amperes**, the unit most
    /// of the reference's own expectations are written in.
    ///
    /// This is not a unit conversion of [`margin_mw`](Self::margin_mw) at some
    /// convenient voltage — it is the same margin expressed in the units the
    /// binding threshold was written in, converted at the voltage *that*
    /// threshold names. When a CNEC's thresholds disagree about voltage or
    /// side, the one that actually bound is the one that governs here.
    pub margin_a: f64,
    /// The voltage the binding threshold was converted at, volts.
    ///
    /// Exposed so a caller can express this CNEC's margin in amperes without
    /// re-deriving which threshold bound — the optimizer needs exactly that
    /// factor to maximize an ampere objective.
    pub conversion_v: f64,
    /// Current at the monitored terminal, amperes.
    ///
    /// Under [`FlowModel::Ac`] this is the real thing, `|S| / (√3·U)`. Under
    /// DC there is no reactive power and no voltage deviation, so it is
    /// `|P| / (√3·U_nominal)` — which is what a DC study means by a current and
    /// what the reference computes in its own DC mode.
    pub current_a: f64,
    /// Active power and current at the branch's **other** terminal — side two
    /// where [`flow_mw`](Self::flow_mw) is side one — as `(MW, A)`.
    ///
    /// Measured in the **same direction along the branch** as
    /// [`flow_mw`](Self::flow_mw): power entering at side one, power *leaving*
    /// at side two. So the two differ only by the branch's own losses, which is
    /// the comparison anyone asking for both sides wants, and their currents
    /// differ again by the two buses' voltages. Reporting side two as the power
    /// *entering* there would negate it, and the two figures would then differ
    /// by roughly twice the flow rather than by the losses.
    ///
    /// The AC path computes both terminals already, to measure a threshold on
    /// the side it names; this stops the second one being discarded. Under DC
    /// there are no losses, so side two's active power is exactly `flow_mw` and
    /// only its current differs, at the far bus's nominal voltage. That is not
    /// a shortcut — it is what a DC study means by the far end.
    pub side_two: (f64, f64),
}

impl CnecResult {
    /// Amperes per MW for this CNEC, at the voltage its binding threshold
    /// names. `0.0` when no voltage is known, which leaves an ampere objective
    /// indifferent to it rather than dividing by zero.
    pub fn amperes_per_mw(&self) -> f64 {
        if self.conversion_v > 0.0 { 1e6 / (3f64.sqrt() * self.conversion_v) } else { 0.0 }
    }

    pub fn is_violated(&self) -> bool {
        self.margin_mw < 0.0
    }
}

/// How one perimeter — one state — fared.
#[derive(Clone, Debug, PartialEq)]
pub struct PerimeterResult {
    pub state: State,
    pub cnecs: Vec<CnecResult>,
    /// The flows below are not a clean screening result.
    ///
    /// The two flow models mean slightly different things by it, and both are
    /// "treat these numbers with suspicion" rather than "these numbers are
    /// wrong". Under [`FlowModel::Dc`] the contingency defeated the Woodbury
    /// update — typically by severing the network — so the flows come from a
    /// full re-solve, which is slower but still correct. Under
    /// [`FlowModel::Ac`] at least one island failed to converge, and there the
    /// flows genuinely are not to be trusted.
    pub severed: bool,
}

impl PerimeterResult {
    /// The worst margin over every CNEC — the quantity a max-min-margin
    /// objective maximizes, negated.
    pub fn min_margin(&self) -> Option<f64> {
        self.cnecs.iter().map(|c| c.margin_mw).fold(None, |acc, m| {
            Some(match acc {
                Some(a) => f64::min(a, m),
                None => m,
            })
        })
    }

    /// The worst margin over the CNECs a max-min-margin objective actually
    /// optimizes.
    ///
    /// Different from [`min_margin`](Self::min_margin), and the difference is
    /// the point: a monitored CNEC is held to a soft floor, not maximized, so
    /// letting one set the minimum reports as "the margin the RAO achieved" a
    /// number no remedial action was ever asked to move.
    pub fn min_optimized_margin(&self, crac: &Crac) -> Option<f64> {
        self.cnecs
            .iter()
            .filter(|c| crac.flow_cnecs[c.cnec].optimized)
            .map(|c| c.margin_mw)
            .fold(None, |acc, m| {
                Some(match acc {
                    Some(a) => f64::min(a, m),
                    None => m,
                })
            })
    }

    pub fn violations(&self) -> impl Iterator<Item = &CnecResult> {
        self.cnecs.iter().filter(|c| c.is_violated())
    }
}

/// A whole security assessment: every perimeter the CRAC defines.
#[derive(Clone, Debug, PartialEq)]
pub struct SecurityResult {
    pub perimeters: Vec<PerimeterResult>,
    /// CNECs skipped because their network element did not resolve.
    pub skipped: Vec<usize>,
}

impl SecurityResult {
    /// The worst margin anywhere. `None` when nothing was evaluated.
    pub fn min_margin(&self) -> Option<f64> {
        self.perimeters.iter().filter_map(PerimeterResult::min_margin).fold(None, |acc, m| {
            Some(match acc {
                Some(a) => f64::min(a, m),
                None => m,
            })
        })
    }

    pub fn is_secure(&self) -> bool {
        self.min_margin().is_none_or(|m| m >= 0.0)
    }

    pub fn violations(&self) -> impl Iterator<Item = (&PerimeterResult, &CnecResult)> {
        self.perimeters.iter().flat_map(|p| p.violations().map(move |c| (p, c)))
    }
}

/// Everything [`evaluate`] needs about the network.
pub struct Network<'a> {
    pub buses: &'a [Bus],
    pub lines: &'a [Line],
    pub transformers: &'a [Transformer],
    /// Flat branch index → the element id the CRAC would use.
    pub branch_ids: &'a [String],
    /// Bus index → the id the CRAC would use, for resolving injections.
    /// May be empty, in which case no injection resolves.
    pub bus_ids: &'a [String],
    /// Branches the network file itself says are out of service.
    ///
    /// Every evaluation starts from this set, and a remedial action that
    /// *closes* a circuit works by removing from it. Without it a standby
    /// circuit is silently in service, which both understates the flows and
    /// makes "close this line" a no-op — an automaton that fires and changes
    /// nothing.
    pub initially_open: &'a [usize],
    /// ISO country code per bus, parallel to `buses`. Empty when the importer
    /// does not supply them, which disables the one thing that reads it —
    /// `search`'s "skip actions far from the most limiting element" filter.
    ///
    /// Only `ucte` fills this in today (from the file's `##Z<cc>` sub-headers).
    /// IIDM states countries on substations, which `iidm.rs` skips.
    pub bus_countries: &'a [Option<String>],
    /// Shunt admittances. Read only by the AC flow model; DC ignores them.
    pub shunts: &'a [ShuntAdm],
    /// Per-bus active generation, per-unit and positive — the participation
    /// weights a distributed slack shares an imbalance out by.
    ///
    /// Travels with the network rather than with the options for the same
    /// reason [`shunts`](Self::shunts) does: it is a property of the machines,
    /// and every layer that builds an [`AcOptions`] from a `Network` needs it.
    /// Empty is allowed and means "weight by net injection instead", which is
    /// what an importer that never retained generation can offer.
    pub generation: &'a [f64],
    /// Tap changers, parallel to `transformers`.
    ///
    /// A CRAC's PST range action *may* carry its own tap-to-angle table and
    /// often does not — the table is a property of the transformer, and a CRAC
    /// written against a network that already describes it has no reason to
    /// repeat it. Without this, such an action has no positions to choose
    /// between and is silently skipped.
    pub tap_changers: &'a [Option<crate::types::TapChanger>],
    pub base_mva: f64,
}

impl Network<'_> {
    fn n_branches(&self) -> usize {
        self.lines.len() + self.transformers.len()
    }

    /// Line-to-line base voltage at a branch's `from` bus, in volts. Needed to
    /// turn an ampere threshold into MW.
    fn branch_voltage(&self, branch: usize, dc: &[crate::linear::btheta::DcBranch]) -> Option<f64> {
        let b = dc.iter().find(|b| b.index == branch)?;
        Some(self.buses.get(b.from)?.u_rated)
    }

    /// The same at the branch's `to` bus — side two. Equal to
    /// [`branch_voltage`](Self::branch_voltage) on a line and different across
    /// a transformer, which is the only place a DC side-two current differs
    /// from a side-one one.
    fn branch_far_voltage(
        &self,
        branch: usize,
        dc: &[crate::linear::btheta::DcBranch],
    ) -> Option<f64> {
        let b = dc.iter().find(|b| b.index == branch)?;
        Some(self.buses.get(b.to)?.u_rated)
    }
}

/// The tightest bounds a CNEC's thresholds impose, accumulated one at a time.
///
/// Each side is tracked with the voltage its own binding threshold named, so an
/// ampere margin can be converted back at the right voltage even when a CNEC's
/// thresholds disagree about which that is.
struct Bounds {
    upper: f64,
    lower: f64,
    u_upper: f64,
    u_lower: f64,
    /// The same two bounds expressed in **amperes**, carried alongside rather
    /// than derived from the megawatt pair.
    ///
    /// They cannot be derived, and that is the whole point. A megawatt margin
    /// is `limit − |P|`; the ampere margin the reference reports is
    /// `limit_in_amperes − I`, and `I` carries reactive power and the bus's
    /// actual voltage while `|P|` carries neither. The two differ by whatever
    /// those contribute, which on the reference's own `epic5` fixture is 150 A
    /// on a 2000 MW threshold — enough to change which action the search takes.
    upper_a: f64,
    lower_a: f64,
}

impl Bounds {
    fn unbounded(u_rated: f64) -> Self {
        Self {
            upper: f64::INFINITY,
            lower: f64::NEG_INFINITY,
            u_upper: u_rated,
            u_lower: u_rated,
            upper_a: f64::INFINITY,
            lower_a: f64::NEG_INFINITY,
        }
    }

    /// Fold in one threshold's bounds, each tightened by the reliability margin
    /// and by whatever headroom reactive flow has already consumed.
    /// `expressed` is the voltage at which this threshold's bound is stated in
    /// amperes, which is **not** always the voltage it was converted from. An
    /// ampere threshold converts to megawatts at the voltage the CRAC says it
    /// was written against, and expressing it back in amperes undoes exactly
    /// that. A megawatt threshold converts from nothing — it is already the
    /// unit the limit is written in — so the voltage that turns it into amperes
    /// is the network's own, which is what the reference's CNEC carries.
    ///
    /// The charge is deliberately absent from the ampere pair. It exists to
    /// make a megawatt margin behave like an ampere one by taking reactive flow
    /// and the voltage deviation off the limit; in the ampere domain the
    /// current already carries both, and charging twice would take them off
    /// again.
    fn tighten(
        &mut self,
        lower: f64,
        upper: f64,
        voltage: f64,
        expressed: f64,
        frm: f64,
        charge: f64,
    ) {
        let per_amp = 3f64.sqrt() * expressed / 1e6;
        let (upper_a, lower_a) = if per_amp > 0.0 {
            ((upper - frm) / per_amp, (lower + frm) / per_amp)
        } else {
            (f64::INFINITY, f64::NEG_INFINITY)
        };
        let upper = upper - frm - charge;
        let lower = lower + frm + charge;
        if upper < self.upper {
            self.upper = upper;
            self.u_upper = voltage;
            self.upper_a = upper_a;
        }
        if lower > self.lower {
            self.lower = lower;
            self.u_lower = voltage;
            self.lower_a = lower_a;
        }
    }

    /// Whether anything was ever folded in. A CNEC whose thresholds are all
    /// inexpressible constrains nothing, which is better than constraining it
    /// with a number nobody wrote.
    fn is_constraining(&self) -> bool {
        self.upper.is_finite() || self.lower.is_finite()
    }

    /// The margin against the signed flow, and the voltage of the bound that
    /// produced it.
    fn margin(&self, flow_mw: f64) -> (f64, f64) {
        let from_upper = self.upper - flow_mw;
        let from_lower = flow_mw - self.lower;
        if from_upper <= from_lower { (from_upper, self.u_upper) } else { (from_lower, self.u_lower) }
    }

    /// The expression voltage of whichever bound the megawatt margin found
    /// binding — recovered from the pair, so the two can never drift apart.
    fn expressed(&self, flow_mw: f64) -> f64 {
        let volts = |bound_mw: f64, bound_a: f64| {
            if bound_a.is_finite() && bound_a.abs() > 0.0 {
                bound_mw.abs() / bound_a.abs() * 1e6 / 3f64.sqrt()
            } else {
                0.0
            }
        };
        if self.upper - flow_mw <= flow_mw - self.lower {
            volts(self.upper, self.upper_a)
        } else {
            volts(self.lower, self.lower_a)
        }
    }

    /// The margin in **amperes** against a signed current, measured against
    /// whichever bound the megawatt margin found binding.
    ///
    /// The binding side is decided once, in megawatts, rather than a second
    /// time here: a CNEC whose two units disagreed about which of its own
    /// bounds is closer would report a margin and a limit belonging to
    /// different constraints.
    fn margin_amperes(&self, flow_mw: f64, flow_a: f64) -> f64 {
        if self.upper - flow_mw <= flow_mw - self.lower {
            self.upper_a - flow_a
        } else {
            flow_a - self.lower_a
        }
    }

    /// The tighter bound's magnitude, for display.
    fn magnitude(&self) -> f64 {
        f64::min(self.upper.abs(), self.lower.abs())
    }
}

/// The signed MW bounds one threshold puts on a flow, plus the voltage it was
/// converted at.
///
/// Returns `(lower, upper, voltage)`, with `NEG_INFINITY`/`INFINITY` for a
/// bound the threshold does not state. **The two are not assumed symmetric.**
/// A CRAC routinely writes a one-sided threshold — `min: -1500, max: null`
/// means "no more than 1500 A in the reverse direction, and nothing at all in
/// the forward one" — and collapsing that to `|flow| ≤ 1500` invents a
/// constraint nobody wrote, in the direction the flow is most likely to go.
/// The reference's own "opposite CNEC" scenarios are exactly this case.
///
/// Returns `None` when the threshold cannot be expressed at all — a
/// [`Unit::PercentImax`] threshold on a CNEC whose `iMax` the CRAC never
/// stated, or a voltage/angle unit on a flow. `None` rather than a default is
/// the point: a fabricated limit is a constraint nobody wrote either.
fn threshold_bounds(
    threshold: &Threshold,
    cnec: &FlowCnec,
    u_rated_v: f64,
    s_base_va: f64,
) -> Option<(f64, f64, f64, f64)> {
    if threshold.min.is_none() && threshold.max.is_none() {
        return None;
    }
    let side = match threshold.side {
        Side::Two => 1,
        _ => 0,
    };
    // A current threshold has to be converted at the voltage it was *written*
    // against, and the CRAC says which: `nominalV`. That is not always the
    // network's own base — a UCTE 380 kV node is routinely operated at 400 kV,
    // and the CRAC states 400. Converting at 380 instead makes every ampere
    // threshold 5% tight, which reads as a network slightly more constrained
    // than it is rather than as an error.
    let voltage = cnec
        .nominal_v
        .and_then(|v| v[side].or(v[0]))
        .map(|kv| kv * 1000.0)
        .filter(|v| *v > 0.0)
        .unwrap_or(u_rated_v);

    // The scale from the threshold's own unit to MW. Always positive, so it
    // carries the signs of `min` and `max` through unchanged — which is the
    // whole reason the bounds can stay directional.
    let scale = match threshold.unit {
        Unit::Megawatt => 1.0,
        Unit::Ampere => current_to_power_pu(1.0, voltage, s_base_va) * s_base_va / 1e6,
        Unit::PercentImax => {
            // Despite the name this is a **fraction**, not a percentage: the
            // reference's own `ThresholdAdder` javadoc says "the min/max value
            // should be between -1 and 1, where 1 = 100%". Dividing by 100 here
            // makes every such threshold a hundred times too tight, which does
            // not look like a bug — it looks like a network that is massively
            // overloaded, and it was caught only by noticing that a 380 kV line
            // had come out with a 33 MW limit.
            let i_max = cnec.i_max?[side].or(cnec.i_max?[0])?;
            current_to_power_pu(i_max, voltage, s_base_va) * s_base_va / 1e6
        }
        Unit::Degree | Unit::Kilovolt => return None,
    };
    let lower = threshold.min.map_or(f64::NEG_INFINITY, |v| v * scale);
    let upper = threshold.max.map_or(f64::INFINITY, |v| v * scale);
    // The voltage that expresses this bound in amperes, which is the voltage it
    // was converted *from* — and a megawatt threshold was converted from
    // nothing. There the network's own base is the answer, because that is the
    // nominal voltage the reference's CNEC carries: it reads it off the network
    // rather than out of the CRAC. On the `epic5` fixture that is 380 kV
    // against the CRAC's stated 400, which is 150 A on a 2000 MW limit.
    let expressed = match threshold.unit {
        Unit::Megawatt if u_rated_v > 0.0 => u_rated_v,
        _ => voltage,
    };
    Some((lower, upper, voltage, expressed))
}

/// Evaluate every perimeter the CRAC defines, in DC.
///
/// Flows are computed once for the base case and then updated per contingency
/// through [`DcSensitivity::multi_outage_flows`], so the whole sweep costs one
/// factorization plus a small dense solve per outage.
pub fn evaluate(crac: &Crac, network: &Network<'_>, resolution: &Resolution) -> SecurityResult {
    evaluate_with(crac, network, resolution, network.initially_open)
}

/// Evaluate with a chosen flow model.
/// A perimeter whose states do **not** all see the same network.
///
/// Ordinarily a perimeter is one network and one open set. The second
/// preventive perimeter is not: it spans every state at once, and the curative
/// stage's switching is in force only in the states downstream of its own
/// contingency. Holding that switching in a single network — which is what
/// standing one network in for every state amounts to — puts a curative branch
/// into the preventive and outage states, where it is not in force, and the
/// preventive CNECs are then measured somewhere they do not live.
///
/// `Held` is the correction. It names the states that see a different network
/// and what is open in them, and every measurement reads a CNEC in the network
/// its own state describes. See `plans/RAO_PLAN.md` §8.10.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Held {
    /// Branches open in [`states`](Self::states) and not elsewhere.
    pub open: Vec<usize>,
    /// Branches open elsewhere and **not** in [`states`](Self::states) — the
    /// curative stage's closes.
    pub close: Vec<usize>,
    /// Phase-shifter positions in force in [`states`](Self::states) and not
    /// elsewhere, as `(branch, tap, angle in the CRAC's degrees)`.
    ///
    /// This is the measurement half of `A(r, s)`. A curative shifter the second
    /// preventive problem carries a column for is *not* in force preventively —
    /// it has not been decided yet — so writing its tap into the shared
    /// transformers would move a preventive flow with a curative decision.
    pub taps: Vec<(usize, i32, f64)>,
    /// The states that see the difference.
    pub states: Vec<State>,
}

impl Held {
    /// The open set of a held state, given the one that governs the rest.
    ///
    /// A delta rather than a stored set, because the set it applies to is not
    /// fixed: the second preventive pass is *choosing* the preventive switching
    /// while this is in force, and each candidate it tries changes what the
    /// curative states inherit.
    /// `network`'s transformers with this perimeter's held shifter positions in
    /// force.
    ///
    /// The network's own step is preferred over the angle, for the same reason
    /// `search::apply_taps` prefers it: a tap is a position the machine has and
    /// the angle is a description of it, and the two disagree in the last digit.
    pub fn transformers_of(&self, network: &Network<'_>) -> Vec<crate::types::Transformer> {
        let mut out = network.transformers.to_vec();
        for &(branch, tap, angle_deg) in &self.taps {
            let Some(i) = branch.checked_sub(network.lines.len()) else { continue };
            let from_network =
                network.tap_changers.get(i).and_then(|c| c.as_ref()).and_then(|c| c.at(tap));
            let Some(slot) = out.get_mut(i) else { continue };
            let ratio = slot.tap.norm();
            slot.tap = match from_network {
                Some(step) => num_complex::Complex::from_polar(ratio, step.arg()),
                None => num_complex::Complex::from_polar(ratio, angle_deg.to_radians()),
            };
        }
        out
    }

    pub fn applied_to(&self, open: &[usize]) -> Vec<usize> {
        let mut out: Vec<usize> =
            open.iter().chain(self.open.iter()).copied().filter(|b| !self.close.contains(b)).collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Evaluate a perimeter whose states do not all see the same open set.
///
/// `open` governs every state; `held`, where given, overrides it for the states
/// it names. With no `held` this is exactly [`evaluate_model`], and it costs
/// one evaluation rather than two — which matters under
/// [`FlowModel::Ac`](FlowModel::Ac), where an evaluation is a Newton-Raphson
/// solve per state.
pub fn evaluate_split(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    open: &[usize],
    held: Option<&Held>,
    model: FlowModel,
    ac: &AcOptions<'_>,
) -> SecurityResult {
    let Some(held) = held.filter(|h| !h.states.is_empty()) else {
        return evaluate_model(crac, network, resolution, open, model, ac);
    };
    // The two networks are genuinely different, so both are evaluated and each
    // state's perimeter is taken from the one that describes it.
    let base = evaluate_model(crac, network, resolution, open, model, ac);
    // The held states may also see a shifter somewhere else, so they may need a
    // network of their own rather than only an open set of their own.
    let transformers;
    let stepped;
    let other_network = if held.taps.is_empty() {
        network
    } else {
        transformers = held.transformers_of(network);
        stepped = Network { transformers: &transformers, ..*network };
        &stepped
    };
    let other =
        evaluate_model(crac, other_network, resolution, &held.applied_to(open), model, ac);
    let mut perimeters: Vec<PerimeterResult> = Vec::with_capacity(base.perimeters.len());
    for perimeter in base.perimeters {
        if held.states.contains(&perimeter.state)
            && let Some(from_held) = other.perimeters.iter().find(|p| p.state == perimeter.state)
        {
            perimeters.push(from_held.clone());
            continue;
        }
        perimeters.push(perimeter);
    }
    SecurityResult { perimeters, skipped: base.skipped }
}

pub fn evaluate_model(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    open: &[usize],
    model: FlowModel,
    ac: &AcOptions<'_>,
) -> SecurityResult {
    match model {
        FlowModel::Dc => evaluate_with(crac, network, resolution, open),
        FlowModel::Ac => evaluate_ac(crac, network, resolution, open, ac),
    }
}

/// Evaluate with a set of branches already opened.
///
/// `open` is how an applied *remedial action* reaches the evaluator: a
/// topological action that disconnects a line is, electrically, the same thing
/// as a contingency that trips it, so both go through the same Woodbury update.
/// A contingency's own outages are unioned with these, which is what makes
/// "this action, then that outage" a single rank-k correction rather than two
/// nested ones.
///
/// Branch indices are flat, and duplicates are harmless.
pub fn evaluate_with(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    open: &[usize],
) -> SecurityResult {
    let s_base_va = network.base_mva * 1e6;
    let options = DcOptions::default();
    let dc = dc_branches(network.lines, network.transformers, options);

    // Base-case flows, per-unit, by flat branch index — before any applied
    // action.
    let mut buses = network.buses.to_vec();
    let solution = dc_power_flow(&mut buses, network.lines, network.transformers, options);
    let intact_flows = solution.branch_p;

    let sensitivity = DcSensitivity::new(network.buses, &dc, network.n_branches());

    // The applied actions themselves are an outage set, so the "base case" a
    // perimeter is measured against is already the post-action network.
    let mut applied: Vec<usize> = open.to_vec();
    applied.sort_unstable();
    applied.dedup();
    let (base_flows, base_severed) = if applied.is_empty() {
        (intact_flows.clone(), false)
    } else {
        match sensitivity.as_ref().and_then(|s| {
            (!s.is_breaking_set(&applied))
                .then(|| s.multi_outage_flows(&intact_flows, &applied))
                .flatten()
        }) {
            Some(f) => (f, false),
            None => (outaged_flows(network, &applied, options), true),
        }
    };

    // Which CNECs belong to each state, and which had to be skipped.
    let mut skipped = Vec::new();
    let mut by_state: HashMap<State, Vec<usize>> = HashMap::new();
    for (i, cnec) in crac.flow_cnecs.iter().enumerate() {
        match resolution.branch(&cnec.network_element) {
            Some(_) => by_state.entry(cnec.state.clone()).or_default().push(i),
            None => skipped.push(i),
        }
    }

    let mut perimeters = Vec::new();
    for state in crac.states() {
        let Some(indices) = by_state.get(&state) else { continue };

        // Post-contingency flows. A preventive state is the base case; anything
        // else opens the contingency's elements.
        let (flows, severed) = match state.contingency {
            None => (base_flows.clone(), base_severed),
            Some(c) => {
                // The contingency's own elements *plus* whatever the applied
                // actions opened: one rank-k correction against the intact
                // network rather than a correction on a correction.
                let mut outages: Vec<usize> = crac.contingencies[c]
                    .elements
                    .iter()
                    .filter_map(|e| resolution.branch(e))
                    .chain(applied.iter().copied())
                    .collect();
                outages.sort_unstable();
                outages.dedup();
                if outages.is_empty() {
                    // Nothing resolvable to open: the honest answer is the base
                    // case, flagged, rather than a silent pretence that the
                    // contingency was simulated.
                    (base_flows.clone(), true)
                } else {
                    match sensitivity.as_ref().and_then(|s| {
                        (!s.is_breaking_set(&outages))
                            .then(|| s.multi_outage_flows(&intact_flows, &outages))
                            .flatten()
                    }) {
                        Some(f) => (f, false),
                        None => (outaged_flows(network, &outages, options), true),
                    }
                }
            }
        };

        let mut cnecs = Vec::new();
        for &i in indices {
            let cnec = &crac.flow_cnecs[i];
            let Some(branch) = resolution.branch(&cnec.network_element) else { continue };
            let Some(u_rated) = network.branch_voltage(branch, &dc) else { continue };
            let flow_mw = flows.get(branch).copied().unwrap_or(0.0) * network.base_mva;

            // The tightest expressible threshold binds. A CNEC whose thresholds
            // are all inexpressible contributes no constraint at all, which is
            // better than contributing a made-up one.
            //
            // The *binding* threshold's own conversion voltage is carried out
            // of the loop, not just the limit: reporting an ampere margin means
            // undoing the conversion that produced the limit, and on a CNEC
            // whose thresholds name different voltages any other choice is a
            // different number.
            let mut bound = Bounds::unbounded(u_rated);
            for t in &cnec.thresholds {
                let Some((lo, hi, v, expressed)) = threshold_bounds(t, cnec, u_rated, s_base_va)
                else {
                    continue;
                };
                bound.tighten(lo, hi, v, expressed, cnec.reliability_margin, 0.0);
            }
            if !bound.is_constraining() {
                continue;
            }
            let (margin_mw, u_bind) = bound.margin(flow_mw);
            // No reactive power and no voltage deviation, so the current is the
            // active power at the bound's own expression voltage — and the
            // ampere margin then comes out as the megawatt one converted, which
            // is what a DC study means by it.
            let current_a = to_amperes(flow_mw.abs(), bound.expressed(flow_mw));
            let margin_a = bound.margin_amperes(flow_mw, current_a.copysign(flow_mw));
            // Lossless, so the far end passes on exactly what the near end
            // took. Only the current differs, at the far bus's own nominal
            // voltage — which is to say across a transformer and nowhere else.
            let u_far = network.branch_far_voltage(branch, &dc).unwrap_or(u_bind);
            cnecs.push(CnecResult {
                cnec: i,
                branch,
                flow_mw,
                margin_mw,
                upper_mw: bound.upper,
                lower_mw: bound.lower,
                limit_mw: bound.magnitude(),
                conversion_v: u_bind,
                margin_a,
                current_a,
                side_two: (flow_mw, to_amperes(flow_mw.abs(), u_far)),
            });
        }
        perimeters.push(PerimeterResult { state, cnecs, severed });
    }

    SecurityResult { perimeters, skipped }
}

/// The AC settings every RAO layer measures with.
///
/// One builder rather than five literals, because the settings are not a
/// caller's taste — they are what the studies these CRACs come from were run
/// with, and a layer that measured differently from its neighbours would score
/// candidates against a network the next layer does not agree exists.
///
/// **The slack is distributed.** Every configuration the reference ships sets
/// `distributedSlack: true` with `PROPORTIONAL_TO_GENERATION_P`, and it is not
/// a refinement: a single slack puts the whole imbalance at one bus, and when a
/// contingency islands a generator the imbalance is that generator's entire
/// output. On the reference's `epic5` fixture, opening both of a node's only
/// two branches strands 1000 MW of net injection, and where it reappears
/// decides the answer — the slack there sits at one end of the Belgium–France
/// tie, so a single slack pushes the whole make-up through France and out over
/// the CNEC being measured. 1165.5 MW against the reference's 1000.
///
/// The weighting is what makes it work, and it is why this looked innocent for
/// a long time: weighting by *net* injection puts 70% of the make-up back
/// inside or next to the country that lost it and lands on 1160.8, barely
/// moving. Weighting by generation lands on 1000.2.
pub fn ac_options<'a>(network: &Network<'a>) -> AcOptions<'a> {
    AcOptions {
        shunts: network.shunts,
        distribute_slack: true,
        slack_weights: network.generation,
        ..Default::default()
    }
}

/// Fall-back flows for a contingency the Woodbury update cannot answer —
/// typically one that severs the network, where there is no rank-`k` correction
/// to apply because the base factorization no longer describes the topology.
fn outaged_flows(network: &Network<'_>, outages: &[usize], options: DcOptions) -> Vec<f64> {
    let mut lines = network.lines.to_vec();
    let mut transformers = network.transformers.to_vec();
    for &branch in outages {
        if branch < lines.len() {
            // Removing a line outright would renumber every later branch, so it
            // is opened by making it non-conducting instead — the same trick
            // `network::build_ybus_with_outages` uses to keep the sparsity
            // pattern.
            lines[branch].r = crate::topology::reduction::OPEN_BRANCH_Z;
            lines[branch].x = crate::topology::reduction::OPEN_BRANCH_Z;
            lines[branch].b_shunt = 0.0;
            lines[branch].g_shunt = 0.0;
        } else if let Some(t) = transformers.get_mut(branch - lines.len()) {
            t.from_status = 0;
            t.to_status = 0;
        }
    }
    let mut buses = network.buses.to_vec();
    dc_power_flow(&mut buses, &lines, &transformers, options).branch_p
}

/// Knobs the AC flow model needs and the DC one has no use for.
///
/// Kept separate from [`Network`] deliberately: shunts, a convergence
/// tolerance, and an iteration cap are meaningless to a linear solve, and
/// hanging them on the shared struct would make every DC caller state values
/// that are never read.
#[derive(Clone, Copy, Debug)]
pub struct AcOptions<'a> {
    /// Shunt admittances, applied to every scenario.
    pub shunts: &'a [ShuntAdm],
    /// Newton-Raphson convergence tolerance, per-unit power mismatch.
    pub tol: f64,
    pub max_iter: usize,
    /// Jacobian backend for the contingency sweep.
    pub backend: JacobianBackend,
    /// Spread the slack across generators rather than leaving it all on one
    /// bus.
    ///
    /// The reference's configurations set `distributedSlack: true` with
    /// `PROPORTIONAL_TO_GENERATION_P`, and every study these CRACs come from
    /// runs that way: a single slack puts the whole loss change at one bus,
    /// which redistributes flows differently from a share taken across the
    /// machines that would actually respond.
    ///
    /// Costs the shared factorization — each scenario is solved on its own
    /// Y-bus — so it is off unless asked for.
    pub distribute_slack: bool,
    /// Per-bus participation weight for [`distribute_slack`](Self::distribute_slack),
    /// which the reference wants proportional to **generation**.
    ///
    /// Empty falls back to the net injection at each generator bus, which is
    /// the only figure the model carries on its own — and which is *not* the
    /// same distribution. A node generating 2000 MW behind 1000 MW of load nets
    /// to the same 1000 as one generating 1000 behind nothing, so netting
    /// concentrates the share on whichever machines happen to sit behind little
    /// load. On the reference's `epic5` fixture that is the difference between
    /// 1160.8 MW and 1000.2 on the CNEC being measured, against its own 1000:
    /// the net-injection weighting puts 70% of the make-up back inside or next
    /// to the country that lost it, which is exactly where it must not go.
    pub slack_weights: &'a [f64],
}

impl Default for AcOptions<'_> {
    fn default() -> Self {
        Self {
            shunts: &[],
            tol: 1e-8,
            max_iter: 30,
            backend: JacobianBackend::Scalar,
            distribute_slack: false,
            slack_weights: &[],
        }
    }
}

/// Measure every CNEC with a full AC power flow per state.
///
/// The contingency sweep goes through [`BatchSolver::solve_contingencies`], so
/// the whole set of states shares one symbolic factorization rather than
/// re-analysing the sparsity pattern per outage.
///
/// Two things differ from the DC path beyond the obvious, and both are physics
/// the linear model cannot see:
///
/// * **Current is measured at the actual voltage.** A bus running at 1.05 pu
///   carries a given MW at 5% less current than nominal, so a DC study reading
///   its ampere thresholds at nominal voltage is conservative there and
///   optimistic wherever voltage has sagged.
/// * **Reactive flow consumes thermal headroom.** An ampere threshold binds
///   `|S|`, not `|P|`. Rather than change what a margin means, the reactive
///   part is charged against the limit, so `margin = limit − |P|` still holds
///   and the remaining limit is the headroom a real-power move can use.
///
/// MW thresholds are left alone by both rules: they bind active power, which is
/// exactly what `flow_mw` reports.
pub fn evaluate_ac(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    open: &[usize],
    ac: &AcOptions<'_>,
) -> SecurityResult {
    let s_base_va = network.base_mva * 1e6;
    let dc = dc_branches(network.lines, network.transformers, DcOptions::default());

    let mut applied: Vec<usize> = open.to_vec();
    applied.sort_unstable();
    applied.dedup();

    // Which CNECs belong to each state, and which had to be skipped.
    let mut skipped = Vec::new();
    let mut by_state: HashMap<State, Vec<usize>> = HashMap::new();
    for (i, cnec) in crac.flow_cnecs.iter().enumerate() {
        match resolution.branch(&cnec.network_element) {
            Some(_) => by_state.entry(cnec.state.clone()).or_default().push(i),
            None => skipped.push(i),
        }
    }

    // One scenario per state that has CNECs, in `crac.states()` order so the
    // reports come back aligned with the perimeters below.
    let states: Vec<State> = crac.states().into_iter().filter(|s| by_state.contains_key(s)).collect();
    let scenarios: Vec<Scenario> = states
        .iter()
        .map(|state| {
            let mut outages = applied.clone();
            if let Some(c) = state.contingency {
                outages.extend(
                    crac.contingencies[c].elements.iter().filter_map(|e| resolution.branch(e)),
                );
            }
            outages.sort_unstable();
            outages.dedup();
            Scenario { bus_overrides: Vec::new(), branch_outages: outages }
        })
        .collect();

    let reports = if ac.distribute_slack {
        Ok(solve_distributing_slack(network, &scenarios, ac))
    } else {
        let solver = BatchSolver::new(ac.backend);
        solver.solve_contingencies(
            network.buses,
            network.lines,
            network.transformers,
            ac.shunts,
            &scenarios,
            ac.tol,
            ac.max_iter,
        )
    };
    // A batch that cannot run at all is reported as every state severed rather
    // than as a panic or a silently empty result: the caller asked what the
    // margins are, and "unknown" is an answer it can act on.
    let Ok(reports) = reports else {
        return SecurityResult {
            perimeters: states
                .into_iter()
                .map(|state| PerimeterResult { state, cnecs: Vec::new(), severed: true })
                .collect(),
            skipped,
        };
    };

    let params = branch_params(network.lines, network.transformers);
    let mut perimeters = Vec::new();
    for ((state, scenario), report) in states.into_iter().zip(&scenarios).zip(&reports) {
        let v = bus_voltages(&report.buses);
        let severed = report.islands.iter().any(|i| i.status != IslandStatus::Converged);
        let out: Vec<bool> = {
            let mut out = vec![false; params.len()];
            for &b in &scenario.branch_outages {
                if let Some(slot) = out.get_mut(b) {
                    *slot = true;
                }
            }
            out
        };

        let mut cnecs = Vec::new();
        for &i in by_state.get(&state).into_iter().flatten() {
            let cnec = &crac.flow_cnecs[i];
            let Some(branch) = resolution.branch(&cnec.network_element) else { continue };
            let Some(u_rated) = network.branch_voltage(branch, &dc) else { continue };
            let Some(param) = params.get(branch) else { continue };

            // An outaged branch carries nothing. `solve_contingencies` removes
            // it from the Y-bus, but `params` still describes it, so evaluating
            // `terminal_flow` here would report the current that *would* flow
            // across a branch that is not there.
            let outaged = out.get(branch).copied().unwrap_or(false);

            // Both terminals, so a threshold can be measured on the side it
            // names. They differ by the branch's own losses, which is precisely
            // the information a DC study does not have.
            let ends = [Terminal::From, Terminal::To].map(|t| {
                if outaged {
                    (0.0, 0.0, u_rated)
                } else {
                    let (p, q) = terminal_flow(param, t, &v);
                    let bus = match t {
                        Terminal::From => param.from,
                        Terminal::To => param.to,
                    };
                    let u_actual = network
                        .buses
                        .get(bus)
                        .map(|b| b.u_rated * v.get(bus).map_or(1.0, |x| x.norm()))
                        .filter(|u| *u > 0.0)
                        .unwrap_or(u_rated);
                    (p * network.base_mva, q * network.base_mva, u_actual)
                }
            });

            // `flow_mw` keeps the DC path's meaning: active power at side one,
            // signed, so the two models' outputs are directly comparable.
            let flow_mw = ends[0].0;

            let mut bound = Bounds::unbounded(u_rated);
            for t in &cnec.thresholds {
                let Some((lo, hi, u_written, expressed)) =
                    threshold_bounds(t, cnec, u_rated, s_base_va)
                else {
                    continue;
                };
                let (p_mw, q_mvar, u_actual) = match t.side {
                    Side::Two => ends[1],
                    _ => ends[0],
                };
                let charge = match t.unit {
                    // An ampere limit binds apparent power, and at the voltage
                    // the bus is actually running at rather than the one the
                    // threshold was written against. Both corrections are taken
                    // off both bounds, since neither is available in either
                    // direction.
                    //
                    // **Signed, not clamped.** The reactive part only ever eats
                    // headroom, but the voltage ratio goes either way: a bus
                    // running above the voltage its threshold was written for
                    // draws *less* current for the same megawatts, which is
                    // headroom gained. Clamping this at zero conflates the two
                    // and silently discards the gain — worth 74 A on a 1300 A
                    // threshold written at 380 kV for a node operating at 400.
                    Unit::Ampere | Unit::PercentImax => {
                        let s_equiv = p_mw.hypot(q_mvar) * u_written / u_actual;
                        s_equiv - p_mw.abs()
                    }
                    // Voltage and angle units never reach here: `threshold_bounds`
                    // returns `None` for them rather than inventing a flow
                    // limit, so the `continue` above has already fired.
                    Unit::Megawatt | Unit::Degree | Unit::Kilovolt => 0.0,
                };
                bound.tighten(lo, hi, u_written, expressed, cnec.reliability_margin, charge);
            }
            if !bound.is_constraining() {
                continue;
            }

            // Report the current on whichever side the CNEC's thresholds
            // name, so an ampere assertion is checked where it was written.
            let (p_mw, q_mvar, u_actual) =
                match cnec.thresholds.first().map(|t| t.side) {
                    Some(Side::Two) => ends[1],
                    _ => ends[0],
                };
            let current_a = if u_actual > 0.0 {
                p_mw.hypot(q_mvar) * 1e6 / (3f64.sqrt() * u_actual)
            } else {
                0.0
            };

            let (margin_mw, u_bind) = bound.margin(flow_mw);
            // The ampere margin is a difference of two ampere quantities, not
            // the megawatt margin converted. `margin_mw` is `limit − |P|` and
            // carries neither reactive power nor the bus's actual voltage; the
            // current carries both, and the reference's own margin is
            // `limit_in_amperes − I`. On its `epic5` fixture the two readings
            // differ by 150 A on a 2000 MW threshold — enough to change which
            // action the search takes, which is how it was found.
            //
            // For an ampere threshold the two agree exactly, because the charge
            // above has already taken the same two effects off the megawatt
            // limit. This is the general form of that trick, not a second one.
            let margin_a = bound.margin_amperes(flow_mw, current_a.copysign(flow_mw));
            // Negated: `ends[1]` is the power *entering* the branch at its far
            // terminal, and side two reports what leaves there.
            let (p_two, q_two, u_two) = (-ends[1].0, -ends[1].1, ends[1].2);
            cnecs.push(CnecResult {
                cnec: i,
                branch,
                flow_mw,
                margin_mw,
                upper_mw: bound.upper,
                lower_mw: bound.lower,
                limit_mw: bound.magnitude(),
                conversion_v: u_bind,
                margin_a,
                current_a,
                side_two: (
                    p_two,
                    if u_two > 0.0 {
                        p_two.hypot(q_two) * 1e6 / (3f64.sqrt() * u_two)
                    } else {
                        0.0
                    },
                ),
            });
        }
        perimeters.push(PerimeterResult { state, cnecs, severed });
    }

    SecurityResult { perimeters, skipped }
}

/// Solve each scenario on its own Y-bus, spreading the slack across generators
/// in proportion to their scheduled active power.
///
/// `BatchSolver::solve_contingencies` shares one symbolic factorization but
/// leaves the slack on a single bus, so distributing it means giving that up.
/// The weights follow the reference's `PROPORTIONAL_TO_GENERATION_P`: each
/// generator bus's own schedule, falling back to an equal share when nothing
/// generates, which keeps a network of pure loads solvable rather than dividing
/// by a zero total.
fn solve_distributing_slack(
    network: &Network<'_>,
    scenarios: &[Scenario],
    ac: &AcOptions<'_>,
) -> Vec<crate::PowerFlowReport> {
    let n = network.buses.len();
    let n_branches = network.n_branches();
    // Generation where the caller supplied it, net injection otherwise. The
    // fallback is the honest one for a network model that never retained the
    // generation, and it is a different distribution — see
    // [`AcOptions::slack_weights`].
    let weights: Vec<f64> = if ac.slack_weights.len() == network.buses.len() {
        network
            .buses
            .iter()
            .zip(ac.slack_weights)
            .map(|(b, w)| match b.bus_type {
                crate::types::BusType::Slack | crate::types::BusType::PV => w.max(0.0),
                crate::types::BusType::PQ => 0.0,
            })
            .collect()
    } else {
        network
            .buses
            .iter()
            .map(|b| match b.bus_type {
                crate::types::BusType::Slack | crate::types::BusType::PV => b.p_spec.max(0.0),
                crate::types::BusType::PQ => 0.0,
            })
            .collect()
    };
    let distribution = if weights.iter().sum::<f64>() > 0.0 {
        SlackDistribution::from_weights(weights)
    } else {
        SlackDistribution::uniform(network.buses)
    };

    scenarios
        .iter()
        .map(|scenario| {
            let mut outaged = vec![false; n_branches];
            for &b in &scenario.branch_outages {
                if let Some(slot) = outaged.get_mut(b) {
                    *slot = true;
                }
            }
            let mut ybus =
                build_ybus_with_outages(n, network.lines, network.transformers, &outaged);
            stamp_shunts(&mut ybus, ac.shunts);
            let mut buses = network.buses.to_vec();
            let mut ybus = ybus.finish();
            let mut slack =
                crate::outerloop::DistributedSlack::new(distribution.clone());
            let (islands, _) = {
                let mut ctx = crate::outerloop::SolveContext::new(&mut buses, &mut ybus);
                let mut list: Vec<&mut dyn crate::outerloop::OuterLoop> = vec![&mut slack];
                crate::outerloop::solve_with_loops(
                    &mut ctx,
                    ac.tol,
                    ac.max_iter,
                    ac.backend,
                    &mut list,
                    distribution.max_outer_iter,
                )
            };
            // Only `buses` and `islands` are read here. The per-island
            // statuses carry the convergence verdict; `stats` describes a
            // single inner solve, of which this path runs several.
            let converged = islands.iter().all(|i| i.status == IslandStatus::Converged);
            crate::PowerFlowReport {
                buses,
                islands,
                stats: crate::solver::SolveStats {
                    status: if converged {
                        crate::solver::SolveStatus::Converged
                    } else {
                        crate::solver::SolveStatus::MaxIterationsReached
                    },
                    mismatch_history: Vec::new(),
                },
                dc: None,
                linear: None,
                outer: None,
            }
        })
        .collect()
}

/// Convert a three-phase power in MW to a current in amperes at `voltage_v`.
///
/// Sign-preserving, because a margin can be negative and an overload of −40 A
/// is not an overload of 40 A.
fn to_amperes(power_mw: f64, voltage_v: f64) -> f64 {
    if voltage_v > 0.0 { power_mw * 1e6 / (3f64.sqrt() * voltage_v) } else { 0.0 }
}

