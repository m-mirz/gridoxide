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
use crate::solver::{
    newton_raphson_distributing_slack, IslandStatus, JacobianBackend, SlackDistribution,
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
}

impl Bounds {
    fn unbounded(u_rated: f64) -> Self {
        Self {
            upper: f64::INFINITY,
            lower: f64::NEG_INFINITY,
            u_upper: u_rated,
            u_lower: u_rated,
        }
    }

    /// Fold in one threshold's bounds, each tightened by the reliability margin
    /// and by whatever headroom reactive flow has already consumed.
    fn tighten(&mut self, lower: f64, upper: f64, voltage: f64, frm: f64, charge: f64) {
        let upper = upper - frm - charge;
        let lower = lower + frm + charge;
        if upper < self.upper {
            self.upper = upper;
            self.u_upper = voltage;
        }
        if lower > self.lower {
            self.lower = lower;
            self.u_lower = voltage;
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
) -> Option<(f64, f64, f64)> {
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
    Some((lower, upper, voltage))
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
                let Some((lo, hi, v)) = threshold_bounds(t, cnec, u_rated, s_base_va) else {
                    continue;
                };
                bound.tighten(lo, hi, v, cnec.reliability_margin, 0.0);
            }
            if !bound.is_constraining() {
                continue;
            }
            let (margin_mw, u_bind) = bound.margin(flow_mw);
            cnecs.push(CnecResult {
                cnec: i,
                branch,
                flow_mw,
                margin_mw,
                upper_mw: bound.upper,
                lower_mw: bound.lower,
                limit_mw: bound.magnitude(),
                conversion_v: u_bind,
                margin_a: to_amperes(margin_mw, u_bind),
                current_a: to_amperes(flow_mw.abs(), u_bind),
            });
        }
        perimeters.push(PerimeterResult { state, cnecs, severed });
    }

    SecurityResult { perimeters, skipped }
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
}

impl Default for AcOptions<'_> {
    fn default() -> Self {
        Self {
            shunts: &[],
            tol: 1e-8,
            max_iter: 30,
            backend: JacobianBackend::Scalar,
            distribute_slack: false,
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
                let Some((lo, hi, u_written)) = threshold_bounds(t, cnec, u_rated, s_base_va)
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
                    // threshold was written against. The headroom the reactive
                    // part consumes is taken off *both* bounds, since it is
                    // unavailable in either direction.
                    Unit::Ampere | Unit::PercentImax => {
                        let s_equiv = p_mw.hypot(q_mvar) * u_written / u_actual;
                        (s_equiv - p_mw.abs()).max(0.0)
                    }
                    // Voltage and angle units never reach here: `threshold_bounds`
                    // returns `None` for them rather than inventing a flow
                    // limit, so the `continue` above has already fired.
                    Unit::Megawatt | Unit::Degree | Unit::Kilovolt => 0.0,
                };
                bound.tighten(lo, hi, u_written, cnec.reliability_margin, charge);
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
            cnecs.push(CnecResult {
                cnec: i,
                branch,
                flow_mw,
                margin_mw,
                upper_mw: bound.upper,
                lower_mw: bound.lower,
                limit_mw: bound.magnitude(),
                conversion_v: u_bind,
                margin_a: to_amperes(margin_mw, u_bind),
                current_a,
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
    let weights: Vec<f64> = network
        .buses
        .iter()
        .map(|b| match b.bus_type {
            crate::types::BusType::Slack | crate::types::BusType::PV => b.p_spec.max(0.0),
            crate::types::BusType::PQ => 0.0,
        })
        .collect();
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
            let (islands, _) = newton_raphson_distributing_slack(
                &mut buses,
                &ybus.finish(),
                ac.tol,
                ac.max_iter,
                ac.backend,
                &distribution,
            );
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
                    q_limit_switches: Vec::new(),
                    q_limit_stabilized: true,
                },
                dc: None,
                linear: None,
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

