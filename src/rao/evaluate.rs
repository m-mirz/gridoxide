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
use crate::types::{Bus, Line, Transformer};

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
    /// The threshold that bound, MW, as an absolute limit on |flow|.
    pub limit_mw: f64,
}

impl CnecResult {
    pub fn is_violated(&self) -> bool {
        self.margin_mw < 0.0
    }
}

/// How one perimeter — one state — fared.
#[derive(Clone, Debug, PartialEq)]
pub struct PerimeterResult {
    pub state: State,
    pub cnecs: Vec<CnecResult>,
    /// True when the contingency severed the network, so the flows below are a
    /// re-solve rather than a Woodbury update, or could not be computed at all.
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

/// Convert one threshold to an absolute MW limit on |flow|.
///
/// Returns `None` when the threshold cannot be expressed — a
/// [`Unit::PercentImax`] threshold on a CNEC whose `iMax` the CRAC never stated,
/// or a voltage/angle unit on a flow. Returning `None` rather than a default is
/// the point: a fabricated limit is a constraint nobody wrote.
fn threshold_mw(
    threshold: &Threshold,
    cnec: &FlowCnec,
    u_rated_v: f64,
    s_base_va: f64,
) -> Option<f64> {
    let magnitude = match (threshold.min, threshold.max) {
        (Some(min), Some(max)) => f64::min(min.abs(), max.abs()),
        (Some(min), None) => min.abs(),
        (None, Some(max)) => max.abs(),
        (None, None) => return None,
    };
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
    match threshold.unit {
        Unit::Megawatt => Some(magnitude),
        Unit::Ampere => {
            Some(current_to_power_pu(magnitude, voltage, s_base_va) * s_base_va / 1e6)
        }
        Unit::PercentImax => {
            // Despite the name this is a **fraction**, not a percentage: the
            // reference's own `ThresholdAdder` javadoc says "the min/max value
            // should be between -1 and 1, where 1 = 100%". Dividing by 100 here
            // makes every such threshold a hundred times too tight, which does
            // not look like a bug — it looks like a network that is massively
            // overloaded, and it was caught only by noticing that a 380 kV line
            // had come out with a 33 MW limit.
            let i_max = cnec.i_max?[side].or(cnec.i_max?[0])?;
            let amperes = i_max * magnitude;
            Some(current_to_power_pu(amperes, voltage, s_base_va) * s_base_va / 1e6)
        }
        Unit::Degree | Unit::Kilovolt => None,
    }
}

/// Evaluate every perimeter the CRAC defines, in DC.
///
/// Flows are computed once for the base case and then updated per contingency
/// through [`DcSensitivity::multi_outage_flows`], so the whole sweep costs one
/// factorization plus a small dense solve per outage.
pub fn evaluate(crac: &Crac, network: &Network<'_>, resolution: &Resolution) -> SecurityResult {
    evaluate_with(crac, network, resolution, network.initially_open)
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
            let limit_mw = cnec
                .thresholds
                .iter()
                .filter_map(|t| threshold_mw(t, cnec, u_rated, s_base_va))
                .map(|l| (l - cnec.reliability_margin).max(0.0))
                .fold(f64::INFINITY, f64::min);
            if !limit_mw.is_finite() {
                continue;
            }
            cnecs.push(CnecResult {
                cnec: i,
                branch,
                flow_mw,
                margin_mw: limit_mw - flow_mw.abs(),
                limit_mw,
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
