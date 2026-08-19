//! The whole optimization: perimeters, in order, each carrying the last one's
//! decisions forward.
//!
//! [`search`](super::search::search) answers one perimeter. This decides *which
//! perimeters there are* and *in what order*, which is the difference between a
//! set of independent answers and a plan.
//!
//! # The decomposition
//!
//! - The **preventive perimeter** is the base case together with every outage
//!   state. Both are secured by the same actions, because an outage instant is
//!   too soon for anyone to do anything: whatever protects it must already have
//!   been done.
//! - Each contingency then gets its own **curative perimeters**, one per
//!   curative instant that has an action available, solved in chronological
//!   order with the preventive decisions applied and each instant's result fixed
//!   before the next begins.
//!
//! Refusing to put every contingency's curative actions into one problem is not
//! an approximation for speed. Curative actions for different contingencies are
//! never taken together — only one contingency happens — so optimizing them
//! jointly would let the answer trade one against another, which is
//! meaningless.
//!
//! # The pull-forward rule
//!
//! A curative CNEC for which **no curative action exists** cannot be secured
//! after the fact, so it is moved into the preventive perimeter. Without this
//! the preventive optimization is free to park a flow between the PATL and the
//! TATL — acceptable at the outage instant, and permanently overloaded
//! afterwards with nothing available to fix it. It is the one rule in this
//! layer that changes an answer rather than just organising the work, and
//! [`Plan::pulled_forward`] reports where it applied.

use crate::opf::Solver;

use super::crac::{Crac, InstantKind, State};
use super::evaluate::{evaluate_with, Network, Resolution};
use super::linear::Setpoint;
use super::search::{search, SearchOptions, SearchResult};

/// What one perimeter was told to do.
#[derive(Clone, Debug)]
pub struct PerimeterPlan {
    /// The states this perimeter secures.
    pub states: Vec<State>,
    /// Indices into [`Crac::network_actions`].
    pub network_actions: Vec<usize>,
    pub setpoints: Vec<Setpoint>,
    pub initial_margin_mw: f64,
    pub final_margin_mw: f64,
    pub leaves: usize,
    /// Branches left open by every decision in force here — this perimeter's
    /// own and everything carried forward into it.
    pub open_branches: Vec<usize>,
    /// The transformers as this perimeter leaves them.
    ///
    /// Together with `open_branches` this is the network the perimeter's
    /// figures describe, which is what lets a caller re-derive any quantity the
    /// plan does not itself report — a per-CNEC margin, say.
    pub transformers: Vec<crate::types::Transformer>,
}

impl PerimeterPlan {
    pub fn improvement(&self) -> f64 {
        self.final_margin_mw - self.initial_margin_mw
    }

    pub fn is_secure(&self) -> bool {
        self.final_margin_mw >= 0.0
    }
}

/// One contingency's curative sequence.
#[derive(Clone, Debug)]
pub struct ScenarioPlan {
    /// Index into [`Crac::contingencies`].
    pub contingency: usize,
    /// One entry per curative instant that had something to decide, in
    /// chronological order.
    pub perimeters: Vec<PerimeterPlan>,
}

/// The complete answer.
#[derive(Clone, Debug)]
pub struct Plan {
    pub preventive: PerimeterPlan,
    pub scenarios: Vec<ScenarioPlan>,
    /// CNEC indices moved into the preventive perimeter because no curative
    /// action could secure them. See the module docs — this is the one rule
    /// here that changes the answer.
    pub pulled_forward: Vec<usize>,
    /// Worst margin anywhere before and after.
    pub initial_margin_mw: f64,
    pub final_margin_mw: f64,
}

impl Plan {
    pub fn is_secure(&self) -> bool {
        self.final_margin_mw >= 0.0
    }

    pub fn improvement(&self) -> f64 {
        self.final_margin_mw - self.initial_margin_mw
    }

    /// Every perimeter, preventive first.
    pub fn perimeters(&self) -> impl Iterator<Item = &PerimeterPlan> {
        std::iter::once(&self.preventive)
            .chain(self.scenarios.iter().flat_map(|s| s.perimeters.iter()))
    }
}

/// Run the whole optimization.
pub fn run(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    solver: &mut dyn Solver,
    options: &SearchOptions,
) -> Plan {
    let states = crac.states();

    // Which curative states can actually be acted on.
    let actionable = |state: &State| -> bool {
        crac.network_actions.iter().any(|a| a.usage_rules.iter().any(|r| r.covers(state)))
            || crac.range_actions.iter().any(|a| a.usage_rules.iter().any(|r| r.covers(state)))
    };

    // The preventive perimeter: base case, every outage state, and any curative
    // or auto state nothing can act on.
    let mut preventive_states: Vec<State> = Vec::new();
    let mut pulled_forward: Vec<usize> = Vec::new();
    for state in &states {
        let kind = crac.instants[state.instant].kind;
        let include = match kind {
            InstantKind::Preventive | InstantKind::Outage => true,
            InstantKind::Auto | InstantKind::Curative => !actionable(state),
        };
        if include {
            preventive_states.push(state.clone());
            if matches!(kind, InstantKind::Auto | InstantKind::Curative) {
                pulled_forward.extend(
                    crac.flow_cnecs
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| c.state == *state)
                        .map(|(i, _)| i),
                );
            }
        }
    }
    if preventive_states.is_empty() {
        preventive_states = states.clone();
    }

    let initial = worst_margin(crac, network, resolution, &[], &states);

    let preventive = search(crac, network, resolution, &preventive_states, solver, options);
    let mut plan_preventive = PerimeterPlan {
        states: preventive_states.clone(),
        network_actions: preventive.network_actions.clone(),
        setpoints: preventive.setpoints.clone(),
        initial_margin_mw: preventive.initial_margin_mw,
        final_margin_mw: preventive.final_margin_mw,
        leaves: preventive.leaves,
        open_branches: preventive.open_branches.clone(),
        transformers: preventive.transformers.clone(),
    };

    // Everything downstream sees the preventive decisions already taken.
    let carried_open = preventive.open_branches.clone();
    let carried_transformers = preventive.transformers.clone();

    let mut scenarios = Vec::new();
    for (contingency, _) in crac.contingencies.iter().enumerate() {
        let mut curative_instants: Vec<usize> = states
            .iter()
            .filter(|s| s.contingency == Some(contingency))
            .filter(|s| {
                matches!(
                    crac.instants[s.instant].kind,
                    InstantKind::Curative | InstantKind::Auto
                )
            })
            .filter(|s| actionable(s))
            .map(|s| s.instant)
            .collect();
        curative_instants.sort_unstable();
        curative_instants.dedup();
        if curative_instants.is_empty() {
            continue;
        }

        // Each curative instant is solved with the previous one's result fixed.
        let mut open = carried_open.clone();
        let mut transformers = carried_transformers.clone();
        let mut perimeters = Vec::new();
        for instant in curative_instants {
            let state = State { instant, contingency: Some(contingency) };
            let view = Network {
                buses: network.buses,
                lines: network.lines,
                transformers: &transformers,
                branch_ids: network.branch_ids,
                base_mva: network.base_mva,
            };
            let result = curative_search(
                crac, &view, resolution, &[state.clone()], solver, options, &open,
            );
            let mut in_force = open.clone();
            in_force.extend(result.open_branches.iter().copied());
            in_force.sort_unstable();
            in_force.dedup();
            perimeters.push(PerimeterPlan {
                states: vec![state],
                network_actions: result.network_actions.clone(),
                setpoints: result.setpoints.clone(),
                initial_margin_mw: result.initial_margin_mw,
                final_margin_mw: result.final_margin_mw,
                leaves: result.leaves,
                open_branches: in_force.clone(),
                transformers: result.transformers.clone(),
            });
            open = in_force;
            transformers = result.transformers;
        }
        scenarios.push(ScenarioPlan { contingency, perimeters });
    }

    // The worst margin after everything, measured once over every state with
    // the preventive decisions in place. Curative decisions apply only in their
    // own scenario, so each scenario's own perimeters carry those figures.
    let final_margin = plan_preventive
        .final_margin_mw
        .min(
            scenarios
                .iter()
                .flat_map(|s| s.perimeters.iter())
                .map(|p| p.final_margin_mw)
                .fold(f64::INFINITY, f64::min),
        );
    plan_preventive.states = preventive_states;
    pulled_forward.sort_unstable();
    pulled_forward.dedup();

    Plan {
        preventive: plan_preventive,
        scenarios,
        pulled_forward,
        initial_margin_mw: initial,
        final_margin_mw: final_margin,
    }
}

/// A search over a curative perimeter, with branches already open from earlier
/// decisions.
///
/// The open set has to be *inside* the leaf evaluation rather than applied to
/// the network first, because a search leaf adds to it — and a branch opened
/// twice is still one branch, which the union handles and a re-application
/// would not.
fn curative_search(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    solver: &mut dyn Solver,
    options: &SearchOptions,
    already_open: &[usize],
) -> SearchResult {
    if already_open.is_empty() {
        return search(crac, network, resolution, perimeter, solver, options);
    }
    // Represent the already-open branches by removing them from the working
    // copy, so the search's own candidates compose with them naturally.
    let mut lines = network.lines.to_vec();
    let mut transformers = network.transformers.to_vec();
    for &branch in already_open {
        if branch < lines.len() {
            lines[branch].r = f64::INFINITY;
            lines[branch].x = f64::INFINITY;
            lines[branch].b_shunt = 0.0;
            lines[branch].g_shunt = 0.0;
        } else if let Some(t) = transformers.get_mut(branch - lines.len()) {
            t.from_status = 0;
            t.to_status = 0;
        }
    }
    let view = Network {
        buses: network.buses,
        lines: &lines,
        transformers: &transformers,
        branch_ids: network.branch_ids,
        base_mva: network.base_mva,
    };
    search(crac, &view, resolution, perimeter, solver, options)
}

fn worst_margin(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    open: &[usize],
    states: &[State],
) -> f64 {
    evaluate_with(crac, network, resolution, open)
        .perimeters
        .iter()
        .filter(|p| states.contains(&p.state))
        .filter_map(|p| p.min_margin())
        .fold(f64::INFINITY, f64::min)
}
