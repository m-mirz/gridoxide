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
use super::automaton::{simulate, AutomatonResult};
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
    /// The buses as this perimeter leaves them, redispatch included.
    pub buses: Vec<crate::types::Bus>,
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
    /// What the automatons did, before any curative perimeter was solved.
    ///
    /// `None` when this contingency has no `auto` state. Automatons are
    /// *simulated*, not chosen, so this is a record of what the equipment did
    /// rather than a decision — but the curative perimeters start from it, so
    /// it belongs in the plan.
    pub automatons: Option<AutomatonResult>,
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
            // An auto state is never pulled forward: its actions are forced
            // rather than optimized, so "nothing can act on it" is decided by
            // the simulator, not by whether an optimizer would choose to.
            InstantKind::Auto => false,
            InstantKind::Curative => !actionable(state),
        };
        if include {
            preventive_states.push(state.clone());
            if matches!(kind, InstantKind::Curative) {
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

    let initial = worst_margin(crac, network, resolution, network.initially_open, &states);

    let preventive = search(crac, network, resolution, &preventive_states, solver, options);
    let mut plan_preventive = PerimeterPlan {
        states: preventive_states.clone(),
        network_actions: preventive.network_actions.clone(),
        setpoints: preventive.setpoints.clone(),
        initial_margin_mw: preventive.initial_margin_mw,
        final_margin_mw: preventive.final_margin_mw,
        leaves: preventive.leaves,
        open_branches: preventive.open_branches.clone(),
        buses: preventive.buses.clone(),
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
            .filter(|s| crac.instants[s.instant].kind == InstantKind::Curative)
            .filter(|s| actionable(s))
            .map(|s| s.instant)
            .collect();
        curative_instants.sort_unstable();
        curative_instants.dedup();

        // Automatons fire before anything curative is decided, and what they
        // leave behind is what the curative perimeters see.
        let mut open = carried_open.clone();
        let mut transformers = carried_transformers.clone();
        let automatons = {
            let view = Network {
                buses: network.buses,
                lines: network.lines,
                transformers: &transformers,
                branch_ids: network.branch_ids,
                bus_ids: network.bus_ids,
                initially_open: network.initially_open,
                bus_countries: network.bus_countries,
                shunts: network.shunts,
                tap_changers: network.tap_changers,
                base_mva: network.base_mva,
            };
            simulate(crac, &view, resolution, contingency, &open, &transformers)
        };
        if let Some(result) = &automatons {
            open = result.open_branches.clone();
            transformers = result.transformers.clone();
        }
        if curative_instants.is_empty() && automatons.is_none() {
            continue;
        }
        let mut perimeters = Vec::new();
        for instant in curative_instants {
            let state = State { instant, contingency: Some(contingency) };
            let view = Network {
                buses: network.buses,
                lines: network.lines,
                transformers: &transformers,
                branch_ids: network.branch_ids,
                bus_ids: network.bus_ids,
                initially_open: network.initially_open,
                bus_countries: network.bus_countries,
                shunts: network.shunts,
                tap_changers: network.tap_changers,
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
                buses: result.buses.clone(),
                transformers: result.transformers.clone(),
            });
            open = in_force;
            transformers = result.transformers;
        }
        scenarios.push(ScenarioPlan { contingency, automatons, perimeters });
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
        )
        .min(
            scenarios
                .iter()
                .filter_map(|s| s.automatons.as_ref())
                .map(|a| a.final_margin_mw)
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
    // No early return for an empty set. "Nothing is open" is a *result* here —
    // an automaton that closes the file's only standby circuit produces exactly
    // that — and handing the unmodified network back would let `search` re-derive
    // the open set from `initially_open` and undo the closure. The set has to be
    // stated, including when it is empty.
    //
    // Represent the already-open branches by removing them from the working
    // copy, so the search's own candidates compose with them naturally.
    let mut lines = network.lines.to_vec();
    let mut transformers = network.transformers.to_vec();
    for &branch in already_open {
        if branch < lines.len() {
            lines[branch].r = crate::topology::reduction::OPEN_BRANCH_Z;
            lines[branch].x = crate::topology::reduction::OPEN_BRANCH_Z;
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
        bus_ids: network.bus_ids,
        // Empty, deliberately: the open state is already baked into
        // `lines`/`transformers` above. Leaving the file's own list here
        // would re-open branches a *closing* remedial action has just shut,
        // because `evaluate` derives its open set from this field. That is
        // silent — every margin stays self-consistent and the optimizer
        // simply measures a network in which the automaton never acted.
        initially_open: &[],
        bus_countries: network.bus_countries,
        shunts: network.shunts,
        tap_changers: network.tap_changers,
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
