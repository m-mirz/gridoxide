//! The whole optimization: perimeters, in order, each carrying the last one's
//! decisions forward.
//!
//! [`search`] answers one perimeter. This decides *which
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
use super::evaluate::{evaluate_model, evaluate_with, Network, PerimeterResult, Resolution, SecurityResult};
use super::linear::Setpoint;
use super::automaton::{simulate, AutomatonResult};
use super::mnec::Baseline;
use super::search::{objective_of, search, search_with_open, SearchOptions, SearchResult};

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
/// Which optimization steps actually ran, and how each turned out.
///
/// The reference reports this as a sentence and its Cucumber suite asserts it
/// 171 times — more than any other single step — because it is the one thing a
/// margin cannot tell you: whether the answer in front of you is the plan the
/// optimizer wanted, the plan it fell back to, or the network untouched.
///
/// The strings are the reference's own, from `OptimizationStepsExecuted`, and
/// they are reproduced verbatim rather than paraphrased: they are an interface,
/// not prose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StepsExecuted {
    /// One preventive optimization, kept.
    #[default]
    FirstPreventiveOnly,
    /// One preventive optimization, and it lost ground — so the plan was
    /// thrown away and the untouched network reported. See §8.3's defect 28.
    FirstPreventiveFellBackToInitial,
    /// A second preventive pass ran and beat the first.
    SecondPreventiveImproved,
    /// A second preventive pass ran and did not beat the first, so the first
    /// pass's answer stands. Running it and declining it is not a failure —
    /// it is the check working.
    SecondPreventiveFellBackToFirst,
    /// A second preventive pass ran and the finished plan still lost ground to
    /// doing nothing.
    SecondPreventiveFellBackToInitial,
}

impl StepsExecuted {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FirstPreventiveOnly => "The RAO only went through first preventive",
            Self::FirstPreventiveFellBackToInitial => {
                "First preventive fell back to initial situation"
            }
            Self::SecondPreventiveImproved => "Second preventive improved first preventive results",
            Self::SecondPreventiveFellBackToFirst => {
                "Second preventive fell back to first preventive results"
            }
            Self::SecondPreventiveFellBackToInitial => {
                "Second preventive fell back to initial situation"
            }
        }
    }

    /// The same step, once the plan has been thrown away.
    fn fell_back(self) -> Self {
        match self {
            Self::FirstPreventiveOnly => Self::FirstPreventiveFellBackToInitial,
            _ => Self::SecondPreventiveFellBackToInitial,
        }
    }
}

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
    /// Which optimization steps ran, and how each turned out.
    pub steps: StepsExecuted,
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

    // Monitored CNECs are judged against the margins they had before *any*
    // remedial action, preventive ones included, so the baseline is measured
    // here — once, on the untouched network — and carried into every perimeter.
    // Measuring it per perimeter would judge a curative MNEC against whatever
    // the preventive stage left it at, which is precisely the degradation the
    // constraint exists to forbid.
    let options = &with_mnec_baseline(crac, network, resolution, options);

    let preventive = search(crac, network, resolution, &preventive_states, solver, options);
    let mut plan_preventive = perimeter_plan(&preventive, preventive_states.clone());

    // A curative perimeter does not search for the best answer it can find. It
    // searches until it is **better than preventive**, by a stated margin, and
    // then stops — `TreeParameters.buildForCurativePerimeter` gives every
    // curative perimeter `AT_TARGET_OBJECTIVE_VALUE`, unconditionally, where the
    // preventive one gets it only under `SECURE_FLOW`.
    //
    // The reasoning is operational rather than mathematical. Curative actions
    // are taken under time pressure by people who did not plan them, so an
    // extra 40 A bought by a third switching operation is not worth having;
    // what matters is that the post-contingency state is no worse than the one
    // the preventive stage already accepted. `curative-min-obj-improvement`
    // says how much better than that is enough, and the reference's default is
    // **zero** — beat preventive at all and stop.
    //
    // Under `SECURE_FLOW` the target is 0 for every perimeter, so whatever the
    // caller set stands.
    let mut scenarios = scenarios_after(crac, network, resolution, solver, options, &preventive);

    // Second preventive: optimize the preventive perimeter again, this time
    // able to see what the curative stage could and could not do about the
    // constraints the first pass left it.
    // Which steps ran, narrowed as each one decides.
    let mut steps = StepsExecuted::FirstPreventiveOnly;

    // The plan as it stands, against doing nothing. Needed twice: to decide
    // whether a cost increase should trigger a second pass, and — at the very
    // end — whether the finished plan is worth keeping at all.
    let all_states = crac.states();
    let untouched = assess(crac, network, resolution, network.initially_open, options);
    let before = objective_of(crac, &untouched, &all_states, &options.linear);
    let judge = |p: &PerimeterPlan, s: &[ScenarioPlan]| {
        objective_of(
            crac,
            &final_assessment(crac, network, resolution, p, s, options),
            &all_states,
            &options.linear,
        )
    };
    let first_objective = judge(&plan_preventive, &scenarios);

    if let Some(second) = second_preventive(
        crac,
        network,
        resolution,
        solver,
        options,
        &preventive,
        &scenarios,
        first_objective < before - 1e-6,
    ) {
        let after = scenarios_after(crac, network, resolution, solver, options, &second);
        // Kept only if the whole plan it leads to is better, measured the same
        // way `postCheckResults` measures the plan against doing nothing. A
        // second preventive that improves its own perimeter and costs a
        // curative one more than it gains is not an improvement, and the first
        // pass is a perfectly good answer to fall back to.
        // Ran, and not yet kept. Running a second pass and declining it is a
        // different outcome from never running one, and the reference reports
        // the difference.
        steps = StepsExecuted::SecondPreventiveFellBackToFirst;
        let second_plan = perimeter_plan(&second, preventive_states.clone());
        if judge(&second_plan, &after) > first_objective + 1e-6 {
            steps = StepsExecuted::SecondPreventiveImproved;
            plan_preventive = second_plan;
            scenarios = after;
        }
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

    let plan = Plan {
        preventive: plan_preventive,
        scenarios,
        pulled_forward,
        initial_margin_mw: initial,
        final_margin_mw: final_margin,
        steps,
    };

    // Having done all that, check it was worth doing.
    //
    // Every perimeter only ever accepts a candidate that improves *its own*
    // objective, so it is tempting to conclude the plan cannot come out worse
    // than the network it started from. It can, because the perimeters do not
    // partition the harm. A preventive action is judged on the base case and
    // the outage states; the damage it does to a **curative** state is invisible
    // there, and by the time a curative perimeter sees it, the preventive
    // decisions are fixed and it can only make the best of them. On the
    // reference's scenario 1.4.4.2 the preventive perimeter improves itself from
    // 590.6 to 681.7 MW by closing two circuits, and those closures take the
    // curative state to −342; the curative perimeter recovers half of it and the
    // plan still ends worse than doing nothing.
    //
    // So the last thing the reference does is compare the finished plan against
    // the untouched network and, if the plan is worse, throw it away —
    // `postCheckResults`, whose `handleCostIncrease` is passed `true` at every
    // call site, so this is a rule rather than a setting. "First preventive fell
    // back to initial situation" is what its own report calls the outcome.
    //
    // Compared on the **objective**, not on the megawatt margin: under an
    // ampere objective two CNECs at different voltages order differently in the
    // two units, and a monitored CNEC's violation is part of the cost the
    // comparison is about.
    // `before` was measured above, where the same comparison decided whether a
    // second preventive pass was worth attempting.
    let after = judge(&plan.preventive, &plan.scenarios);
    if after < before - 1e-6 {
        return unoptimized(network, &plan, initial, steps.fell_back());
    }
    plan
}

/// The plan that does nothing, reported as such.
///
/// Everything the optimizer chose is dropped and every perimeter is reported at
/// the network as it arrived. The automatons go with it, which is the
/// reference's own behaviour — `UnoptimizedRaoResultImpl` wraps the result from
/// *before* they were simulated — and is worth flagging as a position rather
/// than an accident: an automaton is not a choice, so an operator reading this
/// plan is being told what the RAO decided, not what the equipment will do. No
/// vendored scenario reaches this path with an automaton present, so the
/// question has never been put.
fn unoptimized(
    network: &Network<'_>,
    plan: &Plan,
    initial: f64,
    steps: StepsExecuted,
) -> Plan {
    // `leaves` survives, alone among the fields. It is not a claim about the
    // plan — it is the count of what the search evaluated on the way to
    // deciding, and the search did evaluate them. Zeroing it would report that
    // no work was done rather than that the work was rejected, and it is the
    // one number a caller watching cost has to be able to trust.
    let bare = |states: Vec<State>, leaves: usize| PerimeterPlan {
        states,
        network_actions: Vec::new(),
        setpoints: Vec::new(),
        initial_margin_mw: initial,
        final_margin_mw: initial,
        leaves,
        open_branches: network.initially_open.to_vec(),
        buses: network.buses.to_vec(),
        transformers: network.transformers.to_vec(),
    };
    Plan {
        preventive: bare(plan.preventive.states.clone(), plan.preventive.leaves),
        scenarios: plan
            .scenarios
            .iter()
            .map(|s| ScenarioPlan {
                contingency: s.contingency,
                automatons: None,
                perimeters: s
                    .perimeters
                    .iter()
                    .map(|p| bare(p.states.clone(), p.leaves))
                    .collect(),
            })
            .collect(),
        pulled_forward: plan.pulled_forward.clone(),
        initial_margin_mw: initial,
        final_margin_mw: initial,
        steps,
    }
}

/// A search over a curative perimeter, with branches already open from earlier
/// decisions.
///
/// The set is **stated**, not applied to the network first. Writing
/// `OPEN_BRANCH_Z` into a copy of the lines and searching the result looks
/// equivalent and is not: a closing remedial action is expressed by removing a
/// branch *from the open set*, so a set that has been emptied into the
/// impedances leaves a close with nothing to remove and the branch open no
/// matter what the action says. It does not fail — it evaluates as a change
/// that does nothing, which is why the perimeter then reports that no remedial
/// action was worth taking. See [`search_with_open`] for what that cost.
fn curative_search(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    solver: &mut dyn Solver,
    options: &SearchOptions,
    already_open: &[usize],
) -> SearchResult {
    search_with_open(crac, network, resolution, perimeter, solver, options, already_open)
}

/// Evaluate one network with the flow model the run is using.
///
/// The same choice [`search`](super::search) and [`automaton`](super::automaton)
/// make, and for the same reason: what "the truth" means when an answer is
/// judged has to be the model the answer will be reported in.
fn assess(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    open: &[usize],
    options: &SearchOptions,
) -> SecurityResult {
    let ac = super::evaluate::ac_options(network);
    evaluate_model(crac, network, resolution, open, options.linear.flow_model, &ac)
}

/// The plan's own view of every state: each measured in the network *its* own
/// decisions produced.
///
/// A preventive state is read in the post-preventive network, an `auto` state
/// after that contingency's automatons, and a curative state after its curative
/// perimeter. Three different networks, and reading a curative CNEC in the
/// preventive one reports the overload the curative actions exist to remove.
fn final_assessment(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    preventive: &PerimeterPlan,
    scenarios: &[ScenarioPlan],
    options: &SearchOptions,
) -> SecurityResult {
    let mut perimeters: Vec<PerimeterResult> = Vec::new();
    let mut take = |result: SecurityResult, states: &[State]| {
        for p in result.perimeters {
            if states.contains(&p.state) && !perimeters.iter().any(|q| q.state == p.state) {
                perimeters.push(p);
            }
        }
    };

    // Most specific first, so a state a curative perimeter governs is not taken
    // from the preventive network that also reports it.
    for scenario in scenarios {
        for stage in scenario.perimeters.iter().rev() {
            let net = Network {
                buses: &stage.buses,
                transformers: &stage.transformers,
                ..*network
            };
            take(
                assess(crac, &net, resolution, &stage.open_branches, options),
                &stage.states,
            );
        }
        if let Some(automatons) = &scenario.automatons {
            let net = Network {
                buses: &preventive.buses,
                transformers: &automatons.transformers,
                ..*network
            };
            let auto_states: Vec<State> = crac
                .states()
                .into_iter()
                .filter(|s| s.contingency == Some(scenario.contingency))
                .filter(|s| crac.instants[s.instant].kind == InstantKind::Auto)
                .collect();
            take(
                assess(crac, &net, resolution, &automatons.open_branches, options),
                &auto_states,
            );
        }
    }
    let net = Network {
        buses: &preventive.buses,
        transformers: &preventive.transformers,
        ..*network
    };
    let rest = crac.states();
    take(assess(crac, &net, resolution, &preventive.open_branches, options), &rest);
    SecurityResult { perimeters, skipped: Vec::new() }
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
        .filter_map(|p| p.min_optimized_margin(crac))
        .fold(f64::INFINITY, f64::min)
}

/// `options` with the MNEC baseline filled in from the untouched network.
///
/// A no-op when the CRAC declares no monitored CNEC or the configuration
/// disabled the rule — measuring a baseline nothing reads would cost an AC
/// solve per state for nothing.
fn with_mnec_baseline(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    options: &SearchOptions,
) -> SearchOptions {
    let mut options = options.clone();
    if !options.linear.mnec.options.enabled
        || !crac.flow_cnecs.iter().any(|c| c.monitored)
        || !options.linear.mnec.baseline.is_empty()
    {
        return options;
    }
    options.linear.mnec.baseline =
        Baseline::measure(crac, network, resolution, options.linear.flow_model);
    options
}

/// Everything a preventive answer implies: the automatons it leaves to fire,
/// and the curative perimeters that follow.
///
/// Extracted so it can run **twice** — once on the first preventive result and
/// once on the second's — because the curative answer depends on the preventive
/// one and comparing two preventive answers means comparing what each leads to,
/// not just the preventive perimeter in isolation.
#[allow(clippy::too_many_arguments)]
fn scenarios_after(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    solver: &mut dyn Solver,
    options: &SearchOptions,
    preventive: &SearchResult,
) -> Vec<ScenarioPlan> {
    let states = crac.states();
    let actionable = |state: &State| -> bool {
        crac.network_actions.iter().any(|a| a.usage_rules.iter().any(|r| r.covers(state)))
            || crac.range_actions.iter().any(|a| a.usage_rules.iter().any(|r| r.covers(state)))
    };
    let carried_open = preventive.open_branches.clone();
    let carried_transformers = preventive.transformers.clone();
    let curative_target = options.stop_at_target.unwrap_or_else(|| {
        let target = preventive.final_objective + options.curative_min_obj_improvement;
        // `enforce-curative-security` additionally demands a secure perimeter,
        // which can only make the target harder to reach.
        if options.enforce_curative_security { target.max(0.0) } else { target }
    });
    // A curative perimeter gets its own depth where the configuration states
    // one. The reference keeps `max-preventive-search-tree-depth` and
    // `max-curative-search-tree-depth` apart because they answer different
    // questions — how much may be planned, against how much may be carried out
    // under time pressure — and this is the only layer that knows which
    // perimeter it is about to search.
    let options = &SearchOptions {
        stop_at_target: Some(curative_target),
        max_depth: options.curative_max_depth.unwrap_or(options.max_depth),
        ..options.clone()
    };

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
                generation: network.generation,
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
            simulate(
                crac,
                &view,
                resolution,
                contingency,
                &open,
                &transformers,
                options.linear.flow_model,
            )
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
                generation: network.generation,
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
            // The result's own set, not a union with what went in. A union
            // would be safe only while a perimeter could never *close*
            // anything: `result.open_branches` starts from `open` and the
            // search removes from it, so re-adding `open` puts back exactly the
            // branch a closing remedial action just shut.
            let in_force = result.open_branches.clone();
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
    scenarios
}

/// One perimeter's answer, as the plan reports it.
fn perimeter_plan(result: &SearchResult, states: Vec<State>) -> PerimeterPlan {
    PerimeterPlan {
        states,
        network_actions: result.network_actions.clone(),
        setpoints: result.setpoints.clone(),
        initial_margin_mw: result.initial_margin_mw,
        final_margin_mw: result.final_margin_mw,
        leaves: result.leaves,
        open_branches: result.open_branches.clone(),
        buses: result.buses.clone(),
        transformers: result.transformers.clone(),
    }
}

/// Optimize the preventive perimeter a second time, now that the curative
/// stage has shown what it can do.
///
/// # Why a second pass exists at all
///
/// The decomposition is sequential, and that is its one weakness. The first
/// preventive perimeter is judged on the base case and the outage states, so a
/// curative constraint that **no curative action can fix** is invisible to it —
/// it is not in the perimeter, and the perimeter that does contain it comes
/// later and cannot revisit preventive decisions. The result is a preventive
/// answer that spends nothing on a problem only it could have solved.
///
/// So the reference runs the preventive perimeter again with three things
/// changed, and this reproduces them:
///
/// - **Every CNEC is optimized**, whatever its state. That is the whole point:
///   the curative constraints are now in front of the preventive optimizer.
/// - **The automatons are held applied**, so it works on what they cannot fix
///   rather than re-solving what they will.
/// - **The curative decisions are held applied** too, for the same reason.
///
/// # What this does not do
///
/// The reference also **re-optimizes curative range actions** inside that same
/// problem, which needs a set-point per range action *per state* — `A(r, s)` in
/// `plans/RAO_PLAN.md` §7.3, which is declared there and not built: this LP
/// carries one set-point per action. So a shifter the CRAC allows in both
/// instants is held at whatever the curative stage chose rather than re-tuned
/// against the preventive answer, and four of the reference's fifteen
/// second-preventive scenarios turn on exactly that. They are the ones that ask
/// for one PST at two different taps.
#[allow(clippy::too_many_arguments)]
fn second_preventive(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    solver: &mut dyn Solver,
    options: &SearchOptions,
    preventive: &SearchResult,
    scenarios: &[ScenarioPlan],
    cost_increased: bool,
) -> Option<SearchResult> {
    if !options.second_preventive.runs(options, preventive, scenarios, cost_increased) {
        return None;
    }

    // The network the second pass starts from: the automatons' and the curative
    // stage's decisions in force, the preventive stage's *not*. Those are what
    // is being reconsidered.
    //
    // In force **where they are in force**, though, and not everywhere. This
    // set governs the auto and curative states only; the preventive and outage
    // ones keep the file's, because a curative decision has not been taken yet
    // when they are measured. See [`Held`] and §8.10.
    // The **delta**, not a set: what the automatons and the curative stage did
    // to the network the preventive stage handed them. A delta because the set
    // it applies to is not fixed — this pass is choosing the preventive
    // switching while the delta is in force, and every candidate it tries
    // changes what the curative states inherit. Stored as a set, a preventive
    // action the pass takes would reach the preventive states and not the
    // curative ones, and the two halves of the problem would describe different
    // networks.
    let mut opened: Vec<usize> = Vec::new();
    let mut closed: Vec<usize> = Vec::new();
    for scenario in scenarios {
        // What that scenario ends up with, after its automatons and every
        // curative perimeter in turn.
        let after = scenario
            .perimeters
            .last()
            .map(|p| &p.open_branches)
            .or(scenario.automatons.as_ref().map(|a| &a.open_branches));
        let Some(after) = after else { continue };
        opened.extend(after.iter().filter(|b| !preventive.open_branches.contains(b)).copied());
        closed.extend(preventive.open_branches.iter().filter(|b| !after.contains(b)).copied());
    }
    // One network stands in for every contingency, so two scenarios can
    // disagree about a branch. An open wins, as it does everywhere else here:
    // it is the status quo the file states, and holding a branch closed for a
    // contingency whose curative stage did not close it would invent capacity.
    closed.retain(|b| !opened.contains(b));
    opened.sort_unstable();
    opened.dedup();
    closed.sort_unstable();
    closed.dedup();
    let held = super::evaluate::Held {
        open: opened,
        close: closed,
        // Filled in by the LP as it chooses them: a curative shifter's position
        // is a variable of the second preventive problem, not an input to it.
        taps: Vec::new(),
        // The states that see it: everything at or after the automatons, which
        // is where a curative decision has been taken. An **outage** state is
        // not one of them — it is over before an automaton fires — and neither
        // is the preventive state.
        states: crac
            .states()
            .into_iter()
            .filter(|s| {
                matches!(crac.instants[s.instant].kind, InstantKind::Auto | InstantKind::Curative)
            })
            .collect(),
    };
    // A curative **shifter**, by contrast, goes back to where the file had it.
    //
    // The asymmetry with the switching above is not an oversight. This is one
    // network standing in for every state, so anything held in it is held in
    // the preventive and outage states too, where a curative decision is not in
    // force. For a switch that is the price of letting the second pass see the
    // curative CNECs at all — without it the pass reads overloads the curative
    // stage has already removed and overspends preventively to fix them again.
    // A set-point is different in kind: it is a **stale iterate**. It was
    // chosen against the first pass's preventive decisions, which this pass
    // exists to discard, and `scenarios_after` recomputes it from scratch the
    // moment this pass returns. Pinning the search to a number that is about to
    // be thrown away is what lets the two passes settle into a fixed point that
    // neither can leave: 1.4.4.4's curative `pst_be` at −16 caps the second
    // pass's whole landscape at 553 A, so it takes `close_fr1_fr5` and a
    // preventive shifter at its bound for 645 A, where releasing the set-point
    // lets it find 798 A with no network action at all — the reference's own
    // answer. `plans/RAO_PLAN.md` §8.8 has the measurement.
    //
    // Curative redispatch was never held: `buses` comes from `network`
    // untouched. This makes the shifter agree with it. Nothing but the held
    // switching now separates the second pass's network from the file's, so it
    // is handed the network itself.

    // Every CNEC, but still only the preventive perimeter's actions.
    let all = crac.states();
    let preventive_states: Vec<State> = all
        .iter()
        .filter(|s| crac.instants[s.instant].kind == InstantKind::Preventive)
        .cloned()
        .collect();
    let mut linear = options.linear.clone();
    linear.held = Some(held);
    // The curative shifters get a column of their own here and nowhere else.
    // The reference gates this on `re-optimize-curative-range-actions`, which no
    // vendored configuration sets; what the corpus shows is that its answers
    // need it regardless — 1.4.1.2 asserts a preventive tap of 4 that is only
    // reachable if the pass can see the curative `pst_be` moving to −5 to pay
    // for it. See §8.11.
    linear.a_r_s = true;
    let options = SearchOptions {
        linear,
        available_at: Some(preventive_states),
        // The reference's `hint-from-first-preventive-rao`: offer the first
        // pass's winning set as one candidate, so a second pass that agrees
        // with it gets there at the first depth instead of rediscovering it.
        predefined_combinations: if options.second_preventive.hint {
            let hint: Vec<String> = preventive
                .network_actions
                .iter()
                .map(|&a| crac.network_actions[a].id.clone())
                .collect();
            let mut combos = options.predefined_combinations.clone();
            if hint.len() > 1 {
                combos.push(hint);
            }
            combos
        } else {
            options.predefined_combinations.clone()
        },
        ..options.clone()
    };
    // The search starts from the **file's** open set, not the held one: what it
    // returns is a preventive answer, and a preventive answer may not contain
    // one contingency's curative switching — that would put it into force for
    // every other contingency, which is the one thing the perimeter
    // decomposition exists to prevent. The held set reaches the measurements
    // through `LinearOptions::held`, which is where it belongs, so there is
    // nothing to strip back out afterwards.
    let result = search_with_open(
        crac,
        network,
        resolution,
        &all,
        solver,
        &options,
        network.initially_open,
    );
    Some(result)
}

