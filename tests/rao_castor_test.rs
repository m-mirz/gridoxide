//! Perimeters, in order — phase 9 of `plans/RAO_PLAN.md`.
//!
//! [`search`](gridoxide::rao::search) answers one perimeter. This layer decides
//! which perimeters there are and in what order, which is the difference
//! between a set of independent answers and a plan.
//!
//! Two properties carry most of the weight. The **preventive perimeter covers
//! the outage states too**, because an outage instant is too soon for anyone to
//! act — whatever protects it must already have been done. And a curative CNEC
//! with **no curative action** is pulled forward into that perimeter, or the
//! preventive optimization is free to park a flow between the PATL and the TATL:
//! acceptable at the outage instant, permanently overloaded afterwards, with
//! nothing available to fix it.

use std::path::PathBuf;

use gridoxide::opf::ipm::IpmSolver;
use gridoxide::rao::crac::*;
use gridoxide::rao::{crac_json, evaluate_with, run, Network, Resolution, SearchOptions};
use gridoxide::ucte;

fn ucte_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte").join(name)
}

fn rao_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/rao").join(name)
}

struct Case {
    net: ucte::UcteImport,
    crac: Crac,
}

fn case(crac_name: &str) -> Case {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(rao_fixture(crac_name)).expect("crac");
    Case { net, crac }
}

impl Case {
    fn resolution(&self) -> Resolution {
        Resolution::new(&self.crac, &self.net.branch_ids)
    }

    fn network(&self) -> Network<'_> {
        Network {
            buses: &self.net.buses,
            lines: &self.net.lines,
            transformers: &self.net.transformers,
            branch_ids: &self.net.branch_ids,
            bus_ids: &[],
            initially_open: &[],
            bus_countries: &[],
            shunts: &[],
            tap_changers: &[],
            base_mva: self.net.base_mva,
        }
    }

    fn plan(&self) -> gridoxide::rao::Plan {
        let mut solver = IpmSolver::new();
        run(&self.crac, &self.network(), &self.resolution(), &mut solver, &SearchOptions::default())
    }
}

// ---------------------------------------------------------------------------
// The decomposition
// ---------------------------------------------------------------------------

#[test]
fn the_preventive_perimeter_covers_the_outage_states_too() {
    // An outage instant is too soon for anyone to act, so it is secured by
    // preventive actions or not at all. Optimizing it as its own perimeter
    // would offer it actions nobody could take in time.
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();

    let kinds: Vec<InstantKind> =
        plan.preventive.states.iter().map(|s| c.crac.instants[s.instant].kind).collect();
    assert!(kinds.contains(&InstantKind::Preventive), "{kinds:?}");
    assert!(kinds.contains(&InstantKind::Outage), "the outage state must be secured here: {kinds:?}");

    // And no outage state may appear as a curative perimeter of its own.
    for scenario in &plan.scenarios {
        for perimeter in &scenario.perimeters {
            for state in &perimeter.states {
                assert_ne!(c.crac.instants[state.instant].kind, InstantKind::Outage);
            }
        }
    }
}

/// The one rule in this layer that changes an answer rather than organising the
/// work.
#[test]
fn a_curative_cnec_with_no_curative_action_is_pulled_forward() {
    let c = case("crac-topology-helps.json");
    let plan = c.plan();

    assert!(!plan.pulled_forward.is_empty(), "this fixture has unactionable curative CNECs");
    for &index in &plan.pulled_forward {
        let cnec = &c.crac.flow_cnecs[index];
        let kind = c.crac.instants[cnec.state.instant].kind;
        assert!(matches!(kind, InstantKind::Curative | InstantKind::Auto), "{kind:?}");
        // Nothing may be available for it, or pulling it forward would be wrong.
        let actionable = c
            .crac
            .network_actions
            .iter()
            .any(|a| a.usage_rules.iter().any(|r| r.covers(&cnec.state)))
            || c.crac
                .range_actions
                .iter()
                .any(|a| a.usage_rules.iter().any(|r| r.covers(&cnec.state)));
        assert!(!actionable, "`{}` has a curative action and should not be pulled forward", cnec.id);
        // And its state must be in the preventive perimeter.
        assert!(plan.preventive.states.contains(&cnec.state), "{:?}", plan.preventive.states);
    }
}

#[test]
fn an_actionable_curative_state_gets_its_own_perimeter() {
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();
    // This CRAC has a curative network action and an auto range action, so both
    // states are actionable and neither is pulled forward.
    let curative_states: Vec<&State> =
        plan.scenarios.iter().flat_map(|s| s.perimeters.iter()).flat_map(|p| &p.states).collect();
    assert!(!curative_states.is_empty(), "no curative perimeter was created");
    for state in &curative_states {
        assert!(
            !plan.preventive.states.contains(state),
            "state {state:?} is in both the preventive and a curative perimeter"
        );
    }
}

#[test]
fn curative_perimeters_are_solved_in_chronological_order() {
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();
    for scenario in &plan.scenarios {
        let instants: Vec<usize> =
            scenario.perimeters.iter().filter_map(|p| p.states.first().map(|s| s.instant)).collect();
        assert!(
            instants.windows(2).all(|w| w[0] < w[1]),
            "curative instants out of order: {instants:?}"
        );
    }
}

#[test]
fn each_contingency_gets_its_own_scenario() {
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();
    let mut seen: Vec<usize> = plan.scenarios.iter().map(|s| s.contingency).collect();
    let before = seen.len();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), before, "a contingency appeared twice");
    for &contingency in &seen {
        assert!(contingency < c.crac.contingencies.len());
    }
}

// ---------------------------------------------------------------------------
// Carrying decisions forward
// ---------------------------------------------------------------------------

/// A curative perimeter must start from the network the preventive decisions
/// left behind.
///
/// Solving it against the untouched network would optimize a situation that
/// never occurs, and the two answers would be combined into a plan neither
/// describes.
#[test]
fn a_curative_perimeter_starts_from_the_preventive_result() {
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();
    assert!(
        !plan.preventive.network_actions.is_empty()
            || plan.preventive.setpoints.iter().any(|s| s.moved()),
        "this test needs the preventive perimeter to have done something"
    );

    let scenario = plan.scenarios.first().expect("a scenario");
    let perimeter = scenario.perimeters.last().expect("a curative perimeter");
    let state = perimeter.states.first().expect("a state");

    // The same state, measured against the *untouched* network.
    let untouched = evaluate_with(&c.crac, &c.network(), &c.resolution(), &[])
        .perimeters
        .iter()
        .find(|p| p.state == *state)
        .and_then(|p| p.min_margin())
        .expect("perimeter");

    assert!(
        (perimeter.initial_margin_mw - untouched).abs() > 1e-6,
        "the curative perimeter started at {} MW, the same as the untouched network — \
         the preventive decisions were not carried forward",
        perimeter.initial_margin_mw
    );
}

#[test]
fn the_plans_worst_margin_is_the_worst_of_its_perimeters() {
    for name in ["crac-for-12nodes.json", "crac-topology-helps.json"] {
        let c = case(name);
        let plan = c.plan();
        // Automatons too, not just the perimeters. They are a stage of the
        // plan and can carry its worst margin — on this fixture the automaton
        // stage is 2.4e-6 MW below every perimeter — but they are a different
        // type, so `perimeters()` cannot enumerate them and folding only over
        // it would assert an identity that holds by luck.
        let worst = plan
            .perimeters()
            .map(|p| p.final_margin_mw)
            .chain(
                plan.scenarios
                    .iter()
                    .filter_map(|s| s.automatons.as_ref())
                    .map(|a| a.final_margin_mw),
            )
            .fold(f64::INFINITY, f64::min);
        assert!(
            (plan.final_margin_mw - worst).abs() < 1e-9,
            "{name}: plan says {}, perimeters say {worst}",
            plan.final_margin_mw
        );
        assert_eq!(plan.is_secure(), plan.final_margin_mw >= 0.0);
    }
}

#[test]
fn the_plan_never_makes_things_worse_than_doing_nothing() {
    // Every perimeter keeps its starting point unless something beats it, so
    // the plan as a whole cannot be worse than the untouched network.
    for name in ["crac-for-12nodes.json", "crac-topology-helps.json"] {
        let c = case(name);
        let plan = c.plan();
        assert!(
            plan.final_margin_mw >= plan.initial_margin_mw - 1e-9,
            "{name}: {} -> {}",
            plan.initial_margin_mw,
            plan.final_margin_mw
        );
    }
}

#[test]
fn the_topology_fixture_ends_secure() {
    let c = case("crac-topology-helps.json");
    let plan = c.plan();
    assert!(plan.initial_margin_mw < 0.0, "it should start insecure");
    assert!(plan.is_secure(), "one action secures it: {} MW", plan.final_margin_mw);
    assert_eq!(plan.preventive.network_actions.len(), 1);
}

#[test]
fn the_plan_reproduces_itself() {
    let c = case("crac-for-12nodes.json");
    let a = c.plan();
    let b = c.plan();
    assert_eq!(a.preventive.network_actions, b.preventive.network_actions);
    assert_eq!(a.pulled_forward, b.pulled_forward);
    assert!((a.final_margin_mw - b.final_margin_mw).abs() < 1e-12);
    assert_eq!(a.scenarios.len(), b.scenarios.len());
}

#[test]
fn a_crac_with_no_contingency_still_yields_a_preventive_plan() {
    let c = case("crac-for-12nodes.json");
    let mut crac = c.crac.clone();
    crac.flow_cnecs.retain(|x| x.state.is_preventive());
    crac.contingencies.clear();
    let resolution = Resolution::new(&crac, &c.net.branch_ids);
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &c.network(), &resolution, &mut solver, &SearchOptions::default());
    assert!(plan.scenarios.is_empty());
    assert!(!plan.preventive.states.is_empty());
    assert!(plan.final_margin_mw >= plan.initial_margin_mw - 1e-9);
}

/// Depth 0 exercises the decomposition without the search.
#[test]
fn depth_zero_still_decomposes_even_when_it_finds_nothing() {
    let c = case("crac-for-12nodes.json");
    let mut solver = IpmSolver::new();
    let plan = run(
        &c.crac,
        &c.network(),
        &c.resolution(),
        &mut solver,
        &SearchOptions { max_depth: 0, ..Default::default() },
    );
    assert!(plan.preventive.network_actions.is_empty());
    assert!(plan.preventive.states.len() >= 2, "the decomposition should still happen");
    assert!(plan.final_margin_mw >= plan.initial_margin_mw - 1e-9);
}

/// The case for interleaving the two halves, in data.
///
/// On this fixture, over the preventive perimeter (base case *and* outage
/// state):
///
/// | | worst margin | gain |
/// |---|---|---|
/// | do nothing | −182.3 MW | |
/// | open NL1-NL2, shifter untouched | −182.6 MW | **−0.3 — worse** |
/// | optimize the shifter, topology untouched | −179.2 MW | +3.1 |
/// | both, the shifter re-optimized under the new topology | **−132.7 MW** | **+49.6** |
///
/// The combination is worth sixteen times what either half manages alone, and
/// the action *on its own is harmful*. A design that chose the topology first
/// and the set-points afterwards would evaluate the line opening at −182.6,
/// reject it, and stop with the shifter's 3.1 MW. Only re-running the linear
/// optimization inside each leaf finds the 49.6, which is why that is where all
/// the time goes.
///
/// The 49.6 is itself capped by this CRAC's five **MNECs** on `NL2-BE3`, which
/// the shifter loads as it unloads everything else. Told to ignore them
/// (`mnec.options.enabled = false`) the same search takes the shifter all the
/// way to tap −16 and reports −82.9 MW, a gain of +99.4 — and leaves two
/// monitored branches below the floor they are owed. Half the achievable margin
/// is what this constraint costs on this fixture, which is a fair statement of
/// why it is a *soft* constraint and not a hard one.
#[test]
fn an_action_and_a_setpoint_that_help_only_together_are_both_found() {
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();

    // Both halves are in the answer.
    assert!(!plan.preventive.network_actions.is_empty(), "no network action taken");
    assert!(
        plan.preventive.setpoints.iter().any(|s| s.moved()),
        "no range action moved"
    );
    assert!(
        plan.preventive.improvement() > 25.0,
        "the combination should improve substantially, got {:+.1}",
        plan.preventive.improvement()
    );

    // And neither half achieves that on its own.
    let mut solver = IpmSolver::new();
    let range_only = run(
        &c.crac,
        &c.network(),
        &c.resolution(),
        &mut solver,
        &SearchOptions { max_depth: 0, ..Default::default() },
    );
    // Not "nothing" — a little. The claim is proportion: whatever the shifter
    // manages by itself is a rounding error beside the combination.
    assert!(
        range_only.preventive.improvement() * 10.0 < plan.preventive.improvement(),
        "the shifter alone gained {:+.1}, the combination {:+.1} — too close to \
         demonstrate that interleaving matters",
        range_only.preventive.improvement(),
        plan.preventive.improvement()
    );

    let open: Vec<usize> = plan
        .preventive
        .network_actions
        .iter()
        .flat_map(|&a| c.crac.network_actions[a].elementary.iter())
        .flat_map(|e| e.elements())
        .filter_map(|e| c.resolution().branch(e))
        .collect();
    let topology_only = evaluate_with(&c.crac, &c.network(), &c.resolution(), &open)
        .perimeters
        .iter()
        .filter(|p| plan.preventive.states.contains(&p.state))
        .filter_map(|p| p.min_optimized_margin(&c.crac))
        .fold(f64::INFINITY, f64::min);
    assert!(
        topology_only < plan.initial_margin_mw,
        "the action alone should make things worse, got {topology_only} from {}",
        plan.initial_margin_mw
    );
}

/// The soft constraint binds, and it binds *softly*.
///
/// The same search, on the same fixture, told to honour the CRAC's five MNECs
/// and told to ignore them. Ignoring them is worth another 43 MW of margin and
/// costs two monitored branches their floor, which is the trade the rule
/// exists to refuse.
///
/// The floor is `min(0, m₀ − 50)`, not zero: these MNECs start at 42.4 and
/// 28.6 MW, inside the acceptable decrease, so they are allowed to go slightly
/// negative — and do, to −4.1 and −18.0 MW. A test that asserted "no MNEC ends
/// up negative" would pass for the wrong reason on a fixture where they start
/// comfortable, and fail here. See [`gridoxide::rao::mnec`].
#[test]
fn monitored_cnecs_hold_the_optimizer_back_to_the_floor_they_are_owed() {
    let c = case("crac-for-12nodes.json");
    let monitored: Vec<usize> = c
        .crac
        .flow_cnecs
        .iter()
        .enumerate()
        .filter(|(_, x)| x.monitored)
        .map(|(i, _)| i)
        .collect();
    assert!(!monitored.is_empty(), "this fixture is supposed to declare MNECs");

    let mut unconstrained = SearchOptions::default();
    unconstrained.linear.mnec.options.enabled = false;
    let mut solver = IpmSolver::new();
    let loose = run(&c.crac, &c.network(), &c.resolution(), &mut solver, &unconstrained);
    let held = c.plan();

    assert!(
        held.preventive.improvement() < loose.preventive.improvement() - 10.0,
        "the constraint should cost real margin, held {:+.1} vs loose {:+.1}",
        held.preventive.improvement(),
        loose.preventive.improvement()
    );

    // Every MNEC's floor, measured against the untouched network, and where
    // each of the two answers left it.
    let before = evaluate_with(&c.crac, &c.network(), &c.resolution(), &[]);
    let floor = |cnec: usize| {
        let initial = before
            .perimeters
            .iter()
            .flat_map(|p| p.cnecs.iter())
            .find(|x| x.cnec == cnec)
            .expect("every MNEC is evaluated initially")
            .margin_mw;
        f64::min(0.0, initial - 50.0)
    };
    let worst_shortfall = |plan: &gridoxide::rao::Plan| {
        let after_network = Network {
            buses: &plan.preventive.buses,
            transformers: &plan.preventive.transformers,
            ..c.network()
        };
        let after = evaluate_with(
            &c.crac,
            &after_network,
            &c.resolution(),
            &plan.preventive.open_branches,
        );
        after
            .perimeters
            .iter()
            .flat_map(|p| p.cnecs.iter())
            .filter(|x| c.crac.flow_cnecs[x.cnec].monitored)
            .map(|x| x.margin_mw - floor(x.cnec))
            .fold(f64::INFINITY, f64::min)
    };

    assert!(
        worst_shortfall(&held) >= -1e-6,
        "an MNEC was pushed below its floor by {:.2} MW",
        -worst_shortfall(&held)
    );
    assert!(
        worst_shortfall(&loose) < -1.0,
        "with the rule off the search should breach a floor, worst was {:+.2} MW",
        worst_shortfall(&loose)
    );
}

/// A usage limit binds the *plan*, and both halves of the search spend it.
///
/// This CRAC lets `be` use one topological action, one shifter, and two
/// remedial actions in total, in the curative instant — and every action in it
/// is operated by `be`. So "two actions" and "one of each" are the same
/// sentence here, which is what makes it a clean test: a search that counted
/// only network actions would happily open a line and move both shifters, and
/// every margin it then reported would be for a plan the CRAC forbids.
///
/// Asserted as a property of the answer rather than against a fixed action
/// list, because which two actions are best is exactly the kind of thing two
/// heuristic searches may legitimately disagree about. What they may not
/// disagree about is how many.
#[test]
fn a_curative_perimeter_may_not_exceed_the_crac_s_usage_limits() {
    use gridoxide::rao::crac::RangeActionKind;

    let net = ucte::read(ucte_fixture("TestCase16Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep19us3case1.json"),
    )
    .expect("crac");

    let curative = crac
        .usage_limits
        .iter()
        .find(|l| crac.instants[l.instant].kind == InstantKind::Curative)
        .expect("this fixture declares curative usage limits");
    assert_eq!(curative.max_topo_per_tso.get("be"), Some(&1));
    assert_eq!(curative.max_pst_per_tso.get("be"), Some(&1));
    assert_eq!(curative.max_ra_per_tso.get("be"), Some(&2));

    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &[],
        bus_countries: &net.bus_countries,
        shunts: &net.shunts,
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &network, &resolution, &mut solver, &SearchOptions::default());

    let mut checked = 0;
    let mut most = 0;
    for scenario in &plan.scenarios {
        for perimeter in &scenario.perimeters {
            if !perimeter.states.iter().any(|s| s.instant == curative.instant) {
                continue;
            }
            checked += 1;
            let topo = perimeter
                .network_actions
                .iter()
                .filter(|&&i| crac.network_actions[i].operator.as_deref() == Some("be"))
                .count();
            let moved: Vec<&gridoxide::rao::RangeAction> = perimeter
                .setpoints
                .iter()
                .filter(|s| s.moved())
                .map(|s| &crac.range_actions[s.action])
                .filter(|a| a.operator.as_deref() == Some("be"))
                .collect();
            let psts =
                moved.iter().filter(|a| matches!(a.kind, RangeActionKind::Pst { .. })).count();
            assert!(topo <= 1, "{topo} BE topological actions, the cap is 1");
            assert!(psts <= 1, "{psts} BE shifters moved, the cap is 1");
            assert!(topo + moved.len() <= 2, "{} BE actions, the cap is 2", topo + moved.len());
            most = most.max(topo + moved.len());
        }
    }
    assert!(checked > 0, "no curative perimeter was examined, so nothing was tested");
    // …and the caps are not being satisfied by a search that simply does
    // nothing. Some perimeter spends the whole allowance, so the assertions
    // above are on a plan that is actually pressing against them.
    assert_eq!(most, 2, "no perimeter used its full allowance, so nothing was constrained");
}

/// A curative perimeter stops once it beats the preventive one, and stopping
/// means stopping — not even its range actions move.
///
/// This is the reference's rule and it is not an optimization: a curative
/// perimeter is given `AT_TARGET_OBJECTIVE_VALUE` unconditionally, where the
/// preventive one gets it only under `SECURE_FLOW`. Curative actions are taken
/// under time pressure by people who did not plan them, so the question is not
/// "what is the best post-contingency state" but "is it at least as good as the
/// one preventive already accepted".
///
/// `curative_min_obj_improvement` moves that line, and the two ends of it are
/// what this checks: at zero the search stops as early as it is allowed to, and
/// at a target nothing can reach it runs to full depth on the same fixture.
#[test]
fn a_curative_perimeter_stops_once_it_is_better_than_preventive() {
    let net = ucte::read(ucte_fixture("TestCase16Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep13us11case1.json"),
    )
    .expect("crac");
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &[],
        bus_countries: &net.bus_countries,
        shunts: &net.shunts,
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let plan_with = |improvement: f64| {
        let mut solver = IpmSolver::new();
        run(
            &crac,
            &network,
            &resolution,
            &mut solver,
            &SearchOptions {
                curative_min_obj_improvement: improvement,
                ..Default::default()
            },
        )
    };
    let curative_actions = |plan: &gridoxide::rao::Plan| -> usize {
        plan.scenarios
            .iter()
            .flat_map(|s| s.perimeters.iter())
            .map(|p| {
                p.network_actions.len() + p.setpoints.iter().filter(|s| s.moved()).count()
            })
            .sum()
    };

    // Beat preventive at all and stop. The reference's own default.
    let eager = plan_with(0.0);
    // A target no curative perimeter can reach, so the search runs out its
    // depth instead. The reference's `RaoParameters_maxMargin_ampere.json` uses
    // exactly this value for exactly this reason.
    let thorough = plan_with(10_000.0);

    assert_eq!(
        curative_actions(&eager),
        0,
        "this fixture's curative perimeters already beat preventive, so nothing should be done"
    );
    assert!(
        curative_actions(&thorough) > 0,
        "with an unreachable target the same perimeters should act"
    );
    assert!(
        thorough.final_margin_mw >= eager.final_margin_mw - 1e-9,
        "stopping early cannot beat searching on"
    );
}

// ---------------------------------------------------------------------------
// Automatons
// ---------------------------------------------------------------------------

/// An automaton fires because its condition is met, not because it helps.
///
/// That is the whole difference between this and the search: the search asks
/// which action would be best, and the simulator asks which will actually
/// operate. A scheme that makes the objective worse still operates.
#[test]
fn automatons_are_simulated_rather_than_chosen() {
    use gridoxide::rao::crac_json;

    let net = ucte::read(ucte_fixture("TestCase8Nodes_15_11_6_1.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/crac_15_11_6_1.json"),
    )
    .expect("crac");
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &[],
        shunts: &[],
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &network, &resolution, &mut solver, &SearchOptions::default());

    let scenario = plan.scenarios.first().expect("a scenario");
    let automatons = scenario.automatons.as_ref().expect("an auto perimeter");

    // Four of the five available automatons operate.
    assert_eq!(automatons.fired(), 4, "{automatons:?}");
    assert_eq!(automatons.network_actions.len(), 2);
    assert_eq!(automatons.range_actions.len(), 2);

    // And the fifth does not, because an earlier one already relieved the
    // constraint that would have triggered it. That is the property the whole
    // speed-ordered, re-evaluate-between-batches structure exists for.
    let fired: Vec<&str> = automatons
        .network_actions
        .iter()
        .map(|&i| crac.network_actions[i].id.as_str())
        .collect();
    assert!(fired.contains(&"close_fr1_fr2_3"), "{fired:?}");
    assert!(
        !fired.contains(&"close_fr1_fr2_4"),
        "the second circuit should not close: the first already fixed it"
    );
    assert!(fired.contains(&"close_fr5_fr6_2"), "{fired:?}");

    // The phase shifters are sized against the CNECs their own rules name.
    let taps: Vec<(&str, Option<i32>)> = automatons
        .range_actions
        .iter()
        .map(|(i, _, tap)| (crac.range_actions[*i].id.as_str(), *tap))
        .collect();
    assert!(taps.contains(&("pst_fr3_fr4", Some(2))), "{taps:?}");
    assert!(taps.contains(&("pst_fr7_fr8", Some(-3))), "{taps:?}");

    // The perimeter ends secure, having started well overloaded.
    assert!(automatons.initial_margin_mw < -100.0);
    assert!(automatons.final_margin_mw > 0.0, "{}", automatons.final_margin_mw);
}

/// A standby circuit must survive import, or closing it is inexpressible.
#[test]
fn an_out_of_service_branch_can_be_closed_by_an_automaton() {
    let net = ucte::read(ucte_fixture("TestCase8Nodes_15_11_6_1.uct")).expect("network");
    assert!(!net.initially_open.is_empty(), "this fixture has out-of-service circuits");
    for &branch in &net.initially_open {
        // Present in the model with real impedance — that is what makes it
        // closable — and open only by virtue of being in this list.
        assert!(branch < net.n_branches());
        if branch < net.lines.len() {
            assert!(net.lines[branch].x.is_finite() && net.lines[branch].x < 1e6);
        }
    }
}

/// Automatons run before the curative perimeters and hand them their result.
#[test]
fn curative_perimeters_start_from_what_the_automatons_left() {
    use gridoxide::rao::crac_json;

    let net = ucte::read(ucte_fixture("TestCase12Nodes_15_11_5_1.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/crac_15_11_5_1.json"),
    )
    .expect("crac");
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &[],
        shunts: &[],
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &network, &resolution, &mut solver, &SearchOptions::default());

    for scenario in &plan.scenarios {
        let Some(automatons) = &scenario.automatons else { continue };
        for perimeter in &scenario.perimeters {
            // Whatever the automatons opened is still open in the curative
            // perimeter: a curative decision adds to their result, it does not
            // replace it.
            for branch in &automatons.open_branches {
                assert!(
                    perimeter.open_branches.contains(branch),
                    "curative perimeter lost the automatons' switching"
                );
            }
        }
    }
}

/// A **curative** action may close a branch the network file has out of service.
///
/// This is the one the gate found the hard way. The already-open branches a
/// curative perimeter inherits used to be applied by writing `OPEN_BRANCH_Z`
/// into a copy of the lines, and a close is expressed by *removing* a branch
/// from the open set — so once the set had been spent on the impedances there
/// was nothing left to remove, and the branch stayed open however the CRAC read.
///
/// The failure was silent in the worst way: the action was not refused, it
/// evaluated as a change that does nothing, lost to every other candidate, and
/// the perimeter reported that no remedial action was worth taking. Against the
/// reference's own suite that was 21 curative closes declined out of 21
/// offered, while the same actions matched 8 of 8 in preventive and 3 of 3 in
/// auto.
///
/// `close_fr1_fr5` on the scenario the reference numbers 1.3.3.4.
#[test]
fn a_curative_action_can_close_an_out_of_service_branch() {
    let net = ucte::read(ucte_fixture("TestCase16Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep13us3case4.json"),
    )
    .expect("crac");

    let closes = crac
        .network_actions
        .iter()
        .position(|a| a.id == "close_fr1_fr5")
        .expect("the CRAC declares close_fr1_fr5");
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let branch = match &crac.network_actions[closes].elementary[..] {
        [ElementaryAction::TerminalsConnection { element, connected: true }] => {
            resolution.branch(element).expect("the branch resolves")
        }
        other => panic!("expected one closing elementary action, got {other:?}"),
    };
    assert!(
        net.initially_open.contains(&branch),
        "this test needs a branch the file has out of service — otherwise there is \
         nothing to close and the assertion below passes for the wrong reason"
    );

    let network = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &[],
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &network, &resolution, &mut solver, &SearchOptions::default());

    let taken = plan
        .scenarios
        .iter()
        .flat_map(|s| s.perimeters.iter())
        .find(|p| p.network_actions.contains(&closes))
        .expect("some curative perimeter closes fr1-fr5");
    // Not just chosen — actually in force. The open set the perimeter reports
    // is what every downstream margin is measured against, so an action that is
    // recorded and not applied is the same defect one step later.
    assert!(
        !taken.open_branches.contains(&branch),
        "close_fr1_fr5 was chosen but the branch is still in the perimeter's open set"
    );
    assert!(
        plan.preventive.open_branches.contains(&branch),
        "the branch should still be open in the preventive perimeter, which does not close it"
    );
}

/// A curative perimeter gets its own search depth.
///
/// The reference keeps `max-preventive-search-tree-depth` and
/// `max-curative-search-tree-depth` apart because they answer different
/// questions: how much may be planned in advance, against how much may be
/// carried out under time pressure by people who did not plan it. Every
/// vendored configuration sets the two the same, so this moves no assertion in
/// the Cucumber gate — and a configuration that set them differently would
/// otherwise have been scored against the preventive depth without a word,
/// which is the only reason to have it.
#[test]
fn a_curative_perimeter_has_its_own_depth() {
    let c = case("crac-for-12nodes.json");
    let curative_leaves = |options: &SearchOptions| -> usize {
        let mut solver = IpmSolver::new();
        run(&c.crac, &c.network(), &c.resolution(), &mut solver, options)
            .scenarios
            .iter()
            .flat_map(|s| s.perimeters.iter())
            .map(|p| p.leaves)
            .sum()
    };

    // Depth 0 evaluates the root and nothing else, so a curative perimeter held
    // at zero must not evaluate a single leaf however deep preventive goes.
    let shared = SearchOptions {
        max_depth: 1,
        // Take whatever scores best, so the count reflects the depth rather
        // than a threshold.
        absolute_min_impact: -1e9,
        relative_min_impact: -1e9,
        // A curative perimeter stops the moment it beats the preventive one,
        // and these already do — with the default improvement of zero both
        // settings would read as zero leaves for that reason rather than for
        // the depth. Demanding an improvement nothing can reach keeps the
        // search looking, which is what makes the depth observable.
        curative_min_obj_improvement: 1e9,
        ..Default::default()
    };
    let together = curative_leaves(&shared);
    let apart =
        curative_leaves(&SearchOptions { curative_max_depth: Some(0), ..shared.clone() });

    assert!(together > 0, "the curative perimeters should evaluate something at depth 1");
    assert_eq!(apart, 0, "held at curative depth 0, no curative leaf should be evaluated");
}

/// A `relativeToPreviousInstant` range is anchored on where the **perimeter
/// began**, not on tap zero.
///
/// The reference's scenario 1.3.4.3, reduced to the property. `pst_fr` sits at
/// tap 5 in the network, is available only in curative, and carries a single
/// range of ±10 *relative to the previous instant*. The preventive perimeter
/// moves nothing, so the previous instant is tap 5 and the curative box is
/// [−5, 15]. Read as absolute — which is what an unhandled range kind
/// degenerates to — the box is [−10, 10], and the optimizer stops at 10 with
/// five taps of permitted travel it never knew it had.
///
/// Nothing about that is visible in a margin: the answer stays feasible,
/// self-consistent and worse, which is why it survived nineteen other defects.
#[test]
fn a_relative_curative_range_is_anchored_on_the_preventive_result() {
    let net = ucte::read(ucte_fixture("TestCase16Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep13us4case3.json"),
    )
    .expect("crac");

    let pst = crac
        .range_actions
        .iter()
        .position(|r| r.id == "pst_fr")
        .expect("the CRAC declares pst_fr");
    let RangeActionKind::Pst { initial_tap, .. } = &crac.range_actions[pst].kind else {
        panic!("pst_fr should be a phase shifter");
    };
    assert_eq!(*initial_tap, 5, "the fixture's anchor");
    assert!(
        crac.range_actions[pst]
            .ranges
            .iter()
            .all(|r| r.kind == RangeKind::RelativeToPreviousInstant),
        "this test needs the range to be relative to the previous instant"
    );

    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &[],
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &network, &resolution, &mut solver, &SearchOptions::default());

    // Preventive leaves it where the file had it, so the anchor is 5.
    assert!(
        plan.preventive.setpoints.iter().all(|s| s.action != pst || !s.moved()),
        "the preventive perimeter should not move pst_fr — it is curative-only here"
    );

    let tap = plan
        .scenarios
        .iter()
        .flat_map(|s| s.perimeters.iter())
        .flat_map(|p| p.setpoints.iter())
        .find(|s| s.action == pst)
        .and_then(|s| s.tap)
        .expect("the curative perimeter should optimize pst_fr");

    assert!(
        (-5..=15).contains(&tap),
        "tap {tap} is outside the box the previous instant anchors, [-5, 15]"
    );
    assert_eq!(tap, 15, "the reference reaches the top of that box");
}

/// The three range kinds are **intersected**, and they disagree on purpose.
///
/// `SL_ep13us5case3` declares all three on one shifter: absolute [−16, 16],
/// ±10 of the network as imported (tap 5, so [−5, 15]), and ±10 of the previous
/// instant. The preventive perimeter moves it to −5, so the last is [−15, 5]
/// and the intersection is [−5, 5] — narrower than any of the three alone, and
/// narrower than it would be if the two relative kinds shared an anchor.
#[test]
fn the_range_kinds_intersect_rather_than_the_last_one_winning() {
    let net = ucte::read(ucte_fixture("TestCase16Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep13us5case3.json"),
    )
    .expect("crac");
    let pst = crac.range_actions.iter().position(|r| r.id == "pst_fr").expect("pst_fr");
    let kinds: Vec<RangeKind> = crac.range_actions[pst].ranges.iter().map(|r| r.kind).collect();
    assert_eq!(kinds.len(), 3, "this test needs all three kinds on one action");

    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &[],
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &network, &resolution, &mut solver, &SearchOptions::default());

    let preventive = plan
        .preventive
        .setpoints
        .iter()
        .find(|s| s.action == pst)
        .and_then(|s| s.tap)
        .expect("the preventive perimeter optimizes pst_fr here");
    assert_eq!(preventive, -5, "the preventive answer, and the curative anchor");

    for perimeter in plan.scenarios.iter().flat_map(|s| s.perimeters.iter()) {
        let Some(tap) = perimeter.setpoints.iter().find(|s| s.action == pst).and_then(|s| s.tap)
        else {
            continue;
        };
        assert!(
            (-5..=5).contains(&tap),
            "curative tap {tap} is outside the intersection [-5, 5]: absolute [-16, 16], \
             ten of the imported network's tap 5, and ten of the preventive answer -5"
        );
    }
}



/// An automaton range action stays inside its declared range.
///
/// The reference's scenario 1.2.2.4: `pst_fr` sits at tap 5 and may travel ±5
/// taps *relative to the previous instant*, so the auto perimeter's box is
/// [0, 10]. An automaton is not exempt from that. Left unclamped, the shift
/// stops only when it runs out of tap changer — 16 here, six taps past what the
/// CRAC permits, with every margin downstream correspondingly too good.
#[test]
fn an_automaton_may_not_shift_past_its_own_range() {
    let net = ucte::read(ucte_fixture("TestCase16Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep15us11-3case4.json"),
    )
    .expect("crac");
    let pst = crac.range_actions.iter().position(|r| r.id == "pst_fr").expect("pst_fr");
    let RangeActionKind::Pst { initial_tap, .. } = &crac.range_actions[pst].kind else {
        panic!("pst_fr should be a phase shifter");
    };
    assert_eq!(*initial_tap, 5);
    assert_eq!(crac.range_actions[pst].speed, Some(1), "an automaton");

    let plan = ac_plan(&net, &crac);
    let tap = plan
        .scenarios
        .iter()
        .filter_map(|s| s.automatons.as_ref())
        .flat_map(|a| a.range_actions.iter())
        .find(|(i, _, _)| *i == pst)
        .and_then(|(_, _, tap)| *tap)
        .expect("the automaton should shift pst_fr");
    assert!(
        (0..=10).contains(&tap),
        "tap {tap} is outside the ±5 the CRAC allows around the previous instant's 5"
    );
    assert_eq!(tap, 10, "and it needs all of it");
}

/// An automaton shifts until the circuits it watches are secure, and no further.
///
/// The reference's scenario 1.2.2.2, which is the one that shows why a single
/// shift is not enough. The set-point is sized from a linear estimate — margin
/// over sensitivity — and applied to a network that is not linear, so one shot
/// lands on tap −7 with the watched CNEC still short of its limit. Iterating to
/// −8 clears it, at a margin of 0.2 A: the reference's own answer, and visibly
/// the *smallest* set-point that does the job.
///
/// Both halves are asserted, because each fails differently. Stopping early
/// leaves a protection scheme that did not protect; going too far reports a
/// network healthier than the equipment would actually have left.
#[test]
fn an_automaton_stops_at_the_smallest_set_point_that_secures() {
    let net = ucte::read(ucte_fixture("TestCase16Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep15us11-3case2.json"),
    )
    .expect("crac");
    let pst = crac.range_actions.iter().position(|r| r.id == "pst_be").expect("pst_be");

    let plan = ac_plan(&net, &crac);
    let automatons = plan
        .scenarios
        .iter()
        .filter_map(|s| s.automatons.as_ref())
        .find(|a| a.range_actions.iter().any(|(i, _, _)| *i == pst))
        .expect("the automaton should shift pst_be");
    let tap = automatons
        .range_actions
        .iter()
        .find(|(i, _, _)| *i == pst)
        .and_then(|(_, _, tap)| *tap)
        .expect("a tap");

    assert_eq!(tap, -8, "-7 leaves the watched CNEC overloaded, -9 and beyond overshoot");
    assert!(
        automatons.final_margin_mw >= 0.0,
        "the perimeter should end secure, not {} MW",
        automatons.final_margin_mw
    );
}

/// Build an AC plan the way the reference's `@ac` scenarios are configured.
fn ac_plan(net: &ucte::UcteImport, crac: &Crac) -> gridoxide::rao::Plan {
    let resolution = Resolution::with_buses(crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &net.shunts,
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    // `RaoUtil.getFlowUnit`: an AC load flow means an ampere objective, and the
    // automaton ranks its overloads in the same unit.
    let mut options = SearchOptions::default();
    options.linear.flow_model = gridoxide::rao::FlowModel::Ac;
    options.linear.objective_unit = gridoxide::rao::ObjectiveUnit::Ampere;
    let mut solver = IpmSolver::new();
    run(crac, &network, &resolution, &mut solver, &options)
}
