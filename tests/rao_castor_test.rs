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
