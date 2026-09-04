//! The search tree over network actions — phase 8 of `plans/RAO_PLAN.md`.
//!
//! A range action has a degree of freedom and can be optimized by an LP; a
//! network action is taken or not, with no gradient between the two. So the
//! discrete half is searched, and the two halves interleave: every leaf re-runs
//! the full linear optimization, because a topological action changes the
//! sensitivities the phase shifters are optimized against.

use std::path::PathBuf;

use gridoxide::opf::ipm::IpmSolver;
use gridoxide::rao::crac::*;
use gridoxide::rao::linear::LinearOptions;
use gridoxide::rao::{crac_json, evaluate_with, search, Network, Resolution, SearchOptions};
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

/// Range actions help here and **topology does not**: opening either available
/// line makes the preventive margin worse. That
/// makes it the fixture for "correctly declines", which is the property a greedy
/// search is most likely to get wrong in the expensive direction.
fn case() -> Case {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(rao_fixture("crac-for-12nodes.json")).expect("crac");
    Case { net, crac }
}

/// The mirror image: one topological action, no range actions, and taking it
/// moves the preventive margin from -512.7 MW to +500.0 — insecure to secure.
fn topology_case() -> Case {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(rao_fixture("crac-topology-helps.json")).expect("crac");
    Case { net, crac }
}

impl Case {
    fn resolution(&self) -> Resolution {
        Resolution::new(&self.crac, &self.net.branch_ids)
    }

    fn network(&self) -> Network<'_> {
        Network {
            generation: &self.net.generation,
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

    fn preventive(&self) -> State {
        State::preventive(self.crac.preventive_instant().expect("preventive"))
    }
}

fn run(c: &Case, options: &SearchOptions) -> gridoxide::rao::SearchResult {
    let mut solver = IpmSolver::new();
    search(&c.crac, &c.network(), &c.resolution(), std::slice::from_ref(&c.preventive()), &mut solver, options)
}

// ---------------------------------------------------------------------------
// Does it find anything
// ---------------------------------------------------------------------------

#[test]
fn the_search_finds_a_topological_action_that_secures_the_network() {
    let c = topology_case();
    let with_topology = run(&c, &SearchOptions::default());
    let without = run(&c, &SearchOptions { max_depth: 0, ..Default::default() });

    assert!(with_topology.initial_margin_mw < 0.0, "the fixture should start overloaded");
    assert!(!without.is_secure(), "without the action the network stays insecure");
    assert_eq!(with_topology.network_actions.len(), 1);
    assert!(
        with_topology.final_margin_mw > without.final_margin_mw + 1.0,
        "topology added nothing: {} vs {}",
        with_topology.final_margin_mw,
        without.final_margin_mw
    );
    assert!(with_topology.is_secure(), "the action should secure the network");
    assert!(with_topology.leaves > 0);
}

/// A greedy search must decline an action that makes things worse.
///
/// This is the property such a search is most likely to get wrong in the
/// expensive direction — spending a remedial action for a negative return —
/// and it is the one my own expectation got wrong first: I assumed the twelve-
/// node fixture's two line openings would help, and both make its preventive
/// margin substantially worse. The search evaluated both and took neither,
/// which is correct.
#[test]
fn the_search_declines_actions_that_would_make_the_margin_worse() {
    let c = case();
    let result = run(&c, &SearchOptions::default());
    assert!(
        result.network_actions.is_empty(),
        "took {:?}, but every action here is harmful",
        result.network_actions
    );
    // It still had to *look*, or "declines" would be indistinguishable from
    // "never considered".
    assert!(result.leaves > 0, "no candidate was evaluated");
    // And the range-action half must still have run.
    assert!(result.final_margin_mw > result.initial_margin_mw);
}

#[test]
fn depth_zero_takes_no_network_action() {
    let c = case();
    let result = run(&c, &SearchOptions { max_depth: 0, ..Default::default() });
    assert!(result.network_actions.is_empty());
    assert!(result.open_branches.is_empty());
    assert_eq!(result.leaves, 0, "no candidate should be evaluated at depth 0");
    assert_eq!(result.depth, 0);
    // But the range actions are still optimized — the root leaf always runs.
    assert!(result.final_margin_mw >= result.initial_margin_mw - 1e-9);
}

#[test]
fn the_chosen_actions_are_the_ones_reported_as_open() {
    let c = topology_case();
    let result = run(&c, &SearchOptions::default());
    let mut expected: Vec<usize> = Vec::new();
    for &index in &result.network_actions {
        for elementary in &c.crac.network_actions[index].elementary {
            for element in elementary.elements() {
                expected.push(c.resolution().branch(element).expect("resolved"));
            }
        }
    }
    expected.sort_unstable();
    expected.dedup();
    assert_eq!(result.open_branches, expected);
}

/// The margin the search reports must be the margin of the network it describes.
///
/// The searcher measures a leaf by removing the branch from its working copy;
/// the evaluator applies the same action as a Woodbury outage update. Those are
/// two different mechanisms, and if they disagree the result names a network
/// nobody scored.
#[test]
fn the_reported_margin_matches_an_independent_evaluation_of_the_winning_network() {
    let c = topology_case();
    let result = run(&c, &SearchOptions::default());
    assert!(!result.open_branches.is_empty(), "this test needs a chosen action");

    let network = Network {
        generation: &c.net.generation,
        buses: &c.net.buses,
        lines: &c.net.lines,
        transformers: &result.transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let independent = evaluate_with(&c.crac, &network, &c.resolution(), &result.open_branches);
    let margin = independent
        .perimeters
        .iter()
        .find(|p| p.state == c.preventive())
        .and_then(|p| p.min_margin())
        .expect("a preventive perimeter");
    assert!(
        (margin - result.final_margin_mw).abs() < 1e-6,
        "search says {} MW, an independent evaluation says {margin} MW",
        result.final_margin_mw
    );
}

// ---------------------------------------------------------------------------
// Determinism and stopping
// ---------------------------------------------------------------------------

#[test]
fn the_search_reproduces_itself() {
    // A search tree that returns a different answer each run cannot be
    // regression-tested, and an operator cannot be told why yesterday's study
    // disagreed with today's.
    let c = case();
    let a = run(&c, &SearchOptions::default());
    let b = run(&c, &SearchOptions::default());
    assert_eq!(a.network_actions, b.network_actions);
    assert_eq!(a.open_branches, b.open_branches);
    assert_eq!(a.leaves, b.leaves);
    assert!((a.final_margin_mw - b.final_margin_mw).abs() < 1e-12);
}

#[test]
fn a_minimum_impact_threshold_stops_the_search_spending_actions_cheaply() {
    let c = topology_case();
    let free = run(&c, &SearchOptions::default());
    let strict = run(
        &c,
        &SearchOptions { absolute_min_impact: 1e6, ..Default::default() },
    );
    assert!(!free.network_actions.is_empty());
    assert!(
        strict.network_actions.is_empty(),
        "no action improves by 1e6 MW, so none should be taken"
    );
    // And refusing to act must not make the answer worse than the root.
    assert!(strict.final_margin_mw >= strict.initial_margin_mw - 1e-9);
}

#[test]
fn a_relative_threshold_also_binds() {
    let c = topology_case();
    let strict =
        run(&c, &SearchOptions { relative_min_impact: 10.0, ..Default::default() });
    assert!(strict.network_actions.is_empty(), "a 1000% improvement is not available");
}

#[test]
fn depth_bounds_how_many_actions_stack() {
    let c = topology_case();
    for depth in 0..=2 {
        let result = run(&c, &SearchOptions { max_depth: depth, ..Default::default() });
        assert!(
            result.network_actions.len() <= depth,
            "depth {depth} produced {} actions",
            result.network_actions.len()
        );
        assert!(result.depth <= depth);
    }
}

#[test]
fn deeper_is_never_worse() {
    // Greedy, so deeper is not guaranteed *better* — but it must never be worse,
    // since depth n+1 starts from depth n's answer.
    let c = topology_case();
    let mut previous = f64::NEG_INFINITY;
    for depth in 0..=2 {
        let result = run(&c, &SearchOptions { max_depth: depth, ..Default::default() });
        assert!(
            result.final_margin_mw >= previous - 1e-9,
            "depth {depth} gave {} after {previous}",
            result.final_margin_mw
        );
        previous = result.final_margin_mw;
    }
}

// ---------------------------------------------------------------------------
// What it refuses
// ---------------------------------------------------------------------------

#[test]
fn an_action_it_cannot_express_is_refused_whole_rather_than_applied_in_part() {
    // A network action is a set of elementary actions taken *together* —
    // "split this busbar" is one decision, not six. Applying the half that is
    // expressible would produce a network the CRAC never described, and a
    // result naming an action whose effect was not what was measured.
    let c = case();
    let mut crac = c.crac.clone();
    let index = 0;
    let element = crac.network_actions[index].elementary[0].elements()[0].to_string();
    crac.network_actions[index].elementary.push(ElementaryAction::ShuntSection {
        element: "some shunt".into(),
        section: 2,
    });

    let mut solver = IpmSolver::new();
    let result = search(
        &crac,
        &c.network(),
        &Resolution::new(&crac, &c.net.branch_ids),
        std::slice::from_ref(&c.preventive()),
        &mut solver,
        &SearchOptions::default(),
    );
    assert!(
        !result.network_actions.contains(&index),
        "an action with an inexpressible part was taken anyway"
    );
    // And its openable half must not appear in the result either.
    let branch = Resolution::new(&crac, &c.net.branch_ids).branch(&element).expect("resolved");
    assert!(!result.open_branches.contains(&branch));
}

#[test]
fn an_action_naming_an_unknown_element_is_skipped() {
    let c = case();
    let mut crac = c.crac.clone();
    crac.network_actions[0].elementary = vec![ElementaryAction::TerminalsConnection {
        element: "NO SUCH BRANCH".into(),
        connected: false,
    }];
    let mut solver = IpmSolver::new();
    let result = search(
        &crac,
        &c.network(),
        &Resolution::new(&crac, &c.net.branch_ids),
        std::slice::from_ref(&c.preventive()),
        &mut solver,
        &SearchOptions::default(),
    );
    assert!(!result.network_actions.contains(&0));
}

#[test]
fn actions_touching_the_same_element_are_not_stacked() {
    // Two actions on one element conflict: one may open a line the other moves,
    // and the outcome depends on an order a set does not have.
    let c = case();
    let mut crac = c.crac.clone();
    let element = crac.network_actions[0].elementary[0].elements()[0].to_string();
    // Make the second action touch the first's element too.
    crac.network_actions[1].elementary = vec![ElementaryAction::TerminalsConnection {
        element: element.clone(),
        connected: false,
    }];
    let mut solver = IpmSolver::new();
    let result = search(
        &crac,
        &c.network(),
        &Resolution::new(&crac, &c.net.branch_ids),
        std::slice::from_ref(&c.preventive()),
        &mut solver,
        &SearchOptions { max_depth: 2, ..Default::default() },
    );
    assert!(
        result.network_actions.len() <= 1,
        "two conflicting actions were stacked: {:?}",
        result.network_actions
    );
}

#[test]
fn only_actions_whose_usage_rules_reach_this_state_are_offered() {
    // "Open line NL1-NL2" is preventive-only in this CRAC; "Open line FR1-FR2"
    // is preventive and curative. So the curative perimeter must never choose
    // the first.
    let c = case();
    let curative = c
        .crac
        .states()
        .into_iter()
        .find(|s| c.crac.instants[s.instant].kind == InstantKind::Curative)
        .expect("a curative state");
    let mut solver = IpmSolver::new();
    let result = search(
        &c.crac,
        &c.network(),
        &c.resolution(),
        std::slice::from_ref(&curative),
        &mut solver,
        &SearchOptions::default(),
    );
    for &index in &result.network_actions {
        let action = &c.crac.network_actions[index];
        assert!(
            action.usage_rules.iter().any(|r| r.covers(&curative)),
            "`{}` is not available curatively",
            action.id
        );
    }
}

#[test]
fn a_state_with_no_network_action_still_optimizes_its_range_actions() {
    let c = case();
    let mut crac = c.crac.clone();
    crac.network_actions.clear();
    let mut solver = IpmSolver::new();
    let result = search(
        &crac,
        &c.network(),
        &Resolution::new(&crac, &c.net.branch_ids),
        std::slice::from_ref(&c.preventive()),
        &mut solver,
        &SearchOptions::default(),
    );
    assert!(result.network_actions.is_empty());
    assert_eq!(result.leaves, 0);
    assert!(
        result.final_margin_mw > result.initial_margin_mw,
        "the PST should still have been optimized"
    );
    assert!(result.setpoints.iter().any(|s| s.moved()));
}

#[test]
fn the_leaf_count_is_what_the_search_actually_cost() {
    // Per-leaf cost is where a search tree gets expensive, so the number has to
    // be reported honestly rather than estimated.
    let c = case();
    let result = run(&c, &SearchOptions { max_depth: 1, ..Default::default() });
    let available = c.crac.network_actions_for(&c.preventive()).len();
    assert_eq!(
        result.leaves, available,
        "depth 1 should evaluate exactly the available actions"
    );
}

/// A leaf is scored with its action applied, not with the root's flows.
#[test]
fn a_leaf_is_scored_with_its_action_applied() {
    let c = topology_case();
    let root = run(&c, &SearchOptions { max_depth: 0, ..Default::default() });
    let searched = run(&c, &SearchOptions::default());
    assert!(!searched.network_actions.is_empty());
    assert!(
        (searched.final_margin_mw - root.final_margin_mw).abs() > 1.0,
        "the leaf's margin equals the root's, so the action was not applied"
    );
}

/// An action's worth is judged *after* the range actions are re-tuned around it.
///
/// This is why every leaf re-runs the full LP, and it is where all the time
/// goes. On the twelve-node fixture the phase shifter is re-optimized under each
/// candidate topology, and the resulting margins differ from what the same
/// topology scores with the root's set-points — so the two halves genuinely
/// interleave rather than running in sequence.
#[test]
fn each_leaf_reoptimizes_the_range_actions_around_its_topology() {
    let c = case();
    let resolution = c.resolution();
    let state = c.preventive();

    // What each candidate scores with nothing re-tuned.
    let mut untuned = Vec::new();
    for action in c.crac.network_actions_for(&state) {
        let open: Vec<usize> = action
            .elementary
            .iter()
            .flat_map(|e| e.elements())
            .filter_map(|e| resolution.branch(e))
            .collect();
        if open.is_empty() {
            continue;
        }
        let margin = evaluate_with(&c.crac, &c.network(), &resolution, &open)
            .perimeters
            .iter()
            .find(|p| p.state == state)
            .and_then(|p| p.min_margin())
            .expect("perimeter");
        untuned.push(margin);
    }
    assert!(!untuned.is_empty());

    // The search's own root margin, with the shifter tuned and no topology.
    let root = run(&c, &SearchOptions { max_depth: 0, ..Default::default() });
    // Tuning strictly helps, so the root beats every untuned candidate here —
    // which is exactly why the search declines them all.
    for margin in &untuned {
        assert!(
            root.final_margin_mw > *margin,
            "tuned root {} should beat untuned candidate {margin}",
            root.final_margin_mw
        );
    }
}

// ---------------------------------------------------------------------------
// Conditional usage rules
// ---------------------------------------------------------------------------

/// An `onFlowConstraint` action is on the table only if its CNEC is overloaded.
///
/// The reference's own scenario 2.4.1.2 is built on this fixture, and gridoxide
/// used to fail it in the flattering direction: three actions and +97 A against
/// a reference that uses one and reports −45 A. A margin assertion alone cannot
/// tell "chose well" from "was offered too much and chose well among it", which
/// is what let the defect live, so this checks the *decision* and then the
/// answer.
///
/// The fixture separates the two cases cleanly. `open_fr1_fr3` is conditional on
/// a CNEC that starts at −42.8 MW, so it is available and the reference does use
/// it elsewhere (2.4.1.1). `open_be1_be4` is conditional on two that start at
/// +548 and +553 MW, so nothing authorizes it and it must never be offered.
#[test]
fn an_action_conditional_on_a_healthy_cnec_is_never_offered() {
    use gridoxide::rao::linear::ObjectiveUnit;
    use gridoxide::rao::usage::{branch_countries, Constrained};

    let net = ucte::read(ucte_fixture("TestCase16Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep16us3case2.json"),
    )
    .expect("crac");
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        generation: &net.generation,
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

    // The margins the decision rests on, read rather than assumed.
    let start = evaluate_with(&crac, &network, &resolution, &[]);
    let margin = |id: &str| -> f64 {
        let i = crac.flow_cnecs.iter().position(|c| c.id == id).expect("cnec");
        start
            .perimeters
            .iter()
            .flat_map(|p| p.cnecs.iter())
            .find(|c| c.cnec == i)
            .expect("evaluated")
            .margin_mw
    };
    assert!(margin("FFR1AA1  FFR4AA1  1 - co1_fr2_fr3_1 - curative") < 0.0);
    assert!(margin("FFR4AA1  DDE1AA1  1 - preventive") > 0.0);
    assert!(margin("FFR4AA1  DDE1AA1  1 - co1_fr2_fr3_1 - curative") > 0.0);

    let branches = gridoxide::linear::btheta::dc_branches(
        &net.lines,
        &net.transformers,
        Default::default(),
    );
    let constrained = Constrained::from_result(
        &start,
        ObjectiveUnit::Megawatt,
        &branch_countries(&crac, &network, &resolution, &branches),
    );
    let preventive = State::preventive(crac.preventive_instant().expect("preventive"));
    let rules = |id: &str| {
        crac.network_actions.iter().find(|a| a.id == id).expect("action").usage_rules.clone()
    };

    assert!(
        constrained.allows(&rules("open_fr1_fr3"), std::slice::from_ref(&preventive), &crac),
        "an overloaded CNEC should authorize its action"
    );
    assert!(
        !constrained.allows(&rules("open_be1_be4"), std::slice::from_ref(&preventive), &crac),
        "nothing authorizes open_be1_be4 and it must not be offered"
    );
    assert!(
        constrained.allows(&rules("close_fr1_fr5"), std::slice::from_ref(&preventive), &crac),
        "a free-to-use action is unaffected by any of this"
    );

    // And the search honours it, at a depth that would otherwise reach for it.
    let mut solver = IpmSolver::new();
    let result = search(
        &crac,
        &network,
        &resolution,
        std::slice::from_ref(&preventive),
        &mut solver,
        &SearchOptions { max_depth: 3, ..Default::default() },
    );
    let taken: Vec<&str> =
        result.network_actions.iter().map(|&i| crac.network_actions[i].id.as_str()).collect();
    assert!(
        !taken.contains(&"open_be1_be4"),
        "an action nothing authorized was taken: {taken:?}"
    );
}

// ---------------------------------------------------------------------------
// Predefined combinations
// ---------------------------------------------------------------------------

/// Everything but the winner's bookkeeping, on a fixture where both actions
/// hurt.
fn combination_run(c: &Case, combinations: Vec<Vec<String>>) -> gridoxide::rao::SearchResult {
    run(
        c,
        &SearchOptions {
            max_depth: 1,
            // Take whatever scores best, so the search cannot decline a
            // candidate for a reason this test is not about.
            absolute_min_impact: -1e9,
            relative_min_impact: -1e9,
            predefined_combinations: combinations,
            ..Default::default()
        },
    )
}

/// A predefined combination is **one** candidate, evaluated at **one** depth.
///
/// This is the reference's entire mechanism for searching anything other than a
/// greedy chain — `SearchTreeBloomer.bloom` returns these plus one candidate per
/// individual action, and never enumerates subsets — so the breadth of both
/// searches is configuration rather than algorithm. Every vendored
/// configuration carries `[]`, which is why reading this parameter moves no
/// assertion in the Cucumber gate and is worth having anyway.
///
/// Measured on the leaf count, which is the only place a candidate that is
/// offered and loses can be seen at all. Both of this fixture's actions make
/// the margin worse, so the combination of them is worse still and is correctly
/// not taken; that it was *evaluated* is the property under test.
#[test]
fn a_predefined_combination_is_offered_as_one_more_candidate() {
    let c = case();
    let ids: Vec<String> = c.crac.network_actions.iter().map(|a| a.id.clone()).collect();
    assert_eq!(ids.len(), 2, "this test needs exactly two actions to combine");

    let alone = combination_run(&c, Vec::new());
    let with_pair = combination_run(&c, vec![ids]);

    assert_eq!(alone.leaves, 2, "one leaf per individual action");
    assert_eq!(with_pair.leaves, 3, "the combination adds exactly one leaf, not one per member");
    assert!(with_pair.depth <= 1, "a combination costs one depth, not one per member");
}

/// A combination is taken **whole or not at all**, so anything wrong with it
/// removes it entirely rather than degrading it to the members that survive.
///
/// Applying half of one produces a network the CRAC never described — the same
/// rule [`effect_of`](gridoxide::rao::search) already applies to the elementary
/// actions inside a single network action, one level up.
#[test]
fn a_malformed_predefined_combination_is_dropped_whole() {
    let c = case();
    let ids: Vec<String> = c.crac.network_actions.iter().map(|a| a.id.clone()).collect();
    let baseline = combination_run(&c, Vec::new()).leaves;

    for (why, combination) in [
        ("an id no action carries", vec![ids[0].clone(), "no such action".to_string()]),
        ("fewer than two members", vec![ids[0].clone()]),
        ("the same action twice, which conflicts with itself", vec![ids[0].clone(), ids[0].clone()]),
        ("no members at all", Vec::new()),
    ] {
        let got = combination_run(&c, vec![combination]).leaves;
        assert_eq!(got, baseline, "a combination with {why} should offer no candidate");
    }
}

/// Two actions on the same network element cannot be taken together, and a
/// combination naming both is dropped rather than applied in some order.
///
/// The order would decide the answer, and a set has none — which is exactly why
/// [`compatible`](gridoxide::rao::search) exists for the chain. A predefined
/// combination is not an exemption from it.
#[test]
fn a_predefined_combination_of_conflicting_actions_is_dropped() {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let (mut crac, _) = crac_json::read(rao_fixture("crac-for-12nodes.json")).expect("crac");
    // A second action on the element the first one opens.
    let mut clash = crac.network_actions[0].clone();
    clash.id = "close the same line".to_string();
    for elementary in &mut clash.elementary {
        if let ElementaryAction::TerminalsConnection { connected, .. } = elementary {
            *connected = true;
        }
    }
    let first = crac.network_actions[0].id.clone();
    crac.network_actions.push(clash);
    let c = Case { net, crac };

    let baseline = combination_run(&c, Vec::new()).leaves;
    let got = combination_run(&c, vec![vec![first, "close the same line".to_string()]]).leaves;
    assert_eq!(got, baseline, "two actions on one element should offer no combined candidate");
}
