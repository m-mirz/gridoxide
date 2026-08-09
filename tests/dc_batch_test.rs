//! Batched DC power flow, and DC contingency screening via LODF.
//!
//! The load-bearing test is `batch_matches_a_sequential_loop_exactly`. The
//! batch never re-solves anything — it exploits DC's exact linearity to express
//! each scenario as the base solve plus a response to that scenario's injection
//! delta, against one factorization built once. That is either exactly right or
//! quietly wrong, so it is checked against a plain loop over `dc_power_flow` at
//! 1e-12 rather than at any tolerance that would hide a modelling slip.

use std::path::PathBuf;

use gridoxide::batch::{BatchError, BusOverride, Scenario};
use gridoxide::linear::{
    dc_branches, dc_power_flow, DcBatchSolver, DcOptions, DcSensitivity,
};
use gridoxide::pgm::pgm_to_buses_and_branches;
use gridoxide::types::{Bus, BusType, Line, Transformer};

mod common;

fn load(rel: &str) -> (Vec<Bus>, Vec<Line>, Vec<Transformer>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow")
        .join(rel)
        .join("input.json");
    let input = common::load_pgm_input(&path);
    pgm_to_buses_and_branches(input, 1e6, 50.0)
}

/// A spread of scenarios that move several buses at once, including buses with
/// no load in the template — so the batch cannot pass by only handling the
/// easy "scale existing load" shape.
fn scenarios(buses: &[Bus], count: usize) -> Vec<Scenario> {
    (0..count)
        .map(|k| {
            let factor = 0.5 + 0.1 * k as f64;
            let overrides = buses
                .iter()
                .filter(|b| b.bus_type != BusType::Slack)
                .map(|b| BusOverride::new(b.idx).p(b.p_spec * factor - 0.01 * k as f64))
                .collect();
            Scenario::new(overrides)
        })
        .collect()
}

/// **The oracle.** Batched results must equal a sequential loop over
/// `dc_power_flow`, bus for bus and branch for branch.
#[test]
fn batch_matches_a_sequential_loop_exactly() {
    for name in ["symmetric/transmission-case", "symmetric/distribution-case"] {
        let (buses, lines, transformers) = load(name);
        let opts = DcOptions::default();
        let scenarios = scenarios(&buses, 12);

        let batched = DcBatchSolver::new()
            .solve(&buses, &lines, &transformers, opts, &scenarios)
            .unwrap();
        assert_eq!(batched.len(), scenarios.len());

        for (k, (scenario, got)) in scenarios.iter().zip(&batched).enumerate() {
            let mut expected_buses = buses.clone();
            for ov in &scenario.bus_overrides {
                if let Some(p) = ov.p_spec {
                    expected_buses[ov.bus].p_spec = p;
                }
            }
            let expected =
                dc_power_flow(&mut expected_buses, &lines, &transformers, opts);

            for (i, bus) in expected_buses.iter().enumerate() {
                assert!(
                    (bus.voltage_ang - got.voltage_ang[i]).abs() < 1e-12,
                    "{name} scenario {k} bus {i}: batch {} vs loop {}",
                    got.voltage_ang[i],
                    bus.voltage_ang
                );
            }
            for (b, (want, have)) in
                expected.branch_p.iter().zip(&got.branch_p).enumerate().map(|(b, p)| (b, p))
            {
                assert!(
                    (want - have).abs() < 1e-12,
                    "{name} scenario {k} branch {b}: batch {have} vs loop {want}"
                );
            }
            for (i, island) in expected.islands.iter().enumerate() {
                assert!(
                    (island.slack_pickup - got.slack_pickup[i]).abs() < 1e-12,
                    "{name} scenario {k} island {i}: batch {} vs loop {}",
                    got.slack_pickup[i],
                    island.slack_pickup
                );
            }
        }
    }
}

/// Results come back in scenario order whatever the thread count, and the
/// answer does not depend on how many workers produced it.
#[test]
fn results_are_ordered_and_thread_count_independent() {
    let (buses, lines, transformers) = load("symmetric/transmission-case");
    let opts = DcOptions::default();
    let scenarios = scenarios(&buses, 40);

    let single = DcBatchSolver::with_threads(1)
        .unwrap()
        .solve(&buses, &lines, &transformers, opts, &scenarios)
        .unwrap();
    let many = DcBatchSolver::with_threads(4)
        .unwrap()
        .solve(&buses, &lines, &transformers, opts, &scenarios)
        .unwrap();

    assert_eq!(single.len(), scenarios.len());
    assert_eq!(single, many, "thread count changed the batch's answer or its order");

    // And the scenarios really are distinct, so ordering is observable.
    assert!(
        single[0].branch_p != single[1].branch_p,
        "scenarios are identical, so ordering proves nothing"
    );
}

/// ZIP terms belong to the template and survive a `p_spec` override, exactly as
/// they do on the AC path. Getting this wrong would double-count or drop them
/// silently, since DC folds them into the same scalar injection.
#[test]
fn zip_terms_survive_a_p_spec_override() {
    use num_complex::Complex;
    use gridoxide::types::{ZipKind, ZipTerm};

    let mut buses = vec![
        Bus {
            idx: 0,
            bus_type: BusType::Slack,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: 0.0,
            q_max: 0.0,
            u_rated: 0.0,
            zip_terms: Vec::new(),
        },
        Bus {
            idx: 1,
            bus_type: BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: -0.2,
            q_spec: 0.0,
            q_min: 0.0,
            q_max: 0.0,
            u_rated: 0.0,
            zip_terms: vec![ZipTerm {
                s_const: Complex::new(-0.3, 0.0),
                kind: ZipKind::ConstImpedance,
            }],
        },
    ];
    let lines = vec![Line { from: 0, to: 1, r: 0.0, x: 0.1, b_shunt: 0.0, g_shunt: 0.0 }];
    let opts = DcOptions::default();

    let scenario = Scenario::new(vec![BusOverride::new(1).p(-0.5)]);
    let batched = DcBatchSolver::new()
        .solve(&buses, &lines, &[], opts, std::slice::from_ref(&scenario))
        .unwrap();

    buses[1].p_spec = -0.5;
    let expected = dc_power_flow(&mut buses, &lines, &[], opts);

    // -0.5 override plus the -0.3 constant-impedance term = -0.8 total.
    assert!((batched[0].branch_p[0] - 0.8).abs() < 1e-12, "{}", batched[0].branch_p[0]);
    assert!(
        (batched[0].branch_p[0] - expected.branch_p[0]).abs() < 1e-12,
        "batch dropped or double-counted the ZIP term"
    );
}

/// Input that cannot be interpreted is rejected rather than guessed at.
#[test]
fn malformed_input_is_rejected() {
    let (buses, lines, transformers) = load("symmetric/transmission-case");
    let opts = DcOptions::default();
    let batch = DcBatchSolver::new();

    let out_of_range = Scenario::new(vec![BusOverride::new(buses.len()).p(0.0)]);
    assert!(matches!(
        batch.solve(&buses, &lines, &transformers, opts, &[out_of_range]),
        Err(BatchError::BusOutOfRange { scenario: 0, .. })
    ));

    let mut outage = Scenario::new(vec![]);
    outage.branch_outages = vec![0];
    assert!(matches!(
        batch.solve(&buses, &lines, &transformers, opts, &[outage]),
        Err(BatchError::OutagesUnsupported { scenario: 0 })
    ));

    assert_eq!(batch.solve(&buses, &lines, &transformers, opts, &[]).unwrap().len(), 0);
}

/// `outage_flows` is the N-1 primitive: it must agree with actually removing
/// the branch and re-solving, which is the only check that matters for it.
#[test]
fn outage_flows_match_an_actual_outage_resolve() {
    let (buses, lines, transformers) = load("symmetric/transmission-case");
    let opts = DcOptions::default();
    let n_branches = lines.len() + transformers.len();

    let mut base_buses = buses.clone();
    let base = dc_power_flow(&mut base_buses, &lines, &transformers, opts);
    let branches = dc_branches(&lines, &transformers, opts);
    let sensitivity = DcSensitivity::new(&base_buses, &branches, n_branches).unwrap();

    let mut checked = 0;
    for outaged in 0..lines.len() {
        let Some(predicted) = sensitivity.outage_flows(&base.branch_p, outaged) else {
            continue;
        };
        if base.branch_p[outaged].abs() < 1e-9 {
            continue; // Nothing to redistribute; the identity is trivial.
        }

        // Open the line into gridoxide's own half-open representation, the
        // same path a genuinely open branch takes.
        let mut opened = lines.clone();
        opened[outaged].to = opened[outaged].from;
        let mut scratch = buses.clone();
        let actual = dc_power_flow(&mut scratch, &opened, &transformers, opts).branch_p;

        for k in 0..n_branches {
            assert!(
                (predicted[k] - actual[k]).abs() < 1e-9,
                "outage of {outaged}: branch {k} predicted {} but re-solve gives {}",
                predicted[k],
                actual[k]
            );
        }
        assert_eq!(predicted[outaged], 0.0, "the tripped branch must carry nothing");
        checked += 1;
    }
    assert!(checked > 0, "no load-carrying, non-radial branch was checked");
}

/// A radial branch has no post-outage flows, because removing it islands the
/// network rather than rerouting anything.
#[test]
fn outage_flows_refuse_a_radial_branch() {
    let (buses, lines, transformers) = load("symmetric/transmission-case");
    let opts = DcOptions::default();
    let n_branches = lines.len() + transformers.len();

    let mut base_buses = buses.clone();
    let base = dc_power_flow(&mut base_buses, &lines, &transformers, opts);
    let branches = dc_branches(&lines, &transformers, opts);
    let sensitivity = DcSensitivity::new(&base_buses, &branches, n_branches).unwrap();

    let mut radial = 0;
    for branch in 0..n_branches {
        if sensitivity.is_radial(branch) {
            assert!(sensitivity.outage_flows(&base.branch_p, branch).is_none());
            radial += 1;
        }
    }
    assert!(radial > 0, "fixture has no radial branch, so this proves nothing");

    // A mismatched base-flow vector is rejected rather than read out of bounds.
    assert!(sensitivity.outage_flows(&[0.0], 0).is_none());
}

/// **The N-k oracle.** Predicting a simultaneous multi-branch outage must match
/// actually removing every one of them and re-solving.
///
/// See `chaining_single_outages_is_not_a_substitute` for why this cannot be
/// assembled from single-branch LODF columns.
#[test]
fn multi_outage_flows_match_an_actual_n_minus_2_resolve() {
    let (buses, lines, transformers) = load("symmetric/transmission-case");
    let opts = DcOptions::default();
    let n_branches = lines.len() + transformers.len();

    let mut base_buses = buses.clone();
    let base = dc_power_flow(&mut base_buses, &lines, &transformers, opts);
    let branches = dc_branches(&lines, &transformers, opts);
    let sensitivity = DcSensitivity::new(&base_buses, &branches, n_branches).unwrap();

    let mut checked = 0;
    for a in 0..lines.len() {
        for b in (a + 1)..lines.len() {
            let Some(predicted) = sensitivity.multi_outage_flows(&base.branch_p, &[a, b]) else {
                continue;
            };

            let mut opened = lines.clone();
            opened[a].to = opened[a].from;
            opened[b].to = opened[b].from;
            let mut scratch = buses.clone();
            let actual =
                dc_power_flow(&mut scratch, &opened, &transformers, opts).branch_p;

            for k in 0..n_branches {
                assert!(
                    (predicted[k] - actual[k]).abs() < 1e-9,
                    "outage of {a}+{b}: branch {k} predicted {} but re-solve gives {}",
                    predicted[k],
                    actual[k]
                );
            }
            assert_eq!(predicted[a], 0.0);
            assert_eq!(predicted[b], 0.0);
            checked += 1;
        }
    }

    assert!(checked > 0, "no solvable branch pair was checked");
}

/// A one-element set must reproduce the single-branch result exactly — the two
/// entry points share one implementation, and this pins that they stay shared.
#[test]
fn a_single_element_set_matches_the_single_branch_result() {
    let (buses, lines, transformers) = load("symmetric/transmission-case");
    let opts = DcOptions::default();
    let n_branches = lines.len() + transformers.len();

    let mut base_buses = buses.clone();
    let base = dc_power_flow(&mut base_buses, &lines, &transformers, opts);
    let branches = dc_branches(&lines, &transformers, opts);
    let sensitivity = DcSensitivity::new(&base_buses, &branches, n_branches).unwrap();

    for branch in 0..n_branches {
        assert_eq!(
            sensitivity.outage_flows(&base.branch_p, branch),
            sensitivity.multi_outage_flows(&base.branch_p, &[branch]),
            "branch {branch}"
        );
    }
}

/// A set whose removal disconnects the network has no post-outage flows, and
/// says so rather than returning a large number. Built by hand, because it
/// needs a pair that is individually meshed but jointly breaking — exactly the
/// case single-branch screening misses.
#[test]
fn a_breaking_pair_is_reported_even_when_neither_branch_is_radial() {
    let mk = |idx, bus_type| Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec: if bus_type == BusType::Slack { 0.0 } else { -0.3 },
        q_spec: 0.0,
        q_min: 0.0,
        q_max: 0.0,
        u_rated: 0.0,
        zip_terms: Vec::new(),
    };
    let line = |from, to, x| Line { from, to, r: 0.0, x, b_shunt: 0.0, g_shunt: 0.0 };

    // Buses 0-1 joined by two parallel lines, then a tail to bus 2. Neither
    // parallel line is radial on its own; together they are the only path.
    let buses = vec![mk(0, BusType::Slack), mk(1, BusType::PQ), mk(2, BusType::PQ)];
    let lines = vec![line(0, 1, 0.1), line(0, 1, 0.2), line(1, 2, 0.3)];
    let opts = DcOptions::default();

    let mut scratch = buses.clone();
    let base = dc_power_flow(&mut scratch, &lines, &[], opts);
    let branches = dc_branches(&lines, &[], opts);
    let sensitivity = DcSensitivity::new(&scratch, &branches, lines.len()).unwrap();

    // Fixture assumption: neither of the parallel pair is radial by itself.
    assert!(!sensitivity.is_radial(0) && !sensitivity.is_radial(1));
    assert!(sensitivity.outage_flows(&base.branch_p, 0).is_some());
    assert!(sensitivity.outage_flows(&base.branch_p, 1).is_some());

    // Together they are the only path to buses 1 and 2.
    assert!(sensitivity.is_breaking_set(&[0, 1]));
    assert!(sensitivity.multi_outage_flows(&base.branch_p, &[0, 1]).is_none());

    // Branch 2 is radial on its own, so any set containing it breaks too.
    assert!(sensitivity.is_breaking_set(&[2]));
    assert!(sensitivity.is_breaking_set(&[0, 2]));
}

/// Degenerate and malformed sets are handled explicitly rather than falling
/// through to a singular solve that would look like a breaking set.
#[test]
fn empty_and_malformed_outage_sets() {
    let (buses, lines, transformers) = load("symmetric/transmission-case");
    let opts = DcOptions::default();
    let n_branches = lines.len() + transformers.len();

    let mut base_buses = buses.clone();
    let base = dc_power_flow(&mut base_buses, &lines, &transformers, opts);
    let branches = dc_branches(&lines, &transformers, opts);
    let sensitivity = DcSensitivity::new(&base_buses, &branches, n_branches).unwrap();

    // Outaging nothing changes nothing.
    assert_eq!(
        sensitivity.multi_outage_flows(&base.branch_p, &[]).unwrap(),
        base.branch_p
    );
    // A repeated index is malformed input, not a breaking set.
    assert!(sensitivity.multi_outage_flows(&base.branch_p, &[0, 0]).is_none());
    // Out of range, and a mismatched base-flow vector.
    assert!(sensitivity.multi_outage_flows(&base.branch_p, &[n_branches]).is_none());
    assert!(sensitivity.multi_outage_flows(&[0.0], &[0]).is_none());
}

/// N-3 on a densely meshed hand-built network, against the same re-solve
/// oracle — the formulation is not special-cased for k = 2.
#[test]
fn three_simultaneous_outages_match_a_resolve() {
    let mk = |idx, bus_type, p| Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: 0.0,
        q_min: 0.0,
        q_max: 0.0,
        u_rated: 0.0,
        zip_terms: Vec::new(),
    };
    let line = |from, to, x| Line { from, to, r: 0.0, x, b_shunt: 0.0, g_shunt: 0.0 };

    // Five buses, fully meshed enough that removing three still leaves a
    // connected network.
    let buses = vec![
        mk(0, BusType::Slack, 0.0),
        mk(1, BusType::PQ, -0.4),
        mk(2, BusType::PQ, -0.3),
        mk(3, BusType::PQ, -0.2),
        mk(4, BusType::PQ, 0.5),
    ];
    let lines = vec![
        line(0, 1, 0.10),
        line(1, 2, 0.15),
        line(2, 3, 0.20),
        line(3, 4, 0.25),
        line(4, 0, 0.30),
        line(0, 2, 0.35),
        line(1, 3, 0.40),
        line(2, 4, 0.45),
    ];
    let opts = DcOptions::default();

    let mut scratch = buses.clone();
    let base = dc_power_flow(&mut scratch, &lines, &[], opts);
    let branches = dc_branches(&lines, &[], opts);
    let sensitivity = DcSensitivity::new(&scratch, &branches, lines.len()).unwrap();

    let set = [1usize, 5, 7];
    let predicted = sensitivity
        .multi_outage_flows(&base.branch_p, &set)
        .expect("this triple should leave the network connected");

    let mut opened = lines.clone();
    for &l in &set {
        opened[l].to = opened[l].from;
    }
    let mut scratch = buses.clone();
    let actual = dc_power_flow(&mut scratch, &opened, &[], opts).branch_p;

    for k in 0..lines.len() {
        assert!(
            (predicted[k] - actual[k]).abs() < 1e-9,
            "branch {k}: predicted {} but re-solve gives {}",
            predicted[k],
            actual[k]
        );
    }
}

/// Why the simultaneous formulation exists: applying two single-branch LODF
/// columns one after another does **not** give the N-2 answer.
///
/// The second branch's LODF was computed on the *intact* network, so chaining
/// ignores how the first outage changed the way the second one's power
/// redistributes. Algebraically, the sequential route solves
/// `(1 - Ψ_aa)·c_a = f_a` and drops the `-Ψ_ab·c_b` coupling term that the
/// simultaneous system carries — so the two agree only where `Ψ_ab = 0`, which
/// is exactly what a meshed network does not give you.
#[test]
fn chaining_single_outages_is_not_a_substitute() {
    let mk = |idx, bus_type, p| Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: 0.0,
        q_min: 0.0,
        q_max: 0.0,
        u_rated: 0.0,
        zip_terms: Vec::new(),
    };
    let line = |from, to, x| Line { from, to, r: 0.0, x, b_shunt: 0.0, g_shunt: 0.0 };

    let buses = vec![
        mk(0, BusType::Slack, 0.0),
        mk(1, BusType::PQ, -0.4),
        mk(2, BusType::PQ, -0.3),
        mk(3, BusType::PQ, -0.2),
        mk(4, BusType::PQ, 0.5),
    ];
    let lines = vec![
        line(0, 1, 0.10),
        line(1, 2, 0.15),
        line(2, 3, 0.20),
        line(3, 4, 0.25),
        line(4, 0, 0.30),
        line(0, 2, 0.35),
        line(1, 3, 0.40),
        line(2, 4, 0.45),
    ];
    let opts = DcOptions::default();

    let mut scratch = buses.clone();
    let base = dc_power_flow(&mut scratch, &lines, &[], opts);
    let branches = dc_branches(&lines, &[], opts);
    let sensitivity = DcSensitivity::new(&scratch, &branches, lines.len()).unwrap();

    let (a, b) = (1usize, 6usize);

    // Ground truth: open both and re-solve.
    let mut opened = lines.clone();
    opened[a].to = opened[a].from;
    opened[b].to = opened[b].from;
    let mut scratch = buses.clone();
    let actual = dc_power_flow(&mut scratch, &opened, &[], opts).branch_p;

    let simultaneous = sensitivity.multi_outage_flows(&base.branch_p, &[a, b]).unwrap();
    let first = sensitivity.outage_flows(&base.branch_p, a).unwrap();
    let chained = sensitivity.outage_flows(&first, b).unwrap();

    let worst = |v: &[f64]| {
        v.iter().zip(&actual).map(|(x, t)| (x - t).abs()).fold(0.0f64, f64::max)
    };
    assert!(worst(&simultaneous) < 1e-9, "simultaneous is off by {}", worst(&simultaneous));
    assert!(
        worst(&chained) > 1e-3,
        "chaining happened to agree here, so this fixture cannot show the \
         difference (gap {})",
        worst(&chained)
    );
}
