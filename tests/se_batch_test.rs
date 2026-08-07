//! Batched state estimation against power-grid-model's own batch fixtures.
//!
//! Two things are checked, and the first matters more than the second.
//!
//! **Equivalence.** A parallel batch must produce bit-for-bit what a sequential
//! loop over `PersistentEstimator::estimate` produces, at every thread count.
//! Batching exists to be faster, and the moment it is also *different* it is
//! worthless — the same premise `tests/batch_solver_test.rs` rests on for power
//! flow.
//!
//! **Agreement with power-grid-model.** Each fixture ships an
//! `update_batch.json` of per-scenario sensor readings and a
//! `sym_output_batch.json` of the states power-grid-model converged to. Those
//! are compared per scenario.
//!
//! The scenarios arrive as whole documents rather than as measurement diffs, so
//! each is converted independently and then `se::batch::align_scenarios` folds
//! them onto one template. That is not a convenience: `sensor-update-initially-
//! empty`'s base document has two voltage sensors carrying no readings at all,
//! so a template taken from the base would be empty and cover nothing.

mod common;

use std::path::PathBuf;

use gridoxide::measurement::{measurements_from_pgm, Measurement};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_network};
use gridoxide::se::batch::{
    align_scenarios, MeasurementOverride, SeBatchError, SeBatchSolver, SeScenario,
};
use gridoxide::se::nr::{linear_start, PersistentEstimator, SeMethod, SeOptions, SeStatus};
use gridoxide::se::SeNetwork;
use gridoxide::types::Bus;

const S_BASE_VA: f64 = 1e6;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/state_estimation")
        .join(name)
}

struct Loaded {
    buses: Vec<Bus>,
    se_net: SeNetwork,
    template: Vec<Measurement>,
    scenarios: Vec<SeScenario>,
    node_idx: std::collections::HashMap<u64, usize>,
    expected: serde_json::Value,
}

/// Builds the shared network plus one aligned scenario per batch entry.
fn load(name: &str) -> Loaded {
    let dir = fixture_dir(name);
    let base = common::load_json(&dir.join("input.json"));
    let batch = common::load_json(&dir.join("update_batch.json"));
    let expected = common::load_json(&dir.join("sym_output_batch.json"));

    // The network itself comes from the base document: a batch varies readings,
    // never topology, which is the whole premise of the shared factorization.
    let base_input = common::load_pgm_input(&dir.join("input.json"));
    let id_to_idx = node_id_to_idx(&base_input);
    let shunts = pgm_shunts_1ph(&base_input, &id_to_idx, S_BASE_VA);
    let net = pgm_to_network(
        common::load_pgm_input(&dir.join("input.json")),
        S_BASE_VA,
        50.0,
    );
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &shunts);
    let se_net = SeNetwork::new(&net, ybus.finish(), &shunts);

    let per_scenario: Vec<Vec<Measurement>> = batch["data"]
        .as_array()
        .expect("update_batch data is a list of scenarios")
        .iter()
        .map(|scenario| {
            let merged = common::apply_batch_scenario(&base, scenario);
            measurements_from_pgm(&merged, &net, S_BASE_VA).expect("scenario measurements")
        })
        .collect();

    let (template, scenarios) = align_scenarios(&per_scenario).expect("scenarios align");
    Loaded {
        buses: net.buses.clone(),
        se_net,
        template,
        scenarios,
        node_idx: net.node_idx.clone(),
        expected,
    }
}

/// The sequential reference: exactly what the batch must reproduce.
fn sequential(l: &Loaded, options: SeOptions) -> Vec<(Vec<Bus>, SeStatus)> {
    let mut estimator = PersistentEstimator::new(options);
    l.scenarios
        .iter()
        .map(|sc| {
            let mut measurements = l.template.clone();
            for ov in &sc.overrides {
                if let Some(v) = ov.value {
                    measurements[ov.measurement].value = v;
                }
                if let Some(s) = ov.sigma {
                    measurements[ov.measurement].sigma = s;
                }
            }
            let mut buses = l.buses.clone();
            linear_start(&mut buses, &l.se_net, &measurements);
            let report = estimator.estimate(&measurements, &mut buses, &l.se_net);
            (buses, report.status)
        })
        .collect()
}

/// Every batch fixture, whether or not gridoxide can currently solve it. The
/// equivalence check applies to all of them: batching must reproduce a
/// sequential loop even where that loop reports a singular system.
const BATCH_FIXTURES: &[(&str, SeMethod)] = &[
    ("sensor-update-nr", SeMethod::NewtonRaphson),
    ("sensor-update-il", SeMethod::IterativeLinear),
    ("sensor-update-initially-empty", SeMethod::NewtonRaphson),
    ("unbalanced-power-measurements-newton-raphson", SeMethod::NewtonRaphson),
    ("unbalanced-power-measurements-iterative-linear", SeMethod::IterativeLinear),
];

/// Every fixture, with the tolerance its own comparison holds to.
///
/// All are 1e-6 relative but `sensor-update-il`, and the reason there is the
/// method rather than the model. Its network is `sensor-update-nr`'s, and
/// gridoxide's *Newton-Raphson* reproduces power-grid-model's answer on it to
/// 1e-6 — so the measurement set, the weights, the constraints and the network
/// all agree. What differs is how the two tools linearize, and this fixture's
/// data is inconsistent enough (objective ≈ 320, against ~1e-22 on the
/// consistent ones) for that to show: gridoxide lands 7.8 V from
/// power-grid-model on a 20 kV node, both of them about 1,100 V from the
/// Newton-Raphson optimum they are each approximating.
///
/// Checked rather than assumed: removing the `|U|²` weight scaling that
/// `docs/src/state_estimation/iterative.md` records as gridoxide's known
/// departure from power-grid-model here moves the answer by less than a
/// millivolt, so that is not the cause and the remaining difference is
/// somewhere else in the linearization.
const SOLVABLE_FIXTURES: &[(&str, SeMethod, f64)] = &[
    ("sensor-update-nr", SeMethod::NewtonRaphson, 1e-6),
    ("sensor-update-il", SeMethod::IterativeLinear, 1e-3),
    ("sensor-update-initially-empty", SeMethod::NewtonRaphson, 1e-6),
    ("unbalanced-power-measurements-newton-raphson", SeMethod::NewtonRaphson, 1e-6),
    ("unbalanced-power-measurements-iterative-linear", SeMethod::IterativeLinear, 1e-6),
];

/// A parallel batch equals a sequential loop, bit for bit, at every thread
/// count.
///
/// Asserted with `assert_eq!` on the raw `f64`s rather than a tolerance. A
/// batch that merely agrees to 1e-12 has some order-dependence in it, and the
/// point of the shared factorization is that there is none to have.
#[test]
fn a_batch_reproduces_a_sequential_loop_exactly() {
    for &(name, method) in BATCH_FIXTURES {
        let l = load(name);
        let options = SeOptions { method, max_iter: 100, ..SeOptions::default() };
        let reference = sequential(&l, options);

        for threads in [1usize, 2, 4] {
            let solver = SeBatchSolver::with_threads(options, threads).expect("pool");
            let out = solver
                .estimate(&l.buses, &l.se_net, &l.template, &l.scenarios)
                .expect("batch estimates");
            assert_eq!(out.len(), l.scenarios.len(), "{name}: wrong scenario count");
            for (i, (result, (want_buses, want_status))) in out.iter().zip(&reference).enumerate() {
                assert_eq!(
                    result.report.status, *want_status,
                    "{name} scenario {i} at {threads} thread(s): status differs"
                );
                for (got, want) in result.buses.iter().zip(want_buses) {
                    assert_eq!(
                        (got.voltage_mag, got.voltage_ang),
                        (want.voltage_mag, want.voltage_ang),
                        "{name} scenario {i} at {threads} thread(s): bus {} differs",
                        got.idx
                    );
                }
            }
        }
    }
}

/// Every scenario's own state, against the one power-grid-model published for
/// it.
#[test]
fn batch_scenarios_match_pgm_per_scenario() {
    let mut checked = 0;
    for &(name, method, tol) in SOLVABLE_FIXTURES {
        let l = load(name);
        let options = SeOptions { method, max_iter: 100, ..SeOptions::default() };
        let out = SeBatchSolver::new(options)
            .estimate(&l.buses, &l.se_net, &l.template, &l.scenarios)
            .expect("batch estimates");

        let scenarios = l.expected["data"].as_array().expect("batch output");
        assert_eq!(scenarios.len(), out.len(), "{name}: scenario count differs");

        for (i, (result, expected)) in out.iter().zip(scenarios).enumerate() {
            assert_eq!(
                result.report.status,
                SeStatus::Converged,
                "{name} scenario {i}: {:?}",
                result.report
            );
            for node in expected["node"].as_array().expect("node output") {
                let id = node["id"].as_u64().expect("node id");
                // A de-energized node carries `energized: 0` and a state of all
                // zeroes, which is power-grid-model reporting "not solved"
                // rather than a voltage to compare against.
                if node["energized"].as_u64() == Some(0) {
                    continue;
                }
                let idx = l.node_idx[&id];
                let u_rated = l.buses[idx].u_rated;
                let Some(u) = node["u"].as_f64().or_else(|| {
                    node["u_pu"].as_f64().map(|pu| pu * u_rated)
                }) else {
                    continue;
                };
                let got = result.buses[idx].voltage_mag * u_rated;
                assert!(
                    (got - u).abs() <= tol * u.abs().max(1.0),
                    "{name} scenario {i} node {id}: |V| = {got}, PGM says {u}"
                );
                checked += 1;
            }
        }
    }
    // 20 today: 2 scenarios x 1 node, plus 2 fixtures x 3 scenarios x 3 nodes. A
    // floor rather than an equality, so the check catches coverage silently
    // dropping without breaking when a fixture is added.
    assert!(checked >= 20, "expected a meaningful number of node checks, got {checked}");
}

/// `sensor-update-initially-empty` is the fixture that forces the template to
/// come from the scenarios rather than from the base document.
///
/// Its two voltage sensors carry no readings at all in `input.json`, so the
/// base yields an empty measurement set. Anything that built the template from
/// the base would estimate nothing, converge trivially, and look fine.
#[test]
fn a_base_document_with_no_readings_still_yields_a_template() {
    let dir = fixture_dir("sensor-update-initially-empty");
    let base_input = common::load_pgm_input(&dir.join("input.json"));
    let net = pgm_to_network(
        common::load_pgm_input(&dir.join("input.json")),
        S_BASE_VA,
        50.0,
    );
    let from_base = measurements_from_pgm(&base_input, &net, S_BASE_VA).expect("measurements");
    assert!(from_base.is_empty(), "the base document is supposed to carry no readings");

    let l = load("sensor-update-initially-empty");
    assert!(
        !l.template.is_empty(),
        "the template must come from the scenarios, which do carry readings"
    );
}

/// A scenario that switches a voltage angle on or off changes `n_unknowns`, so
/// it cannot share a factorization and is rejected rather than answered.
#[test]
fn changing_whether_an_angle_is_measured_is_rejected() {
    let l = load("sensor-update-nr");
    // Every angle row, not just the first: this fixture has two, and silencing
    // one would leave the reference still determined by the other.
    let angle_rows: Vec<usize> = l
        .template
        .iter()
        .enumerate()
        .filter(|(_, m)| m.kind == gridoxide::measurement::MeasurementKind::VoltageAngle)
        .map(|(i, _)| i)
        .collect();
    assert!(!angle_rows.is_empty(), "fixture is supposed to carry voltage angles");

    let mut scenarios = l.scenarios.clone();
    scenarios[0] = SeScenario::new(
        angle_rows.iter().map(|&i| MeasurementOverride::new(i).absent()).collect(),
    );

    let solver = SeBatchSolver::new(SeOptions::default());
    assert_eq!(
        solver.estimate(&l.buses, &l.se_net, &l.template, &scenarios).unwrap_err(),
        SeBatchError::AngleReferenceChanged { scenario: 0 }
    );
}

/// An override past the end of the template is a caller error, reported with
/// the offending index rather than panicking on the slice.
#[test]
fn an_out_of_range_override_is_reported() {
    let l = load("sensor-update-nr");
    let n = l.template.len();
    let scenarios = vec![SeScenario::new(vec![MeasurementOverride::new(n).value(1.0)])];
    let solver = SeBatchSolver::new(SeOptions::default());
    assert_eq!(
        solver.estimate(&l.buses, &l.se_net, &l.template, &scenarios).unwrap_err(),
        SeBatchError::MeasurementOutOfRange { scenario: 0, measurement: n, n }
    );
}


/// A component with no source is reported at exactly zero, and the rest of the
/// network solves around it.
///
/// `sensor-update-nr`'s network has three connected components and one source.
/// Node 0 is isolated; node 3 carries a voltage sensor but no branch at all;
/// nodes 4/5/6 are joined to each other by links and to nothing else. Only
/// nodes 1 and 2 are reachable from the source, and power-grid-model reports
/// every other node as `energized: 0` with a state of exactly zero — including
/// node 3, whose sensor it simply ignores. Energization is topological.
///
/// This used to leave gridoxide's gain matrix singular: `mask_untouched` pins a
/// column nothing *structurally* touches, which catches isolated node 0, but
/// nodes 4/5/6 are reached by node 4's own zero-injection constraint and node 3
/// by its own voltage sensor. Touched, and determined by nothing.
#[test]
fn a_de_energised_island_is_reported_at_zero() {
    for name in ["sensor-update-nr", "sensor-update-il"] {
        let l = load(name);

        // Three components, exactly one of them energized.
        let components = gridoxide::network::connected_components(&l.se_net.ybus);
        let live = components
            .iter()
            .filter(|c| c.iter().any(|&i| !l.se_net.source_branches[i].is_empty()))
            .count();
        assert!(
            components.len() > 1 && live == 1,
            "{name}: expected several components with exactly one energised"
        );

        let options = SeOptions { method: SeMethod::NewtonRaphson, max_iter: 100, ..SeOptions::default() };
        let out = SeBatchSolver::new(options)
            .estimate(&l.buses, &l.se_net, &l.template, &l.scenarios)
            .expect("batch estimates");

        let expected = l.expected["data"].as_array().expect("batch output");
        let mut zeroed = 0;
        for (result, scenario) in out.iter().zip(expected) {
            assert_eq!(result.report.status, SeStatus::Converged, "{name}: {:?}", result.report);
            for node in scenario["node"].as_array().expect("node output") {
                if node["energized"].as_u64() != Some(0) {
                    continue;
                }
                let idx = l.node_idx[&node["id"].as_u64().expect("node id")];
                assert_eq!(
                    (result.buses[idx].voltage_mag, result.buses[idx].voltage_ang),
                    (0.0, 0.0),
                    "{name}: node {} is de-energised and must be reported at zero",
                    node["id"]
                );
                zeroed += 1;
            }
        }
        assert!(zeroed >= 8, "{name}: expected several de-energised nodes, got {zeroed}");
    }
}

/// Every batch fixture in the tree is exercised by this file.
///
/// The scanners in `tests/measurement_residual_test.rs` both require a
/// `sym_output.json`, which a batch fixture does not have — it ships
/// `sym_output_batch.json` instead. So a batch fixture added later would be
/// invisible to those guards, and this is the one that covers it.
#[test]
fn every_batch_fixture_is_listed_here() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgm/state_estimation");
    let mut unlisted = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("fixture dir") {
        let entry = entry.expect("dir entry");
        if !entry.path().join("update_batch.json").exists() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if !BATCH_FIXTURES.iter().any(|&(n, _)| n == name) {
            unlisted.push(name);
        }
    }
    assert!(
        unlisted.is_empty(),
        "these fixtures ship an update_batch.json but are not in BATCH_FIXTURES: {unlisted:?}"
    );
    assert_eq!(BATCH_FIXTURES.len(), 5, "batch fixtures present");
}
