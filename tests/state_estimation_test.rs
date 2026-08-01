//! The phase-2 gate: gridoxide's WLS estimate against power-grid-model's.
//!
//! `tests/measurement_residual_test.rs` checks the measurement *model* at a
//! state someone else computed. This checks the estimator itself: start from a
//! flat state, run Gauss-Newton on the fixture's sensors alone, and compare the
//! answer against the one power-grid-model published.
//!
//! Per-unit magnitudes are compared at 1e-6, far tighter than measurement
//! noise and deliberately so: both tools solve the same weighted-least-squares
//! problem from the same data, so their optima should agree to solver
//! tolerance, not merely to within the noise.
//!
//! # Angles and the global phase
//!
//! Angles are handled in two regimes, because only one of them has an absolute
//! answer:
//!
//! - **With a voltage angle measured**, the phase is pinned by the data, so
//!   absolute angles must match.
//! - **Without one**, the whole estimate is invariant under a global rotation:
//!   every measurement function depends on angle *differences* only. Any
//!   reference is then equally valid, and power-grid-model's own fixtures do
//!   not agree with each other on which to use — `transmission-case` reports
//!   its source node at exactly 0, while `1os2msr-no-angle` reports its source
//!   node at -0.0130. So this asserts what is actually determined: that
//!   gridoxide's angles match PGM's *up to one constant shared by every bus*.
//!
//! That second check is stronger than it first appears. A genuinely wrong
//! estimate does not produce a uniform offset — it produces per-bus errors. The
//! test therefore requires the offset to be identical across all nodes to 1e-6,
//! which is what distinguishes "a different convention" from "a different
//! answer".

mod common;

use std::path::PathBuf;

use gridoxide::measurement::measurements_from_pgm;
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_network};
use gridoxide::se::bad_data;
use gridoxide::se::constraints::Constraints;
use gridoxide::se::jacobian::StateLayout;
use gridoxide::se::nr::{estimate, flat_start, SeOptions, SeStatus};
use gridoxide::se::observability::analyze;
use gridoxide::se::SeNetwork;
use gridoxide::solver::JacobianBackend;

const S_BASE_VA: f64 = 1e6;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/state_estimation")
        .join(name)
}

/// Estimates `name` from its sensors and compares every node against the
/// fixture's expected output.
fn assert_estimate_matches(name: &str, backend: JacobianBackend, tol: f64) {
    let dir = fixture_dir(name);
    let input = common::load_pgm_input(&dir.join("input.json"));
    let expected = common::load_json(&dir.join("sym_output.json"));

    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, S_BASE_VA);
    let net = pgm_to_network(
        common::load_pgm_input(&dir.join("input.json")),
        S_BASE_VA,
        50.0,
    );
    let measurements = measurements_from_pgm(&input, &net, S_BASE_VA).expect("measurements");

    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &shunts);
    let se_net = SeNetwork::new(&net, ybus.finish(), &shunts);

    let mut buses = net.buses.clone();
    flat_start(&mut buses, &measurements);
    let options = SeOptions { backend, ..SeOptions::default() };
    let report = estimate(&measurements, &mut buses, &se_net, &options);

    assert_eq!(
        report.status,
        SeStatus::Converged,
        "{name} [{backend:?}]: {report:?}"
    );

    // Absolute angles are only determined when something measures one.
    let phase_is_measured = measurements
        .iter()
        .any(|m| m.kind == gridoxide::measurement::MeasurementKind::VoltageAngle);

    let mut offsets = Vec::new();
    for node in expected["data"]["node"].as_array().expect("node output") {
        let id = node["id"].as_u64().expect("node id");
        let idx = net.node_idx[&id];
        let (Some(u_pu), Some(u_angle)) = (
            node["u_pu"].as_f64(),
            node["u_angle"].as_f64(),
        ) else {
            continue;
        };
        assert!(
            (buses[idx].voltage_mag - u_pu).abs() < tol,
            "{name} [{backend:?}] node {id}: |V| = {}, PGM says {u_pu}",
            buses[idx].voltage_mag
        );
        if phase_is_measured {
            assert!(
                (buses[idx].voltage_ang - u_angle).abs() < tol,
                "{name} [{backend:?}] node {id}: angle = {}, PGM says {u_angle}",
                buses[idx].voltage_ang
            );
        } else {
            offsets.push((id, buses[idx].voltage_ang - u_angle));
        }
    }

    if let Some(&(ref_id, reference)) = offsets.first() {
        for &(id, offset) in &offsets {
            assert!(
                (offset - reference).abs() < tol,
                "{name} [{backend:?}] node {id}: angle offset {offset} differs from node \
                 {ref_id}'s {reference} — a uniform offset is a reference convention, a \
                 varying one is a wrong estimate"
            );
        }
    }
}

/// A three-bus radial case with voltage phasor sensors on every node and power
/// sensors on both line ends — the canonical worked example in
/// power-grid-model's own test set.
#[test]
fn estimates_1os2msr() {
    assert_estimate_matches("1os2msr", JacobianBackend::Scalar, 1e-6);
}

/// The same network with the voltage angles removed, so the phase has to come
/// from a pinned reference instead of from the measurements. Exercises the
/// other branch of `StateLayout`'s reference logic.
#[test]
fn estimates_1os2msr_without_angle_measurements() {
    assert_estimate_matches("1os2msr-no-angle", JacobianBackend::Scalar, 1e-6);
}

/// Contains a measurement with an infinite sigma, which must contribute
/// nothing at all rather than poisoning the gain matrix with a zero-weight row.
#[test]
fn estimates_with_an_infinite_sigma_measurement() {
    assert_estimate_matches("inf-measurement-with-injection", JacobianBackend::Scalar, 1e-6);
}

/// The largest fixture, with transformers as well as lines.
#[test]
fn estimates_transmission_case() {
    assert_estimate_matches("transmission-case", JacobianBackend::Scalar, 1e-6);
}

/// A node with no appliance attached injects exactly nothing, and
/// power-grid-model overrides the 0.1 p.u. injection sensor placed on it with
/// that fact — its published answer has both buses at exactly 1.0 angle 0.
///
/// Reproducing it requires treating zero injection as a hard constraint rather
/// than as data to be fitted: a weighted least-squares fit that merely trusts
/// the sensor a little less would still be pulled toward it. This fixture
/// therefore fails outright without `se::constraints`, which is what makes it
/// the phase-4 gate.
#[test]
fn zero_injection_constraint_overrides_a_conflicting_sensor() {
    assert_estimate_matches(
        "node-injection-sensor-and-zero-injection",
        JacobianBackend::Scalar,
        1e-6,
    );
}

/// Every fixture the estimator handles should be observable in its *physical*
/// unknowns.
///
/// Anything reported undetermined must be one of gridoxide's own synthesized
/// buses — a virtual slack bus behind a source, or a three-winding
/// transformer's star point. Those carry no PGM node id and are genuinely
/// unobservable whenever the source's own power is unmeasured, which is a
/// property of gridoxide's network model rather than of the measurement set.
/// A *physical* node turning up here would mean the fixture cannot determine
/// something power-grid-model evidently does.
#[test]
fn fixtures_are_observable_in_their_physical_unknowns() {
    for name in [
        "1os2msr",
        "1os2msr-no-angle",
        "inf-measurement-with-injection",
        "transmission-case",
        "node-injection-sensor-and-zero-injection",
    ] {
        let dir = fixture_dir(name);
        let input = common::load_pgm_input(&dir.join("input.json"));
        let id_to_idx = node_id_to_idx(&input);
        let shunts = pgm_shunts_1ph(&input, &id_to_idx, S_BASE_VA);
        let net = pgm_to_network(
            common::load_pgm_input(&dir.join("input.json")),
            S_BASE_VA,
            50.0,
        );
        let measurements = measurements_from_pgm(&input, &net, S_BASE_VA).expect("measurements");
        let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
        stamp_shunts(&mut ybus, &shunts);
        let se_net = SeNetwork::new(&net, ybus.finish(), &shunts);

        let mut buses = net.buses.clone();
        flat_start(&mut buses, &measurements);
        let layout = StateLayout::new(&buses, &measurements, &se_net);
        let report = analyze(&measurements, &buses, &se_net, &layout);
        assert!(!report.skipped_numerical, "{name}: fixture should be small enough to analyze");

        // Physical nodes occupy the first `id_to_idx.len()` bus indices.
        let n_physical = id_to_idx.len();
        for unknown in report.unobservable.iter().chain(&report.structurally_unmeasured) {
            assert!(
                unknown.bus >= n_physical,
                "{name}: physical bus {} ({:?}) is unobservable — report {report:?}",
                unknown.bus,
                unknown.quantity
            );
        }
    }
}

/// Bad-data analysis on the fixtures: silent where the data agrees, and
/// pointing at the right sensor where it does not.
///
/// Note the consistent fixtures produce a chi-squared of ~0 rather than ~dof.
/// power-grid-model generated their readings from the true state without adding
/// noise, so there is nothing for the estimate to disagree with. Real telemetry
/// would sit near its degrees of freedom; a near-zero statistic here is a
/// property of the fixtures, not a bug.
#[test]
fn bad_data_analysis_flags_only_the_inconsistent_fixture() {
    for (name, should_reject) in [
        ("1os2msr", false),
        ("1os2msr-no-angle", false),
        ("inf-measurement-with-injection", false),
        ("transmission-case", false),
        ("node-injection-sensor-and-zero-injection", true),
    ] {
        let dir = fixture_dir(name);
        let input = common::load_pgm_input(&dir.join("input.json"));
        let id_to_idx = node_id_to_idx(&input);
        let shunts = pgm_shunts_1ph(&input, &id_to_idx, S_BASE_VA);
        let net = pgm_to_network(
            common::load_pgm_input(&dir.join("input.json")),
            S_BASE_VA,
            50.0,
        );
        let measurements = measurements_from_pgm(&input, &net, S_BASE_VA).expect("measurements");
        let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
        stamp_shunts(&mut ybus, &shunts);
        let se_net = SeNetwork::new(&net, ybus.finish(), &shunts);

        let mut buses = net.buses.clone();
        flat_start(&mut buses, &measurements);
        let report = estimate(&measurements, &mut buses, &se_net, &SeOptions::default());

        let layout = StateLayout::new(&buses, &measurements, &se_net);
        let constraints = Constraints::new(&se_net.zero_injection);
        let bad = bad_data::analyze(
            &measurements,
            &report.residuals,
            &buses,
            &se_net,
            &layout,
            &constraints,
            bad_data::Candidates::default(),
        );

        assert_eq!(
            bad.rejects_at(0.05),
            should_reject,
            "{name}: chi-squared {:.3e} on {} dof, p = {:.3e}",
            bad.chi_squared,
            bad.degrees_of_freedom,
            bad.p_value
        );

        if should_reject {
            // The offending sensor is the node-injection one placed on a bus
            // with no appliance; the zero-injection constraint holds the state
            // at the truth, so the whole disagreement lands in that residual.
            let worst = bad.worst().expect("a suspect should be identified");
            assert!(
                matches!(
                    measurements[worst.measurement].target,
                    gridoxide::measurement::Target::NodeInjection(_)
                ),
                "{name}: expected the injection sensor to be worst, got {:?}",
                measurements[worst.measurement]
            );
            assert!(
                worst.normalized_residual > 3.0,
                "{name}: normalized residual {} should exceed the conventional threshold",
                worst.normalized_residual
            );
        }
    }
}

/// The gain matrix is an ordinary square sparse system, so every backend that
/// carries power flow's Jacobian carries it too. `Block` is expected to fall
/// back to the scalar path (its 2x2-per-bus structure does not fit a 2N-1 state
/// vector), and this asserts that the fallback produces the same answer rather
/// than silently mis-assembling.
#[test]
fn every_backend_agrees() {
    for backend in [
        JacobianBackend::Scalar,
        JacobianBackend::Block,
        JacobianBackend::KluNative,
    ] {
        assert_estimate_matches("1os2msr", backend, 1e-6);
        assert_estimate_matches("transmission-case", backend, 1e-6);
    }
}
