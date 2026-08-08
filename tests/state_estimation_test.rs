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
use gridoxide::measurement::{Measurement, MeasurementKind, Target};
use gridoxide::se::bad_data;
use gridoxide::se::measurement_functions;
use gridoxide::se::constraints::Constraints;
use gridoxide::se::jacobian::StateLayout;
use gridoxide::se::nr::{estimate, linear_start, SeMethod, SeOptions, SeStatus};
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
    assert_estimate_matches_with(name, backend, SeMethod::NewtonRaphson, tol, 20);
}

/// As above, but with the method and iteration budget chosen explicitly.
fn assert_estimate_matches_with(
    name: &str,
    backend: JacobianBackend,
    method: SeMethod,
    tol: f64,
    max_iter: usize,
) {
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
    linear_start(&mut buses, &se_net, &measurements);
    let options = SeOptions { backend, method, max_iter, ..SeOptions::default() };
    let report = estimate(&measurements, &mut buses, &se_net, &options);

    assert_eq!(
        report.status,
        SeStatus::Converged,
        "{name} [{backend:?}/{method:?}]: {report:?}"
    );

    // Absolute angles are only determined when something measures one.
    let phase_is_measured = measurements
        .iter()
        .any(|m| m.kind == gridoxide::measurement::MeasurementKind::VoltageAngle);

    let mut offsets = Vec::new();
    let mut checked = 0usize;
    for node in expected["data"]["node"].as_array().expect("node output") {
        let id = node["id"].as_u64().expect("node id");
        let idx = net.node_idx[&id];
        // Magnitude and angle are checked independently of each other, because
        // power-grid-model's validation framework publishes only the fields a
        // fixture is about: `three_winding_transformer` gives `u_pu` with no
        // `u_angle` at all. Requiring both would silently skip every node there
        // and leave the test asserting nothing but convergence.
        // Some fixtures publish only the volt-valued `u`. It is the same
        // quantity in the node's own rating, so deriving the per-unit value
        // keeps them from going unchecked.
        let u_pu = node["u_pu"]
            .as_f64()
            .or_else(|| node["u"].as_f64().map(|u| u / net.buses[idx].u_rated));
        if let Some(u_pu) = u_pu {
            assert!(
                (buses[idx].voltage_mag - u_pu).abs() < tol,
                "{name} [{backend:?}/{method:?}] node {id}: |V| = {}, PGM says {u_pu}",
                buses[idx].voltage_mag
            );
            checked += 1;
        }
        let Some(u_angle) = node["u_angle"].as_f64() else {
            continue;
        };
        if phase_is_measured {
            assert!(
                (buses[idx].voltage_ang - u_angle).abs() < tol,
                "{name} [{backend:?}/{method:?}] node {id}: angle = {}, PGM says {u_angle}",
                buses[idx].voltage_ang
            );
            checked += 1;
        } else {
            offsets.push((id, buses[idx].voltage_ang - u_angle));
        }
    }

    assert!(
        checked + offsets.len() > 0,
        "{name} [{backend:?}/{method:?}]: the fixture published no node state to compare against, \
         so this test asserted nothing but convergence"
    );

    if let Some(&(ref_id, reference)) = offsets.first() {
        for &(id, offset) in &offsets {
            assert!(
                (offset - reference).abs() < tol,
                "{name} [{backend:?}/{method:?}] node {id}: angle offset {offset} differs from node \
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

/// Sensors on all three sides of a three-winding transformer, i.e.
/// `measured_terminal_type` 6/7/8 — the types `measurements_from_pgm` used to
/// reject outright.
///
/// gridoxide models a three-winding transformer as three two-winding legs to a
/// synthesized star bus, so side k's sensor is leg k's `From` terminal. The
/// star bus itself carries no sensor and is zero-injection, which is what makes
/// the three legs determinable from side measurements alone.
#[test]
fn estimates_three_winding_transformer_side_sensors() {
    assert_estimate_matches("three_winding_transformer", JacobianBackend::Scalar, 1e-6);
}

/// An `asym_voltage_sensor` reduced to the symmetric problem by its positive
/// sequence.
///
/// The three phases here sit at 0.1, 0.2 and 0.3 radians once rotated into
/// sequence, so the positive sequence lands at exactly 0.2 with a magnitude
/// 0.33% below the phase readings — a case where the mean of the angles would
/// give the right answer for the wrong reason, but the mean of the *magnitudes*
/// would not.
#[test]
fn estimates_from_an_asymmetric_voltage_phasor() {
    assert_estimate_matches("single-node-source-asym-voltage-sensor", JacobianBackend::Scalar, 1e-6);
}

/// The same sensor without angles, which PGM reduces by the mean of the three
/// magnitudes rather than by a positive sequence — `has_angle()` requires every
/// phase to carry one.
#[test]
fn estimates_from_an_asymmetric_voltage_magnitude() {
    assert_estimate_matches(
        "single-node-source-asym-voltage-sensor-no-angle",
        JacobianBackend::Scalar,
        1e-6,
    );
}

/// Asymmetric and symmetric sensors in one document, which is the case that
/// actually matters: the symmetric path used to see only part of the
/// measurement set and therefore solved a different problem than PGM did.
///
/// The tolerance is 1e-5 rather than the 1e-6 everything else here uses, and
/// the reason is [`topology::IDEAL_CONNECTION_Y`] rather than the sensors. This
/// fixture's two lower nodes are joined by a `link`, and gridoxide stamps a
/// link at `2e5+j2e5` p.u. where power-grid-model uses `1e8+j1e8` — deliberately,
/// because `G = HᵀWH` squares the admittance and PGM's value makes the
/// `node-injection-*` fixtures come back singular. That is a 500x softer
/// connection, so gridoxide separates the two nodes by 5.2e-6 p.u. where PGM
/// separates them by 1.0e-8, and the whole solution shifts by ~3.3e-6.
///
/// Checked rather than assumed: rebuilding with PGM's `1e8` makes this fixture
/// pass at 1e-6 unchanged. The gap is the regularization, not the asymmetric
/// aggregation, which reproduces PGM's `sym_calc_param` exactly.
///
/// [`topology::IDEAL_CONNECTION_Y`]: gridoxide::topology::IDEAL_CONNECTION_Y
#[test]
fn estimates_with_mixed_symmetric_and_asymmetric_sensors() {
    assert_estimate_matches("dummy-test-sym", JacobianBackend::Scalar, 1e-5);
}

/// A current sensor whose angle is measured against the global reference.
///
/// `I = i·e^{j·i_angle}`, so the measurement is linear in the voltages and the
/// estimator reads the functional's `current` directly.
#[test]
fn estimates_from_a_global_angle_current_sensor() {
    assert_estimate_matches("global-current-sensor", JacobianBackend::Scalar, 1e-6);
}

/// The same sensor read as a *local* angle — the shift between the terminal's
/// voltage and its current.
///
/// The fixture is identical to its sibling but for `angle_measurement_type`,
/// and power-grid-model converges to a visibly different state (node 2 at
/// 1.00654 against 1.00323), so this pins that gridoxide distinguishes the two
/// frames rather than treating the field as decoration.
#[test]
fn estimates_from_a_local_angle_current_sensor() {
    assert_estimate_matches("local-current-sensor", JacobianBackend::Scalar, 1e-6);
}

/// Both frames, through the iterative-linear method as well — both fixtures
/// list it in their own `params.json`.
#[test]
fn both_methods_agree_on_current_sensors() {
    for name in ["global-current-sensor", "local-current-sensor"] {
        assert_estimate_matches_with(name, JacobianBackend::Scalar, SeMethod::IterativeLinear, 1e-6, 100);
    }
}

/// Why [`linear_start`] exists, pinned so the reason cannot be lost.
///
/// `three_winding_transformer` has `clock_12 = clock_13 = 11`, i.e. a 30° shift
/// on two of its three legs, putting the true angles at ~0.53 rad before any
/// load flows. Gauss-Newton started at zero does not merely take longer from
/// there — it converges, reports success, and returns a *different* stationary
/// point: the source node at 0.21 p.u. against its own sensor reading 1.00, at
/// an objective nine orders of magnitude worse than the true optimum.
///
/// This asserts the failure rather than just the fix, because a silent
/// convergence to the wrong basin is the dangerous shape here. If some future
/// change makes the flat start succeed too, this test should be revisited
/// deliberately, not deleted for being noisy.
#[test]
fn a_flat_start_finds_the_wrong_basin_through_a_phase_shifting_transformer() {
    let dir = fixture_dir("three_winding_transformer");
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

    let run = |seed: fn(&mut [gridoxide::types::Bus], &SeNetwork, &[Measurement])| {
        let mut buses = net.buses.clone();
        seed(&mut buses, &se_net, &measurements);
        let report = estimate(&measurements, &mut buses, &se_net, &SeOptions::default());
        (report, buses)
    };

    let (flat, flat_buses) = run(|b, _, m| gridoxide::se::nr::flat_start(b, m));
    let (linear, linear_buses) = run(gridoxide::se::nr::linear_start);

    assert_eq!(flat.status, SeStatus::Converged, "the flat start does converge — that is the problem");
    assert_eq!(linear.status, SeStatus::Converged);

    // The linear start finds the true optimum; the flat one finds a stationary
    // point that fits the data incomparably worse.
    assert!(
        linear.objective < 1e-6,
        "linear start should reach the true optimum, got J = {:.3e}",
        linear.objective
    );
    assert!(
        flat.objective > 1.0,
        "the flat start's spurious minimum should be far worse, got J = {:.3e}",
        flat.objective
    );

    // And the difference is visible in the answer, not just the objective: the
    // source node carries a direct voltage measurement of ~1.0 p.u.
    let source_node = net.node_idx[&1];
    assert!((linear_buses[source_node].voltage_mag - 0.9999926174270017).abs() < 1e-6);
    assert!(
        flat_buses[source_node].voltage_mag < 0.5,
        "expected the spurious basin to sit far from 1.0 p.u., got {}",
        flat_buses[source_node].voltage_mag
    );
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
        linear_start(&mut buses, &se_net, &measurements);
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
        linear_start(&mut buses, &se_net, &measurements);
        let report = estimate(&measurements, &mut buses, &se_net, &SeOptions::default());

        let layout = StateLayout::new(&buses, &measurements, &se_net);
        let constraints = Constraints::new(&se_net);
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

/// power-grid-model's own default method, reaching the same answers.
///
/// Its fixtures accept either algorithm against a single expected output, so
/// agreement here is agreement with power-grid-model twice over. The iteration
/// budget is larger than the Newton path's because this method converges
/// linearly rather than quadratically — trading iteration count for a much
/// cheaper iteration is the whole point of it.
#[test]
fn iterative_linear_matches_pgm() {
    for name in [
        "1os2msr",
        "1os2msr-no-angle",
        "inf-measurement-with-injection",
        "transmission-case",
        "node-injection-sensor-and-zero-injection",
    ] {
        assert_estimate_matches_with(
            name,
            JacobianBackend::Scalar,
            SeMethod::IterativeLinear,
            1e-6,
            100,
        );
    }
}

/// The two methods must agree with each other, not merely each with the
/// fixtures — a stricter statement, since it holds bus by bus rather than only
/// where power-grid-model published a value.
#[test]
fn the_two_methods_agree() {
    for name in ["1os2msr", "transmission-case"] {
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

        let solve = |method| {
            let mut buses = net.buses.clone();
            linear_start(&mut buses, &se_net, &measurements);
            let options = SeOptions { method, max_iter: 100, ..SeOptions::default() };
            let report = estimate(&measurements, &mut buses, &se_net, &options);
            assert_eq!(report.status, SeStatus::Converged, "{name} [{method:?}]: {report:?}");
            buses
        };
        let newton = solve(SeMethod::NewtonRaphson);
        let linear = solve(SeMethod::IterativeLinear);

        for (i, (a, b)) in newton.iter().zip(&linear).enumerate() {
            assert!(
                (a.voltage_mag - b.voltage_mag).abs() < 1e-6,
                "{name} bus {i}: Newton {} vs iterative-linear {}",
                a.voltage_mag,
                b.voltage_mag
            );
        }
    }
}

/// Per-node injections across a link — the assertion that decides the whole
/// zero-impedance policy.
///
/// These two fixtures publish `p`/`q` at *each* node of a linked pair: node 1
/// injecting and node 2 absorbing the same power. Both survive only because a
/// link is stamped as a branch. Merging its endpoints — the treatment gridoxide
/// uses for CGMES switches, and the one an earlier draft of this policy applied
/// here too — collapses them into a single bus whose net injection is zero, and
/// these numbers cease to exist.
///
/// A test comparing node *voltages* would pass under either treatment and so
/// would not be testing the decision at all.
#[test]
fn per_node_injections_survive_across_a_link() {
    for name in [
        "node-injection-with-injection-sensor-sym-sensors",
        "node-injection-wo-injection-sensor-sym-sensors",
    ] {
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
        linear_start(&mut buses, &se_net, &measurements);
        let report = estimate(&measurements, &mut buses, &se_net, &SeOptions::default());
        assert_eq!(report.status, SeStatus::Converged, "{name}: {report:?}");

        let nodes = expected["data"]["node"].as_array().expect("node output");
        let mut checked = 0;
        for node in nodes {
            let id = node["id"].as_u64().expect("node id");
            let idx = net.node_idx[&id];
            let (Some(p), Some(q)) = (node["p"].as_f64(), node["q"].as_f64()) else { continue };

            // power-grid-model's node injection: every appliance at the bus,
            // which for gridoxide means the bus injection plus the structural
            // source and shunt contributions. Same composition as
            // `Target::NodeInjection`'s measurement function.
            let probe = |kind| Measurement {
                kind,
                target: Target::NodeInjection(idx),
                value: 0.0,
                sigma: 1.0,
            };
            let rows = vec![
                probe(MeasurementKind::ActivePower),
                probe(MeasurementKind::ReactivePower),
            ];
            let h = measurement_functions(&rows, &buses, &se_net);

            assert!(
                (h[0] * S_BASE_VA - p).abs() < 1e-3,
                "{name} node {id}: p = {} W, PGM says {p}",
                h[0] * S_BASE_VA
            );
            assert!(
                (h[1] * S_BASE_VA - q).abs() < 1e-3,
                "{name} node {id}: q = {} W, PGM says {q}",
                h[1] * S_BASE_VA
            );
            checked += 1;
        }
        assert!(checked >= 2, "{name}: expected both nodes of the link to be checked");

        // And the endpoints really are distinct buses, asserted directly rather
        // than inferred from the numbers above.
        let raw = common::load_json(&dir.join("input.json"));
        let link = &raw["data"]["link"].as_array().expect("link")[0];
        assert_ne!(
            net.node_idx[&link["from_node"].as_u64().unwrap()],
            net.node_idx[&link["to_node"].as_u64().unwrap()],
            "{name}: a merge here would erase the injections just checked"
        );
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
