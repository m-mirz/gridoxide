//! The three-phase measurement model, checked against the Y-bus it sits beside.
//!
//! `SeNetwork::from_3ph` reads each branch's 3×3 blocks a second time to build
//! the estimator's own view of what a terminal measures. The Y-bus is built from
//! the same blocks by `build_ybus_3ph`. If the two ever disagree the estimate
//! converges confidently to the wrong answer — the failure mode
//! `tests/measurement_residual_test.rs` exists to catch on the symmetric side,
//! and this is its phase-domain counterpart.
//!
//! The check does not need a reference implementation. Kirchhoff's law is enough:
//! at a bus with nothing else attached, the currents its branches carry must sum
//! to the injection its own Y-bus row states. One of those comes from the
//! terminal functionals, the other from the Y-bus, so agreement is a statement
//! about the two descriptions rather than about either one being right.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use num_complex::Complex;

use gridoxide::branch_flow::Terminal;
use gridoxide::measurement::Target;
use gridoxide::network::{build_ybus_3ph, stamp_shunts_3ph, stamp_transformers_3ph};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_3ph, pgm_to_3ph_network, pgm_transformers_3ph};
use gridoxide::se::SeNetwork;

const S_BASE_VA: f64 = 1e6;

fn fixture(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

/// Builds the phase-domain network and estimator model for a PGM document.
fn load_3ph(path: &str) -> (SeNetwork, usize) {
    let dir = fixture(path);
    let input = common::load_pgm_input(&dir.join("input.json"));
    let id_to_idx = node_id_to_idx(&input);
    let transformers = pgm_transformers_3ph(&input, &id_to_idx, S_BASE_VA);
    let shunts = pgm_shunts_3ph(&input, &id_to_idx, S_BASE_VA);

    let input2 = common::load_pgm_input(&dir.join("input.json"));
    let (buses, lines, _) = pgm_to_3ph_network(input2, S_BASE_VA, 50.0);
    let n_nodes = buses.len() / 3;

    let mut ybus = build_ybus_3ph(n_nodes, &lines);
    stamp_transformers_3ph(&mut ybus, &transformers);
    stamp_shunts_3ph(&mut ybus, &shunts);

    // The metadata a state-estimation model needs beyond the Y-bus. Source
    // branches are the ones `pgm_to_3ph_network` synthesizes, appended after the
    // document's own lines.
    let n_doc_lines = lines.len() - input.data.source.iter().filter(|s| s.status != 0).count();
    let source_branch_idx: HashMap<u64, usize> = input
        .data
        .source
        .iter()
        .filter(|s| s.status != 0)
        .enumerate()
        .map(|(i, s)| (s.id, n_doc_lines + i))
        .collect();

    let se_net = SeNetwork::from_3ph(
        ybus.finish(),
        &lines,
        &transformers,
        &shunts,
        &source_branch_idx,
        &vec![false; n_nodes],
    );
    (se_net, n_nodes)
}

/// Every terminal functional agrees with the Y-bus about the current it carries.
///
/// Summed over the branches meeting a bus, the terminal currents must equal that
/// bus's own Y-bus row applied to the state — Kirchhoff's law, with the two
/// sides computed from the estimator's model and from the Y-bus respectively.
///
/// Shunts are excluded from the comparison by summing only at buses carrying
/// none, since a shunt sits on the Y-bus diagonal without being any branch's
/// terminal.
#[test]
fn three_phase_terminal_functionals_agree_with_the_ybus() {
    for path in [
        "tests/data/pgm/powerflow/asymmetric/line",
        "tests/data/pgm/powerflow/asymmetric/transformer",
        "tests/data/pgm/powerflow/asymmetric/transmission-case",
        // A document with a `link`. The phase domain used to refuse one
        // outright, so its 3x3 blocks had never been checked against the Y-bus
        // they are supposed to describe.
        "tests/data/pgm/state_estimation/dummy-test-sym",
    ] {
        let (net, n_nodes) = load_3ph(path);
        let n = net.ybus.n();

        // An arbitrary unbalanced state: nothing here should hold only for a
        // balanced one.
        let v: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phase = (i % 3) as f64;
                Complex::from_polar(
                    1.0 + 0.03 * ((i / 3) as f64 % 5.0) - 0.02 * phase,
                    -phase * std::f64::consts::TAU / 3.0 + 0.05 * ((i / 3) as f64 % 7.0),
                )
            })
            .collect();

        // Which branch terminals land on each bus.
        let mut incident: Vec<Vec<(usize, Terminal)>> = vec![Vec::new(); n];
        for b in 0..net.terminals.len() {
            for t in [Terminal::From, Terminal::To] {
                incident[net.terminals[b][t as usize].at].push((b, t));
            }
        }

        let mut compared = 0;
        for bus in 0..n {
            if net.shunt_y[bus].norm() > 0.0 || incident[bus].is_empty() {
                continue;
            }
            // From the estimator's model: the terminal currents leaving the bus.
            let from_terminals: Complex<f64> = incident[bus]
                .iter()
                .map(|&(b, t)| {
                    net.functional(Target::BranchTerminal { branch: b, terminal: t })
                        .expect("terminal resolves")
                        .current(&v)
                })
                .sum();
            // From the Y-bus: the same current, as that row applied to the state.
            let from_ybus: Complex<f64> =
                net.ybus.row(bus).iter().map(|&(k, y)| y * v[k]).sum();

            // Relative: a stiff source branch carries currents of order 1e7
            // per-unit, where an absolute 1e-9 would be asking for more than
            // f64 has.
            let scale = from_ybus.norm().max(1.0);
            assert!(
                (from_terminals - from_ybus).norm() <= 1e-12 * scale,
                "{path} bus {bus} (node {}, phase {}): terminals give {from_terminals}, \
                 the Y-bus gives {from_ybus}",
                bus / 3,
                bus % 3
            );
            compared += 1;
        }
        assert!(
            compared >= n_nodes,
            "{path}: expected to compare at least one bus per node, got {compared}"
        );
    }
}

/// A three-phase terminal functional couples all six phasors of its branch, and
/// the coupling between *phases* is real rather than three decoupled circuits.
///
/// This is what makes the phase domain a different problem rather than three
/// copies of the scalar one, and it is worth asserting rather than assuming: a
/// conversion that quietly produced three independent single-phase branches
/// would still pass a balanced-state check.
#[test]
fn a_three_phase_terminal_couples_its_phases() {
    let (net, _) = load_3ph("tests/data/pgm/powerflow/asymmetric/transmission-case");

    let mut coupled = 0;
    for b in 0..net.terminals.len() {
        let f = &net.terminals[b][Terminal::From as usize];
        assert_eq!(
            f.coefficients.len(),
            6,
            "branch {b}: three phases at each of two ends"
        );
        // A coefficient on a different phase than the terminal's own is a
        // cross-phase term. It vanishes wherever the zero and positive sequences
        // coincide, so this counts branches rather than requiring every one.
        if f
            .coefficients
            .iter()
            .any(|&(k, c)| k % 3 != f.at % 3 && c.norm() > 1e-12)
        {
            coupled += 1;
        }
    }
    assert!(
        coupled > 0,
        "no branch couples its phases — the conversion has produced three \
         independent single-phase networks rather than a three-phase one"
    );
}

/// How many rotational symmetries the phase-domain measurement set has.
///
/// This is the question the plan flagged as needing an experiment rather than a
/// design. A network measured only in magnitudes and powers is invariant under a
/// global rotation, and `StateLayout` pins exactly one angle to remove it. But if
/// the three phases were independent circuits — which they become wherever the
/// zero and positive sequences coincide — there would be *three* such rotations,
/// one per phase, with only one of them pinned. Two undetermined directions, in a
/// gain matrix `mask_untouched` cannot help with because every column is
/// structurally touched.
///
/// Asked directly rather than through a rank count. A rotation is a symmetry
/// exactly when it leaves every measurement function unchanged, and that is
/// exact arithmetic on `h(x)` — where a rank check on this network would be
/// measuring its conditioning instead. Its source branch carries an admittance
/// of order 1e7, so the gain matrix spans some fourteen decades and
/// `observability::RANK_TOLERANCE` cannot see the voltage rows at all.
///
/// The answer is one, and the reason is gridoxide's own network model rather
/// than the fixture's data: every source expands into a virtual slack bus behind
/// a *sequence-parameterised* impedance (`source_impedance_pu_seq`), whose zero
/// and positive sequences differ by `z01_ratio`. That branch couples the phases
/// at the one place every energised component is reachable from, so a per-phase
/// rotation is not a symmetry even when every line in the network is balanced.
///
/// Note the measurement set does the coupling as much as the network does: it is
/// a *flow* through that branch that depends on all six of its phasors. Given
/// only voltage magnitudes the three rotations separate again — see
/// [`magnitudes_alone_leave_the_phase_relationship_undetermined`], which is the
/// same question answered the other way for a set carrying no flows at all.
#[test]
fn only_the_global_rotation_is_a_symmetry_of_the_phase_domain() {
    use gridoxide::measurement::{Measurement, MeasurementKind};
    use gridoxide::se::measurement_functions;
    use gridoxide::types::{Bus, BusType};

    let (net, _) = load_3ph("tests/data/pgm/powerflow/asymmetric/transmission-case");
    let n = net.ybus.n();

    let state = |rotate: &dyn Fn(usize) -> f64| -> Vec<Bus> {
        (0..n)
            .map(|i| Bus {
                idx: i,
                bus_type: BusType::PQ,
                voltage_mag: 1.0 + 0.02 * ((i / 3) as f64 % 4.0),
                voltage_ang: -((i % 3) as f64) * std::f64::consts::TAU / 3.0
                    + 0.03 * ((i / 3) as f64 % 5.0)
                    + rotate(i),
                p_spec: 0.0,
                q_spec: 0.0,
                q_min: f64::NEG_INFINITY,
                q_max: f64::INFINITY,
                u_rated: 10000.0,
                zip_terms: Vec::new(),
            })
            .collect()
    };

    // Magnitudes and branch flows only: no angle anywhere, so any rotation that
    // is a symmetry leaves the whole set unchanged.
    let mut measurements: Vec<Measurement> = (0..n)
        .map(|b| Measurement {
            kind: MeasurementKind::VoltageMagnitude,
            target: Target::Bus(b),
            value: 0.0,
            sigma: 0.01,
        })
        .collect();
    for b in 0..net.terminals.len() {
        for t in [Terminal::From, Terminal::To] {
            for kind in [MeasurementKind::ActivePower, MeasurementKind::ReactivePower] {
                measurements.push(Measurement {
                    kind,
                    target: Target::BranchTerminal { branch: b, terminal: t },
                    value: 0.0,
                    sigma: 0.01,
                });
            }
        }
    }

    const ALPHA: f64 = 0.17;
    let base = measurement_functions(&measurements, &state(&|_| 0.0), &net);
    let global = measurement_functions(&measurements, &state(&|_| ALPHA), &net);
    let phase_a =
        measurement_functions(&measurements, &state(&|i| if i % 3 == 0 { ALPHA } else { 0.0 }), &net);

    // Relative, because the stiff source branch carries currents of order 1e7
    // and an absolute comparison there is measuring f64's mantissa.
    let worst = |other: &[f64]| {
        base.iter()
            .zip(other)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
            .fold(0.0f64, f64::max)
    };

    assert!(
        worst(&global) < 1e-8,
        "rotating every phase together must leave the measurements untouched — it is the \
         symmetry `StateLayout` pins a reference to remove; worst change {}",
        worst(&global)
    );
    assert!(
        worst(&phase_a) > 1e-3,
        "rotating phase a alone must change the measurements. If it does not, the three \
         phases are independent circuits with three separate rotational symmetries, and \
         `StateLayout` removes only one of them — leaving two undetermined directions that \
         no structural check would catch. Worst change {}",
        worst(&phase_a)
    );
}

/// The gate for asymmetric state estimation: a phase-domain estimate of
/// power-grid-model's own `transmission-case`, against the answer it published
/// for that network solved asymmetrically.
///
/// The fixture's sensors are symmetric, which is the case worth doing first —
/// it separates "does the phase-domain model solve" from "are asymmetric sensors
/// read correctly", and only the first is in question here. Its 11 voltage and
/// 24 power sensors describe all three phases at once, and both of their
/// per-unit bases carry over unchanged: line-to-line over `u_rated` and
/// line-to-neutral over `u_rated/√3` are the same number for a balanced set, as
/// are a three-phase total over `s_base` and a per-phase value over `s_base/3`.
#[test]
fn estimates_transmission_case_in_the_phase_domain() {
    use gridoxide::measurement::measurements_from_pgm_3ph;
    use gridoxide::pgm::pgm_3ph_maps;
    use gridoxide::se::nr::{estimate, linear_start, SeOptions, SeStatus};

    let dir = fixture("tests/data/pgm/state_estimation/transmission-case");
    let input = common::load_pgm_input(&dir.join("input.json"));
    let expected = common::load_json(&dir.join("asym_output.json"));

    let maps = pgm_3ph_maps(&input);
    let id_to_idx = node_id_to_idx(&input);
    let transformers = pgm_transformers_3ph(&input, &id_to_idx, S_BASE_VA);
    let shunts = pgm_shunts_3ph(&input, &id_to_idx, S_BASE_VA);
    let (buses, lines, _) = pgm_to_3ph_network(
        common::load_pgm_input(&dir.join("input.json")),
        S_BASE_VA,
        50.0,
    );

    let mut ybus = build_ybus_3ph(buses.len() / 3, &lines);
    stamp_transformers_3ph(&mut ybus, &transformers);
    stamp_shunts_3ph(&mut ybus, &shunts);
    let se_net = SeNetwork::from_3ph(
        ybus.finish(),
        &lines,
        &transformers,
        &shunts,
        &maps.source_branch_idx,
        &maps.zero_injection,
    );

    let u_rated = |bus: usize| buses[bus].u_rated;
    let measurements = measurements_from_pgm_3ph(&input, &maps, S_BASE_VA, &u_rated)
        .expect("measurements");
    assert!(
        measurements.len() > 100,
        "11 voltage and 24 power sensors over three phases should give a large set, got {}",
        measurements.len()
    );

    let mut state = buses.clone();
    linear_start(&mut state, &se_net, &measurements);
    let report = estimate(
        &measurements,
        &mut state,
        &se_net,
        &SeOptions { max_iter: 40, ..SeOptions::default() },
    );
    assert_eq!(report.status, SeStatus::Converged, "{report:?}");

    // Magnitudes are absolute; angles only up to one rotation shared by every
    // phase-bus, since nothing here measures an angle.
    let mut offsets = Vec::new();
    let mut checked = 0;
    for node in expected["data"]["node"].as_array().expect("node output") {
        let id = node["id"].as_u64().expect("node id");
        let k = maps.node_idx[&id];
        let u_pu = node["u_pu"].as_array().expect("per-phase u_pu");
        let u_angle = node["u_angle"].as_array().expect("per-phase u_angle");
        for p in 0..3 {
            let want = u_pu[p].as_f64().expect("u_pu");
            let got = state[3 * k + p].voltage_mag;
            assert!(
                (got - want).abs() < 1e-6,
                "node {id} phase {p}: |V| = {got}, PGM says {want}"
            );
            offsets.push((id, p, state[3 * k + p].voltage_ang - u_angle[p].as_f64().expect("angle")));
            checked += 1;
        }
    }
    assert_eq!(checked, 33, "11 nodes times three phases");

    let (ref_id, ref_p, reference) = offsets[0];
    for &(id, p, offset) in &offsets {
        assert!(
            (offset - reference).abs() < 1e-6,
            "node {id} phase {p}: angle offset {offset} differs from node {ref_id} phase \
             {ref_p}'s {reference} — a uniform offset is a reference convention, a varying \
             one is a wrong estimate"
        );
    }
}


/// Asymmetric sensors describing their three phases separately.
///
/// The `transmission-case` gate above drives the phase domain from *symmetric*
/// sensors, which reach it by replication and a rotation. This one carries a
/// value per phase already, which is the case the asymmetric path exists for.
///
/// A single node behind a source, so the measurement determines the answer
/// outright and power-grid-model reports back exactly what the sensor read.
/// That makes it a check on the *conversion* — per-unit base, phase selection,
/// angle handling — rather than on the estimator.
#[test]
fn estimates_from_an_asymmetric_voltage_phasor_per_phase() {
    use gridoxide::measurement::measurements_from_pgm_3ph;
    use gridoxide::pgm::pgm_3ph_maps;
    use gridoxide::se::nr::{estimate, linear_start, SeOptions, SeStatus};

    let name = "single-node-source-asym-voltage-sensor";
    let dir = fixture("tests/data/pgm/state_estimation").join(name);
    let input = common::load_pgm_input(&dir.join("input.json"));
    let expected = common::load_json(&dir.join("asym_output.json"));

    let maps = pgm_3ph_maps(&input);
    let id_to_idx = node_id_to_idx(&input);
    let transformers = pgm_transformers_3ph(&input, &id_to_idx, S_BASE_VA);
    let shunts = pgm_shunts_3ph(&input, &id_to_idx, S_BASE_VA);
    let (buses, lines, _) = pgm_to_3ph_network(
        common::load_pgm_input(&dir.join("input.json")),
        S_BASE_VA,
        50.0,
    );
    let mut ybus = build_ybus_3ph(buses.len() / 3, &lines);
    stamp_transformers_3ph(&mut ybus, &transformers);
    stamp_shunts_3ph(&mut ybus, &shunts);
    let se_net = SeNetwork::from_3ph(
        ybus.finish(),
        &lines,
        &transformers,
        &shunts,
        &maps.source_branch_idx,
        &maps.zero_injection,
    );

    let u_rated = |bus: usize| buses[bus].u_rated;
    let measurements =
        measurements_from_pgm_3ph(&input, &maps, S_BASE_VA, &u_rated).expect("measurements");
    assert_eq!(measurements.len(), 6, "three phases, each a magnitude and an angle");

    let mut state = buses.clone();
    linear_start(&mut state, &se_net, &measurements);
    let report = estimate(
        &measurements,
        &mut state,
        &se_net,
        &SeOptions { max_iter: 40, ..SeOptions::default() },
    );
    assert_eq!(report.status, SeStatus::Converged, "{name}: {report:?}");

    for node in expected["data"]["node"].as_array().expect("node output") {
        let k = maps.node_idx[&node["id"].as_u64().expect("node id")];
        let u = node["u"].as_array().expect("per-phase u");
        let u_angle = node["u_angle"].as_array().expect("per-phase u_angle");
        // Published `u` is line-to-neutral volts; the state is per-unit on the
        // line-to-neutral base.
        let base = buses[3 * k].u_rated / 3.0f64.sqrt();
        for p in 0..3 {
            let want = u[p].as_f64().expect("u") / base;
            assert!(
                (state[3 * k + p].voltage_mag - want).abs() < 1e-6,
                "{name} phase {p}: |V| = {}, PGM says {want}",
                state[3 * k + p].voltage_mag
            );
            let want_ang = u_angle[p].as_f64().expect("angle");
            assert!(
                (state[3 * k + p].voltage_ang - want_ang).abs() < 1e-6,
                "{name} phase {p}: angle = {}, PGM says {want_ang}",
                state[3 * k + p].voltage_ang
            );
        }
    }
}

/// Magnitudes alone do not determine a phase relationship, and that is a real
/// limit of gridoxide's source model rather than a defect in the conversion.
///
/// This refines the finding in
/// [`only_the_global_rotation_is_a_symmetry_of_the_phase_domain`], which is true
/// of a set containing *flows*: there the source impedance couples the phases,
/// because a flow through it depends on all six of its phasors. Given only
/// voltage magnitudes nothing does, and the three per-phase rotations are three
/// separate symmetries where `StateLayout` removes one. Two undetermined
/// directions, and the gain matrix is singular — correctly, since the question
/// has no answer.
///
/// power-grid-model does answer it, because its source is a *boundary
/// condition*: a fixed, balanced three-phase voltage. gridoxide's is an unknown
/// like any other, behind a synthesized impedance — the same modelling
/// difference that leaves `SeReport::unconstrained` naming a virtual bus per
/// source on the symmetric side.
///
/// The obvious fix is wrong, and it was tried. gridoxide *builds* that virtual
/// bus balanced — one `u_ref` across three phases at 0/−120/+120 — so
/// constraining its angles to differ by exactly ±120° looks like free
/// information, and it removes exactly the two directions in question. It also
/// contradicts the data.
///
/// The counterexample is this fixture's own sibling,
/// [`estimates_from_an_asymmetric_voltage_phasor_per_phase`]. Its sensor reads
/// three phases whose sequence angles are 0.1, 0.2 and 0.3 — deliberately
/// unbalanced — and its node carries no appliance, so the injection there is
/// zero, so the current through the source branch is zero, so
/// `V_virtual = V_node` exactly. The virtual bus is therefore *as unbalanced as
/// the measurement says the node is*, and forcing it balanced moves that
/// fixture's answer from 0.1 to 0.2, the positive-sequence compromise.
///
/// So the balance is a property of the *initial state* gridoxide synthesizes,
/// not of the equivalent it represents: a real network behind a source can be
/// unbalanced, and one of power-grid-model's own fixtures is. The two
/// directions here are genuinely undetermined, and reporting singular is the
/// correct answer rather than a missing feature. power-grid-model answers
/// instead because it has no source-internal bus to be undetermined about.
#[test]
fn magnitudes_alone_leave_the_phase_relationship_undetermined() {
    use gridoxide::measurement::measurements_from_pgm_3ph;
    use gridoxide::pgm::pgm_3ph_maps;
    use gridoxide::se::nr::{estimate, linear_start, SeOptions, SeStatus};

    let dir = fixture("tests/data/pgm/state_estimation")
        .join("single-node-source-asym-voltage-sensor-no-angle");
    let input = common::load_pgm_input(&dir.join("input.json"));
    let maps = pgm_3ph_maps(&input);
    let id_to_idx = node_id_to_idx(&input);
    let transformers = pgm_transformers_3ph(&input, &id_to_idx, S_BASE_VA);
    let shunts = pgm_shunts_3ph(&input, &id_to_idx, S_BASE_VA);
    let (buses, lines, _) = pgm_to_3ph_network(
        common::load_pgm_input(&dir.join("input.json")),
        S_BASE_VA,
        50.0,
    );
    let mut ybus = build_ybus_3ph(buses.len() / 3, &lines);
    stamp_transformers_3ph(&mut ybus, &transformers);
    stamp_shunts_3ph(&mut ybus, &shunts);
    let se_net = SeNetwork::from_3ph(
        ybus.finish(),
        &lines,
        &transformers,
        &shunts,
        &maps.source_branch_idx,
        &maps.zero_injection,
    );

    let u_rated = |bus: usize| buses[bus].u_rated;
    let measurements =
        measurements_from_pgm_3ph(&input, &maps, S_BASE_VA, &u_rated).expect("measurements");
    assert_eq!(measurements.len(), 3, "three magnitudes, no angle anywhere");

    let mut state = buses.clone();
    linear_start(&mut state, &se_net, &measurements);
    let report = estimate(
        &measurements,
        &mut state,
        &se_net,
        &SeOptions { max_iter: 40, ..SeOptions::default() },
    );
    assert_eq!(
        report.status,
        SeStatus::Singular,
        "expected the phase relationship to be undetermined here. If this now converges, \
         something has supplied the two missing directions — check it is the balanced-source \
         constraint described above and not an accident"
    );
    assert!(
        report.unconstrained.is_empty(),
        "the deficiency is rank, not an untouched column: every unknown is reached by some \
         row, and two combinations of them are still undetermined"
    );
}

// ---------------------------------------------------------------------------
// The analyses, in the phase domain
// ---------------------------------------------------------------------------
//
// Observability, bad-data detection, the zero-injection constraints and the
// batch solver all take `(&[Measurement], &[Bus], &SeNetwork, &StateLayout,
// &Constraints)`, so they are domain-agnostic by signature and were *claimed*
// to carry over. A claim by signature is not a gate: the reference handling is
// exactly where a symmetric assumption would hide, since three per-phase
// rotations are three symmetries where `StateLayout` removes one.

/// The whole case, through the single entry point rather than the eight calls
/// above it.
fn case_3ph(path: &str) -> gridoxide::se::case::SeCase {
    let input = common::load_pgm_input(&fixture(path).join("input.json"));
    gridoxide::se::case::SeCase::from_pgm_3ph(&input, S_BASE_VA, 50.0)
        .expect("this fixture uses no unsupported component")
}

fn case_1ph(path: &str) -> gridoxide::se::case::SeCase {
    let input = common::load_pgm_input(&fixture(path).join("input.json"));
    gridoxide::se::case::SeCase::from_pgm(&input, S_BASE_VA, 50.0).expect("builds")
}

/// The entry point builds what the eight-call assembly built.
///
/// Every test above stands the phase-domain model up by hand, in an order that
/// matters — and that assembly is why the estimator was unreachable from
/// anything but this file. If the door and the long way round ever disagree,
/// everything below is testing a different network from everything above.
#[test]
fn the_entry_point_agrees_with_the_assembly() {
    let dir = fixture("tests/data/pgm/state_estimation/transmission-case");
    let input = common::load_pgm_input(&dir.join("input.json"));

    // The canonical assembly, as `estimates_transmission_case_in_the_phase_domain`
    // performs it: `pgm_3ph_maps` for the source branches and the zero
    // injections, not the simplified `load_3ph` the Y-bus tests use.
    let maps = gridoxide::pgm::pgm_3ph_maps(&input);
    let id_to_idx = node_id_to_idx(&input);
    let transformers = pgm_transformers_3ph(&input, &id_to_idx, S_BASE_VA);
    let shunts = pgm_shunts_3ph(&input, &id_to_idx, S_BASE_VA);
    let (buses, lines, _) = pgm_to_3ph_network(input.clone(), S_BASE_VA, 50.0);
    let mut ybus = build_ybus_3ph(buses.len() / 3, &lines);
    stamp_transformers_3ph(&mut ybus, &transformers);
    stamp_shunts_3ph(&mut ybus, &shunts);
    let by_hand = SeNetwork::from_3ph(
        ybus.finish(),
        &lines,
        &transformers,
        &shunts,
        &maps.source_branch_idx,
        &maps.zero_injection,
    );

    let case = case_3ph("tests/data/pgm/state_estimation/transmission-case");

    assert_eq!(case.phases, 3);
    assert_eq!(case.buses.len(), buses.len());
    assert_eq!(case.network.ybus.n(), by_hand.ybus.n());
    assert_eq!(case.network.terminals.len(), by_hand.terminals.len());
    assert_eq!(case.network.zero_injection, by_hand.zero_injection);
    assert_eq!(case.network.energized, by_hand.energized);
    assert_eq!(case.network.source_branches, by_hand.source_branches);
    for (a, b) in case.network.shunt_y.iter().zip(&by_hand.shunt_y) {
        assert!((a - b).norm() < 1e-15);
    }

    let u_rated = |bus: usize| buses[bus].u_rated;
    let expected =
        gridoxide::measurement::measurements_from_pgm_3ph(&input, &maps, S_BASE_VA, &u_rated)
            .unwrap();
    assert_eq!(case.measurements.len(), expected.len());

    // A node's three phases, and the arithmetic that finds them.
    let (id, idx) = case.nodes()[0];
    for phase in 0..3 {
        assert_eq!(case.bus_of(id, phase), Some(3 * idx + phase));
    }
}

/// Observability carries over, and reports the phase domain's own symmetries.
///
/// The interesting part is not the rank but *which* directions are missing. A
/// source's virtual bus is unobservable when nothing measures the source's own
/// power — the same statement as on the symmetric side — and in the phase
/// domain there are three of them per source rather than one, because the
/// virtual node is three buses like any other.
#[test]
fn observability_carries_over_to_the_phase_domain() {
    use gridoxide::se::constraints::Constraints;
    use gridoxide::se::jacobian::StateLayout;
    use gridoxide::se::observability;

    let case = case_3ph("tests/data/pgm/state_estimation/transmission-case");
    let layout = StateLayout::new(&case.buses, &case.measurements, &case.network);
    let constraints = Constraints::new(&case.network);
    let report =
        observability::analyze(&case.measurements, &case.buses, &case.network, &layout, &constraints);

    assert!(report.n_unknowns > 0);
    assert!(report.rank <= report.n_unknowns);

    // Every undetermined unknown sits on a synthesized bus, not a physical one:
    // this fixture measures its physical network well enough to determine it,
    // and a phase-domain estimate that could not say so would be reporting a
    // defect of the model rather than of the data.
    let n_physical = 3 * case.nodes().len();
    for unknown in report.unobservable.iter().chain(&report.structurally_unmeasured) {
        assert!(
            unknown.bus >= n_physical,
            "physical bus {} ({:?}) came back undetermined",
            unknown.bus,
            unknown.quantity
        );
    }

    // The symmetric run of the same document reaches the same verdict about its
    // physical network, which is what "carries over" has to mean.
    let sym = case_1ph("tests/data/pgm/state_estimation/transmission-case");
    let sym_layout = StateLayout::new(&sym.buses, &sym.measurements, &sym.network);
    let sym_report = observability::analyze(
        &sym.measurements,
        &sym.buses,
        &sym.network,
        &sym_layout,
        &Constraints::new(&sym.network),
    );
    for unknown in sym_report.unobservable.iter().chain(&sym_report.structurally_unmeasured) {
        assert!(unknown.bus >= sym.nodes().len());
    }
}

/// Bad-data detection carries over, and names the measurement that was
/// corrupted.
///
/// This is the claim that had never been checked in the phase domain, and it
/// holds — but only once the estimate underneath it does. See
/// [`a_large_error_diverges_and_says_so`] for the part that does not.
#[test]
fn bad_data_detection_finds_the_corrupted_measurement() {
    use gridoxide::se::bad_data::{self, Candidates};
    use gridoxide::se::constraints::Constraints;
    use gridoxide::se::jacobian::StateLayout;
    use gridoxide::se::nr::{estimate, linear_start, SeOptions, SeStatus};

    let mut case = case_3ph("tests/data/pgm/state_estimation/transmission-case");

    // One phase of one node, off by fifteen of its own sigmas. Large enough to
    // be rejected, small enough that Gauss-Newton still reaches an answer to
    // reject it from — the two are not the same threshold, which is the point
    // of the test below.
    let victim = case
        .measurements
        .iter()
        .position(|m| matches!(m.kind, gridoxide::measurement::MeasurementKind::VoltageMagnitude))
        .expect("this fixture has voltage sensors");
    let sigma = case.measurements[victim].sigma;
    case.measurements[victim].value += 15.0 * sigma;

    let mut buses = case.buses.clone();
    linear_start(&mut buses, &case.network, &case.measurements);
    let report = estimate(
        &case.measurements,
        &mut buses,
        &case.network,
        &SeOptions { max_iter: 40, ..SeOptions::default() },
    );
    assert_eq!(report.status, SeStatus::Converged, "objective {:.3e}", report.objective);

    let layout = StateLayout::new(&buses, &case.measurements, &case.network);
    let bad = bad_data::analyze(
        &case.measurements,
        &report.residuals,
        &buses,
        &case.network,
        &layout,
        &Constraints::new(&case.network),
        Candidates::default(),
    );

    assert!(
        bad.rejects_at(0.05),
        "a fifteen-sigma error should be rejected, chi-squared {:.3e} on {} dof",
        bad.chi_squared,
        bad.degrees_of_freedom
    );
    assert_eq!(
        bad.suspects.first().map(|s| s.measurement),
        Some(victim),
        "the largest normalized residual should be the measurement that was corrupted"
    );
    // And it is that measurement by a margin, rather than by a hair.
    let worst = bad.suspects[0].normalized_residual;
    let next = bad.suspects.get(1).map_or(0.0, |s| s.normalized_residual);
    assert!(worst > 5.0 * next.max(1e-12), "{worst} against a runner-up of {next}");
}

/// A large enough error makes Gauss-Newton diverge — and it says so.
///
/// Found while writing the test above, which originally corrupted a measurement
/// by twenty sigmas and then blamed bad-data detection for the nonsense that
/// came back: normalized residuals of `1e24` on measurements with nothing wrong
/// with them. Bad-data detection was reporting faithfully. The *estimate* had
/// diverged to an objective of `3.5e43`, and a residual covariance means
/// nothing about a state that does not solve anything.
///
/// The cliff is not a property of the phase domain, though the phase domain
/// reaches it sooner: on this fixture the phase-domain estimate converges at
/// fifteen sigmas and diverges at twenty, while the symmetric one converges at
/// twenty and diverges at fifty. Gauss-Newton takes an undamped step, so a
/// measurement far enough from consistent throws the first step past the basin
/// and there is nothing to come back to. A line search would raise the cliff;
/// there is none.
///
/// What is gated is therefore not that it survives, but that it **reports**:
/// `MaxIterations` rather than a converged-looking answer. An estimator that
/// returned this state as an estimate would be the dangerous outcome, and the
/// bad-data report built on top of it is exactly what that danger looks like.
#[test]
fn a_large_error_diverges_and_says_so() {
    use gridoxide::se::nr::{estimate, linear_start, SeOptions, SeStatus};

    let mut case = case_3ph("tests/data/pgm/state_estimation/transmission-case");
    let victim = case
        .measurements
        .iter()
        .position(|m| matches!(m.kind, gridoxide::measurement::MeasurementKind::VoltageMagnitude))
        .unwrap();
    let sigma = case.measurements[victim].sigma;
    case.measurements[victim].value += 20.0 * sigma;

    let mut buses = case.buses.clone();
    linear_start(&mut buses, &case.network, &case.measurements);
    let report = estimate(
        &case.measurements,
        &mut buses,
        &case.network,
        &SeOptions { max_iter: 40, ..SeOptions::default() },
    );

    assert_eq!(
        report.status,
        SeStatus::MaxIterations,
        "a diverged estimate must not report success, objective {:.3e}",
        report.objective
    );
    assert!(
        report.objective > 1e6,
        "expected a diverged objective, got {:.3e}",
        report.objective
    );
}

/// A clean phase-domain estimate is not rejected.
///
/// The other half of the test above, and the one that would catch a
/// degrees-of-freedom count that forgot the phase domain has three times the
/// unknowns: an over-tight chi-squared would reject a case with nothing wrong
/// with it, which looks like a finding and is a defect.
#[test]
fn a_clean_phase_domain_estimate_is_not_rejected() {
    use gridoxide::se::bad_data::{self, Candidates};
    use gridoxide::se::constraints::Constraints;
    use gridoxide::se::jacobian::StateLayout;
    use gridoxide::se::nr::{estimate, linear_start, SeOptions};

    let case = case_3ph("tests/data/pgm/state_estimation/transmission-case");
    let mut buses = case.buses.clone();
    linear_start(&mut buses, &case.network, &case.measurements);
    let report = estimate(
        &case.measurements,
        &mut buses,
        &case.network,
        &SeOptions { max_iter: 40, ..SeOptions::default() },
    );

    let layout = StateLayout::new(&buses, &case.measurements, &case.network);
    let bad = bad_data::analyze(
        &case.measurements,
        &report.residuals,
        &buses,
        &case.network,
        &layout,
        &Constraints::new(&case.network),
        Candidates::default(),
    );
    assert!(
        !bad.rejects_at(0.05),
        "nothing is wrong with this data, chi-squared {:.3e} on {} dof, p = {:.3e}",
        bad.chi_squared,
        bad.degrees_of_freedom,
        bad.p_value
    );
    assert!(bad.degrees_of_freedom > 0);
}

/// The batch solver carries over, and agrees with a sequential loop.
///
/// It shares the estimator's own indifference to the domain — a scenario is an
/// override on a measurement index, and an index means the same thing whether a
/// bus is a node or a node's phase. What that leaves worth checking is the part
/// batching adds: a factorization reused across scenarios and threads must give
/// the answer a scenario computed alone would, and in the phase domain the
/// pattern it caches is three times the size with three times the coupling.
#[test]
fn the_batch_solver_carries_over_to_the_phase_domain() {
    use gridoxide::se::batch::{MeasurementOverride, SeBatchSolver, SeScenario};
    use gridoxide::se::nr::{estimate, linear_start, SeOptions, SeStatus};

    let case = case_3ph("tests/data/pgm/state_estimation/transmission-case");

    // Five snapshots of the same network: each nudges one voltage reading, the
    // way a sequence of measurement scans differs from one another.
    let voltages: Vec<usize> = case
        .measurements
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m.kind, gridoxide::measurement::MeasurementKind::VoltageMagnitude))
        .map(|(i, _)| i)
        .take(5)
        .collect();
    assert_eq!(voltages.len(), 5, "this fixture has voltage sensors on several phases");

    let scenarios: Vec<SeScenario> = voltages
        .iter()
        .map(|&i| {
            let m = &case.measurements[i];
            SeScenario::new(vec![MeasurementOverride::new(i).value(m.value + 2.0 * m.sigma)])
        })
        .collect();

    let options = SeOptions { max_iter: 40, ..SeOptions::default() };
    let solver = SeBatchSolver::new(options.clone());
    let batched = solver
        .estimate(&case.buses, &case.network, &case.measurements, &scenarios)
        .expect("the batch runs");
    assert_eq!(batched.len(), scenarios.len());

    for (k, scenario) in scenarios.iter().enumerate() {
        // The same scenario, estimated on its own.
        let mut measurements = case.measurements.clone();
        for ov in &scenario.overrides {
            if let Some(value) = ov.value {
                measurements[ov.measurement].value = value;
            }
        }
        let mut buses = case.buses.clone();
        linear_start(&mut buses, &case.network, &measurements);
        let alone = estimate(&measurements, &mut buses, &case.network, &options);

        assert_eq!(alone.status, SeStatus::Converged);
        assert_eq!(batched[k].report.status, SeStatus::Converged);
        for (a, b) in batched[k].buses.iter().zip(&buses) {
            assert_eq!(
                a.voltage_mag.to_bits(),
                b.voltage_mag.to_bits(),
                "scenario {k}: batching must not change a single bit of the answer"
            );
            assert_eq!(a.voltage_ang.to_bits(), b.voltage_ang.to_bits());
        }
    }
}

/// A `voltage_regulator` is ignored rather than refused.
///
/// It used to make the whole document inestimable in the phase domain. A
/// voltage regulator pins a generator's bus to `u_ref` and lets Q float, which
/// is a boundary condition for a *power flow*; an estimate treats every bus
/// voltage as an unknown to be recovered from measurements, and has no PV bus
/// for one to create. So it cannot change the answer, and refusing it made a
/// document unestimable for a reason that could not have mattered — while the
/// symmetric path read the same document without comment.
#[test]
fn a_voltage_regulator_does_not_stop_an_estimate() {
    use gridoxide::se::case::SeCase;

    let dir = fixture("tests/data/pgm/state_estimation/transmission-case");
    let raw = std::fs::read_to_string(dir.join("input.json")).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&raw).unwrap();

    // A regulator on whatever generator the document has, or on a node if it
    // has none — `pgm_3ph_maps` should not care either way.
    let data = doc.get_mut("data").unwrap().as_object_mut().unwrap();
    let regulated = data
        .get("sym_gen")
        .and_then(|g| g.as_array())
        .and_then(|g| g.first())
        .and_then(|g| g.get("id"))
        .cloned()
        .unwrap_or_else(|| data["node"].as_array().unwrap()[0]["id"].clone());
    data.insert(
        "voltage_regulator".to_string(),
        serde_json::json!([{
            "id": 999_999,
            "regulated_object": regulated,
            "status": 1,
            "u_ref": 1.0
        }]),
    );

    let input: gridoxide::pgm::PgmInput =
        serde_json::from_value(doc).expect("the document still parses");

    let with = SeCase::from_pgm_3ph(&input, S_BASE_VA, 50.0).expect("estimable");
    let without = case_3ph("tests/data/pgm/state_estimation/transmission-case");

    // And it changes nothing: same buses, same measurements, same model.
    assert_eq!(with.phases, without.phases);
    assert_eq!(with.buses.len(), without.buses.len());
    assert_eq!(with.measurements.len(), without.measurements.len());
    assert_eq!(with.network.zero_injection, without.network.zero_injection);
}

/// A `link` is modelled, and modelled as phase-transparent.
///
/// The phase domain refused a document with one until now, which is why six of
/// the committed state-estimation fixtures could not be estimated in it at all.
///
/// The Kirchhoff check above now covers a link-bearing document, but it cannot
/// catch a link that is *consistently* wrong: it compares the terminal
/// functionals against the Y-bus, and both are built from the same
/// `Transformer3PhSeq`. So the value is pinned here independently, against the
/// two things that say what a link is — `pgm::LINK_Y`, and the fact that an
/// ideal connection couples no phase to any other. power-grid-model's own
/// `link.hpp` agrees literally: `calc_param_y_asym(y_link, 0.0, y_link, 0.0,
/// 1.0)`, the same admittance in the positive and zero sequences.
#[test]
fn a_link_is_phase_transparent() {
    use gridoxide::pgm::{pgm_3ph_maps, LINK_Y};

    let dir = fixture("tests/data/pgm/state_estimation/dummy-test-sym");
    let input = common::load_pgm_input(&dir.join("input.json"));
    let maps = pgm_3ph_maps(&input);

    let link = input.data.link.first().expect("this fixture has a link");
    let branch = *maps.branch_idx.get(&link.id).expect("a link takes a branch index");
    assert_eq!(link.from_status, 1);
    assert_eq!(link.to_status, 1);

    let case = case_3ph("tests/data/pgm/state_estimation/dummy-test-sym");
    let from = maps.node_idx[&link.from_node];
    let to = maps.node_idx[&link.to_node];

    for phase in 0..3 {
        let terminal = &case.network.terminals[3 * branch + phase][0];
        assert_eq!(terminal.at, 3 * from + phase);

        // Exactly two live coefficients, both on this phase: `I_p` depends on
        // `V_from,p` and `V_to,p` and on nothing else. A link that coupled the
        // phases would show up as a third.
        //
        // The threshold is relative to the link's own admittance, and has to
        // be. The blocks are assembled by transforming sequence parameters into
        // the phase domain, so the cross-phase terms are a cancellation rather
        // than a structural zero: measured at 3e-11 against a diagonal of 2e5,
        // which is 1.5e-16 relative — round-off, and exactly what a
        // phase-transparent element should leave behind.
        let floor = 1e-9 * LINK_Y.norm();
        let live: Vec<_> =
            terminal.coefficients.iter().filter(|(_, y)| y.norm() > floor).collect();
        assert_eq!(live.len(), 2, "phase {phase}: {:?}", terminal.coefficients);

        let self_y = live.iter().find(|(bus, _)| *bus == 3 * from + phase).unwrap().1;
        let mutual_y = live.iter().find(|(bus, _)| *bus == 3 * to + phase).unwrap().1;
        assert!((self_y - LINK_Y).norm() < 1e-6 * LINK_Y.norm(), "phase {phase}: {self_y}");
        assert!((mutual_y + LINK_Y).norm() < 1e-6 * LINK_Y.norm(), "phase {phase}: {mutual_y}");
    }
}

/// A current sensor on a link is refused in the phase domain too.
///
/// A link's admittance is a chosen regularization constant, so the current
/// through one is an artifact of that choice rather than a physical quantity —
/// power-grid-model refuses it outright and the symmetric path here already
/// did. The phase domain could not, because a document with a link never got
/// that far.
#[test]
fn a_current_sensor_on_a_link_is_refused() {
    use gridoxide::measurement::{measurements_from_pgm_3ph, MeasurementError};
    use gridoxide::pgm::pgm_3ph_maps;

    let dir = fixture("tests/data/pgm/state_estimation/dummy-test-sym");
    let raw = std::fs::read_to_string(dir.join("input.json")).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let data = doc.get_mut("data").unwrap().as_object_mut().unwrap();
    let link_id = data["link"].as_array().unwrap()[0]["id"].clone();
    data.insert(
        "sym_current_sensor".to_string(),
        serde_json::json!([{
            "id": 999_998,
            "measured_object": link_id,
            "measured_terminal_type": 0,
            "angle_measurement_type": 0,
            "i_measured": 1.0,
            "i_angle_measured": 0.0,
            "i_sigma": 0.01,
            "i_angle_sigma": 0.01
        }]),
    );

    let input: gridoxide::pgm::PgmInput = serde_json::from_value(doc).unwrap();
    let maps = pgm_3ph_maps(&input);
    let (buses, _, _) = pgm_to_3ph_network(input.clone(), S_BASE_VA, 50.0);
    let u_rated = |bus: usize| buses[bus].u_rated;

    match measurements_from_pgm_3ph(&input, &maps, S_BASE_VA, &u_rated) {
        Err(MeasurementError::CurrentSensorOnLink { sensor, .. }) => assert_eq!(sensor, 999_998),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// A three-winding transformer estimates in the phase domain, against
/// power-grid-model's own published magnitudes.
///
/// It used to make a document inestimable here — the last of four components
/// that did. gridoxide models one as three two-winding legs to a synthesized
/// star node, which the symmetric path has always done; what the phase domain
/// adds is each leg's winding configuration, and the star node itself, which is
/// three buses like any other node and injects nothing.
///
/// The sensors in this fixture sit on the transformer's three *sides*
/// (`measured_terminal_type` 6, 7 and 8), so this also exercises resolving a
/// side to its leg — which returned "unknown object" while the phase domain
/// modelled no such transformer.
#[test]
fn a_three_winding_transformer_estimates_in_the_phase_domain() {
    use gridoxide::se::nr::{estimate, linear_start, SeOptions, SeStatus};

    let name = "tests/data/pgm/state_estimation/three_winding_transformer";
    let case = case_3ph(name);

    // Three physical nodes, one star node, one source: five nodes, fifteen
    // buses. The star node is the one the symmetric model also synthesizes.
    assert_eq!(case.buses.len(), 15);
    assert_eq!(case.nodes().len(), 3);
    // Three power sensors on the three sides, times three phases, plus a
    // voltage sensor's three.
    assert!(case.measurements.len() >= 12, "{}", case.measurements.len());

    let mut buses = case.buses.clone();
    linear_start(&mut buses, &case.network, &case.measurements);
    let report = estimate(
        &case.measurements,
        &mut buses,
        &case.network,
        &SeOptions { max_iter: 40, ..SeOptions::default() },
    );
    assert_eq!(report.status, SeStatus::Converged, "objective {:.3e}", report.objective);

    // Against the answer power-grid-model published for this document. The
    // magnitudes are per-phase and must each equal the balanced one.
    let expected = common::load_json(&fixture(name).join("sym_output.json"));
    for node in expected["data"]["node"].as_array().unwrap() {
        let id = node["id"].as_u64().unwrap();
        let u_pu = node["u_pu"].as_f64().unwrap();
        for phase in 0..3 {
            let bus = case.bus_of(id, phase).expect("a document node");
            assert!(
                (buses[bus].voltage_mag - u_pu).abs() < 1e-5,
                "node {id} phase {phase}: {} against power-grid-model's {u_pu}",
                buses[bus].voltage_mag
            );
        }
    }

    // The star node injects nothing, and that is asserted rather than assumed:
    // it is the one place a hard zero-injection constraint is unarguable, since
    // nothing is attached to it.
    let n_physical = 3 * case.nodes().len();
    for bus in n_physical..n_physical + 3 {
        assert!(case.network.zero_injection[bus], "the star node injects nothing");
    }
}
