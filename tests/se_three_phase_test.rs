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

    let maps = pgm_3ph_maps(&input).expect("this fixture uses no unsupported component");
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

