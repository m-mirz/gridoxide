//! `run_power_flow` and `PowerFlowOptions`: that the new front door reproduces
//! the old one exactly under default options, that each method reports through
//! its own channel, and that the initializers are interchangeable starting
//! points for the same Newton solve rather than different answers.

use std::path::{Path, PathBuf};

use gridoxide::linear::{DcApproximation, DcIslandStatus, DcOptions, LinearIslandStatus};
use gridoxide::network::{build_ybus, stamp_shunts, ShuntAdm};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches};
use gridoxide::solver::{
    IslandStatus, PowerFlowInit, PowerFlowMethod, PowerFlowOptions, SolveStatus,
};
use gridoxide::types::{Bus, BusType, Line, Transformer};
use gridoxide::{run_power_flow, run_power_flow_analysis_from_ybus};

mod common;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow")
        .join(rel)
        .join("input.json")
}

/// Loads a PGM fixture into the four pieces `run_power_flow` wants.
fn load(path: &Path) -> (Vec<Bus>, Vec<Line>, Vec<Transformer>, Vec<ShuntAdm>) {
    let input = common::load_pgm_input(path);
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    (buses, lines, transformers, shunts)
}

/// The backward-compatibility guarantee: default options must reproduce
/// `run_power_flow_analysis_from_ybus` bit for bit, because that function's
/// hard-coded `1e-6`/`20`/`Scalar`/`linear_initial_guess` *are* the defaults.
#[test]
fn default_options_reproduce_the_original_entry_point() {
    for name in ["symmetric/transmission-case", "symmetric/distribution-case"] {
        let path = fixture(name);

        let (buses, lines, transformers, shunts) = load(&path);
        let mut ybus = build_ybus(buses.len(), &lines, &transformers);
        stamp_shunts(&mut ybus, &shunts);
        let original = run_power_flow_analysis_from_ybus(buses, ybus);

        let (buses, lines, transformers, shunts) = load(&path);
        let via_options = run_power_flow(
            buses,
            &lines,
            &transformers,
            &shunts,
            PowerFlowOptions::default(),
        );

        assert_eq!(
            original.stats.iterations(),
            via_options.stats.iterations(),
            "{name}: iteration count differs"
        );
        for (a, b) in original.buses.iter().zip(&via_options.buses) {
            assert_eq!(a.voltage_mag.to_bits(), b.voltage_mag.to_bits(), "{name}: |V| differs");
            assert_eq!(a.voltage_ang.to_bits(), b.voltage_ang.to_bits(), "{name}: angle differs");
        }
        assert!(via_options.dc.is_none() && via_options.linear.is_none());
    }
}

/// Each method reports through its own channel and leaves the others empty —
/// the `islands`/`stats` vocabulary belongs to Newton, and the direct methods
/// deliberately do not borrow it.
#[test]
fn each_method_reports_through_its_own_channel() {
    let path = fixture("symmetric/transmission-case");

    let (buses, lines, transformers, shunts) = load(&path);
    let newton = run_power_flow(buses, &lines, &transformers, &shunts, PowerFlowOptions::default());
    assert!(!newton.islands.is_empty());
    assert!(newton.stats.iterations() > 0);
    assert!(newton.dc.is_none() && newton.linear.is_none());

    let (buses, lines, transformers, shunts) = load(&path);
    let dc = run_power_flow(
        buses,
        &lines,
        &transformers,
        &shunts,
        PowerFlowOptions { method: PowerFlowMethod::Dc, ..Default::default() },
    );
    let solution = dc.dc.as_ref().expect("DC must report a DcSolution");
    assert!(solution.islands.iter().all(|i| i.status == DcIslandStatus::Solved));
    assert_eq!(dc.stats.status, SolveStatus::Converged);
    // A direct solve takes no iterations, and says so rather than claiming one.
    assert_eq!(dc.stats.iterations(), 0);
    assert!(dc.islands.is_empty() && dc.linear.is_none());

    let (buses, lines, transformers, shunts) = load(&path);
    let linear = run_power_flow(
        buses,
        &lines,
        &transformers,
        &shunts,
        PowerFlowOptions { method: PowerFlowMethod::LinearImpedance, ..Default::default() },
    );
    let solution = linear.linear.as_ref().expect("linear must report a LinearReport");
    assert!(solution.islands.iter().all(|i| i.status == LinearIslandStatus::Solved));
    assert!(linear.islands.is_empty() && linear.dc.is_none());
}

/// `PowerFlowOptions::dc` must actually reach the DC solver rather than being
/// dropped somewhere in the plumbing.
#[test]
fn dc_options_reach_the_solver() {
    let path = fixture("symmetric/transmission-case");
    let solve = |dc: DcOptions| {
        let (buses, lines, transformers, shunts) = load(&path);
        let opts = PowerFlowOptions { method: PowerFlowMethod::Dc, dc, ..Default::default() };
        let report = run_power_flow(buses, &lines, &transformers, &shunts, opts);
        report.buses.iter().map(|b| b.voltage_ang).collect::<Vec<_>>()
    };

    // Uniform r/x on this fixture means the two approximations rescale every
    // angle by the same factor, so the *angles* move even though the flows do
    // not — which is exactly what makes this a usable probe that the option
    // was threaded through.
    let ignore_r = solve(DcOptions::default());
    let ignore_g =
        solve(DcOptions { approximation: DcApproximation::IgnoreG, ..Default::default() });
    let moved = ignore_r
        .iter()
        .zip(&ignore_g)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(moved > 1e-9, "DcOptions did not reach the solver: angles identical");
}

/// Every initializer that converges must land on the *same* solution — they
/// are different starting points for one problem, not different answers to it.
///
/// Iteration counts are deliberately not ranked. Newton's convergence is
/// quadratic, so the final step dominates and a better start does not
/// reliably mean fewer iterations; asserting an ordering would be asserting a
/// coincidence.
#[test]
fn every_converging_initializer_reaches_the_same_solution() {
    for name in ["symmetric/transmission-case", "symmetric/distribution-case"] {
        let path = fixture(name);
        let mut reference: Option<Vec<(f64, f64)>> = None;
        let mut converged = 0;

        for init in [PowerFlowInit::Flat, PowerFlowInit::LinearImpedance, PowerFlowInit::Dc] {
            let (buses, lines, transformers, shunts) = load(&path);
            let opts = PowerFlowOptions { init, ..Default::default() };
            let report = run_power_flow(buses, &lines, &transformers, &shunts, opts);
            if report.stats.status != SolveStatus::Converged {
                continue;
            }
            converged += 1;

            let solution: Vec<(f64, f64)> =
                report.buses.iter().map(|b| (b.voltage_mag, b.voltage_ang)).collect();
            match &reference {
                None => reference = Some(solution),
                Some(expected) => {
                    // Bounded by the solver's own stopping rule, not tighter:
                    // Newton halts once the mismatch drops below `tol`
                    // (1e-6 by default), so two runs that entered the basin
                    // from different directions legitimately stop at slightly
                    // different points inside it. Observed spread here is
                    // ~5e-8.
                    let tol = PowerFlowOptions::default().tol;
                    for (i, ((vm, va), (evm, eva))) in solution.iter().zip(expected).enumerate() {
                        assert!(
                            (vm - evm).abs() < tol && (va - eva).abs() < tol,
                            "{name} bus {i}: {init:?} gave ({vm}, {va}), expected ({evm}, {eva})"
                        );
                    }
                }
            }
        }
        assert!(converged >= 2, "{name}: only {converged} initializer(s) converged");
    }
}

/// The reason this crate warm-starts by default, demonstrated rather than
/// asserted: on `distribution-case` a flat start does not converge inside the
/// default 20 iterations — it is still at a mismatch of order 1 — while
/// **both** warm starts land it in 4–5. DC init is not merely a faster route
/// to an answer flat start would have reached; on this fixture it is the
/// difference between an answer and none.
#[test]
fn both_warm_starts_rescue_a_case_flat_start_cannot_solve() {
    let path = fixture("symmetric/distribution-case");
    let solve = |init| {
        let (buses, lines, transformers, shunts) = load(&path);
        let opts = PowerFlowOptions { init, ..Default::default() };
        run_power_flow(buses, &lines, &transformers, &shunts, opts).stats
    };

    let flat = solve(PowerFlowInit::Flat);
    assert_eq!(
        flat.status,
        SolveStatus::MaxIterationsReached,
        "fixture no longer defeats a flat start, so this proves nothing"
    );
    assert!(flat.final_mismatch() > 1.0, "flat start got closer than expected: {}", flat.final_mismatch());

    for init in [PowerFlowInit::LinearImpedance, PowerFlowInit::Dc] {
        let stats = solve(init);
        assert_eq!(stats.status, SolveStatus::Converged, "{init:?} failed to rescue the case");
        assert!(stats.iterations() < 10, "{init:?} took {} iterations", stats.iterations());
    }
}

/// DC init has to be a genuinely different starting point, or
/// `PowerFlowInit::Dc` is decorative. Checked by comparing the *initial*
/// state each one leaves behind, with Newton capped at zero iterations.
#[test]
fn dc_init_starts_somewhere_other_than_flat() {
    let path = fixture("symmetric/transmission-case");
    let start = |init| {
        let (buses, lines, transformers, shunts) = load(&path);
        let opts = PowerFlowOptions { init, max_iter: 0, ..Default::default() };
        let report = run_power_flow(buses, &lines, &transformers, &shunts, opts);
        report.buses.iter().map(|b| b.voltage_ang).collect::<Vec<_>>()
    };

    let flat = start(PowerFlowInit::Flat);
    let dc = start(PowerFlowInit::Dc);
    assert!(flat.iter().all(|a| *a == 0.0), "flat start should leave every angle at zero");

    let moved = flat.iter().zip(&dc).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    assert!(moved > 1e-3, "DC init barely moved the starting angles ({moved})");
}

/// An initializer must supply a *starting point* and change nothing else.
///
/// `dc_power_flow` is a solver, not an initializer: it normalizes PQ
/// magnitudes to its own |V| = 1 assumption and runs
/// `network::mark_unreferenced_islands`, which rewrites a sourceless island's
/// buses to `Slack`. Both are correct for a DC solve and destructive as a warm
/// start — the first overwrote PV setpoints (which Newton holds fixed, so the
/// converged answer genuinely changed), the second pre-empted Newton's own
/// classification pass and altered the island statuses it reports.
///
/// Neither fixture used elsewhere in this file catches it: PGM data has no PV
/// buses and these cases are fully referenced. Hence the hand-built network.
#[test]
fn the_dc_initializer_changes_nothing_but_the_starting_angles() {
    fn network() -> (Vec<Bus>, Vec<Line>) {
        let mk = |idx, bus_type, voltage_mag, p_spec| Bus {
            idx,
            bus_type,
            voltage_mag,
            voltage_ang: 0.0,
            p_spec,
            q_spec: 0.0,
            q_min: -10.0,
            q_max: 10.0,
            u_rated: 0.0,
            zip_terms: Vec::new(),
        };
        let line = |from, to, r, x| Line { from, to, r, x, b_shunt: 0.0, g_shunt: 0.0 };
        (
            vec![
                mk(0, BusType::Slack, 1.06, 0.0),
                mk(1, BusType::PV, 1.04, 0.2),
                mk(2, BusType::PQ, 1.0, -0.5),
                // A sourceless island, to exercise the classification path.
                mk(3, BusType::PQ, 1.0, -0.1),
                mk(4, BusType::PQ, 1.0, 0.1),
            ],
            vec![
                line(0, 1, 0.02, 0.06),
                line(1, 2, 0.03, 0.09),
                line(0, 2, 0.04, 0.12),
                line(3, 4, 0.02, 0.06),
            ],
        )
    }

    let solve = |init| {
        let (buses, lines) = network();
        let opts = PowerFlowOptions { init, ..Default::default() };
        run_power_flow(buses, &lines, &[], &[], opts)
    };

    let reference = solve(PowerFlowInit::LinearImpedance);
    let dc = solve(PowerFlowInit::Dc);

    // The PV bus's setpoint is an input the solver holds fixed; it must come
    // back untouched, and so must everything downstream of it.
    assert_eq!(reference.buses[1].voltage_mag, 1.04, "fixture assumption broken");
    for (i, (a, b)) in reference.buses.iter().zip(&dc.buses).enumerate() {
        assert!(
            (a.voltage_mag - b.voltage_mag).abs() < 1e-9
                && (a.voltage_ang - b.voltage_ang).abs() < 1e-9,
            "bus {i}: DC init gave ({}, {}), LinearImpedance gave ({}, {})",
            b.voltage_mag,
            b.voltage_ang,
            a.voltage_mag,
            a.voltage_ang
        );
    }

    // And the island report must be the one Newton's own pass produces, not
    // one the initializer pre-decided.
    let statuses = |r: &gridoxide::PowerFlowReport| {
        r.islands.iter().map(|i| (i.bus_indices.clone(), i.status)).collect::<Vec<_>>()
    };
    assert_eq!(statuses(&reference), statuses(&dc));
    assert!(
        statuses(&reference).iter().any(|(_, s)| *s == IslandStatus::NoReferenceBus),
        "fixture no longer contains a sourceless island: {:?}",
        statuses(&reference)
    );
}

/// `enforce_q_limits` must route to the Q-limit outer loop rather than being
/// accepted and ignored.
#[test]
fn enforce_q_limits_reaches_the_outer_loop() {
    let path = fixture("symmetric/transmission-case");
    let (buses, lines, transformers, shunts) = load(&path);
    let opts = PowerFlowOptions { enforce_q_limits: true, ..Default::default() };
    let report = run_power_flow(buses, &lines, &transformers, &shunts, opts);

    assert_eq!(report.stats.status, SolveStatus::Converged);
    // This fixture has no PV buses to switch, so the outer loop stabilizes
    // immediately — the point is that it ran at all and reported doing so.
    assert!(report.stats.q_limit_stabilized);
}
