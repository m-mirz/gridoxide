//! AC contingency: `BatchSolver::solve_contingencies` honoring
//! `Scenario::branch_outages`.
//!
//! The load-bearing test is `contingencies_match_a_direct_solve_of_the_outaged_network`.
//! The fast path keeps an outaged branch's *structural* Y-bus entries so the
//! sparsity pattern — and the symbolic factorization behind it — survives the
//! outage. That is a real optimization with a real way to be wrong, so it is
//! checked against building the outaged network from scratch and solving it
//! directly, which shares none of that machinery.

use std::path::PathBuf;

use gridoxide::batch::{BatchError, BatchSolver, BusOverride, Scenario};
use gridoxide::network::{
    build_ybus, build_ybus_with_outages, stamp_shunts, structural_component_count, ShuntAdm,
};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches};
use gridoxide::solver::{IslandStatus, JacobianBackend, SolveStatus};
use gridoxide::types::{Bus, Line, Transformer};
use gridoxide::run_power_flow_analysis_from_ybus;

mod common;

const TOL: f64 = 1e-6;
const MAX_ITER: usize = 20;

fn load(rel: &str) -> (Vec<Bus>, Vec<Line>, Vec<Transformer>, Vec<ShuntAdm>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow")
        .join(rel)
        .join("input.json");
    let input = common::load_pgm_input(&path);
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    (buses, lines, transformers, shunts)
}

/// Builds the outaged network from scratch — branches dropped outright, no
/// structural placeholders — and solves it. Shares no code with the fast path.
fn direct_solve(
    buses: &[Bus],
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    outaged: &[usize],
) -> Vec<Bus> {
    let keep_lines: Vec<Line> = lines
        .iter()
        .enumerate()
        .filter(|(i, _)| !outaged.contains(i))
        .map(|(_, l)| l.clone())
        .collect();
    let keep_transformers: Vec<Transformer> = transformers
        .iter()
        .enumerate()
        .filter(|(j, _)| !outaged.contains(&(lines.len() + j)))
        .map(|(_, t)| t.clone())
        .collect();
    let mut ybus = build_ybus(buses.len(), &keep_lines, &keep_transformers);
    stamp_shunts(&mut ybus, shunts);
    run_power_flow_analysis_from_ybus(buses.to_vec(), ybus).buses
}

/// **The oracle.** Every single-branch contingency must reproduce a direct
/// solve of the network with that branch actually removed.
///
/// Run at one thread as well as several. A single worker means *every*
/// scenario shares one `PersistentSolver`, which is the sharpest form of the
/// cache-reuse question: `JacobianPattern` caches the admittance each entry
/// was analyzed against, so a pattern carried over from the previous
/// contingency would silently solve the wrong network. Reusing the symbolic
/// factorization across contingencies while re-analyzing that recipe is what
/// this method claims to do, and this is what checks the claim.
#[test]
fn contingencies_match_a_direct_solve_of_the_outaged_network() {
    for name in ["symmetric/transmission-case", "symmetric/distribution-case"] {
        let (buses, lines, transformers, shunts) = load(name);
        let n_branches = lines.len() + transformers.len();

        let scenarios: Vec<Scenario> = (0..n_branches)
            .map(|b| {
                let mut sc = Scenario::new(vec![]);
                sc.branch_outages = vec![b];
                sc
            })
            .collect();

        // Ground truth, computed once per outage from a network built without
        // the branch at all.
        let expected: Vec<(SolveStatus, Vec<Bus>)> = (0..n_branches)
            .map(|b| {
                let mut ybus = {
                    let keep_lines: Vec<Line> = lines
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != b)
                        .map(|(_, l)| l.clone())
                        .collect();
                    let keep_transformers: Vec<Transformer> = transformers
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| lines.len() + j != b)
                        .map(|(_, t)| t.clone())
                        .collect();
                    build_ybus(buses.len(), &keep_lines, &keep_transformers)
                };
                stamp_shunts(&mut ybus, &shunts);
                let report = run_power_flow_analysis_from_ybus(buses.clone(), ybus);
                (report.stats.status, report.buses)
            })
            .collect();

        for threads in [1usize, 4] {
            let reports = BatchSolver::with_threads(JacobianBackend::Scalar, threads)
                .unwrap()
                .solve_contingencies(
                    &buses,
                    &lines,
                    &transformers,
                    &shunts,
                    &scenarios,
                    TOL,
                    MAX_ITER,
                )
                .unwrap();
            assert_eq!(reports.len(), n_branches);

            let mut converged = 0;
            for (b, report) in reports.iter().enumerate() {
                let (want_status, want_buses) = &expected[b];
                assert_eq!(
                    report.stats.status, *want_status,
                    "{name} threads={threads} outage {b}: status differs from a direct solve"
                );
                if report.stats.status != SolveStatus::Converged {
                    // An unsolvable contingency is a legitimate screening
                    // result; the direct solve just has to agree, which the
                    // status check above already established.
                    continue;
                }
                converged += 1;

                for (i, (got, want)) in report.buses.iter().zip(want_buses).enumerate() {
                    assert!(
                        (got.voltage_mag - want.voltage_mag).abs() < TOL
                            && (got.voltage_ang - want.voltage_ang).abs() < TOL,
                        "{name} threads={threads} outage {b} bus {i}: contingency gave \
                         ({}, {}), direct solve gives ({}, {})",
                        got.voltage_mag,
                        got.voltage_ang,
                        want.voltage_mag,
                        want.voltage_ang
                    );
                }
            }
            assert!(
                converged > 0,
                "{name} threads={threads}: no contingency converged, so nothing was compared"
            );
        }
    }
}

/// The fast path's premise, checked directly: taking a branch out by zeroing
/// its contribution must leave the Y-bus's *pattern* identical to the intact
/// network's, or the shared symbolic factorization is invalid.
#[test]
fn an_outaged_ybus_keeps_the_intact_sparsity_pattern() {
    let (buses, lines, transformers, _) = load("symmetric/transmission-case");
    let n = buses.len();

    let intact = build_ybus(n, &lines, &transformers).finish();
    let pattern = |y: &gridoxide::network::YBusSparse| {
        (0..y.n())
            .map(|i| y.row(i).iter().map(|&(j, _)| j).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    let intact_pattern = pattern(&intact);

    for b in 0..(lines.len() + transformers.len()) {
        let mut outaged = vec![false; lines.len() + transformers.len()];
        outaged[b] = true;
        let with_outage =
            build_ybus_with_outages(n, &lines, &transformers, &outaged).finish();
        assert_eq!(
            pattern(&with_outage),
            intact_pattern,
            "outaging branch {b} changed the sparsity pattern"
        );
    }
}

/// A contingency that severs part of the network must report that honestly —
/// `NoReferenceBus` for the orphaned island — rather than coming back as a
/// singular solve, which is what the structural-zero fast path would produce if
/// it were used here.
#[test]
fn a_severing_contingency_reports_an_unreferenced_island() {
    // Slack — 1 — 2, with 1–2 the only path to bus 2.
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
    let line = |from, to| Line { from, to, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 };
    let buses = vec![
        mk(0, gridoxide::types::BusType::Slack, 0.0),
        mk(1, gridoxide::types::BusType::PQ, -0.2),
        mk(2, gridoxide::types::BusType::PQ, -0.1),
    ];
    let lines = vec![line(0, 1), line(1, 2)];

    // Fixture assumption: outaging branch 1 really does split the network.
    assert_eq!(structural_component_count(3, &lines, &[], &[]), 1);
    assert_eq!(structural_component_count(3, &lines, &[], &[false, true]), 2);

    let mut sc = Scenario::new(vec![]);
    sc.branch_outages = vec![1];
    let reports = BatchSolver::new(JacobianBackend::Scalar)
        .solve_contingencies(&buses, &lines, &[], &[], &[sc], TOL, MAX_ITER)
        .unwrap();

    let statuses: Vec<IslandStatus> = reports[0].islands.iter().map(|i| i.status).collect();
    assert!(
        statuses.contains(&IslandStatus::NoReferenceBus),
        "expected an unreferenced island, got {statuses:?}"
    );
    assert!(
        statuses.contains(&IslandStatus::Converged),
        "the still-referenced island should still solve, got {statuses:?}"
    );
    assert_eq!(reports[0].stats.status, SolveStatus::Converged);
    // The severed bus is pinned to the placeholder, not left at a stale value.
    assert_eq!(reports[0].buses[2].voltage_mag, 0.0);
}

/// A worker reuses one solver across the scenarios it handles, so a severing
/// contingency must not leave a stale factorization behind for the next one.
/// Interleaving the two kinds is what would expose that.
#[test]
fn severing_and_non_severing_contingencies_interleave_safely() {
    let (buses, lines, transformers, shunts) = load("symmetric/transmission-case");
    let n_branches = lines.len() + transformers.len();

    // Every branch, twice over, so severing and non-severing scenarios are
    // guaranteed to share workers in both orders.
    let scenarios: Vec<Scenario> = (0..n_branches)
        .chain((0..n_branches).rev())
        .map(|b| {
            let mut sc = Scenario::new(vec![]);
            sc.branch_outages = vec![b];
            sc
        })
        .collect();

    for threads in [1usize, 4] {
        let reports = BatchSolver::with_threads(JacobianBackend::Scalar, threads)
            .unwrap()
            .solve_contingencies(
                &buses,
                &lines,
                &transformers,
                &shunts,
                &scenarios,
                TOL,
                MAX_ITER,
            )
            .unwrap();

        // The same outage appears twice; both occurrences must agree.
        for b in 0..n_branches {
            let first = &reports[b];
            let second = &reports[2 * n_branches - 1 - b];
            assert_eq!(
                first.stats.status, second.stats.status,
                "threads={threads}, outage {b}: status differs between occurrences"
            );
            for (i, (a, c)) in first.buses.iter().zip(&second.buses).enumerate() {
                assert!(
                    (a.voltage_mag - c.voltage_mag).abs() < TOL
                        && (a.voltage_ang - c.voltage_ang).abs() < TOL,
                    "threads={threads}, outage {b} bus {i}: {} vs {}",
                    a.voltage_mag,
                    c.voltage_mag
                );
            }
        }
    }
}

/// Outages compose with bus overrides — a contingency screen usually wants both
/// (an outage at some loading level), and they must not interfere.
#[test]
fn outages_and_bus_overrides_compose() {
    let (buses, lines, transformers, shunts) = load("symmetric/transmission-case");

    let mut scenario = Scenario::new(
        buses
            .iter()
            .filter(|b| b.p_spec != 0.0)
            .map(|b| BusOverride::new(b.idx).p(b.p_spec * 0.8))
            .collect(),
    );
    scenario.branch_outages = vec![0];

    let report = BatchSolver::new(JacobianBackend::Scalar)
        .solve_contingencies(&buses, &lines, &transformers, &shunts, &[scenario], TOL, MAX_ITER)
        .unwrap()
        .pop()
        .unwrap();

    let mut scaled = buses.clone();
    for b in scaled.iter_mut() {
        if b.p_spec != 0.0 {
            b.p_spec *= 0.8;
        }
    }
    let expected = direct_solve(&scaled, &lines, &transformers, &shunts, &[0]);

    assert_eq!(report.stats.status, SolveStatus::Converged);
    for (i, (got, want)) in report.buses.iter().zip(&expected).enumerate() {
        assert!(
            (got.voltage_mag - want.voltage_mag).abs() < TOL
                && (got.voltage_ang - want.voltage_ang).abs() < TOL,
            "bus {i}: {} vs {}",
            got.voltage_mag,
            want.voltage_mag
        );
    }
}

/// An empty outage list must reproduce the plain batch path exactly, so the two
/// entry points cannot drift.
#[test]
fn no_outages_matches_the_plain_batch_path() {
    let (buses, lines, transformers, shunts) = load("symmetric/transmission-case");
    let scenarios: Vec<Scenario> = (0..8)
        .map(|k| {
            Scenario::new(
                buses
                    .iter()
                    .filter(|b| b.p_spec != 0.0)
                    .map(|b| BusOverride::new(b.idx).p(b.p_spec * (0.6 + 0.05 * k as f64)))
                    .collect(),
            )
        })
        .collect();

    let batch = BatchSolver::new(JacobianBackend::Scalar);
    let via_contingencies = batch
        .solve_contingencies(&buses, &lines, &transformers, &shunts, &scenarios, TOL, MAX_ITER)
        .unwrap();

    let mut ybus = build_ybus(buses.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let ybus = ybus.finish();
    let via_plain = batch.solve(&buses, &ybus, &scenarios, TOL, MAX_ITER).unwrap();

    for (k, (a, b)) in via_contingencies.iter().zip(&via_plain).enumerate() {
        assert_eq!(a.stats.status, b.stats.status, "scenario {k}");
        for (i, (x, y)) in a.buses.iter().zip(&b.buses).enumerate() {
            assert_eq!(x.voltage_mag.to_bits(), y.voltage_mag.to_bits(), "scenario {k} bus {i}");
            assert_eq!(x.voltage_ang.to_bits(), y.voltage_ang.to_bits(), "scenario {k} bus {i}");
        }
    }
}

/// Malformed input is rejected, and `solve` still refuses outages it cannot
/// honor from a finished Y-bus.
#[test]
fn malformed_input_is_rejected() {
    let (buses, lines, transformers, shunts) = load("symmetric/transmission-case");
    let n_branches = lines.len() + transformers.len();
    let batch = BatchSolver::new(JacobianBackend::Scalar);

    let mut bad_branch = Scenario::new(vec![]);
    bad_branch.branch_outages = vec![n_branches];
    assert!(matches!(
        batch.solve_contingencies(
            &buses, &lines, &transformers, &shunts, &[bad_branch], TOL, MAX_ITER
        ),
        Err(BatchError::BranchOutOfRange { scenario: 0, .. })
    ));

    let bad_bus = Scenario::new(vec![BusOverride::new(buses.len()).p(0.0)]);
    assert!(matches!(
        batch.solve_contingencies(
            &buses, &lines, &transformers, &shunts, &[bad_bus], TOL, MAX_ITER
        ),
        Err(BatchError::BusOutOfRange { scenario: 0, .. })
    ));

    assert_eq!(
        batch
            .solve_contingencies(&buses, &lines, &transformers, &shunts, &[], TOL, MAX_ITER)
            .unwrap()
            .len(),
        0
    );

    // `solve` takes a finished Y-bus and still cannot honor an outage.
    let mut ybus = build_ybus(buses.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let ybus = ybus.finish();
    let mut outage = Scenario::new(vec![]);
    outage.branch_outages = vec![0];
    assert!(matches!(
        batch.solve(&buses, &ybus, &[outage], TOL, MAX_ITER),
        Err(BatchError::OutagesUnsupported { scenario: 0 })
    ));
}
