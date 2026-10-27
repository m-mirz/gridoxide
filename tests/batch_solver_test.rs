//! `batch::BatchSolver` correctness. The load-bearing property is
//! *equivalence*: a parallel batch must produce exactly what a sequential
//! loop over `PersistentSolver::solve` produces, bit for bit. Everything
//! else in the batch path is an optimization on top of that.

use std::fs;
use std::path::PathBuf;

use gridoxide::batch::{BatchError, BatchSolver, BatchStart, BusOverride, Scenario};
use gridoxide::json::NetworkData;
use gridoxide::network::{YBusSparse, build_ybus, linear_initial_guess};
use gridoxide::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide::types::Bus;

fn load_network() -> (Vec<Bus>, YBusSparse) {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let raw = fs::read_to_string(path).expect("read network.json");
    let network: NetworkData = serde_json::from_str(&raw).expect("parse network.json");
    let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();
    (network.buses, ybus)
}

/// A spread of load scalings on bus 2 — enough variety that a solver
/// wrongly carrying state between scenarios would show up.
fn scenarios(template: &[Bus], n: usize) -> Vec<Scenario> {
    (0..n)
        .map(|k| {
            let f = 0.5 + 1.5 * (k as f64) / (n as f64);
            Scenario::new(vec![
                BusOverride::new(2)
                    .p(template[2].p_spec * f)
                    .q(template[2].q_spec * f),
            ])
        })
        .collect()
}

/// Applies a scenario the same way `BatchSolver` does, then solves it with a
/// plain `PersistentSolver` — the reference the batch must reproduce.
fn solve_sequentially(
    template: &[Bus],
    ybus: &YBusSparse,
    scs: &[Scenario],
    backend: JacobianBackend,
) -> Vec<Vec<Bus>> {
    let mut solver = PersistentSolver::new(backend);
    scs.iter()
        .map(|sc| {
            let mut buses = template.to_vec();
            for ov in &sc.bus_overrides {
                let b = &mut buses[ov.bus];
                if let Some(p) = ov.p_spec {
                    b.p_spec = p;
                }
                if let Some(q) = ov.q_spec {
                    b.q_spec = q;
                }
                if let Some(vm) = ov.voltage_mag {
                    b.voltage_mag = vm;
                }
            }
            linear_initial_guess(&mut buses, ybus);
            solver.solve(&mut buses, ybus, 1e-6, 20);
            buses
        })
        .collect()
}

fn assert_batch_matches_sequential(backend: JacobianBackend) {
    let (template, ybus) = load_network();
    let scs = scenarios(&template, 24);

    let expected = solve_sequentially(&template, &ybus, &scs, backend);

    let batch = BatchSolver::with_threads(backend, 8).expect("build pool");
    let got = batch
        .solve(&template, &ybus, &scs, 1e-6, 20)
        .expect("batch solve");

    assert_eq!(got.len(), scs.len(), "one report per scenario");
    for (k, (report, want)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            report.stats.status,
            SolveStatus::Converged,
            "scenario {k} should converge with backend {backend:?}"
        );
        for (i, (b, w)) in report.buses.iter().zip(want).enumerate() {
            // Bit-exact, not approximate: the batch runs the identical
            // arithmetic in the identical order, just on another thread.
            assert_eq!(
                b.voltage_mag, w.voltage_mag,
                "scenario {k} bus {i} |V| ({backend:?})"
            );
            assert_eq!(
                b.voltage_ang, w.voltage_ang,
                "scenario {k} bus {i} angle ({backend:?})"
            );
        }
    }
}

#[test]
fn batch_matches_sequential_scalar() {
    assert_batch_matches_sequential(JacobianBackend::Scalar);
}

#[test]
fn batch_matches_sequential_klu_native() {
    assert_batch_matches_sequential(JacobianBackend::KluNative);
}

#[test]
fn batch_matches_sequential_block() {
    assert_batch_matches_sequential(JacobianBackend::Block);
}

/// The backend that most needs this test: `KluRealSystem` owns raw
/// SuiteSparse pointers and is `!Send`, so it only works here because each
/// `PersistentSolver` is constructed inside its own `rayon::scope` worker
/// closure and never crosses a thread boundary.
#[cfg(feature = "klu")]
#[test]
fn batch_matches_sequential_klu() {
    assert_batch_matches_sequential(JacobianBackend::Klu);
}

/// Results must come back in scenario order and with identical values no
/// matter how many workers produced them — the batch hands out indices from
/// an atomic counter, so completion order is nondeterministic by design and
/// the ordering has to be restored explicitly.
#[test]
fn results_are_thread_count_invariant() {
    let (template, ybus) = load_network();
    let scs = scenarios(&template, 32);

    let one = BatchSolver::with_threads(JacobianBackend::Scalar, 1).expect("build pool");
    let many = BatchSolver::with_threads(JacobianBackend::Scalar, 8).expect("build pool");

    let a = one
        .solve(&template, &ybus, &scs, 1e-6, 20)
        .expect("1-thread batch");
    let b = many
        .solve(&template, &ybus, &scs, 1e-6, 20)
        .expect("8-thread batch");

    assert_eq!(a.len(), b.len());
    for (k, (ra, rb)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            ra.stats, rb.stats,
            "scenario {k} stats differ across thread counts"
        );
        for (i, (x, y)) in ra.buses.iter().zip(&rb.buses).enumerate() {
            assert_eq!(x.voltage_mag, y.voltage_mag, "scenario {k} bus {i} |V|");
            assert_eq!(x.voltage_ang, y.voltage_ang, "scenario {k} bus {i} angle");
        }
    }
}

/// A scenario that cannot converge must not affect any other scenario in the
/// batch. Divergent contingencies are normal in N-1 screening, which makes
/// per-scenario convergence masking a hard requirement for any GPU path — this
/// is the CPU-side precedent.
#[test]
fn divergent_scenario_does_not_poison_the_batch() {
    let (template, ybus) = load_network();

    let mut scs = scenarios(&template, 8);
    // A load far beyond the network's transfer capability: no solution
    // exists, so Newton cannot converge.
    let bad = 4;
    scs[bad] = Scenario::new(vec![BusOverride::new(2).p(-1.0e6).q(-1.0e6)]);

    let batch = BatchSolver::with_threads(JacobianBackend::Scalar, 4).expect("build pool");
    let got = batch
        .solve(&template, &ybus, &scs, 1e-6, 20)
        .expect("batch solve");

    assert_ne!(
        got[bad].stats.status,
        SolveStatus::Converged,
        "the deliberately unsolvable scenario must not report success"
    );

    // Every other scenario is untouched — compare against a batch that never
    // contained the bad scenario at all.
    let clean = scenarios(&template, 8);
    let reference = batch
        .solve(&template, &ybus, &clean, 1e-6, 20)
        .expect("reference batch");
    for k in (0..8).filter(|&k| k != bad) {
        assert_eq!(
            got[k].stats.status,
            SolveStatus::Converged,
            "scenario {k} should still converge"
        );
        for (i, (x, y)) in got[k].buses.iter().zip(&reference[k].buses).enumerate() {
            assert_eq!(
                x.voltage_mag, y.voltage_mag,
                "scenario {k} bus {i} |V| perturbed by its neighbor"
            );
            assert_eq!(
                x.voltage_ang, y.voltage_ang,
                "scenario {k} bus {i} angle perturbed by its neighbor"
            );
        }
    }
}

/// `branch_outages` is reserved but unimplemented. It must be rejected
/// loudly rather than silently ignored — silently ignoring it would return
/// confident, wrong contingency results.
#[test]
fn branch_outages_are_rejected() {
    let (template, ybus) = load_network();
    let scs = vec![
        Scenario::new(vec![BusOverride::new(2).p(-0.5)]),
        Scenario {
            bus_overrides: Vec::new(),
            branch_outages: vec![0],
        },
    ];

    let batch = BatchSolver::new(JacobianBackend::Scalar);
    let err = batch
        .solve(&template, &ybus, &scs, 1e-6, 20)
        .expect_err("outages must be rejected");
    assert_eq!(err, BatchError::OutagesUnsupported { scenario: 1 });
}

#[test]
fn out_of_range_bus_override_is_rejected() {
    let (template, ybus) = load_network();
    let scs = vec![Scenario::new(vec![BusOverride::new(99).p(-0.5)])];

    let batch = BatchSolver::new(JacobianBackend::Scalar);
    let err = batch
        .solve(&template, &ybus, &scs, 1e-6, 20)
        .expect_err("bad index must be rejected");
    assert_eq!(
        err,
        BatchError::BusOutOfRange {
            scenario: 0,
            bus: 99,
            n_buses: template.len()
        }
    );
}

#[test]
fn empty_batch_is_not_an_error() {
    let (template, ybus) = load_network();
    let batch = BatchSolver::new(JacobianBackend::Scalar);
    let got = batch
        .solve(&template, &ybus, &[], 1e-6, 20)
        .expect("empty batch");
    assert!(got.is_empty());
}

/// `BatchStart::Template` starts every scenario from a solved base case:
/// the same answers as the linear start, and no Newton step for a scenario
/// that is the base.
#[test]
fn a_solved_base_case_is_a_start() {
    let (template, ybus) = load_network();
    let mut base = template.clone();
    linear_initial_guess(&mut base, &ybus);
    let mut solver = PersistentSolver::new(JacobianBackend::KluNative);
    solver.solve(&mut base, &ybus, 1e-10, 20);

    let mut scs = scenarios(&template, 8);
    scs.push(Scenario::new(Vec::new()));
    let linear = BatchSolver::with_threads(JacobianBackend::KluNative, 2)
        .unwrap()
        .solve(&template, &ybus, &scs, 1e-8, 20)
        .unwrap();
    let mut batch = BatchSolver::with_threads(JacobianBackend::KluNative, 2).unwrap();
    batch.set_start(BatchStart::Template);
    let from_base = batch.solve(&base, &ybus, &scs, 1e-8, 20).unwrap();

    for (a, b) in linear.iter().zip(&from_base) {
        assert_eq!(b.stats.status, SolveStatus::Converged);
        for (x, y) in a.buses.iter().zip(&b.buses) {
            assert!((x.voltage_mag - y.voltage_mag).abs() < 1e-7);
            assert!((x.voltage_ang - y.voltage_ang).abs() < 1e-7);
        }
    }
    // `iterations()` counts mismatch evaluations: one, and no Newton step.
    assert_eq!(from_base.last().unwrap().stats.iterations(), 1);
    assert!(linear.last().unwrap().stats.iterations() > 1);
}

/// A dedicated pool keeps each worker's solver between calls. Its Jacobian
/// carries the Y-bus admittances, so a call on a network with other values
/// must not reuse it: the answers must be a fresh solver's.
#[test]
fn a_cached_worker_solver_is_not_reused_on_another_network() {
    let (template, ybus) = load_network();
    let scs = scenarios(&template, 16);
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let network: NetworkData = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let mut lines = network.lines.clone();
    for line in &mut lines {
        line.x *= 1.3;
    }
    let other = build_ybus(template.len(), &lines, &[]).finish();

    let batch = BatchSolver::with_threads(JacobianBackend::KluNative, 4).unwrap();
    let first = batch.solve(&template, &ybus, &scs, 1e-8, 20).unwrap();
    let again = batch.solve(&template, &ybus, &scs, 1e-8, 20).unwrap();
    let changed = batch.solve(&template, &other, &scs, 1e-8, 20).unwrap();
    let fresh = BatchSolver::with_threads(JacobianBackend::KluNative, 4)
        .unwrap()
        .solve(&template, &other, &scs, 1e-8, 20)
        .unwrap();

    for (a, b) in first.iter().zip(&again) {
        for (x, y) in a.buses.iter().zip(&b.buses) {
            assert_eq!(x.voltage_mag, y.voltage_mag);
        }
    }
    for (a, b) in changed.iter().zip(&fresh) {
        assert_eq!(a.stats.status, SolveStatus::Converged);
        for (x, y) in a.buses.iter().zip(&b.buses) {
            assert!((x.voltage_mag - y.voltage_mag).abs() < 1e-12);
            assert!((x.voltage_ang - y.voltage_ang).abs() < 1e-12);
        }
    }
}
