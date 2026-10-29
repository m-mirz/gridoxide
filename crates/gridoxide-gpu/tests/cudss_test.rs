//! cuDSS's uniform batch against the CPU stand-in: the same contract, the
//! same Newton loop, the same answers. Needs an NVIDIA GPU; built only with the
//! `cuda` feature (`cargo test -p gridoxide-gpu --features cuda`).
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use gridoxide::network::{YBusSparse, build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide::types::Bus;
use gridoxide_gpu::{
    BatchOptions, BatchReport, CpuUniformBatch, CsrPattern, CudssUniformBatch, LinearError,
    Outcome, UniformBatch, solve_batch,
};

/// `[[a, 1], [1, b]]`: a 2×2 pattern with every entry stored.
fn pattern() -> CsrPattern {
    CsrPattern {
        n: 2,
        row_ptr: vec![0, 2, 4],
        col_idx: vec![0, 1, 0, 1],
    }
}

fn values(diag: &[(f64, f64)]) -> Vec<f64> {
    diag.iter().flat_map(|&(a, b)| [a, 1.0, 1.0, b]).collect()
}

/// x of [[a, 1], [1, b]] x = [1, 0].
fn expected(a: f64, b: f64) -> [f64; 2] {
    let det = a * b - 1.0;
    [b / det, -1.0 / det]
}

#[test]
fn a_solve_selecting_an_unfactored_system_is_refused() {
    let mut lin = CudssUniformBatch::analyze(&pattern(), 3).unwrap();
    lin.factor(
        &values(&[(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)]),
        &[true, false, true],
    )
    .unwrap();
    let mut x = vec![0.0; 6];
    assert_eq!(
        lin.solve(&[1.0, 0.0].repeat(3), &mut x, &[false, true, false]),
        Err(LinearError::NotFactored { system: 1 })
    );
}

/// Probe B's question on the smallest case, with each factorization kind: an
/// unselected system keeps its factors and its solution.
#[test]
fn masks_leave_unselected_systems_as_they_were() {
    for refactorization in [false, true] {
        let mut lin = CudssUniformBatch::analyze(&pattern(), 3).unwrap();
        lin.use_refactorization(refactorization);
        lin.factor(
            &values(&[(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)]),
            &[true, true, true],
        )
        .unwrap();
        lin.factor(
            &values(&[(8.0, 9.0), (1e300, 1e300), (1e300, 1e300)]),
            &[true, false, false],
        )
        .unwrap();
        let mut x = vec![12345.0; 6];
        lin.solve(&[1.0, 0.0].repeat(3), &mut x, &[true, false, true])
            .unwrap();

        let close =
            |got: &[f64], want: [f64; 2]| got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-14);
        assert!(
            close(&x[0..2], expected(8.0, 9.0)),
            "{refactorization}: {x:?}"
        );
        assert_eq!(&x[2..4], &[12345.0, 12345.0]);
        assert!(
            close(&x[4..6], expected(6.0, 7.0)),
            "{refactorization}: {x:?}"
        );
    }
}

fn load(rel: &str) -> (Vec<Bus>, YBusSparse) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/benchmark-grids")
        .join(rel);
    assert!(
        path.exists(),
        "{} not found — run `git submodule update --init tests/data/benchmark-grids`",
        path.display()
    );
    let net = gridoxide::matpower::read(&path).unwrap();
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let mut base = net.buses.clone();
    linear_initial_guess(&mut base, &ybus);
    let (_, stats) = PersistentSolver::new(JacobianBackend::KluNative)
        .solve_with_stats(&mut base, &ybus, 1e-10, 30);
    assert_eq!(stats.status, SolveStatus::Converged);
    (base, ybus)
}

/// Scenario k scales every injection by 0.6 + 0.4·k/B: deterministic and
/// uneven enough that scenarios converge at different steps.
fn ramp(base: &[Bus], batch: usize) -> Vec<Scenario> {
    (0..batch)
        .map(|k| {
            let f = 0.6 + 0.4 * k as f64 / batch as f64;
            Scenario::new(
                base.iter()
                    .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
                    .map(|b| BusOverride::new(b.idx).p(b.p_spec * f).q(b.q_spec * f))
                    .collect(),
            )
        })
        .collect()
}

fn assert_same(gpu: &BatchReport, cpu: &BatchReport) {
    for (g, c) in gpu.scenarios.iter().zip(&cpu.scenarios) {
        assert_eq!(g.outcome, Outcome::Converged);
        assert_eq!(g.outcome, c.outcome);
        assert_eq!(g.newton_steps, c.newton_steps);
        for (a, b) in g.voltage_mag.iter().zip(&c.voltage_mag) {
            assert!((a - b).abs() < 1e-9, "|V| {a} against {b}");
        }
        for (a, b) in g.voltage_ang.iter().zip(&c.voltage_ang) {
            assert!((a - b).abs() < 1e-9, "angle {a} against {b}");
        }
    }
    assert_eq!(gpu.linear, cpu.linear);
}

/// The whole adaptive loop over the device gives the stand-in's answers, step
/// for step, with and without factor reuse.
#[test]
fn the_newton_loop_on_the_device_matches_the_stand_in() {
    for case in ["matpower/case1354pegase.m", "generated/mvlv1004.m"] {
        let (base, ybus) = load(case);
        let scenarios = ramp(&base, 64);
        for reuse_below in [None, Some(0.1)] {
            let opts = BatchOptions {
                tol: 1e-10,
                reuse_below,
                ..BatchOptions::default()
            };
            let gpu = solve_batch::<CudssUniformBatch>(&base, &ybus, &scenarios, &opts).unwrap();
            let cpu = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &opts).unwrap();
            assert_same(&gpu, &cpu);
        }
    }
}
