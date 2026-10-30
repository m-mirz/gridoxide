//! Phase 3: the adaptive Newton loop on the device ([`GpuBatchSolver`])
//! against the CPU loop on the stand-in — the
//! same outcomes, the same steps and factorizations per scenario, voltages to
//! 1e-9 — for sweeps, N-1, every policy and option, in one chunk and in
//! several. Needs an NVIDIA GPU; built only with the `cuda` feature.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use gridoxide::matpower::MatpowerNetwork;
use gridoxide::network::{YBusSparse, build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide::types::Bus;
use gridoxide_gpu::{
    BatchError, BatchOptions, BatchReport, CpuUniformBatch, GpuBatchSolver, Outcome, StepPolicy,
    solve_batch, solve_contingency_batch,
};

fn read(rel: &str) -> MatpowerNetwork {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/benchmark-grids")
        .join(rel);
    assert!(
        path.exists(),
        "{} not found — run `git submodule update --init tests/data/benchmark-grids`",
        path.display()
    );
    gridoxide::matpower::read(&path).unwrap()
}

fn load(rel: &str) -> (MatpowerNetwork, Vec<Bus>, YBusSparse) {
    let net = read(rel);
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let mut base = net.buses.clone();
    linear_initial_guess(&mut base, &ybus);
    let (_, stats) = PersistentSolver::new(JacobianBackend::KluNative)
        .solve_with_stats(&mut base, &ybus, 1e-10, 30);
    assert_eq!(stats.status, SolveStatus::Converged);
    (net, base, ybus)
}

/// Every injection scaled by `f`.
fn scaled(base: &[Bus], f: f64) -> Scenario {
    Scenario::new(
        base.iter()
            .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
            .map(|b| BusOverride::new(b.idx).p(b.p_spec * f).q(b.q_spec * f))
            .collect(),
    )
}

/// Scenario k scales every injection by 0.6 + 0.4·k/B: uneven enough that
/// scenarios converge at different steps.
fn ramp(base: &[Bus], batch: usize) -> Vec<Scenario> {
    (0..batch)
        .map(|k| scaled(base, 0.6 + 0.4 * k as f64 / batch as f64))
        .collect()
}

fn cpu(
    base: &[Bus],
    ybus: &YBusSparse,
    scenarios: &[Scenario],
    opts: &BatchOptions,
) -> Result<BatchReport, BatchError> {
    solve_batch::<CpuUniformBatch>(base, ybus, scenarios, opts)
}

/// Scenario by scenario, except those in `rough` (whose divergence is
/// chaotic, so only its outcome is compared, and only as not converged).
fn assert_same(gpu: &BatchReport, cpu: &BatchReport, rough: &[usize], what: &str) {
    assert_eq!(gpu.scenarios.len(), cpu.scenarios.len());
    for (s, (g, c)) in gpu.scenarios.iter().zip(&cpu.scenarios).enumerate() {
        if rough.contains(&s) {
            assert_ne!(g.outcome, Outcome::Converged, "{what}: scenario {s}");
            assert_ne!(c.outcome, Outcome::Converged, "{what}: scenario {s}");
            continue;
        }
        assert_eq!(g.outcome, c.outcome, "{what}: scenario {s}");
        assert_eq!(g.newton_steps, c.newton_steps, "{what}: scenario {s}");
        assert_eq!(g.factorizations, c.factorizations, "{what}: scenario {s}");
        assert_eq!(g.on_cpu, c.on_cpu, "{what}: scenario {s}");
        assert_eq!(g.voltage_mag.len(), c.voltage_mag.len());
        for (a, b) in g
            .voltage_mag
            .iter()
            .zip(&c.voltage_mag)
            .chain(g.voltage_ang.iter().zip(&c.voltage_ang))
        {
            assert!(
                (a - b).abs() < 1e-9,
                "{what}: scenario {s}, {a} against {b}"
            );
        }
    }
    assert_eq!(gpu.cpu_handoffs, cpu.cpu_handoffs, "{what}");
}

/// Sweeps in one chunk and in three (the last padded), adaptive and fixed.
#[test]
fn sweeps_match_the_cpu_loop() {
    for case in ["matpower/case1354pegase.m", "generated/mvlv1004.m"] {
        let (_, base, ybus) = load(case);
        let scenarios = ramp(&base, 64);
        for policy in [StepPolicy::Adaptive, StepPolicy::Fixed(4)] {
            let opts = BatchOptions {
                tol: 1e-10,
                policy,
                ..BatchOptions::default()
            };
            let want = cpu(&base, &ybus, &scenarios, &opts).unwrap();
            for capacity in [64, 24] {
                let mut gpu = GpuBatchSolver::new(&base, &ybus, capacity).unwrap();
                let got = gpu.solve(&scenarios, &opts).unwrap();
                let what = format!("{case}, {policy:?}, capacity {capacity}");
                assert_same(&got, &want, &[], &what);
                assert_eq!(
                    got.linear.systems_factored, want.linear.systems_factored,
                    "{what}"
                );
                assert_eq!(
                    got.linear.systems_solved, want.linear.systems_solved,
                    "{what}"
                );
                if capacity == 64 {
                    assert_eq!(got.batch_steps, want.batch_steps, "{what}");
                }
            }
        }
    }
}

/// A fixed step count leaves some of mvlv1004's scenarios short, as on the
/// CPU; the adaptive loop converges them all.
#[test]
fn the_adaptive_loop_converges_what_fixed_steps_miss() {
    let (_, base, ybus) = load("generated/mvlv1004.m");
    let scenarios = ramp(&base, 64);
    let mut gpu = GpuBatchSolver::new(&base, &ybus, 64).unwrap();
    let fixed = BatchOptions {
        policy: StepPolicy::Fixed(4),
        ..BatchOptions::default()
    };
    let short = gpu.solve(&scenarios, &fixed).unwrap();
    let adaptive = gpu.solve(&scenarios, &BatchOptions::default()).unwrap();
    let converged = |r: &BatchReport| {
        r.scenarios
            .iter()
            .filter(|s| s.outcome == Outcome::Converged)
            .count()
    };
    eprintln!(
        "mvlv1004: fixed 4 converges {} of 64, adaptive {}",
        converged(&short),
        converged(&adaptive)
    );
    assert!(converged(&short) < 64);
    assert_eq!(converged(&adaptive), 64);
}

/// Factor reuse and the CPU hand-off.
#[test]
fn options_match_the_cpu_loop() {
    let (_, base, ybus) = load("matpower/case1354pegase.m");
    let scenarios = ramp(&base, 48);
    let mut gpu = GpuBatchSolver::new(&base, &ybus, 48).unwrap();

    let reuse = BatchOptions {
        tol: 1e-10,
        reuse_below: Some(0.1),
        ..BatchOptions::default()
    };
    let got = gpu.solve(&scenarios, &reuse).unwrap();
    let want = cpu(&base, &ybus, &scenarios, &reuse).unwrap();
    assert_same(&got, &want, &[], "factor reuse");
    assert_eq!(got.linear, want.linear);

    // A tail of heavier scenarios that take more steps than the rest, for
    // the hand-off to take.
    let mut tail = scenarios.clone();
    for (k, f) in [(10, 1.25), (20, 1.3), (30, 1.35)] {
        tail[k] = scaled(&base, f);
    }
    let handoff = BatchOptions {
        tol: 1e-10,
        cpu_handoff: Some(5),
        ..BatchOptions::default()
    };
    let scenarios = tail;
    let got = gpu.solve(&scenarios, &handoff).unwrap();
    let want = cpu(&base, &ybus, &scenarios, &handoff).unwrap();
    assert!(got.cpu_handoffs > 0, "the hand-off is exercised");
    assert_same(&got, &want, &[], "CPU hand-off");
}

/// A scenario with no operating point diverges alone; the rest are as if it
/// were not there.
#[test]
fn a_hopeless_scenario_does_not_disturb_the_rest() {
    let (_, base, ybus) = load("matpower/case1354pegase.m");
    let mut scenarios = ramp(&base, 16);
    scenarios[5] = scaled(&base, 50.0);
    let opts = BatchOptions {
        tol: 1e-10,
        ..BatchOptions::default()
    };
    let got = GpuBatchSolver::new(&base, &ybus, 16)
        .unwrap()
        .solve(&scenarios, &opts)
        .unwrap();
    let want = cpu(&base, &ybus, &scenarios, &opts).unwrap();
    eprintln!(
        "hopeless: device {:?} after {} steps, CPU {:?} after {}",
        got.scenarios[5].outcome,
        got.scenarios[5].newton_steps,
        want.scenarios[5].outcome,
        want.scenarios[5].newton_steps
    );
    assert_same(&got, &want, &[5], "hopeless");
}

/// N-1 with outages that split the network among them, in one chunk and in
/// several, adaptive and with the CPU hand-off.
#[test]
fn n1_matches_the_cpu_loop() {
    let (net, base, _) = load("matpower/case1354pegase.m");
    let n_branches = net.lines.len() + net.transformers.len();
    let scenarios: Vec<Scenario> = (0..96)
        .map(|k| Scenario {
            bus_overrides: Vec::new(),
            branch_outages: vec![k * n_branches / 96],
        })
        .collect();
    for (opts, what) in [
        (
            BatchOptions {
                tol: 1e-10,
                ..BatchOptions::default()
            },
            "adaptive",
        ),
        (
            BatchOptions {
                tol: 1e-10,
                cpu_handoff: Some(4),
                ..BatchOptions::default()
            },
            "hand-off",
        ),
    ] {
        let want = solve_contingency_batch::<CpuUniformBatch>(
            &base,
            &net.lines,
            &net.transformers,
            &net.shunts,
            &scenarios,
            &opts,
        )
        .unwrap();
        let splits = want
            .scenarios
            .iter()
            .filter(|r| r.outcome == Outcome::Splits)
            .count();
        assert!(splits > 0, "some outages split the network");
        for capacity in [96, 32] {
            let mut gpu = GpuBatchSolver::with_branches(
                &base,
                &net.lines,
                &net.transformers,
                &net.shunts,
                capacity,
            )
            .unwrap();
            let got = gpu.solve_contingencies(&scenarios, &opts).unwrap();
            assert_same(
                &got,
                &want,
                &[],
                &format!("N-1 {what}, capacity {capacity}"),
            );
        }
    }
}

/// A session solves again without a new analysis, to the same answers; N-1
/// needs one made with the branches.
#[test]
fn a_session_solves_again() {
    let (_, base, ybus) = load("matpower/case1354pegase.m");
    let mut gpu = GpuBatchSolver::new(&base, &ybus, 32).unwrap();
    let opts = BatchOptions::default();
    let first = gpu.solve(&ramp(&base, 40), &opts).unwrap();
    let other = gpu.solve(&ramp(&base, 7), &opts).unwrap();
    let again = gpu.solve(&ramp(&base, 40), &opts).unwrap();
    assert_eq!(other.scenarios.len(), 7);
    assert_same(&again, &first, &[], "solved again");
    assert_eq!(
        gpu.solve_contingencies(&ramp(&base, 2), &opts).unwrap_err(),
        BatchError::BranchesUnknown
    );
}
