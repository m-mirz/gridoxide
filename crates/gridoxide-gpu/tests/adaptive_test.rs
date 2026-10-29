//! The batched Newton loop on real cases, over the CPU stand-in for cuDSS's
//! uniform batch: the answers it must give before a device backend exists.

use std::path::PathBuf;

use gridoxide::batch::{BatchSolver, BatchStart};
use gridoxide::network::{YBusSparse, build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide::types::Bus;
use gridoxide_gpu::{BatchOptions, CpuUniformBatch, Outcome, StepPolicy, solve_batch};

/// A case's solved base state and its Y-bus. Fails loudly without the
/// benchmark-grids submodule, so a missing case cannot pass as green.
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

/// grid-bench's kind of sweep: every injection scaled along a daily curve
/// between 0.6 and 1.0, with 5% deterministic noise per bus and scenario.
fn sweep(base: &[Bus], batch: usize) -> Vec<Scenario> {
    let mut state = 0x5eed_u64;
    let mut uniform = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..batch)
        .map(|k| {
            let t = k as f64 / batch as f64;
            let scale = 0.6 + 0.4 * (0.5 - 0.5 * (2.0 * std::f64::consts::PI * t).cos());
            Scenario::new(
                base.iter()
                    .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
                    .map(|b| {
                        let (u, v) = (uniform().max(1e-300), uniform());
                        let e = (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos();
                        let f = scale * (1.0 + 0.05 * e);
                        BusOverride::new(b.idx).p(b.p_spec * f).q(b.q_spec * f)
                    })
                    .collect(),
            )
        })
        .collect()
}

/// Every scenario converges to what the CPU `BatchSolver` reaches from the
/// same start, in the same number of Newton steps: the loop is Newton-Raphson,
/// only batched.
#[test]
fn every_scenario_matches_the_cpu_batch_solver() {
    let (base, ybus) = load("matpower/case1354pegase.m");
    let scenarios = sweep(&base, 12);
    let opts = BatchOptions {
        tol: 1e-10,
        ..BatchOptions::default()
    };
    let report = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &opts).unwrap();

    let mut cpu = BatchSolver::with_threads(JacobianBackend::KluNative, 2).unwrap();
    cpu.set_start(BatchStart::Template);
    let want = cpu
        .solve_voltages(&base, &ybus, &scenarios, opts.tol, opts.max_iter)
        .unwrap();

    for (got, want) in report.scenarios.iter().zip(&want) {
        assert_eq!(got.outcome, Outcome::Converged);
        assert_eq!(want.stats.status, SolveStatus::Converged);
        // `iterations()` counts mismatch evaluations: one more than steps.
        assert_eq!(got.newton_steps, want.stats.iterations() - 1);
        for (a, b) in got.voltage_mag.iter().zip(&want.voltage_mag) {
            assert!((a - b).abs() < 1e-9, "|V| {a} against {b}");
        }
        for (a, b) in got.voltage_ang.iter().zip(&want.voltage_ang) {
            assert!((a - b).abs() < 1e-9, "angle {a} against {b}");
        }
    }
    assert_eq!(report.linear.systems_factored, report.linear.systems_solved);
}

/// gpusim2grid's fixed four steps leave part of a heavily loaded feeder's
/// sweep short of the tolerance (grid-bench rejects its mvlv1004#sweep); the
/// adaptive loop takes the steps each scenario needs and converges all.
#[test]
fn a_fixed_step_count_misses_what_the_adaptive_loop_converges() {
    let (base, ybus) = load("generated/mvlv1004.m");
    let scenarios = sweep(&base, 40);
    let adaptive = BatchOptions::default();
    let fixed = BatchOptions {
        policy: StepPolicy::Fixed(4),
        ..adaptive
    };
    let fixed = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &fixed).unwrap();
    let adaptive = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &adaptive).unwrap();

    let short = fixed
        .scenarios
        .iter()
        .filter(|s| s.outcome == Outcome::NotConverged)
        .count();
    assert!(short > 0, "four steps were enough for every scenario here");
    assert!(fixed.scenarios.iter().all(|s| s.newton_steps == 4));
    assert!(
        adaptive
            .scenarios
            .iter()
            .all(|s| s.outcome == Outcome::Converged)
    );
}

/// Reusing a scenario's factors while its mismatch contracts fast enough
/// converges every scenario with fewer factorizations than Newton steps, to
/// the same answer within the tolerance's reach.
#[test]
fn factor_reuse_converges_with_fewer_factorizations() {
    let (base, ybus) = load("matpower/case1354pegase.m");
    let scenarios = sweep(&base, 12);
    let newton = BatchOptions {
        tol: 1e-10,
        ..BatchOptions::default()
    };
    let reuse = BatchOptions {
        reuse_below: Some(0.1),
        ..newton
    };
    let newton = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &newton).unwrap();
    let reuse = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &reuse).unwrap();

    for (a, b) in newton.scenarios.iter().zip(&reuse.scenarios) {
        assert_eq!(b.outcome, Outcome::Converged);
        assert!(b.factorizations < b.newton_steps);
        for (x, y) in a.voltage_mag.iter().zip(&b.voltage_mag) {
            assert!((x - y).abs() < 1e-8);
        }
    }
    assert!(reuse.linear.systems_factored < newton.linear.systems_factored);
}

/// Handing every active scenario to the CPU at once gives the CPU's answer
/// and says so per scenario.
#[test]
fn a_cpu_handoff_finishes_on_the_cpu() {
    let (base, ybus) = load("matpower/case1354pegase.m");
    let scenarios = sweep(&base, 6);
    let batch = BatchOptions {
        tol: 1e-10,
        ..BatchOptions::default()
    };
    let handoff = BatchOptions {
        cpu_handoff: Some(6),
        ..batch
    };
    let batch = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &batch).unwrap();
    let handoff = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &handoff).unwrap();

    assert_eq!(handoff.cpu_handoffs, 6);
    assert_eq!(handoff.linear.systems_factored, 0);
    for (a, b) in batch.scenarios.iter().zip(&handoff.scenarios) {
        assert!(b.on_cpu);
        assert_eq!(b.outcome, Outcome::Converged);
        assert_eq!(a.newton_steps, b.newton_steps);
        for (x, y) in a.voltage_mag.iter().zip(&b.voltage_mag) {
            assert!((x - y).abs() < 1e-9);
        }
    }
}

/// A scenario with no solution is flagged on its own; the rest of the batch
/// converges as if it were not there.
#[test]
fn a_hopeless_scenario_does_not_disturb_the_rest() {
    let (base, ybus) = load("matpower/case14.m");
    let mut scenarios = sweep(&base, 4);
    // Fifty times the load: no operating point exists.
    scenarios[2] = Scenario::new(
        base.iter()
            .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
            .map(|b| {
                BusOverride::new(b.idx)
                    .p(b.p_spec * 50.0)
                    .q(b.q_spec * 50.0)
            })
            .collect(),
    );
    let alone: Vec<Scenario> = [0, 1, 3].iter().map(|&k| scenarios[k].clone()).collect();
    let opts = BatchOptions::default();
    let all = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &opts).unwrap();
    let rest = solve_batch::<CpuUniformBatch>(&base, &ybus, &alone, &opts).unwrap();

    assert_ne!(all.scenarios[2].outcome, Outcome::Converged);
    for (k, r) in [0, 1, 3].iter().zip(&rest.scenarios) {
        let a = &all.scenarios[*k];
        assert_eq!(a.outcome, Outcome::Converged);
        assert_eq!(a.newton_steps, r.newton_steps);
        assert_eq!(a.voltage_mag, r.voltage_mag);
    }
}
