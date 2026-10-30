//! N-1 as one uniform batch, over the CPU stand-in: every outage that keeps
//! the network connected solved to what the CPU's in-place
//! `solve_contingencies` reaches, and every one that splits it flagged.

use std::path::PathBuf;

use gridoxide::batch::{BatchSolver, BatchStart};
use gridoxide::network::{build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide_gpu::{
    BatchError, BatchOptions, CpuUniformBatch, Outcome, StepPolicy, solve_batch,
    solve_contingency_batch,
};

struct Case {
    net: gridoxide::matpower::MatpowerNetwork,
    base: Vec<gridoxide::types::Bus>,
}

/// A case and its solved base state. Fails loudly without the
/// benchmark-grids submodule, so a missing case cannot pass as green.
fn load(rel: &str) -> Case {
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
    Case { net, base }
}

/// `count` single-branch outages spread over the whole branch list.
fn outages(case: &Case, count: usize) -> Vec<Scenario> {
    let n_branches = case.net.lines.len() + case.net.transformers.len();
    (0..count)
        .map(|k| Scenario {
            bus_overrides: Vec::new(),
            branch_outages: vec![k * n_branches / count],
        })
        .collect()
}

/// Every connected outage converges to the CPU's answer from the same start
/// in the same number of steps; the splitting ones are flagged, and the CPU
/// agrees that they split.
#[test]
fn every_outage_matches_the_cpu_contingency_solver() {
    let case = load("matpower/case1354pegase.m");
    let net = &case.net;
    let scenarios = outages(&case, 48);
    let opts = BatchOptions {
        tol: 1e-10,
        ..BatchOptions::default()
    };
    let report = solve_contingency_batch::<CpuUniformBatch>(
        &case.base,
        &net.lines,
        &net.transformers,
        &net.shunts,
        &scenarios,
        &opts,
    )
    .unwrap();

    let mut cpu = BatchSolver::with_threads(JacobianBackend::KluNative, 2).unwrap();
    cpu.set_start(BatchStart::Template);
    let want = cpu
        .solve_contingencies(
            &case.base,
            &net.lines,
            &net.transformers,
            &net.shunts,
            &scenarios,
            opts.tol,
            opts.max_iter,
        )
        .unwrap();

    let mut split = 0;
    for (got, want) in report.scenarios.iter().zip(&want) {
        if got.outcome == Outcome::Splits {
            assert!(want.islands.len() > 1, "the CPU sees one island");
            split += 1;
            continue;
        }
        assert_eq!(want.islands.len(), 1);
        assert_eq!(got.outcome, Outcome::Converged);
        assert_eq!(want.stats.status, SolveStatus::Converged);
        assert_eq!(got.newton_steps, want.stats.iterations() - 1);
        for (a, b) in got.voltage_mag.iter().zip(&want.buses) {
            assert!((a - b.voltage_mag).abs() < 1e-9, "|V| {a} against {b:?}");
        }
        for (a, b) in got.voltage_ang.iter().zip(&want.buses) {
            assert!((a - b.voltage_ang).abs() < 1e-9, "angle {a} against {b:?}");
        }
    }
    // case1354pegase has radial branches: the sample must hit some.
    assert!(split > 0 && split < scenarios.len(), "{split} split");
}

/// Outages and injection overrides together, and the factor-reuse policy:
/// still the CPU's answers, within the tolerance's reach.
#[test]
fn outages_combine_with_overrides_and_factor_reuse() {
    let case = load("matpower/case1354pegase.m");
    let net = &case.net;
    let mut scenarios = outages(&case, 16);
    for (k, sc) in scenarios.iter_mut().enumerate() {
        let f = 0.8 + 0.02 * k as f64;
        sc.bus_overrides = case
            .base
            .iter()
            .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
            .map(|b| BusOverride::new(b.idx).p(b.p_spec * f).q(b.q_spec * f))
            .collect();
    }
    let opts = BatchOptions {
        tol: 1e-10,
        ..BatchOptions::default()
    };
    let reuse = BatchOptions {
        reuse_below: Some(0.1),
        ..opts
    };
    let run = |opts: &BatchOptions| {
        solve_contingency_batch::<CpuUniformBatch>(
            &case.base,
            &net.lines,
            &net.transformers,
            &net.shunts,
            &scenarios,
            opts,
        )
        .unwrap()
    };
    let (newton, reuse) = (run(&opts), run(&reuse));
    let mut cpu = BatchSolver::with_threads(JacobianBackend::KluNative, 2).unwrap();
    cpu.set_start(BatchStart::Template);
    let want = cpu
        .solve_contingency_voltages(
            &case.base,
            &net.lines,
            &net.transformers,
            &net.shunts,
            &scenarios,
            opts.tol,
            opts.max_iter,
        )
        .unwrap();
    for ((a, b), w) in newton.scenarios.iter().zip(&reuse.scenarios).zip(&want) {
        if a.outcome == Outcome::Splits {
            assert_eq!(b.outcome, Outcome::Splits);
            continue;
        }
        assert_eq!(a.outcome, Outcome::Converged);
        assert_eq!(b.outcome, Outcome::Converged);
        assert_eq!(a.newton_steps, w.stats.iterations() - 1);
        for ((x, y), z) in a.voltage_mag.iter().zip(&b.voltage_mag).zip(&w.voltage_mag) {
            assert!((x - z).abs() < 1e-9);
            assert!((y - z).abs() < 1e-8);
        }
    }
    assert!(reuse.linear.systems_factored < newton.linear.systems_factored);
}

/// The CPU hand-off takes each outage's admittances in place, as the batch
/// does, and restores them before the next.
#[test]
fn a_cpu_handoff_solves_each_outage_on_its_own_network() {
    let case = load("matpower/case1354pegase.m");
    let net = &case.net;
    let scenarios = outages(&case, 12);
    let opts = BatchOptions {
        tol: 1e-10,
        ..BatchOptions::default()
    };
    let handoff = BatchOptions {
        cpu_handoff: Some(scenarios.len()),
        ..opts
    };
    let run = |opts: &BatchOptions| {
        solve_contingency_batch::<CpuUniformBatch>(
            &case.base,
            &net.lines,
            &net.transformers,
            &net.shunts,
            &scenarios,
            opts,
        )
        .unwrap()
    };
    let (batch, handoff) = (run(&opts), run(&handoff));
    assert_eq!(handoff.linear.systems_factored, 0);
    for (a, b) in batch.scenarios.iter().zip(&handoff.scenarios) {
        assert_eq!(a.outcome, b.outcome);
        if a.outcome == Outcome::Splits {
            continue;
        }
        assert!(b.on_cpu);
        assert_eq!(a.newton_steps, b.newton_steps);
        for (x, y) in a.voltage_mag.iter().zip(&b.voltage_mag) {
            assert!((x - y).abs() < 1e-9);
        }
    }
}

/// N-1 is where a fixed step count is most exposed: outages move the
/// operating point unevenly, so the steps they need spread.
#[test]
fn outages_need_uneven_step_counts() {
    let case = load("matpower/case1354pegase.m");
    let net = &case.net;
    let scenarios = outages(&case, 48);
    let report = solve_contingency_batch::<CpuUniformBatch>(
        &case.base,
        &net.lines,
        &net.transformers,
        &net.shunts,
        &scenarios,
        &BatchOptions::default(),
    )
    .unwrap();
    let steps: Vec<usize> = report
        .scenarios
        .iter()
        .filter(|s| s.outcome == Outcome::Converged)
        .map(|s| s.newton_steps)
        .collect();
    assert!(steps.iter().min() < steps.iter().max(), "{steps:?}");

    let fixed = BatchOptions {
        policy: StepPolicy::Fixed(*steps.iter().min().unwrap()),
        ..BatchOptions::default()
    };
    let fixed = solve_contingency_batch::<CpuUniformBatch>(
        &case.base,
        &net.lines,
        &net.transformers,
        &net.shunts,
        &scenarios,
        &fixed,
    )
    .unwrap();
    assert!(
        fixed
            .scenarios
            .iter()
            .any(|s| s.outcome == Outcome::NotConverged)
    );
}

/// `solve_batch` takes a finished Y-bus, from which no branch can be taken:
/// it refuses an outage rather than solving the intact network.
#[test]
fn solve_batch_refuses_outages() {
    let case = load("matpower/case14.m");
    let mut ybus = build_ybus(
        case.net.buses.len(),
        &case.net.lines,
        &case.net.transformers,
    );
    stamp_shunts(&mut ybus, &case.net.shunts);
    let scenarios = outages(&case, 3);
    let err = solve_batch::<CpuUniformBatch>(
        &case.base,
        &ybus.finish(),
        &scenarios,
        &BatchOptions::default(),
    )
    .unwrap_err();
    assert_eq!(err, BatchError::OutagesUnsupported { scenario: 0 });
}
