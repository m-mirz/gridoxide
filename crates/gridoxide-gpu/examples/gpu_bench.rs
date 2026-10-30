//! Wall-clock time of the device loop ([`GpuBatchSolver`]) against
//! gridoxide's CPU `BatchSolver` on all its threads, on the same scenarios
//! from the same start, with outcomes and voltage differences.
//!
//! ```text
//! cargo run --release -p gridoxide-gpu --features cuda --example gpu_bench -- \
//!     <case.m> [B] [sweep|n1] [capacity] [tol] [handoff=N] [reuse=R]
//! ```
//!
//! The scenarios are `batch_counts`'s: a grid-bench sweep (a daily load curve
//! between 0.6 and 1.0 with 5% noise per bus) or B single-branch outages
//! spread over the branch list (B = 0: every branch). Both solvers start
//! every scenario from the solved base case; the device solves in chunks of
//! `capacity` (default min(B, 1024)). The session's analysis is timed apart:
//! a session keeps it for every later solve. Each solve is timed as the
//! median of three, after one untimed run. `handoff=N` finishes the last N
//! active scenarios of each chunk on the CPU, `reuse=R` keeps a scenario on
//! its factors while its mismatch falls more than R-fold
//! (`BatchOptions::cpu_handoff`, `reuse_below`).
//!
//! For N-1 the CPU is timed on the outages the device solves, those that keep
//! the network connected, so that both columns do the same work; the outages
//! that split it, which the device leaves to the CPU, are timed apart.

use std::time::Instant;

use gridoxide::batch::{BatchSolver, BatchStart};
use gridoxide::network::{build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide_gpu::{BatchOptions, BatchReport, GpuBatchSolver, Outcome};

/// A deterministic standard normal: splitmix64 into Box-Muller.
fn normal(state: &mut u64) -> f64 {
    let mut next = || {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let (u, v) = (next().max(1e-300), next());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

/// The median wall time of three runs of `f`, in ms, after one untimed run;
/// and the last run's result.
fn timed<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut out = f();
    let mut ms = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        out = f();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    (ms[1], out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: gpu_bench <case.m> [B] [sweep|n1] [capacity] [tol]");
    let batch: usize = args.get(2).map_or(1000, |b| b.parse().unwrap());
    let n1 = match args.get(3).map(String::as_str) {
        None | Some("sweep") => false,
        Some("n1") => true,
        Some(other) => panic!("unknown kind {other}: sweep or n1"),
    };
    let tol: f64 = args.get(5).map_or(1e-6, |t| t.parse().unwrap());

    let net = gridoxide::matpower::read(path).unwrap();
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let mut base = net.buses.clone();
    linear_initial_guess(&mut base, &ybus);
    PersistentSolver::new(JacobianBackend::KluNative).solve(&mut base, &ybus, 1e-10, 30);

    let n_branches = net.lines.len() + net.transformers.len();
    let mut rng = 0x5eed_u64;
    let scenarios: Vec<Scenario> = if n1 {
        let count = if batch == 0 { n_branches } else { batch };
        (0..count)
            .map(|k| Scenario {
                bus_overrides: Vec::new(),
                branch_outages: vec![k * n_branches / count],
            })
            .collect()
    } else {
        (0..batch)
            .map(|k| {
                let t = k as f64 / batch as f64;
                let scale = 0.6 + 0.4 * (0.5 - 0.5 * (2.0 * std::f64::consts::PI * t).cos());
                Scenario::new(
                    base.iter()
                        .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
                        .map(|b| {
                            let f = scale * (1.0 + 0.05 * normal(&mut rng));
                            BusOverride::new(b.idx).p(b.p_spec * f).q(b.q_spec * f)
                        })
                        .collect(),
                )
            })
            .collect()
    };
    let capacity: usize = args
        .get(4)
        .map_or(scenarios.len().min(1024), |c| c.parse().unwrap());
    let mut opts = BatchOptions {
        tol,
        ..BatchOptions::default()
    };
    for arg in args.iter().skip(6) {
        match arg.split_once('=') {
            Some(("handoff", n)) => opts.cpu_handoff = Some(n.parse().unwrap()),
            Some(("reuse", r)) => opts.reuse_below = Some(r.parse().unwrap()),
            _ => panic!("unknown option {arg}: handoff=N or reuse=R"),
        }
    }

    let t = Instant::now();
    let mut gpu = if n1 {
        GpuBatchSolver::with_branches(&base, &net.lines, &net.transformers, &net.shunts, capacity)
    } else {
        GpuBatchSolver::new(&base, &ybus, capacity)
    }
    .unwrap();
    let open_ms = t.elapsed().as_secs_f64() * 1e3;
    let (gpu_ms, report): (f64, BatchReport) = timed(|| {
        if n1 {
            gpu.solve_contingencies(&scenarios, &opts)
        } else {
            gpu.solve(&scenarios, &opts)
        }
        .unwrap()
    });
    let mut cpu = BatchSolver::new(JacobianBackend::KluNative);
    cpu.set_start(BatchStart::Template);
    // The scenarios the device solved, and those it left to the CPU.
    let (solved, splits): (Vec<usize>, Vec<usize>) =
        (0..scenarios.len()).partition(|&s| report.scenarios[s].outcome != Outcome::Splits);
    let pick =
        |set: &[usize]| -> Vec<Scenario> { set.iter().map(|&s| scenarios[s].clone()).collect() };
    let cpu_solve = |set: &[Scenario]| {
        if n1 {
            cpu.solve_contingencies(
                &base,
                &net.lines,
                &net.transformers,
                &net.shunts,
                set,
                tol,
                opts.max_iter,
            )
        } else {
            cpu.solve(&base, &ybus, set, tol, opts.max_iter)
        }
        .unwrap()
    };
    let (cpu_ms, cpu_reports) = timed(|| cpu_solve(&pick(&solved)));
    let split_ms = if splits.is_empty() {
        0.0
    } else {
        timed(|| cpu_solve(&pick(&splits))).0
    };

    let count = |o: Outcome| report.scenarios.iter().filter(|r| r.outcome == o).count();
    let cpu_converged = cpu_reports
        .iter()
        .filter(|r| r.stats.status == SolveStatus::Converged)
        .count();
    // Largest voltage difference over the scenarios both converged.
    let mut dv = 0.0f64;
    for (g, c) in solved
        .iter()
        .map(|&s| &report.scenarios[s])
        .zip(&cpu_reports)
    {
        if g.outcome == Outcome::Converged && c.stats.status == SolveStatus::Converged {
            for (i, b) in c.buses.iter().enumerate() {
                dv = dv
                    .max((g.voltage_mag[i] - b.voltage_mag).abs())
                    .max((g.voltage_ang[i] - b.voltage_ang).abs());
            }
        }
    }
    let steps: usize = report.scenarios.iter().map(|r| r.newton_steps).sum();

    println!(
        "{path}: {} buses, {} {}, tol {tol:e}, capacity {capacity}, hand-off {:?}, \
         reuse {:?}, cuDSS {:?}",
        net.buses.len(),
        scenarios.len(),
        if n1 { "outages" } else { "sweep scenarios" },
        opts.cpu_handoff,
        opts.reuse_below,
        gridoxide_gpu::cudss::version()
    );
    println!(
        "  gpu: open (analysis) {open_ms:.1} ms, solve {gpu_ms:.1} ms ({:.3} ms/scenario), \
         {} converged, {} not, {} diverged, {} split; {} batch steps, {:.2} steps/scenario, \
         {} systems factored, {} solved, {} on the CPU",
        gpu_ms / scenarios.len() as f64,
        count(Outcome::Converged),
        count(Outcome::NotConverged),
        count(Outcome::Diverged),
        count(Outcome::Splits),
        report.batch_steps,
        steps as f64 / scenarios.len() as f64,
        report.linear.systems_factored,
        report.linear.systems_solved,
        report.cpu_handoffs
    );
    println!(
        "  cpu: {} threads, solve {cpu_ms:.1} ms ({:.3} ms/scenario) of the {} the gpu solved, \
         {cpu_converged} converged; the {} that split the network {split_ms:.1} ms",
        cpu.threads(),
        cpu_ms / solved.len() as f64,
        solved.len(),
        splits.len()
    );
    println!(
        "  gpu / cpu: {:.2}x the time; max |dV| where both converged: {dv:.1e}",
        gpu_ms / cpu_ms
    );
}
