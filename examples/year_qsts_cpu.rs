//! Investigates a 1-year, 15-minute-resolution QSTS batch (~35,040 scenarios)
//! on the CPU rayon BatchSolver — no GPU memory ceiling, since each worker
//! holds only its own PersistentSolver state, not the whole batch stacked.

use std::env;
use std::fs;
use std::time::Instant;

use gridoxide::batch::{uniform_load_scaling, BatchSolver, Scenario};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches};
use gridoxide::solver::{JacobianBackend, SolveStatus};

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: year_qsts_cpu <input.json> [n_scenarios] [threads]");
    let nb: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(365 * 24 * 4);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(30);

    let raw = fs::read_to_string(&path).expect("read input file");
    let input = serde_json::from_str(&raw).expect("parse PGM input JSON");
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (template, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let mut ybus = build_ybus(template.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let ybus = ybus.finish();

    // Smooth daily+seasonal-ish load curve, deterministic: two superimposed
    // sinusoids over the 15-minute steps in a year, scaled to [0.6, 1.15] of
    // nominal load -- a reasonable QSTS load-shape stand-in.
    println!("case: {path}");
    println!("buses: {}  scenarios: {nb}  threads: {threads}", template.len());

    let steps_per_day = 96.0; // 24h * 4 (15-min steps)
    let scenarios: Vec<Scenario> = (0..nb)
        .map(|k| {
            let t = k as f64;
            let daily = (2.0 * std::f64::consts::PI * t / steps_per_day).sin();
            let seasonal = (2.0 * std::f64::consts::PI * t / (steps_per_day * 365.0)).sin();
            let factor = 0.875 + 0.2 * daily + 0.075 * seasonal;
            uniform_load_scaling(&template, factor)
        })
        .collect();

    let batch = BatchSolver::with_threads(JacobianBackend::KluNative, threads).expect("pool");
    let t0 = Instant::now();
    let reports = batch.solve(&template, &ybus, &scenarios, 1e-8, 20).expect("batch solve");
    let elapsed = t0.elapsed();

    let converged = reports.iter().filter(|r| r.stats.status == SolveStatus::Converged).count();
    let not_converged = nb - converged;

    println!();
    println!("total wall time: {:.2} s", elapsed.as_secs_f64());
    println!("throughput:      {:.1} solves/s", nb as f64 / elapsed.as_secs_f64());
    println!("per-solve:       {:.3} ms", elapsed.as_secs_f64() * 1e3 / nb as f64);
    println!();
    println!("converged:       {converged}/{nb}");
    println!("not converged:   {not_converged}/{nb}");
}
