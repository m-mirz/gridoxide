//! Ad-hoc runtime benchmark for the DC (Bθ) power flow and its sensitivity
//! factors, against a PGM-format JSON network. Not part of the test suite.
//!
//! Usage: cargo run --release --example bench_dc -- <path-to-input.json> [repeat-count] [mode]
//!
//! `repeat-count` (default 1) re-runs the measured work that many times from a
//! fresh clone of the post-parse buses, so `perf record` gets enough samples to
//! profile — one DC solve is fast enough that a single run is mostly noise.
//!
//! `mode` (default "solve") selects what is measured:
//!
//! - `solve` — one full `dc_power_flow` per repeat: branch reduction, island
//!   partition, one factorization per island, back-substitution, and branch
//!   flows. This is the number to compare against pandapower's `rundcpp` or
//!   lightsim2grid's `dc_pf`.
//! - `ptdf` — one `DcSensitivity::new` (the factorization) plus a full sweep of
//!   `ptdf_column` over every bus. This is the access pattern that motivated
//!   `sparse::RealFactorization`: `n` solves against *one* factorization, where
//!   routing through `solver::LinearSolver` would have refactorized `n` times.
//!   The per-column figure is the one worth quoting.
//! - `lodf` — the same, sweeping `lodf_column` over every branch. Radial
//!   branches return `None` without a back-substitution, so the count of
//!   columns actually produced is reported alongside the timing.
//! - `n2` — an N-2 sweep: `multi_outage_flows` over every branch *pair* drawn
//!   from the first `repeat` branches, which is the shape a contingency screen
//!   actually has. Breaking pairs are reported separately, since they return
//!   without a solve.
//! - `batch` — `repeat` load-scaling scenarios through `DcBatchSolver`, which
//!   performs exactly *one* numeric factorization for the whole batch (DC's `B`
//!   depends only on topology, so a scenario changes only the right-hand side).
//!   Compare its per-scenario figure against `solve`'s per-run figure to see
//!   what hoisting the factorization is worth.
//!
//! Note that neither sensitivity mode materializes a dense matrix: at
//! `case9241pegase` those are 1.19 GB and 2.06 GB respectively, and the point
//! of the column API is not paying that.

use std::env;
use std::fs;
use std::time::Instant;

use gridoxide::batch::{uniform_load_scaling, Scenario};
use gridoxide::linear::{dc_branches, dc_power_flow, DcBatchSolver, DcOptions, DcSensitivity};
use gridoxide::pgm::pgm_to_buses_and_branches;

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: bench_dc <input.json> [repeat-count] [mode]");
    let repeat: usize =
        args.next().map(|s| s.parse().expect("repeat-count must be an integer")).unwrap_or(1);
    let mode = args.next().unwrap_or_else(|| "solve".to_string());

    let raw = fs::read_to_string(&path).expect("unable to read input");
    let input = serde_json::from_str(&raw).expect("unable to parse input");
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n_branches = lines.len() + transformers.len();
    let opts = DcOptions::default();

    println!("{} bus(es), {} branch(es), mode {mode}", buses.len(), n_branches);

    let start = Instant::now();
    let mut produced = 0usize;
    match mode.as_str() {
        "solve" => {
            for _ in 0..repeat {
                let mut scratch = buses.clone();
                let solution = dc_power_flow(&mut scratch, &lines, &transformers, opts);
                // Consume the result so the solve cannot be optimized away.
                produced += solution.branch_p.len();
            }
        }
        "ptdf" | "lodf" => {
            for _ in 0..repeat {
                let mut scratch = buses.clone();
                dc_power_flow(&mut scratch, &lines, &transformers, opts);
                let branches = dc_branches(&lines, &transformers, opts);
                let sensitivity = DcSensitivity::new(&scratch, &branches, n_branches)
                    .expect("reduced B is singular, so there are no sensitivities to benchmark");

                produced = 0;
                if mode == "ptdf" {
                    for bus in 0..scratch.len() {
                        if sensitivity.ptdf_column(bus).is_some() {
                            produced += 1;
                        }
                    }
                } else {
                    for branch in 0..n_branches {
                        if sensitivity.lodf_column(branch).is_some() {
                            produced += 1;
                        }
                    }
                }
            }
        }
        "n2" => {
            let mut scratch = buses.clone();
            let base = dc_power_flow(&mut scratch, &lines, &transformers, opts);
            let branches = dc_branches(&lines, &transformers, opts);
            let sensitivity = DcSensitivity::new(&scratch, &branches, n_branches)
                .expect("reduced B is singular, so there are no sensitivities to benchmark");
            let span = repeat.min(n_branches);
            for a in 0..span {
                for b in (a + 1)..span {
                    if sensitivity.multi_outage_flows(&base.branch_p, &[a, b]).is_some() {
                        produced += 1;
                    }
                }
            }
        }
        "batch" => {
            let scenarios: Vec<Scenario> = (0..repeat)
                .map(|k| uniform_load_scaling(&buses, 0.5 + 0.001 * k as f64))
                .collect();
            let results = DcBatchSolver::new()
                .solve(&buses, &lines, &transformers, opts, &scenarios)
                .expect("batch failed");
            produced = results.len();
        }
        other => panic!("unknown mode {other:?}; expected solve, ptdf, lodf, n2 or batch"),
    }
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1e3;
    println!("{total_ms:.3} ms total, {:.3} ms/run over {repeat} run(s)", total_ms / repeat as f64);
    match mode.as_str() {
        "solve" => println!("  {} branch flow(s) per run", produced / repeat.max(1)),
        "n2" => {
            let span = repeat.min(n_branches);
            println!("  {} pair(s) screened, {produced} solvable", span * (span - 1) / 2);
            if produced > 0 {
                println!("  {:.4} ms/solvable pair", total_ms / produced as f64);
            }
        }
        "batch" => {
            println!("  {produced} scenario(s), one factorization for all of them");
            if produced > 0 {
                println!("  {:.4} ms/scenario", total_ms / produced as f64);
            }
        }
        _ => {
            println!("  {produced} column(s) produced per run");
            if produced > 0 {
                println!("  {:.4} ms/column", total_ms / repeat as f64 / produced as f64);
            }
        }
    }
}
