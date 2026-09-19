//! Ad-hoc runtime benchmark for AC N-1 contingency screening. Not part of the
//! test suite.
//!
//! Usage: cargo run --release --example bench_contingency -- <input.json> [max-outages] [backend]
//!
//! Sweeps single-branch outages through `BatchSolver::solve_contingencies` and
//! compares against solving each outaged network independently — the same work
//! with no factorization reuse, which is what a naive contingency loop does.
//!
//! The reuse being measured is narrower than an ordinary batch's. Each outage
//! has its own Y-bus, so the Jacobian's cached per-nonzero recipe must be
//! re-analyzed every scenario; what survives is the *symbolic* factorization,
//! because `build_ybus_with_outages` keeps the outaged branch's structural
//! entries and so leaves the sparsity pattern untouched. Contingencies that
//! sever the network cannot use that and fall back to a full rebuild, so the
//! reported split tells you how much of the sweep took which path.

use std::env;
use std::fs;
use std::time::Instant;

use gridoxide::batch::{BatchSolver, Scenario};
use gridoxide::network::{build_ybus, stamp_shunts, structural_component_count};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches};
use gridoxide::run_power_flow_analysis_from_ybus;
use gridoxide::solver::JacobianBackend;
use gridoxide::types::{Line, Transformer};

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: bench_contingency <input.json> [max-outages] [backend]");
    let max_outages: usize =
        args.next().map(|s| s.parse().expect("max-outages must be an integer")).unwrap_or(200);
    let backend = match args.next().as_deref() {
        None | Some("scalar") => JacobianBackend::Scalar,
        Some("klu_native") => JacobianBackend::KluNative,
        Some(other) => panic!("unknown backend {other:?}"),
    };

    let raw = fs::read_to_string(&path).expect("unable to read input");
    let input = serde_json::from_str(&raw).expect("unable to parse input");
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n_branches = lines.len() + transformers.len();
    let sweep = max_outages.min(n_branches);

    let intact = structural_component_count(buses.len(), &lines, &transformers, &[]);
    let severing = (0..sweep)
        .filter(|&b| {
            let mut o = vec![false; n_branches];
            o[b] = true;
            structural_component_count(buses.len(), &lines, &transformers, &o) != intact
        })
        .count();

    println!(
        "{} bus(es), {n_branches} branch(es); sweeping {sweep} outage(s), {severing} severing",
        buses.len()
    );

    let scenarios: Vec<Scenario> = (0..sweep)
        .map(|b| {
            let mut sc = Scenario::new(vec![]);
            sc.branch_outages = vec![b];
            sc
        })
        .collect();

    // Single-threaded on both sides, so the comparison is about factorization
    // reuse rather than about cores.
    let batch = BatchSolver::with_threads(backend, 1).expect("thread pool");
    let start = Instant::now();
    let reports = batch
        .solve_contingencies(&buses, &lines, &transformers, &shunts, &scenarios, 1e-6, 20)
        .expect("contingency sweep failed");
    let batched_ms = start.elapsed().as_secs_f64() * 1e3;
    let converged = reports
        .iter()
        .filter(|r| r.stats.status == gridoxide::solver::SolveStatus::Converged)
        .count();

    let start = Instant::now();
    for b in 0..sweep {
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
        let mut ybus = build_ybus(buses.len(), &keep_lines, &keep_transformers);
        stamp_shunts(&mut ybus, &shunts);
        let report = run_power_flow_analysis_from_ybus(buses.clone(), ybus);
        std::hint::black_box(report.buses.len());
    }
    let independent_ms = start.elapsed().as_secs_f64() * 1e3;

    println!("  batched     {batched_ms:9.2} ms  ({:.3} ms/outage)", batched_ms / sweep as f64);
    println!(
        "  independent {independent_ms:9.2} ms  ({:.3} ms/outage)",
        independent_ms / sweep as f64
    );
    println!("  speedup     {:9.2}x", independent_ms / batched_ms);
    println!("  {converged}/{sweep} converged");
}
