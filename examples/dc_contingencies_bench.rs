//! Where DC N-1's time goes: `DcSensitivity::contingencies` over every
//! connected outage of a MATPOWER case, at several thread counts, against the
//! triangular solves alone.
//!
//! ```text
//! cargo run --release --features matpower --example dc_contingencies_bench -- <case.m>
//! ```

use std::time::Instant;

use gridoxide::linear::{DcOptions, DcSensitivity, dc_branches, dc_power_flow};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dc_contingencies_bench <case.m>");
    let net = gridoxide::matpower::read(&path).unwrap();
    let opts = DcOptions::default();
    let (lines, transformers) = (&net.lines, &net.transformers);
    let n_branches = lines.len() + transformers.len();
    let mut buses = net.buses.clone();
    let base = dc_power_flow(&mut buses, lines, transformers, opts);
    let angles: Vec<f64> = buses.iter().map(|b| b.voltage_ang).collect();
    let t = Instant::now();
    let sensitivity =
        DcSensitivity::new(&buses, &dc_branches(lines, transformers, opts), n_branches).unwrap();
    let factor_ms = t.elapsed().as_secs_f64() * 1e3;
    let outages: Vec<usize> = (0..n_branches)
        .filter(|&b| !sensitivity.is_radial(b))
        .collect();
    println!(
        "{path}: {} buses, {n_branches} branches, {} outages, factorization {factor_ms:.1} ms",
        buses.len(),
        outages.len()
    );
    for threads in [1, 8, 30] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            let out = pool.install(|| sensitivity.contingencies(&angles, &base.branch_p, &outages));
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            assert!(out.unwrap().solved.iter().all(|&s| s));
        }
        println!(
            "  {threads:2} threads: contingencies {best:8.1} ms ({:.3} ms per outage)",
            best / outages.len() as f64
        );
    }
}
