//! DC N-1 on the device (`GpuDcSolver`) against the CPU's
//! (`DcSensitivity::contingencies` on every core), every connected outage of
//! a MATPOWER case, with the largest difference between the two.
//!
//! ```text
//! cargo run --release -p gridoxide-gpu --features cuda --example gpu_dc_bench -- <case.m> [capacity]
//! ```
//!
//! Each side is timed as the median of three after one untimed run: the
//! device from the base case's CPU solve to the outages' rows on the host,
//! the CPU from the same base solve (its factorization kept, as the device
//! session's).

use std::time::Instant;

use gridoxide::linear::{DcOptions, DcSensitivity, dc_branches, dc_power_flow};
use gridoxide_gpu::GpuDcSolver;

/// The median of three, after one untimed run; each run's result is dropped
/// outside the clock (freeing case9241pegase's 2.9 GB takes a tenth of a
/// second or more, the caller's cost, not the solver's).
fn timed<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut out = f();
    let mut ms = Vec::new();
    for _ in 0..3 {
        drop(out);
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
        .expect("usage: gpu_dc_bench <case.m> [capacity]");
    let net = gridoxide::matpower::read(path).unwrap();
    let opts = DcOptions::default();
    let (lines, transformers) = (&net.lines, &net.transformers);
    let n_br = lines.len() + transformers.len();
    let mut buses = net.buses.clone();
    dc_power_flow(&mut buses, lines, transformers, opts);
    let sensitivity =
        DcSensitivity::new(&buses, &dc_branches(lines, transformers, opts), n_br).unwrap();
    let outages: Vec<usize> = (0..n_br).filter(|&b| !sensitivity.is_radial(b)).collect();
    let capacity: usize = args.get(2).map_or(outages.len(), |c| c.parse().unwrap());

    let t = Instant::now();
    let mut gpu = GpuDcSolver::new(&net.buses, lines, transformers, opts, capacity).unwrap();
    let open_ms = t.elapsed().as_secs_f64() * 1e3;
    let (gpu_ms, got) = timed(|| gpu.solve_contingencies(&net.buses, &outages).unwrap());
    let (cpu_ms, want) = timed(|| {
        let mut b = net.buses.clone();
        let base = dc_power_flow(&mut b, lines, transformers, opts);
        let angles: Vec<f64> = b.iter().map(|x| x.voltage_ang).collect();
        sensitivity
            .contingencies(&angles, &base.branch_p, &outages)
            .unwrap()
    });
    let (mut da, mut df) = (0.0f64, 0.0f64);
    for k in 0..outages.len() {
        assert_eq!(got.solved[k], want.solved[k]);
        for (a, b) in got.angles(k).iter().zip(want.angles(k)) {
            da = da.max((a - b).abs());
        }
        for (a, b) in got.flows(k).iter().zip(want.flows(k)) {
            df = df.max((a - b).abs());
        }
    }
    let gb = outages.len() as f64 * (net.buses.len() + n_br) as f64 * 8.0 / 1e9;
    println!(
        "{path}: {} outages, {gb:.2} GB of angles and flows, capacity {capacity}\n  \
         gpu: open {open_ms:.0} ms, solve {gpu_ms:.1} ms\n  \
         cpu: {} threads, {cpu_ms:.1} ms\n  \
         gpu / cpu {:.2}x; max |dθ| {da:.1e} rad, max |dp| {df:.1e} p.u.",
        outages.len(),
        rayon::current_num_threads(),
        gpu_ms / cpu_ms
    );
}
