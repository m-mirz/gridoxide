//! A DC sweep on the device (`GpuDcSolver::solve_sweep`) against the CPU's
//! (`DcBatchSolver` on every core): `count` scenarios that scale every load
//! of a MATPOWER case by its own factor, with the largest difference between
//! the two.
//!
//! ```text
//! cargo run --release -p gridoxide-gpu --features cuda --example gpu_dc_sweep_bench -- <case.m> [count] [capacity]
//! ```
//!
//! `count` defaults to 1000, `capacity` to `count`. Each side is timed as
//! the median of three after one untimed run, from the scenarios to every
//! scenario's rows on the host. The CPU builds its factorization inside each
//! call; the device session keeps its own, made once (reported as "open").

use std::time::Instant;

use gridoxide::linear::{DcBatchSolver, DcOptions};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide_gpu::GpuDcSolver;

/// The median of three, after one untimed run; each run's result is dropped
/// outside the clock.
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

/// A fixed pseudo-random factor in [0.8, 1.2), so both sides and every run
/// see the same scenarios.
fn factor(k: usize, bus: usize) -> f64 {
    let mut x = (k as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (bus as u64);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    0.8 + 0.4 * (x >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: gpu_dc_sweep_bench <case.m> [count] [capacity]");
    let count: usize = args.get(2).map_or(1000, |c| c.parse().unwrap());
    let capacity: usize = args.get(3).map_or(count, |c| c.parse().unwrap());
    let net = gridoxide::matpower::read(path).unwrap();
    let opts = DcOptions::default();
    let (lines, transformers) = (&net.lines, &net.transformers);
    let n_br = lines.len() + transformers.len();
    let scenarios: Vec<Scenario> = (0..count)
        .map(|k| {
            Scenario::new(
                net.buses
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| b.p_spec != 0.0)
                    .map(|(i, b)| BusOverride::new(i).p(b.p_spec * factor(k, i)))
                    .collect(),
            )
        })
        .collect();

    let t = Instant::now();
    let mut gpu = GpuDcSolver::new(&net.buses, lines, transformers, opts, capacity).unwrap();
    let open_ms = t.elapsed().as_secs_f64() * 1e3;
    let (gpu_ms, got) = timed(|| gpu.solve_sweep(&net.buses, &scenarios).unwrap());
    let cpu = DcBatchSolver::new();
    let (cpu_ms, want) = timed(|| {
        cpu.solve(&net.buses, lines, transformers, opts, &scenarios)
            .unwrap()
    });
    let (mut da, mut df) = (0.0f64, 0.0f64);
    for (k, w) in want.iter().enumerate() {
        for (a, b) in got.angles(k).iter().zip(&w.voltage_ang) {
            da = da.max((a - b).abs());
        }
        for (a, b) in got.flows(k).iter().zip(&w.branch_p) {
            df = df.max((a - b).abs());
        }
    }
    let gb = count as f64 * (net.buses.len() + n_br) as f64 * 8.0 / 1e9;
    println!(
        "{path}: {count} scenarios, {gb:.2} GB of angles and flows, capacity {capacity}\n  \
         gpu: open {open_ms:.0} ms, solve {gpu_ms:.1} ms\n  \
         cpu: {} threads, {cpu_ms:.1} ms\n  \
         gpu / cpu {:.2}x; max |dθ| {da:.1e} rad, max |dp| {df:.1e} p.u.",
        cpu.threads(),
        gpu_ms / cpu_ms
    );
}
