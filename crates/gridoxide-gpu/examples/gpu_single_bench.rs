//! Whether a single AC or DC power flow is worth doing on the device:
//! `GpuBatchSolver::solve_single` and `GpuDcSolver::solve` against
//! gridoxide's CPU solves, on each MATPOWER case given.
//!
//! ```text
//! cargo run --release -p gridoxide-gpu --features cuda --example gpu_single_bench -- <case.m>... [tol=T]
//! ```
//!
//! Every time is the median of five after one untimed run. Two rows per side:
//!
//! - **cold**: a network seen for the first time. CPU: a new solver
//!   (symbolic and numeric factorization) and its solve. Device: a new
//!   session (cuDSS's analysis; for DC also its factorization) and one
//!   solve.
//! - **warm**: the same network again, the analysis kept. CPU: the solver
//!   kept. Device: the session kept.
//!
//! AC starts from the linear initial guess, on both sides with the same
//! tolerance (default 1e-8), through `PersistentSolver` (KLU) and through a
//! capacity-1 `GpuBatchSolver`, which factors every step.
//!
//! DC is `DcSolver::solve` against `GpuDcSolver::solve`, which solves the
//! reduced system on the device and does the rest on the host, as the CPU.

use std::time::Instant;

use gridoxide::linear::{DcOptions, DcSolver};
use gridoxide::network::{build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide_gpu::{BatchOptions, GpuBatchSolver, GpuDcSolver, Outcome};

/// The median wall time of five runs of `f`, in ms, after one untimed run;
/// and the last run's result.
fn timed<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut out = f();
    let mut ms = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        out = f();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    (ms[2], out)
}

fn ac(net: &gridoxide::matpower::MatpowerNetwork, tol: f64) {
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let mut start = net.buses.clone();
    linear_initial_guess(&mut start, &ybus);
    let max_iter = 30;

    let cpu_once = |solver: &mut PersistentSolver| {
        let mut buses = start.clone();
        let (_, stats) = solver.solve_with_stats(&mut buses, &ybus, tol, max_iter);
        (stats, buses)
    };
    let (cpu_cold, _) = timed(|| cpu_once(&mut PersistentSolver::new(JacobianBackend::KluNative)));
    let mut kept = PersistentSolver::new(JacobianBackend::KluNative);
    let (cpu_warm, (stats, cpu_buses)) = timed(|| cpu_once(&mut kept));

    let opts = BatchOptions {
        tol,
        max_iter,
        ..BatchOptions::default()
    };
    let gpu_once = |session: &mut GpuBatchSolver| {
        let mut buses = start.clone();
        session.solve_single(&mut buses, &opts).unwrap()
    };
    let (gpu_cold, _) = timed(|| gpu_once(&mut GpuBatchSolver::new(&start, &ybus, 1).unwrap()));
    let t = Instant::now();
    let mut session = GpuBatchSolver::new(&start, &ybus, 1).unwrap();
    let open = t.elapsed().as_secs_f64() * 1e3;
    let (gpu_warm, r) = timed(|| gpu_once(&mut session));
    let dv = cpu_buses
        .iter()
        .enumerate()
        .map(|(i, b)| {
            (r.voltage_mag[i] - b.voltage_mag)
                .abs()
                .max((r.voltage_ang[i] - b.voltage_ang).abs())
        })
        .fold(0.0, f64::max);
    println!(
        "  AC (tol {tol:e}): cpu {} in {} iterations, gpu {} in {} steps ({} factorizations); \
         max |dV| {dv:.1e}",
        if stats.status == SolveStatus::Converged {
            "converged"
        } else {
            "did not converge"
        },
        stats.iterations(),
        if r.outcome == Outcome::Converged {
            "converged"
        } else {
            "did not converge"
        },
        r.newton_steps,
        r.factorizations,
    );
    println!(
        "    cold: cpu {cpu_cold:8.2} ms   gpu {gpu_cold:8.2} ms   gpu / cpu {:6.2}x",
        gpu_cold / cpu_cold
    );
    println!(
        "    warm: cpu {cpu_warm:8.2} ms   gpu {gpu_warm:8.2} ms   gpu / cpu {:6.2}x   \
         (gpu analysis {open:.1} ms; {:.2} ms per gpu step)",
        gpu_warm / cpu_warm,
        gpu_warm / r.newton_steps.max(1) as f64
    );
}

fn dc(net: &gridoxide::matpower::MatpowerNetwork) {
    let opts = DcOptions::default();
    let (lines, transformers) = (&net.lines, &net.transformers);
    let (cpu_cold, _) = timed(|| {
        let mut buses = net.buses.clone();
        DcSolver::new(&net.buses, lines, transformers, opts).solve(&mut buses)
    });
    let kept = DcSolver::new(&net.buses, lines, transformers, opts);
    let (cpu_warm, _) = timed(|| {
        let mut buses = net.buses.clone();
        kept.solve(&mut buses)
    });

    let (gpu_cold, _) = timed(|| {
        let mut buses = net.buses.clone();
        GpuDcSolver::new(&net.buses, lines, transformers, opts, 1)
            .unwrap()
            .solve(&mut buses)
            .unwrap()
    });
    let mut session = GpuDcSolver::new(&net.buses, lines, transformers, opts, 1).unwrap();
    let (gpu_warm, _) = timed(|| {
        let mut buses = net.buses.clone();
        session.solve(&mut buses).unwrap()
    });
    println!("  DC:");
    println!(
        "    cold: cpu {cpu_cold:8.2} ms   gpu {gpu_cold:8.2} ms   gpu / cpu {:6.2}x",
        gpu_cold / cpu_cold
    );
    println!(
        "    warm: cpu {cpu_warm:8.2} ms   gpu {gpu_warm:8.2} ms   gpu / cpu {:6.2}x",
        gpu_warm / cpu_warm
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut tol = 1e-8;
    let mut paths = Vec::new();
    for arg in &args {
        match arg.strip_prefix("tol=") {
            Some(t) => tol = t.parse().unwrap(),
            None => paths.push(arg.as_str()),
        }
    }
    assert!(
        !paths.is_empty(),
        "usage: gpu_single_bench <case.m>... [tol=T]"
    );
    println!("cuDSS {:?}", gridoxide_gpu::cudss::version());
    for path in paths {
        let net = gridoxide::matpower::read(path).unwrap();
        println!(
            "{path}: {} buses, {} branches",
            net.buses.len(),
            net.lines.len() + net.transformers.len()
        );
        ac(&net, tol);
        dc(&net);
    }
}
