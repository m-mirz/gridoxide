//! Where do the milliseconds actually go?
//!
//! `plans/GPU_PLAN.md` §1 answered this for the CPU by instrumenting
//! `newton_raphson_native_klu_cached` with per-phase timers; that
//! instrumentation was throwaway and never committed, which is why the GPU
//! path's own 91x regression had to be argued from arithmetic rather than
//! measured. This is the committed equivalent for the batch paths.
//!
//! It runs, on one case at one batch size:
//!
//! 1. the CPU `BatchSolver` (the bar every GPU claim must clear),
//! 2. the **stacked** device path (`solve_batch_block_diagonal_device_resident`),
//! 3. the **batched** device path (`solve_batch_block_diagonal_batched_device`),
//!
//! and for (3) reports a per-phase breakdown from CUDA events — assembly,
//! injections+mismatch, cuDSS refactor+solve, update — so "it is 95% inside
//! cuDSS" is a number on a screen rather than an inference.
//!
//! ```bash
//! cargo run --release --features gpu,cudss --example bde_profile -- \
//!     scripts/bench/.case-cache/case9241pegase.json 256 [threads]
//! ```
//!
//! Timings are wall clock for whole solves, so they include setup
//! (`linear_initial_guess` per scenario, symbolic analysis) the way a caller
//! would actually experience it.
//!
//! Two things to keep in mind when reading the phase table:
//!
//! - It will not sum to the wall time. The gap is setup, and it is reported.
//! - The profiling run adds a `cudaStreamSynchronize` per iteration to read
//!   the events back, which the production loop does not have. **Use the
//!   phase table for proportions, and the wall-clock rows above it for
//!   absolute numbers.**

use std::env;
use std::fs;
use std::time::Instant;

use gridoxide::batch::{uniform_load_scaling, BatchSolver, Scenario};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches};
use gridoxide::solver::{JacobianBackend, SolveStatus};

#[cfg(all(feature = "gpu", feature = "cudss"))]
mod phases {
    use gridoxide::device_layout::{build_csr_structure, csr_scatter_map};
    use gridoxide::gpu::{device_memory, GpuBatch, PhaseTimer};
    use gridoxide::jacobian::JacobianPattern;
    use gridoxide::network::YBusSparse;
    use gridoxide::sparse_cudss::CudssBatchedSystem;
    use gridoxide::types::Bus;

    /// Accumulated milliseconds per phase, summed over Newton iterations.
    #[derive(Default)]
    pub struct Breakdown {
        pub iterations: usize,
        pub injections_mismatch: f32,
        pub assemble: f32,
        pub solve: f32,
        pub update: f32,
        /// Host time spent blocked on the convergence-norm copy plus the host
        /// bookkeeping around it — the only per-iteration host cost left.
        pub host_sync: f32,
    }

    impl Breakdown {
        pub fn total(&self) -> f32 {
            self.injections_mismatch + self.assemble + self.solve + self.update + self.host_sync
        }

        pub fn report(&self, wall_ms: f64) {
            let t = self.total().max(f32::MIN_POSITIVE);
            let row = |name: &str, ms: f32| {
                println!("  {name:<26} {ms:9.1} ms  {:5.1}%  {:8.3} ms/iter", 100.0 * ms / t, ms / self.iterations.max(1) as f32);
            };
            println!("per-phase, summed over {} Newton iterations:", self.iterations);
            row("injections + mismatch", self.injections_mismatch);
            row("Jacobian assembly", self.assemble);
            row("cuDSS refactor + solve", self.solve);
            row("Newton update", self.update);
            row("host sync + bookkeeping", self.host_sync);
            println!("  {:<26} {:9.1} ms", "phase total", t);
            println!(
                "  {:<26} {:9.1} ms   (setup: initial guess, analysis, readback)",
                "wall minus phases",
                wall_ms - t as f64
            );
        }
    }

    /// The batched Newton loop again, with a CUDA event pair around each
    /// phase. Deliberately a copy of `bde::solve_batch_block_diagonal_batched_device`
    /// rather than a flag on it: the production path should not carry timing
    /// branches, and a profile that quietly diverges from what ships is worse
    /// than none. Keep the two in sync — the phase order below is the contract.
    pub fn profile(
        buses_template: &[Bus],
        ybus: &YBusSparse,
        states: &[Vec<Bus>],
        tol: f64,
        max_iter: usize,
    ) -> Option<Breakdown> {
        let nb = states.len();
        let block = JacobianPattern::analyze(buses_template, ybus);
        let blk = block.n_unknowns;
        let pairs: Vec<(usize, usize)> =
            block.rows().iter().zip(block.cols()).map(|(&r, &c)| (r as usize, c as usize)).collect();
        let scatter = csr_scatter_map(blk, &pairs);
        let (row_ptr, col_idx, _) = build_csr_structure(blk, &pairs);

        if let Some((free, total)) = device_memory() {
            println!(
                "device memory before batch: {:.1} / {:.1} GiB free",
                free as f64 / (1 << 30) as f64,
                total as f64 / (1 << 30) as f64
            );
        }

        let mut gpu = GpuBatch::new(&block, ybus, buses_template, states, &scatter)?;

        let (t_mis, t_asm, t_solve, t_upd) =
            (PhaseTimer::new()?, PhaseTimer::new()?, PhaseTimer::new()?, PhaseTimer::new()?);

        let mut active = vec![true; nb];
        let mut max_mismatch = vec![0.0f64; nb];
        let mut cudss: Option<CudssBatchedSystem> = None;
        let mut out = Breakdown::default();

        for _ in 0..max_iter {
            t_mis.begin(gpu.stream());
            gpu.power_injections()?;
            gpu.mismatch()?;
            t_mis.end(gpu.stream());

            let host_start = std::time::Instant::now();
            gpu.download_max_mismatch(&mut max_mismatch)?;
            for s in 0..nb {
                if active[s] && max_mismatch[s] < tol {
                    active[s] = false;
                }
            }
            let done = active.iter().all(|&a| !a);
            out.host_sync += host_start.elapsed().as_secs_f32() * 1e3;

            // The mismatch phase ran even on the final iteration; count it, then
            // stop before the solve that iteration does not need.
            out.iterations += 1;
            out.injections_mismatch += t_mis.elapsed_ms()?;
            if done {
                break;
            }

            gpu.upload_active(&active)?;
            t_asm.begin(gpu.stream());
            gpu.zero_masked_rhs()?;
            gpu.assemble()?;
            t_asm.end(gpu.stream());

            if cudss.is_none() {
                cudss = CudssBatchedSystem::new(
                    nb,
                    blk,
                    &row_ptr,
                    &col_idx,
                    gpu.values_ptr(),
                    gpu.rhs_ptr(),
                    gpu.dx_ptr(),
                    gpu.stream().as_u64(),
                );
            }
            t_solve.begin(gpu.stream());
            cudss.as_mut()?.refactor_and_solve()?;
            t_solve.end(gpu.stream());

            t_upd.begin(gpu.stream());
            gpu.apply_update()?;
            t_upd.end(gpu.stream());

            gpu.stream().synchronize()?;
            out.assemble += t_asm.elapsed_ms()?;
            out.solve += t_solve.elapsed_ms()?;
            out.update += t_upd.elapsed_ms()?;
        }

        Some(out)
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: bde_profile <input.json> [n_scenarios] [threads]");
    let nb: usize = args.next().map(|s| s.parse().expect("n_scenarios must be an integer")).unwrap_or(256);
    let threads: usize = args.next().map(|s| s.parse().expect("threads must be an integer")).unwrap_or(0);

    let raw = fs::read_to_string(&path).expect("read input file");
    let input = serde_json::from_str(&raw).expect("parse PGM input JSON");
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (template, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let mut ybus = build_ybus(template.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let ybus = ybus.finish();

    // A spread of load scalings, deterministic and identical for every path.
    let scenarios: Vec<Scenario> =
        (0..nb).map(|k| uniform_load_scaling(&template, 0.8 + 0.4 * k as f64 / (nb.max(2) - 1) as f64)).collect();

    println!("case: {path}");
    println!("buses: {}  scenarios: {nb}", template.len());
    println!();

    // ---- The bar: multithreaded CPU. -------------------------------------
    let batch = match threads {
        0 => BatchSolver::new(JacobianBackend::KluNative),
        n => BatchSolver::with_threads(JacobianBackend::KluNative, n).expect("pool"),
    };
    let t0 = Instant::now();
    let cpu = batch.solve(&template, &ybus, &scenarios, 1e-8, 40).expect("cpu batch");
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let cpu_converged = cpu.iter().filter(|r| r.stats.status == SolveStatus::Converged).count();
    println!(
        "CPU BatchSolver ({} threads):   {cpu_ms:9.1} ms   {:8.1} solves/s   ({cpu_converged}/{nb} converged)",
        batch.threads(),
        nb as f64 / (cpu_ms / 1e3)
    );

    #[cfg(all(feature = "gpu", feature = "cudss"))]
    {
        use gridoxide::bde::{solve_batch_block_diagonal_batched_device, solve_batch_block_diagonal_device_resident};
        use gridoxide::network::linear_initial_guess;

        let t1 = Instant::now();
        let stacked = solve_batch_block_diagonal_device_resident(&template, &ybus, &scenarios, 1e-8, 40);
        let stacked_ms = t1.elapsed().as_secs_f64() * 1e3;
        let stacked_ok = stacked.iter().filter(|r| r.stats.status == SolveStatus::Converged).count();
        println!(
            "GPU stacked (control):         {stacked_ms:9.1} ms   {:8.1} solves/s   ({stacked_ok}/{nb} converged)",
            nb as f64 / (stacked_ms / 1e3)
        );

        let t2 = Instant::now();
        let batched = solve_batch_block_diagonal_batched_device(&template, &ybus, &scenarios, 1e-8, 40);
        let batched_ms = t2.elapsed().as_secs_f64() * 1e3;
        let batched_ok = batched.iter().filter(|r| r.stats.status == SolveStatus::Converged).count();
        println!(
            "GPU batched (cuDSS batch API): {batched_ms:9.1} ms   {:8.1} solves/s   ({batched_ok}/{nb} converged)",
            nb as f64 / (batched_ms / 1e3)
        );

        println!();
        println!("speedup vs CPU:  stacked {:.2}x   batched {:.2}x", cpu_ms / stacked_ms, cpu_ms / batched_ms);
        println!("speedup vs stacked control:  batched {:.2}x", stacked_ms / batched_ms);

        // Worst voltage disagreement against the CPU — a timing run that
        // silently stopped being correct is worthless.
        let worst = batched
            .iter()
            .zip(&cpu)
            .flat_map(|(g, c)| g.buses.iter().zip(&c.buses))
            .map(|(a, b)| (a.voltage_mag - b.voltage_mag).abs().max((a.voltage_ang - b.voltage_ang).abs()))
            .fold(0.0f64, f64::max);
        println!("batched vs CPU, worst |dV|: {worst:.3e}");

        println!();
        let states: Vec<Vec<gridoxide::types::Bus>> = scenarios
            .iter()
            .map(|sc| {
                let mut buses = template.to_vec();
                for ov in &sc.bus_overrides {
                    let b = &mut buses[ov.bus];
                    if let Some(p) = ov.p_spec {
                        b.p_spec = p;
                    }
                    if let Some(q) = ov.q_spec {
                        b.q_spec = q;
                    }
                }
                linear_initial_guess(&mut buses, &ybus);
                buses
            })
            .collect();
        match phases::profile(&template, &ybus, &states, 1e-8, 40) {
            Some(b) => b.report(batched_ms),
            None => println!("per-phase profiling failed (see the CUDA error above)"),
        }
    }

    #[cfg(not(all(feature = "gpu", feature = "cudss")))]
    println!("\n(built without --features gpu,cudss — CPU baseline only)");
}
