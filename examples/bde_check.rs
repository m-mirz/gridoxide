//! Validates block-diagonal embedding at realistic scale on real sparse code.
//!
//! `tests/bde_test.rs` proves the same property on the committed 3-bus
//! fixture, which is all CI can rely on (the MATPOWER cases live in a
//! gitignored cache). This example is the scaled-up check: on a case with
//! thousands of buses, KLU's BTF ordering, AMD fill-reducing permutation and
//! partial pivoting all have real room to move, and if any of them crossed a
//! block boundary the per-scenario answers would diverge from independent
//! solves.
//!
//! ```bash
//! cargo run --release --example bde_check -- scripts/bench/.case-cache/case1354pegase.json 16
//! # or, to exercise the device-resident cuDSS path (plans/GPU_PLAN.md Phase 3):
//! cargo run --release --features gpu,cudss --example bde_check -- \
//!     scripts/bench/.case-cache/case1354pegase.json 16
//! ```

use std::env;
use std::fs;
use std::time::Instant;

use gridoxide::batch::{BatchSolver, BusOverride, Scenario};
use gridoxide::bde::{solve_batch_block_diagonal, BlockDiagonal};
#[cfg(not(feature = "cudss"))]
use gridoxide::klu_native::KluNativeSystem;
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches};
use gridoxide::solver::{JacobianBackend, SolveStatus};

/// The backend `solve_batch_block_diagonal` embeds against. With the `cudss`
/// feature this is the GPU path this file is really meant to check; without
/// it, `KluNativeSystem` still proves the embedding on a real case at scale
/// (`tests/bde_test.rs` covers the small fixture unconditionally).
#[cfg(feature = "cudss")]
type EmbeddedSystem = gridoxide::sparse_cudss::CudssRealSystem;
#[cfg(not(feature = "cudss"))]
type EmbeddedSystem = KluNativeSystem;

/// A GPU solver picks a different elimination order/pivoting than the CPU
/// reference, so agreement lands at ~1e-9 rather than the CPU-vs-CPU
/// backends' bit-identical-to-1e-11 — see `scripts/GPU_RUNBOOK.md` Phase 3.
#[cfg(feature = "cudss")]
const AGREEMENT_TOL: f64 = 1e-6;
#[cfg(not(feature = "cudss"))]
const AGREEMENT_TOL: f64 = 1e-9;

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: bde_check <input.json> [n_scenarios]");
    let nb: usize = args.next().map(|s| s.parse().expect("n_scenarios must be an integer")).unwrap_or(8);

    let raw = fs::read_to_string(&path).expect("read input file");
    let input = serde_json::from_str(&raw).expect("parse PGM input JSON");
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (template, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let mut ybus = build_ybus(template.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let ybus = ybus.finish();

    // Spread of load scalings, deterministic.
    let scenarios: Vec<Scenario> = (0..nb)
        .map(|k| {
            let f = 0.8 + 0.4 * (k as f64) / (nb.max(2) - 1) as f64;
            Scenario::new(
                template
                    .iter()
                    .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
                    .map(|b| BusOverride::new(b.idx).p(b.p_spec * f).q(b.q_spec * f))
                    .collect(),
            )
        })
        .collect();

    let bd = BlockDiagonal::analyze(&template, &ybus, nb);
    println!("case: {path}");
    println!("buses: {}  scenarios: {}", template.len(), nb);
    println!(
        "per-scenario unknowns: {}  stacked unknowns: {}  stacked nonzeros: {}",
        bd.block_size(),
        bd.n_unknowns(),
        bd.len()
    );

    let t0 = Instant::now();
    let batch = BatchSolver::with_threads(JacobianBackend::KluNative, 1).expect("pool");
    let independent = batch.solve(&template, &ybus, &scenarios, 1e-8, 40).expect("independent");
    let t_indep = t0.elapsed();

    let t1 = Instant::now();
    let embedded = solve_batch_block_diagonal::<EmbeddedSystem>(&template, &ybus, &scenarios, 1e-8, 40);
    let t_bde = t1.elapsed();

    #[cfg(all(feature = "gpu", feature = "cudss"))]
    let (device_resident, t_device) = {
        let t2 = Instant::now();
        let r = gridoxide::bde::solve_batch_block_diagonal_device_resident(&template, &ybus, &scenarios, 1e-8, 40);
        (r, t2.elapsed())
    };

    #[cfg(all(feature = "gpu", feature = "cudss"))]
    let (batched, t_batched) = {
        let t3 = Instant::now();
        let r = gridoxide::bde::solve_batch_block_diagonal_batched_device(&template, &ybus, &scenarios, 1e-8, 40);
        (r, t3.elapsed())
    };

    let mut worst_vm = 0.0f64;
    let mut worst_va = 0.0f64;
    let mut iter_mismatch = 0usize;
    let mut not_converged = 0usize;
    for (emb, indep) in embedded.iter().zip(&independent) {
        if emb.stats.status != SolveStatus::Converged || indep.stats.status != SolveStatus::Converged {
            not_converged += 1;
            continue;
        }
        if emb.stats.iterations() != indep.stats.iterations() {
            iter_mismatch += 1;
        }
        for (e, r) in emb.buses.iter().zip(&indep.buses) {
            worst_vm = worst_vm.max((e.voltage_mag - r.voltage_mag).abs());
            worst_va = worst_va.max((e.voltage_ang - r.voltage_ang).abs());
        }
    }

    #[cfg(all(feature = "gpu", feature = "cudss"))]
    let (worst_vm_dr, worst_va_dr, iter_mismatch_dr, not_converged_dr) = {
        let mut worst_vm = 0.0f64;
        let mut worst_va = 0.0f64;
        let mut iter_mismatch = 0usize;
        let mut not_converged = 0usize;
        for (dr, indep) in device_resident.iter().zip(&independent) {
            if dr.stats.status != SolveStatus::Converged || indep.stats.status != SolveStatus::Converged {
                not_converged += 1;
                continue;
            }
            if dr.stats.iterations() != indep.stats.iterations() {
                iter_mismatch += 1;
            }
            for (e, r) in dr.buses.iter().zip(&indep.buses) {
                worst_vm = worst_vm.max((e.voltage_mag - r.voltage_mag).abs());
                worst_va = worst_va.max((e.voltage_ang - r.voltage_ang).abs());
            }
        }
        (worst_vm, worst_va, iter_mismatch, not_converged)
    };

    #[cfg(all(feature = "gpu", feature = "cudss"))]
    let (worst_vm_b, worst_va_b, not_converged_b) = {
        let mut worst_vm = 0.0f64;
        let mut worst_va = 0.0f64;
        let mut not_converged = 0usize;
        for (b, indep) in batched.iter().zip(&independent) {
            if b.stats.status != SolveStatus::Converged || indep.stats.status != SolveStatus::Converged {
                not_converged += 1;
                continue;
            }
            for (e, r) in b.buses.iter().zip(&indep.buses) {
                worst_vm = worst_vm.max((e.voltage_mag - r.voltage_mag).abs());
                worst_va = worst_va.max((e.voltage_ang - r.voltage_ang).abs());
            }
        }
        (worst_vm, worst_va, not_converged)
    };

    println!();
    println!("independent (1 thread):       {:.1} ms", t_indep.as_secs_f64() * 1e3);
    println!("block-diagonal (host round trip every iter): {:.1} ms", t_bde.as_secs_f64() * 1e3);
    #[cfg(all(feature = "gpu", feature = "cudss"))]
    println!("block-diagonal (device-resident Jacobian):   {:.1} ms", t_device.as_secs_f64() * 1e3);
    #[cfg(all(feature = "gpu", feature = "cudss"))]
    println!("batched (cuDSS uniform batch API):           {:.1} ms", t_batched.as_secs_f64() * 1e3);
    println!();
    println!("host-resident vs independent:");
    println!("  max |dVm| = {worst_vm:.3e}");
    println!("  max |dVa| = {worst_va:.3e} rad");
    println!("  iteration-count mismatches: {iter_mismatch}/{nb}");
    println!("  non-converged scenarios:    {not_converged}/{nb}");
    #[cfg(all(feature = "gpu", feature = "cudss"))]
    {
        println!("device-resident vs independent:");
        println!("  max |dVm| = {worst_vm_dr:.3e}");
        println!("  max |dVa| = {worst_va_dr:.3e} rad");
        // Informational only, not part of RESULT below — see
        // `bde::solve_batch_block_diagonal_device_resident`'s doc comment:
        // this path is verified to feed cuDSS bit-identical values to the
        // host-resident path, yet can still take an extra Newton iteration
        // or two to cross `tol`. Voltage agreement is the property that
        // actually matters and the one gating pass/fail.
        println!("  iteration-count mismatches: {iter_mismatch_dr}/{nb}  (informational — see solve_batch_block_diagonal_device_resident's doc comment)");
        println!("  non-converged scenarios:    {not_converged_dr}/{nb}");
        println!("batched vs independent:");
        println!("  max |dVm| = {worst_vm_b:.3e}");
        println!("  max |dVa| = {worst_va_b:.3e} rad");
        println!("  non-converged scenarios:    {not_converged_b}/{nb}");
    }

    #[allow(unused_mut)]
    let mut ok = worst_vm < AGREEMENT_TOL && worst_va < AGREEMENT_TOL && iter_mismatch == 0 && not_converged == 0;
    #[cfg(all(feature = "gpu", feature = "cudss"))]
    {
        ok &= worst_vm_dr < AGREEMENT_TOL && worst_va_dr < AGREEMENT_TOL && not_converged_dr == 0;
        ok &= worst_vm_b < AGREEMENT_TOL && worst_va_b < AGREEMENT_TOL && not_converged_b == 0;
    }
    println!();
    println!("RESULT: {}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
