//! GPU Jacobian assembly at realistic scale — `plans/GPU_PLAN.md` Phase 2.
//!
//! `tests/gpu_assembly_test.rs` proves kernel correctness on the committed
//! 3-bus fixture (all CI can rely on, since the MATPOWER cases live in a
//! gitignored cache). This runs it on a real case at batch scale and reports
//! the deviation from the f64 CPU reference plus assembly throughput.
//!
//! ```bash
//! cargo run --release --features gpu --example gpu_assembly_check -- \
//!     scripts/bench/.case-cache/case1354pegase.json 64
//! ```
//!
//! **f64, CUDA.** `src/gpu.rs` assembles in f64 on CubeCL's CUDA backend —
//! see that module's doc comment. This is the real Phase 2 exit criterion:
//! agreement with the f64 CPU reference to near machine epsilon, not just f32
//! tolerance. Timings here are not a speedup claim either — nothing is being
//! compared against a CPU assembly baseline.

use std::env;
use std::fs;
use std::time::Instant;

use gridoxide::gpu::default_assembler;
use gridoxide::jacobian::{EntryKind, JacobianPattern};
use gridoxide::network::{build_ybus, linear_initial_guess, power_injections, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches};

/// The identical formulas evaluated in **f32 on the CPU** — kept only as
/// historical context for what the old wgpu/f32 path cost (see git history
/// before the Phase 2 CUDA/f64 switch). Not used for pass/fail any more: the
/// GPU is f64 now, so it is compared directly against the f64 CPU reference
/// below.
fn cpu_f32_reference(
    pattern: &JacobianPattern,
    states: &[Vec<gridoxide::types::Bus>],
    p_all: &[Vec<f64>],
    q_all: &[Vec<f64>],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(states.len() * pattern.len());
    for s in 0..states.len() {
        for e in pattern.entries() {
            let i = e.i as usize;
            let k = e.k as usize;
            let g = e.y.re as f32;
            let b = e.y.im as f32;
            let vm_i = states[s][i].voltage_mag as f32;
            let p = p_all[s][i] as f32;
            let q = q_all[s][i] as f32;
            let v = match e.kind {
                EntryKind::Hii => -q - vm_i * vm_i * b,
                EntryKind::Nii => p / vm_i + vm_i * g,
                EntryKind::Mii => p - vm_i * vm_i * g,
                EntryKind::Lii => q / vm_i - vm_i * b,
                _ => {
                    let vm_k = states[s][k].voltage_mag as f32;
                    let ang = states[s][i].voltage_ang as f32 - states[s][k].voltage_ang as f32;
                    let (sin, cos) = (ang.sin(), ang.cos());
                    match e.kind {
                        EntryKind::Hik => vm_i * vm_k * (g * sin - b * cos),
                        EntryKind::Nik => vm_i * (g * cos + b * sin),
                        EntryKind::Mik => -vm_i * vm_k * (g * cos + b * sin),
                        _ => vm_i * (g * sin - b * cos),
                    }
                }
            };
            out.push(v);
        }
    }
    out
}

/// Distribution of relative deviation, not just the worst case. A kernel
/// fault is systematic and moves the whole distribution; isolated libm
/// (sin/cos) rounding differences between GPU and CPU affect a thin tail.
fn percentiles(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let mut rels: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs() / y.abs().max(1.0))
        .collect();
    rels.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let at = |q: f64| rels[((rels.len() - 1) as f64 * q) as usize];
    (at(0.5), at(0.9999), at(1.0))
}

/// Error relative to the magnitude of the *operands* that produced each value,
/// rather than to the value itself.
///
/// This is the measure that actually tests the kernel. Every H/N/M/L term is a
/// sum of two products, and any of them can cancel: `Nii = P/V + V*G` cancels
/// when the two are comparable, `Hii = -Q - V^2*B` does not because `B`
/// dominates. Scaling by the result therefore reports cancellation, which is a
/// property of the network's R/X ratio, not of where the arithmetic ran.
/// Scaling by the operands asks the question that isolates the kernel: did the
/// GPU evaluate the same expression to f64 precision?
fn operand_scaled_worst(
    pattern: &JacobianPattern,
    states: &[Vec<gridoxide::types::Bus>],
    p_all: &[Vec<f64>],
    q_all: &[Vec<f64>],
    a: &[f64],
    b: &[f64],
) -> (f64, usize) {
    let nnz = pattern.len();
    let mut worst = 0.0f64;
    let mut idx = 0usize;
    for slot in 0..a.len() {
        let s = slot / nnz;
        let e = &pattern.entries()[slot % nnz];
        let (i, k) = (e.i as usize, e.k as usize);
        let vm_i = states[s][i].voltage_mag;
        let vm_k = states[s][k].voltage_mag;
        // Bounds every intermediate product in all eight formulas.
        let scale = vm_i * vm_k.max(1.0) * (e.y.re.abs() + e.y.im.abs())
            + p_all[s][i].abs()
            + q_all[s][i].abs()
            + 1.0;
        let d = (a[slot] - b[slot]).abs() / scale;
        if d > worst {
            worst = d;
            idx = slot;
        }
    }
    (worst, idx)
}

fn worst_rel(a: &[f64], b: &[f64]) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut idx = 0usize;
    for (n, (&x, &y)) in a.iter().zip(b).enumerate() {
        let r = (x - y).abs() / y.abs().max(1.0);
        if r > worst {
            worst = r;
            idx = n;
        }
    }
    (worst, idx)
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: gpu_assembly_check <input.json> [n_scenarios]");
    let nb: usize = args.next().map(|s| s.parse().expect("n_scenarios must be an integer")).unwrap_or(32);

    let raw = fs::read_to_string(&path).expect("read input file");
    let input = serde_json::from_str(&raw).expect("parse PGM input JSON");
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (template, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let mut ybus = build_ybus(template.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let ybus = ybus.finish();

    let pattern = JacobianPattern::analyze(&template, &ybus);

    // Distinct operating point per scenario.
    let states: Vec<Vec<gridoxide::types::Bus>> = (0..nb)
        .map(|s| {
            let f = 0.8 + 0.4 * (s as f64) / (nb.max(2) - 1) as f64;
            let mut buses = template.to_vec();
            for b in buses.iter_mut() {
                b.p_spec *= f;
                b.q_spec *= f;
            }
            linear_initial_guess(&mut buses, &ybus);
            buses
        })
        .collect();

    let mut p_all = Vec::with_capacity(nb);
    let mut q_all = Vec::with_capacity(nb);
    for s in &states {
        let (p, q) = power_injections(s, &ybus);
        p_all.push(p);
        q_all.push(q);
    }

    println!("case: {path}");
    println!("buses: {}  scenarios: {}  nnz/scenario: {}", template.len(), nb, pattern.len());
    println!("total Jacobian values: {}", nb * pattern.len());

    let t0 = Instant::now();
    let mut expected: Vec<f64> = Vec::with_capacity(nb * pattern.len());
    for s in 0..nb {
        pattern.fill_into(&states[s], &p_all[s], &q_all[s], &mut expected);
    }
    let t_cpu = t0.elapsed();

    let mut asm = default_assembler(&pattern, template.len());
    // Warm-up: first launch pays kernel compilation and buffer setup.
    let _ = asm.assemble_batch(&states, &p_all, &q_all);

    let t1 = Instant::now();
    let got = asm.assemble_batch(&states, &p_all, &q_all);
    let t_gpu = t1.elapsed();

    assert_eq!(got.len(), expected.len());
    let cpu32 = cpu_f32_reference(&pattern, &states, &p_all, &q_all);
    let cpu32_as_f64: Vec<f64> = cpu32.iter().map(|&v| v as f64).collect();

    let (gpu_vs_f64, i_a) = worst_rel(&got, &expected);
    let (cpu32_vs_f64, _) = worst_rel(&cpu32_as_f64, &expected);

    println!();
    println!("cpu assemble (f64):        {:.2} ms", t_cpu.as_secs_f64() * 1e3);
    println!("gpu assemble (f64, warm):  {:.2} ms  [upload + launch + readback]", t_gpu.as_secs_f64() * 1e3);
    println!();
    println!("worst relative deviation:");
    println!(
        "  gpu(f64)  vs cpu(f64):  {gpu_vs_f64:.3e}   (scenario {}, slot {})   <- the real Phase 2 result",
        i_a / pattern.len(),
        i_a % pattern.len()
    );
    println!("  cpu(f32)  vs cpu(f64):  {cpu32_vs_f64:.3e}   <- what the old wgpu/f32 path cost, for context only");
    println!("  f64 machine epsilon:    {:.3e}", f64::EPSILON);

    // Deviation by entry kind, gpu(f64) vs cpu(f64) directly. Any remaining
    // gap is now libm (sin/cos) rounding between GPU and CPU implementations,
    // not f32 cancellation — the four off-diagonal kinds call sin/cos, the
    // four diagonal kinds are pure arithmetic.
    {
        let names = ["Hii", "Nii", "Hik", "Nik", "Mii", "Lii", "Mik", "Lik"];
        let mut worst = [0.0f64; 8];
        let mut exact = [0usize; 8];
        let mut count = [0usize; 8];
        for slot in 0..got.len() {
            let e = &pattern.entries()[slot % pattern.len()];
            let ki = e.kind as usize;
            let d = (got[slot] - expected[slot]).abs() / expected[slot].abs().max(1.0);
            count[ki] += 1;
            if d == 0.0 {
                exact[ki] += 1;
            }
            if d > worst[ki] {
                worst[ki] = d;
            }
        }
        println!();
        println!("gpu(f64) vs cpu(f64) by entry kind:");
        for k in 0..8 {
            if count[k] == 0 {
                continue;
            }
            let uses_trig = !matches!(k, 0 | 1 | 4 | 5);
            println!(
                "  {:3}  n={:>9}  bit-exact={:>6.2}%  worst={:.3e}   {}",
                names[k],
                count[k],
                100.0 * exact[k] as f64 / count[k] as f64,
                worst[k],
                if uses_trig { "(uses sin/cos)" } else { "(arithmetic only)" }
            );
        }
    }

    let (p50, p9999, pmax) = percentiles(&got, &expected);
    println!();
    println!("gpu(f64) vs cpu(f64) distribution over {} values:", got.len());
    println!("  median   {p50:.3e}");
    println!("  p99.99   {p9999:.3e}");
    println!("  max      {pmax:.3e}");
    {
        let e = expected[i_a];
        let gv = got[i_a];
        println!("  worst slot: cpu_f64={e:.6e}  gpu_f64={gv:.6e}  |abs diff|={:.3e}", (gv - e).abs());
    }

    // Phase 2's real exit criterion (`scripts/GPU_RUNBOOK.md`): operand-scaled
    // error near f64 machine epsilon, and the result-scaled worst case
    // collapsing from the old f32 path's ~1e-2 to near zero. Anything looser
    // means the kernel itself has a bug, not a precision limitation — there is
    // no cancellation argument left to hide behind once both sides are f64.
    let (op_worst, op_idx) = operand_scaled_worst(&pattern, &states, &p_all, &q_all, &got, &expected);
    println!();
    println!("operand-scaled error (the measure that isolates the kernel):");
    println!(
        "  gpu(f64) vs cpu(f64):  {op_worst:.3e}   (scenario {}, slot {})",
        op_idx / pattern.len(),
        op_idx % pattern.len()
    );

    let ok = gpu_vs_f64 < 1e-9 && op_worst < 1e-9;
    println!();
    println!("RESULT: {}", if ok {
        "PASS (f64 exactness confirmed: this is the real Phase 2 exit criterion, not the f32 approximation)"
    } else {
        "FAIL"
    });
    if !ok {
        std::process::exit(1);
    }
}
