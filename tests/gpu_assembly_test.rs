//! gridoxide's CUDA kernels vs. their CPU references.
//!
//! Runs only with `--features gpu` (the last two also need `cudss`), needs an
//! NVIDIA GPU at runtime, and needs the CUDA toolkit at build time — `nvcc`
//! compiles `cuda/gridoxide_kernels.cu`. See `src/gpu.rs`.
//!
//! **These are f64 checks.** `plans/GPU_PLAN.md` Phase 2's stated exit
//! criterion is f64 exactness against `JacobianPattern::fill`'s CPU reference,
//! so the assembly tolerances are near machine epsilon rather than the ~1e-5
//! an f32 kernel would need. `go_power_injections` is looser (1e-12 relative)
//! for a stated reason: it sums each row in a different order than `faer`'s
//! column-major SpMV, which is a rounding difference, not an error.
//!
//! What each test isolates:
//!
//! - assembly vs. `JacobianPattern::fill_into`, and the scenario stride;
//! - `go_power_injections` vs. `network::power_injections`;
//! - the whole device-resident assembly path — CSR scatter, identity masking,
//!   per-block offsets — against the CPU block-diagonal reference.

#![cfg(feature = "gpu")]

use std::fs;
use std::path::PathBuf;

use gridoxide::gpu::GpuAssembler;
use gridoxide::jacobian::JacobianPattern;
use gridoxide::json::NetworkData;
use gridoxide::network::{build_ybus, linear_initial_guess, power_injections, YBusSparse};
use gridoxide::types::{Bus, BusType};

fn load_network() -> (Vec<Bus>, YBusSparse) {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let raw = fs::read_to_string(path).expect("read network.json");
    let network: NetworkData = serde_json::from_str(&raw).expect("parse network.json");
    let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();
    (network.buses, ybus)
}

/// Distinct operating points per scenario, so a kernel that ignored the
/// scenario stride and assembled the same block B times would fail.
fn scenario_states(template: &[Bus], ybus: &YBusSparse, nb: usize) -> Vec<Vec<Bus>> {
    (0..nb)
        .map(|s| {
            let mut buses = template.to_vec();
            let f = 0.6 + 0.25 * s as f64;
            buses[2].p_spec = template[2].p_spec * f;
            buses[2].q_spec = template[2].q_spec * f;
            linear_initial_guess(&mut buses, ybus);
            // Perturb angles too, so sin/cos are not all evaluated near zero.
            for (j, b) in buses.iter_mut().enumerate() {
                if b.bus_type != BusType::Slack {
                    b.voltage_ang += 0.013 * (s + 1) as f64 * (j + 1) as f64;
                }
            }
            buses
        })
        .collect()
}

#[test]
fn gpu_assembly_matches_cpu_reference() {
    let (template, ybus) = load_network();
    let pattern = JacobianPattern::analyze(&template, &ybus);
    let nb = 6;
    let states = scenario_states(&template, &ybus, nb);

    let mut p_all = Vec::new();
    let mut q_all = Vec::new();
    for s in &states {
        let (p, q) = power_injections(s, &ybus);
        p_all.push(p);
        q_all.push(q);
    }

    // CPU reference, in f64, laid out scenario-major exactly as the kernel
    // writes it.
    let mut expected: Vec<f64> = Vec::new();
    for s in 0..nb {
        pattern.fill_into(&states[s], &p_all[s], &q_all[s], &mut expected);
    }

    let mut asm = GpuAssembler::new(&pattern, template.len()).expect("CUDA device available");
    let got = asm.assemble_batch(&states, &p_all, &q_all).expect("assembly launch");

    assert_eq!(got.len(), expected.len(), "one value per (scenario, entry)");
    assert_eq!(got.len(), nb * pattern.len());

    let mut worst_rel = 0.0f64;
    for (idx, (&g, &e)) in got.iter().zip(&expected).enumerate() {
        let denom = e.abs().max(1.0);
        let rel = (g - e).abs() / denom;
        worst_rel = worst_rel.max(rel);
        assert!(
            rel < 1e-9,
            "entry {idx} (scenario {}, slot {}): gpu {g} vs cpu {e} (rel {rel:.3e})",
            idx / pattern.len(),
            idx % pattern.len()
        );
    }
    println!("worst relative deviation: {worst_rel:.3e}");
}

/// Each scenario's block must be assembled from *its own* state. Feeding two
/// scenarios identical states and one different state pins down the stride:
/// blocks 0 and 1 must match each other exactly and block 2 must not.
#[test]
fn scenario_stride_is_respected() {
    let (template, ybus) = load_network();
    let pattern = JacobianPattern::analyze(&template, &ybus);

    let mut a = template.to_vec();
    linear_initial_guess(&mut a, &ybus);
    let mut c = template.to_vec();
    c[2].p_spec *= 3.0;
    linear_initial_guess(&mut c, &ybus);

    let states = vec![a.clone(), a.clone(), c];
    let mut p_all = Vec::new();
    let mut q_all = Vec::new();
    for s in &states {
        let (p, q) = power_injections(s, &ybus);
        p_all.push(p);
        q_all.push(q);
    }

    let mut asm = GpuAssembler::new(&pattern, template.len()).expect("CUDA device available");
    let got = asm.assemble_batch(&states, &p_all, &q_all).expect("assembly launch");
    let nnz = pattern.len();

    assert_eq!(
        got[0..nnz],
        got[nnz..2 * nnz],
        "identical scenario states must produce identical blocks"
    );
    assert_ne!(
        got[0..nnz],
        got[2 * nnz..3 * nnz],
        "a different scenario state must produce a different block — the kernel is ignoring the stride"
    );
}

/// `go_power_injections` vs. `network::power_injections`.
///
/// This kernel is new with the CUDA rewrite — the CubeCL path computed
/// injections on the host, one serial `power_injections` call per scenario per
/// Newton iteration. Moving it on-device is what lets the batched loop keep
/// `vm`/`va` resident, so it needs its own check against the CPU original.
///
/// Agreement is to rounding, not bit-for-bit: `faer`'s column-major SpMV sums
/// each output in a different order than the kernel's row-major one. The
/// tolerance is relative and tight enough that any indexing or stride error
/// fails it by orders of magnitude.
#[test]
fn gpu_power_injections_match_cpu_reference() {
    use gridoxide::device_layout::csr_scatter_map;
    use gridoxide::gpu::GpuBatch;

    let (template, ybus) = load_network();
    let pattern = JacobianPattern::analyze(&template, &ybus);
    let nb = 5;
    let states = scenario_states(&template, &ybus, nb);

    let pairs: Vec<(usize, usize)> =
        pattern.rows().iter().zip(pattern.cols()).map(|(&r, &c)| (r as usize, c as usize)).collect();
    let scatter = csr_scatter_map(pattern.n_unknowns, &pairs);

    let gpu = GpuBatch::new(&pattern, &ybus, &template, &states, &scatter).expect("CUDA device available");
    gpu.power_injections().expect("power injections launch");
    let (p_gpu, q_gpu) = gpu.read_injections().expect("readback");

    let n = template.len();
    let mut worst = 0.0f64;
    for (s, state) in states.iter().enumerate() {
        let (p_ref, q_ref) = power_injections(state, &ybus);
        for i in 0..n {
            for (got, want) in [(p_gpu[s * n + i], p_ref[i]), (q_gpu[s * n + i], q_ref[i])] {
                worst = worst.max((got - want).abs() / want.abs().max(1.0));
            }
        }
    }
    assert!(worst < 1e-12, "GPU power injections disagree with the CPU reference (worst rel {worst:.3e})");
}

/// The device-resident assembly path end to end: CSR-scattered output, mixed
/// active/masked scenarios, read back through the same buffer cuDSS binds to.
///
/// This is the shape `bde`'s loop produces every iteration, and it is the
/// check that rules out the scatter map, the identity-masking kernel path and
/// the per-block CSR offset arithmetic together. It descends from an
/// investigation into an iteration-count discrepancy on the CubeCL path (see
/// `bde::solve_batch_block_diagonal_device_resident`'s doc comment); the
/// conclusion there was that gridoxide feeds cuDSS the right numbers, and this
/// keeps that conclusion pinned down after the rewrite.
#[test]
fn device_resident_masked_scattered_output_matches_cpu() {
    use gridoxide::bde::BlockDiagonal;
    use gridoxide::device_layout::{build_csr_structure, csr_scatter_map, pack_values_slice};
    use gridoxide::gpu::GpuAssembler;

    let (template, ybus) = load_network();
    let nb = 4usize;
    let bd = BlockDiagonal::analyze(&template, &ybus, nb);
    let base = JacobianPattern::analyze(&template, &ybus);

    let states = scenario_states(&template, &ybus, nb);
    let mut p_all = Vec::new();
    let mut q_all = Vec::new();
    for s in &states {
        let (p, q) = power_injections(s, &ybus);
        p_all.push(p);
        q_all.push(q);
    }
    let active = vec![true, false, true, false];

    // CPU reference: BlockDiagonal::fill (entries order, stacked), reordered
    // into CSR order via the stacked pairs/groups.
    let mut cpu_values = Vec::new();
    bd.fill(&states, &p_all, &q_all, &active, &mut cpu_values);
    let full_pairs: Vec<(usize, usize)> = bd.to_triplets(&cpu_values).iter().map(|&(r, c, _)| (r, c)).collect();
    let (_, _, full_groups) = build_csr_structure(bd.n_unknowns(), &full_pairs);
    let cpu_csr = pack_values_slice(&cpu_values, &full_groups);

    // Device path: one single-block scatter map, offset by the kernel.
    let block_pairs: Vec<(usize, usize)> =
        base.rows().iter().zip(base.cols()).map(|(&r, &c)| (r as usize, c as usize)).collect();
    let scatter = csr_scatter_map(base.n_unknowns, &block_pairs);
    let mut asm = GpuAssembler::new(&base, template.len()).expect("CUDA device available");
    asm.set_scatter(&scatter).expect("scatter upload");
    asm.assemble_batch_masked(&states, &p_all, &q_all, &active).expect("assembly launch");
    let gpu_csr = asm.read_values().expect("readback");

    assert_eq!(cpu_csr.len(), gpu_csr.len());
    let worst = gpu_csr.iter().zip(&cpu_csr).map(|(&g, &c)| (g - c).abs() / c.abs().max(1.0)).fold(0.0f64, f64::max);
    assert!(worst < 1e-9, "device-resident masked CSR values disagree with the CPU reference (worst rel {worst:.3e})");
}
