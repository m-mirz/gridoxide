//! GPU Jacobian assembly vs. the CPU reference.
//!
//! Runs only with `--features gpu`, and needs a working GPU driver at runtime
//! (CUDA — see `src/gpu.rs`'s `DefaultRuntime`).
//!
//! **This is an f64 check.** `plans/GPU_PLAN.md` Phase 2's stated exit
//! criterion is f64 exactness against `JacobianPattern::fill`'s CPU reference,
//! reachable now that the kernel runs on CubeCL's CUDA backend instead of
//! wgpu/WGSL (which has no f64 — see git history for that earlier version).
//! The tolerance below is accordingly tight: near f64 machine epsilon, not the
//! ~1e-5 an f32 kernel would need.

#![cfg(feature = "gpu")]

use std::fs;
use std::path::PathBuf;

use gridoxide::gpu::default_assembler;
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

    let asm = default_assembler(&pattern, template.len());
    let got = asm.assemble_batch(&states, &p_all, &q_all);

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

    let asm = default_assembler(&pattern, template.len());
    let got = asm.assemble_batch(&states, &p_all, &q_all);
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
