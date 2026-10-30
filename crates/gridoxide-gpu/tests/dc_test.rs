//! The device's DC N-1 (`GpuDcSolver`) against the CPU's
//! (`DcSensitivity::contingencies`, itself checked against re-solves in
//! gridoxide's `tests/dc_batch_test.rs`): every outage's angles and flows, and
//! which outages have a post-outage state, in one chunk and in several.
//! Needs the `cuda` feature and a GPU.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use gridoxide::linear::{DcContingencies, DcOptions, DcSensitivity, dc_branches, dc_power_flow};
use gridoxide_gpu::GpuDcSolver;

fn case(name: &str) -> gridoxide::matpower::MatpowerNetwork {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/benchmark-grids/matpower")
        .join(format!("{name}.m"));
    assert!(
        path.exists(),
        "{} not found — run `git submodule update --init tests/data/benchmark-grids`",
        path.display()
    );
    gridoxide::matpower::read(&path).unwrap()
}

fn cpu(net: &gridoxide::matpower::MatpowerNetwork, outages: &[usize]) -> DcContingencies {
    let opts = DcOptions::default();
    let n_br = net.lines.len() + net.transformers.len();
    let mut buses = net.buses.clone();
    let base = dc_power_flow(&mut buses, &net.lines, &net.transformers, opts);
    let angles: Vec<f64> = buses.iter().map(|b| b.voltage_ang).collect();
    let branches = dc_branches(&net.lines, &net.transformers, opts);
    DcSensitivity::new(&buses, &branches, n_br)
        .unwrap()
        .contingencies(&angles, &base.branch_p, outages)
        .unwrap()
}

/// Largest difference over the outages both solved, after checking that
/// the same ones are solved and the rest are NaN on the device.
fn compare(got: &DcContingencies, want: &DcContingencies) -> (f64, f64) {
    assert_eq!(
        got.solved, want.solved,
        "the same outages have a post-outage state"
    );
    let (mut da, mut df) = (0.0f64, 0.0f64);
    for k in 0..want.solved.len() {
        if !want.solved[k] {
            assert!(got.angles(k).iter().chain(got.flows(k)).all(|v| v.is_nan()));
            continue;
        }
        for (a, b) in got.angles(k).iter().zip(want.angles(k)) {
            da = da.max((a - b).abs());
        }
        for (a, b) in got.flows(k).iter().zip(want.flows(k)) {
            df = df.max((a - b).abs());
        }
    }
    (da, df)
}

#[test]
fn every_outage_of_case1354pegase_matches_the_cpu() {
    let net = case("case1354pegase");
    let n_br = net.lines.len() + net.transformers.len();
    let outages: Vec<usize> = (0..n_br).collect();
    let want = cpu(&net, &outages);
    assert!(want.solved.iter().any(|&s| !s), "some branches are radial");
    for capacity in [n_br, 300] {
        let mut gpu = GpuDcSolver::new(
            &net.buses,
            &net.lines,
            &net.transformers,
            DcOptions::default(),
            capacity,
        )
        .unwrap();
        let got = gpu.solve_contingencies(&net.buses, &outages).unwrap();
        let (da, df) = compare(&got, &want);
        assert!(
            da < 1e-11 && df < 1e-10,
            "capacity {capacity}: angles {da:e}, flows {df:e}"
        );
    }
}

#[test]
fn case9241pegase_in_chunks_matches_the_cpu() {
    let net = case("case9241pegase");
    let n_br = net.lines.len() + net.transformers.len();
    // Every seventh branch, so the CPU reference stays quick.
    let outages: Vec<usize> = (0..n_br).step_by(7).collect();
    let want = cpu(&net, &outages);
    let mut gpu = GpuDcSolver::new(
        &net.buses,
        &net.lines,
        &net.transformers,
        DcOptions::default(),
        1000,
    )
    .unwrap();
    let got = gpu.solve_contingencies(&net.buses, &outages).unwrap();
    let (da, df) = compare(&got, &want);
    assert!(da < 1e-10 && df < 1e-9, "angles {da:e}, flows {df:e}");
    // A second call on the same session gives the same answer.
    let again = gpu.solve_contingencies(&net.buses, &outages[..50]).unwrap();
    assert_eq!(again.solved, got.solved[..50]);
}

#[test]
fn a_branch_out_of_range_is_refused() {
    let net = case("case1354pegase");
    let n_br = net.lines.len() + net.transformers.len();
    let mut gpu = GpuDcSolver::new(
        &net.buses,
        &net.lines,
        &net.transformers,
        DcOptions::default(),
        8,
    )
    .unwrap();
    assert!(gpu.solve_contingencies(&net.buses, &[0, n_br]).is_err());
}
