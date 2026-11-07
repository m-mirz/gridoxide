//! The device's DC N-1 (`GpuDcSolver`) against the CPU's
//! (`DcSensitivity::contingencies`, itself checked against re-solves in
//! gridoxide's `tests/dc_batch_test.rs`): every outage's angles and flows, and
//! which outages have a post-outage state, in one chunk and in several. And
//! its DC sweeps against the CPU's `DcBatchSolver`. Needs the `cuda` feature
//! and a GPU.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use gridoxide::linear::{
    DcBatchSolver, DcContingencies, DcIslandStatus, DcOptions, DcSensitivity, DcSolver,
    dc_branches, dc_power_flow,
};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide::types::BusType;
use gridoxide_gpu::GpuDcSolver;
use gridoxide_gpu::newton::BatchError;

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

/// Scenarios that scale every load by its own factor, plus the corner
/// cases: no override, a bus overridden twice (the last wins), an override
/// at the reference bus (which only its pickup sees), and one equal to the
/// template's value.
fn sweep_scenarios(net: &gridoxide::matpower::MatpowerNetwork, count: usize) -> Vec<Scenario> {
    let loads: Vec<usize> = (0..net.buses.len())
        .filter(|&i| net.buses[i].p_spec < 0.0)
        .collect();
    let slack = net
        .buses
        .iter()
        .position(|b| b.bus_type == BusType::Slack)
        .unwrap();
    let mut scenarios: Vec<Scenario> = (0..count)
        .map(|k| {
            let f = 0.7 + 0.6 * k as f64 / count as f64;
            Scenario::new(
                loads
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| (j + k) % 3 != 0)
                    .map(|(_, &i)| BusOverride::new(i).p(net.buses[i].p_spec * f))
                    .collect(),
            )
        })
        .collect();
    let (a, b) = (loads[0], loads[1]);
    scenarios[0] = Scenario::new(Vec::new());
    scenarios[1] = Scenario::new(vec![
        BusOverride::new(a).p(-1.0),
        BusOverride::new(b).p(-0.5),
        BusOverride::new(a).p(-2.0),
    ]);
    scenarios[2] = Scenario::new(vec![BusOverride::new(slack).p(3.0)]);
    scenarios[3] = Scenario::new(vec![BusOverride::new(a).p(net.buses[a].p_spec)]);
    scenarios
}

#[test]
fn a_sweep_of_case1354pegase_matches_the_cpu_batch() {
    let net = case("case1354pegase");
    let opts = DcOptions::default();
    let scenarios = sweep_scenarios(&net, 500);
    let want = DcBatchSolver::new()
        .solve(&net.buses, &net.lines, &net.transformers, opts, &scenarios)
        .unwrap();
    for capacity in [scenarios.len(), 64, 1] {
        let mut gpu =
            GpuDcSolver::new(&net.buses, &net.lines, &net.transformers, opts, capacity).unwrap();
        let got = gpu.solve_sweep(&net.buses, &scenarios).unwrap();
        assert_eq!(got.len(), scenarios.len());
        let (mut da, mut df, mut dp) = (0.0f64, 0.0f64, 0.0f64);
        for (k, w) in want.iter().enumerate() {
            for (a, b) in got.angles(k).iter().zip(&w.voltage_ang) {
                da = da.max((a - b).abs());
            }
            for (a, b) in got.flows(k).iter().zip(&w.branch_p) {
                df = df.max((a - b).abs());
            }
            assert_eq!(got.slack_pickup(k).len(), w.slack_pickup.len());
            for (a, b) in got.slack_pickup(k).iter().zip(&w.slack_pickup) {
                dp = dp.max((a - b).abs());
            }
        }
        assert!(
            da < 1e-11 && df < 1e-10 && dp < 1e-10,
            "capacity {capacity}: angles {da:e}, flows {df:e}, pickups {dp:e}"
        );
    }
}

#[test]
fn a_sweep_and_n1_share_a_session() {
    let net = case("case1354pegase");
    let opts = DcOptions::default();
    let scenarios = sweep_scenarios(&net, 40);
    let mut gpu = GpuDcSolver::new(&net.buses, &net.lines, &net.transformers, opts, 16).unwrap();
    let first = gpu.solve_sweep(&net.buses, &scenarios).unwrap();
    let outages: Vec<usize> = (0..200).collect();
    let n1 = gpu.solve_contingencies(&net.buses, &outages).unwrap();
    let (da, df) = compare(&n1, &cpu(&net, &outages));
    assert!(da < 1e-11 && df < 1e-10, "angles {da:e}, flows {df:e}");
    // The same sweep again, to rounding: cuDSS runs without its deterministic
    // mode, so a repeat is not bit-identical (6e-16 in angles, 1e-12 in flows
    // on an A100, two sweeps in a row the same), but N-1 leaves nothing
    // behind.
    let again = gpu.solve_sweep(&net.buses, &scenarios).unwrap();
    let gap = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    };
    assert_eq!(again.angles.len(), first.angles.len());
    assert_eq!(again.flows.len(), first.flows.len());
    let (da, df) = (
        gap(&again.angles, &first.angles),
        gap(&again.flows, &first.flows),
    );
    assert!(
        da < 1e-11 && df < 1e-10,
        "repeat: angles {da:e}, flows {df:e}"
    );
}

#[test]
fn a_sweep_refuses_outages_and_unknown_buses() {
    let net = case("case1354pegase");
    let mut gpu = GpuDcSolver::new(
        &net.buses,
        &net.lines,
        &net.transformers,
        DcOptions::default(),
        8,
    )
    .unwrap();
    let outage = Scenario {
        bus_overrides: Vec::new(),
        branch_outages: vec![0],
    };
    let far = Scenario::new(vec![BusOverride::new(net.buses.len()).p(1.0)]);
    assert!(matches!(
        gpu.solve_sweep(&net.buses, &[Scenario::new(Vec::new()), outage]),
        Err(BatchError::OutagesUnsupported { scenario: 1 })
    ));
    assert!(matches!(
        gpu.solve_sweep(&net.buses, &[far]),
        Err(BatchError::BusOutOfRange { scenario: 0, .. })
    ));
    assert!(gpu.solve_sweep(&net.buses, &[]).unwrap().is_empty());
}

/// A single DC power flow (`GpuDcSolver::solve_single`) against the CPU's
/// `DcSolver::solve`: angles, flows, pickups and statuses, on a case with
/// phase shifters (case9241pegase's transformers) and on case1354pegase, and
/// twice on one session.
#[test]
fn a_single_dc_power_flow_matches_the_cpu() {
    for name in ["case1354pegase", "case9241pegase"] {
        let net = case(name);
        let opts = DcOptions::default();
        let cpu = DcSolver::new(&net.buses, &net.lines, &net.transformers, opts);
        let mut want_buses = net.buses.clone();
        let want = cpu.solve(&mut want_buses);
        let mut gpu = GpuDcSolver::new(&net.buses, &net.lines, &net.transformers, opts, 4).unwrap();
        for round in 0..2 {
            let mut buses = net.buses.clone();
            let got = gpu.solve_single(&mut buses).unwrap();
            let da = buses
                .iter()
                .zip(&want_buses)
                .map(|(g, w)| (g.voltage_ang - w.voltage_ang).abs())
                .fold(0.0, f64::max);
            let df = got
                .branch_p
                .iter()
                .zip(&want.branch_p)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0, f64::max);
            assert_eq!(got.islands.len(), want.islands.len());
            for (g, w) in got.islands.iter().zip(&want.islands) {
                assert_eq!(g.status, w.status);
                assert!((g.slack_pickup - w.slack_pickup).abs() < 1e-9);
            }
            assert!(
                got.islands
                    .iter()
                    .all(|i| i.status != DcIslandStatus::Singular)
            );
            assert!(
                got.max_residual < 1e-9,
                "{name}: residual {:e}",
                got.max_residual
            );
            assert!(
                da < 1e-11 && df < 1e-9,
                "{name}, round {round}: angles {da:e}, flows {df:e}"
            );
        }
    }
}
