//! Phase 2: the device's Newton step against the CPU, kernel by kernel —
//! injections, mismatch and its norm, Jacobian values, update — on one
//! Newton step of every scenario. Needs an NVIDIA GPU; built only with the
//! `cuda` feature.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use gridoxide::jacobian::JacobianPattern;
use gridoxide::network::{
    YBusSparse, branch_stamp, build_ybus, build_ybus_with_outages, effective_injection,
    linear_initial_guess, power_injections, stamp_shunts,
};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide::types::{Bus, BusType, ZipKind, ZipTerm};
use gridoxide_gpu::{
    BatchOptions, BatchReport, CpuUniformBatch, DeviceBatch, Outcome, StepPolicy, UniformBatch,
    csr_jacobian, solve_batch, solve_contingency_batch,
};
use num_complex::Complex;

fn load(rel: &str) -> (Vec<Bus>, YBusSparse) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/benchmark-grids")
        .join(rel);
    assert!(
        path.exists(),
        "{} not found — run `git submodule update --init tests/data/benchmark-grids`",
        path.display()
    );
    let net = gridoxide::matpower::read(&path).unwrap();
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let mut base = net.buses.clone();
    linear_initial_guess(&mut base, &ybus);
    let (_, stats) = PersistentSolver::new(JacobianBackend::KluNative)
        .solve_with_stats(&mut base, &ybus, 1e-10, 30);
    assert_eq!(stats.status, SolveStatus::Converged);
    (base, ybus)
}

/// Scenario k scales every injection by 0.6 + 0.4·k/B and, on every fifth
/// bus that has one, its PV setpoint by 1 + 0.01·k/B.
fn ramp(base: &[Bus], batch: usize) -> Vec<Scenario> {
    (0..batch)
        .map(|k| {
            let f = 0.6 + 0.4 * k as f64 / batch as f64;
            Scenario::new(
                base.iter()
                    .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0 || b.bus_type == BusType::PV)
                    .map(|b| {
                        let ov = BusOverride::new(b.idx).p(b.p_spec * f).q(b.q_spec * f);
                        if b.bus_type == BusType::PV && b.idx % 5 == 0 {
                            ov.vm(b.voltage_mag * (1.0 + 0.01 * k as f64 / batch as f64))
                        } else {
                            ov
                        }
                    })
                    .collect(),
            )
        })
        .collect()
}

/// On every third loaded PQ bus, a fifth of its load moved into each ZIP
/// kind, so the mismatch kernel's ZIP terms are exercised at the size of the
/// network's own loads.
fn with_zip(mut base: Vec<Bus>) -> Vec<Bus> {
    for b in base.iter_mut().filter(|b| b.bus_type == BusType::PQ) {
        let load = Complex::new(b.p_spec, b.q_spec);
        if b.idx % 3 == 0 && load.norm() > 0.0 {
            b.zip_terms = [
                ZipKind::ConstImpedance,
                ZipKind::ConstCurrent,
                ZipKind::ConstPower,
            ]
            .map(|kind| ZipTerm {
                s_const: load * 0.2,
                kind,
            })
            .to_vec();
            b.p_spec *= 0.4;
            b.q_spec *= 0.4;
        }
    }
    base
}

/// The CPU's view of one scenario: buses with its overrides, as
/// `solve_batch` starts them.
fn cpu_state(start: &[Bus], sc: &Scenario) -> Vec<Bus> {
    let mut buses = start.to_vec();
    for ov in &sc.bus_overrides {
        let b = &mut buses[ov.bus];
        b.p_spec = ov.p_spec.unwrap_or(b.p_spec);
        b.q_spec = ov.q_spec.unwrap_or(b.q_spec);
        b.voltage_mag = ov.voltage_mag.unwrap_or(b.voltage_mag);
    }
    buses
}

/// The CPU's injections, mismatch and norm of one scenario, as
/// `newton::run` computes them.
fn cpu_mismatch(buses: &[Bus], ybus: &YBusSparse) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let (p, q) = power_injections(buses, ybus);
    let ns = buses.iter().filter(|b| b.bus_type != BusType::Slack);
    let pq = buses.iter().filter(|b| b.bus_type == BusType::PQ);
    let f: Vec<f64> = ns
        .map(|b| effective_injection(b).0 - p[b.idx])
        .chain(pq.map(|b| effective_injection(b).1 - q[b.idx]))
        .collect();
    let norm = f.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    (p, q, f, norm)
}

/// Largest |a - b| relative to max(1, |b|), over two slices.
fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

/// The device's injections, mismatch and norms against the CPU's, every
/// scenario on its own Y-bus; returns the CPU's mismatch, `batch × n`.
fn compare(dev: &DeviceBatch, states: &[Vec<Bus>], ybuses: &[YBusSparse], what: &str) -> Vec<f64> {
    let n_bus = states[0].len();
    let n = dev.n();
    let (p_dev, q_dev) = dev.injections().unwrap();
    let rhs_dev = dev.rhs().unwrap();
    let norm_dev = dev.norms().unwrap();
    let mut rhs_cpu = Vec::new();
    let (mut ep, mut ef, mut en) = (0.0f64, 0.0f64, 0.0f64);
    for (s, (buses, ybus)) in states.iter().zip(ybuses).enumerate() {
        let (p, q, f, norm) = cpu_mismatch(buses, ybus);
        ep = ep
            .max(worst(&p_dev[s * n_bus..(s + 1) * n_bus], &p))
            .max(worst(&q_dev[s * n_bus..(s + 1) * n_bus], &q));
        ef = ef.max(worst(&rhs_dev[s * n..(s + 1) * n], &f));
        en = en.max((norm_dev[s] - norm).abs() / norm.max(1.0));
        rhs_cpu.extend(f);
    }
    eprintln!("{what}: injections {ep:.1e}, mismatch {ef:.1e}, norm {en:.1e}");
    assert!(ep < 1e-11 && ef < 1e-11 && en < 1e-11, "{what}");
    rhs_cpu
}

/// One Newton step of every scenario on the device, each kernel against the
/// CPU, every scenario with its own Y-bus and Jacobian pattern (the same
/// for a sweep, its outage's for N-1). The update is the CPU's own solve of
/// the CPU's Jacobian, applied to both, so after it the two states differ
/// only by the device's rounding. Returns the device's norms after the step.
fn check_one_step(
    dev: &mut DeviceBatch,
    states: &mut [Vec<Bus>],
    ybuses: &[YBusSparse],
    patterns: &[JacobianPattern],
    what: &str,
) -> Vec<f64> {
    let batch = states.len();
    let n = dev.n();
    dev.mismatch().unwrap();
    let rhs = compare(dev, states, ybuses, what);
    dev.fill_jacobian().unwrap();
    let mut values = Vec::new();
    let mut fill = Vec::new();
    for ((buses, ybus), pattern) in states.iter().zip(ybuses).zip(patterns) {
        let (p, q) = power_injections(buses, ybus);
        pattern.fill(buses, &p, &q, &mut fill);
        values.extend_from_slice(&fill);
    }
    let ej = worst(&dev.jacobian_values().unwrap(), &values);
    eprintln!("{what}: Jacobian {ej:.1e}");
    assert!(ej < 1e-11, "{what}");

    let mut lin = CpuUniformBatch::analyze(dev.csr(), batch).unwrap();
    let all = vec![true; batch];
    lin.factor(&values, &all).unwrap();
    let mut dx = vec![0.0; batch * n];
    lin.solve(&rhs, &mut dx, &all).unwrap();
    dev.update(&dx).unwrap();
    for (s, buses) in states.iter_mut().enumerate() {
        let non_slack = buses
            .iter()
            .filter(|b| b.bus_type != BusType::Slack)
            .map(|b| (b.idx, true));
        let pq = buses
            .iter()
            .filter(|b| b.bus_type == BusType::PQ)
            .map(|b| (b.idx, false));
        let unknowns: Vec<(usize, bool)> = non_slack.chain(pq).collect();
        for (r, (i, angle)) in unknowns.into_iter().enumerate() {
            if angle {
                buses[i].voltage_ang += dx[s * n + r];
            } else {
                buses[i].voltage_mag += dx[s * n + r];
            }
        }
    }
    let (vm, va) = dev.voltages().unwrap();
    let vm_cpu: Vec<f64> = states.iter().flatten().map(|b| b.voltage_mag).collect();
    let va_cpu: Vec<f64> = states.iter().flatten().map(|b| b.voltage_ang).collect();
    assert_eq!(vm, vm_cpu, "{what}: the update is exact");
    assert_eq!(va, va_cpu, "{what}: the update is exact");

    dev.mismatch().unwrap();
    compare(dev, states, ybuses, &format!("{what}, after the step"));
    dev.norms().unwrap()
}

/// The device's norms after one step against the CPU loop's, stopped after
/// the same step.
fn assert_norms(dev: &[f64], cpu: &BatchReport, what: &str) {
    let cpu: Vec<f64> = cpu
        .scenarios
        .iter()
        .filter(|r| r.outcome != Outcome::Splits)
        .map(|r| r.final_mismatch)
        .collect();
    assert_eq!(dev.len(), cpu.len());
    for (d, c) in dev.iter().zip(cpu) {
        assert!(
            (d - c).abs() <= 1e-11 * c.max(1.0),
            "{what}: {d:e} against {c:e}"
        );
    }
}

fn one_step() -> BatchOptions {
    BatchOptions {
        policy: StepPolicy::Fixed(1),
        ..BatchOptions::default()
    }
}

/// A sweep, with ZIP loads and PV setpoint overrides.
#[test]
fn one_newton_step_matches_the_cpu() {
    for case in ["matpower/case1354pegase.m", "generated/mvlv1004.m"] {
        let (base, ybus) = load(case);
        let base = with_zip(base);
        let scenarios = ramp(&base, 32);
        let mut states: Vec<Vec<Bus>> = scenarios.iter().map(|sc| cpu_state(&base, sc)).collect();
        let mut dev = DeviceBatch::new(&base, &ybus, &scenarios).unwrap();
        let ybuses = vec![ybus.clone(); scenarios.len()];
        let patterns: Vec<JacobianPattern> = (0..scenarios.len())
            .map(|_| csr_jacobian(&base, &ybus).0)
            .collect();
        let norms = check_one_step(&mut dev, &mut states, &ybuses, &patterns, case);
        let cpu = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &one_step()).unwrap();
        assert_norms(&norms, &cpu, case);
    }
}

/// N-1: each outage's Y-bus built from scratch without its branch, as the
/// reference for the device's deltas on the shared one.
#[test]
fn one_newton_step_of_n1_matches_the_cpu() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/benchmark-grids/matpower/case1354pegase.m");
    let net = gridoxide::matpower::read(&path).unwrap();
    let (base, _) = load("matpower/case1354pegase.m");
    let n_bus = base.len();
    let n_branches = net.lines.len() + net.transformers.len();
    // Every 13th branch, single outages, and a few double ones.
    let mut scenarios: Vec<Scenario> = (0..n_branches)
        .step_by(13)
        .map(|b| Scenario {
            bus_overrides: Vec::new(),
            branch_outages: vec![b],
        })
        .collect();
    scenarios.extend((0..8).map(|k| Scenario {
        bus_overrides: Vec::new(),
        branch_outages: vec![100 * k + 7, 100 * k + 61],
    }));

    let (mut dev, members) = DeviceBatch::contingencies(
        &base,
        &net.lines,
        &net.transformers,
        &net.shunts,
        &scenarios,
    )
    .unwrap();
    assert!(
        members.len() > scenarios.len() / 2,
        "most outages keep the network connected"
    );
    eprintln!(
        "N-1: {} of {} outages keep the network connected",
        members.len(),
        scenarios.len()
    );
    let built = |outaged: &[bool]| {
        let mut y = build_ybus_with_outages(n_bus, &net.lines, &net.transformers, outaged);
        stamp_shunts(&mut y, &net.shunts);
        y.finish()
    };
    let intact = built(&[]);
    let mut ybuses = Vec::new();
    let mut patterns = Vec::new();
    for &s in &members {
        let mut outaged = vec![false; n_branches];
        for &b in &scenarios[s].branch_outages {
            outaged[b] = true;
        }
        let rebuilt = built(&outaged);
        let mut ybus = intact.clone();
        let mut pattern = csr_jacobian(&base, &intact).0;
        for &b in &scenarios[s].branch_outages {
            for (r, c, _) in branch_stamp(&net.lines, &net.transformers, b).into_entries() {
                let y = rebuilt.get(r, c);
                ybus.set(r, c, y).unwrap();
                pattern.set_admittance(r, c, y);
            }
        }
        ybuses.push(ybus);
        patterns.push(pattern);
    }
    let mut states = vec![base.clone(); members.len()];
    let norms = check_one_step(&mut dev, &mut states, &ybuses, &patterns, "N-1");
    let cpu = solve_contingency_batch::<CpuUniformBatch>(
        &base,
        &net.lines,
        &net.transformers,
        &net.shunts,
        &scenarios,
        &one_step(),
    )
    .unwrap();
    assert_norms(&norms, &cpu, "N-1");
}

/// An inactive scenario's buffers are left exactly as they were by every
/// kernel.
#[test]
fn inactive_scenarios_are_left_alone() {
    let (base, ybus) = load("matpower/case1354pegase.m");
    let scenarios = ramp(&base, 8);
    let mut dev = DeviceBatch::new(&base, &ybus, &scenarios).unwrap();
    let n = dev.n();
    let nnz = dev.csr().nnz();
    dev.mismatch().unwrap();
    dev.fill_jacobian().unwrap();
    let rhs0 = dev.rhs().unwrap();
    let norm0 = dev.norms().unwrap();
    let values0 = dev.jacobian_values().unwrap();
    let (vm0, va0) = dev.voltages().unwrap();

    let active: Vec<bool> = (0..scenarios.len()).map(|s| s % 2 == 0).collect();
    dev.set_active(&active).unwrap();
    dev.update(&vec![1e-3; scenarios.len() * n]).unwrap();
    dev.mismatch().unwrap();
    dev.fill_jacobian().unwrap();
    let rhs1 = dev.rhs().unwrap();
    let norm1 = dev.norms().unwrap();
    let values1 = dev.jacobian_values().unwrap();
    let (vm1, va1) = dev.voltages().unwrap();
    let n_bus = base.len();
    for (s, &on) in active.iter().enumerate() {
        let same = |a: &[f64], b: &[f64], k: usize| a[s * k..(s + 1) * k] == b[s * k..(s + 1) * k];
        assert_eq!(same(&rhs0, &rhs1, n), !on, "scenario {s}'s mismatch");
        assert_eq!(norm0[s] == norm1[s], !on, "scenario {s}'s norm");
        assert_eq!(
            same(&values0, &values1, nnz),
            !on,
            "scenario {s}'s Jacobian"
        );
        assert_eq!(same(&va0, &va1, n_bus), !on, "scenario {s}'s angles");
        assert_eq!(same(&vm0, &vm1, n_bus), !on, "scenario {s}'s magnitudes");
    }
}
