//! MATPOWER import, checked against the case's own equations.
//!
//! Each case is read, solved from a flat start, and the solution substituted
//! into the power-flow equations rebuilt here from the raw matrices with
//! MATPOWER's own formulas (`makeYbus`, `makeSbus`, `bustypes`), sharing
//! nothing with `src/matpower.rs` but the parser. A conversion that solved a
//! different network — a ratio on the wrong side, a lost shift, a setpoint
//! taken from `Vm` instead of `Vg` — converges happily and fails here.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use num_complex::Complex;

use gridoxide::matpower::{self, MatpowerCase};
use gridoxide::solver::SolveStatus;
use gridoxide::{PowerFlowInit, PowerFlowOptions, TapData, run_power_flow};

fn case_path(dir: &str, name: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/benchmark-grids")
        .join(dir)
        .join(format!("{name}.m"));
    if path.exists() {
        Some(path)
    } else {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/benchmark-grids`",
            path.display()
        );
        None
    }
}

/// The largest power mismatch, per unit, of `vm`/`va` (per bus row of
/// `mpc.bus`) in the case's equations: P and Q at PQ buses, P at PV buses,
/// and the setpoints a regulated bus must hold.
fn residual(case: &MatpowerCase, vm: &[f64], va: &[f64]) -> f64 {
    let base = case.base_mva;
    let index: HashMap<u64, usize> = case
        .bus
        .iter()
        .enumerate()
        .map(|(i, b)| (b[0] as u64, i))
        .collect();
    let n = case.bus.len();
    let isolated: Vec<bool> = case.bus.iter().map(|b| b[1] == 4.0).collect();

    let mut y: Vec<HashMap<usize, Complex<f64>>> = vec![HashMap::new(); n];
    for br in &case.branch {
        let (f, t) = (index[&(br[0] as u64)], index[&(br[1] as u64)]);
        if br[10] == 0.0 || isolated[f] || isolated[t] {
            continue;
        }
        let ys = Complex::new(1.0, 0.0) / Complex::new(br[2], br[3]);
        let ratio = if br[8] == 0.0 { 1.0 } else { br[8] };
        let tap = Complex::from_polar(ratio, br[9].to_radians());
        let ytt = ys + Complex::new(0.0, br[4] / 2.0);
        *y[f].entry(f).or_default() += ytt / (ratio * ratio);
        *y[f].entry(t).or_default() += -ys / tap.conj();
        *y[t].entry(f).or_default() += -ys / tap;
        *y[t].entry(t).or_default() += ytt;
    }
    for (i, b) in case.bus.iter().enumerate() {
        if !isolated[i] {
            *y[i].entry(i).or_default() += Complex::new(b[4], b[5]) / base;
        }
    }

    let mut s_spec: Vec<Complex<f64>> = case
        .bus
        .iter()
        .map(|b| -Complex::new(b[2], b[3]) / base)
        .collect();
    let mut vg: Vec<Option<f64>> = vec![None; n];
    for g in &case.gens {
        let at = index[&(g[0] as u64)];
        if g[7] > 0.0 && !isolated[at] {
            s_spec[at] += Complex::new(g[1], g[2]) / base;
            vg[at] = Some(g[5]);
        }
    }

    let v: Vec<Complex<f64>> = (0..n).map(|i| Complex::from_polar(vm[i], va[i])).collect();
    let mut worst: f64 = 0.0;
    for i in (0..n).filter(|&i| !isolated[i]) {
        let current: Complex<f64> = y[i].iter().map(|(&j, &yij)| yij * v[j]).sum();
        let mismatch = v[i] * current.conj() - s_spec[i];
        let kind = case.bus[i][1];
        match (kind, vg[i]) {
            (3.0, Some(set)) => {
                worst = worst.max((vm[i] - set).abs());
                worst = worst.max((va[i] - case.bus[i][8].to_radians()).abs());
            }
            (2.0, Some(set)) => {
                worst = worst.max((vm[i] - set).abs());
                worst = worst.max(mismatch.re.abs());
            }
            _ => worst = worst.max(mismatch.re.abs()).max(mismatch.im.abs()),
        }
    }
    worst
}

/// Reads, solves from a flat start, and returns the largest residual.
fn solve(dir: &str, name: &str) -> Option<(f64, usize)> {
    let path = case_path(dir, name)?;
    let text = std::fs::read_to_string(&path).unwrap();
    let case = matpower::parse(&text).unwrap();
    let net = matpower::read(&path).unwrap();
    let report = run_power_flow(
        net.buses,
        &net.lines,
        &net.transformers,
        &net.shunts,
        TapData::none(),
        PowerFlowOptions {
            init: PowerFlowInit::Flat,
            tol: 1e-8,
            max_iter: 30,
            ..Default::default()
        },
    );
    assert_eq!(
        report.stats.status,
        SolveStatus::Converged,
        "{name} did not converge from a flat start"
    );
    let vm: Vec<f64> = report.buses.iter().map(|b| b.voltage_mag).collect();
    let va: Vec<f64> = report.buses.iter().map(|b| b.voltage_ang).collect();
    Some((residual(&case, &vm, &va), report.stats.iterations()))
}

/// Transmission cases: line charging, off-nominal ratios, phase shifters
/// (PEGASE), PV buses without generators (case3120sp) and a reference angle
/// that is not zero.
#[test]
fn transmission_cases_solve_the_case_they_define() {
    for name in [
        "case14",
        "case118",
        "case300",
        "case_ACTIVSg200",
        "case1354pegase",
        "case2869pegase",
        "case3120sp",
        "case9241pegase",
    ] {
        let Some((worst, iterations)) = solve("matpower", name) else {
            continue;
        };
        assert!(
            worst < 1e-7,
            "{name}: residual {worst:e} p.u. after {iterations} iterations"
        );
    }
}

/// Distribution feeders, in the plain-data copies whose units are already
/// converted: high R/X, heavy loading, radial.
#[test]
fn distribution_feeders_solve_the_case_they_define() {
    for name in [
        "case33bw",
        "case69",
        "case85",
        "case141",
        "case18nbr",
        "case22",
    ] {
        let Some((worst, iterations)) = solve("matpower-plain", name) else {
            continue;
        };
        assert!(
            worst < 1e-7,
            "{name}: residual {worst:e} p.u. after {iterations} iterations"
        );
    }
}

/// `case33bw` as MATPOWER ships it converts its own units with MATLAB code
/// after the matrices; reading only the literals would solve loads 1000 times
/// too large. It is refused instead.
#[test]
fn a_case_that_converts_its_units_in_code_is_refused() {
    let Some(path) = case_path("matpower", "case33bw") else {
        return;
    };
    let err = matpower::read(&path).unwrap_err().to_string();
    assert!(err.contains("modified by MATLAB code"), "{err}");
}
