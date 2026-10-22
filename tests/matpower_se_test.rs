//! State estimation on a MATPOWER case read directly (`matpower::se_case`).
//!
//! The measurements are taken at a solved state with MATPOWER's own formulas
//! (`makeYbus`, `makeSbus`, and the branch admittances behind `PF`/`QF` and
//! `PT`/`QT`), rebuilt here from the raw matrices and sharing nothing with
//! gridoxide but the parser. Noise-free and complete, they determine that
//! state exactly, so an estimate that places a measurement wrongly (a bus, a
//! branch row, an end, a sign, a unit) misses it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use num_complex::Complex;

use gridoxide::matpower::{self, CaseMeasurement, MatpowerCase, MeasuredQuantity as Q};
use gridoxide::se::nr::{SeOptions, SeStatus, estimate, linear_start};
use gridoxide::solver::SolveStatus;
use gridoxide::{PowerFlowInit, PowerFlowOptions, TapData, run_power_flow};

fn case_path(name: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/benchmark-grids/matpower")
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

/// |V| and the injection at every energized bus, and the flow at both ends of
/// every branch in service, at the state `v` (one per row of `mpc.bus`).
fn measurements_at(case: &MatpowerCase, v: &[Complex<f64>]) -> Vec<CaseMeasurement> {
    let base = case.base_mva;
    let index: HashMap<u64, usize> = case
        .bus
        .iter()
        .enumerate()
        .map(|(i, b)| (b[0] as u64, i))
        .collect();
    let live: Vec<bool> = case.bus.iter().map(|b| b[1] != 4.0).collect();
    let measure = |quantity, at, value: f64| CaseMeasurement {
        quantity,
        at,
        value,
        sigma: 1e-3 + 1e-2 * value.abs(),
    };

    let mut current = vec![Complex::new(0.0, 0.0); v.len()];
    let mut out = Vec::new();
    for (row, br) in case.branch.iter().enumerate() {
        let (f, t) = (index[&(br[0] as u64)], index[&(br[1] as u64)]);
        if br[10] == 0.0 || !live[f] || !live[t] {
            continue;
        }
        let ys = Complex::new(1.0, 0.0) / Complex::new(br[2], br[3]);
        let ratio = if br[8] == 0.0 { 1.0 } else { br[8] };
        let tap = Complex::from_polar(ratio, br[9].to_radians());
        let ytt = ys + Complex::new(0.0, br[4] / 2.0);
        let i_from = ytt / (ratio * ratio) * v[f] - ys / tap.conj() * v[t];
        let i_to = -ys / tap * v[f] + ytt * v[t];
        current[f] += i_from;
        current[t] += i_to;
        let (s_from, s_to) = (v[f] * i_from.conj() * base, v[t] * i_to.conj() * base);
        let row = row as u64;
        out.push(measure(Q::ActiveFlowFrom, row, s_from.re));
        out.push(measure(Q::ReactiveFlowFrom, row, s_from.im));
        out.push(measure(Q::ActiveFlowTo, row, s_to.re));
        out.push(measure(Q::ReactiveFlowTo, row, s_to.im));
    }
    for (i, b) in case.bus.iter().enumerate() {
        if !live[i] {
            continue;
        }
        // The shunt is part of the network: its current is in the bus's
        // total, and the injection is what remains, generation less load.
        current[i] += Complex::new(b[4], b[5]) / base * v[i];
        let s = v[i] * current[i].conj() * base;
        let number = b[0] as u64;
        out.push(measure(Q::VoltageMagnitude, number, v[i].norm()));
        out.push(measure(Q::ActiveInjection, number, s.re));
        out.push(measure(Q::ReactiveInjection, number, s.im));
    }
    out
}

/// Plain bus-branch cases, PEGASE's phase shifters, and case3120sp's
/// transformers with line charging and branches of negative impedance: the
/// losses a conversion to PGM would bring.
#[test]
fn estimates_the_state_the_measurements_were_taken_at() {
    for name in ["case14", "case300", "case1354pegase", "case3120sp"] {
        let Some(path) = case_path(name) else {
            return;
        };
        let case = matpower::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let net = matpower::read(&path).unwrap();
        let solved = run_power_flow(
            net.buses.clone(),
            &net.lines,
            &net.transformers,
            &net.shunts,
            TapData::none(),
            PowerFlowOptions {
                init: PowerFlowInit::Flat,
                tol: 1e-10,
                max_iter: 30,
                ..Default::default()
            },
        );
        assert_eq!(solved.stats.status, SolveStatus::Converged, "{name}");
        let truth: Vec<Complex<f64>> = solved
            .buses
            .iter()
            .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
            .collect();

        let se = matpower::se_case(&net, &measurements_at(&case, &truth)).unwrap();
        let mut buses = se.buses.clone();
        linear_start(&mut buses, &se.network, &se.measurements);
        let report = estimate(
            &se.measurements,
            &mut buses,
            &se.network,
            &SeOptions::default(),
        );
        assert_eq!(report.status, SeStatus::Converged, "{name}: {report:?}");
        assert!(
            report.objective < 1e-9,
            "{name}: objective {:.3e}",
            report.objective
        );

        let reference = se.network.reference_bus(&buses);
        let shift = buses[reference].voltage_ang - truth[reference].arg();
        for (i, (b, t)) in buses.iter().zip(&truth).enumerate() {
            if !se.network.energized[i] {
                continue;
            }
            assert!(
                (b.voltage_mag - t.norm()).abs() < 1e-7,
                "{name} bus {}: |V| {} estimated, {} true",
                net.bus_numbers[i],
                b.voltage_mag,
                t.norm()
            );
            assert!(
                (b.voltage_ang - shift - t.arg()).abs() < 1e-7,
                "{name} bus {}: angle {} estimated, {} true",
                net.bus_numbers[i],
                b.voltage_ang - shift,
                t.arg()
            );
        }
    }
}

/// A measurement that names nothing in the solved network is refused, by
/// name, rather than dropped.
#[test]
fn a_measurement_on_nothing_is_refused() {
    let Some(path) = case_path("case14") else {
        return;
    };
    let net = matpower::read(&path).unwrap();
    let one = |quantity, at, sigma| {
        let m = CaseMeasurement {
            quantity,
            at,
            value: 1.0,
            sigma,
        };
        matpower::se_case(&net, &[m])
            .map(|_| ())
            .unwrap_err()
            .to_string()
    };
    assert!(one(Q::VoltageMagnitude, 99, 0.01).contains("does not define"));
    assert!(one(Q::ActiveFlowFrom, 20, 0.01).contains("no such branch in service"));
    assert!(one(Q::ActiveInjection, 2, 0.0).contains("positive sigma"));
    assert!(
        matpower::se_case(&net, &[])
            .map(|_| ())
            .unwrap_err()
            .to_string()
            .contains("no measurements")
    );
}
