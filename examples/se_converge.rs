//! How the iterative-linear estimate approaches its answer, iteration by iteration.
//!
//! The step test is what decides `max_iter`, but it is not what a user cares
//! about; the error against the state the measurements were read from is. This
//! prints both, so the two can be compared.
use std::path::PathBuf;

use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use gridoxide::measurement::{Measurement, MeasurementKind, Target};
use gridoxide::network::{build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_network, PgmInput};
use gridoxide::se::nr::{estimate, flat_start, SeMethod, SeOptions};
use gridoxide::se::SeNetwork;
use gridoxide::solver::{JacobianBackend, PersistentSolver};
use gridoxide::types::Bus;

const S_BASE_VA: f64 = 1e6;
const SIGMA_V: f64 = 1e-3;
const SIGMA_S: f64 = 1e-2;

fn main() {
    let case = std::env::args().nth(1).unwrap_or_else(|| "case1354pegase".into());
    let limit: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(100);
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts/bench/.case-cache")
        .join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path).unwrap();
    let input: PgmInput = serde_json::from_str(&text).unwrap();
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, S_BASE_VA);
    let net = pgm_to_network(serde_json::from_str(&text).unwrap(), S_BASE_VA, 50.0);
    let mut yb = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut yb, &shunts);
    let ybus = yb.finish();

    let mut truth = net.buses.clone();
    linear_initial_guess(&mut truth, &ybus);
    PersistentSolver::new(JacobianBackend::Scalar).solve(&mut truth, &ybus, 1e-10, 40);
    let se_net = SeNetwork::new(&net, ybus, &shunts);
    let measurements = synthesize(&truth, &se_net, &net);

    println!("{case}: {} buses, {} measurements", truth.len(), measurements.len());
    println!("{:>5} {:>12} {:>8} {:>12} {:>12}", "iter", "step", "ratio", "worst dV", "objective");

    let mut previous = f64::NAN;
    for k in 1..=limit {
        let mut buses = net.buses.clone();
        flat_start(&mut buses, &measurements);
        let options = SeOptions {
            method: SeMethod::IterativeLinear,
            max_iter: k,
            ..SeOptions::default()
        };
        let report = estimate(&measurements, &mut buses, &se_net, &options);
        let worst = buses
            .iter()
            .zip(&truth)
            .map(|(x, t)| (x.voltage_mag - t.voltage_mag).abs())
            .fold(0.0f64, f64::max);
        println!(
            "{k:>5} {:>12.3e} {:>8.4} {worst:>12.3e} {:>12.6e}",
            report.last_step,
            report.last_step / previous,
            report.objective
        );
        previous = report.last_step;
        if report.status == gridoxide::se::nr::SeStatus::Converged {
            println!("converged at iteration {}", report.iterations);
            break;
        }
    }
}

/// Same synthesis as `bench_se.rs`, inlined.
fn synthesize(truth: &[Bus], net: &SeNetwork, pgm: &gridoxide::pgm::PgmNetwork) -> Vec<Measurement> {
    let v = bus_voltages(truth);
    let params = branch_params(&pgm.lines, &pgm.transformers);
    let (p_inj, q_inj) = gridoxide::network::power_injections(truth, &net.ybus);
    let mut out = Vec::new();

    for (bus, b) in truth.iter().enumerate() {
        out.push(Measurement {
            kind: MeasurementKind::VoltageMagnitude,
            target: Target::Bus(bus),
            value: b.voltage_mag,
            sigma: SIGMA_V,
        });
        if p_inj[bus].abs() > 1e-9 || q_inj[bus].abs() > 1e-9 {
            out.push(Measurement {
                kind: MeasurementKind::ActivePower,
                target: Target::Bus(bus),
                value: p_inj[bus],
                sigma: SIGMA_S,
            });
            out.push(Measurement {
                kind: MeasurementKind::ReactivePower,
                target: Target::Bus(bus),
                value: q_inj[bus],
                sigma: SIGMA_S,
            });
        }
    }

    for (branch, p) in params.iter().enumerate() {
        if p.from == p.to {
            continue;
        }
        for terminal in [Terminal::From, Terminal::To] {
            let (pf, qf) = terminal_flow(p, terminal, &v);
            let target = Target::BranchTerminal { branch, terminal };
            out.push(Measurement {
                kind: MeasurementKind::ActivePower,
                target,
                value: pf,
                sigma: SIGMA_S,
            });
            out.push(Measurement {
                kind: MeasurementKind::ReactivePower,
                target,
                value: qf,
                sigma: SIGMA_S,
            });
        }
    }
    out
}
