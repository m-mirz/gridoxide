//! Times the two state-estimation methods against each other at scale.
//!
//! Every state-estimation fixture committed to this repo is 12 buses or fewer,
//! so nothing so far says how either method behaves on a real grid — or whether
//! it works there at all. This runs both over the MATPOWER benchmark cases the
//! power-flow benchmarks already use.
//!
//! # How the measurements are made
//!
//! There is no telemetry for these grids, so a measurement set is synthesized
//! from a converged power flow: solve, then read the quantities a real SCADA
//! system would report off that solution, exactly. That makes the estimator's
//! job easy in one specific way — the data is perfectly consistent, so the
//! objective collapses to zero and the residuals carry no information. It is
//! still the right shape for a *timing* comparison, because cost is set by the
//! measurement count and the sparsity pattern, not by how well the data agrees.
//!
//! Redundancy is the ratio of measurements to unknowns. Real estimators run
//! around 1.5-3x; this builds voltage magnitudes everywhere plus branch flows at
//! both ends of every branch, which lands in that range.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example bench_se
//! ```

use std::path::PathBuf;
use std::time::Instant;

use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use gridoxide::measurement::{Measurement, MeasurementKind, Target};
use gridoxide::network::{build_ybus, linear_initial_guess, power_injections, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_network, PgmInput};
use gridoxide::se::nr::{estimate, flat_start, SeMethod, SeOptions, SeStatus};
use gridoxide::se::SeNetwork;
use gridoxide::solver::{JacobianBackend, PersistentSolver};
use gridoxide::types::Bus;

const S_BASE_VA: f64 = 1e6;

/// Sensor accuracies, per-unit. Loosely realistic: voltage transducers are the
/// best-trusted instrument on a substation, power measurements much less so.
const SIGMA_V: f64 = 1e-3;
const SIGMA_S: f64 = 1e-2;

fn main() {
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/bench/.case-cache");
    let mut cases: Vec<String> = std::env::args().skip(1).collect();
    if cases.is_empty() {
        cases = ["case14", "case118", "case300", "case1354pegase", "case2869pegase"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    }

    println!(
        "{:<16} {:>6} {:>7} {:>6} {:>10} {:>7} {:>10} {:>7}  {}",
        "case", "buses", "meas", "redun", "NR (ms)", "iters", "IL (ms)", "iters", "verdict"
    );

    for case in &cases {
        let path = cache.join(format!("{case}.json"));
        let Ok(text) = std::fs::read_to_string(&path) else {
            println!("{case:<16} (not in .case-cache — run scripts/bench/matpower_to_pgm.py)");
            continue;
        };
        let input: PgmInput = match serde_json::from_str(&text) {
            Ok(i) => i,
            Err(e) => {
                println!("{case:<16} parse error: {e}");
                continue;
            }
        };

        let id_to_idx = node_id_to_idx(&input);
        let shunts = pgm_shunts_1ph(&input, &id_to_idx, S_BASE_VA);
        let net = pgm_to_network(serde_json::from_str(&text).unwrap(), S_BASE_VA, 50.0);
        let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
        stamp_shunts(&mut ybus, &shunts);
        let ybus = ybus.finish();

        // The "true" state: an ordinary power-flow solution.
        let mut truth = net.buses.clone();
        linear_initial_guess(&mut truth, &ybus);
        let mut solver = PersistentSolver::new(JacobianBackend::Scalar);
        solver.solve(&mut truth, &ybus, 1e-10, 40);

        let se_net = SeNetwork::new(&net, ybus, &shunts);
        let measurements = synthesize(&truth, &se_net, &net);

        // 2N-1 unknowns; redundancy below ~1 cannot possibly be observable.
        let unknowns = 2 * truth.len() - 1;
        let redundancy = measurements.len() as f64 / unknowns as f64;

        let run = |method| {
            let mut buses = net.buses.clone();
            flat_start(&mut buses, &measurements);
            let options = SeOptions {
                method,
                max_iter: if method == SeMethod::IterativeLinear { 100 } else { 20 },
                ..SeOptions::default()
            };
            let start = Instant::now();
            let report = estimate(&measurements, &mut buses, &se_net, &options);
            (start.elapsed().as_secs_f64() * 1e3, report, buses)
        };

        let (nr_ms, nr, nr_buses) = run(SeMethod::NewtonRaphson);
        let (il_ms, il, il_buses) = run(SeMethod::IterativeLinear);

        // Do the two agree, and did either recover the state the measurements
        // were read from? Timing a method that produced the wrong answer, or no
        // answer, would be meaningless.
        let worst = |a: &[Bus]| {
            a.iter()
                .zip(&truth)
                .map(|(x, t)| (x.voltage_mag - t.voltage_mag).abs())
                .fold(0.0f64, f64::max)
        };
        let verdict = match (nr.status, il.status) {
            (SeStatus::Converged, SeStatus::Converged) => {
                format!("dV nr={:.1e} il={:.1e}", worst(&nr_buses), worst(&il_buses))
            }
            (a, b) => format!("nr={a:?} il={b:?}"),
        };

        println!(
            "{case:<16} {:>6} {:>7} {:>6.2} {:>10.1} {:>7} {:>10.1} {:>7}  {verdict}",
            truth.len(),
            measurements.len(),
            redundancy,
            nr_ms,
            nr.iterations,
            il_ms,
            il.iterations,
        );
    }
}

/// Reads a measurement set off a known state: a voltage magnitude at every bus,
/// and active/reactive flow at both ends of every branch.
///
/// Branch flows rather than bus injections, because that is what a real system
/// mostly has — line-end telemetry is far more common than a metered injection —
/// and because it produces the sparsity pattern the estimator will actually
/// meet.
fn synthesize(truth: &[Bus], net: &SeNetwork, pgm: &gridoxide::pgm::PgmNetwork) -> Vec<Measurement> {
    let v = bus_voltages(truth);
    let params = branch_params(&pgm.lines, &pgm.transformers);
    let (p_inj, q_inj) = power_injections(truth, &net.ybus);
    let mut out = Vec::new();

    for (bus, b) in truth.iter().enumerate() {
        out.push(Measurement {
            kind: MeasurementKind::VoltageMagnitude,
            target: Target::Bus(bus),
            value: b.voltage_mag,
            sigma: SIGMA_V,
        });
        // An injection measurement wherever there is something to inject; a bus
        // that injects nothing is a zero-injection bus and is constrained, not
        // measured.
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
