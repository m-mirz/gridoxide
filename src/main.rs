//! gridoxide's command-line front end.
//!
//! Invoked with no arguments it runs the bundled power-flow demo, which is what
//! it has always done. `estimate <path>` runs state estimation over a PGM
//! document containing sensors.
//!
//! Argument handling is deliberately hand-rolled: two modes and one path do not
//! justify a dependency, and the crate otherwise has none for this.

use std::fs;
use std::path::PathBuf;

use gridoxide::json::NetworkData;
use gridoxide::measurement::measurements_from_pgm;
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_network, PgmInput};
use gridoxide::run_power_flow_analysis;
use gridoxide::se::bad_data::{self, Candidates};
use gridoxide::se::constraints::Constraints;
use gridoxide::se::jacobian::StateLayout;
use gridoxide::se::nr::{estimate, linear_start, SeMethod, SeOptions, SeStatus};
use gridoxide::se::observability;
use gridoxide::se::SeNetwork;
use gridoxide::solver::SolveStatus;

const USAGE: &str = "\
usage:
  gridoxide                     run the bundled power-flow demo
  gridoxide estimate <path> [--iterative-linear]
                                run state estimation over a PGM JSON document
                                containing sym_voltage_sensor/sym_power_sensor.
                                The default method is Newton-Raphson; the flag
                                selects the faster, less exact linearized one.
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => power_flow_demo(),
        Some("estimate") => match args.get(1) {
            Some(path) => {
                let method = if args.iter().any(|a| a == "--iterative-linear") {
                    SeMethod::IterativeLinear
                } else {
                    SeMethod::NewtonRaphson
                };
                if let Err(message) = run_estimate(path, method) {
                    eprintln!("error: {message}");
                    std::process::exit(1);
                }
            }
            None => {
                eprintln!("error: estimate needs a path\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("-h") | Some("--help") | Some("help") => print!("{USAGE}"),
        Some(other) => {
            eprintln!("error: unknown command {other:?}\n\n{USAGE}");
            std::process::exit(2);
        }
    }
}

fn power_flow_demo() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let network_json = fs::read_to_string(path).expect("Unable to read network.json");
    let network_data: NetworkData =
        serde_json::from_str(&network_json).expect("Unable to parse network.json");

    let report = run_power_flow_analysis(network_data);

    // The solver itself is silent so it can be driven from many threads at
    // once (see `batch::BatchSolver`); this reconstructs the progress output
    // it used to print, from `SolveStats::mismatch_history`.
    for (i, max_mis) in report.stats.mismatch_history.iter().enumerate() {
        println!("iter {}: max mismatch = {:.6e}", i + 1, max_mis);
    }
    match report.stats.status {
        SolveStatus::Converged => println!("Converged in {} iterations", report.stats.iterations()),
        SolveStatus::MaxIterationsReached => {
            println!("Failed to converge in {} iterations", report.stats.iterations())
        }
        SolveStatus::Singular => println!("Jacobian is singular. Failed to solve."),
    }

    println!("Final voltages:");
    for b in report.buses.iter() {
        println!(
            "Bus {}: |V| = {:.6}, angle = {:.6} deg",
            b.idx,
            b.voltage_mag,
            b.voltage_ang.to_degrees()
        );
    }

    println!("\nIslands: {} connected component(s)", report.islands.len());
    for (i, island) in report.islands.iter().enumerate() {
        println!("  island {i}: {} bus(es), status = {:?}", island.bus_indices.len(), island.status);
    }
}

/// Estimates the state of the grid in `path` and prints it, along with the two
/// analyses that say whether the answer should be trusted.
fn run_estimate(path: &str, method: SeMethod) -> Result<(), String> {
    let s_base_va = 1e6;
    let raw = fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let input: PgmInput =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?;
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, s_base_va);
    let net = pgm_to_network(
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?,
        s_base_va,
        50.0,
    );
    let measurements =
        measurements_from_pgm(&input, &net, s_base_va).map_err(|e| format!("{path}: {e}"))?;
    if measurements.is_empty() {
        return Err(format!("{path} contains no usable sensors, so there is nothing to estimate"));
    }

    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &shunts);
    let se_net = SeNetwork::new(&net, ybus.finish(), &shunts);

    let mut buses = net.buses.clone();
    linear_start(&mut buses, &se_net, &measurements);
    // The linearized method converges linearly rather than quadratically, so it
    // wants a larger budget for the same tolerance — that trade, a cheaper
    // iteration for more of them, is the point of it.
    let max_iter = match method {
        SeMethod::IterativeLinear => 100,
        SeMethod::NewtonRaphson => 20,
    };
    let report = estimate(
        &measurements,
        &mut buses,
        &se_net,
        &SeOptions { method, max_iter, ..SeOptions::default() },
    );

    println!(
        "{} bus(es), {} measurement(s) after aggregation",
        buses.len(),
        measurements.len()
    );
    match report.status {
        SeStatus::Converged => {
            println!("Converged in {} iteration(s)", report.iterations)
        }
        SeStatus::MaxIterations => println!(
            "Did not converge in {} iteration(s); last step {:.3e}",
            report.iterations, report.last_step
        ),
        SeStatus::Singular => println!(
            "Gain matrix is singular after {} iteration(s) — see the observability \
             report below",
            report.iterations
        ),
    }
    println!("Objective J(x) = {:.6e}", report.objective);

    println!("\nEstimated voltages:");
    let mut ids: Vec<(&u64, &usize)> = net.node_idx.iter().collect();
    ids.sort();
    for (id, &idx) in ids {
        println!(
            "  node {id}: |V| = {:.6} p.u., angle = {:.6} deg",
            buses[idx].voltage_mag,
            buses[idx].voltage_ang.to_degrees()
        );
    }

    let layout = StateLayout::new(&buses, &measurements, &se_net);
    let obs = observability::analyze(&measurements, &buses, &se_net, &layout);
    println!(
        "\nObservability: rank {} of {} unknown(s)",
        obs.rank, obs.n_unknowns
    );
    if obs.skipped_numerical {
        println!("  (too large for the dense rank check; structural analysis only)");
    }
    // Buses beyond the physical node count are gridoxide's own synthesized
    // ones — a virtual slack bus per source, a star point per three-winding
    // transformer — and are expected to be unobservable when the source's own
    // power is unmeasured. Saying which is which avoids a false alarm.
    let n_physical = net.node_idx.len();
    // The two lists overlap by design — a structurally unmeasured column is
    // also rank-deficient — so they are merged before printing rather than
    // reported twice.
    let mut undetermined: Vec<_> =
        obs.unobservable.iter().chain(&obs.structurally_unmeasured).copied().collect();
    undetermined.sort_by_key(|u| (u.bus, format!("{:?}", u.quantity)));
    undetermined.dedup();
    for unknown in undetermined {
        let kind = if unknown.bus >= n_physical { "synthesized" } else { "physical" };
        println!("  undetermined: {kind} bus {} ({:?})", unknown.bus, unknown.quantity);
    }
    if obs.is_observable() {
        println!("  fully observable");
    }

    let constraints = Constraints::new(&se_net.zero_injection);
    let bad = bad_data::analyze(
        &measurements,
        &report.residuals,
        &buses,
        &se_net,
        &layout,
        &constraints,
        Candidates::default(),
    );
    println!(
        "\nBad data: chi-squared {:.4e} on {} dof, p = {:.4e}",
        bad.chi_squared, bad.degrees_of_freedom, bad.p_value
    );
    if bad.rejects_at(0.05) {
        println!("  REJECTED at 5%: the measurements are not merely noisy");
        for suspect in bad.suspects.iter().take(5) {
            let m = &measurements[suspect.measurement];
            println!(
                "  suspect: measurement {} ({:?} on {:?}), normalized residual {:.2}",
                suspect.measurement, m.kind, m.target, suspect.normalized_residual
            );
        }
    } else {
        println!("  not rejected at 5%");
    }

    Ok(())
}
