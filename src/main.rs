use std::fs;
use std::path::PathBuf;
use gridoxide::json::NetworkData;
use gridoxide::run_power_flow_analysis;
use gridoxide::solver::SolveStatus;

fn main() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let network_json = fs::read_to_string(path).expect("Unable to read network.json");
    let network_data: NetworkData = serde_json::from_str(&network_json).expect("Unable to parse network.json");

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
