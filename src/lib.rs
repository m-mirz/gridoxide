pub mod types;
pub mod network;
pub mod solver;
pub mod json;

use network::build_ybus;
use solver::newton_raphson;
use json::NetworkData;
use types::Bus;

/// Runs a power flow analysis on the given network data.
///
/// # Arguments
///
/// * `network_data` - The network data to analyze.
///
/// # Returns
///
/// A vector of buses with the calculated voltage magnitudes and angles.
pub fn run_power_flow_analysis(network_data: NetworkData) -> Vec<Bus> {
    let mut buses = network_data.buses;
    let lines = network_data.lines;

    let ybus = build_ybus(buses.len(), &lines);

    newton_raphson(&mut buses, &ybus, 1e-6, 20);

    buses
}
