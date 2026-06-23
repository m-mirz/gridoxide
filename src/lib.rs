pub mod types;
pub mod network;
pub mod solver;
pub mod json;
pub mod pgm;

use network::build_ybus;
use solver::newton_raphson;
use json::NetworkData;
use types::Bus;
use nalgebra::{DMatrix, Complex};

pub fn run_power_flow_analysis(network_data: NetworkData) -> Vec<Bus> {
    let mut buses = network_data.buses;
    let lines = network_data.lines;

    let ybus = build_ybus(buses.len(), &lines);

    newton_raphson(&mut buses, &ybus, 1e-6, 20);

    buses
}

/// Runs a power flow analysis given a pre-built Y-bus matrix.
/// Intended for the 3-phase case where `buses` is a 3N-element vector and
/// `ybus` is the 3N×3N phase-domain admittance matrix from `build_ybus_3ph`.
pub fn run_power_flow_analysis_from_ybus(
    mut buses: Vec<Bus>,
    ybus: DMatrix<Complex<f64>>,
) -> Vec<Bus> {
    newton_raphson(&mut buses, &ybus, 1e-6, 20);
    buses
}
