pub mod types;
pub mod network;
pub mod solver;
pub mod json;
pub mod pgm;
pub mod sparse;
pub mod block_sparse;
#[cfg(feature = "klu")]
pub mod sparse_klu;
#[cfg(feature = "python")]
mod python;

use network::{build_ybus, linear_initial_guess, YBus};
use solver::newton_raphson;
use json::NetworkData;
use types::Bus;

pub fn run_power_flow_analysis(network_data: NetworkData) -> Vec<Bus> {
    let ybus = build_ybus(network_data.buses.len(), &network_data.lines, &[]);
    run_power_flow_analysis_from_ybus(network_data.buses, ybus)
}

/// Runs a power flow analysis given a pre-built Y-bus matrix.
/// Intended for the 3-phase case where `buses` is a 3N-element vector and
/// `ybus` is the 3N×3N phase-domain admittance matrix from `build_ybus_3ph`.
pub fn run_power_flow_analysis_from_ybus(
    mut buses: Vec<Bus>,
    ybus: YBus,
) -> Vec<Bus> {
    let ybus = ybus.finish();
    linear_initial_guess(&mut buses, &ybus);
    newton_raphson(&mut buses, &ybus, 1e-6, 20);
    buses
}
