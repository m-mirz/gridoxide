pub mod types;
pub mod network;
pub mod solver;
pub mod jacobian;
pub mod batch;
pub mod dc;
pub mod json;
pub mod pgm;
pub mod sparse;
pub mod block_sparse;
pub mod klu_native;
#[cfg(feature = "klu")]
pub mod sparse_klu;
#[cfg(feature = "pardiso")]
pub mod sparse_pardiso;
#[cfg(feature = "cgmes")]
pub mod cgmes;
#[cfg(feature = "python")]
mod python;

use network::{build_ybus, linear_initial_guess, YBus};
use solver::{IslandReport, JacobianBackend, PersistentSolver, SolveStats};
use json::NetworkData;
use types::Bus;

/// Result of a full power-flow analysis: the solved (or placeholder, for
/// unreferenced islands) buses, plus a per-connected-component breakdown of
/// how each one was resolved — see [`solver::IslandReport`]/
/// [`solver::IslandStatus`].
#[derive(Debug)]
pub struct PowerFlowReport {
    pub buses: Vec<Bus>,
    pub islands: Vec<IslandReport>,
    /// Iteration count and per-iteration convergence trace. The solver
    /// itself prints nothing (see [`solver::SolveStats`]); `src/main.rs`
    /// reconstructs the progress output from this.
    pub stats: SolveStats,
}

pub fn run_power_flow_analysis(network_data: NetworkData) -> PowerFlowReport {
    let ybus = build_ybus(network_data.buses.len(), &network_data.lines, &[]);
    run_power_flow_analysis_from_ybus(network_data.buses, ybus)
}

/// Runs a power flow analysis given a pre-built Y-bus matrix.
/// Intended for the 3-phase case too, where `buses` is a 3N-element vector
/// and `ybus` is the 3N×3N phase-domain admittance matrix from
/// `build_ybus_3ph` — a physical bus's 3 phase-rows always share the same
/// `BusType`, so the island partitioning `PersistentSolver::solve` does
/// internally is safe for that case too (worst case, in a network with
/// perfectly balanced zero/positive-sequence impedances, phases may
/// partition into separate components rather than staying grouped by
/// physical bus; each still carries its own valid reference, so this
/// doesn't change correctness, only granularity).
///
/// Every disconnected component of `ybus` is solved in this same call (not
/// just the largest one) — see [`solver::PersistentSolver::solve`], the one
/// canonical solve entry point every public function in this crate
/// (including this one) ultimately goes through. `PowerFlowReport::islands`
/// gives the resulting per-component breakdown.
pub fn run_power_flow_analysis_from_ybus(
    mut buses: Vec<Bus>,
    ybus: YBus,
) -> PowerFlowReport {
    let ybus = ybus.finish();
    linear_initial_guess(&mut buses, &ybus);
    let mut solver = PersistentSolver::new(JacobianBackend::Scalar);
    let (islands, stats) = solver.solve_with_stats(&mut buses, &ybus, 1e-6, 20);
    PowerFlowReport { buses, islands, stats }
}
