pub mod types;
pub mod network;
pub mod switches;
pub mod topology;
pub mod branch_flow;
pub mod measurement;
pub mod se;
pub mod solver;
pub mod jacobian;
pub mod batch;
pub mod bde;
pub mod dc;
pub mod linear;
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

use linear::{
    btheta::{dc_power_flow, DcIslandStatus, DcSolution},
    impedance::{linear_power_flow, LinearIslandStatus, LinearReport},
};
use network::{build_ybus, linear_initial_guess, stamp_shunts, ShuntAdm, YBus};
use solver::{
    IslandReport, JacobianBackend, PersistentSolver, PowerFlowInit, PowerFlowMethod,
    PowerFlowOptions, SolveStats, SolveStatus,
};
use json::NetworkData;
use types::{Bus, Line, Transformer};

/// Result of a full power-flow analysis: the solved (or placeholder, for
/// unreferenced islands) buses, plus a per-connected-component breakdown of
/// how each one was resolved — see [`solver::IslandReport`]/
/// [`solver::IslandStatus`].
#[derive(Debug)]
pub struct PowerFlowReport {
    pub buses: Vec<Bus>,
    /// Newton-Raphson's per-component breakdown. **Empty** for the two direct
    /// methods, which report through [`dc`](Self::dc) and
    /// [`linear`](Self::linear) instead.
    ///
    /// This is not laziness. `IslandStatus::MaxIterationsReached` cannot
    /// happen without iterations, and `IslandStatus::AmbiguousReferenceBus`
    /// means "over-determined, the answer may satisfy neither reference" —
    /// which is true of AC and false of DC, where two references leave the
    /// system perfectly well posed. Mapping onto this vocabulary would state
    /// things about a DC solve that are not so.
    pub islands: Vec<IslandReport>,
    /// Iteration count and per-iteration convergence trace. The solver
    /// itself prints nothing (see [`solver::SolveStats`]); `src/main.rs`
    /// reconstructs the progress output from this.
    ///
    /// For the direct methods this carries only a status — `Converged` or
    /// `Singular` — with an empty history, since they take no iterations.
    pub stats: SolveStats,
    /// Populated only by [`PowerFlowMethod::Dc`]: per-branch active flows,
    /// per-island slack pickup, and the solve's residual.
    pub dc: Option<DcSolution>,
    /// Populated only by [`PowerFlowMethod::LinearImpedance`].
    pub linear: Option<LinearReport>,
}

impl PowerFlowReport {
    /// The shape every method-specific constructor starts from.
    fn empty(buses: Vec<Bus>, status: SolveStatus) -> Self {
        Self {
            buses,
            islands: Vec::new(),
            stats: SolveStats {
                status,
                mismatch_history: Vec::new(),
                q_limit_switches: Vec::new(),
                q_limit_stabilized: true,
            },
            dc: None,
            linear: None,
        }
    }
}

pub fn run_power_flow_analysis(network_data: NetworkData) -> PowerFlowReport {
    let ybus = build_ybus(network_data.buses.len(), &network_data.lines, &[]);
    run_power_flow_analysis_from_ybus(network_data.buses, ybus)
}

/// Runs a power flow by the method named in `opts`.
///
/// Takes branch lists rather than a finished Y-bus because
/// [`PowerFlowMethod::Dc`] is formulated per branch — it needs each branch's
/// own `x` and `tap`, which a Y-bus has already summed away — and never
/// builds one at all. The other two methods assemble their Y-bus here, from
/// these same lists plus `shunts`.
///
/// [`run_power_flow_analysis`] and [`run_power_flow_analysis_from_ybus`] are
/// unchanged and remain the shortest path to an ordinary Newton solve;
/// `PowerFlowOptions::default()` here reproduces exactly what they do.
pub fn run_power_flow(
    mut buses: Vec<Bus>,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    opts: PowerFlowOptions,
) -> PowerFlowReport {
    match opts.method {
        PowerFlowMethod::Dc => {
            let solution = dc_power_flow(&mut buses, lines, transformers, opts.dc);
            let status = if solution.islands.iter().any(|i| i.status == DcIslandStatus::Singular) {
                SolveStatus::Singular
            } else {
                SolveStatus::Converged
            };
            let mut report = PowerFlowReport::empty(buses, status);
            report.dc = Some(solution);
            report
        }
        PowerFlowMethod::LinearImpedance => {
            let ybus = finished_ybus(buses.len(), lines, transformers, shunts);
            let solution = linear_power_flow(&mut buses, &ybus);
            let status =
                if solution.islands.iter().any(|i| i.status == LinearIslandStatus::Singular) {
                    SolveStatus::Singular
                } else {
                    SolveStatus::Converged
                };
            let mut report = PowerFlowReport::empty(buses, status);
            report.linear = Some(solution);
            report
        }
        PowerFlowMethod::NewtonRaphson => {
            let ybus = finished_ybus(buses.len(), lines, transformers, shunts);
            match opts.init {
                PowerFlowInit::Flat => {}
                PowerFlowInit::LinearImpedance => linear_initial_guess(&mut buses, &ybus),
                // Seed angles and *only* angles.
                //
                // `dc_power_flow` is a solver, not an initializer: it also
                // normalizes PQ magnitudes to its own |V| = 1 assumption, and
                // it runs `network::mark_unreferenced_islands`, which rewrites
                // a sourceless island's buses to `Slack` with zero injection.
                // Both are right for a DC solve and wrong here — Newton has
                // its own classification pass, and letting the initializer
                // pre-empt it changes the island statuses that pass reports
                // (a sourceless island comes back `AmbiguousReferenceBus`, or
                // for a single bus, silently `Converged`). See
                // `network::linear_initial_guess`, which states the same
                // invariant for the other initializer.
                //
                // So everything but `voltage_ang` is restored afterwards.
                PowerFlowInit::Dc => {
                    let saved: Vec<(types::BusType, f64, f64, f64)> = buses
                        .iter()
                        .map(|b| (b.bus_type, b.p_spec, b.q_spec, b.voltage_mag))
                        .collect();
                    dc_power_flow(&mut buses, lines, transformers, opts.dc);
                    for (bus, (bus_type, p_spec, q_spec, voltage_mag)) in
                        buses.iter_mut().zip(saved)
                    {
                        bus.bus_type = bus_type;
                        bus.p_spec = p_spec;
                        bus.q_spec = q_spec;
                        bus.voltage_mag = voltage_mag;
                    }
                }
            }

            let mut solver = PersistentSolver::new(opts.backend);
            let (islands, stats) = if opts.enforce_q_limits {
                solver::newton_raphson_enforcing_q_limits_with_stats(
                    &mut buses,
                    &ybus,
                    opts.tol,
                    opts.max_iter,
                    opts.backend,
                    opts.max_outer_iter,
                )
            } else {
                solver.solve_with_stats(&mut buses, &ybus, opts.tol, opts.max_iter)
            };
            PowerFlowReport { buses, islands, stats, dc: None, linear: None }
        }
    }
}

fn finished_ybus(
    n: usize,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
) -> network::YBusSparse {
    let mut ybus = build_ybus(n, lines, transformers);
    stamp_shunts(&mut ybus, shunts);
    ybus.finish()
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
    PowerFlowReport { buses, islands, stats, dc: None, linear: None }
}
