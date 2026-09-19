//! The constant-admittance linearization: replace every load by the
//! admittance that would draw its rated power at `|V| = 1`, and what is left
//! is a linear circuit.
//!
//! \\[ S_i = U_i \overline{I_i} = -U_i \overline{U_i Y_{load}}
//!    \implies Y_{load} = -\overline{S_i} \quad \text{at } |U| = 1 \\]
//!
//! Folding that into the Y-bus diagonal turns the power-flow problem into one
//! complex linear system `Y' U = I`, solvable in a single factorization with
//! no iteration and no convergence to fail. Unlike the DC approximation in
//! [`btheta`](super::btheta) it keeps resistance and produces voltage
//! *magnitudes*, which is what makes it the useful linearization on
//! distribution networks — but it is only exact when every load really is
//! constant-impedance, and it degrades as loading moves away from nominal.
//!
//! This is mathematically power-grid-model's `CalculationMethod.linear`
//! (`power_grid_model/math_solver/linear_pf_solver.hpp`), down to the
//! `Y_load = -conj(S_base)` step. gridoxide has had the algorithm since it
//! was written as Newton-Raphson's warm start, mirroring PGM's own
//! `NewtonRaphsonPFSolver::initialize_derived_solver`; this module is that
//! code promoted to a solver in its own right, with the island handling the
//! warm-start path never needed.
//!
//! # What counts as an unknown
//!
//! Only `PQ` buses. `PV` buses are held at their present voltage, exactly as
//! `Slack` buses are — the method has no mechanism for "magnitude fixed,
//! angle free", since that constraint is not linear in `U`. On PGM data this
//! is free (PGM has no PV buses: sources are slacks), and it is why the
//! comparison with PGM is exact. On data that does carry PV buses, treating
//! them as fully fixed is an approximation on top of the linearization, and
//! the reason this mode is offered alongside Newton-Raphson rather than
//! instead of it.

use num_complex::Complex;

use crate::network::{
    classify, connected_components, effective_injection, mark_unreferenced_islands, Verdict,
    YBusSparse,
};
use crate::sparse;
use crate::types::{Bus, BusType};

/// How one connected component's constant-admittance solve turned out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearIslandStatus {
    /// Solved; this island's `PQ` bus voltages satisfy its linearized
    /// equations exactly.
    Solved,
    /// The reduced system was singular, so this island's voltages were left
    /// at whatever the caller supplied.
    Singular,
    /// No `Slack` bus, so there is nothing to reference the solve against.
    /// Pinned to `V = 0` by
    /// [`network::mark_unreferenced_islands`](crate::network).
    NoReferenceBus,
    /// Two or more `Slack` buses. Unlike the Newton path — which leaves such
    /// a component live and warns that the result may satisfy neither
    /// reference — this solves it, because the system stays linear and well
    /// posed: each fixed voltage simply contributes its own term to the
    /// right-hand side. The result is the true solution of the circuit *as
    /// described*, which is a meaningful answer to an arguably ill-posed
    /// question.
    SolvedMultipleReferences,
}

/// One connected component's constant-admittance outcome.
#[derive(Clone, Debug)]
pub struct LinearIslandReport {
    pub bus_indices: Vec<usize>,
    pub slack_indices: Vec<usize>,
    pub status: LinearIslandStatus,
}

/// The result of a [`linear_power_flow`] solve. Voltages are written back
/// into the `buses` slice; the per-island breakdown lives here.
#[derive(Clone, Debug)]
pub struct LinearReport {
    pub islands: Vec<LinearIslandReport>,
}

/// Solves one reduced constant-admittance system over `unknown_idx`, writing
/// `voltage_mag`/`voltage_ang` back for those buses only.
///
/// Every bus not in `unknown_idx` is treated as a fixed voltage source at its
/// present `voltage_mag`/`voltage_ang` and moves to the right-hand side.
/// Returns `false` if the reduced system is singular, in which case no bus is
/// modified — leaving the caller's starting values intact is what makes this
/// safe to use as a warm start.
///
/// Walks each unknown bus's actual admittance neighbours from the sparse
/// Y-bus's row structure rather than the full unknown×unknown cross product,
/// so the assembly cost is proportional to the number of branches, not to the
/// square of the number of buses.
pub(crate) fn solve_constant_admittance(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    unknown_idx: &[usize],
) -> bool {
    let m = unknown_idx.len();
    if m == 0 {
        return true;
    }

    let mut reduced_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (r, &i) in unknown_idx.iter().enumerate() {
        reduced_pos[i] = Some(r);
    }

    let mut triplets: Vec<(usize, usize, Complex<f64>)> = Vec::new();
    let mut rhs = vec![Complex::new(0.0, 0.0); m];
    for (r, &i) in unknown_idx.iter().enumerate() {
        let (p, q) = effective_injection(&buses[i]);
        let y_load = -Complex::new(p, q).conj();
        let mut diag_seen = false;
        for &(j, y_ij) in ybus.row(i) {
            let y_ij = if j == i {
                diag_seen = true;
                y_ij + y_load
            } else {
                y_ij
            };
            match reduced_pos[j] {
                Some(c) => triplets.push((r, c, y_ij)),
                None => {
                    let u_j = Complex::from_polar(buses[j].voltage_mag, buses[j].voltage_ang);
                    rhs[r] -= y_ij * u_j;
                }
            }
        }
        if !diag_seen {
            triplets.push((r, r, y_load));
        }
    }

    match sparse::solve_complex(m, &triplets, &rhs) {
        Some(sol) => {
            for (r, &i) in unknown_idx.iter().enumerate() {
                buses[i].voltage_mag = sol[r].norm();
                buses[i].voltage_ang = sol[r].arg();
            }
            true
        }
        None => false,
    }
}

/// Runs the constant-admittance linearization as a standalone power flow.
///
/// Partitions into connected components first, so a sourceless island is
/// pinned rather than solved and a singular island is attributable to itself
/// — the same treatment
/// [`btheta::dc_power_flow`](super::btheta::dc_power_flow) gives, and for the
/// same reason: components are decoupled in the Y-bus, so solving them
/// separately costs nothing and reports more.
pub fn linear_power_flow(buses: &mut [Bus], ybus: &YBusSparse) -> LinearReport {
    let components = connected_components(ybus);
    let classified = classify(buses, &components);
    mark_unreferenced_islands(buses, &classified);

    let mut islands = Vec::with_capacity(classified.len());
    for c in &classified {
        if matches!(c.verdict, Verdict::NoReferenceBus) {
            islands.push(LinearIslandReport {
                bus_indices: c.bus_indices.clone(),
                slack_indices: c.slack_indices.clone(),
                status: LinearIslandStatus::NoReferenceBus,
            });
            continue;
        }

        let unknown_idx: Vec<usize> = c
            .bus_indices
            .iter()
            .copied()
            .filter(|&i| buses[i].bus_type == BusType::PQ)
            .collect();
        let ok = solve_constant_admittance(buses, ybus, &unknown_idx);

        let status = if !ok {
            LinearIslandStatus::Singular
        } else if c.slack_indices.len() > 1 {
            LinearIslandStatus::SolvedMultipleReferences
        } else {
            LinearIslandStatus::Solved
        };
        islands.push(LinearIslandReport {
            bus_indices: c.bus_indices.clone(),
            slack_indices: c.slack_indices.clone(),
            status,
        });
    }

    LinearReport { islands }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::build_ybus;
    use crate::types::Line;

    fn bus(idx: usize, bus_type: BusType, p_spec: f64, q_spec: f64) -> Bus {
        Bus {
            idx,
            bus_type,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec,
            q_spec,
            q_min: 0.0,
            q_max: 0.0,
            u_rated: 0.0,
            zip_terms: Vec::new(),
        }
    }

    /// The defining property of the method: a load that really *is* constant
    /// impedance is solved exactly, so the resulting voltages reproduce the
    /// specified power to round-off. Any other load model would leave an
    /// error here.
    #[test]
    fn a_constant_impedance_load_is_reproduced_exactly() {
        let mut buses = vec![bus(0, BusType::Slack, 0.0, 0.0), bus(1, BusType::PQ, -0.2, -0.1)];
        let lines = vec![Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 }];
        let ybus = build_ybus(2, &lines, &[]).finish();
        let report = linear_power_flow(&mut buses, &ybus);
        assert_eq!(report.islands[0].status, LinearIslandStatus::Solved);

        // The model replaced the load by y = -conj(S); at the solved voltage
        // it therefore draws S·|U|², which is what we recover here.
        let u: Vec<Complex<f64>> = buses
            .iter()
            .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
            .collect();
        let i = ybus.mul_vec(&u);
        let s1 = u[1] * i[1].conj();
        let expected = Complex::new(-0.2, -0.1) * buses[1].voltage_mag.powi(2);
        assert!((s1 - expected).norm() < 1e-12, "{s1} vs {expected}");
    }

    /// A load draws current, so the bus feeding it must sit below the source
    /// — the sanity check that the sign of `Y_load` is right way round.
    #[test]
    fn load_depresses_voltage_and_lags_the_source() {
        let mut buses = vec![bus(0, BusType::Slack, 0.0, 0.0), bus(1, BusType::PQ, -0.5, -0.2)];
        let lines = vec![Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 }];
        let ybus = build_ybus(2, &lines, &[]).finish();
        linear_power_flow(&mut buses, &ybus);
        assert!(buses[1].voltage_mag < 1.0, "{}", buses[1].voltage_mag);
        assert!(buses[1].voltage_ang < 0.0, "{}", buses[1].voltage_ang);
    }

    /// Sourceless islands are pinned, not invented — same contract as the AC
    /// and DC paths.
    #[test]
    fn sourceless_island_is_reported_not_invented() {
        let mut buses = vec![
            bus(0, BusType::Slack, 0.0, 0.0),
            bus(1, BusType::PQ, -0.2, 0.0),
            bus(2, BusType::PQ, -0.1, 0.0),
            bus(3, BusType::PQ, 0.1, 0.0),
        ];
        let lines = vec![
            Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 },
            Line { from: 2, to: 3, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 },
        ];
        let ybus = build_ybus(4, &lines, &[]).finish();
        let report = linear_power_flow(&mut buses, &ybus);

        assert_eq!(report.islands.len(), 2);
        assert_eq!(report.islands[0].status, LinearIslandStatus::Solved);
        assert_eq!(report.islands[1].status, LinearIslandStatus::NoReferenceBus);
        assert_eq!(buses[2].voltage_mag, 0.0);
    }

    /// `linear_power_flow` must agree bus-for-bus with the warm-start entry
    /// point it was extracted from, on a network where the two see the same
    /// unknown set. This is what pins the promotion as a refactor rather than
    /// a reimplementation.
    #[test]
    fn agrees_with_the_warm_start_entry_point() {
        let lines = vec![
            Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.03, g_shunt: 0.0 },
            Line { from: 1, to: 2, r: 0.04, x: 0.10, b_shunt: 0.02, g_shunt: 0.0 },
        ];
        let ybus = build_ybus(3, &lines, &[]).finish();

        let make = || {
            vec![
                bus(0, BusType::Slack, 0.0, 0.0),
                bus(1, BusType::PQ, -0.3, -0.1),
                bus(2, BusType::PQ, -0.2, -0.05),
            ]
        };

        let mut via_mode = make();
        linear_power_flow(&mut via_mode, &ybus);

        let mut via_warm_start = make();
        crate::network::linear_initial_guess(&mut via_warm_start, &ybus);

        for i in 0..3 {
            assert!(
                (via_mode[i].voltage_mag - via_warm_start[i].voltage_mag).abs() < 1e-15
                    && (via_mode[i].voltage_ang - via_warm_start[i].voltage_ang).abs() < 1e-15,
                "bus {i} disagrees"
            );
        }
    }
}
