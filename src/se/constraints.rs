//! Zero-injection buses as hard equality constraints.
//!
//! A bus with no load and no generator injects exactly nothing. That is not a
//! measurement — there is no sensor, no noise and no uncertainty — it is a
//! property of the network, known exactly.
//!
//! # Why not just weight it heavily
//!
//! The common shortcut is to feed it in as a pseudo-measurement of zero with a
//! very small sigma. It works, and it is why so many estimators are described
//! as ill-conditioned: the weight matrix then spans many orders of magnitude,
//! and `G = HᵀWH` squares that spread. power-grid-model ships fixtures named
//! `ill-conditioned-by-line-meshed` and `ill-conditioned-by-link-meshed` for
//! exactly this failure.
//!
//! Treating the knowledge as a constraint instead keeps the weights physical.
//! The Lagrangian stationarity conditions give the augmented system
//!
//! ```text
//! [ G   Cᵀ ] [ Δx ]   [ HᵀW r ]
//! [ C   0  ] [ λ  ] = [ −c(x) ]
//! ```
//!
//! where `c(x)` collects the injections that must vanish and `C = ∂c/∂x`. It is
//! symmetric indefinite rather than positive definite, which rules out a
//! Cholesky-style solver — but every gridoxide backend is a general sparse LU,
//! so this needs assembly work only, no new solver.
//!
//! # What it costs
//!
//! The system grows by two rows and columns per constrained bus (`P` and `Q`),
//! and `λ` is discarded. The multipliers are not meaningless — they are the
//! sensitivity of the objective to each constraint — but nothing here consumes
//! them yet.

use crate::types::Bus;

use super::jacobian::{Row, StateLayout};
use super::SeNetwork;

/// The zero-injection constraints of a network, in a fixed order.
///
/// Built once per topology: which buses are constrained never changes during an
/// estimate, so neither does the augmented system's sparsity pattern.
pub struct Constraints {
    /// Constrained buses, ascending. Each contributes two rows, `P` then `Q`.
    pub buses: Vec<usize>,
}

impl Constraints {
    /// The constraints a network implies: its zero injections.
    ///
    /// Notably *not* the phase relationship of a synthesized source's three
    /// virtual buses, which looks like free information and is not — see
    /// `tests/se_three_phase_test.rs::magnitudes_alone_leave_the_phase_relationship_undetermined`.
    pub fn new(net: &SeNetwork) -> Self {
        Self::from_flags(&net.constrained_buses())
    }

    /// Zero-injection constraints from per-bus flags.
    pub fn from_flags(zero_injection: &[bool]) -> Self {
        let buses = zero_injection
            .iter()
            .enumerate()
            .filter(|&(_, &z)| z)
            .map(|(i, _)| i)
            .collect();
        Self { buses }
    }

    /// Number of scalar constraints, two per bus.
    pub fn len(&self) -> usize {
        2 * self.buses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buses.is_empty()
    }

    /// `c(x)`: the injections that ought to be zero, and `C`, their Jacobian.
    ///
    /// The rows of `C` are the same bus-injection partials a `P`/`Q` injection
    /// *measurement* would produce — the difference between a constraint and a
    /// measurement here is entirely in how the system consumes them, not in the
    /// mathematics of the row.
    pub fn evaluate(
        &self,
        buses: &[Bus],
        net: &SeNetwork,
        layout: &StateLayout,
    ) -> (Vec<f64>, Vec<Row>) {
        let (p_inj, q_inj) = crate::network::power_injections(buses, &net.ybus);
        let v: Vec<num_complex::Complex<f64>> = buses
            .iter()
            .map(|b| num_complex::Complex::from_polar(b.voltage_mag, b.voltage_ang))
            .collect();
        let mut values = Vec::with_capacity(self.len());
        let mut rows = Vec::with_capacity(self.len());
        let mut scratch = Vec::new();

        for &bus in &self.buses {
            for active in [true, false] {
                values.push(if active { p_inj[bus] } else { q_inj[bus] });
                rows.push(super::jacobian::injection_row(
                    layout, net, &v, bus, active, &mut scratch,
                ));
            }
        }

        (values, rows)
    }
}

/// Assembles the augmented KKT system from an already-built gain matrix.
///
/// `triplets` and `rhs` are `G` and `HᵀWr`, sized `n`; the result is the
/// `(n + m) × (n + m)` system above. `G`'s own triplets are reused rather than
/// rebuilt, so this composes with the unconstrained path instead of duplicating
/// it.
pub fn augment(
    mut triplets: Vec<(usize, usize, f64)>,
    mut rhs: Vec<f64>,
    n: usize,
    constraint_values: &[f64],
    constraint_rows: &[Row],
) -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    rhs.resize(n + constraint_rows.len(), 0.0);

    for (k, (row, &c)) in constraint_rows.iter().zip(constraint_values).enumerate() {
        let lambda = n + k;
        for &(col, value) in row {
            // C in the lower-left block, and its transpose in the upper-right.
            triplets.push((lambda, col, value));
            triplets.push((col, lambda, value));
        }
        // The constraint is c(x) = 0, so the step has to cancel the current
        // violation: C Δx = −c(x).
        rhs[lambda] = -c;
    }
    (triplets, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_flags_means_no_constraints() {
        let c = Constraints::from_flags(&[false, false]);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn each_bus_contributes_two_rows() {
        let c = Constraints::from_flags(&[true, false, true]);
        assert_eq!(c.buses, vec![0, 2]);
        assert_eq!(c.len(), 4);
    }

    /// The augmented block must be symmetric: `C` below, `Cᵀ` to the right, and
    /// a zero block on the diagonal. An asymmetry here would silently solve a
    /// different problem.
    #[test]
    fn augmented_system_is_symmetric() {
        let n = 3;
        let g = vec![(0, 0, 2.0), (1, 1, 3.0), (2, 2, 4.0)];
        let rows = vec![vec![(0, 1.0), (2, -0.5)]];
        let (triplets, rhs) = augment(g, vec![1.0, 2.0, 3.0], n, &[0.25], &rows);

        let size = n + 1;
        let mut dense = vec![vec![0.0; size]; size];
        for (i, j, v) in triplets {
            dense[i][j] += v;
        }
        for i in 0..size {
            for j in 0..size {
                assert!(
                    (dense[i][j] - dense[j][i]).abs() < 1e-12,
                    "asymmetric at ({i},{j})"
                );
            }
        }
        assert_eq!(dense[n][n], 0.0, "the multiplier block must stay zero");
        assert_eq!(rhs[n], -0.25, "the constraint row drives the violation to zero");
    }
}
