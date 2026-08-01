//! Observability analysis: which parts of the state the measurements actually
//! determine.
//!
//! An unobservable system is not a numerical accident, and treating it as one
//! produces the worst diagnostic in the library: `SeStatus::Singular`, which
//! says a factorization failed and nothing about why. The useful answer names
//! the unknowns nobody is watching.
//!
//! # Two kinds of unobservable
//!
//! **Structural.** A column of `H` that is identically zero: no measurement
//! function mentions that unknown at all. [`super::nr::estimate`] already finds
//! these and pins them, because it has to — they would make the gain matrix
//! singular on their own. In gridoxide they are usually the virtual slack buses
//! synthesized per source.
//!
//! **Numerical.** A column that is present but linearly dependent on the
//! others. Two branch-flow measurements on a radial feeder with no injection
//! measurement between them constrain the same combination of angles twice and
//! leave another combination free. Nothing about the sparsity pattern gives this
//! away; it takes a rank computation.
//!
//! # Why pivoted Cholesky
//!
//! `G = HᵀWH` is symmetric positive *semi*definite by construction, and for
//! that class the standard rank-revealing factorization is a Cholesky with
//! symmetric diagonal pivoting: pivots are taken largest-first, and the
//! factorization stops when the largest remaining diagonal falls below a
//! tolerance. The number of steps taken is the rank, and the unknowns left in
//! the trailing block are the ones the measurements do not pin down.
//!
//! The alternative — reading zero pivots out of the LU that
//! [`crate::solver::LinearSolver`] already computes — was the plan's first
//! choice, but none of the five backends expose their pivots (`faer`'s
//! `ColPivQr` has no rank accessor, and `klu_native::factor` returns only
//! `Option<Numeric>`). Widening the trait would force all five to implement
//! something only this module wants.
//!
//! The cost is that this densifies `G`, so it is `O(n³)` time and `O(n²)`
//! memory. That is fine for an analysis pass on the grids this is useful for and
//! wrong for a large one, so [`analyze`] refuses above [`DENSE_LIMIT`] rather
//! than quietly allocating gigabytes. A sparse rank-revealing method is the
//! follow-up.

use crate::measurement::Measurement;
use crate::types::Bus;

use super::jacobian::{measurement_jacobian, StateLayout};
use super::SeNetwork;

/// Largest state dimension [`analyze`] will densify, chosen so the gain matrix
/// stays under ~32 MB (`2000² × 8` bytes).
pub const DENSE_LIMIT: usize = 2000;

/// Which of a bus's two unknowns an [`Unknown`] refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    Angle,
    Magnitude,
}

/// One state variable, named in terms of the grid rather than of the solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unknown {
    pub bus: usize,
    pub quantity: Quantity,
}

/// What the measurements do and do not determine.
#[derive(Clone, Debug)]
pub struct ObservabilityReport {
    pub n_unknowns: usize,
    /// Rank of the gain matrix. Equal to `n_unknowns` exactly when the state is
    /// fully observable.
    pub rank: usize,
    /// Unknowns no measurement function mentions at all.
    pub structurally_unmeasured: Vec<Unknown>,
    /// Unknowns left undetermined once linear dependence is accounted for.
    ///
    /// A superset of [`structurally_unmeasured`](Self::structurally_unmeasured)
    /// in effect — a zero column is also rank-deficient — though the specific
    /// variables reported can differ, since which member of a dependent group
    /// gets blamed depends on the pivot order. The *count* is what is
    /// meaningful; the names are representatives.
    pub unobservable: Vec<Unknown>,
    /// Set when the system was too large to analyze densely, in which case only
    /// the structural half of the report is filled in.
    pub skipped_numerical: bool,
}

impl ObservabilityReport {
    /// Whether every state variable is determined.
    pub fn is_observable(&self) -> bool {
        self.rank == self.n_unknowns && self.structurally_unmeasured.is_empty()
    }
}

/// Names the bus and quantity a state-vector column refers to.
fn describe(layout: &StateLayout, col: usize) -> Unknown {
    let magnitude_start = layout.vmag(0);
    if col >= magnitude_start {
        Unknown { bus: col - magnitude_start, quantity: Quantity::Magnitude }
    } else {
        let bus = (0..layout.n_buses)
            .find(|&b| layout.theta(b) == Some(col))
            .expect("angle column belongs to some bus");
        Unknown { bus, quantity: Quantity::Angle }
    }
}

/// Pivoted Cholesky of a symmetric positive semidefinite matrix.
///
/// Returns `(rank, permutation)`: the first `rank` entries of the permutation
/// are the columns that were successfully eliminated, and the tail is the
/// deficient block.
///
/// `tol` is relative to the largest diagonal entry, so it is scale-free — which
/// matters here because `G`'s entries carry measurement weights that routinely
/// span several orders of magnitude.
fn pivoted_cholesky_rank(mut a: Vec<Vec<f64>>, tol: f64) -> (usize, Vec<usize>) {
    let n = a.len();
    let mut perm: Vec<usize> = (0..n).collect();
    let max_diag = (0..n).map(|i| a[i][i]).fold(0.0f64, f64::max);
    if max_diag <= 0.0 {
        return (0, perm);
    }
    let threshold = tol * max_diag;

    for k in 0..n {
        // Largest remaining diagonal, the standard pivot choice for PSD input.
        let (pivot, value) = (k..n)
            .map(|i| (i, a[i][i]))
            .fold((k, f64::NEG_INFINITY), |best, cur| if cur.1 > best.1 { cur } else { best });
        if value <= threshold {
            return (k, perm);
        }

        if pivot != k {
            a.swap(k, pivot);
            for row in a.iter_mut() {
                row.swap(k, pivot);
            }
            perm.swap(k, pivot);
        }

        let d = a[k][k].sqrt();
        a[k][k] = d;
        for i in k + 1..n {
            a[i][k] /= d;
        }
        for j in k + 1..n {
            for i in j..n {
                let update = a[i][k] * a[j][k];
                a[i][j] -= update;
                a[j][i] = a[i][j];
            }
        }
    }
    (n, perm)
}

/// Analyzes what `measurements` determine about the state at `buses`.
///
/// The analysis is linearization-dependent: `H` is evaluated at the state it is
/// given, so a flat start and a converged estimate can in principle disagree.
/// In practice observability is a structural property and they do not, but the
/// honest reading of a report is "at this operating point".
pub fn analyze(
    measurements: &[Measurement],
    buses: &[Bus],
    net: &SeNetwork,
    layout: &StateLayout,
) -> ObservabilityReport {
    let n = layout.n_unknowns();
    let rows = measurement_jacobian(measurements, buses, net, layout);

    // Structural half: a column no row mentions with a nonzero coefficient.
    // Note this asks for a nonzero *value*, unlike the estimator's mask, which
    // deliberately asks only for structural presence so its sparsity pattern
    // stays fixed. Here there is no pattern to preserve and the stronger
    // question is the useful one.
    let mut touched = vec![false; n];
    for row in &rows {
        for &(c, v) in row {
            if v != 0.0 {
                touched[c] = true;
            }
        }
    }
    let structurally_unmeasured: Vec<Unknown> = (0..n)
        .filter(|&c| !touched[c])
        .map(|c| describe(layout, c))
        .collect();

    if n > DENSE_LIMIT {
        return ObservabilityReport {
            n_unknowns: n,
            rank: n - structurally_unmeasured.len(),
            structurally_unmeasured,
            unobservable: Vec::new(),
            skipped_numerical: true,
        };
    }

    let mut g = vec![vec![0.0; n]; n];
    for (row, m) in rows.iter().zip(measurements) {
        let w = m.weight();
        if !w.is_finite() || w == 0.0 {
            continue;
        }
        for &(i, hi) in row {
            for &(j, hj) in row {
                g[i][j] += w * hi * hj;
            }
        }
    }

    let (rank, perm) = pivoted_cholesky_rank(g, 1e-10);
    let unobservable = perm[rank..].iter().map(|&c| describe(layout, c)).collect();

    ObservabilityReport {
        n_unknowns: n,
        rank,
        structurally_unmeasured,
        unobservable,
        skipped_numerical: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch_flow::Terminal;
    use crate::measurement::{MeasurementKind, Target};

    fn m(kind: MeasurementKind, target: Target) -> Measurement {
        Measurement { kind, target, value: 0.0, sigma: 0.01 }
    }

    #[test]
    fn identity_is_full_rank() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (rank, _) = pivoted_cholesky_rank(a, 1e-10);
        assert_eq!(rank, 2);
    }

    /// A rank-1 matrix built as `v vᵀ` must be detected as rank 1, and the
    /// pivot order must put the dependent column last.
    #[test]
    fn rank_one_outer_product_is_rank_one() {
        let v = [1.0, 2.0, -1.0];
        let a: Vec<Vec<f64>> = (0..3)
            .map(|i| (0..3).map(|j| v[i] * v[j]).collect())
            .collect();
        let (rank, _) = pivoted_cholesky_rank(a, 1e-10);
        assert_eq!(rank, 1, "v v^T has rank 1");
    }

    /// The tolerance is relative, so a uniformly tiny matrix is still full rank
    /// — otherwise a grid measured entirely by high-sigma sensors would be
    /// reported unobservable purely because its weights are small.
    #[test]
    fn rank_detection_is_scale_free() {
        let a = vec![vec![1e-12, 0.0], vec![0.0, 1e-12]];
        let (rank, _) = pivoted_cholesky_rank(a, 1e-10);
        assert_eq!(rank, 2);
    }

    /// Enough measurements to determine the whole state: full rank, nothing
    /// unobservable.
    #[test]
    fn well_measured_system_is_observable() {
        let (net, buses) = crate::se::tests::two_bus_net();
        let measurements = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0)),
            m(MeasurementKind::VoltageMagnitude, Target::Bus(1)),
            m(MeasurementKind::ActivePower, Target::Bus(1)),
            m(MeasurementKind::ReactivePower, Target::Bus(1)),
            m(
                MeasurementKind::ActivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
            ),
        ];
        let layout = StateLayout::new(&buses, &measurements, &net);
        let report = analyze(&measurements, &buses, &net, &layout);
        assert!(
            report.is_observable(),
            "expected observable, got {report:?}"
        );
        assert_eq!(report.rank, report.n_unknowns);
    }

    /// One voltage magnitude cannot determine three unknowns, and the report
    /// should say which — this is the diagnosis that replaces a bare
    /// `Singular`.
    #[test]
    fn under_measured_system_names_what_is_missing() {
        let (net, buses) = crate::se::tests::two_bus_net();
        let measurements = vec![m(MeasurementKind::VoltageMagnitude, Target::Bus(0))];
        let layout = StateLayout::new(&buses, &measurements, &net);
        let report = analyze(&measurements, &buses, &net, &layout);

        assert!(!report.is_observable());
        assert_eq!(report.rank, 1, "only the measured magnitude is determined");
        assert_eq!(report.unobservable.len(), report.n_unknowns - 1);
        // The measured bus's magnitude is the one thing that *is* observable.
        assert!(
            !report
                .unobservable
                .contains(&Unknown { bus: 0, quantity: Quantity::Magnitude }),
            "the measured quantity must not be reported unobservable: {report:?}"
        );
    }

    /// Measuring only powers leaves the global phase free, which is exactly the
    /// invariance `StateLayout` pins a reference for. With the reference pinned
    /// the remaining angles are determined; this checks the pinning actually
    /// buys observability rather than merely making the matrix square.
    #[test]
    fn pinned_reference_makes_angles_observable() {
        let (net, buses) = crate::se::tests::two_bus_net();
        let measurements = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0)),
            m(MeasurementKind::VoltageMagnitude, Target::Bus(1)),
            m(
                MeasurementKind::ActivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
            ),
            m(
                MeasurementKind::ReactivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
            ),
        ];
        let layout = StateLayout::new(&buses, &measurements, &net);
        assert!(layout.angle_ref.is_some(), "no angle measured, so one is pinned");
        let report = analyze(&measurements, &buses, &net, &layout);
        assert!(report.is_observable(), "{report:?}");
    }
}
