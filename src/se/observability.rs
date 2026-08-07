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
//! The factorization itself is `faer`'s
//! (`linalg::cholesky::llt_pivoting`) — blocked and vectorized, rather than the
//! naive triple loop this module originally carried.
//!
//! The rank *decision*, however, is made here rather than taken from faer's
//! `PivLltInfo::rank`. faer stops at `eps · n · max_diag`, i.e. around 1e-16
//! relative: the threshold for "numerically not positive definite".
//! Observability wants a much looser one. A grid whose measurements determine a
//! direction only barely — a weak coupling that survives at 1e-12 of the
//! leading pivot — is unobservable for every practical purpose, and reporting
//! it as observable would be a worse answer than reporting it as not. So the
//! factor's diagonal is thresholded here at [`RANK_TOLERANCE`], relative to the
//! largest pivot, and faer's own rank is an upper bound on the result.
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

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt_pivoting::factor::{
    cholesky_in_place, cholesky_in_place_scratch,
};
use faer::{Mat, Par};

use crate::measurement::Measurement;
use crate::types::Bus;

use super::jacobian::{measurement_jacobian, StateLayout};
use super::SeNetwork;

/// Largest state dimension [`analyze`] will densify, chosen so the gain matrix
/// stays under ~32 MB (`2000² × 8` bytes).
pub const DENSE_LIMIT: usize = 2000;

/// Relative threshold below which a pivot counts as zero.
///
/// Applied to the ratio of a pivot to the largest one, so it is scale-free —
/// which matters because `G`'s entries carry measurement weights that routinely
/// span several orders of magnitude, and an absolute threshold would call a
/// grid watched entirely by high-sigma sensors unobservable purely for having
/// small weights.
///
/// Deliberately far looser than the `eps · n` faer uses internally: a direction
/// determined only at the 1e-12 level is not usefully determined at all.
pub const RANK_TOLERANCE: f64 = 1e-10;

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
    /// A global-angle current sensor is present with nothing to measure its
    /// angle against.
    ///
    /// Such a sensor reports `I = i·e^{jθ}` with `θ` absolute, which is only
    /// meaningful if something else fixes what that reference *is*. With no
    /// voltage angle in the set, [`StateLayout`] pins a reference bus instead —
    /// and that pin is then a false constraint, rotating the estimate away from
    /// the angle the sensor actually measured. The rank check will not catch it,
    /// because the system is perfectly determined; it is determined to the wrong
    /// thing.
    ///
    /// power-grid-model refuses to run at all here, raising `NotObservableError`
    /// with "Global angle current sensors require at least one voltage angle
    /// measurement as a reference point". gridoxide reports rather than refuses,
    /// on the grounds that this analysis is the place that says what is wrong
    /// and the estimator's job is to answer the question it was asked.
    pub global_current_without_angle_reference: bool,
}

impl ObservabilityReport {
    /// Whether every state variable is determined.
    ///
    /// Deliberately does *not* fold in
    /// [`global_current_without_angle_reference`](Self::global_current_without_angle_reference):
    /// that state is fully determined, just to the wrong reference, and calling
    /// it unobservable would misname it.
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

/// Rank of a symmetric positive semidefinite matrix, and the pivot order.
///
/// Returns `(rank, permutation)`: the first `rank` entries of the permutation
/// are the columns that were eliminated with a pivot above threshold, and the
/// tail is the deficient block — the unknowns the measurements do not
/// determine.
///
/// The factorization is faer's blocked pivoted `LLᵀ`; the rank decision is this
/// module's, for the reason given in the module comment. `None` means the
/// factorization itself failed (a NaN pivot), which for a gain matrix means the
/// inputs were already corrupt rather than merely rank-deficient.
fn pivoted_cholesky_rank(mut a: Mat<f64>, tol: f64) -> Option<(usize, Vec<usize>)> {
    let n = a.nrows();
    if n == 0 {
        return Some((0, Vec::new()));
    }

    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    // `MemBuffer` rather than a hand-rolled byte vector: faer's scratch is
    // allocated as `f64`, and a plain `Vec<u8>` is not guaranteed to be aligned
    // for that, so the stack silently loses bytes to alignment and then runs
    // out.
    let mut buffer = MemBuffer::new(cholesky_in_place_scratch::<usize, f64>(
        n,
        Par::Seq,
        Default::default(),
    ));
    let stack = MemStack::new(&mut buffer);

    let (info, _) = cholesky_in_place(
        a.as_mut(),
        &mut perm,
        &mut perm_inv,
        Par::Seq,
        stack,
        Default::default(),
    )
    .ok()?;

    // faer stopped either at full rank or at its own eps-level threshold; apply
    // the looser observability threshold to the pivots it did compute. The
    // diagonal holds L's entries, so a pivot is the square of its diagonal.
    let pivots: Vec<f64> = (0..info.rank).map(|k| a[(k, k)] * a[(k, k)]).collect();
    let largest = pivots.iter().copied().fold(0.0f64, f64::max);
    let rank = if largest <= 0.0 {
        0
    } else {
        pivots.iter().take_while(|&&p| p > tol * largest).count()
    };

    Some((rank, perm))
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

    // A global-angle current sensor supplies an absolute phase; a voltage angle
    // is what gives that phase a meaning. `StateLayout` pins a reference bus
    // when no voltage angle exists, which would silently contradict the sensor.
    let has_global_current = measurements.iter().any(|m| {
        matches!(
            m.target,
            crate::measurement::Target::BranchTerminalCurrent {
                frame: crate::measurement::AngleFrame::Global,
                ..
            }
        ) && m.weight() > 0.0
    });
    let has_voltage_angle = measurements
        .iter()
        .any(|m| m.kind == crate::measurement::MeasurementKind::VoltageAngle && m.weight() > 0.0);
    let global_current_without_angle_reference = has_global_current && !has_voltage_angle;

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
            global_current_without_angle_reference,
        };
    }

    let mut g = Mat::<f64>::zeros(n, n);
    for (row, m) in rows.iter().zip(measurements) {
        let w = m.weight();
        if !w.is_finite() || w == 0.0 {
            continue;
        }
        for &(i, hi) in row {
            for &(j, hj) in row {
                g[(i, j)] += w * hi * hj;
            }
        }
    }

    let Some((rank, perm)) = pivoted_cholesky_rank(g, RANK_TOLERANCE) else {
        // A gain matrix that cannot be factorized at all is not a statement
        // about observability; report nothing determined rather than guess.
        return ObservabilityReport {
            n_unknowns: n,
            rank: 0,
            structurally_unmeasured,
            unobservable: (0..n).map(|c| describe(layout, c)).collect(),
            skipped_numerical: false,
            global_current_without_angle_reference,
        };
    };
    let unobservable = perm[rank..].iter().map(|&c| describe(layout, c)).collect();

    ObservabilityReport {
        n_unknowns: n,
        rank,
        structurally_unmeasured,
        unobservable,
        skipped_numerical: false,
        global_current_without_angle_reference,
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

    /// Builds a dense matrix from rows, for the kernel tests.
    fn mat(rows: &[&[f64]]) -> Mat<f64> {
        let n = rows.len();
        let mut m = Mat::<f64>::zeros(n, n);
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                m[(i, j)] = v;
            }
        }
        m
    }

    #[test]
    fn identity_is_full_rank() {
        let (rank, _) = pivoted_cholesky_rank(mat(&[&[1.0, 0.0], &[0.0, 1.0]]), RANK_TOLERANCE)
            .expect("factorizable");
        assert_eq!(rank, 2);
    }

    /// A rank-1 matrix built as `v vᵀ` must be detected as rank 1, and the
    /// pivot order must put the dependent column last.
    #[test]
    fn rank_one_outer_product_is_rank_one() {
        let v = [1.0, 2.0, -1.0];
        let rows: Vec<Vec<f64>> = (0..3)
            .map(|i| (0..3).map(|j| v[i] * v[j]).collect())
            .collect();
        let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let (rank, _) = pivoted_cholesky_rank(mat(&refs), RANK_TOLERANCE).expect("factorizable");
        assert_eq!(rank, 1, "v v^T has rank 1");
    }

    /// The tolerance is relative, so a uniformly tiny matrix is still full rank
    /// — otherwise a grid measured entirely by high-sigma sensors would be
    /// reported unobservable purely because its weights are small.
    #[test]
    fn rank_detection_is_scale_free() {
        let (rank, _) = pivoted_cholesky_rank(mat(&[&[1e-12, 0.0], &[0.0, 1e-12]]), RANK_TOLERANCE)
            .expect("factorizable");
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

    /// The case structural detection cannot see: every column is touched, and
    /// the state is still not determined.
    ///
    /// Two measurements at one branch terminal reach all three unknowns of this
    /// network — the near bus's angle and both magnitudes — so nothing is
    /// structurally missing and the estimator's own mask would report a clean
    /// bill. But two numbers cannot fix three unknowns, and only the rank
    /// computation says so. This is the capability the module exists for, on a
    /// network rather than on a hand-built matrix.
    #[test]
    fn structurally_complete_but_rank_deficient_is_caught() {
        let (net, buses) = crate::se::tests::two_bus_net();
        let measurements = vec![
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
        let report = analyze(&measurements, &buses, &net, &layout);

        assert_eq!(report.n_unknowns, 3);
        assert!(
            report.structurally_unmeasured.is_empty(),
            "every unknown is reached by a measurement, so structure alone sees no problem: {report:?}"
        );
        assert_eq!(report.rank, 2, "two measurements cannot determine three unknowns");
        assert!(!report.is_observable());
        assert_eq!(report.unobservable.len(), 1);
    }

    /// Redundant sensors add confidence, not rank.
    ///
    /// A third measurement duplicating the first leaves the rank where it was —
    /// the naive check of "enough measurements for the unknowns" would pass here
    /// and be wrong.
    #[test]
    fn a_duplicate_measurement_adds_no_rank() {
        let (net, buses) = crate::se::tests::two_bus_net();
        let flow = m(
            MeasurementKind::ActivePower,
            Target::BranchTerminal { branch: 0, terminal: Terminal::From },
        );
        let measurements = vec![
            flow,
            m(
                MeasurementKind::ReactivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
            ),
            // Same quantity, same terminal, independently reported.
            flow,
        ];
        let layout = StateLayout::new(&buses, &measurements, &net);
        let report = analyze(&measurements, &buses, &net, &layout);

        assert_eq!(measurements.len(), report.n_unknowns, "as many rows as unknowns");
        assert_eq!(report.rank, 2, "but only two of them are independent");
        assert!(!report.is_observable(), "{report:?}");
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
