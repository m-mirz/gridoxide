//! The optimization layer: a solver-independent problem statement, and
//! whatever backend is available to answer it.
//!
//! This module is the boundary, not the mathematics. It carries no notion of a
//! bus, a generator or a cost curve — those arrive with DC-OPF (see
//! `plans/OPF_PLAN.md`). What lives here is the smallest description of a
//! convex problem that both backends can consume and that the OPF formulation
//! can build without knowing which one will solve it.
//!
//! # Why a boundary at all
//!
//! Because there will be two backends, deliberately. [`highs`] wraps a system
//! HiGHS install; an in-house convex interior-point method follows. They are
//! not redundant:
//!
//! - the in-house one removes the system-install requirement, so it becomes
//!   the portable default;
//! - HiGHS stays as a reference, and **the pair is a validation gate**. On a
//!   convex problem the optimum is unique in objective value — and in the
//!   primal too where the objective is strictly convex — so two independent
//!   solvers *must* agree. A disagreement is a bug in one of them, not a
//!   modelling convention or a different local optimum, which is a far sharper
//!   signal than most cross-tool comparisons give.
//!
//! # Shape of the problem
//!
//! \\[ \min_x\ \tfrac12 x^{\mathsf T} Q x + c^{\mathsf T} x + \text{offset}
//!    \quad\text{s.t.}\quad
//!    \ell_r \le A x \le u_r, \qquad \ell_c \le x \le u_c \\]
//!
//! **Two-sided row bounds** rather than a sense-per-row, because that is one
//! form covering everything the OPF needs without special cases: an equality
//! is `lower == upper` (the \\(B\theta = P\\) balance), a symmetric limit is
//! `-rate ≤ flow ≤ rate`, and a one-sided constraint leaves the other side
//! infinite. It is also HiGHS's own form, so nothing is lost in translation.
//!
//! Matrices are `(row, col, value)` triplets, the shape the rest of this crate
//! already speaks — `sparse::RealFactorization::new`,
//! `network::YBus::into_entries`. Each backend converts to whatever layout it
//! wants internally, where that conversion can be tested in one place.

#[cfg(feature = "opf-highs")]
pub mod highs;
pub mod ac;
pub mod dc;
pub mod ipm;
#[cfg(feature = "opf-ipopt")]
pub mod ipopt;
pub mod nlp;
pub mod model;

/// A convex quadratic program — or a linear one, when [`hessian`] is `None`.
///
/// Every vector is indexed positionally: `col_*` by variable, `row_*` by
/// constraint. Use [`f64::INFINITY`] and its negation for an absent bound.
///
/// [`hessian`]: LinearProgram::hessian
#[derive(Clone, Debug, Default)]
pub struct LinearProgram {
    pub n_vars: usize,
    /// Number of constraint rows. Kept explicitly rather than inferred from
    /// `rows`, so a problem may declare a row that happens to have no
    /// coefficients — which is what a network with an isolated bus produces.
    pub n_rows: usize,

    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
    /// Linear objective coefficients.
    pub col_cost: Vec<f64>,
    /// Constant added to the objective. Carries the `c₀` terms of a
    /// generator cost curve, which change the reported cost but not the
    /// optimum.
    pub offset: f64,

    /// **Lower triangle** of `Q` as `(i, j, value)` with `i >= j`, for the
    /// `½ xᵀQx` term. `None` for a pure LP.
    ///
    /// Lower-triangle-only is the convention every QP solver uses and the one
    /// place a caller is most likely to go wrong: supplying both `(i, j)` and
    /// `(j, i)` double-counts the off-diagonal. [`Self::validate`] rejects an
    /// upper-triangle entry outright rather than letting it be silently
    /// misread.
    pub hessian: Option<Vec<(usize, usize, f64)>>,

    /// Constraint matrix as `(row, col, value)` triplets. Duplicates sum, the
    /// same accumulation semantics `network::YBus` documents.
    pub rows: Vec<(usize, usize, f64)>,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
}

impl LinearProgram {
    /// An unconstrained minimization over `n_vars` free variables, to be filled
    /// in by the caller.
    pub fn new(n_vars: usize) -> Self {
        Self {
            n_vars,
            n_rows: 0,
            col_lower: vec![f64::NEG_INFINITY; n_vars],
            col_upper: vec![f64::INFINITY; n_vars],
            col_cost: vec![0.0; n_vars],
            offset: 0.0,
            hessian: None,
            rows: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        }
    }

    /// Appends one constraint row, returning its index.
    ///
    /// `coefficients` are `(col, value)` pairs for this row only, so a caller
    /// never has to track the global row index while building.
    pub fn add_row(&mut self, coefficients: &[(usize, f64)], lower: f64, upper: f64) -> usize {
        let row = self.n_rows;
        for &(col, value) in coefficients {
            self.rows.push((row, col, value));
        }
        self.row_lower.push(lower);
        self.row_upper.push(upper);
        self.n_rows += 1;
        row
    }

    /// Checks the problem is internally consistent, before a backend sees it.
    ///
    /// Exists because every one of these mistakes is otherwise silent: a short
    /// bound vector reads as a garbage bound, an out-of-range column index
    /// lands on the wrong variable, and an upper-triangle Hessian entry is
    /// simply misinterpreted. Backends call this first so neither has to
    /// re-derive the checks — and so a bug is reported against the *problem*
    /// rather than surfacing as a solver failure.
    pub fn validate(&self) -> Result<(), OpfError> {
        let n = self.n_vars;
        for (name, len) in [
            ("col_lower", self.col_lower.len()),
            ("col_upper", self.col_upper.len()),
            ("col_cost", self.col_cost.len()),
        ] {
            if len != n {
                return Err(OpfError::Shape { what: name, expected: n, got: len });
            }
        }
        for (name, len) in
            [("row_lower", self.row_lower.len()), ("row_upper", self.row_upper.len())]
        {
            if len != self.n_rows {
                return Err(OpfError::Shape { what: name, expected: self.n_rows, got: len });
            }
        }
        for &(row, col, _) in &self.rows {
            if row >= self.n_rows || col >= n {
                return Err(OpfError::Index { row, col });
            }
        }
        if let Some(hessian) = &self.hessian {
            for &(i, j, _) in hessian {
                if i >= n || j >= n {
                    return Err(OpfError::Index { row: i, col: j });
                }
                if j > i {
                    return Err(OpfError::UpperTriangleHessian { row: i, col: j });
                }
            }
        }
        for k in 0..n {
            if self.col_lower[k] > self.col_upper[k] {
                return Err(OpfError::CrossedBounds { index: k, kind: "column" });
            }
        }
        for k in 0..self.n_rows {
            if self.row_lower[k] > self.row_upper[k] {
                return Err(OpfError::CrossedBounds { index: k, kind: "row" });
            }
        }
        Ok(())
    }
}

/// How a solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptStatus {
    /// A point satisfying the optimality conditions was found.
    Optimal,
    /// No point satisfies the constraints.
    Infeasible,
    /// The objective can be driven arbitrarily low within the constraints —
    /// on an OPF this means a missing limit rather than a genuine answer.
    Unbounded,
    /// The solver stopped without deciding: an iteration or time limit, or a
    /// numerical failure. Carries whatever the backend called it, since the
    /// useful detail is backend-specific.
    Other(&'static str),
}

/// A solved (or attempted) problem.
///
/// The vectors are empty unless `status` is [`OptStatus::Optimal`] — there is
/// no meaningful primal or dual at an infeasible point, and returning stale
/// numbers would be worse than returning none.
#[derive(Clone, Debug)]
pub struct Solution {
    pub status: OptStatus,
    /// Objective value, `offset` included.
    pub objective: f64,
    /// Variable values, by column.
    pub primal: Vec<f64>,
    /// `A x` at the solution, by row.
    pub row_activity: Vec<f64>,
    /// Reduced costs, by column.
    pub col_dual: Vec<f64>,
    /// Row duals — the shadow price of each constraint.
    ///
    /// **Sign convention**, which matters more here than anywhere else in the
    /// crate because these become locational marginal prices: for a
    /// *minimization*, `row_dual[k]` is `∂objective/∂b` where `b` is the
    /// binding side of row `k`. So a binding upper limit on a constraint whose
    /// relaxation would *reduce* cost carries a **negative** dual. An inactive
    /// row's dual is zero, by complementary slackness. This is asserted
    /// directly in `tests/opf_highs_test.rs` rather than left to inspection —
    /// a flipped sign would look entirely plausible on every case and be wrong
    /// on all of them.
    pub row_dual: Vec<f64>,
}

impl Solution {
    /// The result of a solve that did not reach an optimum.
    pub fn failed(status: OptStatus) -> Self {
        Self {
            status,
            objective: f64::NAN,
            primal: Vec::new(),
            row_activity: Vec::new(),
            col_dual: Vec::new(),
            row_dual: Vec::new(),
        }
    }

    pub fn is_optimal(&self) -> bool {
        self.status == OptStatus::Optimal
    }
}

/// Something that can answer a [`LinearProgram`].
pub trait Solver {
    /// Solves it. `Err` means the problem could not be *posed* — a malformed
    /// problem or a backend failure; an infeasible or unbounded problem is a
    /// perfectly good `Ok` carrying that [`OptStatus`].
    fn solve(&mut self, problem: &LinearProgram) -> Result<Solution, OpfError>;
}

/// Why a problem could not be posed or solved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpfError {
    Shape { what: &'static str, expected: usize, got: usize },
    Index { row: usize, col: usize },
    UpperTriangleHessian { row: usize, col: usize },
    CrossedBounds { index: usize, kind: &'static str },
    /// The backend rejected the problem or failed internally.
    Backend(String),
}

impl std::fmt::Display for OpfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape { what, expected, got } => {
                write!(f, "{what} has {got} entries, expected {expected}")
            }
            Self::Index { row, col } => {
                write!(f, "entry ({row}, {col}) is outside the problem's dimensions")
            }
            Self::UpperTriangleHessian { row, col } => write!(
                f,
                "Hessian entry ({row}, {col}) is above the diagonal — supply the lower \
                 triangle only, or the off-diagonal term is counted twice"
            ),
            Self::CrossedBounds { index, kind } => {
                write!(f, "{kind} {index} has its lower bound above its upper bound")
            }
            Self::Backend(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for OpfError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_var() -> LinearProgram {
        let mut lp = LinearProgram::new(2);
        lp.col_lower = vec![0.0, 0.0];
        lp.col_upper = vec![10.0, 10.0];
        lp.col_cost = vec![1.0, 1.0];
        lp
    }

    #[test]
    fn add_row_assigns_consecutive_indices_and_keeps_bounds_aligned() {
        let mut lp = two_var();
        assert_eq!(lp.add_row(&[(0, 1.0), (1, 1.0)], 1.0, f64::INFINITY), 0);
        assert_eq!(lp.add_row(&[(0, 1.0)], f64::NEG_INFINITY, 4.0), 1);
        assert_eq!(lp.n_rows, 2);
        assert_eq!(lp.row_lower.len(), 2);
        assert_eq!(lp.rows.len(), 3);
        lp.validate().unwrap();
    }

    /// The check that earns `validate` its keep: an upper-triangle entry is
    /// not an error any solver would report, it is a silently doubled
    /// off-diagonal term.
    #[test]
    fn an_upper_triangle_hessian_entry_is_rejected() {
        let mut lp = two_var();
        lp.hessian = Some(vec![(0, 1, 2.0)]);
        assert_eq!(
            lp.validate(),
            Err(OpfError::UpperTriangleHessian { row: 0, col: 1 })
        );

        lp.hessian = Some(vec![(1, 0, 2.0)]);
        lp.validate().unwrap();
    }

    #[test]
    fn shape_and_index_mistakes_are_caught() {
        let mut lp = two_var();
        lp.col_cost.push(1.0);
        assert!(matches!(lp.validate(), Err(OpfError::Shape { what: "col_cost", .. })));

        let mut lp = two_var();
        lp.add_row(&[(5, 1.0)], 0.0, 1.0);
        assert_eq!(lp.validate(), Err(OpfError::Index { row: 0, col: 5 }));

        let mut lp = two_var();
        lp.col_lower[1] = 20.0;
        assert_eq!(lp.validate(), Err(OpfError::CrossedBounds { index: 1, kind: "column" }));
    }

    #[test]
    fn a_failed_solution_carries_no_stale_numbers() {
        let s = Solution::failed(OptStatus::Infeasible);
        assert!(!s.is_optimal());
        assert!(s.primal.is_empty() && s.row_dual.is_empty());
        assert!(s.objective.is_nan());
    }
}
