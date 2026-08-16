//! The HiGHS backend, against problems small enough to solve on paper.
//!
//! Every expected value here is derived by hand, not taken from a second
//! solver. At this layer there is no oracle and none is needed: these problems
//! have two variables, and what is being tested is the *binding* — that the
//! matrix crossed the FFI in the layout HiGHS expects, that the Hessian
//! convention is right, and that the numbers come back meaning what this crate
//! says they mean.
//!
//! Two checks recur, and they test different things:
//!
//! - the **primal and objective**, which are unambiguous;
//! - **KKT stationarity**, `col_dual = Qx + c − Aᵀ·row_dual`, which pins the
//!   duals' internal consistency without assuming a sign convention.
//!
//! On top of those, one test asserts a specific dual's *sign*, which is the
//! convention itself. That one matters most: these duals become locational
//! marginal prices in DC-OPF, and a flipped sign would look entirely plausible
//! on every case and be wrong on all of them.

use gridoxide::opf::highs::HighsSolver;
use gridoxide::opf::{LinearProgram, OpfError, OptStatus, Solver};

/// Tolerance for quantities a simplex solve pins exactly — an LP optimum sits
/// on a vertex, so it is as precise as the arithmetic.
const TOL: f64 = 1e-9;

/// Tolerance for a QP, and it has to be looser than [`TOL`] for a real reason
/// rather than convenience.
///
/// HiGHS solves a QP by an interior-point method, which converges
/// asymptotically instead of landing on a vertex, so the answer is only ever
/// as precise as the solver's convergence permits. The QP tests below already
/// tighten HiGHS as far as it allows (`set_tolerance(1e-10, 1e-10)`), and even
/// then the observed primal error is ~6e-8 with a stationarity residual of
/// ~1.3e-7 — the Hessian multiplies the primal error through.
///
/// `1e-6` sits just above that floor and still leaves the tests their teeth:
/// the three competing readings of the Hessian convention put `x0` at 1.0,
/// 1.5, or nowhere at all, so they differ by ~0.5 — five orders of magnitude
/// above this bound.
const QP_TOL: f64 = 1e-6;

#[track_caller]
fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < TOL, "{what}: got {got}, want {want}");
}

#[track_caller]
fn close_qp(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < QP_TOL, "{what}: got {got}, want {want}");
}

/// A solver tightened as far as HiGHS allows, for the QP tests.
fn tight_solver() -> HighsSolver {
    let solver = solver();
    solver.set_tolerance(1e-10, 1e-10).expect("HiGHS accepts 1e-10");
    solver
}

fn solver() -> HighsSolver {
    HighsSolver::new().expect("HiGHS should be available — the feature requires it")
}

/// Checks `col_dual = Qx + c − Aᵀ·row_dual` at the returned point.
///
/// This is stationarity of the Lagrangian, and it holds whatever sign
/// convention the backend uses for the duals — so it catches a transposed
/// matrix, a misread Hessian or a mismatched vector without hard-coding what
/// HiGHS reports. The sign convention itself is pinned separately, below.
#[track_caller]
fn assert_stationarity(lp: &LinearProgram, solution: &gridoxide::opf::Solution, tol: f64) {
    let n = lp.n_vars;
    let mut lhs = lp.col_cost.clone();

    if let Some(hessian) = &lp.hessian {
        // Lower triangle: the diagonal contributes once, an off-diagonal to
        // both of its variables.
        for &(i, j, v) in hessian {
            lhs[i] += v * solution.primal[j];
            if i != j {
                lhs[j] += v * solution.primal[i];
            }
        }
    }
    for &(row, col, v) in &lp.rows {
        lhs[col] -= v * solution.row_dual[row];
    }

    for k in 0..n {
        assert!(
            (solution.col_dual[k] - lhs[k]).abs() < tol,
            "stationarity at column {k}: col_dual {} vs Qx + c - A'y {}",
            solution.col_dual[k],
            lhs[k]
        );
    }
}

/// A linear program with a unique vertex optimum.
///
/// ```text
/// min  x0 + 2*x1
/// s.t. x0 + x1 >= 4            (binding)
///      0 <= x0 <= 3            (binding, at the upper bound)
///      x1 >= 0
/// ```
///
/// `x1` is the expensive variable, so `x0` is pushed to its ceiling of 3 and
/// `x1` takes the remaining 1. Optimum `(3, 1)`, objective 5.
#[test]
fn lp_with_a_unique_vertex_optimum() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![0.0, 0.0];
    lp.col_upper = vec![3.0, f64::INFINITY];
    lp.col_cost = vec![1.0, 2.0];
    lp.add_row(&[(0, 1.0), (1, 1.0)], 4.0, f64::INFINITY);

    let solution = solver().solve(&lp).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal);

    close(solution.primal[0], 3.0, "x0");
    close(solution.primal[1], 1.0, "x1");
    close(solution.objective, 5.0, "objective");
    close(solution.row_activity[0], 4.0, "row activity");

    // `x1` sits strictly between its bounds, so its reduced cost is zero and
    // the row dual is forced to `c1 = 2`. `x0` is at its ceiling with reduced
    // cost `c0 − λ = 1 − 2 = −1`.
    close(solution.row_dual[0], 2.0, "row dual");
    close(solution.col_dual[0], -1.0, "reduced cost of x0");
    close(solution.col_dual[1], 0.0, "reduced cost of x1");
    assert_stationarity(&lp, &solution, TOL);
}

/// The same problem with a second, slack row. Its dual must be exactly zero —
/// complementary slackness, and the cheapest check that a non-binding row is
/// not silently contributing to the answer.
#[test]
fn a_slack_row_has_a_zero_dual() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![0.0, 0.0];
    lp.col_upper = vec![3.0, f64::INFINITY];
    lp.col_cost = vec![1.0, 2.0];
    lp.add_row(&[(0, 1.0), (1, 1.0)], 4.0, f64::INFINITY);
    lp.add_row(&[(0, 1.0), (1, 2.0)], f64::NEG_INFINITY, 100.0);

    let solution = solver().solve(&lp).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal);

    // Unchanged from the one-row case.
    close(solution.primal[0], 3.0, "x0");
    close(solution.primal[1], 1.0, "x1");
    close(solution.objective, 5.0, "objective");

    close(solution.row_activity[1], 5.0, "slack row activity");
    assert_eq!(solution.row_dual[1], 0.0, "a slack row must carry no price");
    assert_stationarity(&lp, &solution, TOL);
}

/// A strictly convex QP with an **off-diagonal** Hessian term, minimized in
/// the interior.
///
/// ```text
/// min  1/2 x'Qx − 3*x0 − 3*x1,    Q = [[2, 1], [1, 2]]
/// ```
///
/// Stationarity gives `Qx = [3, 3]`, so `x = (1, 1)` and the objective is
/// `3 − 6 = −3`.
///
/// The off-diagonal is the point. `LinearProgram::hessian` takes the lower
/// triangle only, and the three plausible readings are all distinguishable
/// here: counting `(1,0)` once gives `(1, 1)`; counting it twice makes `Q`
/// singular; dropping it gives `(1.5, 1.5)` and objective `−4.5`. A Hessian
/// with a zero off-diagonal would pass under all three.
#[test]
fn qp_with_an_off_diagonal_hessian_and_an_interior_optimum() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![-10.0, -10.0];
    lp.col_upper = vec![10.0, 10.0];
    lp.col_cost = vec![-3.0, -3.0];
    lp.hessian = Some(vec![(0, 0, 2.0), (1, 0, 1.0), (1, 1, 2.0)]);

    let solution = tight_solver().solve(&lp).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal);

    close_qp(solution.primal[0], 1.0, "x0");
    close_qp(solution.primal[1], 1.0, "x1");
    close_qp(solution.objective, -3.0, "objective");

    // Nothing is binding, so every reduced cost is zero.
    close_qp(solution.col_dual[0], 0.0, "reduced cost of x0");
    close_qp(solution.col_dual[1], 0.0, "reduced cost of x1");
    assert_stationarity(&lp, &solution, QP_TOL);

    // The discriminating check, stated as its own assertion so a failure says
    // *which* misreading happened rather than just "not 1.0".
    assert!(
        (solution.primal[0] - 1.5).abs() > 0.1,
        "x0 = {} is the dropped-off-diagonal answer",
        solution.primal[0]
    );
}

/// The same QP with `x0` capped below its unconstrained optimum.
///
/// With `x0 = 0.5` the stationarity condition in `x1` gives
/// `x1 = (3 − x0)/2 = 1.25`, and `∂f/∂x0 = 2*0.5 + 1.25 − 3 = −0.75` — negative,
/// so the cap is genuinely active and carries that as its reduced cost.
/// Objective: `0.25 + 0.625 + 1.5625 − 1.5 − 3.75 = −2.8125`.
#[test]
fn qp_with_an_active_bound() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![-10.0, -10.0];
    lp.col_upper = vec![0.5, 10.0];
    lp.col_cost = vec![-3.0, -3.0];
    lp.hessian = Some(vec![(0, 0, 2.0), (1, 0, 1.0), (1, 1, 2.0)]);

    let solution = tight_solver().solve(&lp).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal);

    close_qp(solution.primal[0], 0.5, "x0 at its cap");
    close_qp(solution.primal[1], 1.25, "x1");
    close_qp(solution.objective, -2.8125, "objective");
    close_qp(solution.col_dual[0], -0.75, "reduced cost at the active cap");
    close_qp(solution.col_dual[1], 0.0, "reduced cost of the free variable");
    assert_stationarity(&lp, &solution, QP_TOL);
}

/// **The convention itself.** For a minimization, a binding *upper* row whose
/// relaxation would reduce the objective carries a **negative** dual.
///
/// ```text
/// min  −x0    s.t.  x0 <= 5,  0 <= x0 <= 100
/// ```
///
/// `x0` sits strictly inside its column bounds, so its reduced cost is zero
/// and stationarity forces `row_dual = c0 = −1`. Raising the limit by one unit
/// lowers the objective by one, which is exactly what a dual of `−1` says.
///
/// This is the assertion DC-OPF's locational marginal prices rest on.
#[test]
fn a_binding_upper_row_prices_negative_under_minimization() {
    let mut lp = LinearProgram::new(1);
    lp.col_lower = vec![0.0];
    lp.col_upper = vec![100.0];
    lp.col_cost = vec![-1.0];
    lp.add_row(&[(0, 1.0)], f64::NEG_INFINITY, 5.0);

    let solution = solver().solve(&lp).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal);

    close(solution.primal[0], 5.0, "x0 at the row limit");
    close(solution.objective, -5.0, "objective");
    close(solution.row_dual[0], -1.0, "the binding row's price");
    close(solution.col_dual[0], 0.0, "reduced cost of the interior variable");
    assert_stationarity(&lp, &solution, TOL);

    // And the sign is the claim, not just the magnitude.
    assert!(
        solution.row_dual[0] < 0.0,
        "relaxing this limit reduces the objective, so its dual must be negative"
    );
}

/// An infeasible problem is a valid answer, not an error — and it must not
/// come back carrying stale numbers.
#[test]
fn an_infeasible_problem_is_reported_as_such() {
    let mut lp = LinearProgram::new(1);
    lp.col_lower = vec![f64::NEG_INFINITY];
    lp.col_upper = vec![f64::INFINITY];
    lp.col_cost = vec![1.0];
    lp.add_row(&[(0, 1.0)], 2.0, f64::INFINITY);
    lp.add_row(&[(0, 1.0)], f64::NEG_INFINITY, 1.0);

    let solution = solver().solve(&lp).unwrap();
    assert!(!solution.is_optimal(), "status was {:?}", solution.status);
    assert!(
        matches!(solution.status, OptStatus::Infeasible | OptStatus::Other(_)),
        "expected an infeasible verdict, got {:?}",
        solution.status
    );
    assert!(solution.primal.is_empty(), "a failed solve must return no primal values");
}

#[test]
fn an_unbounded_problem_is_reported_as_such() {
    let mut lp = LinearProgram::new(1);
    lp.col_lower = vec![f64::NEG_INFINITY];
    lp.col_upper = vec![f64::INFINITY];
    lp.col_cost = vec![1.0];

    let solution = solver().solve(&lp).unwrap();
    assert!(!solution.is_optimal(), "status was {:?}", solution.status);
    assert!(
        matches!(solution.status, OptStatus::Unbounded | OptStatus::Other(_)),
        "expected an unbounded verdict, got {:?}",
        solution.status
    );
}

/// A malformed problem is rejected before HiGHS sees it, with the mistake
/// named — the whole point of `LinearProgram::validate`.
#[test]
fn a_malformed_problem_is_refused_by_the_boundary() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![0.0, 0.0];
    lp.col_upper = vec![1.0, 1.0];
    lp.col_cost = vec![1.0, 1.0];
    // Upper triangle — would silently double the off-diagonal term.
    lp.hessian = Some(vec![(0, 1, 2.0)]);

    match solver().solve(&lp) {
        Err(e) => assert_eq!(e, OpfError::UpperTriangleHessian { row: 0, col: 1 }),
        Ok(_) => panic!("an upper-triangle Hessian entry should have been refused"),
    }
}

/// The solver is reusable: a second problem must not inherit anything from the
/// first. Cheap to check and easy to get wrong, since HiGHS instances are
/// stateful.
#[test]
fn one_solver_handles_successive_problems() {
    let mut solver = solver();

    let mut first = LinearProgram::new(1);
    first.col_lower = vec![0.0];
    first.col_upper = vec![10.0];
    first.col_cost = vec![1.0];
    first.add_row(&[(0, 1.0)], 3.0, f64::INFINITY);
    let a = solver.solve(&first).unwrap();
    close(a.primal[0], 3.0, "first problem");

    // Different size and a different optimum, on the same instance.
    let mut second = LinearProgram::new(2);
    second.col_lower = vec![0.0, 0.0];
    second.col_upper = vec![3.0, f64::INFINITY];
    second.col_cost = vec![1.0, 2.0];
    second.add_row(&[(0, 1.0), (1, 1.0)], 4.0, f64::INFINITY);
    let b = solver.solve(&second).unwrap();
    close(b.primal[0], 3.0, "second problem x0");
    close(b.primal[1], 1.0, "second problem x1");
    close(b.objective, 5.0, "second problem objective");
}
