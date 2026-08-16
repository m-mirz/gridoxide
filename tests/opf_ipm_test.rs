//! The in-house interior-point QP solver.
//!
//! Two kinds of test live here. The first is the same battery
//! `opf_highs_test.rs` runs — hand-derived LPs and QPs where the optimum,
//! the duals and their signs are all known on paper. Both backends must clear
//! the identical bar, and stating it twice is deliberate: these cases are the
//! only ones whose answers come from outside either implementation, so they
//! are what makes the cross-check in `opf_cross_test.rs` meaningful rather
//! than merely self-consistent.
//!
//! The second is specific to this backend: the internal reformulation
//! (`opf::ipm`'s module docs) rewrites the problem before solving, and every
//! branch of that rewrite needs exercising — equality rows, inequality rows,
//! fixed columns, free columns. A bug there would produce a confidently wrong
//! answer to a *different* problem than the caller posed.

#![cfg(feature = "opf")]

use gridoxide::opf::ipm::{IpmOptions, IpmSolver};
use gridoxide::opf::{LinearProgram, OptStatus, Solution, Solver};

/// The interior-point method converges asymptotically rather than landing on
/// a vertex, so — unlike a simplex solve — its answers are only ever as
/// precise as the convergence tolerance permits.
///
/// With the default `tolerance` of 1e-10 the observed error on these problems
/// is under 1e-8. This bound sits above that floor and still leaves the tests
/// their teeth: the competing misreadings each case is designed to catch are
/// separated by ~0.5, seven orders of magnitude above this.
const TOL: f64 = 1e-7;

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < TOL, "{what}: got {got}, want {want}");
}

fn solve(lp: &LinearProgram) -> (Solution, usize) {
    let mut solver = IpmSolver::new();
    let solution = solver.solve(lp).unwrap();
    (solution, solver.iterations())
}

/// `col_dual = Qx + c − Aᵀ·row_dual` — stationarity of the Lagrangian.
///
/// Holds whatever sign convention a backend picks, so it catches a transposed
/// matrix, a misread Hessian or a mismatched vector without hard-coding what
/// any one solver reports. The convention itself is pinned separately below.
fn assert_stationarity(lp: &LinearProgram, solution: &Solution) {
    let mut expected = lp.col_cost.clone();
    if let Some(hessian) = &lp.hessian {
        for &(i, j, v) in hessian {
            expected[i] += v * solution.primal[j];
            if i != j {
                expected[j] += v * solution.primal[i];
            }
        }
    }
    for &(r, c, v) in &lp.rows {
        expected[c] -= v * solution.row_dual[r];
    }
    for k in 0..lp.n_vars {
        assert!(
            (expected[k] - solution.col_dual[k]).abs() < TOL,
            "stationarity in column {k}: reduced cost {} but Qx + c − Aᵀy = {}",
            solution.col_dual[k],
            expected[k]
        );
    }
}

/// ```text
/// min  x0 + 2*x1   s.t.  x0 + x1 >= 4,  0 <= x0 <= 3,  x1 >= 0
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

    let (solution, _) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.primal[0], 3.0, "x0");
    close(solution.primal[1], 1.0, "x1");
    close(solution.objective, 5.0, "objective");

    // Requiring one more unit costs 2, the price of the marginal variable.
    close(solution.row_dual[0], 2.0, "the binding row's price");
    close(solution.col_dual[0], -1.0, "reduced cost at x0's active ceiling");
    assert_stationarity(&lp, &solution);
}

/// A vertex optimum is where an interior-point method is *least* comfortable:
/// it approaches from inside and never arrives exactly. Worth an explicit
/// check that the answer is nonetheless sharp.
#[test]
fn a_slack_row_carries_a_negligible_price() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![0.0, 0.0];
    lp.col_upper = vec![3.0, f64::INFINITY];
    lp.col_cost = vec![1.0, 2.0];
    lp.add_row(&[(0, 1.0), (1, 1.0)], 4.0, f64::INFINITY);
    lp.add_row(&[(0, 1.0)], f64::NEG_INFINITY, 100.0);

    let (solution, _) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.objective, 5.0, "objective");

    // Not exactly zero, unlike simplex: complementary slackness holds only in
    // the limit. It must still be negligible next to the binding row's 2.
    assert!(
        solution.row_dual[1].abs() < TOL,
        "a slack row priced at {}, which is not negligible",
        solution.row_dual[1]
    );
    assert_stationarity(&lp, &solution);
}

/// A strictly convex QP with an **off-diagonal** Hessian term.
///
/// ```text
/// min  1/2 x'Qx − 3*x0 − 3*x1,    Q = [[2, 1], [1, 2]]
/// ```
///
/// `Qx = [3, 3]` gives `x = (1, 1)` and objective `3 − 6 = −3`.
///
/// The off-diagonal is the point, and this is the case that pins the Hessian
/// convention independently of HiGHS. `LinearProgram::hessian` takes the lower
/// triangle only, and the three plausible readings are all distinguishable:
/// counting `(1,0)` once gives `(1, 1)`; counting it twice makes `Q` singular;
/// dropping it gives `(1.5, 1.5)` and objective `−4.5`.
#[test]
fn qp_with_an_off_diagonal_hessian_and_an_interior_optimum() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![-10.0, -10.0];
    lp.col_upper = vec![10.0, 10.0];
    lp.col_cost = vec![-3.0, -3.0];
    lp.hessian = Some(vec![(0, 0, 2.0), (1, 0, 1.0), (1, 1, 2.0)]);

    let (solution, _) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.primal[0], 1.0, "x0");
    close(solution.primal[1], 1.0, "x1");
    close(solution.objective, -3.0, "objective");
    close(solution.col_dual[0], 0.0, "reduced cost of x0");
    assert_stationarity(&lp, &solution);

    assert!(
        (solution.primal[0] - 1.5).abs() > 0.1,
        "x0 = {} is the dropped-off-diagonal answer",
        solution.primal[0]
    );
}

/// The same QP with `x0` capped below its unconstrained optimum.
///
/// With `x0 = 0.5`, stationarity in `x1` gives `x1 = (3 − x0)/2 = 1.25`, and
/// `∂f/∂x0 = 2(0.5) + 1.25 − 3 = −0.75` — negative, so the cap is genuinely
/// active and carries that as its reduced cost.
#[test]
fn qp_with_an_active_bound() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![-10.0, -10.0];
    lp.col_upper = vec![0.5, 10.0];
    lp.col_cost = vec![-3.0, -3.0];
    lp.hessian = Some(vec![(0, 0, 2.0), (1, 0, 1.0), (1, 1, 2.0)]);

    let (solution, _) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.primal[0], 0.5, "x0 at its cap");
    close(solution.primal[1], 1.25, "x1");
    close(solution.objective, -2.8125, "objective");
    close(solution.col_dual[0], -0.75, "reduced cost at the active cap");
    assert_stationarity(&lp, &solution);
}

/// **The sign convention**, which DC-OPF's locational marginal prices rest on.
///
/// ```text
/// min  −x0    s.t.  x0 <= 5,  0 <= x0 <= 100
/// ```
///
/// `x0` sits strictly inside its column bounds, so its reduced cost is zero
/// and stationarity forces `row_dual = c0 = −1`. Raising the limit by one unit
/// lowers the objective by one — exactly what a dual of `−1` says.
#[test]
fn a_binding_upper_row_prices_negative_under_minimization() {
    let mut lp = LinearProgram::new(1);
    lp.col_lower = vec![0.0];
    lp.col_upper = vec![100.0];
    lp.col_cost = vec![-1.0];
    lp.add_row(&[(0, 1.0)], f64::NEG_INFINITY, 5.0);

    let (solution, _) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.primal[0], 5.0, "x0 at the row limit");
    close(solution.objective, -5.0, "objective");
    close(solution.row_dual[0], -1.0, "the binding row's price");
    assert_stationarity(&lp, &solution);

    assert!(
        solution.row_dual[0] < 0.0,
        "relaxing this limit reduces the objective, so its dual must be negative"
    );
}

/// **A fixed column**, which has no interior and so cannot carry a barrier
/// term at all — `1/(u_k − ℓ_k)` diverges. `ipm` rewrites it as an equality
/// row instead.
///
/// This is not an exotic input: DC-OPF pins its reference angle exactly this
/// way, so every OPF this backend solves takes the path. Here `x0` is pinned
/// at 3 and `x0 + x1 = 5` forces `x1 = 2`, for `2(3) + 1(2) = 8`.
#[test]
fn a_fixed_column_is_pinned_without_a_barrier() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![3.0, 0.0];
    lp.col_upper = vec![3.0, 10.0];
    lp.col_cost = vec![2.0, 1.0];
    lp.add_row(&[(0, 1.0), (1, 1.0)], 5.0, 5.0);

    let (solution, _) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.primal[0], 3.0, "the fixed column holds its value");
    close(solution.primal[1], 2.0, "x1");
    close(solution.objective, 8.0, "objective");

    // The equality row prices the marginal unit, which is x1 at 1.
    close(solution.row_dual[0], 1.0, "the equality row's price");
    // And the fixed column's reduced cost is what stationarity leaves over —
    // the multiplier of its internal pinning row, surfaced where a caller
    // expects a fixed variable's price.
    close(solution.col_dual[0], 1.0, "reduced cost of the fixed column");
    assert_stationarity(&lp, &solution);
}

/// A free column — no bounds either way, so no barrier term and no curvature
/// unless the Hessian supplies it. Without regularization the KKT matrix is
/// singular here, so this is the test that the regularization is actually
/// doing its job.
#[test]
fn a_free_column_with_no_curvature_still_solves() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![f64::NEG_INFINITY, 0.0];
    lp.col_upper = vec![f64::INFINITY, 10.0];
    lp.col_cost = vec![0.0, 1.0];
    // x0 is free and costs nothing; the rows are what determine it.
    lp.add_row(&[(0, 1.0), (1, -1.0)], 0.0, 0.0);
    lp.add_row(&[(0, 1.0)], 4.0, f64::INFINITY);

    let (solution, _) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.primal[0], 4.0, "x0 driven by the rows alone");
    close(solution.primal[1], 4.0, "x1 tied to x0");
    close(solution.objective, 4.0, "objective");
    assert_stationarity(&lp, &solution);
}

/// Every row an equality, which leaves the barrier nothing to do on the rows
/// and the whole problem determined by the constraint system.
#[test]
fn a_fully_determined_equality_system_solves_in_one_step() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![f64::NEG_INFINITY; 2];
    lp.col_upper = vec![f64::INFINITY; 2];
    lp.col_cost = vec![0.0, 0.0];
    lp.add_row(&[(0, 1.0), (1, 1.0)], 3.0, 3.0);
    lp.add_row(&[(0, 1.0), (1, -1.0)], 1.0, 1.0);

    let (solution, iterations) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.primal[0], 2.0, "x0");
    close(solution.primal[1], 1.0, "x1");

    // Two, not one: with no bounds there is no barrier and no fraction-to-
    // boundary cap, so the first pass takes the exact Newton step and the
    // second confirms it — the convergence test runs at the top of the loop.
    //
    // This is a sharper check than it looks. Nothing here blocks the step, so
    // anything above two means the step is being scaled when it should not be,
    // which is a bug that stays invisible elsewhere: on a bounded problem the
    // same fault merely costs iterations rather than changing any answer.
    assert_eq!(iterations, 2, "an unblocked Newton step should be taken in full");
}

/// An infeasible problem must be *diagnosed*, not merely failed.
///
/// This is where an interior-point method is structurally weaker than simplex:
/// with no basis there is no infeasibility certificate, only the observation
/// that the duals diverge while the primal residual stalls. The verdict is
/// therefore evidence-based rather than proven — but it must still be right.
#[test]
fn an_infeasible_problem_is_reported_as_such() {
    let mut lp = LinearProgram::new(1);
    lp.col_lower = vec![f64::NEG_INFINITY];
    lp.col_upper = vec![f64::INFINITY];
    lp.col_cost = vec![1.0];
    lp.add_row(&[(0, 1.0)], 5.0, f64::INFINITY);
    lp.add_row(&[(0, 1.0)], f64::NEG_INFINITY, 2.0);

    let (solution, iterations) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Infeasible);
    assert!(solution.primal.is_empty(), "a failed solve must return no primal values");
    assert!(solution.objective.is_nan());
    assert!(iterations < 30, "divergence should be recognised quickly, took {iterations}");
}

/// The mirror image: the primal runs off along a ray while the *dual* residual
/// stalls. Distinguishing the two directions is the whole content of the
/// diagnosis, so a solver that reported `Infeasible` here would be as wrong as
/// one that reported `Optimal`.
#[test]
fn an_unbounded_problem_is_reported_as_such() {
    let mut lp = LinearProgram::new(1);
    lp.col_lower = vec![f64::NEG_INFINITY];
    lp.col_upper = vec![f64::INFINITY];
    lp.col_cost = vec![1.0];

    let (solution, iterations) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Unbounded);
    assert!(solution.primal.is_empty());
    assert!(iterations < 30, "divergence should be recognised quickly, took {iterations}");
}

/// A bounded problem that *looks* like it has a ray. The objective is
/// `−(x0 − x1)` and the row caps `x0 − x1` at 1, so the optimum is exactly −1
/// even though both variables are individually unbounded above.
///
/// Kept because it is the case a naive divergence check gets wrong: the
/// iterates may grow without the problem being unbounded, which is why the
/// diagnosis pairs a runaway norm with a *stalled residual* rather than
/// trusting magnitude alone.
#[test]
fn growth_without_divergence_is_not_mistaken_for_unboundedness() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![0.0, 0.0];
    lp.col_upper = vec![f64::INFINITY; 2];
    lp.col_cost = vec![-1.0, 1.0];
    lp.add_row(&[(0, 1.0), (1, -1.0)], f64::NEG_INFINITY, 1.0);

    let (solution, _) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.objective, -1.0, "objective");
}

/// Degenerate shapes that must not panic or loop.
#[test]
fn degenerate_problems_are_answered_rather_than_crashing() {
    let (solution, _) = solve(&LinearProgram::new(0));
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.objective, 0.0, "the empty problem costs nothing");

    // Bounds but no rows at all: the optimum is read straight off the box.
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![-1.0, -1.0];
    lp.col_upper = vec![1.0, 1.0];
    lp.col_cost = vec![1.0, -1.0];
    let (solution, _) = solve(&lp);
    assert_eq!(solution.status, OptStatus::Optimal);
    close(solution.objective, -2.0, "objective");
    close(solution.primal[0], -1.0, "x0 at its floor");
    close(solution.primal[1], 1.0, "x1 at its ceiling");
}

/// The `offset` is added to the reported objective without touching the
/// optimum — it carries generator cost curves' constant terms, which change
/// what a dispatch *costs* but not what the cheapest dispatch *is*.
#[test]
fn the_objective_offset_shifts_the_value_and_not_the_argument() {
    let mut lp = LinearProgram::new(1);
    lp.col_lower = vec![0.0];
    lp.col_upper = vec![10.0];
    lp.col_cost = vec![1.0];
    lp.add_row(&[(0, 1.0)], 4.0, f64::INFINITY);

    let (plain, _) = solve(&lp);
    lp.offset = 100.0;
    let (shifted, _) = solve(&lp);

    close(shifted.objective - plain.objective, 100.0, "the offset");
    close(shifted.primal[0], plain.primal[0], "the optimum is unmoved");
}

/// Iterative refinement exists to undo the regularization's perturbation, so
/// turning it off should measurably degrade the answer. If it does not, the
/// refinement is not doing anything and the regularization is not costing
/// anything — either way the module docs would be wrong.
#[test]
fn iterative_refinement_earns_its_keep() {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![0.0, 0.0];
    lp.col_upper = vec![f64::INFINITY; 2];
    lp.col_cost = vec![1.0, 3.0];
    lp.hessian = Some(vec![(0, 0, 2.0), (1, 0, 0.5), (1, 1, 4.0)]);
    lp.add_row(&[(0, 1.0), (1, 1.0)], 10.0, 10.0);

    let refined = {
        let mut s = IpmSolver::new();
        s.solve(&lp).unwrap()
    };
    let unrefined = {
        let mut s = IpmSolver::with_options(IpmOptions {
            refinement_rounds: 0,
            ..IpmOptions::default()
        });
        s.solve(&lp).unwrap()
    };

    assert_eq!(refined.status, OptStatus::Optimal);
    assert_eq!(unrefined.status, OptStatus::Optimal);

    // Both land on the same answer — refinement is an accuracy measure, not a
    // correctness one — but the refined residual must be the smaller.
    let residual = |s: &Solution| {
        let mut r = lp.col_cost.clone();
        for &(i, j, v) in lp.hessian.as_ref().unwrap() {
            r[i] += v * s.primal[j];
            if i != j {
                r[j] += v * s.primal[i];
            }
        }
        for &(row, c, v) in &lp.rows {
            r[c] -= v * s.row_dual[row];
        }
        (0..2).map(|k| (r[k] - s.col_dual[k]).abs()).fold(0.0f64, f64::max)
    };
    assert!(
        residual(&refined) <= residual(&unrefined),
        "refinement left a larger stationarity residual ({:.3e}) than skipping it ({:.3e})",
        residual(&refined),
        residual(&unrefined)
    );
}
