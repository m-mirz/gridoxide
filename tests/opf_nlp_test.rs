//! The nonconvex nonlinear-program solver, on problems small enough to solve
//! by hand.
//!
//! Everything here has a closed-form optimum derived on paper. That matters
//! more for a nonlinear solver than for a linear one: there is no certificate
//! of global optimality to check against, so unless the answer is known
//! independently there is nothing to compare a converged point to except its
//! own first-order conditions — which a subtly wrong Hessian will satisfy
//! perfectly at the wrong point.
//!
//! The cases are chosen to exercise what nonconvexity actually breaks:
//! curved constraints, an indefinite Lagrangian, a bound active at the
//! optimum, and a problem with two local minima where which one is found
//! depends on where the search starts.

#![cfg(feature = "opf")]

use gridoxide::opf::nlp::{solve, NlpOptions, NonlinearProblem};
use gridoxide::opf::OptStatus;

const TOL: f64 = 1e-6;

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < TOL, "{what}: got {got}, want {want}");
}

/// Minimize `x0 + 2·x1` on the circle `x0² + x1² = 1`.
///
/// The optimum is where the objective's gradient is antiparallel to the
/// constraint's: `x = −(1, 2)/√5`, objective `−√5`.
///
/// A *linear* objective on the circle, rather than the obvious `x0² + x1²`,
/// and the reason is worth recording. Minimizing the radius on the circle
/// makes every feasible point optimal, and the reduced Hessian
/// `∇²f − y∇²c = (2 − 2y)I` is exactly zero at the solution's `y = 1`. That
/// problem is degenerate: the solver reaches it to 6e-9 but can never satisfy
/// a first-order test, because the first-order conditions do not distinguish
/// any point on the circle from any other. Testing against it would measure
/// tolerance settings rather than correctness.
///
/// What is tested here is that a *curved equality* is satisfied at the
/// solution — a solver that linearized the constraint once and never revisited
/// it would land off the circle.
struct OnACircle;

impl NonlinearProblem for OnACircle {
    fn n_vars(&self) -> usize {
        2
    }
    fn n_constraints(&self) -> usize {
        1
    }
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![-10.0; 2], vec![10.0; 2])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![1.0], vec![1.0])
    }
    fn objective(&self, x: &[f64]) -> f64 {
        x[0] + 2.0 * x[1]
    }
    fn gradient(&self, _x: &[f64]) -> Vec<f64> {
        vec![1.0, 2.0]
    }
    fn constraints(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] + x[1] * x[1]]
    }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 2.0 * x[0]), (0, 1, 2.0 * x[1])]
    }
    fn lagrangian_hessian(&self, _x: &[f64], y: &[f64]) -> Vec<(usize, usize, f64)> {
        // ∇²f = 0, ∇²c = 2I, and the convention is ∇²f − Σ yᵢ∇²cᵢ.
        let d = -2.0 * y[0];
        vec![(0, 0, d), (1, 1, d)]
    }
    fn initial_point(&self) -> Vec<f64> {
        vec![0.6, -0.4]
    }
}

#[test]
fn a_curved_equality_constraint_is_satisfied_at_the_solution() {
    let solution = solve(&OnACircle, &NlpOptions::default()).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal, "{:?}", solution);

    let root5 = 5.0f64.sqrt();
    close(solution.x[0], -1.0 / root5, "x0");
    close(solution.x[1], -2.0 / root5, "x1");
    close(solution.objective, -root5, "objective");
    close(
        solution.x[0] * solution.x[0] + solution.x[1] * solution.x[1],
        1.0,
        "the constraint itself",
    );
    assert!(solution.violation < 1e-8, "violation {}", solution.violation);
}

/// Minimize `x0 · x1` subject to `x0 + x1 = 2`, with both variables in
/// `[0, 10]`.
///
/// The Lagrangian is **indefinite**: the objective's Hessian is
/// `[[0,1],[1,0]]`, whose eigenvalues are ±1. The interior stationary point
/// `(1, 1)` is a saddle along the feasible line, not a minimum, so a solver
/// that simply drove the KKT residual to zero would stop there and report
/// success. The true minima sit at the corners `(0, 2)` and `(2, 0)`, both
/// with objective 0.
///
/// This is the case that tests the regularization: without raising `γ` when
/// the direction stops being a descent direction, the method walks to the
/// saddle and stays.
struct IndefiniteProduct;

impl NonlinearProblem for IndefiniteProduct {
    fn n_vars(&self) -> usize {
        2
    }
    fn n_constraints(&self) -> usize {
        1
    }
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0; 2], vec![10.0; 2])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![2.0], vec![2.0])
    }
    fn objective(&self, x: &[f64]) -> f64 {
        x[0] * x[1]
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![x[1], x[0]]
    }
    fn constraints(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] + x[1]]
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 1.0), (0, 1, 1.0)]
    }
    fn lagrangian_hessian(&self, _x: &[f64], _y: &[f64]) -> Vec<(usize, usize, f64)> {
        // The constraint is linear, so only the objective contributes.
        vec![(0, 1, 1.0), (1, 0, 1.0)]
    }
    fn initial_point(&self) -> Vec<f64> {
        vec![1.4, 0.6]
    }
}

#[test]
fn an_indefinite_lagrangian_does_not_stall_at_the_saddle() {
    let solution = solve(&IndefiniteProduct, &NlpOptions::default()).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal, "{:?}", solution);
    close(solution.objective, 0.0, "objective at a corner");
    close(solution.x[0] + solution.x[1], 2.0, "the equality constraint");

    // One variable at zero, the other at two — whichever way round.
    let (a, b) = (solution.x[0].min(solution.x[1]), solution.x[0].max(solution.x[1]));
    close(a, 0.0, "the variable driven to its floor");
    close(b, 2.0, "the variable taking the whole budget");

    assert!(
        (solution.objective - 1.0).abs() > 0.5,
        "objective {} is the saddle value, so the solver stopped at a stationary point \
         that is not a minimum",
        solution.objective
    );
}

/// Minimize `(x0 − 2)² + (x1 − 1)²` subject to the *inequality*
/// `x0² + x1² ≤ 1`.
///
/// The unconstrained optimum `(2, 1)` is outside the disc, so the constraint
/// is active and the solution is where the ray from the origin through `(2,1)`
/// meets the unit circle: `(2, 1)/√5`. Objective `(√5 − 1)² ≈ 1.5279`.
///
/// Tests a curved *inequality* — the slack reformulation plus a nonlinear
/// Jacobian row — and gives a nonzero multiplier to check.
struct InsideADisc;

impl NonlinearProblem for InsideADisc {
    fn n_vars(&self) -> usize {
        2
    }
    fn n_constraints(&self) -> usize {
        1
    }
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![-5.0; 2], vec![5.0; 2])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![f64::NEG_INFINITY], vec![1.0])
    }
    fn objective(&self, x: &[f64]) -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2)
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]
    }
    fn constraints(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] + x[1] * x[1]]
    }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 2.0 * x[0]), (0, 1, 2.0 * x[1])]
    }
    fn lagrangian_hessian(&self, _x: &[f64], y: &[f64]) -> Vec<(usize, usize, f64)> {
        let d = 2.0 - 2.0 * y[0];
        vec![(0, 0, d), (1, 1, d)]
    }
    fn initial_point(&self) -> Vec<f64> {
        vec![0.1, 0.1]
    }
}

#[test]
fn a_curved_inequality_binds_where_geometry_says_it_should() {
    let solution = solve(&InsideADisc, &NlpOptions::default()).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal, "{:?}", solution);

    let root5 = 5.0f64.sqrt();
    close(solution.x[0], 2.0 / root5, "x0 on the circle");
    close(solution.x[1], 1.0 / root5, "x1 on the circle");
    close(solution.objective, (root5 - 1.0).powi(2), "objective");
    assert!(solution.violation < 1e-8, "violation {}", solution.violation);
}

/// A problem with **two local minima**, to make explicit what a nonconvex
/// solver does and does not promise.
///
/// Minimize `x⁴ − 4x² + 0.3x` on `[−3, 3]`. The two wells sit near `x ≈ ±1.41`
/// and the linear term breaks the tie, making the negative one slightly
/// deeper. A local method returns whichever well it starts in — and that is
/// correct behaviour, not a defect.
///
/// This is asserted rather than merely noted, because AC-OPF has the same
/// property and every comparison against a published objective rests on it:
/// disagreeing with a reference may mean a different local solution, which is
/// why [`NlpSolution`](gridoxide::opf::nlp::NlpSolution) reports feasibility
/// alongside the objective.
struct TwoWells {
    start: f64,
}

impl NonlinearProblem for TwoWells {
    fn n_vars(&self) -> usize {
        1
    }
    fn n_constraints(&self) -> usize {
        0
    }
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![-3.0], vec![3.0])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![], vec![])
    }
    fn objective(&self, x: &[f64]) -> f64 {
        x[0].powi(4) - 4.0 * x[0] * x[0] + 0.3 * x[0]
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![4.0 * x[0].powi(3) - 8.0 * x[0] + 0.3]
    }
    fn constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![]
    }
    fn lagrangian_hessian(&self, x: &[f64], _y: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 12.0 * x[0] * x[0] - 8.0)]
    }
    fn initial_point(&self) -> Vec<f64> {
        vec![self.start]
    }
}

#[test]
fn a_nonconvex_problem_finds_the_local_minimum_it_started_near() {
    let from_the_right = solve(&TwoWells { start: 2.0 }, &NlpOptions::default()).unwrap();
    let from_the_left = solve(&TwoWells { start: -2.0 }, &NlpOptions::default()).unwrap();

    assert_eq!(from_the_right.status, OptStatus::Optimal);
    assert_eq!(from_the_left.status, OptStatus::Optimal);
    assert!(from_the_right.x[0] > 0.0, "started right, ended at {}", from_the_right.x[0]);
    assert!(from_the_left.x[0] < 0.0, "started left, ended at {}", from_the_left.x[0]);

    // Both are genuine stationary points: f'(x) = 4x³ − 8x + 0.3 ≈ 0.
    for solution in [&from_the_right, &from_the_left] {
        let x = solution.x[0];
        let slope = 4.0 * x.powi(3) - 8.0 * x + 0.3;
        assert!(slope.abs() < 1e-5, "f'({x}) = {slope}, not a stationary point");
    }

    // And the left well really is the deeper one, so the answer genuinely
    // depends on the start rather than both runs finding the same point.
    assert!(
        from_the_left.objective < from_the_right.objective,
        "left {} should be below right {}",
        from_the_left.objective,
        from_the_right.objective
    );
}

/// Bounds active at the optimum, with no constraints at all — the barrier
/// working on its own.
struct BoundedQuadratic;

impl NonlinearProblem for BoundedQuadratic {
    fn n_vars(&self) -> usize {
        2
    }
    fn n_constraints(&self) -> usize {
        0
    }
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![1.0, -0.5], vec![4.0, 0.25])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![], vec![])
    }
    fn objective(&self, x: &[f64]) -> f64 {
        (x[0] - 0.5).powi(2) + (x[1] - 3.0).powi(2)
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * (x[0] - 0.5), 2.0 * (x[1] - 3.0)]
    }
    fn constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![]
    }
    fn lagrangian_hessian(&self, _x: &[f64], _y: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 2.0), (1, 1, 2.0)]
    }
    fn initial_point(&self) -> Vec<f64> {
        vec![2.0, 0.0]
    }
}

#[test]
fn both_bounds_bind_when_the_unconstrained_optimum_is_outside_the_box() {
    let solution = solve(&BoundedQuadratic, &NlpOptions::default()).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal, "{:?}", solution);
    // Unconstrained optimum (0.5, 3.0) is outside on both axes, so each
    // variable stops at its nearer bound.
    close(solution.x[0], 1.0, "x0 at its floor");
    close(solution.x[1], 0.25, "x1 at its ceiling");
    close(solution.objective, 0.25 + 7.5625, "objective");
}

/// A fixed variable — equal bounds, so no interior and no barrier term. The
/// same path DC-OPF's reference angle takes through the convex solver.
struct WithAFixedVariable;

impl NonlinearProblem for WithAFixedVariable {
    fn n_vars(&self) -> usize {
        2
    }
    fn n_constraints(&self) -> usize {
        1
    }
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.75, -10.0], vec![0.75, 10.0])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![2.0], vec![2.0])
    }
    fn objective(&self, x: &[f64]) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * x[0], 2.0 * x[1]]
    }
    fn constraints(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] + x[1]]
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 1.0), (0, 1, 1.0)]
    }
    fn lagrangian_hessian(&self, _x: &[f64], _y: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 2.0), (1, 1, 2.0)]
    }
    fn initial_point(&self) -> Vec<f64> {
        vec![0.75, 0.0]
    }
}

#[test]
fn a_fixed_variable_is_pinned_without_a_barrier_term() {
    let solution = solve(&WithAFixedVariable, &NlpOptions::default()).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal, "{:?}", solution);
    close(solution.x[0], 0.75, "the fixed variable holds");
    close(solution.x[1], 1.25, "the free variable takes the remainder");
    close(solution.objective, 0.5625 + 1.5625, "objective");
}
