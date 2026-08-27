//! The time step: an implicit one-step rule with a Newton solve per step.
//!
//! # The rule, and why there are two of them
//!
//! The trapezoidal rule is the default. It is second-order and A-stable, which
//! is what a stiff phasor-domain DAE wants, and it is what every production
//! RMS simulator uses between events.
//!
//! It is **not** L-stable. Its amplification factor tends to `−1` rather than
//! `0` as `h·λ → −∞`, so a step change excites the fastest mode of the system
//! into a numerical oscillation that alternates sign and decays only as the
//! physical mode does. On a fault application — the one thing an RMS run
//! exists to simulate — that shows up as a visible ringing on the voltages
//! that is entirely an artifact of the integrator.
//!
//! The standard remedy is cheap: take a couple of **backward Euler** steps
//! immediately after each discontinuity, then return to trapezoidal. Backward
//! Euler is L-stable, so it annihilates the mode in one step; it is only
//! first-order, but two steps out of a run of thousands cost nothing
//! measurable in accuracy. [`DynamicsOptions::damping_steps`](
//! super::DynamicsOptions::damping_steps) is that count.
//!
//! Both rules are the same residual with different coefficients — see
//! [`dae`](super::dae) — so nothing here branches on which is active beyond
//! choosing `(a, b)`.
//!
//! # The Newton solve
//!
//! One Newton per step over the whole vector `z = [x; v_re, v_im]`,
//! differential and algebraic together. There is no inner iteration between a
//! device solver and a network solver, so there is no interface error to
//! converge away and no per-step lag between the two halves.
//!
//! The Jacobian's *pattern* is fixed for the lifetime of a topology, so
//! `LinearSolver` analyzes once and every step is a numeric refactorization —
//! exactly the contract `solver::newton_raphson_cached` relies on, reused
//! wholesale.

use num_complex::Complex;

use crate::solver::LinearSolver;

use super::dae::residual;
use super::models::ModelJacobian;
use super::{DynamicSystem, DynamicsOptions, DynamicsStatus, Trajectory};

/// Coefficients `(a, b)` of `x₁ − x₀ − h(a·f₁ + b·f₀) = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    Trapezoidal,
    BackwardEuler,
}

impl Rule {
    pub fn coefficients(self) -> (f64, f64) {
        match self {
            Rule::Trapezoidal => (0.5, 0.5),
            Rule::BackwardEuler => (1.0, 0.0),
        }
    }
}

/// What one step did, so the caller can distinguish "the network came apart"
/// from "the step was too big".
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepOutcome {
    Converged { iterations: usize },
    NotConverged,
    Singular,
}

/// The integration loop, generic over the sparse-LU backend for the same
/// reason `solver::newton_raphson_cached` and `continuation::trace` are: four
/// near-identical copies otherwise.
pub fn integrate<S: LinearSolver>(system: &mut DynamicSystem, opts: &DynamicsOptions) -> (Trajectory, DynamicsStatus, usize, usize) {
    let pattern = system.pattern.clone();
    let layout = pattern.layout().clone();
    let n = layout.n();

    // Built lazily, on the first iteration that has real values to analyze —
    // see `DaePattern::to_triplets`.
    let mut solver: Option<S> = None;

    let mut x = system.x0.clone();
    let mut v = system.v0.clone();

    let mut scratch: Vec<ModelJacobian> =
        (0..layout.n_devices()).map(|d| ModelJacobian::zeros(layout.dev_len[d])).collect();
    let mut values: Vec<f64> = Vec::with_capacity(pattern.nnz());
    let mut f_scratch = vec![0.0; layout.n_diff];
    let mut f0 = vec![0.0; layout.n_diff];
    let mut res = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    let mut traj = Trajectory::new(system.observable_names());
    traj.push(0.0, &x, &v);

    let mut t = 0.0;
    let mut steps = 0usize;
    let mut newton_total = 0usize;
    // Backward-Euler steps still owed. Starting the run itself counts as a
    // discontinuity: the initial point is an equilibrium, so this costs
    // nothing, and it means a run that *begins* with an event is damped the
    // same way one that meets an event later is.
    let mut damping_left = opts.damping_steps;

    let mut status = DynamicsStatus::Completed;

    while t < opts.end_time - 1e-12 {
        let h = opts.step.min(opts.end_time - t);
        let rule = if damping_left > 0 { Rule::BackwardEuler } else { Rule::Trapezoidal };
        let (a, b) = rule.coefficients();
        let (ha, hb) = (h * a, h * b);

        // Derivatives at the start of the step. Trapezoidal needs them;
        // backward Euler multiplies them by zero, and computing them anyway
        // keeps one code path.
        system.derivatives_at(&x, &v, &mut f0);

        let x_prev = x.clone();
        // Explicit-Euler predictor for the differential half; the algebraic
        // half starts from where it was, which is already a solution of the
        // constraint at the previous point.
        for i in 0..layout.n_diff {
            x[i] = x_prev[i] + h * f0[i];
        }

        let mut outcome = StepOutcome::NotConverged;
        for iteration in 1..=opts.max_newton {
            residual(
                &pattern,
                &system.models,
                &x_prev,
                &f0,
                &x,
                &v,
                &system.v_fixed,
                ha,
                hb,
                &system.ybus,
                &mut f_scratch,
                &mut res,
            );

            let norm = res.iter().fold(0.0f64, |m, r| m.max(r.abs()));
            if norm < opts.tol {
                outcome = StepOutcome::Converged { iterations: iteration - 1 };
                break;
            }

            pattern.fill(&system.models, &x, &v, ha, &system.ybus, &mut scratch, &mut values);
            for k in 0..n {
                rhs[k] = -res[k];
            }
            if solver.is_none() {
                solver = S::new(n, &pattern.to_triplets(&values));
            }
            let Some(system_lu) = solver.as_mut() else {
                outcome = StepOutcome::Singular;
                break;
            };
            let dz = match system_lu.factor_and_solve_values(&values, &rhs) {
                Some(dz) => dz,
                None => {
                    outcome = StepOutcome::Singular;
                    break;
                }
            };

            for i in 0..layout.n_diff {
                x[i] += dz[i];
            }
            for i in 0..layout.n_bus {
                v[i] += Complex::new(dz[layout.v_re(i)], dz[layout.v_im(i)]);
            }
            newton_total += 1;
        }

        match outcome {
            StepOutcome::Converged { iterations } => {
                let _ = iterations;
            }
            StepOutcome::NotConverged => {
                status = DynamicsStatus::NewtonFailed { time: t + h };
                break;
            }
            StepOutcome::Singular => {
                status = DynamicsStatus::Singular { time: t + h };
                break;
            }
        }

        t += h;
        steps += 1;
        damping_left = damping_left.saturating_sub(1);
        traj.push(t, &x, &v);
    }

    system.x0 = x;
    system.v0 = v;
    (traj, status, steps, newton_total)
}

/// Solves the algebraic block alone, holding every differential state fixed.
///
/// This is what an event needs (phase 2) and what the builder uses to confirm
/// its own initial point: at a discontinuity the algebraic variables jump while
/// the differential ones do not, so the trapezoidal rule must not be applied
/// across it. Instead `x` is frozen and `Y V − I_inj(x, V) = 0` is re-solved
/// for `V` alone.
///
/// Implemented as an ordinary step of the full system with `ha = 0`: that
/// makes every differential row read `x₁ − x₀ = 0`, pinning `x`, while the
/// algebraic rows are untouched. One assembly, one factorization, no second
/// pattern for the sub-block.
pub fn solve_algebraic<S: LinearSolver>(
    system: &mut DynamicSystem,
    tol: f64,
    max_newton: usize,
) -> StepOutcome {
    let pattern = system.pattern.clone();
    let layout = pattern.layout().clone();
    let n = layout.n();

    let mut solver: Option<S> = None;

    let mut scratch: Vec<ModelJacobian> =
        (0..layout.n_devices()).map(|d| ModelJacobian::zeros(layout.dev_len[d])).collect();
    let mut values: Vec<f64> = Vec::with_capacity(pattern.nnz());
    let mut f_scratch = vec![0.0; layout.n_diff];
    let f0 = vec![0.0; layout.n_diff];
    let mut res = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    let x_prev = system.x0.clone();
    let mut x = system.x0.clone();
    let mut v = system.v0.clone();

    for iteration in 1..=max_newton {
        residual(
            &pattern, &system.models, &x_prev, &f0, &x, &v, &system.v_fixed,
            0.0, 0.0, &system.ybus, &mut f_scratch, &mut res,
        );
        let norm = res.iter().fold(0.0f64, |m, r| m.max(r.abs()));
        if norm < tol {
            system.x0 = x;
            system.v0 = v;
            return StepOutcome::Converged { iterations: iteration - 1 };
        }
        pattern.fill(&system.models, &x, &v, 0.0, &system.ybus, &mut scratch, &mut values);
        for k in 0..n {
            rhs[k] = -res[k];
        }
        if solver.is_none() {
            solver = S::new(n, &pattern.to_triplets(&values));
        }
        let Some(system_lu) = solver.as_mut() else {
            return StepOutcome::Singular;
        };
        let dz = match system_lu.factor_and_solve_values(&values, &rhs) {
            Some(dz) => dz,
            None => return StepOutcome::Singular,
        };
        for i in 0..layout.n_diff {
            x[i] += dz[i];
        }
        for i in 0..layout.n_bus {
            v[i] += Complex::new(dz[layout.v_re(i)], dz[layout.v_im(i)]);
        }
    }
    StepOutcome::NotConverged
}
