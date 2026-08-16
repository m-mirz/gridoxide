//! A primal-dual interior-point method for *nonconvex* nonlinear programs.
//!
//! [`ipm`](super::ipm) solves convex quadratic programs, where the Newton step
//! is exact and can always be taken in full. This solves the general problem,
//! where it cannot: the model is only a local approximation, so a full step may
//! increase the objective, violate the constraints, or head toward a maximum.
//! Nearly all the extra machinery here exists to handle that.
//!
//! \\[ \min_x f(x) \quad\text{s.t.}\quad c_L \le c(x) \le c_U,
//!    \qquad x_L \le x \le x_U \\]
//!
//! # What nonconvexity costs
//!
//! Three things the QP solver does not need:
//!
//! - **A line search.** The step is a direction, not a destination. It is
//!   accepted only if it reduces a merit function trading the objective
//!   against constraint violation — otherwise the step is shortened until it
//!   does.
//! - **Regularization driven by failure, not by structure.** In a convex QP
//!   the tiny fixed \\(\gamma\\) exists only to make a rank-deficient matrix
//!   factorizable. Here the Hessian of the Lagrangian can be genuinely
//!   indefinite, and the Newton direction then points at a saddle or a
//!   maximum. \\(\gamma\\) is raised — by orders of magnitude, repeatedly —
//!   until the step is usable, which pushes the direction toward gradient
//!   descent.
//! - **A gradual barrier.** The QP drives \\(\mu \to 0\\) as fast as
//!   Mehrotra's heuristic allows, because its model is exact. Here \\(\mu\\)
//!   is held while the current barrier subproblem is solved to a loose
//!   tolerance, then decreased. Dropping it too fast produces a step that is
//!   accurate for a problem nobody asked about.
//!
//! # What it shares
//!
//! The reformulation, the augmented system, the regularization and the
//! iterative refinement are the same as [`ipm`](super::ipm)'s — see that
//! module for the derivations, which are not repeated here. A slack per
//! inequality row, fixed columns rewritten as equality rows, and the
//! symmetric quasi-definite KKT matrix all carry over unchanged.
//!
//! The one inherited detail worth repeating is the **common step length**.
//! `ipm` takes a single \\(\alpha\\) for primal and dual whenever the
//! objective is quadratic, because a split step leaves an uncontracted
//! \\((\alpha_p - \alpha_d) Q \Delta u\\) term in the dual residual. Here the
//! Hessian is *always* present, so the step is always common.
//!
//! # Status
//!
//! A KKT point of a nonconvex problem is **locally** optimal. Unlike the
//! convex case there is no proof of global optimality and no certificate to
//! offer, so [`OptStatus::Optimal`] here means "satisfies the first-order
//! conditions", which is what every AC-OPF tool reports and what published
//! objectives are compared against.

use crate::opf::{OpfError, OptStatus};
use crate::sparse::RealFactorization;

/// A twice-differentiable problem.
///
/// Modelled on the interface IPOPT and MATPOWER's MIPS both use, because it
/// is the one that lets the solver ask for exactly what it needs and nothing
/// more — evaluations are the expensive part of a nonlinear solve, and a
/// coarser interface forces recomputation.
///
/// Sparsity patterns may change between calls; nothing is cached across them.
/// That is a deliberate simplification, not a design principle — see the note
/// on the KKT assembly in [`solve`].
pub trait NonlinearProblem {
    fn n_vars(&self) -> usize;
    fn n_constraints(&self) -> usize;

    /// Variable bounds. Infinite entries mean unbounded; equal entries fix the
    /// variable, which is handled without a barrier term.
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>);

    /// Constraint bounds. Equal entries make the row an equality.
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>);

    fn objective(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64]) -> Vec<f64>;
    fn constraints(&self, x: &[f64]) -> Vec<f64>;

    /// \\(\partial c_i/\partial x_j\\) as `(row, col, value)` triplets.
    /// Duplicates sum.
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)>;

    /// \\(\nabla^2 f(x) - \sum_i y_i \nabla^2 c_i(x)\\) — the Hessian of the
    /// Lagrangian in **this module's sign convention**, as full symmetric
    /// triplets.
    ///
    /// The minus sign is the trap. It follows from the Lagrangian being
    /// written \\(L = f - y^{\mathsf T} c\\) here, which is the convention that
    /// makes a row's multiplier read as \\(\partial f/\partial b\\) — the same
    /// choice, for the same reason, that
    /// [`dc`](super::dc) makes so its duals come out as prices. A caller
    /// negating `y` gets a Hessian that is wrong only where the constraints
    /// curve, so it converges on nearly-linear problems and fails on real
    /// ones.
    fn lagrangian_hessian(&self, x: &[f64], y: &[f64]) -> Vec<(usize, usize, f64)>;

    /// Where to start. A strictly interior point is strongly preferred; the
    /// solver pushes any violation inside the bounds but cannot invent a good
    /// initial guess.
    fn initial_point(&self) -> Vec<f64>;
}

#[derive(Clone, Copy, Debug)]
pub struct NlpOptions {
    /// Convergence tolerance on the scaled KKT error.
    pub tolerance: f64,
    pub max_iterations: usize,
    /// Starting barrier parameter.
    pub initial_barrier: f64,
    /// Fraction-to-boundary parameter.
    pub tau: f64,
    /// Armijo constant for the line search.
    pub armijo: f64,
    /// Smallest step the line search will accept before giving up and asking
    /// for more regularization.
    pub min_step: f64,
    /// Starting primal regularization, raised on demand.
    pub regularization: f64,
    pub dual_regularization: f64,
    pub refinement_rounds: usize,
}

impl Default for NlpOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 300,
            initial_barrier: 0.1,
            tau: 0.995,
            armijo: 1e-4,
            min_step: 1e-10,
            regularization: 1e-8,
            dual_regularization: 1e-8,
            refinement_rounds: 2,
        }
    }
}

/// A solved nonlinear program.
#[derive(Clone, Debug)]
pub struct NlpSolution {
    pub status: OptStatus,
    pub objective: f64,
    pub x: Vec<f64>,
    /// Constraint multipliers, in the convention documented on
    /// [`NonlinearProblem::lagrangian_hessian`].
    pub y: Vec<f64>,
    /// Reduced costs on the variables.
    pub z: Vec<f64>,
    pub iterations: usize,
    /// Largest constraint violation at the returned point. Reported alongside
    /// the objective because for a nonconvex problem the objective alone is
    /// not interpretable — a lower value at an infeasible point is not better.
    pub violation: f64,
}

/// The problem rewritten so every constraint is an equality and every bound
/// sits on a variable — the same two transformations
/// [`ipm`](super::ipm) documents.
struct Internal {
    n_x: usize,
    n_total: usize,
    m_eq: usize,
    /// Internal equality row -> original constraint row, or `None` for a row
    /// added to pin a fixed variable.
    source_row: Vec<Option<usize>>,
    /// Original inequality row -> its slack column.
    slack_of: Vec<Option<usize>>,
    /// Fixed-variable pinning rows, as `(internal row, variable, value)`.
    pinned: Vec<(usize, usize, f64)>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    c_lower: Vec<f64>,
    c_upper: Vec<f64>,
}

impl Internal {
    fn build(problem: &dyn NonlinearProblem) -> Self {
        let n_x = problem.n_vars();
        let m = problem.n_constraints();
        let (mut lower, mut upper) = problem.var_bounds();
        let (c_lower, c_upper) = problem.constraint_bounds();

        let mut n_total = n_x;
        let mut slack_of = vec![None; m];
        let mut source_row = Vec::with_capacity(m);
        for r in 0..m {
            source_row.push(Some(r));
            if c_lower[r] != c_upper[r] {
                slack_of[r] = Some(n_total);
                lower.push(c_lower[r]);
                upper.push(c_upper[r]);
                n_total += 1;
            }
        }

        let mut m_eq = m;
        let mut pinned = Vec::new();
        for k in 0..n_x {
            if lower[k] == upper[k] {
                pinned.push((m_eq, k, lower[k]));
                source_row.push(None);
                lower[k] = f64::NEG_INFINITY;
                upper[k] = f64::INFINITY;
                m_eq += 1;
            }
        }

        Self {
            n_x,
            n_total,
            m_eq,
            source_row,
            slack_of,
            pinned,
            lower,
            upper,
            c_lower,
            c_upper,
        }
    }

    /// The internal equality residual `ĉ(u)`.
    fn residual(&self, c: &[f64], u: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.m_eq];
        for (r, &source) in self.source_row.iter().enumerate() {
            if let Some(original) = source {
                out[r] = c[original]
                    - match self.slack_of[original] {
                        Some(slack) => u[slack],
                        None => self.c_lower[original],
                    };
            }
        }
        for &(row, k, value) in &self.pinned {
            out[row] = u[k] - value;
        }
        out
    }

    /// `∇ĉ(u)` in the internal space: the caller's Jacobian, plus `−1` in each
    /// inequality row's slack column, plus the pinning rows.
    fn jacobian(&self, base: &[(usize, usize, f64)]) -> Vec<(usize, usize, f64)> {
        let mut out: Vec<(usize, usize, f64)> = base.to_vec();
        for r in 0..self.slack_of.len() {
            if let Some(slack) = self.slack_of[r] {
                out.push((r, slack, -1.0));
            }
        }
        for &(row, k, _) in &self.pinned {
            out.push((row, k, 1.0));
        }
        out
    }
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn mat_vec(triplets: &[(usize, usize, f64)], x: &[f64], rows: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows];
    for &(r, c, v) in triplets {
        out[r] += v * x[c];
    }
    out
}

fn mat_transpose_vec(
    triplets: &[(usize, usize, f64)],
    y: &[f64],
    cols: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; cols];
    for &(r, c, v) in triplets {
        out[c] += v * y[r];
    }
    out
}

/// Solves a nonlinear program.
///
/// # The KKT assembly, and what is deliberately not optimized
///
/// The matrix is rebuilt and refactorized from triplets every iteration,
/// symbolic analysis included. For a nonlinear problem the *values* change
/// every iteration anyway, so only the symbolic half is wasted — the same
/// waste [`RealSparseSystem`](crate::sparse::RealSparseSystem) exists to
/// remove for Newton power flow, and the same fix would apply here once the
/// pattern is known to be stable. It is not done yet because correctness on
/// nonconvex problems was the phase's risk and premature caching would have
/// made every failure ambiguous between the algorithm and the cache.
pub fn solve(
    problem: &dyn NonlinearProblem,
    options: &NlpOptions,
) -> Result<NlpSolution, OpfError> {
    let p = Internal::build(problem);
    let (n_x, n_total, m_eq) = (p.n_x, p.n_total, p.m_eq);
    let dim = n_total + m_eq;

    let has_lower: Vec<bool> = p.lower.iter().map(|v| v.is_finite()).collect();
    let has_upper: Vec<bool> = p.upper.iter().map(|v| v.is_finite()).collect();
    let n_bounds =
        has_lower.iter().filter(|b| **b).count() + has_upper.iter().filter(|b| **b).count();

    // Start from the caller's point, pushed strictly inside every bound. A
    // point *on* a bound has no barrier value at all, so this is not a
    // refinement but a precondition.
    let mut u = vec![0.0; n_total];
    let start = problem.initial_point();
    u[..n_x].copy_from_slice(&start[..n_x]);
    {
        let c = problem.constraints(&u[..n_x]);
        for r in 0..p.slack_of.len() {
            if let Some(slack) = p.slack_of[r] {
                u[slack] = c[r];
            }
        }
    }
    for k in 0..n_total {
        let (lo, hi) = (p.lower[k], p.upper[k]);
        let margin = match (has_lower[k], has_upper[k]) {
            (true, true) => (1e-2 * (hi - lo)).min(0.1),
            _ => 1e-2,
        };
        if has_lower[k] && u[k] < lo + margin {
            u[k] = lo + margin;
        }
        if has_upper[k] && u[k] > hi - margin {
            u[k] = hi - margin;
        }
        if !u[k].is_finite() {
            u[k] = 0.0;
        }
    }

    let mut y = vec![0.0; m_eq];
    let mut z_l: Vec<f64> = has_lower.iter().map(|&h| if h { 1.0 } else { 0.0 }).collect();
    let mut z_u: Vec<f64> = has_upper.iter().map(|&h| if h { 1.0 } else { 0.0 }).collect();
    let mut mu = options.initial_barrier;
    let mut penalty = 1.0f64;

    let mut iterations = 0;
    for _ in 0..options.max_iterations {
        iterations += 1;

        let x = &u[..n_x];
        let f = problem.objective(x);
        let grad_x = problem.gradient(x);
        let c = problem.constraints(x);
        let jac_x = problem.jacobian(x);

        let a = p.jacobian(&jac_x);
        let r_p = p.residual(&c, &u);

        let mut grad = vec![0.0; n_total];
        grad[..n_x].copy_from_slice(&grad_x);

        let g: Vec<f64> =
            (0..n_total).map(|k| if has_lower[k] { u[k] - p.lower[k] } else { 1.0 }).collect();
        let t: Vec<f64> =
            (0..n_total).map(|k| if has_upper[k] { p.upper[k] - u[k] } else { 1.0 }).collect();

        let aty = mat_transpose_vec(&a, &y, n_total);
        let r_d: Vec<f64> =
            (0..n_total).map(|k| grad[k] - aty[k] - z_l[k] + z_u[k]).collect();

        let complementarity: f64 = (0..n_total)
            .map(|k| {
                (if has_lower[k] { g[k] * z_l[k] } else { 0.0 })
                    + (if has_upper[k] { t[k] * z_u[k] } else { 0.0 })
            })
            .sum();

        // Scaled KKT error, in the style IPOPT uses: dividing by the size of
        // the multipliers keeps the test meaningful when they are large, which
        // on an AC-OPF they routinely are.
        let s_d = (1.0 + norm_inf(&y).max(norm_inf(&z_l).max(norm_inf(&z_u)))) / 1.0;
        let error = (norm_inf(&r_d) / s_d)
            .max(norm_inf(&r_p))
            .max(complementarity / (n_bounds.max(1) as f64) / s_d);

        if error <= options.tolerance && mu <= options.tolerance {
            return Ok(finish(
                problem, &p, &u, &y, &z_l, &z_u, f, &c, iterations, OptStatus::Optimal,
            ));
        }

        // Decrease the barrier once the current subproblem is solved loosely.
        // Solving each one tightly would waste iterations on a barrier
        // parameter that is about to change.
        let sub_error = (norm_inf(&r_d) / s_d)
            .max(norm_inf(&r_p))
            .max((complementarity / (n_bounds.max(1) as f64) - mu).abs() / s_d);
        if sub_error <= 10.0 * mu {
            mu = (0.2 * mu).min(mu.powf(1.5)).max(options.tolerance * 0.1);
        }

        let hessian = problem.lagrangian_hessian(x, &y);

        // Try the step, raising regularization until one is usable. This
        // stands in for the inertia correction a symmetric indefinite
        // factorization would provide: an LU gives no inertia, so instead of
        // detecting a bad direction from the factorization we detect it from
        // the direction failing to be a descent direction, and respond the
        // same way.
        let mut gamma = options.regularization;
        let mut accepted = None;
        for _ in 0..12 {
            let mut kkt: Vec<(usize, usize, f64)> = Vec::new();
            for &(i, j, v) in &hessian {
                kkt.push((i, j, v));
            }
            for k in 0..n_total {
                let d = (if has_lower[k] { z_l[k] / g[k] } else { 0.0 })
                    + (if has_upper[k] { z_u[k] / t[k] } else { 0.0 });
                kkt.push((k, k, d + gamma));
            }
            for &(r, k, v) in &a {
                kkt.push((n_total + r, k, v));
                kkt.push((k, n_total + r, v));
            }
            for r in 0..m_eq {
                kkt.push((n_total + r, n_total + r, -options.dual_regularization));
            }

            let Some(factorization) = RealFactorization::new(dim, &kkt) else {
                gamma *= 100.0;
                continue;
            };

            // Barrier-perturbed right-hand side. No Mehrotra corrector: the
            // predictor's premise is that the model is exact, which is what
            // nonconvexity denies.
            let mut rhs = vec![0.0; dim];
            for k in 0..n_total {
                let mut rho = -r_d[k];
                if has_lower[k] {
                    rho -= (g[k] * z_l[k] - mu) / g[k];
                }
                if has_upper[k] {
                    rho += (t[k] * z_u[k] - mu) / t[k];
                }
                rhs[k] = rho;
            }
            for r in 0..m_eq {
                rhs[n_total + r] = -r_p[r];
            }

            let Some(mut d) = factorization.solve(&rhs) else {
                gamma *= 100.0;
                continue;
            };
            for _ in 0..options.refinement_rounds {
                let mut residual = rhs.clone();
                let du = &d[..n_total];
                let w = &d[n_total..];
                let hdu = mat_vec_symmetric(&hessian, du, n_total);
                let atw = mat_transpose_vec(&a, w, n_total);
                let adu = mat_vec(&a, du, m_eq);
                for k in 0..n_total {
                    let barrier = (if has_lower[k] { z_l[k] / g[k] } else { 0.0 })
                        + (if has_upper[k] { z_u[k] / t[k] } else { 0.0 });
                    residual[k] -= hdu[k] + barrier * du[k] + atw[k];
                }
                for r in 0..m_eq {
                    residual[n_total + r] -= adu[r];
                }
                let Some(correction) = factorization.solve(&residual) else { break };
                for i in 0..dim {
                    d[i] += correction[i];
                }
            }

            let du: Vec<f64> = d[..n_total].to_vec();
            let dy: Vec<f64> = (0..m_eq).map(|r| -d[n_total + r]).collect();
            if !du.iter().all(|v| v.is_finite()) || !dy.iter().all(|v| v.is_finite()) {
                gamma *= 100.0;
                continue;
            }

            let dz_l: Vec<f64> = (0..n_total)
                .map(|k| {
                    if has_lower[k] { (mu - g[k] * z_l[k] - z_l[k] * du[k]) / g[k] } else { 0.0 }
                })
                .collect();
            let dz_u: Vec<f64> = (0..n_total)
                .map(|k| {
                    if has_upper[k] { (mu - t[k] * z_u[k] + z_u[k] * du[k]) / t[k] } else { 0.0 }
                })
                .collect();

            // The merit function's penalty must dominate the multipliers, or
            // reducing it would not imply reducing infeasibility.
            let needed = norm_inf(&y) + 10.0;
            if penalty < needed {
                penalty = needed;
            }

            let directional = (0..n_total)
                .map(|k| {
                    let mut term = grad[k];
                    if has_lower[k] {
                        term -= mu / g[k];
                    }
                    if has_upper[k] {
                        term += mu / t[k];
                    }
                    term * du[k]
                })
                .sum::<f64>()
                - penalty * r_p.iter().map(|v| v.abs()).sum::<f64>();

            if directional > 0.0 {
                // Not a descent direction: the Hessian is indefinite along it.
                gamma *= 100.0;
                continue;
            }

            let alpha_max = fraction_to_boundary(
                n_total, &has_lower, &has_upper, &g, &t, &du, options.tau,
            );
            let merit_here =
                merit(problem, &p, &u, mu, penalty, &has_lower, &has_upper, n_x);

            let mut alpha = alpha_max;
            let mut ok = false;
            while alpha > options.min_step {
                let trial: Vec<f64> =
                    (0..n_total).map(|k| u[k] + alpha * du[k]).collect();
                let inside = (0..n_total).all(|k| {
                    (!has_lower[k] || trial[k] > p.lower[k])
                        && (!has_upper[k] || trial[k] < p.upper[k])
                });
                if inside {
                    let m_trial = merit(
                        problem, &p, &trial, mu, penalty, &has_lower, &has_upper, n_x,
                    );
                    if m_trial.is_finite()
                        && m_trial <= merit_here + options.armijo * alpha * directional
                    {
                        ok = true;
                        break;
                    }
                }
                alpha *= 0.5;
            }

            if !ok {
                gamma *= 100.0;
                continue;
            }
            accepted = Some((du, dy, dz_l, dz_u, alpha));
            break;
        }

        let Some((du, dy, dz_l, dz_u, alpha)) = accepted else {
            return Ok(finish(
                problem,
                &p,
                &u,
                &y,
                &z_l,
                &z_u,
                f,
                &c,
                iterations,
                OptStatus::Other("no acceptable step; the problem may be infeasible here"),
            ));
        };

        for k in 0..n_total {
            u[k] += alpha * du[k];
            z_l[k] += alpha * dz_l[k];
            z_u[k] += alpha * dz_u[k];
        }
        for r in 0..m_eq {
            y[r] += alpha * dy[r];
        }

        // Keep the bound multipliers in a sane band around μ/slack. Without
        // this they drift over many iterations until the barrier term they
        // scale stops resembling the constraint it came from — a slow failure
        // that looks like stalling rather than like a bug.
        for k in 0..n_total {
            if has_lower[k] {
                let center = mu / g[k];
                z_l[k] = z_l[k].clamp(center / 1e10, center * 1e10).max(1e-14);
            }
            if has_upper[k] {
                let center = mu / t[k];
                z_u[k] = z_u[k].clamp(center / 1e10, center * 1e10).max(1e-14);
            }
        }
    }

    let x = &u[..n_x];
    let f = problem.objective(x);
    let c = problem.constraints(x);
    Ok(finish(
        problem,
        &p,
        &u,
        &y,
        &z_l,
        &z_u,
        f,
        &c,
        iterations,
        OptStatus::Other("iteration limit"),
    ))
}

/// `H x` for a full symmetric triplet list.
fn mat_vec_symmetric(
    triplets: &[(usize, usize, f64)],
    x: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for &(r, c, v) in triplets {
        out[r] += v * x[c];
    }
    out
}

/// The largest step keeping every bound slack strictly positive, scaled by
/// `tau`. Infinity when nothing blocks, so an unrestricted direction is taken
/// in full — the same rule, for the same reason,
/// [`ipm`](super::ipm)'s `step_lengths` documents.
fn fraction_to_boundary(
    n_total: usize,
    has_lower: &[bool],
    has_upper: &[bool],
    g: &[f64],
    t: &[f64],
    du: &[f64],
    tau: f64,
) -> f64 {
    let mut alpha = f64::INFINITY;
    for k in 0..n_total {
        if has_lower[k] && du[k] < 0.0 {
            alpha = alpha.min(-g[k] / du[k]);
        }
        if has_upper[k] && du[k] > 0.0 {
            alpha = alpha.min(t[k] / du[k]);
        }
    }
    (tau * alpha).min(1.0)
}

/// Objective plus barrier plus penalized constraint violation.
///
/// The \\(\ell_1\\) penalty is what lets the line search compare a step that
/// improves the objective while worsening feasibility against one that does
/// the reverse. With `penalty` above the largest multiplier — enforced before
/// each search — a reduction here implies progress on the real problem rather
/// than on one of its two halves.
#[allow(clippy::too_many_arguments)]
fn merit(
    problem: &dyn NonlinearProblem,
    p: &Internal,
    u: &[f64],
    mu: f64,
    penalty: f64,
    has_lower: &[bool],
    has_upper: &[bool],
    n_x: usize,
) -> f64 {
    let x = &u[..n_x];
    let mut value = problem.objective(x);
    if !value.is_finite() {
        return f64::INFINITY;
    }
    for k in 0..u.len() {
        if has_lower[k] {
            let s = u[k] - p.lower[k];
            if s <= 0.0 {
                return f64::INFINITY;
            }
            value -= mu * s.ln();
        }
        if has_upper[k] {
            let s = p.upper[k] - u[k];
            if s <= 0.0 {
                return f64::INFINITY;
            }
            value -= mu * s.ln();
        }
    }
    let c = problem.constraints(x);
    let r = p.residual(&c, u);
    value + penalty * r.iter().map(|v| v.abs()).sum::<f64>()
}

#[allow(clippy::too_many_arguments)]
fn finish(
    problem: &dyn NonlinearProblem,
    p: &Internal,
    u: &[f64],
    y: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    objective: f64,
    c: &[f64],
    iterations: usize,
    status: OptStatus,
) -> NlpSolution {
    let n_x = p.n_x;
    let m = problem.n_constraints();

    let mut violation = 0.0f64;
    for r in 0..m {
        violation = violation
            .max((p.c_lower[r] - c[r]).max(0.0))
            .max((c[r] - p.c_upper[r]).max(0.0));
    }

    NlpSolution {
        status,
        objective,
        x: u[..n_x].to_vec(),
        y: (0..m)
            .map(|r| p.source_row.iter().position(|s| *s == Some(r)).map(|i| y[i]).unwrap_or(0.0))
            .collect(),
        z: (0..n_x).map(|k| z_l[k] - z_u[k]).collect(),
        iterations,
        violation,
    }
}
