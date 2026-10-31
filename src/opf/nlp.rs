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
//! # The algorithm
//!
//! IPOPT's, as Wächter & Biegler (2006, Math. Prog. 106) describe it and
//! IPOPT sets its defaults, in the parts that decide whether an AC-OPF
//! converges from a flat start. Each was added because a PGLib case failed
//! without it (see `examples/opf_compare.rs`, which holds the result against
//! IPOPT on the same model):
//!
//! - **Scaling** (`Scaled`): the objective and every constraint row scaled
//!   so no gradient exceeds 100 at the start. Unscaled, a 1e5 $/h cost
//!   drowns the barrier and inflates the multipliers; six of grid-bench's
//!   eleven PGLib cases stalled for that alone.
//! - **Inertia correction** (`factorize_with_inertia`): the KKT matrix is
//!   factorized by an `LDLᵀ` that reports its inertia
//!   ([`SymmetricFactorization`]), and the Hessian block regularized by the
//!   smallest δw that gives it `n` positive and `m` negative eigenvalues,
//!   the condition for the step to minimize the barrier model.
//! - **A filter line search** (`Filter`) on the violation θ and the
//!   barrier objective φ, with up to four second-order corrections when the
//!   full step does not reduce θ. A merit function's penalty weight rejected
//!   good Newton steps on the PEGASE cases (the Maratos effect).
//! - **Feasibility restoration** (`restoration`) when the line search
//!   finds no step: Gauss–Newton steps toward feasibility until the filter
//!   accepts a point.
//! - **A separate step for the bound multipliers**, the largest keeping them
//!   positive, while the constraint multipliers follow the primal step.
//! - **A monotone barrier**: μ is held while its subproblem is solved to a
//!   loose tolerance, then decreased superlinearly.
//!
//! On case9241_pegase the path was compared with IPOPT's iteration by
//! iteration, on the same model. These made it IPOPT's, step for step, up to
//! rounding, and made that case solve (it ended over a 1 MVA branch's limits
//! before):
//!
//! - IPOPT's **bound push**, relative to each bound's size (`State::start`);
//! - IPOPT's **damping** of variables bounded on one side (`damping`);
//! - the **scaling** computed without fixed variables, which IPOPT removes;
//! - **δc only on equality rows**, so the step is IPOPT's unperturbed one
//!   (`Kkt`), and refinement to IPOPT's residual ratio, 1e-10;
//! - the **switching condition** without the θ_min test, which only decides
//!   whether Armijo is required (`Filter::accepts`);
//! - an **acceptable** stop when the line search fails near feasibility at
//!   an acceptable point, as IPOPT's restoration ends there.
//!
//! Not (yet) IPOPT's: the least-squares start of the multipliers (tried; it
//! changed nothing on case9241_pegase and broke the piecewise-cost case5_pjm),
//! the adaptive barrier, the watchdog, and a restoration phase that is an
//! interior point of its own. Algorithm NCL (`NlpOptions::ncl`) is an
//! opt-in outer loop for degenerate problems.
//!
//! # What it shares
//!
//! The reformulation and the augmented system are [`ipm`](super::ipm)'s — see
//! that module for the derivations, which are not repeated here. A slack per
//! inequality row, fixed columns rewritten as equality rows, and the
//! symmetric KKT matrix carry over unchanged; the factorization does not
//! ([`SymmetricFactorization`] in place of `ipm`'s LU, for the inertia).
//!
//! Unlike `ipm`, the primal step and the bound multipliers' step differ:
//! the multipliers set the barrier curvature, and tied to a primal step cut
//! short by a bound they would barely move (an epigraph cost variable,
//! unbounded and without curvature, blocked every step of the piecewise-cost
//! case5_pjm that way).
//!
//! # Status
//!
//! A KKT point of a nonconvex problem is **locally** optimal. Unlike the
//! convex case there is no proof of global optimality and no certificate to
//! offer, so [`OptStatus::Optimal`] here means "satisfies the first-order
//! conditions", which is what every AC-OPF tool reports and what published
//! objectives are compared against.

use crate::ldlt::{Inertia, SymmetricFactorization};
use crate::opf::{OpfError, OptStatus};

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
    /// IPOPT's `acceptable_tol` and `acceptable_iter`: a point whose error
    /// stays below this for that many iterations in a row, or where the
    /// solver would otherwise stop without a step, is reported solved, as
    /// IPOPT reports "solved to acceptable level" (which `opf::ipopt` maps
    /// to `Optimal` too). Near the optimum the error can sit at a floor of
    /// rounding noise above `tolerance`.
    pub acceptable_tolerance: f64,
    pub acceptable_iterations: usize,
    pub max_iterations: usize,
    /// Starting barrier parameter.
    pub initial_barrier: f64,
    /// Fraction-to-boundary parameter: its lower bound τ_min, raised to
    /// `1 − μ` as the barrier vanishes (IPOPT's `tau_min`).
    pub tau: f64,
    /// Armijo constant for the line search.
    pub armijo: f64,
    /// Smallest step the line search will accept before giving up and asking
    /// for more regularization.
    pub min_step: f64,
    /// The first \\(\\delta_w\\) tried when the KKT matrix has the wrong
    /// inertia (IPOPT's `first_hessian_perturbation`); see
    /// `factorize_with_inertia`.
    pub regularization: f64,
    pub dual_regularization: f64,
    /// The most rounds of iterative refinement per solve; it stops earlier
    /// once IPOPT's residual ratio is below 1e-10.
    pub refinement_rounds: usize,
    /// Solve by Algorithm NCL around the interior point (see `ncl`): a
    /// residual on every constraint row, removed by multiplier updates over
    /// a sequence of subproblems. For problems whose active constraints are
    /// degenerate.
    pub ncl: bool,
    /// Scale the objective and every constraint row so that no gradient
    /// exceeds 100 at the starting point (IPOPT's `gradient-based` scaling);
    /// see `Scaled`.
    pub scaling: bool,
    /// One line per iteration on stderr, in IPOPT's columns (objective,
    /// primal and dual infeasibility, log10 of the barrier, step size,
    /// log10 of the regularization, step length, line-search trials), so a
    /// stalled solve can be read against IPOPT's own log of the same model.
    pub log: bool,
}

impl Default for NlpOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            acceptable_tolerance: 1e-6,
            acceptable_iterations: 15,
            max_iterations: 300,
            initial_barrier: 0.1,
            tau: 0.99,
            armijo: 1e-4,
            min_step: 1e-10,
            regularization: 1e-4,
            dual_regularization: 1e-8,
            refinement_rounds: 10,
            scaling: true,
            ncl: false,
            log: false,
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

fn mat_transpose_vec(triplets: &[(usize, usize, f64)], y: &[f64], cols: usize) -> Vec<f64> {
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
    let inner = |problem: &dyn NonlinearProblem| {
        if options.ncl {
            ncl(problem, options)
        } else {
            solve_scaled(problem, options)
        }
    };
    if !options.scaling {
        return inner(problem);
    }
    let scaled = Scaled::new(problem);
    let mut solution = inner(&scaled)?;
    // Back to the caller's units: the objective and every multiplier carry
    // the objective's scale, a constraint multiplier its row's too.
    solution.objective = problem.objective(&solution.x);
    for (r, y) in solution.y.iter_mut().enumerate() {
        *y *= scaled.rows[r] / scaled.objective;
    }
    for z in solution.z.iter_mut() {
        *z /= scaled.objective;
    }
    let c = problem.constraints(&solution.x);
    let (lo, hi) = problem.constraint_bounds();
    solution.violation = (0..c.len())
        .map(|r| (lo[r] - c[r]).max(c[r] - hi[r]).max(0.0))
        .fold(0.0, f64::max);
    Ok(solution)
}

/// Algorithm NCL (Ma, Judd, Orban & Saunders, 2018; MadNCL, Montoison et
/// al., arXiv:2510.05885, is its GPU implementation on MadNLP).
///
/// Each subproblem gives every constraint row a free residual `r`,
///
/// \\[ \min_{x,r}\ f(x) + y_k^{\mathsf T} r + \tfrac{\rho_k}{2}\lVert r \rVert^2
///    \quad\text{s.t.}\quad c_L \le c(x) + r \le c_U,\ x_L \le x \le x_U, \\]
///
/// whose constraint Jacobian `[J I]` has full row rank whatever the active
/// set. Its row multipliers are \\(\lambda = y_k + \rho_k r\\) (stationarity in
/// `r`, in this module's sign convention), which is NCL's update: when the
/// residual is small enough, `y` takes them and the next subproblem asks
/// for less; when not, ρ grows tenfold. It ends when `‖r‖∞ ≤ tolerance`.
///
/// Why: when active constraints are nearly dependent (on case9241_pegase,
/// the limits at both ends of a 1 MVA branch, whose gradients vanish near
/// zero flow), the plain interior point holds both over their limits and
/// cannot move off them. IPOPT solves that case with a multiplier of 1.6e8
/// on one end and the other inactive.
///
/// The outer loop is MadNCL's (`src/solver.jl`): one interior point
/// throughout, each outer iteration a barrier subproblem at a *fixed* μ,
/// solved to `100 μ^1.05` from where the last one ended ([`run`] with
/// [`Barrier::Fixed`]). A subproblem that converges with `‖r‖∞ ≤ η` updates
/// `y` and lowers μ (`min(μ^1.99, 0.2 μ)`, down to 1e-9); otherwise ρ grows
/// tenfold. `η = min(μ^1.1, 0.1 μ)` ties the residual allowed to the barrier,
/// so the residuals vanish as the barrier does. ρ starts at 1e4: MadNCL's
/// 100 on gradients scaled to 1, here scaled to 100 (`Scaled`). MadNCL's
/// extrapolation step is not (yet) here.
fn ncl(problem: &dyn NonlinearProblem, options: &NlpOptions) -> Result<NlpSolution, OpfError> {
    const MAX_OUTER: usize = 60;
    const RHO_MAX: f64 = 1e12;
    const MU_MIN: f64 = 1e-9;
    let (n, m) = (problem.n_vars(), problem.n_constraints());
    let mut sub = Ncl {
        inner: problem,
        y: vec![0.0; m],
        rho: 1e4,
        start: problem
            .initial_point()
            .into_iter()
            .chain(std::iter::repeat_n(0.0, m))
            .collect(),
    };
    let p = Internal::build(&sub);
    let mut state = State::start(&sub, &p, options);
    let mut mu = options.initial_barrier;
    let mut eta = 0.1f64;
    let mut last = None;
    for k in 1..=MAX_OUTER {
        let tolerance = 100.0 * mu.powf(1.05);
        let s = run(
            &sub,
            &p,
            &mut state,
            options,
            Barrier::Fixed { mu, tolerance },
        )?;
        let converged = s.status == OptStatus::Optimal;
        let residual = norm_inf(&state.u[n..n + m]);
        if options.log {
            eprintln!(
                "NCL {k:2}: μ {mu:.1e}  ρ {:.0e}  η {eta:.1e}  ‖r‖∞ {residual:.2e}  {} iterations so far  {:?}",
                sub.rho, state.iterations, s.status
            );
        }
        if converged && residual <= options.tolerance && mu <= options.tolerance {
            return Ok(unwrap_ncl(problem, s, n));
        }
        if converged && residual <= eta.max(options.tolerance) {
            sub.y.clone_from(&s.y);
            mu = mu.powf(1.99).min(0.2 * mu).max(MU_MIN);
        } else {
            sub.rho *= 10.0;
            if sub.rho > RHO_MAX {
                return Ok(NlpSolution {
                    status: OptStatus::Infeasible,
                    ..unwrap_ncl(problem, s, n)
                });
            }
        }
        eta = mu.powf(1.1).min(0.1 * mu);
        last = Some(s);
    }
    Ok(NlpSolution {
        status: OptStatus::Other("NCL: outer iteration limit"),
        ..unwrap_ncl(problem, last.expect("at least one outer iteration"), n)
    })
}

/// A subproblem's solution, back in the original problem's variables.
fn unwrap_ncl(problem: &dyn NonlinearProblem, s: NlpSolution, n: usize) -> NlpSolution {
    let x = s.x[..n].to_vec();
    NlpSolution {
        objective: problem.objective(&x),
        x,
        z: s.z[..n].to_vec(),
        ..s
    }
}

/// An NCL subproblem: the original variables, then one residual per row.
struct Ncl<'a> {
    inner: &'a dyn NonlinearProblem,
    y: Vec<f64>,
    rho: f64,
    start: Vec<f64>,
}

impl NonlinearProblem for Ncl<'_> {
    fn n_vars(&self) -> usize {
        self.inner.n_vars() + self.inner.n_constraints()
    }
    fn n_constraints(&self) -> usize {
        self.inner.n_constraints()
    }
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.inner.n_constraints();
        let (mut lo, mut hi) = self.inner.var_bounds();
        lo.extend(std::iter::repeat_n(f64::NEG_INFINITY, m));
        hi.extend(std::iter::repeat_n(f64::INFINITY, m));
        (lo, hi)
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        self.inner.constraint_bounds()
    }
    fn objective(&self, x: &[f64]) -> f64 {
        let n = self.inner.n_vars();
        let r = &x[n..];
        self.inner.objective(&x[..n])
            + r.iter()
                .zip(&self.y)
                .map(|(r, y)| y * r + 0.5 * self.rho * r * r)
                .sum::<f64>()
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let n = self.inner.n_vars();
        let mut g = self.inner.gradient(&x[..n]);
        g.extend(x[n..].iter().zip(&self.y).map(|(r, y)| y + self.rho * r));
        g
    }
    fn constraints(&self, x: &[f64]) -> Vec<f64> {
        let n = self.inner.n_vars();
        let mut c = self.inner.constraints(&x[..n]);
        for (ci, r) in c.iter_mut().zip(&x[n..]) {
            *ci += r;
        }
        c
    }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.inner.n_vars();
        let mut j = self.inner.jacobian(&x[..n]);
        j.extend((0..self.inner.n_constraints()).map(|r| (r, n + r, 1.0)));
        j
    }
    fn lagrangian_hessian(&self, x: &[f64], y: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.inner.n_vars();
        let mut h = self.inner.lagrangian_hessian(&x[..n], y);
        h.extend((0..self.inner.n_constraints()).map(|r| (n + r, n + r, self.rho)));
        h
    }
    fn initial_point(&self) -> Vec<f64> {
        self.start.clone()
    }
}

/// IPOPT's `gradient-based` scaling (Wächter & Biegler 2006, §3.8): the
/// objective and each constraint row multiplied by `min(1, 100 / ‖∇‖∞)` of
/// their gradient at the starting point, so no gradient there exceeds 100.
///
/// Why it matters on an AC-OPF: costs run to 1e5 $/h with gradients of
/// thousands per p.u., while the barrier parameter starts at 0.1. Unscaled,
/// the barrier terms are noise beside the cost, the multipliers come out in
/// the thousands, and so does the indefiniteness of `y·∇²c`, which the
/// inertia correction then fights with a δw of 1e5 on every iteration.
///
/// The problem's Hessian callback serves the scaled problem unchanged:
/// \\( d_f \nabla^2 f - \sum_i \tilde y_i d_i \nabla^2 c_i
///    = d_f\, H(x,\ \tilde y \circ d / d_f) \\).
struct Scaled<'a> {
    inner: &'a dyn NonlinearProblem,
    objective: f64,
    rows: Vec<f64>,
}

impl<'a> Scaled<'a> {
    const MAX_GRADIENT: f64 = 100.0;
    const MIN_FACTOR: f64 = 1e-8;

    /// Fixed variables (equal bounds) are left out of every gradient norm:
    /// IPOPT removes them from the problem before it scales it.
    fn new(inner: &'a dyn NonlinearProblem) -> Self {
        let x0 = inner.initial_point();
        let (lower, upper) = inner.var_bounds();
        let free: Vec<bool> = (0..x0.len()).map(|k| lower[k] != upper[k]).collect();
        let factor = |norm: f64| (Self::MAX_GRADIENT / norm).clamp(Self::MIN_FACTOR, 1.0);
        let gradient = inner.gradient(&x0);
        let objective = factor(
            (0..x0.len())
                .filter(|&k| free[k])
                .map(|k| gradient[k].abs())
                .fold(0.0, f64::max),
        );
        let mut row_norm = vec![0.0f64; inner.n_constraints()];
        for (r, k, v) in inner.jacobian(&x0) {
            if free[k] {
                row_norm[r] = row_norm[r].max(v.abs());
            }
        }
        Self {
            inner,
            objective,
            rows: row_norm.into_iter().map(factor).collect(),
        }
    }
}

impl NonlinearProblem for Scaled<'_> {
    fn n_vars(&self) -> usize {
        self.inner.n_vars()
    }
    fn n_constraints(&self) -> usize {
        self.inner.n_constraints()
    }
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        self.inner.var_bounds()
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let (mut lo, mut hi) = self.inner.constraint_bounds();
        for r in 0..lo.len() {
            lo[r] *= self.rows[r];
            hi[r] *= self.rows[r];
        }
        (lo, hi)
    }
    fn objective(&self, x: &[f64]) -> f64 {
        self.objective * self.inner.objective(x)
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        self.inner
            .gradient(x)
            .into_iter()
            .map(|g| self.objective * g)
            .collect()
    }
    fn constraints(&self, x: &[f64]) -> Vec<f64> {
        let mut c = self.inner.constraints(x);
        for (r, v) in c.iter_mut().enumerate() {
            *v *= self.rows[r];
        }
        c
    }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut j = self.inner.jacobian(x);
        for e in j.iter_mut() {
            e.2 *= self.rows[e.0];
        }
        j
    }
    fn lagrangian_hessian(&self, x: &[f64], y: &[f64]) -> Vec<(usize, usize, f64)> {
        let y: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(r, v)| v * self.rows[r] / self.objective)
            .collect();
        let mut h = self.inner.lagrangian_hessian(x, &y);
        for e in h.iter_mut() {
            e.2 *= self.objective;
        }
        h
    }
    fn initial_point(&self) -> Vec<f64> {
        self.inner.initial_point()
    }
}

fn solve_scaled(
    problem: &dyn NonlinearProblem,
    options: &NlpOptions,
) -> Result<NlpSolution, OpfError> {
    let p = Internal::build(problem);
    let mut state = State::start(problem, &p, options);
    run(problem, &p, &mut state, options, Barrier::Schedule)
}

/// Where the barrier parameter comes from in [`run`].
#[derive(Clone, Copy)]
enum Barrier {
    /// IPOPT's monotone schedule, to `tolerance`: a whole solve.
    Schedule,
    /// Held at `mu` until the barrier subproblem's error is below
    /// `tolerance`: one subproblem, as NCL's outer loop asks (MadNCL fixes
    /// MadNLP's μ the same way).
    Fixed { mu: f64, tolerance: f64 },
}

/// The interior point's state between calls of [`run`]: the iterate, its
/// multipliers, the barrier and the inertia correction's memory. Kept so an
/// outer loop (NCL) can resume where the last subproblem ended.
struct State {
    u: Vec<f64>,
    y: Vec<f64>,
    z_l: Vec<f64>,
    z_u: Vec<f64>,
    mu: f64,
    delta_last: f64,
    iterations: usize,
}

impl State {
    /// The caller's point, pushed strictly inside every bound (a point *on*
    /// a bound has no barrier value at all), multipliers `y = 0`, `z = 1`.
    fn start(problem: &dyn NonlinearProblem, p: &Internal, options: &NlpOptions) -> Self {
        let (n_x, n_total, m_eq) = (p.n_x, p.n_total, p.m_eq);
        let has_lower: Vec<bool> = p.lower.iter().map(|v| v.is_finite()).collect();
        let has_upper: Vec<bool> = p.upper.iter().map(|v| v.is_finite()).collect();

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
            // IPOPT's push (`bound_push` = `bound_frac` = 1e-2, the same for
            // slacks): at least min(κ₁ max(1, |bound|), κ₂ (u − l)) inside
            // each bound, relative to the bound's size. A flat 1e-2 put the
            // slacks of case9241_pegase's large branch limits at another
            // starting point, and its path elsewhere from the first step.
            const KAPPA_1: f64 = 1e-2;
            const KAPPA_2: f64 = 1e-2;
            let (lo, hi) = (p.lower[k], p.upper[k]);
            let width = hi - lo; // infinite unless both bounds are finite
            if has_lower[k] {
                let push = (KAPPA_1 * lo.abs().max(1.0)).min(KAPPA_2 * width);
                if u[k] < lo + push {
                    u[k] = lo + push;
                }
            }
            if has_upper[k] {
                let push = (KAPPA_1 * hi.abs().max(1.0)).min(KAPPA_2 * width);
                if u[k] > hi - push {
                    u[k] = hi - push;
                }
            }
            if !u[k].is_finite() {
                u[k] = 0.0;
            }
        }

        let z_l: Vec<f64> = has_lower
            .iter()
            .map(|&h| if h { 1.0 } else { 0.0 })
            .collect();
        let z_u: Vec<f64> = has_upper
            .iter()
            .map(|&h| if h { 1.0 } else { 0.0 })
            .collect();
        Self {
            u,
            y: vec![0.0; m_eq],
            z_l,
            z_u,
            mu: options.initial_barrier,
            delta_last: 0.0,
            iterations: 0,
        }
    }
}

/// The interior point's iterations from `state` on, until the barrier
/// (`Barrier`) says it is done or `options.max_iterations` more have run;
/// `state` is left at the last iterate.
fn run(
    problem: &dyn NonlinearProblem,
    p: &Internal,
    state: &mut State,
    options: &NlpOptions,
    barrier: Barrier,
) -> Result<NlpSolution, OpfError> {
    let (n_x, n_total, m_eq) = (p.n_x, p.n_total, p.m_eq);
    let dim = n_total + m_eq;
    let has_lower: Vec<bool> = p.lower.iter().map(|v| v.is_finite()).collect();
    let has_upper: Vec<bool> = p.upper.iter().map(|v| v.is_finite()).collect();
    let n_bounds =
        has_lower.iter().filter(|b| **b).count() + has_upper.iter().filter(|b| **b).count();

    let mut u = std::mem::take(&mut state.u);
    let mut y = std::mem::take(&mut state.y);
    let mut z_l = std::mem::take(&mut state.z_l);
    let mut z_u = std::mem::take(&mut state.z_u);
    let mut delta_last = state.delta_last;
    let mut iterations = state.iterations;
    let mut mu = match barrier {
        Barrier::Fixed { mu, .. } => mu,
        Barrier::Schedule => state.mu,
    };
    let mut filter = Filter::default();
    let mut acceptable_streak = 0usize;
    let mut error = f64::INFINITY;
    let mut first = true;

    let status = 'solve: {
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

            let g: Vec<f64> = (0..n_total)
                .map(|k| if has_lower[k] { u[k] - p.lower[k] } else { 1.0 })
                .collect();
            let t: Vec<f64> = (0..n_total)
                .map(|k| if has_upper[k] { p.upper[k] - u[k] } else { 1.0 })
                .collect();

            let aty = mat_transpose_vec(&a, &y, n_total);
            let r_d: Vec<f64> = (0..n_total)
                .map(|k| grad[k] - aty[k] - z_l[k] + z_u[k])
                .collect();

            // The worst complementarity pair against a target (0 for the
            // problem, μ for the barrier subproblem): a max-norm, as IPOPT's,
            // since an average lets one far-off pair hide behind the rest.
            let complementarity = |target: f64| -> f64 {
                (0..n_total)
                    .map(|k| {
                        let lo = if has_lower[k] {
                            (g[k] * z_l[k] - target).abs()
                        } else {
                            0.0
                        };
                        let hi = if has_upper[k] {
                            (t[k] * z_u[k] - target).abs()
                        } else {
                            0.0
                        };
                        lo.max(hi)
                    })
                    .fold(0.0, f64::max)
            };

            // The scaled KKT error, IPOPT's (Wächter & Biegler 2006, eq. 5): the
            // dual residual divided by the *average* multiplier size, and only
            // once that exceeds s_max = 100. Dividing by the largest multiplier
            // instead lets one large bound multiplier make the dual error look
            // small, and the barrier fall before its subproblem is solved
            // (case9241_pegase then stalled at μ = 2.5e-9 with the dual error
            // stuck at 1e-2).
            const S_MAX: f64 = 100.0;
            let z_sum = norm_1(&z_l) + norm_1(&z_u);
            let s_d = ((norm_1(&y) + z_sum) / ((m_eq + n_bounds).max(1) as f64)).max(S_MAX) / S_MAX;
            let s_c = (z_sum / (n_bounds.max(1) as f64)).max(S_MAX) / S_MAX;
            error = (norm_inf(&r_d) / s_d)
                .max(norm_inf(&r_p))
                .max(complementarity(0.0) / s_c);
            acceptable_streak = if error <= options.acceptable_tolerance {
                acceptable_streak + 1
            } else {
                0
            };

            let sub_error = (norm_inf(&r_d) / s_d)
                .max(norm_inf(&r_p))
                .max(complementarity(mu) / s_c);
            match barrier {
                Barrier::Schedule => {
                    if (error <= options.tolerance && mu <= options.tolerance)
                        || acceptable_streak >= options.acceptable_iterations
                    {
                        break 'solve OptStatus::Optimal;
                    }
                    // Decrease the barrier once the current subproblem is solved
                    // loosely. Solving each one tightly would waste iterations on
                    // a barrier parameter that is about to change.
                    if sub_error <= 10.0 * mu {
                        mu = (0.2 * mu).min(mu.powf(1.5)).max(options.tolerance * 0.1);
                        // A new barrier problem: the old filter does not apply.
                        filter.reset();
                    }
                }
                Barrier::Fixed { tolerance, .. } => {
                    if sub_error <= tolerance {
                        break 'solve OptStatus::Optimal;
                    }
                }
            }
            // The violation bounds from the first point of this run (MadNCL
            // resets them, and the filter, for every subproblem the same way).
            if first {
                filter.start(norm_1(&r_p));
                first = false;
            }

            // IPOPT's fraction-to-boundary parameter, τ = max(τ_min, 1 − μ): close
            // to 1 as the barrier vanishes, so the last steps are not held 0.5%
            // short of every bound they approach.
            let tau = options.tau.max(1.0 - mu);

            // Rows past the problem's own are the pinning rows, linear in u.
            let hessian = problem.lagrangian_hessian(x, &y[..problem.n_constraints()]);

            let sigma: Vec<f64> = (0..n_total)
                .map(|k| {
                    (if has_lower[k] { z_l[k] / g[k] } else { 0.0 })
                        + (if has_upper[k] { z_u[k] / t[k] } else { 0.0 })
                })
                .collect();

            // The step, with IPOPT's inertia correction (`factorize_with_inertia`):
            // the Hessian block is regularized by the smallest δw that gives the
            // KKT matrix the inertia of a minimization. A step of correct inertia
            // can still find no progress in the line search; then `floor` forces
            // a larger δw, pushing the direction toward steepest descent.
            let mut floor = 0.0f64;
            let mut gamma = 0.0f64;
            let mut accepted = None;
            let mut trials = 0usize;
            let mut restore = false;
            for _ in 0..12 {
                let Some((factorization, delta)) = factorize_with_inertia(
                    &hessian,
                    &a,
                    &sigma,
                    p,
                    options,
                    floor,
                    &mut delta_last,
                ) else {
                    break;
                };
                gamma = delta;

                // Barrier-perturbed right-hand side. No Mehrotra corrector: the
                // predictor's premise is that the model is exact, which is what
                // nonconvexity denies.
                let mut rhs = vec![0.0; dim];
                for k in 0..n_total {
                    let mut rho = -r_d[k] - mu * damping(k, &has_lower, &has_upper);
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

                // A solve with iterative refinement against the system as
                // regularized by δw (δc is only there to make it factorizable).
                // IPOPT's residual ratio, ‖r‖∞ / (‖K‖∞ ‖d‖∞ + ‖rhs‖∞), with ‖K‖∞
                // taken as the largest entry: refinement stops below 1e-10.
                let k_norm = hessian
                    .iter()
                    .chain(a.iter())
                    .map(|e| e.2.abs())
                    .chain(sigma.iter().map(|s| s + gamma))
                    .fold(0.0f64, f64::max);
                let solve = |rhs: &[f64]| -> Option<Vec<f64>> {
                    let mut d = factorization.solve(rhs)?;
                    for _ in 0..options.refinement_rounds {
                        let mut residual = rhs.to_vec();
                        let du = &d[..n_total];
                        let w = &d[n_total..];
                        let hdu = mat_vec_symmetric(&hessian, du, n_total);
                        let atw = mat_transpose_vec(&a, w, n_total);
                        let adu = mat_vec(&a, du, m_eq);
                        for k in 0..n_total {
                            residual[k] -= hdu[k] + (sigma[k] + gamma) * du[k] + atw[k];
                        }
                        // Against IPOPT's system, δc = 0: `Kkt` puts δc only
                        // on equality rows, where refinement removes it.
                        for r in 0..m_eq {
                            residual[n_total + r] -= adu[r];
                        }
                        let ratio = norm_inf(&residual)
                            / (k_norm * norm_inf(&d) + norm_inf(rhs)).max(f64::MIN_POSITIVE);
                        if ratio < 1e-10 {
                            break;
                        }
                        let Some(correction) = factorization.solve(&residual) else {
                            break;
                        };
                        for i in 0..dim {
                            d[i] += correction[i];
                        }
                    }
                    d.iter().all(|v| v.is_finite()).then_some(d)
                };
                let Some(d) = solve(&rhs) else {
                    floor = (100.0 * gamma).max(options.regularization);
                    continue;
                };
                let du: Vec<f64> = d[..n_total].to_vec();
                let dy: Vec<f64> = (0..m_eq).map(|r| -d[n_total + r]).collect();

                // Filter line search (`Filter`), backtracking from the
                // fraction-to-boundary step, with second-order corrections when
                // the full step does not reduce the violation.
                let theta = norm_1(&r_p);
                let phi = barrier_objective(f, &u, mu, p, &has_lower, &has_upper);
                let dphi: f64 = (0..n_total)
                    .map(|k| {
                        let mut term = grad[k] + mu * damping(k, &has_lower, &has_upper);
                        if has_lower[k] {
                            term -= mu / g[k];
                        }
                        if has_upper[k] {
                            term += mu / t[k];
                        }
                        term * du[k]
                    })
                    .sum();
                let alpha_max =
                    fraction_to_boundary(n_total, &has_lower, &has_upper, &g, &t, &du, tau);
                let alpha_min = filter.alpha_min(theta, dphi).max(options.min_step);
                let trial_at = |du: &[f64], alpha: f64| -> Vec<f64> {
                    (0..n_total).map(|k| u[k] + alpha * du[k]).collect()
                };

                let mut found: Option<(Vec<f64>, Vec<f64>, f64, bool)> = None;
                let mut alpha = alpha_max;
                let mut first = true;
                while alpha >= alpha_min {
                    if let Some((th, ph, rp)) = evaluate(
                        problem,
                        p,
                        &trial_at(&du, alpha),
                        mu,
                        &has_lower,
                        &has_upper,
                    ) {
                        if let Some(f_type) = filter.accepts(alpha, theta, phi, dphi, th, ph) {
                            found = Some((du.clone(), dy.clone(), alpha, f_type));
                            break;
                        }
                        if first && th >= theta {
                            // Second-order correction (IPOPT's `max_soc` = 4):
                            // the step again, with the constraint residual at the
                            // trial point folded into the right-hand side.
                            let mut c_soc = r_p.clone();
                            let (mut alpha_soc, mut th_prev, mut rp_trial) = (alpha, theta, rp);
                            for _ in 0..4 {
                                for r in 0..m_eq {
                                    c_soc[r] = alpha_soc * c_soc[r] + rp_trial[r];
                                    rhs[n_total + r] = -c_soc[r];
                                }
                                let Some(d_soc) = solve(&rhs) else { break };
                                let du_soc: Vec<f64> = d_soc[..n_total].to_vec();
                                alpha_soc = fraction_to_boundary(
                                    n_total, &has_lower, &has_upper, &g, &t, &du_soc, tau,
                                );
                                let Some((th_s, ph_s, rp_s)) = evaluate(
                                    problem,
                                    p,
                                    &trial_at(&du_soc, alpha_soc),
                                    mu,
                                    &has_lower,
                                    &has_upper,
                                ) else {
                                    break;
                                };
                                if let Some(f_type) =
                                    filter.accepts(alpha, theta, phi, dphi, th_s, ph_s)
                                {
                                    let dy_soc = (0..m_eq).map(|r| -d_soc[n_total + r]).collect();
                                    found = Some((du_soc, dy_soc, alpha_soc, f_type));
                                    break;
                                }
                                if th_s > 0.99 * th_prev {
                                    break;
                                }
                                (th_prev, rp_trial) = (th_s, rp_s);
                            }
                            for r in 0..m_eq {
                                rhs[n_total + r] = -r_p[r];
                            }
                            if found.is_some() {
                                break;
                            }
                        }
                    }
                    first = false;
                    alpha *= 0.5;
                    trials += 1;
                }

                // No acceptable step. Away from feasibility, the restoration
                // phase takes over; near it, a larger δw turns the direction
                // toward steepest descent and the search runs again.
                let Some((du, dy, alpha, f_type)) = found else {
                    if theta > filter.theta_min {
                        filter.add(theta, phi);
                        restore = true;
                        break;
                    }
                    // Near feasibility at an acceptable point, IPOPT ends here
                    // ("solved to acceptable level", after its restoration
                    // phase finds nothing better). The heavily regularized
                    // steps below would only stir the multipliers: on
                    // case9241_pegase they threw the dual error from 5e-8
                    // back to 1e-4 at the optimum.
                    if error <= options.acceptable_tolerance {
                        break 'solve OptStatus::Optimal;
                    }
                    floor = (100.0 * gamma).max(options.regularization);
                    continue;
                };
                if !f_type {
                    filter.add(theta, phi);
                }

                let dz_l: Vec<f64> = (0..n_total)
                    .map(|k| {
                        if has_lower[k] {
                            (mu - g[k] * z_l[k] - z_l[k] * du[k]) / g[k]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let dz_u: Vec<f64> = (0..n_total)
                    .map(|k| {
                        if has_upper[k] {
                            (mu - t[k] * z_u[k] + z_u[k] * du[k]) / t[k]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                accepted = Some((du, dy, dz_l, dz_u, alpha));
                break;
            }

            if restore {
                let theta = norm_1(&r_p);
                match restoration(
                    problem, p, &u, mu, theta, &filter, &has_lower, &has_upper, options,
                ) {
                    Some((restored, steps)) => {
                        if options.log {
                            eprintln!(
                                "{iterations:4}r restoration: θ {theta:.2e} -> {:.2e} in {steps} steps",
                                {
                                    let x = &restored[..n_x];
                                    norm_1(&p.residual(&problem.constraints(x), &restored))
                                }
                            );
                        }
                        u = restored;
                        continue;
                    }
                    None => {
                        break 'solve stopped(
                            error,
                            options,
                            "restoration failed; the problem may be infeasible here",
                        );
                    }
                }
            }

            let Some((du, dy, dz_l, dz_u, alpha)) = accepted else {
                break 'solve stopped(
                    error,
                    options,
                    "no acceptable step; the problem may be infeasible here",
                );
            };

            if options.log {
                if iterations == 1 {
                    eprintln!(
                        "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha    ls"
                    );
                }
                eprintln!(
                    "{:4} {:+.7e} {:.2e} {:.2e} {:6.2} {:.2e} {:6.2} {:.2e} {:3}",
                    iterations,
                    f,
                    norm_inf(&r_p),
                    norm_inf(&r_d) / s_d,
                    mu.log10(),
                    norm_inf(&du),
                    if gamma > 0.0 {
                        gamma.log10()
                    } else {
                        f64::NEG_INFINITY
                    },
                    alpha,
                    trials,
                );
            }
            // The bound multipliers take their own step, the largest that keeps
            // them positive (IPOPT's default; the constraint multipliers follow
            // the primal step). Tied to a primal step cut short by a bound, they
            // would barely move, and the barrier curvature Σ = z/s they set would
            // keep producing the same blocked direction.
            let alpha_z = {
                let mut a = 1.0f64;
                for k in 0..n_total {
                    if has_lower[k] && dz_l[k] < 0.0 {
                        a = a.min(-tau * z_l[k] / dz_l[k]);
                    }
                    if has_upper[k] && dz_u[k] < 0.0 {
                        a = a.min(-tau * z_u[k] / dz_u[k]);
                    }
                }
                a
            };
            for k in 0..n_total {
                u[k] += alpha * du[k];
                z_l[k] += alpha_z * dz_l[k];
                z_u[k] += alpha_z * dz_u[k];
            }
            for r in 0..m_eq {
                y[r] += alpha * dy[r];
            }

            // Keep the bound multipliers in a sane band around μ/slack. Without
            // this they drift over many iterations until the barrier term they
            // scale stops resembling the constraint it came from — a slow failure
            // that looks like stalling rather than like a bug.
            // IPOPT's κΣ = 1e10, around the slacks after the step.
            for k in 0..n_total {
                if has_lower[k] {
                    let center = mu / (u[k] - p.lower[k]);
                    z_l[k] = z_l[k].clamp(center / 1e10, center * 1e10).max(1e-14);
                }
                if has_upper[k] {
                    let center = mu / (p.upper[k] - u[k]);
                    z_u[k] = z_u[k].clamp(center / 1e10, center * 1e10).max(1e-14);
                }
            }
        }

        stopped(error, options, "iteration limit")
    };

    let x = &u[..n_x];
    let f = problem.objective(x);
    let c = problem.constraints(x);
    let solution = finish(problem, p, &u, &y, &z_l, &z_u, f, &c, iterations, status);
    *state = State {
        u,
        y,
        z_l,
        z_u,
        mu,
        delta_last,
        iterations,
    };
    Ok(solution)
}

/// The KKT matrix factorized with IPOPT's inertia correction (Wächter &
/// Biegler 2006, §3.1, and IPOPT's defaults), returning the factorization and
/// the δw it took.
///
/// The step minimizes the barrier model only if the KKT matrix has `n`
/// positive and `m` negative eigenvalues, so it is factorized with δw = 0
/// first, and otherwise with δw from `options.regularization` (1e-4), or a
/// third of the last iteration's δw, raised ×100 (no δw before) or ×8 until
/// the inertia is right; `None` past 1e40. δc (`dual_regularization`) is
/// always on: [`SymmetricFactorization`] does not pivot, and a zero diagonal
/// in the constraint block would stop it. `floor` is a lower bound on δw from
/// the caller.
fn factorize_with_inertia(
    hessian: &[(usize, usize, f64)],
    a: &[(usize, usize, f64)],
    sigma: &[f64],
    p: &Internal,
    options: &NlpOptions,
    floor: f64,
    delta_last: &mut f64,
) -> Option<(Kkt, f64)> {
    let factor = |dw: f64| {
        let diagonal: Vec<f64> = sigma.iter().map(|s| s + dw).collect();
        Kkt::new(hessian, a, &diagonal, p, options.dual_regularization).filter(Kkt::is_minimizing)
    };
    if floor == 0.0
        && let Some(f) = factor(0.0)
    {
        return Some((f, 0.0));
    }
    let grow = if *delta_last == 0.0 { 100.0 } else { 8.0 };
    let mut dw = if *delta_last == 0.0 {
        options.regularization
    } else {
        (*delta_last / 3.0).max(1e-20)
    }
    .max(floor);
    while dw <= 1e40 {
        if let Some(f) = factor(dw) {
            *delta_last = dw;
            return Some((f, dw));
        }
        dw *= grow;
    }
    None
}

/// The KKT system with the inequality slacks eliminated, factorized.
///
/// A slack's column holds only its diagonal \\(d_k = \Sigma_k + \delta_w\\)
/// and a −1 in its own row, so it is eliminated exactly: the row's diagonal
/// becomes \\(-(\delta_c + 1/d_k)\\), its right-hand side gains
/// \\(\mathrm{rhs}_k / d_k\\), and the slack's step is recovered as
/// \\((\mathrm{rhs}_k + w_r)/d_k\\). By Sylvester's law the full matrix has the
/// inertia `(n, m)` exactly when this one has `(n_x, m)`, the slacks'
/// diagonals being positive.
///
/// Why: a binding inequality's slack sits at μ/z from its bound, so near the
/// end Σ = z/s reaches 1e16 beside entries of order one, more than an
/// unpivoted `LDLᵀ` can solve accurately (case9241_pegase stalled on one
/// such slack for 250 iterations). Condensed, it is a −1e-16 on the
/// constraint diagonal, next to δc. This is the first of the condensed KKT
/// forms the GPU path builds on.
struct Kkt {
    factorization: SymmetricFactorization,
    n_x: usize,
    n: usize,
    m: usize,
    /// `(row, slack column, d_k)`.
    slacks: Vec<(usize, usize, f64)>,
}

impl Kkt {
    /// `None` when a slack's `d_k` is not positive (a row without bounds and
    /// δw = 0) or the factorization meets a zero pivot.
    fn new(
        hessian: &[(usize, usize, f64)],
        a: &[(usize, usize, f64)],
        diagonal: &[f64],
        p: &Internal,
        delta_c: f64,
    ) -> Option<Self> {
        let (n_x, n, m) = (p.n_x, p.n_total, p.m_eq);
        let slacks: Vec<(usize, usize, f64)> = p
            .slack_of
            .iter()
            .enumerate()
            .filter_map(|(r, s)| s.map(|k| (r, k, diagonal[k])))
            .collect();
        if slacks.iter().any(|&(_, _, d)| d.is_nan() || d <= 0.0) {
            return None;
        }
        // δc only where the row would otherwise have no diagonal (equality
        // rows): an inequality row has −1/d_k from its slack, and δc there,
        // next to a 1/d_k that is tiny for a binding slack, would change the
        // step itself (case9241_pegase's path then left IPOPT's by the 12th
        // iteration).
        let mut row_diag = vec![-delta_c; m];
        for &(r, _, d) in &slacks {
            row_diag[r] = -1.0 / d;
        }
        let mut kkt: Vec<(usize, usize, f64)> = hessian
            .iter()
            .copied()
            .filter(|&(i, j, _)| i >= j && i < n_x)
            .collect();
        kkt.extend((0..n_x).map(|k| (k, k, diagonal[k])));
        kkt.extend(
            a.iter()
                .filter(|&&(_, k, _)| k < n_x)
                .map(|&(r, k, v)| (n_x + r, k, v)),
        );
        kkt.extend((0..m).map(|r| (n_x + r, n_x + r, row_diag[r])));
        Some(Self {
            factorization: SymmetricFactorization::new(n_x + m, &kkt)?,
            n_x,
            n,
            m,
            slacks,
        })
    }

    /// The inertia of a step that minimizes the barrier model.
    fn is_minimizing(&self) -> bool {
        self.factorization.inertia()
            == Inertia {
                positive: self.n_x,
                negative: self.m,
                zero: 0,
            }
    }

    /// Solves the full system (`n` variables then `m` rows) through the
    /// condensed one.
    fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        let (n_x, n, m) = (self.n_x, self.n, self.m);
        let mut reduced = vec![0.0; n_x + m];
        reduced[..n_x].copy_from_slice(&rhs[..n_x]);
        reduced[n_x..].copy_from_slice(&rhs[n..n + m]);
        for &(r, k, d) in &self.slacks {
            reduced[n_x + r] += rhs[k] / d;
        }
        let sol = self.factorization.solve(&reduced)?;
        let mut out = vec![0.0; n + m];
        out[..n_x].copy_from_slice(&sol[..n_x]);
        out[n..].copy_from_slice(&sol[n_x..]);
        for &(r, k, d) in &self.slacks {
            out[k] = (rhs[k] + sol[n_x + r]) / d;
        }
        out.iter().all(|v| v.is_finite()).then_some(out)
    }
}

/// `H x` for a full symmetric triplet list.
fn mat_vec_symmetric(triplets: &[(usize, usize, f64)], x: &[f64], n: usize) -> Vec<f64> {
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

/// The status when the iteration stops short of `tolerance`: solved if the
/// last error met `acceptable_tolerance`, as IPOPT decides.
fn stopped(error: f64, options: &NlpOptions, why: &'static str) -> OptStatus {
    if error <= options.acceptable_tolerance {
        OptStatus::Optimal
    } else {
        OptStatus::Other(why)
    }
}

/// `a ≤ b`, up to rounding in values of the size of `reference` (IPOPT's
/// `Compare_le`, 10 machine epsilons). Near the optimum θ is rounding noise
/// and φ changes in its last digits; without the allowance every step there
/// is rejected.
fn le(a: f64, b: f64, reference: f64) -> bool {
    a - b <= 10.0 * f64::EPSILON * reference.abs()
}

fn norm_1(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

/// The barrier objective \\(\varphi_\mu = f - \mu \sum \ln s\\) over every
/// bound slack; infinite outside the bounds.
fn barrier_objective(
    f: f64,
    u: &[f64],
    mu: f64,
    p: &Internal,
    has_lower: &[bool],
    has_upper: &[bool],
) -> f64 {
    let mut value = f;
    for k in 0..u.len() {
        value += mu * damping(k, has_lower, has_upper) * u[k];
        for (has, s) in [
            (has_lower[k], u[k] - p.lower[k]),
            (has_upper[k], p.upper[k] - u[k]),
        ] {
            if has {
                if s <= 0.0 {
                    return f64::INFINITY;
                }
                value -= mu * s.ln();
            }
        }
    }
    value
}

/// IPOPT's damping of variables bounded on one side only (`kappa_d`):
/// \\(\varphi_\mu\\) gains \\(\kappa_d \mu\\) times the distance to that bound,
/// whose derivative in `u[k]` this returns (`+κ_d` for a lower bound, `−κ_d`
/// for an upper; 0 for two bounds or none). Without it the barrier pushes
/// such a variable away from its one bound with nothing to stop it; on
/// case9241_pegase the 32,098 branch-limit slacks are such variables, and
/// the term is 1.9e3 of φ at the start, the whole difference to IPOPT's φ.
fn damping(k: usize, has_lower: &[bool], has_upper: &[bool]) -> f64 {
    const KAPPA_D: f64 = 1e-5;
    match (has_lower[k], has_upper[k]) {
        (true, false) => KAPPA_D,
        (false, true) => -KAPPA_D,
        _ => 0.0,
    }
}

/// A trial point's violation θ (ℓ1), barrier objective φ and constraint
/// residual; `None` outside the bounds or where the problem is not finite.
fn evaluate(
    problem: &dyn NonlinearProblem,
    p: &Internal,
    u: &[f64],
    mu: f64,
    has_lower: &[bool],
    has_upper: &[bool],
) -> Option<(f64, f64, Vec<f64>)> {
    let x = &u[..p.n_x];
    let f = problem.objective(x);
    let phi = barrier_objective(f, u, mu, p, has_lower, has_upper);
    if !phi.is_finite() {
        return None;
    }
    let r = p.residual(&problem.constraints(x), u);
    let theta = norm_1(&r);
    theta.is_finite().then_some((theta, phi, r))
}

/// The feasibility restoration phase, simplified from IPOPT's (Wächter &
/// Biegler 2006, §3.3): entered when the filter line search finds no step,
/// left at the first point that reduces the violation θ by the filter's
/// margin and that the filter (already holding the point restoration started
/// from) accepts. Returns that point and the number of steps, or `None`.
///
/// Each step is the minimum-norm Gauss–Newton step toward feasibility,
/// \\(\min \tfrac12 \lVert D\,d \rVert^2\\) s.t. \\(A d = -\hat c\\), with
/// \\(D^2 = \Sigma + \zeta I\\): \\(\Sigma = \mu/s^2\\) the barrier's
/// curvature on every bound slack, which keeps the step away from the
/// bounds, and \\(\zeta = \sqrt\mu\\) a proximity weight, as in IPOPT's
/// restoration objective. Its KKT matrix is quasi-definite, so it always
/// factorizes. The step is cut to the bounds and halved until θ falls by
/// the filter's margin.
///
/// IPOPT instead solves a whole ℓ1 penalty problem by its own interior
/// point; this keeps the part that matters here, moving to a point the
/// filter accepts, without a second barrier problem.
#[allow(clippy::too_many_arguments)]
fn restoration(
    problem: &dyn NonlinearProblem,
    p: &Internal,
    start: &[f64],
    mu: f64,
    theta_start: f64,
    filter: &Filter,
    has_lower: &[bool],
    has_upper: &[bool],
    options: &NlpOptions,
) -> Option<(Vec<f64>, usize)> {
    const MAX_STEPS: usize = 100;
    let (n, m) = (p.n_total, p.m_eq);
    let zeta = mu.sqrt();
    let mut u = start.to_vec();
    for step in 1..=MAX_STEPS {
        let x = &u[..p.n_x];
        let r = p.residual(&problem.constraints(x), &u);
        let theta = norm_1(&r);
        let a = p.jacobian(&problem.jacobian(x));
        let g: Vec<f64> = (0..n)
            .map(|k| if has_lower[k] { u[k] - p.lower[k] } else { 1.0 })
            .collect();
        let t: Vec<f64> = (0..n)
            .map(|k| if has_upper[k] { p.upper[k] - u[k] } else { 1.0 })
            .collect();
        let d2: Vec<f64> = (0..n)
            .map(|k| {
                zeta + (if has_lower[k] {
                    mu / (g[k] * g[k])
                } else {
                    0.0
                }) + (if has_upper[k] {
                    mu / (t[k] * t[k])
                } else {
                    0.0
                })
            })
            .collect();
        let factorization = Kkt::new(&[], &a, &d2, p, options.dual_regularization)?;
        let mut rhs = vec![0.0; n + m];
        for row in 0..m {
            rhs[n + row] = -r[row];
        }
        let mut d = factorization.solve(&rhs)?;
        for _ in 0..options.refinement_rounds {
            let du = &d[..n];
            let atw = mat_transpose_vec(&a, &d[n..], n);
            let adu = mat_vec(&a, du, m);
            let mut residual = rhs.clone();
            for k in 0..n {
                residual[k] -= d2[k] * du[k] + atw[k];
            }
            for row in 0..m {
                residual[n + row] -= adu[row];
            }
            let correction = factorization.solve(&residual)?;
            d.iter_mut().zip(&correction).for_each(|(di, ci)| *di += ci);
        }
        let du = &d[..n];

        let mut alpha = fraction_to_boundary(n, has_lower, has_upper, &g, &t, du, options.tau);
        let next = loop {
            if alpha < options.min_step {
                return None;
            }
            let trial: Vec<f64> = (0..n).map(|k| u[k] + alpha * du[k]).collect();
            if let Some((th, _, _)) = evaluate(problem, p, &trial, mu, has_lower, has_upper)
                && th <= (1.0 - Filter::GAMMA_THETA) * theta
            {
                break trial;
            }
            alpha *= 0.5;
        };
        u = next;
        if let Some((th, ph, _)) = evaluate(problem, p, &u, mu, has_lower, has_upper)
            && th <= (1.0 - Filter::GAMMA_THETA) * theta_start
            && filter.admits(th, ph)
        {
            return Some((u, step));
        }
    }
    None
}

/// IPOPT's filter (Wächter & Biegler 2006, §2.3–2.4, and IPOPT's defaults).
///
/// A trial point is judged on two measures, the violation θ and the barrier
/// objective φ, instead of one merit function trading them off through a
/// penalty weight. That weight is what made the ℓ1 merit search reject good
/// Newton steps (the Maratos effect): on case2869_pegase it cut every step
/// to α ≈ 0.005 for hundreds of iterations.
#[derive(Default)]
struct Filter {
    /// `(θ, φ)` pairs no trial may match or exceed in both.
    entries: Vec<(f64, f64)>,
    theta_max: f64,
    theta_min: f64,
}

impl Filter {
    const GAMMA_THETA: f64 = 1e-5;
    const GAMMA_PHI: f64 = 1e-8;
    const DELTA: f64 = 1.0;
    const S_THETA: f64 = 1.1;
    const S_PHI: f64 = 2.3;
    const ETA_PHI: f64 = 1e-8;
    const GAMMA_ALPHA: f64 = 0.05;

    /// Sets the violation bounds from the starting point's θ.
    fn start(&mut self, theta0: f64) {
        self.theta_max = 1e4 * theta0.max(1.0);
        self.theta_min = 1e-4 * theta0.max(1.0);
    }

    fn reset(&mut self) {
        self.entries.clear();
    }

    /// Records the current point's margin, after a step accepted on
    /// violation (not on objective) decrease.
    fn add(&mut self, theta: f64, phi: f64) {
        self.entries.push((
            (1.0 - Self::GAMMA_THETA) * theta,
            phi - Self::GAMMA_PHI * theta,
        ));
    }

    /// Whether `(th, ph)` is inside the violation bound and not dominated
    /// by any filter entry.
    fn admits(&self, th: f64, ph: f64) -> bool {
        th <= self.theta_max && !self.entries.iter().any(|&(ft, fp)| th >= ft && ph >= fp)
    }

    /// The switching condition (IPOPT's `IsFtype`): the step promises enough
    /// decrease of φ, relative to the violation, to be judged on φ.
    fn switching(&self, alpha: f64, theta: f64, dphi: f64) -> bool {
        dphi < 0.0 && alpha * (-dphi).powf(Self::S_PHI) > Self::DELTA * theta.powf(Self::S_THETA)
    }

    /// `Some(no_entry)` if the trial `(th, ph)` at step `alpha` is accepted
    /// from the current `(theta, phi)` with directional derivative `dphi`,
    /// as IPOPT's `FilterLSAcceptor` decides: near feasibility
    /// (`theta ≤ theta_min`) a step meeting the switching condition must
    /// satisfy Armijo; any other step needs sufficient decrease of θ or φ.
    /// `no_entry` is true when the step meets the switching condition *and*
    /// Armijo, at any θ: such a step adds no filter entry.
    fn accepts(
        &self,
        alpha: f64,
        theta: f64,
        phi: f64,
        dphi: f64,
        th: f64,
        ph: f64,
    ) -> Option<bool> {
        if !self.admits(th, ph) {
            return None;
        }
        let f_type = self.switching(alpha, theta, dphi);
        let armijo = le(ph - phi, Self::ETA_PHI * alpha * dphi, phi);
        let accepted = if f_type && theta <= self.theta_min {
            armijo
        } else {
            le(th, (1.0 - Self::GAMMA_THETA) * theta, theta)
                || le(ph - phi, -Self::GAMMA_PHI * theta, phi)
        };
        accepted.then_some(f_type && armijo)
    }

    /// The smallest step worth trying before declaring the search failed.
    fn alpha_min(&self, theta: f64, dphi: f64) -> f64 {
        let mut a = Self::GAMMA_THETA;
        if dphi < 0.0 {
            a = a.min(Self::GAMMA_PHI * theta / -dphi);
            if theta <= self.theta_min {
                a = a.min(Self::DELTA * theta.powf(Self::S_THETA) / (-dphi).powf(Self::S_PHI));
            }
        }
        Self::GAMMA_ALPHA * a
    }
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
            .map(|r| {
                p.source_row
                    .iter()
                    .position(|s| *s == Some(r))
                    .map(|i| y[i])
                    .unwrap_or(0.0)
            })
            .collect(),
        z: (0..n_x).map(|k| z_l[k] - z_u[k]).collect(),
        iterations,
        violation,
    }
}
