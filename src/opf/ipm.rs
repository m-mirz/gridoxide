//! A primal-dual interior-point method for convex quadratic programs.
//!
//! The second backend behind [`Solver`], and the portable one: pure Rust on
//! the sparse factorization stack this crate already owns, with no system
//! install and nothing to link. That is most of the point — `opf::highs` (the
//! `opf-highs` feature) needs a local HiGHS, which means the whole OPF surface
//! is invisible to CI as long as it is the only backend.
//!
//! The other half of the point is that **two independent solvers are a
//! validation gate**. On a convex problem the optimal objective is unique, so
//! a disagreement between this and HiGHS is a bug in one of them — not a
//! modelling convention, not a different local optimum. `super`'s module docs
//! make the argument at more length.
//!
//! # Algorithm
//!
//! Mehrotra's predictor-corrector, the method MATPOWER's MIPS, IPOPT and OOQP
//! all build on. Per iteration: one factorization of the KKT matrix, then two
//! solves against it — an *affine* step that ignores the barrier, and a
//! *corrected* step that uses the affine step's success to pick how hard to
//! centre. The corrector is nearly free (same factorization, new right-hand
//! side) and typically halves the iteration count, which is why the method is
//! ubiquitous.
//!
//! # The internal form
//!
//! The public [`LinearProgram`] carries two-sided bounds on both rows and
//! columns. A barrier method wants equalities plus bounded variables, so the
//! problem is rewritten once, up front, into
//!
//! \\[ \min_u\ \tfrac12 u^{\mathsf T} Q u + c^{\mathsf T} u
//!    \quad\text{s.t.}\quad \tilde A u = b, \qquad \ell \le u \le \upsilon \\]
//!
//! by two transformations, each of which exists to remove a case the barrier
//! genuinely cannot represent:
//!
//! - **A slack per inequality row.** Row \\(r\\) becomes
//!   \\(A_r x - s_r = 0\\) with \\(\ell_r \le s_r \le \upsilon_r\\), moving the
//!   bound off the row and onto a variable. Rows that are already equalities
//!   (`row_lower == row_upper`) get no slack — giving one a slack would create
//!   a variable pinned between equal bounds, which is the very thing the next
//!   point removes.
//! - **Fixed columns become equality rows.** A variable with
//!   `col_lower == col_upper` has no interior, so \\(\log(u_k - \ell_k)\\) is
//!   undefined and \\(1/(u_k - \ell_k)\\) diverges. Rather than special-casing
//!   it through the iteration, the column is freed and pinned by a new row
//!   \\(e_k^{\mathsf T} x = \ell_k\\).
//!
//!   This is not a corner case here: DC-OPF pins its reference angle exactly
//!   this way (`dc.rs` sets `col_lower == col_upper == 0` on it), so every
//!   single OPF this backend sees exercises the path.
//!
//! After both, no internal variable is fixed and every row is an equality —
//! the form the Newton system below assumes.
//!
//! # The Newton system
//!
//! Eliminating the bound multipliers analytically (they are diagonal, so this
//! costs nothing) leaves the *augmented system*
//!
//! \\[ \begin{bmatrix} Q + D + \gamma I & \tilde A^{\mathsf T} \\\\
//!                     \tilde A & -\delta I \end{bmatrix}
//!    \begin{bmatrix} \Delta u \\\\ w \end{bmatrix}
//!    = \begin{bmatrix} \rho \\\\ -r_p \end{bmatrix} \\]
//!
//! with \\(D\\) the diagonal barrier term and \\(w = -\Delta y\\), chosen so
//! the matrix is symmetric. \\(\gamma, \delta > 0\\) are primal and dual
//! **regularization**: without them the matrix is singular whenever
//! \\(\tilde A\\) is rank-deficient or a free variable carries no curvature —
//! both of which happen in DC-OPF, where bus angles are free and cost nothing.
//! With them it is *quasi-definite*, hence factorizable for any symmetric
//! permutation. The perturbation that buys is then undone by iterative
//! refinement against the unregularized matrix.
//!
//! # What this does not do
//!
//! No presolve, no crossover, no basis. Those are what make HiGHS fast and
//! robust on adversarial inputs, and rediscovering them is not the goal: this
//! backend has to be correct and good enough for OPF-shaped problems, with
//! HiGHS remaining available as the reference.

use crate::opf::{LinearProgram, OpfError, OptStatus, Solution, Solver};
use crate::sparse::RealFactorization;

/// Tuning for [`IpmSolver`]. The defaults are what the test suite runs.
#[derive(Clone, Copy, Debug)]
pub struct IpmOptions {
    /// Relative tolerance on all three optimality measures — primal residual,
    /// dual residual, and the complementarity gap. Each is scaled by the size
    /// of the data it is measured against, so this is meaningful across
    /// problem scales.
    pub tolerance: f64,
    /// Iteration cap. A convex QP that has not converged in this many steps
    /// has almost always hit a numerical problem rather than needing more
    /// time; interior-point methods converge in tens of iterations nearly
    /// independent of size.
    pub max_iterations: usize,
    /// Primal regularization \\(\gamma\\) added to the `(1,1)` block.
    pub primal_regularization: f64,
    /// Dual regularization \\(\delta\\) subtracted from the `(2,2)` block.
    pub dual_regularization: f64,
    /// Rounds of iterative refinement per solve, against the *unregularized*
    /// KKT matrix. Zero disables it — which is a good way to see what the
    /// regularization is costing.
    pub refinement_rounds: usize,
    /// The iterate norm past which divergence is declared, given the matching
    /// residual has not converged.
    ///
    /// Exposed because it is the one threshold that depends on the caller's
    /// scaling rather than on the method. Per-unit OPF quantities sit within a
    /// few orders of unity — angles around 1 radian, powers in the tens, prices
    /// in the hundreds — so the default is roughly eight orders of magnitude
    /// above anything physical. Data scaled far away from that should raise it.
    pub divergence_threshold: f64,
}

impl Default for IpmOptions {
    fn default() -> Self {
        Self {
            // Measured: this puts the objective within ~1e-11 *relative* of
            // HiGHS on every pglib fixture, for at most one iteration more
            // than 1e-9 costs. Tightening further buys little and pushes the
            // barrier terms toward ill-conditioning.
            tolerance: 1e-10,
            max_iterations: 200,
            primal_regularization: 1e-9,
            dual_regularization: 1e-9,
            refinement_rounds: 2,
            divergence_threshold: 1e10,
        }
    }
}

/// A convex QP solver needing nothing but this crate.
#[derive(Clone, Debug, Default)]
pub struct IpmSolver {
    options: IpmOptions,
    iterations: usize,
}

impl IpmSolver {
    pub fn new() -> Self {
        Self { options: IpmOptions::default(), iterations: 0 }
    }

    pub fn with_options(options: IpmOptions) -> Self {
        Self { options, iterations: 0 }
    }

    /// Iterations taken by the last [`solve`](Solver::solve). Reported because
    /// it is the honest measure of how this backend compares to HiGHS, and
    /// because a sudden jump is the first symptom of a conditioning problem.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn options_mut(&mut self) -> &mut IpmOptions {
        &mut self.options
    }
}

/// The problem rewritten as `Ãu = b, ℓ ≤ u ≤ υ` — see the module docs.
struct Internal {
    n_vars: usize,
    n_total: usize,
    m_eq: usize,
    a: Vec<(usize, usize, f64)>,
    b: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Lower triangle of `Q`, indices below `n_vars`.
    q: Vec<(usize, usize, f64)>,
    c: Vec<f64>,
}

impl Internal {
    fn build(lp: &LinearProgram) -> Self {
        let n = lp.n_vars;
        let mut lower = lp.col_lower.clone();
        let mut upper = lp.col_upper.clone();
        let mut c = lp.col_cost.clone();
        let mut a: Vec<(usize, usize, f64)> = lp.rows.clone();
        let mut b = Vec::with_capacity(lp.n_rows);

        // One equality row per original row; inequality rows additionally get
        // a slack column carrying the bound.
        let mut n_total = n;
        for r in 0..lp.n_rows {
            if lp.row_lower[r] == lp.row_upper[r] {
                b.push(lp.row_lower[r]);
            } else {
                a.push((r, n_total, -1.0));
                lower.push(lp.row_lower[r]);
                upper.push(lp.row_upper[r]);
                c.push(0.0);
                n_total += 1;
                b.push(0.0);
            }
        }

        // Fixed columns become equality rows, so no internal variable is
        // pinned between equal bounds.
        let mut m_eq = lp.n_rows;
        for k in 0..n {
            if lower[k] == upper[k] {
                a.push((m_eq, k, 1.0));
                b.push(lower[k]);
                lower[k] = f64::NEG_INFINITY;
                upper[k] = f64::INFINITY;
                m_eq += 1;
            }
        }

        Self {
            n_vars: n,
            n_total,
            m_eq,
            a,
            b,
            lower,
            upper,
            q: lp.hessian.clone().unwrap_or_default(),
            c,
        }
    }

    /// `Qu`, expanding the stored lower triangle to the full symmetric matrix.
    ///
    /// The doubling is the classic place to go wrong: an entry `(i, j)` with
    /// `i > j` stands for *both* `Q[i][j]` and `Q[j][i]`, so it contributes to
    /// two components of the product, while a diagonal entry contributes to
    /// one.
    fn q_mul(&self, u: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_total];
        for &(i, j, v) in &self.q {
            out[i] += v * u[j];
            if i != j {
                out[j] += v * u[i];
            }
        }
        out
    }

    fn a_mul(&self, u: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.m_eq];
        for &(r, k, v) in &self.a {
            out[r] += v * u[k];
        }
        out
    }

    fn at_mul(&self, y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_total];
        for &(r, k, v) in &self.a {
            out[k] += v * y[r];
        }
        out
    }
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

impl Solver for IpmSolver {
    fn solve(&mut self, problem: &LinearProgram) -> Result<Solution, OpfError> {
        problem.validate()?;
        // A barrier method has no way to honour integrality, and quietly
        // returning the relaxation would be the worst available answer: a tap
        // of 4.3 looks like a result and is not one. Refusing is what lets a
        // caller discover, at the boundary, that it needs the MIP backend.
        if problem.has_integers() {
            return Err(OpfError::IntegralityUnsupported {
                backend: "interior-point",
                columns: problem.n_integers(),
            });
        }
        self.iterations = 0;

        let p = Internal::build(problem);
        let n_total = p.n_total;
        let m_eq = p.m_eq;
        let dim = n_total + m_eq;

        let has_lower: Vec<bool> = p.lower.iter().map(|x| x.is_finite()).collect();
        let has_upper: Vec<bool> = p.upper.iter().map(|x| x.is_finite()).collect();
        let n_bounds =
            has_lower.iter().filter(|x| **x).count() + has_upper.iter().filter(|x| **x).count();

        // Start strictly inside every finite bound. Crude next to Mehrotra's
        // own starting-point heuristic, but the first few iterations recover
        // from it and it cannot land on a boundary, which is the only thing
        // that would be fatal.
        let mut u = vec![0.0; n_total];
        for k in 0..n_total {
            u[k] = match (has_lower[k], has_upper[k]) {
                (true, true) => 0.5 * (p.lower[k] + p.upper[k]),
                (true, false) => p.lower[k] + 1.0,
                (false, true) => p.upper[k] - 1.0,
                (false, false) => 0.0,
            };
        }
        let mut y = vec![0.0; m_eq];
        let mut z_l: Vec<f64> = has_lower.iter().map(|&h| if h { 1.0 } else { 0.0 }).collect();
        let mut z_u: Vec<f64> = has_upper.iter().map(|&h| if h { 1.0 } else { 0.0 }).collect();

        let b_scale = 1.0 + norm_inf(&p.b);
        let c_scale = 1.0 + norm_inf(&p.c);

        for iteration in 0..self.options.max_iterations {
            self.iterations = iteration + 1;

            let g: Vec<f64> = (0..n_total)
                .map(|k| if has_lower[k] { u[k] - p.lower[k] } else { 1.0 })
                .collect();
            let t: Vec<f64> = (0..n_total)
                .map(|k| if has_upper[k] { p.upper[k] - u[k] } else { 1.0 })
                .collect();

            let qu = p.q_mul(&u);
            let aty = p.at_mul(&y);
            let r_d: Vec<f64> =
                (0..n_total).map(|k| qu[k] + p.c[k] - aty[k] - z_l[k] + z_u[k]).collect();
            let au = p.a_mul(&u);
            let r_p: Vec<f64> = (0..m_eq).map(|r| au[r] - p.b[r]).collect();

            let complementarity: f64 = (0..n_total)
                .map(|k| {
                    (if has_lower[k] { g[k] * z_l[k] } else { 0.0 })
                        + (if has_upper[k] { t[k] * z_u[k] } else { 0.0 })
                })
                .sum();
            let mu = if n_bounds > 0 { complementarity / n_bounds as f64 } else { 0.0 };

            let objective =
                0.5 * (0..n_total).map(|k| u[k] * qu[k]).sum::<f64>()
                    + (0..n_total).map(|k| p.c[k] * u[k]).sum::<f64>();

            // The gap test uses *total* complementarity, not `mu`. `mu` is the
            // per-bound average, so testing it would make the criterion weaker
            // by a factor of the bound count — silently letting big problems
            // stop further from the optimum than small ones. Total
            // complementarity is the duality gap, which is what "how close to
            // optimal" actually means.
            let primal_ok = norm_inf(&r_p) / b_scale <= self.options.tolerance;
            let dual_ok = norm_inf(&r_d) / c_scale <= self.options.tolerance;
            let gap_ok = complementarity / (1.0 + objective.abs()) <= self.options.tolerance;
            if primal_ok && dual_ok && gap_ok {
                return Ok(finish(problem, &p, &u, &y, objective));
            }

            // Without a basis there is no infeasibility *certificate*, so the
            // diagnosis has to come from how the iterates misbehave — and they
            // misbehave in two clearly distinguishable ways.
            //
            // A **primal-infeasible** problem drives the duals to infinity
            // while the primal residual stalls at a positive floor: the method
            // is chasing a multiplier that would make the impossible rows pay,
            // and no primal point ever satisfies them. A **dual-infeasible**
            // (unbounded) problem is the mirror image — the primal runs off
            // along a ray while the dual residual stalls, since no bounded
            // multiplier can price a direction of infinite descent.
            //
            // Pairing the runaway norm with the residual that failed is what
            // makes this safe. Magnitude alone is not evidence: a legitimately
            // large-but-converging iterate would be misread as divergence,
            // whereas a residual still above tolerance says the method is not
            // merely passing through.
            if !u.iter().all(|x| x.is_finite()) || !y.iter().all(|x| x.is_finite()) {
                return Ok(Solution::failed(diagnose(&u, &y, primal_ok, dual_ok)));
            }
            if norm_inf(&y) > self.options.divergence_threshold && !primal_ok {
                return Ok(Solution::failed(OptStatus::Infeasible));
            }
            if norm_inf(&u) > self.options.divergence_threshold && !dual_ok {
                return Ok(Solution::failed(OptStatus::Unbounded));
            }

            // The KKT matrix: symmetric, regularized to quasi-definite.
            let mut kkt: Vec<(usize, usize, f64)> = Vec::with_capacity(
                p.q.len() * 2 + p.a.len() * 2 + n_total + m_eq,
            );
            for &(i, j, v) in &p.q {
                kkt.push((i, j, v));
                if i != j {
                    kkt.push((j, i, v));
                }
            }
            for k in 0..n_total {
                let d = (if has_lower[k] { z_l[k] / g[k] } else { 0.0 })
                    + (if has_upper[k] { z_u[k] / t[k] } else { 0.0 });
                kkt.push((k, k, d + self.options.primal_regularization));
            }
            for &(r, k, v) in &p.a {
                kkt.push((n_total + r, k, v));
                kkt.push((k, n_total + r, v));
            }
            for r in 0..m_eq {
                kkt.push((n_total + r, n_total + r, -self.options.dual_regularization));
            }

            let Some(factorization) = RealFactorization::new(dim, &kkt) else {
                return Ok(Solution::failed(diagnose(&u, &y, primal_ok, dual_ok)));
            };

            // One factorization, two right-hand sides: the predictor ignores
            // the barrier entirely, the corrector uses how well it did.
            let solve_step = |rc_l: &[f64], rc_u: &[f64]| -> Option<(Vec<f64>, Vec<f64>)> {
                let mut rhs = vec![0.0; dim];
                for k in 0..n_total {
                    let mut rho = -r_d[k];
                    if has_lower[k] {
                        rho -= rc_l[k] / g[k];
                    }
                    if has_upper[k] {
                        rho += rc_u[k] / t[k];
                    }
                    rhs[k] = rho;
                }
                for r in 0..m_eq {
                    rhs[n_total + r] = -r_p[r];
                }

                let mut d = factorization.solve(&rhs)?;
                for _ in 0..self.options.refinement_rounds {
                    let residual = kkt_residual(&p, &has_lower, &has_upper, &g, &t, &z_l, &z_u, &d, &rhs);
                    let correction = factorization.solve(&residual)?;
                    for i in 0..dim {
                        d[i] += correction[i];
                    }
                }

                let du = d[..n_total].to_vec();
                // `w = -Δy` is what the symmetrized system solves for.
                let dy: Vec<f64> = (0..m_eq).map(|r| -d[n_total + r]).collect();
                Some((du, dy))
            };

            let rc_l_aff: Vec<f64> =
                (0..n_total).map(|k| if has_lower[k] { g[k] * z_l[k] } else { 0.0 }).collect();
            let rc_u_aff: Vec<f64> =
                (0..n_total).map(|k| if has_upper[k] { t[k] * z_u[k] } else { 0.0 }).collect();

            let Some((du_aff, _)) = solve_step(&rc_l_aff, &rc_u_aff) else {
                return Ok(Solution::failed(diagnose(&u, &y, primal_ok, dual_ok)));
            };
            let dz_l_aff: Vec<f64> = (0..n_total)
                .map(|k| {
                    if has_lower[k] { (-rc_l_aff[k] - z_l[k] * du_aff[k]) / g[k] } else { 0.0 }
                })
                .collect();
            let dz_u_aff: Vec<f64> = (0..n_total)
                .map(|k| {
                    if has_upper[k] { (-rc_u_aff[k] + z_u[k] * du_aff[k]) / t[k] } else { 0.0 }
                })
                .collect();

            let (ap_aff, ad_aff) = step_lengths(
                n_total, &has_lower, &has_upper, &g, &t, &z_l, &z_u, &du_aff, &dz_l_aff,
                &dz_u_aff, 1.0,
            );

            // How much would the affine step have reduced complementarity?
            // That ratio, cubed, is Mehrotra's centring parameter: a step that
            // worked well earns a more aggressive next one.
            let sigma = if n_bounds > 0 && mu > 0.0 {
                let mu_aff: f64 = (0..n_total)
                    .map(|k| {
                        let mut s = 0.0;
                        if has_lower[k] {
                            s += (g[k] + ap_aff * du_aff[k]) * (z_l[k] + ad_aff * dz_l_aff[k]);
                        }
                        if has_upper[k] {
                            s += (t[k] - ap_aff * du_aff[k]) * (z_u[k] + ad_aff * dz_u_aff[k]);
                        }
                        s
                    })
                    .sum::<f64>()
                    / n_bounds as f64;
                (mu_aff / mu).powi(3).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let target = sigma * mu;
            let rc_l: Vec<f64> = (0..n_total)
                .map(|k| {
                    if has_lower[k] {
                        g[k] * z_l[k] - target + du_aff[k] * dz_l_aff[k]
                    } else {
                        0.0
                    }
                })
                .collect();
            let rc_u: Vec<f64> = (0..n_total)
                .map(|k| {
                    if has_upper[k] {
                        t[k] * z_u[k] - target - du_aff[k] * dz_u_aff[k]
                    } else {
                        0.0
                    }
                })
                .collect();

            let Some((du, dy)) = solve_step(&rc_l, &rc_u) else {
                return Ok(Solution::failed(diagnose(&u, &y, primal_ok, dual_ok)));
            };
            let dz_l: Vec<f64> = (0..n_total)
                .map(|k| if has_lower[k] { (-rc_l[k] - z_l[k] * du[k]) / g[k] } else { 0.0 })
                .collect();
            let dz_u: Vec<f64> = (0..n_total)
                .map(|k| if has_upper[k] { (-rc_u[k] + z_u[k] * du[k]) / t[k] } else { 0.0 })
                .collect();

            // Stop just short of the boundary, so the next iteration still has
            // an interior to work in.
            let (mut alpha_p, mut alpha_d) = step_lengths(
                n_total, &has_lower, &has_upper, &g, &t, &z_l, &z_u, &du, &dz_l, &dz_u, 0.995,
            );

            // **A quadratic objective forces a common step length.** Stepping
            // the primal and dual different distances is standard for linear
            // programs and strictly better there — each side goes as far as its
            // own bounds allow. For a QP it silently breaks the method.
            //
            // The dual residual is `Qu + c − Aᵀy − z_l + z_u`, so it depends on
            // the primal *and* the duals. Substituting the Newton equation into
            // it after a split step leaves
            //
            //     r_d ← (1 − α_d)·r_d + (α_p − α_d)·Q Δu
            //
            // The first term is the contraction the method relies on; the
            // second is pure error, absent only when `Q = 0` or the two step
            // lengths agree. Left in, it stops the dual residual converging —
            // the iterates settle into a limit cycle that looks like slow
            // progress and never terminates. That is exactly how this was
            // found: a randomly generated QP cycled with period 8 until the
            // iteration limit while HiGHS solved it without trouble.
            if !p.q.is_empty() {
                let alpha = alpha_p.min(alpha_d);
                alpha_p = alpha;
                alpha_d = alpha;
            }

            for k in 0..n_total {
                u[k] += alpha_p * du[k];
                z_l[k] += alpha_d * dz_l[k];
                z_u[k] += alpha_d * dz_u[k];
            }
            for r in 0..m_eq {
                y[r] += alpha_d * dy[r];
            }
        }

        Ok(Solution::failed(OptStatus::Other("iteration limit")))
    }
}

/// Which way the iterates ran away, when the method fails outright rather than
/// tripping a threshold cleanly.
///
/// Reached when the factorization fails or the iterates go non-finite — by
/// which point one of the two norms has usually already exploded, so the same
/// evidence used for the threshold test applies. `Other` is the honest answer
/// when neither dominates: the solve failed and this method cannot say why.
fn diagnose(u: &[f64], y: &[f64], primal_ok: bool, dual_ok: bool) -> OptStatus {
    let (nu, ny) = (norm_inf(u), norm_inf(y));
    if !primal_ok && ny > nu.max(1.0) * 1e3 {
        OptStatus::Infeasible
    } else if !dual_ok && nu > ny.max(1.0) * 1e3 {
        OptStatus::Unbounded
    } else {
        OptStatus::Other("interior-point method failed to converge")
    }
}

/// `rhs − M d` for the *unregularized* KKT matrix `M`, which is what iterative
/// refinement has to measure against — refining against the regularized matrix
/// would converge neatly to the regularized answer and buy nothing.
#[allow(clippy::too_many_arguments)]
fn kkt_residual(
    p: &Internal,
    has_lower: &[bool],
    has_upper: &[bool],
    g: &[f64],
    t: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    d: &[f64],
    rhs: &[f64],
) -> Vec<f64> {
    let n_total = p.n_total;
    let m_eq = p.m_eq;
    let du = &d[..n_total];
    let w = &d[n_total..];

    let qdu = p.q_mul(du);
    let atw = p.at_mul(w);
    let adu = p.a_mul(du);

    let mut out = vec![0.0; n_total + m_eq];
    for k in 0..n_total {
        let barrier = (if has_lower[k] { z_l[k] / g[k] } else { 0.0 })
            + (if has_upper[k] { z_u[k] / t[k] } else { 0.0 });
        out[k] = rhs[k] - (qdu[k] + barrier * du[k] + atw[k]);
    }
    for r in 0..m_eq {
        out[n_total + r] = rhs[n_total + r] - adu[r];
    }
    out
}

/// The largest primal and dual steps keeping every bound slack and every
/// multiplier positive, scaled by `tau`.
#[allow(clippy::too_many_arguments)]
fn step_lengths(
    n_total: usize,
    has_lower: &[bool],
    has_upper: &[bool],
    g: &[f64],
    t: &[f64],
    z_l: &[f64],
    z_u: &[f64],
    du: &[f64],
    dz_l: &[f64],
    dz_u: &[f64],
    tau: f64,
) -> (f64, f64) {
    // Infinity, not one: these start as the *unblocked* step, so that a
    // direction no bound restricts is scaled to a full step rather than to
    // `tau`. Starting at one would silently cap every iteration at 99.5% of
    // Newton even with nothing in the way, turning exact convergence into a
    // geometric crawl on any problem whose variables are all free.
    let mut alpha_p = f64::INFINITY;
    let mut alpha_d = f64::INFINITY;
    for k in 0..n_total {
        if has_lower[k] {
            if du[k] < 0.0 {
                alpha_p = alpha_p.min(-g[k] / du[k]);
            }
            if dz_l[k] < 0.0 {
                alpha_d = alpha_d.min(-z_l[k] / dz_l[k]);
            }
        }
        if has_upper[k] {
            if du[k] > 0.0 {
                alpha_p = alpha_p.min(t[k] / du[k]);
            }
            if dz_u[k] < 0.0 {
                alpha_d = alpha_d.min(-z_u[k] / dz_u[k]);
            }
        }
    }
    ((tau * alpha_p).min(1.0), (tau * alpha_d).min(1.0))
}

/// Maps the converged internal point back onto the caller's problem.
///
/// The duals are recomputed from the *original* matrix rather than read out of
/// the internal one, which is what keeps them consistent with the KKT identity
/// the OPF tests assert (`col_dual = Qx + c − Aᵀy`). It also handles fixed
/// columns for free: their pinning rows are internal-only, so their multiplier
/// lands in `col_dual` as a reduced cost, exactly where a caller expects a
/// fixed variable's price to appear.
fn finish(
    lp: &LinearProgram,
    p: &Internal,
    u: &[f64],
    y: &[f64],
    objective: f64,
) -> Solution {
    let n = p.n_vars;
    let primal = u[..n].to_vec();

    let mut row_activity = vec![0.0; lp.n_rows];
    for &(r, k, v) in &lp.rows {
        row_activity[r] += v * primal[k];
    }
    let row_dual = y[..lp.n_rows].to_vec();

    let mut col_dual = vec![0.0; n];
    for &(i, j, v) in &p.q {
        col_dual[i] += v * primal[j];
        if i != j {
            col_dual[j] += v * primal[i];
        }
    }
    for k in 0..n {
        col_dual[k] += lp.col_cost[k];
    }
    for &(r, k, v) in &lp.rows {
        col_dual[k] -= v * row_dual[r];
    }

    Solution {
        status: OptStatus::Optimal,
        objective: objective + lp.offset,
        primal,
        row_activity,
        col_dual,
        row_dual,
    }
}
