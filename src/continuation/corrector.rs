//! The extended Newton corrector and the tangent predictor.
//!
//! Both are the same linear solve against the same bordered matrix with
//! different right-hand sides, which is why they live together and share one
//! symbolic factorization.
//!
//! The corrector is deliberately a transliteration of
//! `solver::newton_raphson_cached`: same unknown ordering, same
//! `effective_injection − power_injections` mismatch, same convergence test on
//! `max|mismatch|`. With `Δλ` pinned to zero it reduces to that loop term for
//! term, and `tests/continuation_test.rs` asserts exactly that — it is the
//! cheapest guard against the augmented form quietly disagreeing with the
//! solver it is supposed to extend.

use crate::network::{effective_injection, power_injections, YBusSparse};
use crate::solver::LinearSolver;
use crate::types::{Bus, BusType};

use super::augmented::{border_columns, AugmentedPattern, Layout, Parametrization};
use super::direction::{BaseSpec, LoadingDirection};

/// The closing equation, fully evaluated: what `p(x, λ)` is and what border row
/// it implies.
#[derive(Clone, Debug)]
pub(crate) enum Constraint {
    /// `λ − target = 0`. Also what the event locator and the target-λ landing
    /// correct with, whatever the trace's nominal parametrization.
    Natural { target: f64 },
    /// `z_k − target = 0`.
    Local { k: usize, target: f64 },
    /// `tᵀ(z − z₀) − σ = 0`.
    Arc { tangent: Vec<f64>, z0: Vec<f64>, step: f64 },
}

impl Constraint {
    fn residual(&self, layout: &Layout, buses: &[Bus], lambda: f64) -> f64 {
        match self {
            Constraint::Natural { target } => lambda - target,
            Constraint::Local { k, target } => layout.get(buses, lambda, *k) - target,
            Constraint::Arc { tangent, z0, step } => {
                let mut acc = -step;
                for k in 0..=layout.n_unknowns {
                    acc += tangent[k] * (layout.get(buses, lambda, k) - z0[k]);
                }
                acc
            }
        }
    }

    /// The columns this constraint's border row occupies, and its values there.
    ///
    /// Two of the three are a single entry; only pseudo-arclength needs the
    /// whole row, and the module docs for `augmented` record what that costs.
    fn border(&self, layout: &Layout) -> (Vec<usize>, Vec<f64>) {
        let n = layout.n_unknowns;
        match self {
            Constraint::Natural { .. } => (border_columns(Parametrization::Natural, n, n), vec![1.0]),
            Constraint::Local { k, .. } => {
                (border_columns(Parametrization::Local, *k, n), vec![1.0])
            }
            Constraint::Arc { tangent, .. } => (
                border_columns(Parametrization::PseudoArcLength, 0, n),
                tangent[..=n].to_vec(),
            ),
        }
    }
}

/// How a corrector call finished.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CorrectorStatus {
    Converged,
    MaxIterations,
    Singular,
}

pub(crate) struct CorrectorOutcome {
    pub status: CorrectorStatus,
    pub iterations: usize,
    pub lambda: f64,
}

/// One stretch of curve over which the bus types — and therefore the unknown
/// count and the sparsity pattern — do not change.
///
/// A Q-limit switch ends a segment: it moves `n_unknowns`, which is exactly the
/// condition `PersistentSolver::reset` documents, so the pattern, the direction
/// rows and the cached factorization are all rebuilt.
pub(crate) struct Segment<S: LinearSolver> {
    pub layout: Layout,
    pattern: AugmentedPattern,
    /// The direction in Newton row order, rebuilt with the segment.
    ds: Vec<f64>,
    solver: Option<S>,
    values: Vec<f64>,
    /// Augmented solves performed — reported so the cost model is visible
    /// rather than folded into a wall-clock number.
    pub solves: usize,
    /// Symbolic factorizations rebuilt because the continuation index moved.
    pub reanalyses: usize,
}

impl<S: LinearSolver> Segment<S> {
    pub fn analyze(
        buses: &[Bus],
        ybus: &YBusSparse,
        dir: &LoadingDirection,
        border_cols: Vec<usize>,
    ) -> Self {
        let layout = Layout::analyze(buses);
        let pattern = AugmentedPattern::analyze(buses, ybus, border_cols);
        let ds = layout.direction_rows(&dir.d_p, &dir.d_q);
        let values = Vec::with_capacity(pattern.len());
        Self { layout, pattern, ds, solver: None, values, solves: 0, reanalyses: 0 }
    }

    /// Re-analyzes if the border row now needs different columns.
    ///
    /// [`Parametrization::Local`] re-picks its continuation index as the curve
    /// bends, and the index *is* the border row's only column — so the pattern
    /// changes with it and the cached factorization goes with it. This is rare:
    /// far from the nose the tangent is dominated by λ, so `k` sits at `n` and
    /// does not move; near the nose it changes once or twice. A re-analysis
    /// costs what a bus-type change already costs.
    fn ensure_border(&mut self, buses: &[Bus], ybus: &YBusSparse, cols: &[usize]) {
        if self.pattern.border_cols() == cols {
            return;
        }
        self.pattern = AugmentedPattern::analyze(buses, ybus, cols.to_vec());
        self.solver = None;
        self.reanalyses += 1;
    }

    pub fn n(&self) -> usize {
        self.layout.n_unknowns
    }

    /// The ordinary power-flow mismatch, in the Newton row order: `ΔP` at every
    /// non-slack bus, then `ΔQ` at every PQ bus.
    fn mismatch(&self, buses: &[Bus], ybus: &YBusSparse) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (p_calc, q_calc) = power_injections(buses, ybus);
        let mut m = vec![0.0; self.layout.n_unknowns];
        for (row, &i) in self.layout.non_slack_idx.iter().enumerate() {
            let (p_eff, _) = effective_injection(&buses[i]);
            m[row] = p_eff - p_calc[i];
        }
        for (row, &i) in self.layout.pq_idx.iter().enumerate() {
            let (_, q_eff) = effective_injection(&buses[i]);
            m[self.layout.n_angle + row] = q_eff - q_calc[i];
        }
        (m, p_calc, q_calc)
    }

    /// One bordered solve at the current state, with the given border row and
    /// right-hand side. `None` when the backend reports singularity.
    fn solve_bordered(
        &mut self,
        buses: &[Bus],
        ybus: &YBusSparse,
        p_calc: &[f64],
        q_calc: &[f64],
        border_cols: &[usize],
        border: &[f64],
        rhs: &[f64],
    ) -> Option<Vec<f64>> {
        self.ensure_border(buses, ybus, border_cols);
        let (pattern, ds, values) = (&self.pattern, &self.ds, &mut self.values);
        pattern.fill(buses, p_calc, q_calc, ds, border, values);
        if self.solver.is_none() {
            self.solver = S::new(pattern.dim(), &pattern.to_triplets(values));
        }
        let solver = self.solver.as_mut()?;
        self.solves += 1;
        solver.factor_and_solve_values(values, rhs)
    }

    /// The unit tangent to the solution curve at the current (converged) state.
    ///
    /// Solves the bordered system against `[0; …; 0; 1]` with `border` as the
    /// bottom row. The choice of that row is what keeps the walk moving
    /// forward — see `continuation::tangent_border`, which builds it. Bordering
    /// with `e_n` instead pins `t_λ = 1` exactly and can only ever report an
    /// increasing λ: right for the very first step, and fatal for detecting the
    /// fold if used throughout.
    ///
    /// At the nose this vector *is* the null vector of `J`, which is why the
    /// weakest-bus ranking falls out of the predictor rather than needing an
    /// eigensolver.
    pub fn tangent(
        &mut self,
        buses: &[Bus],
        ybus: &YBusSparse,
        border_cols: &[usize],
        border: &[f64],
        sign: f64,
    ) -> Option<Vec<f64>> {
        let (p_calc, q_calc) = power_injections(buses, ybus);
        let mut rhs = vec![0.0; self.layout.n_unknowns + 1];
        rhs[self.layout.n_unknowns] = sign;
        let mut t = self.solve_bordered(buses, ybus, &p_calc, &q_calc, border_cols, border, &rhs)?;
        let norm = t.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(norm > 0.0) || !norm.is_finite() {
            return None;
        }
        for v in &mut t {
            *v /= norm;
        }
        Some(t)
    }

    /// Extended Newton on `[g; p] = 0`, mutating `buses` and returning the
    /// corrected λ.
    ///
    /// The right-hand side is `[mismatch; −p]` and the top-right block is
    /// `−Δs`; with `Δλ = 0` both reduce to the plain Newton step.
    #[allow(clippy::too_many_arguments)]
    pub fn correct(
        &mut self,
        buses: &mut [Bus],
        ybus: &YBusSparse,
        base: &BaseSpec,
        dir: &LoadingDirection,
        constraint: &Constraint,
        mut lambda: f64,
        tol: f64,
        max_iter: usize,
    ) -> CorrectorOutcome {
        let n = self.layout.n_unknowns;
        for it in 0..max_iter {
            dir.apply(base, lambda, buses);
            let (mismatch, p_calc, q_calc) = self.mismatch(buses, ybus);
            let p_res = constraint.residual(&self.layout, buses, lambda);

            let worst = mismatch.iter().fold(0.0f64, |a, &b| a.max(b.abs())).max(p_res.abs());
            if worst < tol {
                return CorrectorOutcome {
                    status: CorrectorStatus::Converged,
                    iterations: it,
                    lambda,
                };
            }

            let (border_cols, border) = constraint.border(&self.layout);
            let mut rhs = Vec::with_capacity(n + 1);
            rhs.extend_from_slice(&mismatch);
            rhs.push(-p_res);

            let Some(dz) =
                self.solve_bordered(buses, ybus, &p_calc, &q_calc, &border_cols, &border, &rhs)
            else {
                return CorrectorOutcome {
                    status: CorrectorStatus::Singular,
                    iterations: it,
                    lambda,
                };
            };
            for k in 0..n {
                self.layout.add(buses, k, dz[k]);
            }
            lambda += dz[n];
        }
        // One last consistency write, so `buses` always describes the λ being
        // reported even on a failed correction.
        dir.apply(base, lambda, buses);
        CorrectorOutcome { status: CorrectorStatus::MaxIterations, iterations: max_iter, lambda }
    }

    /// The reactive injection at every bus, for the event functions.
    pub fn q_injections(&self, buses: &[Bus], ybus: &YBusSparse) -> Vec<f64> {
        power_injections(buses, ybus).1
    }
}

/// Whether any bus still carries a voltage-dependent term.
///
/// Neither `jacobian::JacobianPattern::fill` nor its reference oracle
/// `solver::build_jacobian_triplets` includes `∂s_eff/∂|V|` for ZIP terms. For
/// an ordinary solve that is a converged-answer-preserving inexactness: the
/// mismatch is exact, so only the step direction is off. For continuation it is
/// not — the Jacobian's singularity *is* the answer, so a Jacobian missing
/// terms puts the nose in the wrong place while still looking entirely
/// plausible.
pub(crate) fn zip_buses(buses: &[Bus]) -> Vec<usize> {
    buses
        .iter()
        .filter(|b| b.bus_type != BusType::Slack && !b.zip_terms.is_empty())
        .map(|b| b.idx)
        .collect()
}
