//! The bordered `(n+1) × (n+1)` system, and why it is the one that works.
//!
//! # Why not a bordered *solve*
//!
//! The tempting cheap route is to keep the plain `n × n` Jacobian, reuse its
//! factorization, and recover `Δλ` by block elimination — two solves against
//! `J` plus a scalar. It is rejected on a mathematical basis, not a performance
//! one: `J` is *exactly* the matrix that goes singular at the nose, so the
//! Schur complement `d − bᵀJ⁻¹c` is the quotient of two quantities that both
//! vanish there. Every backend in this crate reports singularity only as a
//! non-finite result vector (`sparse::RealSparseSystem::solve_values`) and none
//! offers a condition estimate, so there is no way to detect the transition
//! either. The augmented matrix, by contrast, stays non-singular *through* the
//! nose.
//!
//! # Why it stays non-singular
//!
//! At a simple fold `J` has a one-dimensional null space `span(v)`, and
//! transversality gives `−Δs ∉ range(J)`. The bordered matrix
//!
//! ```text
//! [ J    -Δs ]
//! [ bᵀ     d ]
//! ```
//!
//! is then non-singular **iff `bᵀv ≠ 0`**. For [`Parametrization::PseudoArcLength`]
//! `b = t_x`, and at the nose `t = (v, 0)`, so `bᵀv = ‖v‖² ≠ 0`. For
//! [`Parametrization::Local`] `b = e_k` with `k = argmax|t_i| = argmax|v_i|`, so
//! `bᵀv = v_k ≠ 0`. That is the entire reason this form exists.
//!
//! # The border row must stay sparse, and this is measured
//!
//! The λ **column** is emitted at every row regardless of whether `Δs` is zero
//! there, so the pattern does not depend on which buses carry load. That is
//! free: COLAMD's `dense_col` threshold trips on it, the column is ordered last,
//! and the factors are then exactly the bordered factorization. Measured on
//! `case1354pegase` (n = 2449), a bordered solve with a sparse border row costs
//! **1.4×** a plain Newton solve.
//!
//! The border **row** is a different story, and the naive choice is a disaster.
//! Emitting it densely — which [`Parametrization::PseudoArcLength`] genuinely
//! needs, since its row *is* the previous tangent — costs **92×** the plain
//! solve at the same size, while the symbolic analysis stays at 1.0×. So this is
//! numeric fill during factorization, not a bad ordering, and no amount of
//! reasoning about COLAMD's dense-row handling makes it go away. At 119 buses
//! the same comparison is 1.7×, which is why the problem is invisible on small
//! fixtures.
//!
//! Hence [`Parametrization::Local`] is the default: its row is a single entry
//! `e_k`, and it carries the *same* transversality guarantee at the nose
//! (`bᵀv = v_k ≠ 0`, because `k` is chosen as the tangent's largest component).
//! `k` changes rarely — far from the nose the tangent is dominated by λ, so
//! `k = n` and this degenerates to `Natural`; near the nose a voltage component
//! takes over — and a change costs one re-analysis, the same price a bus-type
//! change already pays.
//!
//! So the pattern is fixed for as long as the *continuation index* is, rather
//! than for the whole segment. [`border_columns`] says which columns the row
//! occupies, and `corrector::Segment` re-analyzes when they change.

use crate::jacobian::JacobianPattern;
use crate::network::YBusSparse;
use crate::types::{Bus, BusType};

/// Which equation closes the `n` power-flow equations in `n+1` unknowns.
///
/// All three share one sparsity pattern, so switching between them costs
/// nothing and they can be mixed along a single trace — which the driver does:
/// [`Natural`](Self::Natural) is what lands exactly on a target λ and what the
/// event locator corrects with, whatever the trace's nominal choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Parametrization {
    /// `λ − λ_pred = 0`. Cheapest, and the only one whose continuation
    /// parameter is meaningful to a caller — but it **cannot pass the nose**,
    /// where λ stops being a valid parameter for the curve.
    Natural,
    /// `z_k − z_k_pred = 0`, with `k` the largest component of the tangent.
    /// The classical Ajjarapu–Christy choice, and the default: it passes the
    /// nose, and its border row is one entry rather than `n+1`. See the module
    /// docs for why that second property decides the matter at any real size.
    #[default]
    Local,
    /// `tᵀ(z − z₀) − σ = 0`. The most robust in principle — its bordering row
    /// is non-orthogonal to the null vector by construction rather than by a
    /// search — and what MATPOWER defaults to. **But its row is dense**, which
    /// costs 92× a plain solve at n = 2449 against `Local`'s 1.4×. Worth having
    /// for small networks and for cross-checking `Local`; not the default.
    PseudoArcLength,
}

/// The Newton unknown layout, derived once per segment.
///
/// Reproduces `solver::newton_raphson_cached`'s ordering exactly — every
/// non-slack bus contributes an angle unknown in bus order, then every PQ bus a
/// magnitude unknown — because the corrector's mismatch vector and the
/// augmented Jacobian's rows must agree with it term for term.
pub struct Layout {
    pub non_slack_idx: Vec<usize>,
    pub pq_idx: Vec<usize>,
    pub n_angle: usize,
    pub n_unknowns: usize,
    pub n_buses: usize,
}

impl Layout {
    pub fn analyze(buses: &[Bus]) -> Self {
        let non_slack_idx: Vec<usize> =
            buses.iter().filter(|b| b.bus_type != BusType::Slack).map(|b| b.idx).collect();
        let pq_idx: Vec<usize> =
            buses.iter().filter(|b| b.bus_type == BusType::PQ).map(|b| b.idx).collect();
        let n_angle = non_slack_idx.len();
        let n_unknowns = n_angle + pq_idx.len();
        Self { non_slack_idx, pq_idx, n_angle, n_unknowns, n_buses: buses.len() }
    }

    /// Re-expresses a tangent written in `from`'s layout in this one.
    ///
    /// A `PV → PQ` switch grows the unknown count by one, so the tangent that
    /// was steering the walk no longer even has the right length — but it still
    /// carries the direction of travel, which is the part that matters. Mapping
    /// it bus by bus (the newly-freed magnitude gets a zero component) gives the
    /// next tangent solve a bordering row that points the right way, so the walk
    /// continues forward across the event instead of doubling back.
    pub fn embed(&self, from: &Layout, t: &[f64]) -> Vec<f64> {
        let mut ang_of = vec![usize::MAX; self.n_buses];
        for (row, &i) in from.non_slack_idx.iter().enumerate() {
            ang_of[i] = row;
        }
        let mut vm_of = vec![usize::MAX; self.n_buses];
        for (row, &i) in from.pq_idx.iter().enumerate() {
            vm_of[i] = from.n_angle + row;
        }
        let mut out = vec![0.0; self.n_unknowns + 1];
        for (row, &i) in self.non_slack_idx.iter().enumerate() {
            if ang_of[i] != usize::MAX {
                out[row] = t[ang_of[i]];
            }
        }
        for (row, &i) in self.pq_idx.iter().enumerate() {
            if vm_of[i] != usize::MAX {
                out[self.n_angle + row] = t[vm_of[i]];
            }
        }
        out[self.n_unknowns] = t[from.n_unknowns];
        out
    }


    /// The value of unknown `k`, with `k == n_unknowns` meaning λ itself.
    pub fn get(&self, buses: &[Bus], lambda: f64, k: usize) -> f64 {
        if k < self.n_angle {
            buses[self.non_slack_idx[k]].voltage_ang
        } else if k < self.n_unknowns {
            buses[self.pq_idx[k - self.n_angle]].voltage_mag
        } else {
            lambda
        }
    }

    /// Adds `delta` to unknown `k`. λ is the caller's own scalar, so `k ==
    /// n_unknowns` is handled by the caller rather than here.
    pub fn add(&self, buses: &mut [Bus], k: usize, delta: f64) {
        if k < self.n_angle {
            buses[self.non_slack_idx[k]].voltage_ang += delta;
        } else if k < self.n_unknowns {
            buses[self.pq_idx[k - self.n_angle]].voltage_mag += delta;
        }
    }

    /// `Δs` in Newton row order: `Δp` on every non-slack P row, `Δq` on every
    /// PQ Q row. The same layout as the mismatch vector, which is what lets the
    /// λ column be appended row-for-row.
    pub fn direction_rows(&self, d_p: &[f64], d_q: &[f64]) -> Vec<f64> {
        let mut ds = vec![0.0; self.n_unknowns];
        for (row, &i) in self.non_slack_idx.iter().enumerate() {
            ds[row] = d_p[i];
        }
        for (row, &i) in self.pq_idx.iter().enumerate() {
            ds[self.n_angle + row] = d_q[i];
        }
        ds
    }
}

/// Which columns the border row structurally occupies, for a parametrization
/// whose continuation index is `k` in a system of `n` unknowns.
///
/// One entry for the two sparse parametrizations, all `n+1` for
/// pseudo-arclength. See the module docs for why this distinction is worth
/// making rather than always emitting the dense row.
pub fn border_columns(param: Parametrization, k: usize, n: usize) -> Vec<usize> {
    match param {
        Parametrization::Natural => vec![n],
        Parametrization::Local => vec![k],
        Parametrization::PseudoArcLength => (0..=n).collect(),
    }
}

/// A [`JacobianPattern`] plus the border, in one fixed emission order:
/// the Jacobian's own entries, then the λ column top to bottom, then the
/// border row's own columns in order.
pub struct AugmentedPattern {
    jac: JacobianPattern,
    n: usize,
    border_cols: Vec<usize>,
}

impl AugmentedPattern {
    pub fn analyze(buses: &[Bus], ybus: &YBusSparse, border_cols: Vec<usize>) -> Self {
        let jac = JacobianPattern::analyze(buses, ybus);
        let n = jac.n_unknowns;
        Self { jac, n, border_cols }
    }

    /// The columns this pattern's border row occupies. A caller whose
    /// parametrization now wants different ones must re-analyze.
    pub fn border_cols(&self) -> &[usize] {
        &self.border_cols
    }

    /// The augmented dimension, `n + 1`.
    pub fn dim(&self) -> usize {
        self.n + 1
    }

    pub fn len(&self) -> usize {
        self.jac.len() + self.n + self.border_cols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `(row, col)` pairs in emission order, with the values supplied — the
    /// shape every [`LinearSolver`](crate::solver::LinearSolver) backend
    /// analyzes once.
    pub fn to_triplets(&self, values: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::with_capacity(self.len());
        for (k, ((&r, &c), &v)) in
            self.jac.rows().iter().zip(self.jac.cols()).zip(values).enumerate()
        {
            let _ = k;
            out.push((r as usize, c as usize, v));
        }
        let mut at = self.jac.len();
        for r in 0..self.n {
            out.push((r, self.n, values[at]));
            at += 1;
        }
        for &c in &self.border_cols {
            out.push((self.n, c, values[at]));
            at += 1;
        }
        out
    }

    /// Refills every value at its fixed offset.
    ///
    /// `ds` is the direction in Newton row order (see
    /// [`Layout::direction_rows`]) and `border` holds the row's values at
    /// [`border_cols`](Self::border_cols), in that order.
    /// The λ column carries `∂g/∂λ = −Δs`; getting that sign wrong is the
    /// single easiest way to produce a curve that looks plausible and is wrong,
    /// which is why the test suite re-checks every returned point against an
    /// independently-built power flow.
    pub fn fill(
        &self,
        buses: &[Bus],
        p_calc: &[f64],
        q_calc: &[f64],
        ds: &[f64],
        border: &[f64],
        values: &mut Vec<f64>,
    ) {
        values.clear();
        self.jac.fill_into(buses, p_calc, q_calc, values);
        values.extend(ds.iter().map(|d| -d));
        values.extend_from_slice(border);
        debug_assert_eq!(values.len(), self.len());
    }
}

/// The index of the tangent's largest component — the unknown that is locally
/// the best continuation parameter, and the one [`Parametrization::Local`]
/// pins.
///
/// Choosing the *largest* component is what makes the bordered matrix
/// non-singular at the nose: there `b = e_k` and the tangent is the Jacobian's
/// null vector `v`, so `bᵀv = v_k`, which is the largest entry of `v` and
/// therefore certainly not zero.
pub fn dominant_index(tangent: &[f64]) -> usize {
    let mut best = 0;
    let mut best_abs = 0.0;
    for (i, t) in tangent.iter().enumerate() {
        if t.abs() > best_abs {
            best_abs = t.abs();
            best = i;
        }
    }
    best
}
