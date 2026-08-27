//! The differential-algebraic system: its variable layout, its residual, and
//! its Jacobian's sparsity pattern.
//!
//! # The system
//!
//! ```text
//! ẋ = f(x, V)                        the devices
//! Y V − I_inj(x, V) = 0              the network
//! ```
//!
//! integrated by a one-step implicit rule written as
//!
//! ```text
//! x₁ − x₀ − h(a·f₁ + b·f₀) = 0       (a, b) = (½, ½) trapezoidal
//!                                            (1, 0)  backward Euler
//! ```
//!
//! so that both rules share one residual and one Jacobian, differing only in
//! the scalar `h·a` this module is handed. That is not a convenience: the
//! integrator switches between them *within a run* (§5 of the plan — backward
//! Euler for two steps after each event, to kill the trapezoidal rule's
//! ringing), and two separate assemblies would be two places to get the
//! damping wrong.
//!
//! # Why rectangular current balance
//!
//! In rectangular coordinates `V = v_re + j·v_im` the network block is the
//! real form of the admittance matrix,
//!
//! ```text
//! [ i_re ]   [ G  −B ] [ v_re ]
//! [ i_im ] = [ B   G ] [ v_im ]
//! ```
//!
//! which is **constant** for the lifetime of a topology — no dependence on `x`
//! or `V` at all. Every moving part is then a device stamp, and every device
//! stamp is local: `∂f/∂x` is one dense block per device, `∂f/∂V` and
//! `∂I/∂x` couple that block to its own bus's two columns/rows, and `∂I/∂V` is
//! 2×2 on one diagonal.
//!
//! A polar power-mismatch Jacobian — what `jacobian::JacobianPattern` builds
//! for the power flow — has none of that structure: every entry moves every
//! iteration. This module is modelled on that one's *shape* (analyze once,
//! refill values at fixed offsets, hand the values to
//! [`LinearSolver::factor_and_solve_values`](crate::solver::LinearSolver)) and
//! reuses none of its arithmetic.
//!
//! # Variable order
//!
//! ```text
//! z = [ device 0 states … device n states | v_re₀ v_im₀ v_re₁ v_im₁ … ]
//! ```
//!
//! Devices first, then the network with each bus's two components adjacent.
//! No hand-tuned permutation beyond that: every backend behind
//! `LinearSolver` runs its own fill-reducing ordering, and second-guessing
//! COLAMD or AMD from here would only make the pattern harder to read.

use num_complex::Complex;

use crate::network::YBusSparse;

use super::models::{DynamicModel, ModelJacobian};

/// Where each device's states and each bus's voltage components live in `z`.
#[derive(Clone, Debug)]
pub struct DaeLayout {
    pub n_bus: usize,
    /// Total differential states, summed over devices.
    pub n_diff: usize,
    /// Start index in `z` of each device's state block.
    pub dev_offset: Vec<usize>,
    /// How many states each device owns.
    pub dev_len: Vec<usize>,
    /// Which bus each device sits at.
    pub dev_bus: Vec<usize>,
}

impl DaeLayout {
    pub fn new(dev_len: &[usize], dev_bus: &[usize], n_bus: usize) -> Self {
        let mut dev_offset = Vec::with_capacity(dev_len.len());
        let mut acc = 0usize;
        for &len in dev_len {
            dev_offset.push(acc);
            acc += len;
        }
        Self {
            n_bus,
            n_diff: acc,
            dev_offset,
            dev_len: dev_len.to_vec(),
            dev_bus: dev_bus.to_vec(),
        }
    }

    /// Total unknowns: every differential state plus two per bus.
    pub fn n(&self) -> usize {
        self.n_diff + 2 * self.n_bus
    }

    /// Row/column index of bus `i`'s real voltage component.
    pub fn v_re(&self, i: usize) -> usize {
        self.n_diff + 2 * i
    }

    /// Row/column index of bus `i`'s imaginary voltage component.
    pub fn v_im(&self, i: usize) -> usize {
        self.n_diff + 2 * i + 1
    }

    pub fn n_devices(&self) -> usize {
        self.dev_len.len()
    }
}

/// The Jacobian's sparsity pattern, analyzed once per topology, refilled every
/// Newton iteration.
///
/// The `(row, col)` arrays are handed to `LinearSolver::new` exactly once; from
/// then on only [`fill`](Self::fill)'s `f64` values move, in the same
/// positional order, which is the contract
/// `LinearSolver::factor_and_solve_values` is written against.
#[derive(Clone, Debug)]
pub struct DaePattern {
    layout: DaeLayout,
    rows: Vec<u32>,
    cols: Vec<u32>,
    /// First slot of each device's `∂f/∂x` block (`len × len`, row-major).
    dfdx_slot: Vec<usize>,
    /// First slot of each device's `∂f/∂V` block (`len × 2`, row-major).
    dfdv_slot: Vec<usize>,
    /// First slot of each device's `∂I/∂x` block (`2 × len`, row-major).
    didx_slot: Vec<usize>,
    /// For bus `i`, the slot of the first of its `4 · row(i).len()` network
    /// entries. Neighbours are in `YBusSparse::row(i)` order.
    net_row_slot: Vec<usize>,
    /// For bus `i`, the slot of its own diagonal 2×2 group, or `usize::MAX` if
    /// the Y-bus has no diagonal entry there (a bus with no branch, no shunt
    /// and no device — structurally singular, and reported as such rather than
    /// silently patched).
    net_diag_slot: Vec<usize>,
    /// Buses whose two algebraic equations are replaced by `V = V₀`.
    fixed: Vec<bool>,
    nnz: usize,
}

impl DaePattern {
    /// Builds the pattern from the device set and the finished Y-bus.
    ///
    /// `ybus` must already carry every constant stamp — each device's Norton
    /// admittance and every load converted to an admittance — because the
    /// network block is assembled straight out of it and nothing later adds to
    /// it.
    ///
    /// `fixed` names buses held at a constant voltage (an infinite bus). Their
    /// two rows become `v_re = v_re₀`, `v_im = v_im₀`. This is exact, unlike
    /// the usual trick of a very small source impedance, which is what the
    /// analytic gate in `tests/dynamics_smib_test.rs` needs.
    pub fn analyze(layout: DaeLayout, ybus: &YBusSparse, fixed: &[usize]) -> Self {
        let n_dev = layout.n_devices();
        let mut rows: Vec<u32> = Vec::new();
        let mut cols: Vec<u32> = Vec::new();

        let push = |r: usize, c: usize, rows: &mut Vec<u32>, cols: &mut Vec<u32>| {
            rows.push(r as u32);
            cols.push(c as u32);
        };

        // 1. ∂F_diff/∂x — one dense block per device, including the identity
        //    that the implicit rule contributes on the diagonal.
        let mut dfdx_slot = Vec::with_capacity(n_dev);
        for d in 0..n_dev {
            dfdx_slot.push(rows.len());
            let (off, len) = (layout.dev_offset[d], layout.dev_len[d]);
            for i in 0..len {
                for j in 0..len {
                    push(off + i, off + j, &mut rows, &mut cols);
                }
            }
        }

        // 2. ∂F_diff/∂V — each device against its own bus's two columns.
        let mut dfdv_slot = Vec::with_capacity(n_dev);
        for d in 0..n_dev {
            dfdv_slot.push(rows.len());
            let (off, len, bus) = (layout.dev_offset[d], layout.dev_len[d], layout.dev_bus[d]);
            for i in 0..len {
                push(off + i, layout.v_re(bus), &mut rows, &mut cols);
                push(off + i, layout.v_im(bus), &mut rows, &mut cols);
            }
        }

        // 3. ∂F_alg/∂x — each device's own bus's two rows against its states.
        let mut didx_slot = Vec::with_capacity(n_dev);
        for d in 0..n_dev {
            didx_slot.push(rows.len());
            let (off, len, bus) = (layout.dev_offset[d], layout.dev_len[d], layout.dev_bus[d]);
            for j in 0..len {
                push(layout.v_re(bus), off + j, &mut rows, &mut cols);
            }
            for j in 0..len {
                push(layout.v_im(bus), off + j, &mut rows, &mut cols);
            }
        }

        // 4. ∂F_alg/∂V — the real form of Y, plus (at fill time) each device's
        //    own −∂I/∂V folded onto its bus's diagonal group. Folding rather
        //    than emitting a second entry at the same (row, col) keeps the
        //    triplet list duplicate-free, which the positional-values contract
        //    needs.
        let mut net_row_slot = Vec::with_capacity(layout.n_bus);
        let mut net_diag_slot = vec![usize::MAX; layout.n_bus];
        for i in 0..layout.n_bus {
            net_row_slot.push(rows.len());
            for &(j, _) in ybus.row(i) {
                if i == j {
                    net_diag_slot[i] = rows.len();
                }
                push(layout.v_re(i), layout.v_re(j), &mut rows, &mut cols);
                push(layout.v_re(i), layout.v_im(j), &mut rows, &mut cols);
                push(layout.v_im(i), layout.v_re(j), &mut rows, &mut cols);
                push(layout.v_im(i), layout.v_im(j), &mut rows, &mut cols);
            }
        }

        let mut is_fixed = vec![false; layout.n_bus];
        for &b in fixed {
            is_fixed[b] = true;
        }

        let nnz = rows.len();
        Self {
            layout,
            rows,
            cols,
            dfdx_slot,
            dfdv_slot,
            didx_slot,
            net_row_slot,
            net_diag_slot,
            fixed: is_fixed,
            nnz,
        }
    }

    pub fn layout(&self) -> &DaeLayout {
        &self.layout
    }

    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Whether this bus's two algebraic equations were replaced by `V = V₀`.
    pub fn is_fixed(&self, bus: usize) -> bool {
        self.fixed[bus]
    }

    /// The triplet list for `LinearSolver::new`, pairing the pattern's fixed
    /// `(row, col)` arrays with values from an actual [`fill`](Self::fill).
    ///
    /// The values matter even though only the pattern is being analyzed:
    /// `KluNative` and `Klu` *factor* numerically inside `new`, so handing
    /// them a placeholder matrix of ones is handing them a singular one, and
    /// they correctly refuse it. `solver::newton_raphson_cached` builds its
    /// triplets the same way and for the same reason — first fill, then
    /// analyze.
    pub fn to_triplets(&self, values: &[f64]) -> Vec<(usize, usize, f64)> {
        (0..self.nnz)
            .map(|k| (self.rows[k] as usize, self.cols[k] as usize, values[k]))
            .collect()
    }

    /// A bus whose Y-bus row has no diagonal entry, if any. Such a bus makes
    /// the algebraic block structurally singular, and saying which one it is
    /// beats letting the factorization fail with no explanation.
    pub fn bus_without_diagonal(&self) -> Option<usize> {
        (0..self.layout.n_bus)
            .find(|&i| !self.fixed[i] && self.net_diag_slot[i] == usize::MAX)
    }

    /// Writes the Jacobian's values into `values`, in slot order.
    ///
    /// `ha` is `h·a` for the active integration rule — `h/2` for trapezoidal,
    /// `h` for backward Euler. `scratch` is one [`ModelJacobian`] per device,
    /// reused across iterations so a step allocates nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn fill(
        &self,
        models: &[Box<dyn DynamicModel>],
        x: &[f64],
        v: &[Complex<f64>],
        ha: f64,
        ybus: &YBusSparse,
        scratch: &mut [ModelJacobian],
        values: &mut Vec<f64>,
    ) {
        values.clear();
        values.resize(self.nnz, 0.0);

        for (d, model) in models.iter().enumerate() {
            let (off, len, bus) = (
                self.layout.dev_offset[d],
                self.layout.dev_len[d],
                self.layout.dev_bus[d],
            );
            let jac = &mut scratch[d];
            jac.clear();
            model.jacobian(&x[off..off + len], v[bus], jac);

            // ∂F_diff/∂x = I − h·a·∂f/∂x.
            let base = self.dfdx_slot[d];
            for i in 0..len {
                for j in 0..len {
                    let identity = if i == j { 1.0 } else { 0.0 };
                    values[base + i * len + j] = identity - ha * jac.dfdx[i * len + j];
                }
            }

            // ∂F_diff/∂V = −h·a·∂f/∂V.
            let base = self.dfdv_slot[d];
            for i in 0..len {
                values[base + i * 2] = -ha * jac.dfdv[i * 2];
                values[base + i * 2 + 1] = -ha * jac.dfdv[i * 2 + 1];
            }

            // ∂F_alg/∂x = −∂I/∂x, unless the bus's equations were replaced.
            let base = self.didx_slot[d];
            if !self.fixed[bus] {
                for j in 0..len {
                    values[base + j] = -jac.didx[j];
                    values[base + len + j] = -jac.didx[len + j];
                }
            }
        }

        // ∂F_alg/∂V = Y_real, then −∂I/∂V folded onto the device diagonals.
        for i in 0..self.layout.n_bus {
            let mut slot = self.net_row_slot[i];
            if self.fixed[i] {
                // Replaced rows: identity on this bus's own diagonal, zero
                // everywhere else along the row.
                for &(j, _) in ybus.row(i) {
                    let one = if i == j { 1.0 } else { 0.0 };
                    values[slot] = one;
                    values[slot + 1] = 0.0;
                    values[slot + 2] = 0.0;
                    values[slot + 3] = one;
                    slot += 4;
                }
                continue;
            }
            for &(_, y) in ybus.row(i) {
                values[slot] = y.re;
                values[slot + 1] = -y.im;
                values[slot + 2] = y.im;
                values[slot + 3] = y.re;
                slot += 4;
            }
        }

        for (d, _) in models.iter().enumerate() {
            let bus = self.layout.dev_bus[d];
            if self.fixed[bus] {
                continue;
            }
            let diag = self.net_diag_slot[bus];
            if diag == usize::MAX {
                continue;
            }
            let didv = scratch[d].didv;
            values[diag] -= didv[0];
            values[diag + 1] -= didv[1];
            values[diag + 2] -= didv[2];
            values[diag + 3] -= didv[3];
        }
    }
}

/// Evaluates the residual `F(z₁)` for the implicit step.
///
/// `x0`/`f0` are the previous point and its derivatives; `x`/`v` the current
/// Newton iterate. `ha` and `hb` are `h·a` and `h·b` for the active rule, so
/// backward Euler is just `hb = 0` — see the module doc.
///
/// The algebraic half is evaluated at the new point unconditionally, which is
/// what makes this a DAE step rather than an explicit one: the network is a
/// constraint, not something integrated.
#[allow(clippy::too_many_arguments)]
pub fn residual(
    pattern: &DaePattern,
    models: &[Box<dyn DynamicModel>],
    x0: &[f64],
    f0: &[f64],
    x: &[f64],
    v: &[Complex<f64>],
    v_fixed: &[Complex<f64>],
    ha: f64,
    hb: f64,
    ybus: &YBusSparse,
    f_scratch: &mut [f64],
    out: &mut [f64],
) {
    let layout = &pattern.layout;

    for (d, model) in models.iter().enumerate() {
        let (off, len, bus) = (layout.dev_offset[d], layout.dev_len[d], layout.dev_bus[d]);
        model.derivatives(&x[off..off + len], v[bus], &mut f_scratch[off..off + len]);
        for i in off..off + len {
            out[i] = x[i] - x0[i] - ha * f_scratch[i] - hb * f0[i];
        }
    }

    // Network: (Y V)ᵢ − I_injᵢ, split into real and imaginary rows.
    for i in 0..layout.n_bus {
        if pattern.fixed[i] {
            out[layout.v_re(i)] = v[i].re - v_fixed[i].re;
            out[layout.v_im(i)] = v[i].im - v_fixed[i].im;
            continue;
        }
        let mut acc = Complex::new(0.0, 0.0);
        for &(j, y) in ybus.row(i) {
            acc += y * v[j];
        }
        out[layout.v_re(i)] = acc.re;
        out[layout.v_im(i)] = acc.im;
    }

    for (d, model) in models.iter().enumerate() {
        let (off, len, bus) = (layout.dev_offset[d], layout.dev_len[d], layout.dev_bus[d]);
        if pattern.fixed[bus] {
            continue;
        }
        let inj = model.injection(&x[off..off + len], v[bus]);
        out[layout.v_re(bus)] -= inj.re;
        out[layout.v_im(bus)] -= inj.im;
    }
}
