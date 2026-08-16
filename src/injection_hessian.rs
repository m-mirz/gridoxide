//! Second derivatives of the bus power injections.
//!
//! [`ac_sensitivity`](crate::ac_sensitivity) differentiates a converged
//! operating point once. This differentiates it twice — the piece any
//! interior-point method needs to solve AC-OPF, where the Hessian of the
//! Lagrangian is
//!
//! \\[ \nabla^2_{xx} L = \nabla^2 f
//!    + \sum_i \lambda_i \nabla^2 P_i + \sum_i \mu_i \nabla^2 Q_i \\]
//!
//! The objective's own \\(\nabla^2 f\\) is trivial; the constraint terms are
//! not, and they are what lives here.
//!
//! # What it computes, and why in that form
//!
//! Never one \\(\nabla^2 P_i\\) at a time — always the **multiplier-weighted
//! sum** over every bus at once, which is the only form an optimizer asks
//! for. Each individual \\(\nabla^2 P_i\\) is a sparse \\(2n \times 2n\\)
//! matrix and there are \\(2n\\) of them; materializing them separately would
//! cost \\(O(n)\\) matrices to build one. Summing as we go makes the whole
//! thing one pass over the Y-bus, the same cost as assembling the Jacobian.
//!
//! # State layout
//!
//! The **full** \\(2n\\) space: \\(\theta_0 \dots \theta_{n-1}\\) at indices
//! \\(0 \dots n-1\\), then \\(|V|_0 \dots |V|_{n-1}\\) at \\(n \dots 2n-1\\).
//! No slack angle removed, no PV magnitude removed.
//!
//! That is deliberately *not* the power flow's reduced state. Newton solves
//! with bus types fixed, so it drops the slack angle and every PV magnitude;
//! AC-OPF makes generator voltages **decision variables**, so the rows Newton
//! discards are exactly the ones the optimizer needs. Emitting the full space
//! and letting the caller select is both more general and easier to check —
//! the reduced form is a submatrix, whereas the reverse would require
//! rebuilding.
//!
//! # Conventions
//!
//! Both functions return **full symmetric** `(row, col, value)` triplets, with
//! each off-diagonal appearing twice. That is the opposite of
//! [`LinearProgram::hessian`](crate::opf::LinearProgram::hessian), which wants
//! the lower triangle only — a caller crossing into the optimizer must filter
//! to `row >= col`, and dropping that step silently doubles every off-diagonal
//! term. Full symmetric is right *here* because these are matrices to
//! multiply and finite-difference, not yet an objective to minimize.
//!
//! Duplicate `(row, col)` pairs may be emitted and **sum**, the same
//! accumulation semantics [`YBus`](crate::network::YBus) documents.

use crate::network::YBusSparse;
use crate::types::Bus;

/// The polar injection equations' shared per-pair trigonometry.
///
/// For an ordered pair \\((i, k)\\) with \\(\theta_{ik} = \theta_i -
/// \theta_k\\):
///
/// \\[ c_{ik} = G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik}, \qquad
///     s_{ik} = G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik} \\]
///
/// Every derivative below is a product of these with voltage magnitudes,
/// because differentiating in \\(\theta\\) just rotates one into the other:
/// \\(\partial c/\partial\theta_i = -s\\), \\(\partial s/\partial\theta_i =
/// c\\), and the \\(\theta_k\\) derivatives are their negatives. Naming them
/// once is what keeps the second-derivative bookkeeping below tractable.
#[derive(Clone, Copy, Debug)]
struct Pair {
    c: f64,
    s: f64,
}

fn pair(buses: &[Bus], i: usize, k: usize, y: num_complex::Complex<f64>) -> Pair {
    let angle = buses[i].voltage_ang - buses[k].voltage_ang;
    let (sin, cos) = angle.sin_cos();
    Pair { c: y.re * cos + y.im * sin, s: y.re * sin - y.im * cos }
}

/// First derivatives of every injection, over the full \\(2n\\) state.
///
/// Rows are the equations — \\(P_0 \dots P_{n-1}\\) then \\(Q_0 \dots
/// Q_{n-1}\\) — and columns the state, so entry \\((i, j)\\) is
/// \\(\partial P_i/\partial x_j\\) for \\(i < n\\).
///
/// This duplicates what [`JacobianPattern`](crate::jacobian::JacobianPattern)
/// computes, and duplicating it is the point. That one is written for Newton:
/// it emits only the reduced rows and columns, and it reads the diagonal
/// blocks out of precomputed `p_calc`/`q_calc` (`H_ii = −Q_i − V_i²B_ii` and
/// friends). This one sums over neighbours directly and shares no line of code
/// with it. Where they overlap they must agree — which makes the mature,
/// heavily exercised assembler an independent oracle for this one, and this
/// one in turn the thing the Hessian is finite-differenced against. Neither
/// check would mean much if the two shared an implementation.
pub fn injection_jacobian(buses: &[Bus], ybus: &YBusSparse) -> Vec<(usize, usize, f64)> {
    let n = buses.len();
    let mut out = Vec::new();

    for i in 0..n {
        let vm_i = buses[i].voltage_mag;
        let y_ii = ybus.get(i, i);

        // Self terms: the k = i contribution to P_i is V_i²G_ii and to Q_i is
        // −V_i²B_ii, neither of which depends on any angle.
        let mut dp_dtheta_i = 0.0;
        let mut dq_dtheta_i = 0.0;
        let mut dp_dv_i = 2.0 * vm_i * y_ii.re;
        let mut dq_dv_i = -2.0 * vm_i * y_ii.im;

        for &(k, y) in ybus.row(i) {
            if k == i {
                continue;
            }
            let vm_k = buses[k].voltage_mag;
            let Pair { c, s } = pair(buses, i, k, y);

            dp_dtheta_i += vm_i * vm_k * (-s);
            dq_dtheta_i += vm_i * vm_k * c;
            dp_dv_i += vm_k * c;
            dq_dv_i += vm_k * s;

            // Off-diagonal columns.
            out.push((i, k, vm_i * vm_k * s));
            out.push((i, n + k, vm_i * c));
            out.push((n + i, k, -vm_i * vm_k * c));
            out.push((n + i, n + k, vm_i * s));
        }

        out.push((i, i, dp_dtheta_i));
        out.push((i, n + i, dp_dv_i));
        out.push((n + i, i, dq_dtheta_i));
        out.push((n + i, n + i, dq_dv_i));
    }

    out
}

/// \\(\sum_i \lambda_i \nabla^2 P_i + \sum_i \mu_i \nabla^2 Q_i\\), over the
/// full \\(2n\\) state.
///
/// `lambda` weights the active-power equations and `mu` the reactive ones;
/// both are indexed by bus and must have length `buses.len()`. In an AC-OPF
/// these are the balance constraints' multipliers, so passing the ones a
/// solver hands back gives the constraint half of the Lagrangian's Hessian
/// directly.
///
/// # Structure
///
/// The result is symmetric and has the Jacobian's sparsity — no more. That is
/// worth stating because it is not obvious: \\(\nabla^2 P_i\\) has *no*
/// entry at \\((\theta_k, \theta_m)\\) for two distinct neighbours \\(k, m\\)
/// of \\(i\\), even though both appear in \\(P_i\\). Each term of \\(P_i\\)
/// involves exactly one neighbour, so differentiating twice can never couple
/// two of them. A second-order method therefore needs no more fill than the
/// first-order one, which is the reason AC-OPF stays tractable on a network's
/// natural sparsity.
///
/// # Panics
///
/// If `lambda` or `mu` is not `buses.len()` long — a mismatch there would
/// otherwise silently weight the wrong bus.
pub fn weighted_injection_hessian(
    buses: &[Bus],
    ybus: &YBusSparse,
    lambda: &[f64],
    mu: &[f64],
) -> Vec<(usize, usize, f64)> {
    let n = buses.len();
    assert_eq!(lambda.len(), n, "lambda must carry one multiplier per bus");
    assert_eq!(mu.len(), n, "mu must carry one multiplier per bus");

    let mut out = Vec::new();

    for i in 0..n {
        let (w, z) = (lambda[i], mu[i]);
        if w == 0.0 && z == 0.0 {
            continue;
        }
        let vm_i = buses[i].voltage_mag;
        let y_ii = ybus.get(i, i);

        // The self term V_i²G_ii (and −V_i²B_ii) is quadratic in |V_i| and
        // independent of every angle, so it contributes to exactly one entry.
        out.push((n + i, n + i, 2.0 * (w * y_ii.re - z * y_ii.im)));

        // Accumulated across neighbours, because ∂²P_i/∂θ_i² and
        // ∂²P_i/∂|V_i|∂θ_i are sums over the whole row.
        let mut d2_theta_i = 0.0;
        let mut d2_v_i_theta_i = 0.0;

        for &(k, y) in ybus.row(i) {
            if k == i {
                continue;
            }
            let vm_k = buses[k].voltage_mag;
            let Pair { c, s } = pair(buses, i, k, y);

            // Two weighted combinations carry everything. `t` is the angle-
            // angle and magnitude-magnitude coefficient; `r` the mixed one.
            // They differ because differentiating in θ rotates c into −s and
            // s into c, so the mixed derivatives pick up the *other* pairing.
            let t = vm_i * vm_k * (w * c + z * s);
            let u = w * c + z * s;
            let r = -w * s + z * c;

            // θθ: the same magnitude with alternating sign, since P_i depends
            // on θ_i and θ_k only through their difference.
            d2_theta_i -= t;
            out.push((i, k, t));
            out.push((k, i, t));
            out.push((k, k, -t));

            // |V||V|: bilinear in V_i·V_k, so only the cross terms survive.
            out.push((n + i, n + k, u));
            out.push((n + k, n + i, u));

            // Mixed. Both orderings are emitted because the caller gets a
            // full symmetric matrix; mixed partials commute, so each value
            // appears at (row, col) and (col, row).
            d2_v_i_theta_i += vm_k * r;
            out.push((n + i, k, -vm_k * r));
            out.push((k, n + i, -vm_k * r));
            out.push((n + k, i, vm_i * r));
            out.push((i, n + k, vm_i * r));
            out.push((n + k, k, -vm_i * r));
            out.push((k, n + k, -vm_i * r));
        }

        out.push((i, i, d2_theta_i));
        out.push((n + i, i, d2_v_i_theta_i));
        out.push((i, n + i, d2_v_i_theta_i));
    }

    out
}

/// Sums triplets into a dense matrix. For tests and small problems only —
/// the whole point of the sparse form is not to do this.
pub fn to_dense(triplets: &[(usize, usize, f64)], dim: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; dim]; dim];
    for &(r, c, v) in triplets {
        m[r][c] += v;
    }
    m
}
