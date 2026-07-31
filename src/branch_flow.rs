//! Branch terminal power flows and their partial derivatives.
//!
//! Until this module, [`network::power_injections`](crate::network::power_injections)
//! was the only place in the crate where complex power was formed at all, and it
//! is strictly bus-level: `S_k = V_k · (Y V)_k^*`. That answers "what does the
//! network inject at this bus", not "what flows through this branch, measured at
//! this end of it" — which is what a line/transformer power sensor reports, and
//! what PGM's own `sym_output.json` records per `line` entry.
//!
//! Both quantities are needed by state estimation: bus-injection measurements
//! reuse `power_injections` and the `H`/`N`/`M`/`L` derivative blocks already in
//! [`crate::jacobian`], while branch power measurements need what is here.
//!
//! # Model
//!
//! Every branch is reduced to its four π-model admittance entries
//! `[yff, yft, ytf, ytt]` — the same `[Complex<f64>; 4]` that
//! [`network::branch_calc_param`](crate::network::branch_calc_param) already
//! produces for transformers, and that [`network::build_ybus`](crate::network::build_ybus)
//! forms inline for lines. The terminal flows are then
//!
//! ```text
//! S_from = V_i · (yff·V_i + yft·V_j)^*
//! S_to   = V_j · (ytf·V_i + ytt·V_j)^*
//! ```
//!
//! Expanding with `V = v·e^(jθ)`, `y = g + jb` and `θ_ij = θ_i − θ_j` gives the
//! real form this module evaluates, for the terminal whose own bus is "near" and
//! whose opposite bus is "far":
//!
//! ```text
//! P = v_near²·g_self + v_near·v_far·( g_mut·cos θ + b_mut·sin θ)
//! Q = −v_near²·b_self + v_near·v_far·( g_mut·sin θ − b_mut·cos θ)
//! ```
//!
//! where the from-terminal reads `(g_self, b_self)` off `yff` and `(g_mut,
//! b_mut)` off `yft`, and the to-terminal reads them off `ytt` and `ytf` with
//! bus `j` as near and bus `i` as far. The two terminals are therefore the same
//! expression under a relabelling, which is why [`terminal_flow`] takes a
//! [`Terminal`] and reorders its inputs rather than implementing two
//! near-duplicate formulas.

use num_complex::Complex;

use crate::network::branch_calc_param;
use crate::types::{Bus, Line, Transformer};

/// Which end of a branch a flow is measured at. Matches the first two variants
/// of PGM's `MeasuredTerminalType` (`branch_from` = 0, `branch_to` = 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terminal {
    From,
    To,
}

/// A branch reduced to the four π-model admittance entries plus its endpoints.
///
/// Deliberately independent of whether the branch started as a [`Line`] or a
/// [`Transformer`]: once reduced, terminal flows do not care, and state
/// estimation addresses branches by a single flat index across both.
#[derive(Clone, Copy, Debug)]
pub struct BranchParams {
    pub from: usize,
    pub to: usize,
    /// `[yff, yft, ytf, ytt]`, the same layout `branch_calc_param` returns.
    pub y: [Complex<f64>; 4],
}

impl BranchParams {
    /// `(y_self, y_mutual)` as seen from `terminal`, i.e. the two entries whose
    /// row is that terminal's own bus.
    fn seen_from(&self, terminal: Terminal) -> (Complex<f64>, Complex<f64>) {
        match terminal {
            Terminal::From => (self.y[0], self.y[1]),
            Terminal::To => (self.y[3], self.y[2]),
        }
    }

    /// `(near_bus, far_bus)` for `terminal`.
    fn buses(&self, terminal: Terminal) -> (usize, usize) {
        match terminal {
            Terminal::From => (self.from, self.to),
            Terminal::To => (self.to, self.from),
        }
    }
}

/// Reduces a [`Line`] to its π-model entries.
///
/// Mirrors the stamping [`build_ybus`](crate::network::build_ybus) does inline:
/// series admittance `1/(r + jx)`, total shunt split equally between the ends.
///
/// A `from == to` line is gridoxide's representation of a *half-open* branch —
/// `pgm::pgm_to_buses_and_branches` collapses a line with one terminal open into
/// a self-loop carrying only that line's shunt, discarding which end was open.
/// Its entire flow is therefore reported at [`Terminal::From`] and zero at
/// [`Terminal::To`]; a caller needing PGM's actual per-end convention for such a
/// branch has to consult the original `from_status`/`to_status`, which no longer
/// exist on `Line`.
pub fn line_params(line: &Line) -> BranchParams {
    let zero = Complex::new(0.0, 0.0);
    if line.from == line.to {
        return BranchParams {
            from: line.from,
            to: line.to,
            y: [Complex::new(line.g_shunt, line.b_shunt), zero, zero, zero],
        };
    }
    let y_series = Complex::new(1.0, 0.0) / Complex::new(line.r, line.x);
    let y_shunt_half = Complex::new(line.g_shunt / 2.0, line.b_shunt / 2.0);
    let y_diag = y_series + y_shunt_half;
    BranchParams {
        from: line.from,
        to: line.to,
        y: [y_diag, -y_series, -y_series, y_diag],
    }
}

/// Reduces a [`Transformer`] to its π-model entries, reusing
/// [`branch_calc_param`] verbatim — including its open-terminal rules, so a
/// transformer with a disconnected end contributes only its shunt, exactly as
/// it does to the Y-bus.
pub fn transformer_params(t: &Transformer) -> BranchParams {
    BranchParams {
        from: t.from,
        to: t.to,
        y: branch_calc_param(t.y_series, t.y_shunt, t.tap, t.from_status, t.to_status),
    }
}

/// Reduces every branch in a network to [`BranchParams`], lines first and
/// transformers second.
///
/// That order is the crate's flat branch index, and it matches the order
/// [`build_ybus`](crate::network::build_ybus) stamps them in. Note that it is
/// *not* the order of the original PGM input: fully-open lines are dropped
/// during conversion and virtual source branches are appended, so mapping a PGM
/// object id to an index here requires the map built by
/// [`pgm::pgm_to_network`](crate::pgm::pgm_to_network), never arithmetic on
/// input positions.
pub fn branch_params(lines: &[Line], transformers: &[Transformer]) -> Vec<BranchParams> {
    lines
        .iter()
        .map(line_params)
        .chain(transformers.iter().map(transformer_params))
        .collect()
}

/// Complex bus voltages in the form the flow formulas consume.
pub fn bus_voltages(buses: &[Bus]) -> Vec<Complex<f64>> {
    buses
        .iter()
        .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
        .collect()
}

/// Active and reactive power flowing *into* `branch` at `terminal`, in per-unit
/// on the system base.
///
/// Sign convention matches PGM's `p_from`/`p_to`: positive means power entering
/// the branch at that terminal, so a lossless branch would report
/// `p_from = −p_to`.
pub fn terminal_flow(branch: &BranchParams, terminal: Terminal, v: &[Complex<f64>]) -> (f64, f64) {
    let (near, far) = branch.buses(terminal);
    let (y_self, y_mut) = branch.seen_from(terminal);
    let (v_near, th_near) = (v[near].norm(), v[near].arg());
    let (v_far, th_far) = (v[far].norm(), v[far].arg());
    let (s, c) = (th_near - th_far).sin_cos();

    let p = v_near * v_near * y_self.re + v_near * v_far * (y_mut.re * c + y_mut.im * s);
    let q = -v_near * v_near * y_self.im + v_near * v_far * (y_mut.re * s - y_mut.im * c);
    (p, q)
}

/// Partial derivatives of one terminal's `(P, Q)` with respect to the four state
/// variables it depends on.
///
/// "near" is the terminal's own bus, "far" the opposite one. Every other state
/// variable has zero partial, which is what makes the measurement Jacobian's
/// rows sparse: a branch power measurement touches exactly two buses.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlowDerivs {
    pub dp_dtheta_near: f64,
    pub dp_dtheta_far: f64,
    pub dp_dv_near: f64,
    pub dp_dv_far: f64,
    pub dq_dtheta_near: f64,
    pub dq_dtheta_far: f64,
    pub dq_dv_near: f64,
    pub dq_dv_far: f64,
}

/// Analytic derivatives of [`terminal_flow`].
///
/// The angle partials are exactly antisymmetric (`∂/∂θ_far = −∂/∂θ_near`) since
/// both flows depend on the angle difference alone; they are still returned as
/// separate fields so callers assembling a Jacobian row do not have to remember
/// to negate.
///
/// `tests::derivs_match_finite_differences` checks all eight against central
/// differences on an asymmetric branch, which is the real guard here — these
/// formulas are easy to write plausibly and wrongly.
pub fn terminal_flow_derivs(
    branch: &BranchParams,
    terminal: Terminal,
    v: &[Complex<f64>],
) -> FlowDerivs {
    let (near, far) = branch.buses(terminal);
    let (y_self, y_mut) = branch.seen_from(terminal);
    let (v_near, th_near) = (v[near].norm(), v[near].arg());
    let (v_far, th_far) = (v[far].norm(), v[far].arg());
    let (s, c) = (th_near - th_far).sin_cos();

    // The two recurring combinations: `cos_term` appears in P's mutual part and
    // in ∂Q/∂θ, `sin_term` in Q's mutual part and in −∂P/∂θ.
    let cos_term = y_mut.re * c + y_mut.im * s;
    let sin_term = y_mut.re * s - y_mut.im * c;

    FlowDerivs {
        dp_dtheta_near: -v_near * v_far * sin_term,
        dp_dtheta_far: v_near * v_far * sin_term,
        dp_dv_near: 2.0 * v_near * y_self.re + v_far * cos_term,
        dp_dv_far: v_near * cos_term,
        dq_dtheta_near: v_near * v_far * cos_term,
        dq_dtheta_far: -v_near * v_far * cos_term,
        dq_dv_near: -2.0 * v_near * y_self.im + v_far * sin_term,
        dq_dv_far: v_near * sin_term,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_line() -> Line {
        Line { from: 0, to: 1, r: 0.05, x: 0.2, b_shunt: 0.04, g_shunt: 0.01 }
    }

    /// Two buses at different magnitudes *and* angles, so no derivative term can
    /// vanish by accident.
    fn test_voltages() -> Vec<Complex<f64>> {
        vec![
            Complex::from_polar(1.03, 0.07),
            Complex::from_polar(0.97, -0.05),
        ]
    }

    /// The flows a branch reports must be the same power the Y-bus attributes to
    /// that branch — for a two-bus network with a single branch, the sum of the
    /// terminal flows across all branches at a bus is that bus's injection.
    #[test]
    fn terminal_flows_sum_to_bus_injection() {
        let line = test_line();
        let p = line_params(&line);
        let v = test_voltages();

        let (p_from, q_from) = terminal_flow(&p, Terminal::From, &v);
        let (p_to, q_to) = terminal_flow(&p, Terminal::To, &v);

        // S_bus0 = V_0 · (Y V)_0^*, built from this branch's entries alone.
        let i0 = p.y[0] * v[0] + p.y[1] * v[1];
        let s0 = v[0] * i0.conj();
        let i1 = p.y[2] * v[0] + p.y[3] * v[1];
        let s1 = v[1] * i1.conj();

        assert!((p_from - s0.re).abs() < 1e-12, "p_from vs injection");
        assert!((q_from - s0.im).abs() < 1e-12, "q_from vs injection");
        assert!((p_to - s1.re).abs() < 1e-12, "p_to vs injection");
        assert!((q_to - s1.im).abs() < 1e-12, "q_to vs injection");
    }

    /// A lossless, shunt-free branch conserves power: whatever enters one end
    /// leaves the other.
    #[test]
    fn lossless_branch_conserves_active_power() {
        let line = Line { from: 0, to: 1, r: 0.0, x: 0.2, b_shunt: 0.0, g_shunt: 0.0 };
        let p = line_params(&line);
        let v = test_voltages();
        let (p_from, _) = terminal_flow(&p, Terminal::From, &v);
        let (p_to, _) = terminal_flow(&p, Terminal::To, &v);
        assert!((p_from + p_to).abs() < 1e-12, "p_from={p_from} p_to={p_to}");
    }

    /// All eight partials against central differences. The step is 1e-6 and the
    /// tolerance 1e-6, which leaves ~4 orders of margin over the ~1e-10
    /// truncation error of a central difference at this scale.
    #[test]
    fn derivs_match_finite_differences() {
        let branch = line_params(&test_line());
        let base = test_voltages();

        for terminal in [Terminal::From, Terminal::To] {
            let d = terminal_flow_derivs(&branch, terminal, &base);
            let (near, far) = branch.buses(terminal);
            let h = 1e-6;

            // Perturb one polar component of one bus and re-evaluate.
            let bump = |bus: usize, d_mag: f64, d_ang: f64| {
                let mut v = base.clone();
                v[bus] = Complex::from_polar(
                    base[bus].norm() + d_mag,
                    base[bus].arg() + d_ang,
                );
                terminal_flow(&branch, terminal, &v)
            };
            let central = |bus: usize, mag: bool| {
                let (pp, qp) = if mag { bump(bus, h, 0.0) } else { bump(bus, 0.0, h) };
                let (pm, qm) = if mag { bump(bus, -h, 0.0) } else { bump(bus, 0.0, -h) };
                ((pp - pm) / (2.0 * h), (qp - qm) / (2.0 * h))
            };

            let (dp_dth_near, dq_dth_near) = central(near, false);
            let (dp_dth_far, dq_dth_far) = central(far, false);
            let (dp_dv_near, dq_dv_near) = central(near, true);
            let (dp_dv_far, dq_dv_far) = central(far, true);

            let close = |a: f64, b: f64, what: &str| {
                assert!((a - b).abs() < 1e-6, "{terminal:?} {what}: analytic {a}, numeric {b}");
            };
            close(d.dp_dtheta_near, dp_dth_near, "dP/dtheta_near");
            close(d.dp_dtheta_far, dp_dth_far, "dP/dtheta_far");
            close(d.dp_dv_near, dp_dv_near, "dP/dv_near");
            close(d.dp_dv_far, dp_dv_far, "dP/dv_far");
            close(d.dq_dtheta_near, dq_dth_near, "dQ/dtheta_near");
            close(d.dq_dtheta_far, dq_dth_far, "dQ/dtheta_far");
            close(d.dq_dv_near, dq_dv_near, "dQ/dv_near");
            close(d.dq_dv_far, dq_dv_far, "dQ/dv_far");
        }
    }

    /// A self-loop line is a half-open branch: shunt-only, all of it on the
    /// from-terminal.
    #[test]
    fn self_loop_line_is_shunt_only() {
        let line = Line { from: 2, to: 2, r: 0.0, x: 0.0, b_shunt: 0.03, g_shunt: 0.0 };
        let p = line_params(&line);
        assert_eq!(p.y[1], Complex::new(0.0, 0.0));
        assert_eq!(p.y[3], Complex::new(0.0, 0.0));

        let v = vec![Complex::new(1.0, 0.0); 3];
        let (p_from, q_from) = terminal_flow(&p, Terminal::From, &v);
        assert!(p_from.abs() < 1e-12);
        // S = V·(y·V)^* with y = j0.03 gives Q = −0.03 at |V| = 1.
        assert!((q_from + 0.03).abs() < 1e-12, "q_from={q_from}");
        assert_eq!(terminal_flow(&p, Terminal::To, &v), (0.0, 0.0));
    }

    /// Transformer branches go through `branch_calc_param`, so an off-nominal
    /// tap must show up as an asymmetry between the two terminals that a plain
    /// line does not have.
    #[test]
    fn transformer_tap_breaks_terminal_symmetry() {
        let t = Transformer {
            from: 0,
            to: 1,
            from_status: 1,
            to_status: 1,
            y_series: Complex::new(1.0, 0.0) / Complex::new(0.01, 0.1),
            y_shunt: Complex::new(0.0, 0.0),
            tap: Complex::new(1.05, 0.0),
        };
        let p = transformer_params(&t);
        assert!((p.y[0] - p.y[3]).norm() > 1e-9, "tap should make yff != ytt");

        let v = test_voltages();
        let (p_from, _) = terminal_flow(&p, Terminal::From, &v);
        let (p_to, _) = terminal_flow(&p, Terminal::To, &v);
        // Still a passive element: losses are positive, so the flows cannot be
        // equal and opposite.
        assert!(p_from + p_to > 0.0, "losses should be positive");
    }
}
