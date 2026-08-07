//! Every quantity the estimator measures, as one shape.
//!
//! A power measurement — a bus injection, a branch terminal flow, a source's
//! output, a shunt's draw, power-grid-model's node injection — is always the
//! same thing written five ways: a *current*, linear in the complex bus
//! voltages, evaluated against the voltage at one bus.
//!
//! ```text
//! I = Σ c_k·V_k        S = V_at · conj(I)
//! ```
//!
//! `c` and `at` are all that distinguish the five. A bus injection's `c` is its
//! Y-bus row; a branch terminal's is that branch's own half-row; a shunt's is a
//! single negated admittance. Writing them once means the measurement model,
//! its Jacobian and the iterative-linear method's row builder stop being three
//! independent statements of the same physics that can disagree — a
//! disagreement that used to surface only as `the_two_methods_agree` failing,
//! with nothing to say which side was wrong.
//!
//! It is also what keeps three-phase tractable. A three-phase branch terminal
//! flow depends on six phasors rather than two, which under a fixed-arity
//! `FlowDerivs` means twenty-four named fields and a second finite-difference
//! test; here it is a longer coefficient list and the same loop.

use num_complex::Complex;

use crate::branch_flow::Terminal;
use crate::measurement::Target;

use super::SeNetwork;

/// The imaginary unit, spelled once so the derivative expressions below read as
/// their formulas.
const J: Complex<f64> = Complex::new(0.0, 1.0);

/// A terminal current as a sparse linear functional of the complex bus
/// voltages, plus the bus whose voltage turns it into a power.
#[derive(Clone, Debug, PartialEq)]
pub struct CurrentFunctional {
    /// The bus whose voltage forms the power `S = V_at · conj(I)`.
    pub at: usize,
    /// `I = Σ c_k·V_k`, as `(bus, c)`.
    ///
    /// Duplicate buses are allowed and sum, which is what makes gridoxide's
    /// half-open branch — represented as a self-loop, so `from == to` — come
    /// out right, and what lets a node injection concatenate its Y-bus row with
    /// its source and shunt terms instead of merging them.
    pub coefficients: Vec<(usize, Complex<f64>)>,
}

/// One bus's contribution to a quantity's derivative, as the pair of partials
/// with respect to that bus's polar state.
///
/// Complex because the quantity is: for a power, `re` is `∂P` and `im` is `∂Q`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Partial {
    pub bus: usize,
    pub d_dtheta: Complex<f64>,
    pub d_dvmag: Complex<f64>,
}

impl CurrentFunctional {
    /// `I = Σ c_k·V_k`.
    pub fn current(&self, v: &[Complex<f64>]) -> Complex<f64> {
        self.coefficients.iter().map(|&(k, c)| c * v[k]).sum()
    }

    /// `S = V_at · conj(I)`, i.e. `(P, Q)` as one complex number.
    pub fn power(&self, v: &[Complex<f64>]) -> Complex<f64> {
        v[self.at] * self.current(v).conj()
    }

    /// Partials of `S` with respect to every bus it depends on, appended to
    /// `out` in coefficient order.
    ///
    /// ```text
    /// ∂S/∂θ_k    = −j·V_at·conj(c_k·V_k)      + [k = at]·j·S
    /// ∂S/∂|V_k|  =    V_at·conj(c_k·e^{jθ_k}) + [k = at]·S/|V_at|
    /// ```
    ///
    /// Every coefficient produces an entry, including one whose value is
    /// numerically zero. `H`'s sparsity pattern has to depend on the topology
    /// alone and not on the state — dropping a zero would shrink the row at a
    /// flat start, where many `sin(θ_i − θ_k)` vanish exactly, and grow it again
    /// next iteration, invalidating the cached symbolic factorization.
    ///
    /// An *empty* functional produces nothing at all rather than a lone `at`
    /// entry. Its quantity is identically zero with no dependence on anything,
    /// and a row of structural zeros would defeat
    /// [`mask_untouched`](super::jacobian::mask_untouched), which is what pins a
    /// genuinely disconnected bus.
    pub fn power_partials(&self, v: &[Complex<f64>], out: &mut Vec<Partial>) {
        power_partials_of(self.at, &self.coefficients, v, out)
    }

    /// `I_local = S/|V_at|` — see [`local_current_partials_of`].
    pub fn local_current(&self, v: &[Complex<f64>]) -> Complex<f64> {
        local_current_of(self.at, &self.coefficients, v)
    }

    pub fn current_partials(&self, v: &[Complex<f64>], out: &mut Vec<Partial>) {
        current_partials_of(&self.coefficients, v, out)
    }

    pub fn local_current_partials(&self, v: &[Complex<f64>], out: &mut Vec<Partial>) {
        local_current_partials_of(self.at, &self.coefficients, v, out)
    }
}

/// [`CurrentFunctional::power_partials`] against borrowed parts.
///
/// Callers that already hold a coefficient slice — `se::constraints` hands over
/// a Y-bus row directly — use this rather than building a functional to own a
/// copy of it, which would be an allocation per constraint row per iteration.
pub fn power_partials_of(
    at: usize,
    coefficients: &[(usize, Complex<f64>)],
    v: &[Complex<f64>],
    out: &mut Vec<Partial>,
) {
    {
        if coefficients.is_empty() {
            return;
        }
        let v_at = v[at];
        let s = v_at * coefficients.iter().map(|&(k, c)| c * v[k]).sum::<Complex<f64>>().conj();
        let inv_mag = if v_at.norm() > 0.0 { v_at.norm().recip() } else { 0.0 };
        let mut saw_at = false;

        for &(k, c) in coefficients {
            let unit_k = unit(v[k]);
            let mut d_dtheta = -J * v_at * (c * v[k]).conj();
            let mut d_dvmag = v_at * (c * unit_k).conj();
            if k == at {
                saw_at = true;
                d_dtheta += J * s;
                d_dvmag += s * inv_mag;
            }
            out.push(Partial { bus: k, d_dtheta, d_dvmag });
        }

        // `at` is one of the coefficients for every target gridoxide builds, so
        // this does not fire today. It is here because the identity `S = V_at ·
        // conj(I)` does not require it, and a functional that ever separates the
        // two would otherwise silently lose the leading factor's derivative.
        if !saw_at {
            out.push(Partial { bus: at, d_dtheta: J * s, d_dvmag: s * inv_mag });
        }
    }
}

/// Partials of `I` itself, rather than of the power it forms.
///
/// A *global-angle* current sensor measures `I` directly, and `I` is linear in
/// the voltages, so these are simply the coefficients rotated:
///
/// ```text
/// ∂I/∂θ_k   = j·c_k·V_k
/// ∂I/∂|V_k| = c_k·e^{jθ_k}
/// ```
pub fn current_partials_of(
    coefficients: &[(usize, Complex<f64>)],
    v: &[Complex<f64>],
    out: &mut Vec<Partial>,
) {
    for &(k, c) in coefficients {
        let unit_k = unit(v[k]);
        out.push(Partial { bus: k, d_dtheta: J * c * v[k], d_dvmag: c * unit_k });
    }
}

/// Partials of `S/|V_at|`, the *local-angle* current frame.
///
/// A local-angle sensor measures the phase shift between voltage and current
/// rather than an absolute angle, so what it reads is
/// `I_local = conj(I)·e^{jθ_at}`. Substituting `conj(I) = S/V_at` collapses that
/// to `S/|V_at|` — the same power the estimator already differentiates, divided
/// by a magnitude. power-grid-model reaches the identical place from the other
/// direction, computing its local-angle Jacobian as the power form
/// post-multiplied by `abs_u_chi_inv`.
///
/// Only the leading magnitude carries the extra term, since `|V_at|` depends on
/// no angle at all:
///
/// ```text
/// ∂/∂θ_k   = (∂S/∂θ_k)/|V_at|
/// ∂/∂|V_k| = (∂S/∂|V_k|)/|V_at| − [k = at]·S/|V_at|²
/// ```
pub fn local_current_partials_of(
    at: usize,
    coefficients: &[(usize, Complex<f64>)],
    v: &[Complex<f64>],
    out: &mut Vec<Partial>,
) {
    let start = out.len();
    power_partials_of(at, coefficients, v, out);
    if out.len() == start {
        return;
    }
    let mag = v[at].norm();
    if mag <= 0.0 {
        return;
    }
    let s = local_current_of(at, coefficients, v) * mag;
    for p in out[start..].iter_mut() {
        p.d_dtheta /= mag;
        p.d_dvmag /= mag;
        if p.bus == at {
            p.d_dvmag -= s / (mag * mag);
        }
    }
}

/// `S/|V_at|`, the quantity a local-angle current sensor reads.
pub fn local_current_of(
    at: usize,
    coefficients: &[(usize, Complex<f64>)],
    v: &[Complex<f64>],
) -> Complex<f64> {
    let mag = v[at].norm();
    if mag <= 0.0 {
        return Complex::new(0.0, 0.0);
    }
    let i: Complex<f64> = coefficients.iter().map(|&(k, c)| c * v[k]).sum();
    v[at] * i.conj() / mag
}

/// A unit phasor in `z`'s direction, or `1` where `z` has no direction.
fn unit(z: Complex<f64>) -> Complex<f64> {
    if z.norm() > 0.0 { z / z.norm() } else { Complex::new(1.0, 0.0) }
}

impl SeNetwork {
    /// The current functional a measurement target names, or `None` if the
    /// target does not resolve in this network.
    ///
    /// This is the single description of what each target measures. Everything
    /// downstream — `h(x)`, the Jacobian, the iterative-linear row builder —
    /// reads it rather than restating it.
    pub fn functional(&self, target: Target) -> Option<CurrentFunctional> {
        Some(match target {
            // A bus injection is its own Y-bus row: `I_i = Σ_k Y_ik·V_k`.
            Target::Bus(bus) => {
                CurrentFunctional { at: bus, coefficients: self.ybus.row(bus).to_vec() }
            }
            // A current sensor names the same terminal as a power sensor and so
            // resolves to the same functional. What differs is which quantity of
            // it the estimator reads — `current` or `local_current` rather than
            // `power` — not the coefficients.
            Target::BranchTerminal { branch, terminal }
            | Target::BranchTerminalCurrent { branch, terminal, .. } => {
                let b = self.branches.get(branch)?;
                let (near, far) = b.buses(terminal);
                let (y_self, y_mut) = b.seen_from(terminal);
                CurrentFunctional { at: near, coefficients: vec![(near, y_self), (far, y_mut)] }
            }
            // A source's injection into its node is the negation of the current
            // flowing from that node into the synthesized branch, and negating
            // the coefficients negates the power with it.
            Target::SourceInjection { branch } => {
                let b = self.branches.get(branch)?;
                let (near, far) = b.buses(Terminal::To);
                let (y_self, y_mut) = b.seen_from(Terminal::To);
                CurrentFunctional { at: near, coefficients: vec![(near, -y_self), (far, -y_mut)] }
            }
            // A shunt consumes `S = |V|²·conj(y_sh)`; as an injection that is
            // negated, which is the `-y_sh` here.
            Target::ShuntInjection { bus } => CurrentFunctional {
                at: bus,
                coefficients: vec![(bus, -self.shunt_y[bus])],
            },
            // power-grid-model's node injection is the sum over *every*
            // appliance at the bus. gridoxide's bus injection covers loads and
            // generators; sources and shunts are structural and are added back.
            // All three share `at`, so concatenating their coefficients adds
            // their powers.
            Target::NodeInjection(bus) => {
                let mut coefficients = self.ybus.row(bus).to_vec();
                for &branch in &self.source_branches[bus] {
                    let b = &self.branches[branch];
                    let (near, far) = b.buses(Terminal::To);
                    let (y_self, y_mut) = b.seen_from(Terminal::To);
                    debug_assert_eq!(near, bus, "a source branch must feed the bus it is listed on");
                    coefficients.push((near, -y_self));
                    coefficients.push((far, -y_mut));
                }
                coefficients.push((bus, -self.shunt_y[bus]));
                CurrentFunctional { at: bus, coefficients }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch_flow::{terminal_flow, terminal_flow_derivs};

    fn state(buses: &[crate::types::Bus]) -> Vec<Complex<f64>> {
        buses.iter().map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang)).collect()
    }

    /// The functional form reproduces `branch_flow::terminal_flow` exactly.
    ///
    /// This is the identity the whole module rests on: a terminal flow *is*
    /// `V_near · conj(y_self·V_near + y_mut·V_far)`, written out longhand in
    /// `branch_flow` and as two coefficients here.
    #[test]
    fn a_branch_terminal_functional_matches_terminal_flow() {
        let (net, mut buses) = super::super::tests::two_bus_net();
        buses[0].voltage_mag = 1.04;
        buses[0].voltage_ang = 0.11;
        buses[1].voltage_mag = 0.97;
        buses[1].voltage_ang = -0.07;
        let v = state(&buses);

        for terminal in [Terminal::From, Terminal::To] {
            let f = net
                .functional(Target::BranchTerminal { branch: 0, terminal })
                .expect("branch 0 exists");
            let s = f.power(&v);
            let (p, q) = terminal_flow(&net.branches[0], terminal, &v);
            assert!((s.re - p).abs() < 1e-12, "{terminal:?}: P {} vs {p}", s.re);
            assert!((s.im - q).abs() < 1e-12, "{terminal:?}: Q {} vs {q}", s.im);
        }
    }

    /// And its partials reproduce `terminal_flow_derivs`' eight fields.
    ///
    /// `branch_flow`'s closed forms stay as the public branch-flow API and are
    /// finite-difference tested in their own module, so agreeing with them is a
    /// check against an independent derivation rather than against a restatement
    /// of this one.
    #[test]
    fn branch_terminal_partials_match_the_closed_form() {
        let (net, mut buses) = super::super::tests::two_bus_net();
        buses[0].voltage_mag = 1.04;
        buses[0].voltage_ang = 0.11;
        buses[1].voltage_mag = 0.97;
        buses[1].voltage_ang = -0.07;
        let v = state(&buses);

        for terminal in [Terminal::From, Terminal::To] {
            let f = net
                .functional(Target::BranchTerminal { branch: 0, terminal })
                .expect("branch 0 exists");
            let mut partials = Vec::new();
            f.power_partials(&v, &mut partials);

            let d = terminal_flow_derivs(&net.branches[0], terminal, &v);
            let (near, _far) = net.branches[0].buses(terminal);
            let want = |bus: usize| -> (f64, f64, f64, f64) {
                if bus == near {
                    (d.dp_dtheta_near, d.dq_dtheta_near, d.dp_dv_near, d.dq_dv_near)
                } else {
                    (d.dp_dtheta_far, d.dq_dtheta_far, d.dp_dv_far, d.dq_dv_far)
                }
            };
            assert_eq!(partials.len(), 2, "a two-bus branch touches two buses");
            for p in &partials {
                let (dp_dth, dq_dth, dp_dv, dq_dv) = want(p.bus);
                assert!((p.d_dtheta.re - dp_dth).abs() < 1e-10, "dP/dθ at {}", p.bus);
                assert!((p.d_dtheta.im - dq_dth).abs() < 1e-10, "dQ/dθ at {}", p.bus);
                assert!((p.d_dvmag.re - dp_dv).abs() < 1e-10, "dP/dV at {}", p.bus);
                assert!((p.d_dvmag.im - dq_dv).abs() < 1e-10, "dQ/dV at {}", p.bus);
            }
        }
    }

    /// Every functional's partials against central differences of its own
    /// `power`, which is the guard that does not depend on any other module
    /// being right.
    #[test]
    fn partials_match_finite_differences_for_every_target() {
        let (net, mut buses) = super::super::tests::two_bus_net();
        // A state where nothing vanishes by symmetry.
        buses[0].voltage_mag = 1.03;
        buses[0].voltage_ang = 0.13;
        buses[1].voltage_mag = 0.96;
        buses[1].voltage_ang = -0.09;

        let targets = [
            Target::Bus(1),
            Target::NodeInjection(1),
            Target::ShuntInjection { bus: 1 },
            Target::BranchTerminal { branch: 0, terminal: Terminal::From },
            Target::BranchTerminal { branch: 0, terminal: Terminal::To },
            Target::SourceInjection { branch: 0 },
        ];

        const H: f64 = 1e-6;
        for target in targets {
            let f = net.functional(target).expect("target resolves");
            let mut raw = Vec::new();
            f.power_partials(&state(&buses), &mut raw);

            // Duplicate buses sum, and they genuinely occur: a node injection
            // lists its Y-bus row *and* its source branch's terms, which land on
            // the same two buses and largely cancel. Comparing an individual
            // entry against a finite difference would be comparing half a
            // derivative.
            let mut partials: Vec<Partial> = Vec::new();
            for e in &raw {
                match partials.iter_mut().find(|p| p.bus == e.bus) {
                    Some(p) => {
                        p.d_dtheta += e.d_dtheta;
                        p.d_dvmag += e.d_dvmag;
                    }
                    None => partials.push(*e),
                }
            }
            if matches!(target, Target::NodeInjection(_)) {
                assert!(raw.len() > partials.len(), "the duplicate-summing path should be exercised");
            }

            for p in &partials {
                let mut shifted = buses.clone();
                shifted[p.bus].voltage_ang += H;
                let up = f.power(&state(&shifted));
                shifted[p.bus].voltage_ang -= 2.0 * H;
                let down = f.power(&state(&shifted));
                let fd = (up - down) / (2.0 * H);
                assert!(
                    (fd - p.d_dtheta).norm() < 1e-5,
                    "{target:?} d/dθ at bus {}: analytic {} vs finite difference {fd}",
                    p.bus,
                    p.d_dtheta
                );

                let mut shifted = buses.clone();
                shifted[p.bus].voltage_mag += H;
                let up = f.power(&state(&shifted));
                shifted[p.bus].voltage_mag -= 2.0 * H;
                let down = f.power(&state(&shifted));
                let fd = (up - down) / (2.0 * H);
                assert!(
                    (fd - p.d_dvmag).norm() < 1e-5,
                    "{target:?} d/d|V| at bus {}: analytic {} vs finite difference {fd}",
                    p.bus,
                    p.d_dvmag
                );
            }
        }
    }

    /// The three identities the local-angle frame rests on.
    ///
    /// A local-angle sensor reads `conj(I)·e^{jθ_at}`; that equals `S/|V_at|`
    /// once `conj(I) = S/V_at` is substituted, which is what lets the estimator
    /// reuse the power path for it instead of deriving a third quantity.
    #[test]
    fn local_current_is_the_power_over_the_magnitude() {
        let (net, mut buses) = super::super::tests::two_bus_net();
        buses[0].voltage_mag = 1.041;
        buses[0].voltage_ang = 0.137;
        buses[1].voltage_mag = 0.958;
        buses[1].voltage_ang = -0.061;
        let v = state(&buses);

        let f = net
            .functional(Target::BranchTerminal { branch: 0, terminal: Terminal::From })
            .expect("branch 0");
        let local = f.local_current(&v);
        let s = f.power(&v);
        let i = f.current(&v);
        let v_at = v[f.at];

        assert!((local - s / v_at.norm()).norm() < 1e-12, "S/|V| identity");
        assert!(
            (local - i.conj() * Complex::from_polar(1.0, v_at.arg())).norm() < 1e-12,
            "conj(I)·e^{{jθ}} identity"
        );
        // And it is genuinely a different quantity from the global current, so
        // a test confusing the two would not pass by coincidence.
        assert!((local - i).norm() > 1e-3, "the two frames should differ here");
    }

    /// Both current forms' partials against central differences of their own
    /// values.
    #[test]
    fn current_partials_match_finite_differences() {
        let (net, mut buses) = super::super::tests::two_bus_net();
        buses[0].voltage_mag = 1.031;
        buses[0].voltage_ang = 0.121;
        buses[1].voltage_mag = 0.967;
        buses[1].voltage_ang = -0.074;

        let f = net
            .functional(Target::BranchTerminal { branch: 0, terminal: Terminal::To })
            .expect("branch 0");

        const H: f64 = 1e-6;
        for (name, value, partials) in [
            (
                "global",
                Box::new(|v: &[Complex<f64>]| f.current(v)) as Box<dyn Fn(&[Complex<f64>]) -> Complex<f64>>,
                {
                    let mut o = Vec::new();
                    f.current_partials(&state(&buses), &mut o);
                    o
                },
            ),
            (
                "local",
                Box::new(|v: &[Complex<f64>]| f.local_current(v)),
                {
                    let mut o = Vec::new();
                    f.local_current_partials(&state(&buses), &mut o);
                    o
                },
            ),
        ] {
            // Sum duplicates before comparing, for the same reason the power
            // check does.
            let mut summed: Vec<Partial> = Vec::new();
            for e in &partials {
                match summed.iter_mut().find(|p| p.bus == e.bus) {
                    Some(p) => {
                        p.d_dtheta += e.d_dtheta;
                        p.d_dvmag += e.d_dvmag;
                    }
                    None => summed.push(*e),
                }
            }
            for p in &summed {
                let mut shifted = buses.clone();
                shifted[p.bus].voltage_ang += H;
                let up = value(&state(&shifted));
                shifted[p.bus].voltage_ang -= 2.0 * H;
                let fd = (up - value(&state(&shifted))) / (2.0 * H);
                assert!((fd - p.d_dtheta).norm() < 1e-5, "{name} d/dθ at {}", p.bus);

                let mut shifted = buses.clone();
                shifted[p.bus].voltage_mag += H;
                let up = value(&state(&shifted));
                shifted[p.bus].voltage_mag -= 2.0 * H;
                let fd = (up - value(&state(&shifted))) / (2.0 * H);
                assert!((fd - p.d_dvmag).norm() < 1e-5, "{name} d/d|V| at {}", p.bus);
            }
        }
    }

    /// A functional with no coefficients yields no partials at all, so an
    /// isolated bus's column stays untouched and `mask_untouched` can pin it.
    #[test]
    fn an_empty_functional_produces_no_partials() {
        let f = CurrentFunctional { at: 0, coefficients: Vec::new() };
        let v = vec![Complex::new(1.0, 0.0)];
        assert_eq!(f.power(&v), Complex::new(0.0, 0.0));
        let mut partials = Vec::new();
        f.power_partials(&v, &mut partials);
        assert!(partials.is_empty());
    }
}
