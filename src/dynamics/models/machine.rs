//! Synchronous machine models.
//!
//! Phase 1 ships the classical model only. [`GenCls`] is deliberately the
//! first: it is the model the equal-area criterion is written for, so it is
//! the one with a closed-form answer to compare against (gate G2), and it
//! exercises every part of the DAE machinery — a differential block, a
//! network coupling in both directions, and a nonlinear injection — without
//! the parameter surface of a round-rotor model.
//!
//! The transient and subtransient models (`GenTransient`, `GenRound`) are
//! phase 3 and slot in beside this one with no change to
//! [`DynamicModel`](super::DynamicModel).

use num_complex::Complex;

use super::{DynamicModel, InitError, ModelJacobian};

/// The classical (second-order) synchronous machine: a constant voltage
/// magnitude behind transient reactance, with the swing equation for the
/// rotor.
///
/// States: `δ` (rotor angle relative to the synchronous frame, rad) and `ω`
/// (rotor speed, per unit, so `1.0` is synchronous).
///
/// \\[ \dot{\delta} = \Omega_b(\omega - 1), \qquad
///    \dot{\omega} = \frac{P_m - P_e - D(\omega - 1)}{2H} \\]
///
/// **`P_e` here is the air-gap power, not the terminal power.** With armature
/// resistance the two differ by the stator copper loss, and it is the air-gap
/// power that the rotor actually feels:
/// `Re(E·conj(I)) = Re(V·conj(I)) + |I|²·r_a`. Using the terminal power
/// instead produces a machine that is damped by its own resistance, which is
/// wrong in a way that looks entirely plausible — the swing decays, just not
/// for a real reason.
#[derive(Clone, Debug)]
pub struct GenCls {
    /// Inertia constant, seconds, **already converted to the network base**.
    h: f64,
    /// Damping coefficient, per unit on the network base.
    d: f64,
    /// Norton admittance `1/(r_a + j x'_d)`, network base. Constant, and
    /// stamped into `Y` rather than evaluated per step.
    y: Complex<f64>,
    /// Series impedance `r_a + j x'_d`, network base. Kept alongside `y`
    /// because initialization needs it directly and inverting twice invites
    /// the two to disagree.
    z: Complex<f64>,
    /// `2π·f_nom`, rad/s. The one place a physical time unit enters.
    omega_base: f64,
    /// Constant internal EMF magnitude, latched by [`initialize`](
    /// DynamicModel::initialize). Zero until then.
    e_mag: f64,
    /// Mechanical power, per unit on the network base, latched likewise.
    /// Constant for this model — there is no governor in phase 1.
    p_m: f64,
}

/// [`GenCls`]'s parameters **as a data file states them**: on the machine's
/// own MVA rating, which is not the network's.
///
/// Kept as a separate type so the conversion has exactly one place to happen
/// ([`GenCls::new`]) and cannot be skipped by a caller building the struct
/// directly. A machine whose `H` was never converted swings at the wrong
/// frequency and nothing else looks wrong — see `plans/RMS_PLAN.md` §9.5.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenClsParams {
    /// Inertia constant, seconds, on `mbase`.
    pub h: f64,
    /// Damping coefficient, per unit on `mbase`. Often zero.
    pub d: f64,
    /// Armature resistance, per unit on `mbase`. Often zero.
    pub ra: f64,
    /// Transient reactance `x'_d`, per unit on `mbase`.
    pub xdp: f64,
    /// The machine's own MVA rating, in the same units as the network's
    /// `s_base`.
    pub mbase: f64,
}

const STATE_NAMES: [&str; 2] = ["delta", "omega"];

impl GenCls {
    /// Converts `params` from the machine base to the network base and builds
    /// the model.
    ///
    /// With `k = mbase / s_base`: impedances scale as `1/k` and inertia and
    /// damping as `k`. Both directions are easy to invert by accident, so the
    /// arithmetic lives here and only here.
    pub fn new(params: GenClsParams, s_base: f64, f_nom: f64) -> Result<Self, InitError> {
        if params.h <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "h", value: params.h });
        }
        if params.mbase <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "mbase", value: params.mbase });
        }
        if params.xdp <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "xdp", value: params.xdp });
        }
        let k = params.mbase / s_base;
        let z = Complex::new(params.ra / k, params.xdp / k);
        Ok(Self {
            h: params.h * k,
            d: params.d * k,
            y: Complex::new(1.0, 0.0) / z,
            z,
            omega_base: std::f64::consts::TAU * f_nom,
            e_mag: 0.0,
            p_m: 0.0,
        })
    }

    /// The latched internal EMF magnitude. Exposed for the analytic gate,
    /// which needs `P_max = E·V/X` to predict the critical clearing angle.
    pub fn e_mag(&self) -> f64 {
        self.e_mag
    }

    /// The latched mechanical power, network base.
    pub fn p_m(&self) -> f64 {
        self.p_m
    }

    /// Inertia constant on the **network** base — what the swing equation
    /// actually divides by, and what the closed-form `t_cc` needs.
    pub fn h(&self) -> f64 {
        self.h
    }

    pub fn omega_base(&self) -> f64 {
        self.omega_base
    }

    /// The internal EMF phasor at rotor angle `delta`.
    fn emf(&self, delta: f64) -> Complex<f64> {
        Complex::from_polar(self.e_mag, delta)
    }

    /// Air-gap power `Re(E·conj(I))` with `I = (E − V)·y`, expanded so the
    /// derivative below reads off the same two quantities.
    ///
    /// Returns `(p_e, w)` where `w = E·conj(V)·conj(y)`; the caller's
    /// derivatives are `∂p_e/∂δ = Im(w)` and `∂p_e/∂V = −E·conj(y)` read
    /// componentwise.
    fn air_gap_power(&self, e: Complex<f64>, v: Complex<f64>) -> (f64, Complex<f64>) {
        let w = e * v.conj() * self.y.conj();
        (e.norm_sqr() * self.y.re - w.re, w)
    }
}

impl DynamicModel for GenCls {
    fn n_states(&self) -> usize {
        2
    }

    fn state_names(&self) -> &[&'static str] {
        &STATE_NAMES
    }

    fn norton_admittance(&self) -> Option<Complex<f64>> {
        Some(self.y)
    }

    fn derivatives(&self, x: &[f64], v: Complex<f64>, out: &mut [f64]) {
        let (delta, omega) = (x[0], x[1]);
        let e = self.emf(delta);
        let (p_e, _) = self.air_gap_power(e, v);
        out[0] = self.omega_base * (omega - 1.0);
        out[1] = (self.p_m - p_e - self.d * (omega - 1.0)) / (2.0 * self.h);
    }

    fn injection(&self, x: &[f64], _v: Complex<f64>) -> Complex<f64> {
        self.emf(x[0]) * self.y
    }

    fn jacobian(&self, x: &[f64], v: Complex<f64>, out: &mut ModelJacobian) {
        let e = self.emf(x[0]);
        let (_, w) = self.air_gap_power(e, v);
        let two_h = 2.0 * self.h;

        // ∂P_e/∂δ = Im(w); ∂P_e/∂v_re = −Re(E·conj(y)); ∂P_e/∂v_im = −Im(E·conj(y)).
        let dpe_ddelta = w.im;
        let eyc = e * self.y.conj();

        // df/dx, row-major 2×2.
        out.dfdx[0] = 0.0;
        out.dfdx[1] = self.omega_base;
        out.dfdx[2] = -dpe_ddelta / two_h;
        out.dfdx[3] = -self.d / two_h;

        // df/dV, row-major 2×2. δ̇ does not see the network at all.
        out.dfdv[0] = 0.0;
        out.dfdv[1] = 0.0;
        out.dfdv[2] = eyc.re / two_h;
        out.dfdv[3] = eyc.im / two_h;

        // dI/dx: I = E·y and dE/dδ = jE, so dI/dδ = jEy. Independent of ω.
        let jey = Complex::new(0.0, 1.0) * e * self.y;
        out.didx[0] = jey.re;
        out.didx[1] = 0.0;
        out.didx[2] = jey.im;
        out.didx[3] = 0.0;

        // dI/dV is zero: the V·y half of the source lives in the Y-bus.
        out.didv = [0.0; 4];
    }

    fn initialize(&mut self, v: Complex<f64>, s: Complex<f64>) -> Result<Vec<f64>, InitError> {
        if v.norm() == 0.0 {
            return Err(InitError::ZeroTerminalVoltage);
        }
        // S = V·conj(I) defines the current the machine pushes into the bus.
        let i = (s / v).conj();
        let e = v + self.z * i;
        self.e_mag = e.norm();
        // At equilibrium the rotor is at synchronous speed, so the damping
        // term vanishes and P_m is exactly the air-gap power. Latching it from
        // the *solved* operating point rather than reading it from the file is
        // what makes gate G1 hold to machine precision: any mismatch between
        // the two would show up as a rotor that starts moving at t = 0.
        self.p_m = (e * i.conj()).re;
        Ok(vec![e.arg(), 1.0])
    }
}
