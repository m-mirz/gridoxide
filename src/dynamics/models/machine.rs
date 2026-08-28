//! Synchronous machine models.
//!
//! # The interface, and why it is not [`DynamicModel`]
//!
//! A machine is never alone on a bus in a realistic case: an exciter sets its
//! field voltage and a governor its mechanical power. [`Machine`] therefore
//! takes `E_fd` and `P_m` as *inputs* and reports how sensitive it is to each,
//! and [`GeneratingUnit`](super::unit::GeneratingUnit) is what turns a machine
//! plus its controls into one [`DynamicModel`]. A machine with no controls goes
//! through the same path, with both inputs held at what the operating point
//! implied.
//!
//! # The dq frame
//!
//! Every model here works in its own rotor frame and does the rotation itself;
//! nothing outside this file knows about `d` and `q` axes. The convention is
//! the usual one — the `q` axis leads the `d` axis by 90°, and the rotor angle
//! `δ` is the angle of the `q` axis in the network frame:
//!
//! ```text
//! v_d = v_re·sin δ − v_im·cos δ        i_re = i_d·sin δ + i_q·cos δ
//! v_q = v_re·cos δ + v_im·sin δ        i_im = −i_d·cos δ + i_q·sin δ
//! ```
//!
//! The check that fixes the convention: a classical machine's internal EMF is
//! `E'∠δ`, which these send to `(e_d, e_q) = (0, E')` — purely `q`-axis, which
//! is what "constant voltage behind transient reactance" means.
//!
//! # The Norton stamp, and saliency
//!
//! Each model declares a constant [`norton_admittance`](Machine::norton_admittance)
//! that is stamped into `Y` once at build, and
//! [`injection`](Machine::injection) returns
//!
//! ```text
//! I_inj = I_machine(x, V) + y_norton·V
//! ```
//!
//! so the `y_norton·V` the network row adds is exactly cancelled and the
//! machine's true current is what enters the balance. The stamp is therefore
//! *purely* a conditioning device — it keeps `Y` diagonally dominant at
//! generator buses without changing a single answer.
//!
//! For a non-salient machine the two cancel completely and `∂I/∂V` is zero.
//! For a salient one (`x'_q ≠ x'_d`) the machine is not an impedance in the
//! network frame at all — its response to a voltage depends on the rotor's
//! orientation — so a residual `∂I/∂V` remains. That residual *is* the
//! saliency, it is handled in the analytic Jacobian, and it is why the stamp is
//! chosen as the average of the two axes: that makes the leftover as small as
//! it can be.

use num_complex::Complex;

use super::{DynamicModel, InitError};

/// The four blocks a machine contributes, plus its two input sensitivities.
#[derive(Clone, Debug)]
pub struct MachineJacobian {
    /// `∂f/∂x`, `n × n` row-major.
    pub dfdx: Vec<f64>,
    /// `∂f/∂V`, `n × 2` row-major, columns `(v_re, v_im)`.
    pub dfdv: Vec<f64>,
    /// `∂f/∂E_fd`, length `n`.
    pub dfde: Vec<f64>,
    /// `∂f/∂P_m`, length `n`.
    pub dfdp: Vec<f64>,
    /// `∂I/∂x`, `2 × n` row-major, rows `(i_re, i_im)`.
    pub didx: Vec<f64>,
    /// `∂I/∂V`, `2 × 2` row-major.
    pub didv: [f64; 4],
}

impl MachineJacobian {
    pub fn zeros(n: usize) -> Self {
        Self {
            dfdx: vec![0.0; n * n],
            dfdv: vec![0.0; n * 2],
            dfde: vec![0.0; n],
            dfdp: vec![0.0; n],
            didx: vec![0.0; 2 * n],
            didv: [0.0; 4],
        }
    }

    pub fn clear(&mut self) {
        self.dfdx.fill(0.0);
        self.dfdv.fill(0.0);
        self.dfde.fill(0.0);
        self.dfdp.fill(0.0);
        self.didx.fill(0.0);
        self.didv = [0.0; 4];
    }
}

/// What a machine's initialization produces: its own states, plus the field
/// voltage and mechanical power the operating point turned out to require.
///
/// Those two are what an exciter and a governor are then initialized to
/// *reproduce*, which is how their references get fixed. See
/// [`unit`](super::unit).
#[derive(Clone, Debug, PartialEq)]
pub struct MachineInit {
    pub states: Vec<f64>,
    pub e_fd: f64,
    pub p_m: f64,
}

/// A synchronous machine: rotor dynamics plus whatever flux states the model
/// carries, driven by a field voltage and a mechanical power.
pub trait Machine: std::fmt::Debug {
    fn n_states(&self) -> usize;
    fn state_names(&self) -> &[&'static str];

    /// Which state is the rotor speed, in per unit. A governor and a stabilizer
    /// both read it, and neither should have to know the model's layout.
    fn omega_index(&self) -> usize;

    /// The constant admittance stamped into `Y` at this machine's bus. See the
    /// module doc: it is a conditioning device, cancelled inside
    /// [`injection`](Self::injection).
    fn norton_admittance(&self) -> Complex<f64>;

    fn derivatives(&self, x: &[f64], v: Complex<f64>, e_fd: f64, p_m: f64, out: &mut [f64]);

    /// The machine's own current plus `y_norton·V`.
    fn injection(&self, x: &[f64], v: Complex<f64>) -> Complex<f64>;

    fn jacobian(
        &self,
        x: &[f64],
        v: Complex<f64>,
        e_fd: f64,
        p_m: f64,
        out: &mut MachineJacobian,
    );

    /// Choose states so every derivative is zero at this terminal condition,
    /// and report the `E_fd` and `P_m` that requires.
    fn initialize(&mut self, v: Complex<f64>, s: Complex<f64>) -> Result<MachineInit, InitError>;
}

/// Converts an impedance from the machine's own MVA base to the network's, and
/// an inertia or damping constant the other way.
///
/// With `k = mbase / s_base`, impedances scale as `1/k` and inertia as `k`.
/// Both directions are easy to invert by accident, so every model routes
/// through here. A machine whose `H` was never converted swings at the wrong
/// frequency and nothing else looks wrong.
fn base_ratio(mbase: f64, s_base: f64) -> Result<f64, InitError> {
    if mbase <= 0.0 {
        return Err(InitError::NonPositiveParameter { name: "mbase", value: mbase });
    }
    Ok(mbase / s_base)
}

fn require_positive(name: &'static str, value: f64) -> Result<f64, InitError> {
    if value <= 0.0 {
        return Err(InitError::NonPositiveParameter { name, value });
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// Classical
// ---------------------------------------------------------------------------

/// The classical (second-order) machine: a constant voltage magnitude behind
/// transient reactance, with the swing equation for the rotor.
///
/// States: `δ` (rad) and `ω` (per unit, so `1.0` is synchronous).
///
/// \\[ \dot{\delta} = \Omega_b(\omega - 1), \qquad
///    \dot{\omega} = \frac{P_m - P_e - D(\omega - 1)}{2H} \\]
///
/// **`P_e` is the air-gap power, not the terminal power.** With armature
/// resistance the two differ by the stator copper loss, and it is the air-gap
/// power the rotor feels: `Re(E·conj(I)) = Re(V·conj(I)) + |I|²·r_a`. Using the
/// terminal power instead gives a machine damped by its own resistance — wrong
/// in a way that looks entirely plausible, since the swing still decays.
///
/// **`E_fd` has no effect here.** A classical machine's flux is constant by
/// assumption, which is the assumption. An exciter attached to one initializes
/// consistently and then does nothing; that is honest rather than useful, and
/// [`GenTransient`] is the first model with a field winding to excite.
#[derive(Clone, Debug)]
pub struct GenCls {
    h: f64,
    d: f64,
    y: Complex<f64>,
    z: Complex<f64>,
    omega_base: f64,
    /// Constant internal EMF magnitude, latched by [`Machine::initialize`].
    e_mag: f64,
}

/// [`GenCls`]'s parameters as a data file states them: on the machine's own MVA
/// rating, which is not the network's.
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

const GENCLS_STATES: [&str; 2] = ["delta", "omega"];

impl GenCls {
    pub fn new(params: GenClsParams, s_base: f64, f_nom: f64) -> Result<Self, InitError> {
        require_positive("h", params.h)?;
        require_positive("xdp", params.xdp)?;
        let k = base_ratio(params.mbase, s_base)?;
        let z = Complex::new(params.ra / k, params.xdp / k);
        Ok(Self {
            h: params.h * k,
            d: params.d * k,
            y: Complex::new(1.0, 0.0) / z,
            z,
            omega_base: std::f64::consts::TAU * f_nom,
            e_mag: 0.0,
        })
    }

    /// The latched internal EMF magnitude, network base. Exposed because the
    /// analytic transient-stability gates need `P_max = E·V/X`.
    pub fn e_mag(&self) -> f64 {
        self.e_mag
    }

    /// Inertia constant on the **network** base — what the swing equation
    /// divides by, and what a closed-form `t_cc` needs.
    pub fn h(&self) -> f64 {
        self.h
    }

    pub fn omega_base(&self) -> f64 {
        self.omega_base
    }

    fn emf(&self, delta: f64) -> Complex<f64> {
        Complex::from_polar(self.e_mag, delta)
    }

    /// Air-gap power `Re(E·conj(I))` with `I = (E − V)·y`, expanded so the
    /// derivatives read off the same two quantities: `∂P_e/∂δ = Im(w)` and
    /// `∂P_e/∂V = −E·conj(y)` componentwise.
    fn air_gap_power(&self, e: Complex<f64>, v: Complex<f64>) -> (f64, Complex<f64>) {
        let w = e * v.conj() * self.y.conj();
        (e.norm_sqr() * self.y.re - w.re, w)
    }
}

impl Machine for GenCls {
    fn n_states(&self) -> usize {
        2
    }

    fn state_names(&self) -> &[&'static str] {
        &GENCLS_STATES
    }

    fn omega_index(&self) -> usize {
        1
    }

    fn norton_admittance(&self) -> Complex<f64> {
        self.y
    }

    fn derivatives(&self, x: &[f64], v: Complex<f64>, _e_fd: f64, p_m: f64, out: &mut [f64]) {
        let (delta, omega) = (x[0], x[1]);
        let (p_e, _) = self.air_gap_power(self.emf(delta), v);
        out[0] = self.omega_base * (omega - 1.0);
        out[1] = (p_m - p_e - self.d * (omega - 1.0)) / (2.0 * self.h);
    }

    fn injection(&self, x: &[f64], _v: Complex<f64>) -> Complex<f64> {
        // I_machine + y·V = (E − V)·y + y·V = E·y, so the cancellation is
        // exact and nothing here depends on V at all.
        self.emf(x[0]) * self.y
    }

    fn jacobian(
        &self,
        x: &[f64],
        v: Complex<f64>,
        _e_fd: f64,
        _p_m: f64,
        out: &mut MachineJacobian,
    ) {
        let e = self.emf(x[0]);
        let (_, w) = self.air_gap_power(e, v);
        let two_h = 2.0 * self.h;
        let eyc = e * self.y.conj();

        out.dfdx[0] = 0.0;
        out.dfdx[1] = self.omega_base;
        out.dfdx[2] = -w.im / two_h;
        out.dfdx[3] = -self.d / two_h;

        out.dfdv[0] = 0.0;
        out.dfdv[1] = 0.0;
        out.dfdv[2] = eyc.re / two_h;
        out.dfdv[3] = eyc.im / two_h;

        out.dfde[0] = 0.0;
        out.dfde[1] = 0.0;
        out.dfdp[0] = 0.0;
        out.dfdp[1] = 1.0 / two_h;

        let jey = Complex::new(0.0, 1.0) * e * self.y;
        out.didx[0] = jey.re;
        out.didx[1] = 0.0;
        out.didx[2] = jey.im;
        out.didx[3] = 0.0;

        out.didv = [0.0; 4];
    }

    fn initialize(&mut self, v: Complex<f64>, s: Complex<f64>) -> Result<MachineInit, InitError> {
        if v.norm() == 0.0 {
            return Err(InitError::ZeroTerminalVoltage);
        }
        let i = (s / v).conj();
        let e = v + self.z * i;
        self.e_mag = e.norm();
        // At equilibrium the rotor is synchronous, so the damping term vanishes
        // and P_m is exactly the air-gap power. Latched from the *solved*
        // operating point rather than read from a file, which is what makes the
        // equilibrium gate hold to machine precision.
        Ok(MachineInit {
            states: vec![e.arg(), 1.0],
            e_fd: self.e_mag,
            p_m: (e * i.conj()).re,
        })
    }
}

// ---------------------------------------------------------------------------
// Fourth-order transient
// ---------------------------------------------------------------------------

/// The fourth-order ("two-axis") machine: a field winding on the `d` axis and
/// one damper winding on the `q` axis, both with transient time constants.
///
/// States: `δ`, `ω`, `e'_q`, `e'_d`.
///
/// ```text
/// δ̇      = Ω_b(ω − 1)
/// ω̇      = [P_m − P_e − D(ω − 1)] / 2H
/// T'_d0 ė'_q = −e'_q − (x_d − x'_d)·i_d + E_fd
/// T'_q0 ė'_d = −e'_d + (x_q − x'_q)·i_q
/// ```
///
/// with the stator relations, from which the currents follow by a 2×2 solve:
///
/// ```text
/// v_d = e'_d − r_a·i_d + x'_q·i_q          det = r_a² + x'_d·x'_q
/// v_q = e'_q − r_a·i_q − x'_d·i_d
/// ```
///
/// and the air-gap power **including the saliency term**:
///
/// ```text
/// P_e = e'_d·i_d + e'_q·i_q + (x'_q − x'_d)·i_d·i_q
/// ```
///
/// That last term is the reluctance torque, and dropping it is a common and
/// invisible mistake: the machine still swings, just at a slightly wrong
/// frequency and to a slightly wrong equilibrium.
///
/// This is the first model with a field winding, so it is the first one an
/// exciter can actually act on — `E_fd` enters `ė'_q` directly.
#[derive(Clone, Debug)]
pub struct GenTransient {
    h: f64,
    d: f64,
    ra: f64,
    xd: f64,
    xdp: f64,
    xq: f64,
    xqp: f64,
    td0p: f64,
    tq0p: f64,
    omega_base: f64,
    y: Complex<f64>,
}

/// [`GenTransient`]'s parameters, on the machine's own MVA rating.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenTransientParams {
    pub h: f64,
    pub d: f64,
    pub ra: f64,
    /// Synchronous reactances.
    pub xd: f64,
    pub xq: f64,
    /// Transient reactances. Must be strictly below the synchronous ones — a
    /// machine whose transient reactance exceeds its synchronous one has a
    /// negative field time constant in disguise.
    pub xdp: f64,
    pub xqp: f64,
    /// Open-circuit transient time constants, seconds.
    pub td0p: f64,
    pub tq0p: f64,
    pub mbase: f64,
}

const GENTRANSIENT_STATES: [&str; 4] = ["delta", "omega", "eqp", "edp"];

impl GenTransient {
    pub fn new(params: GenTransientParams, s_base: f64, f_nom: f64) -> Result<Self, InitError> {
        require_positive("h", params.h)?;
        require_positive("xdp", params.xdp)?;
        require_positive("xqp", params.xqp)?;
        require_positive("td0p", params.td0p)?;
        require_positive("tq0p", params.tq0p)?;
        if params.xd <= params.xdp {
            return Err(InitError::NonPositiveParameter {
                name: "xd - xdp",
                value: params.xd - params.xdp,
            });
        }
        if params.xq < params.xqp {
            return Err(InitError::NonPositiveParameter {
                name: "xq - xqp",
                value: params.xq - params.xqp,
            });
        }
        let k = base_ratio(params.mbase, s_base)?;
        let (ra, xdp, xqp) = (params.ra / k, params.xdp / k, params.xqp / k);
        // The stamp is the average of the two transient axes, which is what
        // makes the residual ∂I/∂V — the saliency — as small as it can be.
        let z_norton = Complex::new(ra, 0.5 * (xdp + xqp));
        Ok(Self {
            h: params.h * k,
            d: params.d * k,
            ra,
            xd: params.xd / k,
            xdp,
            xq: params.xq / k,
            xqp,
            td0p: params.td0p,
            tq0p: params.tq0p,
            omega_base: std::f64::consts::TAU * f_nom,
            y: Complex::new(1.0, 0.0) / z_norton,
        })
    }

    pub fn h(&self) -> f64 {
        self.h
    }

    /// Network-frame voltage into rotor coordinates.
    fn to_dq(delta: f64, v: Complex<f64>) -> (f64, f64) {
        let (sd, cd) = delta.sin_cos();
        (v.re * sd - v.im * cd, v.re * cd + v.im * sd)
    }

    /// Rotor-frame current back into the network frame.
    fn from_dq(delta: f64, i_d: f64, i_q: f64) -> Complex<f64> {
        let (sd, cd) = delta.sin_cos();
        Complex::new(i_d * sd + i_q * cd, -i_d * cd + i_q * sd)
    }

    fn det(&self) -> f64 {
        self.ra * self.ra + self.xdp * self.xqp
    }

    /// Armature currents from the stator equations, given the flux states and
    /// the terminal voltage in rotor coordinates.
    fn currents(&self, eqp: f64, edp: f64, v_d: f64, v_q: f64) -> (f64, f64) {
        let (a, b) = (edp - v_d, eqp - v_q);
        let det = self.det();
        ((self.ra * a + self.xqp * b) / det, (-self.xdp * a + self.ra * b) / det)
    }

    fn air_gap_power(&self, eqp: f64, edp: f64, i_d: f64, i_q: f64) -> f64 {
        edp * i_d + eqp * i_q + (self.xqp - self.xdp) * i_d * i_q
    }
}

impl Machine for GenTransient {
    fn n_states(&self) -> usize {
        4
    }

    fn state_names(&self) -> &[&'static str] {
        &GENTRANSIENT_STATES
    }

    fn omega_index(&self) -> usize {
        1
    }

    fn norton_admittance(&self) -> Complex<f64> {
        self.y
    }

    fn derivatives(&self, x: &[f64], v: Complex<f64>, e_fd: f64, p_m: f64, out: &mut [f64]) {
        let (delta, omega, eqp, edp) = (x[0], x[1], x[2], x[3]);
        let (v_d, v_q) = Self::to_dq(delta, v);
        let (i_d, i_q) = self.currents(eqp, edp, v_d, v_q);
        let p_e = self.air_gap_power(eqp, edp, i_d, i_q);

        out[0] = self.omega_base * (omega - 1.0);
        out[1] = (p_m - p_e - self.d * (omega - 1.0)) / (2.0 * self.h);
        out[2] = (-eqp - (self.xd - self.xdp) * i_d + e_fd) / self.td0p;
        out[3] = (-edp + (self.xq - self.xqp) * i_q) / self.tq0p;
    }

    fn injection(&self, x: &[f64], v: Complex<f64>) -> Complex<f64> {
        let (delta, eqp, edp) = (x[0], x[2], x[3]);
        let (v_d, v_q) = Self::to_dq(delta, v);
        let (i_d, i_q) = self.currents(eqp, edp, v_d, v_q);
        Self::from_dq(delta, i_d, i_q) + self.y * v
    }

    fn jacobian(
        &self,
        x: &[f64],
        v: Complex<f64>,
        _e_fd: f64,
        _p_m: f64,
        out: &mut MachineJacobian,
    ) {
        let (delta, eqp, edp) = (x[0], x[2], x[3]);
        let (sd, cd) = delta.sin_cos();
        let (v_d, v_q) = Self::to_dq(delta, v);
        let (i_d, i_q) = self.currents(eqp, edp, v_d, v_q);
        let det = self.det();
        let two_h = 2.0 * self.h;

        // `a = e'_d − v_d`, `b = e'_q − v_q` are the only things the currents
        // depend on, so every derivative below is (∂a/∂z, ∂b/∂z) pushed through
        // one fixed 2×2 inverse. Columns: δ, ω, e'_q, e'_d, v_re, v_im.
        //
        //   ∂v_d/∂δ = v_q,  ∂v_q/∂δ = −v_d
        //   ∂v_d/∂(v_re, v_im) = (sin δ, −cos δ)
        //   ∂v_q/∂(v_re, v_im) = (cos δ,  sin δ)
        let dab: [(f64, f64); 6] = [
            (-v_q, v_d),   // δ
            (0.0, 0.0),    // ω
            (0.0, 1.0),    // e'_q
            (1.0, 0.0),    // e'_d
            (-sd, -cd),    // v_re
            (cd, -sd),     // v_im
        ];
        let mut di_d = [0.0; 6];
        let mut di_q = [0.0; 6];
        for (k, (da, db)) in dab.iter().enumerate() {
            di_d[k] = (self.ra * da + self.xqp * db) / det;
            di_q[k] = (-self.xdp * da + self.ra * db) / det;
        }

        // ∂P_e/∂z, with the explicit e'_d and e'_q terms where they apply.
        let sal = self.xqp - self.xdp;
        let mut dpe = [0.0; 6];
        for (k, dpe_k) in dpe.iter_mut().enumerate() {
            *dpe_k = edp * di_d[k] + eqp * di_q[k] + sal * (i_q * di_d[k] + i_d * di_q[k]);
        }
        dpe[2] += i_q; // ∂P_e/∂e'_q picks up i_q directly
        dpe[3] += i_d; // and ∂P_e/∂e'_d picks up i_d

        // `dfdx` is row-major n×n: entry (row, col) lives at `row * n + col`.
        let n = 4;
        // δ̇ = Ω_b(ω − 1) sees nothing but ω.
        out.dfdx[1] = self.omega_base;
        // ω̇
        for k in 0..n {
            out.dfdx[n + k] = -dpe[k] / two_h;
        }
        out.dfdx[n + 1] -= self.d / two_h;
        // ė'_q and ė'_d
        for k in 0..n {
            out.dfdx[2 * n + k] = -(self.xd - self.xdp) * di_d[k] / self.td0p;
            out.dfdx[3 * n + k] = (self.xq - self.xqp) * di_q[k] / self.tq0p;
        }
        out.dfdx[2 * n + 2] -= 1.0 / self.td0p;
        out.dfdx[3 * n + 3] -= 1.0 / self.tq0p;

        for (col, k) in [(0usize, 4usize), (1, 5)] {
            out.dfdv[2 + col] = -dpe[k] / two_h;
            out.dfdv[4 + col] = -(self.xd - self.xdp) * di_d[k] / self.td0p;
            out.dfdv[6 + col] = (self.xq - self.xqp) * di_q[k] / self.tq0p;
        }

        out.dfde[2] = 1.0 / self.td0p;
        out.dfdp[1] = 1.0 / two_h;

        // I_machine = (i_d·sin δ + i_q·cos δ) + j(−i_d·cos δ + i_q·sin δ), so
        // δ moves it both through the currents and through the rotation.
        for k in 0..4 {
            let (mut d_re, mut d_im) =
                (di_d[k] * sd + di_q[k] * cd, -di_d[k] * cd + di_q[k] * sd);
            if k == 0 {
                d_re += i_d * cd - i_q * sd;
                d_im += i_d * sd + i_q * cd;
            }
            out.didx[k] = d_re;
            out.didx[n + k] = d_im;
        }

        // ∂I_inj/∂V is the machine's own plus the Norton stamp's, which is what
        // leaves zero for a non-salient machine and the saliency for a salient
        // one.
        let (g, b) = (self.y.re, self.y.im);
        for (col, k) in [(0usize, 4usize), (1, 5)] {
            out.didv[col] = di_d[k] * sd + di_q[k] * cd;
            out.didv[2 + col] = -di_d[k] * cd + di_q[k] * sd;
        }
        out.didv[0] += g;
        out.didv[1] += -b;
        out.didv[2] += b;
        out.didv[3] += g;
    }

    fn initialize(&mut self, v: Complex<f64>, s: Complex<f64>) -> Result<MachineInit, InitError> {
        if v.norm() == 0.0 {
            return Err(InitError::ZeroTerminalVoltage);
        }
        let i = (s / v).conj();

        // The rotor angle comes from the *steady-state* q-axis reference
        // E_q = V + (r_a + j·x_q)·I. That is not a convenience: choosing δ this
        // way is exactly what makes ė'_d vanish, since it forces the d-axis
        // component of E_q to zero, which reduces the stator relation for e'_d
        // to `(x_q − x'_q)·i_q` — the very expression ė'_d = 0 demands.
        let e_q = v + Complex::new(self.ra, self.xq) * i;
        let delta = e_q.arg();

        let (v_d, v_q) = Self::to_dq(delta, v);
        let (i_d, i_q) = Self::to_dq(delta, i);

        let eqp = v_q + self.ra * i_q + self.xdp * i_d;
        let edp = v_d + self.ra * i_d - self.xqp * i_q;
        let e_fd = eqp + (self.xd - self.xdp) * i_d;
        let p_m = self.air_gap_power(eqp, edp, i_d, i_q);

        Ok(MachineInit { states: vec![delta, 1.0, eqp, edp], e_fd, p_m })
    }
}

/// A bare machine as a [`DynamicModel`], for the common case of no controls.
///
/// Exactly [`GeneratingUnit::machine_only`](super::unit::GeneratingUnit::machine_only);
/// this is the shorter spelling.
pub fn bare(machine: Box<dyn Machine>) -> Box<dyn DynamicModel> {
    Box::new(super::unit::GeneratingUnit::machine_only(machine))
}
