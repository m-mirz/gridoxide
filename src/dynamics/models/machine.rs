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
//! # The `ω ≈ 1` approximation, and what it costs
//!
//! By default every stator equation here omits the rotor speed. The full form
//! carries it on the speed-voltage terms, and writes the swing equation in
//! **torque** rather than power:
//!
//! ```text
//! full:          v_d = −r_a·i_d − ω·λ_q      2H·ω̇ = c_m − c_e − D·Δω
//! here:          v_d = −r_a·i_d − λ_q        2H·ω̇ = P_m − P_e − D·Δω
//! ```
//!
//! The two differ by exactly a factor of `ω`, so they agree at synchronous
//! speed and diverge in proportion to the speed deviation. Neglecting the
//! speed voltages is the classical RMS assumption — Kundur §13.3 states it
//! explicitly — and it is what makes the phasor-domain formulation coherent in
//! the first place; keeping them, as Sauer & Pai and Dynawo do, is the other
//! defensible choice.
//!
//! It is not free, and its size is known rather than guessed:
//! `tests/dynamics_reference_test.rs` measures it against Dynawo on Kundur's
//! Example 13.2 at **0.6% of terminal power for a 0.9% speed deviation**.
//!
//! **Both forms are available.** `with_speed_voltages(true)` on any machine
//! here — or `"speed_voltages": true` in a document's `dynamics` section —
//! switches to the full one. It is off by default because the approximation is
//! what makes the phasor formulation coherent in the first place, and because
//! every closed-form gate in this crate (the equal-area criterion above all) is
//! derived from the power form. Turning it on drives the measured 0.6% offset
//! against Dynawo to under 0.05%, which is what closes the loop on the
//! attribution rather than leaving it an assertion.
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
use serde::{Deserialize, Serialize};

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
    /// Whether to carry the rotor speed on the speed-voltage terms and write
    /// the swing equation in torque. See the module doc.
    speed_voltages: bool,
    /// The armature resistance and transient reactance separately, since the
    /// full form needs them apart rather than as one impedance.
    ra: f64,
    xdp: f64,
}

/// [`GenCls`]'s parameters as a data file states them: on the machine's own MVA
/// rating, which is not the network's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
        let (ra, xdp) = (params.ra / k, params.xdp / k);
        let z = Complex::new(ra, xdp);
        Ok(Self {
            h: params.h * k,
            d: params.d * k,
            y: Complex::new(1.0, 0.0) / z,
            z,
            omega_base: std::f64::consts::TAU * f_nom,
            e_mag: 0.0,
            speed_voltages: false,
            ra,
            xdp,
        })
    }

    /// Carry the rotor speed on the speed-voltage terms, and write the swing
    /// equation in torque. See the module doc for what this costs and buys.
    pub fn with_speed_voltages(mut self, on: bool) -> Self {
        self.speed_voltages = on;
        self
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

    /// The speed factor on the flux-derived terms: the rotor speed in the full
    /// form, one in the approximate one. `dw` is its derivative with respect to
    /// `ω`, which is what makes the two forms one code path.
    fn speed(&self, omega: f64) -> (f64, f64) {
        if self.speed_voltages {
            (omega, 1.0)
        } else {
            (1.0, 0.0)
        }
    }

    /// The stator solve, and its sensitivity to the rotor speed.
    ///
    /// `I = (w·E − V)/(r_a + j·w·x'_d)`. With `w = 1` this is the ordinary
    /// Norton current; with `w = ω` it is the full form, in which the machine's
    /// admittance itself moves with speed — which is why the constant stamp can
    /// no longer cancel exactly and `∂I/∂V` stops being zero.
    fn stator(&self, e: Complex<f64>, v: Complex<f64>, omega: f64) -> Stator {
        let (w, dw) = self.speed(omega);
        let y = Complex::new(1.0, 0.0) / Complex::new(self.ra, w * self.xdp);
        let current = (e * w - v) * y;
        // dy/dω = −j·x'_d·y²·dw, by differentiating 1/(r_a + j·w·x'_d).
        let dy = -Complex::new(0.0, 1.0) * self.xdp * y * y * dw;
        Stator { y, current, di_domega: e * dw * y + (e * w - v) * dy, w }
    }

    /// The air-gap **torque**, `Re(E·conj(I))` with `E` the unscaled internal
    /// EMF. At `w = 1` this is the air-gap power, which is why the two forms
    /// coincide at synchronous speed.
    fn torque(e: Complex<f64>, current: Complex<f64>) -> f64 {
        (e * current.conj()).re
    }

    /// The mechanical term the swing equation subtracts the torque from:
    /// `P_m` in the approximate form, `P_m/ω` in the full one.
    fn mechanical(&self, p_m: f64, omega: f64) -> (f64, f64) {
        if self.speed_voltages {
            (p_m / omega, -p_m / (omega * omega))
        } else {
            (p_m, 0.0)
        }
    }
}

/// What one stator solve produces, plus what it needs for the `ω` column.
struct Stator {
    y: Complex<f64>,
    current: Complex<f64>,
    di_domega: Complex<f64>,
    w: f64,
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
        let e = self.emf(delta);
        let c_e = Self::torque(e, self.stator(e, v, omega).current);
        let (mech, _) = self.mechanical(p_m, omega);
        out[0] = self.omega_base * (omega - 1.0);
        out[1] = (mech - c_e - self.d * (omega - 1.0)) / (2.0 * self.h);
    }

    fn injection(&self, x: &[f64], v: Complex<f64>) -> Complex<f64> {
        // In the approximate form `I_machine + y·V = (E − V)·y + y·V = E·y`,
        // so the constant stamp cancels exactly and nothing depends on V. In
        // the full form the machine's own admittance moves with speed, so a
        // residual remains — which is the speed voltage, made visible.
        self.stator(self.emf(x[0]), v, x[1]).current + self.y * v
    }

    fn jacobian(
        &self,
        x: &[f64],
        v: Complex<f64>,
        _e_fd: f64,
        p_m: f64,
        out: &mut MachineJacobian,
    ) {
        let (delta, omega) = (x[0], x[1]);
        let e = self.emf(delta);
        let stator = self.stator(e, v, omega);
        let two_h = 2.0 * self.h;
        let (_, dmech) = self.mechanical(p_m, omega);

        // Everything follows from ∂I, chained through `c_e = Re(E·conj(I))`.
        // Writing it this way rather than expanding the torque means the two
        // forms differ in exactly one place — the stator solve — instead of in
        // every derivative.
        let i = stator.current;
        let di_ddelta = Complex::new(0.0, 1.0) * e * stator.w * stator.y;
        let di_dvre = -stator.y;
        let di_dvim = -Complex::new(0.0, 1.0) * stator.y;
        let de_ddelta = Complex::new(0.0, 1.0) * e;

        let dce_ddelta = (de_ddelta * i.conj()).re + (e * di_ddelta.conj()).re;
        let dce_domega = (e * stator.di_domega.conj()).re;
        let dce_dvre = (e * di_dvre.conj()).re;
        let dce_dvim = (e * di_dvim.conj()).re;

        out.dfdx[0] = 0.0;
        out.dfdx[1] = self.omega_base;
        out.dfdx[2] = -dce_ddelta / two_h;
        out.dfdx[3] = (dmech - dce_domega - self.d) / two_h;

        out.dfdv[0] = 0.0;
        out.dfdv[1] = 0.0;
        out.dfdv[2] = -dce_dvre / two_h;
        out.dfdv[3] = -dce_dvim / two_h;

        out.dfde[0] = 0.0;
        out.dfde[1] = 0.0;
        out.dfdp[0] = 0.0;
        out.dfdp[1] = if self.speed_voltages { 1.0 / (omega * two_h) } else { 1.0 / two_h };

        out.didx[0] = di_ddelta.re;
        out.didx[1] = stator.di_domega.re;
        out.didx[2] = di_ddelta.im;
        out.didx[3] = stator.di_domega.im;

        // ∂I_inj/∂V = ∂I_machine/∂V + y_const, zero in the approximate form and
        // the speed voltage itself in the full one.
        let d_re = di_dvre + self.y;
        let d_im = di_dvim + Complex::new(0.0, 1.0) * self.y;
        out.didv = [d_re.re, d_im.re, d_re.im, d_im.im];
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
    speed_voltages: bool,
}

/// [`GenTransient`]'s parameters, on the machine's own MVA rating.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
            speed_voltages: false,
        })
    }

    /// Carry the rotor speed on the speed-voltage terms, and write the swing
    /// equation in torque. See the module doc.
    pub fn with_speed_voltages(mut self, on: bool) -> Self {
        self.speed_voltages = on;
        self
    }

    pub fn h(&self) -> f64 {
        self.h
    }

    /// The speed factor on the flux-derived terms, and its derivative.
    fn speed(&self, omega: f64) -> (f64, f64) {
        if self.speed_voltages { (omega, 1.0) } else { (1.0, 0.0) }
    }

    fn mechanical(&self, p_m: f64, omega: f64) -> (f64, f64) {
        if self.speed_voltages {
            (p_m / omega, -p_m / (omega * omega))
        } else {
            (p_m, 0.0)
        }
    }

    /// Network-frame voltage into rotor coordinates.
    pub(crate) fn to_dq(delta: f64, v: Complex<f64>) -> (f64, f64) {
        let (sd, cd) = delta.sin_cos();
        (v.re * sd - v.im * cd, v.re * cd + v.im * sd)
    }

    /// Rotor-frame current back into the network frame.
    pub(crate) fn from_dq(delta: f64, i_d: f64, i_q: f64) -> Complex<f64> {
        let (sd, cd) = delta.sin_cos();
        Complex::new(i_d * sd + i_q * cd, -i_d * cd + i_q * sd)
    }

    /// `det` of the 2×2 stator solve. The speed factor enters squared, which is
    /// what makes the full form's `ω` column more than a scaling.
    fn det(&self, w: f64) -> f64 {
        self.ra * self.ra + w * w * self.xdp * self.xqp
    }

    /// Armature currents from the stator equations, given the flux states, the
    /// terminal voltage in rotor coordinates, and the speed factor.
    fn currents(&self, eqp: f64, edp: f64, v_d: f64, v_q: f64, w: f64) -> (f64, f64, f64, f64) {
        let (a, b) = (w * edp - v_d, w * eqp - v_q);
        let det = self.det(w);
        (
            (self.ra * a + w * self.xqp * b) / det,
            (-w * self.xdp * a + self.ra * b) / det,
            a,
            b,
        )
    }

    /// Air-gap **torque**. The expression does not change between the two
    /// forms; only the currents it is evaluated at do.
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
        let (w, _) = self.speed(omega);
        let (v_d, v_q) = Self::to_dq(delta, v);
        let (i_d, i_q, _, _) = self.currents(eqp, edp, v_d, v_q, w);
        let p_e = self.air_gap_power(eqp, edp, i_d, i_q);
        let (mech, _) = self.mechanical(p_m, omega);

        out[0] = self.omega_base * (omega - 1.0);
        out[1] = (mech - p_e - self.d * (omega - 1.0)) / (2.0 * self.h);
        out[2] = (-eqp - (self.xd - self.xdp) * i_d + e_fd) / self.td0p;
        out[3] = (-edp + (self.xq - self.xqp) * i_q) / self.tq0p;
    }

    fn injection(&self, x: &[f64], v: Complex<f64>) -> Complex<f64> {
        let (delta, omega, eqp, edp) = (x[0], x[1], x[2], x[3]);
        let (w, _) = self.speed(omega);
        let (v_d, v_q) = Self::to_dq(delta, v);
        let (i_d, i_q, _, _) = self.currents(eqp, edp, v_d, v_q, w);
        Self::from_dq(delta, i_d, i_q) + self.y * v
    }

    fn jacobian(
        &self,
        x: &[f64],
        v: Complex<f64>,
        _e_fd: f64,
        p_m: f64,
        out: &mut MachineJacobian,
    ) {
        let (delta, omega, eqp, edp) = (x[0], x[1], x[2], x[3]);
        let (w, dw) = self.speed(omega);
        let (_, dmech) = self.mechanical(p_m, omega);
        let (sd, cd) = delta.sin_cos();
        let (v_d, v_q) = Self::to_dq(delta, v);
        let (i_d, i_q, a, b) = self.currents(eqp, edp, v_d, v_q, w);
        let det = self.det(w);
        let two_h = 2.0 * self.h;

        // `a = e'_d − v_d`, `b = e'_q − v_q` are the only things the currents
        // depend on, so every derivative below is (∂a/∂z, ∂b/∂z) pushed through
        // one fixed 2×2 inverse. Columns: δ, ω, e'_q, e'_d, v_re, v_im.
        //
        //   ∂v_d/∂δ = v_q,  ∂v_q/∂δ = −v_d
        //   ∂v_d/∂(v_re, v_im) = (sin δ, −cos δ)
        //   ∂v_q/∂(v_re, v_im) = (cos δ,  sin δ)
        let dab: [(f64, f64); 6] = [
            (-v_q, v_d),       // δ
            (0.0, 0.0),        // ω — the speed factor is not linear here, see below
            (0.0, w),          // e'_q
            (w, 0.0),          // e'_d
            (-sd, -cd),        // v_re
            (cd, -sd),         // v_im
        ];
        let mut di_d = [0.0; 6];
        let mut di_q = [0.0; 6];
        for (k, (da, db)) in dab.iter().enumerate() {
            di_d[k] = (self.ra * da + w * self.xqp * db) / det;
            di_q[k] = (-w * self.xdp * da + self.ra * db) / det;
        }
        // The `ω` column cannot go through the same fixed inverse: the speed
        // factor appears in the coefficients *and* in `det`, so it has to be
        // differentiated as a product. `dw` is zero in the approximate form,
        // which leaves this column zero and the two forms identical.
        let ddet = 2.0 * w * dw * self.xdp * self.xqp;
        di_d[1] = (self.ra * dw * edp + dw * self.xqp * b + w * self.xqp * dw * eqp) / det
            - i_d * ddet / det;
        di_q[1] = (-dw * self.xdp * a - w * self.xdp * dw * edp + self.ra * dw * eqp) / det
            - i_q * ddet / det;

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
        out.dfdx[n + 1] += (dmech - self.d) / two_h;
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
        out.dfdp[1] = if self.speed_voltages { 1.0 / (omega * two_h) } else { 1.0 / two_h };

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

// ---------------------------------------------------------------------------
// Sixth-order subtransient
// ---------------------------------------------------------------------------

/// The sixth-order subtransient machine: a field winding and a damper on the
/// `d` axis, two dampers on the `q` axis.
///
/// States: `δ`, `ω`, `e'_q`, `e'_d`, `ψ_1d`, `ψ_2q`.
///
/// # Why the sign conventions here are derived and not cited
///
/// Published statements of this model disagree with each other about the sign
/// of `ψ_2q` and of `e'_d`, and a formulation copied from one source and
/// checked against another produces a machine that is internally consistent
/// and physically wrong. The conventions below are instead **pinned by two
/// requirements**, each of which the code can be checked against:
///
/// 1. **The steady state must reduce to [`GenTransient`]'s.** Setting the flux
///    derivatives to zero gives `ψ_1d = e'_q − (x'_d − x_l)i_d` and
///    `ψ_2q = e'_d + (x'_q − x_l)i_q`; substituting those into the
///    subtransient EMFs must turn the subtransient stator equations into the
///    transient ones, term for term. It does, and only for one choice of
///    signs.
/// 2. **The two axes must map onto each other** under
///    `(e'_q, ψ_1d, i_d) ↔ (e'_d, ψ_2q, −i_q)`. That is what fixes the sign of
///    the damper-coupling correction in `ė'_d`, which requirement 1 cannot see
///    because the correction vanishes at steady state.
///
/// # The equations
///
/// ```text
/// e''_q = a_d·e'_q + b_d·ψ_1d          a_d = (x''_d − x_l)/(x'_d − x_l)
/// e''_d = a_q·e'_d + b_q·ψ_2q          b_d = (x'_d − x''_d)/(x'_d − x_l)
///                                      a_d + b_d = 1, and likewise on q
///
/// v_d = e''_d − r_a·i_d + x''_q·i_q    det = r_a² + x''_d·x''_q
/// v_q = e''_q − r_a·i_q − x''_d·i_d
///
/// Δ_d = ψ_1d − e'_q + (x'_d − x_l)·i_d      (zero at steady state)
/// Δ_q = ψ_2q − e'_d − (x'_q − x_l)·i_q      (zero at steady state)
///
/// T'_d0 ·ė'_q  = −e'_q − (x_d − x'_d)(i_d − K_d·Δ_d) + E_fd
/// T'_q0 ·ė'_d  = −e'_d + (x_q − x'_q)(i_q + K_q·Δ_q)
/// T''_d0·ψ̇_1d = −Δ_d
/// T''_q0·ψ̇_2q = −Δ_q
/// ```
///
/// with `K_d = b_d/(x'_d − x_l)` and `K_q = b_q/(x'_q − x_l)`. That the two
/// damper equations are just `−Δ/T''` is not a rearrangement for tidiness — it
/// is the same `Δ` the correction terms use, and seeing one expression serve
/// both is itself a check that the signs agree.
///
/// # What this buys over the fourth order
///
/// The damper windings. Immediately after a disturbance the machine's
/// effective impedance is `x''`, not `x'` — the dampers hold their flux and
/// oppose the change — and only after their time constants elapse does the
/// response settle onto the transient reactances. A fourth-order machine
/// misses that first, largest current excursion entirely.
#[derive(Clone, Debug)]
pub struct GenRound {
    h: f64,
    d: f64,
    ra: f64,
    xd: f64,
    xdp: f64,
    xdpp: f64,
    xq: f64,
    xqp: f64,
    xqpp: f64,
    xl: f64,
    td0p: f64,
    tq0p: f64,
    td0pp: f64,
    tq0pp: f64,
    ad: f64,
    bd: f64,
    aq: f64,
    bq: f64,
    kd: f64,
    kq: f64,
    omega_base: f64,
    y: Complex<f64>,
    speed_voltages: bool,
}

/// [`GenRound`]'s parameters, on the machine's own MVA rating.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenRoundParams {
    pub h: f64,
    pub d: f64,
    pub ra: f64,
    pub xd: f64,
    pub xq: f64,
    pub xdp: f64,
    pub xqp: f64,
    /// Subtransient reactances. Below the transient ones, and both above the
    /// leakage reactance — the ordering `x_l < x'' < x' < x` is what makes the
    /// interpolation coefficients lie in `[0, 1]`, and a set that violates it
    /// is not a machine.
    pub xdpp: f64,
    pub xqpp: f64,
    /// Stator leakage reactance.
    pub xl: f64,
    pub td0p: f64,
    pub tq0p: f64,
    /// Open-circuit subtransient time constants, seconds. Short — tens of
    /// milliseconds — which is what makes this model stiff and what the
    /// integrator's backward-Euler damping was put in for.
    pub td0pp: f64,
    pub tq0pp: f64,
    pub mbase: f64,
}

const GENROUND_STATES: [&str; 6] = ["delta", "omega", "eqp", "edp", "psi1d", "psi2q"];

impl GenRound {
    pub fn new(params: GenRoundParams, s_base: f64, f_nom: f64) -> Result<Self, InitError> {
        require_positive("h", params.h)?;
        require_positive("td0p", params.td0p)?;
        require_positive("tq0p", params.tq0p)?;
        require_positive("td0pp", params.td0pp)?;
        require_positive("tq0pp", params.tq0pp)?;
        // x_l < x'' < x' < x, on both axes.
        for (name, lo, hi) in [
            ("xdpp - xl", params.xl, params.xdpp),
            ("xdp - xdpp", params.xdpp, params.xdp),
            ("xd - xdp", params.xdp, params.xd),
            ("xqpp - xl", params.xl, params.xqpp),
            ("xqp - xqpp", params.xqpp, params.xqp),
            ("xq - xqp", params.xqp, params.xq),
        ] {
            if hi <= lo {
                return Err(InitError::NonPositiveParameter { name, value: hi - lo });
            }
        }

        let k = base_ratio(params.mbase, s_base)?;
        let (xd, xdp, xdpp) = (params.xd / k, params.xdp / k, params.xdpp / k);
        let (xq, xqp, xqpp) = (params.xq / k, params.xqp / k, params.xqpp / k);
        let (ra, xl) = (params.ra / k, params.xl / k);

        let bd = (xdp - xdpp) / (xdp - xl);
        let bq = (xqp - xqpp) / (xqp - xl);
        // The stamp is the average of the two subtransient axes: it is the
        // impedance the machine actually presents at the instant a disturbance
        // arrives, which is what makes it the right conditioning choice here.
        let z_norton = Complex::new(ra, 0.5 * (xdpp + xqpp));
        Ok(Self {
            h: params.h * k,
            d: params.d * k,
            ra,
            xd,
            xdp,
            xdpp,
            xq,
            xqp,
            xqpp,
            xl,
            td0p: params.td0p,
            tq0p: params.tq0p,
            td0pp: params.td0pp,
            tq0pp: params.tq0pp,
            ad: 1.0 - bd,
            bd,
            aq: 1.0 - bq,
            bq,
            kd: bd / (xdp - xl),
            kq: bq / (xqp - xl),
            omega_base: std::f64::consts::TAU * f_nom,
            y: Complex::new(1.0, 0.0) / z_norton,
            speed_voltages: false,
        })
    }

    /// Carry the rotor speed on the speed-voltage terms, and write the swing
    /// equation in torque. See the module doc.
    pub fn with_speed_voltages(mut self, on: bool) -> Self {
        self.speed_voltages = on;
        self
    }

    pub fn h(&self) -> f64 {
        self.h
    }

    /// The speed factor on the flux-derived terms, and its derivative.
    fn speed(&self, omega: f64) -> (f64, f64) {
        if self.speed_voltages { (omega, 1.0) } else { (1.0, 0.0) }
    }

    fn mechanical(&self, p_m: f64, omega: f64) -> (f64, f64) {
        if self.speed_voltages {
            (p_m / omega, -p_m / (omega * omega))
        } else {
            (p_m, 0.0)
        }
    }

    /// The subtransient impedance the machine presents to a sudden change.
    /// Exposed because that is the property distinguishing this model from
    /// [`GenTransient`], and a gate needs to be able to name it.
    pub fn subtransient_impedance(&self) -> Complex<f64> {
        Complex::new(self.ra, 0.5 * (self.xdpp + self.xqpp))
    }

    /// The subtransient EMFs, interpolated between the transient flux states
    /// and the damper fluxes.
    fn subtransient_emf(&self, eqp: f64, edp: f64, psi1d: f64, psi2q: f64) -> (f64, f64) {
        (self.ad * eqp + self.bd * psi1d, self.aq * edp + self.bq * psi2q)
    }

    fn det(&self, w: f64) -> f64 {
        self.ra * self.ra + w * w * self.xdpp * self.xqpp
    }

    fn currents(
        &self,
        eqpp: f64,
        edpp: f64,
        v_d: f64,
        v_q: f64,
        w: f64,
    ) -> (f64, f64, f64, f64) {
        let (a, b) = (w * edpp - v_d, w * eqpp - v_q);
        let det = self.det(w);
        (
            (self.ra * a + w * self.xqpp * b) / det,
            (-w * self.xdpp * a + self.ra * b) / det,
            a,
            b,
        )
    }
}

impl Machine for GenRound {
    fn n_states(&self) -> usize {
        6
    }

    fn state_names(&self) -> &[&'static str] {
        &GENROUND_STATES
    }

    fn omega_index(&self) -> usize {
        1
    }

    fn norton_admittance(&self) -> Complex<f64> {
        self.y
    }

    fn derivatives(&self, x: &[f64], v: Complex<f64>, e_fd: f64, p_m: f64, out: &mut [f64]) {
        let (delta, omega, eqp, edp, psi1d, psi2q) = (x[0], x[1], x[2], x[3], x[4], x[5]);
        let (w, _) = self.speed(omega);
        let (v_d, v_q) = GenTransient::to_dq(delta, v);
        let (eqpp, edpp) = self.subtransient_emf(eqp, edp, psi1d, psi2q);
        let (i_d, i_q, _, _) = self.currents(eqpp, edpp, v_d, v_q, w);
        let (mech, _) = self.mechanical(p_m, omega);

        let p_e = edpp * i_d + eqpp * i_q + (self.xqpp - self.xdpp) * i_d * i_q;
        let delta_d = psi1d - eqp + (self.xdp - self.xl) * i_d;
        let delta_q = psi2q - edp - (self.xqp - self.xl) * i_q;

        out[0] = self.omega_base * (omega - 1.0);
        out[1] = (mech - p_e - self.d * (omega - 1.0)) / (2.0 * self.h);
        out[2] = (-eqp - (self.xd - self.xdp) * (i_d - self.kd * delta_d) + e_fd) / self.td0p;
        out[3] = (-edp + (self.xq - self.xqp) * (i_q + self.kq * delta_q)) / self.tq0p;
        out[4] = -delta_d / self.td0pp;
        out[5] = -delta_q / self.tq0pp;
    }

    fn injection(&self, x: &[f64], v: Complex<f64>) -> Complex<f64> {
        let (delta, omega, eqp, edp, psi1d, psi2q) = (x[0], x[1], x[2], x[3], x[4], x[5]);
        let (w, _) = self.speed(omega);
        let (v_d, v_q) = GenTransient::to_dq(delta, v);
        let (eqpp, edpp) = self.subtransient_emf(eqp, edp, psi1d, psi2q);
        let (i_d, i_q, _, _) = self.currents(eqpp, edpp, v_d, v_q, w);
        GenTransient::from_dq(delta, i_d, i_q) + self.y * v
    }

    fn jacobian(
        &self,
        x: &[f64],
        v: Complex<f64>,
        _e_fd: f64,
        p_m: f64,
        out: &mut MachineJacobian,
    ) {
        let (delta, omega, eqp, edp, psi1d, psi2q) = (x[0], x[1], x[2], x[3], x[4], x[5]);
        let (w, dw) = self.speed(omega);
        let (_, dmech) = self.mechanical(p_m, omega);
        let (sd, cd) = delta.sin_cos();
        let (v_d, v_q) = GenTransient::to_dq(delta, v);
        let (eqpp, edpp) = self.subtransient_emf(eqp, edp, psi1d, psi2q);
        let (i_d, i_q, a, b) = self.currents(eqpp, edpp, v_d, v_q, w);
        let det = self.det(w);
        let two_h = 2.0 * self.h;
        let n = 6;

        // Columns, in order: δ, ω, e'_q, e'_d, ψ_1d, ψ_2q, v_re, v_im.
        //
        // `a = e''_d − v_d` and `b = e''_q − v_q` are all the currents depend
        // on, and the subtransient EMFs are linear in the flux states, so each
        // column is one `(∂a, ∂b)` pair pushed through one fixed 2×2 inverse.
        let dab: [(f64, f64); 8] = [
            (-v_q, v_d),             // δ
            (0.0, 0.0),              // ω — handled below, not linear here
            (0.0, w * self.ad),      // e'_q
            (w * self.aq, 0.0),      // e'_d
            (0.0, w * self.bd),      // ψ_1d
            (w * self.bq, 0.0),      // ψ_2q
            (-sd, -cd),              // v_re
            (cd, -sd),               // v_im
        ];
        // ∂e''_q and ∂e''_d, in the same column order.
        let deqpp = [0.0, 0.0, self.ad, 0.0, self.bd, 0.0, 0.0, 0.0];
        let dedpp = [0.0, 0.0, 0.0, self.aq, 0.0, self.bq, 0.0, 0.0];

        let mut di_d = [0.0; 8];
        let mut di_q = [0.0; 8];
        for (k, (da, db)) in dab.iter().enumerate() {
            di_d[k] = (self.ra * da + w * self.xqpp * db) / det;
            di_q[k] = (-w * self.xdpp * da + self.ra * db) / det;
        }
        // The speed factor appears in the coefficients *and* in `det`, so the
        // `ω` column is a product rule rather than the fixed inverse above.
        // `dw` is zero in the approximate form, leaving this column zero.
        let ddet = 2.0 * w * dw * self.xdpp * self.xqpp;
        di_d[1] = (self.ra * dw * edpp + dw * self.xqpp * b + w * self.xqpp * dw * eqpp) / det
            - i_d * ddet / det;
        di_q[1] = (-dw * self.xdpp * a - w * self.xdpp * dw * edpp + self.ra * dw * eqpp) / det
            - i_q * ddet / det;

        let sal = self.xqpp - self.xdpp;
        let (xdd, xqq) = (self.xdp - self.xl, self.xqp - self.xl);
        let mut dpe = [0.0; 8];
        let mut d_delta_d = [0.0; 8];
        let mut d_delta_q = [0.0; 8];
        for k in 0..8 {
            dpe[k] = edpp * di_d[k]
                + eqpp * di_q[k]
                + sal * (i_q * di_d[k] + i_d * di_q[k])
                + i_d * dedpp[k]
                + i_q * deqpp[k];
            d_delta_d[k] = xdd * di_d[k];
            d_delta_q[k] = -xqq * di_q[k];
        }
        // Clippy would rather these three were iterator chains; they are one
        // fused pass over parallel arrays and read better as an indexed loop.
        d_delta_d[4] += 1.0; // ∂Δ_d/∂ψ_1d
        d_delta_d[2] -= 1.0; // ∂Δ_d/∂e'_q
        d_delta_q[5] += 1.0; // ∂Δ_q/∂ψ_2q
        d_delta_q[3] -= 1.0; // ∂Δ_q/∂e'_d

        // `dfdx` is row-major n×n.
        out.dfdx[1] = self.omega_base;
        for k in 0..n {
            out.dfdx[n + k] = -dpe[k] / two_h;
            out.dfdx[2 * n + k] =
                -(self.xd - self.xdp) * (di_d[k] - self.kd * d_delta_d[k]) / self.td0p;
            out.dfdx[3 * n + k] =
                (self.xq - self.xqp) * (di_q[k] + self.kq * d_delta_q[k]) / self.tq0p;
            out.dfdx[4 * n + k] = -d_delta_d[k] / self.td0pp;
            out.dfdx[5 * n + k] = -d_delta_q[k] / self.tq0pp;
        }
        out.dfdx[n + 1] += (dmech - self.d) / two_h;
        out.dfdx[2 * n + 2] -= 1.0 / self.td0p;
        out.dfdx[3 * n + 3] -= 1.0 / self.tq0p;

        // `dfdv` is row-major n×2; columns 6 and 7 above are v_re and v_im.
        for (col, k) in [(0usize, 6usize), (1, 7)] {
            out.dfdv[2 + col] = -dpe[k] / two_h;
            out.dfdv[4 + col] =
                -(self.xd - self.xdp) * (di_d[k] - self.kd * d_delta_d[k]) / self.td0p;
            out.dfdv[6 + col] =
                (self.xq - self.xqp) * (di_q[k] + self.kq * d_delta_q[k]) / self.tq0p;
            out.dfdv[8 + col] = -d_delta_d[k] / self.td0pp;
            out.dfdv[10 + col] = -d_delta_q[k] / self.tq0pp;
        }

        out.dfde[2] = 1.0 / self.td0p;
        out.dfdp[1] = if self.speed_voltages { 1.0 / (omega * two_h) } else { 1.0 / two_h };

        for k in 0..n {
            let (mut d_re, mut d_im) =
                (di_d[k] * sd + di_q[k] * cd, -di_d[k] * cd + di_q[k] * sd);
            if k == 0 {
                d_re += i_d * cd - i_q * sd;
                d_im += i_d * sd + i_q * cd;
            }
            out.didx[k] = d_re;
            out.didx[n + k] = d_im;
        }

        let (g, b) = (self.y.re, self.y.im);
        for (col, k) in [(0usize, 6usize), (1, 7)] {
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

        // The same q-axis locator the fourth-order model uses, and for the same
        // reason: at steady state the machine looks like x_q behind its
        // terminal on the q axis, whatever damper windings it has. Working that
        // through the subtransient stator equations gives
        // `v_d = x_q·i_q − r_a·i_d` exactly, which is the statement that the
        // d-axis component of `V + (r_a + j·x_q)·I` is zero.
        let e_q = v + Complex::new(self.ra, self.xq) * i;
        let delta = e_q.arg();

        let (v_d, v_q) = GenTransient::to_dq(delta, v);
        let (i_d, i_q) = GenTransient::to_dq(delta, i);

        // Work inward from the stator, then outward through the steady-state
        // flux relations. Every step is forced; there is nothing to choose.
        // Initialization is at synchronous speed, where the two forms coincide
        // exactly, so nothing here needs to know which is in use.
        let eqpp = v_q + self.ra * i_q + self.xdpp * i_d;
        let edpp = v_d + self.ra * i_d - self.xqpp * i_q;
        let edp = (self.xq - self.xqp) * i_q;
        let psi2q = edp + (self.xqp - self.xl) * i_q;
        let eqp = eqpp + (self.xdp - self.xdpp) * i_d;
        let psi1d = eqp - (self.xdp - self.xl) * i_d;

        let e_fd = eqp + (self.xd - self.xdp) * i_d;
        let p_m = edpp * i_d + eqpp * i_q + (self.xqpp - self.xdpp) * i_d * i_q;

        Ok(MachineInit {
            states: vec![delta, 1.0, eqp, edp, psi1d, psi2q],
            e_fd,
            p_m,
        })
    }
}

// ---------------------------------------------------------------------------
// Fifth-order, salient pole
// ---------------------------------------------------------------------------

/// The fifth-order machine: a field winding and a damper on the `d` axis, and
/// **one** damper on the `q` axis.
///
/// States: `δ`, `ω`, `e'_q`, `ψ_1d`, `e''_d`.
///
/// This is the model a salient-pole machine gets, and it is what Dynawo calls
/// `GeneratorSynchronousThreeWindings` and PSS/E calls `GENSAL`. Its two axes
/// are borrowed wholesale from the models either side of it: the `d` axis is
/// [`GenRound`]'s, with the same interpolated subtransient EMF and the same
/// damper-coupling correction, and the `q` axis is [`GenTransient`]'s, with a
/// single winding and therefore a single flux state.
///
/// ```text
/// e''_q = a_d·e'_q + b_d·ψ_1d          Δ_d = ψ_1d − e'_q + (x'_d − x_l)·i_d
///
/// T'_d0 ·ė'_q  = −e'_q − (x_d − x'_d)(i_d − K_d·Δ_d) + E_fd
/// T''_d0·ψ̇_1d = −Δ_d
/// T''_q0·ė''_d = −e''_d + (x_q − x''_q)·i_q
/// ```
///
/// There is no `x'_q` and no `T'_q0`, and that is the physical content rather
/// than a simplification: one winding gives one time constant. A salient-pole
/// rotor has no path for a `q`-axis transient because it has no `q`-axis field
/// — the flux there sees the interpolar gap, and only the damper bars respond.
///
/// PSS/E's `GENSAL` additionally forces `x''_q = x''_d`; Dynawo states the two
/// separately. This model allows them to differ, which reads both.
#[derive(Clone, Debug)]
pub struct GenSalient {
    h: f64,
    d: f64,
    ra: f64,
    xd: f64,
    xdp: f64,
    xdpp: f64,
    xq: f64,
    xqpp: f64,
    xl: f64,
    td0p: f64,
    td0pp: f64,
    tq0pp: f64,
    ad: f64,
    bd: f64,
    kd: f64,
    omega_base: f64,
    y: Complex<f64>,
    speed_voltages: bool,
}

/// [`GenSalient`]'s parameters, on the machine's own MVA rating.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenSalientParams {
    pub h: f64,
    pub d: f64,
    pub ra: f64,
    pub xd: f64,
    pub xq: f64,
    pub xdp: f64,
    pub xdpp: f64,
    /// The single `q`-axis subtransient reactance. There is no `x'_q`.
    pub xqpp: f64,
    pub xl: f64,
    pub td0p: f64,
    pub td0pp: f64,
    /// The single `q`-axis open-circuit time constant.
    pub tq0pp: f64,
    pub mbase: f64,
}

const GENSALIENT_STATES: [&str; 5] = ["delta", "omega", "eqp", "psi1d", "edpp"];

impl GenSalient {
    pub fn new(params: GenSalientParams, s_base: f64, f_nom: f64) -> Result<Self, InitError> {
        require_positive("h", params.h)?;
        require_positive("td0p", params.td0p)?;
        require_positive("td0pp", params.td0pp)?;
        require_positive("tq0pp", params.tq0pp)?;
        for (name, lo, hi) in [
            ("xdpp - xl", params.xl, params.xdpp),
            ("xdp - xdpp", params.xdpp, params.xdp),
            ("xd - xdp", params.xdp, params.xd),
            ("xq - xqpp", params.xqpp, params.xq),
        ] {
            if hi <= lo {
                return Err(InitError::NonPositiveParameter { name, value: hi - lo });
            }
        }

        let k = base_ratio(params.mbase, s_base)?;
        let (xdp, xdpp, xl) = (params.xdp / k, params.xdpp / k, params.xl / k);
        let (ra, xqpp) = (params.ra / k, params.xqpp / k);
        let bd = (xdp - xdpp) / (xdp - xl);
        let z_norton = Complex::new(ra, 0.5 * (xdpp + xqpp));
        Ok(Self {
            h: params.h * k,
            d: params.d * k,
            ra,
            xd: params.xd / k,
            xdp,
            xdpp,
            xq: params.xq / k,
            xqpp,
            xl,
            td0p: params.td0p,
            td0pp: params.td0pp,
            tq0pp: params.tq0pp,
            ad: 1.0 - bd,
            bd,
            kd: bd / (xdp - xl),
            omega_base: std::f64::consts::TAU * f_nom,
            y: Complex::new(1.0, 0.0) / z_norton,
            speed_voltages: false,
        })
    }

    /// Carry the rotor speed on the speed-voltage terms, and write the swing
    /// equation in torque. See the module doc.
    pub fn with_speed_voltages(mut self, on: bool) -> Self {
        self.speed_voltages = on;
        self
    }

    pub fn h(&self) -> f64 {
        self.h
    }

    fn speed(&self, omega: f64) -> (f64, f64) {
        if self.speed_voltages { (omega, 1.0) } else { (1.0, 0.0) }
    }

    fn mechanical(&self, p_m: f64, omega: f64) -> (f64, f64) {
        if self.speed_voltages {
            (p_m / omega, -p_m / (omega * omega))
        } else {
            (p_m, 0.0)
        }
    }

    fn det(&self, w: f64) -> f64 {
        self.ra * self.ra + w * w * self.xdpp * self.xqpp
    }

    fn currents(&self, eqpp: f64, edpp: f64, v_d: f64, v_q: f64, w: f64) -> (f64, f64, f64, f64) {
        let (a, b) = (w * edpp - v_d, w * eqpp - v_q);
        let det = self.det(w);
        (
            (self.ra * a + w * self.xqpp * b) / det,
            (-w * self.xdpp * a + self.ra * b) / det,
            a,
            b,
        )
    }
}

impl Machine for GenSalient {
    fn n_states(&self) -> usize {
        5
    }

    fn state_names(&self) -> &[&'static str] {
        &GENSALIENT_STATES
    }

    fn omega_index(&self) -> usize {
        1
    }

    fn norton_admittance(&self) -> Complex<f64> {
        self.y
    }

    fn derivatives(&self, x: &[f64], v: Complex<f64>, e_fd: f64, p_m: f64, out: &mut [f64]) {
        let (delta, omega, eqp, psi1d, edpp) = (x[0], x[1], x[2], x[3], x[4]);
        let (w, _) = self.speed(omega);
        let (v_d, v_q) = GenTransient::to_dq(delta, v);
        let eqpp = self.ad * eqp + self.bd * psi1d;
        let (i_d, i_q, _, _) = self.currents(eqpp, edpp, v_d, v_q, w);
        let (mech, _) = self.mechanical(p_m, omega);

        let c_e = edpp * i_d + eqpp * i_q + (self.xqpp - self.xdpp) * i_d * i_q;
        let delta_d = psi1d - eqp + (self.xdp - self.xl) * i_d;

        out[0] = self.omega_base * (omega - 1.0);
        out[1] = (mech - c_e - self.d * (omega - 1.0)) / (2.0 * self.h);
        out[2] = (-eqp - (self.xd - self.xdp) * (i_d - self.kd * delta_d) + e_fd) / self.td0p;
        out[3] = -delta_d / self.td0pp;
        out[4] = (-edpp + (self.xq - self.xqpp) * i_q) / self.tq0pp;
    }

    fn injection(&self, x: &[f64], v: Complex<f64>) -> Complex<f64> {
        let (delta, omega, eqp, psi1d, edpp) = (x[0], x[1], x[2], x[3], x[4]);
        let (w, _) = self.speed(omega);
        let (v_d, v_q) = GenTransient::to_dq(delta, v);
        let eqpp = self.ad * eqp + self.bd * psi1d;
        let (i_d, i_q, _, _) = self.currents(eqpp, edpp, v_d, v_q, w);
        GenTransient::from_dq(delta, i_d, i_q) + self.y * v
    }

    fn jacobian(
        &self,
        x: &[f64],
        v: Complex<f64>,
        _e_fd: f64,
        p_m: f64,
        out: &mut MachineJacobian,
    ) {
        let (delta, omega, eqp, psi1d, edpp) = (x[0], x[1], x[2], x[3], x[4]);
        let (w, dw) = self.speed(omega);
        let (_, dmech) = self.mechanical(p_m, omega);
        let (sd, cd) = delta.sin_cos();
        let (v_d, v_q) = GenTransient::to_dq(delta, v);
        let eqpp = self.ad * eqp + self.bd * psi1d;
        let (i_d, i_q, a, b) = self.currents(eqpp, edpp, v_d, v_q, w);
        let det = self.det(w);
        let two_h = 2.0 * self.h;
        let n = 5;

        // Columns: δ, ω, e'_q, ψ_1d, e''_d, v_re, v_im.
        let dab: [(f64, f64); 7] = [
            (-v_q, v_d),
            (0.0, 0.0),
            (0.0, w * self.ad),
            (0.0, w * self.bd),
            (w, 0.0),
            (-sd, -cd),
            (cd, -sd),
        ];
        let deqpp = [0.0, 0.0, self.ad, self.bd, 0.0, 0.0, 0.0];
        let dedpp = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let mut di_d = [0.0; 7];
        let mut di_q = [0.0; 7];
        for (k, (da, db)) in dab.iter().enumerate() {
            di_d[k] = (self.ra * da + w * self.xqpp * db) / det;
            di_q[k] = (-w * self.xdpp * da + self.ra * db) / det;
        }
        let ddet = 2.0 * w * dw * self.xdpp * self.xqpp;
        di_d[1] = (self.ra * dw * edpp + dw * self.xqpp * b + w * self.xqpp * dw * eqpp) / det
            - i_d * ddet / det;
        di_q[1] = (-dw * self.xdpp * a - w * self.xdpp * dw * edpp + self.ra * dw * eqpp) / det
            - i_q * ddet / det;

        let sal = self.xqpp - self.xdpp;
        let xdd = self.xdp - self.xl;
        let mut dce = [0.0; 7];
        let mut d_delta_d = [0.0; 7];
        for k in 0..7 {
            dce[k] = edpp * di_d[k]
                + eqpp * di_q[k]
                + sal * (i_q * di_d[k] + i_d * di_q[k])
                + i_d * dedpp[k]
                + i_q * deqpp[k];
            d_delta_d[k] = xdd * di_d[k];
        }
        d_delta_d[3] += 1.0; // ∂Δ_d/∂ψ_1d
        d_delta_d[2] -= 1.0; // ∂Δ_d/∂e'_q

        out.dfdx[1] = self.omega_base;
        for k in 0..n {
            out.dfdx[n + k] = -dce[k] / two_h;
            out.dfdx[2 * n + k] =
                -(self.xd - self.xdp) * (di_d[k] - self.kd * d_delta_d[k]) / self.td0p;
            out.dfdx[3 * n + k] = -d_delta_d[k] / self.td0pp;
            out.dfdx[4 * n + k] = (self.xq - self.xqpp) * di_q[k] / self.tq0pp;
        }
        out.dfdx[n + 1] += (dmech - self.d) / two_h;
        out.dfdx[2 * n + 2] -= 1.0 / self.td0p;
        out.dfdx[4 * n + 4] -= 1.0 / self.tq0pp;

        for (col, k) in [(0usize, 5usize), (1, 6)] {
            out.dfdv[2 + col] = -dce[k] / two_h;
            out.dfdv[4 + col] =
                -(self.xd - self.xdp) * (di_d[k] - self.kd * d_delta_d[k]) / self.td0p;
            out.dfdv[6 + col] = -d_delta_d[k] / self.td0pp;
            out.dfdv[8 + col] = (self.xq - self.xqpp) * di_q[k] / self.tq0pp;
        }

        out.dfde[2] = 1.0 / self.td0p;
        out.dfdp[1] = if self.speed_voltages { 1.0 / (omega * two_h) } else { 1.0 / two_h };

        for k in 0..n {
            let (mut d_re, mut d_im) =
                (di_d[k] * sd + di_q[k] * cd, -di_d[k] * cd + di_q[k] * sd);
            if k == 0 {
                d_re += i_d * cd - i_q * sd;
                d_im += i_d * sd + i_q * cd;
            }
            out.didx[k] = d_re;
            out.didx[n + k] = d_im;
        }

        let (g, b_y) = (self.y.re, self.y.im);
        for (col, k) in [(0usize, 5usize), (1, 6)] {
            out.didv[col] = di_d[k] * sd + di_q[k] * cd;
            out.didv[2 + col] = -di_d[k] * cd + di_q[k] * sd;
        }
        out.didv[0] += g;
        out.didv[1] += -b_y;
        out.didv[2] += b_y;
        out.didv[3] += g;
    }

    fn initialize(&mut self, v: Complex<f64>, s: Complex<f64>) -> Result<MachineInit, InitError> {
        if v.norm() == 0.0 {
            return Err(InitError::ZeroTerminalVoltage);
        }
        let i = (s / v).conj();

        // The same q-axis locator the other models use. It still holds: at
        // steady state `e''_d = (x_q − x''_q)·i_q`, so the subtransient stator
        // relation reduces to `v_d = x_q·i_q − r_a·i_d`, which is exactly the
        // statement that E_q has no d-axis component.
        let e_q = v + Complex::new(self.ra, self.xq) * i;
        let delta = e_q.arg();

        let (v_d, v_q) = GenTransient::to_dq(delta, v);
        let (i_d, i_q) = GenTransient::to_dq(delta, i);

        let eqpp = v_q + self.ra * i_q + self.xdpp * i_d;
        let edpp = (self.xq - self.xqpp) * i_q;
        let eqp = eqpp + (self.xdp - self.xdpp) * i_d;
        let psi1d = eqp - (self.xdp - self.xl) * i_d;

        let e_fd = eqp + (self.xd - self.xdp) * i_d;
        let p_m = edpp * i_d + eqpp * i_q + (self.xqpp - self.xdpp) * i_d * i_q;

        Ok(MachineInit { states: vec![delta, 1.0, eqp, psi1d, edpp], e_fd, p_m })
    }
}
