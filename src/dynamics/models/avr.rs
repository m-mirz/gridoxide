//! Excitation systems.
//!
//! An exciter compares the terminal voltage against a reference and drives the
//! machine's field voltage to close the gap. It is what turns a generator from
//! a fixed voltage source behind a reactance into something that *holds* a
//! voltage — and it is the first control here with a fast enough time constant
//! to matter for the integrator's damping (see
//! [`integrator`](crate::dynamics::integrator)).
//!
//! # Limits, and why they are non-windup
//!
//! Real exciters have a ceiling and a floor on `E_fd`, and a study of a severe
//! fault is shaped by them: a machine whose field voltage is pinned at its
//! ceiling cannot support its terminal voltage any harder, however large the
//! error.
//!
//! They are implemented as **non-windup** limits, which is the distinction that
//! matters. With windup the state keeps integrating past the ceiling while the
//! output is pinned there, so when the voltage error finally reverses the
//! output stays pinned for however long the state takes to travel back — a
//! delay with no physical basis, and one that can be seconds on a slow
//! exciter. A non-windup limit holds the state *at* the boundary and lets it
//! leave the instant its derivative reverses.
//!
//! The cost is a right-hand side that is not smooth at the boundary, so the
//! analytic Jacobian is exact on each side and undefined exactly on it. That is
//! the same character as [`ZipLoad`](super::load::ZipLoad)'s low-voltage
//! cutoff, and it is handled the same way: the oracle checks each side and
//! never straddles.

use serde::{Deserialize, Serialize};

use std::cell::Cell;

use super::{Control, InitError, LimitState, Limits};

/// The IEEE simplified excitation system, `SEXS`.
///
/// ```text
///                    1 + s·T_a          K
///   V_ref + v_s − |V| ──► ───────── ──► ───────── ──► E_fd
///                    1 + s·T_b        1 + s·T_e
/// ```
///
/// States: the lead-lag's internal state, and the field voltage itself.
///
/// ```text
/// ẋ₁ = [u·(1 − T_a/T_b) − x₁] / T_b        y₁ = x₁ + (T_a/T_b)·u
/// ẋ₂ = (K·y₁ − x₂) / T_e                   E_fd = x₂
/// ```
///
/// `u` here is the *whole* input including the latched reference; the
/// [`Control`] interface passes only `v_s − |V|` and this adds `V_ref`.
///
/// At equilibrium `y₁ = u` regardless of the lead-lag's time constants — the
/// lead-lag is unity at DC — so `E_fd = K·u` and the reference follows:
/// `V_ref = E_fd/K + |V|`. The steady-state voltage error `E_fd/K` is real and
/// is the reason a high `K` is what makes a machine hold its terminal voltage
/// closely.
#[derive(Clone, Debug)]
pub struct Sexs {
    k: f64,
    ta_over_tb: f64,
    tb: f64,
    te: f64,
    limits: Limits,
    /// Which limit is holding `E_fd` this step, decided once at its start.
    limit: Cell<LimitState>,
    /// Latched by [`Control::initialize`], never read from a file.
    v_ref: f64,
}

/// [`Sexs`]'s parameters. Per unit on the machine's own base, and
/// dimensionless besides the time constants, so nothing here needs converting.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SexsParams {
    /// Regulator gain. Large — 100 to 400 is ordinary.
    pub k: f64,
    /// Lead time constant, seconds. Zero is allowed (no lead).
    pub ta: f64,
    /// Lag time constant, seconds.
    pub tb: f64,
    /// Exciter time constant, seconds.
    pub te: f64,
    /// Field-voltage ceiling and floor. Unbounded when absent.
    #[serde(default)]
    pub limits: Limits,
}

const SEXS_STATES: [&str; 2] = ["avr_lead", "efd"];

impl Sexs {
    pub fn new(params: SexsParams) -> Result<Self, InitError> {
        if params.k <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "avr k", value: params.k });
        }
        if params.tb <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "avr tb", value: params.tb });
        }
        if params.te <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "avr te", value: params.te });
        }
        if params.ta < 0.0 {
            return Err(InitError::NonPositiveParameter { name: "avr ta", value: params.ta });
        }
        Ok(Self {
            k: params.k,
            ta_over_tb: params.ta / params.tb,
            tb: params.tb,
            te: params.te,
            limits: params.limits,
            limit: Cell::new(LimitState::Free),
            v_ref: 0.0,
        })
    }

    /// The latched voltage reference, per unit. Exposed so a test can check
    /// that the machine really is driven back to the voltage it started at.
    pub fn v_ref(&self) -> f64 {
        self.v_ref
    }

    /// `Ė_fd` before any limit is applied.
    fn raw_field_derivative(&self, x: &[f64], u: f64) -> f64 {
        let w = self.v_ref + u;
        let y1 = x[0] + self.ta_over_tb * w;
        (self.k * y1 - x[1]) / self.te
    }
}

impl Control for Sexs {
    fn n_states(&self) -> usize {
        2
    }

    fn state_names(&self) -> &[&'static str] {
        &SEXS_STATES
    }

    fn latch(&self, x: &[f64], u: f64) {
        let raw = self.raw_field_derivative(x, u);
        self.limit.set(self.limits.latch(x[1], raw));
    }

    fn derivatives(&self, x: &[f64], u: f64, out: &mut [f64]) {
        let w = self.v_ref + u;
        out[0] = (w * (1.0 - self.ta_over_tb) - x[0]) / self.tb;
        out[1] = self.limits.held(self.limit.get(), self.raw_field_derivative(x, u));
    }

    fn jacobian(&self, _x: &[f64], _u: f64, dfdx: &mut [f64], dfdu: &mut [f64]) {
        dfdx[0] = -1.0 / self.tb;
        dfdx[1] = 0.0;
        dfdu[0] = (1.0 - self.ta_over_tb) / self.tb;

        // A held state's derivative is a constant zero, so its whole row is —
        // the exact Jacobian of what the step is actually solving, since the
        // active set is fixed for its duration.
        if self.limit.get() == LimitState::Free {
            dfdx[2] = self.k / self.te;
            dfdx[3] = -1.0 / self.te;
            dfdu[1] = self.k * self.ta_over_tb / self.te;
        }
    }

    fn output(&self, x: &[f64], _u: f64) -> f64 {
        self.limits.under(self.limit.get(), x[1]).0
    }

    fn output_jacobian(&self, x: &[f64], _u: f64, dydx: &mut [f64]) -> f64 {
        dydx[0] = 0.0;
        dydx[1] = self.limits.under(self.limit.get(), x[1]).1;
        // No feedthrough: the field voltage is a state, so a step in terminal
        // voltage reaches E_fd only through T_e.
        0.0
    }

    fn project(&self, x: &mut [f64]) {
        x[1] = self.limits.clamp(x[1]).0;
    }

    fn initialize(&mut self, y: f64, u: f64) -> Result<Vec<f64>, InitError> {
        // An operating point needing more field voltage than the ceiling allows
        // has no equilibrium; saying so beats initializing to a state the model
        // would immediately leave.
        self.limits.require("exciter field voltage", y)?;
        // At DC the lead-lag is unity, so E_fd = K·(V_ref + u) and the
        // reference is whatever makes that reproduce the field voltage the
        // machine turned out to need.
        let w = y / self.k;
        self.v_ref = w - u;
        Ok(vec![w * (1.0 - self.ta_over_tb), y])
    }
}

/// A purely proportional voltage regulator: `E_fd = K·(V_ref + v_s − |V|)`.
///
/// **No states at all.** The field voltage is an algebraic function of the
/// terminal voltage, so a change in `|V|` reaches `ė'_q` within the same
/// instant rather than through a lag. That is not a simplification of
/// [`Sexs`] — it is a different device, and it is the one Dynawo's
/// `VRProportional` implements, so mapping a Dynawo case onto it is exact
/// rather than approximate.
///
/// It is also the regulator Kundur's worked examples use, which makes it the
/// natural counterpart for the textbook cases.
///
/// A zero-state control is a legitimate member of this library for the same
/// reason a ZIP load is a legitimate device: it contributes derivatives to
/// nothing and sensitivities to everything downstream, and the chain rule in
/// [`unit`](super::unit) needs no special case for it.
#[derive(Clone, Debug)]
pub struct VrProportional {
    k: f64,
    limits: Limits,
    limit: Cell<LimitState>,
    v_ref: f64,
}

const NO_AVR_STATES: [&str; 0] = [];

impl VrProportional {
    pub fn new(k: f64) -> Result<Self, InitError> {
        Self::limited(k, Limits::NONE)
    }

    /// With a field-voltage ceiling and floor. No states means no windup to
    /// worry about: the clamp is on the output and nothing integrates behind
    /// it.
    pub fn limited(k: f64, limits: Limits) -> Result<Self, InitError> {
        if k <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "avr gain", value: k });
        }
        Ok(Self { k, limits, limit: Cell::new(LimitState::Free), v_ref: 0.0 })
    }

    pub fn v_ref(&self) -> f64 {
        self.v_ref
    }
}

impl Control for VrProportional {
    fn n_states(&self) -> usize {
        0
    }

    fn state_names(&self) -> &[&'static str] {
        &NO_AVR_STATES
    }

    fn derivatives(&self, _x: &[f64], _u: f64, _out: &mut [f64]) {}

    fn jacobian(&self, _x: &[f64], _u: f64, _dfdx: &mut [f64], _dfdu: &mut [f64]) {}

    fn latch(&self, _x: &[f64], u: f64) {
        // A pure gain has no state, so "latching" is fixing which branch of the
        // clamp the step uses — the same reason, and the same smoothness.
        let raw = self.k * (self.v_ref + u);
        self.limit.set(if raw > self.limits.upper() {
            LimitState::AtMax
        } else if raw < self.limits.lower() {
            LimitState::AtMin
        } else {
            LimitState::Free
        });
    }

    fn output(&self, _x: &[f64], u: f64) -> f64 {
        self.limits.under(self.limit.get(), self.k * (self.v_ref + u)).0
    }

    fn output_jacobian(&self, _x: &[f64], u: f64, _dydx: &mut [f64]) -> f64 {
        self.k * self.limits.under(self.limit.get(), self.k * (self.v_ref + u)).1
    }

    fn initialize(&mut self, y: f64, u: f64) -> Result<Vec<f64>, InitError> {
        self.limits.require("exciter field voltage", y)?;
        self.v_ref = y / self.k - u;
        Ok(Vec::new())
    }
}
