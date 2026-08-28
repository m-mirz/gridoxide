//! Excitation systems.
//!
//! An exciter compares the terminal voltage against a reference and drives the
//! machine's field voltage to close the gap. It is what turns a generator from
//! a fixed voltage source behind a reactance into something that *holds* a
//! voltage — and it is the first control here with a fast enough time constant
//! to matter for the integrator's damping (see
//! [`integrator`](crate::dynamics::integrator)).
//!
//! # Limits are deliberately absent
//!
//! Real exciters have ceiling and floor limits on `E_fd`, and a real study of a
//! severe fault is shaped by them. They are not implemented here, and that is a
//! considered omission rather than an oversight: a hard clamp makes the
//! right-hand side non-smooth, so the analytic Jacobian acquires a
//! discontinuity that the step's Newton solve can chatter against, and doing it
//! properly needs non-windup limiter logic plus the state to remember whether a
//! limit is active. That is a piece of machinery in its own right.
//!
//! Half-implemented limits would be worse than none, because they would look
//! present. A model with no limits is at least honestly unlimited, and its
//! `E_fd` can be read to see whether a study would have hit one.

use serde::{Deserialize, Serialize};

use super::{Control, InitError};

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
            v_ref: 0.0,
        })
    }

    /// The latched voltage reference, per unit. Exposed so a test can check
    /// that the machine really is driven back to the voltage it started at.
    pub fn v_ref(&self) -> f64 {
        self.v_ref
    }
}

impl Control for Sexs {
    fn n_states(&self) -> usize {
        2
    }

    fn state_names(&self) -> &[&'static str] {
        &SEXS_STATES
    }

    fn derivatives(&self, x: &[f64], u: f64, out: &mut [f64]) {
        let w = self.v_ref + u;
        let y1 = x[0] + self.ta_over_tb * w;
        out[0] = (w * (1.0 - self.ta_over_tb) - x[0]) / self.tb;
        out[1] = (self.k * y1 - x[1]) / self.te;
    }

    fn jacobian(&self, _x: &[f64], _u: f64, dfdx: &mut [f64], dfdu: &mut [f64]) {
        dfdx[0] = -1.0 / self.tb;
        dfdx[1] = 0.0;
        dfdu[0] = (1.0 - self.ta_over_tb) / self.tb;

        dfdx[2] = self.k / self.te;
        dfdx[3] = -1.0 / self.te;
        dfdu[1] = self.k * self.ta_over_tb / self.te;
    }

    fn output(&self, x: &[f64], _u: f64) -> f64 {
        x[1]
    }

    fn output_jacobian(&self, _x: &[f64], _u: f64, dydx: &mut [f64]) -> f64 {
        dydx[0] = 0.0;
        dydx[1] = 1.0;
        // No feedthrough: the field voltage is a state, so a step in terminal
        // voltage reaches E_fd only through T_e.
        0.0
    }

    fn initialize(&mut self, y: f64, u: f64) -> Result<Vec<f64>, InitError> {
        // At DC the lead-lag is unity, so E_fd = K·(V_ref + u) and the
        // reference is whatever makes that reproduce the field voltage the
        // machine turned out to need.
        let w = y / self.k;
        self.v_ref = w - u;
        Ok(vec![w * (1.0 - self.ta_over_tb), y])
    }
}
