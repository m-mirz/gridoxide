//! Power system stabilizers.
//!
//! A stabilizer adds a small signal to the exciter's input, in phase with the
//! rotor speed, to damp the electromechanical oscillation. It exists because a
//! *high-gain* exciter — the thing that makes a machine hold its voltage well —
//! reduces the natural damping of that oscillation, sometimes to negative. The
//! stabilizer buys the damping back without giving up the voltage regulation.
//!
//! That is the whole reason the signal graph in [`unit`](super::unit) has three
//! blocks and not two: the stabilizer's output is an input to the exciter, not
//! to the machine.
//!
//! Output limits are absent, as they are for the other controls — see
//! [`avr`](super::avr).

use super::{Control, InitError};

/// A speed-input stabilizer: a washout followed by two lead-lag stages.
///
/// ```text
///        s·T_w        1 + s·T₁     1 + s·T₃
///   Δω ──►───────── ──►────────► ──►────────► v_s
///        1 + s·T_w    1 + s·T₂     1 + s·T₄
/// ```
///
/// ```text
/// ẋ₁ = (u − x₁)/T_w                     y₁ = u − x₁
/// ẋ₂ = [K·y₁·(1 − T₁/T₂) − x₂]/T₂       y₂ = x₂ + (T₁/T₂)·K·y₁
/// ẋ₃ = [y₂·(1 − T₃/T₄) − x₃]/T₄         v_s = x₃ + (T₃/T₄)·y₂
/// ```
///
/// **The washout is what makes this initialize to nothing.** It passes no
/// steady signal at all, so `v_s = 0` at any equilibrium whatever the speed is,
/// and the stabilizer has no reference to latch. That is not a convenience: a
/// stabilizer that contributed at steady state would be shifting the voltage
/// setpoint, which is the exciter's job and not its own.
///
/// The lead-lag stages exist to advance the phase of the signal enough to
/// compensate the lag the exciter and the field winding introduce, so that what
/// arrives at the rotor is in phase with speed and therefore damping. Getting
/// that phase wrong makes a stabilizer that *reduces* damping, which is why
/// tuning them is a subject in itself.
#[derive(Clone, Debug)]
pub struct Stab1 {
    k: f64,
    tw: f64,
    t1_over_t2: f64,
    t2: f64,
    t3_over_t4: f64,
    t4: f64,
}

/// [`Stab1`]'s parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stab1Params {
    /// Stabilizer gain.
    pub k: f64,
    /// Washout time constant, seconds. Long compared with the oscillation —
    /// 5 to 10 s is usual — so that it removes the steady component without
    /// distorting the swing.
    pub tw: f64,
    pub t1: f64,
    pub t2: f64,
    pub t3: f64,
    pub t4: f64,
}

const STAB1_STATES: [&str; 3] = ["pss_washout", "pss_lead1", "pss_lead2"];

impl Stab1 {
    pub fn new(params: Stab1Params) -> Result<Self, InitError> {
        for (name, value) in
            [("pss tw", params.tw), ("pss t2", params.t2), ("pss t4", params.t4)]
        {
            if value <= 0.0 {
                return Err(InitError::NonPositiveParameter { name, value });
            }
        }
        Ok(Self {
            k: params.k,
            tw: params.tw,
            t1_over_t2: params.t1 / params.t2,
            t2: params.t2,
            t3_over_t4: params.t3 / params.t4,
            t4: params.t4,
        })
    }

    /// The washout's output and the first stage's, given states and input.
    fn stages(&self, x: &[f64], u: f64) -> (f64, f64) {
        let y1 = u - x[0];
        let y2 = x[1] + self.t1_over_t2 * self.k * y1;
        (y1, y2)
    }
}

impl Control for Stab1 {
    fn n_states(&self) -> usize {
        3
    }

    fn state_names(&self) -> &[&'static str] {
        &STAB1_STATES
    }

    fn derivatives(&self, x: &[f64], u: f64, out: &mut [f64]) {
        let (y1, y2) = self.stages(x, u);
        out[0] = (u - x[0]) / self.tw;
        out[1] = (self.k * y1 * (1.0 - self.t1_over_t2) - x[1]) / self.t2;
        out[2] = (y2 * (1.0 - self.t3_over_t4) - x[2]) / self.t4;
    }

    fn jacobian(&self, _x: &[f64], _u: f64, dfdx: &mut [f64], dfdu: &mut [f64]) {
        let n = 3;
        // ∂y₁/∂x₁ = −1, ∂y₁/∂u = 1.
        dfdx[0] = -1.0 / self.tw;
        dfdu[0] = 1.0 / self.tw;

        let g2 = self.k * (1.0 - self.t1_over_t2) / self.t2;
        dfdx[n] = -g2;
        dfdx[n + 1] = -1.0 / self.t2;
        dfdu[1] = g2;

        // ∂y₂/∂x₁ = −(T₁/T₂)·K, ∂y₂/∂x₂ = 1, ∂y₂/∂u = (T₁/T₂)·K.
        let g3 = (1.0 - self.t3_over_t4) / self.t4;
        let dy2_dx1 = -self.t1_over_t2 * self.k;
        dfdx[2 * n] = g3 * dy2_dx1;
        dfdx[2 * n + 1] = g3;
        dfdx[2 * n + 2] = -1.0 / self.t4;
        dfdu[2] = g3 * self.t1_over_t2 * self.k;
    }

    fn output(&self, x: &[f64], u: f64) -> f64 {
        let (_, y2) = self.stages(x, u);
        x[2] + self.t3_over_t4 * y2
    }

    fn output_jacobian(&self, _x: &[f64], _u: f64, dydx: &mut [f64]) -> f64 {
        let a = self.t3_over_t4 * self.t1_over_t2 * self.k;
        dydx[0] = -a;
        dydx[1] = self.t3_over_t4;
        dydx[2] = 1.0;
        // The washout and both lead-lags each pass their input straight
        // through, so a speed step reaches the exciter with no delay at all.
        a
    }

    fn initialize(&mut self, y: f64, u: f64) -> Result<Vec<f64>, InitError> {
        // A washout passes nothing at steady state, so the only equilibrium
        // with a constant input is the zero state, and the only output it can
        // produce is zero. Anything else asked for here is a caller error, not
        // something to accommodate.
        if y != 0.0 {
            return Err(InitError::OutsideLimits { name: "pss output", value: y });
        }
        let _ = u;
        Ok(vec![u, 0.0, 0.0])
    }
}
