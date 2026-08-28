//! Turbine-governor models.
//!
//! A governor reads the rotor speed and adjusts the mechanical power to oppose
//! a deviation. Without one, `P_m` is a constant and the frequency of an
//! islanded system never returns after a disturbance — it settles at whatever
//! the swing equation's new equilibrium is, or does not settle at all.
//!
//! Rate and position limits on the valve are absent for the same reason
//! exciter limits are — see [`avr`](super::avr).

use serde::{Deserialize, Serialize};

use super::{Control, InitError};

/// `TGOV1`: a steam turbine-governor, the simplest model in wide use.
///
/// ```text
///                 1              1 + s·T₂
///   Δω ──► droop ──► ─────── ──► ─────────── ──► P_m  (minus D_t·Δω)
///                 1 + s·T₁       1 + s·T₃
/// ```
///
/// ```text
/// ẋ₁ = [(P_ref − Δω/R) − x₁] / T₁
/// ẋ₂ = (x₁ − x₂) / T₃
/// P_m = x₂ + (T₂/T₃)(x₁ − x₂) − D_t·Δω
/// ```
///
/// `R` is the droop: a machine with `R = 0.05` gives up 5% of its speed range
/// for the whole of its power range, so a 1% frequency dip calls for 20% more
/// power. That is the number that decides how the burden of a lost generator is
/// shared, and it is why several machines with different droops settle at one
/// common frequency but different outputs.
///
/// `D_t` is turbine damping, a direct speed-to-power term that bypasses both
/// lags. It is the only feedthrough here, and it is why `P_m` responds to a
/// speed step instantly as well as through `T₁`.
#[derive(Clone, Debug)]
pub struct Tgov1 {
    r: f64,
    t1: f64,
    t2_over_t3: f64,
    t3: f64,
    dt: f64,
    /// Latched by [`Control::initialize`].
    p_ref: f64,
}

/// [`Tgov1`]'s parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tgov1Params {
    /// Droop, per unit. `0.05` is the usual setting.
    pub r: f64,
    /// Governor time constant, seconds.
    pub t1: f64,
    /// Turbine lead, seconds.
    pub t2: f64,
    /// Turbine lag, seconds.
    pub t3: f64,
    /// Turbine damping, per unit. Often zero.
    pub dt: f64,
}

const TGOV1_STATES: [&str; 2] = ["gov_valve", "gov_turbine"];

impl Tgov1 {
    pub fn new(params: Tgov1Params) -> Result<Self, InitError> {
        if params.r <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "gov r", value: params.r });
        }
        if params.t1 <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "gov t1", value: params.t1 });
        }
        if params.t3 <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "gov t3", value: params.t3 });
        }
        Ok(Self {
            r: params.r,
            t1: params.t1,
            t2_over_t3: params.t2 / params.t3,
            t3: params.t3,
            dt: params.dt,
            p_ref: 0.0,
        })
    }

    /// The latched power reference, per unit on the network base.
    pub fn p_ref(&self) -> f64 {
        self.p_ref
    }

    /// The droop, per unit — what decides how a machine shares a disturbance.
    pub fn r(&self) -> f64 {
        self.r
    }
}

impl Control for Tgov1 {
    fn n_states(&self) -> usize {
        2
    }

    fn state_names(&self) -> &[&'static str] {
        &TGOV1_STATES
    }

    fn derivatives(&self, x: &[f64], u: f64, out: &mut [f64]) {
        let demand = self.p_ref - u / self.r;
        out[0] = (demand - x[0]) / self.t1;
        out[1] = (x[0] - x[1]) / self.t3;
    }

    fn jacobian(&self, _x: &[f64], _u: f64, dfdx: &mut [f64], dfdu: &mut [f64]) {
        dfdx[0] = -1.0 / self.t1;
        dfdx[1] = 0.0;
        dfdu[0] = -1.0 / (self.r * self.t1);

        dfdx[2] = 1.0 / self.t3;
        dfdx[3] = -1.0 / self.t3;
        dfdu[1] = 0.0;
    }

    fn output(&self, x: &[f64], u: f64) -> f64 {
        x[1] + self.t2_over_t3 * (x[0] - x[1]) - self.dt * u
    }

    fn output_jacobian(&self, _x: &[f64], _u: f64, dydx: &mut [f64]) -> f64 {
        dydx[0] = self.t2_over_t3;
        dydx[1] = 1.0 - self.t2_over_t3;
        -self.dt
    }

    fn initialize(&mut self, y: f64, u: f64) -> Result<Vec<f64>, InitError> {
        // At equilibrium both lags have settled, so x₁ = x₂ and P_m is exactly
        // x₂ plus the damping term. The reference absorbs whatever speed
        // deviation the operating point has — normally none.
        let x = y + self.dt * u;
        self.p_ref = x + u / self.r;
        Ok(vec![x, x])
    }
}
