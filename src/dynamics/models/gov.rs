//! Turbine-governor models.
//!
//! A governor reads the rotor speed and adjusts the mechanical power to oppose
//! a deviation. Without one, `P_m` is a constant and the frequency of an
//! islanded system never returns after a disturbance — it settles at whatever
//! the swing equation's new equilibrium is, or does not settle at all.
//!
//! Valve **position** limits are implemented, and non-windup, for the reason
//! [`avr`](super::avr) sets out: a governor whose valve is wide open cannot
//! open further however far the frequency falls, and a wound-up integrator
//! would keep it there long after the frequency recovered. Valve *rate* limits
//! are not implemented — they constrain the derivative rather than the state,
//! which is a different and more intrusive piece of machinery.

use serde::{Deserialize, Serialize};

use std::cell::Cell;

use super::{Control, InitError, LimitState, Limits};

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
    limits: Limits,
    /// Which limit is holding the valve this step, decided once at its start.
    limit: Cell<LimitState>,
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
    /// Valve position limits, per unit on the network base. Unbounded when
    /// absent.
    #[serde(default)]
    pub limits: Limits,
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
            limits: params.limits,
            limit: Cell::new(LimitState::Free),
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

    /// The valve's derivative before any limit is applied.
    fn raw_valve_derivative(&self, x: &[f64], u: f64) -> f64 {
        (self.p_ref - u / self.r - x[0]) / self.t1
    }
}

impl Control for Tgov1 {
    fn n_states(&self) -> usize {
        2
    }

    fn state_names(&self) -> &[&'static str] {
        &TGOV1_STATES
    }

    fn latch(&self, x: &[f64], u: f64) {
        self.limit.set(self.limits.latch(x[0], self.raw_valve_derivative(x, u)));
    }

    fn derivatives(&self, x: &[f64], u: f64, out: &mut [f64]) {
        out[0] = self.limits.held(self.limit.get(), self.raw_valve_derivative(x, u));
        out[1] = (self.limits.under(self.limit.get(), x[0]).0 - x[1]) / self.t3;
    }

    fn jacobian(&self, x: &[f64], _u: f64, dfdx: &mut [f64], dfdu: &mut [f64]) {
        if self.limit.get() == LimitState::Free {
            dfdx[0] = -1.0 / self.t1;
            dfdu[0] = -1.0 / (self.r * self.t1);
        }
        dfdx[1] = 0.0;

        dfdx[2] = self.limits.under(self.limit.get(), x[0]).1 / self.t3;
        dfdx[3] = -1.0 / self.t3;
        dfdu[1] = 0.0;
    }

    fn output(&self, x: &[f64], u: f64) -> f64 {
        let valve = self.limits.under(self.limit.get(), x[0]).0;
        x[1] + self.t2_over_t3 * (valve - x[1]) - self.dt * u
    }

    fn output_jacobian(&self, x: &[f64], _u: f64, dydx: &mut [f64]) -> f64 {
        dydx[0] = self.t2_over_t3 * self.limits.under(self.limit.get(), x[0]).1;
        dydx[1] = 1.0 - self.t2_over_t3;
        -self.dt
    }

    fn project(&self, x: &mut [f64]) {
        x[0] = self.limits.clamp(x[0]).0;
    }

    fn initialize(&mut self, y: f64, u: f64) -> Result<Vec<f64>, InitError> {
        // At equilibrium both lags have settled, so x₁ = x₂ and P_m is exactly
        // x₂ plus the damping term. The reference absorbs whatever speed
        // deviation the operating point has — normally none.
        let x = y + self.dt * u;
        self.limits.require("governor valve position", x)?;
        self.p_ref = x + u / self.r;
        Ok(vec![x, x])
    }
}

/// A purely proportional governor: `P_m = P_ref − K·Δω`.
///
/// No states, no lags: the mechanical power follows the speed within the
/// instant. Real turbines do not, which is what [`Tgov1`]'s two time constants
/// are for — but this is exactly Dynawo's `GoverProportional`, so a Dynawo case
/// maps onto it without inventing time constants that were never stated.
///
/// The gain is the reciprocal of a droop: `K = 1/R`. Stated as a gain here
/// rather than as a droop because that is what the files carry, and converting
/// once at the boundary beats converting at every use.
#[derive(Clone, Debug)]
pub struct GoverProportional {
    k: f64,
    limits: Limits,
    limit: Cell<LimitState>,
    p_ref: f64,
}

const NO_GOV_STATES: [&str; 0] = [];

impl GoverProportional {
    /// `k` is per unit on the **network** base: a gain of 20 is a 5% droop.
    pub fn new(k: f64) -> Result<Self, InitError> {
        Self::limited(k, Limits::NONE)
    }

    /// With mechanical-power limits. No states, so nothing winds up behind the
    /// clamp.
    pub fn limited(k: f64, limits: Limits) -> Result<Self, InitError> {
        if k <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "gov gain", value: k });
        }
        Ok(Self { k, limits, limit: Cell::new(LimitState::Free), p_ref: 0.0 })
    }

    pub fn p_ref(&self) -> f64 {
        self.p_ref
    }

    /// The equivalent droop, for comparison with [`Tgov1::r`].
    pub fn droop(&self) -> f64 {
        1.0 / self.k
    }
}

impl Control for GoverProportional {
    fn n_states(&self) -> usize {
        0
    }

    fn state_names(&self) -> &[&'static str] {
        &NO_GOV_STATES
    }

    fn derivatives(&self, _x: &[f64], _u: f64, _out: &mut [f64]) {}

    fn jacobian(&self, _x: &[f64], _u: f64, _dfdx: &mut [f64], _dfdu: &mut [f64]) {}

    fn latch(&self, _x: &[f64], u: f64) {
        let raw = self.p_ref - self.k * u;
        self.limit.set(if raw > self.limits.upper() {
            LimitState::AtMax
        } else if raw < self.limits.lower() {
            LimitState::AtMin
        } else {
            LimitState::Free
        });
    }

    fn output(&self, _x: &[f64], u: f64) -> f64 {
        self.limits.under(self.limit.get(), self.p_ref - self.k * u).0
    }

    fn output_jacobian(&self, _x: &[f64], u: f64, _dydx: &mut [f64]) -> f64 {
        -self.k * self.limits.under(self.limit.get(), self.p_ref - self.k * u).1
    }

    fn initialize(&mut self, y: f64, u: f64) -> Result<Vec<f64>, InitError> {
        self.limits.require("governor mechanical power", y)?;
        self.p_ref = y + self.k * u;
        Ok(Vec::new())
    }
}
