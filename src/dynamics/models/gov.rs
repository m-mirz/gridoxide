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
//! would keep it there long after the frequency recovered.
//!
//! Valve **rate** limits are implemented too, and they constrain a different
//! thing: not where the valve may be but how fast it may travel. A steam valve
//! that can open in a fifth of a second and a hydro gate that takes five are
//! the same model with different rates, and the difference decides whether a
//! machine can arrest a frequency excursion at all.
//!
//! The two compose in one order and not the other. The rate limit clips the
//! derivative first; the position limit then decides whether the valve may move
//! at all. Reversing them would let a valve pinned at its ceiling still
//! "travel" at its rate limit, which is nothing.

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
    /// How fast the valve may travel, per unit per second.
    rate: Limits,
    /// Which limit is holding the valve this step, decided once at its start.
    /// Covers both the position and the rate: while either holds, the valve's
    /// derivative is a constant and its Jacobian row is zero.
    limit: Cell<LimitState>,
    /// The rate-clipped derivative for this step, if the rate limit binds.
    held_rate: Cell<Option<f64>>,
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
    /// Valve **rate** limits, per unit per second — how fast it may travel,
    /// closing (`min`, negative) and opening (`max`). Unbounded when absent.
    #[serde(default)]
    pub rate: Limits,
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
            rate: params.rate,
            limit: Cell::new(LimitState::Free),
            held_rate: Cell::new(None),
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

    /// The rate limits, per unit per second.
    pub fn rate(&self) -> Limits {
        self.rate
    }

    /// The valve's derivative under whichever limits are latched for this step.
    fn valve_derivative(&self, x: &[f64], u: f64) -> f64 {
        let raw = self.held_rate.get().unwrap_or_else(|| self.raw_valve_derivative(x, u));
        self.limits.held(self.limit.get(), raw)
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
        // The rate limit clips the derivative first; the position limit then
        // decides whether the valve may move at all. In the other order a valve
        // pinned at its ceiling would still be "travelling" at its rate limit.
        let raw = self.raw_valve_derivative(x, u);
        let (clipped, scale) = self.rate.clamp(raw);
        self.held_rate.set((scale == 0.0).then_some(clipped));
        self.limit.set(self.limits.latch(x[0], clipped));
    }

    fn derivatives(&self, x: &[f64], u: f64, out: &mut [f64]) {
        out[0] = self.valve_derivative(x, u);
        out[1] = (self.limits.under(self.limit.get(), x[0]).0 - x[1]) / self.t3;
    }

    fn jacobian(&self, x: &[f64], _u: f64, dfdx: &mut [f64], dfdu: &mut [f64]) {
        // Free of *both* limits, or the derivative is a constant and its whole
        // row is zero — the exact Jacobian of what the step is solving, since
        // the active set is fixed for its duration.
        if self.limit.get() == LimitState::Free && self.held_rate.get().is_none() {
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
