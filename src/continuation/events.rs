//! Discrete events along the curve, and locating them exactly.
//!
//! # Why the outer-loop driver is not reused
//!
//! [`outerloop::solve_with_loops`](crate::outerloop::solve_with_loops) drives
//! `PersistentSolver::solve`, an ordinary `n × n` Newton at fixed injections; it
//! cannot solve the bordered system at all. More to the point, its fixed-point
//! driver flips a bus the moment a step overshoots a limit — which is precisely
//! the imprecision event location exists to remove, and would make this module
//! dead code.
//!
//! So the corrector runs with bus types **frozen**, and this module finds the λ
//! at which a machine actually saturates. What it does *not* do is re-implement
//! the switching rule:
//! [`ReactiveLimits::check`](crate::outerloop::ReactiveLimits) remains the only
//! code in the crate that flips a bus type and writes `q_spec = limit`, and it
//! is called once, at the located crossing. There is no duplicated rule to drift
//! and no double counting, because it is called nowhere else during a step.
//!
//! The two dispatch loops are handled differently again — see
//! [`LoadingDirection::scale_loads_with_pickup`](super::direction::LoadingDirection::scale_loads_with_pickup)
//! for why they are folded into the direction analytically instead.

use crate::types::{Bus, BusType};

/// Which limit a generator ran into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QLimitKind {
    Max,
    Min,
}

/// Something discrete that happened at a particular λ.
#[derive(Clone, Debug, PartialEq)]
pub enum ContinuationEvent {
    /// A `PV` bus reached a reactive limit and was switched to `PQ`, pinned at
    /// that limit. `lambda` is the located crossing, not the step that
    /// straddled it.
    QLimit { bus: usize, limit: QLimitKind, lambda: f64, iterations: usize },
    /// The safety net fired: an accepted point violated a limit the locator did
    /// not bracket. The switch is applied anyway, so the curve stays
    /// outer-loop-consistent, but the λ is the step's, not the true crossing.
    MissedQLimit { bus: usize, lambda: f64 },
    /// A corrector failed and the step was halved.
    StepRejected { lambda: f64, step: f64 },
}

/// The reactive-limit event function at every bus that is still `PV`.
///
/// Positive means violated, so the earliest crossing of the running maximum is
/// the first machine to saturate. Buses with infinite limits — which is what
/// every importer writes when the source document states none — never
/// contribute.
pub(crate) fn q_violations(buses: &[Bus], q_calc: &[f64]) -> Vec<(usize, f64)> {
    buses
        .iter()
        .filter(|b| b.bus_type == BusType::PV)
        .filter_map(|b| {
            let q = q_calc[b.idx];
            let over = q - b.q_max;
            let under = b.q_min - q;
            let e = over.max(under);
            e.is_finite().then_some((b.idx, e))
        })
        .collect()
}

/// The largest event value over every still-`PV` bus: the scalar the locator
/// brackets, so the *earliest* crossing along the step is the one found.
pub(crate) fn worst_violation(buses: &[Bus], q_calc: &[f64]) -> f64 {
    q_violations(buses, q_calc).into_iter().map(|(_, e)| e).fold(f64::NEG_INFINITY, f64::max)
}

/// Regula falsi with Illinois damping over `[lo, hi]`, given the bracketing
/// values `f_lo < 0 <= f_hi`.
///
/// Each iteration costs a full corrector, so plain bisection would be wasteful;
/// Illinois damping keeps the superlinear rate without losing the bracket the
/// way unmodified regula falsi does when one endpoint stalls.
pub struct Illinois {
    pub lo: f64,
    pub hi: f64,
    pub f_lo: f64,
    pub f_hi: f64,
}

impl Illinois {
    /// The next trial point.
    pub fn next(&self) -> f64 {
        let denom = self.f_hi - self.f_lo;
        if !denom.is_finite() || denom == 0.0 {
            return 0.5 * (self.lo + self.hi);
        }
        let x = self.hi - self.f_hi * (self.hi - self.lo) / denom;
        // Never leave the bracket, and never stall on an endpoint.
        let (a, b) = (self.lo.min(self.hi), self.lo.max(self.hi));
        let margin = 1e-3 * (b - a);
        x.clamp(a + margin, b - margin)
    }

    /// Narrows the bracket with the value measured at `x`.
    pub fn update(&mut self, x: f64, f_x: f64) {
        if f_x >= 0.0 {
            self.hi = x;
            self.f_hi = f_x;
            self.f_lo *= 0.5; // Illinois: halve the stale endpoint's weight
        } else {
            self.lo = x;
            self.f_lo = f_x;
            self.f_hi *= 0.5;
        }
    }

    pub fn width(&self) -> f64 {
        (self.hi - self.lo).abs()
    }
}
