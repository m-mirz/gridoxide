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

// Moved to `roots` so dynamics can locate its events without depending on
// continuation; re-exported so this path keeps working.
pub use crate::roots::Illinois;

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
    QLimit {
        bus: usize,
        limit: QLimitKind,
        lambda: f64,
        iterations: usize,
    },
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
    q_violations(buses, q_calc)
        .into_iter()
        .map(|(_, e)| e)
        .fold(f64::NEG_INFINITY, f64::max)
}
