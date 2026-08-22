//! Reactive-limit enforcement: the `PV → PQ` switch.
//!
//! A `PV` bus holds its voltage by producing whatever reactive power that
//! takes. A real machine cannot: past `q_max` (or below `q_min`) it saturates,
//! the voltage falls away from target, and the bus is no longer a `PV` bus at
//! all — it is a `PQ` bus injecting exactly its limit.
//!
//! This is the standard MATPOWER-style **one-directional** switch: a bus that
//! moves `PV → PQ` stays there for the rest of the run. The reverse switch
//! (a `PQ`-clamped bus whose voltage overshoots its target, which should
//! return to `PV`) is deliberately not implemented — it is where the
//! literature's oscillation reports come from, and the one-directional form is
//! what every reference gridoxide is checked against uses by default.

use crate::network::power_injections;
use crate::types::BusType;

use super::{Invalidates, OuterLoop, OuterLoopContext, OuterLoopStatus};

/// Which buses this loop moved off voltage control, and whether it settled.
#[derive(Clone, Debug, Default)]
pub struct QLimitReport {
    /// Buses switched `PV → PQ`, in switch order.
    pub switches: Vec<usize>,
}

/// Enforces `q_min`/`q_max` on every `PV` bus by switching violators to `PQ`.
///
/// Reports [`Invalidates::Pattern`]: a bus type change moves `n_unknowns`, so
/// the Jacobian's sparsity pattern changes and the cached symbolic
/// factorization cannot be kept. That is the opposite of
/// [`DistributedSlack`](super::DistributedSlack), which only moves `p_spec`.
#[derive(Clone, Debug, Default)]
pub struct ReactiveLimits {
    switches: Vec<usize>,
}

impl ReactiveLimits {
    pub fn new() -> Self {
        Self::default()
    }

    /// What it did, readable after the solve.
    pub fn report(&self) -> QLimitReport {
        QLimitReport { switches: self.switches.clone() }
    }

    /// Buses switched `PV → PQ`, in switch order.
    pub fn switches(&self) -> &[usize] {
        &self.switches
    }
}

impl OuterLoop for ReactiveLimits {
    fn name(&self) -> &'static str {
        "ReactiveLimits"
    }

    fn invalidates(&self) -> Invalidates {
        Invalidates::Pattern
    }

    fn check(&mut self, ctx: &mut OuterLoopContext<'_, '_>) -> OuterLoopStatus {
        let (_, q_calc) = power_injections(ctx.net.buses, ctx.net.ybus);
        let mut switched = false;
        for b in ctx.net.buses.iter_mut() {
            if b.bus_type != BusType::PV {
                continue;
            }
            let q = q_calc[b.idx];
            if q < b.q_min {
                b.bus_type = BusType::PQ;
                b.q_spec = b.q_min;
                self.switches.push(b.idx);
                switched = true;
            } else if q > b.q_max {
                b.bus_type = BusType::PQ;
                b.q_spec = b.q_max;
                self.switches.push(b.idx);
                switched = true;
            }
        }
        if switched {
            OuterLoopStatus::Unstable
        } else {
            OuterLoopStatus::Stable
        }
    }
}
