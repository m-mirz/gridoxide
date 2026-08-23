//! The loading direction: what "more load" means for a particular study.
//!
//! λ_max is a property of the *direction*, not of the network. Two studies of
//! the same system that stress different buses, or share the pickup
//! differently, get different noses, and neither is more correct than the
//! other. That is why this is an explicit input with named constructors rather
//! than something the solver infers.
//!
//! [`Bus`](crate::types::Bus) stores only net `p_spec`/`q_spec` — every
//! importer nets generation and load together before the solver sees them
//! (`pgm::accumulate` is the clearest example) — so a direction is a per-bus
//! `(Δp, Δq)` vector, and "load" has to be read off the sign of the net
//! injection unless the caller knows better and supplies its own.

use crate::types::{Bus, BusType};

/// The specified injections at the converged base case, captured once.
///
/// Taken **after** `network::mark_unreferenced_islands` has run and never
/// re-derived from `buses` afterwards. Both halves of that matter:
///
/// - a sourceless island is zeroed to `Slack, V = 0, P = Q = 0` by that pass,
///   so snapshotting afterwards gives those buses a zero direction for free
///   rather than stressing an island with nothing to stress it;
/// - [`ReactiveLimits`](crate::outerloop::ReactiveLimits) rewrites `q_spec` in
///   place when it clamps a generator, so re-deriving the base later would fold
///   a machine's reactive limit into the *load* direction and quietly
///   un-enforce it on the next step.
#[derive(Clone, Debug, PartialEq)]
pub struct BaseSpec {
    pub p: Vec<f64>,
    pub q: Vec<f64>,
}

impl BaseSpec {
    pub fn capture(buses: &[Bus]) -> Self {
        Self {
            p: buses.iter().map(|b| b.p_spec).collect(),
            q: buses.iter().map(|b| b.q_spec).collect(),
        }
    }
}

/// Per-bus `(Δp, Δq)` in per-unit: the injection change per unit of λ.
///
/// `s_spec(λ) = s_base + λ · Δs`, so λ = 0 is the base case and λ = 1 is the
/// base case plus one full direction.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadingDirection {
    /// Indexed by [`Bus::idx`](crate::types::Bus::idx).
    pub d_p: Vec<f64>,
    pub d_q: Vec<f64>,
}

impl LoadingDirection {
    /// The default: `Δp_i = p_spec_i`, `Δq_i = q_spec_i` at every bus that is a
    /// net consumer (`p_spec < 0`), zero elsewhere. Constant power factor,
    /// generation held at schedule.
    ///
    /// The slack picks up both the load increase and the incremental losses.
    /// Its own entry is irrelevant either way — a slack bus contributes no P
    /// equation, so nothing in the augmented system ever reads it — but at high
    /// λ its output can become physically absurd. That is a property of the
    /// scenario, not a defect; [`scale_loads_with_pickup`](Self::scale_loads_with_pickup)
    /// is the realistic alternative.
    pub fn scale_loads(buses: &[Bus]) -> Self {
        let consumer = |b: &Bus| b.bus_type != BusType::Slack && b.p_spec < 0.0;
        Self {
            d_p: buses.iter().map(|b| if consumer(b) { b.p_spec } else { 0.0 }).collect(),
            d_q: buses.iter().map(|b| if consumer(b) { b.q_spec } else { 0.0 }).collect(),
        }
    }

    /// [`scale_loads`](Self::scale_loads) restricted to `subset`: a local or
    /// zonal stress direction rather than a system-wide one.
    pub fn scale_buses(buses: &[Bus], subset: &[usize]) -> Self {
        let mut d = Self::zero(buses.len());
        for &i in subset {
            if i >= buses.len() || buses[i].bus_type == BusType::Slack || buses[i].p_spec >= 0.0 {
                continue;
            }
            d.d_p[i] = buses[i].p_spec;
            d.d_q[i] = buses[i].q_spec;
        }
        d
    }

    /// [`scale_loads`](Self::scale_loads), with the active-power pickup shared
    /// over `weights` instead of dumped on the slack:
    /// `Δp_gen,i = −ŵ_i · Σ_j Δp_load,j`, with `ŵ` normalized over the supplied
    /// weights.
    ///
    /// This is the **analytic** equivalent of running
    /// [`DistributedSlack`](crate::outerloop::DistributedSlack) inside the
    /// corrector, and the analytic form is required rather than merely cheaper.
    /// That loop moves `p_spec` as a function of the *solved state*; such a
    /// dependence is invisible to `∂g/∂λ`, so running it inside the corrector
    /// would silently corrupt the tangent — the one vector continuation needs
    /// exact. Its effect is exactly linear in λ, so folding it into `Δp` is
    /// both faithful and free.
    ///
    /// Weights at slack buses are dropped: a slack has no P equation, so a
    /// share allocated to it would vanish and the pickup would no longer sum to
    /// the load increase.
    pub fn scale_loads_with_pickup(buses: &[Bus], weights: &[f64]) -> Self {
        let mut d = Self::scale_loads(buses);
        let increase: f64 = d.d_p.iter().sum(); // negative: load is a negative injection
        let eligible = |i: usize| i < buses.len() && buses[i].bus_type != BusType::Slack;
        let total: f64 = weights
            .iter()
            .enumerate()
            .filter(|&(i, w)| eligible(i) && *w > 0.0)
            .map(|(_, w)| *w)
            .sum();
        if total <= 0.0 {
            return d;
        }
        for (i, w) in weights.iter().enumerate() {
            if eligible(i) && *w > 0.0 {
                d.d_p[i] -= increase * (w / total);
            }
        }
        d
    }

    /// A study direction from a market or scenario tool, taken as given.
    pub fn explicit(d_p: Vec<f64>, d_q: Vec<f64>) -> Self {
        Self { d_p, d_q }
    }

    /// MATPOWER's base-case-to-target-case form: `Δ = target − base`.
    pub fn to_target(buses: &[Bus], p_target: &[f64], q_target: &[f64]) -> Self {
        Self {
            d_p: buses.iter().map(|b| p_target.get(b.idx).copied().unwrap_or(b.p_spec) - b.p_spec).collect(),
            d_q: buses.iter().map(|b| q_target.get(b.idx).copied().unwrap_or(b.q_spec) - b.q_spec).collect(),
        }
    }

    pub fn zero(n: usize) -> Self {
        Self { d_p: vec![0.0; n], d_q: vec![0.0; n] }
    }

    /// `Σ over net consumers of −Δp_i`: the per-unit *load* increase per unit
    /// of λ, and the multiplier behind the reported margin.
    ///
    /// Counts only the consuming half, so a direction with generation pickup
    /// reports the same margin as the same load ramp without it — the margin is
    /// how much more load the system carries, not how much more the machines
    /// produce.
    pub fn total_active_load_increase(&self) -> f64 {
        self.d_p.iter().filter(|&&dp| dp < 0.0).map(|dp| -dp).sum()
    }

    /// True when nothing moves — a direction that would make continuation a
    /// no-op, and a caller mistake worth refusing rather than looping over.
    pub fn is_empty(&self) -> bool {
        self.d_p.iter().chain(&self.d_q).all(|v| *v == 0.0)
    }

    /// Stop moving this bus's reactive injection.
    ///
    /// Called when [`ReactiveLimits`](crate::outerloop::ReactiveLimits) clamps a
    /// generator: it has just written `q_spec = q_min|q_max`, and the next
    /// `q_spec ← q_base + λ·Δq` write would overwrite that clamp and silently
    /// un-enforce the limit. Zeroing `Δq` and re-basing `q_base` on the clamped
    /// value is what makes the limit stick for the rest of the trace.
    ///
    /// `Δp` is deliberately untouched: under `scale_loads` a generator's `Δp` is
    /// already zero, and under a pickup direction it is that machine's active
    /// share, which a *reactive* limit says nothing about.
    pub(crate) fn freeze_reactive(&mut self, bus: usize, base: &mut BaseSpec, clamped_q: f64) {
        self.d_q[bus] = 0.0;
        base.q[bus] = clamped_q;
    }

    /// Writes `s_spec(λ) = s_base + λ·Δs` into `buses`.
    ///
    /// Applied against the saved base rather than multiplicatively against the
    /// current value, so stepping back and forth along the curve — which the
    /// event locator does on every bisection trial — cannot accumulate drift.
    pub(crate) fn apply(&self, base: &BaseSpec, lambda: f64, buses: &mut [Bus]) {
        for b in buses.iter_mut() {
            b.p_spec = base.p[b.idx] + lambda * self.d_p[b.idx];
            b.q_spec = base.q[b.idx] + lambda * self.d_q[b.idx];
        }
    }
}
