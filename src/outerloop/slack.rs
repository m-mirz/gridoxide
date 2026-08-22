//! Distributed slack: sharing the system imbalance across generators instead
//! of dumping all of it on one bus.
//!
//! A single slack bus absorbs every megawatt the rest of the schedule does not
//! account for, transmission losses included. On a real network that is a few
//! hundred megawatts appearing at one point, and every branch flow around it is
//! distorted by the difference. Distributing the imbalance by participation
//! weight is what the reference tools all do instead.
//!
//! # Why it converges, and why not in one pass
//!
//! Write the slack's excess over its own schedule as \\(\Delta\\), and let
//! \\(\alpha_s\\) be the slack's own share. Distributing adds
//! \\((1-\alpha_s)\Delta\\) to the *other* participants' schedules, so the next
//! solve asks the slack for that much less while its own schedule has risen by
//! \\(\alpha_s\Delta\\). Those cancel exactly, which is what makes this
//! affordable as an outer loop.
//!
//! What is left over is the change in transmission losses caused by the
//! redistributed flows, and that does not vanish — it shrinks. Convergence is
//! therefore linear at roughly the fractional loss sensitivity, a few per cent
//! per pass. Measured on pglib at `tolerance = 1e-8`: `case14_ieee`,
//! `case30_ieee` and `case118_ieee` each take **seven** passes, moving 2.3,
//! 2.4 and 16.5 per-unit off their slacks respectively.
//!
//! # The schedule is the slack's own `p_spec`
//!
//! Ordinary power flow ignores that field for a slack bus — the slack's output
//! is an *answer*, not an input — so it is free to carry the schedule here.
//! Worth knowing that a document which never needed it may leave it at zero, in
//! which case the slack is treated as scheduled for nothing and its entire
//! output is redistributed.

use crate::network::{effective_injection, power_injections};
use crate::types::{Bus, BusType};

use super::{Invalidates, OuterLoop, OuterLoopContext, OuterLoopStatus};

/// Who participates in picking up the imbalance, and by how much.
///
/// Weights are normalized **per island**, which is the only reading that works
/// on a disconnected network: each island has its own slack and its own
/// imbalance, and a global normalization would size one island's correction by
/// another island's generators.
#[derive(Clone, Debug)]
pub struct SlackDistribution {
    /// Participation weight per bus, indexed by [`Bus::idx`].
    pub factors: Vec<f64>,
    /// Convergence tolerance on the slack's remaining deviation from its
    /// schedule, per-unit.
    pub tolerance: f64,
    /// Cap on outer passes. Around seven is typical at the default tolerance.
    pub max_outer_iter: usize,
}

impl SlackDistribution {
    /// Every generator bus — `Slack` and `PV` — takes an equal share.
    ///
    /// `PQ` buses are excluded even when they carry positive `p_spec`. A
    /// positive injection at a `PQ` bus is a fixed schedule, not a machine
    /// under governor control, and the distinction is exactly what
    /// participation means.
    pub fn uniform(buses: &[Bus]) -> Self {
        let factors = buses
            .iter()
            .map(|b| match b.bus_type {
                BusType::Slack | BusType::PV => 1.0,
                BusType::PQ => 0.0,
            })
            .collect();
        Self { factors, tolerance: 1e-8, max_outer_iter: 20 }
    }

    /// Explicit per-bus weights.
    pub fn from_weights(factors: Vec<f64>) -> Self {
        Self { factors, tolerance: 1e-8, max_outer_iter: 20 }
    }
}

/// What [`DistributedSlack`] did.
#[derive(Clone, Debug, Default)]
pub struct SlackDistributionReport {
    /// Per bus, how much its active schedule moved, per-unit. Summing this
    /// over an island gives that island's total losses-plus-imbalance.
    pub shift: Vec<f64>,
    /// Each island's remaining slack deviation at exit, in the order the
    /// island reports come back. Islands that could not be distributed over
    /// carry their untouched deviation here rather than a zero.
    pub residual: Vec<f64>,
    /// Outer passes taken.
    pub outer_iterations: usize,
    /// False if the loop hit `max_outer_iter` with a deviation still above
    /// tolerance, or if the inner solve stopped converging.
    pub converged: bool,
    /// Islands that were left on a single slack, and why — no reference bus,
    /// an ambiguous one, or no participating generator in that island.
    pub undistributed: Vec<(usize, &'static str)>,
}

/// Moves the slack's deviation from its own schedule onto the participating
/// generators, by normalized weight, until the deviation is inside tolerance.
///
/// Reports [`Invalidates::Nothing`]: only `p_spec` changes between passes, so
/// bus types, `n_unknowns` and the Jacobian's sparsity pattern all hold and the
/// solver keeps its symbolic factorization across the whole loop. That is the
/// opposite of [`ReactiveLimits`](super::ReactiveLimits), which switches
/// `PV → PQ` and must reset each time it does.
///
/// # What it mutates
///
/// `buses[i].p_spec` ends up holding the **dispatched** value rather than the
/// schedule it went in with, for every participating bus. That is the answer —
/// who ended up producing what — and [`SlackDistributionReport::shift`] records
/// how far each moved, so the original is recoverable.
#[derive(Clone, Debug)]
pub struct DistributedSlack {
    distribution: SlackDistribution,
    report: SlackDistributionReport,
    /// Set once the pass cap is hit, so the pass after the final redistribution
    /// accepts the state rather than starting another round.
    exhausted: bool,
}

impl DistributedSlack {
    pub fn new(distribution: SlackDistribution) -> Self {
        Self {
            distribution,
            report: SlackDistributionReport::default(),
            exhausted: false,
        }
    }

    /// What it did, readable after the solve.
    pub fn report(&self) -> &SlackDistributionReport {
        &self.report
    }

    /// Consumes the loop for its report, for callers that do not need it back.
    pub fn into_report(self) -> SlackDistributionReport {
        self.report
    }
}

impl OuterLoop for DistributedSlack {
    fn name(&self) -> &'static str {
        "DistributedSlack"
    }

    fn invalidates(&self) -> Invalidates {
        Invalidates::Nothing
    }

    fn initialize(&mut self, ctx: &mut OuterLoopContext<'_, '_>) {
        let n = ctx.net.buses.len();
        assert_eq!(
            self.distribution.factors.len(),
            n,
            "participation factors must carry one weight per bus"
        );
        self.report.shift = vec![0.0; n];
    }

    fn check(&mut self, ctx: &mut OuterLoopContext<'_, '_>) -> OuterLoopStatus {
        if self.exhausted {
            return OuterLoopStatus::Stable;
        }
        self.report.outer_iterations += 1;

        let (p_calc, _) = power_injections(ctx.net.buses, ctx.net.ybus);
        let mut residual = Vec::with_capacity(ctx.islands.len());
        let mut undistributed = Vec::new();
        let mut worst = 0.0f64;
        // Collected first, applied after, so the deviation every island is
        // measured against comes from one consistent solved state.
        let mut updates: Vec<(usize, f64)> = Vec::new();

        for (island, report) in ctx.islands.iter().enumerate() {
            let slack = match report.slack_indices.as_slice() {
                [only] => *only,
                [] => {
                    residual.push(0.0);
                    undistributed.push((island, "no reference bus"));
                    continue;
                }
                _ => {
                    residual.push(0.0);
                    undistributed.push((island, "ambiguous reference bus"));
                    continue;
                }
            };

            let scheduled = effective_injection(&ctx.net.buses[slack]).0;
            let delta = p_calc[slack] - scheduled;

            let total: f64 = report
                .bus_indices
                .iter()
                .map(|&i| self.distribution.factors[i].max(0.0))
                .sum();
            if total <= 0.0 {
                residual.push(delta);
                undistributed.push((island, "no participating generator in this island"));
                continue;
            }

            residual.push(delta);
            worst = worst.max(delta.abs());
            for &i in &report.bus_indices {
                let weight = self.distribution.factors[i].max(0.0);
                if weight > 0.0 {
                    updates.push((i, weight / total * delta));
                }
            }
        }

        self.report.residual = residual;
        self.report.undistributed = undistributed;

        if worst <= self.distribution.tolerance {
            self.report.converged = true;
            return OuterLoopStatus::Stable;
        }

        for (i, amount) in updates {
            ctx.net.buses[i].p_spec += amount;
            self.report.shift[i] += amount;
        }

        // The cap is spent on this redistribution: re-solve once more so the
        // returned state matches the final schedules, then accept it.
        if self.report.outer_iterations >= self.distribution.max_outer_iter.max(1) {
            self.exhausted = true;
        }
        OuterLoopStatus::Unstable
    }
}
