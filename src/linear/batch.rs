//! Batched DC power flow: many scenarios over one topology, one factorization
//! in total.
//!
//! [`batch::BatchSolver`](crate::batch::BatchSolver) can reuse only the
//! *symbolic* half of its factorization across scenarios, because Newton's
//! Jacobian changes numerically at every iteration of every scenario. DC has
//! no such problem: `B` depends on the topology alone, so a bus-injection
//! scenario changes only the right-hand side. That makes batching here a
//! qualitatively cheaper operation, not merely a parallel one.
//!
//! The implementation leans on DC being *exactly* linear rather than
//! re-solving anything:
//!
//! \\[ \theta_s = \theta_{base} + \Delta\theta(\Delta P_s), \qquad
//!    P^{f}_s = P^{f}_{base} + \Delta P^{f}(\Delta P_s) \\]
//!
//! One base solve establishes the phase-shift and reference-angle terms; every
//! scenario is then that base plus [`DcSensitivity::response`] to its own
//! injection delta. Since `response` solves against a factorization built once
//! in [`DcSensitivity::new`], a batch of ten thousand scenarios performs
//! exactly one numeric factorization.
//!
//! This is not an approximation, and
//! `tests/dc_batch_test.rs::batch_matches_a_sequential_loop_exactly` asserts it
//! against a plain loop over [`dc_power_flow`] to 1e-12.
//!
//! # Contingencies
//!
//! [`Scenario::branch_outages`](crate::batch::Scenario::branch_outages) is
//! rejected here, as it is on the AC path — but for the opposite reason. On AC
//! it is unimplemented because an outage gives each scenario its own sparsity
//! pattern. On DC it is *unnecessary*: an outage's effect on flows is
//! [`DcSensitivity::outage_flows`], one solve and no refactorization at all, so
//! folding it into a scenario type designed around per-bus overrides would be
//! a worse API than the one that already exists. See
//! `docs/src/powerflow/dc.md`.

use std::sync::Mutex;

use crate::batch::{BatchError, Scenario};
use crate::types::{Bus, Line, Transformer};

use super::btheta::{dc_branches, dc_power_flow, DcIslandReport, DcSolution};
use super::sensitivity::DcSensitivity;
use super::DcOptions;

/// One scenario's DC answer.
#[derive(Clone, Debug, PartialEq)]
pub struct DcBatchResult {
    /// Bus voltage angles, radians, one per bus.
    pub voltage_ang: Vec<f64>,
    /// Active power entering each branch at its `from` terminal, per-unit,
    /// indexed by the flat branch index — the same space
    /// [`DcSolution::branch_p`] uses.
    pub branch_p: Vec<f64>,
    /// Total active power each island's reference bus(es) supply, per-unit,
    /// in the same order as [`DcBatchSolver::islands`].
    ///
    /// DC is lossless, so this is exactly the negation of everything else the
    /// island injects — computed from the scenario's injections directly
    /// rather than from the solved flows.
    pub slack_pickup: Vec<f64>,
}

/// Solves many DC scenarios over one shared topology.
///
/// ```no_run
/// # use gridoxide::batch::{BusOverride, Scenario};
/// # use gridoxide::linear::{batch::DcBatchSolver, DcOptions};
/// # use gridoxide::types::{Bus, Line, Transformer};
/// # fn example(buses: Vec<Bus>, lines: Vec<Line>, transformers: Vec<Transformer>) {
/// let scenarios: Vec<Scenario> = (0..10_000)
///     .map(|k| Scenario::new(vec![BusOverride::new(3).p(-0.1 * k as f64)]))
///     .collect();
/// let batch = DcBatchSolver::new();
/// let results = batch
///     .solve(&buses, &lines, &transformers, DcOptions::default(), &scenarios)
///     .unwrap();
/// assert_eq!(results.len(), 10_000);
/// # }
/// ```
pub struct DcBatchSolver {
    /// Built once and reused across `solve` calls. `None` uses rayon's global
    /// pool, which `faer` also uses internally — sharing it is what keeps
    /// nested parallelism from oversubscribing the machine.
    pool: Option<rayon::ThreadPool>,
}

impl Default for DcBatchSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DcBatchSolver {
    /// Uses rayon's global thread pool (honors `RAYON_NUM_THREADS`).
    pub fn new() -> Self {
        Self { pool: None }
    }

    /// Uses a dedicated pool of exactly `threads` workers, built once here and
    /// reused by every `solve` call.
    pub fn with_threads(threads: usize) -> Result<Self, BatchError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| BatchError::ThreadPool(e.to_string()))?;
        Ok(Self { pool: Some(pool) })
    }

    pub fn threads(&self) -> usize {
        match &self.pool {
            Some(p) => p.current_num_threads(),
            None => rayon::current_num_threads(),
        }
    }

    /// Solves every scenario, returning one [`DcBatchResult`] each **in
    /// scenario order regardless of thread count**.
    ///
    /// Each scenario starts from `buses_template` with its own overrides
    /// applied, so scenarios never influence one another.
    /// `BusOverride::voltage_mag` is ignored: DC has no magnitude state to
    /// set. `BusOverride::q_spec` is ignored too, for the same reason.
    ///
    /// Errors only for input that cannot be interpreted — an out-of-range bus,
    /// a branch outage, or a network whose reduced `B` is singular. A scenario
    /// cannot "fail to converge", since nothing iterates.
    pub fn solve(
        &self,
        buses_template: &[Bus],
        lines: &[Line],
        transformers: &[Transformer],
        opts: DcOptions,
        scenarios: &[Scenario],
    ) -> Result<Vec<DcBatchResult>, BatchError> {
        for (i, sc) in scenarios.iter().enumerate() {
            if !sc.branch_outages.is_empty() {
                return Err(BatchError::OutagesUnsupported { scenario: i });
            }
            for ov in &sc.bus_overrides {
                if ov.bus >= buses_template.len() {
                    return Err(BatchError::BusOutOfRange {
                        scenario: i,
                        bus: ov.bus,
                        n_buses: buses_template.len(),
                    });
                }
            }
        }
        if scenarios.is_empty() {
            return Ok(Vec::new());
        }

        let prepared = Prepared::new(buses_template, lines, transformers, opts)
            .ok_or(BatchError::DcSingular)?;

        let collected: Mutex<Vec<(usize, DcBatchResult)>> =
            Mutex::new(Vec::with_capacity(scenarios.len()));
        let run = || {
            use rayon::prelude::*;
            scenarios
                .par_iter()
                .enumerate()
                .for_each(|(i, sc)| {
                    let result = prepared.solve_one(sc);
                    collected.lock().expect("batch collector poisoned").push((i, result));
                });
        };
        match &self.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }

        let mut collected = collected.into_inner().expect("batch collector poisoned");
        collected.sort_by_key(|(i, _)| *i);
        Ok(collected.into_iter().map(|(_, r)| r).collect())
    }

    /// The base solve's per-island breakdown, in the order
    /// [`DcBatchResult::slack_pickup`] indexes.
    ///
    /// Computed from the template, so it is the same for every scenario —
    /// island membership depends on topology, which a bus override cannot
    /// change.
    pub fn islands(
        buses_template: &[Bus],
        lines: &[Line],
        transformers: &[Transformer],
        opts: DcOptions,
    ) -> Vec<DcIslandReport> {
        let mut scratch = buses_template.to_vec();
        dc_power_flow(&mut scratch, lines, transformers, opts).islands
    }
}

/// The base solve plus the one factorization every scenario shares.
struct Prepared {
    base_ang: Vec<f64>,
    base: DcSolution,
    sensitivity: DcSensitivity,
    /// Per-bus injection the base solve used, so a scenario's delta can be
    /// formed without re-deriving it.
    base_injection: Vec<f64>,
    /// Per-bus sum of the template's ZIP terms. A `BusOverride` replaces
    /// `p_spec` alone, so a scenario's injection is its override plus this —
    /// ZIP terms belong to the template and survive an override, exactly as
    /// they do on the AC path.
    zip_sum: Vec<f64>,
    /// Per island, the non-slack member buses — all that
    /// [`DcBatchResult::slack_pickup`] needs, since DC is lossless.
    island_members: Vec<Vec<usize>>,
}

impl Prepared {
    fn new(
        buses_template: &[Bus],
        lines: &[Line],
        transformers: &[Transformer],
        opts: DcOptions,
    ) -> Option<Self> {
        let mut scratch = buses_template.to_vec();
        let base = dc_power_flow(&mut scratch, lines, transformers, opts);
        let base_ang: Vec<f64> = scratch.iter().map(|b| b.voltage_ang).collect();

        let branches = dc_branches(lines, transformers, opts);
        let n_branches = lines.len() + transformers.len();
        // Built from the *marked* buses `dc_power_flow` left behind, so an
        // unreferenced island is excluded from both consistently.
        let sensitivity = DcSensitivity::new(&scratch, &branches, n_branches)?;

        let zip_sum: Vec<f64> = buses_template.iter().map(zip_total).collect();
        let base_injection: Vec<f64> =
            buses_template.iter().zip(&zip_sum).map(|(b, z)| b.p_spec + z).collect();
        let island_members = base
            .islands
            .iter()
            .map(|island| {
                island
                    .bus_indices
                    .iter()
                    .copied()
                    .filter(|&i| !island.slack_indices.contains(&i))
                    .collect()
            })
            .collect();

        Some(Self { base_ang, base, sensitivity, base_injection, zip_sum, island_members })
    }

    fn solve_one(&self, scenario: &Scenario) -> DcBatchResult {
        // The injection delta is all DC responds to. A delta at a slack bus is
        // harmless: `response` builds its right-hand side from each island's
        // *unknowns*, so a reference's injection is never read — which is
        // correct, since a slack's injection is an output of the solve.
        let mut delta = vec![0.0; self.base_injection.len()];
        let mut scenario_injection = self.base_injection.clone();
        for ov in &scenario.bus_overrides {
            if let Some(p) = ov.p_spec {
                let new = p + self.zip_sum[ov.bus];
                delta[ov.bus] = new - self.base_injection[ov.bus];
                scenario_injection[ov.bus] = new;
            }
        }

        let (d_ang, d_flow) = self
            .sensitivity
            .response(&delta)
            .expect("delta is one entry per bus and the factorization is non-singular");

        let voltage_ang: Vec<f64> =
            self.base_ang.iter().zip(&d_ang).map(|(a, d)| a + d).collect();
        let branch_p: Vec<f64> =
            self.base.branch_p.iter().zip(&d_flow).map(|(p, d)| p + d).collect();
        // Lossless, so each island's reference supplies exactly the negation
        // of everything else in it.
        let slack_pickup = self
            .island_members
            .iter()
            .map(|members| -members.iter().map(|&i| scenario_injection[i]).sum::<f64>())
            .collect();

        DcBatchResult { voltage_ang, branch_p, slack_pickup }
    }
}

/// A bus's ZIP terms summed as active power. Every `ZipKind` collapses to its
/// own `s_const` at the DC assumption `|V| = 1`, which is the same reduction
/// `btheta::dc_injection` performs (private to that module).
fn zip_total(bus: &Bus) -> f64 {
    bus.zip_terms.iter().map(|z| z.s_const.re).sum()
}
