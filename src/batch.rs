//! Batched power flow: one topology, many scenarios, solved across all cores.
//!
//! This is the workload where parallelism actually pays. A single solve is
//! ~60% sparse LU (see `plans/GPU_PLAN.md` §1), which neither threads nor a
//! GPU help with; but N-1 screening, time series/QSTS and Monte Carlo run
//! *thousands* of independent solves over one unchanging topology, and those
//! scale essentially linearly.
//!
//! Two properties make that cheap here:
//!
//! - Every scenario shares one [`YBusSparse`], so the Jacobian's sparsity
//!   pattern is identical across the whole batch. Each worker keeps one
//!   [`PersistentSolver`] and amortizes a single symbolic factorization
//!   across its entire share of the batch — measured at ~45% of solve time
//!   on a 9,241-bus case when redone per solve.
//! - Scenarios are fully independent, so the only coordination is handing
//!   out indices.
//!
//! This is also the CPU baseline any future GPU work has to beat.
//! `plans/GPU_PLAN.md` §6 is explicit that beating a single-threaded CPU
//! solver is not a result.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::network::{linear_initial_guess, YBusSparse};
use crate::solver::{JacobianBackend, PersistentSolver};
use crate::types::Bus;
use crate::PowerFlowReport;

/// A per-bus change applied on top of the batch's shared bus template.
/// `None` fields leave the template's value alone.
///
/// Deliberately cannot change `bus_type`: switching `PV`↔`PQ` changes
/// `n_unknowns`, which invalidates the cached symbolic factorization that
/// makes batching worth doing at all. A caller that needs per-scenario bus
/// types wants separate batches, one per type assignment.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BusOverride {
    pub bus: usize,
    pub p_spec: Option<f64>,
    pub q_spec: Option<f64>,
    /// Voltage setpoint for a `Slack`/`PV` bus. Ignored for `PQ` buses,
    /// whose magnitude is an unknown the solver overwrites.
    pub voltage_mag: Option<f64>,
}

impl BusOverride {
    pub fn new(bus: usize) -> Self {
        Self { bus, ..Default::default() }
    }

    pub fn p(mut self, p_spec: f64) -> Self {
        self.p_spec = Some(p_spec);
        self
    }

    pub fn q(mut self, q_spec: f64) -> Self {
        self.q_spec = Some(q_spec);
        self
    }

    pub fn vm(mut self, voltage_mag: f64) -> Self {
        self.voltage_mag = Some(voltage_mag);
        self
    }
}

/// One scenario in a batch: a set of bus-value overrides against the shared
/// template.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scenario {
    pub bus_overrides: Vec<BusOverride>,
    /// Branch indices to take out of service.
    ///
    /// **Not implemented yet** — a non-empty list is rejected with
    /// [`BatchError::OutagesUnsupported`] rather than silently ignored. The
    /// field exists so N-1 contingency support can land additively: it is
    /// the one thing that forces a per-scenario Y-bus (and therefore a
    /// per-scenario sparsity pattern), which is a materially different
    /// design from the shared-pattern fast path above.
    pub branch_outages: Vec<usize>,
}

impl Scenario {
    pub fn new(bus_overrides: Vec<BusOverride>) -> Self {
        Self { bus_overrides, branch_outages: Vec::new() }
    }
}

/// Builds a scenario scaling every bus's `p_spec`/`q_spec` by `factor` —
/// the standard time-series/QSTS and Monte Carlo shape, and what
/// `scripts/bench/bench_batch.py` drives.
///
/// Buses with zero injection stay zero, so this touches only buses that
/// actually carry load or generation.
pub fn uniform_load_scaling(buses_template: &[Bus], factor: f64) -> Scenario {
    let overrides = buses_template
        .iter()
        .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
        .map(|b| BusOverride::new(b.idx).p(b.p_spec * factor).q(b.q_spec * factor))
        .collect();
    Scenario::new(overrides)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchError {
    /// `Scenario::branch_outages` was non-empty. See that field's doc comment.
    OutagesUnsupported { scenario: usize },
    /// A `BusOverride::bus` was not a valid index into the bus template.
    BusOutOfRange { scenario: usize, bus: usize, n_buses: usize },
    /// `rayon` refused to build the requested thread pool.
    ThreadPool(String),
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchError::OutagesUnsupported { scenario } => write!(
                f,
                "scenario {scenario} requests branch outages, which are not implemented yet \
                 (see batch::Scenario::branch_outages)"
            ),
            BatchError::BusOutOfRange { scenario, bus, n_buses } => write!(
                f,
                "scenario {scenario} overrides bus {bus}, but the template has only {n_buses} bus(es)"
            ),
            BatchError::ThreadPool(e) => write!(f, "building the rayon thread pool failed: {e}"),
        }
    }
}

impl std::error::Error for BatchError {}

/// Solves many scenarios over one shared topology, in parallel.
///
/// ```no_run
/// # use gridoxide::batch::{BatchSolver, Scenario, BusOverride};
/// # use gridoxide::solver::JacobianBackend;
/// # use gridoxide::network::YBusSparse;
/// # use gridoxide::types::Bus;
/// # fn example(buses: Vec<Bus>, ybus: &YBusSparse) {
/// let scenarios: Vec<Scenario> = (0..256)
///     .map(|k| Scenario::new(vec![BusOverride::new(3).p(-0.1 * k as f64)]))
///     .collect();
/// let batch = BatchSolver::new(JacobianBackend::KluNative);
/// let reports = batch.solve(&buses, ybus, &scenarios, 1e-6, 20).unwrap();
/// assert_eq!(reports.len(), 256);
/// # }
/// ```
pub struct BatchSolver {
    backend: JacobianBackend,
    /// Built once and reused across `solve` calls. `None` uses rayon's
    /// global pool, which is also what `faer` uses internally — sharing it
    /// is what prevents nested parallelism from oversubscribing the machine.
    pool: Option<rayon::ThreadPool>,
}

impl BatchSolver {
    /// Uses rayon's global thread pool (honors `RAYON_NUM_THREADS`).
    pub fn new(backend: JacobianBackend) -> Self {
        Self { backend, pool: None }
    }

    /// Uses a dedicated pool of exactly `threads` workers. Built once here
    /// and reused by every `solve` call, so repeated batches never pay
    /// thread-creation cost.
    pub fn with_threads(backend: JacobianBackend, threads: usize) -> Result<Self, BatchError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| BatchError::ThreadPool(e.to_string()))?;
        Ok(Self { backend, pool: Some(pool) })
    }

    /// Workers this solver will use for a batch.
    pub fn threads(&self) -> usize {
        match &self.pool {
            Some(p) => p.current_num_threads(),
            None => rayon::current_num_threads(),
        }
    }

    /// Solves every scenario and returns one [`PowerFlowReport`] each, **in
    /// scenario order regardless of thread count**.
    ///
    /// Each scenario starts from `buses_template` with its overrides applied
    /// and a fresh [`linear_initial_guess`], so scenarios never influence
    /// each other's starting point — results are identical to a sequential
    /// loop over `PersistentSolver::solve`, which `tests/batch_solver_test.rs`
    /// asserts directly.
    ///
    /// A scenario that fails to converge is *not* an error: its report
    /// carries `SolveStatus::MaxIterationsReached`/`Singular` and the rest of
    /// the batch is unaffected. Divergent contingency scenarios are normal
    /// and must not poison the batch.
    pub fn solve(
        &self,
        buses_template: &[Bus],
        ybus: &YBusSparse,
        scenarios: &[Scenario],
        tol: f64,
        max_iter: usize,
    ) -> Result<Vec<PowerFlowReport>, BatchError> {
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

        let next = AtomicUsize::new(0);
        let collected: Mutex<Vec<(usize, PowerFlowReport)>> =
            Mutex::new(Vec::with_capacity(scenarios.len()));
        let n_workers = self.threads().max(1).min(scenarios.len());
        let backend = self.backend;

        let run = || {
            rayon::scope(|s| {
                for _ in 0..n_workers {
                    s.spawn(|_| {
                        // Constructed *inside* the spawned closure, so it
                        // never crosses a thread boundary. This is what
                        // keeps the `!Send` FFI backends usable here:
                        // `KluRealSystem` owns raw `*mut klu_symbolic` and
                        // `PardisoRealSystem` a `[*mut c_void; 64]`, so a
                        // `PersistentSolver` holding either is not `Send`.
                        // `rayon::scope`/`spawn` only require the *closure*
                        // to be `Send`; work-stealing steals unstarted
                        // tasks and never migrates one mid-execution, so
                        // this local stays on one thread for its lifetime.
                        // (`rayon`'s `map_init` would demand `T: Send` and
                        // silently exclude those two backends.)
                        let mut solver = PersistentSolver::new(backend);

                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= scenarios.len() {
                                break;
                            }

                            let mut buses = buses_template.to_vec();
                            for ov in &scenarios[i].bus_overrides {
                                let b = &mut buses[ov.bus];
                                if let Some(p) = ov.p_spec {
                                    b.p_spec = p;
                                }
                                if let Some(q) = ov.q_spec {
                                    b.q_spec = q;
                                }
                                if let Some(vm) = ov.voltage_mag {
                                    b.voltage_mag = vm;
                                }
                            }

                            linear_initial_guess(&mut buses, ybus);
                            let (islands, stats) =
                                solver.solve_with_stats(&mut buses, ybus, tol, max_iter);
                            collected
                                .lock()
                                .expect("batch result mutex poisoned by a panicking worker")
                                .push((i, PowerFlowReport { buses, islands, stats }));
                        }
                    });
                }
            });
        };

        match &self.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }

        let mut out = collected.into_inner().expect("batch result mutex poisoned");
        out.sort_by_key(|(i, _)| *i);
        Ok(out.into_iter().map(|(_, r)| r).collect())
    }
}
