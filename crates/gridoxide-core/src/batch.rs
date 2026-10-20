//! Batched power flow: one topology, many scenarios, solved across all cores.
//!
//! This is the workload where parallelism actually pays. A single solve is
//! ~60% sparse LU, which neither threads nor a
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
//! [`BatchSolver::solve_contingencies`] extends this to **branch outages**,
//! which change the network rather than its bus values. The first property
//! above appears to rule that out — and it was long assumed to — but only the
//! *sparsity pattern* has to hold still, not the Y-bus itself:
//! [`network::build_ybus_with_outages`](crate::network::build_ybus_with_outages)
//! takes a branch out while keeping its structural entries, so the symbolic
//! factorization still carries across the sweep. What cannot carry is the
//! Jacobian's cached per-nonzero recipe, which stores the admittance it was
//! analyzed against; that is re-derived per scenario via
//! [`PersistentSolver::invalidate_admittances`], at O(nonzeros) rather than a
//! fresh symbolic analysis. Measured at 2.0–2.7x against independent solves,
//! single-threaded, before any parallelism.
//!
//! This is also the CPU baseline any future GPU work has to beat: beating a
//! single-threaded CPU solver is not a result.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::PowerFlowReport;
use crate::network::{
    ShuntAdm, YBusSparse, build_ybus, build_ybus_with_outages, linear_initial_guess, stamp_shunts,
    structural_component_count,
};
pub use crate::scenario::{BatchError, BusOverride, Scenario, uniform_load_scaling};
use crate::solver::{JacobianBackend, PersistentSolver};
use crate::types::{Bus, Line, Transformer};

/// Solves many scenarios over one shared topology, in parallel.
///
/// ```no_run
/// # use gridoxide_core::batch::{BatchSolver, Scenario, BusOverride};
/// # use gridoxide_core::solver::JacobianBackend;
/// # use gridoxide_core::network::YBusSparse;
/// # use gridoxide_core::types::Bus;
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
        Self {
            backend,
            pool: None,
        }
    }

    /// Uses a dedicated pool of exactly `threads` workers. Built once here
    /// and reused by every `solve` call, so repeated batches never pay
    /// thread-creation cost.
    pub fn with_threads(backend: JacobianBackend, threads: usize) -> Result<Self, BatchError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| BatchError::ThreadPool(e.to_string()))?;
        Ok(Self {
            backend,
            pool: Some(pool),
        })
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
                                .push((
                                    i,
                                    PowerFlowReport {
                                        buses,
                                        islands,
                                        stats,
                                        dc: None,
                                        linear: None,
                                        outer: None,
                                    },
                                ));
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

    /// Solves every scenario **honoring [`Scenario::branch_outages`]**,
    /// returning one [`PowerFlowReport`] each in scenario order.
    ///
    /// Takes branch lists rather than a finished Y-bus, because an outage
    /// changes the network rather than its bus values, and a Y-bus has already
    /// summed the branches away.
    ///
    /// # Two paths, and why
    ///
    /// Most contingencies leave the network connected. Those take the fast
    /// path: [`network::build_ybus_with_outages`](crate::network::build_ybus_with_outages)
    /// rebuilds the Y-bus with the outaged branches contributing nothing but
    /// **keeping their structural entries**, so the sparsity pattern — and
    /// therefore `jacobian::JacobianPattern` and the symbolic factorization
    /// behind it — is bit-for-bit the intact network's. A worker reuses one
    /// [`PersistentSolver`] across every such scenario it handles, exactly as
    /// [`solve`](Self::solve) does for injection-only batches. That is what
    /// this method exists to make possible, and it is why the old
    /// "an outage forces its own sparsity pattern" objection turns out to be
    /// avoidable in the common case.
    ///
    /// A contingency that *splits* the network cannot use that trick:
    /// [`connected_components`](crate::network::connected_components) reads the
    /// same structural entries and cannot tell a zero-valued one from a live
    /// one, so it would report the severed buses as still connected and their
    /// now-referenceless rows would come back as a singular solve instead of an
    /// honest `NoReferenceBus`. Those scenarios are detected up front with
    /// [`structural_component_count`]
    /// — removing branches can only *increase* the component count, so a count
    /// above the intact network's is exactly the "something split" condition —
    /// and take a slow path: the outaged branches are dropped outright and the
    /// worker's solver is reset around that scenario, since its pattern
    /// genuinely differs.
    ///
    /// A scenario that fails to converge is not an error: its report carries
    /// the failure and the rest of the batch is unaffected. Islanding into a
    /// sourceless region is not a failure at all — that island comes back
    /// `IslandStatus::NoReferenceBus`, which for N-1 screening is a result.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_contingencies(
        &self,
        buses_template: &[Bus],
        lines: &[Line],
        transformers: &[Transformer],
        shunts: &[ShuntAdm],
        scenarios: &[Scenario],
        tol: f64,
        max_iter: usize,
    ) -> Result<Vec<PowerFlowReport>, BatchError> {
        let n_branches = lines.len() + transformers.len();
        for (i, sc) in scenarios.iter().enumerate() {
            for &b in &sc.branch_outages {
                if b >= n_branches {
                    return Err(BatchError::BranchOutOfRange {
                        scenario: i,
                        branch: b,
                        n_branches,
                    });
                }
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

        let n = buses_template.len();
        let intact_components = structural_component_count(n, lines, transformers, &[]);

        // Decided once, up front, so workers never repeat the analysis: which
        // branches are out, and whether taking them out split the network.
        let plans: Vec<(Vec<bool>, bool)> = scenarios
            .iter()
            .map(|sc| {
                let mut outaged = vec![false; n_branches];
                for &b in &sc.branch_outages {
                    outaged[b] = true;
                }
                let splits = !sc.branch_outages.is_empty()
                    && structural_component_count(n, lines, transformers, &outaged)
                        != intact_components;
                (outaged, splits)
            })
            .collect();

        let next = AtomicUsize::new(0);
        let collected: Mutex<Vec<(usize, PowerFlowReport)>> =
            Mutex::new(Vec::with_capacity(scenarios.len()));
        let n_workers = self.threads().max(1).min(scenarios.len());
        let backend = self.backend;

        let run = || {
            rayon::scope(|s| {
                for _ in 0..n_workers {
                    s.spawn(|_| {
                        // Inside the closure for the same reason as in
                        // `solve`: it is what keeps the `!Send` FFI backends
                        // usable here.
                        let mut solver = PersistentSolver::new(backend);

                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= scenarios.len() {
                                break;
                            }
                            let (outaged, splits) = &plans[i];

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

                            let mut ybus = if *splits {
                                let keep_lines: Vec<Line> = lines
                                    .iter()
                                    .enumerate()
                                    .filter(|(k, _)| !outaged[*k])
                                    .map(|(_, l)| l.clone())
                                    .collect();
                                let keep_transformers: Vec<Transformer> = transformers
                                    .iter()
                                    .enumerate()
                                    .filter(|(k, _)| !outaged[lines.len() + k])
                                    .map(|(_, t)| t.clone())
                                    .collect();
                                // This scenario's pattern differs from the one
                                // the worker has been reusing.
                                solver.reset();
                                build_ybus(n, &keep_lines, &keep_transformers)
                            } else {
                                // Same pattern, different values: the symbolic
                                // factorization survives, the Jacobian's cached
                                // admittances do not.
                                solver.invalidate_admittances();
                                build_ybus_with_outages(n, lines, transformers, outaged)
                            };
                            stamp_shunts(&mut ybus, shunts);
                            let ybus = ybus.finish();

                            linear_initial_guess(&mut buses, &ybus);
                            let (islands, stats) =
                                solver.solve_with_stats(&mut buses, &ybus, tol, max_iter);
                            if *splits {
                                // ...and must not leak into the next one.
                                solver.reset();
                            }

                            collected
                                .lock()
                                .expect("batch result mutex poisoned by a panicking worker")
                                .push((
                                    i,
                                    PowerFlowReport {
                                        buses,
                                        islands,
                                        stats,
                                        dc: None,
                                        linear: None,
                                        outer: None,
                                    },
                                ));
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
