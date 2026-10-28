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

use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::PowerFlowReport;
use num_complex::Complex;

use crate::network::{
    ShuntAdm, YBusSparse, branch_stamp, build_ybus, build_ybus_with_outages, linear_initial_guess,
    stamp_shunts, structural_component_count,
};
pub use crate::scenario::{BatchError, BusOverride, Scenario, uniform_load_scaling};
use crate::solver::{IslandReport, JacobianBackend, PersistentSolver, SolveStats};
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
    start: BatchStart,
    /// Distinguishes this solver's cached worker solvers from another's.
    id: u64,
}

/// One scenario's result from [`BatchSolver::solve_voltages`].
#[derive(Clone, Debug)]
pub struct ScenarioVoltages {
    /// Per bus, per unit.
    pub voltage_mag: Vec<f64>,
    /// Per bus, radians.
    pub voltage_ang: Vec<f64>,
    pub stats: SolveStats,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// A dedicated pool's worker solver, kept between [`BatchSolver::solve`]
    /// calls with the key it was built for: `(BatchSolver::id, network
    /// fingerprint)`. Setting a worker's solver up (the network's
    /// components, the Jacobian's pattern, the symbolic factorization) costs
    /// about three scenarios' solves on case1354pegase, and every call paid
    /// it again on every worker. Thread-local, so a `!Send` backend never
    /// crosses a thread; taken out while in use, so a task rayon runs on
    /// this thread while one waits finds none and builds its own.
    static WORKER_SOLVER: RefCell<Option<((u64, u64), PersistentSolver)>> =
        const { RefCell::new(None) };
}

/// What a cached worker solver depends on: the Y-bus, structure and values
/// (the Jacobian's recipe carries the admittances), and the bus types,
/// which decide the Jacobian's unknowns.
fn network_fingerprint(buses: &[Bus], ybus: &YBusSparse) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ybus.n().hash(&mut h);
    for i in 0..ybus.n() {
        for &(j, y) in ybus.row(i) {
            (j, y.re.to_bits(), y.im.to_bits()).hash(&mut h);
        }
    }
    for b in buses {
        std::mem::discriminant(&b.bus_type).hash(&mut h);
    }
    h.finish()
}

/// Where every scenario of a batch starts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchStart {
    /// A fresh [`linear_initial_guess`] on the scenario's own buses.
    #[default]
    Linear,
    /// The template's voltages as given, overrides applied: pass a solved
    /// base case as the template to start every scenario from its solution,
    /// as a sweep or N-1 study from a solved operating point is run.
    Template,
}

impl BatchSolver {
    /// Uses rayon's global thread pool (honors `RAYON_NUM_THREADS`).
    pub fn new(backend: JacobianBackend) -> Self {
        Self {
            backend,
            pool: None,
            start: BatchStart::default(),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Sets where every scenario of later batches starts.
    pub fn set_start(&mut self, start: BatchStart) {
        self.start = start;
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
            start: BatchStart::default(),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
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
    /// and, under [`BatchStart::Linear`], a fresh [`linear_initial_guess`];
    /// under [`BatchStart::Template`], from the template's own voltages.
    /// Either way scenarios never influence each other's starting point —
    /// results are identical to a sequential loop over
    /// `PersistentSolver::solve`, which `tests/batch_solver_test.rs` asserts
    /// directly.
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
        self.solve_each(
            buses_template,
            ybus,
            scenarios,
            tol,
            max_iter,
            |buses, islands, stats| PowerFlowReport {
                buses: std::mem::take(buses),
                islands,
                stats,
                dc: None,
                linear: None,
                outer: None,
            },
        )
    }

    /// [`solve`](Self::solve), keeping only each scenario's voltages and
    /// convergence: what a sweep reads.
    ///
    /// Each worker solves its scenarios in one reused bus buffer, where
    /// `solve` hands every scenario's whole bus vector to its report. Those
    /// are what a batch allocates most of (13.5 MB for 100 scenarios of
    /// case1354pegase), which glibc returns to the system after each batch
    /// and faults back in on the next: a tenth of a single-threaded sweep.
    pub fn solve_voltages(
        &self,
        buses_template: &[Bus],
        ybus: &YBusSparse,
        scenarios: &[Scenario],
        tol: f64,
        max_iter: usize,
    ) -> Result<Vec<ScenarioVoltages>, BatchError> {
        self.solve_each(
            buses_template,
            ybus,
            scenarios,
            tol,
            max_iter,
            |buses, _, stats| ScenarioVoltages {
                voltage_mag: buses.iter().map(|b| b.voltage_mag).collect(),
                voltage_ang: buses.iter().map(|b| b.voltage_ang).collect(),
                stats,
            },
        )
    }

    /// Every scenario, solved in a worker's bus buffer and handed to `keep`,
    /// which makes the scenario's result; in scenario order.
    fn solve_each<R: Send>(
        &self,
        buses_template: &[Bus],
        ybus: &YBusSparse,
        scenarios: &[Scenario],
        tol: f64,
        max_iter: usize,
        keep: impl Fn(&mut Vec<Bus>, Vec<IslandReport>, SolveStats) -> R + Sync,
    ) -> Result<Vec<R>, BatchError> {
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
        let collected: Mutex<Vec<(usize, R)>> = Mutex::new(Vec::with_capacity(scenarios.len()));
        let n_workers = self.threads().max(1).min(scenarios.len());
        let backend = self.backend;
        let start = self.start;
        // Only a dedicated pool's threads keep a solver between calls: the
        // global pool's are everyone's, and would hold it indefinitely.
        let key = self
            .pool
            .is_some()
            .then(|| (self.id, network_fingerprint(buses_template, ybus)));

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
                        //
                        // Or taken from this thread's cache, when it was
                        // built for this solver and this network.
                        let cached = key.and_then(|key| {
                            WORKER_SOLVER
                                .with(|slot| slot.borrow_mut().take())
                                .filter(|(k, _)| *k == key)
                        });
                        let mut solver = cached
                            .map(|(_, solver)| solver)
                            .unwrap_or_else(|| PersistentSolver::new(backend));
                        let mut buses: Vec<Bus> = Vec::with_capacity(buses_template.len());

                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= scenarios.len() {
                                break;
                            }

                            buses.clear();
                            buses.extend_from_slice(buses_template);
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

                            if start == BatchStart::Linear {
                                linear_initial_guess(&mut buses, ybus);
                            }
                            let (islands, stats) =
                                solver.solve_with_stats(&mut buses, ybus, tol, max_iter);
                            let result = keep(&mut buses, islands, stats);
                            collected
                                .lock()
                                .expect("batch result mutex poisoned by a panicking worker")
                                .push((i, result));
                        }
                        if let Some(key) = key {
                            WORKER_SOLVER.with(|slot| *slot.borrow_mut() = Some((key, solver)));
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
    /// path: each worker keeps a copy of the intact Y-bus, built by
    /// [`network::build_ybus_with_outages`](crate::network::build_ybus_with_outages)
    /// with every branch's structural entries, and takes the outaged
    /// branches off it in place ([`YBusSparse::set`], their stamps from
    /// [`branch_stamp`]), telling its solver's cached Jacobian of the entries
    /// that changed ([`PersistentSolver::set_admittance`]) and restoring both
    /// afterwards. The sparsity pattern — and therefore
    /// `jacobian::JacobianPattern` and the symbolic factorization behind it —
    /// stays the intact network's, so a worker reuses one
    /// [`PersistentSolver`] across every such scenario it handles, exactly as
    /// [`solve`](Self::solve) does for injection-only batches. Rebuilding the
    /// Y-bus and re-deriving the Jacobian per outage instead was two thirds
    /// of an N-1 sweep of case1354pegase.
    ///
    /// A contingency that *splits* the network cannot use that trick:
    /// [`connected_components`](crate::network::connected_components) reads the
    /// same structural entries and cannot tell a zero-valued one from a live
    /// one, so it would report the severed buses as still connected and their
    /// now-referenceless rows would come back as a singular solve instead of an
    /// honest `NoReferenceBus`. Those scenarios are detected, by the worker
    /// that solves them, with [`structural_component_count`]
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
        self.contingencies_each(
            buses_template,
            lines,
            transformers,
            shunts,
            scenarios,
            tol,
            max_iter,
            |buses, islands, stats| PowerFlowReport {
                buses: std::mem::take(buses),
                islands,
                stats,
                dc: None,
                linear: None,
                outer: None,
            },
        )
    }

    /// [`solve_contingencies`](Self::solve_contingencies), keeping only each
    /// scenario's voltages and convergence, as
    /// [`solve_voltages`](Self::solve_voltages) does for a batch: one reused
    /// bus buffer per worker instead of a whole bus vector per scenario.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_contingency_voltages(
        &self,
        buses_template: &[Bus],
        lines: &[Line],
        transformers: &[Transformer],
        shunts: &[ShuntAdm],
        scenarios: &[Scenario],
        tol: f64,
        max_iter: usize,
    ) -> Result<Vec<ScenarioVoltages>, BatchError> {
        self.contingencies_each(
            buses_template,
            lines,
            transformers,
            shunts,
            scenarios,
            tol,
            max_iter,
            |buses, _, stats| ScenarioVoltages {
                voltage_mag: buses.iter().map(|b| b.voltage_mag).collect(),
                voltage_ang: buses.iter().map(|b| b.voltage_ang).collect(),
                stats,
            },
        )
    }

    /// Every contingency, solved in a worker's bus buffer and handed to
    /// `keep`; in scenario order.
    #[allow(clippy::too_many_arguments)]
    fn contingencies_each<R: Send>(
        &self,
        buses_template: &[Bus],
        lines: &[Line],
        transformers: &[Transformer],
        shunts: &[ShuntAdm],
        scenarios: &[Scenario],
        tol: f64,
        max_iter: usize,
        keep: impl Fn(&mut Vec<Bus>, Vec<IslandReport>, SolveStats) -> R + Sync,
    ) -> Result<Vec<R>, BatchError> {
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

        // Each scenario's plan: which branches are out, whether taking them
        // out splits the network, and what the outaged branches add to the
        // Y-bus. Made by the worker that solves it: the connectivity check is
        // a search of the whole network, and made up front for every scenario
        // it was a serial share of the sweep that grew with its length.
        let plan = |sc: &Scenario| {
            let mut outaged = vec![false; n_branches];
            for &b in &sc.branch_outages {
                outaged[b] = true;
            }
            let splits = !sc.branch_outages.is_empty()
                && structural_component_count(n, lines, transformers, &outaged)
                    != intact_components;
            let stamp: Vec<(usize, usize, Complex<f64>)> = sc
                .branch_outages
                .iter()
                .flat_map(|&b| branch_stamp(lines, transformers, b).into_entries())
                .collect();
            (outaged, splits, stamp)
        };

        // The intact network, which every worker copies once and changes in
        // place for each outage that keeps it connected.
        let intact = {
            let mut ybus = build_ybus_with_outages(n, lines, transformers, &[]);
            stamp_shunts(&mut ybus, shunts);
            ybus.finish()
        };
        // A dedicated pool's workers keep their solver between calls, as in
        // `solve`: every outage is undone before the next, so a worker's
        // solver ends each call holding the intact network's admittances.
        let key = self
            .pool
            .is_some()
            .then(|| (self.id, network_fingerprint(buses_template, &intact)));

        let next = AtomicUsize::new(0);
        let collected: Mutex<Vec<(usize, R)>> = Mutex::new(Vec::with_capacity(scenarios.len()));
        let n_workers = self.threads().max(1).min(scenarios.len());
        let backend = self.backend;
        let start = self.start;

        let run = || {
            rayon::scope(|s| {
                for _ in 0..n_workers {
                    s.spawn(|_| {
                        // Inside the closure for the same reason as in
                        // `solve`: it is what keeps the `!Send` FFI backends
                        // usable here.
                        let cached = key.and_then(|key| {
                            WORKER_SOLVER
                                .with(|slot| slot.borrow_mut().take())
                                .filter(|(k, _)| *k == key)
                        });
                        let mut solver = cached
                            .map(|(_, solver)| solver)
                            .unwrap_or_else(|| PersistentSolver::new(backend));
                        let mut ybus = intact.clone();
                        let mut undo: Vec<(usize, usize, Complex<f64>)> = Vec::new();
                        let mut buses: Vec<Bus> = Vec::with_capacity(buses_template.len());

                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= scenarios.len() {
                                break;
                            }
                            let (outaged, splits, stamp) = plan(&scenarios[i]);
                            let (outaged, splits, stamp) = (&outaged, &splits, &stamp);

                            buses.clear();
                            buses.extend_from_slice(buses_template);
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

                            let split_ybus = splits.then(|| {
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
                                let mut ybus = build_ybus(n, &keep_lines, &keep_transformers);
                                stamp_shunts(&mut ybus, shunts);
                                ybus.finish()
                            });
                            if !splits {
                                // Same pattern, different values: the outaged
                                // branches come off the intact Y-bus in place,
                                // and only the entries they touch change in the
                                // Jacobian the worker keeps. The values they
                                // replace are restored exactly afterwards, so
                                // no rounding carries into the next outage.
                                undo.clear();
                                for &(r, c, y) in stamp {
                                    let old = ybus.get(r, c);
                                    ybus.set(r, c, old - y)
                                        .expect("the intact Y-bus stores every branch's entries");
                                    undo.push((r, c, old));
                                }
                                for &(r, c, _) in stamp {
                                    solver.set_admittance(r, c, ybus.get(r, c));
                                }
                            }
                            let solve_on = split_ybus.as_ref().unwrap_or(&ybus);

                            if start == BatchStart::Linear {
                                linear_initial_guess(&mut buses, solve_on);
                            }
                            let (islands, stats) =
                                solver.solve_with_stats(&mut buses, solve_on, tol, max_iter);
                            if *splits {
                                // ...and must not leak into the next one.
                                solver.reset();
                            } else {
                                for &(r, c, old) in undo.iter().rev() {
                                    ybus.set(r, c, old);
                                }
                                for &(r, c, _) in stamp {
                                    solver.set_admittance(r, c, ybus.get(r, c));
                                }
                            }

                            let result = keep(&mut buses, islands, stats);
                            collected
                                .lock()
                                .expect("batch result mutex poisoned by a panicking worker")
                                .push((i, result));
                        }
                        if let Some(key) = key {
                            WORKER_SOLVER.with(|slot| *slot.borrow_mut() = Some((key, solver)));
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
