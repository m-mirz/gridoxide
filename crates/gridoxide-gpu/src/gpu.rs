//! The adaptive Newton loop on the device: [`GpuBatchSolver`].
//!
//! The loop of [`newton`](crate::newton) with its state and work on the
//! device. The kernels of
//! [`DeviceBatch`](crate::device::DeviceBatch) fill the Jacobians and
//! mismatches in place, and cuDSS factors and solves those same buffers on
//! the same stream, writing the update where the update kernel reads it.
//! Per step the host gets back one number per scenario, its ‖F‖∞, and
//! decides with the CPU loop's own rule which scenarios stop
//! (`newton::decide`); that is the step's one synchronization. Voltages come
//! back once, at the end, or for the scenarios handed to the CPU.
//!
//! A solver is a session for one network: the cuDSS analysis and the device
//! buffers are made once for `capacity` scenarios and kept, so that solving
//! another set of scenarios costs no analysis. A set larger than the
//! capacity is solved in chunks of it, the last one padded with inactive
//! scenarios, which the masks keep out of every kernel and every cuDSS call.
//!
//! The options are [`BatchOptions`]'s, with the CPU loop's meaning:
//! adaptive or fixed steps, factor reuse, and the CPU hand-off of the last
//! few scenarios.
//!
//! Scenarios that start from the same voltages on the same network have the
//! same Jacobian at the first step: it is factorized once for the session, and
//! that step is one solve with a column per scenario (`CudssShared`). Still
//! Newton's step, for a fifth of a sweep's time. Not for N-1, whose Y-buses
//! differ, a scenario that sets a voltage, or factor reuse, which keeps a
//! scenario's own factors.
//!
//! Those factors also serve further steps, if [`BatchOptions::chord_steps`]
//! asks: a chord method, each step a column of the same solve, a sixteenth of
//! a Newton step's cost on case9241pegase but converging only linearly. Every
//! scenario then finishes on Newton's steps, to the same tolerance on the same
//! mismatch.
//!
//! A single power flow ([`GpuBatchSolver::solve_single`]) is a batch of
//! one from the buses it is given. One system is cuDSS's worst case, a
//! roughly fixed cost per system, so it pays only on large grids; make its
//! session with `capacity` 1, or the other slots are masked out at every
//! step.

use gridoxide_core::network::{ShuntAdm, YBusSparse};
use gridoxide_core::scenario::Scenario;
use gridoxide_core::types::{Bus, Line, Transformer};

use crate::cuda::Range;
use crate::cudss::{CudssBatch, CudssShared};
use crate::device::DeviceBatch;
use crate::newton::{
    BatchError, BatchOptions, BatchReport, Contingencies, Delta, HandOff, Outcome, ScenarioResult,
    StepPolicy, check_buses, contingencies, decide, finish_on_cpu, intact_ybus, needs_factors,
};
use crate::uniform::LinearCounts;

/// The fewest scenarios for which a shared first step pays: it costs a
/// session one more analysis, and a column of a narrow solve costs up to 20
/// times what it does in a wide one.
const MIN_SHARED: usize = 32;

/// The branches an N-1 session takes out.
struct Branches {
    lines: Vec<Line>,
    transformers: Vec<Transformer>,
    shunts: Vec<ShuntAdm>,
}

/// A session of batched power flows on one network; see the module
/// documentation.
pub struct GpuBatchSolver {
    // First: their descriptors point into `dev`'s buffers and they run on
    // `dev`'s stream, so they go before them.
    lin: CudssBatch,
    /// The first step's system, made by the first sweep that shares it.
    shared: Option<CudssShared>,
    dev: DeviceBatch,
    start: Vec<Bus>,
    /// The Y-bus the device holds (the intact one for N-1), for the CPU
    /// hand-off.
    ybus: YBusSparse,
    branches: Option<Branches>,
}

impl GpuBatchSolver {
    /// A session for sweeps on `ybus` from `start` (its bus types, and the
    /// voltages and injections every scenario starts from), `capacity`
    /// scenarios at a time.
    pub fn new(start: &[Bus], ybus: &YBusSparse, capacity: usize) -> Result<Self, BatchError> {
        Self::open(start, ybus.clone(), None, capacity)
    }

    /// A session that also solves N-1
    /// ([`solve_contingencies`](Self::solve_contingencies)) on the intact
    /// network of `lines`, `transformers` and `shunts`.
    pub fn with_branches(
        start: &[Bus],
        lines: &[Line],
        transformers: &[Transformer],
        shunts: &[ShuntAdm],
        capacity: usize,
    ) -> Result<Self, BatchError> {
        let ybus = intact_ybus(start.len(), lines, transformers, shunts);
        let branches = Branches {
            lines: lines.to_vec(),
            transformers: transformers.to_vec(),
            shunts: shunts.to_vec(),
        };
        Self::open(start, ybus, Some(branches), capacity)
    }

    fn open(
        start: &[Bus],
        ybus: YBusSparse,
        branches: Option<Branches>,
        capacity: usize,
    ) -> Result<Self, BatchError> {
        let capacity = capacity.max(1);
        let _r = Range::new(c"batch.analysis");
        let dev = DeviceBatch::with_capacity(start, &ybus, capacity)?;
        let (values, rhs, dx) = dev.linear_buffers();
        // SAFETY: the buffers are `dev`'s, sized for its capacity, and `dev`
        // and its stream outlive `lin`, which the struct drops first.
        let lin =
            unsafe { CudssBatch::analyze(dev.csr(), capacity, dev.stream(), values, rhs, dx)? };
        Ok(Self {
            lin,
            shared: None,
            dev,
            start: start.to_vec(),
            ybus,
            branches,
        })
    }

    /// Scenarios per chunk.
    pub fn capacity(&self) -> usize {
        self.dev.batch()
    }

    /// As [`CudssUniformBatch::use_refactorization`](crate::cudss::CudssUniformBatch::use_refactorization).
    pub fn use_refactorization(&mut self, on: bool) {
        self.lin.use_refactorization(on);
    }

    /// Solves every scenario from `start`'s state, as
    /// [`solve_batch`](crate::newton::solve_batch) does, and refuses the same
    /// scenarios.
    pub fn solve(
        &mut self,
        scenarios: &[Scenario],
        opts: &BatchOptions,
    ) -> Result<BatchReport, BatchError> {
        check_buses(&self.start, scenarios)?;
        if let Some(scenario) = scenarios
            .iter()
            .position(|sc| !sc.branch_outages.is_empty())
        {
            return Err(BatchError::OutagesUnsupported { scenario });
        }
        let deltas = vec![Delta::default(); scenarios.len()];
        self.solve_all(scenarios, &deltas, opts)
    }

    /// A single power flow of `buses`, from their voltages with their
    /// injections and setpoints, voltages written back into them as the
    /// CPU's `PersistentSolver::solve` does. `buses` must be the session's
    /// network: the same count and bus types, which are checked, and the same
    /// ZIP terms, which are not (the device took them with the network).
    /// Later [`solve`](Self::solve)s still start from the session's own
    /// start. Branch outages and the CPU hand-off of `opts` apply as to a
    /// batch of one.
    pub fn solve_single(
        &mut self,
        buses: &mut [Bus],
        opts: &BatchOptions,
    ) -> Result<ScenarioResult, BatchError> {
        let n_buses = self.start.len();
        if buses.len() != n_buses {
            return Err(BatchError::NotTheSessionsBuses {
                n_buses,
                given: buses.len(),
                bus: None,
            });
        }
        if let Some(bus) = (0..n_buses).find(|&i| buses[i].bus_type != self.start[i].bus_type) {
            return Err(BatchError::NotTheSessionsBuses {
                n_buses,
                given: n_buses,
                bus: Some(bus),
            });
        }
        let kept = std::mem::replace(&mut self.start, buses.to_vec());
        self.dev.set_start(&self.start);
        let report = self.solve(&[Scenario::default()], opts);
        self.dev.set_start(&kept);
        self.start = kept;
        let r = report?
            .scenarios
            .pop()
            .expect("one result for one scenario");
        for (i, b) in buses.iter_mut().enumerate() {
            b.voltage_mag = r.voltage_mag[i];
            b.voltage_ang = r.voltage_ang[i];
        }
        Ok(r)
    }

    /// N-1, as [`solve_contingency_batch`](crate::newton::solve_contingency_batch)
    /// does it: an outage that splits the network is reported as
    /// [`Outcome::Splits`]. Needs a session made by
    /// [`with_branches`](Self::with_branches).
    pub fn solve_contingencies(
        &mut self,
        scenarios: &[Scenario],
        opts: &BatchOptions,
    ) -> Result<BatchReport, BatchError> {
        let Some(br) = &self.branches else {
            return Err(BatchError::BranchesUnknown);
        };
        let Contingencies {
            members, deltas, ..
        } = contingencies(
            &self.start,
            &br.lines,
            &br.transformers,
            &br.shunts,
            scenarios,
        )?;
        let batch: Vec<Scenario> = members.iter().map(|&s| scenarios[s].clone()).collect();
        let solved = self.solve_all(&batch, &deltas, opts)?;
        let mut results: Vec<Option<ScenarioResult>> = vec![None; scenarios.len()];
        for (&s, r) in members.iter().zip(solved.scenarios) {
            results[s] = Some(r);
        }
        Ok(BatchReport {
            scenarios: results
                .into_iter()
                .map(|r| {
                    r.unwrap_or(ScenarioResult {
                        outcome: Outcome::Splits,
                        newton_steps: 0,
                        factorizations: 0,
                        final_mismatch: f64::NAN,
                        on_cpu: false,
                        voltage_mag: Vec::new(),
                        voltage_ang: Vec::new(),
                    })
                })
                .collect(),
            ..solved
        })
    }

    /// The scenarios in chunks of the capacity.
    fn solve_all(
        &mut self,
        scenarios: &[Scenario],
        deltas: &[Delta],
        opts: &BatchOptions,
    ) -> Result<BatchReport, BatchError> {
        let before = self.counts();
        let mut report = BatchReport {
            scenarios: Vec::with_capacity(scenarios.len()),
            batch_steps: 0,
            linear: LinearCounts::default(),
            cpu_handoffs: 0,
        };
        let capacity = self.capacity();
        for (chunk, chunk_deltas) in scenarios.chunks(capacity).zip(deltas.chunks(capacity)) {
            self.solve_chunk(chunk, chunk_deltas, opts, &mut report)?;
        }
        let after = self.counts();
        report.linear = LinearCounts {
            factor_calls: after.factor_calls - before.factor_calls,
            solve_calls: after.solve_calls - before.solve_calls,
            systems_factored: after.systems_factored - before.systems_factored,
            systems_solved: after.systems_solved - before.systems_solved,
        };
        Ok(report)
    }

    /// What cuDSS has done in this session.
    fn counts(&self) -> LinearCounts {
        let (a, b) = (
            self.lin.counts(),
            self.shared.as_ref().map(|s| s.counts()).unwrap_or_default(),
        );
        LinearCounts {
            factor_calls: a.factor_calls + b.factor_calls,
            solve_calls: a.solve_calls + b.solve_calls,
            systems_factored: a.systems_factored + b.systems_factored,
            systems_solved: a.systems_solved + b.systems_solved,
        }
    }

    /// Whether every scenario of a chunk has the session's Jacobian at the
    /// first step: `start`'s voltages on the intact network. `solve_single`
    /// starts elsewhere, and is below [`MIN_SHARED`].
    fn shares_first_step(scenarios: &[Scenario], deltas: &[Delta], opts: &BatchOptions) -> bool {
        scenarios.len() >= MIN_SHARED
            && opts.reuse_below.is_none()
            && deltas.iter().all(|d| d.at.is_empty())
            && scenarios
                .iter()
                .all(|sc| sc.bus_overrides.iter().all(|o| o.voltage_mag.is_none()))
    }

    /// A step on the session's first-step factors, for the `active`
    /// scenarios: the solve into the update buffer, after the session's one
    /// factorization, which reads a first step's Jacobian.
    fn shared_step(&mut self, active: &[bool]) -> Result<(), BatchError> {
        let (values, rhs, dx) = self.dev.linear_buffers();
        if self.shared.is_none() {
            let _r = Range::new(c"batch.analysis");
            // SAFETY: as in `open`.
            let shared = unsafe {
                CudssShared::analyze(
                    self.dev.csr(),
                    self.capacity(),
                    self.dev.stream(),
                    values,
                    rhs,
                    dx,
                )?
            };
            self.shared = Some(shared);
        }
        let shared = self.shared.as_mut().expect("made above");
        if !shared.factored() {
            let first = active
                .iter()
                .position(|&a| a)
                .expect("a step has an active scenario");
            let one: Vec<bool> = (0..active.len()).map(|s| s == first).collect();
            let _r = Range::new(c"batch.cudss_factor");
            self.dev.set_active(&one)?;
            self.dev.fill_jacobian()?;
            let nnz = self.dev.csr().nnz();
            // SAFETY: slot `first` of the batch's values, `nnz` of them.
            unsafe { shared.factor(values.add(first * nnz))? };
        }
        let _r = Range::new(c"batch.cudss_solve");
        let width = active.iter().rposition(|&a| a).map_or(0, |s| s + 1);
        shared.solve(width, active.iter().filter(|&&a| a).count())?;
        Ok(())
    }

    /// One chunk: the loop of `newton::run`, step for step.
    fn solve_chunk(
        &mut self,
        scenarios: &[Scenario],
        deltas: &[Delta],
        opts: &BatchOptions,
        report: &mut BatchReport,
    ) -> Result<(), BatchError> {
        let batch = self.capacity();
        let m = scenarios.len();
        let n_bus = self.start.len();
        let _chunk = Range::new(c"batch.chunk");
        {
            let _r = Range::new(c"batch.load");
            self.dev.load(scenarios, deltas)?;
        }

        let mut active: Vec<bool> = (0..batch).map(|s| s < m).collect();
        let mut outcome: Vec<Option<Outcome>> = vec![None; m];
        let mut steps = vec![0usize; m];
        let mut factorizations = vec![0usize; m];
        let mut norm = vec![f64::INFINITY; m];
        let mut last_norm = vec![f64::INFINITY; m];
        let mut on_cpu: Vec<Option<Vec<Bus>>> = vec![None; m];
        // Who steps on the shared factors, and the steps past the first.
        let shared = Self::shares_first_step(scenarios, deltas, opts);
        let mut on_shared: Vec<bool> = (0..batch).map(|s| shared && s < m).collect();
        let mut chord = vec![0usize; m];
        let chord_steps = match opts.policy {
            StepPolicy::Adaptive => opts.chord_steps,
            StepPolicy::Fixed(_) => 0,
        };

        loop {
            // 1. Mismatch on the device, flags on the host.
            let norms = {
                let _r = Range::new(c"batch.mismatch");
                self.dev.set_active(&active)?;
                self.dev.mismatch()?;
                self.dev.norms()?
            };
            for s in 0..m {
                if !active[s] {
                    continue;
                }
                norm[s] = norms[s];
                if let Some(o) = decide(opts, norm[s], steps[s] - chord[s]) {
                    outcome[s] = Some(o);
                    active[s] = false;
                } else if steps[s] > 0
                    && (chord[s] >= chord_steps || norm[s] > opts.chord_below * last_norm[s])
                {
                    on_shared[s] = false;
                }
            }
            let n_active = active.iter().filter(|&&a| a).count();
            if n_active == 0 {
                break;
            }

            // The last few, on the CPU.
            if opts.policy == StepPolicy::Adaptive
                && let Some(limit) = opts.cpu_handoff
                && n_active <= limit
            {
                let (vm, va) = self.dev.voltages()?;
                let handed: Vec<HandOff> = (0..m)
                    .filter(|&s| active[s])
                    .map(|s| {
                        let mut buses = self.start.clone();
                        for ov in &scenarios[s].bus_overrides {
                            let b = &mut buses[ov.bus];
                            b.p_spec = ov.p_spec.unwrap_or(b.p_spec);
                            b.q_spec = ov.q_spec.unwrap_or(b.q_spec);
                        }
                        for (i, b) in buses.iter_mut().enumerate() {
                            b.voltage_mag = vm[s * n_bus + i];
                            b.voltage_ang = va[s * n_bus + i];
                        }
                        HandOff {
                            scenario: s,
                            buses,
                            steps: steps[s],
                        }
                    })
                    .collect();
                for (h, (taken, final_norm, o)) in finish_on_cpu(handed, &self.ybus, deltas, opts) {
                    let s = h.scenario;
                    steps[s] += taken;
                    factorizations[s] += taken;
                    norm[s] = final_norm;
                    outcome[s] = Some(o);
                    on_cpu[s] = Some(h.buses);
                    active[s] = false;
                    report.cpu_handoffs += 1;
                }
                break;
            }

            // 2 and 3 for those on the session's one Jacobian. First: this
            // solve writes every column, the batch's below only its own.
            let shared_now: Vec<bool> = (0..batch).map(|s| active[s] && on_shared[s]).collect();
            if shared_now.iter().any(|&a| a) {
                self.shared_step(&shared_now)?;
                for s in (0..m).filter(|&s| shared_now[s]) {
                    if steps[s] == 0 {
                        // Newton's step, on that factorization.
                        factorizations[s] += 1;
                    } else {
                        chord[s] += 1;
                    }
                }
            }
            let newton: Vec<bool> = (0..batch).map(|s| active[s] && !on_shared[s]).collect();
            if newton.iter().any(|&a| a) {
                // 2. Jacobians where new factors are needed, one masked
                // factorization.
                let refactor: Vec<bool> = (0..batch)
                    .map(|s| {
                        newton[s] && needs_factors(opts, factorizations[s], norm[s], last_norm[s])
                    })
                    .collect();
                if refactor.iter().any(|&r| r) {
                    {
                        let _r = Range::new(c"batch.jacobian");
                        self.dev.set_active(&refactor)?;
                        self.dev.fill_jacobian()?;
                    }
                    let _r = Range::new(c"batch.cudss_factor");
                    self.lin.factor(&refactor)?;
                    for s in (0..m).filter(|&s| refactor[s]) {
                        factorizations[s] += 1;
                    }
                }

                // 3. One masked solve into the update buffer.
                let _r = Range::new(c"batch.cudss_solve");
                self.lin.solve(&newton)?;
            }
            {
                let _r = Range::new(c"batch.update");
                self.dev.set_active(&active)?;
                self.dev.update_in_place()?;
            }
            for s in (0..m).filter(|&s| active[s]) {
                steps[s] += 1;
                last_norm[s] = norm[s];
            }
            report.batch_steps += 1;
        }

        let _r = Range::new(c"batch.results");
        let (vm, va) = self.dev.voltages()?;
        for s in 0..m {
            let (voltage_mag, voltage_ang) = match &on_cpu[s] {
                Some(buses) => (
                    buses.iter().map(|b| b.voltage_mag).collect(),
                    buses.iter().map(|b| b.voltage_ang).collect(),
                ),
                None => (
                    vm[s * n_bus..(s + 1) * n_bus].to_vec(),
                    va[s * n_bus..(s + 1) * n_bus].to_vec(),
                ),
            };
            report.scenarios.push(ScenarioResult {
                outcome: outcome[s].expect("every scenario leaves the loop flagged"),
                newton_steps: steps[s],
                factorizations: factorizations[s],
                final_mismatch: norm[s],
                on_cpu: on_cpu[s].is_some(),
                voltage_mag,
                voltage_ang,
            });
        }
        Ok(())
    }
}
