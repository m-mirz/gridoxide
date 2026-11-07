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
//! capacity streams through its slots: a slot whose scenario ends takes the
//! next one, so the batch stays full until the last scenarios; an empty slot
//! is masked out of every kernel and every cuDSS call.
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
//! [`GpuDcSolver`](crate::dc::GpuDcSolver) has the same three calls:
//! `solve_single`, `solve_sweep` and `solve_contingencies`. Two things differ
//! because the problems do: a session here keeps its start, which a shared
//! first step needs, where a DC sweep takes its template with each call; and
//! a scenario here reports its own outcome, as a DC solve cannot fail one by
//! one.
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

    /// Scenarios at a time: the session's slots.
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
    pub fn solve_sweep(
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
    /// Later [`solve_sweep`](Self::solve_sweep)s still start from the session's own
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
        let report = self.solve_sweep(&[Scenario::default()], opts);
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

    /// The scenarios through the session's slots.
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
        if !scenarios.is_empty() {
            self.solve_stream(scenarios, deltas, opts, &mut report)?;
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
    /// The session's factorization is made, if it is not yet, from a slot of
    /// `fresh`: one still at the start, whose Jacobian is the shared one.
    fn shared_step(&mut self, active: &[bool], fresh: &[bool]) -> Result<(), BatchError> {
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
            let first = fresh
                .iter()
                .position(|&a| a)
                .expect("the first shared step has a scenario at the start");
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

    /// Every scenario through the session's `capacity` slots: the loop of
    /// `newton::run`, step for step, each slot taking the next scenario as
    /// soon as its own ends, or chunk by chunk (below). In chunks, a chunk
    /// takes as many steps as its slowest scenario, every other slot masked
    /// out meanwhile at a sixth of a step's cost: on case9241pegase's N-1 a
    /// scenario takes 3.8 steps on average and its slowest 12.
    fn solve_stream(
        &mut self,
        scenarios: &[Scenario],
        deltas: &[Delta],
        opts: &BatchOptions,
        report: &mut BatchReport,
    ) -> Result<(), BatchError> {
        let batch = self.capacity();
        let total = scenarios.len();
        let n_bus = self.start.len();
        let _stream = Range::new(c"batch.chunk");
        let first = total.min(batch);
        {
            let _r = Range::new(c"batch.load");
            self.dev.load(&scenarios[..first], &deltas[..first])?;
        }
        // Per slot: its scenario, and that scenario's Newton state.
        let mut slot: Vec<Option<usize>> = (0..batch).map(|k| (k < first).then_some(k)).collect();
        let mut next = first;
        let shared = Self::shares_first_step(scenarios, deltas, opts) && batch >= MIN_SHARED;
        // A sweep on the shared first step moves through its steps nearly in
        // step, and refilled slots would cost every step both solves (case9241
        // pegase's 10k: 6.29 s against 4.65): there a chunk ends before the
        // next starts. Elsewhere (N-1) slots refill: 11.4 s to 9.4 on
        // case9241pegase's every outage.
        let refill = !shared;
        let mut on_shared = vec![shared; batch];
        let mut steps = vec![0usize; batch];
        let mut chord = vec![0usize; batch];
        let mut factorizations = vec![0usize; batch];
        let mut norm = vec![f64::INFINITY; batch];
        let mut last_norm = vec![f64::INFINITY; batch];
        let chord_steps = match opts.policy {
            StepPolicy::Adaptive => opts.chord_steps,
            StepPolicy::Fixed(_) => 0,
        };
        let mut results: Vec<Option<ScenarioResult>> = vec![None; total];

        loop {
            let active: Vec<bool> = slot.iter().map(Option::is_some).collect();
            // 1. Mismatch on the device, flags on the host.
            let norms = {
                let _r = Range::new(c"batch.mismatch");
                self.dev.set_active(&active)?;
                self.dev.mismatch()?;
                self.dev.norms()?
            };
            let mut ended: Vec<(usize, Outcome)> = Vec::new();
            for k in (0..batch).filter(|&k| active[k]) {
                norm[k] = norms[k];
                if let Some(o) = decide(opts, norm[k], steps[k] - chord[k]) {
                    ended.push((k, o));
                } else if steps[k] > 0
                    && (chord[k] >= chord_steps || norm[k] > opts.chord_below * last_norm[k])
                {
                    on_shared[k] = false;
                }
            }
            if !ended.is_empty() {
                let _r = Range::new(c"batch.results");
                let which: Vec<usize> = ended.iter().map(|&(k, _)| k).collect();
                let (vm, va) = self.dev.voltages_of(&which)?;
                for (i, &(k, outcome)) in ended.iter().enumerate() {
                    let s = slot[k].take().expect("an active slot holds a scenario");
                    results[s] = Some(ScenarioResult {
                        outcome,
                        newton_steps: steps[k],
                        factorizations: factorizations[k],
                        final_mismatch: norm[k],
                        on_cpu: false,
                        voltage_mag: vm[i * n_bus..(i + 1) * n_bus].to_vec(),
                        voltage_ang: va[i * n_bus..(i + 1) * n_bus].to_vec(),
                    });
                }
                // The ended slots take the next scenarios, which start over;
                // in chunks where the first step is shared (see above).
                let mut fills = Vec::new();
                if !refill && next < total && slot.iter().all(Option::is_none) {
                    let end = (next + batch).min(total);
                    let _r = Range::new(c"batch.load");
                    self.dev.load(&scenarios[next..end], &deltas[next..end])?;
                    for k in 0..batch {
                        slot[k] = (next + k < end).then_some(next + k);
                        (steps[k], chord[k], factorizations[k]) = (0, 0, 0);
                        (norm[k], last_norm[k]) = (f64::INFINITY, f64::INFINITY);
                        on_shared[k] = shared;
                    }
                    next = end;
                    continue;
                }
                for k in which {
                    if next == total || !refill {
                        break;
                    }
                    slot[k] = Some(next);
                    (steps[k], chord[k], factorizations[k]) = (0, 0, 0);
                    (norm[k], last_norm[k]) = (f64::INFINITY, f64::INFINITY);
                    on_shared[k] = shared;
                    fills.push((k, &scenarios[next]));
                    next += 1;
                }
                if !fills.is_empty() {
                    let _r = Range::new(c"batch.load");
                    self.dev.refill(&fills)?;
                    let slot_deltas: Vec<Option<&Delta>> =
                        slot.iter().map(|s| s.map(|s| &deltas[s])).collect();
                    self.dev.set_deltas(&slot_deltas)?;
                    // A full factorization for the new systems: as fast as a
                    // refactorization on the pivots of the system before, and
                    // on case9241pegase's N-1 one more outage converges.
                    let filled: Vec<usize> = fills.iter().map(|&(k, _)| k).collect();
                    self.lin.unfactor(&filled);
                    // Their mismatch first, with everyone else's again.
                    continue;
                }
            }
            let active: Vec<bool> = slot.iter().map(Option::is_some).collect();
            let n_active = active.iter().filter(|&&a| a).count();
            if n_active == 0 {
                break;
            }

            // The last few, on the CPU, once no scenario is waiting.
            if next == total
                && opts.policy == StepPolicy::Adaptive
                && let Some(limit) = opts.cpu_handoff
                && n_active <= limit
            {
                let which: Vec<usize> = (0..batch).filter(|&k| active[k]).collect();
                let (vm, va) = self.dev.voltages_of(&which)?;
                let handed: Vec<HandOff> = which
                    .iter()
                    .enumerate()
                    .map(|(i, &k)| {
                        let s = slot[k].expect("an active slot holds a scenario");
                        let mut buses = self.start.clone();
                        for ov in &scenarios[s].bus_overrides {
                            let b = &mut buses[ov.bus];
                            b.p_spec = ov.p_spec.unwrap_or(b.p_spec);
                            b.q_spec = ov.q_spec.unwrap_or(b.q_spec);
                        }
                        for (j, b) in buses.iter_mut().enumerate() {
                            b.voltage_mag = vm[i * n_bus + j];
                            b.voltage_ang = va[i * n_bus + j];
                        }
                        HandOff {
                            scenario: s,
                            buses,
                            steps: steps[k],
                        }
                    })
                    .collect();
                let slot_of: std::collections::HashMap<usize, usize> =
                    which.iter().map(|&k| (slot[k].unwrap(), k)).collect();
                for (h, (taken, final_norm, outcome)) in
                    finish_on_cpu(handed, &self.ybus, deltas, opts)
                {
                    let k = slot_of[&h.scenario];
                    results[h.scenario] = Some(ScenarioResult {
                        outcome,
                        newton_steps: steps[k] + taken,
                        factorizations: factorizations[k] + taken,
                        final_mismatch: final_norm,
                        on_cpu: true,
                        voltage_mag: h.buses.iter().map(|b| b.voltage_mag).collect(),
                        voltage_ang: h.buses.iter().map(|b| b.voltage_ang).collect(),
                    });
                    slot[k] = None;
                    report.cpu_handoffs += 1;
                }
                break;
            }

            // 2 and 3 for those on the session's one Jacobian. First: this
            // solve writes every column, the batch's below only its own.
            let shared_now: Vec<bool> = (0..batch).map(|k| active[k] && on_shared[k]).collect();
            if shared_now.iter().any(|&a| a) {
                let fresh: Vec<bool> = (0..batch).map(|k| shared_now[k] && steps[k] == 0).collect();
                self.shared_step(&shared_now, &fresh)?;
                for k in (0..batch).filter(|&k| shared_now[k]) {
                    if steps[k] == 0 {
                        // Newton's step, on that factorization.
                        factorizations[k] += 1;
                    } else {
                        chord[k] += 1;
                    }
                }
            }
            let newton: Vec<bool> = (0..batch).map(|k| active[k] && !on_shared[k]).collect();
            if newton.iter().any(|&a| a) {
                // 2. Jacobians where new factors are needed, one masked
                // factorization.
                let refactor: Vec<bool> = (0..batch)
                    .map(|k| {
                        newton[k] && needs_factors(opts, factorizations[k], norm[k], last_norm[k])
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
                    for k in (0..batch).filter(|&k| refactor[k]) {
                        factorizations[k] += 1;
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
            for k in (0..batch).filter(|&k| active[k]) {
                steps[k] += 1;
                last_norm[k] = norm[k];
            }
            report.batch_steps += 1;
        }
        report.scenarios.extend(
            results
                .into_iter()
                .map(|r| r.expect("every scenario leaves the loop flagged")),
        );
        Ok(())
    }
}
