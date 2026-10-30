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

use gridoxide_core::network::{ShuntAdm, YBusSparse};
use gridoxide_core::scenario::Scenario;
use gridoxide_core::types::{Bus, Line, Transformer};

use crate::cudss::CudssBatch;
use crate::device::DeviceBatch;
use crate::newton::{
    BatchError, BatchOptions, BatchReport, Contingencies, Delta, HandOff, Outcome, ScenarioResult,
    StepPolicy, check_buses, contingencies, decide, finish_on_cpu, intact_ybus, needs_factors,
};
use crate::uniform::LinearCounts;

/// The branches an N-1 session takes out.
struct Branches {
    lines: Vec<Line>,
    transformers: Vec<Transformer>,
    shunts: Vec<ShuntAdm>,
}

/// A session of batched power flows on one network; see the module
/// documentation.
pub struct GpuBatchSolver {
    // First: its descriptors point into `dev`'s buffers and it runs on
    // `dev`'s stream, so it goes before them.
    lin: CudssBatch,
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
        let dev = DeviceBatch::with_capacity(start, &ybus, capacity)?;
        let (values, rhs, dx) = dev.linear_buffers();
        // SAFETY: the buffers are `dev`'s, sized for its capacity, and `dev`
        // and its stream outlive `lin`, which the struct drops first.
        let lin =
            unsafe { CudssBatch::analyze(dev.csr(), capacity, dev.stream(), values, rhs, dx)? };
        Ok(Self {
            lin,
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
        let before = self.lin.counts();
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
        let after = self.lin.counts();
        report.linear = LinearCounts {
            factor_calls: after.factor_calls - before.factor_calls,
            solve_calls: after.solve_calls - before.solve_calls,
            systems_factored: after.systems_factored - before.systems_factored,
            systems_solved: after.systems_solved - before.systems_solved,
        };
        Ok(report)
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
        self.dev.load(scenarios, deltas)?;

        let mut active: Vec<bool> = (0..batch).map(|s| s < m).collect();
        let mut outcome: Vec<Option<Outcome>> = vec![None; m];
        let mut steps = vec![0usize; m];
        let mut factorizations = vec![0usize; m];
        let mut norm = vec![f64::INFINITY; m];
        let mut last_norm = vec![f64::INFINITY; m];
        let mut on_cpu: Vec<Option<Vec<Bus>>> = vec![None; m];

        loop {
            // 1. Mismatch on the device, flags on the host.
            self.dev.set_active(&active)?;
            self.dev.mismatch()?;
            let norms = self.dev.norms()?;
            for s in 0..m {
                if !active[s] {
                    continue;
                }
                norm[s] = norms[s];
                if let Some(o) = decide(opts, norm[s], steps[s]) {
                    outcome[s] = Some(o);
                    active[s] = false;
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

            // 2. Jacobians where new factors are needed, one masked factorization.
            let refactor: Vec<bool> = (0..batch)
                .map(|s| active[s] && needs_factors(opts, factorizations[s], norm[s], last_norm[s]))
                .collect();
            if refactor.iter().any(|&r| r) {
                self.dev.set_active(&refactor)?;
                self.dev.fill_jacobian()?;
                self.lin.factor(&refactor)?;
                for s in (0..m).filter(|&s| refactor[s]) {
                    factorizations[s] += 1;
                }
            }

            // 3. One masked solve into the update buffer, and the update.
            self.lin.solve(&active)?;
            self.dev.set_active(&active)?;
            self.dev.update_in_place()?;
            for s in (0..m).filter(|&s| active[s]) {
                steps[s] += 1;
                last_norm[s] = norm[s];
            }
            report.batch_steps += 1;
        }

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
