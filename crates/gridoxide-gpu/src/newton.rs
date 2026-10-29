//! Batched Newton-Raphson over a [`UniformBatch`], every scenario stopped at
//! its own convergence.
//!
//! The loop the device will run, written once over the linear backend so it
//! is tested on the CPU first ([`CpuUniformBatch`](crate::uniform::CpuUniformBatch)).
//! Each step:
//!
//! 1. every active scenario's mismatch and its ‖F‖∞, which flags it converged
//!    (below `tol`), diverged (non-finite or above `diverge_above`) or active;
//! 2. the Jacobians of the active scenarios that need new factors, into the
//!    batch's value buffer, and one masked factorization;
//! 3. one masked solve of every active scenario, and its update.
//!
//! A scenario leaves the masks once it is flagged, so converged scenarios
//! stop costing factorizations and solves, where a fixed step count (the
//! [`StepPolicy::Fixed`] that gpusim2grid runs) spends the same work on every
//! scenario and reports none of them. Two options trade further:
//! [`BatchOptions::reuse_below`] keeps a scenario on its last factors while its
//! mismatch contracts fast enough, and [`BatchOptions::cpu_handoff`] finishes
//! the last few active scenarios on the CPU instead of running a nearly empty
//! batch.
//!
//! N-1 ([`solve_contingency_batch`]) is the same loop: an outage that keeps
//! the network connected leaves the Y-bus pattern, and so the Jacobian
//! pattern, as it is and changes only the values at the outaged branches'
//! few positions. Each scenario carries those values as a delta on the intact
//! Y-bus, applied around its mismatch and Jacobian fill and restored exactly
//! afterwards, as the CPU's `solve_contingencies` applies outages in place;
//! the device keeps the same deltas per scenario. An outage that splits the
//! network has another pattern and is not a member of the batch.
//!
//! Buffers are laid out as the device needs them: `B × nnz` Jacobian values in
//! CSR order, `B × n` right-hand sides and updates, system-major. Bus state is
//! still one `Vec<Bus>` per scenario here, the CPU's layout; the device keeps
//! it in flat arrays.

use std::collections::BTreeMap;

use gridoxide_core::jacobian::JacobianPattern;
use gridoxide_core::network::{
    ShuntAdm, YBusSparse, branch_stamp, build_ybus_with_outages, effective_injection,
    power_injections, stamp_shunts, structural_component_count,
};
use gridoxide_core::scenario::Scenario;
use gridoxide_core::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide_core::types::{Bus, BusType, Line, Transformer};
use num_complex::Complex;

use crate::uniform::{CsrPattern, LinearCounts, LinearError, UniformBatch};

/// When a scenario stops stepping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepPolicy {
    /// At its own convergence, divergence, or `max_iter`.
    Adaptive,
    /// After exactly this many Newton steps, converged or not, as
    /// gpusim2grid's batch sweeps run (`nb_iter`); the mismatch is checked
    /// only then.
    Fixed(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct BatchOptions {
    /// ‖F‖∞ in per unit.
    pub tol: f64,
    pub max_iter: usize,
    pub policy: StepPolicy,
    /// A mismatch above this, or non-finite, is divergence.
    pub diverge_above: f64,
    /// Refactor a scenario only when its mismatch fell by less than this
    /// factor since its last step (0.1: less than tenfold); otherwise solve
    /// against its last factors. `None` refactors every active scenario
    /// every step, as Newton-Raphson does.
    pub reuse_below: Option<f64>,
    /// Once this few scenarios remain active, finish them on the CPU.
    pub cpu_handoff: Option<usize>,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            max_iter: 20,
            policy: StepPolicy::Adaptive,
            diverge_above: 1e10,
            reuse_below: None,
            cpu_handoff: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Converged,
    /// Non-finite or above `diverge_above`.
    Diverged,
    /// Out of steps with the mismatch above `tol`.
    NotConverged,
    /// The scenario's outage splits the network, so it is not solved here:
    /// its pattern is not the batch's, and an island without a reference has
    /// no solution to report. The CPU's `BatchSolver::solve_contingencies`
    /// solves it island by island.
    Splits,
}

#[derive(Clone, Debug)]
pub struct ScenarioResult {
    pub outcome: Outcome,
    /// Newton updates applied, on the batch and on the CPU.
    pub newton_steps: usize,
    /// Of those, the steps that refactored the Jacobian.
    pub factorizations: usize,
    /// ‖F‖∞ at the last evaluation; NaN for [`Outcome::Splits`].
    pub final_mismatch: f64,
    /// Whether the CPU finished it (`cpu_handoff`).
    pub on_cpu: bool,
    /// Empty for [`Outcome::Splits`].
    pub voltage_mag: Vec<f64>,
    pub voltage_ang: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct BatchReport {
    pub scenarios: Vec<ScenarioResult>,
    /// Steps the batch took: one masked factorization and solve each.
    pub batch_steps: usize,
    pub linear: LinearCounts,
    /// Scenarios finished on the CPU.
    pub cpu_handoffs: usize,
}

/// Why a batch was refused before it ran.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchError {
    /// [`solve_batch`] was given a scenario with branch outages, which a
    /// finished Y-bus cannot honour; [`solve_contingency_batch`] takes the
    /// branch lists.
    OutagesUnsupported {
        scenario: usize,
    },
    BusOutOfRange {
        scenario: usize,
        bus: usize,
        n_buses: usize,
    },
    BranchOutOfRange {
        scenario: usize,
        branch: usize,
        n_branches: usize,
    },
    /// N-1 was asked of a GPU session made without the network's branches
    /// (`GpuBatchSolver::new` rather than `with_branches`).
    BranchesUnknown,
    Linear(LinearError),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::OutagesUnsupported { scenario } => write!(
                f,
                "scenario {scenario} requests branch outages, which `solve_batch` cannot \
                 honour from a finished Y-bus — use `solve_contingency_batch`"
            ),
            BatchError::BusOutOfRange {
                scenario,
                bus,
                n_buses,
            } => write!(
                f,
                "scenario {scenario} overrides bus {bus}, but there are {n_buses} buses"
            ),
            BatchError::BranchOutOfRange {
                scenario,
                branch,
                n_branches,
            } => write!(
                f,
                "scenario {scenario} takes out branch {branch}, but there are {n_branches} branches"
            ),
            BatchError::BranchesUnknown => write!(
                f,
                "N-1 needs the network's branches: make the session with `with_branches`"
            ),
            BatchError::Linear(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BatchError {}

impl From<LinearError> for BatchError {
    fn from(e: LinearError) -> Self {
        BatchError::Linear(e)
    }
}

/// Solves every scenario from `start`'s voltages: `start` carries the
/// network's bus types and injections and the voltages every scenario starts
/// from (a solved base case, for a sweep from one operating point), and each
/// scenario's overrides replace injections and setpoints.
///
/// Every scenario has `start`'s bus types, so they share one Jacobian
/// pattern, which is what makes them a uniform batch. The network must be one
/// component with its reference; islands are the CPU solver's business. A
/// scenario with branch outages is refused, not ignored.
pub fn solve_batch<U: UniformBatch>(
    start: &[Bus],
    ybus: &YBusSparse,
    scenarios: &[Scenario],
    opts: &BatchOptions,
) -> Result<BatchReport, BatchError> {
    check_buses(start, scenarios)?;
    if let Some(scenario) = scenarios
        .iter()
        .position(|sc| !sc.branch_outages.is_empty())
    {
        return Err(BatchError::OutagesUnsupported { scenario });
    }
    let deltas = vec![Delta::default(); scenarios.len()];
    run::<U>(start, &mut ybus.clone(), scenarios, &deltas, opts)
}

/// N-1 as one batch: each scenario's `branch_outages` taken off the intact
/// network built from `lines`, `transformers` and `shunts`, on top of its bus
/// overrides; see the module documentation. A scenario whose outage splits
/// the network is reported as [`Outcome::Splits`] and costs the batch
/// nothing.
pub fn solve_contingency_batch<U: UniformBatch>(
    start: &[Bus],
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    scenarios: &[Scenario],
    opts: &BatchOptions,
) -> Result<BatchReport, BatchError> {
    let Contingencies {
        mut ybus,
        members,
        deltas,
    } = contingencies(start, lines, transformers, shunts, scenarios)?;
    let batch: Vec<Scenario> = members.iter().map(|&s| scenarios[s].clone()).collect();
    let solved = run::<U>(start, &mut ybus, &batch, &deltas, opts)?;
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

/// N-1's batch before it runs: the intact Y-bus, the scenarios whose
/// outages keep the network connected, and each one's delta.
pub(crate) struct Contingencies {
    pub(crate) ybus: YBusSparse,
    pub(crate) members: Vec<usize>,
    pub(crate) deltas: Vec<Delta>,
}

/// Checks the scenarios and sorts out which outages split the network; what
/// [`solve_contingency_batch`] and the device's
/// [`DeviceBatch::contingencies`](crate::device::DeviceBatch::contingencies)
/// share.
pub(crate) fn contingencies(
    start: &[Bus],
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    scenarios: &[Scenario],
) -> Result<Contingencies, BatchError> {
    check_buses(start, scenarios)?;
    let n_branches = lines.len() + transformers.len();
    for (scenario, sc) in scenarios.iter().enumerate() {
        if let Some(&branch) = sc.branch_outages.iter().find(|&&b| b >= n_branches) {
            return Err(BatchError::BranchOutOfRange {
                scenario,
                branch,
                n_branches,
            });
        }
    }

    let n = start.len();
    let ybus = intact_ybus(n, lines, transformers, shunts);
    let intact_components = structural_component_count(n, lines, transformers, &[]);
    let mut members = Vec::new();
    let mut deltas = Vec::new();
    for (s, sc) in scenarios.iter().enumerate() {
        let mut outaged = vec![false; n_branches];
        for &b in &sc.branch_outages {
            outaged[b] = true;
        }
        if structural_component_count(n, lines, transformers, &outaged) != intact_components {
            continue;
        }
        members.push(s);
        deltas.push(Delta::of_outage(
            &ybus,
            lines,
            transformers,
            &sc.branch_outages,
        ));
    }

    Ok(Contingencies {
        ybus,
        members,
        deltas,
    })
}

/// The intact network's Y-bus that N-1's deltas apply to.
pub(crate) fn intact_ybus(
    n: usize,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
) -> YBusSparse {
    let mut ybus = build_ybus_with_outages(n, lines, transformers, &[]);
    stamp_shunts(&mut ybus, shunts);
    ybus.finish()
}

pub(crate) fn check_buses(start: &[Bus], scenarios: &[Scenario]) -> Result<(), BatchError> {
    for (scenario, sc) in scenarios.iter().enumerate() {
        if let Some(ov) = sc.bus_overrides.iter().find(|ov| ov.bus >= start.len()) {
            return Err(BatchError::BusOutOfRange {
                scenario,
                bus: ov.bus,
                n_buses: start.len(),
            });
        }
    }
    Ok(())
}

/// One scenario's change to the shared Y-bus: at each position it touches,
/// the value with its outage and the intact one, to restore exactly. Empty
/// for a scenario without outages.
#[derive(Clone, Debug, Default)]
pub(crate) struct Delta {
    pub(crate) at: Vec<(usize, usize)>,
    pub(crate) outaged: Vec<Complex<f64>>,
    intact: Vec<Complex<f64>>,
}

impl Delta {
    /// The outaged branches' stamps subtracted from the intact values in
    /// stamp order, as the CPU's `solve_contingencies` subtracts them, so
    /// both see the same bits.
    fn of_outage(
        ybus: &YBusSparse,
        lines: &[Line],
        transformers: &[Transformer],
        branches: &[usize],
    ) -> Self {
        let mut values: BTreeMap<(usize, usize), Complex<f64>> = BTreeMap::new();
        for &b in branches {
            for (r, c, y) in branch_stamp(lines, transformers, b).into_entries() {
                *values.entry((r, c)).or_insert_with(|| ybus.get(r, c)) -= y;
            }
        }
        Self {
            at: values.keys().copied().collect(),
            outaged: values.values().copied().collect(),
            intact: values.keys().map(|&(r, c)| ybus.get(r, c)).collect(),
        }
    }

    /// Writes `values` at the delta's positions into the Y-bus and the
    /// Jacobian pattern's copies of it.
    fn write(&self, values: &[Complex<f64>], ybus: &mut YBusSparse, pattern: &mut JacobianPattern) {
        self.write_ybus(values, ybus);
        for (&(r, c), &y) in self.at.iter().zip(values) {
            pattern.set_admittance(r, c, y);
        }
    }

    /// [`write`](Self::write) into the Y-bus alone.
    fn write_ybus(&self, values: &[Complex<f64>], ybus: &mut YBusSparse) {
        for (&(r, c), &y) in self.at.iter().zip(values) {
            ybus.set(r, c, y)
                .expect("the intact Y-bus stores every branch's entries");
        }
    }
}

/// The batched loop over scenarios that share `start`'s pattern, each with
/// its delta on `ybus`; `ybus` is left as it came.
fn run<U: UniformBatch>(
    start: &[Bus],
    ybus: &mut YBusSparse,
    scenarios: &[Scenario],
    deltas: &[Delta],
    opts: &BatchOptions,
) -> Result<BatchReport, BatchError> {
    let batch = scenarios.len();
    let (mut pattern, csr) = csr_jacobian(start, ybus);
    let n = pattern.n_unknowns;
    let nnz = csr.nnz();

    let non_slack: Vec<usize> = start
        .iter()
        .filter(|b| b.bus_type != BusType::Slack)
        .map(|b| b.idx)
        .collect();
    let pq: Vec<usize> = start
        .iter()
        .filter(|b| b.bus_type == BusType::PQ)
        .map(|b| b.idx)
        .collect();

    let mut states: Vec<Vec<Bus>> = scenarios
        .iter()
        .map(|sc| {
            let mut buses = start.to_vec();
            for ov in &sc.bus_overrides {
                let b = &mut buses[ov.bus];
                b.p_spec = ov.p_spec.unwrap_or(b.p_spec);
                b.q_spec = ov.q_spec.unwrap_or(b.q_spec);
                b.voltage_mag = ov.voltage_mag.unwrap_or(b.voltage_mag);
            }
            buses
        })
        .collect();

    let mut lin = U::analyze(&csr, batch)?;
    let mut values = vec![0.0; batch * nnz];
    let mut rhs = vec![0.0; batch * n];
    let mut x = vec![0.0; batch * n];
    let mut fill = Vec::with_capacity(nnz);

    let mut active = vec![true; batch];
    let mut outcome: Vec<Option<Outcome>> = vec![None; batch];
    let mut steps = vec![0usize; batch];
    let mut factorizations = vec![0usize; batch];
    let mut norm = vec![f64::INFINITY; batch];
    let mut last_norm = vec![f64::INFINITY; batch];
    let mut on_cpu = vec![false; batch];
    let mut injections: Vec<(Vec<f64>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); batch];
    let mut batch_steps = 0;
    let mut cpu_handoffs = 0;

    loop {
        // 1. Mismatch and flags.
        for s in 0..batch {
            if !active[s] {
                continue;
            }
            deltas[s].write(&deltas[s].outaged, ybus, &mut pattern);
            let buses = &states[s];
            let (p, q) = power_injections(buses, ybus);
            deltas[s].write(&deltas[s].intact, ybus, &mut pattern);
            let f = &mut rhs[s * n..(s + 1) * n];
            for (k, &i) in non_slack.iter().enumerate() {
                f[k] = effective_injection(&buses[i]).0 - p[i];
            }
            for (k, &i) in pq.iter().enumerate() {
                f[non_slack.len() + k] = effective_injection(&buses[i]).1 - q[i];
            }
            // NaN once any entry is NaN: `f64::max` would let a later
            // finite entry replace it, and a NaN scenario pass as converged.
            norm[s] = f.iter().fold(0.0f64, |m, v| {
                if m.is_nan() || v.is_nan() {
                    f64::NAN
                } else {
                    m.max(v.abs())
                }
            });
            injections[s] = (p, q);

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
            let mut solver = PersistentSolver::new(JacobianBackend::KluNative);
            for s in 0..batch {
                if !active[s] {
                    continue;
                }
                let (taken, final_norm, o) = finish_on_cpu(
                    &mut solver,
                    &mut states[s],
                    ybus,
                    &deltas[s],
                    opts,
                    steps[s],
                );
                steps[s] += taken;
                factorizations[s] += taken;
                norm[s] = final_norm;
                outcome[s] = Some(o);
                on_cpu[s] = true;
                active[s] = false;
                cpu_handoffs += 1;
            }
            break;
        }

        // 2. Jacobians where new factors are needed, and one masked factorization.
        let refactor: Vec<bool> = (0..batch)
            .map(|s| active[s] && needs_factors(opts, factorizations[s], norm[s], last_norm[s]))
            .collect();
        for s in (0..batch).filter(|&s| refactor[s]) {
            let (p, q) = &injections[s];
            deltas[s].write(&deltas[s].outaged, ybus, &mut pattern);
            pattern.fill(&states[s], p, q, &mut fill);
            deltas[s].write(&deltas[s].intact, ybus, &mut pattern);
            values[s * nnz..(s + 1) * nnz].copy_from_slice(&fill);
            factorizations[s] += 1;
        }
        if refactor.iter().any(|&r| r) {
            lin.factor(&values, &refactor)?;
        }

        // 3. One masked solve, and the update.
        lin.solve(&rhs, &mut x, &active)?;
        for s in (0..batch).filter(|&s| active[s]) {
            let dx = &x[s * n..(s + 1) * n];
            let (dth, dvm) = dx.split_at(non_slack.len());
            for (&i, &d) in non_slack.iter().zip(dth) {
                states[s][i].voltage_ang += d;
            }
            for (&i, &d) in pq.iter().zip(dvm) {
                states[s][i].voltage_mag += d;
            }
            steps[s] += 1;
            last_norm[s] = norm[s];
        }
        batch_steps += 1;
    }

    let scenarios = (0..batch)
        .map(|s| ScenarioResult {
            outcome: outcome[s].expect("every scenario leaves the loop flagged"),
            newton_steps: steps[s],
            factorizations: factorizations[s],
            final_mismatch: norm[s],
            on_cpu: on_cpu[s],
            voltage_mag: states[s].iter().map(|b| b.voltage_mag).collect(),
            voltage_ang: states[s].iter().map(|b| b.voltage_ang).collect(),
        })
        .collect();
    Ok(BatchReport {
        scenarios,
        batch_steps,
        linear: lin.counts(),
        cpu_handoffs,
    })
}

/// The Jacobian pattern of `start`'s bus types on `ybus`, its entries put in
/// CSR order (which cuDSS takes and the fill then writes directly), and that
/// order as a [`CsrPattern`]: what the CPU loop and the device share.
pub fn csr_jacobian(start: &[Bus], ybus: &YBusSparse) -> (JacobianPattern, CsrPattern) {
    let mut pattern = JacobianPattern::analyze(start, ybus);
    let mut order: Vec<usize> = (0..pattern.len()).collect();
    order.sort_by_key(|&k| (pattern.rows()[k], pattern.cols()[k]));
    pattern.permute(&order);
    let csr = csr_of(&pattern, pattern.n_unknowns);
    (pattern, csr)
}

/// What a scenario's mismatch norm after `steps` Newton steps says about it:
/// done with an outcome, or `None` to step on. The one rule the CPU loop and
/// the device loop share.
pub(crate) fn decide(opts: &BatchOptions, norm: f64, steps: usize) -> Option<Outcome> {
    let diverged = !norm.is_finite() || norm > opts.diverge_above;
    match opts.policy {
        StepPolicy::Adaptive => {
            if diverged {
                Some(Outcome::Diverged)
            } else if norm < opts.tol {
                Some(Outcome::Converged)
            } else if steps >= opts.max_iter {
                Some(Outcome::NotConverged)
            } else {
                None
            }
        }
        StepPolicy::Fixed(k) => {
            if steps < k {
                None
            } else if diverged {
                Some(Outcome::Diverged)
            } else if norm < opts.tol {
                Some(Outcome::Converged)
            } else {
                Some(Outcome::NotConverged)
            }
        }
    }
}

/// Whether an active scenario gets new factors this step
/// ([`BatchOptions::reuse_below`]).
pub(crate) fn needs_factors(
    opts: &BatchOptions,
    factorizations: usize,
    norm: f64,
    last_norm: f64,
) -> bool {
    match opts.reuse_below {
        None => true,
        Some(ratio) => factorizations == 0 || norm > ratio * last_norm,
    }
}

/// Finishes one scenario on the CPU ([`BatchOptions::cpu_handoff`]) from
/// the state the batch left in `buses`, on `ybus` with its `delta` applied
/// for the solve and restored after: the steps it took, its last mismatch
/// and its outcome. `solver` is shared by the scenarios handed off together.
pub(crate) fn finish_on_cpu(
    solver: &mut PersistentSolver,
    buses: &mut [Bus],
    ybus: &mut YBusSparse,
    delta: &Delta,
    opts: &BatchOptions,
    steps: usize,
) -> (usize, f64, Outcome) {
    let budget = opts.max_iter - steps;
    // The solver analyses its Jacobian at its first solve and takes later
    // deltas in place, as the CPU's N-1 does.
    delta.write_ybus(&delta.outaged, ybus);
    for &(r, c) in &delta.at {
        solver.set_admittance(r, c, ybus.get(r, c));
    }
    let (_, stats) = solver.solve_with_stats(buses, ybus, opts.tol, budget);
    delta.write_ybus(&delta.intact, ybus);
    for &(r, c) in &delta.at {
        solver.set_admittance(r, c, ybus.get(r, c));
    }
    // `iterations()` counts mismatch evaluations, the first at the state the
    // batch left, so it is one more than the steps when the solve converged.
    let taken = match stats.status {
        SolveStatus::Converged => stats.iterations().saturating_sub(1),
        _ => stats.iterations(),
    };
    let norm = stats.final_mismatch();
    let outcome = match stats.status {
        SolveStatus::Converged => Outcome::Converged,
        _ if !norm.is_finite() => Outcome::Diverged,
        _ => Outcome::NotConverged,
    };
    (taken, norm, outcome)
}

/// The pattern's entries, already in CSR order, as a [`CsrPattern`].
fn csr_of(pattern: &JacobianPattern, n: usize) -> CsrPattern {
    let mut row_ptr = vec![0usize; n + 1];
    for &r in pattern.rows() {
        row_ptr[r as usize + 1] += 1;
    }
    for r in 0..n {
        row_ptr[r + 1] += row_ptr[r];
    }
    CsrPattern {
        n,
        row_ptr,
        col_idx: pattern.cols().iter().map(|&c| c as usize).collect(),
    }
}
