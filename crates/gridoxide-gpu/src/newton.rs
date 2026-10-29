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
//! Buffers are laid out as the device needs them: `B × nnz` Jacobian values in
//! CSR order, `B × n` right-hand sides and updates, system-major. Bus state is
//! still one `Vec<Bus>` per scenario here, the CPU's layout; the device keeps
//! it in flat arrays.

use gridoxide_core::jacobian::JacobianPattern;
use gridoxide_core::network::{YBusSparse, effective_injection, power_injections};
use gridoxide_core::scenario::Scenario;
use gridoxide_core::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide_core::types::{Bus, BusType};

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
}

#[derive(Clone, Debug)]
pub struct ScenarioResult {
    pub outcome: Outcome,
    /// Newton updates applied, on the batch and on the CPU.
    pub newton_steps: usize,
    /// Of those, the steps that refactored the Jacobian.
    pub factorizations: usize,
    /// ‖F‖∞ at the last evaluation.
    pub final_mismatch: f64,
    /// Whether the CPU finished it (`cpu_handoff`).
    pub on_cpu: bool,
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

/// Solves every scenario from `start`'s voltages: `start` carries the
/// network's bus types and injections and the voltages every scenario starts
/// from (a solved base case, for a sweep from one operating point), and each
/// scenario's overrides replace injections and setpoints.
///
/// Every scenario has `start`'s bus types, so they share one Jacobian
/// pattern, which is what makes them a uniform batch. The network must be one
/// component with its reference; islands are the CPU solver's business.
pub fn solve_batch<U: UniformBatch>(
    start: &[Bus],
    ybus: &YBusSparse,
    scenarios: &[Scenario],
    opts: &BatchOptions,
) -> Result<BatchReport, LinearError> {
    let batch = scenarios.len();
    let mut pattern = JacobianPattern::analyze(start, ybus);
    let n = pattern.n_unknowns;
    // CSR order, which cuDSS takes and the fill then writes directly.
    let mut order: Vec<usize> = (0..pattern.len()).collect();
    order.sort_by_key(|&k| (pattern.rows()[k], pattern.cols()[k]));
    pattern.permute(&order);
    let csr = csr_of(&pattern, n);
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
            let buses = &states[s];
            let (p, q) = power_injections(buses, ybus);
            let f = &mut rhs[s * n..(s + 1) * n];
            for (k, &i) in non_slack.iter().enumerate() {
                f[k] = effective_injection(&buses[i]).0 - p[i];
            }
            for (k, &i) in pq.iter().enumerate() {
                f[non_slack.len() + k] = effective_injection(&buses[i]).1 - q[i];
            }
            norm[s] = f.iter().fold(
                0.0f64,
                |m, v| if v.is_nan() { f64::NAN } else { m.max(v.abs()) },
            );
            injections[s] = (p, q);

            let done = match opts.policy {
                StepPolicy::Adaptive => {
                    if !(norm[s].is_finite()) || norm[s] > opts.diverge_above {
                        Some(Outcome::Diverged)
                    } else if norm[s] < opts.tol {
                        Some(Outcome::Converged)
                    } else if steps[s] >= opts.max_iter {
                        Some(Outcome::NotConverged)
                    } else {
                        None
                    }
                }
                StepPolicy::Fixed(k) => {
                    if steps[s] < k {
                        None
                    } else if !norm[s].is_finite() || norm[s] > opts.diverge_above {
                        Some(Outcome::Diverged)
                    } else if norm[s] < opts.tol {
                        Some(Outcome::Converged)
                    } else {
                        Some(Outcome::NotConverged)
                    }
                }
            };
            if let Some(o) = done {
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
                let budget = opts.max_iter - steps[s];
                let (_, stats) = solver.solve_with_stats(&mut states[s], ybus, opts.tol, budget);
                // `iterations()` counts mismatch evaluations, the first at
                // the state the batch left, so it is one more than the steps
                // when the solve converged.
                let taken = match stats.status {
                    SolveStatus::Converged => stats.iterations().saturating_sub(1),
                    _ => stats.iterations(),
                };
                steps[s] += taken;
                factorizations[s] += taken;
                norm[s] = stats.final_mismatch();
                outcome[s] = Some(match stats.status {
                    SolveStatus::Converged => Outcome::Converged,
                    _ if !norm[s].is_finite() => Outcome::Diverged,
                    _ => Outcome::NotConverged,
                });
                on_cpu[s] = true;
                active[s] = false;
                cpu_handoffs += 1;
            }
            break;
        }

        // 2. Jacobians where new factors are needed, and one masked factorization.
        let refactor: Vec<bool> = (0..batch)
            .map(|s| {
                active[s]
                    && match opts.reuse_below {
                        None => true,
                        Some(ratio) => factorizations[s] == 0 || norm[s] > ratio * last_norm[s],
                    }
            })
            .collect();
        for s in (0..batch).filter(|&s| refactor[s]) {
            let (p, q) = &injections[s];
            pattern.fill(&states[s], p, q, &mut fill);
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
