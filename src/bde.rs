//! Block-diagonal embedding: solve B scenarios as one sparse system.
//!
//! Every scenario in a batch shares one topology, so every scenario's Jacobian
//! shares one sparsity pattern. Rather than B separate sparse solves, stack
//! them:
//!
//! ```text
//!         ┌ J₁              ┐   ┌ Δx₁ ┐   ┌ f₁ ┐
//!         │    J₂           │   │ Δx₂ │   │ f₂ │
//!   J  =  │       ⋱         │ , │  ⋮  │ = │ ⋮  │
//!         └             J_B ┘   └ Δx_B┘   └ f_B┘
//! ```
//!
//! This is the design `plans/GPU_PLAN.md` §3 adopts, and the reason it matters
//! is §3 property 2: **it needs no batched solver API**. You hand the library
//! one ordinary sparse matrix. That is what makes the AMD path viable even
//! though rocSOLVER's `csrrf_*` refactorization routines are not batched, and
//! it is why the same code can later target cuDSS, rocSOLVER, or anything else
//! that factors a sparse matrix.
//!
//! The equivalence is exact, not approximate:
//!
//! - **No fill crosses a block.** Fill at `(i, j)` needs a path `i → k → j`
//!   through eliminated vertices; the graph is B disconnected components, so
//!   no such path exists between blocks. Total fill is the sum of per-block
//!   fills.
//! - **Pivoting stays inside its block.** A column in block `s` has nonzeros
//!   only in block `s`'s rows, so partial-pivot search can never reach another
//!   scenario.
//! - **The elimination tree is a forest** of B independent trees, which is
//!   precisely the wide parallelism a GPU refactorization needs — and the
//!   reason a single 18k-unknown Jacobian cannot fill a GPU but 256 stacked
//!   ones can.
//!
//! This module exists to prove all of that on **real sparse code, on the CPU,
//! before any GPU spend**. `scripts/bench/jax_oracle.py` already established it
//! with dense linear algebra; this extends the result to the actual sparse
//! backends the GPU path will mirror.
//!
//! It is deliberately *not* faster than [`crate::batch::BatchSolver`] on a CPU
//! — one big factorization beats B small ones only when the hardware wants
//! wide independent work, which is a GPU property. Treat this as an
//! architecture validator and the host-side half of Phase 3.

use crate::batch::Scenario;
use crate::jacobian::JacobianPattern;
use crate::network::{effective_injection, linear_initial_guess, power_injections, YBusSparse};
use crate::solver::{LinearSolver, SolveStats, SolveStatus};
use crate::types::{Bus, BusType};

/// The stacked sparsity pattern for `n_scenarios` copies of one topology's
/// Jacobian, plus the per-block recipe needed to refill it.
pub struct BlockDiagonal {
    block: JacobianPattern,
    n_scenarios: usize,
    /// Unknowns per scenario — the block stride.
    block_size: usize,
    rows: Vec<u32>,
    cols: Vec<u32>,
}

impl BlockDiagonal {
    /// Derives the stacked pattern by replicating a single-scenario
    /// [`JacobianPattern`] `n_scenarios` times, offsetting block `s`'s rows
    /// and columns by `s * block_size`.
    ///
    /// All scenarios must share `buses`' bus-type assignment and `ybus` —
    /// that shared pattern is the entire premise. `batch::Scenario` cannot
    /// override `bus_type` precisely so this stays true.
    pub fn analyze(buses: &[Bus], ybus: &YBusSparse, n_scenarios: usize) -> Self {
        let block = JacobianPattern::analyze(buses, ybus);
        let block_size = block.n_unknowns;
        let nnz = block.len();

        let mut rows = Vec::with_capacity(nnz * n_scenarios);
        let mut cols = Vec::with_capacity(nnz * n_scenarios);
        for s in 0..n_scenarios {
            let off = (s * block_size) as u32;
            rows.extend(block.rows().iter().map(|&r| r + off));
            cols.extend(block.cols().iter().map(|&c| c + off));
        }

        Self { block, n_scenarios, block_size, rows, cols }
    }

    /// Total unknowns across the batch — `n_scenarios * block_size`.
    pub fn n_unknowns(&self) -> usize {
        self.n_scenarios * self.block_size
    }

    /// Unknowns per scenario.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Nonzeros in the stacked matrix.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The stacked pattern as `(row, col, value)` triplets — the shape every
    /// [`LinearSolver`] consumes for its one-time symbolic analysis.
    pub fn to_triplets(&self, values: &[f64]) -> Vec<(usize, usize, f64)> {
        self.rows
            .iter()
            .zip(&self.cols)
            .zip(values)
            .map(|((&r, &c), &v)| (r as usize, c as usize, v))
            .collect()
    }

    /// Fills the stacked value array from each scenario's own state.
    ///
    /// `active[s] == false` writes an identity block instead of scenario
    /// `s`'s Jacobian (see [`JacobianPattern::fill_identity_into`]), which —
    /// paired with a zero right-hand side — makes that scenario's update
    /// exactly zero while leaving the sparsity pattern untouched.
    pub fn fill(
        &self,
        states: &[Vec<Bus>],
        p_calc: &[Vec<f64>],
        q_calc: &[Vec<f64>],
        active: &[bool],
        values: &mut Vec<f64>,
    ) {
        values.clear();
        for s in 0..self.n_scenarios {
            if active[s] {
                self.block.fill_into(&states[s], &p_calc[s], &q_calc[s], values);
            } else {
                self.block.fill_identity_into(values);
            }
        }
    }
}

/// One scenario's outcome from [`solve_batch_block_diagonal`].
pub struct BdeScenarioResult {
    pub buses: Vec<Bus>,
    pub stats: SolveStats,
}

/// Runs Newton-Raphson on **all** scenarios at once through a single stacked
/// sparse factorization per iteration, with per-scenario convergence masking.
///
/// Scenarios converge at different iteration counts and some contingency
/// scenarios never converge at all; `plans/GPU_PLAN.md` §3 makes masking a
/// hard requirement rather than an optimization. A scenario that has converged
/// (or gone singular, or hit `max_iter`) is masked out by writing an identity
/// into its block — its `Δx` is then exactly zero and it stops moving, while
/// every other scenario continues undisturbed and the stacked matrix keeps the
/// pattern its cached symbolic factorization was built for.
///
/// Returns one result per scenario, in scenario order.
///
/// Unlike [`crate::batch::BatchSolver`] this does **not** partition islands —
/// it is an architecture validator for the GPU path, not a production entry
/// point. Use `BatchSolver` for real work on a CPU.
pub fn solve_batch_block_diagonal<S: LinearSolver>(
    buses_template: &[Bus],
    ybus: &YBusSparse,
    scenarios: &[Scenario],
    tol: f64,
    max_iter: usize,
) -> Vec<BdeScenarioResult> {
    let nb = scenarios.len();
    if nb == 0 {
        return Vec::new();
    }

    let non_slack_idx: Vec<usize> = buses_template
        .iter()
        .filter(|b| !matches!(b.bus_type, BusType::Slack))
        .map(|b| b.idx)
        .collect();
    let pq_idx: Vec<usize> = buses_template
        .iter()
        .filter(|b| matches!(b.bus_type, BusType::PQ))
        .map(|b| b.idx)
        .collect();
    let n_angle = non_slack_idx.len();
    let blk = n_angle + pq_idx.len();

    // Per-scenario starting state, built exactly the way `BatchSolver` builds
    // it, so the two are comparable scenario for scenario.
    let mut states: Vec<Vec<Bus>> = scenarios
        .iter()
        .map(|sc| {
            let mut buses = buses_template.to_vec();
            for ov in &sc.bus_overrides {
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
            buses
        })
        .collect();

    let pattern = BlockDiagonal::analyze(buses_template, ybus, nb);
    let mut cache: Option<S> = None;

    let mut active = vec![true; nb];
    let mut history: Vec<Vec<f64>> = vec![Vec::new(); nb];
    let mut status = vec![SolveStatus::MaxIterationsReached; nb];

    let mut values: Vec<f64> = Vec::with_capacity(pattern.len());
    let mut rhs: Vec<f64> = vec![0.0; pattern.n_unknowns()];
    let mut p_all: Vec<Vec<f64>> = vec![Vec::new(); nb];
    let mut q_all: Vec<Vec<f64>> = vec![Vec::new(); nb];

    for _ in 0..max_iter {
        rhs.iter_mut().for_each(|v| *v = 0.0);

        for s in 0..nb {
            // Injections are recomputed even for masked scenarios: `fill`
            // ignores them, but keeping the arrays sized and valid avoids a
            // second code path, and the cost is bounded by how quickly
            // scenarios drop out.
            let (p_calc, q_calc) = power_injections(&states[s], ybus);
            if !active[s] {
                p_all[s] = p_calc;
                q_all[s] = q_calc;
                continue;
            }

            let base = s * blk;
            let mut max_mis = 0.0f64;
            for (r, &i) in non_slack_idx.iter().enumerate() {
                let (p_eff, _) = effective_injection(&states[s][i]);
                let v = p_eff - p_calc[i];
                rhs[base + r] = v;
                max_mis = max_mis.max(v.abs());
            }
            for (r, &i) in pq_idx.iter().enumerate() {
                let (_, q_eff) = effective_injection(&states[s][i]);
                let v = q_eff - q_calc[i];
                rhs[base + n_angle + r] = v;
                max_mis = max_mis.max(v.abs());
            }

            history[s].push(max_mis);
            if max_mis < tol {
                status[s] = SolveStatus::Converged;
                active[s] = false;
                // Zero this block's rhs so the masked identity yields Δx = 0.
                rhs[base..base + blk].iter_mut().for_each(|v| *v = 0.0);
            }

            p_all[s] = p_calc;
            q_all[s] = q_calc;
        }

        if active.iter().all(|&a| !a) {
            break;
        }

        pattern.fill(&states, &p_all, &q_all, &active, &mut values);

        if cache.is_none() {
            cache = S::new(pattern.n_unknowns(), &pattern.to_triplets(&values));
        }
        let Some(system) = cache.as_mut() else {
            for s in 0..nb {
                if active[s] {
                    status[s] = SolveStatus::Singular;
                }
            }
            break;
        };
        let Some(dx) = system.factor_and_solve_values(&values, &rhs) else {
            for s in 0..nb {
                if active[s] {
                    status[s] = SolveStatus::Singular;
                }
            }
            break;
        };

        for s in 0..nb {
            if !active[s] {
                continue;
            }
            let base = s * blk;
            for (r, &i) in non_slack_idx.iter().enumerate() {
                states[s][i].voltage_ang += dx[base + r];
            }
            for (r, &i) in pq_idx.iter().enumerate() {
                states[s][i].voltage_mag += dx[base + n_angle + r];
            }
        }
    }

    states
        .into_iter()
        .zip(history)
        .zip(status)
        .map(|((buses, mismatch_history), status)| BdeScenarioResult {
            buses,
            stats: SolveStats {
                status,
                mismatch_history,
                q_limit_switches: Vec::new(),
                q_limit_stabilized: true,
            },
        })
        .collect()
}
