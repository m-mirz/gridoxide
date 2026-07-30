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
//! **That property is real, and it is also the trap.** Stacking is
//! *mathematically* equivalent to B independent solves (proof below) but it is
//! not *computationally* equivalent on a GPU: a general sparse direct solver
//! handed one 10-million-row matrix has no way to know it is really B
//! independent 2,450-row problems, and pays scheduling and bookkeeping over the
//! whole thing. Measured, that cost was ~95% of the device-resident path's
//! runtime and left it ~91x slower than the 30-thread CPU
//! [`BatchSolver`](crate::batch::BatchSolver). On NVIDIA the fix is to stop
//! stacking and use cuDSS's uniform batch entry point instead — see
//! [`solve_batch_block_diagonal_batched_device`], which is the path to use, and
//! [`crate::sparse_cudss::CudssBatchedSystem`] for the mechanism. The stacking
//! functions here are kept as the A/B control that makes that claim measurable.
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

use rayon::prelude::*;

use crate::batch::Scenario;
use crate::jacobian::JacobianPattern;
use crate::network::{effective_injection, linear_initial_guess, power_injections, YBusSparse};
use crate::solver::{LinearSolver, SolveStats, SolveStatus};
use crate::types::{Bus, BusType};

/// Applies each scenario's overrides to the shared template and computes its
/// starting point, exactly the way [`crate::batch::BatchSolver`] does, so the
/// two are comparable scenario for scenario.
///
/// **Parallel across scenarios**, because `linear_initial_guess` is a full
/// linearized sparse solve per scenario — at batch 4,096 that is thousands of
/// sparse solves, and running them on one thread put a large serial region in
/// front of every GPU path's Newton loop while the CPU baseline it is measured
/// against spread the identical work over every core. Ordering is preserved:
/// `par_iter().map().collect()` is deterministic.
fn build_states(buses_template: &[Bus], ybus: &YBusSparse, scenarios: &[Scenario]) -> Vec<Vec<Bus>> {
    scenarios
        .par_iter()
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
        .collect()
}

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

/// Packs per-scenario states, mismatch traces and statuses into results.
fn collect_results(
    states: Vec<Vec<Bus>>,
    history: Vec<Vec<f64>>,
    status: Vec<SolveStatus>,
) -> Vec<BdeScenarioResult> {
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

/// Every scenario `Singular`, with its starting state — the outcome when
/// device setup fails before a single Newton iteration can run. A GPU that
/// isn't there is reported the same way a singular matrix is, rather than
/// panicking, so a batch never takes the process down.
#[cfg(all(feature = "gpu", feature = "cudss"))]
fn singular_results(states: Vec<Vec<Bus>>, nb: usize) -> Vec<BdeScenarioResult> {
    collect_results(states, vec![Vec::new(); nb], vec![SolveStatus::Singular; nb])
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

    let mut states = build_states(buses_template, ybus, scenarios);

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

    collect_results(states, history, status)
}

/// Device-resident counterpart to [`solve_batch_block_diagonal`]: the
/// Jacobian is assembled directly into cuDSS's own device buffer by
/// [`crate::gpu::GpuAssembler`] and never round-trips through host memory —
/// `plans/GPU_PLAN.md` §6 Phase 3's actual payoff, vs. the host-resident path
/// both `solve_batch_block_diagonal` and `sparse_cudss::CudssRealSystem::
/// factor_and_solve_values` still use (values assembled on the CPU, or
/// assembled on the GPU then downloaded and re-uploaded — either way a full
/// host↔device round trip of the batch's `nnz`-sized array every iteration).
///
/// Only the mismatch/right-hand-side and solution vectors still cross that
/// boundary each iteration, and deliberately so: they are `O(n_unknowns)`,
/// not `O(nnz)` — `plans/GPU_PLAN.md` §1 measured Jacobian assembly at
/// 35-41% of iteration time and mismatch evaluation at only 4-6%, so this is
/// exactly the split worth eliminating a round trip for and the split not
/// worth the extra complexity of a GPU mismatch kernel.
///
/// Masking is real here, not simplified away: `active` feeds the same GPU
/// kernel that assembles the real Newton values (see `gpu::assemble_kernel`'s
/// doc comment), writing an identity block for any converged/diverged
/// scenario — required for correctness, not just efficiency, since one
/// singular block would otherwise be free to fail the whole batch's shared
/// factorization.
///
/// **A known, investigated characteristic**: this path's per-scenario
/// converged voltages agree with an independent CPU solve to ~1e-12
/// (`bde_test.rs`'s `bde_device_resident_matches_independent`), but it
/// sometimes takes one or two more Newton iterations to get there than the
/// host-resident path does on identical data. Every gridoxide-side
/// explanation was ruled out directly (see `tests/gpu_assembly_test.rs`'s
/// device-resident checks): the GPU-assembled, CSR-scattered, masked values
/// are bit-identical to what the CPU would compute in the same order, and the
/// stacked-CSR offset arithmetic is exact, so cuDSS receives the identical
/// numbers either way. Neither an explicit `cudssMatrixSetValues`
/// notification nor `CUDSS_CONFIG_DETERMINISTIC_MODE` changed it. The leading
/// unconfirmed explanation is that cuDSS's own factorization takes a
/// measurably different (internally consistent, still correct) path depending
/// on which allocator provided the values buffer.
///
/// **Status: superseded.** This is now the A/B *control* for
/// [`solve_batch_block_diagonal_batched_device`], not the recommended path —
/// see that function for why handing cuDSS one stacked matrix instead of a
/// uniform batch costs ~95% of the runtime. It is kept because a performance
/// claim needs something to be measured against; `examples/bde_profile.rs`
/// runs both.
#[cfg(all(feature = "gpu", feature = "cudss"))]
pub fn solve_batch_block_diagonal_device_resident(
    buses_template: &[Bus],
    ybus: &YBusSparse,
    scenarios: &[Scenario],
    tol: f64,
    max_iter: usize,
) -> Vec<BdeScenarioResult> {
    use crate::gpu::GpuAssembler;
    use crate::sparse_cudss::{build_csr_structure, csr_scatter_map, CudssRealSystem};

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

    let mut states = build_states(buses_template, ybus, scenarios);

    let pattern = BlockDiagonal::analyze(buses_template, ybus, nb);

    // The GPU kernel writes each entry to its CSR position directly — computed
    // once, from a single block's (row, col) pairs, since every scenario's
    // block shares the same relative structure (`csr_scatter_map`'s own doc
    // comment explains why one single-block map serves every scenario).
    let block_pairs: Vec<(usize, usize)> =
        pattern.block.rows().iter().zip(pattern.block.cols()).map(|(&r, &c)| (r as usize, c as usize)).collect();
    let scatter = csr_scatter_map(pattern.block.n_unknowns, &block_pairs);

    let Some(mut assembler) = GpuAssembler::new(&pattern.block, buses_template.len()) else {
        return singular_results(states, nb);
    };
    if assembler.set_scatter(&scatter).is_none() {
        return singular_results(states, nb);
    }

    // The *stacked* CSR structure cuDSS's matrix wraps — built once, from
    // the full block-diagonal (row, col) pairs (already correctly offset by
    // `BlockDiagonal::analyze`), exactly like the host-resident path builds
    // it from `pattern.to_triplets(&values)` inside `CudssRealSystem::new`.
    let full_pairs: Vec<(usize, usize)> = pattern.rows.iter().zip(&pattern.cols).map(|(&r, &c)| (r as usize, c as usize)).collect();
    let (row_ptr, col_idx, _groups) = build_csr_structure(pattern.n_unknowns(), &full_pairs);

    let mut active = vec![true; nb];
    let mut history: Vec<Vec<f64>> = vec![Vec::new(); nb];
    let mut status = vec![SolveStatus::MaxIterationsReached; nb];

    let mut rhs: Vec<f64> = vec![0.0; pattern.n_unknowns()];
    let mut p_all: Vec<Vec<f64>> = vec![Vec::new(); nb];
    let mut q_all: Vec<Vec<f64>> = vec![Vec::new(); nb];
    // Constructed on the first iteration, once the GPU assembler has written
    // valid initial values into the buffer cuDSS's matrix will point at —
    // see `CudssRealSystem::new_device_resident`'s precondition.
    let mut cudss: Option<CudssRealSystem> = None;

    for _ in 0..max_iter {
        rhs.iter_mut().for_each(|v| *v = 0.0);

        for s in 0..nb {
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
                rhs[base..base + blk].iter_mut().for_each(|v| *v = 0.0);
            }

            p_all[s] = p_calc;
            q_all[s] = q_calc;
        }

        if active.iter().all(|&a| !a) {
            break;
        }

        // Assemble directly into the persistent device buffer — no host
        // round trip for the (large) Jacobian values, masking active[]
        // scenarios via the kernel's identity fallback.
        if assembler.assemble_batch_masked(&states, &p_all, &q_all, &active).is_none() {
            for s in 0..nb {
                if active[s] {
                    status[s] = SolveStatus::Singular;
                }
            }
            break;
        }
        // Required barrier, not a performance nicety: `CudssRealSystem` runs
        // on the default stream while the assembler runs on its own, so
        // cuDSS's refactorization must not be enqueued before the kernel
        // above has finished writing the buffer it reads. (The batched path
        // shares one stream with cuDSS and needs nothing here — see
        // `solve_batch_block_diagonal_batched_device`.)
        if assembler.stream().synchronize().is_none() {
            for s in 0..nb {
                if active[s] {
                    status[s] = SolveStatus::Singular;
                }
            }
            break;
        }

        if cudss.is_none() {
            let Some(ptr) = assembler.values_ptr() else {
                for s in 0..nb {
                    if active[s] {
                        status[s] = SolveStatus::Singular;
                    }
                }
                break;
            };
            cudss = CudssRealSystem::new_device_resident(pattern.n_unknowns(), &row_ptr, &col_idx, ptr);
        }
        let Some(system) = cudss.as_mut() else {
            for s in 0..nb {
                if active[s] {
                    status[s] = SolveStatus::Singular;
                }
            }
            break;
        };
        let Some(dx) = system.solve_device_resident(&rhs) else {
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

    collect_results(states, history, status)
}

/// **The batched GPU path**: every scenario's Jacobian, mismatch, linear solve
/// and Newton update stays on the device, and the linear solve goes through
/// cuDSS's *uniform batch* API rather than one stacked block-diagonal matrix.
///
/// # Why this exists
///
/// [`solve_batch_block_diagonal_device_resident`] already removed the
/// Jacobian's host round trip and was still ~91x slower than the 30-thread CPU
/// [`BatchSolver`](crate::batch::BatchSolver) on case1354pegase at batch 4,096.
/// The round trip was never the problem: at that batch size the serial host
/// mismatch loop is ~1.5% of an iteration, the Jacobian upload another ~1.5%,
/// and ~95% is inside cuDSS refactorizing and solving a single 10-million-row
/// stacked matrix. `plans/GPU_PLAN.md` §3 property 2 chose that stacking
/// deliberately — it needs no batched solver API, which is what kept the
/// AMD/rocSOLVER path open — but a general sparse direct solver handed one
/// enormous matrix cannot exploit the fact that it is really B independent
/// 2,450-row problems. See [`crate::sparse_cudss::CudssBatchedSystem`] for the
/// mechanism.
///
/// # What changed against the control path
///
/// | | device-resident (control) | batched (this) |
/// |---|---|---|
/// | Linear solve | one `B*blk`-row general matrix | `B` uniform batched systems |
/// | Symbolic analysis | on `B*blk` rows | on `blk` rows, once |
/// | CSR structure uploaded | `O(n_total)` + `O(nnz_total)` | one block, shared by all `B` |
/// | Mismatch evaluation | serial host loop, `B` calls/iteration | one kernel |
/// | Host↔device per iteration | `rhs` down + `Δx` up, `O(B*blk)` each | `B` f64 down, `B` u32 up |
/// | Synchronization | stream sync per iteration, after a whole-batch stall | one, on the convergence copy |
///
/// The last two rows are why the loop below looks so different from its
/// siblings: there is no `rhs` vector, no `p_all`/`q_all`, and no `Δx` on the
/// host at all. The only host-visible per-iteration quantity is
/// `max_mismatch[s]`, because whether a scenario has converged is a decision
/// only the host can make — and the convergence, masking, `mismatch_history`
/// and `SolveStatus` bookkeeping around it is deliberately kept identical to
/// [`solve_batch_block_diagonal`]'s, so the two stay comparable line for line.
///
/// Masking works exactly as in the other paths: a converged or diverged
/// scenario gets an identity block from the assembly kernel and a zeroed
/// right-hand side, so its Δx is exactly zero. That is a correctness
/// requirement, not an optimization — one singular block is otherwise free to
/// fail the whole batch's factorization.
///
/// Returns every scenario as `Singular` if the device cannot be set up at all.
#[cfg(all(feature = "gpu", feature = "cudss"))]
pub fn solve_batch_block_diagonal_batched_device(
    buses_template: &[Bus],
    ybus: &YBusSparse,
    scenarios: &[Scenario],
    tol: f64,
    max_iter: usize,
) -> Vec<BdeScenarioResult> {
    use crate::device_layout::{build_csr_structure, csr_scatter_map};
    use crate::gpu::GpuBatch;
    use crate::sparse_cudss::CudssBatchedSystem;

    let nb = scenarios.len();
    if nb == 0 {
        return Vec::new();
    }

    let mut states = build_states(buses_template, ybus, scenarios);

    // One block, not a stacked matrix — that is the whole point. Every
    // scenario shares this pattern, this scatter map and this CSR structure,
    // so all three are `O(one block)` regardless of batch size.
    let block = JacobianPattern::analyze(buses_template, ybus);
    let blk = block.n_unknowns;
    let block_pairs: Vec<(usize, usize)> =
        block.rows().iter().zip(block.cols()).map(|(&r, &c)| (r as usize, c as usize)).collect();
    let scatter = csr_scatter_map(blk, &block_pairs);
    let (row_ptr, col_idx, _groups) = build_csr_structure(blk, &block_pairs);

    let Some(mut gpu) = GpuBatch::new(&block, ybus, buses_template, &states, &scatter) else {
        return singular_results(states, nb);
    };

    let mut active = vec![true; nb];
    let mut history: Vec<Vec<f64>> = vec![Vec::new(); nb];
    let mut status = vec![SolveStatus::MaxIterationsReached; nb];
    let mut max_mismatch = vec![0.0f64; nb];
    // Constructed on the first iteration, once the assembly kernel has been
    // enqueued into the buffer cuDSS's batched matrix points at — see
    // `CudssBatchedSystem::new`'s precondition. Because it is handed this
    // object's stream, its own analysis and factorization are ordered after
    // that kernel with no host synchronization.
    let mut cudss: Option<CudssBatchedSystem> = None;

    // Runs the whole Newton loop; `None` anywhere means a CUDA or cuDSS call
    // failed, and every still-active scenario is reported `Singular` — the
    // same contract the CPU paths give a singular factorization.
    let outcome = (|| -> Option<()> {
        for _ in 0..max_iter {
            gpu.power_injections()?;
            gpu.mismatch()?;
            // The loop's one synchronization point, and its only
            // device-to-host transfer: one f64 per scenario.
            gpu.download_max_mismatch(&mut max_mismatch)?;

            for s in 0..nb {
                if !active[s] {
                    continue;
                }
                history[s].push(max_mismatch[s]);
                if max_mismatch[s] < tol {
                    status[s] = SolveStatus::Converged;
                    active[s] = false;
                }
            }
            if active.iter().all(|&a| !a) {
                return Some(());
            }

            gpu.upload_active(&active)?;
            // After the mask update, so a scenario that converged on *this*
            // iteration gets Δx = 0 rather than one more step.
            gpu.zero_masked_rhs()?;
            gpu.assemble()?;

            if cudss.is_none() {
                cudss = CudssBatchedSystem::new(
                    nb,
                    blk,
                    &row_ptr,
                    &col_idx,
                    gpu.values_ptr(),
                    gpu.rhs_ptr(),
                    gpu.dx_ptr(),
                    gpu.stream().as_u64(),
                );
            }
            cudss.as_mut()?.refactor_and_solve()?;
            gpu.apply_update()?;
        }
        Some(())
    })();

    if outcome.is_none() {
        for s in 0..nb {
            if active[s] {
                status[s] = SolveStatus::Singular;
            }
        }
    }

    // The one readback of the whole solve. Best-effort: if it fails, the
    // states still hold each scenario's starting point and the statuses above
    // already say the solve did not complete.
    let _ = gpu.download_voltages_into(&mut states);

    collect_results(states, history, status)
}
