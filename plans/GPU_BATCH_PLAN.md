# Batched AC power flow on NVIDIA GPUs

Status: **plan**, 2026-10-28, against `2b0c730`. Supersedes the `gpu-nvidia` branch's
`plans/GPU_PLAN.md`, whose lessons are in [What we know](#what-we-know).

The goal: solve a batch of AC power flows (an operating-point sweep, N-1) on one NVIDIA GPU,
**every scenario to its own convergence**, faster than gpusim2grid on the same GPU, with the
code in its own crate. The method: cuDSS's uniform batch for the sparse LU, and
`CUDSS_DATA_UBATCH_MASK` so that a scenario that has converged stops costing factorizations.

## What we know

### The old `gpu-nvidia` branch

Eleven commits on a base from before the three-crate split, so it cannot be rebased. It tried
block-diagonal embedding (one huge matrix), then cuDSS's non-uniform batch API
(`cudssMatrixCreateBatchCsr`); both stayed 4-9x behind gridoxide's 30-thread CPU
`BatchSolver` on an A100, because symbolic analysis ran per matrix. Its last attempt at the
uniform batch looped `CUDSS_CONFIG_UBATCH_INDEX` over the systems one at a time and got wrong
numbers for every index but 0. Worth keeping from it:

- the CUDA kernels in `cuda/gridoxide_kernels.cu` (Jacobian assembly, injections, mismatch,
  update) and the scatter maps of `src/device_layout.rs`;
- `build.rs`'s nvcc handling (the `NVCC` environment variable, not `cc::Build::compiler`) and
  the run path to cuDSS's libraries;
- cuDSS's OpenMP threading layer (`cudssSetThreadingLayer`), about 3.2x on analysis;
- its three FFI bugs, all hand-written signatures that did not match the headers.

### gpusim2grid (Grid2op/gpusim2grid, `416ae9f`, in grid-bench's GPU tabs)

It gets the uniform batch right, and its shape is the one to copy
(`contingency/strategies/cudss_batch_solver.cuh`): **one plain** `cudssMatrixCreateCsr`
whose value buffer holds all B systems (`B·nnz`), dense right-hand side and solution as
`n × 1` column-major descriptors over `B·n` buffers, `CUDSS_CONFIG_UBATCH_SIZE = B` before one
analysis, then factorization, refactorization and solve on the whole batch in single calls.
No per-index loop.

What it leaves on the table (grid-bench, A100):

- **A fixed step count.** Its sweep runs `nb_iter = 4` Newton steps with no convergence test,
  so the host never waits on the device, and fails the oracle where 4 are not enough:
  mvlv1004#sweep (25 of 100 scenarios at up to 3.3e-3 MVA) and case1354pegase#sweep-1k.
- **Analysis on every call**, 10% of its time; the session rebuilds its solver per run.
- The sparse LU is 78% of its time (cuDSS refactorizations and solves); every other kernel
  together is under 1%. Transfers are not the cost.
- On 100-scenario sweeps it is slower than lightsim2grid on 30 CPU threads (0.96 against
  0.27 ms per scenario on case1354pegase): a GPU pays only at scale.

### `CUDSS_DATA_UBATCH_MASK`

Present in cuDSS 0.8.0 and 1.0.0, set with `cudssDataSet`. Per NVIDIA's documentation: a
**host** array of one `int` per system, non-zero selects; honoured by factorization and
solve; an unselected system's solution is left exactly as it was; the mask must be set
before the analysis or the first factorization (its device storage is allocated then), so
start with all ones; a solve may only select systems some factorization since the last
analysis covered; mutually exclusive with `UBATCH_INDEX`; not with the BTF/COLAMD
reorderings, hybrid or multi-GPU modes.

Unverified, and the design depends on both: whether masked work **costs less** in
proportion to the systems selected, and whether `REFACTORIZATION` (not only
`FACTORIZATION`) honours the mask. Phase 0 measures them.

## Beating a fixed step count

1. **Stop per scenario.** After each step a kernel computes ‖F‖∞ per scenario and writes a
   flag: active, converged, or diverged (non-finite or growing). The flags (B ints) come
   back to the host, one small copy and one synchronize per step, and become the mask of
   the next refactorization and solve. The loop ends when nothing is active or at
   `max_iter`; a scenario that did not converge is reported as such, never hidden. This
   alone fixes gpusim2grid's two oracle failures.
2. **Stop paying for converged scenarios.** Jacobian, injection and update kernels run over
   a compacted list of active scenarios; the masked cuDSS calls skip the rest, if phase 0
   shows masking saves work. Honest expectation: on sweeps from a solved base case most
   scenarios take 3-4 steps either way, so the gain there is correctness; it is in uneven
   batches (N-1, wide sweeps, heavily loaded feeders) that a fixed count must cover the
   worst scenario.
3. **Reuse factors (experiment).** Refactor only scenarios whose mismatch is not
   contracting fast enough; solve every active one against its latest factors. The mask
   lets a refactorization and a solve select different sets. Fewer factorizations, the
   dominant cost, for possibly more steps; kept only if the oracle and grid-bench agree.
4. **Hand the stragglers to the CPU.** When only a few scenarios remain active the GPU is
   idle; below a threshold, copy their states back and finish them on the CPU
   `BatchSolver`.
5. **Analyse once per network**, not per call: the session keeps its cuDSS analysis, as
   `solve_batch` keeps its workers' solvers since #26.

## The crate

- **`crates/gridoxide-gpu`**, depending on `gridoxide-core` (buses, Y-bus,
  `JacobianPattern`), owning all CUDA and cuDSS code, NVIDIA only.
- **Bindings generated** with bindgen from the installed `cudss.h`, as gridoxide already
  does for KLU, HiGHS and IPOPT; no hand-written signatures.
- **Build:** nvcc through `cc` with the old branch's fixes; cuDSS found through
  `CUDSS_DIR`, with a run path to its libraries.
- **A workspace member but not a default member,** so `cargo build`, `cargo test` and CI
  never need CUDA; `gridoxide` gets a `gpu` feature that re-exports it.
- **Python:** a `GpuBatchSolver` (or `solve_batch(device="gpu")`) behind that feature, with
  `solve_batch`'s scenario format and `start="last"`.
- **Kernels** ported from the old branch, the Jacobian fill rewritten to mirror today's
  branch-free design: compute per-pair values into slots, then gather into storage (CSR)
  order.

## Phases

0. **Probe** ([`probes/cudss_ubatch`](../probes/cudss_ubatch/README.md), a rented A100, one
   session). Real Jacobians of case1354pegase, case2869pegase and case9241pegase, B up to
   2048, cuDSS 0.8.0 and 1.0.0: (A) the uniform batch matches the CPU; (B) the mask's
   semantics; (C) whether masked work scales with the selection; (D) the per-step cost of
   driving the mask. **Gate:** if (A) fails, stop; if (C) shows no saving, drop items 2 and 3
   above and keep per-scenario stopping and the CPU hand-off.
1. **Crate skeleton:** bindgen, build script, a stub that builds without CUDA, workspace and
   feature wiring.
2. **Device data and kernels:** CSR pattern, batched buffers, injections, mismatch, Jacobian
   fill, update, each checked against the CPU on one Newton step.
3. **Adaptive Newton loop:** flags, mask, all-converged exit, divergence handling, analysis
   kept per session.
4. **Stragglers to the CPU,** then the factor-reuse experiment, each kept only if grid-bench
   shows it pays.
5. **Python API, tests, grid-bench:** every scenario against the CPU `BatchSolver` to 1e-8, a
   GPU Docker image, and the GPU tabs against gpusim2grid and ExaPF.jl on the same A100.

## Targets

- **Correct:** every scenario of every sweep and N-1 case passes grid-bench's oracle, where
  gpusim2grid fails two.
- **Fast:** faster than gpusim2grid on `#sweep-1k`, `#sweep-10k` and `#n1-full`, same GPU.
  On 100-scenario batches, beating the 30-thread CPU is unlikely and not a target.

## Risks

- **Masked work may not shrink** with the selection, which removes the main speed-up
  (phase 0, C).
- **The host-side mask forces a sync per step.** Small next to a factorization (phase 0, D),
  but it gives up gpusim2grid's sync-free loop.
- **cuDSS's licence** governs what an image may redistribute; grid-bench's gpusim2grid image
  already ships its runtime libraries.
- **No NVIDIA GPU on the development machine:** every phase from 0 needs a rented one, and
  the probe's local syntax check (`probes/cudss_ubatch/stub`) is what keeps signature
  mistakes off that clock.
