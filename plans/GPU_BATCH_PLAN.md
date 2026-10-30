# Batched AC power flow on NVIDIA GPUs

Status: **plan**, 2026-10-28, against `2b0c730`; phases 0-4 done on an A100, 2026-10-29,
phase 5 on 2026-10-30, grid-bench's GPU tabs included ([Phase 0 on an A100](#phase-0-on-an-a100),
[phases](#phases)).
**The GPU beats 30 CPU threads only on case9241pegase's sweeps** (0.85-0.90x the time);
everywhere else, N-1 included, it takes 1.9-5x the CPU's time ([phase 5](#phases)). **cuDSS 0.8.0 computes wrong
solutions** in a uniform batch (see [the correction](#correction-a-fails-on-080)); only
1.0.0 is fit to use. Supersedes the `gpu-nvidia` branch's
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
`FACTORIZATION`) honours the mask. Phase 0 measured them, and found 0.8.0 departing from the
documentation: [Phase 0 on an A100](#phase-0-on-an-a100).

## Phase 0 on an A100

Run on 2026-10-29 on an A100-SXM4-40GB (driver 580, CUDA 12.8), cuDSS 0.8.0 and 1.0.0, with
`probes/cudss_ubatch/run.sh`; its follow-up `explore.cpp` answers what the probe's first run
raised. Its raw output goes to `probes/cudss_ubatch/results/`, which is not tracked; the
numbers that matter are in the tables below.

- **A, the uniform batch: passes in 1.0.0, fails in 0.8.0** (as first reported here: passes
  in both; see the correction below). In 1.0.0 every system of case1354pegase,
  case2869pegase and case9241pegase matches the CPU to 1e-12 at every B from 64 to 2048,
  after the factorization and after a refactorization on new values. The gate is passed,
  for 1.0.0.
- **B, the mask: as documented in 1.0.0, not in 0.8.0.** In both, a masked
  `REFACTORIZATION` is accepted and refactors the selected systems correctly, without
  reading the others' values (1e300 there does no harm). But in 0.8.0, at every B, a
  factorization **drops the factors of the systems it does not select**: they solve to
  zeros until refactored, and a `REFACTORIZATION` restores them. And 0.8.0's masked solve
  writes every system's solution buffer, the unselected ones from cuDSS's own copy of their
  last solution. 1.0.0 keeps the factors and leaves the unselected solutions untouched.
- **C, masked cost: it depends on the version** (tables below). 0.8.0's mask saves only the
  part of a step's cost above a large fixed cost: one selected system of 2048 still costs
  half the full batch. 1.0.0's scales: one of 2048 costs 9-12% of the full batch.
- **D, the mask's price: 16-22 µs per step** (B flags device to host, a synchronize,
  `cudssDataSet`), at every B. Nothing next to a factorization.

Refactorization and solve of all B systems, ms (median of 9, `explore time`):

| B | case1354pegase 0.8.0 | 1.0.0 | case9241pegase 0.8.0 | 1.0.0 |
|---|---|---|---|---|
| 1 | 0.41 | 0.49 | 1.00 | 0.98 |
| 15 | 1.17 | 1.44 | 5.54 | 4.28 |
| 16 | 16.1 | 1.51 | 122 | 5.66 |
| 64 | 15.4 | 5.21 | 134 | 20.0 |
| 256 | 18.5 | 19.9 | 150 | 61.5 |
| 1024 | 25.3 | 63.1 | 206 | 244 |
| 2048 | 34.0 | 126 | 277 | 486 |

The same at B = 2048 with 2048, 205 and 1 systems selected (probe C, contiguous):

| | case1354pegase 0.8.0 | 1.0.0 | case9241pegase 0.8.0 | 1.0.0 |
|---|---|---|---|---|
| 2048 | 33.9 | 126 | 277 | 485 |
| 205 | 18.2 | 22.2 | 148 | 100 |
| 1 | 16.7 | 10.8 | 131 | 57.7 |

- **0.8.0 has two regimes, switching at exactly B = 16** on both cases: up to 15 systems
  cost little more than one; from 16 on, a step has a fixed cost of about 15 ms
  (case1354pegase) or 120 ms (case9241pegase), above which it grows slowly, to 0.017 and
  0.135 ms per system at B = 2048. None of the knobs tried moves the switch: reordering
  (default, AMD), factorization algorithm, pivoting off, superpanels off, nested-dissection
  levels, deterministic mode, cuDSS's threading layer.
- **1.0.0 has no switch**: its cost is about linear in B, 0.06 and 0.24 ms per system, 3.7x
  and 1.75x 0.8.0's at B = 2048. At B = 4096 case9241pegase runs out of the A100's 40 GB.
- **0.8.0 is not safe with two handles at once in one process.** With the device tests run
  in parallel threads, each with its own handle and stream, some outages' solutions come
  back NaN; never with one thread, never with 1.0.0. (A process-wide lock around each
  `cudssExecute` turned out not to be enough; see the correction.)
- The threading layer changes nothing for refactorization and solve. Analysis takes 30 ms
  (case1354pegase) to 90 ms (case9241pegase) in 0.8.0 and about 100 ms in 1.0.0, worth keeping
  per session (item 5 below).

What it means:

- **Per-scenario stopping works in both versions.** The active set only shrinks, and every
  step refactors and solves the same set, which is all 0.8.0's mask supports.
- **Factor reuse needs 1.0.0**, whose unselected systems keep their factors.
- **Superseded by the correction below:** the estimate that follows assumed 0.8.0 computes
  correct solutions. Rough per-batch LU costs for 1000 connected outages of
  N-1, from the costs above and the step profiles of [N-1](#n-1) (adaptive: about 2.2
  factorizations per outage, all active for two steps; reuse: about 1.2 factorizations and
  3.5 solves per outage). These are estimates, not runs:

  | | 0.8.0 fixed 4 | 0.8.0 adaptive | 1.0.0 adaptive | 1.0.0 reuse < 0.1 |
  |---|---|---|---|---|
  | case1354pegase | ≈100 ms | ≈70 ms | ≈140 ms | ≈125 ms |
  | case9241pegase | ≈820 ms | ≈535 ms | ≈600 ms | ≈515 ms |

  On those numbers 0.8.0 with adaptive stopping looked best or close to it on both.
- **On 0.8.0 the tail costs the fixed cost.** Once few scenarios remain active, a step still
  costs 15-120 ms, so the CPU hand-off (item 4) matters more than the CPU counts suggest.

### Correction: A fails on 0.8.0

Found in phase 3, when the device loop reported 115 of a 1000-scenario case1354pegase sweep
as diverged where the CPU converges every one. The probe's comparison was blind to NaN
(`std::max(err, NaN)` keeps `err`), so probe A reported a pass while 0.8.0 returned NaN
solutions. Fixed, it gives (A, first factorization; the refactorization alike):

| dump | B | 0.8.0 | 1.0.0 |
|---|---|---|---|
| case1354pegase, 300 sweep scenarios | 300 | 39 systems NaN or wrong, from index 217 | pass |
| case1354pegase, 512 distinct | 512 | 70, from 216 | pass |
| case1354pegase, 64 tiled | 1024 | 84, from 218 | pass |
| case1354pegase, 64 tiled | 2048 | 1, index 336 | pass |
| case2869pegase, case9241pegase, 64 tiled | 1024, 2048 | pass | pass |

Which systems fail changes from run to run, never below index 216 (twice the A100's 108
SMs). The same sequence of calls from Rust and from C++ fails alike once the comparison
sees NaN. Separately, 0.8.0 also returns NaNs when other GPU work runs alongside it in the
process (device tests in parallel threads), even with its calls serialized. Neither fault
is something a caller can work around reliably: **0.8.0 is not fit for the uniform batch,
and 1.0.0 is the version to use.** gpusim2grid runs 0.8.0, so its grid-bench results on
case1354pegase deserve a second look.

Decided on 2026-10-29: gridoxide-gpu supports cuDSS 1.0 or later only. Its build script
refuses an older `cudss.h`; the 0.8.0 stand-in, the per-version factor rule, the lock and
`BatchError::ReuseUnsupported` are gone. `tests/cudss_test.rs`'s
`a_large_batch_matches_the_stand_in` (512 scenarios of case1354pegase) guards against a
cuDSS that fails past index 216 again, which batches of 64 cannot see. The probe still runs
0.8.0, to reproduce all this.
  The device alternative would be a second analysis at B = 15 for the stragglers (1.2 and
  5.5 ms per step).

## Beating a fixed step count

1. **Stop per scenario.** After each step a kernel computes ‖F‖∞ per scenario and writes a
   flag: active, converged, or diverged (non-finite or growing). The flags (B ints) come
   back to the host, one small copy and one synchronize per step, and become the mask of
   the next refactorization and solve. The loop ends when nothing is active or at
   `max_iter`; a scenario that did not converge is reported as such, never hidden. This
   alone fixes gpusim2grid's two oracle failures.
2. **Stop paying for converged scenarios.** Jacobian, injection and update kernels run over
   a compacted list of active scenarios; the masked cuDSS calls skip the rest. Phase 0: in
   1.0.0 the saving scales with the selection; in 0.8.0 it is only the part of a step above
   its fixed cost. Honest expectation: on sweeps from a solved base case most
   scenarios take 3-4 steps either way, so the gain there is correctness; it is in uneven
   batches (N-1, wide sweeps, heavily loaded feeders) that a fixed count must cover the
   worst scenario.
3. **Reuse factors (experiment).** Refactor only scenarios whose mismatch is not
   contracting fast enough; solve every active one against its latest factors. The mask
   lets a refactorization and a solve select different sets. Fewer factorizations, the
   dominant cost, for possibly more steps; kept only if the oracle and grid-bench agree.
   Phase 0: cuDSS 1.0.0 only, since 0.8.0 drops the factors of the systems a factorization
   leaves out.
4. **Hand the stragglers to the CPU.** When only a few scenarios remain active the GPU is
   idle; below a threshold, copy their states back and finish them on the CPU
   `BatchSolver`.
5. **Analyse once per network**, not per call: the session keeps its cuDSS analysis, as
   `solve_batch` keeps its workers' solvers since #26.

## Counts on the CPU, before any device

`crates/gridoxide-gpu` already runs the loop above over a CPU stand-in for the uniform batch
(`CpuUniformBatch`, which enforces the mask's documented rules), so the work each policy does
can be counted now: `cargo run --release -p gridoxide-gpu --example batch_counts -- <case.m>`.
Sweeps of grid-bench's kind (100 scenarios from the solved base case, a daily load curve
between 0.6 and 1.0, 5% noise per bus), tolerance 1e-6 p.u.:

| case | fixed 4: converged | adaptive: systems factored | reuse < 0.1: factored / solved / batch steps |
|---|---|---|---|
| case14 | 100 | 254 (−37%) | 100 / 379 / 5 |
| case1354pegase | 100 | 337 (−16%) | 170 / 488 / 6 |
| case2869pegase | 100 | 328 (−18%) | 164 / 479 / 7 |
| case9241pegase | 100 | 357 (−11%) | 213 / 516 / 7 |
| mvlv1004 | **60** | 401 (±0%, all 100 converge) | 282 / 479 / 7 |
| mvlv10616 | 100 | 351 (−12%) | 246 / 451 / 6 |

(The fixed policy factors and solves 400 systems in 4 batch steps.) What it means:

- **Correctness is the sure gain.** Four fixed steps leave 40 of mvlv1004's 100 scenarios
  short of the tolerance, the failure grid-bench shows for gpusim2grid; the adaptive loop
  converges all, for the same work.
- **Adaptive stopping saves 11-18% of factorizations on the transmission cases**, if masked
  device work scales with the selection (phase 0, C).
- **Factor reuse halves the factorizations against fixed steps** (−47 to −59%, −30% on
  mvlv1004), for 20-30% more solves and 5-7 batch steps instead of 4, each with its mask
  sync. gpusim2grid's own phase times put a refactorization at about 2.6 solves on
  case9241pegase (5.8 against 2.2 ms per step), which would make reuse about 26% cheaper
  than fixed steps there, adaptive stopping alone about 11%. Phase 0 replaces that estimate
  with measurements.
- **The CPU hand-off never triggered**: on these sweeps the scenarios converge together, with
  no tail of a few stragglers. It stays an option for uneven batches (N-1).

### N-1

`solve_contingency_batch` runs the same loop over outages: one that keeps the network
connected changes only the Y-bus values at the outaged branches' positions, not its pattern,
so every such outage is a member of the same uniform batch. Each scenario carries its
changed values as a delta on the intact Y-bus, applied around its mismatch and Jacobian
fill and restored exactly afterwards, as the CPU's `solve_contingencies` applies outages in
place; on the device the deltas (a few entries per outage) are what each scenario adds to the
shared Y-bus. An outage that splits the network has another pattern and is reported as
`Outcome::Splits`, for the CPU to solve island by island. Tested against the CPU's
`solve_contingencies` on case1354pegase: same steps, voltages to 1e-9, the same outages
split.

`batch_counts -- <case.m> 100 1e-6 n1`, 100 single-branch outages spread over the branch
list, from the solved base case:

| case | split | fixed 4: systems factored | adaptive: factored, max steps | reuse < 0.1: factored / solved / batch steps |
|---|---|---|---|---|
| case14 | 5 | 380 | 265 (−30%), 4 | 130 / 380 / 5 |
| case1354pegase | 22 | 312 | 202 (−35%), 3 | 94 / 295 / 6 |
| case2869pegase | 17 | 332 | 210 (−37%), 4 | 100 / 302 / 6 |
| case9241pegase | 8 | 368 | 217 (−41%), 3 | 106 / 302 / 5 |

Every connected outage converges under every policy. Adaptive stopping saves 30-41% of
factorizations here, against 11-18% on sweeps: most outages barely move the operating
point and converge in two or three steps (2.4-2.8 on average), where a fixed count pays
for the worst. Factor reuse cuts factorizations by 66-71% against fixed steps. The radial feeders (mvlv1004,
mvlv10616) split on every outage, as radial networks must, so their N-1 is not a batch
at all.

Every branch's outage (`batch_counts -- <case.m> 0 1e-6 n1`, in batches of 1000: the
stand-in keeps two copies of each scenario's Jacobian, so case9241pegase's 16049 outages in
one batch would need about 40 GB, and a device batch is bounded the same way):

| case | outages | split | fixed 4: short | factored: fixed 4 / adaptive / reuse < 0.1 | batch steps: adaptive / hand-off |
|---|---|---|---|---|---|
| case1354pegase | 1991 | 561 | 3 | 5720 / 3804 (−33%) / 1801 (−69%) | 20 / 3 (36 to the CPU) |
| case2869pegase | 4582 | 778 | 1 | 15216 / 9547 (−37%) / 4488 (−71%) | 5 / 3 (64 to the CPU) |
| case9241pegase | 16049 | 1665 | 30 | 57536 / 34953 (−39%) / 17344 (−70%) | 173 / 50 over 17 batches (312 to the CPU) |

(Batch steps for one batch of the whole list on the first two; case9241pegase ran as 17
batches of up to 1000, 3.4 GB peak, and its steps add up over them.)

- **Here fixed steps fail, and the full lists show it.** On case1354pegase four steps leave
  three outages short: branch 1325 converges in 5 adaptive steps; branches 75 and 1754 do
  not converge in 20, and neither does the CPU's `solve_contingencies` (mismatch 1.4e3 and
  1.9 p.u. after 20 steps; the CPU ends at 7.9e3 and 0.52). They are reported as not
  converged, where a fixed count reports nothing. On case9241pegase four steps leave 30
  outages short; the adaptive loop converges 20 of them and reports 10 as not converged in
  20 steps (not yet checked against the CPU one by one).
- **Factor reuse can change a hard outage's outcome.** With reuse < 0.3, two of
  case9241pegase's hard outages diverge instead of running out of steps; the converged
  set is the same. Reuse stays an experiment, with < 0.1 the safer threshold.
- **The CPU hand-off pays on N-1.** Unlike the sweeps, a few outages need many more steps
  than the rest: on case1354pegase, the batch would run 20 steps for two hopeless
  scenarios. Handing the last 5% to the CPU ends the batch after 3 steps, and cuts
  case9241pegase's batch steps from 173 to 50.

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
   above and keep per-scenario stopping and the CPU hand-off. *Done* (2026-10-29): (A)
   passes; (B) and (C) split by version, so item 3 needs 1.0.0, and item 2 saves more on
   1.0.0 than on 0.8.0; see [Phase 0 on an A100](#phase-0-on-an-a100).
1. **Crate skeleton:** bindgen, build script, a stub that builds without CUDA, workspace and
   feature wiring. *Done, untested on a device:* `crates/gridoxide-gpu` is a workspace
   member that `cargo publish --workspace` skips (`publish = false`), with the
   `UniformBatch` trait, the CPU stand-in and the adaptive loop (phase 3's logic), tested
   against the CPU `BatchSolver`. Behind the `cuda` feature, `build.rs` generates the
   bindings from `cudss.h` (`CUDSS_DIR`, `CUDA_HOME`) and `CudssUniformBatch` implements
   the trait in gpusim2grid's shape, with host copies of values, right-hand sides and
   solutions per call (phase 2 moves them onto the device). `REFACTORIZATION` stays off
   (`use_refactorization`) until probe B shows it honours the mask. It type-checks against
   cuDSS 0.8.0 and 1.0.0 through the probe's stub headers; `run.sh` runs its device tests
   (`tests/cudss_test.rs`: the mask contract, and the Newton loop step for step against
   the stand-in) after the probe. *Tested on the A100*: the backend uses `REFACTORIZATION`
   by default, and since the correction above builds against cuDSS 1.0 or later only.
2. **Device data and kernels:** CSR pattern, batched buffers, injections, mismatch, Jacobian
   fill, update, each checked against the CPU on one Newton step. *Done* (2026-10-29):
   `DeviceBatch` (`src/device.rs`, `cuda/kernels.cu`) holds the network and every
   scenario's state on the device, with the fill in gridoxide-core's two passes (per-bus
   and per-pair slots, then a gather into CSR order) and ‖F‖∞ reduced per scenario. N-1
   keeps the shared Y-bus: each outage's changed values drive two fix-up kernels that
   recompute only the rows' injections and the fill slots they touch. On case1354pegase
   and mvlv1004 (sweeps with ZIP loads and PV setpoints) and 116 connected outages of
   case1354pegase, every kernel matches the CPU to 6e-12 relative, the update is exact,
   and after one step every scenario's mismatch is the CPU loop's to 1e-11
   (`tests/device_test.rs`). The step's buffers still pass through the host to cuDSS;
   phase 3 hands them over on the device.
3. **Adaptive Newton loop:** flags, mask, all-converged exit, divergence handling, analysis
   kept per session. *Done* (2026-10-29): `GpuBatchSolver` (`src/gpu.rs`) is a session per
   network that keeps its cuDSS analysis and device buffers for `capacity` scenarios and
   solves any number in chunks of it, sweeps and N-1. cuDSS factors and solves
   `DeviceBatch`'s own buffers on its stream; per step the host reads back the B norms and
   decides with the CPU loop's own rule (`newton::decide`, shared), so the step's one
   synchronization carries the norms rather than flags. Factor reuse (1.0.0) and the CPU
   hand-off work as on the CPU. Against the CPU loop (`tests/gpu_test.rs`): same outcomes,
   steps and factorizations per scenario, voltages to 1e-9, for sweeps and N-1, in one
   chunk and in several. `examples/gpu_bench.rs`, cuDSS 1.0.0, against the CPU
   `BatchSolver` on 30 threads from the same start, tol 1e-6 (every scenario agrees to
   2.3e-12; analysis, kept by the session, not included):

   | case | scenarios | GPU solve | CPU solve | GPU / CPU | systems factored |
   |---|---|---|---|---|---|
   | case1354pegase | 1000 sweep | 258 ms | 114 ms | 2.27x | 3373 |
   | case2869pegase | 1000 sweep | 411 ms | 267 ms | 1.54x | 3287 |
   | case9241pegase | 1000 sweep | 1334 ms | 1478 ms | 0.90x | 3568 |
   | case1354pegase | 1430 connected outages (N-1) | 634 ms | 125 ms | **5.06x** | 3804 |
   | case2869pegase | 3804 connected outages (N-1) | 1782 ms | 744 ms | **2.40x** | 9547 |

   The systems factored are the CPU stand-in's counts exactly. At 1.0.0's 0.06-0.24 ms per
   refactorization and solve, the LU is most of the GPU time; the GPU pays only on the
   larger cases, and on N-1 not even there.

   **Corrected in phase 5** (2026-10-30): the N-1 rows first read 1.25x and 0.85x, because
   the CPU column also solved the outages that split the network (561 and 778), which the
   GPU reports as `Outcome::Splits` without solving. Those are the CPU's expensive part:
   229 and 783 ms on 30 threads, a full rebuild each. `gpu_bench` now times the CPU on
   the outages the GPU solved and the splits apart; the rows above are its numbers.
4. **Stragglers to the CPU,** then the factor-reuse experiment, each kept only if grid-bench
   shows it pays. *Measured* (2026-10-29, `gpu_bench`, cuDSS 1.0.0, tol 1e-6, GPU solve
   time; the CPU `BatchSolver` on 30 threads in the last column). The hand-off now solves
   its scenarios in parallel on rayon's pool, each on a fresh solver, in both loops; run one
   by one it cost case9241pegase's N-1 3 s.

   | case | adaptive | hand-off 50 | reuse < 0.1 | both | CPU, same outages |
   |---|---|---|---|---|---|
   | case1354pegase, 1000 sweep | 258 ms | 259 ms (never taken) | 216 ms | | 114 ms |
   | case2869pegase, 1000 sweep | 410 ms | 410 ms (never taken) | 349 ms | | 267 ms |
   | case9241pegase, 1000 sweep | 1335 ms | 1334 ms (never taken) | 1240 ms | | 1478 ms |
   | case1354pegase, N-1 (1430 connected) | 636 ms | 489 ms | 569 ms | 485 ms | 125 ms |
   | case2869pegase, N-1 (3804 connected) | 1793 ms | 1939 ms | 1588 ms | 1785 ms | 744 ms |
   | case9241pegase, N-1 (14384 connected) | 25.7 s | 25.4 s | 23.5 s | 23.7 s | 13.5 s |

   - **Factor reuse pays everywhere**: 7-16% on sweeps, 9-11% on N-1, from half the
     factorizations for about 45% more solves. It ends each scenario at a different
     iterate within the tolerance (voltages 2e-7 to 7e-7 from the CPU's at tol 1e-6,
     against 1e-12 without), and a hopeless outage's outcome can change: case9241pegase
     reported 10 not converged in one run and 9 not converged and 1 diverged in another.
     It stays an option, off by default; grid-bench's oracle decides whether it can be on.
   - **The hand-off pays only where a few outages drag a chunk on**: 23% on
     case1354pegase's N-1, 1% on case9241pegase's, 8% worse on case2869pegase's, never taken
     on sweeps (their scenarios converge together). It stays an option, off by default, its
     threshold to be set per case if at all.
   - ~~With either, the GPU is now at 0.75-0.98x the CPU's time on N-1~~ (corrected in
     phase 5: the CPU column then included the splitting outages; on the same outages the
     GPU's best is 1.7-3.9x the CPU's time on N-1). On case9241pegase's sweep it is at
     0.84-0.90x, and still 1.3-2x it on the smaller sweeps. The LU is
     0.24 ms per system on case9241pegase, so of 25 s for its N-1 about 8 s are
     factorizations and solves: the rest, the chunks' tail steps and the per-step host
     round trip, is where phase 5's profile has to look first. (Phase 5 profiled a full
     sweep step, not N-1's tail: there the round trip costs about 0.1 ms per step, and
     cuDSS is 89-95% of the step.)
5. **Python API, tests, grid-bench:** every scenario against the CPU `BatchSolver` to 1e-8, a
   GPU Docker image, and the GPU tabs against gpusim2grid and ExaPF.jl on the same A100.
   *Done* (2026-10-30):
   - `gridoxide` has a `gpu` feature (re-exporting the crate as `gridoxide::gpu`), and
     `gridoxide-gpu` is now published with the workspace; its dev-dependency on gridoxide
     carries no version, so the cycle stays out of the registry, and its tests and examples
     stay out of the package. It `links = "cudss"` and hands its library directories on as
     `DEP_CUDSS_LIB_DIRS`, which gridoxide's build script puts on the extension's run path.
   - Python: `GpuBatchSolver(model, capacity=1024, reuse_below=None, cpu_handoff=None)`,
     a session from the model's last `solve()` (`start="last"`), with `solve_batch` (and
     `set_scenarios`) in `solve_batch`'s format and `solve_contingencies` in its format,
     returning the CPU's `BatchResult` and `ContingencyResult`. Outages that split the
     network are solved on the CPU, so every outage gets a result; the device's own
     statuses are `converged`, `max_iterations` and `diverged`.
   - `python/tests/test_gpu.py`: a 300-scenario sweep and every outage of case1354pegase
     against the CPU, outcome, iteration count and voltages to 1e-8, in one chunk and in
     several; hand-off and factor reuse. cuDSS is not deterministic by default: the same
     call twice differs by up to 1e-13.
   - `docker/gpu/Dockerfile`: CUDA 12.8.1, cuDSS 1.0.0 pinned, the wheel built with
     `python,gpu`; the tests pass in it on the A100 (`--gpus all`).
   - **The N-1 comparison was lopsided** (see phase 3's correction). Like for like,
     the GPU loses every N-1 case: 5.1x, 2.4x and 1.9x the CPU's time on case1354pegase,
     case2869pegase and case9241pegase.
   - **10k sweeps change nothing**: the session solves in chunks of its capacity, so the
     time per scenario is the 1k sweep's (`gpu_bench`, 10000 scenarios, tol 1e-6):

     | case | GPU | CPU, 30 threads | GPU / CPU |
     |---|---|---|---|
     | case1354pegase | 2.79 s | 1.07 s | 2.61x |
     | case2869pegase | 4.49 s | 2.40 s | 1.87x |
     | case9241pegase, capacity 1024 | 11.5 s | 13.6 s | 0.85x |
     | case9241pegase, capacity 2048 | 11.7 s | 13.5 s | 0.87x |

     From Python, scenario conversion adds about 240 ms per 1000 scenarios of
     case9241pegase (6.3 million tuples); `set_scenarios` keeps it out of the timed call,
     as on the CPU.
   - **One batch step under Nsight Systems** (2024.6.2, `--trace=cuda`; `gpu_bench`, a
     1024-scenario sweep in one chunk, the first step of the last timed solve, all 1024
     active):

     | | case1354pegase | case9241pegase |
     |---|---|---|
     | step | 65.3 ms | 253.6 ms |
     | GPU busy | 99.8% | 99.9% |
     | kernel launches | 20 | 20 |
     | largest launch | 2.3M blocks | 11.2M blocks |
     | factorization (`factorize_v3_ker`, `factorize_ker`) | 67% | 77% |
     | triangular solves (`fwd_ker`, `bwd_ker`) | 22% | 18% |
     | our kernels (Jacobian, injections, mismatch) | 1.1% | 2.6% |

     The GPU is full at B = 1024 on both cases: cuDSS factors the whole batch in one launch,
     and the gaps between work add up to about 0.1 ms per step. So a larger batch cannot
     help (it is more such steps in a row), a small grid does not underuse the GPU, and
     our own kernels cannot change the outcome.
   - **What grid size changes is the CPU's cost.** The CPU `BatchSolver`'s speedup from 1
     to 30 threads is 23.6x on case1354pegase, 24.4x on case2869pegase and 20.8x on
     case9241pegase, so shared memory bandwidth costs it little. On one core, its time per
     scenario grows 11.8x from case1354pegase to case9241pegase (2.5 to 29.4 ms) for 6.8x
     the buses; the GPU's time per system and step grows 3.9x (64 to 248 µs). Per step,
     the A100 is worth about 12 cores on case1354pegase and about 33 on case9241pegase; at
     21-24x on 30 threads, it wins once it is worth more than that, between
     case2869pegase and case9241pegase. Why the costs grow so differently is not
     measured yet: presumably fill-in and caches for KLU on one core, while cuDSS's launch
     grows with the matrix (about 2300 against 11000 blocks per system) and its time grows
     more slowly than its work. Counting the LU factors' nonzeros for both cases against
     their factorization times would test it.
   - **The LU factors** (`examples/lu_counts.rs`; the solved base case's Jacobian; KLU's
     count from `KluNativeSystem::factor_stats`, cuDSS's `CUDSS_DATA_LU_NNZ`, per system):

     | | case1354pegase | case2869pegase | case9241pegase | growth |
     |---|---|---|---|---|
     | Jacobian nnz | 15803 | 36591 | 129412 | 8.2x |
     | KLU nnz(L+U) | 22005 (1.39x) | 55019 (1.50x) | 215298 (1.66x) | 9.8x |
     | cuDSS LU_NNZ | 25475 (1.61x) | 64987 (1.78x) | 247494 (1.91x) | 9.7x |
     | KLU factorization | 0.128 Mflop | 0.378 Mflop | 2.405 Mflop | 18.8x |
     | CPU, one core, per scenario | 2.5 ms | 6.1 ms | 29.4 ms | 11.8x |
     | cuDSS factorization, per system | 42.6 µs | | 189.6 µs | 4.5x |

     Fill-in stays small, but the work per factorization grows 18.8x for 6.8x the buses,
     and the CPU follows it part of the way. cuDSS does not follow it at all: over the
     batch its factorization runs at 3 and 13 GFLOP/s, 0.03-0.1% of the A100's FP64 peak,
     and moving its factors once would take 0.3-0.7% of the time at the A100's bandwidth.
     So its cost is a fixed overhead per system, presumably the sequential steps of each
     sparse elimination, and a small grid pays it for little arithmetic: the GPU wins once
     a factorization has enough work. (Not confirmed with Nsight Compute; the flop counts
     are KLU's ordering, and cuDSS's factors are 16% larger.)

   - **grid-bench** (2026-10-30, its branch `gridoxide-gpu`, `e35773e`): gridoxide's GPU
     batch and N-1 as the tool `gridoxide_gpu`, from `d105bf4`, against gpusim2grid and
     ExaPF.jl's GPU batch rerun the same day on the same A100, grid-bench's protocol
     (base case from flat start in the timed call, voltages copied back, tol 1e-8 p.u.)
     and oracle. ms per scenario; lightsim2grid on 30 threads of the machine in
     parentheses:

     | case | gridoxide | gpusim2grid | ExaPF (GPU) | (lightsim2grid) |
     |---|---|---|---|---|
     | case1354pegase#sweep | 0.70 | 0.96 | 0.86 | (0.15) |
     | case9241pegase#sweep | 4.43 | 6.94 | 10.2 | (2.04) |
     | mvlv1004#sweep | 0.57 | 25 rejected | 0.81 | (0.10) |
     | case1354pegase#sweep-1k | 0.27 | 0.15, 507 NaN | 0.22 | (0.12) |
     | case9241pegase#sweep-1k | 1.63 | 1.37 | out of memory | (1.79) |
     | case1354pegase#sweep-10k | 0.30 | 0.13 | out of memory | (0.13) |
     | case2869pegase#sweep-10k | 0.51 | 0.28 | out of memory | (0.33) |
     | case9241pegase#sweep-10k | 1.52 | 0.95 | out of memory | (1.84) |
     | case2869pegase#n1 | 1.37 | 160 NaN | 167 not converged | (0.25) |
     | case1354pegase#n1-full | 0.31 | 0.11, 148 failed | 0.24 | (0.091) |
     | case2869pegase#n1-full | 0.61 | 0.26, 1 failed | out of memory | (0.27) |
     | case9241pegase#n1-full | 2.24 | 0.86, 16 failed | out of memory | (1.26) |

     gridoxide is accepted on every scenario of all 22 cases, the only GPU tool there of
     which that holds: the correctness target is met. The speed target is not. On
     100 scenarios it is the fastest GPU tool (1.4-1.6x gpusim2grid), but on the large
     sweeps gpusim2grid is 1.2-2.3x faster where it is accepted: cuDSS 0.8.0's lower
     cost per system (phase 0) and four fixed steps against every scenario to 1e-8.
     (A first version of this said 1.2-2.6x, counting the `#n1-full` cases, on which
     gpusim2grid is not accepted; on N-1 it is accepted only on case1354pegase#n1 and
     case9241pegase#n1, 1.1-1.3x faster.) What closing the gap could take:
     [GPU_BATCH_TODO.md](GPU_BATCH_TODO.md).
     Against 30 CPU threads gridoxide is ahead only on case9241pegase's large sweeps
     (1.1-1.2x). gpusim2grid and ExaPF.jl, both on cuDSS 0.8.0, fail case2869pegase#n1,
     which gridoxide on 1.0.0 solves whole.

## Targets

- **Correct:** every scenario of every sweep and N-1 case passes grid-bench's oracle, where
  gpusim2grid fails two.
- **Fast:** faster than gpusim2grid on `#sweep-1k`, `#sweep-10k` and `#n1-full`, same GPU.
  On 100-scenario batches, beating the 30-thread CPU is unlikely and not a target.

## Risks

- **cuDSS 1.0.0 costs more per system** than 0.8.0 at large B (1.75-3.7x at B = 2048),
  which is why the GPU beats 30 CPU threads only on the larger cases (phase 3). Its masks
  scale, so the adaptive loop keeps its savings.
- **The host-side mask forces a sync per step.** Small next to a factorization (phase 0, D),
  but it gives up gpusim2grid's sync-free loop.
- **cuDSS's licence** governs what an image may redistribute; grid-bench's gpusim2grid image
  already ships its runtime libraries.
- **cuDSS's undocumented behaviour:** 0.8.0's faults were found by measurement, not
  documentation, and a later cuDSS may have its own. The device tests (the mask contract,
  a 512-scenario batch, parallel test threads) are what a new version must pass first.
- **GPU time is rented:** phase 0 ran on an A100 that may not stay available, and the
  probe's local syntax check (`probes/cudss_ubatch/stub`) is what keeps signature mistakes
  off that clock.
