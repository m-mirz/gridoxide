# GPU TODO

What is left on the GPU: the AC-OPF, the batch power flow, what else could
run there. The cuDSS findings behind the batch's design are in
[probes/cudss_ubatch/README.md](probes/cudss_ubatch/README.md).

# AC-OPF on the GPU: what is left to be faster

Where we stand (A100, 1e-6, deterministic mode off, `opf_compare`; grid-bench
has yet to run it): 1.7-2.3x faster than ExaModelsPower/MadNLP's stored
results on the GOC cases, and solving the three it fails. case24464_goc takes
1.58 s (MadNLP: 3.63): 0.69 s of iterations at 12.4 ms each, nearly all cuDSS
(factorization 6.4, solves 5.5), 0.64 s in the first iteration and about 0.4 s
of setup around them. So what is left first is the once-a-solve work.

- [x] **The interior point on the device.** `InteriorPoint::Device`: the
  decisions stay on the host (`nlp::drive`), behind `nlp::Engine`. A steady
  iteration on case24464_goc from 55 ms to 12.4; solves 2.0-3.25x faster from
  case2869_pegase up, on the same path as the host's vectors to every printed
  digit. The watchdog's return to its saved point has a test of its own: no
  PGLib run here takes it on the device.
- [ ] **The setup around the iterations,** about 0.4 s of case24464_goc's
  1.58: scaling, the patterns, `DeviceInteriorPoint::new`, each solve anew.
  Not yet broken down.
- [ ] **Fewer cuDSS solves.** 4.75 an iteration (refinement rounds,
  second-order corrections), 1.65 ms each without deterministic mode, now
  nearly half of an iteration. Fewer refinement rounds bought only 4-5% and
  moved paths; try cuDSS's own iterative refinement.
- [ ] **cuDSS's analysis**, once a solve (most of the first iteration's
  0.64 s on case24464_goc with `CUDSS_THREADING_LIB`): reuse it across solves
  of the same network, or hand cuDSS a reordering it can use without nested
  dissection's cost.
- [ ] **The remaining iteration gap.** case10000_goc takes 81 iterations
  (IPOPT 82, MadNLP 63); case13659_pegase 207-208 (deterministic mode; 216-280 without). On the default set the gap to
  IPOPT is case2869_pegase alone, and it is the end game, not the barrier:
  from the proportional start (100 iterations, IPOPT 61) nlp takes IPOPT's
  steps to every printed digit for 46 iterations, then, at μ = 10^-8.6 with
  the objective already final, fails the inertia test and adds δw ~ 1e-2.
  IPOPT adds none and stops on its acceptable streak; after our δw steps the
  dual error rises past 1e-6, which breaks that streak. faer's intranode
  pivoted `LBLᵀ` does not cure it (92 iterations; 78 against 68 from flat),
  perhaps because it pivots only within a supernode. Next: find out why the
  test fails there, and what to do if it is only noise near the optimum.
  MadNLP's gap on the GOC cases is a separate question;
  candidates are IPOPT's adaptive barrier and MadNLP's relaxed equalities.
- [x] **The pattern checks** (`fits`): `NonlinearProblem::fixed_pattern`
  skips them. Not `matches` on HyKKT's matrix, whose pattern moves with the
  rows it counts as equalities.
- [x] **Triplets on the device path:** gone with the interior point on the
  device, which reads the model's values where they are. The restoration
  phase, on the host's vectors, still builds them.
- [ ] **HyKKT on the device.** Its CG and residuals still run on the host;
  only worth it if it beats the augmented system once they do not.
- [ ] **Small cases:** below case1354 the per-call device latency
  dominates; a CPU fallback by size would keep the CPU's times there. The
  interior point on the device costs case14_ieee and case118_ieee 12 ms more
  (0.68x and 0.78x); it is even on case300_ieee.
- [x] **Restoration's solver,** kept across restorations: on the GPU a new
  one was a cuDSS handle, stream and buffers and a full analysis. On
  case13659_pegase, the one case with restorations, 13.8 s to 11.1 s (flat,
  3 of them) and 13.3 s to 11.6 s (proportional, 2), at the same iterations;
  peak device memory about 925 MiB either way.
- [x] **The KKT solves** on the device: `kkt::CudssSolver` behind
  `opf::nlp`'s `SymmetricSolver` (cuDSS `LDLᵀ` with inertia, and Cholesky
  for HyKKT). See docs/src/opf/index.md, "On the GPU".
- [x] **The model and KKT assembly** on the device: the KKT assembly,
  equilibration and refinement products, and the model with
  `AcOpf::evaluate_on_device` (case24464_goc: about 80 ms an iteration, from
  208).

# GPU batch power flow

gridoxide-gpu is correct: grid-bench accepts every scenario of all 22 GPU
cases, which no other GPU tool there manages. It is not fast enough.

## Where it stands

On one A100 on 2026-10-30 (grid-bench branch `gridoxide-gpu`, cuDSS 1.0.0,
tol 1e-8), before `load`'s fix and cuDSS's settings below; the settings alone take 1.2-2.2x
off these sweeps; grid-bench has yet to run them in its containers:

- **Sweeps:** gpusim2grid is 1.2-2.3x faster on `#sweep-1k` and
  `#sweep-10k`. The gap grows from 1k to 10k, because gpusim2grid's time per
  scenario falls by 30-55% and ours does not.
- **N-1:** gpusim2grid passes the oracle only on two 200-outage cases, where
  it is 1.1-1.3x faster. It fails every `#n1-full` case.
- **Against 30 CPU threads:** the GPU is ahead only on case9241pegase's
  sweeps. On N-1 it takes 1.9-5.1x the CPU's time on the same outages.

What we know about the causes:

- cuDSS is 87-92% of the device's work and 73-83% of the wall time of a
  batch solve (Nsight Systems, `scripts/bench/README.md`, section 11). During
  the Newton steps the GPU is busy throughout; its 9-16% idle is what falls
  outside them, `DeviceBatch::load` (6-12%) and the results (4-7%).
- Of the device's work, factorization is 51-58% and the triangular solves
  33-41%.
- cuDSS's cost is roughly fixed per system: at its default settings its
  factorization ran at 0.03-0.1% of FP64 peak. That favours large grids.
- cuDSS 1.0.0 costs 1.75-3.7x what 0.8.0 does per system. We cannot use
  0.8.0: it returns NaN in a uniform batch.
- gpusim2grid runs four fixed steps. We take every scenario to 1e-8.

These are ruled out as causes: the per-step host round trip (about 0.1 ms
of a 65-254 ms step) and our own kernels (3-5% of the device's work).

## Measure first

- [ ] Count steps and factorizations per scenario in grid-bench, against
  gpusim2grid's fixed four, to see how much of the gap the tolerance
  explains. Record them in the adapter's `extra_info`.
- [ ] Profile the tail of a 10k sweep, when few scenarios are still active.
  A masked step still costs 9-12% of a full one.
- [ ] Find out why gpusim2grid's time per scenario falls from 1k to 10k and
  ours does not.

## Candidates

- [x] **`DeviceBatch::load` on the host,** 12-26% of a batch solve's wall
  time, now 4-8%: it fills the scenarios in parallel from `start`'s columns.
  Solves 8-23% faster.
- [ ] **Keep the batch full.** As scenarios converge, refill their slots
  instead of finishing each chunk before starting the next. A refilled slot
  needs a masked `FACTORIZATION`.
- [x] **Factor reuse in grid-bench:** not worth it now. The oracle accepts
  it, but with cuDSS's settings below `reuse_below=0.1` is 3-12% slower than
  none: its extra Newton steps outweigh the factorizations it saves.
- [x] **cuDSS 1.0's settings:** multiblock factorization and six levels of
  nested dissection: 1.3-2.7x at B = 1024, half the memory on case9241pegase,
  the same steps and accuracy, the oracle accepts every scenario.
- [x] **The same settings for the AC-OPF:** no gain, `kkt::CudssSolver` keeps
  cuDSS's defaults. On case2869_pegase to case30000_goc six levels are 5-21%
  slower, multiblock within 5%, 12 levels within 4% either way, AMD about
  twice as slow. Iterations are unchanged but for rounding.
- [ ] **grid-bench on these settings,** in its containers: the adapter's
  docstring and results still describe the old ones.
- [ ] **Newer cuDSS releases.** On each release, rerun the probe and the
  device tests, and adopt the release if it is correct and cheaper.
- [ ] **Mixed precision.** Factor in FP32 and keep the mismatch in FP64.
  This is an experiment, kept only if the oracle accepts it.

# Beyond batch power flows

The device solves AC sweeps (`GpuBatchSolver.solve_batch`), AC N-1
(`GpuBatchSolver.solve_contingencies`), DC N-1
(`GpuDcSolver.solve_contingencies`), DC sweeps (`GpuDcSolver.solve_sweep`),
single AC and DC power flows (`GpuBatchSolver.solve_single`,
`GpuDcSolver.solve`) and the AC-OPF (above). Everything else is CPU only.

- [x] **DC sweeps.** `GpuDcSolver::solve_sweep`, one multi-column solve per
  chunk on the session's factorization, through N-1's two-slot pipeline.
  Slower than `DcBatchSolver` on 30 threads: 1000 scenarios take 45 ms
  against 12 on case1354pegase, 290 against 94 on case9241pegase
  (`examples/gpu_dc_sweep_bench.rs`).
- [ ] **A linear start for AC.** The CPU's `start="linear"` has no device
  counterpart; every scenario starts from the last `solve()`. A DC solve per
  scenario on the device would give it one.
- [x] **Single AC or DC power flow.** `GpuBatchSolver::solve_single` (a
  batch of one from the given buses) and `GpuDcSolver::solve` (the CPU's
  `DcSolver` with its reduced solve on the device, through
  `DcSolver::solve_with`). `examples/gpu_single_bench.rs`, against the CPU:
  single AC on a warm session 2.4 ms against 3.0 on case1354pegase, 7.9
  against 41 on case9241pegase; cold 2.75x slower and 0.93x; single DC warm
  1.8x slower and 0.8x.
- [x] **The Python tests of these two:** `python/tests/test_gpu.py` passes
  on the A100.
- [ ] **State estimation.** Batched SE over measurement sets. Large work;
  only after the batch is competitive.

# From the review of PR #30

Correctness:

- [x] **A device setup error reads as infeasibility.** `MakeSolver` returns
  the error, and it ends the solve as `OpfError::Backend`.
- [x] **The pinned buffer after a failed factorization.**
  `CudssSolver::try_factorize` synchronizes on the error path too.
- [x] **`try_solve` on a 0×0 system** did not panic after all: cuDSS's
  analysis fails on an empty matrix, so no factorization reached it. An empty
  system now factorizes and solves, as on the host.
- [x] **Python's `max_iter`** and `ac_opf`'s `max_iterations` default to
  3000, as `NlpOptions` does.
- [x] **FMA in `kernels.cu`.** The device residual was further from the
  host's than FMA: it went through the equilibrated `D K D`. `go_kkt_symv`
  now takes `K` as assembled, without FMA. Against the CPU's iterations on
  96 runs (default set and case9241_pegase, deterministic), the summed gap
  falls from 89 to 46 and the worst from 33 to 18. Dropping FMA alone did
  not help: `case118_ieee__api` went from 83 iterations to 382 (CPU: 50).
  The rest of `kernels.cu` keeps FMA: the Newton step's kernels do not
  follow the CPU's rounding.

Cleanups:

- [x] **The Y-bus and bus template** left `AcOpf`; the test accessors build
  them on demand.
- [x] **`quad_col`** left `go_ac_model`.
- [x] **The row count** is computed once, in `AcOpf::build`.
