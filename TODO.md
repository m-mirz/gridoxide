# AC-OPF on the GPU: what is left to be faster

Where we stand (A100, grid-bench's settings, benchmark mode): faster than
ExaModelsPower/MadNLP up to case2869_pegase, 1.1-1.6x slower on the GOC
cases, and solving the three it fails. Measured on case24464_goc at about
80 ms an iteration (MadNLP: 65).

- [ ] **The interior point on the device.** Its vector work (residuals,
  KKT error, condense/expand around the solves, line search, step update)
  runs on the host: about 30 ms an iteration on case24464_goc. Keeping
  `u`, `y`, `z` on the device, as MadNLP does, removes it and the copies.
- [ ] **Fewer cuDSS solves.** 4.75 an iteration (refinement rounds,
  second-order corrections), 1.65 ms each without deterministic mode. Fewer
  refinement rounds bought only 4-5% and moved paths; try refinement on the
  device in one call, or cuDSS's own iterative refinement.
- [ ] **cuDSS's analysis**, once a solve (0.7 s on case24464_goc with
  `CUDSS_THREADING_LIB`): reuse it across solves of the same network, or
  hand cuDSS a reordering it can use without nested dissection's cost.
- [ ] **The remaining iteration gap.** case10000_goc takes 81 iterations
  (IPOPT 82, MadNLP 63); case13659_pegase 211. Trace against IPOPT
  (`opf-ipopt`) and MadNLP where they differ; candidates are IPOPT's
  adaptive barrier and MadNLP's relaxed equalities.
- [ ] **The pattern checks** (`fits`, `matches`): about 1 ms each an
  iteration on case24464_goc, for patterns that never change. A
  `NonlinearProblem` flag for a fixed pattern would skip them, and with it
  `AcOpf::jacobian` and `lagrangian_hessian` could hand over values alone,
  instead of rebuilding the triplets on every call on the device path.
- [ ] **HyKKT on the device.** Its CG and residuals still run on the host;
  only worth it if it beats the augmented system once they do not.
- [ ] **Small cases:** below case1354 the per-call device latency
  dominates; a CPU fallback by size would keep the CPU's times there.
- [ ] **Restoration's solver.** Each restoration makes a new KKT solver: on
  the GPU a new cuDSS handle, stream and buffers and a full analysis. Keep
  it across restorations.

# From the review of PR #30

Correctness:

- [ ] **A device setup error reads as infeasibility.** The `cudss_*_solver`
  hooks turn `CudssSolver::new`'s error into `None`; `Augmented::new` and
  `Hybrid::new` pass it on, the δw loop takes it for a failed factorization,
  retries about 20 times with a new handle each, and the run ends in "no
  acceptable step; the problem may be infeasible here". Stop the run with
  the device's error.
- [ ] **The pinned buffer after a failed factorization.**
  `CudssSolver::try_factorize` returns through `?` after ANALYSIS, the INFO
  reset or FACTORIZATION without reaching `cudaStreamSynchronize`, and
  `Assembly::run` counts on the last factorization having synchronized: the
  next retry can write `host_trip` while its copy is in flight. Synchronize
  on the error path too.
- [ ] **`try_solve` on a 0×0 system** panics: `solution` holds `n.max(1)`
  entries and `download` asserts equal lengths. `try_multiply` guards
  `n == 0`; do the same.
- [ ] **Python's `max_iter` defaults to 300**, `NlpOptions` to 3000:
  bus-injection case13659_pegase needs 322, so it fails from Python at the
  default.
- [ ] **FMA in `kernels.cu`.** `opf.cu` is built with `-fmad=false`,
  `kernels.cu` is not, so the device residual (`k_kkt_symv`) is not the
  host's arithmetic and refinement can stop a round apart from the CPU's.

Docs:

- [ ] **Stale numbers.** docs/src/opf/index.md ("On the GPU") and
  plans/GPU_BATCH_TODO.md still say 4-5.5x slower than ExaModelsPower on
  the GOC cases and 93 ms an iteration on case24464_goc; the figures at the
  top of this file are the current ones.

Cleanups:

- [ ] **The Y-bus and bus template in `AcOpf::build`.** Only the
  `*_for_test` accessors still read them; every build pays for them.
- [ ] **`quad_col`** is uploaded and stored in `go_ac_model`, but no kernel
  reads it.
- [ ] **The row count** is computed three times (`AcOpf::build`,
  `AcKernels::device_data`, `n_constraints`). If they drift, the device
  model rejects every call and the CPU silently answers instead.
