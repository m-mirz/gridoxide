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
  (IPOPT 82, MadNLP 63); case13659_pegase 211. On the default set the gap to
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
- [ ] **Triplets on the device path.** `AcOpf::jacobian` and
  `lagrangian_hessian` still rebuild the triplets from fixed patterns on
  every call. Handing over values alone goes through the scaling, NCL and
  restoration wrappers. Measured on case24464_goc: 3.4 ms of about 73 an
  iteration (Jacobian 2.1, Hessian 1.3, beyond the device's own 1.6).
- [ ] **HyKKT on the device.** Its CG and residuals still run on the host;
  only worth it if it beats the augmented system once they do not.
- [ ] **Small cases:** below case1354 the per-call device latency
  dominates; a CPU fallback by size would keep the CPU's times there.
- [x] **Restoration's solver,** kept across restorations: on the GPU a new
  one was a cuDSS handle, stream and buffers and a full analysis. On
  case13659_pegase, the one case with restorations, 13.8 s to 11.1 s (flat,
  3 of them) and 13.3 s to 11.6 s (proportional, 2), at the same iterations;
  peak device memory about 925 MiB either way.

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
