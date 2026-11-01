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
  `NonlinearProblem` flag for a fixed pattern would skip them.
- [ ] **HyKKT on the device.** Its CG and residuals still run on the host;
  only worth it if it beats the augmented system once they do not.
- [ ] **Small cases:** below case1354 the per-call device latency
  dominates; a CPU fallback by size would keep the CPU's times there.
