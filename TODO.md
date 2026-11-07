# GPU TODO

Numbers: grid-bench on one A100, 2026-11-07, at gridoxide v0.4.0 (branch
`gridoxide-gpu-opf`), against the other tools' 2026-10-31 results. cuDSS findings:
[probes/cudss_ubatch/README.md](probes/cudss_ubatch/README.md); profiles and
settings: `scripts/bench/README.md`, section 11.

# AC-OPF

Every case accepted, at grid-bench's limit of 500 iterations. 1.8-2.4x faster
than ExaModelsPower from case2869_pegase up but on case13659_pegase; it also
solves case9241_pegase and case30000_goc, which that does not.

- [ ] **case13659_pegase's crawl:** 206-254 iterations and 4.66 s, against
  ExaModelsPower's 63 and 2.10. The cause is how bus injection writes branch
  limits: |S(V)|² ≤ rate², nonconvex in the voltages, whose multipliers make the
  Hessian indefinite (inertia off by hundreds, δw of 1e4 to 1e9, steps of
  1e-3). Without branch limits bus injection takes 64 iterations; branch flow,
  whose limit is p² + q² ≤ rate² on flow variables, takes 63-66 with them, as
  ExaModelsPower and PowerModels (flow variables too) do. But a branch-flow
  iteration costs 2-2.5x a bus one (case24464_goc: factorization 17.6 ms
  against 6.3, solves 11.6 against 5.4): cuDSS factors 4 more variables and
  rows a branch. At grid-bench's settings, bus / branch flow: case13659_pegase
  218 / 63 iterations, 4.43 / 1.98 s; case24464_goc 1.59 / 3.92 s;
  case9241_pegase 0.78 / 1.63 s. Next: **condense the flow variables out of
  branch flow's KKT system**, exactly, as the inequality slacks are now:
  branch flow's path at about bus injection's cost an iteration. Prototype on
  the host first, for the iterations and the conditioning (the eliminated rows'
  barrier terms), then the device. Either formulation is within grid-bench's
  rules, MATPOWER's problem; the adapter's docstring states which.
- [ ] **The rest of the iteration gap.** case10000_goc 81 (MadNLP 63),
  case19402_goc 70 (62), case10480_goc 66 (60). Candidates: IPOPT's adaptive
  barrier, MadNLP's relaxed equalities.
- [ ] **False inertia alarms,** at 1e-8 only. On case2869_pegase the `LDLᵀ`
  reports one negative eigenvalue too many where a dense pivoted factorization
  gives the right inertia, and 48 of 100 iterations follow. The factorization
  needs curing, not the test: raising δc, a curvature test (gated or not) and
  faer's `LBLᵀ` all made it worse or broke other cases.
- [ ] **cuDSS's analysis,** once a solve: about 0.6 s of case24464_goc's 1.6.
  Reuse it across solves, or give cuDSS a cheaper reordering.
- [ ] **The setup around the iterations,** about 0.4 s of the same 1.6. Not yet
  broken down.
- [ ] **Fewer cuDSS solves:** 4.75 an iteration, nearly half its time. Try
  cuDSS's own iterative refinement.
- [ ] **Small cases:** the device costs case14_ieee and case118_ieee 12 ms. A
  fallback by size would avoid it.
- [ ] **HyKKT on the device,** only if it would beat the augmented system.

# Batch power flow

Every scenario accepted, which no other GPU tool manages. Sweeps lead
gpusim2grid everywhere; case9241pegase's 10k takes 0.59 ms a scenario, 0.47 of
it the solve. N-1 is the weak spot: 0.90-1.01x gridoxide's CPU on `#n1-full`,
0.54-0.70x on 200 outages.

- [ ] **Profile `#n1-full`,** then the tail of a 10k sweep (a masked step still
  costs 9-12% of a full one).
- [ ] **Keep the batch full:** refill the slots of converged scenarios. Only if
  the tail's profile asks for it.
- [ ] **Mixed precision:** factor in FP32, mismatch in FP64, if the oracle
  accepts it.

# Beyond

- [ ] **A linear start for AC:** a DC solve per scenario on the device.

# Settled

Measured; not to be retried without a reason.

- **The batch's cuDSS settings on the AC-OPF:** no gain, or 5-21% slower.
- **Factor reuse in the batch:** 3-12% slower than none.
- **cuDSS 0.8.0:** cheaper, but NaN in a uniform batch.
- **DC sweeps on the GPU:** 3x slower than `DcBatchSolver` on 30 threads.
- **The batch's host round trip and own kernels:** not the bottleneck.
- **FMA in `kernels.cu`:** the Newton step's kernels keep it; `K`'s product
  does not.
