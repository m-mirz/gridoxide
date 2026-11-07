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
  ExaModelsPower's 63 and 2.10. The inertia is off by hundreds at μ = 0.1, a
  truly indefinite Hessian; 130 of 208 iterations carry a δw of 1e4 to 1e9, at
  steps of about 1e-3.
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
