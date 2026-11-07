# GPU TODO

What is left on the GPU. Numbers are grid-bench's, in its containers, on one
A100 and 30 cores on 2026-11-07 at gridoxide febde7c, against the stored
2026-10-31 results of the other tools on the same machine. The cuDSS findings
behind the batch's design are in
[probes/cudss_ubatch/README.md](probes/cudss_ubatch/README.md).

# AC-OPF

The oracle accepts every solved case. 1.8-2.4x faster than
ExaModelsPower/MadNLP from case2869_pegase up (case24464_goc 1.62 s against
3.63), and it solves case9241_pegase and case30000_goc, which MadNLP does not.
case13659_pegase stops at grid-bench's 200 iterations: it needs 207-211.

- [ ] **The inertia test's false alarms.** First for the iteration gap. On
  case2869_pegase (proportional start, 1e-8) the last 48 of 100 iterations
  follow a δw at an optimum already reached, and each δw is a false alarm:
  at the first, the unpivoted `LDLᵀ` reports `(6757, 19486)`, a dense pivoted
  factorization of the same matrix the expected `(6758, 19485)`, and the
  eigenvalue nearest zero of `D K D` is 5.6e-7, far above rounding. Raising δc
  is not the fix: 1e-7 takes that case from 131 iterations to 69 (flat) but
  fails case13659_pegase, and 1e-6 fails more. Next: MadNLP's inertia-free
  curvature test, and whether the same false alarms hold back
  case13659_pegase, the one that matters at grid-bench's tolerance (at 1e-6
  case2869_pegase takes 45, MadNLP 46). faer's intranode pivoted `LBLᵀ` did
  not cure it (92 iterations; 78 against 68 from flat).
- [ ] **The rest of the iteration gap.** case10000_goc takes 81 iterations
  (IPOPT 82, MadNLP 63), case19402_goc 70 (62), case10480_goc 66 (60).
  Candidates are IPOPT's adaptive barrier and MadNLP's relaxed equalities.
- [ ] **cuDSS's analysis**, once a solve: most of the first iteration's 0.64 s
  on case24464_goc, as much as all the other iterations (0.69 s at 12.4 ms).
  Reuse it across solves of the same network, or hand cuDSS a reordering
  without nested dissection's cost.
- [ ] **The setup around the iterations,** about 0.4 s of case24464_goc's
  1.6: scaling, the patterns, `DeviceInteriorPoint::new`, each solve anew. Not
  yet broken down.
- [ ] **Fewer cuDSS solves.** 4.75 an iteration, 1.65 ms each without
  deterministic mode, nearly half of an iteration. Fewer refinement rounds
  bought only 4-5% and moved paths; try cuDSS's own iterative refinement.
- [ ] **Small cases.** The device costs case14_ieee and case118_ieee 12 ms
  (0.68x and 0.78x the host's vectors, even on case300_ieee). A fallback by
  size would keep the host's times there.
- [ ] **HyKKT on the device.** Its CG and residuals run on the host. Only
  worth it if it beats the augmented system once they do not.

# Batch power flow

The oracle accepts every scenario of every case, which no other GPU tool
there manages. What is left to be faster is N-1, and short sweeps.

- **Sweeps:** ahead of gpusim2grid on all of them, 1.01-1.6x on `#sweep-1k`
  and `#sweep-10k`, 2.3-10x on the 100-scenario ones. Against gridoxide on 30
  CPU threads: 1.5-1.6x ahead on case9241pegase's large sweeps, level on
  case1354pegase's and case2869pegase's, 0.83-0.88x on mvlv10616's, and
  0.23-0.82x on 100 scenarios.
- **N-1:** 0.54-0.70x gridoxide's CPU on 200 outages and 0.91-1.06x on
  `#n1-full`, 0.53-0.89x lightsim2grid there. gpusim2grid is about 1.5x faster on
  `#n1-full`, and the oracle rejects it on all three.
- **Steps:** 2.7-4.4 a scenario on average, so taking every scenario to 1e-8
  costs about nothing against gpusim2grid's fixed four.
- **Where the time goes** (Nsight Systems, `scripts/bench/README.md`, section
  11): cuDSS is 87-92% of the device's work and 73-83% of the wall time,
  factorization 51-58% of the device's work and the triangular solves 33-41%.

- [ ] **Profile `#n1-full`,** where the GPU trails most at scale, then the
  tail of a 10k sweep, when few scenarios are still active: a masked step
  still costs 9-12% of a full one.
- [ ] **Keep the batch full.** As scenarios converge, refill their slots
  instead of finishing each chunk before starting the next. A refilled slot
  needs a masked `FACTORIZATION`. Only if the tail's profile asks for it.
- [ ] **Mixed precision.** Factor in FP32 and keep the mismatch in FP64. An
  experiment, kept only if the oracle accepts it.

# Beyond batch power flows

The device solves AC sweeps, AC and DC N-1, DC sweeps, single AC and DC power
flows and the AC-OPF. Everything else is CPU only.

- [ ] **A linear start for AC.** The CPU's `start="linear"` has no device
  counterpart; every scenario starts from the last `solve()`. A DC solve per
  scenario on the device would give it one.

# Settled

Measured, and not to be tried again without a reason:

- **The batch's cuDSS settings on the AC-OPF:** no gain, `kkt::CudssSolver`
  keeps the defaults. Six levels are 5-21% slower, multiblock within 5%, 12
  levels within 4% either way, AMD about twice as slow.
- **Factor reuse in the batch:** `reuse_below=0.1` is 3-12% slower than none,
  its extra Newton steps outweighing the factorizations it saves.
- **cuDSS 0.8.0:** 1.75-3.7x cheaper per system than 1.0.0, but NaN in a
  uniform batch.
- **DC sweeps on the GPU:** slower than `DcBatchSolver` on 30 threads (45 ms
  against 12 on case1354pegase, 290 against 94 on case9241pegase).
- **The batch's host round trip and own kernels:** about 0.1 ms of a
  65-254 ms step, and 3-5% of the device's work.
- **FMA in `kernels.cu`:** the device residual takes `K` as assembled, without
  FMA; dropping FMA alone moved `case118_ieee__api` from 83 iterations to 382
  (CPU: 50). The Newton step's kernels keep it.
- **`matches` on HyKKT's matrix:** its pattern moves with the rows it counts
  as equalities, so `fixed_pattern` cannot skip it.
