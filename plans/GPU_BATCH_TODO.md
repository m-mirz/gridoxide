# GPU batch: TODO

gridoxide-gpu is correct: grid-bench accepts every scenario of all 22 GPU
cases, which no other GPU tool there manages. It is not fast enough. The
cuDSS findings behind the design are in
[probes/cudss_ubatch/README.md](../probes/cudss_ubatch/README.md).

## Where it stands

On one A100 on 2026-10-30 (grid-bench branch `gridoxide-gpu`, cuDSS 1.0.0,
tol 1e-8):

- **Sweeps:** gpusim2grid is 1.2-2.3x faster on `#sweep-1k` and
  `#sweep-10k`. The gap grows from 1k to 10k, because gpusim2grid's time per
  scenario falls by 30-55% and ours does not.
- **N-1:** gpusim2grid passes the oracle only on two 200-outage cases, where
  it is 1.1-1.3x faster. It fails every `#n1-full` case.
- **Against 30 CPU threads:** the GPU is ahead only on case9241pegase's
  sweeps. On N-1 it takes 1.9-5.1x the CPU's time on the same outages.

What we know about the causes:

- cuDSS is 89-95% of a full step, and the GPU is 99.8% busy.
- cuDSS's cost is roughly fixed per system: its factorization runs at
  0.03-0.1% of FP64 peak. That favours large grids.
- cuDSS 1.0.0 costs 1.75-3.7x what 0.8.0 does per system. We cannot use
  0.8.0: it returns NaN in a uniform batch.
- gpusim2grid runs four fixed steps. We take every scenario to 1e-8.

These are ruled out as causes: the per-step host round trip (about 0.1 ms
of a 65-254 ms step), our own kernels (1-3% of a step) and the copy of the
voltages back to the host.

## Measure first

- [ ] Count steps and factorizations per scenario in grid-bench, against
  gpusim2grid's fixed four, to see how much of the gap the tolerance
  explains. Record them in the adapter's `extra_info`.
- [ ] Profile the tail of a 10k sweep, when few scenarios are still active.
  A masked step still costs 9-12% of a full one.
- [ ] Find out why gpusim2grid's time per scenario falls from 1k to 10k and
  ours does not.

## Candidates

- [ ] **Keep the batch full.** As scenarios converge, refill their slots
  instead of finishing each chunk before starting the next. A refilled slot
  needs a masked `FACTORIZATION`.
- [ ] **Factor reuse in grid-bench.** `reuse_below=0.1` saves 7-16% on
  sweeps. The oracle grades residuals, so it should accept the results;
  run it and see.
- [ ] **cuDSS 1.0's settings:** reordering, pivoting and nested-dissection
  levels, timed at B = 1024.
- [ ] **Newer cuDSS releases.** On each release, rerun the probe and the
  device tests, and adopt the release if it is correct and cheaper.
- [ ] **Mixed precision.** Factor in FP32 and keep the mismatch in FP64.
  This is an experiment, kept only if the oracle accepts it.
