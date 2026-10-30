# GPU batch: TODO

Starting points for closing the large-batch gap to gpusim2grid, from
[GPU_BATCH_PLAN.md](GPU_BATCH_PLAN.md)'s phase 5 and grid-bench's
`gridoxide_gpu` adapters (its branch `gridoxide-gpu`).

## The gap

grid-bench on one A100-SXM4-40GB, 2026-10-30, gridoxide `d105bf4` on cuDSS 1.0.0
against gpusim2grid `416ae9f` on cuDSS 0.8.0, ms per scenario. Only the large
cases on which gpusim2grid is accepted (every scenario through the oracle):

| case | gridoxide | gpusim2grid | gap |
|---|---|---|---|
| case2869pegase#sweep-1k | 0.44 | 0.37 | 1.17x |
| case9241pegase#sweep-1k | 1.63 | 1.37 | 1.19x |
| mvlv10616#sweep-1k | 1.54 | 1.32 | 1.17x |
| case1354pegase#sweep-10k | 0.30 | 0.13 | 2.34x |
| case2869pegase#sweep-10k | 0.51 | 0.28 | 1.82x |
| case9241pegase#sweep-10k | 1.52 | 0.95 | 1.59x |
| mvlv10616#sweep-10k | 1.34 | 0.96 | 1.40x |

On N-1 gpusim2grid is accepted only on two 200-outage cases (case1354pegase#n1
1.13x, case9241pegase#n1 1.32x); it fails every `#n1-full`, so there is no
large N-1 gap to close. The gap grows from 1k to 10k, where gpusim2grid's
time per scenario falls by 30-55 % and gridoxide's stays.

What is known of its causes:
- cuDSS 1.0.0 costs 1.75-3.7x 0.8.0 per system at large batches (phase 0's
  probe, B = 2048). 0.8.0 is not an option: it returns NaN in a uniform batch
  (507 of case1354pegase#sweep-1k's 1000 scenarios, in gpusim2grid, that day).
- gpusim2grid runs four fixed steps; gridoxide takes every scenario to 1e-8
  p.u. Its step counts in grid-bench were not recorded.
- A full step on the A100 is 89-95 % cuDSS and 99.8 % busy (phase 5's
  profile), so the gain is in the linear algebra or in doing less of it.

## Measure first

- [ ] Steps and factorizations per scenario in grid-bench's runs, against
  gpusim2grid's fixed four: how much of the gap is the tolerance. Record them
  in the adapter's `extra_info`, as `scenarios` already is.
- [ ] Profile a 10k sweep's tail: the steps of each chunk once few scenarios
  are left active, which phase 5's profile (one full step) did not cover.
  Masked steps cost 9-12 % of a full one even with one system selected
  (phase 0, C).
- [ ] Why gpusim2grid's time per scenario falls from 1k to 10k and
  gridoxide's does not (both are chunked, 2048 and 1024). Capacity 2048 timed
  the same for gridoxide on case9241pegase#sweep-10k; check the smaller cases.

## Candidates

- [ ] **Keep the batch full.** Refill a chunk's slots as its scenarios
  converge, instead of finishing each chunk before the next: no tail steps
  until the last chunk, and every step at full occupancy. A refilled slot
  needs a full `FACTORIZATION` for its first step, which cuDSS 1.0's mask
  allows per system; check that a masked factorization of a few slots
  costs no more than the refactorization it replaces.
- [ ] **Factor reuse in grid-bench.** `reuse_below=0.1` saved 7-16 % on
  sweeps in `gpu_bench` (phase 4). grid-bench's oracle grades residuals, not
  distance to a reference, so a scenario ending elsewhere within the
  tolerance should still be accepted; run it and see. If it is, it can be the
  adapter's setting, with its justification.
- [ ] **cuDSS 1.0's own settings.** Phase 0 tried reordering, factorization
  algorithm, pivoting, superpanels and nested-dissection levels on 0.8.0's
  B = 16 switch only. Time them on 1.0.0 at B = 1024 with probe C.
- [ ] **A newer cuDSS.** Rerun probe A (correctness past index 216) and C
  (cost per system) on each release after 1.0.0; adopt one that is correct
  and cheaper. The device tests are what it must pass first (plan, Risks).
- [ ] **Mixed precision.** Factor in FP32 and keep the mismatch in FP64:
  Newton tolerates an inexact Jacobian, at the price of more steps. An
  experiment, kept only if the oracle and grid-bench agree.

## Not the cause

- The per-step host round trip: about 0.1 ms of a 65-254 ms step.
- Our own kernels (Jacobian, injections, mismatch): 1-3 % of a step.
- The timed call's base case and voltage copy: `voltages()` copies a
  1000 × 9241 batch in 0.14 s; the base-case solve is one CPU power flow.
