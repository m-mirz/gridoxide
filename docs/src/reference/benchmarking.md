# Benchmarking and Profiling

**`scripts/bench/README.md` is the single source of truth for every benchmark number in this
project.** This page is a map into it, not a copy of it — the numbers live there, next to the scripts
that produce them, so they can be updated in one place when re-measured.

## Profiling

For profiling with `perf`, set:

```bash
sysctl kernel.perf_event_paranoid=1
```

## What is measured, and where

`scripts/bench/README.md` is organized as a numbered sequence of benchmarks:

| Section | What it covers |
|---|---|
| §1–3 | Generating a synthetic radial MV/LV benchmark grid, then timing gridoxide and power-grid-model on it |
| Interpreting results | How to read the numbers, including the cold-vs-warm distinction |
| §4 | The 12-case real IEEE/MATPOWER test-case suite, against five other solvers |
| §4b | Batched power flow — the multi-core CPU baseline (`batch::BatchSolver`) |
| §4c | The JAX oracle validating the block-diagonal embedding |
| §4d | Block-diagonal embedding on real sparse code |
| §5 | Cross-validating CGMES import against pypowsybl |
| §6 | The CGMES conformance test configurations |
| §7 | State estimation — both gridoxide methods against both power-grid-model methods |

## The two benchmark shapes

**Synthetic radial distribution grid** (§1–3). `examples/bench_network.rs` and
`scripts/bench/bench_gridoxide_native.py` time gridoxide against power-grid-model on generated MV/LV
topology at controllable scale. Its `cold` mode measures N independent flat-start solves with no
shared state; the optional `warm` mode measures repeated solves through a `PersistentSolver` — see
[Backends and Factorization Reuse](../solvers/backends.md).

**Real power-system test cases** (§4). Twelve real IEEE/MATPOWER grids, 14 to 9,241 buses, comparing
gridoxide against five independent solvers: power-grid-model,
[lightsim2grid](https://github.com/m-mirz/lightsim2grid), RTE's
[powsybl-open-loadflow](https://github.com/powsybl/powsybl-open-loadflow) (via pypowsybl),
pandapower's default solver, and [VeraGrid](https://github.com/SanPen/VeraGrid).

Two results from that second benchmark are worth stating here because they shaped the code:

- gridoxide and pandapower's own native path are the only two of the six that converge on all 12
  cases. The other four each fail on a subset of the same handful of genuinely hard cases (RTE's own
  real production grids), confirmed by cross-checking against powsybl-open-loadflow directly — not a
  gridoxide gap.
- Compared warm-vs-warm, `Klu` is frequently *faster* than lightsim2grid's own KLU-backed C++ solver
  on this real transmission-topology data, even though PGM still clearly beats every gridoxide
  backend on the synthetic radial-distribution topology. The comparison genuinely depends on grid
  topology, not just implementation language.
