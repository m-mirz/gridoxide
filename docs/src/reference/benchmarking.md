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
| §8 | DC (Bθ) power flow against the AC solve, and the cost of a PTDF/LODF column |
| §9 | AC N-1 contingency screening: factorization reuse across an outage sweep |
| §10 | Node-breaker topology import, and what retaining switches costs |

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


## RMS dynamics

`cargo run --release --features dynamics --example dynamics_scale` times a ring of alternating
generator and load buses, every generator carrying a sixth-order machine with an exciter and a
governor — ten differential states per unit on top of the network's two per bus — through a bolted
fault and its clearing, at a 5 ms step.

| buses | units | unknowns | build | 3 s run | per step | Newton/step |
|---|---|---|---|---|---|---|
| 16 | 8 | 112 | 0.7 ms | 30 ms | 0.050 ms | 1.34 |
| 64 | 32 | 448 | 0.15 ms | 160 ms | 0.266 ms | 1.34 |
| 256 | 128 | 1 792 | 0.5 ms | 682 ms | 1.14 ms | 1.34 |
| 1 024 | 512 | 7 168 | 2.1 ms | 3.15 s | 5.25 ms | 1.34 |
| 4 096 | 2 048 | 28 672 | 9.2 ms | 16.4 s | 27.3 ms | 1.34 |

Two things are worth reading off this.

**Time per step is very close to linear in the system size** — 256× the unknowns costs 546× the
step, an exponent of 1.14. That is what the formulation was chosen for: the network block is the
constant real form of the Y-bus, every device stamp is local, and the pattern is analyzed **once for
the whole run** because every event is value-only. Nothing re-analyzes, whatever happens.

**Newton iterations per step are flat at 1.34** across four orders of magnitude. A step normally
converges on the first correction and occasionally needs a second; the count does not grow with the
system, which is the signature of an exact analytic Jacobian rather than an approximated one.

The `build` column is the power flow plus initialization, and it is negligible against the run —
which is the right shape, since a study sweeps many scenarios over one build.
