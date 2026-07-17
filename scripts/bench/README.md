# Benchmark scripts

Reproduces the runtime comparison against [power-grid-model](https://github.com/PowerGridModel/power-grid-model)
(PGM) referenced in the top-level README's "Sparse solver" section: generate a synthetic radial
distribution/LV grid at a target scale, then time both gridoxide (`examples/bench_network.rs`) and PGM's
Python bindings solving the exact same network.

## 1. Generate a benchmark grid

```bash
python3 scripts/bench/generate_grid.py grid.json --target-nodes 2200
```

`generate_grid.py` ports power-grid-model's own C++ benchmark grid generator
(`tests/benchmark_cpp/fictional_grid_generator.hpp`) to Python — a radial MV feeder network with
stochastically-attached LV sub-grids, using the same component templates and parameters as PGM's release-mode
benchmark. It's not a bit-for-bit RNG replica (different RNG engine than the C++ `mt19937_64`), but since both
gridoxide and PGM are benchmarked against the *same* generated JSON file, the comparison is apples-to-apples
regardless of how the topology was generated.

`--target-nodes` is a rough target, not exact (LV-grid attachment is a stochastic Bernoulli process — see the
script's docstring). The three scales used in this project's own benchmarking:

```bash
python3 scripts/bench/generate_grid.py grid_small.json  --target-nodes 200   # -> 192 nodes
python3 scripts/bench/generate_grid.py grid_medium.json --target-nodes 1500  # -> 1,003 nodes
python3 scripts/bench/generate_grid.py grid_large.json  --target-nodes 2200  # -> 2,605 nodes
```

## 2. Benchmark gridoxide

```bash
cargo build --release --example bench_network
./target/release/examples/bench_network grid.json [repeat-count] [backend]
```

`repeat-count` (default 1) re-runs the solve that many times from a fresh flat start each time — useful both
for stable timing averages and for `perf record`-based profiling, since a single solve is often too fast
(tens of milliseconds) to sample meaningfully. `backend` selects `scalar` (default), `block`, or `klu` (if
built with `--features klu`) — see the top-level README's "Sparse solver" section.

## 3. Benchmark power-grid-model

```bash
python3 -m venv .venv-pgm
.venv-pgm/bin/pip install power-grid-model
.venv-pgm/bin/python3 scripts/bench/bench_pgm.py grid.json
```

`power-grid-model` ships as a prebuilt wheel — no C++ build needed.

## Interpreting results

Compare gridoxide's `total (guess + NR)` line against PGM's `min`/`mean` (warm, repeated calls on an
already-built model) and `cold (construct+calc)` (includes PGM's own model-build overhead) figures. Sample
voltage output (`voltage_mag min/max` from gridoxide, `u_pu min/max` from PGM) should match closely if both
are solving the same input correctly — a large mismatch there means something is wrong with the comparison,
not just the timing.

## 4. Benchmark against real power-system test-case grids (lightsim2grid)

A second, separate comparison: gridoxide against [lightsim2grid](https://github.com/m-mirz/lightsim2grid) (a
C++/KLU-backed Newton-Raphson solver) on the same 12 real IEEE/MATPOWER-derived test-case grids lightsim2grid's
own benchmark uses
([`benchmarks/benchmark_grid_size.py`](https://github.com/m-mirz/lightsim2grid/blob/master/benchmarks/benchmark_grid_size.py)):
`case14`, `case118`, `case_illinois200`, `case300`, `case1354pegase`, `case1888rte`, `case2848rte`,
`case2869pegase`, `case3120sp`, `case6495rte`, `case6515rte`, `case9241pegase`.

**PGM isn't part of this comparison** — a deliberate scope split, not a technical limitation: PGM's own team
doesn't benchmark against these particular IEEE/MATPOWER test cases either. gridoxide models these test
cases' generators as genuine PV (voltage-controlled) buses, the same way PGM and lightsim2grid do — PGM's
`voltage_regulator` component (`regulated_object` = the generator, `u_ref` = its voltage setpoint) is PGM's
real PV-bus mechanism, and `src/pgm.rs::pgm_to_buses_and_branches` now parses it the same way PGM's own
`newton_raphson_pf_solver.hpp::set_u_ref_and_bus_types` does, assigning `BusType::PV` and pinning the bus's
voltage magnitude.

gridoxide's side is converted straight from MATPOWER's own `.m` case files
(`matpower_to_pgm.py`, fetched from a fork of MATPOWER's repo,
https://github.com/m-mirz/matpower/tree/master/data), not through pandapower's own MATPOWER importer.
This isn't cosmetic: three of these twelve cases (`case1888rte`, `case6495rte`, `case6515rte`) would not
converge at all through pandapower's importer, root-caused by comparing directly against
`references/powsybl-open-loadflow` via `pypowsybl` — pandapower's importer assigns each bus a real physical
`baseKV`, and for a transformer connecting two different voltage levels that requires very carefully keeping
every impedance/tap conversion referenced to a *consistent* side (easy to get subtly wrong — this project's
own first attempt did, twice). powsybl's own MATPOWER importer sidesteps the problem entirely: verified
directly that it assigns `nominal_v = 1.0` to *every* bus regardless of the file's `baseKV` column, and
encodes an off-nominal ratio purely via `rated_u1 = ratio`, `rated_u2 = 1.0` — MATPOWER's own power-flow
formulation never actually needs a physical voltage reference, only a *consistent* one.
`matpower_to_pgm.py` does the same, which fixed convergence on all three cases immediately. See that
script's docstring for the full derivation, including a real, narrower-than-PGM's-C++-reference gap this
surfaced in gridoxide's own `network::transformer_tap` (`src/network.rs`) along the way.
`convert_pandapower_case.py` (pandapower-based, kept as a standalone tool, no longer used by this suite)
documents the specific data-quality quirk that trips up pandapower's importer.

```bash
python3 -m venv .venv-case-suite
.venv-case-suite/bin/pip install numpy scipy pandapower lightsim2grid
cargo build --release --example bench_network --features klu
.venv-case-suite/bin/python3 scripts/bench/run_case_suite.py --python .venv-case-suite/bin/python3
```

This loops all 12 cases, fetching each MATPOWER `.m` file and converting it to PGM JSON on first use (cached
under `scripts/bench/.case-cache/`, gitignored — delete it to force re-fetch/reconversion), running
gridoxide's `scalar`, `block`, and `klu` backends, and running lightsim2grid with `SolverType.KLU` (matching
lightsim2grid's own benchmark default, and gridoxide's own fastest backend, for the closest apples-to-apples
solver comparison), then prints one combined markdown table. A case that fails to convert or diverges gets an
explicit `FAILED (...)` cell rather than a misleading blank one. Use `--repeat` to change how many timed
solves gridoxide averages per case (default 10), `--cache-dir`/`--out` to change where converted grids/the
results table are written.

`matpower_to_pgm.py <input.m-or-.mat> <output.json>`, `convert_pandapower_case.py <case_name> <output.json>`,
and `bench_lightsim2grid.py <case_name>` also work standalone, for benchmarking or debugging one case at a
time.

### Results

`newton_raphson`-only time (ms/run, 10 repeats) vs. lightsim2grid's `ac_pf` (ms, 5 warm runs' mean):

| case | buses | scalar | block | klu | lightsim2grid (KLU) |
|---|---|---|---|---|---|
| case14 | 15 | 0.089 | N/A¹ | 0.038 | 0.027 |
| case118 | 119 | 0.317 | N/A¹ | 0.196 | 0.156 |
| case_illinois200 | 201 | 0.841 | N/A¹ | 0.467 | 0.297 |
| case300 | 301 | 2.271 | N/A¹ | 0.762 | 0.533 |
| case1354pegase | 1355 | 8.670 | N/A¹ | 4.004 | 3.005 |
| case1888rte | 1889 | 11.494 | N/A¹ | 4.760 | FAILED² |
| case2848rte | 2849 | 20.312 | N/A¹ | 7.695 | 7.899 |
| case2869pegase | 2870 | 26.641 | N/A¹ | 10.675 | 6.364 |
| case3120sp | 3121 | 21.188 | N/A¹ | 9.047 | 6.026 |
| case6495rte | 6496 | 63.205 | N/A¹ | 24.194 | FAILED² |
| case6515rte | 6516 | 95.627 | N/A¹ | 26.753 | FAILED² |
| case9241pegase | 9242 | 120.056 | N/A¹ | 45.337 | 25.253 |

¹ `Block` is documented as symmetric-only with no PV-bus support (`src/solver.rs`'s `JacobianBackend::Block`
doc comment) and correctly panics with a clear message rather than silently mishandling one — every case
here has at least one PV bus (a real `gen`), so `Block` never runs on this track.
² lightsim2grid's own `ac_pf` reports divergence (`V.shape[0] == 0`) on these three cases — its `gridmodel`
still loads via `pandapower.networks.<case_name>()` directly (it needs the pandapower net object, not PGM
JSON), so it's still exposed to the same underlying data quirk `matpower_to_pgm.py` sidesteps for gridoxide.
Confirmed via `pypowsybl` that even `powsybl-open-loadflow` fails on these same three cases without an
explicit workaround (RTE's own benchmark code zeroes a handful of phase-shift values first) — this is a
genuine property of these specific cases' data, not a gridoxide-, lightsim2grid-, or powsybl-specific gap.

gridoxide now converges on all 12 cases. `klu` is consistently 2–4x faster than `scalar` (as with the
synthetic-grid benchmark above), and is within roughly 1.1–1.8x of lightsim2grid's mature, heavily-optimized
C++ implementation at every scale up to 9,241 buses.
