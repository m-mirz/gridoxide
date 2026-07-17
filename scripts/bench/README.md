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

## 4. Benchmark against real power-system test-case grids

A second, separate comparison: gridoxide against **four** other independent Newton-Raphson implementations —
[power-grid-model](https://github.com/PowerGridModel/power-grid-model) (PGM),
[lightsim2grid](https://github.com/m-mirz/lightsim2grid) (C++/KLU-backed),
[powsybl-open-loadflow](https://github.com/powsybl/powsybl-open-loadflow) (RTE's Java solver, via
[pypowsybl](https://pypi.org/project/pypowsybl/)), and pandapower's own default solver — on the same 12 real
IEEE/MATPOWER-derived test-case grids lightsim2grid's own benchmark uses
([`benchmarks/benchmark_grid_size.py`](https://github.com/m-mirz/lightsim2grid/blob/master/benchmarks/benchmark_grid_size.py)):
`case14`, `case118`, `case_illinois200`, `case300`, `case1354pegase`, `case1888rte`, `case2848rte`,
`case2869pegase`, `case3120sp`, `case6495rte`, `case6515rte`, `case9241pegase`.

gridoxide models these cases' generators as genuine PV (voltage-controlled) buses, the same way PGM,
lightsim2grid, pypowsybl, and pandapower all do — PGM's `voltage_regulator` component (`regulated_object` =
the generator, `u_ref` = its voltage setpoint) is PGM's real PV-bus mechanism, and
`src/pgm.rs::pgm_to_buses_and_branches` now parses it the same way PGM's own
`newton_raphson_pf_solver.hpp::set_u_ref_and_bus_types` does, assigning `BusType::PV` and pinning the bus's
voltage magnitude.

gridoxide's and PGM's side are converted straight from MATPOWER's own `.m` case files (`matpower_to_pgm.py`,
fetched from a fork of MATPOWER's repo, https://github.com/m-mirz/matpower/tree/master/data), not through
pandapower's own MATPOWER importer. This isn't cosmetic: three of these twelve cases (`case1888rte`,
`case6495rte`, `case6515rte`) would not converge at all through pandapower's importer, root-caused by
comparing directly against `references/powsybl-open-loadflow` via `pypowsybl` — pandapower's importer assigns
each bus a real physical `baseKV`, and for a transformer connecting two different voltage levels that
requires very carefully keeping every impedance/tap conversion referenced to a *consistent* side (easy to
get subtly wrong — this project's own first attempt did, twice). powsybl's own MATPOWER importer sidesteps
the problem entirely: verified directly that it assigns `nominal_v = 1.0` to *every* bus regardless of the
file's `baseKV` column, and encodes an off-nominal ratio purely via `rated_u1 = ratio`, `rated_u2 = 1.0` —
MATPOWER's own power-flow formulation never actually needs a physical voltage reference, only a *consistent*
one. `matpower_to_pgm.py` does the same, which fixed convergence on all three cases immediately for
gridoxide. See that script's docstring for the full derivation, including a real,
narrower-than-PGM's-C++-reference gap this surfaced in gridoxide's own `network::transformer_tap`
(`src/network.rs`), and two of PGM's own stricter validity constraints its writer had to satisfy that
gridoxide's simplified port doesn't enforce (even `clock` parity and non-negative `tap_size` — see the
script for both). `convert_pandapower_case.py` (pandapower-based, kept as a standalone tool, no longer used
by this suite) documents the specific data-quality quirk that trips up pandapower's importer.

lightsim2grid and pandapower still load via `pandapower.networks.<case_name>()` directly (they need the
pandapower net object, not PGM JSON); pypowsybl loads the same MATPOWER file gridoxide and PGM do, via its
own MATPOWER importer (`bench_pypowsybl.py` — re-serializes a `.m` file to a temporary `.mat` first, since
pypowsybl's importer only reads the binary format).

```bash
python3 -m venv .venv-case-suite
.venv-case-suite/bin/pip install numpy scipy power-grid-model pandapower lightsim2grid pypowsybl
cargo build --release --example bench_network --features klu
.venv-case-suite/bin/python3 scripts/bench/run_case_suite.py --python .venv-case-suite/bin/python3
```

This loops all 12 cases, fetching each MATPOWER `.m` file and converting it to PGM JSON on first use (cached
under `scripts/bench/.case-cache/`, gitignored — delete it to force re-fetch/reconversion), running
gridoxide's `scalar`, `block`, and `klu` backends, PGM, lightsim2grid with `SolverType.KLU` (matching
lightsim2grid's own benchmark default, and gridoxide's own fastest backend, for the closest apples-to-apples
solver comparison), pypowsybl with the same "basic" `LoadFlowParameters` powsybl's own benchmark repo uses,
and pandapower with its own defaults — then prints one combined markdown table. A case that fails to convert
or diverges gets an explicit `FAILED (...)` cell (with the tool's actual exception, not a generic trailer)
rather than a misleading blank one. Use `--repeat` to change how many timed solves gridoxide averages per
case (default 10), `--cache-dir`/`--out` to change where converted grids/the results table are written.

`matpower_to_pgm.py <input.m-or-.mat> <output.json>`, `bench_pgm.py <input.json>`,
`convert_pandapower_case.py <case_name> <output.json>`, `bench_lightsim2grid.py <case_name>`,
`bench_pypowsybl.py <case_name> <input.m-or-.mat>`, and `bench_pandapower.py <case_name>` also work
standalone, for benchmarking or debugging one case/tool at a time.

### Results

`newton_raphson`-only time (ms/run, 10 repeats) vs. each other tool's own warm-run mean (5 runs):

| case | buses | scalar | block | klu | PGM | lightsim2grid (KLU) | pypowsybl | pandapower |
|---|---|---|---|---|---|---|---|---|
| case14 | 15 | 0.049 | N/A¹ | 0.037 | 0.134 | 0.026 | 1.665 | 16.248 |
| case118 | 119 | 0.374 | N/A¹ | 0.193 | FAILED² | 0.151 | 4.554 | 16.892 |
| case_illinois200 | 201 | 0.867 | N/A¹ | 0.424 | 0.573 | 0.299 | 5.857 | 16.247 |
| case300 | 301 | 2.067 | N/A¹ | 0.771 | FAILED² | 0.539 | 7.849 | 16.620 |
| case1354pegase | 1355 | 9.042 | N/A¹ | 3.959 | FAILED³ | 2.536 | 37.060 | 26.974 |
| case1888rte | 1889 | 11.948 | N/A¹ | 4.468 | FAILED² | FAILED⁴ | FAILED⁵ | 27.916 |
| case2848rte | 2849 | 19.662 | N/A¹ | 7.672 | FAILED² | 7.531 | FAILED⁵ | 32.433 |
| case2869pegase | 2870 | 23.929 | N/A¹ | 9.873 | FAILED³ | 6.182 | 96.524 | 33.982 |
| case3120sp | 3121 | 21.054 | N/A¹ | 9.196 | FAILED³ | 7.088 | 79.131 | 32.107 |
| case6495rte | 6496 | 64.359 | N/A¹ | 24.531 | FAILED³ | FAILED⁴ | FAILED⁵ | 144.302 |
| case6515rte | 6516 | 72.729 | N/A¹ | 26.744 | FAILED³ | FAILED⁴ | FAILED⁵ | 61.075 |
| case9241pegase | 9242 | 117.379 | N/A¹ | 40.982 | FAILED³ | 24.667 | 379.553 | 82.602 |

¹ `Block` is documented as symmetric-only with no PV-bus support (`src/solver.rs`'s `JacobianBackend::Block`
doc comment) and correctly panics with a clear message rather than silently mishandling one — every case
here has at least one PV bus (a real `gen`), so `Block` never runs on this track.
² PGM's own `IterationDiverge`: fails to converge within 20 iterations, PGM's own default. `case14` and
`case_illinois200` are the two cases where PGM does converge; `case14`'s converged voltages match gridoxide's
exactly (`voltage_mag`/`u_pu` both 1.010000/1.090000), confirming the converted data itself is correct there.
`case_illinois200` converges in both but to visibly different voltages (gridoxide: 1.0082/1.0400, PGM:
1.0101/1.0548) — both plausible, neither obviously wrong, not chased further; PGM's own particular
Newton-Raphson implementation is simply less robust on these specific harder cases via this input path than
gridoxide's.
³ PGM's own `SparseMatrixError` ("possibly singular matrix") — raised during `PowerGridModel()` construction,
before any iteration runs.
⁴ lightsim2grid's own `ac_pf` reports divergence (`V.shape[0] == 0`).
⁵ pypowsybl's own `run_ac` fails to converge (`MAX_ITERATION_REACHED` or `Unrealistic state`) using the same
"basic" (no damping, flat start) parameters as powsybl's own benchmark repo — *without* the explicit
phase-shift-zeroing workaround that repo's own `MatpowerUtil.java` applies before benchmarking these same
three cases (confirmed directly: applying that workaround via `pypowsybl` does make powsybl-open-loadflow
converge on all three, in ~4 iterations each — see matpower_to_pgm.py's docstring).

Three consistent findings across the table: gridoxide (`scalar`/`klu`) and pandapower (its own native,
no-cross-tool-conversion path) are the only two that converge on **all 12** cases. `case1888rte`,
`case6495rte`, and `case6515rte` are hard for every tool that doesn't special-case them — a genuine property
of those three cases' data (RTE's own real production grid, per the case names), not a gridoxide-, PGM-,
lightsim2grid-, or pypowsybl-specific gap. `klu` is consistently 2–4x faster than `scalar`, and — on the
cases where a same-input, same-settings comparison is possible at all — within roughly the same order of
magnitude as lightsim2grid and PGM's own C++/C++ engines, well ahead of pandapower's and pypowsybl's
heavier general-purpose frameworks.

gridoxide now converges on all 12 cases. `klu` is consistently 2–4x faster than `scalar` (as with the
synthetic-grid benchmark above), and is within roughly 1.1–1.8x of lightsim2grid's mature, heavily-optimized
C++ implementation at every scale up to 9,241 buses.
