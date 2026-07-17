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
`case2869pegase`, `case3120sp`, `case6495rte`, `case6515rte`, `case9241pegase` — all bundled directly in
pandapower's own `pandapower.networks` module, nothing to download separately.

**PGM isn't part of this comparison** — a deliberate scope split, not a technical limitation: PGM's own team
doesn't benchmark against these particular IEEE/MATPOWER test cases either. gridoxide models these test
cases' generators as genuine PV (voltage-controlled) buses, the same way PGM and lightsim2grid do — PGM's
`voltage_regulator` component (`regulated_object` = the generator, `u_ref` = its voltage setpoint) is PGM's
real PV-bus mechanism, and `src/pgm.rs::pgm_to_buses_and_branches` now parses it the same way PGM's own
`newton_raphson_pf_solver.hpp::set_u_ref_and_bus_types` does, assigning `BusType::PV` and pinning the bus's
voltage magnitude. See `convert_pandapower_case.py`'s docstring for the full rationale, including a known
data-quality quirk in how pandapower's MATPOWER-derived loaders encode transformers (large `sn_mva`/
`vk_percent` values that push PGM's `uk` field outside its own documented valid range) that affects every
one of these 12 cases.

```bash
python3 -m venv .venv-case-suite
.venv-case-suite/bin/pip install pandapower power-grid-model-io lightsim2grid
cargo build --release --example bench_network --features klu
.venv-case-suite/bin/python3 scripts/bench/run_case_suite.py --python .venv-case-suite/bin/python3
```

This loops all 12 cases, converting each to PGM JSON on first use (cached under
`scripts/bench/.case-cache/`, gitignored — delete it to force reconversion), running gridoxide's `scalar`,
`block`, and `klu` backends, and running lightsim2grid with `SolverType.KLU` (matching lightsim2grid's own
benchmark default, and gridoxide's own fastest backend, for the closest apples-to-apples solver comparison),
then prints one combined markdown table. A case that fails to convert or diverges gets an explicit `FAILED
(...)` cell rather than a misleading blank one. Use `--repeat` to change how many timed solves gridoxide
averages per case (default 10), `--cache-dir`/`--out` to change where converted grids/the results table are
written.

`convert_pandapower_case.py <case_name> <output.json>` and `bench_lightsim2grid.py <case_name>` also work
standalone, for benchmarking or debugging one case at a time.

### Results

`newton_raphson`-only time (ms/run, 10 repeats) vs. lightsim2grid's `ac_pf` (ms, 5 warm runs' mean):

| case | buses | scalar | block | klu | lightsim2grid (KLU) |
|---|---|---|---|---|---|
| case14 | 15 | 0.045 | N/A¹ | 0.040 | 0.024 |
| case118 | 119 | 0.324 | N/A¹ | 0.192 | 0.144 |
| case_illinois200 | 201 | 1.245 | N/A¹ | 0.508 | 0.289 |
| case300 | 301 | 2.145 | N/A¹ | 0.922 | 0.566 |
| case1354pegase | 1355 | 8.949 | N/A¹ | 3.898 | 2.572 |
| case1888rte | 1889 | 13.341 | N/A¹ | 5.242 | FAILED² |
| case2848rte | 2849 | 19.055 | N/A¹ | 8.171 | 7.804 |
| case2869pegase | 2870 | 23.485 | N/A¹ | 10.073 | 7.203 |
| case3120sp | 3121 | 23.788 | N/A¹ | 9.338 | 6.394 |
| case6495rte | — | FAILED³ | N/A¹ | FAILED³ | FAILED² |
| case6515rte | — | FAILED³ | N/A¹ | FAILED³ | FAILED² |
| case9241pegase | 9242 | 120.869 | N/A¹ | 42.355 | 25.389 |

¹ `Block` is documented as symmetric-only with no PV-bus support (`src/solver.rs`'s `JacobianBackend::Block`
doc comment) and correctly panics with a clear message rather than silently mishandling one — every case
here has at least one PV bus (a real `gen`), so `Block` never runs on this track.
² lightsim2grid's own `ac_pf` reports divergence (`V.shape[0] == 0`) on these cases.
³ gridoxide's Newton-Raphson doesn't converge within 20 iterations from a flat start on these two cases.
Notably, lightsim2grid also fails on both — independent evidence these two specific real-world RTE cases are
genuinely hard to solve from a flat start, not a gridoxide-specific gap. `case1888rte` is the interesting
counter-example: gridoxide converges cleanly there while lightsim2grid diverges.

`klu` is consistently 2–3x faster than `scalar` (as with the synthetic-grid benchmark above), and is within
roughly 1.2–1.7x of lightsim2grid's mature, heavily-optimized C++ implementation at every scale up to 9,241
buses — without matching lightsim2grid's convergence robustness on the two hardest cases.
