# Benchmark scripts

Reproduces the runtime comparison against [power-grid-model](https://github.com/PowerGridModel/power-grid-model)
(PGM) referenced in the top-level README's "Sparse solver" section: generate a synthetic radial
distribution/LV grid at a target scale, then time both gridoxide (`examples/bench_network.rs`) and PGM's
Python bindings solving the exact same network.

## Python bindings

gridoxide also has its own native Python bindings (`src/python.rs`, a `gridoxide` extension module built
with [maturin](https://www.maturin.rs/)), so every script in this directory can run purely in Python — no
subprocess call into a compiled Rust binary, no parsing its stdout.

```bash
pip install maturin
maturin develop --release --features python         # scalar + block backends
maturin develop --release --features python,klu     # + the klu backend (needs a C compiler, libclang — see
                                                      # the top-level README's "Experimental backends" section)
```

(`maturin develop` needs an active virtualenv — either run it from inside one, or set `VIRTUAL_ENV=/path/to/venv`
first.) This builds `#[cfg(feature = "python")]`-gated code that's otherwise not compiled at all — a plain
`cargo build`/`cargo test` is completely unaffected. Only ever build the `python` feature via `maturin`, never
combined with `cargo build --examples`/`cargo test` in one invocation: `pyo3`'s `extension-module` feature
(needed so the built `.so` doesn't try to link `libpython` itself, since Python `dlopen`s it and supplies the
symbols at runtime) makes a normal standalone binary or test harness fail to link — a well-known, expected
PyO3 constraint, not a gridoxide-specific issue.

```python
import gridoxide

model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="klu")
model.solve()
print(model.voltage_mag(), model.voltage_ang())
```

`PowerFlowModel` wraps `solver::PersistentSolver` directly (see the top-level README's "Reusing factorization
across repeated solves") — repeated `.solve()` calls on the same model reuse cached symbolic factorization,
matching exactly how every other tool benchmarked here (PGM's `PowerGridModel`, lightsim2grid's `GridModel`,
pandapower's `net`) is itself used: construct once, call the solve method repeatedly, time each call yourself
with `time.perf_counter()`. `scripts/bench/bench_gridoxide_native.py` does exactly this.

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
./target/release/examples/bench_network grid.json [repeat-count] [backend] [mode]
```

`repeat-count` (default 1) re-runs the solve that many times from a fresh flat start each time — useful both
for stable timing averages and for `perf record`-based profiling, since a single solve is often too fast
(tens of milliseconds) to sample meaningfully. `backend` selects `scalar` (default), `block`, or `klu` (if
built with `--features klu`) — see the top-level README's "Sparse solver" section. `mode` selects `cold`
(default — every repeat calls `newton_raphson_with_backend` fresh, redoing symbolic factorization every time)
or `warm` (one `solver::PersistentSolver` is reused across all repeats, so only the first pays for symbolic
factorization) — see the top-level README's "Reusing factorization across repeated solves" section. `warm` is
the fair comparison against PGM's `min`/`mean` below and against every other tool in step 4, all of which
reuse their own persistent model/solver object across their repeated timed calls.

`bench_gridoxide_native.py` (needs the Python bindings — see "Python bindings" above) is the pure-Python
equivalent, always warm (`PowerFlowModel` wraps `PersistentSolver` directly, so there's no separate
`cold`/`warm` mode the way this CLI binary has one):

```bash
python3 scripts/bench/bench_gridoxide_native.py grid.json [backend]
```

## 3. Benchmark power-grid-model

```bash
python3 -m venv .venv-pgm
.venv-pgm/bin/pip install power-grid-model
.venv-pgm/bin/python3 scripts/bench/bench_pgm.py grid.json
```

`power-grid-model` ships as a prebuilt wheel — no C++ build needed.

## Interpreting results

Compare gridoxide's `total (guess + NR)` line against PGM's `min`/`mean` (warm, repeated calls on an
already-built model) and `cold (construct+calc)` (includes PGM's own model-build overhead) figures — use
gridoxide's `warm` mode for this (see step 2), since PGM's `min`/`mean` are themselves warm (repeated calls on
one persistent `PowerGridModel`), not `cold`. Sample voltage output (`voltage_mag min/max` from gridoxide,
`u_pu min/max` from PGM) should match closely if both are solving the same input correctly — a large mismatch
there means something is wrong with the comparison, not just the timing.

`newton_raphson`-only time (ms/run, 50 warm repeats) vs. PGM's own `mean` (5 warm runs):

| Nodes | Scalar | Block | Klu | PGM |
|---|---|---|---|---|
| 192 | 1.50 | 0.67 | 0.45 | 0.42 |
| 1,003 | 8.38 | 3.15 | 2.47 | 0.93 |
| 2,605 | 21.67 | 8.10 | 6.71 | 2.49 |

PGM is clearly faster than any gridoxide backend on this synthetic radial distribution/LV topology, even
warm-vs-warm — a real, standing gap (see the top-level README's "Experimental backends" section). Contrast
with step 4 below, where gridoxide's `Klu` backend is frequently *faster* than lightsim2grid's own KLU-backed
C++ solver on real transmission-topology grids — the comparison depends heavily on the grid's topology, not
just implementation language or which C library both ultimately call into.

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

gridoxide's own side runs through its native Python bindings (`bench_gridoxide_native.py`, see "Python
bindings" below) rather than a subprocess call into a compiled `bench_network.rs` binary — every tool in this
comparison is driven the same way now: a small Python script constructing one persistent model/solver object
and timing repeated `solve()`/`ac_pf()`/`calculate_power_flow()`/`runpp()` calls on it with
`time.perf_counter()`.

```bash
python3 -m venv .venv-case-suite
.venv-case-suite/bin/pip install maturin numpy scipy power-grid-model pandapower lightsim2grid pypowsybl
VIRTUAL_ENV=.venv-case-suite .venv-case-suite/bin/maturin develop --release --features python,klu
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

Every tool's own warm-run mean (5 timed calls on one persistent model/solver object, `time.perf_counter()`),
gridoxide included (`bench_gridoxide_native.py`/`PowerFlowModel`, always warm — see "Python bindings" above):

| case | buses | scalar | block | klu | PGM | lightsim2grid (KLU) | pypowsybl | pandapower |
|---|---|---|---|---|---|---|---|---|
| case14 | 15 | 0.043 | 0.044 | 0.026 | 0.226 | 0.026 | 1.665 | 17.377 |
| case118 | 119 | 0.202 | 0.171 | 0.106 | FAILED¹ | 0.157 | 4.844 | 15.976 |
| case_illinois200 | 201 | 0.662 | 0.306 | 0.257 | 0.360 | 0.309 | 7.013 | 16.418 |
| case300 | 301 | 1.701 | 0.607 | 0.596 | FAILED¹ | 0.538 | 8.326 | 18.117 |
| case1354pegase | 1355 | 6.820 | 3.074 | 2.413 | FAILED² | 2.556 | 37.686 | 25.496 |
| case1888rte | 1889 | 11.505 | 3.895 | 3.562 | FAILED¹ | FAILED³ | FAILED⁴ | 28.751 |
| case2848rte | 2849 | 20.224 | 6.372 | 5.274 | FAILED¹ | 7.996 | FAILED⁴ | 33.078 |
| case2869pegase | 2870 | 26.854 | 8.852 | 7.820 | FAILED² | 6.660 | 107.720 | 34.977 |
| case3120sp | 3121 | 19.025 | 6.887 | 5.397 | FAILED² | 6.572 | 83.408 | 33.200 |
| case6495rte | 6496 | 66.981 | 19.588 | 17.621 | FAILED² | FAILED³ | FAILED⁴ | 152.003 |
| case6515rte | 6516 | 77.745 | 23.947 | 21.146 | FAILED² | FAILED³ | FAILED⁴ | 58.419 |
| case9241pegase | 9242 | 116.537 | 37.282 | 29.545 | FAILED² | 25.732 | 404.777 | 87.930 |

`block` used to show `N/A` here — it panicked on any bus modeled as `PV` (via PGM's `voltage_regulator`,
which every one of these 12 real cases uses for its generators), since its 2×2-block-per-bus indexing
assumed every non-slack bus was `PQ`. Fixed by giving a `PV` bus's block a dummy `ΔVmag = 0` row instead of
a real Q-mismatch row (`solver::build_jacobian_blocks`) — mathematically equivalent to the scalar backend's
actual dimension reduction, confirmed by convergence to the same voltages in the same iteration count on
every case above. Fixing this also surfaced a real, previously-latent bug in `BlockLu::refactor` unrelated to
`PV` buses themselves: it scattered `adj.row(perm[j])`'s values where it needed `adj.col(perm[j])`'s — silently
wrong on any matrix that isn't value-symmetric (the real Jacobian isn't; every hand-written unit test in
`block_sparse.rs` happened to use symmetric off-diagonal blocks, which is why it went unnoticed) — see
`BlockAdjacency::col` and the new `solve_asymmetric_off_diagonal_matches_dense_reference` test.

¹ PGM's own `IterationDiverge`: fails to converge within 20 iterations, PGM's own default. `case14` and
`case_illinois200` are the two cases where PGM does converge; `case14`'s converged voltages match gridoxide's
exactly (`voltage_mag`/`u_pu` both 1.010000/1.090000), confirming the converted data itself is correct there.
`case_illinois200` converges in both but to visibly different voltages (gridoxide: 1.0082/1.0400, PGM:
1.0101/1.0548) — both plausible, neither obviously wrong, not chased further; PGM's own particular
Newton-Raphson implementation is simply less robust on these specific harder cases via this input path than
gridoxide's.
² PGM's own `SparseMatrixError` ("possibly singular matrix") — raised during `PowerGridModel()` construction,
before any iteration runs.
³ lightsim2grid's own `ac_pf` reports divergence (`V.shape[0] == 0`).
⁴ pypowsybl's own `run_ac` fails to converge (`MAX_ITERATION_REACHED` or `Unrealistic state`) using the same
"basic" (no damping, flat start) parameters as powsybl's own benchmark repo — *without* the explicit
phase-shift-zeroing workaround that repo's own `MatpowerUtil.java` applies before benchmarking these same
three cases (confirmed directly: applying that workaround via `pypowsybl` does make powsybl-open-loadflow
converge on all three, in ~4 iterations each — see matpower_to_pgm.py's docstring).

gridoxide (`scalar`/`block`/`klu`) and pandapower (its own native, no-cross-tool-conversion path) are the
only two of five tools that converge on **all 12** cases. `case1888rte`, `case6495rte`, and `case6515rte` are
hard for every tool that doesn't special-case them — a genuine property of those three cases' data (RTE's own
real production grid, per the case names), not a gridoxide-, PGM-, lightsim2grid-, or pypowsybl-specific gap.

`block` is consistently faster than `scalar` (2–3x on the larger cases: `case9241pegase` 37.28ms vs
116.54ms) despite `scalar` being backed by `faer`, a mature general-purpose sparse solver, and `block`'s LU
being a from-scratch, no-partial-pivoting implementation — the payoff of factoring at 2×2-block granularity
(half as many colamd-ordered elimination steps, no interleaved-scalar bookkeeping) is apparently large enough
to outweigh that. `klu` is faster still, consistently ~10-20% ahead of `block` on the larger cases (e.g.
`case9241pegase`: 29.55ms vs 37.28ms). Comparing gridoxide's `klu` against lightsim2grid's own KLU-backed C++
solver, the two are roughly competitive: `klu` is faster on some cases (`case2848rte`: 5.27ms vs 8.00ms),
lightsim2grid faster on others (`case9241pegase`: 29.55ms vs 25.73ms; `case300`: 0.60ms vs 0.54ms, though at
`case300`'s sub-millisecond scale that gap is close to run-to-run timing noise). This is the *warm* comparison
(both gridoxide's `PowerFlowModel` and lightsim2grid's `GridModel` reuse one persistent solver object across
their 5 timed calls — see the top-level README's "Reusing factorization across repeated solves"); the
earlier `cold` numbers (fresh symbolic factorization every repeat) made gridoxide look 1.3–1.7x *slower*
across the board, which `perf`-profiling one case (`case9241pegase`) traced to that redone-every-time
ordering step, something lightsim2grid's own benchmark never does — not a genuine solver-speed gap. Both
`pypowsybl` and `pandapower` are markedly slower than every C-backed solver here, consistent with being
heavier, more general-purpose Python frameworks not specifically tuned for repeated single-scenario power flow.
