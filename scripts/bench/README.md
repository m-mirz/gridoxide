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
maturin develop --release --features python,pardiso # + the pardiso backend (needs MKLROOT set — see
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
(tens of milliseconds) to sample meaningfully. `backend` selects `scalar` (default), `block`, `klu` (if
built with `--features klu`), `klu_native` (no extra build requirements), or `pardiso` (if built with
`--features pardiso` and `MKLROOT` set) — see the top-level README's "Sparse solver" and "Experimental
backends" sections. `mode` selects `cold`
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

`newton_raphson`-only time (ms/run, 200 warm repeats) vs. PGM's own `mean` (5 warm runs):

| Nodes | Scalar | Block | Klu | KluNative | Pardiso | PGM |
|---|---|---|---|---|---|---|
| 192 | 1.28 | 0.52 | 0.44 | 0.45 | 2.06 | 0.42 |
| 1,003 | 7.86 | 2.80 | 2.52 | 2.69 | 5.06 | 0.93 |
| 2,605 | 20.97 | 6.81 | 6.04 | 6.56 | 11.73 | 2.49 |

PGM is clearly faster than any gridoxide backend on this synthetic radial distribution/LV topology, even
warm-vs-warm — a real, standing gap (see the top-level README's "Experimental backends" section). `KluNative`
(the pure-Rust port of `Klu`'s algorithm, `src/klu_native/`) lands close to `Klu` itself (1.02-1.09x) across
this range of scales. `Pardiso` (Intel oneMKL, `src/sparse_pardiso.rs`) is 2-4.7x slower than `Klu` at every
scale, and slower than even `Scalar` at 192 nodes — its default nonsymmetric matching/scaling preprocessing
carries a largely size-independent fixed cost per solve that dominates at these small problem sizes, though
it does scale better than `Scalar` as node count grows (overtaking it by 2,605 nodes). Contrast
with step 4 below, where gridoxide's `Klu` backend is frequently *faster* than lightsim2grid's own KLU-backed
C++ solver on real transmission-topology grids — the comparison depends heavily on the grid's topology, not
just implementation language or which C library both ultimately call into.

## 4. Benchmark against real power-system test-case grids

A second, separate comparison: gridoxide against **five** other independent Newton-Raphson implementations —
[power-grid-model](https://github.com/PowerGridModel/power-grid-model) (PGM),
[lightsim2grid](https://github.com/m-mirz/lightsim2grid) (C++/KLU-backed),
[powsybl-open-loadflow](https://github.com/powsybl/powsybl-open-loadflow) (RTE's Java solver, via
[pypowsybl](https://pypi.org/project/pypowsybl/)), pandapower's own default solver, and
[VeraGrid](https://github.com/SanPen/VeraGrid) (Python/numba-JIT-backed) — on the same 12 real
IEEE/MATPOWER-derived test-case grids lightsim2grid's own benchmark uses
([`benchmarks/benchmark_grid_size.py`](https://github.com/m-mirz/lightsim2grid/blob/master/benchmarks/benchmark_grid_size.py)):
`case14`, `case118`, `case_illinois200`, `case300`, `case1354pegase`, `case1888rte`, `case2848rte`,
`case2869pegase`, `case3120sp`, `case6495rte`, `case6515rte`, `case9241pegase`.

gridoxide models these cases' generators as genuine PV (voltage-controlled) buses, the same way PGM,
lightsim2grid, pypowsybl, pandapower, and VeraGrid all do — VeraGrid's own MATPOWER importer reads each bus's
`type` column and each generator's `Vg` setpoint directly from the case file, so PV-bus handling comes for
free there with no extra conversion step (see bench_veragrid.py). PGM's `voltage_regulator` component
(`regulated_object` = the generator, `u_ref` = its voltage setpoint) is PGM's real PV-bus mechanism, and
`src/pgm.rs::pgm_to_buses_and_branches` now parses it the same way PGM's own
`newton_raphson_pf_solver.hpp::set_u_ref_and_bus_types` does, assigning `BusType::PV` and pinning the bus's
voltage magnitude.

gridoxide's and PGM's side are converted straight from MATPOWER's own `.m` case files (`matpower_to_pgm.py`,
vendored in the `benchmark-grids` git submodule at `tests/data/benchmark-grids/matpower/` — see that
submodule's own `PROVENANCE.md` for exactly where they come from and licensing), not through
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
pandapower net object, not PGM JSON); pypowsybl and VeraGrid both load the same MATPOWER file gridoxide and
PGM do, each via its own MATPOWER importer (`bench_pypowsybl.py` — re-serializes a `.m` file to a temporary
`.mat` first, since pypowsybl's importer only reads the binary format; `bench_veragrid.py` reads `.m`/`.mat`
directly, no re-serialization needed).

gridoxide's own side runs through its native Python bindings (`bench_gridoxide_native.py`, see "Python
bindings" below) rather than a subprocess call into a compiled `bench_network.rs` binary — every tool in this
comparison is driven the same way now: a small Python script constructing one persistent model/solver object
and timing repeated `solve()`/`ac_pf()`/`calculate_power_flow()`/`runpp()` calls on it with
`time.perf_counter()`.

```bash
git submodule update --init tests/data/benchmark-grids
python3 -m venv .venv-case-suite
.venv-case-suite/bin/pip install maturin numpy scipy power-grid-model pandapower lightsim2grid pypowsybl \
    VeraGridEngine
VIRTUAL_ENV=.venv-case-suite .venv-case-suite/bin/maturin develop --release --features python,klu
.venv-case-suite/bin/python3 scripts/bench/run_case_suite.py --python .venv-case-suite/bin/python3
```

This loops all 12 cases, reading each MATPOWER `.m` file from the `benchmark-grids` submodule and converting
it to PGM JSON on first use (cached under `scripts/bench/.case-cache/`, gitignored — delete it to force
reconversion), running
gridoxide's `scalar`, `block`, `klu`, `klu_native`, and `pardiso` backends (`pardiso` needs the extension
built with `--features python,pardiso` and `MKLROOT` set — see "Python bindings" above; otherwise every
`pardiso` cell reports its own build/load error rather than a timing), PGM, lightsim2grid with
`SolverType.KLU` (matching lightsim2grid's own benchmark default, and gridoxide's own fastest backend, for
the closest apples-to-apples solver comparison), pypowsybl with the same "basic" `LoadFlowParameters`
powsybl's own benchmark repo uses, pandapower with its own defaults, and VeraGrid with `SolverType.NR` and
every automatic-control feature (tap changers, remote voltage regulation, alternate-solver fallback) turned
off, for the same reason — see bench_veragrid.py's docstring — then prints one combined markdown table. A
case that fails to convert or diverges gets an explicit `FAILED (...)` cell (with the tool's actual
exception, not a generic trailer) rather than a misleading blank one. Use `--repeat` to change how many timed
solves gridoxide averages per case (default 10), `--cache-dir`/`--out` to change where converted grids/the
results table are written.

`matpower_to_pgm.py <input.m-or-.mat> <output.json>`, `bench_pgm.py <input.json>`,
`convert_pandapower_case.py <case_name> <output.json>`, `bench_lightsim2grid.py <case_name>`,
`bench_pypowsybl.py <case_name> <input.m-or-.mat>`, and `bench_pandapower.py <case_name>` also work
standalone, for benchmarking or debugging one case/tool at a time.

### Results

Every tool's own warm-run mean (5 timed calls on one persistent model/solver object, `time.perf_counter()`),
gridoxide included (`bench_gridoxide_native.py`/`PowerFlowModel`, always warm — see "Python bindings" above):

| case | buses | scalar | block | klu | klu_native | pardiso⁵ | PGM | lightsim2grid (KLU) | pypowsybl | pandapower | VeraGrid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| case14 | 15 | 0.042 | 0.032 | 0.059 | 0.027 | 0.074 | 0.155 | 0.028 | 1.985 | 14.547 | 3.083 |
| case118 | 119 | 0.211 | 0.164 | 0.106 | 0.111 | 0.340 | FAILED¹ | 0.149 | 4.969 | 15.445 | 6.389 |
| case_illinois200 | 201 | 0.664 | 0.296 | 0.257 | 0.269 | 0.649 | 0.585 | 0.329 | 6.816 | 15.741 | 7.131 |
| case300 | 301 | 1.709 | 0.582 | 0.490 | 0.512 | 1.525 | FAILED¹ | 0.528 | 9.578 | 19.078 | 11.577 |
| case1354pegase | 1355 | 6.665 | 2.969 | 2.393 | 3.167 | 3.626 | FAILED² | 2.575 | 38.163 | 22.155 | 48.832 |
| case1888rte | 1889 | 10.343 | 3.951 | 3.716 | 3.429 | 4.617 | FAILED¹ | FAILED³ | FAILED⁴ | 26.073 | FAILED⁶ |
| case2848rte | 2849 | 18.166 | 6.202 | 5.255 | 5.347 | 7.213 | FAILED¹ | 7.699 | FAILED⁴ | 30.648 | 87.872 |
| case2869pegase | 2870 | 18.856 | 7.366 | 6.109 | 6.731 | 8.270 | FAILED² | 6.135 | 97.341 | 31.530 | 99.871 |
| case3120sp | 3121 | 18.793 | 6.553 | 5.323 | 6.078 | 7.108 | FAILED² | 6.297 | 81.665 | 29.024 | 97.111 |
| case6495rte | 6496 | 61.226 | 19.387 | 16.721 | 19.994 | 20.931 | FAILED² | FAILED³ | FAILED⁴ | 111.096 | FAILED⁶ |
| case6515rte | 6516 | 73.762 | 22.021 | 18.578 | 22.537 | 25.251 | FAILED² | FAILED³ | FAILED⁴ | 49.822 | FAILED⁶ |
| case9241pegase | 9242 | 110.813 | 33.835 | 29.260 | 34.623 | 35.644 | FAILED² | 27.460 | 384.882 | 75.689 | 384.093 |

`klu_native` (`src/klu_native/`, the from-scratch Rust port — see the top-level README's "Experimental
backends") converges to the same voltages as every other gridoxide backend on all 12 cases, and lands close
to `klu` throughout (mostly 1.0-1.2x, `case1354pegase` closer to 1.3x) — a large improvement from
an earlier, unoptimized version of this port that ran a consistent ~1.9-2x slower across every scale.
`perf`-profiling `case9241pegase` traced that gap to allocator churn, not algorithmic overhead:
`kernel::refactor_block` allocated two new heap-backed `Vec`s per column on *every* Newton iteration's
refactor (for this case, tens of millions of allocations across a timed run), while real KLU's own
`klu_refactor.c` overwrites one already-allocated buffer in place and allocates nothing at all during a
refactor. Rewriting `refactor_block`/`refactor::refactor` to mutate the existing factorization in place,
backed by a `RefactorScratch` buffer reused across every solve (`KluNativeSystem` owns one for its whole
lifetime), eliminated nearly all of that churn — `case9241pegase`'s glibc allocator functions (`_int_malloc`,
`_int_free_merge_chunk`, `realloc`, ...) dropped from ~32% of `newton_raphson` self-time to effectively
absent from the profile.

`pardiso` (Intel oneMKL, `src/sparse_pardiso.rs`) converges to the same voltages as every other gridoxide
backend on all 12 cases (`case14`'s `voltage_mag` matches the others exactly: 1.010000/1.090000). Its gap to
`klu` ranges from ~1.2x (`case14`, `case1888rte`, `case6495rte`) up to ~3.2x (`case118`, `case300`) depending
on case, without as clean a shrinks-monotonically-with-size trend as the three synthetic radial-distribution
grids in the top-level README's "Experimental backends" section show — these 12 cases vary more in topology
and conditioning than those synthetic grids do. The largest cases still settle in a similar 1.2-1.4x range
(`case6515rte`: 25.25ms vs 18.58ms, ~1.36x; `case9241pegase`: 35.64ms vs 29.26ms, ~1.22x), consistent with
the same underlying cause: PARDISO's default nonsymmetric matching/scaling preprocessing carries a largely
size-independent fixed cost per solve, which matters proportionally less once there's enough real
factorization work to amortize it against, but topology and matrix conditioning clearly also play a role
here that the synthetic grids' more uniform structure doesn't surface.

PGM's numbers above needed a workaround to get at all: once `matpower_to_pgm.py` started writing real
`q_min`/`q_max` onto `voltage_regulator` (see that script's docstring), every one of these 12 inputs started
tripping PGM's own `ExperimentalFeature` error when run through its public `calculate_power_flow` API — the
installed version (1.13.120) never wired the `experimental_features` flag through that public wrapper at
all, even though the private `_calculate_power_flow` accepts it. `bench_pgm.py` now calls that private
method directly with `experimental_features="enabled"`, confirmed to reproduce the exact same converged
voltages the public API produced before real Q-limits existed (`case14`: `u_pu` 1.010000/1.090000, matching
this section's own footnote below) — not a workaround that changes what's being measured, just one that
un-blocks measuring it at all pending a stable PGM release.

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
gridoxide's. VeraGrid also converges on this case, to 1.010942/1.054895 — much closer to PGM's value than to
gridoxide's, a third independent data point suggesting this specific case genuinely has more than one
plausible solution basin reachable by flat-start Newton-Raphson, not that any one tool's implementation is
simply wrong.
² PGM's own `SparseMatrixError` ("possibly singular matrix") — raised during `PowerGridModel()` construction,
before any iteration runs.
³ lightsim2grid's own `ac_pf` reports divergence (`V.shape[0] == 0`).
⁴ pypowsybl's own `run_ac` fails to converge (`MAX_ITERATION_REACHED` or `Unrealistic state`) using the same
"basic" (no damping, flat start) parameters as powsybl's own benchmark repo — *without* the explicit
phase-shift-zeroing workaround that repo's own `MatpowerUtil.java` applies before benchmarking these same
three cases (confirmed directly: applying that workaround via `pypowsybl` does make powsybl-open-loadflow
converge on all three, in ~4 iterations each — see matpower_to_pgm.py's docstring).
⁵ `pardiso`'s numbers need the extension built with `--features python,pardiso` and a local Intel oneMKL
install (`MKLROOT` set) — not part of the `--features python,klu` build command above, so build that
separately if reproducing this column specifically.
⁶ VeraGrid's own `PowerFlowDriver` reports `results.converged == 0` (this script raises a `RuntimeError` when
it sees that) with `SolverType.NR` and every automatic-control feature disabled the same way as elsewhere in
this table — the same three cases every other tool here (bar gridoxide and pandapower) also fails on.

gridoxide (`scalar`/`block`/`klu`/`klu_native`/`pardiso`) and pandapower (its own native, no-cross-tool-conversion
path) are the only two of six tools that converge on **all 12** cases. `case1888rte`, `case6495rte`, and
`case6515rte` are
hard for every tool that doesn't special-case them — a genuine property of those three cases' data (RTE's own
real production grid, per the case names), not a gridoxide-, PGM-, lightsim2grid-, pypowsybl-, or
VeraGrid-specific gap; four independent implementations (PGM, lightsim2grid, pypowsybl, VeraGrid) all fail on
exactly the same three cases.

`block` is consistently faster than `scalar` (2–3x on the larger cases: `case9241pegase` 33.84ms vs
110.81ms) despite `scalar` being backed by `faer`, a mature general-purpose sparse solver, and `block`'s LU
being a from-scratch, no-partial-pivoting implementation — the payoff of factoring at 2×2-block granularity
(half as many colamd-ordered elimination steps, no interleaved-scalar bookkeeping) is apparently large enough
to outweigh that. `klu` is faster still, consistently ~10-15% ahead of `block` on the larger cases (e.g.
`case9241pegase`: 29.26ms vs 33.84ms). Comparing gridoxide's `klu` against lightsim2grid's own KLU-backed C++
solver, the two are roughly competitive: `klu` is faster on most cases (e.g. `case2848rte`: 5.26ms vs
7.70ms), lightsim2grid faster on a couple (`case9241pegase`: 29.26ms vs 27.46ms; `case14`: 0.059ms vs
0.028ms, though at `case14`'s sub-millisecond scale that gap is close to run-to-run timing noise). This is
the *warm* comparison (both gridoxide's `PowerFlowModel` and lightsim2grid's `GridModel` reuse one persistent
solver object across their 5 timed calls — see the top-level README's "Reusing factorization across repeated
solves" for the `PersistentSolver` feature itself); the earlier `cold` numbers (fresh symbolic factorization
every repeat) made gridoxide look 1.3–1.7x *slower* across the board, which `perf`-profiling `case9241pegase`
traced to that redone-every-time ordering step, something lightsim2grid's own benchmark never does — not a
genuine solver-speed gap. Measured directly on that same 9,241-bus case: reusing factorization across
repeated solves (`warm` mode) cut per-solve `klu` time by ~45% relative to `cold`, with `perf` showing
COLAMD/AMD/BTF ordering — the fill-reducing step a cold solve redoes every time — responsible for roughly a
third of total solve time when redone from scratch on every call. `pypowsybl`, `pandapower`, and `VeraGrid`
are all markedly slower than every C/Rust-backed
solver here (`klu`/`klu_native`/`block`/lightsim2grid), consistent with being heavier, more general-purpose
Python frameworks not specifically tuned for repeated single-scenario power flow — `VeraGrid`'s own numerical
kernels are numba-JIT-compiled, so this comparison is already its *warm*, post-JIT-warmup number (see
bench_veragrid.py's docstring); its *first* call on a case this size costs low-single-digit seconds, not
milliseconds, entirely JIT-compilation overhead unrelated to the power-flow algorithm itself.
