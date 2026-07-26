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
gridoxide. See `gridoxide.matpower`'s module docstring (`python/gridoxide/matpower.py` — the conversion logic
moved into the pip package itself; `matpower_to_pgm.py` is now only a thin CLI wrapper around it) for the
full derivation, including a real,
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

`check_matpower_residual.py [<case>...|--all]` is the correctness counterpart to all of the above, and the
one to reach for first when a converted case looks subtly wrong: it checks a solved case against the
MATPOWER file's own power-flow equations, with no second tool involved, and exits nonzero on any residual
above `--tol` (default 1e-3 MVA). See the accuracy section below.

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

These timings predate the four correctness fixes described in the accuracy section below, and were not fully
re-measured after them — spot-checking `klu_native` on `case14`/`case2869pegase`/`case9241pegase` afterwards
gave 0.027/6.914/35.347 ms against the table's 0.027/6.731/34.623, i.e. run-to-run noise. That's expected:
the fixes add shunt admittances to Y-bus diagonal entries that already exist, so they change neither sparsity
structure nor iteration counts materially. The *voltages* those runs converged to did change, though — see
below.

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
`q_min`/`q_max` onto `voltage_regulator` (see `gridoxide.matpower`'s module docstring), every one of these 12 inputs started
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
`case_illinois200` are the two cases where PGM does converge, and it now agrees with gridoxide to 0.0000% on
both (see the accuracy section's own table).

`case_illinois200` is worth recording as a cautionary note. It used to converge in both tools but to visibly
different voltages (gridoxide: 1.0082/1.0400, PGM: 1.0101/1.0548), with VeraGrid landing at 1.010942/1.054895
— much closer to PGM. That was read here as "three independent data points suggesting this case genuinely has
more than one plausible solution basin reachable by flat-start Newton-Raphson, not that any one tool's
implementation is simply wrong." The real explanation was mundane and entirely gridoxide's: it was discarding
this case's four bus shunts (190 MVAr), so PGM and VeraGrid agreed with each other because they were both
right. Two tools agreeing against a third is evidence about the third, not about the problem having multiple
solutions — see the accuracy section's "What the cross-tool table missed" for the bug itself.
² PGM's own `SparseMatrixError` ("possibly singular matrix") — raised during `PowerGridModel()` construction,
before any iteration runs.
³ lightsim2grid's own `ac_pf` reports divergence (`V.shape[0] == 0`).
⁴ pypowsybl's own `run_ac` fails to converge (`MAX_ITERATION_REACHED` or `Unrealistic state`) using the same
"basic" (no damping, flat start) parameters as powsybl's own benchmark repo — *without* the explicit
phase-shift-zeroing workaround that repo's own `MatpowerUtil.java` applies before benchmarking these same
three cases (confirmed directly: applying that workaround via `pypowsybl` does make powsybl-open-loadflow
converge on all three, in ~4 iterations each — see `gridoxide.matpower`'s module docstring).
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
real production grid, per the case names); four independent implementations (PGM, lightsim2grid, pypowsybl,
VeraGrid) all fail on exactly the same three cases.

**gridoxide's own convergence on those three cases does not demonstrate a more robust solver, and shouldn't be
read as one.** Its MATPOWER→PGM conversion rounds `transformer.clock` to the nearest 60°, and every phase
shift in these cases is well under 30°, so all of them are rounded to zero — 4 branches in `case1888rte` (up
to 9.95°), 17 in `case6495rte` and 16 in `case6515rte` (16.60° each). Zeroing phase shifts is *precisely* the
workaround powsybl's own `MatpowerUtil.java` applies before benchmarking these same three cases, and footnote
4 below already records that applying it via `pypowsybl` makes powsybl-open-loadflow converge on all three in
~4 iterations. So gridoxide is converging on the same easier problem powsybl needs the workaround to reach,
having applied the workaround implicitly in its own data conversion, rather than solving the harder published
network the other tools diverge on. See the accuracy section's footnote 5 and `gridoxide.matpower`'s module
docstring; `check_matpower_residual.py` quantifies what the zeroed shifts cost (a five-figure MVA
power-balance residual on `case1888rte`).

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

### Accuracy results

A MATPOWER `.m` case file is power-flow *input* (the `bus`/`gen` matrices), not a published solved-case
snapshot the way CGMES's `SvVoltage` is (section 6 below). That makes it tempting to conclude there's no
ground truth here at all, and an earlier version of this section did exactly that — then read the resulting
tool-vs-tool table as evidence that independent solvers simply land on slightly different valid solutions.
**That was wrong, and it hid a real bug for as long as it stood.** A case file has no reference *solution*,
but it fully specifies the power-flow *equations*, and any correct solution must satisfy them:

```bash
python3 scripts/bench/check_matpower_residual.py --all
```

`check_matpower_residual.py` rebuilds Ybus directly from the `.m` file (following `makeYbus.m`'s own
conventions) and evaluates `dS = V .* conj(Ybus @ V) - S_specified` on the converged solution, checking P at
every PV/PQ bus and Q at every PQ bus (a PV bus's reactive output and a slack bus's P/Q are free variables,
so those residuals are meaningless and skipped). This is an *absolute* check — it needs no second tool — and
it is the check that should be trusted when it disagrees with the cross-tool table below. See "What the
cross-tool table missed" at the end of this section for what it caught.

`accuracy_case_suite.py` separately reports every other tool's own per-bus voltage-magnitude
deviation from **gridoxide's own** converged solution, matched by MATPOWER bus number rather than array
position (each tool's own bus-ID convention was confirmed empirically, not assumed — see the script's
docstring: VeraGrid sets `Bus.code` to the MATPOWER id directly, pandapower's built-in cases set
`net.bus["name"]` to it, lightsim2grid's `ac_pf` output reuses that same pandapower indexing (confirmed to
match `net.res_bus` to ~1e-8), and pypowsybl's own `Line`/`TwoWindingsTransformer` elements — not its bus ids
themselves, see below — are named `"LINE-<from>-<to>"`/`"TWT-<from>-<to>"`). Using gridoxide as that anchor
is a presentational choice only, and a treacherous one: its five backends agreeing to bit-identical voltages
on all 12 cases (confirmed above) says the five backends share a front end, not that the front end is right,
and a table of deviations *from* the anchor cannot distinguish "every other tool is slightly off" from "the
anchor is off and every other tool agrees with each other." Read this table together with the residual check
above, which has no anchor at all.

```bash
python3 scripts/bench/accuracy_case_suite.py --out accuracy.md
```

Needs the same environment as this section's own `run_case_suite.py` setup, but runs every tool in-process in
one interpreter (no subprocess-per-tool the way the timing suite does) since it needs each tool's full
per-bus array, not just a `mean=Xms` line to regex out of stdout.

**A real bug in this benchmark's own pypowsybl-matching logic, found and fixed while building this table.**
pypowsybl's own bus ids look like `"VL-<n>_<k>"`, and it's tempting to read `<n>` as the MATPOWER bus number
(this project's own CGMES benchmark, section 6, does exactly that for a different bus-id scheme) — but for
pypowsybl's MATPOWER importer specifically, that's wrong for any bus that's only ever a transformer's
*secondary* side: confirmed directly on case14, where transformers `TWT-4-7`/`TWT-4-9`/`TWT-5-6` connect
`VL-4_0`→`VL-4_1`, `VL-4_0`→`VL-4_2`, and `VL-5_0`→`VL-5_1` — i.e. pypowsybl names a transformer's secondary
voltage level after its *primary* side's own bus number with an incrementing suffix, not the secondary's own
true number. `"VL-4_1"` is really MATPOWER bus **7**, `"VL-4_2"` is bus **9** — naively parsing the leading
number silently compared gridoxide's real bus 4 against pypowsybl's bus 7 or bus 9 instead, and dropped bus
4's own two duplicate-keyed dict entries down to one merged, wrong value. Fixed by reading each `Line`'s/
`TwoWindingsTransformer`'s own id string instead (`"LINE-<from>-<to>"`/`"TWT-<from>-<to>"`, with a `"#<k>"`
disambiguator on the last number for parallel branches between the same bus pair, e.g. case118's
`"LINE-42-49#0"` — confirmed to always carry the two true endpoint numbers, checked directly on case14
through case9241pegase) — see `pypowsybl_bus_to_matpower_id` in the script for the fix itself.

The fix is directly confirmed by cross-checking against VeraGrid, an entirely independent tool with its own
MATPOWER importer and no shared code path with pypowsybl's bus-naming scheme at all: after the fix,
pypowsybl's own deviation-from-gridoxide numbers now match VeraGrid's almost exactly on every case that
converges on both (`case9241pegase` and `case1354pegase`: identical to 4 decimal places on both, currently
`0.0000%/0.0008%/2.3527%` and `0.0000%/0.0018%/2.7037%`) — two independently-implemented tools landing on the
same answer once measured correctly is strong evidence the numbers below are real, not another matching
artifact. That agreement is also what made the *gridoxide*-side bugs diagnosable: the closer these two
independent tools matched each other, the harder it was to keep attributing their shared distance from
gridoxide to ordinary solver-to-solver variation (see "What the cross-tool table missed" below):

| case | PGM | pypowsybl | VeraGrid | pandapower | lightsim2grid |
|---|---|---|---|---|---|
| case14 | n=14 median=0.0000% p90=0.0000% max=0.0000% | n=14 median=0.0024% p90=0.0139% max=0.0674% | n=14 median=0.0024% p90=0.0139% max=0.0674% | n=14 median=0.0024% p90=0.0139% max=0.0674% | n=14 median=0.0024% p90=0.0139% max=0.0674% |
| case118 | FAILED¹ | n=118 median=0.0000% p90=0.0001% max=0.1740% | n=118 median=0.0000% p90=0.0001% max=0.1114% | n=118 median=0.0000% p90=0.0002% max=1.3665% | n=118 median=0.0000% p90=0.0001% max=0.1114% |
| case_illinois200 | n=200 median=0.0000% p90=0.0000% max=0.0000% | n=200 median=0.0148% p90=0.0536% max=0.1531% | n=200 median=0.0767% p90=0.2268% max=1.0768% | n=199 median=1.3989% p90=3.0931% max=5.8083% | n=199 median=1.3989% p90=3.0931% max=5.8083% |
| case300 | FAILED¹ | n=300 median=0.0049% p90=0.1984% max=0.7585% | n=300 median=0.0007% p90=0.1696% max=0.7585% | n=300 median=0.1741% p90=4.5251% max=10.0565% | n=300 median=0.0112% p90=4.4378% max=10.0471% |
| case1354pegase | FAILED² | n=1354 median=0.0000% p90=0.0018% max=2.7037%⁵ | n=1354 median=0.0000% p90=0.0018% max=2.7037%⁵ | n=223³ median=1.5827% p90=4.2504% max=9.9509% | n=223³ median=1.5827% p90=4.2504% max=9.9509% |
| case1888rte | FAILED¹ | FAILED | FAILED | n=1790 median=0.8890% p90=3.1720% max=26.9861% | FAILED |
| case2848rte | FAILED¹ | FAILED | n=2848 median=0.0069% p90=0.1249% max=2.8521%⁵ | n=2767 median=1.4167% p90=4.5991% max=22.2243% | n=2767 median=1.5308% p90=4.9142% max=97.9196%⁴ |
| case2869pegase | FAILED² | n=2869 median=0.0000% p90=0.0011% max=2.7529%⁵ | n=2869 median=0.0000% p90=0.0009% max=2.7529%⁵ | n=889³ median=1.8198% p90=4.7823% max=11.8474% | n=889³ median=1.8198% p90=4.7823% max=11.8474% |
| case3120sp | FAILED² | n=3120 median=0.0004% p90=0.0100% max=2.4033% | n=3120 median=0.0130% p90=0.2114% max=1.7435% | n=3119 median=1.6588% p90=5.6333% max=27.6617%⁴ | n=3119 median=1.6588% p90=5.6333% max=27.6617%⁴ |
| case6495rte | FAILED² | FAILED | FAILED | n=6490 median=1.7131% p90=4.9194% max=81.7230%⁴ | FAILED |
| case6515rte | FAILED² | FAILED | FAILED | n=6510 median=1.8431% p90=5.2888% max=81.7200%⁴ | FAILED |
| case9241pegase | FAILED² | n=9241 median=0.0000% p90=0.0008% max=2.3527%⁵ | n=9241 median=0.0000% p90=0.0008% max=2.3527%⁵ | n=9240 median=2.4184% p90=6.8006% max=29.9954%⁴ | n=9240 median=2.4184% p90=6.8006% max=29.9954%⁴ |

¹ PGM's own `IterationDiverge` (fails to converge within 20 iterations) — same footnote-1 cases as the timing
table above.

² PGM's own `SparseMatrixError` ("possibly singular matrix") — same footnote-2 cases as the timing table
above.

³ pandapower/lightsim2grid's own `n` on `case1354pegase`/`case2869pegase` (both PEGASE-derived reductions of
the larger `case9241pegase` network) is far smaller than the case's real bus count — checked directly, and
it's neither a naming collision (`net.bus["name"]` has 1,354 fully unique values for `case1354pegase`) nor a
simple off-by-one/constant-offset shift against gridoxide's own MATPOWER-numbered ids (`net.bus["name"]`
covers ids 2–9,240 vs. gridoxide/PGM's own 3–9,241 for the same case, but the *particular* missing/extra ids
don't follow a fixed offset). pandapower's own bundled `case1354pegase()`/`case2869pegase()` nets apparently
derive their per-bus numbering from the larger 9,241-bus PEGASE network differently than this project's own
vendored `case1354pegase.m`/`case2869pegase.m` files do — a real, unresolved discrepancy specific to these
two cases' own bundled data, not a bug in this benchmark script's matching logic (case14 through case300, and
`case9241pegase` itself, all match pandapower's `net.bus["name"]` to gridoxide's ids with `n` equal to the
full bus count).

⁴ pandapower/lightsim2grid's own `max` deviation on the largest cases (22–98%, `case2848rte` through
`case9241pegase`) is now far outside every other tool's own `max` on the same cases (≤2.9%,
pypowsybl/VeraGrid, which agree with each other almost exactly per the fix above) — a gap that got
*relatively* much larger once the gridoxide-side bugs below were fixed, since those fixes pulled
pypowsybl/VeraGrid down by an order of magnitude and left pandapower/lightsim2grid roughly where they were.
Both load `pandapower.networks.<case>()` rather than the vendored `.m` file, and their bundled data provably
differs from it on some cases (see footnote 3), so part of this is a data difference rather than a solver
difference; `check_matpower_residual.py` deliberately doesn't report them for that reason. Not chased further.

⁵ These remaining `max` figures (2.3–2.9%, concentrated on the "pegase" and RTE cases) are not solver
disagreement: they are the endpoints of branches whose phase shift this benchmark's own MATPOWER→PGM
conversion rounds to zero, because PGM's `transformer.clock` cannot represent a continuous shift angle. Every
one of these cases contains such branches (6 in `case1354pegase` up to 66 in `case9241pegase` — see
`gridoxide.matpower`'s module docstring for the full per-case list), and `check_matpower_residual.py
--zero-phase-shifts` confirms the attribution directly: gridoxide's residual against the as-published network
is nonzero on exactly these 7 cases and drops to exactly 0.0000 MVA against the same network with every shift
zeroed. pypowsybl and VeraGrid both honor the real shift angle, so the gap is gridoxide's conversion, not
their solve.

**What the cross-tool table missed: four real bugs, not "two valid solutions".** The numbers above used to
look very different — `case14` read `median=0.313% p90=1.788% max=2.154%` for *every* one of the five tools,
to three decimal places, and the pegase cases ran to double-digit `max`. Five independently-implemented
solvers reproducing each other's deviation from the anchor that precisely is not five tools each being
slightly wrong; it is those five agreeing with each other and the anchor being wrong. An earlier version of
this section instead concluded the gap was "the ordinary result of two independently-implemented nonlinear
Newton-Raphson solvers converging to two individually valid but not bit-identical solutions" — after
explicitly ruling out a transformer-tap-convention bug (via a standalone 2-bus tap test, still a valid result)
and a too-small `SOURCE_SK` (real, but slack-adjacent only). Both of those investigations were sound; the
conclusion drawn once they came back negative was not, because a suspiciously *identical* column across
independent tools was never treated as the diagnostic it is. `check_matpower_residual.py` settles it without
reference to any tool, and found four distinct defects:

1. **gridoxide ignored every PGM `shunt` component on the PGM-JSON path.** `PowerFlowModel.from_pgm_json`
   (`src/python.rs`) and `examples/bench_network.rs` both built the Y-bus from lines and transformers only,
   never calling `pgm::pgm_shunts_1ph` + `network::stamp_shunts` — which `from_cgmes` right next to it does
   call, and which every `tests/pgm_*_test.rs` calls by hand, so the helpers were well covered while the two
   real entry points silently skipped them. Every bus shunt in all 12 cases was discarded: 1 in `case14`
   (19 MVAr), 14 in `case118`, and 7,327 totalling ~89.5 GVAr in `case9241pegase`. On `case14` this left bus
   9's reactive balance off by exactly `Bs·|V|²` = 20.3010 MVAr while PGM, reading the *same* converted JSON,
   satisfied it exactly — which is what localized the bug to gridoxide's ingestion rather than the conversion.
2. **The converter dropped branch charging on the transformer path.** A MATPOWER branch with an off-nominal
   ratio *and* nonzero `b` became a PGM `transformer`, which has no field for charging susceptance, and the
   `b` was simply not written (4 such branches in `case300`, 52 in `case3120sp`, 11 in `case2848rte`).
   `makeYbus.m` folds it into `Ytt = ys + j·b/2` with `Yff = Ytt/|tap|²`, leaving the series terms untouched,
   so it decomposes exactly into two ordinary `shunt` components — now emitted, driving `case300`'s residual
   from 3.2 MVAr to 0.
3. **The converter flipped the sign of every negative reactance on the transformer path.** `uk` was computed
   as `np.hypot(r, x)`, which is unsigned, so a series-capacitive branch (`x < 0` — real series compensation:
   10 such branches in `case3120sp`, ~40 in each RTE case) came back inductive.
   `network::transformer_admittances_ex` already handles this correctly — it takes `|uk|` for the magnitude
   and applies `sign(uk)` to the recovered reactance — so carrying the sign through `uk` was the whole fix,
   worth 537 MVA of residual on `case3120sp`.
4. **The converter dropped generator reactive output at PQ-typed buses.** `sym_gen.q_specified` was
   hardcoded to `0.0`. That is correct at a PV bus, where the `voltage_regulator` makes Q a free variable, but
   a generator at a bus MATPOWER types PQ has its `Qg` honored as a fixed injection — unusual but real, e.g.
   `case1888rte`'s bus 1005 at `Qg = -19` MVAr, `case2848rte`'s 2534 and 642. The residual at those buses was
   exactly `-Qg`.

With all four fixed, gridoxide's solution satisfies the as-published MATPOWER equations to 0.0000 MVA on 5 of
the 12 cases, and on the other 7 to 0.0000 MVA once the reference network's phase shifts are zeroed the same
way the conversion zeroes them (footnote 5) — the one remaining discrepancy, and a conversion limitation
rather than a solver one. The cross-tool medians above fell by roughly two orders of magnitude as a result
(`case9241pegase` pypowsybl/VeraGrid: `median 0.628% p90 4.008% max 20.838%` → `median 0.0000% p90 0.0008%
max 2.3527%`), and PGM now agrees with gridoxide to 0.0000% on both cases where it converges.

**Building this table surfaced and fixed a real regression.** `case3120sp` previously diverged catastrophically
on *every* gridoxide backend (`voltage_mag` blowing up to roughly -1548/930 p.u.) — confirmed independent of
these Python bindings via the plain `bench_network` Rust binary too, so not specific to this benchmark's own
code. This directly contradicted this same README's section 4 timing table above, which shows `case3120sp`
converging cleanly on every backend (`klu_native`: 6.078ms) — not stale data, either: reconverting
`case3120sp.m` fresh via the current `matpower_to_pgm.py` produced byte-identical JSON to the cached copy.
Bisected (via `git worktree`, testing the commit immediately before) to `network::dc_angle_guess`, a
DC-power-flow-style `PV`-bus angle pre-pass added in a later commit specifically to help CGMES FullGrid
converge, but wired in *unconditionally* at the top of `linear_initial_guess` — every solve on every backend,
CGMES or not, went through it. Its own original justification never actually panned out either (FullGrid
still doesn't converge with or without it), while the full `cargo test --features cgmes` suite (130+ tests
across every backend and CGMES fixture) passes identically with or without it — so it provided no
demonstrated benefit anywhere in this repo while silently breaking a previously-working, previously-published
case. Removed outright (not patched around) in `src/network.rs`; every backend now converges on
`case3120sp` again, matching its pre-regression iteration trace exactly.

## 4b. Batched power flow — the multi-core CPU baseline

`bench_batch.py` measures `batch::BatchSolver`: one topology, many injection scenarios, solved
across cores. This is the time-series / QSTS / Monte Carlo shape, and it exists specifically to be
the baseline that any future GPU work has to beat — `plans/GPU_PLAN.md` §6 is explicit that beating
a *single-threaded* CPU solver is not a result.

```bash
maturin develop --release --features python,klu
python3 scripts/bench/bench_batch.py .case-cache/case9241pegase.json klu 256
```

Scenarios are ±20% uniform load scalings from a fixed seed. The script asserts that voltages agree
across every thread count to < 1e-12, so a scaling number is never reported without the
corresponding correctness check.

**case9241pegase, 256 scenarios, `klu` backend, AMD Ryzen 7 250 (8 physical cores / 16 SMT threads,
16 MiB shared L3).** *Steady state* — the third of three consecutive runs, by which point the
numbers are stable to within ~2%:

| threads | batch (ms) | per solve (ms) | solves/s | speedup |
|---|---|---|---|---|
| 1 | 7,177 | 28.04 | 35.7 | 1.00x |
| 2 | 4,033 | 15.75 | 63.5 | 1.78x |
| 4 | 2,643 | 10.33 | 96.8 | 2.72x |
| 8 | 2,024 | 7.91 | 126.5 | **3.55x** |
| 16 | 1,973 | 7.71 | 129.7 | 3.64x (SMT, not extra cores) |

**case1354pegase, same setup:**

| threads | batch (ms) | per solve (ms) | solves/s | speedup |
|---|---|---|---|---|
| 1 | 610 | 2.384 | 419.5 | 1.00x |
| 2 | 350 | 1.369 | 730.6 | 1.74x |
| 4 | 234 | 0.914 | 1,094.4 | 2.61x |
| 8 | 174 | 0.680 | 1,471.3 | **3.51x** |
| 16 | 185 | 0.721 | 1,386.4 | 3.30x (SMT — *slower* than 8) |

**Read absolute numbers on this machine with suspicion.** It is a thermally constrained laptop APU,
and a cold first run reads far better than anything reproducible: the very first 8-thread
case9241pegase measurement taken on this part was 3.65x, and it never recurred. Every table here is
a steady-state reading, and any A/B comparison below was taken *interleaved* — old build, new
build, old build, new build — because comparing two numbers captured minutes apart on this hardware
measures the heatsink, not the code.

### Precomputed Jacobian offsets

`jacobian::JacobianPattern` derives the Jacobian's sparsity pattern and per-nonzero recipe once per
topology, then refills one reused `Vec<f64>` each iteration instead of rebuilding a
`Vec<(usize, usize, f64)>` from scratch. Every `LinearSolver` backend already discarded the
`(row, col)` half after construction — each caches its own positional mapping into its CSC layout
and reads nothing but `entries[i].2` — so `factor_and_solve_values` hands over just the values.

Measured by interleaved A/B — alternating `maturin develop` between the parent commit and this
one, two rounds, min-of-3 each, so both builds see the same thermal state. ms/solve:

| | case9241 t1 | case9241 t8 | case1354 t1 | case1354 t8 |
|---|---|---|---|---|
| before | 34.33 / 33.46 | 11.47 / 11.43 | 2.560 / 2.576 | 0.717 / 0.741 |
| after | 29.95 / 30.09 | 10.63 / 10.56 | 2.395 / 2.365 | 0.690 / 0.693 |
| **gain** | **11.4%** | **7.5%** | **7.3%** | **5.1%** |

`plans/GPU_PLAN.md` §1 measures assembly at ~36% of iteration time, so on the large case
single-threaded roughly a third of that stage was allocation and index rebuilding rather than
arithmetic that matters.

The gain is *smaller* at 8 threads than at 1 (7.5% vs 11.4%), not larger: once all cores are
running, the solve is memory-stalled in the LU, so assembly is a smaller share of the total and
speeding it up buys proportionally less. The gain is also smaller on the small case, where the
triplet array (~20k nonzeros, ~0.5 MB) fits comfortably in cache and rebuilding it was never
especially expensive; on case9241pegase it is ~150k nonzeros, ~3.6 MB, rebuilt every iteration.

Values are bit-for-bit identical to the previous implementation — `src/jacobian.rs`'s tests compare
`f64::to_bits`, not a tolerance.

### Why this is sub-linear, and why that is not a solver defect

Both cases land at ~3.5x on 8 physical cores — 44% parallel efficiency. Two effects account for
most of the gap:

| Factor | Effect | Cumulative ceiling |
|---|---|---|
| 8 physical cores (16 logical is SMT2) | 8x | 8.0x |
| All-core clock throttle (2,977 -> 2,236 MHz, measured from `/proc/cpuinfo` under load) | x0.75 | 6.0x |
| Shared-L3 / memory-bandwidth contention | remainder | **~3.5x observed** |

The last row was isolated by running **8 single-threaded processes concurrently** instead of 8
threads: separate address spaces share no allocator arenas, no locks and no false sharing, so if
in-process contention were the cause, processes would scale noticeably better. They did not —
3.96x for processes vs. 3.91x for threads, measured back to back on an equally-warm machine. The
ceiling is hardware, not the batch solver.

A caution against over-reading this: an earlier draft of this section claimed scaling degrades with
grid size (63% efficiency on case1354pegase vs 46% on case9241pegase) and attributed it to
case9241pegase's LU factors overflowing the 16 MiB L3. That was an artifact of comparing a
cold-machine case1354pegase run against a warm case9241pegase one. Measured under equal thermal
conditions the two cases scale within 0.05x of each other, and the tidy cache story does not
survive. The working-set effect is real but shows up in the *Jacobian assembly* gain above (11.4%
vs 7.3%), not in thread scaling.

**This machine is a poor benchmark device for this workload.** Re-measure on a desktop or server
part before quoting any of this anywhere load-bearing. The same caveat `plans/GPU_PLAN.md` §5
raises about this machine's *GPU* applies just as much to its CPU.

### What this implies for the GPU plan

The CPU batch path is **memory-bandwidth bound, not compute bound**. That is an argument *for* the
GPU direction rather than against it: bandwidth is precisely what datacenter GPUs have in bulk (an
MI300X has roughly two orders of magnitude more than this APU's LPDDR). It also sharpens what a GPU
claim has to say — the number to beat on this machine is ~127 solves/s steady-state on
case9241pegase, but a publishable claim needs a *server* CPU baseline, since this part is throttle-
and bandwidth-limited in a way a desktop or EPYC/Xeon host would not be.

## 4c. JAX oracle — validating the block-diagonal embedding

`jax_oracle.py` is an independent reimplementation of the batched AC power flow in JAX (f64, CPU,
dense), written to answer questions the Rust solver cannot answer about itself. It is **not** a
performance prototype — it is deliberately the slowest power flow in this repo and must never be
quoted as a speed number.

```bash
python3 -m venv .venv-jax
.venv-jax/bin/pip install jax numpy maturin
VIRTUAL_ENV=$PWD/.venv-jax .venv-jax/bin/maturin develop --release --features python,klu
.venv-jax/bin/python scripts/bench/jax_oracle.py .case-cache/case118.json 8
```

It consumes gridoxide's *own* Y-bus and bus arrays (`ybus_triplets`, `bus_spec`, `initial_guess`,
`zip_term_counts`) rather than re-deriving the model from the input file. That is the point: if the
oracle built its own model, a disagreement could equally be a tap-ratio or shunt-stamping
difference in the converter as a solver bug, and the comparison would establish nothing about the
solver.

Three checks per case. Max |dVm| against `klu`:

| case | buses | 1. oracle vs klu | 2. **BDE vs independent** | 3. oracle vs BatchSolver |
|---|---|---|---|---|
| case14 | 15 | 4.4e-16 | 4.4e-16 | 6.7e-16 |
| case118 | 119 | 1.1e-15 | 1.3e-15 | 1.2e-15 |
| case_illinois200 | 201 | 6.4e-15 | 2.7e-14 | 6.5e-11 |
| case300 | 301 | 1.5e-14 | 3.5e-14 | 6.5e-14 |
| case1354pegase | 1,355 | 3.9e-14 | 1.2e-13 | 7.5e-14 |
| case1888rte | 1,889 | 1.4e-13 | 4.6e-13 | 3.1e-12 |

**Column 2 is the one that matters.** `plans/GPU_PLAN.md` §3 property 2 claims that stacking B
scenarios into one block-diagonal matrix and taking a single LU is equivalent to B independent
solves — which is what lets the AMD path work without a batched refactorization API, and is the
architectural load-bearing wall under Phases 3-5. It is now checked numerically rather than
asserted: agreement is at machine precision, and the per-scenario iteration counts match exactly,
on every case.

Scope limits, stated rather than discovered later: constant-power injections only (ZIP terms are
asserted absent, not handled); dense Jacobian, so B is auto-capped to bound the dense solve and
case9241pegase is out of reach; no Q-limit enforcement or island partitioning, matching
`PersistentSolver::solve`'s defaults.

### A bug the oracle found in its own scaffolding

The first run disagreed with `klu` by 1.1e-2 in |V| and 0.44 rad in angle. The cause was in the
export, not either solver: PyO3 maps `Vec<u8>` to Python `bytes` rather than a list of ints, so
`np.asarray(kinds)` produced a 0-d array, every bus mask collapsed to a single index, and the
oracle "converged" in 4 iterations on a one-unknown problem. `bus_spec` now returns `Vec<u32>`, and
the oracle validates the array's shape and value set on load instead of trusting it.

Worth recording because of what it implies about check 2. In that broken run, BDE-vs-independent
*passed* — both paths shared the same wrong indexing, so they agreed with each other perfectly
while both being wrong. A self-consistency check between two code paths cannot detect a fault in
what they share. That is exactly why check 1 (against a genuinely separate implementation) has to
pass before check 2 means anything.

## 5. Cross-validate CGMES import against pypowsybl

`cross_validate_cgmes_microgrid_be.py` checks gridoxide's CGMES import + solve against pypowsybl's own,
independent CGMES import + AC load flow, on the same ENTSO-E MicroGrid-BE-MAS conformance files
`tests/cgmes_microgrid_be_test.rs` already checks against that fixture's own published `SvVoltage` values.
That Rust test's doc comment already claimed pypowsybl "also deviates from this fixture's published SV
values by a comparable few percent" as a one-off manual finding — this script is what actually computes and
asserts it:

```bash
pip install pypowsybl
python3 scripts/bench/cross_validate_cgmes_microgrid_be.py
```

It runs `examples/cgmes_microgrid_be_dump.rs` (`cargo run --release --example cgmes_microgrid_be_dump
--features cgmes`) as a subprocess to get gridoxide's own per-`TopologicalNode` solved voltages (predates
`PowerFlowModel.from_cgmes` in `src/python.rs`, added since — see section 6 below, which uses the native
binding directly instead) — then zips the same BE-MAS + boundary files together for pypowsybl (its CGMES
importer needs one archive, not a directory) and solves with the same "BASIC" `LoadFlowParameters`
`bench_pypowsybl.py` already uses. See the script's own docstring for how pypowsybl's bus IDs (not
TopologicalNode mRIDs) get matched back to gridoxide's, and for how angles are made comparable despite the
two tools not natively sharing a reference bus (pypowsybl's OpenLoadFlow is pinned, via the `slackBusesIds`
provider parameter, to the same bus gridoxide uses as its own slack). Worst observed per-bus deviation is
0.22% voltage / 0.07° angle (re-checked after section 6's two CGMES importer fixes — the voltage figure is
unchanged, the angle figure was previously quoted as "~0.3°", which was the *tolerance* rather than anything
actually measured); `--tol`/`--angle-tol` default to 0.01 (1%) / 0.3°.

## 6. Benchmark against CGMES conformance test configurations

A third comparison, this time on real CGMES conformance test configurations rather than MATPOWER cases:
gridoxide (`KluNative` backend) against pypowsybl/powsybl-open-loadflow, both importing and solving the
*same* CGMES profile files directly — no PGM-JSON or MATPOWER conversion step on either side. pypowsybl
is the only one of this project's three vendored references (`references/`) with any native CGMES import at
all — confirmed directly: neither power-grid-model nor lightsim2grid has a single CGMES-related file
anywhere in their own trees.

```bash
maturin develop --release --features python,cgmes
python3 scripts/bench/bench_gridoxide_cgmes.py <fixture_name> <profile.xml>...
python3 scripts/bench/bench_pypowsybl_cgmes.py <fixture_name> <profile.xml>...
```

`bench_gridoxide_cgmes.py` uses `gridoxide.PowerFlowModel.from_cgmes` (added to `src/python.rs` specifically
for this benchmark) directly — no subprocess, unlike section 5's cross-validation script. `bench_pypowsybl_cgmes.py`
zips the given profile files into a temp archive first (pypowsybl's CGMES importer needs one file/archive,
not a list of paths — the same constraint section 5's script already worked around). Both use the same
"BASIC" `LoadFlowParameters`/flat-start conventions as every other pypowsybl comparison in this directory.

Both scripts also report each tool's own deviation from the fixture's published `SvVoltage` (parsed directly
from the SV profile by the shared `cgmes_sv.py` helper) — an accuracy check against the fixture's own
reference solution, independently per tool, not a tool-vs-tool comparison. On gridoxide's side this is exact:
`PowerFlowModel.bus_index_for_mrid` looks up a solved bus directly by its `TopologicalNode` mRID, the same
lookup `tests/cgmes_common::assert_matches_sv` uses on the Rust test side. On pypowsybl's side it's a
heuristic: pypowsybl's own bus IDs aren't TopologicalNode mRIDs, so `cgmes_sv.py`'s
`match_powsybl_buses_to_tn` reconstructs the mapping from the TP profile's own
`TopologicalNode.ConnectivityNodeContainer` references (pypowsybl's bus-view ID is
`"<container-mRID>_<index>"`), resolving any container holding more than one `TopologicalNode` via
nearest-magnitude matching against the *published* voltage (kept independent of gridoxide's own solve,
unlike section 5's `cross_validate_cgmes_microgrid_be.py`, which matches against gridoxide's solved value
instead since its own purpose is a tool-vs-tool cross-check, not an accuracy-vs-reference metric).

### Timing results

`solve()`/`run_ac()` warm-run mean (5 timed calls on one persistent model, `time.perf_counter()`), plus each
side's own cold (construct+solve) figure:

| fixture | gridoxide nodes | pypowsybl nodes | gridoxide mean (ms) | pypowsybl mean (ms) | gridoxide cold (ms) | pypowsybl cold (ms) |
|---|---|---|---|---|---|---|
| PowerFlow | 2 | 2 | 0.007 | 1.398 | 0.55 | 19.26 |
| MiniGrid | 15 | 11 | 0.019 | 1.306 (no solve)¹ | 3.86 | 65.00 (no solve)¹ |
| PST_PhaseTapChangerTable_Type3 | 2 | 2 | 0.006 | 1.386 | 0.60 | 22.96 |
| MicroGrid-BE-MAS | 13 | 7 | 0.023 | 0.872 | 1.52 | 35.86 |
| MicroGrid-Type2-HVDC-MAS | 6 | 4 | 0.010 | 0.711 (no solve)⁵ | 0.91 | 27.33 (no solve)⁵ |
| SmallGrid | 163 | 120 | 0.188 | 2.555 | 67.91 | 519.63 |
| Svedala | 191 | 104 | 0.188 | 2.511 | 65.71 | 543.74 |
| RealGrid | 6,252 | 5,806 | 18.258 | 136.959 | 1,449.74 | 5,890.94 |

gridoxide's node counts run higher than pypowsybl's own bus counts on every fixture — gridoxide's multi-island
support (`solver::PersistentSolver::solve`) keeps every disconnected switchyard stub/spare bus as its own
tiny island with a placeholder solution, while pypowsybl's `connected_component_mode=MAIN` (this benchmark's
own, deliberate choice, matching every other pypowsybl comparison in this directory) solves only the largest
connected component and drops the rest. gridoxide's `KluNative` backend is faster than pypowsybl on every
fixture where both tools actually solve one — which excludes `MiniGrid` and `MicroGrid-Type2-HVDC-MAS`, whose
pypowsybl cells are marked `(no solve)`: those measure how long pypowsybl takes to hit its iteration limit
or to decide there is nothing to calculate, not how long it takes to solve a load flow, so they are not a
speed comparison in either direction. On the rest, the ratio shrinks as fixtures get larger (from ~200x on
the trivial 2-bus `PowerFlow`/`PST`
cases, where both tools are sub-millisecond and fixed per-call overhead dominates any real comparison, down
to ~14x on `SmallGrid`/`Svedala` and ~7.5x on RealGrid: 18.3ms vs 137.0ms warm, 1.45s vs 5.89s cold) —
consistent with section 4's MATPOWER-case comparison, where every gridoxide backend already beat pypowsybl by
a wide margin at every scale. Model construction dominates pypowsybl's cold figure far more than gridoxide's:
pypowsybl's own CGMES importer (Java, general-purpose IIDM construction) costs tens to hundreds of
milliseconds even on tiny 2-bus fixtures, while gridoxide's cold figure barely differs from its own warm
solve time at small scale (`PowerFlow`: 0.55ms cold vs 0.007ms warm solve — decode+convert+Y-bus-build is the
whole gap).

These are noticeably faster than an earlier version of this table (RealGrid warm mean was 24.8ms) — not
random variance: `network::dc_angle_guess`, the DC-power-flow-style angle pre-pass removed as part of fixing
the `case3120sp` regression (section 4 above), ran one extra sparse linear solve on *every* call to
`linear_initial_guess`, i.e. every warm-repeated solve in this benchmark too, not just cold ones. Removing it
cut every fixture's own warm-run time here, most visibly on RealGrid (~26% faster) where that extra solve's
own ~6,000-bus sparse system was the most expensive to redo repeatedly.

This timing table predates the two CGMES importer fixes described under "Accuracy results" below and was not
re-measured after them, for the same reason as section 4's: neither fix changes matrix structure or iteration
count. Spot-checking afterwards gave RealGrid 17.61 ms warm / 1,425 ms cold and Svedala 0.172 ms warm /
70.2 ms cold against the table's 18.258/1,449.74 and 0.188/65.71, with node counts identical — run-to-run
noise. The solved *voltages* did change; see the accuracy table.

### Accuracy results

Deviation from each fixture's own published `SvVoltage` (median/p90/max relative voltage-magnitude error,
`n` = number of `TopologicalNode`s matched to a solved bus):

| fixture | gridoxide n | gridoxide median | gridoxide p90 | gridoxide max | pypowsybl n | pypowsybl median | pypowsybl p90 | pypowsybl max |
|---|---|---|---|---|---|---|---|---|
| PowerFlow | 2 | 0.0000% | 0.0001% | 0.0001% | 2 | 0.0001% | 0.0001% | 0.0001% |
| MiniGrid | 11 | 1.248% | 2.334% | 2.387% | — | NOT CONVERGED¹ | NOT CONVERGED¹ | NOT CONVERGED¹ |
| PST_PhaseTapChangerTable_Type3 | 2 | 0.0001% | 0.0001% | 0.0002% | 2 | 0.0002% | 0.0002% | 0.0002% |
| MicroGrid-BE-MAS | 7 | 0.527% | 2.573% | 2.612% | 7 | 0.477% | 2.612% | 2.612% |
| MicroGrid-Type2-HVDC-MAS | 2² | 3.315%² | 3.315%² | 3.315%² | — | NO CALCULATION⁵ | NO CALCULATION⁵ | NO CALCULATION⁵ |
| SmallGrid | 127 | 0.002% | 0.217% | 0.447% | 7³ | 0.002%³ | 2.833%³ | 2.833%³ |
| Svedala | 108 | 0.213% | 1.000% | 2.865% | 96 | 0.285% | 1.627% | 4.071% |
| RealGrid | 6,051 | 0.018% | 0.424% | 93.475%⁴ | 5,806 | 0.099% | 0.822% | 82.068%⁴ |

¹ pypowsybl fails to converge on MiniGrid under this benchmark's flat-start "BASIC" parameters
(`status=Reached Newton-Raphson max iterations limit`, 16 iterations). This cell used to read
`n=2 / 0.000% / 0.000% / 0.000%` with the non-convergence relegated to this footnote — which was actively
misleading, because 0.000% is not a neutral placeholder, it is the *best possible score in the column*, and
column-scanning readers (or anything grepping these tables) would read pypowsybl as having solved MiniGrid
perfectly. The figures came from the two boundary buses whose voltage never has to move from its fixed
initial value: comparing the input SV profile against itself. `bench_pypowsybl_cgmes.py` now prints
`NOT CONVERGED` in the deviation line itself, on stdout — previously it warned only on stderr, which the
`2>/dev/null` used to produce these tables discarded, which is exactly how the 0.000% got recorded here.
² gridoxide's own `n=2` here reflects that this fixture's published `SvVoltage` set includes some
`TopologicalNode`s (the DC-side/converter switchyard detail resolved away by `cgmes_resolve_dc_converters`
into ordinary AC bus injections — see `src/dc.rs`) with no corresponding AC bus in gridoxide's own solved
output at all, not a matching failure. This is the one row in this table **not** re-measured after the two
CGMES importer fixes described below: reproducing its original `n=2` needs the exact profile-file set the
original run used, and this fixture only loads at all with its `-BD-MAS` boundary files included, which
yields `n=4` (two of them de-energized boundary nodes reported at 100%). The fixture has no out-of-service
`SynchronousMachine` at all, so the in-service fix cannot have moved it; treat these four figures as
provisional rather than as a comparable before/after.
³ pypowsybl's own bus IDs aren't TopologicalNode mRIDs (see above), and the nearest-magnitude
container-based reconstruction this benchmark uses to recover that mapping breaks down badly on SmallGrid
specifically: only 7 of 118 candidate containers matched at all (checked directly — SmallGrid's own TP
profile groups many more `TopologicalNode`s per `ConnectivityNodeContainer`, up to 9 in one container, than
pypowsybl's own resulting bus-view groups them into, so the container-mRID half of the `"<container>_<index>"`
join key mismatches for most buses). This is a real limitation of this benchmark's own pypowsybl-side
matching heuristic on a fixture with this much switchgear detail (838 `Disconnector`s + 427 `Breaker`s),
not a pypowsybl solve-accuracy issue — gridoxide's own `n=127` figure has no equivalent uncertainty, since
`bus_index_for_mrid` looks up the mRID directly rather than reconstructing it.
⁴ RealGrid's `max` is a defect in the *fixture*, and it is now pinned down rather than inferred. Both tools
showing a similarly large outlier (93% / 82%) alongside an otherwise tight distribution was previously read
as corroboration of "a real data quirk in that specific `TopologicalNode`" — the right conclusion, but
reached by tool agreement, which section 4 shows is not by itself reliable evidence. It is now checkable
without any tool:

```bash
python3 scripts/bench/check_cgmes_sv_consistency.py RealGrid
```

`check_cgmes_sv_consistency.py` compares each two-winding transformer's published voltage ratio against the
ratio its own `ratedU` values declare, adjusted by the actual `RatioTapChanger.step` in the SSH profile. On
RealGrid, 83 of 1,461 such transformers miss their own nameplate ratio by more than 5%, and the four worst
miss it by **70–87%** — the LV sides of four 63/20 kV units, published at 10.76–11.88 kV where every piece of
equipment data in the fixture (the `TopologicalNode`'s own `BaseVoltage`, the transformer end's `ratedU`, and
all five `SynchronousMachine.ratedU` values on each node) says 20 kV. No tap position can produce that: those
are exactly the four buses at the top of gridoxide's own error list, and the SV profile simply disagrees with
the EQ profile about them. The next one down, `_fad2025e` at 14.2%, is a 225/150 kV pair whose two parallel
transformers both sit at `step=0` (neutral tap, ratio exactly 1.5) against a published ratio of 1.7275.

Excluding the 140 buses that sit on a >5%-inconsistent transformer, RealGrid's own figures become
`n=5911 median=0.017% p90=0.423% max=4.318%` — i.e. the entire 93% headline is fixture data, and the real
worst-case is 4.3%. The median and p90 barely move, confirming the rest of the distribution was never
affected by it.
⁵ pypowsybl doesn't actually solve MicroGrid-Type2-HVDC-MAS at all: `run_ac`'s own result reports
`status=NO_CALCULATION, iteration_count=0, status_text="Network has no generator with voltage control
enabled"` — checked directly, and now reported by `bench_pypowsybl_cgmes.py` on stdout in the deviation line
itself (it used to warn only on stderr, which the `2>/dev/null` used to produce these tables discarded).
`iteration_count=0` is the cleanest statement of the problem available: not a solve that went wrong, but no
solve at all. CGMES's `VsConverter`/`CsConverter` HVDC converters apparently aren't recognized by
pypowsybl's importer as an AC voltage-controlling source the way a `SynchronousMachine` is, so it declines to
run any Newton-Raphson iteration and its `get_buses()` simply echoes back whatever voltage state its CGMES
importer initialized directly from the input SV profile — both the 0.000% "deviation" (comparing the input
file against itself) and the timing figures (time to conclude there's nothing to calculate, not time to
actually solve a load flow) are artifacts of this, not a real result. gridoxide's own DC-aware solve
(`cgmes_resolve_dc_converters`/`src/dc.rs`) is the only one of the two that actually solves this fixture's
power flow at all.

Where both tools have a genuinely comparable `n` (PowerFlow, PST, MicroGrid-BE, Svedala, RealGrid), their
accuracy against the published reference is close — consistent with section 5's cross-validation finding that
gridoxide and pypowsybl solve real CGMES fixtures to comparable accuracy, not with one tool being
systematically more correct than the other.

**Two CGMES importer bugs found by working outward from the worst-deviating buses.** Both were found the
same way as section 4's: take the single worst bus, ask what the fixture's own data says should happen
there, and keep going until the number is explained rather than attributed.

1. **`build_two_winding` applied only one end's structural voltage ratio.** A `PowerTransformerEnd`'s
   nameplate `ratedU` need not equal the system `nominalVoltage` of the bus it sits on, and where they
   differ, that difference *is* an off-nominal ideal-transformer ratio, independent of any tap changer. The
   importer folded in end 1's `ratedU / bus1.u_rated` but silently dropped end 2's. Invisible on every
   hand-authored fixture in this repo (all of which have `ratedU == nominalVoltage` throughout), but
   pervasive on real data: **666 of RealGrid's 1,509** two-winding transformers and 4 of Svedala's 53, by up
   to 7.2%. RealGrid's then-worst non-data-defect bus was the LV side of a 63 kV / 42.0168 kV unit whose bus
   base is 45 kV — a dropped 45/42.0168 = 1.0709 ratio showing up as a 7.6% solved-voltage error. Fixing it
   cut RealGrid's median deviation **5x, from 0.088% to 0.018%**.
2. **An out-of-service `SynchronousMachine` still regulated voltage.** The machine loop gated on terminal
   connectivity and `RegulatingCondEq.controlEnabled`, but never on `Equipment.inService` — even though
   `equipment_in_service` was already applied to lines, switches and shunts elsewhere in the same file. All
   six of Svedala's `inService=false` machines also carry `controlEnabled=true`, and one of them
   (`_f4cde1f4`, `p=q=0`) regulates a *remote* terminal, so it pinned an in-service bus to its 21 kV setpoint
   against the fixture's published 20.134 kV — Svedala's single worst bus. The other five re-typed their own
   already-de-energized buses back to PV, which is why `tests/cgmes_svedala_test.rs`'s de-energized-bus count
   moved from 78 to the correct 83 (that fixture's own `TopologicalIsland` lists 108 of 191 buses).

Net effect on the two fixtures with enough buses to be statistically meaningful: RealGrid's median fell from
0.088% to 0.018% and its p90 from 0.525% to 0.424%; Svedala's p90 halved (2.121% → 1.000%) and its max fell
by a third (4.351% → 2.865%). Svedala's *median* moved the other way, 0.081% → 0.213% — worth stating plainly
rather than burying: on a 108-bus fixture that is a shift of about a thousandth of a per-unit on the middle
bus, both changes are individually justified by the fixture's own equipment data, and both moved the tail
substantially in the right direction, but the median did not improve. Timings are unaffected (RealGrid warm
mean 18.26 → 17.61 ms, node counts identical), since neither fix changes matrix structure.
