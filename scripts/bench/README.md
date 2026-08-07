# Benchmark scripts

Timing and accuracy harnesses comparing gridoxide against other power-system solvers:

| § | Comparison | Against |
|---|---|---|
| 1–3 | Synthetic radial MV/LV grids | power-grid-model |
| 4 | 12 MATPOWER test cases | PGM, lightsim2grid, pypowsybl, pandapower, VeraGrid |
| 4b | Batched power flow (multi-core) | itself, thread scaling |
| 4c–4d | Block-diagonal embedding | JAX oracle, independent solves |
| 5–6 | CGMES conformance configurations | pypowsybl |
| 7 | State estimation | power-grid-model |

## Python bindings

Every script here runs purely in Python via gridoxide's native bindings (`src/python.rs`, built with
[maturin](https://www.maturin.rs/)) — no subprocess into a Rust binary, no stdout parsing.

```bash
pip install maturin
maturin develop --release --features python         # scalar + block backends
maturin develop --release --features python,klu     # + klu (needs a C compiler, libclang)
maturin develop --release --features python,pardiso # + pardiso (needs MKLROOT set)
```

`maturin develop` needs an active virtualenv (or `VIRTUAL_ENV=/path/to/venv`). The `python` feature is
`#[cfg]`-gated, so plain `cargo build`/`cargo test` is unaffected — but never combine `--features python`
with `cargo build --examples`/`cargo test` in one invocation: PyO3's `extension-module` makes standalone
binaries fail to link. That's an expected PyO3 constraint, not a gridoxide issue.

```python
import gridoxide

model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="klu")
model.solve()
print(model.voltage_mag(), model.voltage_ang())
```

`PowerFlowModel` wraps `solver::PersistentSolver`, so repeated `.solve()` calls reuse the cached symbolic
factorization — matching how every other tool here is driven (construct once, time repeated solve calls with
`time.perf_counter()`). All comparisons below are warm-vs-warm.

## 1. Generate a synthetic benchmark grid

```bash
python3 scripts/bench/generate_grid.py grid_small.json  --target-nodes 200   # -> 192 nodes
python3 scripts/bench/generate_grid.py grid_medium.json --target-nodes 1500  # -> 1,003 nodes
python3 scripts/bench/generate_grid.py grid_large.json  --target-nodes 2200  # -> 2,605 nodes
```

Ports PGM's own C++ benchmark generator (`tests/benchmark_cpp/fictional_grid_generator.hpp`): a radial MV
feeder with stochastically-attached LV sub-grids. Not a bit-for-bit RNG replica, but both tools read the
*same* generated JSON, so the comparison is apples-to-apples. `--target-nodes` is approximate — LV attachment
is a Bernoulli process.

## 2. Benchmark gridoxide

```bash
cargo build --release --example bench_network
./target/release/examples/bench_network grid.json [repeat-count] [backend] [mode]
python3 scripts/bench/bench_gridoxide_native.py grid.json [backend]   # pure-Python, always warm
```

- `repeat-count` (default 1) — re-runs the solve, for stable averages and for `perf record` sampling.
- `backend` — `scalar` (default), `block`, `klu`, `klu_native`, `pardiso`. See `docs/src/solvers/backends.md`.
- `mode` — `cold` (default, fresh symbolic factorization per repeat) or `warm` (one `PersistentSolver` reused).
  Use `warm` for any comparison against another tool.

## 3. Benchmark power-grid-model

```bash
python3 -m venv .venv-pgm
.venv-pgm/bin/pip install power-grid-model      # prebuilt wheel, no C++ build
.venv-pgm/bin/python3 scripts/bench/bench_pgm.py grid.json
```

Compare gridoxide's `total (guess + NR)` against PGM's `min`/`mean`. Voltage outputs (`voltage_mag min/max`
vs `u_pu min/max`) should match closely — a large mismatch means the comparison itself is broken.

### Results — synthetic radial grids

`newton_raphson`-only ms/run (200 warm repeats) vs PGM's `mean` (5 warm runs):

| Nodes | Scalar | Block | Klu | KluNative | Pardiso | PGM |
|---|---|---|---|---|---|---|
| 192 | 1.28 | 0.52 | 0.44 | 0.45 | 2.06 | 0.42 |
| 1,003 | 7.86 | 2.80 | 2.52 | 2.69 | 5.06 | 0.93 |
| 2,605 | 20.97 | 6.81 | 6.04 | 6.56 | 11.73 | 2.49 |

**PGM is faster than every gridoxide backend on this topology, warm-vs-warm — a real, standing gap.**
`KluNative` tracks `Klu` closely (1.02–1.09x). `Pardiso` is 2–4.7x slower than `Klu`: its default nonsymmetric
matching/scaling preprocessing carries a roughly size-independent fixed cost that dominates at these sizes.
Contrast with §4, where gridoxide's `Klu` is often *faster* than lightsim2grid's KLU-backed C++ solver on
transmission topologies — the result depends heavily on grid topology, not implementation language.

## 4. Benchmark against MATPOWER test cases

gridoxide against five independent Newton-Raphson implementations —
[PGM](https://github.com/PowerGridModel/power-grid-model),
[lightsim2grid](https://github.com/m-mirz/lightsim2grid) (C++/KLU),
[powsybl-open-loadflow](https://github.com/powsybl/powsybl-open-loadflow) via
[pypowsybl](https://pypi.org/project/pypowsybl/), pandapower, and
[VeraGrid](https://github.com/SanPen/VeraGrid) (numba-JIT) — on the same 12 cases lightsim2grid's own
benchmark uses: `case14`, `case118`, `case_illinois200`, `case300`, `case1354pegase`, `case1888rte`,
`case2848rte`, `case2869pegase`, `case3120sp`, `case6495rte`, `case6515rte`, `case9241pegase`.

```bash
git submodule update --init tests/data/benchmark-grids
python3 -m venv .venv-case-suite
.venv-case-suite/bin/pip install maturin numpy scipy power-grid-model pandapower lightsim2grid pypowsybl \
    VeraGridEngine
VIRTUAL_ENV=.venv-case-suite .venv-case-suite/bin/maturin develop --release --features python,klu
.venv-case-suite/bin/python3 scripts/bench/run_case_suite.py --python .venv-case-suite/bin/python3
```

Loops all 12 cases, converting each MATPOWER `.m` to PGM JSON on first use (cached in
`scripts/bench/.case-cache/`, gitignored). A case that fails to convert or diverges gets an explicit
`FAILED (...)` cell with the tool's real exception. `--repeat` sets timed solves per case (default 10),
`--cache-dir`/`--out` set output locations.

Each tool runs at its closest-comparable settings: lightsim2grid with `SolverType.KLU` (its own benchmark
default), pypowsybl with the same "basic" `LoadFlowParameters` powsybl's benchmark repo uses, pandapower with
its defaults, VeraGrid with `SolverType.NR` and every automatic control (tap changers, remote voltage
regulation, alternate-solver fallback) off — see `bench_veragrid.py`'s docstring.

**Data paths.** gridoxide and PGM read MATPOWER `.m` files converted by `gridoxide.matpower`
(`python/gridoxide/matpower.py`; `matpower_to_pgm.py` is a thin CLI wrapper), *not* pandapower's importer.
Three cases (`case1888rte`, `case6495rte`, `case6515rte`) will not converge through pandapower's importer at
all: it assigns each bus a physical `baseKV`, which requires keeping every impedance/tap conversion referenced
to a consistent side. powsybl's own importer sidesteps this by assigning `nominal_v = 1.0` to every bus and
encoding off-nominal ratio as `rated_u1 = ratio, rated_u2 = 1.0` — MATPOWER's formulation needs a *consistent*
voltage reference, not a physical one. `matpower_to_pgm.py` does the same. lightsim2grid and pandapower load
via `pandapower.networks.<case>()` (they need the net object); pypowsybl and VeraGrid read the same `.m` file
through their own importers. See `gridoxide.matpower`'s module docstring for the full derivation.

Generators are modelled as genuine PV buses in every tool. gridoxide parses PGM's `voltage_regulator`
component the same way PGM's own `newton_raphson_pf_solver.hpp::set_u_ref_and_bus_types` does
(`src/pgm.rs::pgm_to_buses_and_branches`).

Individual tools also run standalone: `matpower_to_pgm.py`, `bench_pgm.py`, `bench_lightsim2grid.py`,
`bench_pypowsybl.py`, `bench_pandapower.py`, `bench_veragrid.py`, `convert_pandapower_case.py`.

### Timing results

Warm-run mean, 5 timed calls on one persistent model per tool, ms:

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

¹ PGM `IterationDiverge` (20-iteration default). ² PGM `SparseMatrixError`, raised during construction.
³ lightsim2grid `ac_pf` divergence (`V.shape[0] == 0`). ⁴ pypowsybl `MAX_ITERATION_REACHED`/`Unrealistic
state` under the same "basic" parameters powsybl's own benchmark repo uses — *without* the phase-shift-zeroing
workaround its `MatpowerUtil.java` applies to these same three cases (applying it makes powsybl converge on
all three in ~4 iterations). ⁵ needs `--features python,pardiso` and `MKLROOT`. ⁶ VeraGrid
`results.converged == 0` with `SolverType.NR` and automatic controls disabled.

These timings predate the correctness fixes below and were not fully re-measured; spot-checks agreed to
run-to-run noise (`klu_native` case14/case2869pegase/case9241pegase: 0.027/6.914/35.347 vs 0.027/6.731/34.623).
The fixes add shunts to existing Y-bus diagonals, changing neither sparsity nor iteration counts. Solved
*voltages* did change — see the accuracy section.

### Reading the timing table

**gridoxide converging on the three RTE cases is not a robustness result.** Its MATPOWER→PGM conversion rounds
`transformer.clock` to the nearest 60°, and every phase shift in these cases is under 30°, so all are zeroed —
4 branches in `case1888rte`, 17 in `case6495rte`, 16 in `case6515rte`. That is precisely the workaround
powsybl's benchmark applies (footnote 4). gridoxide is solving the same easier problem, having applied the
workaround implicitly during conversion. `check_matpower_residual.py` quantifies the cost (a five-figure MVA
residual on `case1888rte`). pandapower converges via its own native path.

**Backends.** `block` beats `scalar` 2–3x on the larger cases (case9241pegase: 33.84 vs 110.81 ms) despite
`scalar` using `faer` and `block`'s LU having no partial pivoting — factoring at 2×2-block granularity halves
the elimination steps. `klu` is ~10–15% ahead of `block`. `klu_native` (`src/klu_native/`) converges to the
same voltages as every other backend on all 12 cases and lands within 1.0–1.3x of `klu`, after `perf` traced
an earlier 1.9–2x gap to allocator churn: `kernel::refactor_block` allocated two `Vec`s per column per Newton
iteration where real KLU's `klu_refactor.c` allocates nothing. Rewriting it to mutate in place against a
reused `RefactorScratch` dropped glibc allocator functions from ~32% of `newton_raphson` self-time to
effectively absent. `pardiso` matches the other backends' voltages exactly and sits 1.2–3.2x behind `klu`,
without the clean size trend the synthetic grids show — these cases vary more in topology and conditioning.

**vs lightsim2grid.** Roughly competitive: `klu` faster on most cases (case2848rte 5.26 vs 7.70 ms),
lightsim2grid faster on a couple (case9241pegase 29.26 vs 27.46 ms). Earlier `cold` numbers made gridoxide
look 1.3–1.7x slower, which `perf` traced to redone symbolic factorization — not a solver-speed gap. On
case9241pegase, warm cut per-solve `klu` time ~45%, with COLAMD/AMD/BTF ordering about a third of a cold solve.

**Python frameworks.** pypowsybl, pandapower and VeraGrid are markedly slower than every C/Rust-backed solver
here. VeraGrid's numbers are already post-JIT-warmup; its *first* call on a large case costs seconds.

Two fixes came out of building this table:

- `block` used to be `N/A` — it panicked on any `PV` bus, since its 2×2-per-bus indexing assumed `PQ`. Fixed
  by giving a `PV` block a dummy `ΔVmag = 0` row (`solver::build_jacobian_blocks`), equivalent to the scalar
  backend's dimension reduction. This surfaced a latent bug in `BlockLu::refactor`: it scattered
  `adj.row(perm[j])` where it needed `adj.col(perm[j])` — silently wrong on any value-asymmetric matrix (the
  real Jacobian is one; every unit test happened to use symmetric off-diagonal blocks). See
  `solve_asymmetric_off_diagonal_matches_dense_reference`.
- `case3120sp` diverged catastrophically on *every* backend (`voltage_mag` ≈ -1548/930 p.u.), contradicting
  this table. Bisected to `network::dc_angle_guess`, a DC-power-flow angle pre-pass added for CGMES FullGrid
  but wired unconditionally into `linear_initial_guess`. Its own justification never panned out (FullGrid
  still doesn't converge either way) and the full `cargo test --features cgmes` suite passes identically
  without it, so it was removed outright.

PGM's numbers needed a workaround to obtain at all: once the converter began writing real `q_min`/`q_max` onto
`voltage_regulator`, all 12 inputs tripped PGM's `ExperimentalFeature` error through the public
`calculate_power_flow` (version 1.13.120 never wired `experimental_features` through the public wrapper).
`bench_pgm.py` calls the private `_calculate_power_flow` with `experimental_features="enabled"`, confirmed to
reproduce the same converged voltages the public API gave before Q-limits existed.

### Accuracy results

A MATPOWER `.m` file is power-flow *input*, not a solved snapshot — but it fully specifies the *equations*,
and any correct solution must satisfy them. This is the check to trust when it disagrees with the cross-tool
table:

```bash
python3 scripts/bench/check_matpower_residual.py --all
```

Rebuilds Ybus from the `.m` file (following `makeYbus.m`) and evaluates `dS = V .* conj(Ybus @ V) -
S_specified`, checking P at every PV/PQ bus and Q at every PQ bus. Absolute, no second tool involved.

```bash
python3 scripts/bench/accuracy_case_suite.py --out accuracy.md
```

Reports every other tool's per-bus voltage-magnitude deviation from **gridoxide's own** solution, matched by
MATPOWER bus number (each tool's ID convention confirmed empirically — see the script's docstring). Needs the
same environment as `run_case_suite.py` but runs all tools in one interpreter. **Using gridoxide as the anchor
is presentational and treacherous**: its five backends agreeing bit-identically says they share a front end,
not that the front end is right, and deviations *from* an anchor cannot distinguish "every other tool is off"
from "the anchor is off." Read alongside the residual check.

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

¹ ² Same PGM failures as the timing table.

³ pandapower/lightsim2grid's `n` on the two PEGASE reductions is far below the real bus count. Not a naming
collision and not a constant offset — pandapower's bundled `case1354pegase()`/`case2869pegase()` derive their
per-bus numbering from the 9,241-bus network differently than the vendored `.m` files do. Unresolved, and
specific to those two cases' bundled data; every other case matches at full bus count.

⁴ pandapower/lightsim2grid `max` on the largest cases (22–98%) is far outside every other tool's (≤2.9%).
Both load `pandapower.networks.<case>()` rather than the vendored `.m`, and that data provably differs
(footnote 3), so part of this is a data difference. `check_matpower_residual.py` deliberately excludes them.

⁵ These remaining 2.3–2.9% maxima are endpoints of branches whose phase shift the conversion rounds to zero,
because PGM's `transformer.clock` can't represent a continuous angle (6 such branches in `case1354pegase` up
to 66 in `case9241pegase`). `check_matpower_residual.py --zero-phase-shifts` confirms it directly: gridoxide's
residual is nonzero on exactly these 7 cases and drops to exactly 0.0000 MVA against the phase-zeroed network.
pypowsybl and VeraGrid honour the real angle, so this is gridoxide's conversion, not their solve.

**The cross-tool table missed four real bugs.** `case14` used to read `median=0.313% p90=1.788% max=2.154%`
for *every* one of the five tools to three decimals. Five independent solvers reproducing each other's
deviation that precisely is not five tools being slightly wrong — it is the anchor being wrong. An earlier
version of this section instead concluded the gap was two valid solutions from independent Newton-Raphson
implementations. `check_matpower_residual.py` settles it without reference to any tool, and found:

1. **gridoxide ignored every PGM `shunt` on the PGM-JSON path.** `PowerFlowModel.from_pgm_json` and
   `examples/bench_network.rs` built Y-bus from lines and transformers only, never calling
   `pgm::pgm_shunts_1ph` + `network::stamp_shunts` — which `from_cgmes` beside it does call, so the helpers
   were well covered while the real entry points skipped them. Discarded 1 shunt in `case14` (19 MVAr) up to
   7,327 totalling ~89.5 GVAr in `case9241pegase`.
2. **The converter dropped branch charging on the transformer path.** A branch with off-nominal ratio *and*
   nonzero `b` became a PGM `transformer`, which has no charging field. `makeYbus.m` folds it into
   `Ytt = ys + j·b/2`, `Yff = Ytt/|tap|²`, which decomposes exactly into two `shunt` components — now emitted,
   driving `case300`'s residual from 3.2 MVAr to 0.
3. **The converter flipped the sign of every negative reactance on the transformer path.** `uk` used
   `np.hypot(r, x)`, which is unsigned, so series-capacitive branches (10 in `case3120sp`, ~40 per RTE case)
   came back inductive. `network::transformer_admittances_ex` already handles signed `uk` correctly — 537 MVA
   of residual on `case3120sp`.
4. **The converter dropped generator reactive output at PQ-typed buses.** `sym_gen.q_specified` was hardcoded
   to `0.0`. Correct at a PV bus, wrong where MATPOWER types the bus PQ and `Qg` is a fixed injection
   (`case1888rte` bus 1005 at -19 MVAr, `case2848rte` 2534 and 642). Residual there was exactly `-Qg`.

With all four fixed, gridoxide satisfies the as-published equations to 0.0000 MVA on 5 of 12 cases, and on the
other 7 once the reference's phase shifts are zeroed the same way the conversion zeroes them (footnote 5).
Cross-tool medians fell ~2 orders of magnitude (case9241pegase pypowsybl/VeraGrid: `median 0.628% p90 4.008%
max 20.838%` → `median 0.0000% p90 0.0008% max 2.3527%`), and PGM now agrees to 0.0000% where it converges.

`case_illinois200` is a cautionary note: it converged in both tools to visibly different voltages, with
VeraGrid landing much closer to PGM. That was read as "this case has multiple solution basins." The real cause
was bug 1 — gridoxide discarding four bus shunts (190 MVAr). Two tools agreeing against a third is evidence
about the third.

A separate pypowsybl-side matching bug in this benchmark: pypowsybl bus IDs look like `"VL-<n>_<k>"`, but
`<n>` is *not* the MATPOWER bus number for any bus that is only ever a transformer's secondary side —
pypowsybl names a secondary voltage level after its *primary* side's bus number with an incrementing suffix
(on case14, `"VL-4_1"` is really bus 7). Fixed by reading each `Line`/`TwoWindingsTransformer` id instead
(`"LINE-<from>-<to>"`, with `"#<k>"` for parallel branches) — see `pypowsybl_bus_to_matpower_id`. Confirmed by
VeraGrid, an entirely independent importer, matching pypowsybl's post-fix deviations to 4 decimals on every
shared case.

## 4b. Batched power flow — the multi-core CPU baseline

`bench_batch.py` measures `batch::BatchSolver`: one topology, many injection scenarios across cores. This is
the time-series/QSTS/Monte-Carlo shape, and exists to be the baseline any future GPU work must beat —
`plans/GPU_PLAN.md` §6 is explicit that beating a *single-threaded* CPU solver is not a result.

```bash
maturin develop --release --features python,klu
python3 scripts/bench/bench_batch.py .case-cache/case9241pegase.json klu 256
```

Scenarios are ±20% uniform load scalings from a fixed seed. The script asserts voltages agree across every
thread count to <1e-12, so a scaling number is never reported without its correctness check.

**AMD Ryzen 7 250 (8 physical cores / 16 SMT threads, 16 MiB shared L3), 256 scenarios, `klu`.** Steady state
(third consecutive run, stable to ~2%):

| threads | case9241pegase ms/solve | speedup | case1354pegase ms/solve | speedup |
|---|---|---|---|---|
| 1 | 28.04 | 1.00x | 2.384 | 1.00x |
| 2 | 15.75 | 1.78x | 1.369 | 1.74x |
| 4 | 10.33 | 2.72x | 0.914 | 2.61x |
| 8 | 7.91 | **3.55x** | 0.680 | **3.51x** |
| 16 | 7.71 | 3.64x (SMT) | 0.721 | 3.30x (SMT, *slower*) |

**Read absolute numbers on this machine with suspicion.** It is a thermally constrained laptop APU; a cold
first run reads far better than anything reproducible (the very first 8-thread case9241pegase measurement was
3.65x and never recurred). Every table here is steady-state, and every A/B below was taken *interleaved* —
old, new, old, new — because two numbers captured minutes apart measure the heatsink, not the code.
**Re-measure on a desktop or server part before quoting any of this.**

### Precomputed Jacobian offsets

`jacobian::JacobianPattern` derives the sparsity pattern and per-nonzero recipe once per topology, then refills
one reused `Vec<f64>` per iteration instead of rebuilding a `Vec<(usize, usize, f64)>`. Every `LinearSolver`
backend already discarded the `(row, col)` half after construction, so `factor_and_solve_values` hands over
just the values. Interleaved A/B, two rounds, min-of-3 each, ms/solve:

| | case9241 t1 | case9241 t8 | case1354 t1 | case1354 t8 |
|---|---|---|---|---|
| before | 34.33 / 33.46 | 11.47 / 11.43 | 2.560 / 2.576 | 0.717 / 0.741 |
| after | 29.95 / 30.09 | 10.63 / 10.56 | 2.395 / 2.365 | 0.690 / 0.693 |
| **gain** | **11.4%** | **7.5%** | **7.3%** | **5.1%** |

`plans/GPU_PLAN.md` §1 measures assembly at ~36% of iteration time, so single-threaded on the large case
roughly a third of that stage was allocation and index rebuilding. The gain is *smaller* at 8 threads: once
all cores run, the solve is memory-stalled in the LU, so assembly is a smaller share. It is also smaller on
the small case, whose triplet array (~20k nnz, ~0.5 MB) fits in cache; case9241pegase's is ~150k nnz, ~3.6 MB,
rebuilt every iteration. Values are bit-for-bit identical — `src/jacobian.rs` compares `f64::to_bits`.

### Why sub-linear scaling is not a solver defect

Both cases land at ~3.5x on 8 physical cores (44% efficiency):

| Factor | Effect | Cumulative ceiling |
|---|---|---|
| 8 physical cores (16 logical is SMT2) | 8x | 8.0x |
| All-core clock throttle (2,977 → 2,236 MHz, from `/proc/cpuinfo` under load) | x0.75 | 6.0x |
| Shared-L3 / memory-bandwidth contention | remainder | **~3.5x observed** |

The last row was isolated by running **8 single-threaded processes** instead of 8 threads: separate address
spaces share no allocator arenas, locks or false sharing, so in-process contention would show up as better
process scaling. It did not — 3.96x processes vs 3.91x threads, back to back on an equally warm machine. The
ceiling is hardware.

An earlier draft claimed scaling degrades with grid size (63% on case1354pegase vs 46% on case9241pegase) and
blamed L3 overflow. That was an artifact of comparing a cold-machine run against a warm one; under equal
thermal conditions the two scale within 0.05x. The working-set effect is real but shows up in the Jacobian
assembly gain (11.4% vs 7.3%), not in thread scaling.

**Implication for the GPU plan.** The CPU batch path is memory-bandwidth bound, not compute bound — an
argument *for* the GPU direction, since bandwidth is what datacenter GPUs have in bulk. The number to beat
here is ~127 solves/s steady-state on case9241pegase, but a publishable claim needs a *server* CPU baseline.

## 4c. JAX oracle — validating the block-diagonal embedding

`jax_oracle.py` is an independent reimplementation of batched AC power flow in JAX (f64, CPU, dense), written
to answer questions the Rust solver cannot answer about itself. It is **not** a performance prototype — it is
deliberately the slowest power flow in this repo and must never be quoted as a speed number.

```bash
python3 -m venv .venv-jax
.venv-jax/bin/pip install jax numpy maturin
VIRTUAL_ENV=$PWD/.venv-jax .venv-jax/bin/maturin develop --release --features python,klu
.venv-jax/bin/python scripts/bench/jax_oracle.py .case-cache/case118.json 8
```

It consumes gridoxide's *own* Y-bus and bus arrays (`ybus_triplets`, `bus_spec`, `initial_guess`,
`zip_term_counts`) rather than re-deriving the model. That is the point: if the oracle built its own model, a
disagreement could equally be a converter difference as a solver bug. Max |dVm| against `klu`:

| case | buses | 1. oracle vs klu | 2. **BDE vs independent** | 3. oracle vs BatchSolver |
|---|---|---|---|---|
| case14 | 15 | 4.4e-16 | 4.4e-16 | 6.7e-16 |
| case118 | 119 | 1.1e-15 | 1.3e-15 | 1.2e-15 |
| case_illinois200 | 201 | 6.4e-15 | 2.7e-14 | 6.5e-11 |
| case300 | 301 | 1.5e-14 | 3.5e-14 | 6.5e-14 |
| case1354pegase | 1,355 | 3.9e-14 | 1.2e-13 | 7.5e-14 |
| case1888rte | 1,889 | 1.4e-13 | 4.6e-13 | 3.1e-12 |

**Column 2 is the one that matters.** `plans/GPU_PLAN.md` §3 property 2 claims stacking B scenarios into one
block-diagonal matrix and taking a single LU is equivalent to B independent solves — what lets the AMD path
work without a batched refactorization API, and the load-bearing wall under Phases 3–5. Now checked
numerically: machine-precision agreement and exactly matching per-scenario iteration counts on every case.

Scope limits: constant-power injections only (ZIP terms asserted absent), dense Jacobian so B is auto-capped
and case9241pegase is out of reach, no Q-limit enforcement or island partitioning.

**A bug the oracle found in its own scaffolding.** The first run disagreed with `klu` by 1.1e-2 in |V|. PyO3
maps `Vec<u8>` to Python `bytes`, so `np.asarray(kinds)` produced a 0-d array, every bus mask collapsed to one
index, and the oracle "converged" in 4 iterations on a one-unknown problem. `bus_spec` now returns `Vec<u32>`
and the oracle validates array shape on load. Worth recording for what it implies about check 2: in that
broken run, BDE-vs-independent *passed* — both paths shared the same wrong indexing. A self-consistency check
between two code paths cannot detect a fault in what they share, which is why check 1 must pass first.

## 4d. Block-diagonal embedding on real sparse code

`src/bde.rs` stacks B scenarios' Jacobians into one block-diagonal sparse matrix and takes a single
factorization per Newton iteration. §4c established equivalence with *dense* linear algebra; this extends it
to the sparse backends the GPU path mirrors, where the claim is far less obvious — KLU applies BTF ordering,
AMD permutation and partial pivoting to the stacked matrix, none of which know about the block structure.

```bash
cargo run --release --example bde_check -- scripts/bench/.case-cache/case1354pegase.json 16
```

16 scenarios, `klu_native`, versus independent per-scenario solves:

| case | buses | stacked unknowns | stacked nnz | max \|dVm\| | iter mismatches | independent (1 thread) | block-diagonal |
|---|---|---|---|---|---|---|---|
| case118 | 119 | 2,928 | 17,456 | **0** | 0/16 | 2.6 ms | 7.1 ms |
| case300 | 301 | 8,512 | 59,968 | **0** | 0/16 | 10.2 ms | 23.5 ms |
| case1354pegase | 1,355 | 39,184 | 253,552 | **0** | 0/16 | 47.4 ms | 130.6 ms |
| case2869pegase | 2,870 | 83,664 | 586,160 | **0** | 0/16 | 113.6 ms | 310.9 ms |

**Bit-exact, not merely within tolerance**, as theory predicts: BTF finds the B disconnected components, AMD
orders within each, and no fill crosses a block, so partial pivoting cannot reach across. Iteration counts
match per scenario, so the embedding doesn't perturb convergence. `tests/bde_test.rs` additionally checks that
the stacked pattern contains no entry linking two scenarios, and that a masked scenario's block is exactly the
identity *in the same stored positions* — masking must preserve sparsity, since dropping a converged
scenario's block would invalidate the cached symbolic factorization.

**It is ~2.7x slower on a CPU, and that is expected.** One large factorization beats B small ones only on
hardware that wants wide independent work — a GPU property. `bde.rs` is an architecture validator and the
host-side half of Phase 3, **not** a CPU optimization; use `batch::BatchSolver` for real CPU work.

## 5. Cross-validate CGMES import against pypowsybl

`cross_validate_cgmes_microgrid_be.py` checks gridoxide's CGMES import + solve against pypowsybl's own, on the
same ENTSO-E MicroGrid-BE-MAS conformance files `tests/cgmes_microgrid_be_test.rs` checks against published
`SvVoltage`. That test's doc comment claimed pypowsybl "also deviates by a comparable few percent" as a manual
finding; this script computes and asserts it.

```bash
pip install pypowsybl
python3 scripts/bench/cross_validate_cgmes_microgrid_be.py
```

Runs `examples/cgmes_microgrid_be_dump.rs` as a subprocess (predates `PowerFlowModel.from_cgmes`; §6 uses the
native binding), then zips BE-MAS + boundary files for pypowsybl and solves with the same "BASIC"
`LoadFlowParameters` used everywhere here. pypowsybl's OpenLoadFlow is pinned via `slackBusesIds` to
gridoxide's own slack so angles are comparable. Worst observed deviation: 0.22% voltage / 0.07° angle;
`--tol`/`--angle-tol` default to 1% / 0.3°.

## 6. Benchmark against CGMES conformance test configurations

gridoxide (`KluNative`) against pypowsybl/powsybl-open-loadflow, both importing and solving the *same* CGMES
profile files — no conversion step on either side. pypowsybl is the only vendored reference with native CGMES
import at all (neither PGM nor lightsim2grid has a single CGMES file in its tree).

```bash
maturin develop --release --features python,cgmes
python3 scripts/bench/bench_gridoxide_cgmes.py <fixture_name> <profile.xml>...
python3 scripts/bench/bench_pypowsybl_cgmes.py <fixture_name> <profile.xml>...
```

`bench_gridoxide_cgmes.py` uses `PowerFlowModel.from_cgmes` directly; `bench_pypowsybl_cgmes.py` zips profiles
into a temp archive first (pypowsybl's importer needs one archive). Both also report deviation from the
fixture's published `SvVoltage` (parsed by the shared `cgmes_sv.py`). On gridoxide's side this is exact —
`bus_index_for_mrid` looks a bus up by `TopologicalNode` mRID. On pypowsybl's side it's a heuristic:
`match_powsybl_buses_to_tn` reconstructs the mapping from the TP profile's `ConnectivityNodeContainer`
references (pypowsybl's bus-view ID is `"<container-mRID>_<index>"`), resolving multi-node containers by
nearest-magnitude matching against the *published* voltage — kept independent of gridoxide's own solve.

### Timing results

Warm mean (5 timed calls on one persistent model) plus each side's cold (construct+solve):

| fixture | gridoxide nodes | pypowsybl nodes | gridoxide mean (ms) | pypowsybl mean (ms) | gridoxide cold (ms) | pypowsybl cold (ms) |
|---|---|---|---|---|---|---|
| PowerFlow | 2 | 2 | 0.007 | 1.398 | 0.55 | 19.26 |
| MiniGrid | 15 | 11 | 0.019 | 1.306 (no solve)¹ | 3.86 | 65.00 (no solve)¹ |
| PST_PhaseTapChangerTable_Type3 | 2 | 2 | 0.006 | 1.386 | 0.60 | 22.96 |
| MicroGrid-BE-MAS | 13 | 7 | 0.023 | 0.872 | 1.52 | 35.86 |
| MicroGrid-Type2-HVDC-MAS | 6 | 4 | 0.010 | 0.711 (no solve)⁵ | 0.91 | 27.33 |
| SmallGrid | 163 | 120 | 0.188 | 2.555 | 67.91 | 519.63 |
| Svedala | 191 | 104 | 0.188 | 2.511 | 65.71 | 543.74 |
| RealGrid | 6,252 | 5,806 | 18.258 | 136.959 | 1,449.74 | 5,890.94 |

gridoxide's node counts run higher because `PersistentSolver::solve` keeps every disconnected switchyard stub
as its own island with a placeholder solution, while pypowsybl's `connected_component_mode=MAIN` solves only
the largest component. gridoxide is faster on every fixture where both actually solve — excluding MiniGrid and
MicroGrid-Type2-HVDC-MAS, whose `(no solve)` cells measure time to hit an iteration limit or decide there is
nothing to calculate. On the rest the ratio shrinks with size, from ~200x on the 2-bus fixtures (where fixed
per-call overhead dominates) to ~14x on SmallGrid/Svedala and ~7.5x on RealGrid. Construction dominates
pypowsybl's cold figure far more: its Java CGMES importer costs tens to hundreds of ms even on 2-bus fixtures,
while gridoxide's cold barely exceeds its warm solve at small scale (PowerFlow: 0.55 vs 0.007 ms).

These are faster than an earlier version (RealGrid warm was 24.8 ms) because removing `network::dc_angle_guess`
(§4) eliminated one extra sparse solve per `linear_initial_guess` call — ~26% on RealGrid, whose ~6,000-bus
system was the most expensive to redo. The table predates the two importer fixes below and was not
re-measured; spot-checks gave RealGrid 17.61/1,425 and Svedala 0.172/70.2 with identical node counts.

### Accuracy results

Deviation from each fixture's published `SvVoltage` (`n` = `TopologicalNode`s matched to a solved bus):

| fixture | gridoxide n | median | p90 | max | pypowsybl n | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| PowerFlow | 2 | 0.0000% | 0.0001% | 0.0001% | 2 | 0.0001% | 0.0001% | 0.0001% |
| MiniGrid | 11 | 1.248% | 2.334% | 2.387% | — | NOT CONVERGED¹ | — | — |
| PST_PhaseTapChangerTable_Type3 | 2 | 0.0001% | 0.0001% | 0.0002% | 2 | 0.0002% | 0.0002% | 0.0002% |
| MicroGrid-BE-MAS | 7 | 0.527% | 2.573% | 2.612% | 7 | 0.477% | 2.612% | 2.612% |
| MicroGrid-Type2-HVDC-MAS | 2² | 3.315%² | 3.315%² | 3.315%² | — | NO CALCULATION⁵ | — | — |
| SmallGrid | 127 | 0.002% | 0.217% | 0.447% | 7³ | 0.002%³ | 2.833%³ | 2.833%³ |
| Svedala | 108 | 0.213% | 1.000% | 2.865% | 96 | 0.285% | 1.627% | 4.071% |
| RealGrid | 6,051 | 0.018% | 0.424% | 93.475%⁴ | 5,806 | 0.099% | 0.822% | 82.068%⁴ |

¹ pypowsybl hits its Newton-Raphson iteration limit on MiniGrid under flat-start "BASIC" parameters. This cell
used to read `0.000%` with the non-convergence in a footnote — actively misleading, since 0.000% is the *best
possible score in the column*. Those figures came from the two boundary buses whose voltage never moves, i.e.
comparing the input SV profile against itself. `bench_pypowsybl_cgmes.py` now prints `NOT CONVERGED` on
stdout; it previously warned only on stderr, which the `2>/dev/null` used to produce these tables discarded.

² gridoxide's `n=2` reflects that this fixture's published SV set includes `TopologicalNode`s (DC-side detail
resolved away by `cgmes_resolve_dc_converters`, `src/dc.rs`) with no AC bus in the solved output — not a
matching failure. This is the one row **not** re-measured after the two fixes below: reproducing `n=2` needs
the original profile-file set, and including the `-BD-MAS` boundary files the fixture requires yields `n=4`.
Treat these four figures as provisional.

³ pypowsybl's container-based bus reconstruction breaks down on SmallGrid: only 7 of 118 candidate containers
matched, because its TP profile groups far more `TopologicalNode`s per container (up to 9) than pypowsybl's
bus view does, so the join key mismatches. A limitation of this benchmark's matching heuristic on a fixture
with 838 `Disconnector`s + 427 `Breaker`s, not a pypowsybl accuracy issue. gridoxide's `n=127` has no such
uncertainty.

⁴ **RealGrid's `max` is a defect in the fixture**, now pinned down rather than inferred from tool agreement:

```bash
python3 scripts/bench/check_cgmes_sv_consistency.py RealGrid
```

Compares each two-winding transformer's published voltage ratio against what its own `ratedU` values declare,
adjusted by the actual `RatioTapChanger.step`. On RealGrid, 83 of 1,461 transformers miss their nameplate
ratio by >5%, and the four worst by **70–87%** — LV sides of four 63/20 kV units published at 10.76–11.88 kV
where the `TopologicalNode`'s `BaseVoltage`, the end's `ratedU`, and all five `SynchronousMachine.ratedU`
values say 20 kV. No tap position produces that; those are exactly the four buses at the top of gridoxide's
error list. Excluding the 140 buses on a >5%-inconsistent transformer, RealGrid becomes `n=5911 median=0.017%
p90=0.423% max=4.318%` — the entire 93% headline is fixture data, and the median and p90 barely move.

⁵ pypowsybl doesn't solve MicroGrid-Type2-HVDC-MAS at all: `run_ac` reports `status=NO_CALCULATION,
iteration_count=0, status_text="Network has no generator with voltage control enabled"`. CGMES
`VsConverter`/`CsConverter` aren't recognized as AC voltage-controlling sources, so `get_buses()` echoes back
the state its importer initialized from the input SV profile — both the 0.000% "deviation" and the timing are
artifacts. gridoxide's DC-aware solve (`src/dc.rs`) is the only one of the two that solves this fixture.

Where both tools have comparable `n` (PowerFlow, PST, MicroGrid-BE, Svedala, RealGrid), accuracy against the
published reference is close — consistent with §5, and not with either tool being systematically more correct.

**Two CGMES importer bugs, found by working outward from the worst-deviating buses:**

1. **`build_two_winding` applied only one end's structural voltage ratio.** A `PowerTransformerEnd`'s `ratedU`
   need not equal its bus's `nominalVoltage`, and that difference *is* an off-nominal ideal-transformer ratio
   independent of any tap changer. The importer folded in end 1's `ratedU / bus1.u_rated` but dropped end 2's.
   Invisible on every hand-authored fixture (all have `ratedU == nominalVoltage`), pervasive on real data:
   **666 of RealGrid's 1,509** two-winding transformers and 4 of Svedala's 53, by up to 7.2%. Fixing it cut
   RealGrid's median deviation **5x, 0.088% → 0.018%**.
2. **An out-of-service `SynchronousMachine` still regulated voltage.** The machine loop gated on terminal
   connectivity and `RegulatingCondEq.controlEnabled` but never `Equipment.inService`, though
   `equipment_in_service` was already applied to lines, switches and shunts in the same file. All six of
   Svedala's `inService=false` machines also carry `controlEnabled=true`, and one (`_f4cde1f4`, `p=q=0`)
   regulates a *remote* terminal, pinning an in-service bus to 21 kV against the published 20.134 kV —
   Svedala's worst bus. `tests/cgmes_svedala_test.rs`'s de-energized count moved from 78 to the correct 83.

Net: RealGrid median 0.088% → 0.018%, p90 0.525% → 0.424%; Svedala p90 halved (2.121% → 1.000%) and max fell a
third (4.351% → 2.865%). Svedala's *median* moved the other way, 0.081% → 0.213% — about a thousandth of a
per-unit on the middle bus of a 108-bus fixture. Both changes are justified by the fixture's own equipment
data and both moved the tail substantially, but the median did not improve.

## 7. State estimation against power-grid-model

Every section above times a *power flow*; this one times a *state estimation*: gridoxide's two methods against
power-grid-model's two, on the same MATPOWER cases from byte-identical measurements. PGM is the only vendored
reference that does state estimation at all, so this is a two-tool comparison.

```bash
# 1. synthesize measurements and write the augmented documents both tools read
cargo run --release --example bench_se -- case14 case118 case300 case1354pegase case2869pegase \
    --emit /tmp/se_bench

# 2. time both tools on them
VIRTUAL_ENV=$PWD/.venv-ls2g .venv-ls2g/bin/maturin develop --release --features python
.venv-ls2g/bin/python scripts/bench/bench_se_pgm.py
```

The measurement set is generated once by gridoxide and written *into* the input document, so neither tool gets
a set tuned to it: a `sym_voltage_sensor` on every node and a `sym_power_sensor` on both ends of every line and
transformer, ~4x redundancy. Values come from gridoxide's power flow for a blunt reason — PGM's power flow does
not converge on any of these converted cases (on case300 the deviation *grows* to 1394 after 200 iterations).
That is worth chasing separately; it doesn't tilt this comparison, since both tools estimate from the same
numbers.

**The data is perfectly consistent, with no noise at all.** That is the right shape for timing, where cost
follows measurement count and sparsity, and the wrong shape for judging robustness. Bad-data behaviour needs a
different harness.

### Timing results

Best of 5 warm calls on one persistent model per tool, ms. Both sides amortize symbolic factorization across
those calls — PGM through `PowerGridModel`, gridoxide through the `PersistentEstimator` that
`StateEstimationModel` holds:

| case | buses | measurements | PGM nr | PGM il | gridoxide nr | gridoxide il |
|---|---|---|---|---|---|---|
| case14 | 14 | 94 | 0.2 | 0.2 | **0.1** | 0.1 |
| case118 | 118 | 862 | **0.7** | 0.5 | 1.4 | 0.6 |
| case300 | 300 | 1,944 | `SparseMatrixError`¹ | **0.8** | 4.3 | 1.3 |
| case1354pegase | 1,354 | 9,318 | `SparseMatrixError`¹ | **3.5** | 24.5 | 6.3 |
| case2869pegase | 2,869 | 21,197 | `SparseMatrixError`¹ | **8.1** | 16.5 | 16.5 |

¹ PGM's Newton-Raphson estimator raises `SparseMatrixError` ("possibly singular matrix! ... might mean the
system is not fully observable") on every case from 300 buses up, on documents its *own* iterative-linear
method estimates from the same sensors without complaint. The suggested cause does not fit: gridoxide's
Newton-Raphson converges on the same measurements to 1e-14, and gridoxide's observability analysis flags
exactly one unobservable state per case — the virtual slack bus it appends per active source (`src/pgm.rs`),
which sits at index `n_nodes`, carries no sensor by construction, is pinned rather than estimated, and is not
part of PGM's problem at all. Every physical node is observable. Reproduces from a plain `PowerGridModel`
construct-and-call with no gridoxide code in the path; not investigated further.

That extra bus is also why gridoxide's bus counts run one higher than the `buses` column (`bench_se.rs` prints
301 for case300), the same way §6's node counts exceed pypowsybl's.

**Newton-Raphson.** gridoxide is the only one of the two that answers at all above 300 buses, which outweighs
any timing statement about the cases where both run. Where both run they are within 2x either way: gridoxide
faster on case14 (0.1 vs 0.2 ms), PGM faster on case118 (0.7 vs 1.4 ms).

**Iterative-linear.** PGM is faster, by a ratio remarkably stable with size — 1.6x at case300, 1.8x at
case1354pegase, 2.0x at case2869pegase. This used to read that flatness as "a constant-factor gap in
per-iteration work, not an asymptotic one". **That inference was wrong, and backwards.** A stable ratio is
equally consistent with two ratios that happen to be stable, which is what this is.

```bash
.venv-ls2g/bin/python3 scripts/bench/se_iterations.py /tmp/se_bench
```

`se_iterations.py` measures the iteration count identically for both tools — the smallest `max_iterations`
that does not fail — having first checked the two convergence criteria are the same quantity (both take
`max over buses of |Δu|`, phase-normalized, against a 1e-8 default):

| case | PGM its | gridoxide its | iterations | ms per iteration |
|---|---|---|---|---|
| case300 | 9 | 29 | 3.2x | **0.62x** |
| case1354pegase | 10 | 28 | 2.8x | **0.71x** |
| case2869pegase | 10 | 33 | 3.3x | **0.71x** |

**gridoxide's iterations are 30-40% cheaper than PGM's; it just takes three times as many.** Both tools
reach the same answer — max |Δu| between their solutions is 7.6e-9 to 5.2e-7, agreement at their shared
tolerance — so this is one problem with one optimum and two paths to it. The standing gap is a
convergence-rate gap, and the linear algebra the earlier reading pointed at is already ahead.

Taking the damping out does not close it: undamped, gridoxide's map locks into a period-2 limit cycle at
~1.7e-1 rather than converging, so the under-relaxation is load-bearing rather than overhead.
`docs/src/state_estimation/iterative.md` has the trace and rules out the two obvious mechanisms — the `|U|²`
weight scaling PGM deliberately omits, and the zero-injection KKT constraints it has no equivalent of.
Neither removes the cycle.

**Accuracy is not a differentiator on this data.** Where PGM answers, the two tools agree to 5.8e-15 V on
case14 and 3.3e-15 V on case118. Against the state the measurements were read from, gridoxide's
Newton-Raphson lands at ~1e-14 p.u. throughout and its iterative-linear at 8.7e-10 through 1.1e-6, degrading
with size — the linearization bias inherent to the method, documented in
`docs/src/state_estimation/iterative.md`, not an implementation artifact.

A caveat this table used to carry is now retired: gridoxide's column no longer includes a fresh symbolic
factorization per call. `StateEstimationModel` holds a `crate::se::nr::PersistentEstimator` whose cache
survives across `solve()` calls, so the symbolic phase is paid once per model on both sides.

### Sanity-checking against `examples/bench_se.rs`

`examples/bench_se.rs` times gridoxide's two methods against *each other* and reports 117 ms for
Newton-Raphson on case1354pegase where the table above reports 24.5 ms. Both are correct; they are not the
same problem. `bench_se.rs` also synthesizes bus-injection measurements (11,189 scalar rows vs the emitted
document's 9,318), and an injection row is a full Y-bus row rather than a two-bus branch row, so its gain
matrix is substantially denser. The emitted document drops them because PGM has no counterpart sensor.
**Compare numbers within one harness, never across the two.**
