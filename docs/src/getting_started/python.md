# Python Bindings

`gridoxide` ships as a pip-installable package exposing the solver to Python:

```bash
pip install gridoxide
```

Prebuilt wheels are published for Linux (x86_64), Windows, and macOS (arm64).

```python
import gridoxide

model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="klu_native")
model.solve()
print(model.voltage_mag())  # per-unit magnitude, one entry per node
print(model.voltage_ang())  # angle in radians, one entry per node
```

Grids are loaded from [power-grid-model (PGM)](https://github.com/PowerGridModel/power-grid-model)
JSON input files. `python/README.md` in the repository is the package's own PyPI landing page and
carries a full worked example of that input format.

## API

- `PowerFlowModel.from_pgm_json(path, backend="scalar", tol=1e-6, max_iter=20, s_base_va=1e6,
  freq_hz=50.0)` — loads a PGM JSON file and builds the Y-bus admittance matrix.
- `model.n_nodes` — number of buses, including one virtual slack bus per active `source`.
- `model.solve()` — runs Newton-Raphson from a flat/linear-initial-guess start; raises
  `RuntimeError` if it doesn't converge within `max_iter` iterations.
- `model.reset()` — discards the cached symbolic factorization; call before the next `solve()` if
  the topology has changed.
- `model.voltage_mag()` / `model.voltage_ang()` — per-bus results in node order.

## Reusing factorization across repeated solves

`PowerFlowModel` wraps the Rust `solver::PersistentSolver` — it *is* that API, not a reimplementation
of it — so repeated `.solve()` calls on one model reuse the cached symbolic factorization exactly as
described in [Backends and Factorization Reuse](../solvers/backends.md). Construct one model per
topology, then solve as many times as needed:

```python
model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="scalar")
for scenario in scenarios:
    apply_scenario(model, scenario)  # changes p/q values only
    model.solve()
    results.append(model.voltage_mag())
```

Call `model.reset()` if the topology itself changes between solves, not just bus values.

This is what lets the benchmark suite run its whole comparison in pure Python
(`scripts/bench/bench_gridoxide_native.py`), timing gridoxide with the same
`time.perf_counter()`-around-a-persistent-solve-object methodology every other tool there already
uses (PGM's `PowerGridModel`, lightsim2grid's `GridModel`, pandapower's `net`), rather than shelling
out to a compiled Rust binary and parsing its stdout.

## Backends available from Python

| Backend | Notes |
|---|---|
| `"scalar"` (default) | Sparse LU via `faer`, no special build requirements. |
| `"block"` | Block-structured variant (one 2×2 block per bus); faster on some topologies. |
| `"klu_native"` | From-scratch Rust translation of SuiteSparse KLU, always available in the wheel. |

Two further backends exist in the source tree but are **not** in the published wheel, since they
need extra system dependencies at build time — build from source with the matching Cargo feature:

- `"klu"` — links vendored SuiteSparse C directly (`--features python,klu`).
- `"pardiso"` — Intel oneMKL's PARDISO solver (`--features python,pardiso`, needs `MKLROOT` set).

See [Backends and Factorization Reuse](../solvers/backends.md) for what each one actually does and
how they compare.

## Building input grids

Two helpers produce or convert PGM JSON, so a working grid doesn't require hand-writing one. Both
ship in the pip package itself (they used to live only under `scripts/bench/`) and are installed as
console scripts:

- `gridoxide.generate_grid` — synthetic radial MV/LV distribution grid generator at any scale, pure
  stdlib, no extra dependencies:

  ```python
  from gridoxide.generate_grid import generate
  generate(target_nodes=2200, seed=42, out_path="grid.json")  # ~2,600 nodes
  ```

  or `gridoxide-generate-grid grid.json --target-nodes 2200 --seed 42`.

- `gridoxide.matpower` (needs `pip install gridoxide[matpower]`) — converts a raw MATPOWER `.m`/`.mat`
  case into PGM JSON:

  ```python
  from gridoxide.matpower import convert
  convert("case14.m", "case14.json")
  ```

  or `gridoxide-matpower case14.m case14.json`.

`scripts/bench/generate_grid.py` and `matpower_to_pgm.py` are thin CLI wrappers delegating to these
same package modules, so the conversion logic only lives in one place.

There is deliberately no pandapower-based converter: it would pull in the full pandapower +
power-grid-model-io dependency chain, and `gridoxide.matpower` already covers the same real-world
test-case grids straight from their original MATPOWER sources. If you already have a
`pandapower.pandapowerNet` and pandapower installed, `scripts/bench/convert_pandapower_case.py` in
the main repo is a standalone (not packaged) converter.

## How the extension is built

`src/python.rs` exposes `PersistentSolver` and PGM-JSON loading as `gridoxide._gridoxide`, a private
compiled extension module built with [maturin](https://www.maturin.rs/):

```bash
maturin develop --release --features python,klu
```

It is gated entirely behind the opt-in `python` Cargo feature and compiled by nothing else, so a
plain `cargo build`/`cargo test` never touches it — the feature must never be combined with a plain
`cargo` invocation.

This is a mixed Rust/Python maturin project (`pyproject.toml`'s `python-source = "python"` plus
`module-name = "gridoxide._gridoxide"`): pure-Python code lives in `python/gridoxide/` and ships in
the same wheel as the compiled extension, re-exported through `python/gridoxide/__init__.py` so
callers only ever write `import gridoxide`.

`python/tests/` holds a pytest suite (`scalar`/`block` only — no `klu`, matching what's published)
checked against this project's own committed PGM reference fixtures, run by
`.github/workflows/python.yml` on every push/PR. `.github/workflows/pypi.yml` builds wheels
(Linux/Windows/macOS) plus an sdist and publishes to PyPI via
[trusted publishing](https://docs.pypi.org/trusted-publishers/) on `v*` tags. The published wheel
**deliberately omits the `klu` backend** — LGPL-2.1-or-later vendored SuiteSparse source, plus a C
compiler and libclang needed on every target platform. See
[Provenance and Licensing](../reference/provenance.md).
