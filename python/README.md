# gridoxide

Python bindings for [`gridoxide`](https://github.com/m-mirz/gridoxide), a Rust AC power flow solver
using the Newton-Raphson method with a sparse Jacobian (via [`faer`](https://crates.io/crates/faer)).

Grids are loaded from [power-grid-model (PGM)](https://github.com/PowerGridModel/power-grid-model) JSON
input files. Repeated solves against the same topology reuse a cached symbolic factorization, so scenario
sweeps and time-series runs only pay the fill-reducing-ordering cost once.

## Install

```bash
pip install gridoxide
```

Prebuilt wheels are published for Linux (x86_64), Windows, and macOS (arm64).

## Quickstart

```python
import gridoxide

model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="scalar")
model.solve()
print(model.voltage_mag())  # per-unit voltage magnitude, one entry per node
print(model.voltage_ang())  # voltage angle in radians, one entry per node
```

`grid.json` is a PGM-format input file, e.g.:

```json
{
  "version": "1.0",
  "type": "input",
  "is_batch": false,
  "attributes": {},
  "data": {
    "node": [
      {"id": 1, "u_rated": 10500.0},
      {"id": 2, "u_rated": 10500.0}
    ],
    "line": [
      {"id": 3, "from_node": 1, "to_node": 2, "from_status": 1, "to_status": 1,
       "r1": 0.25, "x1": 0.2, "c1": 1e-06, "tan1": 0.0,
       "r0": 0.25, "x0": 0.2, "c0": 1e-06, "tan0": 0.0, "i_n": 1000.0}
    ],
    "sym_load": [
      {"id": 9, "node": 2, "status": 1, "type": 0, "p_specified": 10000.0, "q_specified": 2000.0}
    ],
    "source": [
      {"id": 4, "node": 1, "status": 1, "u_ref": 1.0, "sk": 1e10, "rx_ratio": 0.1, "z01_ratio": 1.0}
    ]
  }
}
```

## Reusing factorization across repeated solves

`PowerFlowModel` wraps gridoxide's `PersistentSolver`: construct one per topology, then call `.solve()`
as many times as needed. Only the first call pays for symbolic factorization; later calls only need
values (`p_specified`/`q_specified`) to change, not the topology.

```python
model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="scalar")
for scenario in scenarios:
    apply_scenario(model, scenario)  # e.g. edit and re-load p/q values
    model.solve()
    results.append(model.voltage_mag())
```

Call `model.reset()` if the topology itself changes between solves (not just bus values).

## Building input grids

Two helpers produce or convert PGM JSON, so a working grid doesn't require hand-writing one:

- `gridoxide.generate_grid` — generates a synthetic radial MV/LV distribution grid at any scale, no
  extra dependencies needed:

  ```python
  from gridoxide.generate_grid import generate
  generate(target_nodes=2200, seed=42, out_path="grid.json")  # ~2,600 nodes
  ```

  or from the shell: `gridoxide-generate-grid grid.json --target-nodes 2200 --seed 42`.

- `gridoxide.matpower` (needs `pip install gridoxide[matpower]`) — converts a raw MATPOWER case
  (`.m` or `.mat`) into PGM JSON:

  ```python
  from gridoxide.matpower import convert
  convert("case14.m", "case14.json")
  ```

  or `gridoxide-matpower case14.m case14.json`.

`gridoxide` deliberately has no pandapower-based converter: it would need the full pandapower +
power-grid-model-io dependency chain, and `gridoxide.matpower` already covers the same real-world
test-case grids straight from their original MATPOWER source files without it. If you already have a
`pandapower.pandapowerNet` object and pandapower installed, see
[`scripts/bench/convert_pandapower_case.py`](https://github.com/m-mirz/gridoxide/blob/main/scripts/bench/convert_pandapower_case.py)
in the main repo for a standalone (not part of this package) converter.

## API

- `PowerFlowModel.from_pgm_json(path, backend="scalar", tol=1e-6, max_iter=20, s_base_va=1e6, freq_hz=50.0)`
  — loads a PGM JSON file and builds the Y-bus admittance matrix.
- `model.n_nodes` — number of buses, including one virtual slack bus per active `source`.
- `model.solve()` — runs Newton-Raphson from a flat/linear-initial-guess start; raises `RuntimeError` if
  it doesn't converge within `max_iter` iterations.
- `model.reset()` — discards the cached symbolic factorization; call before the next `solve()` if the
  topology has changed.
- `model.voltage_mag()` / `model.voltage_ang()` — per-bus results in node order (per-unit magnitude,
  radians).

### Backends

`backend` selects the linear solver used inside the Newton-Raphson Jacobian:

| Backend | Notes |
|---|---|
| `"scalar"` (default) | Sparse LU via `faer`, no special build requirements. |
| `"block"` | Block-structured variant of the same solver; faster on some topologies, but panics on PV buses. |
| `"klu_native"` | From-scratch Rust translation of SuiteSparse KLU, always available in this wheel. |

Two additional backends exist in the source tree but are **not** included in the published wheel, since
they need extra system dependencies at build time — build gridoxide from source with the matching Cargo
feature to use them:

- `"klu"` — links vendored SuiteSparse C directly (`--features python,klu`).
- `"pardiso"` — Intel oneMKL's PARDISO solver (`--features python,pardiso`, needs `MKLROOT` set).

## License

gridoxide's own code is Apache-2.0. A default build (and this published wheel) also always bundles
`src/klu_native/`, a from-scratch Rust translation of vendored SuiteSparse AMD/BTF/KLU source
(BSD-3-Clause for the AMD-derived pieces, LGPL-2.1-or-later for the BTF/KLU-derived pieces) — see the
[main README's License section](https://github.com/m-mirz/gridoxide#license) for the full breakdown.
