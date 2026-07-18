#!/usr/bin/env python3
"""Runs power-grid-model's Python bindings on a PGM-format JSON network and
times `calculate_power_flow`, for direct comparison against gridoxide's own
`examples/bench_network.rs` on the exact same input file.

Usage: python3 bench_pgm.py <input.json>

Requires the `power-grid-model` package (a prebuilt wheel, no C++ build
needed): `pip install power-grid-model` (in a venv, e.g. `python3 -m venv
.venv && .venv/bin/pip install power-grid-model`).

Calls the private `_calculate_power_flow` (not the public `calculate_power_flow`)
with `experimental_features="enabled"`, deliberately: since `matpower_to_pgm.py`
started writing real `q_min`/`q_max` onto `voltage_regulator` (see that script's
docstring), every one of these input files now trips PGM's own
`ExperimentalFeature` error ("Voltage Regulator with Qmin/Qmax limits is an
experimental feature") when run through the public API — confirmed directly
(installed version 1.13.120) that `experimental_features` is a real parameter
of `_calculate_power_flow` but the public `calculate_power_flow` wrapper never
forwards it, so there is currently no supported way to opt in other than
calling the private method. This reproduces the exact same converged voltages
the public API produced before `matpower_to_pgm.py` started including Q-limits
(`case14`: `u_pu` 1.01/1.09, matching `scripts/bench/README.md`'s own footnote) —
confirmed directly, not assumed. Revisit once a future power-grid-model release
stabilizes this and threads the flag through the public API.
"""
import sys
import time

from power_grid_model import PowerGridModel, CalculationMethod
from power_grid_model.utils import json_deserialize

path = sys.argv[1] if len(sys.argv) > 1 else "grid_bench_input.json"
with open(path) as f:
    raw = f.read()

dataset = json_deserialize(raw)

t0 = time.perf_counter()
model = PowerGridModel(dataset)
t1 = time.perf_counter()
print(f"model construction: {(t1 - t0) * 1e3:.3f} ms", file=sys.stderr)

# Warm-up (first-call overhead), then timed runs.
model._calculate_power_flow(  # noqa: SLF001
    calculation_method=CalculationMethod.newton_raphson, symmetric=True, experimental_features="enabled"
)

n_node = len(dataset["node"])
times = []
for _ in range(5):
    t0 = time.perf_counter()
    result = model._calculate_power_flow(  # noqa: SLF001
        calculation_method=CalculationMethod.newton_raphson, symmetric=True, experimental_features="enabled"
    )
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"nodes={n_node}")
print(f"calculate_power_flow (warm, 5 runs, ms): {[f'{t * 1e3:.3f}' for t in times]}")
print(f"min={min(times) * 1e3:.3f}ms mean={sum(times) / len(times) * 1e3:.3f}ms")

u_pu = result["node"]["u_pu"]
print(f"sample u_pu[0:5] = {u_pu[:5]}")
print(f"u_pu min/max = {u_pu.min():.6f} / {u_pu.max():.6f}")

# Also time full cold construction+calc, comparable to gridoxide's "total"
# (parse + build + solve) figure.
t0 = time.perf_counter()
model2 = PowerGridModel(dataset)
model2._calculate_power_flow(  # noqa: SLF001
    calculation_method=CalculationMethod.newton_raphson, symmetric=True, experimental_features="enabled"
)
t1 = time.perf_counter()
print(f"cold (construct+calc): {(t1 - t0) * 1e3:.3f} ms")
