#!/usr/bin/env python3
"""Runs power-grid-model's Python bindings on a PGM-format JSON network and
times `calculate_power_flow`, for direct comparison against gridoxide's own
`examples/bench_network.rs` on the exact same input file.

Usage: python3 bench_pgm.py <input.json>

**As of the `power-grid-model` `main` branch (post "Activate q-limit handling,
remove experimental feature"), this needs a from-source build, not `pip install
power-grid-model`.** The latest PyPI release at the time of writing (1.12.110)
predates the `voltage_regulator` component `matpower_to_pgm.py` emits for every
generator, so it fails immediately with `PowerGridSerializationError: Cannot
find component with name: voltage_regulator!` — not a convergence failure, a
hard incompatibility. Build from source instead (needs Python >=3.12 and a
C++23 compiler, e.g. gcc>=14 or clang>=18 — see that repo's
`docs/advanced_documentation/build-guide.md`):
`CC=gcc-14 CXX=g++-14 uv build --wheel -o dist && pip install dist/*.whl`.

This used to call the private `_calculate_power_flow` with
`experimental_features="enabled"`, because `voltage_regulator`'s `q_min`/`q_max`
tripped PGM's `ExperimentalFeature` error through the public API on 1.13.120.
That gate is gone on current `main` (q-limit handling is no longer
experimental), confirmed directly: the public `calculate_power_flow` now
produces the same converged voltages the private-API workaround did. Revert to
the private-API call only if testing against an older PGM build that still
gates this feature.
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
model.calculate_power_flow(calculation_method=CalculationMethod.newton_raphson, symmetric=True)

n_node = len(dataset["node"])
times = []
for _ in range(5):
    t0 = time.perf_counter()
    result = model.calculate_power_flow(calculation_method=CalculationMethod.newton_raphson, symmetric=True)
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
model2.calculate_power_flow(calculation_method=CalculationMethod.newton_raphson, symmetric=True)
t1 = time.perf_counter()
print(f"cold (construct+calc): {(t1 - t0) * 1e3:.3f} ms")
