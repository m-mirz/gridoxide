#!/usr/bin/env python3
"""Runs power-grid-model's Python bindings on a PGM-format JSON network and
times `calculate_power_flow`, for direct comparison against gridoxide's own
`examples/bench_network.rs` on the exact same input file.

Usage: python3 bench_pgm.py <input.json>

Requires the `power-grid-model` package (a prebuilt wheel, no C++ build
needed): `pip install power-grid-model` (in a venv, e.g. `python3 -m venv
.venv && .venv/bin/pip install power-grid-model`).
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
