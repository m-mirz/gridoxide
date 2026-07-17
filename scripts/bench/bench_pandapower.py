#!/usr/bin/env python3
"""Runs pandapower's own default power-flow solver on one of pandapower's
bundled power-system test-case grids (see cases.py) and times `pp.runpp`,
for comparison against gridoxide, PGM, lightsim2grid, and pypowsybl on the
same underlying case.

Usage: python3 bench_pandapower.py <case_name>

`case_name` is a function in `pandapower.networks` (e.g. "case14").

Requires `pandapower`: pip install pandapower

Uses `pp.runpp`'s own defaults unmodified — `algorithm="nr"` (its PYPOWER-
derived Newton-Raphson, numba-accelerated when available), single slack
(`distributed_slack=False`), no reactive-limit enforcement
(`enforce_q_lims=False`) — the closest match to gridoxide's own flat-start,
single-slack, no-reactive-limit-enforcement Newton-Raphson, and to how the
other tools in this suite are configured (see bench_pypowsybl.py's
docstring). This is the same solver `pandapower.networks.<case_name>()`'s
own bundled data was presumably validated against, and — unlike
lightsim2grid/pypowsybl, which load the same pandapower net object here —
it's pandapower's own implementation, not a third-party backend swapped in
underneath it.
"""
import sys
import time

import pandapower as pp
import pandapower.networks as pn

case_name = sys.argv[1] if len(sys.argv) > 1 else "case14"

t0 = time.perf_counter()
net = getattr(pn, case_name)()
t1 = time.perf_counter()
print(f"model construction: {(t1 - t0) * 1e3:.3f} ms", file=sys.stderr)


def run() -> None:
    pp.runpp(net)


# Warm-up (first-call overhead, including numba JIT if available), then timed runs.
run()

times = []
for _ in range(5):
    t0 = time.perf_counter()
    run()
    t1 = time.perf_counter()
    times.append(t1 - t0)

n_bus = len(net.bus)
print(f"nodes={n_bus}")
print(f"runpp (warm, 5 runs, ms): {[f'{t * 1e3:.3f}' for t in times]}")
print(f"min={min(times) * 1e3:.3f}ms mean={sum(times) / len(times) * 1e3:.3f}ms")

vm_pu = net.res_bus["vm_pu"]
print(f"vm_pu min/max = {vm_pu.min():.6f} / {vm_pu.max():.6f}")

# Also time full cold construction+calc, comparable to gridoxide's "total"
# (parse + build + solve) figure.
t0 = time.perf_counter()
net2 = getattr(pn, case_name)()
pp.runpp(net2)
t1 = time.perf_counter()
print(f"cold (construct+calc): {(t1 - t0) * 1e3:.3f} ms")
