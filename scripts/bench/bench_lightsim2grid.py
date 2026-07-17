#!/usr/bin/env python3
"""Runs lightsim2grid directly (no grid2op) on one of pandapower's bundled
power-system test-case grids (see cases.py) and times `GridModel.ac_pf`, for
comparison against gridoxide's `examples/bench_network.rs` and PGM's
`bench_pgm.py` on the same underlying case.

Usage: python3 bench_lightsim2grid.py <case_name>

`case_name` is a function in `pandapower.networks` (e.g. "case14"). Unlike
`bench_pgm.py`, this loads the case directly via `pandapower.networks`
rather than a converted PGM JSON file — lightsim2grid's `init_from_pandapower`
needs the pandapower net object itself, not gridoxide's PGM-format input.

Requires `pandapower` and `lightsim2grid`: pip install pandapower lightsim2grid

Only `SolverType.KLU` is benchmarked here (matching lightsim2grid's own
`benchmarks/benchmark_grid_size.py`, which fixes `ls_solver_type =
SolverType.KLU`), for the closest comparison against gridoxide's own `Klu`
backend. `grid.ac_pf(V_init, max_it, tol)` is the same primitive
`LightSimBackend.runpf()` calls internally (confirmed by reading
`lightSimBackend.py`) — the direct analogue of PGM's
`model.calculate_power_flow(...)`.

Note: lightsim2grid solves the pandapower net's `gen` elements as true PV
buses directly, the same way gridoxide now does via PGM's own
`voltage_regulator` component (see convert_pandapower_case.py's docstring)
— voltages between the two should track pandapower's own `runpp` closely.
See convert_pandapower_case.py's docstring for a known transformer-encoding
data-quality caveat on these specific real-world cases, which also applies
here.
"""
import sys
import time

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from lightsim2grid import SolverType
from lightsim2grid.gridmodel import init_from_pandapower

case_name = sys.argv[1] if len(sys.argv) > 1 else "case14"

t0 = time.perf_counter()
net = getattr(pn, case_name)()
pp.runpp(net)
grid = init_from_pandapower(net)
grid.change_solver(SolverType.KLU)
t1 = time.perf_counter()
print(f"model construction: {(t1 - t0) * 1e3:.3f} ms", file=sys.stderr)

n_bus = len(grid.get_bus_vn_kv())


def run_ac_pf():
    v_init = np.ones(n_bus, dtype=complex) * grid.get_init_vm_pu()
    v = grid.ac_pf(v_init, 20, 1e-6)
    if v.shape[0] == 0:
        raise RuntimeError(f"lightsim2grid ac_pf diverged for {case_name}")
    return v


# Warm-up (first-call overhead), then timed runs.
run_ac_pf()

times = []
v = None
for _ in range(5):
    t0 = time.perf_counter()
    v = run_ac_pf()
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"nodes={n_bus}")
print(f"ac_pf (warm, 5 runs, ms): {[f'{t * 1e3:.3f}' for t in times]}")
print(f"min={min(times) * 1e3:.3f}ms mean={sum(times) / len(times) * 1e3:.3f}ms")

vm = np.abs(v)
print(f"sample vm[0:5] = {vm[:5]}")
print(f"vm min/max = {vm.min():.6f} / {vm.max():.6f}")

# Also time full cold construction+calc, comparable to gridoxide's "total"
# (parse + build + solve) figure.
t0 = time.perf_counter()
net2 = getattr(pn, case_name)()
pp.runpp(net2)
grid2 = init_from_pandapower(net2)
grid2.change_solver(SolverType.KLU)
v_init2 = np.ones(n_bus, dtype=complex) * grid2.get_init_vm_pu()
grid2.ac_pf(v_init2, 20, 1e-6)
t1 = time.perf_counter()
print(f"cold (construct+calc): {(t1 - t0) * 1e3:.3f} ms")
