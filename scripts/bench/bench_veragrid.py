#!/usr/bin/env python3
"""Runs VeraGrid (https://github.com/SanPen/VeraGrid) directly on a
MATPOWER case file and times its `PowerFlowDriver`, for comparison against
gridoxide, PGM, lightsim2grid, pypowsybl, and pandapower on the same
underlying case.

Usage: python3 bench_veragrid.py <case_name> <input.m-or-.mat>

`case_name` is only used for log messages; `input.m-or-.mat` is the
MATPOWER case file (see cases.py / matpower_to_pgm.py). VeraGrid's own
`parse_matpower_file` reads MATPOWER's `bus`/`gen`/`branch` matrices
directly (both `.m` and `.mat`), including each bus's `type` column
(1=PQ, 2=PV, 3=ref) and each generator's `Vg` voltage setpoint — so PV-bus
handling comes for free from the raw case file, the same way it does for
gridoxide (via PGM's `voltage_regulator`) and pypowsybl (via its own
MATPOWER importer), with no extra conversion step needed here.

Requires `VeraGridEngine` (the headless engine package — not `VeraGrid`,
which pulls in its Qt-based desktop GUI and its far heavier dependency
set): pip install VeraGridEngine

Uses `SolverType.NR` (Newton-Raphson, the same algorithm gridoxide/PGM/
lightsim2grid/pandapower all use here) with every automatic-control
feature this benchmark's other tools also disable for comparability
turned off: `retry_with_other_methods=False` (no automatic fallback to a
different algorithm if NR itself struggles — this script reports NR's own
result, unmodified, the same way bench_pypowsybl.py reports
powsybl-open-loadflow's raw, unmodified result), `distributed_slack=False`
(single slack, already VeraGrid's own default), `control_q=False`
(already VeraGrid's own default — no reactive-limit enforcement, matching
every other tool in this suite), and `control_taps_modules=False`/
`control_taps_phase=False`/`control_remote_voltage=False` (VeraGrid's own
defaults for these are all `True` — automatic tap-changer and remote
voltage regulation during the power-flow solve itself, the same category
of feature bench_pypowsybl.py explicitly disables via
`phase_shifter_regulation_on=False`/`transformer_voltage_control_on=False`
for the same reason: a fixed-topology, no-extra-control Newton-Raphson is
what every other tool here is being compared against).

VeraGrid's numerical kernels are numba-JIT-compiled — like pandapower, the
*first* solve call pays a one-off JIT-compilation cost (seconds, not
milliseconds) that has nothing to do with the actual power-flow algorithm;
the warm-up call below absorbs that before any timed run.
"""
import sys
import time

import numpy as np
import VeraGridEngine as vg

case_name = sys.argv[1] if len(sys.argv) > 1 else "case14"
input_path = sys.argv[2] if len(sys.argv) > 2 else f"{case_name}.m"

t0 = time.perf_counter()
grid, logger = vg.parse_matpower_file(input_path)
t1 = time.perf_counter()
print(f"model construction: {(t1 - t0) * 1e3:.3f} ms", file=sys.stderr)

n_bus = grid.get_bus_number()

options = vg.PowerFlowOptions(
    solver_type=vg.SolverType.NR,
    retry_with_other_methods=False,
    distributed_slack=False,
    control_q=False,
    control_taps_modules=False,
    control_taps_phase=False,
    control_remote_voltage=False,
    verbose=0,
)
driver = vg.PowerFlowDriver(grid, options)


def run() -> None:
    driver.run()
    if not driver.results.converged:
        raise RuntimeError(f"VeraGrid power flow did not converge for {case_name}")


# Warm-up (first-call overhead, including numba JIT compilation), then timed runs.
run()

times = []
for _ in range(5):
    t0 = time.perf_counter()
    run()
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"nodes={n_bus}")
print(f"run (warm, 5 runs, ms): {[f'{t * 1e3:.3f}' for t in times]}")
print(f"min={min(times) * 1e3:.3f}ms mean={sum(times) / len(times) * 1e3:.3f}ms")

vm = np.abs(driver.results.voltage)
print(f"vm min/max = {vm.min():.6f} / {vm.max():.6f}")

# Also time full cold construction+calc, comparable to gridoxide's "total"
# (parse + build + solve) figure.
t0 = time.perf_counter()
grid2, _ = vg.parse_matpower_file(input_path)
driver2 = vg.PowerFlowDriver(grid2, options)
driver2.run()
t1 = time.perf_counter()
print(f"cold (construct+calc): {(t1 - t0) * 1e3:.3f} ms")
