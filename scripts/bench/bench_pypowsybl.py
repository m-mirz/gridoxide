#!/usr/bin/env python3
"""Runs powsybl-open-loadflow (via pypowsybl) directly on a MATPOWER case
file and times `loadflow.run_ac`, for comparison against gridoxide, PGM,
and lightsim2grid on the same underlying case.

Usage: python3 bench_pypowsybl.py <case_name> <input.m-or-.mat>

`case_name` is only used for log messages; `input.m-or-.mat` is the
MATPOWER case file (see cases.py / matpower_to_pgm.py). pypowsybl's own
MATPOWER importer only reads `.mat` (binary), not `.m` (plain text) — a
`.m` input is re-serialized to a temporary `.mat` first via
matpower_to_pgm.py's `load_mpc()` + `scipy.io.savemat()`. Getting this
right needed two things confirmed empirically, not assumed: the struct
must be nested under a top-level `mpc` key (a flat top-level
bus/gen/branch/baseMVA layout raises "expected structure named 'mpc' not
found"), and `mpc` needs an explicit `version` field (its absence raises
"expected MATPOWER variables not found: [... version ...]" even though
MATPOWER case files don't strictly need one for power flow).

Requires `pypowsybl` and `scipy`: pip install pypowsybl scipy

Uses the same "basic" `LoadFlowParameters` powsybl's own benchmark repo
does (uniform/flat voltage init, no distributed slack, no reactive limits,
phase shifter regulation off, main connected component only — see
https://github.com/powsybl/powsybl-benchmark's `LoadFlowParametersType.BASIC`),
for the closest comparison against gridoxide's own flat-start, single-slack,
no-reactive-limit-enforcement Newton-Raphson.

Deliberately does *not* apply the "FIX RTE cases" phase-shift-zeroing
workaround `powsybl-benchmark`'s own `MatpowerUtil.java` applies before
benchmarking case1888rte/case6495rte/case6515rte — this script reports
whatever powsybl-open-loadflow does on the raw case, unmodified, the same
way bench_lightsim2grid.py reports lightsim2grid's raw (also unmodified)
result. Confirmed directly (see matpower_to_pgm.py's docstring and this
project's own investigation) that powsybl-open-loadflow itself fails to
converge on those three cases without that workaround — this is a genuine
property of that data, not a bug in this script.
"""
import atexit
import os
import sys
import tempfile
import time
from pathlib import Path

import pypowsybl.loadflow as lf
import pypowsybl.network as pn
import scipy.io as sio

from matpower_to_pgm import load_mpc

case_name = sys.argv[1] if len(sys.argv) > 1 else "case14"
input_path = Path(sys.argv[2] if len(sys.argv) > 2 else f"{case_name}.m")

if input_path.suffix == ".m":
    mpc = load_mpc(input_path)
    mpc.setdefault("version", "2")
    tmp_mat = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
    sio.savemat(tmp_mat.name, {"mpc": mpc})
    m_path = tmp_mat.name
    atexit.register(os.unlink, m_path)
else:
    m_path = str(input_path)

t0 = time.perf_counter()
network = pn.load(m_path)
t1 = time.perf_counter()
print(f"model construction: {(t1 - t0) * 1e3:.3f} ms", file=sys.stderr)

n_bus = len(network.get_buses())

params = lf.Parameters(
    voltage_init_mode=lf.VoltageInitMode.UNIFORM_VALUES,
    distributed_slack=False,
    use_reactive_limits=False,
    phase_shifter_regulation_on=False,
    transformer_voltage_control_on=False,
    connected_component_mode=lf.ConnectedComponentMode.MAIN,
)


def run_ac() -> None:
    result = lf.run_ac(network, parameters=params)
    if result[0].status != lf.ComponentStatus.CONVERGED:
        raise RuntimeError(f"pypowsybl load flow did not converge for {case_name}: {result[0].status_text}")


# Warm-up (first-call overhead), then timed runs.
run_ac()

times = []
for _ in range(5):
    t0 = time.perf_counter()
    run_ac()
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"nodes={n_bus}")
print(f"run_ac (warm, 5 runs, ms): {[f'{t * 1e3:.3f}' for t in times]}")
print(f"min={min(times) * 1e3:.3f}ms mean={sum(times) / len(times) * 1e3:.3f}ms")

voltages = network.get_buses()["v_mag"]
print(f"vm min/max = {voltages.min():.6f} / {voltages.max():.6f}")

# Also time full cold construction+calc, comparable to gridoxide's "total"
# (parse + build + solve) figure.
t0 = time.perf_counter()
network2 = pn.load(m_path)
lf.run_ac(network2, parameters=params)
t1 = time.perf_counter()
print(f"cold (construct+calc): {(t1 - t0) * 1e3:.3f} ms")
