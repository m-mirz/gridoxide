#!/usr/bin/env python3
"""Runs powsybl-open-loadflow (via pypowsybl) directly on a CGMES
conformance test configuration and times `loadflow.run_ac`, for comparison
against gridoxide (`bench_gridoxide_cgmes.py`) on the *same* CGMES input —
the CGMES counterpart to `bench_pypowsybl.py`'s own MATPOWER-case
benchmark. pypowsybl is the only tool among this project's three vendored
references (`references/`) with any native CGMES import at all — confirmed
directly: neither `power-grid-model` nor `lightsim2grid` has a single
CGMES-related file anywhere in their own trees.

Usage: python3 bench_pypowsybl_cgmes.py <fixture_name> <profile.xml>...

Requires `pypowsybl`: pip install pypowsybl

Uses the same "BASIC"-style `LoadFlowParameters` `bench_pypowsybl.py`
already uses (uniform/flat voltage init, no distributed slack, no reactive
limits, phase shifter regulation off, main connected component only) — the
closest match to gridoxide's own flat-start, single-slack, no-reactive-
limit-enforcement `KluNative` Newton-Raphson.

pypowsybl's CGMES importer needs a single file or archive, not a list of
paths (confirmed via `help(pn.load)`: `file: Union[str, PathLike]`) — the
profile files are zipped into one temp archive first, the same approach
`cross_validate_cgmes_microgrid_be.py` already established for this reason.
This script's own CLI signature still matches `bench_gridoxide_cgmes.py`'s
exactly: fixture name, then any number of profile file paths.

Also reports pypowsybl's own deviation from the fixture's published
`SvVoltage`, matching the metric `bench_gridoxide_cgmes.py` reports —
independent per-tool accuracy against the fixture's own reference solution,
not a tool-vs-tool comparison. Since pypowsybl's own bus IDs aren't
TopologicalNode mRIDs, this needs `cgmes_sv.py`'s
`match_powsybl_buses_to_tn` (TP-file-derived container grouping, resolved
via nearest-magnitude matching against the *published* SvVoltage — kept
independent of gridoxide's own solve, unlike `cross_validate_cgmes_microgrid_be.py`'s
version of the same idea, which matches against gridoxide's solved values
instead since its purpose there is a tool-vs-tool cross-check, not an
accuracy-vs-reference metric).
"""
import math
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pypowsybl.loadflow as lf
import pypowsybl.network as pn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cgmes_sv import deviation_stats, match_powsybl_buses_to_tn, parse_sv_voltages, parse_tn_containers

fixture_name = sys.argv[1] if len(sys.argv) > 1 else "fixture"
profile_paths = [Path(p) for p in sys.argv[2:]]
if not profile_paths:
    print("usage: bench_pypowsybl_cgmes.py <fixture_name> <profile.xml>...", file=sys.stderr)
    sys.exit(1)

print(f"[{fixture_name}] pypowsybl (powsybl-open-loadflow)", file=sys.stderr)

tmp_dir = tempfile.TemporaryDirectory()
zip_path = Path(tmp_dir.name) / "profiles.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    for p in profile_paths:
        zf.write(p, arcname=p.name)

t0 = time.perf_counter()
network = pn.load(str(zip_path))
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


def run_ac():
    result = lf.run_ac(network, parameters=params)
    return result[0]


# Warm-up (first-call overhead), then timed runs.
warm_result = run_ac()
converged = warm_result.status == lf.ComponentStatus.CONVERGED
if not converged:
    # Deliberately on stdout, not just stderr: when this doesn't converge,
    # `network.get_buses()` still returns a full voltage vector — whatever the
    # CGMES importer initialized it to from the input SV profile, untouched by
    # any iteration. Comparing that against the published SV compares the input
    # against itself and yields a 0.0000% "deviation", i.e. the *best possible*
    # score, for a load flow that never ran. The tables in this directory's own
    # README were produced with `2>/dev/null`, so a stderr-only warning was
    # silently dropped and exactly that misleading 0.0000% is what got recorded.
    print(f"NOT CONVERGED: pypowsybl load flow did not converge for {fixture_name}: "
          f"{warm_result.status_text} (iterations={warm_result.iteration_count})")

times = []
for _ in range(5):
    t0 = time.perf_counter()
    run_ac()
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"nodes={n_bus}")
print(f"run_ac (warm, 5 runs, ms): {[f'{t * 1e3:.3f}' for t in times]}")
print(f"min={min(times) * 1e3:.3f}ms mean={sum(times) / len(times) * 1e3:.3f}ms")
print(f"status={warm_result.status_text}")

voltages = network.get_buses()["v_mag"]
print(f"vm min/max = {voltages.min():.6f} / {voltages.max():.6f}")

# Cold construction+calc, comparable to gridoxide's own "cold" figure.
t0 = time.perf_counter()
network2 = pn.load(str(zip_path))
lf.run_ac(network2, parameters=params)
t1 = time.perf_counter()
print(f"cold (construct+calc): {(t1 - t0) * 1e3:.3f} ms")

# Accuracy vs. the fixture's own published SvVoltage.
sv_paths = [p for p in profile_paths if "_SV" in p.name or "SV_" in p.name]
tp_paths = [p for p in profile_paths if "_TP" in p.name or "TP_" in p.name]
if sv_paths and tp_paths:
    expected = {}
    for sv_path in sv_paths:
        expected.update(parse_sv_voltages(sv_path))
    tn_to_container = {}
    for tp_path in tp_paths:
        tn_to_container.update(parse_tn_containers(tp_path))
    powsybl_buses = network.get_buses()
    tn_to_bus = match_powsybl_buses_to_tn(tn_to_container, expected, powsybl_buses)
    errors = []
    for tn_mrid, bus_id in tn_to_bus.items():
        exp_kv, _angle_deg = expected[tn_mrid]
        v_mag = powsybl_buses.loc[bus_id, "v_mag"]
        # NaN v_mag means this bus fell outside the main connected component
        # (connected_component_mode=MAIN never solves it) — not comparable.
        if math.isnan(v_mag):
            continue
        errors.append(abs(v_mag - exp_kv) / exp_kv)
    stats = deviation_stats(errors)
    if converged:
        print(f"deviation vs published SvVoltage: n={stats['n']} median={stats['median']:.4%} "
              f"p90={stats['p90']:.4%} max={stats['max']:.4%}")
    else:
        # Report the non-convergence in the accuracy line itself rather than a
        # percentage. The numbers are still shown, but only after the reason
        # they mean nothing, so they can't be lifted into a results table as
        # though they were a solve. See the `NOT CONVERGED` note above.
        print(f"deviation vs published SvVoltage: NOT CONVERGED ({warm_result.status_text}) — "
              f"no solved state to compare; the values below are pypowsybl's own unsolved "
              f"initial state, echoing the input SV profile back at itself: "
              f"n={stats['n']} median={stats['median']:.4%} p90={stats['p90']:.4%} max={stats['max']:.4%}")
else:
    print("deviation vs published SvVoltage: n/a (no SV/TP profile given)", file=sys.stderr)

tmp_dir.cleanup()
