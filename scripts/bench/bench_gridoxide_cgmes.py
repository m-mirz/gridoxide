#!/usr/bin/env python3
"""Runs gridoxide's CGMES import + solve directly via its native Python
bindings (`gridoxide.PowerFlowModel.from_cgmes`, `src/python.rs`) and times
repeated `solve()` calls with the `KluNative` backend by default — the CGMES
counterpart to `bench_gridoxide_native.py`'s own PGM-JSON benchmark, and the
gridoxide side of `bench_pypowsybl_cgmes.py`'s comparison.

Usage: python3 bench_gridoxide_cgmes.py <fixture_name> <profile.xml>...

Build the bindings first (needs both `python` and `cgmes` features):

    maturin develop --release --features python,cgmes

Also reports each solved bus's deviation from the fixture's own published
`SvVoltage` (parsed directly from the SV profile via `cgmes_sv.py`) — not a
tool-vs-tool comparison, an independent accuracy check against the
fixture's own reference solution, exactly like `assert_matches_sv`/
`assert_matches_sv_percentile` already do on the Rust test side
(`tests/cgmes_common/mod.rs`). `bus_index_for_mrid`/`voltage_kv` are the
Python-binding equivalents of that same Rust-side lookup.
"""
import sys
import time
from pathlib import Path

import gridoxide

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cgmes_sv import deviation_stats, parse_sv_voltages

fixture_name = sys.argv[1] if len(sys.argv) > 1 else "fixture"
profile_paths = sys.argv[2:]
if not profile_paths:
    print("usage: bench_gridoxide_cgmes.py <fixture_name> <profile.xml>...", file=sys.stderr)
    sys.exit(1)

backend = "klu_native"

print(f"[{fixture_name}] gridoxide ({backend})", file=sys.stderr)

t0 = time.perf_counter()
model = gridoxide.PowerFlowModel.from_cgmes(profile_paths, backend=backend)
t1 = time.perf_counter()
print(f"model construction: {(t1 - t0) * 1e3:.3f} ms", file=sys.stderr)

n_node = model.n_nodes

# Warm-up (first-call overhead, including first symbolic factorization),
# then timed runs — same methodology as bench_gridoxide_native.py.
model.solve()

times = []
for _ in range(5):
    t0 = time.perf_counter()
    model.solve()
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"nodes={n_node}")
print(f"solve (warm, 5 runs, ms): {[f'{t * 1e3:.3f}' for t in times]}")
print(f"min={min(times) * 1e3:.3f}ms mean={sum(times) / len(times) * 1e3:.3f}ms")

vm = model.voltage_mag()
print(f"voltage_mag min/max = {min(vm):.6f} / {max(vm):.6f}")

# Cold construction+solve, comparable to the other tools' own figure.
t0 = time.perf_counter()
model2 = gridoxide.PowerFlowModel.from_cgmes(profile_paths, backend=backend)
model2.solve()
t1 = time.perf_counter()
print(f"cold (construct+solve): {(t1 - t0) * 1e3:.3f} ms")

# Accuracy vs. the fixture's own published SvVoltage, if an SV profile was
# given among profile_paths.
sv_paths = [p for p in profile_paths if "_SV" in Path(p).name or "SV_" in Path(p).name]
if sv_paths:
    expected = {}
    for sv_path in sv_paths:
        expected.update(parse_sv_voltages(Path(sv_path)))
    voltage_kv = model.voltage_kv()
    errors = []
    for tn_mrid, (exp_kv, _angle_deg) in expected.items():
        idx = model.bus_index_for_mrid(tn_mrid)
        if idx is None:
            continue
        # A solved voltage_mag of exactly 0.0 is PowerFlowModel.solve()'s
        # fixed placeholder for a sourceless island (see its own doc
        # comment), not a real solution to compare against — skip it.
        if voltage_kv[idx] == 0.0:
            continue
        errors.append(abs(voltage_kv[idx] - exp_kv) / exp_kv)
    stats = deviation_stats(errors)
    print(f"deviation vs published SvVoltage: n={stats['n']} median={stats['median']:.4%} "
          f"p90={stats['p90']:.4%} max={stats['max']:.4%}")
else:
    print("deviation vs published SvVoltage: n/a (no SV profile given)", file=sys.stderr)
