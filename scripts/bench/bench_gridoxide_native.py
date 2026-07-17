#!/usr/bin/env python3
"""Runs gridoxide directly via its native Python bindings (the `gridoxide`
extension module, built with `maturin` from `src/python.rs`) on a PGM-format
JSON network, and times repeated `solve()` calls — the same methodology
`bench_pgm.py`/`bench_lightsim2grid.py`/`bench_pypowsybl.py`/
`bench_pandapower.py` already use, so the whole `scripts/bench/` comparison
can run without shelling out to a compiled `bench_network.rs` binary and
parsing its stdout at all.

Usage: python3 bench_gridoxide_native.py <input.json> [backend]

`backend` (default "scalar") selects `scalar`, `block`, or `klu` — the
`gridoxide` module must have been built with the matching Cargo feature
(`klu` needs `maturin develop --features python,klu`; see this directory's
README for the exact command).

Build the bindings first:

    pip install maturin
    maturin develop --release --features python        # scalar + block
    maturin develop --release --features python,klu     # + klu

`PowerFlowModel.solve()` reuses cached symbolic factorization across calls
on the same model (it wraps `solver::PersistentSolver` directly) — this
script's repeated-call timing is a *warm* number by construction, the same
way every other tool's `bench_*.py` script is warm (one persistent model
object, reused across the 5 timed calls). There is no separate cold/warm
mode here the way `bench_network.rs` has one, since a fresh
`PowerFlowModel` per call would just be reimplementing that binary's `cold`
mode in slower Python for no benefit.
"""
import sys
import time

import gridoxide

path = sys.argv[1] if len(sys.argv) > 1 else "grid_bench_input.json"
backend = sys.argv[2] if len(sys.argv) > 2 else "scalar"

t0 = time.perf_counter()
model = gridoxide.PowerFlowModel.from_pgm_json(path, backend=backend)
t1 = time.perf_counter()
print(f"model construction: {(t1 - t0) * 1e3:.3f} ms", file=sys.stderr)

n_node = model.n_nodes

# Warm-up (first-call overhead — including the first, uncached symbolic
# factorization), then timed runs.
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

# Also time full cold construction+calc, comparable to the other tools'
# "cold (construct+calc)" figure.
t0 = time.perf_counter()
model2 = gridoxide.PowerFlowModel.from_pgm_json(path, backend=backend)
model2.solve()
t1 = time.perf_counter()
print(f"cold (construct+calc): {(t1 - t0) * 1e3:.3f} ms")
