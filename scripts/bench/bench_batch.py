#!/usr/bin/env python3
"""Measures gridoxide's batched power flow scaling across CPU cores.

This is the baseline any future GPU work has to beat. `plans/GPU_PLAN.md` is
explicit about why it exists: a single solve is ~60% sparse LU, which caps a
perfect GPU port of everything else at ~1.65x before PCIe latency eats it, so
the GPU target is *batches* (N-1 screening, time series, Monte Carlo). Quoting
a GPU speedup against a single-threaded CPU solver would be meaningless —
this script produces the number that isn't.

Usage: python3 bench_batch.py <input.json> [backend] [n_scenarios]

`backend` (default "klu_native") selects `scalar`, `block`, `klu`,
`klu_native`, or `pardiso`, exactly as `bench_gridoxide_native.py` does. Build
the bindings first:

    maturin develop --release --features python,klu

Scenarios are `n_scenarios` uniform load scalings drawn from a fixed seed, so
runs are comparable across invocations and machines. That is the time-series /
QSTS shape: one topology, one sparsity pattern, only injections varying — the
case `batch::BatchSolver` is built for, where each worker amortizes a single
symbolic factorization across its whole share of the batch.

Timing methodology matches every other `bench_*.py` here: one warm-up call
(which pays first-touch and the first symbolic factorization), then timed
repeats, reporting the minimum.
"""
import os
import random
import sys
import time

import gridoxide


def physical_cores():
    """Physical cores, not SMT siblings. Sparse LU is memory-bound, so SMT
    buys nearly nothing here and reporting speedup against a logical-core
    count would overstate the headroom a GPU has to beat."""
    try:
        ids = set()
        with open("/proc/cpuinfo") as f:
            core, pkg = None, None
            for line in f:
                if line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif line.startswith("physical id"):
                    pkg = line.split(":")[1].strip()
                elif not line.strip() and core is not None:
                    ids.add((pkg, core))
                    core, pkg = None, None
            if core is not None:
                ids.add((pkg, core))
        if ids:
            return len(ids)
    except OSError:
        pass
    return os.cpu_count() or 1

path = sys.argv[1] if len(sys.argv) > 1 else "grid_bench_input.json"
backend = sys.argv[2] if len(sys.argv) > 2 else "klu_native"
n_scenarios = int(sys.argv[3]) if len(sys.argv) > 3 else 256

REPEATS = 3

t0 = time.perf_counter()
model = gridoxide.PowerFlowModel.from_pgm_json(path, backend=backend)
t1 = time.perf_counter()
print(f"model construction: {(t1 - t0) * 1e3:.3f} ms", file=sys.stderr)

# +/-20% load scalings. Fixed seed so the workload is identical across runs.
rng = random.Random(20260726)
scales = [rng.uniform(0.8, 1.2) for _ in range(n_scenarios)]

default_threads = gridoxide.PowerFlowModel.default_threads()
n_physical = physical_cores()
thread_counts = [t for t in (1, 2, 4, 8, 16, 32) if t <= default_threads]
if default_threads not in thread_counts:
    thread_counts.append(default_threads)

print(f"nodes={model.n_nodes}  backend={backend}  scenarios={n_scenarios}")
print(f"cores: {n_physical} physical, {default_threads} logical (rayon default)")
print()

results = {}
reference = None

for threads in thread_counts:
    # Warm-up also builds (and then caches) this thread count's rayon pool,
    # so the timed runs measure solving, not pool construction.
    out = model.solve_batch_scaled(scales, threads=threads)

    if not all(r.converged for r in out):
        n_bad = sum(1 for r in out if not r.converged)
        print(f"WARNING: {n_bad}/{n_scenarios} scenarios did not converge", file=sys.stderr)

    if reference is None:
        reference = [list(r.voltage_mag) for r in out]
    else:
        worst = max(
            abs(a - b)
            for r, ref in zip(out, reference)
            for a, b in zip(r.voltage_mag, ref)
        )
        assert worst < 1e-12, f"threads={threads} disagrees with threads=1 by {worst:.3e}"

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        model.solve_batch_scaled(scales, threads=threads)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results[threads] = min(times)

base = results[thread_counts[0]]
print(f"{'threads':>8} {'batch (ms)':>12} {'per solve (ms)':>15} {'solves/s':>10} {'speedup':>8}   ")
for threads in thread_counts:
    t = results[threads]
    note = "  (SMT, not extra cores)" if threads > n_physical else ""
    print(
        f"{threads:>8} {t * 1e3:>12.2f} {t / n_scenarios * 1e3:>15.4f} "
        f"{n_scenarios / t:>10.1f} {base / t:>7.2f}x{note}"
    )

print()
print(f"voltages agree across all thread counts to < 1e-12 ({n_scenarios} scenarios)")
best = min(results, key=results.get)
print(f"best: {best} threads, {n_scenarios / results[best]:.1f} solves/s "
      f"({base / results[best]:.2f}x over 1 thread)")
if n_physical in results:
    eff = (base / results[n_physical]) / n_physical
    print(f"parallel efficiency at {n_physical} physical cores: {eff * 100:.0f}%")
    print(
        "Sub-linear scaling here is expected and is not a batch-solver defect: sparse LU "
        "is memory-bound, so all-core clock throttling and shared-L3/memory-bandwidth "
        "contention both bite. To confirm on any given machine, run N single-threaded "
        "processes concurrently -- separate address spaces share no allocator or locks, "
        "so if processes scale no better than threads, the ceiling is hardware."
    )
