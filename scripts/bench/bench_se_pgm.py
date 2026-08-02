"""State estimation: gridoxide against power-grid-model, same measurements.

`examples/bench_se.rs` times gridoxide's two methods against each other. This
times both of them against power-grid-model's, which is the comparison that says
whether the numbers there are any good.

The measurement set is generated once by `examples/bench_se.rs --emit` and
written into an augmented input document that *both* tools read, so neither gets
a set tuned to it and both estimate from byte-identical sensors:

  * a `sym_voltage_sensor` on every node, reading that node's solved voltage;
  * a `sym_power_sensor` on both ends of every line and transformer, reading
    that terminal's solved flow.

The values come from gridoxide's power flow rather than power-grid-model's for a
blunt reason: power-grid-model's power flow does not converge on any of these
converted MATPOWER cases, with or without the voltage regulator — the deviation
*grows* to 1394 after 200 iterations on case300. That is a finding about the
converted documents worth chasing separately. It does not tilt the comparison
here, because both tools estimate from the same numbers; only their provenance
differs, and a converged power flow is a converged power flow.

The data is therefore perfectly consistent — there is no noise, so the objective
collapses and the residuals carry nothing. That is the right shape for timing
(cost follows measurement count and sparsity, not how well the data agrees) and
the wrong shape for judging robustness. Bad-data behaviour needs a different
harness.

Build the bindings first:

    VIRTUAL_ENV=$PWD/.venv-ls2g .venv-ls2g/bin/maturin develop --release --features python

Then:

    .venv-ls2g/bin/python scripts/bench/bench_se_pgm.py [case ...]
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from power_grid_model import (
    CalculationMethod,
    ComponentType,
    DatasetType,
    PowerGridModel,
    initialize_array,
)

import gridoxide

CACHE = Path(__file__).resolve().parent / ".case-cache"
DEFAULT_CASES = ["case14", "case118", "case300", "case1354pegase", "case2869pegase"]

# Sensor accuracies in SI, matching examples/bench_se.rs's per-unit choices at a
# 1 MVA base: voltage transducers are the best-trusted instrument on a
# substation, power measurements much less so.
U_SIGMA_REL = 1e-3
S_SIGMA_VA = 1e-2 * 1e6

REPEATS = 5


def load_se_document(path):
    """Reads an emitted SE document into power-grid-model's array format."""
    raw = json.loads(path.read_text())["data"]
    # power-grid-model rejects a `voltage_regulator` carrying q_min/q_max as
    # experimental, and `matpower_to_pgm.py` emits them. Only those fields are
    # stripped; the component itself stays, so both tools see the same network.
    for entry in raw.get("voltage_regulator", []):
        entry.pop("q_min", None)
        entry.pop("q_max", None)

    data = {}
    for component, entries in raw.items():
        if not entries:
            continue
        arr = initialize_array(DatasetType.input, ComponentType[component], len(entries))
        for i, e in enumerate(entries):
            for field, value in e.items():
                if field in arr.dtype.names:
                    arr[field][i] = value
        data[ComponentType[component]] = arr
    n_meas = 2 * len(raw.get("sym_power_sensor", [])) + len(raw.get("sym_voltage_sensor", []))
    return raw, data, n_meas


def timed(fn):
    """Best of REPEATS, to take scheduler noise out of the comparison."""
    times = []
    result = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1e3)
    return min(times), statistics.median(times), result


def main():
    cases = sys.argv[1:] or DEFAULT_CASES
    tmp = Path("/tmp/claude-1000/se_bench")
    tmp.mkdir(parents=True, exist_ok=True)

    header = f"{'case':<16}{'buses':>7}{'meas':>8}  {'PGM nr':>9}{'PGM il':>9}  {'gx nr':>9}{'gx il':>9}   {'agreement':>10}"
    print(header)
    print("-" * len(header))

    for name in cases:
        path = tmp / f"{name}_se.json"
        if not path.exists():
            print(f"{name:<16} no SE document — run:")
            print(f"{'':<16}   cargo run --release --example bench_se -- {name} --emit {tmp}")
            continue
        raw, augmented, n_meas = load_se_document(path)
        try:
            model = PowerGridModel(augmented)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<16} power-grid-model rejected the document: {str(exc)[:60]}")
            continue

        results = {}
        for label, method in (("nr", CalculationMethod.newton_raphson),
                              ("il", CalculationMethod.iterative_linear)):
            try:
                best, _, out = timed(
                    lambda m=method: model.calculate_state_estimation(calculation_method=m)
                )
                results[f"pgm_{label}"] = best
                results[f"pgm_{label}_u"] = out[ComponentType.node]["u"].copy()
            except Exception as exc:  # noqa: BLE001
                results[f"pgm_{label}"] = None
                results[f"pgm_{label}_err"] = type(exc).__name__

        for label, method in (("nr", "newton_raphson"), ("il", "iterative_linear")):
            try:
                gx = gridoxide.StateEstimationModel.from_pgm_json(
                    str(path), method=method, max_iter=100 if label == "il" else 20
                )
                best, _, _ = timed(gx.solve)
                results[f"gx_{label}"] = best
                u_rated = np.array([n["u_rated"] for n in raw["node"]])
                results[f"gx_{label}_u"] = np.array(gx.voltage_mag()[: len(u_rated)]) * u_rated
            except Exception as exc:  # noqa: BLE001
                results[f"gx_{label}"] = None
                results[f"gx_{label}_err"] = type(exc).__name__

        def fmt(key):
            v = results.get(key)
            return f"{v:9.1f}" if v is not None else f"{results.get(key + '_err', 'fail')[:9]:>9}"

        agreement = "-"
        if results.get("pgm_nr_u") is not None and results.get("gx_nr_u") is not None:
            worst = float(np.max(np.abs(results["pgm_nr_u"] - results["gx_nr_u"])))
            agreement = f"{worst:.2e} V"

        print(
            f"{name:<16}{len(raw['node']):>7}{n_meas:>8}  "
            f"{fmt('pgm_nr')}{fmt('pgm_il')}  {fmt('gx_nr')}{fmt('gx_il')}   {agreement:>10}"
        )

    print("\nTimes are milliseconds, best of", REPEATS, "runs. Measurements are scalar rows")
    print("(a power sensor contributes two), so both tools see the same count.")
    print()
    print("Caveat on the gridoxide column: `estimate()` builds its symbolic")
    print("factorization fresh on every call — `cache` starts as None in")
    print("`se::nr::estimate_with` — so these numbers include setup that power flow")
    print("amortizes away via `PersistentSolver`. State estimation has no equivalent")
    print("yet, and that is likely a fair part of the gap at scale.")


if __name__ == "__main__":
    main()
