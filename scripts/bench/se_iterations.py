"""Where the iterative-linear gap is: iteration count, or per-iteration work?

`bench_se_pgm.py` times the two tools. It cannot say *why* one is faster, and
§7 of `scripts/bench/README.md` used to answer that by inference — a time ratio
that stayed flat at 1.6-2.0x across an order of magnitude of problem size was
read as "a constant-factor gap in the per-iteration work, not an asymptotic
one". That inference needs the iteration counts to be comparable, which was
never measured. This measures them.

Neither tool reports an iteration count through its public API, so the count is
obtained the same way for both: **the smallest `max_iterations` budget that does
not fail**. One procedure, no internal APIs, and no asymmetry in what is being
counted. It is a bisection, so it costs about `log2(300)` solves per cell.

The two convergence criteria were checked to be the same quantity before
comparing anything, which is the control this whole comparison rests on:

  * power-grid-model's `iterate_unknown` (`iterative_linear_se_solver.hpp`)
    returns `max over buses of |u_new - u_old|`, phase-offset-normalized, and
    loops `while (max_dev > err_tol)`;
  * gridoxide's `se::iterative::estimate` computes `raw_step` as
    `.map(|(a, b)| (a - b).norm()).fold(0.0, f64::max)` over the same
    normalized voltages.

Same norm, same quantity, and both default to 1e-8. The one difference is that
gridoxide tests `raw_step * relaxation` where power-grid-model has no relaxation
at all and always takes the full step.

Run it against the documents `examples/bench_se.rs --emit` writes:

    cargo run --release --example bench_se -- case14 case118 case300 \
        case1354pegase case2869pegase --emit /tmp/se_bench
    VIRTUAL_ENV=$PWD/.venv-ls2g .venv-ls2g/bin/python3 -m maturin develop \
        --release --features python
    .venv-ls2g/bin/python3 scripts/bench/se_iterations.py /tmp/se_bench
"""

import json
import sys
import time
from pathlib import Path

from power_grid_model import (
    CalculationMethod,
    ComponentType,
    DatasetType,
    PowerGridModel,
    initialize_array,
)

import gridoxide

DEFAULT_CASES = ["case14", "case118", "case300", "case1354pegase", "case2869pegase"]

# A budget no case here should need. It only has to be an upper bound: the
# bisection below reports "diverge" if even this does not converge.
CEILING = 300
REPEATS = 5


def load(path):
    """Reads an emitted SE document into power-grid-model's array format."""
    raw = json.loads(path.read_text())["data"]
    # power-grid-model rejects a `voltage_regulator` carrying q_min/q_max as
    # experimental, exactly as `bench_se_pgm.py` works around.
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
    return raw, data


def smallest_budget(attempt):
    """Smallest k in [1, CEILING] for which `attempt(k)` succeeds, else None.

    Monotone by construction — a method converging within k iterations also
    converges within k+1 — so this bisects rather than scanning.
    """
    if not attempt(CEILING):
        return None
    lo, hi = 1, CEILING
    while lo < hi:
        mid = (lo + hi) // 2
        if attempt(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def best_ms(fn):
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return min(times)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/se_bench")
    cases = sys.argv[2:] or DEFAULT_CASES

    header = (
        f"{'case':<15}{'buses':>6}   {'PGM ms':>7}{'it':>4}{'ms/it':>8}   "
        f"{'gx ms':>7}{'it':>4}{'ms/it':>8}   {'time x':>7}{'iter x':>7}{'per-it x':>9}"
    )
    print(header)
    print("-" * len(header))

    for name in cases:
        path = root / f"{name}_se.json"
        if not path.exists():
            print(f"{name:<15} no SE document at {path} — run bench_se.rs --emit first")
            continue
        raw, data = load(path)
        try:
            model = PowerGridModel(data)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<15} power-grid-model rejected the document: {type(exc).__name__}")
            continue

        def pgm_attempt(k):
            try:
                model.calculate_state_estimation(
                    calculation_method=CalculationMethod.iterative_linear, max_iterations=k
                )
                return True
            except Exception:  # noqa: BLE001
                return False

        def gx_attempt(k):
            try:
                m = gridoxide.StateEstimationModel.from_pgm_json(
                    str(path), method="iterative_linear", max_iter=k
                )
                m.solve()
                return True
            except Exception:  # noqa: BLE001
                return False

        pgm_it = smallest_budget(pgm_attempt)
        gx_it = smallest_budget(gx_attempt)
        pgm_ms = best_ms(
            lambda: model.calculate_state_estimation(
                calculation_method=CalculationMethod.iterative_linear
            )
        )
        gx = gridoxide.StateEstimationModel.from_pgm_json(
            str(path), method="iterative_linear", max_iter=CEILING
        )
        gx_ms = best_ms(gx.solve)

        if pgm_it is None or gx_it is None:
            print(f"{name:<15}{len(raw['node']):>6}   "
                  f"{pgm_ms:>7.2f}{'div' if pgm_it is None else pgm_it:>4}{'':>8}   "
                  f"{gx_ms:>7.2f}{'div' if gx_it is None else gx_it:>4}")
            continue

        pgm_pi, gx_pi = pgm_ms / pgm_it, gx_ms / gx_it
        print(
            f"{name:<15}{len(raw['node']):>6}   "
            f"{pgm_ms:>7.2f}{pgm_it:>4}{pgm_pi:>8.3f}   "
            f"{gx_ms:>7.2f}{gx_it:>4}{gx_pi:>8.3f}   "
            f"{gx_ms / pgm_ms:>7.2f}{gx_it / pgm_it:>7.2f}{gx_pi / pgm_pi:>9.2f}"
        )

    print()
    print("Times are milliseconds, best of", REPEATS, "runs; `it` is the smallest")
    print("max_iterations that converges. `per-it x` below 1 means gridoxide's own")
    print("iterations are individually cheaper and the gap is iteration count.")


if __name__ == "__main__":
    main()
