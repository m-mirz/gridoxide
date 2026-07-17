#!/usr/bin/env python3
"""Runs gridoxide (scalar/block/klu backends), PGM, lightsim2grid, pypowsybl,
and pandapower head-to-head across all 12 real power-system test-case grids
in cases.py, and prints/saves one combined markdown timing table.

Usage: python3 run_case_suite.py [--python PYTHON] [--cache-dir DIR] [--out FILE]

`--python` (default: the interpreter running this script) needs the
`gridoxide` extension module built and installed (`maturin develop
--release --features python,klu` — see this directory's README), `numpy`
and `scipy` (matpower_to_pgm.py, bench_pypowsybl.py), `power-grid-model`
(bench_pgm.py), `pandapower` (bench_pandapower.py, and bench_lightsim2grid.py
since lightsim2grid needs a pandapower net directly), `lightsim2grid`, and
`pypowsybl`.

gridoxide's and PGM's side of this benchmark are converted straight from
MATPOWER's own `.m` case files (matpower_to_pgm.py), not through
pandapower's own MATPOWER importer — see that script's docstring for why:
pandapower's importer (via power-grid-model-io's PandaPowerConverter)
introduced a real, now-fixed-by-avoiding-it class of from-side/to-side
base-mismatch bug for transformers, and three of these twelve cases
(case1888rte, case6495rte, case6515rte) would not converge at all through
that path. lightsim2grid and pandapower still load via
`pandapower.networks.<case_name>()` directly (they need the pandapower net
object, not PGM JSON); pypowsybl loads the same MATPOWER `.m`/`.mat` file
gridoxide and PGM do, via its own MATPOWER importer (see
bench_pypowsybl.py).

gridoxide now models these cases' generators as genuine PV (voltage-
controlled) buses via PGM's `voltage_regulator` component
(`src/pgm.rs::pgm_to_buses_and_branches`, mirroring PGM's own
`newton_raphson_pf_solver.hpp::set_u_ref_and_bus_types`) — so PGM is a full
column here, not excluded on a PV-support technicality.

gridoxide's own timings come from `bench_gridoxide_native.py` — the
`gridoxide` Python extension module (`src/python.rs`, built via `maturin`),
not a subprocess call into a compiled `bench_network.rs` binary. Every tool
in this comparison is now driven the same way: a small Python script that
constructs one persistent model/solver object and times repeated `solve()`
calls on it with `time.perf_counter()`, printing `min=Xms mean=Yms`. This
also means gridoxide's numbers here are inherently *warm* (`PowerFlowModel`
wraps `solver::PersistentSolver` directly, reusing cached symbolic
factorization across all 5 timed calls) — not cosmetic: every other tool
here also reuses its own persistent model/solver object across its own
repeated timed calls (lightsim2grid's `ac_pf`, PGM's `calculate_power_flow`),
so a `cold` (fresh symbolic factorization every call) number wouldn't
actually be comparable to them — confirmed by profiling (`perf`) a
9,241-bus case, where symbolic factorization alone was responsible for most
of what first looked like a solver-speed gap against lightsim2grid.
"""
import argparse
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from cases import CASE_NAMES, matpower_filename

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
CONVERT_SCRIPT = SCRIPT_DIR / "matpower_to_pgm.py"
BENCH_GRIDOXIDE_SCRIPT = SCRIPT_DIR / "bench_gridoxide_native.py"
BENCH_PGM_SCRIPT = SCRIPT_DIR / "bench_pgm.py"
LS2G_SCRIPT = SCRIPT_DIR / "bench_lightsim2grid.py"
PYPOWSYBL_SCRIPT = SCRIPT_DIR / "bench_pypowsybl.py"
PANDAPOWER_SCRIPT = SCRIPT_DIR / "bench_pandapower.py"
MATPOWER_RAW_URL = "https://raw.githubusercontent.com/m-mirz/matpower/master/data/{filename}"

MEAN_RE = re.compile(r"min=[\d.]+ms mean=([\d.]+)ms")
NODES_RE = re.compile(r"nodes=(\d+)")


def fetch_matpower_case(case_name: str, cache_dir: Path) -> tuple[Path | None, str | None]:
    filename = matpower_filename(case_name)
    m_path = cache_dir / filename
    if m_path.exists():
        return m_path, None
    try:
        urllib.request.urlretrieve(MATPOWER_RAW_URL.format(filename=filename), m_path)
    except OSError as e:
        return None, f"download failed: {e}"
    return m_path, None


def convert_case(python: str, case_name: str, cache_dir: Path) -> tuple[Path | None, Path | None, str | None]:
    """Returns (matpower_path, pgm_json_path, error)."""
    m_path, fetch_err = fetch_matpower_case(case_name, cache_dir)
    if m_path is None:
        return None, None, fetch_err
    out_path = cache_dir / f"{case_name}.json"
    if out_path.exists():
        return m_path, out_path, None
    proc = subprocess.run(
        [python, str(CONVERT_SCRIPT), str(m_path), str(out_path)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0 or not out_path.exists():
        return m_path, None, (extract_error(proc.stderr) if proc.stderr else "conversion failed")
    return m_path, out_path, None


def run_gridoxide(python: str, json_path: Path, backend: str) -> tuple[float | None, int | None, str | None]:
    proc = subprocess.run(
        [python, str(BENCH_GRIDOXIDE_SCRIPT), str(json_path), backend],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        if "only supports symmetric power flow with no PV buses" in proc.stderr:
            return None, None, "N/A (Block backend doesn't support PV buses)"
        if "ModuleNotFoundError" in proc.stderr and "gridoxide" in proc.stderr:
            return None, None, "gridoxide extension module not built (see this directory's README)"
        return None, None, (extract_error(proc.stderr) if proc.stderr else "failed")
    mean_match = MEAN_RE.search(proc.stdout)
    nodes_match = NODES_RE.search(proc.stdout)
    if not mean_match:
        return None, None, "could not parse output"
    return float(mean_match.group(1)), (int(nodes_match.group(1)) if nodes_match else None), None


def extract_error(stderr: str) -> str:
    """The last line of a Python traceback is often a generic trailer
    shared across unrelated exceptions (e.g. PGM's own errors all end with
    the same "Try validate_input_data()..." hint) — prefer the actual
    "SomeError: message" line a few lines up, which is the one that
    actually distinguishes what went wrong."""
    if not stderr:
        return "failed"
    lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
    # PGM's own generic instructional trailer, shared across unrelated
    # exceptions (SparseMatrixError, IterationDiverge, ...) — never the
    # line that actually says what went wrong.
    lines = [l for l in lines if "validate_input_data" not in l and "validate_batch_data" not in l]
    for line in reversed(lines):
        if re.match(r"^[\w.]+:\s", line):
            return line
    return lines[-1] if lines else "failed"


def run_subprocess_mean(python: str, script: Path, args: list[str], timeout: int = 600) -> tuple[float | None, str | None]:
    proc = subprocess.run(
        [python, str(script), *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        return None, extract_error(proc.stderr)
    mean_match = MEAN_RE.search(proc.stdout)
    if not mean_match:
        return None, "could not parse output"
    return float(mean_match.group(1)), None


def fmt(value: float | None, err: str | None) -> str:
    if value is not None:
        return f"{value:.3f} ms"
    if err and err.startswith("N/A"):
        return err
    return f"FAILED ({err})" if err else "FAILED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / ".case-cache")
    parser.add_argument("--out", type=Path, default=None, help="also write the table to this file")
    args = parser.parse_args()

    args.cache_dir.mkdir(exist_ok=True)

    rows = []
    for case_name in CASE_NAMES:
        print(f"=== {case_name} ===", file=sys.stderr)
        m_path, json_path, conv_err = convert_case(args.python, case_name, args.cache_dir)
        if json_path is None:
            rows.append((case_name, "-", f"FAILED ({conv_err})", "-", "-", "-", "-", "-", "-"))
            continue

        n_nodes = None
        backend_times = {}
        for backend in ("scalar", "block", "klu"):
            t, n, err = run_gridoxide(args.python, json_path, backend)
            n_nodes = n_nodes or n
            backend_times[backend] = fmt(t, err)

        pgm_time, pgm_err = run_subprocess_mean(args.python, BENCH_PGM_SCRIPT, [str(json_path)])
        ls2g_time, ls2g_err = run_subprocess_mean(args.python, LS2G_SCRIPT, [case_name])
        pypowsybl_time, pypowsybl_err = run_subprocess_mean(args.python, PYPOWSYBL_SCRIPT, [case_name, str(m_path)])
        pandapower_time, pandapower_err = run_subprocess_mean(args.python, PANDAPOWER_SCRIPT, [case_name])

        rows.append((
            case_name,
            str(n_nodes) if n_nodes else "?",
            backend_times["scalar"],
            backend_times["block"],
            backend_times["klu"],
            fmt(pgm_time, pgm_err),
            fmt(ls2g_time, ls2g_err),
            fmt(pypowsybl_time, pypowsybl_err),
            fmt(pandapower_time, pandapower_err),
        ))

    header = ["case", "buses", "gridoxide scalar", "gridoxide block", "gridoxide klu",
              "PGM", "lightsim2grid (KLU)", "pypowsybl", "pandapower"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)

    print()
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
