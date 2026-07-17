#!/usr/bin/env python3
"""Runs gridoxide (scalar/block/klu backends) and lightsim2grid, head-to-head,
across all 12 real power-system test-case grids in cases.py, and prints/saves
one combined markdown timing table.

Usage: python3 run_case_suite.py [--python PYTHON] [--repeat N]
                                  [--cache-dir DIR] [--out FILE]

`--python` (default: the interpreter running this script) needs `numpy` and
`scipy` (for matpower_to_pgm.py) plus `pandapower` and `lightsim2grid` (for
bench_lightsim2grid.py, which needs a pandapower net directly since
lightsim2grid isn't fed PGM JSON).

gridoxide's side of this benchmark is converted straight from MATPOWER's
own `.m` case files (matpower_to_pgm.py), not through pandapower's own
MATPOWER importer — see that script's docstring for why: pandapower's
importer (via power-grid-model-io's PandaPowerConverter) introduced a real,
now-fixed-by-avoiding-it class of from-side/to-side base-mismatch bug for
transformers, and three of these twelve cases (case1888rte, case6495rte,
case6515rte) would not converge at all through that path.

Why no PGM column here (unlike the rest of scripts/bench/): this is a
deliberate scope split, not a technical limitation — PGM does support PV
buses (via its `voltage_regulator` component, the same mechanism gridoxide
now parses too, see matpower_to_pgm.py), but PGM's own team doesn't
benchmark against these particular IEEE/MATPOWER test cases either. This
track instead compares gridoxide head-to-head against lightsim2grid across
a realistic range of real-world grid sizes/sparsity patterns. For a
voltage+timing comparison against PGM on a synthetic PV-free network, see
generate_grid.py + bench_pgm.py in this same directory.
"""
import argparse
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from cases import CASE_NAMES, matpower_filename

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRIDOXIDE_BIN = REPO_ROOT / "target" / "release" / "examples" / "bench_network"
CONVERT_SCRIPT = Path(__file__).resolve().parent / "matpower_to_pgm.py"
LS2G_SCRIPT = Path(__file__).resolve().parent / "bench_lightsim2grid.py"
MATPOWER_RAW_URL = "https://raw.githubusercontent.com/m-mirz/matpower/master/data/{filename}"

MEAN_RE = re.compile(r"min=[\d.]+ms mean=([\d.]+)ms")
NODES_RE = re.compile(r"nodes=(\d+)")
NR_RE = re.compile(r"newton_raphson: [\d.]+ ms total, ([\d.]+) ms/run")


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


def convert_case(python: str, case_name: str, cache_dir: Path) -> tuple[Path | None, str | None]:
    out_path = cache_dir / f"{case_name}.json"
    if out_path.exists():
        return out_path, None
    m_path, fetch_err = fetch_matpower_case(case_name, cache_dir)
    if m_path is None:
        return None, fetch_err
    proc = subprocess.run(
        [python, str(CONVERT_SCRIPT), str(m_path), str(out_path)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0 or not out_path.exists():
        return None, (proc.stderr.strip().splitlines()[-1] if proc.stderr else "conversion failed")
    return out_path, None


def run_gridoxide(json_path: Path, backend: str, repeat: int) -> tuple[float | None, int | None, str | None]:
    if not GRIDOXIDE_BIN.exists():
        return None, None, "bench_network not built (cargo build --release --example bench_network --features klu)"
    proc = subprocess.run(
        [str(GRIDOXIDE_BIN), str(json_path), str(repeat), backend],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        if "only supports symmetric power flow with no PV buses" in proc.stderr:
            return None, None, "N/A (Block backend doesn't support PV buses)"
        return None, None, (proc.stderr.strip().splitlines()[-1] if proc.stderr else "failed")
    if "Failed to converge" in proc.stdout:
        return None, None, "did not converge in 20 iterations"
    nr_match = NR_RE.search(proc.stdout)
    nodes_match = NODES_RE.search(proc.stdout)
    if not nr_match:
        return None, None, "could not parse output"
    return float(nr_match.group(1)), (int(nodes_match.group(1)) if nodes_match else None), None


def run_lightsim2grid(python: str, case_name: str) -> tuple[float | None, str | None]:
    proc = subprocess.run(
        [python, str(LS2G_SCRIPT), case_name],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        return None, (proc.stderr.strip().splitlines()[-1] if proc.stderr else "failed")
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
    parser.add_argument("--repeat", type=int, default=10, help="gridoxide solve repeat count per backend")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / ".case-cache")
    parser.add_argument("--out", type=Path, default=None, help="also write the table to this file")
    args = parser.parse_args()

    args.cache_dir.mkdir(exist_ok=True)

    rows = []
    for case_name in CASE_NAMES:
        print(f"=== {case_name} ===", file=sys.stderr)
        json_path, conv_err = convert_case(args.python, case_name, args.cache_dir)
        if json_path is None:
            rows.append((case_name, "-", f"FAILED ({conv_err})", "-", "-", "-"))
            continue

        n_nodes = None
        backend_times = {}
        for backend in ("scalar", "block", "klu"):
            t, n, err = run_gridoxide(json_path, backend, args.repeat)
            n_nodes = n_nodes or n
            backend_times[backend] = fmt(t, err)

        ls2g_time, ls2g_err = run_lightsim2grid(args.python, case_name)

        rows.append((
            case_name,
            str(n_nodes) if n_nodes else "?",
            backend_times["scalar"],
            backend_times["block"],
            backend_times["klu"],
            fmt(ls2g_time, ls2g_err),
        ))

    header = ["case", "buses", "gridoxide scalar", "gridoxide block", "gridoxide klu", "lightsim2grid (KLU)"]
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
