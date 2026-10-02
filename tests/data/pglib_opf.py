"""Builds the OPF test fixtures from the pglib-opf cases in `benchmark-grids/`.

The Rust tests read the PGM network document and its `.opf.json` companion,
not MATLAB, so each case is converted with `python/gridoxide/matpower.py` into
a scratch directory before use. Nothing derived is committed, so there is no
copy to drift from the converter.

Writes, for each case in `CASES` plus `PIECEWISE`, `<name>.m`, `<name>.json`
and `<name>.opf.json` into the output directory:

    python3 tests/data/pglib_opf.py <out_dir>

Needs numpy (the converter's `.m` parser) but not the compiled extension: the
converter is loaded by path rather than through the `gridoxide` package.

Why pglib rather than `benchmark-grids/matpower/`: those power-flow cases
leave branches unrated (`case14`, `case118` and `case300` have `rateA = 0` on
every branch), so no flow limit ever binds and congestion goes untested. Every
pglib branch is rated. pglib's `case14_ieee` is the same network as
`matpower/case14` with deliberately different limits; they are not
interchangeable. The published DC and AC objectives the tests compare against
are in `benchmark-grids/pglib/BASELINE.md`.
"""

import importlib.util
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests" / "data" / "benchmark-grids" / "pglib"

CASES = [
    "pglib_opf_case3_lmbd",
    "pglib_opf_case5_pjm",
    "pglib_opf_case14_ieee",
    "pglib_opf_case30_ieee",
    "pglib_opf_case118_ieee",
]

# case5_pjm with its `gencost` rewritten from model 2 (polynomial) to model 1
# (piecewise-linear). The rewrite is exact: every generator in case5_pjm has
# c2 = 0, so three collinear points reproduce its cost precisely while still
# producing two segments, so the multi-segment path runs. The optimum must
# therefore equal the original case's, which makes it a test with a known
# answer rather than a comparison against another implementation.
PIECEWISE = "case5_pjm_pwl"
PIECEWISE_BASE = "pglib_opf_case5_pjm"


def _matpower():
    spec = importlib.util.spec_from_file_location(
        "gridoxide_matpower", ROOT / "python" / "gridoxide" / "matpower.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def piecewise_variant(text: str, mpc: dict) -> str:
    """Rewrites each linear polynomial cost as breakpoints at 0, Pmax/2, Pmax."""
    rows = []
    for gen, cost in zip(mpc["gen"], mpc["gencost"]):
        model, startup, shutdown, n_cost = cost[:4]
        c2, c1, c0 = cost[4:7]
        assert model == 2 and n_cost == 3 and c2 == 0.0, "rewrite is exact only for linear costs"
        p_max = gen[8]
        points = [(p, c1 * p + c0) for p in (0.0, p_max / 2, p_max)]
        rows.append(
            "\t".join(
                ["", "1", f"{startup}", f"{shutdown}", "3"]
                + [f"{v:f}" for point in points for v in point]
            )
            + ";"
        )
    gencost = "mpc.gencost = [\n" + "\n".join(rows) + "\n];"
    return re.sub(r"mpc\.gencost\s*=\s*\[.*?\];", lambda _: gencost, text, flags=re.DOTALL)


def generate(out_dir: Path) -> Path:
    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"{SOURCE} is missing; run `git submodule update --init tests/data/benchmark-grids`"
        )
    matpower = _matpower()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in CASES:
        shutil.copyfile(SOURCE / f"{name}.m", out_dir / f"{name}.m")

    base = out_dir / f"{PIECEWISE_BASE}.m"
    (out_dir / f"{PIECEWISE}.m").write_text(
        piecewise_variant(base.read_text(), matpower.parse_matpower_m(base))
    )

    for name in CASES + [PIECEWISE]:
        matpower.convert(out_dir / f"{name}.m", out_dir / f"{name}.json")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    generate(Path(sys.argv[1]))
