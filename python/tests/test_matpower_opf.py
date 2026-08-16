"""The MATPOWER converter's OPF half, and a guard against committed drift.

`tests/data/pglib-opf/` holds both the upstream `.m` files and the documents
this converter produces from them. That is convenient for the Rust tests — they
read JSON, not MATLAB — but it means two representations of the same data live
side by side and can drift apart silently.

So the first test here regenerates every committed document and compares. If
the converter changes and the fixtures are not refreshed, this fails and says
which file.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from gridoxide.matpower import (  # noqa: E402
    COST_MODEL_PIECEWISE,
    COST_MODEL_POLYNOMIAL,
    convert,
    load_mpc,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "data" / "pglib-opf"
CASES = sorted(p.stem for p in FIXTURES.glob("*.m"))


def test_there_are_fixtures_to_check():
    assert CASES, f"no .m cases found under {FIXTURES}"


@pytest.mark.parametrize("case", CASES)
def test_committed_documents_match_a_fresh_conversion(case):
    """The drift guard. Regenerate and compare, byte-equivalent after parsing."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        convert(FIXTURES / f"{case}.m", tmp / f"{case}.json")

        for suffix in (".json", ".opf.json"):
            fresh = json.loads((tmp / f"{case}{suffix}").read_text())
            committed = json.loads((FIXTURES / f"{case}{suffix}").read_text())
            assert fresh == committed, (
                f"{case}{suffix} is stale — regenerate with "
                f"`python -m gridoxide.matpower {case}.m {case}.json`"
            )


@pytest.mark.parametrize("case", CASES)
def test_cost_curves_are_transcribed_faithfully(case):
    """Both MATPOWER cost models, checked against the raw `.m` rather than
    against the converter's own output — so a transcription that got dropped
    fails here rather than passing trivially.

    Model 2 (polynomial) stores coefficients highest-degree first and the
    document stores them ascending, so `coefficients[k]` multiplies `p**k`.
    Model 1 (piecewise-linear) stores a flat run of alternating `x, y`, which
    becomes a list of pairs.

    Both are covered rather than only the one every case happened to use:
    piecewise costs went unexercised for exactly that reason, and both OPF
    formulations silently ignored them as a result.
    """
    mpc = load_mpc(FIXTURES / f"{case}.m")
    gencost = np.atleast_2d(mpc["gencost"])
    gen = np.atleast_2d(mpc["gen"])
    opf = json.loads((FIXTURES / f"{case}.opf.json").read_text())

    by_index = {g["index"]: g for g in opf["generator"]}
    checked = 0
    for row in range(len(gen)):
        entry = by_index.get(row)
        if entry is None or "cost" not in entry or row >= len(gencost):
            continue
        model = int(gencost[row, 0])
        n_cost = int(gencost[row, 3])
        raw = [float(v) for v in gencost[row, 4 : 4 + n_cost * (1 if model == COST_MODEL_POLYNOMIAL else 2)]]

        if model == COST_MODEL_POLYNOMIAL:
            assert entry["cost"]["model"] == "polynomial"
            assert entry["cost"]["coefficients"] == list(reversed(raw)), (
                f"{case} generator {row}: raw {raw} should reverse to "
                f"{entry['cost']['coefficients']}"
            )
        else:
            assert entry["cost"]["model"] == "piecewise_linear"
            pairs = [[raw[i], raw[i + 1]] for i in range(0, len(raw), 2)]
            assert entry["cost"]["points"] == pairs, (
                f"{case} generator {row}: raw {raw} should pair up to "
                f"{entry['cost']['points']}"
            )
            # The breakpoints must ascend, or "the segment p sits on" is not
            # well defined and every downstream reading of the curve is
            # ambiguous.
            xs = [x for x, _ in pairs]
            assert xs == sorted(xs), f"{case} generator {row}: breakpoints not ascending"
        checked += 1
    assert checked > 0, f"{case}: no cost rows were actually compared"


@pytest.mark.parametrize("case", CASES)
def test_generator_limits_come_from_the_right_columns(case):
    """`Pmax`/`Pmin` are columns 8 and 9, `Qmax`/`Qmin` are 3 and 4 — adjacent
    to `Pg`/`Qg`, which are the *current* output rather than a limit, and easy
    to take by mistake."""
    mpc = load_mpc(FIXTURES / f"{case}.m")
    gen = np.atleast_2d(mpc["gen"])
    opf = json.loads((FIXTURES / f"{case}.opf.json").read_text())

    by_index = {g["index"]: g for g in opf["generator"]}
    for row in range(len(gen)):
        entry = by_index.get(row)
        if entry is None:
            continue
        assert entry["p_max"] == pytest.approx(float(gen[row, 8]))
        assert entry["p_min"] == pytest.approx(float(gen[row, 9]))
        assert entry["node"] == int(gen[row, 0])


@pytest.mark.parametrize("case", CASES)
def test_every_pglib_branch_carries_a_rating(case):
    """The property that makes these fixtures usable for OPF at all — see
    `tests/data/pglib-opf/README.md`."""
    opf = json.loads((FIXTURES / f"{case}.opf.json").read_text())
    unlimited = [b for b in opf["branch_limit"] if b["unlimited"]]
    assert not unlimited, f"{case}: {len(unlimited)} unrated branches"
    assert all(b["rate_a"] > 0.0 for b in opf["branch_limit"])


def test_a_case_without_ratings_is_marked_unlimited_not_dropped():
    """The contrast, on a case that genuinely has none.

    `benchmark-grids`' `case14` has `rateA = 0` on all 20 branches. That must
    come through as *unlimited*, not as a binding zero and not as a missing
    entry — a missing entry would be indistinguishable from a converter bug.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "tests" / "data" / "benchmark-grids" / "matpower" / "case14.m"
    )
    if not source.exists():
        pytest.skip("benchmark-grids submodule not initialized")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        convert(source, tmp / "case14.json")
        opf = json.loads((tmp / "case14.opf.json").read_text())

    assert len(opf["branch_limit"]) == 20
    assert all(b["unlimited"] for b in opf["branch_limit"])
    assert all(b["rate_a"] == 0.0 for b in opf["branch_limit"])


def test_piecewise_costs_are_read_as_point_pairs():
    """The converter's model-1 path on a minimal synthetic case.

    `case5_pjm_pwl.m` now covers it on a realistic network too, but this stays:
    it pins the raw `x, y, x, y` unpacking against a curve small enough to read
    at a glance, which is the part a realistic fixture makes harder to see.

    Worth noting what this test did *not* catch. The converter was correct all
    along — it was the two OPF formulations that ignored the curve it produced,
    leaving the generator free. Testing a converter against its own output
    format says nothing about whether anything downstream reads it.
    """
    case = """
function mpc = tiny
mpc.version = '2';
mpc.baseMVA = 100.0;
mpc.bus = [
\t1\t3\t0\t0\t0\t0\t1\t1.0\t0\t345\t1\t1.1\t0.9;
\t2\t1\t50\t10\t0\t0\t1\t1.0\t0\t345\t1\t1.1\t0.9;
];
mpc.gen = [
\t1\t50\t0\t100\t-100\t1.0\t100\t1\t200\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0;
];
mpc.branch = [
\t1\t2\t0.01\t0.1\t0\t150\t150\t150\t0\t0\t1\t-360\t360;
];
mpc.gencost = [
\t1\t0\t0\t3\t0\t0\t100\t1500\t200\t4000;
];
"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "tiny.m").write_text(case)
        convert(tmp / "tiny.m", tmp / "tiny.json")
        opf = json.loads((tmp / "tiny.opf.json").read_text())

    cost = opf["generator"][0]["cost"]
    assert cost["model"] == "piecewise_linear"
    assert cost["points"] == [[0.0, 0.0], [100.0, 1500.0], [200.0, 4000.0]]
    # And the branch is rated, so it is not silently marked unlimited.
    assert opf["branch_limit"][0]["rate_a"] == 150.0
    assert opf["branch_limit"][0]["unlimited"] is False
    assert COST_MODEL_PIECEWISE == 1 and COST_MODEL_POLYNOMIAL == 2
