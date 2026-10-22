"""Smoke tests for `gridoxide.matpower` (needs the `matpower` extra — see
that module's docstring). Confirms the MATPOWER `.m` -> PGM JSON converter,
now packaged inside `gridoxide` itself rather than only living in
`scripts/bench/`, works end to end: parse a `.m` case, convert it, and feed
the result straight into `gridoxide.PowerFlowModel`.

`baseMVA = 1` here so the case's own per-unit base matches
`PowerFlowModel.from_pgm_json`'s default `s_base_va=1e6` (1 MVA) exactly —
sidesteps having to pass a matching `s_base_va` explicitly, which isn't
what's under test here.
"""
import json
from pathlib import Path

import pytest

import gridoxide
from gridoxide.matpower import convert

CASE_M = """
function mpc = test_case
mpc.version = '2';
mpc.baseMVA = 1;
mpc.bus = [
\t1\t3\t0\t0\t0\t0\t1\t1.0\t0\t1\t1\t1.1\t0.9;
\t2\t1\t0.05\t0.01\t0\t0\t1\t1.0\t0\t1\t1\t1.1\t0.9;
];
mpc.branch = [
\t1\t2\t0.02\t0.06\t0\t250\t250\t250\t0\t0\t1\t-360\t360;
];
"""


def test_convert_matpower_m_to_pgm_json(tmp_path):
    m_path = tmp_path / "test_case.m"
    m_path.write_text(CASE_M)
    out_path = tmp_path / "test_case.json"

    convert(m_path, out_path)

    data = json.loads(out_path.read_text())["data"]
    assert [n["id"] for n in data["node"]] == [1, 2]
    assert len(data["line"]) == 1
    assert len(data["source"]) == 1
    assert data["source"][0]["node"] == 1
    assert len(data["sym_load"]) == 1
    assert data["sym_load"][0]["node"] == 2
    assert data["sym_load"][0]["p_specified"] == pytest.approx(0.05e6)
    assert data["sym_load"][0]["q_specified"] == pytest.approx(0.01e6)


def test_convert_then_solve_converges(tmp_path):
    m_path = tmp_path / "test_case.m"
    m_path.write_text(CASE_M)
    out_path = tmp_path / "test_case.json"
    convert(m_path, out_path)

    model = gridoxide.PowerFlowModel.from_pgm_json(str(out_path))
    model.solve()
    vm = model.voltage_mag()
    # 2 physical nodes + 1 virtual slack bus gridoxide adds per active
    # `source` (src/pgm/mod.rs), appended after the physical nodes.
    assert len(vm) == 3
    assert vm[-1] == pytest.approx(1.0, abs=1e-6)  # virtual slack bus holds u_ref


def test_repeated_convert_calls_do_not_share_id_state(tmp_path):
    """Two `convert()` calls in the same process must each produce
    self-consistent output (this used to be backed by a shared
    module-level id counter in scripts/bench/matpower_to_pgm.py; the
    packaged version uses a fresh counter per call instead)."""
    m_path = tmp_path / "test_case.m"
    m_path.write_text(CASE_M)
    out_path_1 = tmp_path / "out1.json"
    out_path_2 = tmp_path / "out2.json"

    convert(m_path, out_path_1)
    convert(m_path, out_path_2)

    data1 = json.loads(out_path_1.read_text())["data"]
    data2 = json.loads(out_path_2.read_text())["data"]
    assert data1["source"][0]["id"] == data2["source"][0]["id"]


# `PowerFlowModel.from_matpower`: the native reader, no PGM conversion.

CASE14 = Path(__file__).resolve().parents[2] / "tests/data/benchmark-grids/matpower/case14.m"
CASE33BW = CASE14.with_name("case33bw.m")

needs_case14 = pytest.mark.skipif(not CASE14.exists(), reason="benchmark-grids submodule not checked out")


@needs_case14
def test_from_matpower_holds_the_case_setpoints():
    model = gridoxide.PowerFlowModel.from_matpower(str(CASE14), tol=1e-8)
    model.solve(init="flat")
    numbers = model.bus_numbers()
    assert numbers == list(range(1, 15))
    vm = dict(zip(numbers, model.voltage_mag()))
    va = dict(zip(numbers, model.voltage_ang()))
    # The generators' Vg, which case14's Vm column happens to round to.
    for bus, vg in {1: 1.06, 2: 1.045, 3: 1.01, 6: 1.07, 8: 1.09}.items():
        assert vm[bus] == pytest.approx(vg, abs=1e-12)
    assert va[1] == 0.0
    # MATPOWER's runpf answer for bus 14.
    assert vm[14] == pytest.approx(1.0355, abs=1e-4)


@needs_case14
def test_from_matpower_refuses_a_case_that_edits_its_matrices():
    with pytest.raises(ValueError, match="modified by MATLAB code"):
        gridoxide.PowerFlowModel.from_matpower(str(CASE33BW))


def test_from_matpower_reads_a_written_case(tmp_path):
    path = tmp_path / "test_case.m"
    path.write_text(CASE_M.replace("mpc.branch", "mpc.gen = [\n\t1\t0\t0\t0\t0\t1.0\t1\t1;\n];\nmpc.branch"))
    model = gridoxide.PowerFlowModel.from_matpower(str(path))
    model.solve()
    assert model.bus_numbers() == [1, 2]
    assert model.voltage_mag()[1] < 1.0


def test_from_matpower_without_a_file_raises():
    with pytest.raises(RuntimeError, match="reading"):
        gridoxide.PowerFlowModel.from_matpower("/nonexistent/case.m")
