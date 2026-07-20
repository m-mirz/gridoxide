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
    # `source` (src/pgm.rs), appended after the physical nodes.
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
