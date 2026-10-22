"""Smoke tests for the `gridoxide` Python extension module (src/python.rs).
Build it first: `maturin develop --release --features python` (from the
repo root; no `klu` needed — these tests only exercise `scalar`/`block`,
the two backends the published wheel ships).

Uses `tests/data/pgm/powerflow/symmetric/line/input.json`, one of this
project's own committed PGM test fixtures (same one the Rust integration
tests solve), with its known-correct converged voltages from the
neighboring `sym_output.json` — so this is a real correctness check, not
just an import/smoke check.
"""
import json
from pathlib import Path

import pytest

import gridoxide

FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "tests/data/pgm/powerflow/symmetric/line"
INPUT_JSON = FIXTURE_DIR / "input.json"

# Expected u_pu per node id, from sym_output.json — node ids sort to
# gridoxide's 0-based indices in the same order (1 -> 0, 2 -> 1, 5 -> 2).
EXPECTED_U_PU = [1.0000134893760249, 1.000207096167134, 1.0002750640565956]


@pytest.mark.parametrize("backend", ["scalar", "block"])
def test_solve_matches_reference_output(backend):
    model = gridoxide.PowerFlowModel.from_pgm_json(str(INPUT_JSON), backend=backend)
    # 3 physical nodes + 1 virtual slack bus gridoxide adds per active
    # `source` (src/pgm/mod.rs) — see EXPECTED_U_PU's comment for why the first
    # 3 entries still line up with the physical node ids.
    assert model.n_nodes == 4

    model.solve()
    vm = model.voltage_mag()
    assert len(vm) == 4
    for actual, expected in zip(vm[:3], EXPECTED_U_PU):
        assert actual == pytest.approx(expected, abs=1e-5)


def test_repeated_solve_reuses_persistent_solver():
    """A second solve() call on the same model (same topology, same
    values) must reproduce the first solve's result exactly — this is
    PersistentSolver's whole point (reusing cached symbolic
    factorization across calls shouldn't change the answer)."""
    model = gridoxide.PowerFlowModel.from_pgm_json(str(INPUT_JSON), backend="scalar")
    model.solve()
    first = model.voltage_mag()
    model.solve()
    second = model.voltage_mag()
    assert first == second


def test_reset_then_solve_still_converges():
    model = gridoxide.PowerFlowModel.from_pgm_json(str(INPUT_JSON), backend="scalar")
    model.solve()
    model.reset()
    model.solve()
    vm = model.voltage_mag()
    for actual, expected in zip(vm, EXPECTED_U_PU):
        assert actual == pytest.approx(expected, abs=1e-5)


def test_ideal_sources_drop_the_virtual_bus(tmp_path):
    """An ideal source holds its node at `u_ref` with no bus of its own, and a
    very stiff impedance source tends to it."""
    data = json.loads(INPUT_JSON.read_text())
    n_nodes = len(data["data"]["node"])
    ideal = gridoxide.PowerFlowModel.from_pgm_json(str(INPUT_JSON), tol=1e-10, ideal_sources=True)
    ideal.solve()
    assert len(ideal.voltage_mag()) == n_nodes
    for source in data["data"]["source"]:
        source["sk"] = 1e20
    stiff_input = tmp_path / "stiff.json"
    stiff_input.write_text(json.dumps(data))
    stiff = gridoxide.PowerFlowModel.from_pgm_json(str(stiff_input), tol=1e-10)
    stiff.solve()
    assert len(stiff.voltage_mag()) == n_nodes + len(data["data"]["source"])
    assert ideal.voltage_mag() == pytest.approx(stiff.voltage_mag()[:n_nodes], abs=1e-9)
    assert ideal.voltage_ang() == pytest.approx(stiff.voltage_ang()[:n_nodes], abs=1e-9)


def test_missing_file_raises():
    with pytest.raises(RuntimeError):
        gridoxide.PowerFlowModel.from_pgm_json("/nonexistent/path/does/not/exist.json")


def test_invalid_json_raises():
    with pytest.raises(ValueError):
        gridoxide.PowerFlowModel.from_pgm_json(str(Path(__file__)))  # this .py file isn't valid PGM JSON


def test_block_backend_matches_scalar_on_pv_bus(tmp_path):
    """JacobianBackend::Block handles PV buses by replacing their (absent)
    Q-mismatch row with a dummy `ΔVmag = 0` equation rather than reducing
    the system's dimension like Scalar does (crates/gridoxide-core/src/solver.rs's
    `build_jacobian_blocks`); the two formulations must converge to the
    same solution. The Rust-side equivalent is
    tests/block_jacobian_test.rs::block_backend_matches_scalar_backend_with_pv_bus
    — this checks the same property through the Python bindings. A real PV
    bus needs a sym_gen with a voltage_regulator pointing at it (PGM's
    actual PV mechanism — see src/pgm/mod.rs's PgmVoltageRegulator)."""
    data = json.loads(INPUT_JSON.read_text())
    data["data"]["sym_gen"] = [
        {"id": 20, "node": 2, "status": 1, "type": 0, "p_specified": 0.0, "q_specified": 0.0}
    ]
    data["data"]["voltage_regulator"] = [
        {"id": 21, "regulated_object": 20, "status": 1, "u_ref": 1.0}
    ]
    pv_input = tmp_path / "pv_input.json"
    pv_input.write_text(json.dumps(data))

    solved = {}
    for backend in ("scalar", "block"):
        model = gridoxide.PowerFlowModel.from_pgm_json(str(pv_input), backend=backend)
        model.solve()
        solved[backend] = (model.voltage_mag(), model.voltage_ang())

    # Node id 2 is the regulated (PV) bus, index 1 in sorted-id order — its
    # magnitude being pinned to the regulator's u_ref is what makes this a
    # PV bus at all, so assert it before comparing the two backends.
    assert solved["scalar"][0][1] == pytest.approx(1.0, abs=1e-12)

    mag_s, ang_s = solved["scalar"]
    mag_b, ang_b = solved["block"]
    assert mag_b == pytest.approx(mag_s, abs=1e-9)
    assert ang_b == pytest.approx(ang_s, abs=1e-9)


# A fixture a flat start cannot solve inside the default 20 iterations while
# both warm starts can (tests/power_flow_methods_test.rs,
# both_warm_starts_rescue_a_case_flat_start_cannot_solve), so each `init`
# visibly reaches the solver rather than being accepted and ignored.
DISTRIBUTION_CASE = INPUT_JSON.parent.parent / "distribution-case" / "input.json"


def test_init_picks_the_starting_point():
    model = gridoxide.PowerFlowModel.from_pgm_json(str(DISTRIBUTION_CASE))
    with pytest.raises(RuntimeError, match="did not converge"):
        model.solve(init="flat")

    model.solve()
    default = model.voltage_mag()
    model.solve(init="dc")
    dc = model.voltage_mag()
    model.solve(init="linear_impedance")
    assert model.voltage_mag() == default
    assert dc == pytest.approx(default, abs=1e-6)


def test_flat_start_solves_what_it_can():
    model = gridoxide.PowerFlowModel.from_pgm_json(str(INPUT_JSON))
    model.solve(init="flat")
    for actual, expected in zip(model.voltage_mag()[:3], EXPECTED_U_PU):
        assert actual == pytest.approx(expected, abs=1e-5)


def test_unknown_init_raises():
    model = gridoxide.PowerFlowModel.from_pgm_json(str(INPUT_JSON))
    with pytest.raises(ValueError, match="unknown init"):
        model.solve(init="warm")
