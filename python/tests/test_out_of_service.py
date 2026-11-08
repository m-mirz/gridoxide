"""`PowerFlowModel.with_out_of_service`: a CGMES contingency of any equipment, on ReliCapGrid.
Skipped without the cgmes feature or the tests/data/relicapgrid submodule."""

from pathlib import Path

import pytest

import gridoxide

ROOT = Path(__file__).resolve().parents[2] / "tests/data/relicapgrid/Instance"
LOAD = "ebf0cd8b-a357-4e97-aa67-de5b263733d6"   # Svedala's ConformLoad CT72_T1_LAST
SVEDALA_LINE = "f9afd596-8e07-433a-9256-0f12f71c1603"   # an ACLineSegment in service


def svedala():
    if not hasattr(gridoxide.PowerFlowModel, "from_cgmes"):
        pytest.skip("built without the cgmes feature")
    if not ROOT.exists():
        pytest.skip(f"{ROOT} not found; relicapgrid submodule not checked out")
    return gridoxide.PowerFlowModel.from_cgmes(
        [str(p) for area in ("Svedala", "commonData") for p in (ROOT / area / "Grid/cimxml").glob("*.xml")]
    )


def test_an_outage_leaves_the_model_as_it_was():
    model = svedala()
    model.solve()
    voltages = model.voltage_mag()
    outage = model.with_out_of_service([LOAD])
    outage.solve()
    assert outage.voltage_mag() != voltages
    model.solve()
    assert model.voltage_mag() == voltages


def test_the_scheduled_power_loses_the_load():
    model = svedala()
    (_, before, _), (_, after, _) = model.bus_spec(), model.with_out_of_service([LOAD]).bus_spec()
    moved = [a - b for a, b in zip(after, before) if abs(a - b) > 1e-9]
    assert len(moved) == 1 and moved[0] > 0


def test_an_mrid_with_or_without_its_underscore():
    model = svedala()
    a, b = model.with_out_of_service([LOAD]), model.with_out_of_service([f"_{LOAD}"])
    assert a.bus_spec() == b.bus_spec()


def test_unknown_equipment_is_refused_and_changes_nothing():
    model = svedala()
    before = model.bus_spec()
    with pytest.raises(ValueError, match="no equipment"):
        model.with_out_of_service([LOAD, "00000000-0000-0000-0000-000000000000"])
    assert model.with_out_of_service([]).bus_spec() == before


def test_it_needs_a_model_from_cgmes(tmp_path):
    if not hasattr(gridoxide.PowerFlowModel, "from_cgmes"):
        pytest.skip("built without the cgmes feature")
    case = tmp_path / "case.m"
    case.write_text(
        "function mpc = case2\nmpc.version = '2';\nmpc.baseMVA = 100;\n"
        "mpc.bus = [1 3 0 0 0 0 1 1 0 230 1 1.1 0.9; 2 1 10 0 0 0 1 1 0 230 1 1.1 0.9];\n"
        "mpc.gen = [1 10 0 100 -100 1 100 1 100 0];\n"
        "mpc.branch = [1 2 0.01 0.1 0 0 0 0 0 0 1 -360 360];\n"
    )
    with pytest.raises(RuntimeError, match="from_cgmes"):
        gridoxide.PowerFlowModel.from_matpower(str(case)).with_out_of_service([])


def test_terminal_flows_needs_a_solve():
    model = svedala()
    with pytest.raises(RuntimeError, match="solve"):
        model.terminal_flows()
    model.solve()
    flows = model.terminal_flows()
    assert flows and all(amps >= 0 for _, _, amps in flows.values())
    assert max(abs(p) for p, _, _ in flows.values()) > 100


def test_terminal_flows_lose_a_tripped_line():
    model = svedala()
    model.solve()
    terminals = len(model.terminal_flows())
    outage = model.with_out_of_service([SVEDALA_LINE])
    outage.solve()
    assert len(outage.terminal_flows()) == terminals - 2


def test_terminal_flows_refuses_a_solve_that_moved_taps():
    model = svedala()
    model.solve(control_taps=True)
    with pytest.raises(RuntimeError, match="control_taps"):
        model.terminal_flows()


def test_terminal_flows_on_a_model_not_from_cgmes(tmp_path):
    if not hasattr(gridoxide.PowerFlowModel, "terminal_flows"):
        pytest.skip("built without the cgmes feature")
    case = tmp_path / "case.m"
    case.write_text(
        "function mpc = case2\nmpc.version = '2';\nmpc.baseMVA = 100;\n"
        "mpc.bus = [1 3 0 0 0 0 1 1 0 230 1 1.1 0.9; 2 1 10 0 0 0 1 1 0 230 1 1.1 0.9];\n"
        "mpc.gen = [1 10 0 100 -100 1 100 1 100 0];\n"
        "mpc.branch = [1 2 0.01 0.1 0 0 0 0 0 0 1 -360 360];\n"
    )
    model = gridoxide.PowerFlowModel.from_matpower(str(case))
    model.solve()
    with pytest.raises(RuntimeError, match="from_cgmes"):
        model.terminal_flows()


def weights_on_first_generator(model):
    kinds, _, _ = model.bus_spec()
    weights = [0.0] * model.n_nodes
    weights[next(i for i, kind in enumerate(kinds) if kind == 1)] = 1.0
    return weights


def test_slack_weights_put_the_imbalance_where_they_say():
    model = svedala()
    weights = weights_on_first_generator(model)
    model.solve(slack_weights=weights)
    shift = model.slack_shift()
    assert [i for i, moved in enumerate(shift) if moved] == [weights.index(1.0)]


def test_slack_weights_are_one_per_bus_and_exclude_equal_sharing():
    model = svedala()
    with pytest.raises(ValueError, match="one per bus"):
        model.solve(slack_weights=[1.0])
    with pytest.raises(ValueError, match="give one"):
        model.solve(distribute_slack=True, slack_weights=weights_on_first_generator(model))
