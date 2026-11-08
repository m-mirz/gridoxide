"""`PowerFlowModel.with_out_of_service`: a CGMES contingency of any equipment, on ReliCapGrid.
Skipped without the cgmes feature or the tests/data/relicapgrid submodule."""

from pathlib import Path

import pytest

import gridoxide

ROOT = Path(__file__).resolve().parents[2] / "tests/data/relicapgrid/Instance"
LOAD = "ebf0cd8b-a357-4e97-aa67-de5b263733d6"   # Svedala's ConformLoad CT72_T1_LAST


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
