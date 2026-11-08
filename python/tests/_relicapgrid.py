"""What the ReliCapGrid tests share: the 2026 SV-IOP's dataset, `tests/data/relicapgrid`, read
through cimoxide as a reference independent of gridoxide's own use of cimoxide-model. They skip
without the submodule, the cgmes feature or `pip install cimoxide`, their one test dependency.
"""

import functools
from pathlib import Path

import pytest

import gridoxide

cimoxide = pytest.importorskip("cimoxide")
if not hasattr(gridoxide, "security_analysis"):
    pytest.skip("built without the cgmes feature", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2] / "tests/data/relicapgrid/Instance"
if not ROOT.exists():
    pytest.skip(f"{ROOT} not found; relicapgrid submodule not checked out", allow_module_level=True)

AREAS = ("Belgovia", "Britheim", "DC-Espheim-Svedala", "DC-Nordheim-Galia", "Espheim", "Galia",
         "Nordheim", "Portheim", "Svedala")
SAR = ROOT / "Jotunheim/NetworkCode/2D_SAR_1030Z_Jotunheim.xml"


def norm(value):
    """An id without its '#', 'urn:uuid:' or leading '_'."""
    value = str(value)
    if "#" in value:
        value = value.rsplit("#", 1)[1]
    return value.removeprefix("urn:uuid:").removeprefix("_")


def bare_type(type_name):
    return type_name.rsplit(":", 1)[-1]


def listed(directory, pattern="*.xml"):
    return sorted(str(p) for p in (ROOT / directory).glob(pattern))


def grid_files(*areas):
    """The areas' EQ, SSH, TP and SV, with the common data's BaseVoltages. Not the boundary:
    gridoxide gives a terminal off the loaded topology a bus only beside an EquivalentInjection."""
    return sorted(p for area in (*areas, "commonData") for p in listed(f"{area}/Grid/cimxml"))


def cgm_files():
    """The CGM as ReliCapGrid's CI assembles it (group cgmes-CGM-Jotunheim): every IGM's EQ and
    updated SSH, Jotunheim's topology and state, the boundary and the common data."""
    return (listed("Jotunheim/ChangeSet/cimxml")
            + listed("Jotunheim/Grid/cimxml", "*_SV_*") + listed("Jotunheim/Grid/cimxml", "*_TP_*")
            + [p for area in AREAS for p in listed(f"{area}/Grid/cimxml", "*_EQ_*")]
            + listed("boundaryData/Grid/cimxml") + listed("commonData/Grid/cimxml"))


def co_files():
    return sorted(str(p) for p in ROOT.glob("*/NetworkCode/cimxml/*_CO*.xml"))


@functools.cache
def objects_of(paths):
    """What cimoxide decodes from *paths* (a tuple), by normalised id."""
    return {norm(k): v for k, v in cimoxide.CimDataset.decode_files(list(paths)).entries().items()}


@functools.cache
def contingencies():
    """Contingency → (the file's area, its name, its equipment), from every CO file."""
    found = {}
    for path in co_files():
        objects = cimoxide.CimDataset.decode_file(path).entries()
        for mrid, obj in objects.items():
            if bare_type(obj["_type"]).endswith("Contingency"):
                elements = [e for e in objects.values() if bare_type(e["_type"]) == "ContingencyEquipment"
                            and norm(e["ContingencyElement.Contingency"]) == norm(mrid)]
                found[norm(mrid)] = (Path(path).stem.removesuffix("_CO"), obj["IdentifiedObject.name"],
                                     [norm(e["ContingencyEquipment.Equipment"]) for e in elements])
    return found


def with_targets(paths, targets, out):
    """*paths*, with the SSH's RegulatingControl.targetValue of each control in *targets* replaced."""
    written = []
    for path in paths:
        if "_SSH_" in path:
            dataset = cimoxide.CimDataset.decode_file(path)
            for mrid, control in dataset.entries().items():
                if norm(mrid) in targets:
                    dataset[mrid] = dict(control, **{"RegulatingControl.targetValue": str(targets[norm(mrid)])})
            path = out / Path(path).name
            path.write_text(dataset.to_xml_for_profile("SSH"), encoding="utf-8")
            path = str(path)
        written.append(path)
    return written


def espheim_at_the_voltage_its_sv_holds():
    """Its three controls whose SSH target (220 kV) contradicts the SV (231 kV) → 231."""
    objects = objects_of(tuple(grid_files("Espheim")))
    return {k: 231.0 for k, v in objects.items() if bare_type(v["_type"]) == "RegulatingControl"
            and v.get("IdentifiedObject.name") in ("TannrsCk SM", "RegContr_ExNetInj_1", "RegContr_ExNetInj_2")
            and float(v["RegulatingControl.targetValue"]) == 220}
