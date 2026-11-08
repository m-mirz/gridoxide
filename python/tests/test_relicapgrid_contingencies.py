"""CO_UC1's execution: the contingencies of ReliCapGrid's CO files, read by cimoxide and solved.

A contingency's equipment goes out of service (contingentStatus outOfService, the only one the files
use). Each outage is checked to remove its own equipment and nothing else, against the same outage
written into an SSH through cimoxide, and the base network to survive it. Then every contingency
runs: on the IGM that names it, and on the CGM. Limits are test_relicapgrid_security_analysis's.
"""

import functools
from pathlib import Path

import pytest

import gridoxide
from _relicapgrid import (ROOT, bare_type, cimoxide, contingencies, espheim_at_the_voltage_its_sv_holds, grid_files,
                          norm, with_targets)
from test_relicapgrid_power_flow import case_files

CONTINGENCIES = [
    ("7e31c67d-67ba-4592-8ac1-9e806d697c8e", "d9622e7f-5bf0-4e7e-b766-b8596c6fe4ae", "ACLineSegment"),
    ("e9eab3fe-c328-4f78-9bc1-77adb59f6ba7", "ed0c5d75-4a54-43c8-b782-b20d7431630b", "ACLineSegment"),
    ("8cdec4c6-10c3-40c1-9eeb-7f6ae8d9b3fe", "b85e5fb8-7e2b-4264-a059-edab9a838116", "ACLineSegment"),
    ("e05bbe20-9d4a-40da-9777-8424d216785d", "f1c13f90-6d89-4a37-a51c-94742ad2dd72", "PowerTransformer"),
    ("62eac668-82a1-4739-8cfc-de929b78ef7e", "0489d903-c766-11e1-8775-005056c00008", "LinearShuntCompensator"),
    ("ab92defa-ef8e-4d13-a242-5b1a9fe956f6", "8464a151-ec39-4e7a-8321-06c41e63958c", "NonlinearShuntCompensator"),
    ("66627b36-aecd-4141-ab99-4bd8a74d18a8", "038ce404-30dc-4289-b9db-7076cb870b8e", "CsConverter"),
    ("504b48fb-de1f-4f7e-928f-8585daaf7b72", "be09fc02-de3f-49e4-aa84-94803bcc5d76", "DCLineSegment"),
    ("37997e71-cb7d-4a8c-baa6-2a1594956da9", "550ebe0d-f2b2-48c1-991f-cebea43a21aa", "SynchronousMachine"),
    ("f3a8c241-7b5e-4d92-b1c6-8e3f092a4d71", "ebf0cd8b-a357-4e97-aa67-de5b263733d6", "ConformLoad"),
    ("9d17b84c-33b5-4a68-b8b9-ed5b31038d40", "04566cf8-c766-11e1-8775-005056c00008", "ACLineSegment"),
    ("13334fdf-9cc2-4341-adb6-1281269040b4", "0475dbd8-c766-11e1-8775-005056c00008", "ACLineSegment"),
]


def same(a, b):
    return norm(a) == norm(b)


@functools.cache
def base(area):
    return gridoxide.PowerFlowModel.from_cgmes(case_files(area))


@functools.cache
def decoded(area):
    return {norm(k): v for k, v in cimoxide.CimDataset.decode_files(case_files(area)).entries().items()}


def ybus(model):
    rows, columns, g, b = model.ybus_triplets()
    return {(r, c): complex(x, y) for r, c, x, y in zip(rows, columns, g, b)}


def changed(before, after):
    """The buses whose admittance entries differ."""
    return {bus for key in before.keys() | after.keys()
            if abs(before.get(key, 0) - after.get(key, 0)) > 1e-12 for bus in key}


def buses_of(objects, model, equipment):
    """The buses *equipment*'s terminals are on."""
    nodes = {k: norm(v["ConnectivityNode.TopologicalNode"]) for k, v in objects.items()
             if "ConnectivityNode.TopologicalNode" in v}
    found = set()
    for terminal in objects.values():
        if bare_type(terminal["_type"]) == "Terminal" and same(terminal.get("Terminal.ConductingEquipment", ""),
                                                                       equipment):
            node = terminal.get("Terminal.TopologicalNode")
            node = norm(node) if node else nodes[norm(terminal["Terminal.ConnectivityNode"])]
            found.add(model.bus_index_for_mrid(f"_{node}"))
    return found


# The IGM each of CO_UC1's contingencies is on, by the file naming the equipment.
HOME = {"7e31c67d": "Belgovia", "e9eab3fe": "Belgovia", "8cdec4c6": "Espheim", "e05bbe20": "Svedala",
        "62eac668": "Espheim", "ab92defa": "Svedala", "66627b36": "DC-Espheim-Svedala",
        "504b48fb": "DC-Espheim-Svedala", "37997e71": "Belgovia", "f3a8c241": "Svedala",
        "9d17b84c": "Espheim", "13334fdf": "Espheim"}


def home(contingency):
    return HOME[contingency[:8]]


@pytest.mark.parametrize("contingency, equipment, cls", CONTINGENCIES, ids=[c[2] + "-" + c[1][:8] for c in CONTINGENCIES])
def test_an_outage_removes_its_equipment_and_nothing_else(contingency, equipment, cls):
    """What each class takes away: a branch from the admittances between its ends, a
    shunt from its bus's, an injection from the scheduled power, and an HVDC link's
    power from both its ends."""
    area = home(contingency)
    objects, intact = decoded(area), base(area)
    out = intact.with_out_of_service([equipment])
    (_, p, _), (_, p_out, _) = intact.bus_spec(), out.bus_spec()
    moved = {bus: after - before for bus, (before, after) in enumerate(zip(p, p_out)) if abs(after - before) > 1e-9}
    admittances = changed(ybus(intact), ybus(out))
    ends = buses_of(objects, intact, equipment)

    if cls in ("ACLineSegment", "PowerTransformer"):
        assert admittances == ends and len(ends) == 2 and not moved
    elif cls in ("LinearShuntCompensator", "NonlinearShuntCompensator"):
        assert admittances == ends and len(ends) == 1 and not moved
    elif cls in ("ConformLoad", "SynchronousMachine"):
        attribute = "EnergyConsumer.p" if cls == "ConformLoad" else "RotatingMachine.p"
        assert not admittances and moved.keys() == ends
        assert sum(moved.values()) == pytest.approx(float(objects[equipment][attribute]) / 100)
    else:  # CsConverter, DCLineSegment: the link's power, at both its ends
        converters = [v for v in objects.values() if bare_type(v["_type"]) == "CsConverter"]
        assert not admittances and len(moved) == 2
        assert sorted(moved.values()) == pytest.approx(sorted(float(c["ACDCConverter.p"]) / 100 for c in converters),
                                                       abs=1e-3)


@pytest.mark.parametrize("contingency, equipment, cls", CONTINGENCIES, ids=[c[2] + "-" + c[1][:8] for c in CONTINGENCIES])
def test_the_outage_equals_one_written_into_the_ssh(contingency, equipment, cls, tmp_path):
    """The same outage reached by another road: cimoxide sets Equipment.inService in
    the IGM's SSH, exports it, and gridoxide reads the files."""
    area = home(contingency)
    named = contingencies()[contingency][2]
    paths = []
    for path in case_files(area):
        if "_SSH_" in path:
            dataset = cimoxide.CimDataset.decode_file(path)
            for mrid, obj in dataset.entries().items():
                if norm(mrid) in named:
                    dataset[mrid] = dict(obj, **{"Equipment.inService": "false"})
            path = tmp_path / Path(path).name
            path.write_text(dataset.to_xml_for_profile("SSH"), encoding="utf-8")
            path = str(path)
        paths.append(path)
    limits = {"enforce_q_limits": area != "Espheim"}
    derived = base(area).with_out_of_service(named)
    derived.solve(**limits)
    edited = gridoxide.PowerFlowModel.from_cgmes(paths)
    edited.solve(**limits)
    assert edited.n_nodes == derived.n_nodes
    assert edited.voltage_mag() == pytest.approx(derived.voltage_mag(), abs=1e-9)
    assert edited.voltage_ang() == pytest.approx(derived.voltage_ang(), abs=1e-9)


def test_execution_leaves_the_base_network_alone():
    """The report's expected outcome: execution does not modify the base network."""
    model = gridoxide.PowerFlowModel.from_cgmes(case_files("Svedala"))
    model.solve(enforce_q_limits=True)
    voltages, angles = model.voltage_mag(), model.voltage_ang()
    admittances = ybus(model)
    for contingency, equipment, cls in CONTINGENCIES:
        if home(contingency) == "Svedala":
            outage = model.with_out_of_service([equipment])
            outage.solve(enforce_q_limits=True)
            assert outage.voltage_mag() != voltages
    model.solve(enforce_q_limits=True)
    assert model.voltage_mag() == voltages and model.voltage_ang() == angles
    assert ybus(model) == admittances


def test_equipment_no_model_holds_is_refused():
    with pytest.raises(ValueError, match="no equipment"):
        base("Svedala").with_out_of_service(["00000000-0000-0000-0000-000000000000"])


def run(area, names, **limits):
    """(solved, the names of those that stalled, equipment no file of *area* holds)."""
    model = base(area) if isinstance(area, str) else area
    solved, stalled, absent = 0, [], []
    for contingency in names:
        name, equipment = contingencies()[contingency][1:]
        try:
            outage = model.with_out_of_service(equipment)
        except ValueError:
            absent.append(name)
            continue
        try:
            outage.solve(**limits)
            solved += 1
        except RuntimeError:
            stalled.append(name)
    return solved, sorted(stalled), absent


def of_file(area):
    return [k for k, (file_area, _, _) in contingencies().items() if file_area == area]


# Espheim's contingencies that stall on its IGM, without limits as its base needs. All but
# three stall with its controls at the voltage its SV holds as well: that contradiction
# explains few of them. The base is low (0.77 p.u.) and they trip heavily loaded lines, but
# not converging does not prove there is no solution.
STALLED = sorted(["CO154_8_5", "CO48_23_24", "CO51_24_70", "CO54_26_30", "CO63_30_38", "CO75_38_37", "CO76_38_65",
                  "CO83_42_49", "CO84_42_49", "CO85_43_44", "CO86_44_45", "CO133_69_70", "CO153_8_30"])


@pytest.mark.parametrize("area, solved, absent", [
    ("Belgovia", 16, []),
    ("Britheim", 2, []),
    ("DC-Espheim-Svedala", 2, []),
    ("Galia", 11, ["CO_HVDC"]),   # its equipment is DC-Nordheim-Galia's
    ("Svedala", 88, []),
])
def test_every_contingency_of_an_igm_solves(area, solved, absent):
    """With reactive limits, on the IGM whose CO file names them."""
    assert run(area, of_file(area), enforce_q_limits=True) == (solved, [], absent)


def test_the_contingencies_between_two_igms_solve_on_both():
    model = gridoxide.PowerFlowModel.from_cgmes(grid_files("Belgovia", "Svedala"))
    assert run(model, of_file("Svedala-Belgovia"), enforce_q_limits=True) == (4, [], [])


def test_espheims_contingencies():
    """Without limits, as its base stalls with them (test_power_flow.KNOWN)."""
    assert run("Espheim", of_file("Espheim")) == (188, STALLED, [])


def test_espheims_contingencies_with_the_voltage_its_sv_holds(tmp_path):
    """With 231 kV its base solves with limits, and 172 contingencies with it."""
    held = espheim_at_the_voltage_its_sv_holds()
    model = gridoxide.PowerFlowModel.from_cgmes(with_targets(case_files("Espheim"), held, tmp_path))
    limited, stalled_limited, _ = run(model, of_file("Espheim"), enforce_q_limits=True)
    assert limited == 172
    unlimited, stalled, _ = run(model, of_file("Espheim"))
    assert unlimited == 191 and set(STALLED) - set(stalled) == {"CO133_69_70", "CO51_24_70", "CO85_43_44"}


def test_every_contingency_runs_on_the_cgm():
    """The CGM holds every equipment. Without limits, as it stalls with them."""
    assert run("Jotunheim", contingencies()) == (312, STALLED, [])


