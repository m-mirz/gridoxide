"""PF_UC1 (SV-IOP 2026 report §1.1, Annex 2): every case of the report's Annex 2, solved.

Convergence is not taken as the result. An IGM or the CGM has to stand up against the state it
publishes (SvVoltage, read through cimoxide). A combination has no state of its own, each IGM's SV
describing that IGM alone, so it has to hold one angle reference per component and voltages in
range. Problems are pinned in KNOWN, each with its cause. See docs/src/cgmes/sv_iop_2026.md.
"""

import collections
import functools
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

import gridoxide
from _relicapgrid import (AREAS, ROOT, bare_type, cgm_files, cimoxide, espheim_at_the_voltage_its_sv_holds, grid_files,
                          norm, with_targets)

VOLTAGE_TOLERANCE = 1.0    # % of the published voltage
ANGLE_TOLERANCE = 0.5      # degrees, spread of solved - published within a component
VOLTAGE_RANGE = (0.8, 1.2)  # p.u. of nominal

IGMS = {
    "Belgovia": ("Belgovia",),
    "Britheim": ("Britheim",),
    "DC-Espheim-Svedala": ("DC-Espheim-Svedala",),
    "DC-Nordheim-Galia": ("DC-Nordheim-Galia",),
    "Espheim": ("Espheim",),
    "Galia": ("Galia",),
    "Nordheim": ("Nordheim",),
    "Portheim": ("Portheim",),
    "Svedala": ("Svedala",),
}
COMBINATIONS = {
    "Nordheim + DC-Nordheim-Galia": ("Nordheim", "DC-Nordheim-Galia"),
    "Espheim + DC-Espheim-Svedala": ("Espheim", "DC-Espheim-Svedala"),
    "Svedala + DC-Espheim-Svedala": ("Svedala", "DC-Espheim-Svedala"),
    "All except Espheim": ("Belgovia", "Britheim", "DC-Espheim-Svedala", "DC-Nordheim-Galia", "Galia",
                           "Nordheim", "Portheim", "Svedala"),
    "Combination 1": ("Belgovia", "Svedala"),
    "Combination 2": ("Belgovia", "Svedala", "Galia"),
    "Combination 3": ("Belgovia", "Svedala", "Galia", "Espheim", "Britheim"),
}
CGM = "Jotunheim"


def case_files(case):
    if case == CGM:
        return cgm_files()
    return grid_files(*{**IGMS, **COMBINATIONS}[case])


def published(paths):
    """TopologicalNode → (name, kV, degrees) of each SvVoltage, as cimoxide decodes them."""
    objects = {norm(k): v for k, v in cimoxide.CimDataset.decode_files(paths).entries().items()}
    state = {}
    for obj in objects.values():
        if bare_type(obj["_type"]) == "SvVoltage":
            node = norm(obj["SvVoltage.TopologicalNode"])
            state[node] = (objects.get(node, {}).get("IdentifiedObject.name", node),
                           float(obj["SvVoltage.v"]), float(obj["SvVoltage.angle"]))
    return state


@dataclass
class Solution:
    model: object
    state: dict
    problems: list


def compare(model, state, against_sv=True):
    """What is wrong with a converged *model*: its references and voltages, and if
    *against_sv*, how far it is from the published *state*."""
    problems = []
    magnitude, angle, kv = model.voltage_mag(), model.voltage_ang(), model.voltage_kv()
    on_bus = collections.defaultdict(list)
    for node, (name, v, _deg) in state.items():
        bus = model.bus_index_for_mrid(f"_{node}")
        if bus is None:
            problems.append(f"{name}: has a SvVoltage but no bus")
        elif v and not magnitude[bus]:
            problems.append(f"{name}: published at {v:g} kV, solved de-energised")
        elif v:
            on_bus[bus].append(node)
    if not on_bus:
        problems.append("no published voltage to compare with")
    for island in model.islands():
        component = island["buses"]
        if island["status"] == "no_reference":
            continue
        if island["status"] != "converged":
            # Each reference held to its own published angle: the voltages say nothing.
            problems.append(f"{island['status']} in a component of {len(component)} buses")
            continue
        off = [(100 * abs(kv[bus] - state[node][1]) / state[node][1], state[node][0], state[node][1], kv[bus])
               for bus in component for node in on_bus[bus]]
        if off and against_sv:
            pct, name, want, got = max(off, key=lambda o: (round(o[0], 6), o[1]))
            if pct > VOLTAGE_TOLERANCE:
                problems.append(f"{name}: solved {got:.1f} kV, published {want:.1f} kV")
        turned = {bus: math.degrees(angle[bus]) - state[on_bus[bus][0]][2] for bus in component if on_bus[bus]}
        if against_sv and turned and (spread := max(turned.values()) - min(turned.values())) > ANGLE_TOLERANCE:
            problems.append(f"{state[on_bus[max(turned, key=turned.get)][0]][0]}: angle {spread:.0f} degrees "
                            "off the published angles")
        outside = [bus for bus in component if on_bus[bus]
                   and not VOLTAGE_RANGE[0] <= magnitude[bus] <= VOLTAGE_RANGE[1]]
        if outside:
            problems.append(f"{len(outside)} buses out of range, {magnitude[outside[0]]:.2f} p.u. of nominal")
    return sorted(problems)


def solve(paths, against_sv=True, **options):
    """Solve *paths*, with reactive limits unless told otherwise; a Solution whose model
    is None if gridoxide refuses or does not converge."""
    state = published(paths)
    try:
        model = gridoxide.PowerFlowModel.from_cgmes(paths)
        model.solve(**{"enforce_q_limits": True, **options})
    except RuntimeError as error:
        return Solution(None, state, [str(error).split(" (component", 1)[0]])
    return Solution(model, state, compare(model, state, against_sv))


@functools.cache
def case_solution(case):
    return solve(case_files(case), against_sv=case not in COMBINATIONS)


STALLED = "power flow did not converge within 20 iterations"
ESPHEIM_WHY = ("Espheim's SSH has TannrsCk SM and the two ExternalNetworkInjections hold CONNECTIVITY_NODE527 at "
               "220 kV; its SV has them holding it at 231 kV (1.05 times that), within their reactive limits. With "
               "the SSH's target the machines' limits cannot all hold: Newton stalls once they switch "
               "(test_espheim_solves_with_the_target_its_sv_holds). The report: only VeraGrid solves Espheim.")
OPPOSITE_SIGNS = ["BP HVDC GA-NH NH side: angle 2 degrees off the published angles",
                  "TAC Ap: angle 7 degrees off the published angles"]
PORTHEIM = ["2 buses out of range, 1.75 p.u. of nominal"]

# Problems by case: (why, problems). All of these are in the data as supplied.
KNOWN = {
    "DC-Nordheim-Galia": ("A converter's targetPpcc is +70 where its ACDCConverter.p, and the SV, are -70.01; Britheim "
                          "writes both with one sign. gridoxide takes targetPpcc as load-sign, like Britheim: the "
                          "two IGMs disagree on the convention (the report's §4.1 HVDC power convention).",
                          OPPOSITE_SIGNS),
    "Espheim": (ESPHEIM_WHY, [STALLED]),
    "Portheim": ("The case's own SV has every node at 700 kV on a 400 kV base; gridoxide's two buses (eight "
                 "TopologicalNodes joined by closed switches) reproduce it from the angle reference. The report "
                 "notes too much imbalance on Portheim.", PORTHEIM),
    "Espheim + DC-Espheim-Svedala": (ESPHEIM_WHY, [STALLED]),
    "All except Espheim": ("Portheim, as alone.", PORTHEIM),
    "Combination 3": (ESPHEIM_WHY, [STALLED]),
    CGM: ("Jotunheim's SV holds 38 of its 72 voltage controls off their SSH_2 targets, up to 1.32 times (ERIKSHAMN_G1: "
          "20.9 kV, 27.7 in the SV), so it is no solution of that SSH_2. With those targets the reactive limits "
          "cannot all hold, and it converges only without them.", [STALLED]),
}

SOLVED = [case for case in IGMS if case != "Espheim"]


@pytest.mark.parametrize("case", [*IGMS, *COMBINATIONS, CGM])
def test_power_flow(case):
    assert case_solution(case).problems == KNOWN.get(case, ("", []))[1]


@pytest.mark.parametrize("case", [c for c, (_why, problems) in KNOWN.items() if problems == [STALLED]])
def test_stalled_cases_converge_without_reactive_limits(case):
    """What stalls is the reactive limits on those targets, not the network: without
    them every component converges, at one reference each."""
    solution = solve(case_files(case), against_sv=False, enforce_q_limits=False)
    assert solution.model
    solved = [i for i in solution.model.islands() if i["status"] != "no_reference"]
    assert solved and all(i["status"] == "converged" and len(i["references"]) == 1 for i in solved)


def test_espheim_solves_with_the_target_its_sv_holds(tmp_path):
    """Espheim's stall is its data's: with the three controls on the voltage its SV
    holds them at, it solves with reactive limits, and without them to its SV's voltages."""
    paths = case_files("Espheim")
    held = espheim_at_the_voltage_its_sv_holds()
    assert len(held) == 3
    fixed = with_targets(paths, held, tmp_path)
    assert solve(fixed).model
    unlimited = solve(fixed, enforce_q_limits=False)
    assert unlimited.model and not [p for p in unlimited.problems if "kV" in p]


@pytest.mark.parametrize("case", SOLVED)
def test_the_boundary_changes_nothing(case):
    """BDS_UC1's merge of the boundary with an IGM: its six files bring the IGM no
    bus and no flow, the duplicated objects included."""
    root = ROOT / "boundaryData" / "Grid" / "cimxml"
    merged = solve(case_files(case) + sorted(str(p) for p in root.glob("*.xml")))
    assert merged.problems == case_solution(case).problems
    assert merged.model.voltage_kv() == pytest.approx(case_solution(case).model.voltage_kv(), abs=1e-9)


@pytest.mark.parametrize("area", ["Belgovia", "Galia", "Svedala"])
def test_notices_a_broken_hypothesis(area, tmp_path):
    """With half as much load again in the SSH, the same SV no longer fits."""
    paths = []
    for path in grid_files(area):
        if "_SSH_" in path:
            dataset = cimoxide.CimDataset.decode_file(path)
            loads = {k: v for k, v in dataset.entries().items() if "EnergyConsumer.p" in v}
            assert loads
            for mrid, load in loads.items():
                dataset[mrid] = dict(load, **{"EnergyConsumer.p": str(1.5 * float(load["EnergyConsumer.p"]))})
            path = tmp_path / Path(path).name
            path.write_text(dataset.to_xml_for_profile("SSH"), encoding="utf-8")
            path = str(path)
        paths.append(path)
    assert solve(paths).problems


@pytest.mark.parametrize("case", SOLVED)
def test_exported_files_solve_the_same(case, tmp_path):
    """gridoxide reads what cimoxide writes: each IGM file exported for its profile
    gives the voltages of the file ReliCapGrid supplies."""
    paths = []
    for path in grid_files(*IGMS[case]):
        code = next((c for c in ("EQ", "SSH", "TP", "SV") if f"_{c}_" in path), None)
        if code:
            out = tmp_path / Path(path).name
            out.write_text(cimoxide.CimDataset.decode_file(path).to_xml_for_profile(code), encoding="utf-8")
            path = str(out)
        paths.append(path)
    model, exported = case_solution(case).model, solve(paths).model
    assert exported.n_nodes == model.n_nodes
    assert exported.voltage_mag() == pytest.approx(model.voltage_mag(), abs=1e-9)
    assert exported.voltage_ang() == pytest.approx(model.voltage_ang(), abs=1e-9)
