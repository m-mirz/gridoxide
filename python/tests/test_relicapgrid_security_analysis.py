"""SAR_UC1: `gridoxide.security_analysis` on ReliCapGrid's CGM, and the SAR it writes.

The flows are compared with the SvPowerFlow of IGMs that agree with their SV, each result is
recomputed from its limit, and cimoxide validates the dataset in the company of the supplied SAR.
"""

import collections
import functools
import statistics

import pytest

import gridoxide
from _relicapgrid import (ROOT, SAR, bare_type, cgm_files, cimoxide, co_files, contingencies, grid_files, norm,
                          objects_of)
from test_relicapgrid_power_flow import solve

SUPPLIED_HEADER = "b59f6b96-1333-44fd-b9cc-23419db61281"
SUPPLIED_TERMINAL = "3b3075b8-e0e5-66e9-447e-d7e11f76788f"
HEADER = dict(publisher="https://energy.referencedata.eu/test/party/RCC-Jotunheim",
              is_version_of="https://energy.referencedata.eu/Test/Model/Jotunheim-SAR",
              rights_holder="ENTSO-E", generated_at="2026-11-08T00:00:00Z",
              title="20220616_Jotunheim_SAR_gridoxide",
              description="Security analysis of ReliCapGrid's CGM by gridoxide.")


@functools.cache
def cgm():
    return objects_of(tuple(cgm_files()))


@functools.cache
def analysis(sharing="capability"):
    return gridoxide.security_analysis(cgm_files(), co_files(), sharing=sharing)


@functools.cache
def violations():
    return analysis().violations()


@functools.cache
def sar_xml():
    return analysis().to_sar(**HEADER)


@functools.cache
def sar():
    return {norm(k): v for k, v in cimoxide.CimDataset.decode_str(sar_xml()).entries().items()}


def results(kind):
    return [v for v in sar().values() if bare_type(v["_type"]) == kind]


def name_of(terminal):
    return cgm().get(norm(cgm()[norm(terminal)]["Terminal.ConductingEquipment"]), {}).get("IdentifiedObject.name")


# --- the flows ---------------------------------------------------------------

@pytest.mark.parametrize("case, median, largest", [
    ("Belgovia", 0.01, 5), ("Britheim", 0.5, 1.5), ("Galia", 0.02, 7), ("Svedala", 0.1, 12)])
def test_flows_agree_with_the_published_flows(case, median, largest):
    """MVA between terminal_flows and SvPowerFlow, on IGMs that agree with their SV."""
    solution = solve(grid_files(case))
    assert not solution.problems
    objects = objects_of(tuple(grid_files(case)))
    published = {norm(v["SvPowerFlow.Terminal"]): complex(float(v["SvPowerFlow.p"]), float(v["SvPowerFlow.q"]))
                 for v in objects.values() if bare_type(v["_type"]) == "SvPowerFlow"}
    off = [abs(complex(p, q) - published[norm(t)])
           for t, (p, q, _) in solution.model.terminal_flows().items() if norm(t) in published]
    assert len(off) > 5
    assert statistics.median(off) < median and max(off) < largest


def test_a_terminals_current_follows_from_its_power_and_voltage():
    """|S| / (sqrt(3) kV), with kV its node's as solved."""
    model = solve(grid_files("Galia")).model
    objects = objects_of(tuple(grid_files("Galia")))
    nodes = {k: norm(v["ConnectivityNode.TopologicalNode"]) for k, v in objects.items()
             if "ConnectivityNode.TopologicalNode" in v}
    kv = model.voltage_kv()
    checked = 0
    for terminal, (p, q, amps) in model.terminal_flows().items():
        data = objects[norm(terminal)]
        node = data.get("Terminal.TopologicalNode")
        node = norm(node) if node else nodes.get(norm(data.get("Terminal.ConnectivityNode", "")))
        bus = model.bus_index_for_mrid(f"_{node}")
        if bus is not None and kv[bus]:
            assert amps == pytest.approx(1000 * abs(complex(p, q)) / (3 ** 0.5 * kv[bus]), rel=1e-9)
            checked += 1
    assert checked > 10


# --- the limits --------------------------------------------------------------

def test_every_tatl_in_the_data_is_500_a_and_none_above_its_patl():
    """Why none is monitored: 558 forbid for ten minutes what their PATL allows for good."""
    tatl = [k for k, v in cgm().items() if bare_type(v["_type"]) == "CurrentLimit"
            and cgm().get(norm(v["OperationalLimit.OperationalLimitType"]), {}).get(
                "OperationalLimitType.kind", "").endswith("tatl")]
    assert len(tatl) == 578
    assert {float(cgm()[k]["CurrentLimit.value"]) for k in tatl} == {500.0}
    assert not {norm(v["limit"]) for v in violations()} & set(tatl)


def test_a_result_is_its_limit_and_current():
    """Recomputed from the data: a violation's limit, and its percent, 1100 A on 1000 A being 110
    as the profile defines it. The supplied SAR writes the ratio, 1.0632 on 1673.4 A and 1574 A."""
    for v in violations():
        limit = cgm()[norm(v["limit"])]
        amps = float(limit.get("CurrentLimit.value") or limit["CurrentLimit.normalValue"])
        assert v["limit_a"] == amps and v["i"] > amps
        assert v["percent"] == pytest.approx(100 * v["i"] / amps)
    [supplied] = [v for v in cimoxide.CimDataset.decode_file(str(SAR)).entries().values()
                  if "PowerFlowResult.value" in v]
    assert float(supplied["PowerFlowResult.value"]) == pytest.approx(1673.4026 / 1574, abs=1e-3)


# --- the analysis ------------------------------------------------------------

def test_the_counts():
    a = analysis()
    assert (a.solved, len(a.stalled), a.absent) == (304, 21, [])
    base = [v for v in violations() if not v["contingency"]]
    after = [v for v in violations() if v["contingency"]]
    violated = {v["limit"] for v in base}
    new = [v for v in after if v["limit"] not in violated]
    assert (len(base), len(after)) == (13, 4003)
    assert (len(new), len({v["contingency"] for v in new}), len({v["terminal"] for v in new})) == (214, 81, 55)


def test_all_that_stall_are_espheims():
    assert {contingencies()[norm(c)][0] for c in analysis().stalled} == {"Espheim"}


@pytest.mark.parametrize("sharing, solved, stalled, base, behind_reference", [
    ("capability", 304, 21, 13, (303, 127)),
    ("reference", 312, 13, 17, (311, 596)),
    ("equal", 287, 38, 21, (286, 127)),
])
def test_the_answer_depends_on_who_takes_the_imbalance(sharing, solved, stalled, base, behind_reference):
    """With BO-TR2_3, behind the CGM's one reference: the contingencies violating it, the worst."""
    a = analysis(sharing)
    found = [v for v in a.violations() if v["contingency"] and name_of(v["terminal"]) == "BO-TR2_3"]
    assert (a.solved, len(a.stalled)) == (solved, stalled)
    assert len([v for v in a.violations() if not v["contingency"]]) == base
    assert (len({v["contingency"] for v in found}), round(max(v["percent"] for v in found))) == behind_reference


def test_the_supplied_sars_base_case_cannot_tell_them_apart():
    """On its one terminal (-1.531 MW, 0.962 Mvar) both agree with it: a dead-end stub under 2 MW, set
    by local data. So it is little evidence for either, or for the solver. It is closer than the
    CGM's own SvPowerFlow, -1.559 MW and 0.979 Mvar."""
    supplied = pytest.approx((-1.53093, 0.961664), abs=0.01)
    for sharing in ("capability", "reference"):
        flows = {norm(k): v for k, v in analysis(sharing).base_flows().items()}
        assert flows[SUPPLIED_TERMINAL][:2] == supplied


def test_the_cgms_state_leaves_the_imbalance_on_one_machine():
    """Why the published state does not settle the sharing, though it is how it was made: of the
    machines the SV has a flow for, only two move from their schedule, and BO-G1, scheduled at
    52.8 MW, runs at -481.8 MW: a generator as a motor."""
    published = {norm(v["SvPowerFlow.Terminal"]): float(v["SvPowerFlow.p"])
                 for v in cgm().values() if bare_type(v["_type"]) == "SvPowerFlow"}
    moved = {}
    for terminal, obj in cgm().items():
        machine = cgm().get(norm(obj.get("Terminal.ConductingEquipment", "")), {})
        if (bare_type(machine.get("_type", "")) == "SynchronousMachine" and terminal in published
                and machine.get("Equipment.inService") != "false"):
            if abs(published[terminal] - float(machine["RotatingMachine.p"])) > 0.5:
                moved[machine["IdentifiedObject.name"]] = -published[terminal]
    assert moved.keys() == {"BO-G1", "SM_BH1"}
    assert moved["BO-G1"] == pytest.approx(-481.8, abs=0.1)


def test_normal_pf_is_scaled_per_igm():
    """Why capability is maxOperatingP: in a CGM normalPF would give Belgovia's 2 units Svedala's 38's
    share, though they have 700 MW to its 16,750."""
    def fleet(area):
        objects = objects_of(tuple(str(p) for p in (ROOT / area / "Grid/cimxml").glob("*.xml")))
        units = [objects[norm(v["RotatingMachine.GeneratingUnit"])] for v in objects.values()
                 if bare_type(v["_type"]) == "SynchronousMachine" and "RotatingMachine.GeneratingUnit" in v]
        return (len(units), sum(float(u["GeneratingUnit.normalPF"]) for u in units),
                sum(float(u["GeneratingUnit.maxOperatingP"]) for u in units))

    assert fleet("Belgovia") == pytest.approx((2, 100, 700))
    assert fleet("Svedala") == pytest.approx((38, 100, 16750))


def test_no_machine_is_shared_past_its_limits():
    """In the base case and every contingency solved, with maxOperatingP as the share."""
    assert analysis().past_limits() == []


def test_equipment_a_contingency_trips_has_no_result_for_it():
    """Unlike the supplied SAR, which reports a violation on the line its contingency trips."""
    for v in violations():
        if v["contingency"]:
            assert norm(cgm()[norm(v["terminal"])]["Terminal.ConductingEquipment"]) not in contingencies()[
                norm(v["contingency"])][2]


def test_an_unknown_sharing_is_refused():
    with pytest.raises(ValueError, match="sharing"):
        gridoxide.security_analysis(cgm_files(), co_files(), sharing="nonsense")


# --- the dataset -------------------------------------------------------------

def test_every_violation_is_in_the_dataset_as_computed():
    written = collections.Counter(
        (norm(v.get("ContingencyPowerFlowResult.Contingency", "")) or None, norm(v["PowerFlowResult.ACDCTerminal"]),
         norm(v["PowerFlowResult.OperationalLimit"]))
        for v in sar().values() if "PowerFlowResult.OperationalLimit" in v)
    assert written == collections.Counter(
        (norm(v["contingency"]) if v["contingency"] else None, norm(v["terminal"]), norm(v["limit"]))
        for v in violations())
    for result in results("ContingencyPowerFlowResult"):
        limit = cgm()[norm(result["PowerFlowResult.OperationalLimit"])]
        amps = float(result["PowerFlowResult.valueA"])
        limit_a = float(limit.get("CurrentLimit.value") or limit["CurrentLimit.normalValue"])
        assert amps > limit_a
        assert float(result["PowerFlowResult.value"]) == pytest.approx(100 * amps / limit_a, abs=1e-3)
        assert result["PowerFlowResult.isViolation"] == "true"
        assert float(result["PowerFlowResult.absoluteValue"]) == amps


def test_the_datasets_results():
    assert (len(results("ContingencyPowerFlowResult")), len(results("BaseCasePowerFlowResult"))) == (4003, 68)
    assert sum(v["PowerFlowResult.isViolation"] == "true" for v in results("BaseCasePowerFlowResult")) == 13


def test_a_violated_terminal_has_its_base_case():
    base = {norm(v["PowerFlowResult.ACDCTerminal"]) for v in results("BaseCasePowerFlowResult")}
    assert {norm(v["PowerFlowResult.ACDCTerminal"]) for v in results("ContingencyPowerFlowResult")} <= base


def test_stalled_contingencies_have_no_result():
    named = {norm(v["ContingencyPowerFlowResult.Contingency"]) for v in results("ContingencyPowerFlowResult")}
    assert not named & {norm(c) for c in analysis().stalled}


def test_the_header_keeps_its_references_and_literals():
    xml = sar_xml()
    assert "<dcat:version>1.0.0</dcat:version>" in xml
    assert "Imbalance: capability." in xml
    for reference in ("dcterms:accessRights", "dcterms:license", "dcterms:publisher", "prov:wasGeneratedBy",
                      "dcterms:conformsTo"):
        assert f"<{reference} rdf:resource=" in xml, reference
    assert len(analysis().requires) == 7
    for dataset in analysis().requires:
        assert f'<dcterms:requires rdf:resource="{dataset}"/>' in xml


def test_nothing_of_the_template_is_left():
    assert "00000000-0000-0000-0000" not in sar_xml()


def test_the_dataset_is_reproducible():
    assert gridoxide.security_analysis(cgm_files(), co_files()).to_sar(**HEADER) == sar_xml()


# --- validated by cimoxide, the test dependency ------------------------------

def findings(paths):
    return cimoxide.validate_files([str(p) for p in paths], common=True, quality=True)


def rule(finding):
    rule_id = str(finding.rule_id)
    return rule_id.rsplit("#", 1)[-1].split(":", 1)[-1]


def test_cimoxide_finds_nothing_on_the_results_or_new_on_the_header(tmp_path):
    """In the model it reports on, with the CO files it names: nothing on a result, and nothing on the
    header that the supplied SAR's header does not have. The supplied SAR's results have an
    IdentifiedObject.mRID the profile does not allow, and these do not."""
    path = tmp_path / "2D_SAR_gridoxide.xml"
    path.write_text(sar_xml(), encoding="utf-8")
    context = cgm_files() + co_files()
    ours = findings(context + [path])
    header = next(k for k, v in sar().items() if bare_type(v["_type"]) == "Dataset")
    assert not [f for f in ours if norm(f.object_id) in sar() and norm(f.object_id) != header]
    known = {rule(f) for f in findings(context + [SAR]) if norm(f.object_id) == SUPPLIED_HEADER}
    assert {rule(f) for f in ours if norm(f.object_id) == header} <= known


def test_a_result_without_its_value_is_reported(tmp_path):
    """A broken SAR: the profile requires PowerFlowResult.value beside a limit."""
    xml = sar_xml()
    start = xml.index("<nc:PowerFlowResult.value>")
    end = xml.index("</nc:PowerFlowResult.value>", start) + len("</nc:PowerFlowResult.value>")
    path = tmp_path / "2D_SAR_broken.xml"
    path.write_text(xml[:start] + xml[end:], encoding="utf-8")
    assert any("value" in rule(f).lower() for f in findings(cgm_files() + co_files() + [path])
               if norm(f.object_id) in sar())
