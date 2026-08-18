#!/usr/bin/env python3
"""Regenerate the pypowsybl reference solutions for `tests/iidm_test.rs`.

Same arrangement as `ucte_reference.py`, and for the same reason: an importer
that parses is not an importer that is right. Run from the repo root with a venv
that has pypowsybl:

    .venv-pypowsybl/bin/python scripts/bench/iidm_reference.py

It rewrites `tests/data/iidm/<case>.pypowsybl.json` for every `.xiidm` fixture
beside it; commit the result, since the Rust suite reads the JSON and needs
neither Python nor pypowsybl.

Unlike the UCTE script this does **not** force a slack bus. IIDM carries no
slack of its own, so gridoxide picks one and pypowsybl picks one, and they need
not agree. The Rust side compares angles relative to a shared bus and compares
branch flows directly, which is legitimate here because every one of these
fixtures is lossless or near-lossless — the slack absorbs almost nothing, so
moving it does not move the flows. The comparison reports how far apart the two
answers are rather than assuming; a fixture where the choice mattered would show
up immediately as a large deviation rather than as a silent bias.

Buses are keyed by the **bus-breaker** view where that view names them, since
gridoxide labels a bus-breaker bus by the id its file gave. Node-breaker
voltage levels have no such name on either side, so those buses simply do not
intersect and the branch flows carry the comparison.
"""

import json
import math
import pathlib
import sys

import pypowsybl as pp
import pypowsybl.loadflow as lf

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "tests" / "data" / "iidm"


def clean(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return float(x)


def reference(path):
    network = pp.network.load(str(path))
    parameters = lf.Parameters(
        distributed_slack=False,
        use_reactive_limits=False,
        transformer_voltage_control_on=False,
        phase_shifter_regulation_on=False,
        voltage_init_mode=pp.loadflow.VoltageInitMode.UNIFORM_VALUES,
        provider_parameters={"voltageRemoteControl": "false"},
    )
    status = str(lf.run_ac(network, parameters=parameters)[0].status)

    levels = network.get_voltage_levels()
    buses = {}
    for bus_id, row in network.get_bus_breaker_view_buses().iterrows():
        nominal = levels.loc[row["voltage_level_id"], "nominal_v"]
        buses[bus_id.strip()] = {
            "v_pu": clean(row["v_mag"] / nominal),
            "angle_deg": clean(row["v_angle"]),
        }

    branches = {}
    for frame in (network.get_lines(), network.get_2_windings_transformers()):
        for element_id, row in frame.iterrows():
            branches[element_id] = {
                "p1": clean(row["p1"]),
                "q1": clean(row["q1"]),
                "p2": clean(row["p2"]),
                "q2": clean(row["q2"]),
            }
    return {"status": status, "buses": buses, "branches": branches}


def main():
    cases = sorted(FIXTURES.glob("*.xiidm"))
    if not cases:
        sys.exit(f"no .xiidm fixtures under {FIXTURES}")
    for case in cases:
        try:
            document = reference(case)
        except Exception as error:
            print(f"  skip {case.name}: {error}")
            continue
        case.with_suffix(".pypowsybl.json").write_text(json.dumps(document, indent=1) + "\n")
        print(
            f"  {case.name}: {document['status']}, "
            f"{len(document['buses'])} buses, {len(document['branches'])} branches"
        )


if __name__ == "__main__":
    main()
