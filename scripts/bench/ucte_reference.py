#!/usr/bin/env python3
"""Regenerate the pypowsybl reference solutions for `tests/ucte_test.rs`.

The UCTE importer's real gate is not "does it parse" — every column could be
off by one and the file would still load. It is "does the imported network
solve to the same answer as an independent implementation". This script
produces that answer.

Usage (from the repo root, with a venv that has pypowsybl):

    .venv-pypowsybl/bin/python scripts/bench/ucte_reference.py

It rewrites `tests/data/ucte/<case>.pypowsybl.json` for every `.uct` fixture
beside it. Re-run it when a fixture is added, and commit the result: the Rust
test reads the JSON, so the suite itself needs neither Python nor pypowsybl.

Two settings matter and both are deliberate:

* **The load flow is stripped of every outer loop** — no distributed slack, no
  reactive limits, no transformer or phase-shifter regulation. gridoxide's
  plain `newton_raphson` has none of those either, so leaving any of them on
  would compare two different problems and blame the importer for the gap.
* **The slack is forced to the bus gridoxide picks.** No vendored fixture
  declares a type-3 (`U and theta constant`) node, so both tools choose one.
  These fixtures happen to be lossless, which makes the choice almost
  invisible; that is luck, not a property, so the rule is mirrored here rather
  than relied upon.
"""

import json
import math
import pathlib
import sys

import pypowsybl as pp
import pypowsybl.loadflow as lf

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "tests" / "data" / "ucte"


def gridoxide_slack(path):
    """Mirror of `ucte::SlackPolicy::LargestGeneration`.

    The node with the largest generation, preferring one that regulates
    voltage, ties broken on the node code so the choice is reproducible.
    Generation is stored negative in UCTE, hence the sign flip.
    """
    best = None
    block = None
    for raw in path.read_bytes().split(b"\n"):
        line = raw.rstrip(b"\r").decode("latin-1")
        if line.startswith("##"):
            tag = line[2:].strip()
            block = "N" if (tag == "N" or tag.startswith("Z")) else tag[:1]
            continue
        if block != "N" or len(line) < 25:
            continue
        code = line[:8]
        regulating = line[24] in "23"
        try:
            generation = -float(line[49:56].strip() or 0.0)
        except ValueError:
            generation = 0.0
        if generation <= 0.0:
            continue
        key = (regulating, generation, [-ord(c) for c in code])
        if best is None or key > best[0]:
            best = (key, code)
    return best[1].strip() if best else None


def clean(x):
    # A de-energised element reports NaN, which is not valid JSON. Emit null so
    # the Rust side can skip it rather than choke on the whole document.
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return float(x)


def reference(path):
    network = pp.network.load(str(path))
    slack = gridoxide_slack(path)
    parameters = lf.Parameters(
        distributed_slack=False,
        use_reactive_limits=False,
        transformer_voltage_control_on=False,
        phase_shifter_regulation_on=False,
        voltage_init_mode=pp.loadflow.VoltageInitMode.UNIFORM_VALUES,
        provider_parameters={
            "slackBusSelectionMode": "NAME",
            "slackBusesIds": slack or "",
            "voltageRemoteControl": "false",
        },
    )
    status = str(lf.run_ac(network, parameters=parameters)[0].status)

    # The bus-breaker view is keyed on the UCTE node code itself. The bus view
    # merges the two ends of a 380/380 phase shifter into one voltage level and
    # would collapse two distinct nodes onto one key.
    levels = network.get_voltage_levels()
    buses = {}
    for bus_id, row in network.get_bus_breaker_view_buses().iterrows():
        nominal = levels.loc[row["voltage_level_id"], "nominal_v"]
        buses[bus_id.strip()] = {
            "v_pu": clean(row["v_mag"] / nominal),
            "angle_deg": clean(row["v_angle"]),
        }

    branches = {}
    for frame, kind in (
        (network.get_lines(), "line"),
        (network.get_2_windings_transformers(), "xfmr"),
    ):
        for element_id, row in frame.iterrows():
            branches[element_id] = {
                "kind": kind,
                "p1": clean(row["p1"]),
                "q1": clean(row["q1"]),
                "p2": clean(row["p2"]),
                "q2": clean(row["q2"]),
            }
    return {"slack": slack, "status": status, "buses": buses, "branches": branches}


def main():
    cases = sorted(FIXTURES.glob("*.uct"))
    if not cases:
        sys.exit(f"no .uct fixtures under {FIXTURES}")
    for case in cases:
        try:
            document = reference(case)
        except Exception as error:  # a deliberately malformed fixture
            print(f"  skip {case.name}: {error}")
            continue
        out = case.with_suffix(".pypowsybl.json")
        out.write_text(json.dumps(document, indent=1) + "\n")
        print(
            f"  {case.name}: {document['status']}, slack {document['slack']}, "
            f"{len(document['buses'])} buses, {len(document['branches'])} branches"
        )


if __name__ == "__main__":
    main()
