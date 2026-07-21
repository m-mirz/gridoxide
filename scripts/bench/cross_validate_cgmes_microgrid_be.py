#!/usr/bin/env python3
"""Cross-validates gridoxide's CGMES import + solve against pypowsybl's own
independent CGMES import + AC load flow, on the *same* ENTSO-E MicroGrid-BE-MAS
conformance files `tests/cgmes_microgrid_be_test.rs` already checks against
its fixture's own published `SvVoltage` values. That Rust test's own doc
comment already claimed pypowsybl "also deviates from this fixture's
published SV values by a comparable few percent" — this script is what
actually computes and asserts that, instead of it being a one-off manual
finding recorded only in prose.

Usage: python3 cross_validate_cgmes_microgrid_be.py [--tol 0.01] [--angle-tol 0.3]

Requires `pypowsybl`: pip install pypowsybl

How the two sides are obtained:
- gridoxide's own solve comes from `examples/cgmes_microgrid_be_dump.rs`
  (built with `cargo run --release --example cgmes_microgrid_be_dump
  --features cgmes`, run here as a subprocess and its JSON stdout parsed) —
  there's no Python binding for CGMES import, only for PGM-JSON (see
  src/python.rs), so this is the simplest way to get gridoxide's per-bus
  result into this script.
- pypowsybl loads the same 5 files (EQ_BD + EQ/SSH/TP/SV for BE-MAS) zipped
  together on the fly (pypowsybl's CGMES importer needs one archive/file,
  not a directory — confirmed empirically: pointing it at the BE-MAS
  directory alone fails with "nominalVoltage not found for BaseVoltage
  ...", since the referenced BaseVoltage is only defined in the separate
  boundary set) and solves with the same "BASIC" `LoadFlowParameters`
  bench_pypowsybl.py already uses for its own gridoxide-vs-pypowsybl
  benchmark comparison (uniform voltage init, no distributed slack, no
  reactive limits, phase shifter regulation off, main connected component
  only) — the closest match to gridoxide's own flat-start, single-slack,
  no-reactive-limit-enforcement Newton-Raphson.

Matching buses between the two tools: gridoxide's dump is keyed by
TopologicalNode mRID directly. pypowsybl's bus IDs are *not* TopologicalNode
mRIDs — empirically, IIDM's bus-view bus ID is
"<ConnectivityNodeContainer mRID>_<index>" (confirmed by cross-referencing
the TP file's own `TopologicalNode.ConnectivityNodeContainer` field). Most
containers (here: VoltageLevels) hold exactly one TopologicalNode, so
index is always 0 — but one container in this fixture holds two (a
Series Compensator creates a second electrical node inside the same
voltage level), giving bus indices 0 and 1 with no documented way to know
which is which from IIDM's API alone. Resolved by nearest-voltage-magnitude
matching against gridoxide's own solved magnitudes for that container's
candidate TopologicalNodes — safe here since the two candidates differ by
several kV, far more than the few-percent, tool-to-tool deviation this
script is itself measuring.

Angles: pypowsybl (OpenLoadFlow) is pinned to gridoxide's own slack bus via
the `slackBusesIds` provider parameter (with `referenceBusSelectionMode`
left at its default `FIRST_SLACK`, so that slack bus also becomes the angle
reference), so both tools' angle solutions share the same physical pivot
bus. That still leaves an arbitrary constant offset between the two: OpenLoadFlow
fixes its reference bus's own angle at 0 deg, while gridoxide's slack angle
is set to this fixture's *published* SV angle (e.g. 340.9585 deg — see
`src/cgmes.rs`'s own comment on why, near where it reads `TopologicalIsland`).
So each side's angles are separately rebased to 0 at that shared bus before
comparing — a single global rotation correction, not a per-bus fudge, since
only relative angle differences between buses are physically meaningful in
AC power flow.

Finding the pypowsybl bus ID for gridoxide's slack TopologicalNode requires
running pypowsybl once *without* any override first (its natural, unforced
solve) so `match_buses` can resolve that TN to a concrete bus ID the normal
way — only then can a second, slack-pinned solve be run for the actual
comparison.

Tolerance: empirically, per-bus deviation between gridoxide and pypowsybl on
this fixture now tops out around 0.22% (the 10.5 kV bus) now that gridoxide
models `StaticVarCompensator`'s voltage regulation (see `src/cgmes.rs`'s
Step 8) — before that, the Series-Compensator-side substation (which hosts
an in-service, voltage-mode-regulating SVC targeting 229.5 kV per its
`RegulatingControl`) was gridoxide's worst mismatch at 3.3%, since that
device wasn't converted at all and its voltage pin was silently dropped.
The remaining sub-1% gap is consistent with `tests/cgmes_microgrid_be_test.rs`'s
own "few percent" deviation from the published SV values (a
boundary-truncated sub-model solved with simple fixed-injection
equivalents, not a correctness bug in either tool - see that test's doc
comment). `--tol` defaults to 0.01 (1%), comfortably above the observed max
with headroom for minor pypowsybl-version differences.

With the same reference bus pinned on both sides, per-bus angle deviation
now tops out around 0.07 degrees, now that gridoxide also converts
`ACLineSegment.gch` (shunt conductance — real, not just reactive, line
charging) into a `Line.g_shunt` term stamped into the Y-bus (`src/cgmes.rs`'s
Step 4, `src/network.rs`'s `build_ybus`). Before that fix, gridoxide simply
dropped `gch` (despite its own comment already documenting the field),
undercounting real-power losses on `BE-Line_6`/`BE-Line_2` (~6 MW and ~3 MW
respectively at nominal voltage) — both of which feed straight into the
SVC-regulated substation. A bus with a hard voltage-magnitude pin can only
close a *reactive*-power mismatch by adjusting its own injection; it has no
equivalent slack for *active* power, so the missing MW instead surfaced
entirely as extra angle at that bus and its electrically-adjacent Series
Compensator node (0.34 degrees, vs 0.00-0.10 degrees everywhere else in the
unregulated part of the network) rather than as a voltage error.
`--angle-tol` defaults to 0.3 degrees, still comfortable headroom over the
observed max.
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pypowsybl.loadflow as lf
import pypowsybl.network as pn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE = REPO_ROOT / "tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGid-BaseCase"
BE_DIR = BASE / "MicroGrid-BE-MAS"
CIM_NS = "http://iec.ch/TC57/CIM100#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def run_gridoxide() -> dict:
    """Runs examples/cgmes_microgrid_be_dump.rs and parses its JSON. That
    binary also prints Newton-Raphson iteration diagnostics to stdout
    (src/solver.rs's own `println!`, shared with every other solver
    caller — not specific to this example) ahead of the JSON, so only the
    last line is parsed."""
    proc = subprocess.run(
        ["cargo", "run", "--release", "--example", "cgmes_microgrid_be_dump", "--features", "cgmes"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    last_line = proc.stdout.strip().splitlines()[-1]
    entries = json.loads(last_line)
    return {e["mrid"].lstrip("_"): e for e in entries}


def build_be_mas_zip(tmp_dir: Path) -> Path:
    """Zips the same 5 files tests/cgmes_microgrid_be_test.rs reads (BE-MAS's
    own EQ/SSH/TP/SV plus the separate EQ_BD boundary set) into one archive —
    pypowsybl's CGMES importer needs a single file/archive, not a directory."""
    eq_bd = BASE / "MicroGrid-BD-MAS/20171002T0930Z_ENTSO-E_EQ_BD_2.xml"
    files = [
        eq_bd,
        BE_DIR / "20210325T1530Z_1D_BE_EQ_001.xml",
        BE_DIR / "20210325T1530Z_1D_BE_SSH_001.xml",
        BE_DIR / "20210325T1530Z_1D_BE_TP_001.xml",
        BE_DIR / "20210325T1530Z_1D_BE_SV_001.xml",
    ]
    zip_path = tmp_dir / "microgrid_be.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    return zip_path


def parse_tn_containers() -> dict[str, str]:
    """Maps each TopologicalNode mRID to its own ConnectivityNodeContainer
    mRID (a VoltageLevel here) by reading the TP file directly — this is
    the relationship pypowsybl's bus-view IDs ("<container>_<index>") are
    built from, confirmed empirically (see module docstring)."""
    tp_path = BE_DIR / "20210325T1530Z_1D_BE_TP_001.xml"
    tree = ET.parse(tp_path)
    out = {}
    for tn in tree.getroot().findall(f"{{{CIM_NS}}}TopologicalNode"):
        tn_id = tn.get(f"{{{RDF_NS}}}ID", "").lstrip("_")
        container = tn.find(f"{{{CIM_NS}}}TopologicalNode.ConnectivityNodeContainer")
        if tn_id and container is not None:
            container_id = container.get(f"{{{RDF_NS}}}resource", "").lstrip("#_")
            out[tn_id] = container_id
    return out


def run_pypowsybl(zip_path: Path, slack_bus_id: str | None = None) -> "pandas.DataFrame":
    """Loads the zipped BE-MAS case and solves it with the same "BASIC"
    LoadFlowParameters bench_pypowsybl.py already uses (see module
    docstring). When `slack_bus_id` is given, pins OpenLoadFlow's slack (and,
    per the default `referenceBusSelectionMode=FIRST_SLACK`, its angle
    reference too) to that bus via the `slackBusesIds` provider parameter —
    see the module docstring's "Angles" section. Returns the bus-view
    dataframe (indexed by pypowsybl bus id, with `v_mag`/`v_angle` columns)."""
    network = pn.load(str(zip_path))
    provider_parameters = {"slackBusesIds": slack_bus_id} if slack_bus_id else {}
    params = lf.Parameters(
        voltage_init_mode=lf.VoltageInitMode.UNIFORM_VALUES,
        distributed_slack=False,
        use_reactive_limits=False,
        phase_shifter_regulation_on=False,
        transformer_voltage_control_on=False,
        connected_component_mode=lf.ConnectedComponentMode.MAIN,
        provider_parameters=provider_parameters,
    )
    result = lf.run_ac(network, parameters=params)
    if result[0].status != lf.ComponentStatus.CONVERGED:
        raise RuntimeError(f"pypowsybl load flow did not converge: {result[0].status_text}")
    return network.get_buses()


def match_buses(gridoxide: dict, tn_to_container: dict, powsybl_buses) -> list[tuple[str, float, float, str]]:
    """Pairs each TopologicalNode mRID with its pypowsybl bus id, grouping by
    shared container and resolving multi-TN containers (see module
    docstring) via nearest-voltage-magnitude matching. Returns
    (tn_id, gridoxide_kv, pypowsybl_kv, pypowsybl_bus_id) tuples."""
    by_container: dict[str, list[str]] = {}
    for tn_id, container_id in tn_to_container.items():
        if tn_id in gridoxide:
            by_container.setdefault(container_id, []).append(tn_id)

    powsybl_by_container: dict[str, list[tuple[str, float]]] = {}
    for bus_id, v_mag in powsybl_buses["v_mag"].items():
        container_id, _, _idx = bus_id.rpartition("_")
        powsybl_by_container.setdefault(container_id, []).append((bus_id, v_mag))

    matches = []
    for container_id, tn_ids in by_container.items():
        candidates = list(powsybl_by_container.get(container_id, []))
        for tn_id in tn_ids:
            gx_kv = gridoxide[tn_id]["voltage_mag"] * gridoxide[tn_id]["u_rated"] / 1e3
            if not candidates:
                continue
            best = min(candidates, key=lambda c: abs(c[1] - gx_kv))
            candidates.remove(best)
            best_bus_id, best_v_mag = best
            matches.append((tn_id, gx_kv, best_v_mag, best_bus_id))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tol", type=float, default=0.01, help="max relative voltage-magnitude deviation (default 0.01)")
    parser.add_argument("--angle-tol", type=float, default=0.3, help="max slack-relative angle deviation in degrees (default 0.3)")
    args = parser.parse_args()

    if not BE_DIR.exists():
        print(f"skipping: {BE_DIR} not found — run "
              "`git submodule update --init tests/data/CGMES-Test-Configurations`", file=sys.stderr)
        return

    gridoxide = run_gridoxide()
    tn_to_container = parse_tn_containers()
    slack_tn_id = next(tn_id for tn_id, e in gridoxide.items() if e["bus_type"] == "Slack")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = build_be_mas_zip(Path(tmp))
        # Pass 1: unforced solve, only to resolve gridoxide's slack TN to a
        # concrete pypowsybl bus id via the normal nearest-magnitude matching
        # (see module docstring's "Angles" section for why this needs its
        # own solve rather than reusing the pinned one below).
        unforced_buses = run_pypowsybl(zip_path)
        unforced_matches = match_buses(gridoxide, tn_to_container, unforced_buses)
        slack_bus_id = next(bus_id for tn_id, _, _, bus_id in unforced_matches if tn_id == slack_tn_id)

        # Pass 2: the actual comparison, with pypowsybl's slack (and,
        # transitively, its angle reference) pinned to that same bus.
        powsybl_buses = run_pypowsybl(zip_path, slack_bus_id=slack_bus_id)

    matches = match_buses(gridoxide, tn_to_container, powsybl_buses)
    assert len(matches) == len(gridoxide), \
        f"expected to match all {len(gridoxide)} TopologicalNodes, matched {len(matches)}"

    gx_slack_ang_deg = math.degrees(gridoxide[slack_tn_id]["voltage_ang"])
    ps_slack_ang_deg = powsybl_buses.loc[slack_bus_id, "v_angle"]

    print(f"{'TopologicalNode':<38} {'gridoxide (kV)':>15} {'pypowsybl (kV)':>15} {'v diff':>8}"
          f" {'gx angle':>9} {'ps angle':>9} {'a diff':>7}")
    worst_v = 0.0
    worst_a = 0.0
    failures = []
    for tn_id, gx_kv, ps_kv, bus_id in sorted(matches, key=lambda m: m[0]):
        v_diff = abs(gx_kv - ps_kv) / gx_kv
        # Wrap into (-180, 180]: both angles are slack-relative, but
        # gridoxide's raw values run through 0/360 wherever the fixture's
        # published SV angle for its slack happens to fall, so an unwrapped
        # value (or a naive difference of two unwrapped values) can
        # spuriously read ~360° apart instead of ~0°.
        gx_ang_deg = (math.degrees(gridoxide[tn_id]["voltage_ang"]) - gx_slack_ang_deg + 180) % 360 - 180
        ps_ang_deg = (powsybl_buses.loc[bus_id, "v_angle"] - ps_slack_ang_deg + 180) % 360 - 180
        a_diff = abs((gx_ang_deg - ps_ang_deg + 180) % 360 - 180)
        worst_v = max(worst_v, v_diff)
        worst_a = max(worst_a, a_diff)
        print(f"{tn_id:<38} {gx_kv:>15.3f} {ps_kv:>15.3f} {v_diff:>7.2%}"
              f" {gx_ang_deg:>8.2f}° {ps_ang_deg:>8.2f}° {a_diff:>6.2f}°")
        if v_diff >= args.tol:
            failures.append(f"{tn_id} voltage deviates {v_diff:.2%}, exceeding tolerance {args.tol:.0%}")
        if a_diff >= args.angle_tol:
            failures.append(f"{tn_id} angle deviates {a_diff:.2f}°, exceeding tolerance {args.angle_tol:.2f}°")

    print(f"\nworst voltage deviation: {worst_v:.2%} (tolerance {args.tol:.0%})")
    print(f"worst angle deviation: {worst_a:.2f}° (tolerance {args.angle_tol:.2f}°)")
    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print("OK: gridoxide and pypowsybl agree within tolerance on every bus")


if __name__ == "__main__":
    main()
