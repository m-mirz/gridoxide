"""Shared helpers for reading a CGMES fixture's own published `SvVoltage`
values and `TopologicalNode`/`ConnectivityNodeContainer` relationships
directly out of its SV/TP profile XML — used by both `bench_gridoxide_cgmes.py`
and `bench_pypowsybl_cgmes.py` to compute each tool's own deviation from the
fixture's published reference solution (not a tool-vs-tool comparison), so
the two benchmark scripts can report accuracy independently without needing
to match bus IDs between gridoxide and pypowsybl at all.

`parse_tn_containers` and the "<container>_<index>" pypowsybl bus-ID
convention it supports are the same relationship
`cross_validate_cgmes_microgrid_be.py`'s own module docstring documents in
more detail — see that file for the full explanation of why pypowsybl bus
IDs aren't TopologicalNode mRIDs directly.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

CIM_NS = "http://iec.ch/TC57/CIM100#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def parse_sv_voltages(sv_path: Path) -> dict[str, tuple[float, float]]:
    """Returns {tn_mrid (with leading '_'): (v_kv, angle_deg)} from an SV
    profile's own `SvVoltage` instances."""
    tree = ET.parse(sv_path)
    out = {}
    for sv in tree.getroot().findall(f"{{{CIM_NS}}}SvVoltage"):
        tn = sv.find(f"{{{CIM_NS}}}SvVoltage.TopologicalNode")
        v = sv.find(f"{{{CIM_NS}}}SvVoltage.v")
        angle = sv.find(f"{{{CIM_NS}}}SvVoltage.angle")
        if tn is None or v is None or angle is None:
            continue
        tn_id = "_" + tn.get(f"{{{RDF_NS}}}resource", "").lstrip("#_")
        out[tn_id] = (float(v.text), float(angle.text))
    return out


def parse_tn_containers(tp_path: Path) -> dict[str, str]:
    """Returns {tn_mrid (with leading '_'): container_mrid (*no* leading
    '_')} from a TP profile's own `TopologicalNode.ConnectivityNodeContainer`
    references — the relationship pypowsybl's own bus-view IDs
    ("<container>_<index>") are built from. The container id deliberately
    carries no leading underscore, unlike every other mrid this module
    handles: pypowsybl's own bus id is "<raw-uuid>_<index>" with no
    underscore prefix on the uuid part, confirmed empirically (this is the
    one place a CGMES mrid convention has to match a *pypowsybl* string
    format instead of CGMES's own rdf:ID convention)."""
    tree = ET.parse(tp_path)
    out = {}
    for tn in tree.getroot().findall(f"{{{CIM_NS}}}TopologicalNode"):
        tn_id = "_" + tn.get(f"{{{RDF_NS}}}ID", "").lstrip("_")
        container = tn.find(f"{{{CIM_NS}}}TopologicalNode.ConnectivityNodeContainer")
        if container is not None:
            container_id = container.get(f"{{{RDF_NS}}}resource", "").lstrip("#_")
            out[tn_id] = container_id
    return out


def match_powsybl_buses_to_tn(
    tn_to_container: dict[str, str],
    expected: dict[str, tuple[float, float]],
    powsybl_buses,
) -> dict[str, str]:
    """Pairs each TopologicalNode mRID (that has a published SvVoltage) with
    a pypowsybl bus id, grouping by shared container and resolving
    multi-TopologicalNode containers via nearest-magnitude matching against
    the *published* v_kv (not gridoxide's own solved value, unlike
    `cross_validate_cgmes_microgrid_be.py`'s `match_buses` — this keeps the
    pypowsybl-side accuracy metric independent of gridoxide's own solve).
    Returns {tn_mrid: pypowsybl_bus_id}, omitting any TN a container has no
    spare pypowsybl bus candidate left for.
    """
    by_container: dict[str, list[str]] = {}
    for tn_id, container_id in tn_to_container.items():
        if tn_id in expected:
            by_container.setdefault(container_id, []).append(tn_id)

    powsybl_by_container: dict[str, list[tuple[str, float]]] = {}
    for bus_id, v_mag in powsybl_buses["v_mag"].items():
        container_id, _, _idx = bus_id.rpartition("_")
        powsybl_by_container.setdefault(container_id, []).append((bus_id, v_mag))

    out = {}
    for container_id, tn_ids in by_container.items():
        candidates = list(powsybl_by_container.get(container_id, []))
        for tn_id in tn_ids:
            if not candidates:
                continue
            exp_kv, _ = expected[tn_id]
            best = min(candidates, key=lambda c: abs(c[1] - exp_kv))
            candidates.remove(best)
            out[tn_id] = best[0]
    return out


def deviation_stats(errors: list[float]) -> dict[str, float]:
    """Summarizes a list of relative voltage-magnitude errors (fractions,
    not percent) as median/p90/max, matching
    `cgmes_common::assert_matches_sv_percentile`'s own metric choice on the
    Rust side."""
    if not errors:
        return {"n": 0, "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    errs = sorted(errors)
    n = len(errs)
    return {
        "n": n,
        "median": errs[n // 2],
        "p90": errs[n * 9 // 10],
        "max": errs[-1],
    }
