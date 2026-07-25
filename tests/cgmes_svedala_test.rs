mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;
use gridoxide::types::BusType;

/// ENTSO-E's Svedala conformance test configuration — a real, Swedish-
/// network-styled model (191 `TopologicalNode`s, no switch merging needed at
/// all here despite 1028 `Disconnector`s + 436 `Breaker`s: unlike SmallGrid,
/// this fixture's own TP profile already fully pre-merges every closed one).
/// Exercises `RatioTapChangerTable`/`RatioTapChangerTablePoint` and
/// `StaticVarCompensator` at real scale, but nothing this converter doesn't
/// already handle — no new CGMES gaps found here.
///
/// 78 buses are de-energized (`TopologicalIsland`-absent, forced `Slack` at
/// V=0, matching RealGrid's own precedent) — including 7 small, genuinely
/// isolated 2-bus pairs where *both* buses are independently de-energized,
/// producing `IslandStatus::AmbiguousReferenceBus` (two forced `Slack`
/// buses in the same tiny disconnected component). Harmless: both buses in
/// each pair are already correctly pinned at V=0 by the de-energization
/// step itself, before island classification ever runs — the "ambiguous"
/// label is just descriptive of *how* they ended up de-energized, not an
/// unresolved solver problem.
///
/// Percentile-based voltage match (like RealGrid, not MiniGrid/SmallGrid's
/// hard per-bus check) — median error is excellent (0.08%) but there's a
/// real, non-negligible tail (p99 ~4.3%, max ~4.7%), the same
/// `newton_raphson`-doesn't-enforce-Q-limits pattern RealGrid's own doc
/// comment documents, just at a smaller scale.
#[test]
fn test_cgmes_svedala() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/Svedala/Svedala-Merged");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("Svedala_EQBD.xml");
    let eq = dir.join("Svedala_EQ.xml");
    let ssh = dir.join("Svedala_SSH.xml");
    let tp = dir.join("Svedala_TP.xml");
    let sv = dir.join("Svedala_SV.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    assert_eq!(tn_mrids.len(), 191, "expected 191 physical TopologicalNode buses");
    let expected = cgmes_common::expected_voltages(&ds);

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    // No closed switch needs merging here — unlike SmallGrid, this
    // fixture's own TP already fully pre-merges every one.
    assert_eq!(buses.len(), 191, "expected no bus merging (TP already pre-merged)");
    assert_eq!(lines.len(), 90, "expected 90 ACLineSegment lines");
    assert_eq!(transformers.len(), 53, "expected 53 two-winding transformers");
    assert_eq!(shunts.len(), 46, "expected 46 LinearShuntCompensator shunts");

    let n_deenergized = buses.iter().filter(|b| b.bus_type == BusType::Slack && b.voltage_mag == 0.0).count();
    assert_eq!(n_deenergized, 78, "expected 78 de-energized (TopologicalIsland-absent) buses");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv_percentile(&result, &bus_index, &expected, 5e-3, 3e-2, 6e-2);
}
