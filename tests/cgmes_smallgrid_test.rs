mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;
use gridoxide::types::BusType;

/// ENTSO-E's SmallGrid conformance test configuration — despite the name, a
/// genuinely large, real-topology-style network (167 physical
/// `TopologicalNode`s before merging), the first fixture with substantial
/// switchgear: 838 `Disconnector`s + 427 `Breaker`s + 1 bare `Switch`,
/// exercising `merge_closed_switches`'s topological reduction at real scale
/// for the first time (4 buses genuinely merge away, 167 -> 163) — unlike
/// FullGrid's own switch-merging (where none of MiniGrid/MicroGrid-BE/
/// RealGrid needed it at all, since their own TP profiles already pre-merge
/// every closed switch), this is the *second* fixture needing it, and the
/// first at a scale where it actually reduces the bus count.
///
/// That merge exposed a real, separate bug (now fixed, not specific to this
/// fixture): `cgmes_common::assert_matches_sv`/`cgmes_realgrid_test.rs`'s own
/// local `assert_voltage_match` both used to resolve `TopologicalNode` mrid
/// -> bus index via `tn_mrids.iter().position(...)` — a *pre-merge* list
/// position — indexed directly into the *post-merge* (and now genuinely
/// smaller) `buses` array. Silent on every fixture that never actually
/// merged anything, but a hard panic here (`buses.len()=163` indexed with a
/// stale position up to 166) the moment a real merge happens. Fixed via a
/// new `cgmes::cgmes_topological_node_bus_index` public function (a proper
/// mrid -> post-merge-index map, not a list position) that both helpers now
/// use.
///
/// Otherwise a clean, well-modeled network — no CGMES class this converter
/// doesn't already handle appears here at all (no `PowerElectronicsConnection`,
/// `EquivalentBranch`, HVDC, or exotic tap-changer variants) — so despite
/// its real-network styling, voltage *magnitude* match against the
/// published `SvVoltage` is excellent (max per-bus error 0.45%, zero buses
/// over 5%). Angle match is looser (`tol=2e-2` rad, vs. MiniGrid/PST's
/// tighter values) — still small (well under a degree at the worst bus) but
/// large enough relative to `assert_matches_sv`'s single shared tolerance
/// (used for both magnitude, in p.u., and angle, in rad) that a magnitude-
/// only tolerance would have been too tight for angle. Still `assert_matches_sv`'s
/// hard per-bus check, not RealGrid's percentile allowance — no bus here
/// has a plausible reason to be a genuine outlier.
#[test]
fn test_cgmes_smallgrid() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/SmallGrid/SmallGrid-Merged");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("SmallGrid_EQBD.xml");
    let eq = dir.join("SmallGrid_EQ.xml");
    let ssh = dir.join("SmallGrid_SSH.xml");
    let tp = dir.join("SmallGrid_TP.xml");
    let sv = dir.join("SmallGrid_SV.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    assert_eq!(tn_mrids.len(), 167, "expected 167 physical TopologicalNode buses before switch merging");
    let expected = cgmes_common::expected_voltages(&ds);

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    // 4 buses genuinely merge away via closed Disconnectors/Breakers.
    assert_eq!(buses.len(), 163, "expected 4 TopologicalNodes to merge via closed switches, 167 -> 163");
    assert_eq!(lines.len(), 185, "expected 185 ACLineSegment lines");
    assert_eq!(transformers.len(), 14, "expected 14 two-winding transformers");
    assert_eq!(shunts.len(), 14, "expected 14 LinearShuntCompensator shunts");

    let n_deenergized = buses.iter().filter(|b| b.bus_type == BusType::Slack && b.voltage_mag == 0.0).count();
    assert_eq!(n_deenergized, 40, "expected 40 de-energized (TopologicalIsland-absent) buses");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv(&result, &bus_index, &expected, 2e-2);
}
