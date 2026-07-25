mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's PowerFlow conformance test configuration — a tiny (2-bus,
/// 1-line, 1-transformer), self-contained (no boundary EQ needed) fixture
/// specifically exercising `PhaseTapChangerTabular`/`PhaseTapChangerTable`
/// (both already handled) and a `SynchronousMachine` with a
/// `ReactiveCapabilityCurve` (a P-dependent Q-limit curve this converter
/// doesn't read — falls back to unlimited Q, harmless here since nothing in
/// this test enforces Q limits at all). Converges to a near-exact match
/// against the published `SvVoltage` (both buses' error rounds to 0.0000),
/// matching `cgmes_pst_phase_tap_changer_linear_test.rs`'s own precedent for
/// small, self-contained, fully-modeled fixtures.
#[test]
fn test_cgmes_powerflow() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/PowerFlow/PowerFlow");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq = dir.join("PowerFlow_EQ.xml");
    let ssh = dir.join("PowerFlow_SSH.xml");
    let tp = dir.join("PowerFlow_TP.xml");
    let sv = dir.join("PowerFlow_SV.xml");
    let ds = load_profiles(&[&eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    assert_eq!(tn_mrids.len(), 2, "expected 2 physical TopologicalNode buses");
    let expected = cgmes_common::expected_voltages(&ds);
    assert_eq!(expected.len(), 2, "fixture should have 2 SvVoltage entries");

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    assert_eq!(buses.len(), 2);
    assert_eq!(lines.len(), 1, "expected 1 ACLineSegment line");
    assert_eq!(transformers.len(), 1, "expected 1 two-winding transformer");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv(&result, &bus_index, &expected, 1e-3);
}
