mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's PST (Phase Shifting Transformer) conformance test
/// configuration, `PhaseTapChangerTable`/`PhaseTapChangerTabular` variant
/// ("Type3" — a table-lookup phase tap changer, distinct from the
/// `PhaseTapChangerLinear` fixtures `cgmes_pst_phase_tap_changer_linear_test.rs`
/// already covers). Same 2-bus/1-transformer shape as those, small and
/// self-contained (no boundary EQ needed), matching published SV values
/// almost to machine precision.
#[test]
fn test_cgmes_pst_phase_tap_changer_table_type3() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/PST/PST_PhaseTapChangerTable_Type3");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq = dir.join("PST_Type3_EQ.xml");
    let ssh = dir.join("PST_Type3_SSH.xml");
    let tp = dir.join("PST_Type3_TP.xml");
    let sv = dir.join("PST_Type3_SV.xml");
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
