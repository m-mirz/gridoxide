mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's MicroGrid conformance test configuration, Dutch area
/// ("MicroGrid-NL-MAS" = Model As Supplied) — the counterpart to
/// `cgmes_microgrid_be_test.rs`'s own BE area, same `MicroGid-BaseCase`
/// bundle, same shared `MicroGrid-BD-MAS` boundary set. Matches published
/// `SvVoltage` noticeably more tightly than the BE area (max ~0.07% here vs.
/// BE's few-percent boundary-truncation gap) — this area's own boundary
/// `ACLineSegment`s apparently don't hit the same nominal-voltage-mismatch
/// case BE's own test doc comment describes.
#[test]
fn test_cgmes_microgrid_nl_mas() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGid-BaseCase/MicroGrid-NL-MAS");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("../MicroGrid-BD-MAS/20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let eq = dir.join("20210325T1530Z_1D_NL_EQ_001.xml");
    let ssh = dir.join("20210325T1530Z_1D_NL_SSH_001.xml");
    let tp = dir.join("20210325T1530Z_1D_NL_TP_001.xml");
    let sv = dir.join("20210325T1530Z_1D_NL_SV_001.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let expected = cgmes_common::expected_voltages(&ds);
    assert_eq!(expected.len(), 3, "fixture should have 3 SvVoltage entries");

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    // 3 physical TopologicalNode buses, plus synthesized 3-winding star
    // points and boundary ConnectivityNode buses.
    assert!(buses.len() > 3, "expected at least the 3 physical TopologicalNode buses, got {}", buses.len());

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv(&result, &bus_index, &expected, 5e-3);
}
