mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's MicroGrid "Type2" conformance test configuration, Belgian area
/// — another different scenario/snapshot from both `cgmes_microgrid_be_test.rs`
/// (`MicroGid-BaseCase`) and `cgmes_microgrid_type1_test.rs` (`Type1`), using
/// its own `MicroGrid-Type2-BD-MAS` boundary set (the same one
/// `cgmes_microgrid_hvdc_test.rs`'s HVDC fixture shares, confirming this
/// area and that one belong to the same wider Type2 scenario).
#[test]
fn test_cgmes_microgrid_type2_be_mas() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type2/MicroGrid-Type2-BE-MAS");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("../MicroGrid-Type2-BD-MAS/20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let eq = dir.join("20210401T1730Z_1D_BE_EQ_1.xml");
    let ssh = dir.join("20210401T1730Z_1D_BE_SSH_1.xml");
    let tp = dir.join("20210401T1730Z_1D_BE_TP_1.xml");
    let sv = dir.join("20210401T1730Z_1D_BE_SV_1.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let expected = cgmes_common::expected_voltages(&ds);
    assert_eq!(expected.len(), 11, "fixture should have 11 SvVoltage entries");

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    assert!(buses.len() > 11, "expected at least the 11 physical TopologicalNode buses, got {}", buses.len());
    assert_eq!(lines.len(), 10, "expected 10 ACLineSegment lines");
    assert_eq!(transformers.len(), 8, "expected 8 two-winding transformers");
    assert_eq!(shunts.len(), 4, "expected 4 LinearShuntCompensator shunts");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv(&result, &bus_index, &expected, 1e-1);
}

/// The Dutch-area counterpart — unlike `Type1`'s own NL-MAS (which fails to
/// convert standalone at all, a genuine data gap), this one converts and
/// matches published `SvVoltage` tightly (max ~0.08%).
#[test]
fn test_cgmes_microgrid_type2_nl_mas() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type2/MicroGrid-Type2-NL-MAS");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("../MicroGrid-Type2-BD-MAS/20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let eq = dir.join("20210401T1730Z_1D_NL_EQ_1.xml");
    let ssh = dir.join("20210401T1730Z_1D_NL_SSH_1.xml");
    let tp = dir.join("20210401T1730Z_1D_NL_TP_1.xml");
    let sv = dir.join("20210401T1730Z_1D_NL_SV_1.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let expected = cgmes_common::expected_voltages(&ds);
    assert_eq!(expected.len(), 5, "fixture should have 5 SvVoltage entries");

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    assert!(buses.len() > 5, "expected at least the 5 physical TopologicalNode buses, got {}", buses.len());
    assert_eq!(lines.len(), 5, "expected 5 ACLineSegment lines");
    assert_eq!(transformers.len(), 3, "expected 3 two-winding transformers");
    assert_eq!(shunts.len(), 1, "expected 1 LinearShuntCompensator shunt");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv(&result, &bus_index, &expected, 5e-3);
}

/// The "Merged" scenario: BE + NL + HVDC areas' equipment all loaded
/// together against one shared "ASSEMBLED" TP/SV (26 buses total —
/// confirmed via a direct search that some of the assembled solution's own
/// `TopologicalNode`s only resolve once the HVDC area's own EQ/SSH are
/// loaded alongside BE/NL's, unlike `Type1`'s own two-area-only Merged
/// case). AC-only here (no `cgmes_resolve_dc_converters` call): this
/// specific merged bundle pulls in a *different* `CsConverter` instance
/// (not either of `cgmes_microgrid_hvdc_test.rs`'s own two `VsConverter`s)
/// missing its own required `baseS` field — a real, separate data gap in
/// this specific file combination, out of scope here since this test's own
/// purpose is validating multi-area AC profile merging, not DC resolution
/// (already covered by the dedicated HVDC fixture).
#[test]
fn test_cgmes_microgrid_type2_merged() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type2/MicroGrid-Type2-Merged");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let be_eq = dir.join("20210401T1730Z_1D_BE_EQ_1.xml");
    let be_ssh = dir.join("20210401T1730Z_1D_BE_SSH_1.xml");
    let nl_eq = dir.join("20210401T1730Z_1D_NL_EQ_1.xml");
    let nl_ssh = dir.join("20210401T1730Z_1D_NL_SSH_1.xml");
    let hvdc_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type2/MicroGrid-Type2-HVDC-MAS");
    let hvdc_eq = hvdc_dir.join("20210401T1730Z_1D_HVDC_EQ_1.xml");
    let hvdc_ssh = hvdc_dir.join("20210401T1730Z_1D_HVDC_SSH_1.xml");
    let tp = dir.join("20210401T1730Z_1D_ASSEMBLED_TP_1.xml");
    let sv = dir.join("20210401T1730Z_1D_ASSEMBLED_SV_1.xml");
    let ds = load_profiles(&[&eq_bd, &be_eq, &be_ssh, &nl_eq, &nl_ssh, &hvdc_eq, &hvdc_ssh, &tp, &sv])
        .expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    assert_eq!(tn_mrids.len(), 26, "expected 26 physical TopologicalNode buses across all three areas");
    let expected = cgmes_common::expected_voltages(&ds);

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    assert_eq!(buses.len(), 26);
    assert_eq!(lines.len(), 17, "expected 17 ACLineSegment lines");
    assert_eq!(transformers.len(), 13, "expected 13 two-winding transformers");
    assert_eq!(shunts.len(), 5, "expected 5 LinearShuntCompensator shunts");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    // Looser than the single-area tests above: the HVDC converters' own AC
    // injections come from their static SSH values here, not a resolved DC
    // network solve (see this test's own doc comment).
    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv(&result, &bus_index, &expected, 3e-1);
}
