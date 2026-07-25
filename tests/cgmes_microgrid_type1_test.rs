mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's MicroGrid "Type1" conformance test configuration, Belgian area
/// — a different scenario/snapshot from `cgmes_microgrid_be_test.rs`'s own
/// `MicroGid-BaseCase/MicroGrid-BE-MAS` fixture (same shared boundary set,
/// different `MicroGrid-Type1-BE-MAS` directory/snapshot).
///
/// `MicroGrid-Type1-NL-MAS` (the Dutch-area counterpart) is deliberately
/// *not* tested here: loaded standalone, it genuinely fails to convert —
/// `TopologicalNode "NL_TR_BUS2"` references a `BaseVoltage` mrid that
/// isn't defined anywhere in this area's own EQ file *or* the shared
/// boundary set (confirmed via `grep -rl` across the whole `MicroGrid-Type1`
/// submodule tree: zero matches). Not a gridoxide bug — a genuinely
/// incomplete standalone "Model As Supplied" submission; the same reference
/// resolves fine once BE's own EQ is loaded alongside it (see
/// `test_cgmes_microgrid_type1_merged` below, which needs both areas).
#[test]
fn test_cgmes_microgrid_type1_be_mas() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type1/MicroGrid-Type1-BE-MAS");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("../MicroGrid-BD-MAS/20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let eq = dir.join("20210323T1730Z_1D_BE_EQ_1.xml");
    let ssh = dir.join("20210323T1730Z_1D_BE_SSH_1.xml");
    let tp = dir.join("20210323T1730Z_1D_BE_TP_1.xml");
    let sv = dir.join("20210323T1730Z_1D_BE_SV_1.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let expected = cgmes_common::expected_voltages(&ds);
    assert_eq!(expected.len(), 7, "fixture should have 7 SvVoltage entries");

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    assert!(buses.len() > 7, "expected at least the 7 physical TopologicalNode buses, got {}", buses.len());
    assert_eq!(lines.len(), 8, "expected 8 ACLineSegment lines");
    assert_eq!(transformers.len(), 6, "expected 6 two-winding transformers");
    assert_eq!(shunts.len(), 2, "expected 2 LinearShuntCompensator shunts");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv(&result, &bus_index, &expected, 3e-2);
}

/// The "Merged" scenario: both `MicroGrid-Type1-BE-MAS` and
/// `MicroGrid-Type1-NL-MAS`'s own equipment (EQ/SSH) loaded together against
/// one shared "ASSEMBLED" TP/SV — the genuinely two-area, cross-boundary
/// merged model neither area's own standalone submission fully represents
/// alone (see this file's top doc comment on why NL-MAS alone doesn't even
/// convert). The most complete single-model check of this converter's
/// multi-file profile-merging (`load_profiles`'s own MRID-keyed dataset
/// union) available in this test suite.
#[test]
fn test_cgmes_microgrid_type1_merged() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let be_eq = dir.join("20210323T1730Z_1D_BE_EQ_1.xml");
    let be_ssh = dir.join("20210323T1730Z_1D_BE_SSH_1.xml");
    let nl_eq = dir.join("20210323T1730Z_1D_NL_EQ_1.xml");
    let nl_ssh = dir.join("20210323T1730Z_1D_NL_SSH_1.xml");
    let tp = dir.join("20210323T1730Z_1D_ASSEMBLED_TP_1.xml");
    let sv = dir.join("20210323T1730Z_1D_ASSEMBLED_SV_1.xml");
    let ds = load_profiles(&[&eq_bd, &be_eq, &be_ssh, &nl_eq, &nl_ssh, &tp, &sv])
        .expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    assert_eq!(tn_mrids.len(), 17, "expected 17 physical TopologicalNode buses across both areas");
    let expected = cgmes_common::expected_voltages(&ds);

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    assert_eq!(buses.len(), 17);
    assert_eq!(lines.len(), 13, "expected 13 ACLineSegment lines");
    assert_eq!(transformers.len(), 9, "expected 9 two-winding transformers");
    assert_eq!(shunts.len(), 3, "expected 3 LinearShuntCompensator shunts");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv(&result, &bus_index, &expected, 5e-2);
}
