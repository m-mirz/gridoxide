mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's MicroGrid "Type3" conformance test configuration — a 24-hour
/// time series (22:30 through the next day's 21:30, hourly), split into
/// `IGMs` (Individual Grid Models: BE and NL each solved on their own, one
/// EQ per area shared across all 24 hours — only `SSH`/`TP`/`SV` vary
/// hour-to-hour) and `CGMs` (Common Grid Models: BE+NL merged, against a
/// shared "ASSEMBLED" `TP`/`SV` plus an hour-specific BE `SSH` override —
/// `CGMs` itself carries no `EQ` at all, reusing `IGMs`' own). Testing all
/// 24 hours would mean 24x the assertions for no real additional coverage
/// (same topology, same converter code paths, just a different operating
/// point) — this tests the first hour (22:30) only, as representative.
///
/// Same underlying BE/NL areas `cgmes_microgrid_type1_test.rs`'s own
/// `Type1-BE-MAS`/`-NL-MAS`/`-Merged` fixtures use (confirmed: identical
/// `TopologicalNode` mrids appear in both), just a different scenario/time
/// series built on the same base topology.
#[test]
fn test_cgmes_microgrid_type3_igm_be() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type3/IGMs");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let eq = dir.join("20210422T2230Z_1D_BE_EQ_001.xml");
    let ssh = dir.join("20210422T2230Z_1D_BE_SSH_001.xml");
    let tp = dir.join("20210422T2230Z_1D_BE_TP_001.xml");
    let sv = dir.join("20210422T2230Z_1D_BE_SV_001.xml");
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

#[test]
fn test_cgmes_microgrid_type3_igm_nl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type3/IGMs");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let eq = dir.join("20210422T2230Z_1D_NL_EQ_001.xml");
    let ssh = dir.join("20210422T2230Z_1D_NL_SSH_001.xml");
    let tp = dir.join("20210422T2230Z_1D_NL_TP_001.xml");
    let sv = dir.join("20210422T2230Z_1D_NL_SV_001.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let expected = cgmes_common::expected_voltages(&ds);
    assert_eq!(expected.len(), 3, "fixture should have 3 SvVoltage entries");

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    assert!(buses.len() > 3, "expected at least the 3 physical TopologicalNode buses, got {}", buses.len());
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

/// The "CGM" (Common Grid Model, merged) scenario for the same 22:30 hour:
/// both areas' own `EQ` (from `IGMs`, since `CGMs` itself carries none) plus
/// NL's unmodified `IGMs`-own `SSH` and BE's `CGMs`-specific `SSH` override,
/// against the shared "ASSEMBLED" `TP`/`SV`.
#[test]
fn test_cgmes_microgrid_type3_cgm() {
    let igm_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type3/IGMs");
    let cgm_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type3/CGMs");
    if !igm_dir.exists() || !cgm_dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            igm_dir.display()
        );
        return;
    }

    let eq_bd = igm_dir.join("20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let be_eq = igm_dir.join("20210422T2230Z_1D_BE_EQ_001.xml");
    let nl_eq = igm_dir.join("20210422T2230Z_1D_NL_EQ_001.xml");
    let nl_ssh = igm_dir.join("20210422T2230Z_1D_NL_SSH_001.xml");
    let be_ssh = cgm_dir.join("20210422T2230Z_1D_BE_SSH_002.xml");
    let tp = cgm_dir.join("20210422T2230Z_1D_ASSEMBLED_TP_001.xml");
    let sv = cgm_dir.join("20210422T2230Z_1D_ASSEMBLED_SV_001.xml");
    let ds = load_profiles(&[&eq_bd, &be_eq, &nl_eq, &be_ssh, &nl_ssh, &tp, &sv])
        .expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    assert_eq!(tn_mrids.len(), 15, "expected 15 physical TopologicalNode buses across both areas");
    let expected = cgmes_common::expected_voltages(&ds);

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");
    assert_eq!(buses.len(), 16, "expected 1 synthesized bus beyond the 15 physical ones");
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
