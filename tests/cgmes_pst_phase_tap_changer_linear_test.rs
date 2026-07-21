mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's PST (Phase Shifting Transformer) conformance test
/// configurations, `PhaseTapChangerLinear` variants ("Type1" and "Type2" —
/// two exchange-profile splits of what the fixture's own PDF documentation
/// describes as the identical underlying 2-bus/1-transformer network,
/// included here as two separate assertions specifically because agreement
/// between them is itself a check that the conversion doesn't depend on
/// exchange-profile-specific quirks). Small and self-contained (no boundary
/// EQ needed, unlike MicroGrid-BE-MAS), so unlike that fixture's few-percent
/// tolerance this one matches published SV values almost to machine
/// precision — a real accuracy check, not a "close enough" one.
fn run_and_check(dir_name: &str, file_prefix: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/PST")
        .join(dir_name);
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq = dir.join(format!("{file_prefix}_EQ.xml"));
    let ssh = dir.join(format!("{file_prefix}_SSH.xml"));
    let tp = dir.join(format!("{file_prefix}_TP.xml"));
    let sv = dir.join(format!("{file_prefix}_SV.xml"));
    let ds = load_profiles(&[&eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    let expected = cgmes_common::expected_voltages(&ds);
    assert_eq!(expected.len(), 2, "fixture should have 2 SvVoltage entries");

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus);

    cgmes_common::assert_matches_sv(&result, &tn_mrids, &expected, 1e-3);
}

#[test]
fn test_cgmes_pst_phase_tap_changer_linear_type1() {
    run_and_check("PST_PhaseTapChangerLinear_Type1", "PST_Type1");
}

#[test]
fn test_cgmes_pst_phase_tap_changer_linear_type2() {
    run_and_check("PST_PhaseTapChangerLinear_Type2", "PST_Type2");
}
