mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_resolve_dc_converters, cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's MicroGrid Type2 HVDC conformance test configuration — a single,
/// self-contained point-to-point VSC-VSC HVDC link (2 `VsConverter`, 1
/// `DCLineSegment`, no switchyard clutter at all: no `DCSwitch`/`DCBreaker`/
/// `DCDisconnector`/`DCChopper`/`DCBusbar`/`DCShunt`/`DCSeriesDevice`), a much
/// smaller and cleaner HVDC fixture than FullGrid's own — the first fixture
/// this converter's `cgmes_resolve_dc_converters` (its HVDC support) is
/// validated against end-to-end through a full CGMES round-trip.
///
/// One `VsConverter` is a DC-voltage slack (`pPccControl=udc`,
/// `targetUdc=150`), the other a fixed-AC-power follower
/// (`pPccControl=pPcc`, `targetPpcc=150`) — exercising the loss-curve
/// self-consistency loop in `cgmes_resolve_dc_converters`, not just the
/// trivial `UdcSlack`/`FixedIdc` static-target paths RealGrid-scale fixtures
/// don't exercise at all (no CSC/`dcCurrent` link here). Also exposed a real
/// gap, now fixed: `ACDCConverter.PccTerminal` is optional in CIM and is
/// genuinely absent from both converters here — `cgmes_resolve_dc_converters`
/// now falls back to the converter's own regular `Terminal` (sequence 1)
/// when it's unset, the same "point of common coupling defaults to the
/// equipment's own terminal" convention CIM uses elsewhere.
///
/// Each converter's AC terminal sits in a *different* electrical area (VSC1
/// on `_246d0822...`, VSC_2 on `_e0741c5c...`) — expected for a VSC-HVDC
/// link, which by design provides no AC-domain synchronism between its two
/// ends. The fixture's own single `TopologicalIsland` declares only *one*
/// `AngleRefTopologicalNode` (on VSC1's side), so the VSC_2 side has no local
/// angle reference of its own in this Model-As-Supplied export and is left
/// de-energized by gridoxide's `NoReferenceBus` handling — the same
/// "boundary-truncated sub-model" limitation `cgmes_microgrid_be_test.rs`
/// already documents for its own fixture, not a new gap. Only the
/// referenced (VSC1) side is checked against the published `SvVoltage`
/// here; asserting the un-referenced side would just be checking gridoxide's
/// own zeroing behavior against nonzero published values.
///
/// Tolerance matches `cgmes_microgrid_be_test.rs`'s own (`4e-2`) for the
/// same reason: a small, boundary-truncated area solved with a fixed-P/Q
/// `EquivalentInjection` standing in for the rest of the interconnected
/// system on VSC1's own side.
#[test]
fn test_cgmes_microgrid_type2_hvdc() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type2/MicroGrid-Type2-HVDC-MAS");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq_bd = dir.join("../MicroGrid-Type2-BD-MAS/20171002T0930Z_ENTSO-E_EQ_BD_2.xml");
    let eq = dir.join("20210401T1730Z_1D_HVDC_EQ_1.xml");
    let ssh = dir.join("20210401T1730Z_1D_HVDC_SSH_1.xml");
    let tp = dir.join("20210401T1730Z_1D_HVDC_TP_1.xml");
    let sv = dir.join("20210401T1730Z_1D_HVDC_SV_1.xml");
    let ds = load_profiles(&[&eq_bd, &eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    assert_eq!(tn_mrids.len(), 4, "expected 4 physical AC TopologicalNode buses");
    let expected = cgmes_common::expected_voltages(&ds);

    let s_base_va = 100e6;
    let (mut buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("AC conversion failed");
    assert_eq!(lines.len(), 2, "expected 2 ACLineSegment lines");
    assert_eq!(transformers.len(), 2, "expected 2 two-winding transformers");

    let dc = cgmes_resolve_dc_converters(&ds, &mut buses, s_base_va)
        .expect("DC resolution failed")
        .expect("expected HVDC equipment in this fixture");
    assert!(dc.status.converged, "DC network solve did not converge: {:?}", dc.status);
    assert!(dc.status.isolated_buses.is_empty(), "expected no isolated DC buses, got {:?}", dc.status.isolated_buses);
    assert_eq!(dc.dc_bus_mrids.len(), 4, "expected 4 DCTopologicalNode buses (2 positive poles + 2 grounded middle poles)");
    // The UdcSlack side should have solved to (very close to) its own fixed
    // 150 kV target; the FixedP follower side, one line resistance away,
    // solves to something close by but not identical.
    let slack_v = dc.voltages_kv.iter().find(|&&v| (v - 150.0).abs() < 1e-6);
    assert!(slack_v.is_some(), "expected one DC bus solved at exactly the 150 kV UdcSlack target, got {:?}", dc.voltages_kv);

    let mut ybus = build_ybus(buses.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    // Only VSC1's own area has a local angle reference in this fixture (see
    // the module doc comment) — filter to just its two physical buses.
    let vsc1_area: [&str; 2] = ["_246d0822-a3f4-4894-9c22-22c9cfd1104c", "_da2d770e-6ace-47a7-8274-2fbb4cc0d52c"];
    let expected_vsc1_area: Vec<_> = expected.into_iter().filter(|e| vsc1_area.contains(&e.tn_mrid.as_str())).collect();
    assert_eq!(expected_vsc1_area.len(), 2, "expected published SvVoltage for both of VSC1's own area buses");
    let bus_index = cgmes_topological_node_bus_index(&ds).expect("bus index lookup failed");
    cgmes_common::assert_matches_sv(&result, &bus_index, &expected_vsc1_area, 4e-2);
}
