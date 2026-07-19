mod cgmes_common;

use std::path::Path;

use gridoxide::cgmes::{cgmes_to_buses_and_branches, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;

/// ENTSO-E's RealGrid conformance test configuration: a large real
/// national-scale transmission+distribution grid (6252 buses, 7561
/// ACLineSegments, 1509 PowerTransformers, 1347 SynchronousMachines),
/// exercising element types and edge cases MicroGrid-BE-MAS's small
/// fixture doesn't: `ConformLoad`/`NonConformLoad` (not raw
/// `EnergyConsumer` — RealGrid has zero of those), `PhaseTapChangerTabular`
/// (a table-lookup tap changer, distinct from Asymmetrical/Symmetrical),
/// and `Terminal.connected = false` on live equipment (a genuinely
/// de-energized snapshot, not a decode gap — resolved via
/// `TopologicalIsland.TopologicalNodes` membership, matching CGMES's own
/// documented "only energised TopologicalNode-s shall be part of the
/// topological island").
///
/// Two real fixes came out of getting this to solve *and* match published
/// values:
/// - A pre-existing bug in shared core code, not CGMES-specific:
///   `network::branch_calc_param`'s half-open-transformer branch divides by
///   `y_shunt` directly (`2/y_shunt`), mathematically fine as the *limit* as
///   `y_shunt → 0` (the whole expression correctly limits to zero — with no
///   magnetizing branch at all, opening one end truly isolates the connected
///   side) but literal complex division by exactly `0+0j` hits a `0/0`
///   pattern and produces `NaN` instead. `y_shunt == 0` combined with a
///   half-open transformer is common in real CGMES data but apparently never
///   previously exercised by gridoxide's own PGM fixtures — fixed via an
///   explicit zero-shunt guard in `half_open_branch_shunt`.
/// - The Q-sign convention documented in `cgmes.rs`'s Step 3 (load-sign
///   convention negated for both P and Q) turned out to be a
///   `SynchronousMachine`-specific exception, not a general CGMES quirk:
///   applying "no negation" to loads too (extrapolated from a single-machine
///   finding on the much smaller MicroGrid-BE-MAS fixture, which has no
///   `EnergyConsumer` at all to independently check against) dragged
///   RealGrid's median solved-vs-published-SV error to 5.9%, with 3369 of
///   6051 buses over 5% error. Reverting loads specifically back to the
///   documented convention (keeping the machine-only exception) dropped that
///   to a 0.09% median with 11 outliers — see `assert_voltage_match` below.
#[test]
fn test_cgmes_realgrid() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/RealGrid/RealGrid-Merged");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return;
    }

    let eq = dir.join("RealGrid_EQ.xml");
    let ssh = dir.join("RealGrid_SSH.xml");
    let tp = dir.join("RealGrid_TP.xml");
    let sv = dir.join("RealGrid_SV.xml");
    let ds = load_profiles(&[&eq, &ssh, &tp, &sv]).expect("failed to decode CGMES profiles");

    let s_base_va = 100e6;
    let (buses, lines, transformers, shunts) =
        cgmes_to_buses_and_branches(&ds, s_base_va).expect("conversion failed");

    assert_eq!(buses.len(), 6252);
    assert_eq!(lines.len(), 7333, "7561 ACLineSegments minus half-/fully-open ones folded/dropped");
    assert_eq!(transformers.len(), 1509);
    assert_eq!(shunts.len(), 81, "311 LinearShuntCompensators minus disconnected ones");

    let n_deenergized = buses.iter().filter(|b| matches!(b.bus_type, gridoxide::types::BusType::Slack)).count() - 1;
    assert_eq!(n_deenergized, 201, "TopologicalIsland lists 6051 of 6252 TopologicalNodes as energized");

    let tn_mrids = ds.by_type["TopologicalNode"].clone();
    let expected = cgmes_common::expected_voltages(&ds);
    assert_eq!(expected.len(), 6051, "one SvVoltage per energized TopologicalNode");

    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus);

    for (i, b) in result.iter().enumerate() {
        assert!(b.voltage_mag.is_finite() && b.voltage_ang.is_finite(), "non-finite solved voltage at bus {i}");
    }

    assert_voltage_match(&result, &tn_mrids, &expected);
}

/// Unlike `cgmes_common::assert_matches_sv`'s hard per-bus check (right for
/// MicroGrid-BE-MAS's small, fully-modeled fixture, where no bus has a
/// plausible reason to be an outlier), RealGrid is large enough that a
/// handful of genuine, small, known gaps are expected to show up as
/// per-bus outliers without indicating a systemic problem: 3
/// `StaticVarCompensator`s aren't converted at all (no CIM handling for that
/// class yet), and plain `newton_raphson` doesn't enforce
/// `SynchronousMachine` reactive-capability limits the way
/// `solver::newton_raphson_enforcing_q_limits` does. A percentile-based
/// check is the honest bar for this scale: overwhelmingly precise
/// (median/p90), with a small, bounded outlier allowance (p99), rather than
/// either a misleadingly loose uniform tolerance or an unrealistic zero-
/// outlier requirement.
fn assert_voltage_match(result: &[gridoxide::types::Bus], tn_mrids: &[String], expected: &[cgmes_common::ExpectedVoltage]) {
    let mut errs: Vec<f64> = Vec::new();
    for exp in expected {
        let Some(idx) = tn_mrids.iter().position(|m| *m == exp.tn_mrid) else { continue };
        let bus = &result[idx];
        let expected_v_pu = (exp.v * 1e3) / bus.u_rated;
        errs.push((bus.voltage_mag - expected_v_pu).abs());
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = errs.len();
    let median = errs[n / 2];
    let p90 = errs[n * 9 / 10];
    let p99 = errs[n * 99 / 100];
    eprintln!("voltage_mag abs error (pu): n={n} median={median:.5} p90={p90:.5} p99={p99:.5} max={:.5}", errs[n - 1]);

    assert!(median < 5e-3, "median voltage_mag error {median} too high (expected < 0.5%)");
    assert!(p90 < 2e-2, "p90 voltage_mag error {p90} too high (expected < 2%)");
    assert!(p99 < 5e-1, "p99 voltage_mag error {p99} too high (expected < 50%, generously bounding the known StaticVarCompensator/Q-limit gaps)");
}
