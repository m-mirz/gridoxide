use std::path::Path;

use gridoxide::cgmes::CimDataset;
use gridoxide::types::Bus;

/// A single `SvVoltage` entry from a CGMES SV profile, keyed by the
/// `TopologicalNode` it belongs to.
pub struct ExpectedVoltage {
    pub tn_mrid: String,
    pub v: f64,
    pub angle_deg: f64,
}

/// Reads every `SvVoltage` entry out of an already-decoded dataset.
pub fn expected_voltages(ds: &CimDataset) -> Vec<ExpectedVoltage> {
    let mut out = Vec::new();
    let Some(mrids) = ds.by_type.get("SvVoltage") else { return out };
    for mrid in mrids {
        let sv: &cimstructs::SvVoltage = ds.entries[mrid]
            .element
            .as_any()
            .downcast_ref()
            .expect("SvVoltage downcast");
        let (Some(tn), Some(v), Some(angle_deg)) = (&sv.topological_node, sv.v, sv.angle) else { continue };
        out.push(ExpectedVoltage { tn_mrid: tn.mrid.clone(), v, angle_deg });
    }
    out
}

/// Asserts every solved bus matches its fixture's own published `SvVoltage`
/// within `tol` (relative to the bus's own rated voltage, so the tolerance is
/// in per-unit regardless of voltage level) — the CGMES analogue of
/// `tests/common`'s `assert_sym_node` for PGM fixtures.
///
/// `bus_index` must be `cgmes::cgmes_topological_node_bus_index(ds)`'s own
/// output, *not* a `by_type(ds, "TopologicalNode")` list position — bus
/// indices aren't 1:1 with that list's own order once
/// `cgmes_to_buses_and_branches`'s closed-switch merge (Step 2.5) combines
/// any `TopologicalNode`s together, confirmed a real, live bug on SmallGrid
/// (838 `Disconnector`s + 427 `Breaker`s genuinely merge some buses): the
/// old plain `tn_mrids.iter().position(...)` pattern this function used to
/// use either panics (position exceeds the now-smaller `solved.len()`) or,
/// worse, silently compares against the wrong bus.
pub fn assert_matches_sv(
    solved: &[Bus], bus_index: &std::collections::HashMap<String, usize>, expected: &[ExpectedVoltage], tol: f64,
) {
    for exp in expected {
        let Some(&idx) = bus_index.get(&exp.tn_mrid) else { continue };
        let bus = &solved[idx];
        // SvVoltage.v is in kV; bus.u_rated is in V (gridoxide's own convention).
        let expected_v_pu = (exp.v * 1e3) / bus.u_rated;
        assert!(
            (bus.voltage_mag - expected_v_pu).abs() < tol,
            "TopologicalNode {}: voltage_mag {} vs expected {} (tol {tol})",
            exp.tn_mrid, bus.voltage_mag, expected_v_pu,
        );
        let expected_ang_rad = exp.angle_deg.to_radians();
        let diff = (bus.voltage_ang - expected_ang_rad).rem_euclid(2.0 * std::f64::consts::PI);
        let diff = if diff > std::f64::consts::PI { diff - 2.0 * std::f64::consts::PI } else { diff };
        assert!(
            diff.abs() < tol,
            "TopologicalNode {}: voltage_ang {} vs expected {} (tol {tol})",
            exp.tn_mrid, bus.voltage_ang, expected_ang_rad,
        );
    }
}

/// Percentile-based voltage match, for fixtures large enough that a
/// handful of small, known per-bus gaps are expected without indicating a
/// systemic problem (plain `newton_raphson` doesn't enforce
/// `SynchronousMachine`/`StaticVarCompensator` reactive-capability limits
/// the way `solver::newton_raphson_enforcing_q_limits` does) — unlike
/// `assert_matches_sv`'s hard per-bus check, right only for small, fully-
/// modeled fixtures where no bus has a plausible reason to be an outlier.
/// Originally local to `cgmes_realgrid_test.rs`; promoted here once
/// SmallGrid/Svedala needed the identical shape rather than a second copy.
///
/// Checks `voltage_mag` only (not angle, unlike `assert_matches_sv` — real,
/// large fixtures' angle references are less directly comparable at scale).
/// `median_tol`/`p90_tol`/`p99_tol` are the fixture's own empirically-tuned
/// thresholds, not shared constants — each large fixture's own known-gap
/// profile differs.
pub fn assert_matches_sv_percentile(
    solved: &[Bus], bus_index: &std::collections::HashMap<String, usize>, expected: &[ExpectedVoltage],
    median_tol: f64, p90_tol: f64, p99_tol: f64,
) {
    let mut errs: Vec<f64> = Vec::new();
    for exp in expected {
        let Some(&idx) = bus_index.get(&exp.tn_mrid) else { continue };
        let bus = &solved[idx];
        let expected_v_pu = (exp.v * 1e3) / bus.u_rated;
        errs.push((bus.voltage_mag - expected_v_pu).abs());
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = errs.len();
    let median = errs[n / 2];
    let p90 = errs[n * 9 / 10];
    let p99 = errs[n * 99 / 100];
    eprintln!("voltage_mag abs error (pu): n={n} median={median:.5} p90={p90:.5} p99={p99:.5} max={:.5}", errs[n - 1]);

    assert!(median < median_tol, "median voltage_mag error {median} too high (expected < {median_tol})");
    assert!(p90 < p90_tol, "p90 voltage_mag error {p90} too high (expected < {p90_tol})");
    assert!(p99 < p99_tol, "p99 voltage_mag error {p99} too high (expected < {p99_tol})");
}

pub fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGid-BaseCase/MicroGrid-BE-MAS")
}
