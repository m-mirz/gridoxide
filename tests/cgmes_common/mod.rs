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
pub fn assert_matches_sv(
    solved: &[Bus], tn_mrids: &[String], expected: &[ExpectedVoltage], tol: f64,
) {
    for exp in expected {
        let Some(idx) = tn_mrids.iter().position(|m| *m == exp.tn_mrid) else { continue };
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

pub fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGid-BaseCase/MicroGrid-BE-MAS")
}
