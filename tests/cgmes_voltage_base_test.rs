//! Giving each galvanically-connected group of buses one voltage base.
//!
//! An `ACLineSegment` has no ratio, so its two ends are one conductor at one
//! physical voltage. Per-unit only means anything if `|V| = 1.0` is the same
//! volts at both — that is what a base is. CGMES does not guarantee it: the
//! same level is 380 kV in Belgium and 400 kV in the Netherlands, 220 and 225
//! either side of another border, and a tie line between them carries two
//! different `BaseVoltage.nominalVoltage` values on one wire.
//!
//! What that costs is measured here rather than asserted, because the number is
//! the argument: before this, MicroGrid-Type1 solved 4.4% above its own
//! published solution.

mod cgmes_common;

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_to_network, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::solver::{PowerFlowOptions, SolveStatus};

const S_BASE_VA: f64 = 100e6;

fn config(name: &str) -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(name);
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init \
             tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

fn profiles(dir: &Path) -> Vec<PathBuf> {
    // The four a power flow is built from. DL, GL and DY sit in the same
    // directory and are not network data.
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "xml"))
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_uppercase();
            ["_EQ", "_SSH", "_TP", "_SV", "EQ_BD"].iter().any(|k| n.contains(k))
        })
        .collect();
    paths.sort();
    paths
}

fn load(name: &str) -> Option<(gridoxide::cgmes::CimDataset, gridoxide::cgmes::CgmesNetwork)> {
    let dir = config(name)?;
    let paths = profiles(&dir);
    let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    let ds = load_profiles(&refs).expect("decode");
    let net = cgmes_to_network(&ds, S_BASE_VA).expect("convert");
    Some((ds, net))
}

/// **The invariant.** After conversion, no line joins two buses that disagree
/// about their voltage base — on any fixture.
///
/// This is the property, and it is worth stating as one rather than as "the
/// cross-border fixtures are fixed": a line between two bases is not a
/// modelling approximation, it is a per-unit system that does not close.
#[test]
fn no_line_spans_two_voltage_bases() {
    for name in [
        "MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged",
        "FullGrid/FullGrid-Merged",
        "SmallGrid/SmallGrid-Merged",
        "MiniGrid/MiniGrid-Merged",
        "RealGrid/RealGrid-Merged",
        "Svedala/Svedala-Merged",
    ] {
        let Some((_, net)) = load(name) else { continue };
        for (i, l) in net.lines.iter().enumerate() {
            let (a, b) = (net.buses[l.from].u_rated, net.buses[l.to].u_rated);
            if a <= 0.0 || b <= 0.0 {
                continue;
            }
            assert!(
                (a - b).abs() / a.max(b) < 1e-9,
                "{name}: line {i} joins bus {} at {:.1} kV to bus {} at {:.1} kV — one conductor \
                 cannot be two voltage levels, and per-unit across it does not close",
                l.from,
                a / 1e3,
                l.to,
                b / 1e3
            );
        }
    }
}

/// Which fixtures needed it, recorded. The cross-border merged ones do; the
/// single-TSO ones do not and must come through untouched.
#[test]
fn only_the_cross_border_fixtures_needed_harmonizing() {
    /// A fixture, how many buses it should move, and which nominals it should
    /// merge, as `(kept kV, replaced kV)`.
    struct Case {
        fixture: &'static str,
        buses: usize,
        merged: &'static [(u64, u64)],
    }
    let expected = [
        Case { fixture: "MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged", buses: 4,
               merged: &[(400, 380), (225, 220)] },
        Case { fixture: "FullGrid/FullGrid-Merged", buses: 3,
               merged: &[(400, 380), (225, 220)] },
        Case { fixture: "SmallGrid/SmallGrid-Merged", buses: 0, merged: &[] },
        Case { fixture: "MiniGrid/MiniGrid-Merged", buses: 0, merged: &[] },
        Case { fixture: "RealGrid/RealGrid-Merged", buses: 0, merged: &[] },
        Case { fixture: "Svedala/Svedala-Merged", buses: 0, merged: &[] },
    ];

    for Case { fixture: name, buses, merged: pairs } in expected {
        let Some((_, net)) = load(name) else { continue };
        let r = &net.base_harmonization;
        assert_eq!(
            r.buses, buses,
            "{name}: {} bus(es) moved, expected {buses}; merged {:?}",
            r.buses, r.merged
        );
        for (keep, replaced) in pairs {
            assert!(
                r.merged
                    .iter()
                    .any(|(k, rp, _)| k.round() as u64 == *keep && rp.round() as u64 == *replaced),
                "{name}: expected {replaced} kV to be merged onto {keep} kV, got {:?}",
                r.merged
            );
        }
        if pairs.is_empty() {
            assert_eq!(r.groups, 0, "{name}: nothing should have been harmonized");
        }
    }
}

/// The volts are what the document states; only the per-unit numbers move.
///
/// A base is a choice, not a measurement, so harmonizing must leave every
/// reported voltage where it was. Checked against the fixture's own published
/// `SvVoltage`, in kV.
#[test]
fn harmonizing_moves_the_per_unit_numbers_and_not_the_volts() {
    let Some((ds, net)) = load("MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged") else { return };
    let idx = cgmes_topological_node_bus_index(&ds).expect("bus index");

    let report = gridoxide::run_power_flow(
        net.buses.clone(),
        &net.lines,
        &net.transformers,
        &net.shunts,
        gridoxide::TapData::none(),
        PowerFlowOptions { enforce_q_limits: true, max_iter: 60, max_outer_iter: 60, ..Default::default() },
    );
    assert_eq!(report.stats.status, SolveStatus::Converged);

    // Buses that were moved onto another base still report the volts the
    // document publishes for them — the per-unit value absorbed the change.
    let mut worst: f64 = 0.0;
    for e in cgmes_common::expected_voltages(&ds) {
        let Some(&b) = idx.get(&e.tn_mrid) else { continue };
        if net.buses[b].u_rated <= 0.0 {
            continue;
        }
        let kv = report.buses[b].voltage_mag * net.buses[b].u_rated / 1e3;
        worst = worst.max((kv - e.v).abs() / e.v.abs().max(1e-9));
    }
    // Recorded. Before harmonizing this fixture solved 4.4% high with a 0.96%
    // median; the assertion below would have failed by a factor of three.
    assert!(
        worst < 0.015,
        "worst voltage error against the published solution is {worst:.4}, where harmonizing \
         the bases should hold it near 0.013"
    );
}
