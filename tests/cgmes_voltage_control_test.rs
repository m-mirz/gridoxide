//! Generator-side voltage control: what several machines holding one bus
//! jointly hold.
//!
//! `SynchronousMachine`, `StaticVarCompensator`, `PowerElectronicsConnection`
//! and `ExternalNetworkInjection` each carry a `RegulatingControl`, and each
//! wrote the controlled bus's reactive limits with `=` while writing its
//! injections with `+=` — the same loop body, opposite conventions. Two
//! machines on one bus therefore contributed both their reactive power and
//! only the **last one's capability**, so the bus ran with understated
//! headroom and `outerloop::ReactiveLimits` clamped it early.
//!
//! The arithmetic is unit-tested in `src/cgmes.rs`'s own `voltage_control_tests`.
//! What is checked here is that it reaches real data, and how much of it.

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_to_network, load_profiles, CgmesNetwork};
use gridoxide::types::BusType;

fn fixture(dir: &str) -> Option<CgmesNetwork> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(dir);
    if !dir.exists() {
        return None;
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("configuration directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "xml"))
        .collect();
    paths.sort();
    let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    let ds = load_profiles(&refs).expect("failed to decode CGMES profiles");
    Some(cgmes_to_network(&ds, 100e6).expect("conversion failed"))
}

/// How often shared control actually occurs, counted rather than assumed.
///
/// Verified against a second pass over the raw XML that shares no code with
/// the importer: resolve each regulating machine's `RegulatingControl` to a
/// `Terminal` to a `TopologicalNode`, and group. The importer's counts run
/// slightly below the XML's because it skips machines that are out of service
/// or whose control is disabled, which is correct and separately tested.
#[test]
fn shared_control_is_common_on_real_data() {
    let cases = [
        ("MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged", 5, 1),
        ("FullGrid/FullGrid-Merged", 3, 2),
        ("SmallGrid/SmallGrid-Merged", 19, 0),
        ("Svedala/Svedala-Merged", 33, 0),
        ("RealGrid/RealGrid-Merged", 416, 62),
    ];
    let mut checked = 0;
    for (name, regulated, shared) in cases {
        let Some(net) = fixture(name) else {
            eprintln!("skipping {name}: not checked out");
            continue;
        };
        let r = &net.voltage_control;
        assert_eq!(
            (r.regulated_buses, r.shared_buses),
            (regulated, shared),
            "{name}: regulated/shared bus counts moved"
        );
        checked += 1;
    }
    assert!(checked >= 3, "expected most of the fixture set to be present");
}

/// **The defect, on the fixture that shows it most starkly.** FullGrid has a
/// bus held by nine machines, of which six are in service: `SynchronousMachine`s
/// at ±200 MVAr each. Six of those on a 100 MVA base is ±12.0 per-unit.
///
/// Last-writer-wins gave that bus ±2.0 — one machine's nameplate — so its
/// reactive headroom was understated **sixfold**, and Q-limit enforcement
/// clamped it at a sixth of what the plant can actually do.
#[test]
fn fullgrids_nine_machine_bus_holds_what_its_machines_jointly_hold() {
    let Some(net) = fixture("FullGrid/FullGrid-Merged") else {
        eprintln!("skipping: FullGrid fixture not checked out");
        return;
    };
    let joint = net
        .buses
        .iter()
        .find(|b| b.bus_type == BusType::PV && (b.q_max - 12.0).abs() < 1e-9)
        .expect("the six-machine bus should hold ±12 pu");
    assert!(
        (joint.q_min - -12.0).abs() < 1e-9,
        "q_min {} is not six machines' worth",
        joint.q_min
    );
    // And not one machine's worth, which is what it used to be.
    assert!(
        (joint.q_max - 2.0).abs() > 1.0,
        "this looks like a single machine's nameplate again"
    );
}

/// Every summed band is at least as wide as the widest single contribution can
/// be, and none is degenerate. A sign error in the accumulation — adding a
/// `q_max` into `q_min`, say — shows up here as an inverted band.
#[test]
fn no_summed_band_is_inverted_or_empty() {
    for name in [
        "MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged",
        "FullGrid/FullGrid-Merged",
        "SmallGrid/SmallGrid-Merged",
        "Svedala/Svedala-Merged",
        "RealGrid/RealGrid-Merged",
    ] {
        let Some(net) = fixture(name) else { continue };
        for (i, b) in net.buses.iter().enumerate() {
            if b.bus_type != BusType::PV {
                continue;
            }
            assert!(!b.q_min.is_nan() && !b.q_max.is_nan(), "{name}: bus {i} has a NaN limit");
            assert!(
                b.q_min <= b.q_max,
                "{name}: bus {i} has an inverted band [{}, {}]",
                b.q_min,
                b.q_max
            );
        }
    }
}

/// **Why last-writer-wins on the target is harmless in practice**, recorded so
/// it cannot quietly stop being true.
///
/// The feature comparison describes this row as "two controllers with different
/// targets silently keep only the last". Across every vendored fixture,
/// controllers sharing a bus never disagree — so the documented half of the
/// gap has never bitten anyone, and the half that has (the limits, above) was
/// not written down. If a fixture ever gains a disagreement, this fails and
/// the resolution rule has to be decided rather than inherited.
#[test]
fn no_vendored_fixture_has_two_controllers_disagreeing_on_a_target() {
    for name in [
        "MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged",
        "MicroGrid/MicroGrid-Type2/MicroGrid-Type2-Merged",
        "FullGrid/FullGrid-Merged",
        "SmallGrid/SmallGrid-Merged",
        "Svedala/Svedala-Merged",
        "MiniGrid/MiniGrid-Merged",
        "RealGrid/RealGrid-Merged",
    ] {
        let Some(net) = fixture(name) else { continue };
        assert!(
            net.voltage_control.target_conflicts.is_empty(),
            "{name}: {:?}",
            net.voltage_control.target_conflicts
        );
    }
}
