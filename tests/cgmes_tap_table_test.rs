//! CGMES tap tables: every position, not just the one the SSH names.
//!
//! `TapChangerIndex::effect_for_end` computed one step and threw the rest
//! away, which is all a fixed-tap power flow ever wanted. `src/ucte.rs` and
//! `src/iidm.rs` retained the whole table; CGMES — the deepest importer in the
//! tree — did not, and a `PstRangeAction` on a CGMES network was therefore
//! inexpressible.
//!
//! **The gate is agreement with the path it replaces.** Reading the current
//! position back out of the table must reproduce `Transformer::tap` bit for
//! bit, on every fixture. That is what makes the table a replacement for the
//! single-step computation rather than a second opinion about it — the
//! composition it goes through (invert when the changer sits on the `to` side,
//! divide by the nameplate-vs-bus structural ratio) is fiddly enough that a
//! near-miss would be easy to ship.

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_to_network, load_profiles};

fn config(dir: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(dir);
    path.exists().then_some(path)
}

/// Every `.xml` in a configuration directory, which is how the fixture tests
/// already load a merged model.
fn profiles(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("configuration directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "xml"))
        .collect();
    v.sort();
    v
}

fn network(dir: &str) -> Option<gridoxide::cgmes::CgmesNetwork> {
    let dir = config(dir)?;
    let paths = profiles(&dir);
    let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    let ds = load_profiles(&refs).expect("failed to decode CGMES profiles");
    Some(cgmes_to_network(&ds, 100e6).expect("conversion failed"))
}

/// The load-bearing one. Across every fixture that has a tap changer at all,
/// the retained table's current entry is the tap the single-step path put on
/// the transformer — to the bit, not to a tolerance.
#[test]
fn the_current_position_reads_back_exactly() {
    let configs = [
        "Svedala/Svedala-Merged",
        "FullGrid/FullGrid-Merged",
        "SmallGrid/SmallGrid-Merged",
        "PowerFlow/PowerFlow",
        "PST/PST_PhaseTapChangerLinear_Type1",
        "PST/PST_PhaseTapChangerLinear_Type2",
        "PST/PST_PhaseTapChangerTable_Type3",
        "MiniGrid/MiniGrid-Merged",
    ];

    let mut checked = 0;
    let mut skipped = Vec::new();
    for name in configs {
        let Some(net) = network(name) else {
            skipped.push(name);
            continue;
        };
        assert_eq!(
            net.tap_changers.len(),
            net.transformers.len(),
            "{name}: the tap table must be parallel to the transformers"
        );
        for (i, changer) in net.tap_changers.iter().enumerate() {
            let Some(changer) = changer else { continue };
            let from_table = changer
                .current()
                .unwrap_or_else(|| panic!("{name}: transformer {i} is at a position outside its own table"));
            let on_branch = net.transformers[i].tap;
            assert_eq!(
                (from_table.re.to_bits(), from_table.im.to_bits()),
                (on_branch.re.to_bits(), on_branch.im.to_bits()),
                "{name}: transformer {i} at position {} — table says {from_table}, branch carries {on_branch}",
                changer.position
            );
            // And the series admittance, where the changer moves it.
            if let Some(series) = &changer.series {
                let idx = (changer.position - changer.low) as usize;
                assert_eq!(
                    (series[idx].re.to_bits(), series[idx].im.to_bits()),
                    (net.transformers[i].y_series.re.to_bits(), net.transformers[i].y_series.im.to_bits()),
                    "{name}: transformer {i}'s per-step admittance disagrees with the branch"
                );
            }
            checked += 1;
        }
    }

    assert!(
        checked >= 20,
        "expected the fixture set to contribute a good number of tap changers, got {checked} \
         (skipped: {skipped:?})"
    );
}

/// Svedala's eleven `RatioTapChangerTable`-driven changers, at real scale.
/// Table-driven is the case with no formula to fall back on, so a missing row
/// must produce no changer rather than a table with holes.
#[test]
fn svedala_retains_every_position_of_every_changer() {
    let Some(net) = network("Svedala/Svedala-Merged") else {
        eprintln!("skipping: Svedala fixture not checked out");
        return;
    };
    let changers: Vec<_> = net.tap_changers.iter().flatten().collect();
    assert_eq!(changers.len(), 11, "Svedala declares eleven tap changers");

    for c in &changers {
        assert!(c.len() >= 2, "a one-position changer is not a changer");
        assert_eq!(c.steps.len(), (c.high() - c.low + 1) as usize);
        assert!(
            c.position >= c.low && c.position <= c.high(),
            "position {} outside [{}, {}]",
            c.position,
            c.low,
            c.high()
        );
        // Every position must be reachable — the point of retaining the table.
        for p in c.low..=c.high() {
            assert!(c.at(p).is_some(), "position {p} missing");
            assert!(c.ratio(p).unwrap() > 0.0, "position {p} has a non-positive ratio");
        }
    }
}

/// A phase shifter's table is monotone in neither ratio nor, necessarily,
/// angle — but it must be *ordered by position*, and `nearest_to_angle` must
/// round to a position the table actually holds.
#[test]
fn a_phase_shifter_rounds_to_a_position_it_has() {
    let Some(net) = network("PST/PST_PhaseTapChangerTable_Type3") else {
        eprintln!("skipping: PST Type3 fixture not checked out");
        return;
    };
    let changer = net
        .tap_changers
        .iter()
        .flatten()
        .next()
        .expect("the PST fixture has a phase tap changer");

    for p in changer.low..=changer.high() {
        let angle = changer.angle_deg(p).expect("every position has an angle");
        let rounded = changer.nearest_to_angle(angle).expect("rounds to something");
        // Rounding an angle the table itself holds must return a position with
        // that same angle — not necessarily `p`, since two positions may share
        // an angle, which is why this compares the angle rather than the index.
        assert!(
            (changer.angle_deg(rounded).unwrap() - angle).abs() < 1e-12,
            "position {p} at {angle}° rounded to {rounded} at {}°",
            changer.angle_deg(rounded).unwrap()
        );
    }
}

/// Moving a tap must move the branch — and, for a changer whose reactance
/// varies with position, must move both halves together. Writing the ratio and
/// leaving the impedance behind is the failure mode `TapChanger::series`
/// exists to prevent: the flow still changes in the right direction, so the
/// answer looks plausible.
#[test]
fn setting_a_position_moves_the_branch_and_stays_in_range() {
    let Some(mut net) = network("PST/PST_PhaseTapChangerLinear_Type1") else {
        eprintln!("skipping: PST Type1 fixture not checked out");
        return;
    };
    let i = net
        .tap_changers
        .iter()
        .position(|c| c.is_some())
        .expect("the PST fixture has a tap changer");
    let changer = net.tap_changers[i].as_mut().unwrap();
    let (low, high) = (changer.low, changer.high());
    let varies = changer.series.is_some();

    let before = net.transformers[i].tap;
    let before_y = net.transformers[i].y_series;
    assert!(changer.set_position(&mut net.transformers[i], high));
    assert_ne!(net.transformers[i].tap, before, "the extreme tap must differ from the current one");
    if varies {
        assert_ne!(
            net.transformers[i].y_series, before_y,
            "this changer's reactance varies with position, so the branch admittance must too"
        );
    }

    // Out of range leaves both the changer and the branch untouched, so a
    // caller sweeping a range never silently pins at an endpoint.
    let at_high = net.transformers[i].tap;
    let changer = net.tap_changers[i].as_mut().unwrap();
    assert!(!changer.set_position(&mut net.transformers[i], high + 1));
    assert!(!changer.set_position(&mut net.transformers[i], low - 1));
    assert_eq!(net.transformers[i].tap, at_high);
    assert_eq!(net.tap_changers[i].as_ref().unwrap().position, high);
}
