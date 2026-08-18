//! CGMES `OperationalLimit` import — phase 2 of `plans/RAO_PLAN.md`.
//!
//! Before this, a CGMES network reached gridoxide with **no ratings at all**:
//! `cgmes_to_buses_and_branches` produced impedances and nothing else, so "is
//! this network secure" was unanswerable from CGMES input even though every
//! conformity fixture in the tree carries the answer. These tests pin that the
//! answer now arrives, and — just as important — that a failure to read one is
//! reported rather than silently rendered as "unlimited".

use std::path::{Path, PathBuf};

use cimdecoder_alias::CimDataset;
use gridoxide::cgmes::{self, load_profiles};

mod cimdecoder_alias {
    pub use gridoxide::cgmes::CimDataset;
}

fn configurations() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/CGMES-Test-Configurations/v3.0")
}

/// Load an EQ/SSH/TP/SV bundle, or `None` when the submodule is not checked
/// out — the same skip every other CGMES test in this suite performs.
fn try_load(dir: PathBuf, prefix: &str) -> Option<CimDataset> {
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return None;
    }
    let profiles: Vec<PathBuf> = ["EQ", "SSH", "TP", "SV"]
        .iter()
        .map(|p| dir.join(format!("{prefix}_{p}.xml")))
        .filter(|p| p.exists())
        .collect();
    let refs: Vec<&Path> = profiles.iter().map(|p| p.as_path()).collect();
    Some(load_profiles(&refs).expect("failed to decode CGMES profiles"))
}

fn try_load_pst_type1() -> Option<CimDataset> {
    try_load(configurations().join("PST/PST_PhaseTapChangerLinear_Type1"), "PST_Type1")
}

fn try_load_small_grid() -> Option<CimDataset> {
    try_load(configurations().join("SmallGrid/SmallGrid-Merged"), "SmallGrid")
}

/// PATL and TATL, at the smallest scale that has both.
#[test]
fn a_pst_fixture_yields_permanent_and_temporary_ratings() {
    let Some(ds) = try_load_pst_type1() else { return };
    let (limits, report) = cgmes::cgmes_operational_limits(&ds).expect("limits");

    assert!(report.current_limits > 0, "no CurrentLimit converted: {report:?}");
    assert_eq!(report.unattached, 0, "every limit should resolve to equipment");
    assert_eq!(report.without_value, 0);
    assert!(!limits.is_empty());

    // This fixture declares one PATL type (isInfiniteDuration) and one TATL
    // type (acceptableDuration 1200 s), so both branches of the classifier must
    // be exercised somewhere in the file.
    let permanent = limits.values().filter(|l| l.tightest_patl().is_some()).count();
    let temporary: usize = limits.values().map(|l| l.terminals.iter().map(|t| t.tatl.len()).sum::<usize>()).sum();
    assert!(permanent > 0, "no permanent rating found");
    assert!(temporary > 0, "no temporary rating found");

    // The 1200-second TATL is the one the file names; it must survive with its
    // duration, since the duration is the whole reason a TATL is not a PATL.
    let has_1200 = limits.values().any(|l| {
        l.terminals.iter().any(|t| {
            t.tatl.iter().any(|x| x.acceptable_duration_s == Some(1200.0))
        })
    });
    assert!(has_1200, "the TATL_1200 type did not survive with its duration");
}

/// `normalValue` is the field the conformity fixtures actually populate.
///
/// A reader that only consulted `value` would find every limit empty, convert
/// nothing, and report success — which is why this is a test of its own rather
/// than an implementation detail.
#[test]
fn ratings_are_read_from_normal_value_when_value_is_absent() {
    let Some(ds) = try_load_pst_type1() else { return };
    let (limits, _) = cgmes::cgmes_operational_limits(&ds).expect("limits");
    let values: Vec<f64> = limits.values().filter_map(|l| l.tightest_patl()).collect();
    assert!(!values.is_empty(), "no PATL values read");
    assert!(values.iter().all(|v| *v > 0.0), "a rating came through as zero: {values:?}");
    // The fixture's PATLs are around 1312 A; anything near zero would mean the
    // value was defaulted rather than read.
    assert!(values.iter().any(|v| *v > 100.0), "values look defaulted: {values:?}");
}

#[test]
fn limits_land_on_the_terminal_their_set_names() {
    let Some(ds) = try_load_pst_type1() else { return };
    let (limits, _) = cgmes::cgmes_operational_limits(&ds).expect("limits");
    // A two-terminal branch with per-terminal sets must end up with entries on
    // both sides, not both stacked on side 0.
    let two_sided = limits.values().filter(|l| l.terminals.len() >= 2).count();
    assert!(two_sided > 0, "no equipment received limits on more than one terminal");
    for l in limits.values() {
        assert!(l.terminals.len() <= 3, "more terminals than any branch has: {l:?}");
    }
}

#[test]
fn a_larger_grid_imports_every_limit_it_declares() {
    let Some(ds) = try_load_small_grid() else { return };
    let (limits, report) = cgmes::cgmes_operational_limits(&ds).expect("limits");
    assert!(report.current_limits > 100, "only {} limits: {report:?}", report.current_limits);
    assert_eq!(report.unattached, 0, "unattached limits: {report:?}");
    assert!(limits.len() > 50, "only {} pieces of equipment carry limits", limits.len());
    // Every rating must be a positive current. A zero here would be read
    // downstream as a branch that is always overloaded.
    for (mrid, l) in &limits {
        for t in &l.terminals {
            if let Some(patl) = t.patl_a {
                assert!(patl > 0.0, "{mrid} has a non-positive PATL {patl}");
            }
            for tatl in &t.tatl {
                assert!(tatl.value_a > 0.0, "{mrid} has a non-positive TATL");
            }
        }
    }
}

/// Every declared limit must land somewhere.
///
/// The strongest available check, and a cheap one: the number of `CurrentLimit`
/// objects converted has to equal the number of ratings that came out. A limit
/// dropped on the floor reads downstream as *unlimited*, which is the most
/// dangerous possible default — a security analysis on a network with no limits
/// reports everything secure. Counting both sides makes that impossible to miss.
#[test]
fn no_declared_limit_is_lost_between_the_file_and_the_result() {
    for (dir, prefix) in [
        ("PST/PST_PhaseTapChangerLinear_Type1", "PST_Type1"),
        ("SmallGrid/SmallGrid-Merged", "SmallGrid"),
        ("MicroGrid/MicroGid-BaseCase/MicroGrid-BE-MAS", "MicroGrid-BE"),
    ] {
        let Some(ds) = try_load(configurations().join(dir), prefix) else { continue };
        let (limits, report) = cgmes::cgmes_operational_limits(&ds).expect("limits");
        let produced: usize = limits
            .values()
            .map(|l| {
                l.terminals
                    .iter()
                    .map(|t| usize::from(t.patl_a.is_some()) + t.tatl.len())
                    .sum::<usize>()
            })
            .sum();
        assert_eq!(
            produced, report.current_limits,
            "{dir}: {} CurrentLimit objects converted but {produced} ratings produced",
            report.current_limits
        );
        assert_eq!(report.unattached, 0, "{dir}: {report:?}");
        assert_eq!(report.without_value, 0, "{dir}: {report:?}");
    }
}
