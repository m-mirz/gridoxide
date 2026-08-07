//! A guard against a fixture component being silently dropped.
//!
//! `PgmData` deliberately does not carry `#[serde(deny_unknown_fields)]`:
//! power-grid-model's documents contain components gridoxide has no reason to
//! model, and failing to parse a whole document over one of them would be worse
//! than ignoring it. The cost is that a component gridoxide *should* model, but
//! does not yet, parses cleanly and then vanishes — which is exactly how
//! `asym_voltage_sensor` and `asym_power_sensor` went unread for as long as they
//! did, with a fixture in the tree the whole time and no diagnostic anywhere.
//!
//! So the check is inverted: every component appearing in any committed fixture
//! must be either a field of `PgmData` or on [`DELIBERATELY_IGNORED`] with a
//! reason. Adding a fixture that uses something new fails here until someone
//! decides which it is.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Every component `PgmData` reads, as its JSON key.
///
/// Kept by hand rather than derived. That is the point: this list changing is a
/// deliberate act, and the test exists to make sure the *fixtures* cannot
/// outrun it silently.
const PARSED_COMPONENTS: &[&str] = &[
    "node",
    "line",
    "link",
    "source",
    "sym_load",
    "asym_load",
    "sym_gen",
    "asym_gen",
    "shunt",
    "transformer",
    "three_winding_transformer",
    "voltage_regulator",
    "sym_voltage_sensor",
    "sym_power_sensor",
    "asym_voltage_sensor",
    "asym_power_sensor",
    "sym_current_sensor",
    "asym_current_sensor",
];

/// Components a fixture may contain that gridoxide knowingly does not read,
/// each with the reason it is not a gap.
///
/// Empty today, and that is worth stating rather than leaving implicit: every
/// component in every committed fixture is currently parsed. The list exists so
/// that the first genuine exception has somewhere to go with its justification,
/// instead of being waved through by relaxing the test.
const DELIBERATELY_IGNORED: &[(&str, &str)] = &[];

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgm")
}

fn input_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            input_files(&path, out);
        } else if path.file_name().is_some_and(|n| n == "input.json") {
            out.push(path);
        }
    }
}

#[test]
fn every_fixture_component_is_parsed_or_deliberately_ignored() {
    let mut files = Vec::new();
    input_files(&fixture_root(), &mut files);
    assert!(
        files.len() > 20,
        "expected to find the PGM fixture tree, got {} input.json files",
        files.len()
    );

    let ignored: BTreeSet<&str> = DELIBERATELY_IGNORED.iter().map(|&(name, _)| name).collect();
    let parsed: BTreeSet<&str> = PARSED_COMPONENTS.iter().copied().collect();

    // Reported all at once, with the fixture that introduced each, so a new
    // component does not have to be discovered one test run at a time.
    let mut unexplained: Vec<(String, String)> = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).expect("fixture readable");
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let Some(data) = value.get("data").and_then(|d| d.as_object()) else {
            continue;
        };
        for key in data.keys() {
            if parsed.contains(key.as_str()) || ignored.contains(key.as_str()) {
                continue;
            }
            let shown = file
                .strip_prefix(fixture_root().parent().unwrap().parent().unwrap())
                .unwrap_or(file)
                .display()
                .to_string();
            unexplained.push((key.clone(), shown));
        }
    }
    unexplained.sort();
    unexplained.dedup_by(|a, b| a.0 == b.0);

    assert!(
        unexplained.is_empty(),
        "these components appear in fixtures but are neither read by `PgmData` nor listed in \
         DELIBERATELY_IGNORED — add them to one or the other rather than to neither: {unexplained:#?}"
    );
}

/// The ignore list must not accumulate entries that no fixture uses any more.
#[test]
fn the_ignore_list_has_no_stale_entries() {
    let mut files = Vec::new();
    input_files(&fixture_root(), &mut files);
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for file in &files {
        let text = std::fs::read_to_string(file).expect("fixture readable");
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        if let Some(data) = value.get("data").and_then(|d| d.as_object()) {
            seen.extend(data.keys().cloned());
        }
    }
    let stale: Vec<&str> = DELIBERATELY_IGNORED
        .iter()
        .map(|&(name, _)| name)
        .filter(|name| !seen.contains(*name))
        .collect();
    assert!(
        stale.is_empty(),
        "no fixture uses these any more, so the exemption is obsolete: {stale:?}"
    );
}
