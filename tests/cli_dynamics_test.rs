//! The `gridoxide dynamics` subcommand.
//!
//! The physics is gated elsewhere — against closed forms in `dynamics_test.rs`
//! and `dynamics_events_test.rs`, and against Dynawo in
//! `dynamics_reference_test.rs`. What is pinned here is the hand-rolled
//! argument handling, that each flag reaches the code it names, and that the
//! summary reports the things a reader of a transient-stability run actually
//! wants: whether each machine stayed in step, and how deep the voltage went.

use std::path::PathBuf;
use std::process::{Command, Output};

fn document() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/dynamics/smib.json")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridoxide"))
        .args(args)
        .output()
        .expect("failed to run the gridoxide binary")
}

fn dynamics(extra: &[&str]) -> Output {
    let path = document();
    let mut args = vec!["dynamics", path.to_str().unwrap()];
    args.extend_from_slice(extra);
    run(&args)
}

fn stdout_of(out: &Output) -> String {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn it_runs_a_document_and_reports_what_the_machines_did() {
    let text = stdout_of(&dynamics(&["--stop", "6", "--step", "0.005"]));

    assert!(text.contains("3 bus(es), 13 differential state(s), 3 scheduled event(s)"), "{text}");
    // The initial derivative is printed because a nonzero one invalidates
    // everything after it, and a reader should not have to ask.
    assert!(text.contains("initial state derivative:"), "{text}");
    assert!(text.contains("ran to 6.000 s"), "{text}");

    // The summary is per machine, not per state.
    assert!(text.contains("machine"), "{text}");
    assert!(text.contains("min speed"), "{text}");
    assert!(text.contains("G1"), "{text}");
    // The fault collapses a bus, and the run says which and when.
    assert!(text.contains("lowest voltage"), "{text}");
}

#[test]
fn the_csv_flag_writes_the_trajectory_and_observe_narrows_it() {
    let out = tempfile();
    let text = stdout_of(&dynamics(&[
        "--stop", "3", "--csv", out.to_str().unwrap(), "--observe", "G1.omega",
    ]));
    assert!(text.contains("wrote 1 column(s)"), "{text}");

    let csv = std::fs::read_to_string(&out).expect("the csv was written");
    let header = csv.lines().next().unwrap();
    assert_eq!(header, "time,G1.omega");
    // Two rows per event instant, so the discontinuities stay visible in the
    // exported record rather than being smoothed by whoever plots it.
    let rows = csv.lines().count() - 1;
    assert!(rows > 600, "expected a row per step, got {rows}");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn an_observe_pattern_that_matches_nothing_is_an_error() {
    let out = tempfile();
    let result = dynamics(&["--stop", "1", "--csv", out.to_str().unwrap(), "--observe", "nope"]);
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("matched none of the"), "{stderr}");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn the_backends_are_selectable_and_agree() {
    let scalar = stdout_of(&dynamics(&["--stop", "3", "--backend", "scalar"]));
    let klu = stdout_of(&dynamics(&["--stop", "3", "--backend", "klu-native"]));

    // The per-machine summary is printed to six figures, so identical text is a
    // real agreement rather than a rounded one.
    let line = |text: &str| {
        text.lines().find(|l| l.trim_start().starts_with("G1")).unwrap_or_default().to_string()
    };
    assert_eq!(line(&scalar), line(&klu), "the backend is a performance choice, not an answer");

    let bad = dynamics(&["--backend", "quantum"]);
    assert!(!bad.status.success());
    assert!(
        String::from_utf8_lossy(&bad.stderr).contains("expected scalar or klu-native"),
        "{}",
        String::from_utf8_lossy(&bad.stderr)
    );
}

#[test]
fn a_missing_path_and_a_bad_document_are_both_named() {
    let missing = run(&["dynamics"]);
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("needs a document path"));

    let absent = run(&["dynamics", "/nonexistent/case.json"]);
    assert!(!absent.status.success());
    let stderr = String::from_utf8_lossy(&absent.stderr);
    assert!(stderr.contains("/nonexistent/case.json"), "{stderr}");
}

/// A unique scratch path. The binary writes it; the test removes it.
fn tempfile() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "gridoxide-dynamics-{}-{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    path
}
