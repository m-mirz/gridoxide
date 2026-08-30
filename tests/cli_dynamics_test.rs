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

#[test]
fn the_machine_formulation_is_selectable_and_changes_the_answer() {
    let swing = |extra: &[&str]| {
        let text = stdout_of(&dynamics(&[&["--stop", "4"], extra].concat()));
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with("G1"))
            .unwrap_or_default()
            .to_string();
        line.split_whitespace()
            .nth(3)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or_else(|| panic!("no swing in {line:?}"))
    };

    let approximate = swing(&["--no-speed-voltages"]);
    let full = swing(&["--speed-voltages"]);
    assert!(approximate > 0.5 && full > 0.5, "both should swing: {approximate} {full}");
    // A modelling choice that moved nothing would mean the flag was not
    // reaching the equations; one that moved a great deal would mean something
    // other than a factor of omega had changed.
    let gap = (approximate - full).abs();
    assert!(gap > 1e-4, "the flag must reach the machines, gap {gap:e}");
    assert!(gap < 0.05 * approximate, "but it is a per-cent effect, not a new model: {gap:e}");

    // The document's own setting is the default.
    assert_eq!(swing(&[]), approximate);
}

#[test]
fn the_modes_flag_reports_the_linearized_system() {
    let text = stdout_of(&dynamics(&["--modes", "4"]));

    assert!(text.contains("13 mode(s) over 13 differential state(s)"), "{text}");
    // Which method answered is part of the answer: "every mode" and "the four
    // nearest a point" support very different conclusions.
    assert!(text.contains("every mode, by a dense decomposition"), "{text}");
    assert!(text.contains("eigenvalue"), "{text}");
    assert!(text.contains("damping"), "{text}");
    assert!(text.contains("participation"), "{text}");
    // The swing mode is the rotor's, and the output names it rather than
    // leaving a reader to work it out from an eigenvector.
    assert!(text.contains("G1.omega"), "{text}");
    assert!(text.contains("every mode decays"), "{text}");

    // Four modes asked for, four printed — plus the header and the verdict.
    let rows = text.lines().filter(|l| l.contains("j  ")).count();
    assert_eq!(rows, 4, "{text}");

    // It answers instead of running, because it is a different question.
    assert!(!text.contains("ran to"), "{text}");

    // A participation factor never prints as a bare `0%`. On a large system a
    // mode is spread across every machine at a fraction of a per cent each, and
    // rounding that away reads as *no* participation when the truth is the
    // opposite — that the mode belongs to all of them.
    assert!(!text.contains(" 0%"), "{text}");
}

/// Naming a frequency selects the sparse method and says so.
///
/// The interesting property is that it is *selectable*, not merely a fallback
/// past a size threshold: asking about one band of a system small enough for
/// the dense method is a legitimate question, and the answer has to say it is
/// answering that one — a caller who reads "no unstable modes" off a list of
/// four modes near 1 Hz has been misled unless the output told them.
#[test]
fn a_named_frequency_selects_the_sparse_method() {
    let text = stdout_of(&dynamics(&["--modes", "3", "--modes-freq", "1.27"]));

    assert!(text.contains("sparse Arnoldi"), "{text}");
    assert!(text.contains("NOT every mode the system has"), "{text}");
    assert!(text.contains("1.270 Hz"), "{text}");
    assert!(text.contains("residual"), "{text}");

    let rows = text.lines().filter(|l| l.contains("j  ")).count();
    assert_eq!(rows, 3, "{text}");

    // The swing mode is what sits nearest 1.27 Hz, and both methods find the
    // same eigenvalue — the dense run above prints it too.
    assert!(text.contains("-0.73018"), "{text}");
    assert!(text.contains("G1.omega"), "{text}");
}

/// The literal form of the same flag.
#[test]
fn the_shift_can_be_named_outright() {
    let text = stdout_of(&dynamics(&["--modes", "2", "--modes-near", "-0.4,7.98"]));
    assert!(text.contains("sparse Arnoldi"), "{text}");
    assert!(text.contains("-0.4000+7.9800j"), "{text}");
    assert!(text.contains("-0.73018"), "{text}");
}

/// Flags that contradict each other are refused rather than silently ranked.
#[test]
fn the_aiming_flags_are_checked_against_each_other() {
    let out = dynamics(&["--modes", "2", "--modes-near", "-0.4,7.98", "--modes-freq", "1.0"]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--modes-near names the shift outright"), "{err}");

    // A damping ratio says how far off the axis, not where along it.
    let out = dynamics(&["--modes", "2", "--modes-damping", "0.05"]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--modes-freq to say where"), "{err}");

    let out = dynamics(&["--modes", "2", "--modes-near", "nonsense"]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("expected <re>,<im>"), "{err}");
}

/// The sensitivity report answers "what should we change".
#[test]
fn the_sensitivity_flag_ranks_the_parameters() {
    let text = stdout_of(&dynamics(&["--modes", "2", "--modes-sensitivity"]));

    assert!(text.contains("sensitivity of the least-damped mode"), "{text}");
    assert!(text.contains("dlambda/dp"), "{text}");
    assert!(text.contains("dzeta/dp"), "{text}");
    // The machine's inertia and damping coefficient, named against the device
    // they belong to rather than by index.
    assert!(text.contains("G1.h"), "{text}");
    assert!(text.contains("G1.d"), "{text}");
    // And nothing else: a reactance moves the equilibrium, so it is not on
    // offer — see `Machine::tunable`.
    assert!(!text.contains("G1.xdp"), "{text}");
}
