//! The `gridoxide sensitivity` subcommand.
//!
//! The numerics are validated against a finite-difference re-solve in
//! `ac_sensitivity_test.rs`; what is pinned here is the hand-rolled argument
//! handling and the two directions actually reaching the right code.

use std::path::PathBuf;
use std::process::{Command, Output};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/symmetric/distribution-case/input.json")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridoxide"))
        .args(args)
        .output()
        .expect("failed to run the gridoxide binary")
}

fn sensitivity(extra: &[&str]) -> Output {
    let path = fixture();
    let mut args = vec!["sensitivity", path.to_str().unwrap()];
    args.extend_from_slice(extra);
    run(&args)
}

fn stdout_of(out: &Output) -> String {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn forward_direction_reports_branch_and_voltage_response() {
    let text = stdout_of(&sensitivity(&["--dp", "2"]));
    assert!(text.contains("branch flow response"), "{text}");
    assert!(text.contains("bus voltage response"), "{text}");
    // The index space is stated, because "branch 8" is meaningless otherwise.
    assert!(text.contains("lines first"), "{text}");
}

#[test]
fn adjoint_direction_reports_injections_and_taps() {
    let text = stdout_of(&sensitivity(&["--watch", "8"]));
    assert!(text.contains("what moves the active flow on branch 8"), "{text}");
    assert!(text.contains("by injection"), "{text}");
    assert!(text.contains("by tap"), "{text}");
}

/// The two directions compute the same quantity, so a number that appears in
/// one must appear in the other. This is the cheapest end-to-end check that
/// the CLI wires each flag to the direction it claims.
#[test]
fn the_two_directions_agree_through_the_cli() {
    let forward = stdout_of(&sensitivity(&["--dp", "2"]));
    let adjoint = stdout_of(&sensitivity(&["--watch", "8"]));

    // Forward: dP of branch 8 with respect to injection at bus 2.
    let from_forward = forward
        .lines()
        .find(|l| l.trim_start().starts_with("branch    8:"))
        .and_then(|l| l.split("dP").nth(1))
        .and_then(|l| l.split(',').next())
        .map(|s| s.trim().to_string())
        .expect("branch 8 in the forward output");

    // Adjoint: the same number, read from bus 2's row.
    let from_adjoint = adjoint
        .lines()
        .find(|l| l.trim_start().starts_with("bus    2:"))
        .and_then(|l| l.split("dP").nth(1))
        .and_then(|l| l.split(',').next())
        .map(|s| s.trim().to_string())
        .expect("bus 2 in the adjoint output");

    assert_eq!(from_forward, from_adjoint, "\nforward:\n{forward}\nadjoint:\n{adjoint}");
}

/// Tap variables are branch-indexed, not bus-indexed, and a line has no tap —
/// so `--dk` on a line is a legitimate query with an all-zero answer, which
/// prints as an empty response rather than an error.
#[test]
fn tap_variables_are_accepted_for_transformers() {
    let text = stdout_of(&sensitivity(&["--dk", "8"]));
    assert!(text.contains("--dk 8: branch flow response"), "{text}");
    assert!(text.contains("branch    8:"), "{text}");

    let text = stdout_of(&sensitivity(&["--dalpha", "8"]));
    assert!(text.contains("--dalpha 8: branch flow response"), "{text}");
}

#[test]
fn several_variables_can_be_asked_for_at_once() {
    let text = stdout_of(&sensitivity(&["--dp", "2", "--dq", "2", "--watch", "8"]));
    assert!(text.contains("--dp 2:"), "{text}");
    assert!(text.contains("--dq 2:"), "{text}");
    assert!(text.contains("what moves the active flow"), "{text}");
}

#[test]
fn the_to_terminal_is_selectable() {
    let from = stdout_of(&sensitivity(&["--dp", "2"]));
    let to = stdout_of(&sensitivity(&["--dp", "2", "--terminal", "to"]));
    assert!(to.contains("To terminal"), "{to}");
    assert_ne!(from, to, "the two terminals should not report identical flows");
}

#[test]
fn asking_for_nothing_is_an_error() {
    let out = sensitivity(&[]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("at least one of"), "{err}");
}

#[test]
fn rejects_an_out_of_range_index_and_an_unknown_terminal() {
    let out = sensitivity(&["--dp", "999"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("out of range"));

    let out = sensitivity(&["--terminal", "sideways"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--terminal"));
}

#[test]
fn needs_a_path() {
    let out = run(&["sensitivity"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs a path"));
}
