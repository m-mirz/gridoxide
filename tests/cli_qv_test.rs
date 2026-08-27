//! The `gridoxide qv` subcommand.
//!
//! The physics is gated in `qv_test.rs` against a closed-form two-bus curve;
//! what is pinned here is the hand-rolled argument handling and that each flag
//! reaches the code it names.

use std::path::PathBuf;
use std::process::{Command, Output};

fn case14() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf/pglib_opf_case14_ieee.json")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridoxide"))
        .args(args)
        .output()
        .expect("failed to run the gridoxide binary")
}

fn qv(extra: &[&str]) -> Output {
    let path = case14();
    let mut args = vec!["qv", path.to_str().unwrap()];
    args.extend_from_slice(extra);
    run(&args)
}

fn stdout_of(out: &Output) -> String {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn it_ranks_buses_by_reactive_margin() {
    let text = stdout_of(&qv(&["--weakest", "5", "--v-min", "0.2"]));
    assert!(text.contains("weakest 5 bus(es) by reactive margin"), "{text}");
    assert!(text.contains("margin (MVAr)"), "{text}");
    // The two voltage-stability views measure different things, and the output
    // says so rather than leaving a reader to assume they should agree.
    assert!(text.contains("known not to agree"), "{text}");
}

#[test]
fn a_single_bus_reports_its_margin_and_the_curve() {
    let plain = stdout_of(&qv(&["--bus", "13", "--v-min", "0.2"]));
    assert!(plain.contains("reactive margin"), "{plain}");
    assert!(plain.contains("MVAr, at |V|"), "{plain}");
    assert!(plain.contains("(interpolated)"), "the minimum is refined by default");
    assert!(!plain.contains("Q needed (MVAr)"), "the curve is opt-in");

    let curve = stdout_of(&qv(&["--bus", "13", "--v-min", "0.2", "--curve"]));
    assert!(curve.contains("Q needed (MVAr)"), "{curve}");
}

/// A sweep that stops above the nose must say the number is a bound, not a
/// margin — understating a bus's weakness is the direction that misleads.
#[test]
fn a_truncated_sweep_says_the_margin_is_a_lower_bound() {
    let text = stdout_of(&qv(&["--bus", "13", "--v-min", "0.85"]));
    assert!(text.contains("LOWER BOUND"), "{text}");
    assert!(text.contains("lower --v-min"), "the output should say how to fix it");
}

#[test]
fn no_refine_reports_a_sampled_minimum() {
    let text = stdout_of(&qv(&["--bus", "13", "--v-min", "0.2", "--no-refine"]));
    assert!(text.contains("reactive margin"), "{text}");
    assert!(!text.contains("(interpolated)"), "--no-refine means the raw sample");
}

#[test]
fn a_voltage_controlled_bus_is_called_out() {
    // Bus 7 already holds a voltage, so the curve moves an existing machine's
    // setpoint rather than adding a condenser — a different reading of the same
    // number, and worth saying.
    let text = stdout_of(&qv(&["--bus", "7", "--v-min", "0.2"]));
    assert!(text.contains("already voltage-controlled"), "{text}");
}

#[test]
fn bad_flags_are_refused_rather_than_guessed() {
    for args in [
        vec!["--bus", "999"],
        vec!["--bus", "three"],
        vec!["--step", "slowly"],
        vec!["--bus", "3", "--weakest", "5"],
    ] {
        let out = qv(&args);
        assert!(!out.status.success(), "{args:?} should have been refused");
    }
}

#[test]
fn a_missing_path_is_an_error_not_a_panic() {
    let out = run(&["qv", "/nonexistent/network.json"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("reading"));
}
