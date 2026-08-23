//! The `gridoxide continuation` subcommand.
//!
//! The numerics are gated in `continuation_test.rs` (against a closed form) and
//! `continuation_events_test.rs` (against a brute-force bisection). What is
//! pinned here is the hand-rolled argument handling, and that each flag reaches
//! the code it names.

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

fn continuation(extra: &[&str]) -> Output {
    let path = case14();
    let mut args = vec!["continuation", path.to_str().unwrap()];
    args.extend_from_slice(extra);
    run(&args)
}

fn stdout_of(out: &Output) -> String {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn it_reports_the_limit_the_margin_and_the_weakest_buses() {
    let text = stdout_of(&continuation(&[]));
    assert!(text.contains("loadability limit"), "{text}");
    assert!(text.contains("lambda_max"), "{text}");
    assert!(text.contains("margin"), "{text}");
    assert!(text.contains("weakest buses"), "{text}");
    // λ_max means nothing without the direction it was measured along, so the
    // output says so rather than leaving the reader to assume.
    assert!(text.contains("property of that direction"), "{text}");
}

#[test]
fn enforcing_reactive_limits_lists_each_machine_and_lowers_the_limit() {
    let free = stdout_of(&continuation(&[]));
    let limited = stdout_of(&continuation(&["--enforce-q-limits"]));

    assert!(limited.contains("reactive limits reached"), "{limited}");
    assert!(limited.contains("stopped holding its voltage"), "{limited}");

    let lambda = |text: &str| -> f64 {
        text.lines()
            .find(|l| l.trim_start().starts_with("lambda_max"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("no lambda_max in:\n{text}"))
    };
    assert!(
        lambda(&limited) < lambda(&free),
        "reactive limits should shrink the margin: {} vs {}",
        lambda(&limited),
        lambda(&free)
    );
}

#[test]
fn the_curve_flag_prints_the_walk() {
    let text = stdout_of(&continuation(&["--curve"]));
    assert!(text.contains("dlambda/dsigma"), "{text}");
    assert!(text.contains("upper"), "{text}");
}

#[test]
fn a_target_lambda_stops_short_of_the_nose() {
    let text = stdout_of(&continuation(&["--target-lambda", "0.3"]));
    assert!(text.contains("TargetReached"), "{text}");
    // Stopping early means there is no collapse point to report, and saying so
    // beats reporting the last point reached as if it were the limit.
    assert!(text.contains("no collapse point was found"), "{text}");
}

#[test]
fn bad_flags_are_refused_rather_than_guessed() {
    for args in [
        vec!["--parametrization", "wibble"],
        vec!["--target-lambda", "soon"],
        vec!["--step", "quickly"],
        vec!["--target-lambda", "0.3", "--lower-branch"],
    ] {
        let out = continuation(&args);
        assert!(!out.status.success(), "{args:?} should have been refused");
    }
}

#[test]
fn a_missing_path_is_an_error_not_a_panic() {
    let out = run(&["continuation", "/nonexistent/network.json"]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("reading"), "{err}");
}
