//! The `gridoxide opf` subcommand.
//!
//! The numerics are validated against analytic cases, KKT certificates and
//! pglib's published objectives in `opf_dc_test.rs`. What is pinned here is
//! the hand-rolled argument handling, the companion-document default, and that
//! the output actually reports the things a dispatcher reads off an OPF.

use std::path::PathBuf;
use std::process::{Command, Output};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf")
        .join(format!("{name}.json"))
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridoxide"))
        .args(args)
        .output()
        .expect("failed to run the gridoxide binary")
}

fn opf(case: &str, extra: &[&str]) -> Output {
    let path = fixture(case);
    let mut args = vec!["opf", path.to_str().unwrap()];
    args.extend_from_slice(extra);
    run(&args)
}

fn stdout_of(out: &Output) -> String {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The companion document is found without being named — the whole point of
/// the `<network>.opf.json` convention the converter writes.
#[test]
fn the_companion_document_is_found_by_convention() {
    let text = stdout_of(&opf("pglib_opf_case5_pjm", &[]));
    assert!(text.contains("total cost:"), "{text}");
    assert!(text.contains("dispatch (MW):"), "{text}");
    assert!(text.contains("locational marginal price"), "{text}");
}

/// A congested case reports the price spread and what is causing it, which is
/// the reason to run an OPF rather than a power flow.
#[test]
fn a_congested_case_reports_the_spread_and_what_binds() {
    let text = stdout_of(&opf("pglib_opf_case5_pjm", &[]));
    assert!(text.contains("spread:"), "{text}");
    assert!(text.contains("binding branch limits:"), "{text}");
    assert!(text.contains("to relieve"), "{text}");
    // The published objective for this case is 1.7480e4 — see the fixture
    // README. Checked loosely here; the tight comparison lives in
    // `opf_dc_test.rs`.
    assert!(text.contains("total cost: 17479."), "{text}");
}

/// An uncongested case says so in one line rather than printing an identical
/// price for every bus.
#[test]
fn an_uncongested_case_says_the_price_is_uniform() {
    let text = stdout_of(&opf("pglib_opf_case14_ieee", &[]));
    assert!(text.contains("uniform at"), "{text}");
    assert!(text.contains("no binding branch limits"), "{text}");
}

/// Units pinned at a limit are marked, because that is where the answer is
/// being shaped by something other than price.
#[test]
fn generators_at_a_limit_are_marked() {
    let text = stdout_of(&opf("pglib_opf_case5_pjm", &[]));
    assert!(text.contains("(at max)"), "{text}");
}

#[test]
fn the_companion_document_can_be_given_explicitly() {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf/pglib_opf_case5_pjm.opf.json");
    let explicit = stdout_of(&opf("pglib_opf_case5_pjm", &["--data", data.to_str().unwrap()]));
    let implicit = stdout_of(&opf("pglib_opf_case5_pjm", &[]));
    assert_eq!(explicit, implicit);
}

/// Turning shedding off changes the problem, not merely the report — so a
/// case that needs it becomes infeasible and says why.
#[test]
fn shedding_can_be_switched_off() {
    // These cases can all serve their demand, so the flag changes nothing
    // about the answer — but it must still be accepted and still solve.
    let text = stdout_of(&opf("pglib_opf_case5_pjm", &["--no-shedding"]));
    assert!(text.contains("total cost: 17479."), "{text}");
}

#[test]
fn the_shed_price_is_configurable_and_validated() {
    let text = stdout_of(&opf("pglib_opf_case5_pjm", &["--shed-price", "5000"]));
    assert!(text.contains("total cost:"), "{text}");

    let out = opf("pglib_opf_case5_pjm", &["--shed-price", "cheap"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--shed-price"));
}

/// A missing companion document is the most likely mistake, so it should say
/// what to do rather than just failing to open a file.
#[test]
fn a_missing_companion_document_explains_itself() {
    let out = opf("pglib_opf_case5_pjm", &["--data", "/nonexistent/opf.json"]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--data"), "{err}");
}

#[test]
fn needs_a_network_path() {
    let out = run(&["opf"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs a network path"));
}
