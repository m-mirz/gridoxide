//! The `gridoxide short-circuit` subcommand.
//!
//! `main.rs`'s argument handling is hand-rolled, so the parsing is worth
//! pinning even though the solver underneath it is validated thoroughly in
//! `pgm_short_circuit_test.rs`. Everything here drives the built binary through
//! `CARGO_BIN_EXE_gridoxide`, which is the only way to exercise that parsing.

use std::path::PathBuf;
use std::process::{Command, Output};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/short_circuit")
        .join(name)
        .join("input.json")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridoxide"))
        .args(args)
        .output()
        .expect("failed to run the gridoxide binary")
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn reports_fault_current_sources_and_voltages() {
    let path = fixture("three_phase_c_maximum");
    let out = run(&["short-circuit", path.to_str().unwrap()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    let text = stdout_of(&out);
    for section in [
        "fault currents (A):",
        "source contributions (A):",
        "node voltages (p.u.):",
        "symmetrical components of node voltage (p.u.):",
    ] {
        assert!(text.contains(section), "missing {section:?} in:\n{text}");
    }
    // Every node and both sources reported, by their document ids.
    for id_line in ["fault     10:", "source     4:", "node      3 :"] {
        assert!(text.contains(id_line), "missing {id_line:?} in:\n{text}");
    }
}

/// The default is `c_max`, and it has to be, because the maximum current is
/// what equipment ratings are sized against — a default of `c_min` would
/// silently under-report the number most users are after.
#[test]
fn defaults_to_maximum_voltage_scaling() {
    let path = fixture("three_phase_c_maximum");
    let default = stdout_of(&run(&["short-circuit", path.to_str().unwrap()]));
    let explicit =
        stdout_of(&run(&["short-circuit", path.to_str().unwrap(), "--scaling", "max"]));
    assert_eq!(default, explicit);
    assert!(default.contains("c_max"), "{default}");
}

/// `c_min` exists to find the *smallest* fault current, so it must actually
/// produce one smaller than `c_max` — a check that the flag is wired to the
/// solver and not merely to the banner.
#[test]
fn minimum_scaling_gives_a_smaller_fault_current() {
    let path = fixture("three_phase_c_maximum");
    let big = stdout_of(&run(&["short-circuit", path.to_str().unwrap(), "--scaling", "max"]));
    let small = stdout_of(&run(&["short-circuit", path.to_str().unwrap(), "--scaling", "min"]));
    assert!(small.contains("c_min"), "{small}");

    let current = |text: &str| -> f64 {
        // The section header also begins with "fault", so match on the shape
        // of a data row instead.
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with("fault ") && l.contains(": a = "))
            .expect("a fault current line");
        line.split("a = ").nth(1).unwrap().split(',').next().unwrap().trim().parse().unwrap()
    };
    let (big, small) = (current(&big), current(&small));
    assert!(small < big, "c_min current {small} should be below c_max {big}");
}

/// A document with no `fault` is not an error — it is a network under its
/// scaled source voltages — but it should say so rather than print an empty
/// table.
#[test]
fn a_document_without_faults_says_so() {
    let dir = std::env::temp_dir().join("gridoxide_sc_cli_no_fault");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("input.json");
    std::fs::write(
        &path,
        r#"{"version":"1.0","type":"input","is_batch":false,"attributes":{},"data":{
             "node":[{"id":1,"u_rated":10000.0}],
             "source":[{"id":2,"node":1,"status":1,"u_ref":1.0,"sk":1e10,"rx_ratio":0.1}]}}"#,
    )
    .unwrap();

    let out = run(&["short-circuit", path.to_str().unwrap()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout_of(&out);
    assert!(text.contains("no active `fault`"), "{text}");
}

#[test]
fn rejects_an_unknown_scaling() {
    let path = fixture("three_phase_c_maximum");
    let out = run(&["short-circuit", path.to_str().unwrap(), "--scaling", "sideways"]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--scaling"), "{err}");
}

#[test]
fn needs_a_path() {
    let out = run(&["short-circuit"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs a path"));
}
