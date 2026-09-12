//! The `gridoxide rao` subcommand's `--parameters` flag.
//!
//! `main.rs`'s argument handling is hand-rolled, so it is pinned here by
//! driving the built binary rather than calling into the library.
//!
//! The flag exists because the optimizer could be told to minimize cost, hold
//! MNECs, run a second preventive pass or measure in amperes — and the binary
//! could ask for none of it. The Cucumber gate validated 1862 assertions' worth
//! of behaviour no user could reach.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn data(sub: &str, name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(sub).join(name)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridoxide")).args(args).output().expect("run gridoxide")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// A `RaoParameters` document changes the answer, and changes it to the
/// reference's own.
///
/// This is the reference's scenario 3.4.1.1 — "selection of cheapest of 3
/// equivalent network actions" — and it is built so the two objectives cannot
/// agree. Three actions each secure the network; the cheapest buys the least
/// room. So the test is a **pair**: asserting only that `--parameters` produces
/// `closeBeFr4` would pass just as well if the flag did nothing and the default
/// happened to pick it.
#[test]
fn rao_parameters_select_the_objective() {
    let network = data("ucte", "2Nodes4ParallelLines.uct");
    let crac = data("rao/features", "crac-92-1-1.json");
    let config = data("rao/features", "RaoParameters_dc_minObjective.json");

    // Max-min-margin, the default: buy the most room, whatever it costs.
    let margin = stdout(&run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
    ]));
    assert!(margin.contains("closeBeFr2"), "default run was:\n{margin}");
    assert!(margin.contains("closeBeFr3"), "default run was:\n{margin}");

    // MIN_COST: the reference asserts `closeBeFr4` alone, for a margin of 250 —
    // less room than the two-action answer above, and cheaper.
    let cost = stdout(&run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--parameters",
        config.to_str().unwrap(),
    ]));
    assert!(cost.contains("closeBeFr4"), "costly run was:\n{cost}");
    assert!(!cost.contains("closeBeFr2"), "costly run was:\n{cost}");
    assert!(!cost.contains("closeBeFr3"), "costly run was:\n{cost}");
    assert!(cost.contains("250.0 MW"), "costly run was:\n{cost}");
}

/// A `--parameters` path that does not exist is an invocation error, not a
/// silent fall back to the defaults.
///
/// Falling back is the failure this whole subcommand is built against: a study
/// that asked for MIN_COST and got max-min-margin has no way to tell from its
/// own output.
#[test]
fn a_missing_parameters_file_is_refused() {
    let network = data("ucte", "2Nodes4ParallelLines.uct");
    let crac = data("rao/features", "crac-92-1-1.json");
    let out = run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--parameters",
        "/nonexistent/RaoParameters.json",
    ]);
    assert_eq!(out.status.code(), Some(2), "a bad invocation exits 2");
}

/// `--depth` beats the file, because a flag the user typed should beat a file
/// they pointed at.
#[test]
fn an_explicit_depth_wins_over_the_configuration() {
    let network = data("ucte", "2Nodes4ParallelLines.uct");
    let crac = data("rao/features", "crac-92-1-1.json");
    let config = data("rao/features", "RaoParameters_dc_minObjective.json");
    let out = run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--parameters",
        config.to_str().unwrap(),
        "--depth",
        "1",
    ]);
    assert_eq!(out.status.code(), Some(0), "{}", stdout(&out));
}
