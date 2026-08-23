//! The `gridoxide solve` subcommand.
//!
//! There was no CLI route to an AC power flow at all before this: `switches
//! --solve` ran one as a side effect of listing switches, `dc` ran the linear
//! one, and the outer loops were reachable only from Rust. Everything here
//! drives the built binary, which is the only way to exercise `main.rs`'s
//! hand-rolled argument handling.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn config(name: &str) -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(name);
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init \
             tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridoxide"))
        .args(args)
        .output()
        .expect("failed to run the gridoxide binary")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn solve_needs_a_path() {
    let out = run(&["solve"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("solve needs a network path"));
}

#[test]
fn an_unreadable_network_is_reported_rather_than_panicking() {
    let out = run(&["solve", "/nonexistent/network.uct"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&out.stderr).is_empty());
}

/// A bare solve reports islands and a converged status, with no outer loop
/// section at all — nothing was asked for, so nothing should be claimed.
#[test]
fn a_bare_solve_runs_no_outer_loop() {
    let Some(dir) = config("Svedala/Svedala-Merged") else { return };
    let out = run(&["solve", dir.to_str().unwrap()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);
    assert!(text.contains("Converged"), "{text}");
    assert!(!text.contains("outer loops:"), "no loop was configured: {text}");
    assert!(!text.contains("tap controllers:"), "{text}");
}

/// The composition, from the command line: tap control, reactive limits and
/// distributed slack in one solve. This is what no single entry point could
/// express before the outer-loop layer existed.
#[test]
fn all_three_controls_run_together() {
    let Some(dir) = config("Svedala/Svedala-Merged") else { return };
    let out = run(&[
        "solve",
        dir.to_str().unwrap(),
        "--control-taps",
        "--enforce-q-limits",
        "--distribute-slack",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);

    assert!(text.contains("outer loops:"), "{text}");
    for loop_name in ["DistributedSlack", "ReactiveLimits", "PhaseControl", "TransformerVoltageControl"] {
        assert!(text.contains(loop_name), "missing {loop_name} in:\n{text}");
    }
    assert!(text.contains("converged = true"), "{text}");
    // Svedala's eleven controls all reach their deadbands, which is the
    // headline number a reader wants.
    assert!(text.contains("11 of 11 inside their deadbands"), "{text}");
    assert!(text.contains("slack distribution:"), "{text}");
}

/// `--control-taps` on a network with no controls says so by silence: the
/// section is absent rather than reporting zero controllers as a success.
#[test]
fn control_taps_on_an_uncontrolled_network_reports_nothing() {
    let Some(dir) = config("SmallGrid/SmallGrid-Merged") else { return };
    let out = run(&["solve", dir.to_str().unwrap(), "--control-taps"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);
    assert!(text.contains("0 regulating control(s)"), "{text}");
    assert!(!text.contains("tap controllers:"), "{text}");
}

/// The tap-report counts reach the console, so a control that could not be
/// used is visible without reading Rust.
#[test]
fn the_import_report_is_printed() {
    let Some(dir) = config("PowerFlow/PowerFlow") else { return };
    let out = run(&["solve", dir.to_str().unwrap(), "--control-taps"]);
    let text = stdout(&out);
    assert!(text.contains("tap controls: 0 read, 1 disabled"), "{text}");
}

/// A malformed flag value is a usage error, not a panic or a silent default.
#[test]
fn a_bad_flag_value_is_rejected() {
    let Some(dir) = config("PowerFlow/PowerFlow") else { return };
    let out = run(&["solve", dir.to_str().unwrap(), "--max-iter", "banana"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--max-iter"));
}

/// UCTE reaches the same subcommand, which is the point of loading by
/// extension rather than by flag.
#[test]
fn a_ucte_network_solves_through_the_same_path() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte/TestCase12Nodes.uct");
    if !path.exists() {
        eprintln!("skipping: no UCTE fixture");
        return;
    }
    let out = run(&["solve", path.to_str().unwrap(), "--enforce-q-limits"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);
    assert!(text.contains("Converged"), "{text}");
    assert!(text.contains("ReactiveLimits"), "{text}");
}

// ---------------------------------------------------------------------------
// Area interchange
// ---------------------------------------------------------------------------

/// CGMES supplies both halves: the areas, from `ControlArea`/`TieFlow`, and the
/// schedule, from `netInterchange`. Nothing has to be stated on the command
/// line.
#[test]
fn cgmes_supplies_both_the_areas_and_the_schedule() {
    let Some(dir) = config("MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged") else { return };
    let out = run(&["solve", dir.to_str().unwrap(), "--area-interchange"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);

    assert!(text.contains("2 control area(s), 10 tie flow(s)"), "{text}");
    assert!(text.contains("0 contested bus(es)"), "{text}");
    assert!(text.contains("area interchange over"), "{text}");
    assert!(text.contains("AreaInterchange"), "{text}");
    assert!(text.contains("converged = true"), "{text}");
    // BE is not the slack's area, so its declared -236.977 MW is met exactly.
    assert!(text.contains("BE"), "{text}");
    assert!(text.contains("-236.977 MW exported"), "{text}");
    // And one area is named as the dependent one.
    assert!(text.contains("(dependent: takes the residual)"), "{text}");
}

/// UCTE supplies areas from its `##Z` country codes and no schedule at all, so
/// every target is zero — each country asked to serve its own load.
#[test]
fn ucte_supplies_areas_from_country_codes() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte/TestCase12Nodes.uct");
    if !path.exists() {
        eprintln!("skipping: no UCTE fixture");
        return;
    }
    let out = run(&["solve", path.to_str().unwrap(), "--area-interchange"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);
    for country in ["BE", "DE", "FR", "NL"] {
        assert!(text.contains(country), "missing {country} in:\n{text}");
    }
    assert!(text.contains("converged = true"), "{text}");
}

/// The two active-power controls are alternatives, and saying so beats picking
/// one — area interchange subsumes distributed slack.
#[test]
fn the_two_active_power_controls_are_refused_together() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte/TestCase12Nodes.uct");
    if !path.exists() {
        return;
    }
    let out = run(&["solve", path.to_str().unwrap(), "--area-interchange", "--distribute-slack"]);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("subsumes"), "{err}");
}

/// Asking for area control on a format that states no areas is a usage error,
/// not a silent no-op.
#[test]
fn area_interchange_without_areas_is_refused() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/iidm/nordic32.xiidm");
    if !path.exists() {
        eprintln!("skipping: no IIDM fixture");
        return;
    }
    let out = run(&["solve", path.to_str().unwrap(), "--area-interchange"]);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("states none"), "{err}");
}
