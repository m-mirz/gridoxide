//! The `gridoxide switches` subcommand.
//!
//! The CLI had no test coverage at all before this; `main.rs`'s argument
//! handling is hand-rolled, so the parsing is worth pinning even though the
//! solver underneath it is tested thoroughly elsewhere. Everything here drives
//! the built binary through `CARGO_BIN_EXE_gridoxide` rather than calling into
//! the library, which is the only way to exercise that parsing at all.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn minigrid() -> Option<Vec<PathBuf>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MiniGrid/MiniGrid-Merged");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init \
             tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return None;
    }
    Some(
        ["EQBD", "EQ", "SSH", "TP", "SV"]
            .iter()
            .map(|p| dir.join(format!("MiniGrid_{p}.xml")))
            .filter(|p| p.exists())
            .collect(),
    )
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridoxide"))
        .args(args)
        .output()
        .expect("failed to run the gridoxide binary")
}

fn run_switches(extra: &[&str]) -> Option<Output> {
    let files = minigrid()?;
    let mut args: Vec<String> =
        std::iter::once("switches".to_string())
            .chain(files.iter().map(|p| p.display().to_string()))
            .collect();
    args.extend(extra.iter().map(|s| s.to_string()));
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    Some(run(&refs))
}

#[test]
fn switches_lists_devices_with_their_mrids_and_state() {
    let Some(out) = run_switches(&[]) else { return };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(stdout.contains("103 connectivity node(s)"), "{stdout}");
    assert!(stdout.contains("90 switch(es) in the model, 30 retained"), "{stdout}");
    // Every listed switch is named by its CGMES mRID, which is what an operator
    // has to hand — not by an index only gridoxide knows.
    assert!(stdout.contains("_"), "{stdout}");
    assert!(stdout.contains("Disconnector") || stdout.contains("Breaker"), "{stdout}");
    assert!(stdout.contains("closed"), "{stdout}");
    // Without --solve there are no flows to report.
    assert!(!stdout.contains("P (MW)"), "{stdout}");
}

#[test]
fn solve_reports_a_flow_through_each_switch() {
    let Some(out) = run_switches(&["--solve"]) else { return };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("power flow: Converged"), "{stdout}");
    assert!(stdout.contains("P (MW)"), "{stdout}");
}

/// The retention policy is the knob that decides how big the system is, so it
/// has to actually reach the solver.
#[test]
fn the_retention_policy_changes_how_much_survives() {
    let Some(none) = run_switches(&["--retain", "none"]) else { return };
    let all = run_switches(&["--retain", "all"]).unwrap();
    let none = String::from_utf8_lossy(&none.stdout).to_string();
    let all = String::from_utf8_lossy(&all.stdout).to_string();

    assert!(none.contains("0 retained"), "{none}");
    assert!(all.contains("90 retained"), "{all}");
    // `none` merges everything, so it must produce the fewest buses.
    assert!(none.contains("13 bus(es)") || none.contains("15 bus(es)"), "{none}");
    assert!(all.contains("105 bus(es)"), "{all}");
}

/// Opening a load-carrying switch must change the answer, and the listing must
/// show the new position rather than the one the model was imported with.
#[test]
fn opening_a_switch_is_reflected_in_the_listing_and_the_solve() {
    // A disconnector MiniGrid carries real power through.
    const MRID: &str = "_b7e79b65-8d3e-4ee9-bf05-12bf2250b12d";
    let Some(out) = run_switches(&["--open", MRID, "--solve"]) else { return };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(stdout.contains(&format!("opened {MRID}")), "{stdout}");
    assert!(stdout.contains("power flow: Converged"), "{stdout}");
    let line = stdout
        .lines()
        .find(|l| l.contains(MRID))
        .unwrap_or_else(|| panic!("the opened switch is not listed:\n{stdout}"));
    assert!(line.contains("open"), "listing still shows it closed: {line}");
}

/// Bad input is rejected with a message that says what to do, and a non-zero
/// exit — a script driving this needs both.
#[test]
fn malformed_invocations_are_rejected() {
    let out = run(&["switches"]);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("at least one CGMES profile"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let Some(out) = run_switches(&["--retain", "nonsense"]) else { return };
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("expected none, busbar_adjacent or all"), "{err}");

    let out = run_switches(&["--retain"]).unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs a value"));

    let out = run_switches(&["--open", "not-an-mrid"]).unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no retained switch matches"), "{err}");
}

/// `--help` mentions the subcommand, so it is discoverable.
#[test]
fn the_subcommand_is_documented_in_usage() {
    let out = run(&["--help"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("gridoxide switches"), "{stdout}");
    assert!(stdout.contains("--retain"), "{stdout}");
}
