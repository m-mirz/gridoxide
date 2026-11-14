//! `dc`, `sensitivity`, `qv` and `continuation` read every network format, not
//! only power-grid-model JSON.
//!
//! The solvers are gated elsewhere; what is pinned here is that each command
//! reaches them from a MATPOWER case, with its shunts (case14 has one, at bus
//! 9), and that `--base-mva` is refused where the file states its own base.

use std::path::PathBuf;
use std::process::{Command, Output};

fn case14() -> Option<PathBuf> {
    let path = PathBuf::from(crate::REPO).join("tests/data/benchmark-grids/matpower/case14.m");
    if !path.exists() {
        eprintln!(
            "skipping: {} is absent (fetch tests/data/benchmark-grids)",
            path.display()
        );
        return None;
    }
    Some(path)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridox"))
        .args(args)
        .output()
        .expect("failed to run the gridox binary")
}

fn stdout_of(out: &Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn dc_reads_matpower_and_says_it_drops_the_shunt() {
    let Some(case) = case14() else { return };
    let text = stdout_of(&run(&["dc", case.to_str().unwrap()]));
    assert!(text.contains("14 bus(es), 20 branch(es)"), "{text}");
    assert!(text.contains("1 shunt(s) ignored"), "{text}");
}

#[test]
fn the_ac_commands_read_matpower() {
    let Some(case) = case14() else { return };
    let path = case.to_str().unwrap();

    let text = stdout_of(&run(&["sensitivity", path, "--dp", "3"]));
    assert!(text.contains("14 bus(es), 20 branch(es)"), "{text}");

    let text = stdout_of(&run(&["qv", path, "--bus", "8"]));
    assert!(text.contains("14 bus(es)"), "{text}");

    let text = stdout_of(&run(&["continuation", path]));
    assert!(text.contains("loadability limit"), "{text}");
}

/// The shunt at bus 9 (index 8) is passed on: its 19 MVAr hold the bus's
/// voltage up, so dropping it would move the reactive margin there.
#[test]
fn the_shunt_reaches_the_ac_commands() {
    let Some(case) = case14() else { return };
    let with = stdout_of(&run(&["qv", case.to_str().unwrap(), "--bus", "8"]));

    let text = std::fs::read_to_string(&case).unwrap();
    let without = text.replace("19\t1\t1.056", "0\t1\t1.056");
    assert_ne!(
        text, without,
        "case14's bus 9 row changed; update this test"
    );
    let dir = std::env::temp_dir().join(format!("gridox-formats-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let stripped = dir.join("case14.m");
    std::fs::write(&stripped, without).unwrap();
    let without = stdout_of(&run(&["qv", stripped.to_str().unwrap(), "--bus", "8"]));
    let _ = std::fs::remove_dir_all(&dir);

    assert_ne!(with, without, "the shunt made no difference");
}

#[test]
fn base_mva_is_for_pgm_documents_only() {
    let Some(case) = case14() else { return };
    let out = run(&["qv", case.to_str().unwrap(), "--base-mva", "50"]);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("MATPOWER states its own"), "{err}");
}
