//! The commands on inputs other than power-grid-model JSON: `dc`,
//! `sensitivity`, `qv` and `continuation` on every network format, `estimate`
//! on a MATPOWER case with a measurement file, and `opf` on a MATPOWER
//! case's own costs.
//!
//! The solvers are gated elsewhere; what is pinned here is that each command
//! reaches them from a MATPOWER case, with its shunts (case14 has one, at bus
//! 9), and that what does not apply to the format is refused.

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

/// grid-bench's `case14~exact` measurement set (`cases/measurements.py`):
/// |V| and P, Q injections at every bus, without noise, from the solved case.
fn exact_measurements() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/case14-exact.meas.json")
}

/// Exact measurements put the weighted least-squares optimum on the true state,
/// which the measurement file records, so the estimate must land on it.
#[test]
fn estimate_reads_matpower_with_its_measurements() {
    let Some(case) = case14() else { return };
    let meas = exact_measurements();
    let text = stdout_of(&run(&[
        "estimate",
        case.to_str().unwrap(),
        "--measurements",
        meas.to_str().unwrap(),
    ]));
    assert!(text.contains("Converged"), "{text}");

    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meas).unwrap()).unwrap();
    let truth = doc["true_state"].as_object().unwrap();
    for (bus, state) in truth {
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("node {bus}:")))
            .unwrap_or_else(|| panic!("no node {bus} in {text}"));
        let numbers: Vec<f64> = line
            .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
            .filter_map(|s| s.parse().ok())
            .collect();
        let (vm, va) = (numbers[numbers.len() - 2], numbers[numbers.len() - 1]);
        assert!((vm - state[0].as_f64().unwrap()).abs() < 1e-6, "{line}");
        assert!((va - state[1].as_f64().unwrap()).abs() < 1e-5, "{line}");
    }
}

#[test]
fn a_matpower_estimate_needs_measurements_that_fit_the_case() {
    let Some(case) = case14() else { return };
    let path = case.to_str().unwrap();

    let out = run(&["estimate", path]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--measurements"));

    // A flow whose stated ends are not its row's is another case's.
    let wrong = serde_json::json!({"measurements": [
        {"kind": "p_from", "branch_row": 0, "from_bus": 1, "to_bus": 3, "value": 1.0, "sigma": 0.1}
    ]});
    let file = std::env::temp_dir().join(format!("gridox-wrong-{}.json", std::process::id()));
    std::fs::write(&file, wrong.to_string()).unwrap();
    let out = run(&["estimate", path, "--measurements", file.to_str().unwrap()]);
    let _ = std::fs::remove_file(&file);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("to_bus 3"), "{err}");
}

/// MATPOWER's `runopf` reports 8081.53 $/h for case14.
#[cfg(feature = "opf")]
#[test]
fn ac_opf_reads_matpower() {
    let Some(case) = case14() else { return };
    let path = case.to_str().unwrap();
    let text = stdout_of(&run(&["opf", path]));
    let cost: f64 = text
        .lines()
        .find_map(|l| l.strip_prefix("total cost: "))
        .and_then(|l| l.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no cost in {text}"));
    assert!((cost - 8081.53).abs() < 0.05, "{cost}");

    // The DC OPF and --data are a PGM document's.
    for extra in [&["--dc"][..], &["--data", "x.json"]] {
        let mut args = vec!["opf", path];
        args.extend_from_slice(extra);
        assert_eq!(run(&args).status.code(), Some(1), "{args:?}");
    }
}
