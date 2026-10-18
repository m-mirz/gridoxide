//! The `gridoxide estimate` subcommand, in both domains.
//!
//! The numbers are gated elsewhere — against power-grid-model's own state
//! estimation fixtures in `tests/se_pgm_test.rs`, and against its *asymmetric*
//! answer for the same document in `tests/se_three_phase_test.rs`. What is
//! pinned here is that the command reaches them: the phase-domain estimator
//! existed for some time with nothing outside the test suite able to call it,
//! because standing its model up took eight calls in a particular order.
//!
//! The other thing pinned here is that the report says *which* domain answered.
//! A per-phase voltage list and a per-node one are different claims about the
//! same network, and a reader who mistakes one for the other has been misled by
//! the tool rather than by the data.

use std::path::PathBuf;
use std::process::{Command, Output};

fn document() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/state_estimation/transmission-case/input.json")
}

fn estimate(extra: &[&str]) -> Output {
    let path = document();
    let mut args = vec!["estimate", path.to_str().unwrap()];
    args.extend_from_slice(extra);
    Command::new(env!("CARGO_BIN_EXE_gridoxide"))
        .args(&args)
        .output()
        .expect("failed to run the gridoxide binary")
}

fn stdout_of(out: &Output) -> String {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn it_estimates_symmetrically_and_names_the_domain() {
    let text = stdout_of(&estimate(&[]));

    assert!(text.contains("symmetric, one per node"), "{text}");
    assert!(text.contains("Converged in"), "{text}");
    assert!(text.contains("Estimated voltages:"), "{text}");
    // One row per node, and no phase labels anywhere.
    assert!(text.contains("node 1: |V| ="), "{text}");
    assert!(!text.contains("phase"), "{text}");

    assert!(text.contains("Observability: rank"), "{text}");
    assert!(text.contains("Bad data: chi-squared"), "{text}");
    assert!(text.contains("not rejected at 5%"), "{text}");
}

/// `--asymmetric` estimates in the phase domain, and says so.
#[test]
fn the_asymmetric_flag_estimates_in_the_phase_domain() {
    let text = stdout_of(&estimate(&["--asymmetric"]));

    assert!(text.contains("phase domain, three per node"), "{text}");
    assert!(text.contains("Converged in"), "{text}");

    // Three rows per node, labelled by phase rather than by bus index — a
    // reader should not have to know that a node's phases are `3n`, `3n+1`,
    // `3n+2` to read the answer.
    for phase in ["a", "b", "c"] {
        assert!(text.contains(&format!("node 1 phase {phase}: |V| =")), "{text}");
    }

    // Three times the buses and roughly three times the measurements.
    assert!(text.contains("36 bus(es)"), "{text}");

    assert!(text.contains("Observability: rank"), "{text}");
    assert!(text.contains("Bad data: chi-squared"), "{text}");
    assert!(text.contains("not rejected at 5%"), "{text}");
}

/// The two domains agree about a balanced network, which is the check that the
/// phase-domain path is solving the same problem rather than merely solving.
///
/// This fixture is balanced, so each node's three phases must come back at the
/// symmetric magnitude with the angles 120 degrees apart. A phase-domain
/// estimator that had its sequence transform backwards would still converge,
/// and would still look plausible read on its own.
#[test]
fn the_two_domains_agree_about_a_balanced_network() {
    let symmetric = stdout_of(&estimate(&[]));
    let asymmetric = stdout_of(&estimate(&["--asymmetric"]));

    let magnitude = |text: &str, prefix: &str| -> f64 {
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with(prefix))
            .unwrap_or_else(|| panic!("no line starting {prefix:?} in:\n{text}"));
        let after = line.split("|V| = ").nth(1).unwrap();
        after.split_whitespace().next().unwrap().parse().unwrap()
    };
    let angle = |text: &str, prefix: &str| -> f64 {
        let line = text.lines().find(|l| l.trim_start().starts_with(prefix)).unwrap();
        let after = line.split("angle = ").nth(1).unwrap();
        after.split_whitespace().next().unwrap().parse().unwrap()
    };

    for node in [1u32, 2, 3] {
        let sym_mag = magnitude(&symmetric, &format!("node {node}: "));
        let sym_ang = angle(&symmetric, &format!("node {node}: "));
        for (phase, rotation) in [("a", 0.0), ("b", -120.0), ("c", 120.0)] {
            let prefix = format!("node {node} phase {phase}: ");
            assert!(
                (magnitude(&asymmetric, &prefix) - sym_mag).abs() < 1e-5,
                "node {node} phase {phase} magnitude"
            );
            assert!(
                (angle(&asymmetric, &prefix) - (sym_ang + rotation)).abs() < 1e-4,
                "node {node} phase {phase} angle"
            );
        }
    }
}

/// A document with no sensors is refused rather than estimated, in either
/// domain.
///
/// With no measurements the gain matrix is identically zero, so "singular"
/// would be a true statement about the wrong problem. The document is the same
/// one every test above uses, with its sensor arrays emptied — so the refusal
/// is about the sensors and not about some other property of a different file.
#[test]
fn a_document_without_sensors_is_refused() {
    let raw = std::fs::read_to_string(document()).expect("the fixture is committed");
    let mut doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let data = doc.get_mut("data").unwrap().as_object_mut().unwrap();
    for key in data.keys().cloned().collect::<Vec<_>>() {
        if key.ends_with("_sensor") {
            data.insert(key, serde_json::json!([]));
        }
    }

    let dir = std::env::temp_dir().join("gridoxide-estimate-no-sensors");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("input.json");
    std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();

    for extra in [vec![], vec!["--asymmetric"]] {
        let mut args = vec!["estimate", path.to_str().unwrap()];
        args.extend_from_slice(&extra);
        let out = Command::new(env!("CARGO_BIN_EXE_gridoxide"))
            .args(&args)
            .output()
            .expect("runs");
        assert!(!out.status.success(), "{extra:?} should have been refused");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("no usable sensors"), "{extra:?}: {err}");
    }

    std::fs::remove_dir_all(&dir).ok();
}
