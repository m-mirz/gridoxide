//! The `gridoxide security` subcommand.
//!
//! `main.rs`'s argument handling is hand-rolled, so it is pinned here by
//! driving the built binary rather than calling into the library.
//!
//! The exit code carries meaning and is the point of most of these: 0 secure,
//! 1 insecure, 2 the invocation was wrong. Conflating the last two is how a
//! study silently passes because the CRAC failed to load.

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

#[test]
fn an_insecure_network_reports_its_overloads_and_exits_one() {
    let network = data("ucte", "TestCase12Nodes.uct");
    let crac = data("rao", "crac-for-12nodes.json");
    let out = run(&["security", network.to_str().unwrap(), "--crac", crac.to_str().unwrap()]);
    let text = stdout(&out);

    assert!(text.contains("INSECURE"), "{text}");
    assert!(text.contains("OVERLOAD"), "{text}");
    // Every perimeter the CRAC defines should be named.
    for instant in ["preventive", "outage", "auto", "curative"] {
        assert!(text.contains(instant), "no `{instant}` perimeter in output:\n{text}");
    }
    assert_eq!(out.status.code(), Some(1), "an insecure network must exit 1:\n{text}");
}

#[test]
fn the_json_form_is_machine_readable() {
    let network = data("ucte", "TestCase12Nodes.uct");
    let crac = data("rao", "crac-for-12nodes.json");
    let out = run(&[
        "security",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--json",
    ]);
    let text = stdout(&out);
    let doc: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("not JSON: {e}\n{text}"));

    assert_eq!(doc["secure"], false);
    assert!(doc["min_margin_mw"].as_f64().expect("a margin") < 0.0);
    assert_eq!(doc["unresolved"], 0);
    assert_eq!(doc["skipped_cnecs"], 0);
    let perimeters = doc["perimeters"].as_array().expect("perimeters");
    assert_eq!(perimeters.len(), 4);
    // The base case is the one with no contingency, and there is exactly one.
    let base: Vec<_> = perimeters.iter().filter(|p| p["contingency"].is_null()).collect();
    assert_eq!(base.len(), 1);
    assert!(!base[0]["cnecs"].as_array().expect("cnecs").is_empty());
}

#[test]
fn a_secure_network_exits_zero() {
    // Same network, but with every threshold relaxed far beyond any flow. This
    // is the case that distinguishes "found no violations" from "found no
    // CNECs", which would otherwise both print nothing and exit 0.
    let network = data("ucte", "TestCase12Nodes.uct");
    let source = std::fs::read_to_string(data("rao", "crac-for-12nodes.json")).expect("crac");
    let (crac, _) = gridoxide::rao::crac_json::parse(&source).expect("parse");
    let mut relaxed = crac.clone();
    for cnec in &mut relaxed.flow_cnecs {
        for threshold in &mut cnec.thresholds {
            threshold.unit = gridoxide::rao::Unit::Megawatt;
            threshold.min = Some(-1e6);
            threshold.max = Some(1e6);
        }
    }
    let path = std::env::temp_dir().join("gridoxide-security-relaxed.rao.json");
    std::fs::write(&path, relaxed.to_json().expect("serialize")).expect("write");

    let out = run(&["security", network.to_str().unwrap(), "--crac", path.to_str().unwrap()]);
    let text = stdout(&out);
    assert!(text.contains("SECURE"), "{text}");
    assert!(!text.contains("INSECURE"), "{text}");
    assert!(text.contains("0 overload(s)"), "{text}");
    assert_eq!(out.status.code(), Some(0), "{text}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn the_native_companion_document_is_accepted_too() {
    // `--crac` takes either format; the native one is tried first so that a
    // malformed companion reports its own error rather than "not a CRAC".
    let network = data("ucte", "TestCase12Nodes.uct");
    let source = std::fs::read_to_string(data("rao", "crac-for-12nodes.json")).expect("crac");
    let (crac, _) = gridoxide::rao::crac_json::parse(&source).expect("parse");
    let path = std::env::temp_dir().join("gridoxide-security-native.rao.json");
    std::fs::write(&path, crac.to_json().expect("serialize")).expect("write");

    let out = run(&["security", network.to_str().unwrap(), "--crac", path.to_str().unwrap()]);
    let text = stdout(&out);
    assert!(text.contains("INSECURE"), "{text}");
    // Same answer as the OpenRAO-format run: a round trip must not change the
    // verdict.
    assert!(text.contains("182.3"), "{text}");
    assert_eq!(out.status.code(), Some(1));
    let _ = std::fs::remove_file(path);
}

#[test]
fn a_broken_invocation_exits_two_rather_than_one() {
    let network = data("ucte", "TestCase12Nodes.uct");

    // No --crac at all.
    let out = run(&["security", network.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(2), "missing --crac should exit 2");

    // A CRAC that does not exist.
    let out = run(&["security", network.to_str().unwrap(), "--crac", "/nonexistent.json"]);
    assert_eq!(out.status.code(), Some(2));

    // A network whose format cannot be told.
    let crac = data("rao", "crac-for-12nodes.json");
    let out = run(&["security", "network.wat", "--crac", crac.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("expected a .uct or .xiidm"));
}

#[test]
fn an_unresolvable_element_is_warned_about_rather_than_ignored() {
    let network = data("ucte", "TestCase12Nodes.uct");
    let source = std::fs::read_to_string(data("rao", "crac-for-12nodes.json")).expect("crac");
    let (mut crac, _) = gridoxide::rao::crac_json::parse(&source).expect("parse");
    crac.flow_cnecs[0].network_element = "NO SUCH BRANCH".into();
    let path = std::env::temp_dir().join("gridoxide-security-unresolved.rao.json");
    std::fs::write(&path, crac.to_json().expect("serialize")).expect("write");

    let out = run(&["security", network.to_str().unwrap(), "--crac", path.to_str().unwrap()]);
    let text = stdout(&out);
    assert!(text.contains("warning"), "an unresolved element must be warned about:\n{text}");
    assert!(text.contains("NO SUCH BRANCH"), "{text}");
    assert!(text.contains("CNEC(s) skipped"), "{text}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn iidm_networks_work_as_well_as_ucte() {
    let network = data("iidm", "TestCase12Nodes.xiidm");
    let crac = data("rao", "crac-for-12nodes.json");
    let out = run(&[
        "security",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--json",
    ]);
    let text = stdout(&out);
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(doc["unresolved"], 0, "the CRAC should resolve against the IIDM form too");

    // And it must reach the same verdict as the UCTE form of the same network,
    // since the two importers agree on the flows.
    let ucte = data("ucte", "TestCase12Nodes.uct");
    let other = run(&["security", ucte.to_str().unwrap(), "--crac", crac.to_str().unwrap(), "--json"]);
    let expected: serde_json::Value = serde_json::from_str(&stdout(&other)).expect("json");
    assert_eq!(doc["secure"], expected["secure"]);
    let a = doc["min_margin_mw"].as_f64().unwrap();
    let b = expected["min_margin_mw"].as_f64().unwrap();
    assert!((a - b).abs() < 1e-6, "IIDM says {a} MW, UCTE says {b} MW");
}

// ---------------------------------------------------------------------------
// `gridoxide rao`
// ---------------------------------------------------------------------------

#[test]
fn the_rao_command_reports_what_to_do() {
    let network = data("ucte", "TestCase12Nodes.uct");
    let crac = data("rao", "crac-topology-helps.json");
    let out = run(&["rao", network.to_str().unwrap(), "--crac", crac.to_str().unwrap()]);
    let text = stdout(&out);
    assert!(text.contains("APPLY"), "no action recommended:\n{text}");
    assert!(text.contains("Open tie-line FR DE"), "{text}");
    // The preventive perimeter goes from insecure to secure.
    assert!(text.contains("-460.8 -> 500.0"), "{text}");
    assert!(text.contains("SECURE"), "{text}");
    // Pulling a curative CNEC forward changes the answer, so it is reported.
    assert!(text.contains("no curative action"), "{text}");
    assert!(text.contains("preventive perimeter"), "{text}");
}

#[test]
fn the_rao_command_emits_json() {
    let network = data("ucte", "TestCase12Nodes.uct");
    let crac = data("rao", "crac-for-12nodes.json");
    let out = run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--json",
    ]);
    let text = stdout(&out);
    let doc: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("not JSON: {e}\n{text}"));
    let perimeters = doc["perimeters"].as_array().expect("perimeters");
    assert!(!perimeters.is_empty());

    // The first is the preventive perimeter, which covers several instants.
    let preventive = &perimeters[0];
    assert!(preventive["contingency"].is_null(), "{preventive}");
    let instants = preventive["instants"].as_array().expect("instants");
    assert!(instants.len() >= 2, "the preventive perimeter should span states: {preventive}");
    assert!(
        preventive["final_margin_mw"].as_f64().unwrap()
            > preventive["initial_margin_mw"].as_f64().unwrap(),
        "{preventive}"
    );
    // Both halves are used on this fixture, and neither helps alone.
    assert!(!preventive["network_actions"].as_array().unwrap().is_empty(), "{preventive}");
    let setpoints = preventive["setpoints"].as_array().expect("setpoints");
    assert_eq!(setpoints.len(), 1);
    assert!(setpoints[0]["tap"].as_i64().is_some(), "a PST set-point should carry its tap");
    assert!(preventive["leaves"].as_u64().unwrap() > 0, "candidates should have been evaluated");
    assert!(doc["final_margin_mw"].as_f64().unwrap() > doc["initial_margin_mw"].as_f64().unwrap());
}

#[test]
fn the_rao_depth_flag_is_honoured_and_validated() {
    let network = data("ucte", "TestCase12Nodes.uct");
    let crac = data("rao", "crac-topology-helps.json");

    let out = run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--depth",
        "0",
        "--json",
    ]);
    let doc: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("json");
    for perimeter in doc["perimeters"].as_array().expect("perimeters") {
        assert!(
            perimeter["network_actions"].as_array().unwrap().is_empty(),
            "depth 0 took an action: {perimeter}"
        );
    }
    assert_eq!(doc["pulled_forward"].as_u64().unwrap(), 2, "{doc}");

    let out = run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--depth",
        "banana",
    ]);
    assert_eq!(out.status.code(), Some(2), "a bad --depth should exit 2");
}

// ---------------------------------------------------------------------------
// `--validate-ac`
// ---------------------------------------------------------------------------

#[test]
fn ac_validation_is_absent_unless_asked_for() {
    // A consumer that never asked should not have to distinguish "AC said
    // nothing" from "AC was never run", so the key is missing rather than null.
    let network = data("ucte", "TestCase12Nodes.uct");
    let crac = data("rao", "crac-for-12nodes.json");
    let out = run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--json",
    ]);
    let text = stdout(&out);
    assert!(!text.contains("ac_validation"), "{text}");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("valid json");
    assert!(doc.get("ac_validation").is_none());
}

#[test]
fn ac_validation_reports_both_models_per_perimeter() {
    let network = data("ucte", "TestCase12Nodes.uct");
    let crac = data("rao", "crac-for-12nodes.json");
    let out = run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--validate-ac",
        "--json",
    ]);
    let text = stdout(&out);
    let doc: serde_json::Value = serde_json::from_str(&text).expect("valid json");
    let ac = doc.get("ac_validation").expect("ac_validation present");

    let perimeters = ac["perimeters"].as_array().expect("perimeters");
    assert_eq!(perimeters.len(), doc["perimeters"].as_array().unwrap().len());
    for p in perimeters {
        // Both figures, so the disagreement is visible rather than inferred.
        assert!(p["dc_margin_mw"].is_number(), "{p}");
        assert!(p["ac_margin_mw"].is_number(), "{p}");
        assert!(
            ["accepted", "diverged", "insecure", "regressed"]
                .contains(&p["verdict"].as_str().unwrap_or("")),
            "unexpected verdict in {p}"
        );
    }
}

#[test]
fn a_rejected_plan_exits_one_even_when_the_search_called_it_secure() {
    // The exit code is what a script acts on. A plan the DC search liked and
    // the AC check rejected has to be reported as a failure, or the second
    // stage is decoration.
    let network = data("ucte", "TestCase12Nodes.uct");
    let crac = data("rao", "crac-for-12nodes.json");
    let out = run(&[
        "rao",
        network.to_str().unwrap(),
        "--crac",
        crac.to_str().unwrap(),
        "--validate-ac",
    ]);
    let text = stdout(&out);
    assert!(text.contains("AC re-validation"), "{text}");
    assert_eq!(out.status.code(), Some(1), "{text}");
}
