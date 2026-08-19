//! gridoxide against powsybl-open-rao's own Cucumber expectations — the
//! external gate of `plans/RAO_PLAN.md` §8.3.
//!
//! Every other check in this repository is one gridoxide wrote for itself. A
//! finite-differenced derivative proves a derivative; a screening-versus-resolve
//! comparison proves two of gridoxide's own paths agree. Neither says whether
//! the answers are *right* in the sense an operator cares about.
//!
//! These do. `tests/data/rao/features/dc_scenarios.feature` holds eight
//! scenarios copied verbatim from the reference implementation's test suite,
//! written by its authors against inputs it ships, stating margins to the
//! decimal and naming which remedial actions should be used.
//!
//! # The tolerance is theirs, not ours
//!
//! `max(5 MW, 1.5% of the expected value)` — `RaoSteps.flowMegawattTolerance`
//! in the reference. Using the tolerance its own authors judge it by is the only
//! defensible choice; inventing one here would let a disagreement be tuned away.
//!
//! # What it found
//!
//! Five defects, in an afternoon, all of the same shape: internally consistent,
//! externally wrong, and invisible to any test gridoxide could write for itself.
//! An inverted tap sign that left every margin correct and every tap label
//! mirrored; ampere thresholds converted at the network's base rather than the
//! voltage the CRAC names; an LP optimizing a different limit from the one being
//! measured; a `for CORE CC` step this harness was dropping, which rewrites
//! every nominal voltage and so moves every susceptance by 11%; and a
//! tap-rounding rule that settled on the wrong side of the optimum and stayed
//! there, convergent and wrong.
//!
//! # What a failure means
//!
//! Not necessarily a bug. These are two heuristic search trees, and the plan's
//! §8.3 says so: agreement on the *objective* is meaningful, a different set of
//! actions reaching the same margin is not a defect. So the harness reports
//! every assertion — matched and unmatched — rather than stopping at the first,
//! and the test asserts on the aggregate. A regression shows up as a count.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use gridoxide::opf::ipm::IpmSolver;
use gridoxide::rao::crac::*;
use gridoxide::rao::{crac_json, evaluate_with, run, Network, Resolution, SearchOptions};
use gridoxide::ucte;

fn features_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/rao/features")
}

/// The reference's own tolerance, from `RaoSteps.flowMegawattTolerance`.
fn megawatt_tolerance(expected: f64) -> f64 {
    f64::max(5.0, 0.015 * expected.abs())
}

// ---------------------------------------------------------------------------
// Parsing the feature file
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Expect {
    Secured(bool),
    WorstMargin { value: f64, cnec: Option<String> },
    CnecMargin { cnec: String, value: f64 },
    PstTap { action: String, tap: i32 },
    ActionUsed { action: String },
    ActionCount(usize),
    /// A step this harness does not implement, kept so it is counted rather
    /// than quietly dropped.
    Unsupported(String),
}

#[derive(Debug, Default, Clone)]
struct Scenario {
    name: String,
    network: String,
    /// The scenario asked for `for CORE CC`, which rewrites the network's
    /// nominal voltages before anything else happens.
    core_cc: bool,
    crac: String,
    config: String,
    expectations: Vec<Expect>,
}

fn quoted(line: &str) -> Option<String> {
    let start = line.find('"')? + 1;
    let rest = &line[start..];
    Some(rest[..rest.find('"')?].to_string())
}

/// The first number in a line, after any quoted text has been removed so an id
/// containing digits cannot be mistaken for one.
fn number_after_quotes(line: &str) -> Option<f64> {
    let mut stripped = String::new();
    let mut inside = false;
    for c in line.chars() {
        if c == '"' {
            inside = !inside;
            continue;
        }
        if !inside {
            stripped.push(c);
        }
    }
    let mut token = String::new();
    for c in stripped.chars() {
        if c.is_ascii_digit() || c == '.' || (c == '-' && token.is_empty()) {
            token.push(c);
        } else if !token.is_empty() {
            if let Ok(v) = token.parse::<f64>() {
                return Some(v);
            }
            token.clear();
        }
    }
    token.parse::<f64>().ok()
}

fn parse(text: &str) -> Vec<Scenario> {
    let mut scenarios = Vec::new();
    let mut current: Option<Scenario> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("Scenario:") {
            if let Some(s) = current.take() {
                scenarios.push(s);
            }
            current = Some(Scenario {
                name: rest.split_whitespace().next().unwrap_or("?").to_string(),
                ..Default::default()
            });
            continue;
        }
        let Some(scenario) = current.as_mut() else { continue };

        if line.starts_with("Given network file is") {
            scenario.network = quoted(line).unwrap_or_default();
            // `for CORE CC` is not decoration. The reference's own
            // `CoreCcPreprocessor` rewrites every voltage level — 380 kV to 400,
            // 220 to 225 — and since the nominal voltage *is* the per-unit base
            // that moves every susceptance by 11%. Dropping the suffix silently
            // is how two of these scenarios looked like optimizer defects.
            scenario.core_cc = line.contains("for CORE CC");
        } else if line.starts_with("Given crac file is") {
            scenario.crac = quoted(line).unwrap_or_default();
        } else if line.starts_with("Given configuration file is") {
            scenario.config = quoted(line).unwrap_or_default();
        } else if line.starts_with("Then") {
            scenario.expectations.push(expectation(line));
        }
    }
    if let Some(s) = current {
        scenarios.push(s);
    }
    scenarios
}

fn expectation(line: &str) -> Expect {
    if line.contains("security status should be") {
        return match quoted(line).as_deref() {
            Some("SECURED") => Expect::Secured(true),
            Some("UNSECURED") => Expect::Secured(false),
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("the worst margin is") && line.contains(" MW") {
        let cnec = line.contains("on cnec").then(|| quoted(line)).flatten();
        return match number_after_quotes(line) {
            Some(value) => Expect::WorstMargin { value, cnec },
            None => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("margin on cnec") && line.contains(" MW") {
        return match (quoted(line), number_after_quotes(line)) {
            (Some(cnec), Some(value)) => Expect::CnecMargin { cnec, value },
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("the tap of PstRangeAction") {
        return match (quoted(line), number_after_quotes(line)) {
            (Some(action), Some(tap)) => Expect::PstTap { action, tap: tap as i32 },
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("remedial action") && line.contains("is used in preventive") {
        return match quoted(line) {
            Some(action) => Expect::ActionUsed { action },
            None => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("remedial actions are used in preventive") {
        return match number_after_quotes(line) {
            Some(n) => Expect::ActionCount(n as usize),
            None => Expect::Unsupported(line.to_string()),
        };
    }
    Expect::Unsupported(line.to_string())
}

// ---------------------------------------------------------------------------
// Running one scenario
// ---------------------------------------------------------------------------

/// Resolve a path from the feature file by its basename.
///
/// The steps keep the reference's own layout (`epic4/SL_ep4us2_4MR_MW.json`)
/// so the text stays verbatim; the files are vendored flat.
fn resolve(reference: &str) -> PathBuf {
    let name = reference.rsplit('/').next().unwrap_or(reference);
    let flat = features_dir().join(name);
    if flat.exists() {
        return flat;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte").join(name)
}

/// Map the reference's `RaoParameters` onto [`SearchOptions`].
///
/// Only the knobs that change an answer are read. A parameter this does not
/// understand is ignored *silently on purpose*: the alternative is to fail on
/// the dozens of AC, loop-flow and solver settings these files carry, none of
/// which apply to a DC run.
fn options_from(config: &Path) -> SearchOptions {
    let mut options = SearchOptions::default();
    let Ok(text) = std::fs::read_to_string(config) else { return options };
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(&text) else { return options };

    if let Some(topology) = doc.get("topological-actions-optimization") {
        if let Some(v) = topology.get("absolute-minimum-impact-threshold").and_then(|v| v.as_f64()) {
            options.absolute_min_impact = v;
        }
        if let Some(v) = topology.get("relative-minimum-impact-threshold").and_then(|v| v.as_f64()) {
            options.relative_min_impact = v;
        }
    }
    if let Some(range) = doc.get("range-actions-optimization") {
        if let Some(v) = range.get("pst-ra-min-impact-threshold").and_then(|v| v.as_f64()) {
            options.linear.pst_penalty = v;
        }
        if let Some(v) = range.get("injection-ra-min-impact-threshold").and_then(|v| v.as_f64()) {
            options.linear.injection_penalty = v;
        }
    }
    let extension = doc
        .get("extensions")
        .and_then(|e| e.get("open-rao-search-tree-parameters"));
    if let Some(range) = extension.and_then(|e| e.get("range-actions-optimization")) {
        if let Some(v) = range.get("max-mip-iterations").and_then(|v| v.as_u64()) {
            options.linear.max_iterations = v as usize;
        }
        if let Some(v) = range.get("pst-sensitivity-threshold").and_then(|v| v.as_f64()) {
            options.linear.sensitivity_threshold = v;
        }
    }
    if let Some(topology) = extension.and_then(|e| e.get("topological-actions-optimization")) {
        if let Some(v) = topology.get("max-preventive-search-tree-depth").and_then(|v| v.as_u64()) {
            // The reference's default is i32::MAX; anything that large is a
            // depth bound in name only, and running it would evaluate every
            // combination of a corpus this harness has no time budget for.
            options.max_depth = (v as usize).min(3);
        }
    }
    options
}

struct Outcome {
    matched: Vec<String>,
    mismatched: Vec<String>,
    unsupported: Vec<String>,
}

fn check(scenario: &Scenario) -> Outcome {
    let mut outcome =
        Outcome { matched: Vec::new(), mismatched: Vec::new(), unsupported: Vec::new() };

    let options = if scenario.core_cc {
        ucte::UcteOptions::core_capacity_calculation()
    } else {
        ucte::UcteOptions::default()
    };
    let net = ucte::read_with(resolve(&scenario.network), &options).expect("network");
    let (crac, _) = crac_json::read(resolve(&scenario.crac)).expect("crac");
    let search_options = options_from(&resolve(&scenario.config));
    let resolution = Resolution::new(&crac, &net.branch_ids);
    let view = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        base_mva: net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &view, &resolution, &mut solver, &search_options);

    // Per-CNEC margins with the preventive decisions in force — what the
    // reference calls "after PRA".
    let after_pra = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &plan.preventive.transformers,
        branch_ids: &net.branch_ids,
        base_mva: net.base_mva,
    };
    let evaluated =
        evaluate_with(&crac, &after_pra, &resolution, &plan.preventive.open_branches);
    let margins: HashMap<&str, f64> = evaluated
        .perimeters
        .iter()
        .flat_map(|p| p.cnecs.iter())
        .map(|c| (crac.flow_cnecs[c.cnec].id.as_str(), c.margin_mw))
        .collect();

    let used: Vec<&str> = plan
        .preventive
        .network_actions
        .iter()
        .map(|&a| crac.network_actions[a].id.as_str())
        .chain(
            plan.preventive
                .setpoints
                .iter()
                .filter(|s| s.moved())
                .map(|s| crac.range_actions[s.action].id.as_str()),
        )
        .collect();

    let mut record = |ok: bool, detail: String| {
        if ok {
            outcome.matched.push(detail);
        } else {
            outcome.mismatched.push(detail);
        }
    };

    for expectation in &scenario.expectations {
        match expectation {
            Expect::Secured(want) => {
                let got = plan.is_secure();
                record(got == *want, format!("security {got} (expected {want})"));
            }
            Expect::WorstMargin { value, cnec: None } => {
                let got = plan.final_margin_mw;
                record(
                    (got - value).abs() <= megawatt_tolerance(*value),
                    format!("worst margin {got:.2} MW (expected {value})"),
                );
            }
            Expect::WorstMargin { value, cnec: Some(id) } => {
                let got = margins.get(id.as_str()).copied();
                record(
                    got.is_some_and(|g| (g - value).abs() <= megawatt_tolerance(*value)),
                    format!("worst margin on `{id}` {got:?} (expected {value})"),
                );
            }
            Expect::CnecMargin { cnec, value } => {
                let got = margins.get(cnec.as_str()).copied();
                record(
                    got.is_some_and(|g| (g - value).abs() <= megawatt_tolerance(*value)),
                    format!("margin on `{cnec}` {got:?} (expected {value})"),
                );
            }
            Expect::PstTap { action, tap } => {
                let got = plan
                    .preventive
                    .setpoints
                    .iter()
                    .find(|s| crac.range_actions[s.action].id == *action)
                    .and_then(|s| s.tap);
                record(got == Some(*tap), format!("tap of `{action}` {got:?} (expected {tap})"));
            }
            Expect::ActionUsed { action } => {
                let got = used.contains(&action.as_str());
                record(got, format!("`{action}` used: {got}"));
            }
            Expect::ActionCount(n) => {
                let got = used.len();
                record(got == *n, format!("{got} action(s) used (expected {n})"));
            }
            Expect::Unsupported(line) => outcome.unsupported.push(line.clone()),
        }
    }
    outcome
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// Run every vendored scenario and report, then assert on the aggregate.
///
/// The count in the final assertion is a **recorded baseline**, not a target.
/// Raising it is progress; a drop is a regression and the printed report says
/// which assertion moved.
#[test]
fn the_reference_implementations_own_expectations() {
    let text = std::fs::read_to_string(features_dir().join("dc_scenarios.feature"))
        .expect("feature file");
    let scenarios = parse(&text);
    assert_eq!(scenarios.len(), 8, "expected 8 vendored scenarios");

    let (mut matched, mut mismatched, mut unsupported) = (0usize, 0usize, 0usize);
    let mut report = String::new();
    for scenario in &scenarios {
        let outcome = check(scenario);
        matched += outcome.matched.len();
        mismatched += outcome.mismatched.len();
        unsupported += outcome.unsupported.len();
        report.push_str(&format!(
            "\n{}  ({} matched, {} mismatched, {} unsupported)\n",
            scenario.name,
            outcome.matched.len(),
            outcome.mismatched.len(),
            outcome.unsupported.len()
        ));
        for detail in &outcome.matched {
            report.push_str(&format!("    ok    {detail}\n"));
        }
        for detail in &outcome.mismatched {
            report.push_str(&format!("    DIFF  {detail}\n"));
        }
        for line in &outcome.unsupported {
            report.push_str(&format!("    skip  {line}\n"));
        }
    }
    let total = matched + mismatched;
    println!(
        "{report}\n{matched}/{total} checkable assertions match the reference \
         ({unsupported} steps unsupported)"
    );

    assert!(
        matched >= BASELINE_MATCHED,
        "{matched}/{total} matched, baseline is {BASELINE_MATCHED} — a drop is a regression:\n{report}"
    );
}

/// How many of the reference's assertions currently hold: **42 of 42**.
///
/// All eight scenarios match completely — every margin, every tap, every named
/// action, the action count and the security status — at the reference's own
/// tolerance.
///
/// It is still a recorded number rather than an assertion of perfection. These
/// are two heuristic search trees and `plans/RAO_PLAN.md` §8.3 says up front
/// that a different set of actions reaching the same margin is not a defect; a
/// future scenario may legitimately disagree. Raising this is progress, a drop
/// is a regression, and the printed report says which assertion moved.
const BASELINE_MATCHED: usize = 42;

#[test]
fn every_scenario_names_inputs_that_exist() {
    // A scenario whose fixture is missing would otherwise be silently skipped,
    // and the gate would pass by testing nothing.
    let text = std::fs::read_to_string(features_dir().join("dc_scenarios.feature"))
        .expect("feature file");
    for scenario in parse(&text) {
        for reference in [&scenario.network, &scenario.crac, &scenario.config] {
            assert!(!reference.is_empty(), "{}: a Given step is missing", scenario.name);
            assert!(
                resolve(reference).exists(),
                "{}: `{reference}` resolves to {} which does not exist",
                scenario.name,
                resolve(reference).display()
            );
        }
        assert!(!scenario.expectations.is_empty(), "{}: nothing to check", scenario.name);
    }
}
