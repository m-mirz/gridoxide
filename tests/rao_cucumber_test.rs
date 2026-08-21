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
//! Eight defects, all of the same shape: internally consistent, externally
//! wrong, and invisible to any test gridoxide could write for itself.
//!
//! An inverted tap sign that left every margin correct and every tap label
//! mirrored; ampere thresholds converted at the network's base rather than the
//! voltage the CRAC names; an LP optimizing a different limit from the one being
//! measured; a `for CORE CC` step this harness was dropping, which rewrites
//! every nominal voltage and so moves every susceptance by 11%; a tap-rounding
//! rule that settled on the wrong side of the optimum and stayed there,
//! convergent and wrong.
//!
//! Implementing automaton simulation then found four more, three of them
//! structural: a network file's out-of-service circuits were dropped at import,
//! so an automaton that *closes* a standby circuit — one of the commonest there
//! is — could not be applied at all; `initially_open` never reached the
//! evaluation, so those circuits were silently in service; a PST range action
//! whose CRAC omits its tap table was skipped, when the table is a property of
//! the transformer and the network already describes it; and each automaton was
//! sized against the perimeter's worst CNEC rather than the one its own rule
//! names, which asks a scheme to relieve a flow it has no influence over.
//!
//! And on broadening the corpus from 8 scenarios to 22, three more in the
//! redispatch path — which had never been exercised at all. Injection elements
//! were resolved as *branches* when a redispatch names generators and loads,
//! which are buses, so every injection range action was silently dropped. Once
//! they resolved, the chosen set-point was never written to the network, so the
//! measurement saw no change and the action was rejected. And once it was
//! applied, nothing enforced that a redispatch must balance — so the optimizer
//! invented generation and reported margins no network could achieve.
//!
//! Then two more while building the automaton simulator: out-of-service
//! branches were dropped at import, which makes "close this standby circuit" —
//! among the commonest automatons there is — inexpressible; and a PST range
//! action whose CRAC omits its tap table (they routinely do, since the table
//! belongs to the transformer) had no positions to choose between and was
//! skipped in silence, in the automaton *and* in the linear optimizer.
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
use gridoxide::rao::evaluate::{evaluate_model, AcOptions, FlowModel};
use gridoxide::rao::linear::ObjectiveUnit;
use gridoxide::rao::{crac_json, run, Network, Resolution, SearchOptions};
use gridoxide::ucte;
use serde_json::Value;

fn features_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/rao/features")
}

/// The reference's own tolerance, from `RaoSteps.flowMegawattTolerance`.
fn flow_tolerance(expected: f64) -> f64 {
    f64::max(5.0, 0.015 * expected.abs())
}

// ---------------------------------------------------------------------------
// Parsing the feature file
// ---------------------------------------------------------------------------

/// The unit a margin expectation is written in.
///
/// The reference writes most of its expectations in amperes — 194 of the 202
/// worst-margin steps across the `@ac` corpus — and the two are not
/// interchangeable: converting one to the other needs the voltage the *binding*
/// threshold names, which is a property of the CNEC rather than of the step.
/// So the unit is carried through to the comparison and the evaluator reports
/// both.
#[derive(Debug, Clone, Copy, PartialEq)]
enum MarginUnit {
    Megawatt,
    Ampere,
}

/// A margin in both units, as the evaluator reported it.
#[derive(Debug, Clone, Copy)]
struct Margin {
    mw: f64,
    a: f64,
}

impl Margin {
    fn in_unit(&self, unit: MarginUnit) -> f64 {
        match unit {
            MarginUnit::Megawatt => self.mw,
            MarginUnit::Ampere => self.a,
        }
    }
}

#[derive(Debug, Clone)]
enum Expect {
    Secured(bool),
    WorstMargin { value: f64, cnec: Option<String>, stage: Stage, unit: MarginUnit },
    CnecMargin { cnec: String, value: f64, stage: Stage, unit: MarginUnit },
    /// `the initial margin on cnec "X" should be N MW` — the *untouched*
    /// network, before any remedial action. Distinct from the after-PRA one,
    /// and a scenario routinely asserts both for the same CNEC: reading them
    /// as the same step compares the optimized answer against the starting
    /// point and fails on every scenario that improves anything.
    InitialCnecMargin { cnec: String, value: f64, unit: MarginUnit },
    PstTap { action: String, tap: i32, at: Where },
    ActionUsed { action: String, at: Where },
    ActionCount { count: usize, at: Where },
    /// A step this harness does not implement, kept so it is counted rather
    /// than quietly dropped.
    Unsupported(String),
}

/// How far through the plan a margin assertion is measured.
///
/// `after PRA` is the network with only the preventive decisions in force;
/// `after ARA` adds that contingency's automatons; `after CRA` adds its
/// curative decisions too. They are three different networks, and measuring an
/// `auto` CNEC against the preventive one reports the overload the automatons
/// exist to remove.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Stage {
    Pra,
    Ara,
    Cra,
}

fn stage_of(line: &str) -> Stage {
    if line.contains("after ARA") {
        Stage::Ara
    } else if line.contains("after CRA") {
        Stage::Cra
    } else {
        Stage::Pra
    }
}

/// Which perimeter an assertion is about.
///
/// A scenario says either "in preventive" or `after "<contingency>" at
/// "<instant>"`. Reading the second as the first is not a small error: it asks
/// the preventive perimeter about a decision only an automaton or a curative
/// perimeter could have made, and the answer is always "nothing happened".
#[derive(Debug, Clone, PartialEq)]
enum Where {
    Preventive,
    After { contingency: String, instant: String },
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
    let stripped = unquoted(line);
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

/// `after "<contingency>" at "<instant>"`, if the step says so.
fn where_of(line: &str) -> Where {
    let Some(rest) = line.split_once(" after ").map(|(_, r)| r) else {
        return Where::Preventive;
    };
    let quotes: Vec<&str> = rest.split('"').skip(1).step_by(2).collect();
    match (quotes.first(), quotes.get(1)) {
        (Some(contingency), Some(instant)) => Where::After {
            contingency: contingency.trim().to_string(),
            instant: instant.trim().to_string(),
        },
        _ => Where::Preventive,
    }
}

/// The unit a margin step is written in, or `None` if it names neither.
///
/// Matched as a standalone token **outside quotes**, which is the only rule
/// that works for both shapes the reference uses. The unit is not reliably the
/// last token — `the worst margin is -773.0 MW on cnec "…- curative"` ends with
/// the id — and it cannot be found with `contains`, because a quoted CNEC id
/// routinely contains a bare `A` between spaces. Stripping the quoted spans
/// first removes every id from consideration, and what remains is prose the
/// reference wrote.
fn margin_unit(line: &str) -> Option<MarginUnit> {
    let mut unit = None;
    for token in unquoted(line).split_whitespace() {
        match token {
            "MW" => unit = Some(MarginUnit::Megawatt),
            "A" => unit = Some(MarginUnit::Ampere),
            _ => {}
        }
    }
    unit
}

/// `line` with every double-quoted span removed, so ids cannot be mistaken for
/// prose.
fn unquoted(line: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for c in line.chars() {
        if c == '"' {
            inside = !inside;
            // A space in place of the id keeps neighbouring words apart, so
            // `is "X"MW` cannot fuse into one token.
            out.push(' ');
            continue;
        }
        if !inside {
            out.push(c);
        }
    }
    out
}

fn expectation(line: &str) -> Expect {
    if line.contains("security status should be") {
        return match quoted(line).as_deref() {
            Some("SECURED") => Expect::Secured(true),
            Some("UNSECURED") => Expect::Secured(false),
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("the worst margin is") {
        let Some(unit) = margin_unit(line) else { return Expect::Unsupported(line.to_string()) };
        let cnec = line.contains("on cnec").then(|| quoted(line)).flatten();
        return match number_after_quotes(line) {
            Some(value) => Expect::WorstMargin { value, cnec, stage: stage_of(line), unit },
            None => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("margin on cnec") {
        let Some(unit) = margin_unit(line) else { return Expect::Unsupported(line.to_string()) };
        let initial = line.contains("initial margin on cnec");
        return match (quoted(line), number_after_quotes(line)) {
            (Some(cnec), Some(value)) if initial => {
                Expect::InitialCnecMargin { cnec, value, unit }
            }
            (Some(cnec), Some(value)) => {
                Expect::CnecMargin { cnec, value, stage: stage_of(line), unit }
            }
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("the tap of PstRangeAction") {
        return match (quoted(line), number_after_quotes(line)) {
            (Some(action), Some(tap)) => {
                Expect::PstTap { action, tap: tap as i32, at: where_of(line) }
            }
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("remedial action") && line.contains(" is used") && !line.contains("not used") {
        return match quoted(line) {
            Some(action) => Expect::ActionUsed { action, at: where_of(line) },
            None => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("remedial actions are used") {
        return match number_after_quotes(line) {
            Some(n) => Expect::ActionCount { count: n as usize, at: where_of(line) },
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
/// Which flow model the scenario's configuration asks for.
///
/// The reference states it outright — `load-flow-parameters.dc` — so this reads
/// the file rather than the `@ac`/`@dc` tag. The tag is a label on the scenario;
/// the field is the setting the run actually used, and when a scenario is
/// retagged the field is the one that stays true.
fn flow_model_from(config: &Path) -> FlowModel {
    let Ok(text) = std::fs::read_to_string(config) else { return FlowModel::Dc };
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(&text) else { return FlowModel::Dc };
    let dc = doc
        .pointer(
            "/extensions/open-rao-search-tree-parameters/load-flow-and-sensitivity-computation\
             /sensitivity-parameters/load-flow-parameters/dc",
        )
        .and_then(|v| v.as_bool());
    match dc {
        Some(false) => FlowModel::Ac,
        // Absent means the reference's own default, which is DC for these
        // files — and a config this cannot read is treated as DC rather than
        // silently upgraded to a slower model that changes every number.
        _ => FlowModel::Dc,
    }
}

fn options_from(config: &Path) -> SearchOptions {
    let mut options = SearchOptions::default();
    // `RaoUtil.getFlowUnit`: megawatts for a DC load flow, amperes for an AC
    // one. Not a setting of its own — the objective follows the flow model, and
    // the minimum-impact thresholds below are stated in whichever unit results.
    options.linear.flow_model = flow_model_from(config);
    options.linear.objective_unit = match options.linear.flow_model {
        FlowModel::Ac => ObjectiveUnit::Ampere,
        FlowModel::Dc => ObjectiveUnit::Megawatt,
    };
    let Ok(text) = std::fs::read_to_string(config) else { return options };
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(&text) else { return options };

    // `SECURE_FLOW` means "stop once secure" rather than "maximize" —
    // `TreeParameters.buildForPreventivePerimeter` turns it into
    // `AT_TARGET_OBJECTIVE_VALUE` with a target of zero.
    if doc.pointer("/objective-function/type").and_then(|v| v.as_str()) == Some("SECURE_FLOW") {
        options.stop_at_target = Some(0.0);
    }

    // The two knobs live under the search-tree extension, not beside the
    // thresholds above.
    if let Some(topology) = doc.pointer(
        "/extensions/open-rao-search-tree-parameters/topological-actions-optimization",
    ) {
        let skip = topology
            .get("skip-actions-far-from-most-limiting-element")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if skip {
            options.skip_far_actions = Some(
                topology
                    .get("max-number-of-boundaries-for-skipping-actions")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
            );
        }
    }
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
    // MNEC handling is gated on the configuration mentioning it at all, exactly
    // as `ObjectiveFunctionCreator` gates its virtual cost evaluator on both
    // `mnec-parameters` blocks being present. A file that never mentions MNECs
    // leaves them unconstrained however the CRAC labels them.
    let base_mnec = doc.get("mnec-parameters");
    let extension_mnec = extension.and_then(|e| e.get("mnec-parameters"));
    options.linear.mnec.options.enabled = base_mnec.is_some() && extension_mnec.is_some();
    if let Some(v) = base_mnec.and_then(|m| m.get("acceptable-margin-decrease")).and_then(Value::as_f64)
    {
        options.linear.mnec.options.acceptable_margin_decrease = v;
    }
    if let Some(m) = extension_mnec {
        if let Some(v) = m.get("violation-cost").and_then(Value::as_f64) {
            options.linear.mnec.options.violation_cost = v;
        }
        if let Some(v) = m.get("constraint-adjustment-coefficient").and_then(Value::as_f64) {
            options.linear.mnec.options.constraint_adjustment_coefficient = v;
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

/// The margin of `cnec` at `stage`, falling back to the after-PRA map when the
/// stage produced nothing for it — a CNEC in a contingency with no automaton or
/// no curative perimeter is unchanged from PRA, which is the honest answer
/// rather than a missing one.
fn pick(
    stage: Stage,
    cnec: &str,
    pra: &HashMap<&str, Margin>,
    ara: &HashMap<String, Margin>,
    cra: &HashMap<String, Margin>,
) -> Option<Margin> {
    let staged = match stage {
        Stage::Pra => None,
        Stage::Ara => ara.get(cnec).copied(),
        Stage::Cra => cra.get(cnec).copied(),
    };
    staged.or_else(|| pra.get(cnec).copied())
}

/// The tap a phase shifter sits on when nothing has moved it — its position in
/// the network as imported.
///
/// Falls back to the CRAC's declared `initialTap` when the network has no tap
/// changer for the element, which is the same precedence `linear::tap_table`
/// uses and for the same reason: the equipment is the authority, the CRAC is a
/// description of it.
fn resting_tap(
    crac: &Crac,
    resolution: &Resolution,
    net: &ucte::UcteImport,
    action: &str,
) -> Option<i32> {
    let range = crac.range_actions.iter().find(|r| r.id == action)?;
    let RangeActionKind::Pst { element, initial_tap, .. } = &range.kind else { return None };
    let branch = resolution.branch(element)?;
    let changer = branch
        .checked_sub(net.lines.len())
        .and_then(|i| net.tap_changers.get(i))
        .and_then(|c| c.as_ref());
    Some(changer.map_or(*initial_tap, |c| c.position))
}

/// The worst margin across every **optimized** CNEC, each measured at its own
/// stage.
///
/// An `auto` CNEC is read after the automatons and a `curative` one after the
/// curative decisions, because those are the networks those CNECs exist in.
///
/// Monitored CNECs are left out because the reference leaves them out: its
/// functional cost filters on `isOptimized` before taking the minimum, and an
/// MNEC that starts overloaded would otherwise be reported as the worst margin
/// of a run that was never asked to repair it.
fn worst_at_own_stage(
    crac: &Crac,
    unit: MarginUnit,
    pra: &HashMap<&str, Margin>,
    ara: &HashMap<String, Margin>,
    cra: &HashMap<String, Margin>,
) -> Option<f64> {
    crac.flow_cnecs
        .iter()
        .filter(|c| c.optimized)
        .filter_map(|c| {
            let stage = match crac.instants[c.state.instant].kind {
                InstantKind::Auto => Stage::Ara,
                InstantKind::Curative => Stage::Cra,
                _ => Stage::Pra,
            };
            pick(stage, &c.id, pra, ara, cra).map(|m| m.in_unit(unit))
        })
        .fold(None, |acc: Option<f64>, m| Some(acc.map_or(m, |a| a.min(m))))
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
    // The search stays on DC whatever the model: it is what makes the tree
    // finish, and phase 11's whole argument is that AC is where the answer gets
    // *checked*. What the model changes here is every margin the scenario
    // asserts on.
    let model = flow_model_from(&resolve(&scenario.config));
    let ac_options = AcOptions { shunts: &net.shunts, ..Default::default() };
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let view = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &net.shunts,
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &view, &resolution, &mut solver, &search_options);

    // Per-CNEC margins with the preventive decisions in force — what the
    // reference calls "after PRA".
    // One margin map per stage. A CNEC is looked up in the map its step names,
    // because the three describe genuinely different networks.
    let margins_at = |stage: Stage| -> HashMap<String, Margin> {
        let mut out = HashMap::new();
        for scenario in &plan.scenarios {
            let contingency = Some(scenario.contingency);
            let (open, transformers, buses) = match stage {
                Stage::Cra => scenario
                    .perimeters
                    .last()
                    .map(|p| (&p.open_branches, &p.transformers, &p.buses)),
                Stage::Ara => scenario
                    .automatons
                    .as_ref()
                    .map(|a| (&a.open_branches, &a.transformers, &plan.preventive.buses)),
                Stage::Pra => None,
            }
            .unwrap_or((
                &plan.preventive.open_branches,
                &plan.preventive.transformers,
                &plan.preventive.buses,
            ));
            let view = Network {
                buses,
                lines: &net.lines,
                transformers,
                branch_ids: &net.branch_ids,
                bus_ids: &net.node_codes,
                initially_open: &net.initially_open,
                bus_countries: &net.bus_countries,
                shunts: &net.shunts,
                tap_changers: &net.tap_changers,
                base_mva: net.base_mva,
            };
            for perimeter in
                evaluate_model(&crac, &view, &resolution, open, model, &ac_options).perimeters
            {
                if perimeter.state.contingency != contingency {
                    continue;
                }
                for c in &perimeter.cnecs {
                    out.insert(
                        crac.flow_cnecs[c.cnec].id.clone(),
                        Margin { mw: c.margin_mw, a: c.margin_a },
                    );
                }
            }
        }
        out
    };
    let ara_margins = margins_at(Stage::Ara);
    let cra_margins = margins_at(Stage::Cra);

    let after_pra = Network {
        // The plan's own buses, not the file's: a redispatch lives only there,
        // and re-evaluating against the original buses reports a network in
        // which no injection ever moved.
        buses: &plan.preventive.buses,
        lines: &net.lines,
        transformers: &plan.preventive.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &net.shunts,
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let evaluated =
        evaluate_model(
            &crac,
            &after_pra,
            &resolution,
            &plan.preventive.open_branches,
            model,
            &ac_options,
        );
    let margins: HashMap<&str, Margin> = evaluated
        .perimeters
        .iter()
        .flat_map(|p| p.cnecs.iter())
        .map(|c| {
            (crac.flow_cnecs[c.cnec].id.as_str(), Margin { mw: c.margin_mw, a: c.margin_a })
        })
        .collect();

    // The network's *own* open set, not an empty one. "Untouched" means before
    // any remedial action, not with every branch closed — and a UCTE file
    // routinely ships couplers out of service (TestCase16Nodes has two). Passing
    // `&[]` here measures a network that does not exist, and does it silently:
    // every margin stays self-consistent and the whole initial situation is
    // simply a different one.
    let untouched =
        evaluate_model(&crac, &view, &resolution, view.initially_open, model, &ac_options);
    let initial_margins: HashMap<&str, Margin> = untouched
        .perimeters
        .iter()
        .flat_map(|p| p.cnecs.iter())
        .map(|c| {
            (crac.flow_cnecs[c.cnec].id.as_str(), Margin { mw: c.margin_mw, a: c.margin_a })
        })
        .collect();

    // What each perimeter decided, keyed the way the steps address it.
    let decisions = |at: &Where| -> (Vec<String>, HashMap<String, i32>) {
        let mut used = Vec::new();
        let mut taps = HashMap::new();
        fn take_setpoints(
            crac: &Crac,
            setpoints: &[gridoxide::rao::Setpoint],
            used: &mut Vec<String>,
            taps: &mut HashMap<String, i32>,
        ) {
            for s in setpoints {
                // "Used" means *moved*; a tap assertion asks where the shifter
                // ended up, which is a question with an answer even when it
                // stayed put.
                if s.moved() {
                    used.push(crac.range_actions[s.action].id.clone());
                }
                if let Some(tap) = s.tap {
                    taps.insert(crac.range_actions[s.action].id.clone(), tap);
                }
            }
        }
        match at {
            Where::Preventive => {
                used.extend(
                    plan.preventive
                        .network_actions
                        .iter()
                        .map(|&a| crac.network_actions[a].id.clone()),
                );
                take_setpoints(&crac, &plan.preventive.setpoints, &mut used, &mut taps);
            }
            Where::After { contingency, instant } => {
                for scenario in &plan.scenarios {
                    if crac.contingencies[scenario.contingency].id.trim() != contingency.trim() {
                        continue;
                    }
                    if crac.instants.iter().any(|i| {
                        i.id == *instant && i.kind == InstantKind::Auto
                    }) {
                        if let Some(a) = &scenario.automatons {
                            used.extend(
                                a.network_actions
                                    .iter()
                                    .map(|&i| crac.network_actions[i].id.clone()),
                            );
                            for (i, _, tap) in &a.range_actions {
                                used.push(crac.range_actions[*i].id.clone());
                                if let Some(tap) = tap {
                                    taps.insert(crac.range_actions[*i].id.clone(), *tap);
                                }
                            }
                        }
                    }
                    for perimeter in &scenario.perimeters {
                        let matches = perimeter
                            .states
                            .iter()
                            .any(|s| crac.instants[s.instant].id == *instant);
                        if !matches {
                            continue;
                        }
                        used.extend(
                            perimeter
                                .network_actions
                                .iter()
                                .map(|&a| crac.network_actions[a].id.clone()),
                        );
                        take_setpoints(&crac, &perimeter.setpoints, &mut used, &mut taps);
                    }
                }
            }
        }
        (used, taps)
    };

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
            Expect::WorstMargin { value, cnec: None, unit, .. } => {
                // In MW the plan's own objective is the answer. In amperes it
                // is not the same quantity: the worst margin in amperes can
                // fall on a different CNEC, because the conversion voltage
                // differs per CNEC. So it is re-derived from the per-CNEC
                // margins, each measured at its own stage — the same rule the
                // named form already uses.
                let got = match unit {
                    MarginUnit::Megawatt => Some(plan.final_margin_mw),
                    MarginUnit::Ampere => worst_at_own_stage(
                        &crac,
                        *unit,
                        &margins,
                        &ara_margins,
                        &cra_margins,
                    ),
                };
                record(
                    got.is_some_and(|g| (g - value).abs() <= flow_tolerance(*value)),
                    format!("worst margin {got:?} {unit:?} (expected {value})"),
                );
            }
            Expect::WorstMargin { value, cnec: Some(id), unit, .. } => {
                // A worst-margin step names the CNEC that *ends up* carrying the
                // worst margin, so it is measured at that CNEC's own stage —
                // an `auto` CNEC after the automatons, a `curative` one after
                // the curative decisions. Reading it after PRA reports the
                // overload those actions exist to remove, which is the value
                // before anything happened rather than the answer.
                let stage = crac
                    .flow_cnecs
                    .iter()
                    .find(|c| c.id == *id)
                    .map(|c| match crac.instants[c.state.instant].kind {
                        InstantKind::Auto => Stage::Ara,
                        InstantKind::Curative => Stage::Cra,
                        _ => Stage::Pra,
                    })
                    .unwrap_or(Stage::Pra);
                let got = pick(stage, id, &margins, &ara_margins, &cra_margins)
                    .map(|m| m.in_unit(*unit));
                record(
                    got.is_some_and(|g| (g - value).abs() <= flow_tolerance(*value)),
                    format!("worst margin on `{id}` {got:?} (expected {value})"),
                );
            }
            Expect::InitialCnecMargin { cnec, value, unit } => {
                let got = initial_margins.get(cnec.as_str()).map(|m| m.in_unit(*unit));
                record(
                    got.is_some_and(|g| (g - value).abs() <= flow_tolerance(*value)),
                    format!("initial margin on `{cnec}` {got:?} (expected {value})"),
                );
            }
            Expect::CnecMargin { cnec, value, stage, unit } => {
                let got = pick(*stage, cnec, &margins, &ara_margins, &cra_margins)
                    .map(|m| m.in_unit(*unit));
                record(
                    got.is_some_and(|g| (g - value).abs() <= flow_tolerance(*value)),
                    format!("margin on `{cnec}` {got:?} (expected {value})"),
                );
            }
            Expect::PstTap { action, tap, at } => {
                // A shifter with no set-point in this perimeter still has a
                // tap: the one it is sitting on. The reference's
                // `getOptimizedTapOnState` answers for every state, activated
                // or not, and a CRAC routinely makes a PST available only at
                // `auto` while a scenario asks where it stood in preventive.
                // Reporting `None` there fails an assertion that is simply
                // asking "unchanged?".
                let got = decisions(at)
                    .1
                    .get(action)
                    .copied()
                    .or_else(|| resting_tap(&crac, &resolution, &net, action));
                record(
                    got == Some(*tap),
                    format!("tap of `{action}` {got:?} (expected {tap}) {at:?}"),
                );
            }
            Expect::ActionUsed { action, at } => {
                let (used, _) = decisions(at);
                let got = used.iter().any(|u| u == action);
                record(got, format!("`{action}` used: {got} {at:?}"));
            }
            Expect::ActionCount { count, at } => {
                let got = decisions(at).0.len();
                record(got == *count, format!("{got} action(s) used (expected {count}) {at:?}"));
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
    for (file, expected, baseline) in [
        ("dc_scenarios.feature", 25, BASELINE_MATCHED_DC),
        ("ac_scenarios.feature", 38, BASELINE_MATCHED_AC),
        ("ac_scenarios_16nodes.feature", 93, BASELINE_MATCHED_AC16),
    ] {
        run_gate(file, expected, baseline);
    }
}

/// Run one vendored feature file and assert on its aggregate.
///
/// The two files are scored separately on purpose. They exercise different flow
/// models, and a single total would let a gain in one hide a regression in the
/// other.
fn run_gate(file: &str, expected_scenarios: usize, baseline: usize) {
    let text = std::fs::read_to_string(features_dir().join(file)).expect("feature file");
    let scenarios = parse(&text);
    assert_eq!(scenarios.len(), expected_scenarios, "in {file}");

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
        "{report}\n{file}: {matched}/{total} checkable assertions match the reference \
         ({unsupported} steps unsupported)"
    );

    assert!(
        matched >= baseline,
        "{file}: {matched}/{total} matched, baseline is {baseline} — a drop is a regression:\n{report}"
    );
}

/// How many of the reference's assertions currently hold: **132 of 136**,
/// across all 25 scenarios.
///
/// It was 124 of 124 before the three MNEC scenarios (5.2.1.2 to 5.2.1.4)
/// joined it. Eight of their twelve assertions hold; the four that do not are
/// two taps and the two margins that follow from them, and they are the
/// `BestTapFinder` divergence described on [`BASELINE_MATCHED_AC`].
///
/// It is a recorded number rather than an assertion of perfection. These are
/// two heuristic search trees and §8.3 says up front that a different set of
/// actions reaching the same margin is not a defect; a scenario added later may
/// legitimately disagree. Raising this is progress, a drop is a regression, and
/// the printed report says which assertion moved.
const BASELINE_MATCHED_DC: usize = 132;

/// The same, for the 38 AC scenarios in `ac_scenarios.feature`: **190 of 201**.
///
/// Lower than the DC file's score, and expected to be. These scenarios are
/// judged on margins the reference measured with an AC load flow that also
/// distributes slack and enforces reactive limits, while gridoxide's search
/// still chooses its actions on DC sensitivities. Where the two models rank two
/// candidates differently, the search takes the other one and every assertion
/// downstream of that choice moves together.
///
/// # The one divergence that is not a coin toss
///
/// Three of the MNEC scenarios — 5.2.1.3, 5.2.1.4 here and 5.2.3.3 above —
/// disagree by exactly **one tap**, always in the direction of the reference
/// paying an MNEC violation gridoxide declines to pay. On 5.2.1.3 the
/// reference's tap −7 scores 192.05 MW of margin against a 7.67 MW violation
/// penalty — 184.38 — while gridoxide's tap −6 scores 188.42 with no violation
/// at all. On 5.2.3.3 it is −198.52 A against −183.10 A. gridoxide wins both on
/// the reference's own objective.
///
/// That is not luck. The reference rounds a continuous set-point with
/// `BestTapFinder`, which reconsiders the second-nearest tap only when the
/// angle lands within 15% of the midpoint between them, and which compares the
/// two on **minimum margin alone** — its javadoc says so, and warns that
/// "if virtual costs are an important part of the optimization, it is highly
/// recommended to use APPROXIMATED_INTEGERS taps … rather than relying on the
/// best tap finder to round the taps". These CRACs put the LP's optimum
/// *exactly on the MNEC bound*, which is 89% of the way to the next tap: outside
/// the band, so the reference never looks, and blind to the penalty if it did.
/// `PstControl::bracketing_taps` always looks, and scores with the penalty
/// included.
///
/// Matching these four assertions would mean reproducing that rounding, at the
/// cost of a worse answer. They are left as recorded disagreements.
const BASELINE_MATCHED_AC: usize = 190;

/// The same, for the 93 AC scenarios on `TestCase16Nodes`: **664 of 844**.
///
/// The largest of the three files and the newest, so the furthest from
/// settled. It is here to find defects, and it does.
///
/// Its four MNEC scenarios (1.3.6.1, 1.3.6.5 to 1.3.6.7) contribute 39 of 42:
/// three of them match outright, and the three misses are all in 1.3.6.6's
/// curative perimeter on `co1_fr2_fr3_1` — which is where a good part of this
/// file's remaining disagreements sit, MNECs or no MNECs.
///
/// The **2.4 usage-rule family matches in full** — 147 of 147, up from 100 —
/// since conditional usage rules are answered against the perimeter's flows
/// rather than assumed true. Enforcing the CRAC's **usage limits** then took
/// 2.6 from 73 of 134 to 103, and 2.2 from 39 of 63 to 54.
///
/// What is still open, by size: 1.3 curative (85 of 420 wrong), 1.2 automatons
/// (50 of 119), 2.6 (31 of 134). The 2.6 remainder has changed character
/// completely — it was "gridoxide spends actions the CRAC forbids" and is now
/// "gridoxide stops before the reference does", the same greedy-chain limit
/// that shows up wherever three actions are needed and each is worth little on
/// its own.
const BASELINE_MATCHED_AC16: usize = 664;

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
