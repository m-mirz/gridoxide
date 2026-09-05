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
use gridoxide::rao::evaluate::{evaluate_model, FlowModel};
use gridoxide::rao::linear::ObjectiveUnit;
use gridoxide::rao::{
    crac_json, run, Network, Resolution, SearchOptions, SecondPreventiveCondition,
};
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

/// One CNEC's reading in one network: its margin in both units, and the flow
/// that produced it at both terminals.
///
/// The flows travel with the margin rather than in a map of their own because
/// they are read at the same three stages and out of the same
/// [`CnecResult`](gridoxide::rao::CnecResult) — a second set of maps would be a
/// second chance to look a stage up in the wrong one.
#[derive(Debug, Clone, Copy)]
struct Margin {
    mw: f64,
    a: f64,
    /// `(MW, A)` at side one and side two. Under AC these differ by the
    /// branch's losses and by the two buses' voltages; under DC side two is the
    /// negation of side one.
    flow: [(f64, f64); 2],
}

impl Margin {
    fn of(c: &gridoxide::rao::CnecResult) -> Self {
        Self {
            mw: c.margin_mw,
            a: c.margin_a,
            flow: [(c.flow_mw, c.current_a), c.side_two],
        }
    }

    fn in_unit(&self, unit: MarginUnit) -> f64 {
        match unit {
            MarginUnit::Megawatt => self.mw,
            MarginUnit::Ampere => self.a,
        }
    }

    /// The flow at one side, in the unit the step is written in. `side` is the
    /// reference's own 1-based numbering.
    ///
    /// The ampere figure is **signed by the active power** at that terminal.
    /// `CnecResult::current_a` is a magnitude, correctly — a current is one, and
    /// it is what a threshold is compared against. But the reference's
    /// `getFlow(cnec, side, AMPERE)` divides the *signed* megawatts by
    /// `√3·U`, so a step reading `-1444.0 A on side 2` is naming a direction
    /// and not just a size. Comparing a magnitude against it fails on every
    /// reverse flow while agreeing to five significant figures, which is a
    /// confusing way to be right.
    fn flow_in_unit(&self, unit: MarginUnit, side: usize) -> Option<f64> {
        let (mw, a) = *self.flow.get(side.checked_sub(1)?)?;
        Some(match unit {
            MarginUnit::Megawatt => mw,
            MarginUnit::Ampere => a.copysign(mw),
        })
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
    /// `the value of the objective function after CRA should be N` — the
    /// reference's **cost**, which is the negated worst margin plus whatever
    /// the monitored CNECs are violating. `None` means "initially", before any
    /// remedial action.
    ObjectiveValue { value: f64, stage: Option<Stage> },
    PstTap { action: String, tap: i32, at: Where },
    ActionUsed { action: String, at: Where },
    /// `the remedial action "X" is not used ...` — the negative, and worth
    /// having for exactly the reason it is easy to skip: nothing else in this
    /// harness penalizes taking an action the reference declines, which is the
    /// shape two recorded defects already had.
    ActionNotUsed { action: String, at: Where },
    ActionCount { count: usize, at: Where },
    /// `the flow on cnec "X" after PRA should be N A on side N`, and the
    /// `initial flow` form with `stage: None`. Sides are the reference's own
    /// 1-based numbering.
    CnecFlow { cnec: String, value: f64, unit: MarginUnit, side: usize, stage: Option<Stage> },
    /// `the "upper"/"lower" threshold on cnec "X" should be N A` — the bound
    /// itself, not the distance to it.
    CnecThreshold { cnec: String, upper: bool, value: f64, unit: MarginUnit },
    /// `PST "X" in network file with PRA is on tap N` — where the *network*
    /// left the shifter, addressed by network element rather than by range
    /// action. Distinct from [`PstTap`](Self::PstTap): a scenario asserts this
    /// for a shifter no range action names.
    NetworkTap { element: String, tap: i32 },
    /// `line "X" in network file with PRA has connection status to "X"`.
    Connected { element: String, connected: bool },
    /// `the execution details should be "X"` — which optimization steps ran and
    /// how each turned out. The most-asserted step in the whole suite, and the
    /// one thing no margin can tell you: whether the answer in front of you is
    /// the plan the optimizer wanted, the plan it fell back to, or the network
    /// untouched.
    Steps(String),
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
    /// `I launch rao with a time limit of N seconds`, where the reference has
    /// one. A **negative** limit is the suite's way of saying the run had no
    /// time for a second preventive pass — 1.4.4.5 and 1.4.4.6 are the same
    /// scenario at −1 and 600 seconds, and they assert different answers.
    /// gridoxide has no wall clock in the optimizer, so the limit is honoured
    /// where it is a statement about *what runs* rather than about how long.
    time_limit: Option<f64>,
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

/// The `n`th double-quoted span, zero-based.
///
/// Needed because the reference does not put the subject first in every step:
/// `the "upper" threshold on cnec "X"` names the bound before the CNEC.
fn quoted_nth(line: &str, n: usize) -> Option<String> {
    line.split('"').skip(1).step_by(2).nth(n).map(str::to_string)
}

/// The terminal a flow step names, in the reference's 1-based numbering.
fn side_of(line: &str) -> Option<usize> {
    line.split_once(" on side ")?.1.split_whitespace().next()?.parse().ok()
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
        } else if line.contains("launch rao with a time limit of") {
            scenario.time_limit = number_after_quotes(line);
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
    if line.contains("the value of the objective function") {
        // "before optimisation" is the reference's own synonym for
        // "initially" — both call `getCost(null)`.
        let stage = if line.contains("initially") || line.contains("before optimisation") {
            None
        } else {
            Some(stage_of(line))
        };
        return match number_after_quotes(line) {
            Some(value) => Expect::ObjectiveValue { value, stage },
            None => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("flow on cnec") {
        let Some(unit) = margin_unit(line) else { return Expect::Unsupported(line.to_string()) };
        let Some(side) = side_of(line) else { return Expect::Unsupported(line.to_string()) };
        // `initial` is the untouched network; every other form names a stage.
        let stage = (!line.contains("initial flow")).then(|| stage_of(line));
        return match (quoted(line), number_after_quotes(line)) {
            (Some(cnec), Some(value)) => Expect::CnecFlow { cnec, value, unit, side, stage },
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("threshold on cnec") {
        let Some(unit) = margin_unit(line) else { return Expect::Unsupported(line.to_string()) };
        let upper = match quoted(line).as_deref() {
            Some("upper") => true,
            Some("lower") => false,
            _ => return Expect::Unsupported(line.to_string()),
        };
        return match (quoted_nth(line, 1), number_after_quotes(line)) {
            (Some(cnec), Some(value)) => Expect::CnecThreshold { cnec, upper, value, unit },
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("the setpoint of RangeAction") {
        // Parsed and deliberately **not** compared: the two sides name
        // different quantities. gridoxide's `Setpoint::value` for an injection
        // range action is the *shift* it applies, starting from zero; the
        // reference's `getOptimizedSetPointOnState` is the generator's
        // **absolute** target, which it recovers as `targetP / key`. On 2.3.1.4
        // both actions are used, both are named correctly and the margin is
        // exact to 500.0 — only the number's origin differs, and gridoxide
        // cannot produce the reference's without per-generator injections the
        // bus-aggregating UCTE importer does not keep (`Bus::p_spec` is
        // generation minus load). Recording it as a failure would cap the ratio
        // for something that is not wrong, so it is skipped **with its reason
        // attached** rather than silently.
        return Expect::Unsupported(format!(
            "{line}  [not comparable: gridoxide reports the shift, the reference the \
             generator's absolute target]"
        ));
    }
    if line.contains("in network file with PRA is on tap") {
        return match (quoted(line), number_after_quotes(line)) {
            (Some(element), Some(tap)) => Expect::NetworkTap { element, tap: tap as i32 },
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("in network file with PRA has connection status to") {
        return match (quoted(line), quoted_nth(line, 1).as_deref()) {
            (Some(element), Some("true")) => Expect::Connected { element, connected: true },
            (Some(element), Some("false")) => Expect::Connected { element, connected: false },
            _ => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("the execution details should be") {
        return match quoted(line) {
            Some(text) => Expect::Steps(text),
            None => Expect::Unsupported(line.to_string()),
        };
    }
    if line.contains("remedial action") && line.contains(" is not used") {
        return match quoted(line) {
            Some(action) => Expect::ActionNotUsed { action, at: where_of(line) },
            None => Expect::Unsupported(line.to_string()),
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

    // A curative perimeter always stops at a target, and the target is stated
    // relative to the preventive perimeter's own objective:
    // `TreeParameters.buildForCurativePerimeter`. Both halves of that live in
    // different places in the file — the improvement under the search-tree
    // extension, the security flag beside the objective's type.
    if let Some(v) = doc
        .pointer("/extensions/open-rao-search-tree-parameters/objective-function\
                  /curative-min-obj-improvement")
        .and_then(Value::as_f64)
    {
        options.curative_min_obj_improvement = v;
    }
    if let Some(v) =
        doc.pointer("/objective-function/enforce-curative-security").and_then(Value::as_bool)
    {
        options.enforce_curative_security = v;
    }

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
    // `second-preventive-rao`. Off unless the configuration says otherwise,
    // which is the reference's own default.
    if let Some(second) = extension.and_then(|e| e.get("second-preventive-rao")) {
        options.second_preventive.condition =
            match second.get("execution-condition").and_then(|v| v.as_str()) {
                Some("POSSIBLE_CURATIVE_IMPROVEMENT") => {
                    SecondPreventiveCondition::PossibleCurativeImprovement
                }
                Some("COST_INCREASE") => SecondPreventiveCondition::CostIncrease,
                _ => SecondPreventiveCondition::Disabled,
            };
        options.second_preventive.hint = second
            .get("hint-from-first-preventive-rao")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    }
    if let Some(topology) = extension.and_then(|e| e.get("topological-actions-optimization")) {
        // The reference's default is i32::MAX; anything that large is a depth
        // bound in name only, and running it would evaluate every combination
        // of a corpus this harness has no time budget for.
        if let Some(v) = topology.get("max-preventive-search-tree-depth").and_then(|v| v.as_u64()) {
            options.max_depth = (v as usize).min(3);
        }
        // Read separately, because the reference states it separately. Every
        // vendored configuration sets the two the same, so this moves nothing
        // here — and a configuration that did not would otherwise have been
        // scored against the preventive depth without a word.
        if let Some(v) = topology.get("max-curative-search-tree-depth").and_then(|v| v.as_u64()) {
            options.curative_max_depth = Some((v as usize).min(3));
        }
        // The reference's whole mechanism for offering anything but a greedy
        // chain. Every vendored configuration carries `[]`, which is the
        // finding that killed the "gridoxide needs a combinatorial search"
        // diagnosis: on the scenarios that fail, the reference reaches its
        // answer with the same single-action chain.
        if let Some(list) = topology.get("predefined-combinations").and_then(|v| v.as_array()) {
            options.predefined_combinations = list
                .iter()
                .filter_map(|c| c.as_array())
                .map(|c| c.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .collect();
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
    transformers: &[gridoxide::types::Transformer],
    action: &str,
) -> Option<i32> {
    let range = crac.range_actions.iter().find(|r| r.id == action)?;
    let RangeActionKind::Pst { element, initial_tap, .. } = &range.kind else { return None };
    // Read out of the network **this state** is in, not out of the file. A
    // shifter the preventive stage moved to −16 and that no curative action may
    // touch is at −16 in the curative state, and the reference's
    // `getOptimizedTapOnState` says so: it answers for every state, from the
    // set-points in force there. Falling back to the file's position instead
    // reports the plan undoing a decision it never revisited.
    Some(
        tap_in_network(crac, resolution, net, &[], transformers, element)
            .unwrap_or(*initial_tap),
    )
}

/// Where a phase shifter *in the network* ended up, addressed by network
/// element rather than by range action.
///
/// Distinct from [`resting_tap`], which answers for a range action. A scenario
/// asks this of a shifter no range action need name, so the position is
/// recovered from the transformer the plan left behind: the tap changer's own
/// steps are the only authority on which position an angle corresponds to, and
/// the search reports angles.
fn tap_in_network(
    crac: &Crac,
    resolution: &Resolution,
    net: &ucte::UcteImport,
    setpoints: &[gridoxide::rao::Setpoint],
    transformers: &[gridoxide::types::Transformer],
    element: &str,
) -> Option<i32> {
    // A range action on this element already knows the answer, and knows it
    // exactly. Under the continuous tap model the optimizer leaves the
    // transformer on an angle *between* two steps and the set-point carries the
    // position it was rounded to, so recovering the position from the angle
    // afterwards can land on the other neighbour — the same off-by-one
    // `BestTapFinder` exists to settle.
    let by_action = setpoints.iter().find_map(|s| match &crac.range_actions[s.action].kind {
        RangeActionKind::Pst { element: e, .. } if e == element => s.tap,
        _ => None,
    });
    if by_action.is_some() {
        return by_action;
    }
    let branch = resolution.branch(element)?;
    let i = branch.checked_sub(net.lines.len())?;
    let changer = net.tap_changers.get(i)?.as_ref()?;
    let angle = transformers.get(i)?.tap.arg();
    (changer.low..=changer.high())
        .filter(|p| changer.at(*p).is_some())
        .min_by(|a, b| {
            let d = |p: &i32| (changer.at(*p).unwrap().arg() - angle).abs();
            d(a).partial_cmp(&d(b)).unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// A CNEC's declared threshold in one direction, in the unit the step names.
///
/// Read from the **CRAC**, not from the evaluator. The step asserts what the
/// file says — `frm` is zero on every CNEC that carries one of these steps —
/// while the evaluator's bound has already been tightened by the reliability
/// margin and, under AC, charged for reactive flow. Those are the right numbers
/// for a margin and the wrong ones for this question.
fn declared_threshold(crac: &Crac, cnec: &str, upper: bool, unit: MarginUnit) -> Option<f64> {
    let cnec = crac.flow_cnecs.iter().find(|c| c.id == cnec)?;
    let want = match unit {
        MarginUnit::Ampere => Unit::Ampere,
        MarginUnit::Megawatt => Unit::Megawatt,
    };
    cnec.thresholds
        .iter()
        .filter(|t| t.unit == want)
        .filter_map(|t| if upper { t.max } else { t.min })
        .fold(None, |acc: Option<f64>, v| {
            Some(acc.map_or(v, |a| if upper { a.min(v) } else { a.max(v) }))
        })
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
    let mut search_options = options_from(&resolve(&scenario.config));
    // A run given no time does not get a second preventive pass. That is the
    // whole of what the time limit means here: gridoxide's optimizer has no
    // wall clock, and inventing one to reproduce a wall-clock decision would be
    // reproducing the symptom rather than the rule. 1.4.4.5 and 1.4.4.6 are the
    // same scenario at −1 and 600 seconds and assert different answers, which
    // is exactly the distinction being made.
    if scenario.time_limit.is_some_and(|t| t <= 0.0) {
        search_options.second_preventive.condition = SecondPreventiveCondition::Disabled;
    }
    let search_options = search_options;
    // The search stays on DC whatever the model: it is what makes the tree
    // finish, and phase 11's whole argument is that AC is where the answer gets
    // *checked*. What the model changes here is every margin the scenario
    // asserts on.
    let model = flow_model_from(&resolve(&scenario.config));
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let view = Network {
        generation: &net.generation,
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
    // The same settings the optimizer measures with, from the same builder: a
    // harness that scored the answer under a different slack from the one that
    // produced it would be marking its own homework wrong.
    let ac_options = gridoxide::rao::ac_options(&view);
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
                generation: &net.generation,
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
                        Margin::of(c),
                    );
                }
            }
        }
        out
    };
    let ara_margins = margins_at(Stage::Ara);
    let cra_margins = margins_at(Stage::Cra);

    let after_pra = Network {
        generation: &net.generation,
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
            (crac.flow_cnecs[c.cnec].id.as_str(), Margin::of(c))
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
            (crac.flow_cnecs[c.cnec].id.as_str(), Margin::of(c))
        })
        .collect();

    // The reference's "value of the objective function" is its **cost**: the
    // negated worst margin over the optimized CNECs, plus whatever the
    // monitored ones are violating. In the objective's unit, which follows the
    // flow model rather than being stated by the step.
    //
    // Which network each CNEC is read in depends on the instant asked about.
    // "After PRA" reads *every* CNEC, curative ones included, in the post-PRA
    // network — that is `prePerimeterResultForAllFollowingStates`. "After CRA"
    // takes the worst across perimeters, each measured in the network its own
    // decisions produced, which is `Math::max` over the per-state results.
    let objective_unit = match search_options.linear.objective_unit {
        ObjectiveUnit::Ampere => MarginUnit::Ampere,
        ObjectiveUnit::Megawatt => MarginUnit::Megawatt,
    };
    let cost_at = |stage: Option<Stage>| -> Option<f64> {
        let read = |cnec: &FlowCnec| -> Option<f64> {
            let margin = match stage {
                None => initial_margins.get(cnec.id.as_str()).copied(),
                Some(Stage::Cra) => {
                    let own = match crac.instants[cnec.state.instant].kind {
                        InstantKind::Auto => Stage::Ara,
                        InstantKind::Curative => Stage::Cra,
                        _ => Stage::Pra,
                    };
                    pick(own, &cnec.id, &margins, &ara_margins, &cra_margins)
                }
                Some(s) => pick(s, &cnec.id, &margins, &ara_margins, &cra_margins),
            };
            margin.map(|m| m.in_unit(objective_unit))
        };
        let worst = crac
            .flow_cnecs
            .iter()
            .filter(|c| c.optimized)
            .filter_map(read)
            .fold(None, |acc: Option<f64>, m| Some(acc.map_or(m, |a| a.min(m))))?;
        let mnec = search_options.linear.mnec.options;
        let violated: f64 = if mnec.enabled {
            crac.flow_cnecs
                .iter()
                .filter(|c| c.monitored)
                .filter_map(|c| {
                    let initial =
                        initial_margins.get(c.id.as_str())?.in_unit(objective_unit);
                    let floor = f64::min(0.0, initial - mnec.acceptable_margin_decrease);
                    Some(mnec.violation_cost * (floor - read(c)?).max(0.0))
                })
                .sum()
        } else {
            0.0
        };
        Some(violated - worst)
    };

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
                    if crac.instants.iter().any(|i| i.id == *instant && i.kind == InstantKind::Auto)
                        && let Some(a) = &scenario.automatons
                    {
                        {
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

    // The transformers as that point in the plan left them. A shifter with no
    // set-point in this perimeter still has a tap — the one an earlier
    // perimeter put it on — and only this network knows which.
    let transformers_at = |at: &Where| -> &[gridoxide::types::Transformer] {
        let Where::After { contingency, instant } = at else {
            return &plan.preventive.transformers;
        };
        for scenario in &plan.scenarios {
            if crac.contingencies[scenario.contingency].id.trim() != contingency.trim() {
                continue;
            }
            if crac.instants.iter().any(|i| i.id == *instant && i.kind == InstantKind::Auto)
                && let Some(a) = &scenario.automatons
            {
                return &a.transformers;
            }
            if let Some(perimeter) = scenario.perimeters.iter().find(|p| {
                p.states.iter().any(|s| crac.instants[s.instant].id == *instant)
            }) {
                return &perimeter.transformers;
            }
            // A contingency with no perimeter at this instant is still after
            // the automatons, where there are any.
            if let Some(a) = &scenario.automatons {
                return &a.transformers;
            }
        }
        &plan.preventive.transformers
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
            Expect::ObjectiveValue { value, stage } => {
                let got = cost_at(*stage);
                record(
                    got.is_some_and(|g| (g - value).abs() <= flow_tolerance(*value)),
                    format!("objective function {got:?} (expected {value})"),
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
                    .or_else(|| {
                        resting_tap(&crac, &resolution, &net, transformers_at(at), action)
                    });
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
            Expect::Steps(want) => {
                let got = plan.steps.as_str();
                record(got == want, format!("execution details {got:?} (expected {want:?})"));
            }
            Expect::ActionNotUsed { action, at } => {
                let (used, _) = decisions(at);
                let got = used.iter().any(|u| u == action);
                record(!got, format!("`{action}` used: {got} (expected false) {at:?}"));
            }
            Expect::CnecFlow { cnec, value, unit, side, stage } => {
                let reading = match stage {
                    None => initial_margins.get(cnec.as_str()).copied(),
                    Some(s) => pick(*s, cnec, &margins, &ara_margins, &cra_margins),
                };
                let got = reading.and_then(|m| m.flow_in_unit(*unit, *side));
                record(
                    got.is_some_and(|g| (g - value).abs() <= flow_tolerance(*value)),
                    format!("flow on `{cnec}` side {side} {got:?} (expected {value})"),
                );
            }
            Expect::CnecThreshold { cnec, upper, value, unit } => {
                let got = declared_threshold(&crac, cnec, *upper, *unit);
                record(
                    got.is_some_and(|g| (g - value).abs() <= flow_tolerance(*value)),
                    format!(
                        "{} threshold on `{cnec}` {got:?} (expected {value})",
                        if *upper { "upper" } else { "lower" }
                    ),
                );
            }
            Expect::NetworkTap { element, tap } => {
                let got = tap_in_network(
                    &crac,
                    &resolution,
                    &net,
                    &plan.preventive.setpoints,
                    &plan.preventive.transformers,
                    element,
                );
                record(
                    got == Some(*tap),
                    format!("network tap of `{element}` {got:?} (expected {tap})"),
                );
            }
            Expect::Connected { element, connected } => {
                let got = resolution
                    .branch(element)
                    .map(|b| !plan.preventive.open_branches.contains(&b));
                record(
                    got == Some(*connected),
                    format!("`{element}` connected: {got:?} (expected {connected})"),
                );
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
    for (file, expected, baseline, pending) in FILES {
        run_gate(file, expected, baseline, pending);
    }

    // Every recorded reason has to name a scenario that is actually run.
    // `run_gate` can only notice an entry that has stopped disagreeing in a
    // file it appears in; one naming nothing at all would sit there forever,
    // looking like diligence.
    let mut known: Vec<String> = Vec::new();
    for (file, _, _, _) in FILES {
        let text = std::fs::read_to_string(features_dir().join(file)).expect("feature file");
        for scenario in parse(&text) {
            if let Some(id) = scenario.name.split_whitespace().next() {
                known.push(id.trim_end_matches(':').to_string());
            }
        }
    }
    let orphans: Vec<&str> = RECORDED_DISAGREEMENTS
        .iter()
        .map(|(s, _)| *s)
        .filter(|s| !known.iter().any(|k| k == s))
        .collect();
    assert!(orphans.is_empty(), "RECORDED_DISAGREEMENTS names scenarios nothing runs: {orphans:?}");
}

/// The vendored feature files, with their scenario counts and recorded
/// baselines. Scored separately on purpose — a gain in one must not hide a
/// regression in another.
const FILES: [(&str, usize, usize, Option<&str>); 4] = [
    ("dc_scenarios.feature", 25, BASELINE_MATCHED_DC, None),
    ("ac_scenarios.feature", 38, BASELINE_MATCHED_AC, None),
    ("ac_scenarios_16nodes.feature", 93, BASELINE_MATCHED_AC16, None),
    (
        "second_preventive.feature",
        15,
        BASELINE_MATCHED_2P,
        // The fourth field says a whole corpus is allowed to disagree, and why.
        // It exists so "the capability is partly built" cannot be confused with
        // "nobody has looked", which is the distinction the per-scenario check
        // beside it enforces everywhere else. Delete it when the six scenarios
        // below are settled — leaving it is how a corpus stops being measured.
        Some("second preventive holds curative range actions rather than re-optimizing them"),
    ),
];

/// Run one vendored feature file and assert on its aggregate.
///
/// The two files are scored separately on purpose. They exercise different flow
/// models, and a single total would let a gain in one hide a regression in the
/// other.
fn run_gate(file: &str, expected_scenarios: usize, baseline: usize, pending: Option<&str>) {
    let text = std::fs::read_to_string(features_dir().join(file)).expect("feature file");
    let scenarios = parse(&text);
    assert_eq!(scenarios.len(), expected_scenarios, "in {file}");

    let (mut matched, mut mismatched, mut unsupported) = (0usize, 0usize, 0usize);
    let mut report = String::new();
    // Scenarios that disagree without a recorded reason, and recorded reasons
    // that have stopped applying. Both are failures and neither touches the
    // ratio: see [`RECORDED_DISAGREEMENTS`].
    let (mut unexplained, mut stale) = (Vec::new(), Vec::new());
    for scenario in &scenarios {
        let outcome = check(scenario);
        matched += outcome.matched.len();
        mismatched += outcome.mismatched.len();
        unsupported += outcome.unsupported.len();
        match (recorded_reason(&scenario.name), outcome.mismatched.is_empty()) {
            (None, false) => unexplained.push(scenario.name.clone()),
            (Some(_), true) => stale.push(scenario.name.clone()),
            _ => {}
        }
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
        if let Some(reason) = recorded_reason(&scenario.name) {
            report.push_str(&format!("    why   {reason}\n"));
        }
    }
    let total = matched + mismatched;
    let note = pending.map_or(String::new(), |why| format!(" — {why}"));
    println!(
        "{report}\n{file}: {matched}/{total} checkable assertions match the reference \
         ({unsupported} steps unsupported){note}"
    );

    assert!(
        matched >= baseline,
        "{file}: {matched}/{total} matched, baseline is {baseline} — a drop is a regression:\n{report}"
    );
    assert!(
        unexplained.is_empty() || pending.is_some(),
        "{file}: {unexplained:?} disagree with the reference and nothing says why. Either that is a \
         defect, or it is a disagreement worth standing behind — and standing behind one means \
         measuring both answers on the reference's own objective and adding it to \
         RECORDED_DISAGREEMENTS with that measurement. Do not add an entry you have not \
         measured.\n{report}"
    );
    assert!(
        stale.is_empty() || pending.is_some(),
        "{file}: {stale:?} are listed in RECORDED_DISAGREEMENTS and no longer disagree. Delete \
         those entries — a standing excuse for something already fixed will one day excuse a \
         regression instead."
    );
}

/// Where the two implementations disagree **and gridoxide is staying put**.
///
/// Every entry has been measured: both answers evaluated on the reference's own
/// objective, in the unit its own configuration selects, with its own MNEC
/// violation cost applied. None of them is gridoxide being worse.
///
/// The list buys the gate a property a count cannot have — that **no scenario
/// disagrees for a reason nobody has looked at**. That is sharper than the
/// baselines beside it, and it is the one that catches a new defect hiding
/// inside an old total.
///
/// These are **not** excused. Their assertions still count as mismatched in the
/// ratio, exactly as before. Moving them out of the denominator is the one
/// thing this must never do: a gate that stops counting what it has decided not
/// to fix stops being a measurement.
///
/// Adding an entry means doing the measurement first. An entry without one is a
/// claim to be better, dressed up as a record of being better.
const RECORDED_DISAGREEMENTS: &[(&str, &str)] = &[
    (
        "1.3.2.6",
        "same worst margin, reached with one curative action instead of two — the alternative \
         optimum risk 1 names, and this CRAC has no MNEC to break the tie",
    ),
    ("1.3.2.8", "gridoxide 461.3 A against the reference's 433 on the binding CNEC, one tap apart"),
    (
        "1.3.6.6",
        "gridoxide 630.0 A against 612, spending one curative action the reference declines; the \
         MNEC it moves lands at 21.8 A against a floor of 0, so nothing is violated to get there",
    ),
    ("1.3.8.2", "same worst margin, one tap apart on a non-binding CNEC"),
    (
        "5.2.1.3",
        "BestTapFinder: gridoxide's tap -6 scores 188.42 with no MNEC violation; the reference's \
         -7 scores 192.05 and pays 0.76 of violation, for 184.41",
    ),
    ("5.2.1.4", "BestTapFinder: as 5.2.1.3 — 188.42 against 184.41"),
    (
        "5.2.3.2",
        "BestTapFinder: gridoxide's -11 scores -156.28 clean; the reference's -12 scores -146.33 \
         and pays 1.42 of violation at cost 15, for -167.60",
    ),
    (
        "5.2.3.3",
        "BestTapFinder: gridoxide's -8 scores -186.15 clean; the reference's -9 scores -176.19 \
         and pays 1.70 of violation, for -201.62",
    ),
];

/// The reason recorded for a scenario, if any.
///
/// Matched on the leading identifier, so the feature file's own punctuation —
/// some scenario names carry a trailing colon and some do not — cannot decide
/// whether a disagreement counts as explained.
fn recorded_reason(name: &str) -> Option<&'static str> {
    let id = name.split_whitespace().next()?.trim_end_matches(':');
    RECORDED_DISAGREEMENTS.iter().find(|(s, _)| *s == id).map(|(_, why)| *why)
}

/// How many of the reference's assertions currently hold: **175 of 181**,
/// across all 25 scenarios.
///
/// It was 150 of 156 before `the execution details should be` left the skip
/// bucket. That step was skipped 171 times across the four files — more than
/// every other unsupported step put together — on the grounds that it named
/// second-preventive bookkeeping. It does not: it names which optimization
/// steps ran and how each turned out, and it is the one thing no margin can
/// tell you, because a plan and the plan it fell back to have different
/// margins but the same *shape*. All 25 of this file's hold.
///
/// It was 124 of 124 before the three MNEC scenarios (5.2.1.2 to 5.2.1.4)
/// joined it. Eight of their twelve assertions hold; the four that do not are
/// two taps and the two margins that follow from them, and they are the
/// `BestTapFinder` divergence described on [`BASELINE_MATCHED_AC`].
///
/// It was 138 of 142 before the per-side flow, threshold, resting-tap and
/// connection-status steps were taken out of the skip bucket. Twelve of the
/// fourteen new assertions hold; the two that do not are the third and fourth
/// sighting of that same `BestTapFinder` divergence, now visible as a tap
/// position in the network as well as a margin.
///
/// It is a recorded number rather than an assertion of perfection. These are
/// two heuristic search trees and §8.3 says up front that a different set of
/// actions reaching the same margin is not a defect; a scenario added later may
/// legitimately disagree. Raising this is progress, a drop is a regression, and
/// the printed report says which assertion moved.
const BASELINE_MATCHED_DC: usize = 175;

/// The same, for the 38 AC scenarios in `ac_scenarios.feature`: **276 of 282**.
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
/// Four of the MNEC scenarios — 5.2.1.3, 5.2.1.4, 5.2.3.2 here and 5.2.3.3 above —
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
/// Matching these assertions would mean reproducing that rounding, at the cost
/// of a worse answer. They are left as recorded disagreements, and with family
/// 3.2 complete they are now the whole of what this file still misses.
///
/// 5.2.3.2 joined them when the **ampere margin** was corrected — an ampere
/// margin is now the limit in amperes less the current, rather than the
/// megawatt margin converted, which is exactly the reference's own number
/// (−146.33 A against its −146.3) at the reference's own tap. That moved the
/// objective enough to expose the same rounding question one scenario further
/// on: at tap −12 the worst margin is −146.33 with an MNEC violated by 1.4 A,
/// at −11 it is −156.28 with none, and scoring the virtual cost makes −11 the
/// better answer by 4.2. The reference does not score it and takes −12. Three
/// assertions here, against three gained in 3.2 on the same change.
///
/// # The slack, which family 3.2 turned on
///
/// It was 232 before the slack was **distributed** and weighted by
/// **generation**, which took 3.2 from 24 of 33 to 33 of 33 and moved nothing
/// else. Both halves matter and the second is the one that hides: netting
/// generation against load puts 70% of an islanded node's make-up back inside
/// or next to the country that lost it, so distributing barely changes the
/// answer and the whole idea looks refuted. See
/// [`the_slack_is_shared_out_in_proportion_to_generation`] in
/// `tests/rao_evaluate_test.rs`.
///
/// # What the per-side flow steps bought
///
/// It was 192 of 203 before the `flow on cnec … on side N` steps left the skip
/// bucket — 41 new assertions, 40 of which hold. Getting there needed two
/// conventions stated that the evaluator had never had to commit to, and
/// neither was guessable from the margins:
///
/// - An ampere flow is **signed by the active power**. `CnecResult::current_a`
///   is a magnitude, correctly, but `getFlow(cnec, side, AMPERE)` divides the
///   signed megawatts by `√3·U`, so a step reading `-1444.0 A` is naming a
///   direction. The magnitudes had agreed to five figures all along.
/// - **Side two is measured in the same direction as side one** — power
///   entering at one end, leaving at the other — so the two differ by the
///   branch's losses. Reporting the power *entering* side two negates it, and
///   the pair then differ by twice the flow.
const BASELINE_MATCHED_AC: usize = 276;

/// The same, for the 93 AC scenarios on `TestCase16Nodes`: **963 of 977**.
///
/// The largest of the three files and the newest, so the furthest from
/// settled. It is here to find defects, and it does.
///
/// Its four MNEC scenarios (1.3.6.1, 1.3.6.5 to 1.3.6.7) contribute 39 of 42:
/// three of them match outright, and the three misses are all in 1.3.6.6's
/// curative perimeter on `co1_fr2_fr3_1` — which is where a good part of this
/// file's remaining disagreements sit, MNECs or no MNECs.
///
/// The `value of the objective function` steps are checked here too — 39 of
/// them, of which 31 hold. Every one of the eight that does not sits in a
/// scenario whose margins already disagree, so they add no new *kind* of
/// failure; they make the existing ones visible in one more place, which is
/// what a gate is for.
///
/// The **2.4 usage-rule family matches in full** — 147 of 147, up from 100 —
/// since conditional usage rules are answered against the perimeter's flows
/// rather than assumed true. Enforcing the CRAC's **usage limits** then took
/// 2.6 from 73 of 134 to 103, and 2.2 from 39 of 63 to 54.
///
/// Then the **curative stop criterion** — a curative perimeter searches until
/// it beats the preventive one and no further — took 1.2 from 69 of 119 to 94
/// and 1.3 from 335 of 420 to 346.
///
/// Then making a curative **close** possible at all — the perimeter's
/// already-open branches are now a stated set rather than an impedance, so an
/// action that removes one from it actually reconnects the branch — took 1.3
/// from 372 to 404, 2.6 from 103 to 130, and 2.2 to 63 of 63. It was worth 68
/// assertions on this file and none on the other two, which have no curative
/// close between them. Nothing regressed: every other family is unchanged to
/// the assertion.
///
/// Then the curative **range** actions, which is where that left the residue.
/// Two causes, and the smaller one was in this harness:
///
/// - A `relativeToPreviousInstant` range was read as absolute, so a shifter the
///   CRAC allowed ten taps either side of the preventive answer got ten taps
///   either side of *zero*. On 1.3.4.3 that is tap 10 where the reference
///   reaches 15, with five taps of permitted travel the optimizer never knew it
///   had — and nothing about it visible in a margin, because the answer stays
///   feasible, self-consistent and worse. Worth 39.
/// - A shifter with no set-point in a perimeter was reported at the **file's**
///   tap rather than at the one an earlier perimeter put it on, so a plan that
///   moved a PST in preventive and never revisited it read as having undone the
///   move. The reference's `getOptimizedTapOnState` answers for every state
///   from the set-points in force there, and now so does this. Worth 6.
///
/// Then the **automaton simulator**, which that left as the largest cause.
/// Three things, of which only the first was suspected:
///
/// - It sized its shift against **DC** margins while the run measured in AC. On
///   `co2_be1_be3` the DC overload is −120.5 MW where AC says −70.8, so the
///   formula asked for roughly twice the travel it needed.
/// - It ignored the range action's **own range**, stopping only when it ran out
///   of tap changer — 16 where the CRAC allowed 10.
/// - It shifted **once**. The set-point is sized from a linear estimate and
///   applied to a network that is not linear, so one shot lands short: tap −7
///   on 1.2.2.2 with the watched CNEC still overloaded, where −8 clears it.
///
/// Took 1.2 from 98 of 121 to 111.
///
/// The last of that family was not the automaton at all: a **range action whose
/// starting set-point is outside its own range** was optimized rather than
/// dropped. `SL_ep15us11-3case2_withPstCra` declares four range actions on one
/// shifter and names one of them `useless_pst`, permitting tap 0 and nothing
/// else — and by the curative perimeter an automaton has put that shifter on
/// −8. Kept, it is a second control on a device that already has one, pinned to
/// a position the machine is not at, and the curative perimeter moves nothing.
/// Dropping it, as `doesPrePerimeterSetpointRespectRange` does, takes 1.2 to
/// 118 of 121 and 1.2.2.5 to 22 of 22.
///
/// The last three were an ordering rule, not a sizing one: an automaton that
/// states **no speed fires first**, since the reference's `DEFAULT_SPEED` is
/// zero. The tempting reading is the opposite — "no speed stated" looks like
/// "no claim to be fast" — and it costs answers rather than order. On 1.2.2.4
/// the untimed `open_be1_be4` opens a Belgian circuit and the two shifters that
/// follow are sized against what that leaves behind, so `pst_be` needs one tap;
/// fired last they are sized against an overload the opening was about to
/// remove, and spend four. **Family 1.2 is now 121 of 121.**
///
/// Then the rule that the perimeters cannot see between them: **a plan that
/// ends worse than doing nothing is thrown away**. Each perimeter accepts only
/// candidates that improve its own objective, but they do not partition the
/// harm — a preventive action is judged on the base case and the outage states,
/// and what it costs a *curative* state is invisible there. On 1.4.4.2 closing
/// two circuits takes the preventive perimeter from 590.6 to 681.7 MW and the
/// curative state to −342; the curative perimeter recovers half and the plan
/// still ends below where it began. The reference compares the finished plan
/// against the untouched network and discards it — `postCheckResults` — and its
/// own report calls the outcome "First preventive fell back to initial
/// situation". **Family 1.4 is now 9 of 9.**
///
/// What is still open, by size: 3.2 (9 of 33), 5.2 MNEC (9 of 68), 1.3 (14 of
/// 458). Families 1.2, 1.4, 2.2 and 2.6 are complete. Of the remaining tap
/// disagreements six are the `BestTapFinder` divergence recorded on
/// [`BASELINE_MATCHED_AC`], and the rest are one tap apart.
const BASELINE_MATCHED_AC16: usize = 963;

/// The 15 second-preventive scenarios: **91 of 123**, from 45 of 108 before the
/// capability existed.
///
/// 14 of the 15 `execution details` assertions hold. The one that does not is
/// 1.4.1.5, and it is honest: gridoxide's second pass ran and was **declined**,
/// so it says so, where the reference's improved. That scenario is already one
/// of the four §8.7 records as a genuine gap — the step is reporting the gap
/// rather than adding one.
///
/// The corpus was vendored first and scored at 45 with nothing implemented,
/// which is the order everything else in this file was built in and the only
/// order that works: the gate found all twenty-nine defects in §8.3, and
/// building a capability with nothing to check it against is how the
/// twenty-ninth survived three refuted hypotheses.
///
/// # What the 33 that remain are
///
/// Four scenarios need one range action at **two set-points** — a PST at +5 in
/// preventive and −5 in curative, say. The reference re-optimizes curative range
/// actions inside the second preventive problem, which needs a set-point per
/// action *per state*: `A(r, s)` in `plans/RAO_PLAN.md` §7.3, declared there and
/// not built, because this LP carries one set-point per action. Until it does, a
/// shifter the CRAC allows in both instants is held where the curative stage put
/// it rather than re-tuned, and 1.4.1.1.3, 1.4.1.2, 1.4.1.5 and 1.4.1.6 turn on
/// exactly that.
///
/// 1.4.4.4 used to be counted with them and was not one of them. It was the
/// second pass **inheriting a curative set-point**: `pst_be` at −16, chosen
/// against preventive decisions the second pass was in the act of discarding,
/// capped its whole landscape at 553 A. Released, the pass finds the
/// reference's answer with no network action at all, and 1.4.5.1 — the scenario
/// named for fixing CRA set-points — went from four of five to five of five.
/// `plans/RAO_PLAN.md` §8.8 has the sweep, and
/// `the_second_preventive_pass_does_not_inherit_a_curative_set_point` pins it.
///
/// The one assertion that change cost is 1.4.1.6's `security status`, and it
/// was matching by accident: `plan.final_margin_mw` read 41.2 MW there while
/// the plan's own network measured −32.4 A, because the figure was taken in the
/// second pass's network with another contingency's curative shifter still in
/// it. Both numbers now agree, and both disagree with the reference — which is
/// the honest report of a scenario that was already wrong.
const BASELINE_MATCHED_2P: usize = 91;

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
