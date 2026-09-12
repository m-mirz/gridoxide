//! Reading the reference's `RaoParameters` document.
//!
//! OpenRAO states a run's configuration as a JSON file, and every scenario in
//! the vendored Cucumber corpus names one. This maps the knobs that change an
//! answer onto [`SearchOptions`].
//!
//! # Why this is in the library
//!
//! It was in the gate's own harness, and only there, which meant the optimizer
//! could be *told* to minimize cost, hold MNECs, run a second preventive pass or
//! measure in amperes — and `gridoxide rao` could ask for none of it. The gate
//! validated behaviour the binary had no way to reach.
//!
//! It stays validated: the harness calls this module rather than its own copy,
//! so the same 1862 assertions that check the optimizer check this reading of
//! its settings.
//!
//! # What it does not read, and why that is not an oversight
//!
//! A parameter this does not understand is ignored **silently on purpose**: the
//! alternative is to fail on the dozens of AC, loop-flow and solver settings
//! these files carry, none of which apply to a DC run. Specifically unread, each
//! for a stated reason:
//!
//! - `pst-model` — `TapModel::Discrete` is declared and unbuilt, and
//!   `plans/RAO_PLAN.md` §8.21 measured that as unjustified rather than assumed
//!   it: all 24 scenarios whose configuration asks for `APPROXIMATED_INTEGERS`
//!   already match with the continuous model and rounding.
//! - `linear-optimization-solver` — the solver follows the objective. A cost LP
//!   needs a simplex or MIP backend (§8.14), and the caller supplies one.
//! - `ra-range-shrinking`, `available-cpus`, `sensitivity-failure-overcost`,
//!   `do-not-optimize-curative-cnecs-for-tsos-without-cras` — no counterpart
//!   exists in `src/rao/`.
//! - `load-flow-parameters` beyond `dc` — and one of these is worth knowing
//!   about: `distributedSlack` and `balanceType` are *implemented*, but wired to
//!   the values every vendored configuration states rather than to the file (see
//!   [`evaluate`](super::evaluate)). A document setting them otherwise is
//!   ignored.
//!
//! Depth is read as the file states it. The gate clamps it afterwards for its
//! own reasons — a bound in name only would have it evaluate every combination
//! of a corpus it has no time budget for — and that is a harness policy, not a
//! reading of the document, so it does not belong here.

use serde_json::Value;

use super::costly::{Costly, CostlyOptions};
use super::evaluate::FlowModel;
use super::linear::{ObjectiveKind, ObjectiveUnit};
use super::search::{SearchOptions, SecondPreventiveCondition};

/// Read a `RaoParameters` document from JSON text.
///
/// A document that will not parse yields the defaults, for the same reason an
/// unknown key is ignored: a configuration file is a description of a run, and
/// refusing to run at all because one field is unreadable serves nobody. The
/// caller that wants strictness can parse it itself and call
/// [`from_document`].
pub fn from_json(text: &str) -> SearchOptions {
    match serde_json::from_str::<Value>(text) {
        Ok(doc) => from_document(&doc),
        Err(_) => SearchOptions::default(),
    }
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
pub fn flow_model(doc: &serde_json::Value) -> FlowModel {
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

pub fn from_document(doc: &serde_json::Value) -> SearchOptions {
    let mut options = SearchOptions::default();
    // `RaoUtil.getFlowUnit`: megawatts for a DC load flow, amperes for an AC
    // one. Not a setting of its own — the objective follows the flow model, and
    // the minimum-impact thresholds below are stated in whichever unit results.
    options.linear.flow_model = flow_model(doc);
    options.linear.objective_unit = match options.linear.flow_model {
        FlowModel::Ac => ObjectiveUnit::Ampere,
        FlowModel::Dc => ObjectiveUnit::Megawatt,
    };
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

    // `MIN_COST`: minimize what the plan costs rather than maximize what it
    // buys. Read here rather than left to fall through, which is what happened
    // before this and is how a costly configuration was silently answered in
    // the wrong currency.
    if doc.pointer("/objective-function/type").and_then(|v| v.as_str()) == Some("MIN_COST") {
        let mut costly = CostlyOptions::default();
        if let Some(c) = doc.pointer(
            "/extensions/open-rao-search-tree-parameters/costly-min-margin-parameters",
        ) {
            if let Some(v) = c.get("shifted-violation-penalty").and_then(Value::as_f64) {
                costly.violation_penalty = v;
            }
            if let Some(v) = c.get("shifted-violation-threshold").and_then(Value::as_f64) {
                costly.violation_threshold = v;
            }
        }
        options.linear.objective_kind = ObjectiveKind::MinCost(Costly { options: costly });
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
        if let Some(v) = topology.get("max-preventive-search-tree-depth").and_then(|v| v.as_u64()) {
            options.max_depth = v as usize;
        }
        // Read separately, because the reference states it separately. Every
        // vendored configuration sets the two the same, so this moves nothing
        // here — and a configuration that did not would otherwise have been
        // scored against the preventive depth without a word.
        if let Some(v) = topology.get("max-curative-search-tree-depth").and_then(|v| v.as_u64()) {
            options.curative_max_depth = Some(v as usize);
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

