//! Simulating automatons — the `auto` instant.
//!
//! An automaton is **not optimized**. A protection scheme fires when its
//! trigger condition is met, whether or not that helps anything else, and the
//! job here is to reproduce what the equipment does rather than to choose what
//! it should do. That is the whole difference between this module and
//! [`search`](super::search): the search asks "which of these would be best?",
//! and this asks "which of these will actually operate?".
//!
//! # The order matters, and it is `speed`
//!
//! Automatons fire in ascending order of their stated speed, in batches, and
//! **the trigger conditions are re-evaluated between batches but not within
//! one**. Both halves of that are load-bearing, and two vendored scenarios pin
//! them from opposite sides:
//!
//! - Re-evaluating *between* batches is why a fast automaton that relieves an
//!   overload stops a slower one from ever seeing the condition that would have
//!   triggered it. One scenario has five automatons at four speeds, of which the
//!   speed-2 one must not fire because the speed-1 one already fixed its CNEC.
//! - **Not** re-evaluating *within* a batch is why an automaton whose CNEC is
//!   healthy when the batch begins stays out of it, even if a sibling firing
//!   alongside pushes that CNEC into overload. Another scenario has two
//!   speed-less automatons — therefore one batch — where the second's CNEC sits
//!   at +43.5 MW until the first fires. Equipment that sampled the grid
//!   simultaneously would not see that, and the reference does not fire it.
//!
//! Range actions are the exception, and the reference's own description is
//! explicit about the asymmetry: "First, **all** automatic network actions are
//! applied… **Then**, automatic range actions are applied one by one, as long
//! as some of the perimeter's CNECs are overloaded." A range action has a
//! set-point to size against the flows as they stand, so it necessarily sees
//! the state its predecessors left.
//!
//! # Two kinds, handled differently
//!
//! **Network actions** are binary and are simply applied when triggered.
//!
//! **Range actions** have a set-point to compute, and there is nothing to
//! optimize against — so the set-point comes from a formula rather than an LP:
//!
//! \\[ A_{\text{new}} = A_{\text{current}} + \operatorname{sign}(F(c))\,\frac{\min(0,\ \text{margin}(c))}{\sigma} \\]
//!
//! taking \\(c\\) as the worst-overloaded CNEC the action is watching: shift
//! just far enough to clear it, capped by the action's own range. Once an action
//! has moved in one direction it will not be moved back, since a protection
//! scheme does not hunt.

use crate::linear::btheta::dc_branches;
use crate::linear::sensitivity::DcSensitivity;
use crate::linear::DcOptions;
use crate::types::Transformer;

use super::crac::{Crac, ElementaryAction, InstantKind, RangeActionKind, State, UsageRule};
use super::evaluate::{evaluate_with, Network, Resolution};
use super::linear::{phase_shift_sensitivity, tap_table};

/// What fired, and what it left behind.
#[derive(Clone, Debug, Default)]
pub struct AutomatonResult {
    /// Indices into [`Crac::network_actions`] that operated.
    pub network_actions: Vec<usize>,
    /// `(range action index, set-point, tap)` for each that moved.
    pub range_actions: Vec<(usize, f64, Option<i32>)>,
    /// The open set after everything fired.
    pub open_branches: Vec<usize>,
    /// The transformers after everything fired.
    pub transformers: Vec<Transformer>,
    /// Worst margin in the auto perimeter, before and after.
    pub initial_margin_mw: f64,
    pub final_margin_mw: f64,
    /// Speed batches actually processed, for reporting.
    pub batches: usize,
}

impl AutomatonResult {
    pub fn fired(&self) -> usize {
        self.network_actions.len() + self.range_actions.len()
    }
}

/// Cap on how many times a batch may re-fire before the simulation gives up.
///
/// A protection scheme that keeps triggering itself is a modelling error, not
/// something to iterate to convergence, so this is a guard rather than a
/// tolerance.
const MAX_ROUNDS: usize = 20;

/// Simulate the automatons of one contingency's `auto` state.
///
/// `open` and `transformers` come in as the preventive perimeter left them and
/// go out as the automatons leave them, which is what the curative perimeters
/// then start from.
pub fn simulate(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    contingency: usize,
    open: &[usize],
    transformers: &[Transformer],
) -> Option<AutomatonResult> {
    let instant = crac
        .instants
        .iter()
        .position(|i| i.kind == InstantKind::Auto)?;
    let state = State { instant, contingency: Some(contingency) };
    if !crac.flow_cnecs.iter().any(|c| c.state == state) {
        return None;
    }

    let mut result = AutomatonResult {
        open_branches: open.to_vec(),
        transformers: transformers.to_vec(),
        ..Default::default()
    };
    result.initial_margin_mw =
        margin(crac, network, resolution, &state, &result.open_branches, &result.transformers);
    result.final_margin_mw = result.initial_margin_mw;

    // Speeds, ascending. An action without one goes last: an unstated speed is
    // not "instant", and assuming it were would let it pre-empt equipment the
    // file actually timed.
    let mut speeds: Vec<i64> = Vec::new();
    for action in crac.network_actions.iter().filter(|a| covers(&a.usage_rules, &state)) {
        speeds.push(action.speed.unwrap_or(i64::MAX));
    }
    for action in crac.range_actions.iter().filter(|a| covers(&a.usage_rules, &state)) {
        speeds.push(action.speed.unwrap_or(i64::MAX));
    }
    speeds.sort_unstable();
    speeds.dedup();

    for speed in speeds {
        result.batches += 1;

        // One snapshot for the whole batch's *network* actions. Equipment in the
        // same speed class samples the grid at the same moment; asking each in
        // turn what its sibling just did is a different — and, on the vendored
        // evidence, wrong — model.
        let snapshot =
            violated(crac, network, resolution, &state, &result.open_branches, &result.transformers);
        for (index, action) in crac.network_actions.iter().enumerate() {
            if result.network_actions.contains(&index) {
                continue;
            }
            if action.speed.unwrap_or(i64::MAX) != speed {
                continue;
            }
            if !triggered(&action.usage_rules, &state, &snapshot) {
                continue;
            }
            if !apply_network_action(action, resolution, &mut result) {
                continue;
            }
            result.network_actions.push(index);
        }

        // Then range actions, one at a time, each sized against the flows as
        // they stand.
        let mut rounds = 0;
        loop {
            rounds += 1;
            if rounds > MAX_ROUNDS {
                break;
            }
            let violations =
                violated(crac, network, resolution, &state, &result.open_branches, &result.transformers);
            let mut moved = false;
            for (index, action) in crac.range_actions.iter().enumerate() {
                if result.range_actions.iter().any(|(i, _, _)| *i == index) {
                    continue;
                }
                if action.speed.unwrap_or(i64::MAX) != speed {
                    continue;
                }
                if !triggered(&action.usage_rules, &state, &violations) {
                    continue;
                }
                let RangeActionKind::Pst { element, tap_to_angle, .. } = &action.kind else {
                    continue;
                };
                let Some(branch) = resolution.branch(element) else { continue };
                let table =
                    tap_table(tap_to_angle, network.tap_changers, branch, network.lines.len());
                // Sized against the CNECs *this* action watches, not the worst
                // in the perimeter. A scheme is wired to a particular circuit;
                // sizing it against somebody else's overload asks it to relieve
                // a flow it may have no influence over at all, and the near-zero
                // sensitivity then makes it decline to act.
                let watched = watched_cnecs(&action.usage_rules, &state, &violations);
                if watched.is_empty() {
                    continue;
                }
                let Some((tap, angle)) = shift_to_relieve(
                    crac, network, resolution, &state, &result, branch, &table, &watched,
                ) else {
                    continue;
                };
                if let Some(i) = branch.checked_sub(network.lines.len()) {
                    // The network's own step at the chosen tap — see the note in
                    // `linear::apply`. A tap changer is equipment; a CRAC's table
                    // describes it, and the equipment wins when they disagree.
                    let from_network = network
                        .tap_changers
                        .get(i)
                        .and_then(|c| c.as_ref())
                        .and_then(|c| c.at(tap));
                    if let Some(t) = result.transformers.get_mut(i) {
                        let ratio = t.tap.norm();
                        t.tap = match from_network {
                            Some(step) => num_complex::Complex::from_polar(ratio, step.arg()),
                            None => num_complex::Complex::from_polar(ratio, (-angle).to_radians()),
                        };
                    }
                }
                result.range_actions.push((index, angle, Some(tap)));
                moved = true;
                break;
            }
            if !moved {
                break;
            }
        }
    }

    result.final_margin_mw =
        margin(crac, network, resolution, &state, &result.open_branches, &result.transformers);
    Some(result)
}

fn covers(rules: &[UsageRule], state: &State) -> bool {
    rules.iter().any(|r| r.covers(state))
}

/// Whether an automaton's trigger condition is met *now*.
///
/// A bare `OnInstant` or `OnContingencyState` rule is unconditional — the
/// equipment operates on the contingency itself. An `OnConstraint` rule names
/// the CNEC it watches and fires only while that CNEC is over its threshold,
/// which is what makes the order of firing matter.
fn triggered(rules: &[UsageRule], state: &State, violations: &[String]) -> bool {
    let mut conditional = false;
    for rule in rules.iter().filter(|r| r.covers(state)) {
        match rule {
            UsageRule::OnInstant { .. } | UsageRule::OnContingencyState { .. } => return true,
            UsageRule::OnConstraint { cnec, .. } => {
                conditional = true;
                if violations.iter().any(|v| v == cnec) {
                    return true;
                }
            }
            UsageRule::OnFlowConstraintInCountry { .. } => {
                conditional = true;
                if !violations.is_empty() {
                    return true;
                }
            }
        }
    }
    // No rule reached this state at all.
    let _ = conditional;
    false
}

/// The CNECs an action's own rules name, restricted to those actually
/// violated.
///
/// An unconditional rule watches everything, since it fires on the contingency
/// rather than on a constraint.
fn watched_cnecs(rules: &[UsageRule], state: &State, violations: &[String]) -> Vec<String> {
    let mut named = Vec::new();
    for rule in rules.iter().filter(|r| r.covers(state)) {
        match rule {
            UsageRule::OnConstraint { cnec, .. } => {
                if violations.iter().any(|v| v == cnec) {
                    named.push(cnec.clone());
                }
            }
            _ => return violations.to_vec(),
        }
    }
    named
}

fn apply_network_action(
    action: &super::crac::NetworkAction,
    resolution: &Resolution,
    result: &mut AutomatonResult,
) -> bool {
    let mut opened = Vec::new();
    let mut closed = Vec::new();
    for elementary in &action.elementary {
        match elementary {
            ElementaryAction::TerminalsConnection { element, connected } => {
                let Some(branch) = resolution.branch(element) else { return false };
                if *connected {
                    closed.push(branch);
                } else {
                    opened.push(branch);
                }
            }
            ElementaryAction::Switch { element, open } => {
                let Some(branch) = resolution.branch(element) else { return false };
                if *open {
                    opened.push(branch);
                } else {
                    closed.push(branch);
                }
            }
            // Anything else is refused whole rather than applied in part, on the
            // same principle the search uses: a network action is one decision.
            _ => return false,
        }
    }
    result.open_branches.extend(opened);
    result.open_branches.retain(|b| !closed.contains(b));
    result.open_branches.sort_unstable();
    result.open_branches.dedup();
    true
}

/// The set-point that just clears the worst CNEC this action is watching.
///
/// Returns the tap and its angle, or `None` when nothing would help — the
/// sensitivity is negligible, the range is exhausted, or the shift needed is
/// the direction the action has already been moved away from.
#[allow(clippy::too_many_arguments)]
fn shift_to_relieve(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    state: &State,
    result: &AutomatonResult,
    branch: usize,
    tap_to_angle: &[(i32, f64)],
    violations: &[String],
) -> Option<(i32, f64)> {
    if tap_to_angle.is_empty() {
        return None;
    }
    let view = Network {
        buses: network.buses,
        lines: network.lines,
        transformers: &result.transformers,
        branch_ids: network.branch_ids,
        bus_ids: network.bus_ids,
        initially_open: network.initially_open,
        bus_countries: network.bus_countries,
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };
    let assessment = evaluate_with(crac, &view, resolution, &result.open_branches);
    let perimeter = assessment.perimeters.iter().find(|p| p.state == *state)?;

    // The worst CNEC this automaton is watching, among those actually violated.
    let worst = perimeter
        .cnecs
        .iter()
        .filter(|c| violations.iter().any(|v| *v == crac.flow_cnecs[c.cnec].id))
        .min_by(|a, b| a.margin_mw.total_cmp(&b.margin_mw))?;

    let options = DcOptions::default();
    let branches = dc_branches(network.lines, &result.transformers, options);
    let sensitivity = DcSensitivity::new(
        network.buses,
        &branches,
        network.lines.len() + result.transformers.len(),
    )?;
    let column = phase_shift_sensitivity(&sensitivity, &branches, branch)?;
    // MW per degree of the CRAC's angle — the same sign convention the linear
    // optimizer uses, and for the same reason.
    let sigma = -column.get(worst.branch).copied().unwrap_or(0.0)
        * network.base_mva
        * std::f64::consts::PI
        / 180.0;
    if sigma.abs() < 1e-6 {
        return None;
    }

    // Shift just far enough to bring |flow| back to the limit.
    let overload = worst.margin_mw.min(0.0);
    let needed = worst.flow_mw.signum() * overload / sigma;
    let current = tap_to_angle
        .iter()
        .find(|(t, _)| Some(*t) == current_tap(result, branch, network.lines.len(), tap_to_angle))
        .map(|(_, a)| *a)
        .unwrap_or(0.0);
    let target = current + needed;

    // Capped by the table, and rounded *away* from the current position so the
    // shift is at least as large as required rather than fractionally short.
    let mut best: Option<(i32, f64)> = None;
    for &(tap, angle) in tap_to_angle {
        let far_enough = if needed >= 0.0 { angle >= target } else { angle <= target };
        let right_way = if needed >= 0.0 { angle >= current } else { angle <= current };
        if !right_way {
            continue;
        }
        if far_enough {
            let better = best.is_none_or(|(_, b)| (angle - current).abs() < (b - current).abs());
            if better {
                best = Some((tap, angle));
            }
        }
    }
    // Nothing reaches it: go as far as the range allows, which is what a real
    // scheme does rather than declining to act.
    let chosen = best.or_else(|| {
        tap_to_angle
            .iter()
            .filter(|(_, a)| if needed >= 0.0 { *a >= current } else { *a <= current })
            .max_by(|a, b| (a.1 - current).abs().total_cmp(&(b.1 - current).abs()))
            .copied()
    })?;
    if (chosen.1 - current).abs() < 1e-9 {
        return None;
    }
    Some(chosen)
}

/// The tap the transformer is actually at, read from its live angle.
///
/// Converted out of gridoxide's convention into the CRAC's, which is the one
/// the tap table is written in — the same negation the linear optimizer applies
/// and for the same reason.
fn current_tap(
    result: &AutomatonResult,
    branch: usize,
    lines: usize,
    tap_to_angle: &[(i32, f64)],
) -> Option<i32> {
    let live = branch
        .checked_sub(lines)
        .and_then(|i| result.transformers.get(i))
        .map(|t| -t.tap.arg().to_degrees())
        .unwrap_or(0.0);
    tap_to_angle
        .iter()
        .min_by(|a, b| (a.1 - live).abs().total_cmp(&(b.1 - live).abs()))
        .map(|(t, _)| *t)
}

/// Ids of the CNECs currently over their threshold in this state.
fn violated(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    state: &State,
    open: &[usize],
    transformers: &[Transformer],
) -> Vec<String> {
    let view = Network {
        buses: network.buses,
        lines: network.lines,
        transformers,
        branch_ids: network.branch_ids,
        bus_ids: network.bus_ids,
        initially_open: network.initially_open,
        bus_countries: network.bus_countries,
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };
    evaluate_with(crac, &view, resolution, open)
        .perimeters
        .iter()
        .filter(|p| p.state == *state)
        .flat_map(|p| p.violations())
        .map(|c| crac.flow_cnecs[c.cnec].id.clone())
        .collect()
}

fn margin(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    state: &State,
    open: &[usize],
    transformers: &[Transformer],
) -> f64 {
    let view = Network {
        buses: network.buses,
        lines: network.lines,
        transformers,
        branch_ids: network.branch_ids,
        bus_ids: network.bus_ids,
        initially_open: network.initially_open,
        bus_countries: network.bus_countries,
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };
    evaluate_with(crac, &view, resolution, open)
        .perimeters
        .iter()
        .find(|p| p.state == *state)
        .and_then(|p| p.min_margin())
        .unwrap_or(f64::INFINITY)
}
