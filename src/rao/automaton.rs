//! Simulating automatons — the `auto` instant.
//!
//! An automaton is **not optimized**. A protection scheme fires when its
//! trigger condition is met, whether or not that helps anything else, and the
//! job here is to reproduce what the equipment does rather than to choose what
//! it should do. That is the whole difference between this module and
//! [`search`](mod@super::search): the search asks "which of these would be best?",
//! and this asks "which of these will actually operate?".
//!
//! # The order matters, and it is `speed`
//!
//! Automatons fire in ascending order of their speed — [`DEFAULT_SPEED`] for
//! one that states none, which puts it **first** — in batches, and **the
//! trigger conditions are re-evaluated between batches but not within one**.
//! Both halves of that are load-bearing, and two vendored scenarios pin them
//! from opposite sides:
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
use super::evaluate::{evaluate_model, CnecResult, FlowModel, Network, Resolution};
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

/// How many times one automaton may re-size its own set-point.
///
/// The reference's `MAX_NUMBER_OF_SENSI_IN_AUTO_SETPOINT_SHIFT`, and like
/// [`MAX_ROUNDS`] a guard rather than a tolerance: the loop is expected to stop
/// because the perimeter is secure or because the direction reversed, not
/// because it ran out of turns.
const MAX_SHIFTS: usize = 11;

/// How far the assumed sensitivity is shrunk each time the same CNEC comes back
/// as the worst one, and the floor it shrinks to.
///
/// Shrinking the sensitivity *grows* the next step, which is the point: a
/// shifter whose true response is weaker than its linear one otherwise creeps
/// toward the limit a fraction of a tap at a time and hits [`MAX_SHIFTS`] still
/// overloaded. Both numbers are the reference's.
const SENSI_UNDERESTIMATOR_STEP: f64 = 0.15;
const SENSI_UNDERESTIMATOR_MIN: f64 = 0.5;

/// Where an automaton that states no speed fires: **first**.
///
/// The reference's `DEFAULT_SPEED`, and it is worth spelling out because the
/// opposite reading is the tempting one. "No speed stated" looks like "no claim
/// to be fast", which argues for firing such an action last so it cannot
/// pre-empt equipment the file actually timed. The reference reads it the other
/// way: an unstated speed is zero, so an untimed automaton goes before
/// everything that named a speed at all.
///
/// It changes answers rather than just ordering. On scenario 1.2.2.4 the
/// untimed `open_be1_be4` opens a Belgian circuit, and the two phase shifters
/// that follow are sized against the network that leaves behind. Fired last
/// instead, they size themselves against an overload the line opening was about
/// to remove and spend four taps where one is enough — with every margin after
/// that correspondingly adrift.
///
/// Actions that all leave the speed out still share one batch, so the rule that
/// a batch samples the grid once is untouched; what moves is where that batch
/// sits relative to the timed ones.
const DEFAULT_SPEED: i64 = 0;

/// An automaton's firing order, with [`DEFAULT_SPEED`] for one that states none.
fn speed_of(speed: Option<i64>) -> i64 {
    speed.unwrap_or(DEFAULT_SPEED)
}

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
    model: FlowModel,
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
        margin(crac, network, resolution, &state, &result.open_branches, &result.transformers, model);
    result.final_margin_mw = result.initial_margin_mw;

    // Where each shifter stood when this perimeter began, which is what a
    // `relativeToPreviousInstant` range is anchored on. For an auto perimeter
    // the previous instant is preventive, and `transformers` arrives as the
    // preventive perimeter left it — so this is measured before any automaton
    // fires and never re-measured, exactly as [`search`](super::search) does it.
    let anchors = super::linear::taps_now(
        crac,
        &Network { transformers, ..*network },
        resolution,
    );

    // Speeds, ascending — see [`DEFAULT_SPEED`] for where an action that states
    // none belongs, which is not where it looks like it belongs.
    let mut speeds: Vec<i64> = Vec::new();
    for action in crac.network_actions.iter().filter(|a| covers(&a.usage_rules, &state)) {
        speeds.push(speed_of(action.speed));
    }
    for action in crac.range_actions.iter().filter(|a| covers(&a.usage_rules, &state)) {
        speeds.push(speed_of(action.speed));
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
            violated(crac, network, resolution, &state, &result.open_branches, &result.transformers, model);
        for (index, action) in crac.network_actions.iter().enumerate() {
            if result.network_actions.contains(&index) {
                continue;
            }
            if speed_of(action.speed) != speed {
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
                violated(crac, network, resolution, &state, &result.open_branches, &result.transformers, model);
            let mut moved = false;
            for (index, action) in crac.range_actions.iter().enumerate() {
                if result.range_actions.iter().any(|(i, _, _)| *i == index) {
                    continue;
                }
                if speed_of(action.speed) != speed {
                    continue;
                }
                if !triggered(&action.usage_rules, &state, &violations) {
                    continue;
                }
                let RangeActionKind::Pst { element, tap_to_angle, initial_tap } = &action.kind
                else {
                    continue;
                };
                let Some(branch) = resolution.branch(element) else { continue };
                let table =
                    tap_table(tap_to_angle, network.tap_changers, branch, network.lines.len());
                // Restricted to what the CRAC actually permits, *before* the
                // shift is sized. An automaton is not exempt from its own
                // range: the reference clamps the computed set-point to
                // `[minAdmissible, maxAdmissible]` and so must this, or a
                // shifter allowed five taps of travel runs to the end of the
                // tap changer instead — 16 where the CRAC said 10, and every
                // margin downstream too good to be true.
                let previous = anchors
                    .iter()
                    .find(|(i, _)| *i == index)
                    .map_or(*initial_tap, |(_, t)| *t);
                let (low, high) = super::linear::tap_bounds(action, &table, *initial_tap, previous);
                let table: Vec<(i32, f64)> =
                    table.into_iter().filter(|(_, a)| *a >= low - 1e-9 && *a <= high + 1e-9).collect();
                // Sized against the CNECs *this* action watches, not the worst
                // in the perimeter. A scheme is wired to a particular circuit;
                // sizing it against somebody else's overload asks it to relieve
                // a flow it may have no influence over at all, and the near-zero
                // sensitivity then makes it decline to act.
                let watched = watched_cnecs(&action.usage_rules, &state, &violations);
                if watched.is_empty() {
                    continue;
                }
                let Some((tap, angle)) = shift_until_secure(
                    crac, network, resolution, &state, &result, branch, &table, &watched, model,
                ) else {
                    continue;
                };
                if let Some(i) = branch.checked_sub(network.lines.len()) {
                    apply_tap(network, &mut result.transformers, i, tap, angle);
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
        margin(crac, network, resolution, &state, &result.open_branches, &result.transformers, model);
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

/// Shift this automaton's set-point until the CNECs it watches are secure.
///
/// Returns the tap it settles on and that tap's angle, or `None` when it does
/// not move at all.
///
/// # Why this iterates
///
/// The shift is sized from a *linear* estimate — margin over sensitivity — and
/// the network it is applied to is not linear. One shot is therefore
/// systematically wrong in whichever direction the curvature runs, and the
/// error is not small: on the reference's own scenario 1.2.2.2 a single shift
/// lands on tap −7 with the watched CNEC still 5 A short of secure, where the
/// answer is −8.
///
/// So it re-measures after every move and shifts again, which is what the
/// reference does. Three things stop it, all of them needed:
///
/// - **Nothing is violated.** The goal is a secure perimeter, not the best
///   margin available. An automaton is protection equipment: it acts until the
///   thing it watches is inside its limit and then it is finished.
/// - **The direction reverses.** A scheme does not hunt. Once the estimate
///   starts asking for a move back the way it came, the previous position was
///   the answer.
/// - **[`MAX_SHIFTS`] iterations.** A guard, not a tolerance.
///
/// # The sensitivity under-estimator
///
/// When the same CNEC comes back as the worst one twice running, the linear
/// estimate is converging too slowly to reach zero — so the assumed sensitivity
/// is shrunk by [`SENSI_UNDERESTIMATOR_STEP`], down to
/// [`SENSI_UNDERESTIMATOR_MIN`], which makes the next step *larger*. Without it
/// a shifter whose true response is weaker than its linear one creeps toward
/// the limit a fraction of a tap at a time and hits the iteration guard while
/// still overloaded. It is the reference's own device, with the reference's own
/// constants.
#[allow(clippy::too_many_arguments)]
fn shift_until_secure(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    state: &State,
    result: &AutomatonResult,
    branch: usize,
    tap_to_angle: &[(i32, f64)],
    watched: &[String],
    model: FlowModel,
) -> Option<(i32, f64)> {
    let start = current_tap(result, branch, network.lines.len(), tap_to_angle)?;
    let angle_of = |tap: i32| tap_to_angle.iter().find(|(t, _)| *t == tap).map(|(_, a)| *a);
    let start_angle = angle_of(start)?;

    let mut transformers = result.transformers.clone();
    let (mut tap, mut angle) = (start, start_angle);
    let mut direction = 0.0f64;
    // CNECs this shifter provably cannot help — the sensitivity is negligible,
    // so no set-point secures them. Excluded rather than given up on, so the
    // loop moves to the next violated CNEC instead of stopping at the worst.
    let mut hopeless: Vec<usize> = Vec::new();
    let mut previous: Option<usize> = None;
    let mut underestimator = 1.0f64;

    for _ in 0..MAX_SHIFTS {
        let view = Network {
            generation: network.generation,
            buses: network.buses,
            lines: network.lines,
            transformers: &transformers,
            branch_ids: network.branch_ids,
            bus_ids: network.bus_ids,
            initially_open: network.initially_open,
            bus_countries: network.bus_countries,
            shunts: network.shunts,
            tap_changers: network.tap_changers,
            base_mva: network.base_mva,
        };
        let assessment = measure(crac, &view, resolution, &result.open_branches, model);
        let perimeter = assessment.perimeters.iter().find(|p| p.state == *state)?;

        // The worst violated CNEC this automaton watches, ranked in the unit
        // the flow model implies — megawatts under DC, amperes under AC. Not
        // cosmetic: two CNECs at different voltages order differently in the
        // two units, so the unit decides which one the shift is sized against.
        let Some(worst) = perimeter
            .cnecs
            .iter()
            .filter(|c| watched.iter().any(|v| *v == crac.flow_cnecs[c.cnec].id))
            .filter(|c| !hopeless.contains(&c.cnec))
            .filter(|c| c.is_violated())
            .min_by(|a, b| overload(a, model).total_cmp(&overload(b, model)))
        else {
            break;
        };

        underestimator = if previous == Some(worst.cnec) {
            (underestimator - SENSI_UNDERESTIMATOR_STEP).max(SENSI_UNDERESTIMATOR_MIN)
        } else {
            1.0
        };
        previous = Some(worst.cnec);

        let options = DcOptions::default();
        let branches = dc_branches(network.lines, &transformers, options);
        let Some(sensitivity) = DcSensitivity::new(
            network.buses,
            &branches,
            network.lines.len() + transformers.len(),
        ) else {
            break;
        };
        let Some(column) = phase_shift_sensitivity(&sensitivity, &branches, branch) else { break };
        // MW per degree of the CRAC's angle — the same sign convention the
        // linear optimizer uses, and for the same reason.
        let sigma = -column.get(worst.branch).copied().unwrap_or(0.0)
            * network.base_mva
            * std::f64::consts::PI
            / 180.0
            * underestimator;
        if sigma.abs() < 1e-6 {
            hopeless.push(worst.cnec);
            previous = None;
            continue;
        }

        // Just far enough to bring the flow back to the limit. Margin and
        // sensitivity are both in megawatts, so the ratio is the same number in
        // either unit — only the choice of CNEC above depended on that.
        let needed = worst.flow_mw.signum() * worst.margin_mw.min(0.0) / sigma;
        let target = angle + needed;
        let Some(next) = round_away(tap_to_angle, angle, target) else { break };

        let step = (next.1 - angle).signum() * f64::from(u8::from((next.1 - angle).abs() > 1e-9));
        // At a bound, or asked to turn round. Either way the position it is
        // already on is the answer.
        if step == 0.0 || (direction != 0.0 && step != direction) {
            break;
        }
        direction = step;

        if let Some(i) = branch.checked_sub(network.lines.len()) {
            apply_tap(network, &mut transformers, i, next.0, next.1);
        }
        (tap, angle) = next;
    }

    if tap == start {
        return None;
    }
    // The estimate got it into the neighbourhood; this settles which tap.
    Some(smallest_securing(
        crac, network, resolution, state, result, branch, tap_to_angle, watched, &hopeless, model,
        start, tap,
    ))
}

/// The tap nearest `start` that secures every watched CNEC, searching outward
/// towards `reached`.
///
/// # Why the shift is not simply believed
///
/// The reference computes the set-point that puts the worst margin at exactly
/// zero, rounds to the first tap beyond it, and stops. That is the *smallest*
/// set-point which secures the CNEC — and it is only the smallest because the
/// sensitivity it divides by is the true one, taken from the same analysis that
/// measures the flow.
///
/// gridoxide's gradient is the DC phase-shift sensitivity, and under an AC flow
/// model it is not that number: on scenario 1.2.2.2 it reports 5.44 MW per
/// degree where the network delivers 8.8. The direction is right — a DC
/// sensitivity gets a phase shifter's sign and rough size right — but the
/// distance is 60% too far, and rounding *away* then turns that into tap −11
/// where −8 secures the circuit. Believing an approximate gradient to the tap
/// is the one thing this algorithm cannot afford, because there is no
/// keep-it-only-if-it-improved filter behind it: whatever it lands on is the
/// answer.
///
/// So the specification is reproduced rather than the arithmetic. "Shift until
/// the CNECs are secure, and no further" is a statement about measured flows,
/// and measuring is what makes it independent of how good the gradient was.
/// The scan runs from `start` outward, so the first tap that secures everything
/// wins; when none does, the furthest reached stands, which is the honest
/// answer to "this scheme cannot save the circuit".
#[allow(clippy::too_many_arguments)]
fn smallest_securing(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    state: &State,
    result: &AutomatonResult,
    branch: usize,
    tap_to_angle: &[(i32, f64)],
    watched: &[String],
    hopeless: &[usize],
    model: FlowModel,
    start: i32,
    reached: i32,
) -> (i32, f64) {
    let angle_of = |tap: i32| tap_to_angle.iter().find(|(t, _)| *t == tap).map(|(_, a)| *a);
    let reached_angle = angle_of(reached).unwrap_or_default();
    let Some(i) = branch.checked_sub(network.lines.len()) else { return (reached, reached_angle) };

    // Every tap strictly between the two, ordered from `start` outward, then
    // the one actually reached. Ordered by tap number rather than by angle
    // because a tap changer's map need not be monotone.
    let mut between: Vec<i32> = tap_to_angle
        .iter()
        .map(|(t, _)| *t)
        .filter(|t| if reached > start { *t > start && *t < reached } else { *t < start && *t > reached })
        .collect();
    between.sort_by_key(|t| (t - start).abs());

    for tap in between {
        let Some(angle) = angle_of(tap) else { continue };
        let mut transformers = result.transformers.clone();
        apply_tap(network, &mut transformers, i, tap, angle);
        let view = Network {
            generation: network.generation,
            buses: network.buses,
            lines: network.lines,
            transformers: &transformers,
            branch_ids: network.branch_ids,
            bus_ids: network.bus_ids,
            initially_open: network.initially_open,
            bus_countries: network.bus_countries,
            shunts: network.shunts,
            tap_changers: network.tap_changers,
            base_mva: network.base_mva,
        };
        let secure = measure(crac, &view, resolution, &result.open_branches, model)
            .perimeters
            .iter()
            .find(|p| p.state == *state)
            .is_some_and(|p| {
                p.cnecs
                    .iter()
                    .filter(|c| watched.iter().any(|v| *v == crac.flow_cnecs[c.cnec].id))
                    // The same exclusions the shift itself made. A CNEC this
                    // shifter has no influence over cannot decide how far it
                    // travels, or one overload nothing can fix would drive the
                    // machine to the end of its range for nothing.
                    .filter(|c| !hopeless.contains(&c.cnec))
                    .all(|c| !c.is_violated())
            });
        if secure {
            return (tap, angle);
        }
    }
    (reached, reached_angle)
}

/// How far past its limit a CNEC is, in the unit the flow model implies.
///
/// `RaoUtil.getFlowUnit`: megawatts for a DC load flow, amperes for an AC one.
fn overload(cnec: &CnecResult, model: FlowModel) -> f64 {
    match model {
        FlowModel::Ac => cnec.margin_a,
        FlowModel::Dc => cnec.margin_mw,
    }
}

/// The first tap at or beyond `target`, travelling away from `from`.
///
/// **Away**, not nearest. A shift sized to put the margin at exactly zero lands
/// between two taps, and the nearer one is on the wrong side of it about half
/// the time — which for protection equipment means leaving the circuit
/// overloaded. The reference's `roundUpAngleToTapWrtInitialSetpoint` rounds the
/// same way and for the same reason. `tap_to_angle` is already restricted to
/// what the action's range permits, so running out of table is running out of
/// range: the furthest permitted tap in that direction is the answer, which is
/// what a real scheme does rather than declining to act.
fn round_away(tap_to_angle: &[(i32, f64)], from: f64, target: f64) -> Option<(i32, f64)> {
    let forward = target >= from;
    let onward = tap_to_angle
        .iter()
        .filter(|(_, a)| if forward { *a >= from } else { *a <= from });
    onward
        .clone()
        .filter(|(_, a)| if forward { *a >= target } else { *a <= target })
        .min_by(|a, b| (a.1 - from).abs().total_cmp(&(b.1 - from).abs()))
        .or_else(|| onward.max_by(|a, b| (a.1 - from).abs().total_cmp(&(b.1 - from).abs())))
        .copied()
}

/// Put one transformer on a tap, preferring the network's own step.
///
/// See the note in [`linear::apply`](super::linear): a tap changer is
/// equipment, a CRAC's table describes it, and the equipment wins when they
/// disagree.
fn apply_tap(
    network: &Network<'_>,
    transformers: &mut [Transformer],
    index: usize,
    tap: i32,
    angle: f64,
) {
    let from_network =
        network.tap_changers.get(index).and_then(|c| c.as_ref()).and_then(|c| c.at(tap));
    if let Some(t) = transformers.get_mut(index) {
        let ratio = t.tap.norm();
        t.tap = match from_network {
            Some(step) => num_complex::Complex::from_polar(ratio, step.arg()),
            None => num_complex::Complex::from_polar(ratio, (-angle).to_radians()),
        };
    }
}

/// Evaluate with the flow model the run is using, as
/// [`search`](super::search) does.
fn measure(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    open: &[usize],
    model: FlowModel,
) -> super::evaluate::SecurityResult {
    let ac = super::evaluate::ac_options(network);
    evaluate_model(crac, network, resolution, open, model, &ac)
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
    model: FlowModel,
) -> Vec<String> {
    let view = Network {
        generation: network.generation,
        buses: network.buses,
        lines: network.lines,
        transformers,
        branch_ids: network.branch_ids,
        bus_ids: network.bus_ids,
        initially_open: network.initially_open,
        bus_countries: network.bus_countries,
        shunts: network.shunts,
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };
    measure(crac, &view, resolution, open, model)
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
    model: FlowModel,
) -> f64 {
    let view = Network {
        generation: network.generation,
        buses: network.buses,
        lines: network.lines,
        transformers,
        branch_ids: network.branch_ids,
        bus_ids: network.bus_ids,
        initially_open: network.initially_open,
        bus_countries: network.bus_countries,
        shunts: network.shunts,
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };
    measure(crac, &view, resolution, open, model)
        .perimeters
        .iter()
        .find(|p| p.state == *state)
        .and_then(|p| p.min_margin())
        .unwrap_or(f64::INFINITY)
}
