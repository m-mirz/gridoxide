//! The search tree over network actions.
//!
//! Range actions have a degree of freedom and can be optimized by an LP.
//! Network actions cannot: a switch is open or closed, and there is no gradient
//! between the two. So the discrete half is *searched* — greedily, one action
//! at a time, keeping whichever candidate improves the objective most.
//!
//! ```text
//! evaluate(root); optimize_range_actions(root)
//! for depth in 0..max_depth:
//!     for each available action not yet applied:
//!         leaf = root + action
//!         evaluate(leaf)
//!         optimize_range_actions(leaf)      <- the full LP, every leaf
//!     keep the best leaf if it improved enough; else stop
//! ```
//!
//! # Every leaf re-runs the linear optimization, and that is the point
//!
//! It is also where all the time goes, which makes it tempting to skip. Do not:
//! a topological action changes the sensitivities the phase shifters are
//! optimized against, so choosing the topology first and the set-points
//! afterwards gives a worse answer than choosing them together. An action that
//! looks poor on its own may be the best one once the shifters are re-tuned
//! around it.
//!
//! # Determinism under parallelism
//!
//! Candidates are evaluated in a fixed order and ties are broken on the action
//! id, so a run reproduces itself. This matters more than it sounds: a search
//! tree that returns a different answer each time cannot be regression-tested,
//! and an operator cannot be told why yesterday's study disagreed with today's.

use crate::opf::Solver;

use super::crac::{Crac, ElementaryAction, NetworkAction, State};
use super::evaluate::{Network, Resolution};
use super::linear::{optimize, LinearOptions, NetworkMut, Setpoint};
use super::limits::Limits;
use super::usage::Constrained;

#[derive(Clone, Debug)]
pub struct SearchOptions {
    /// How many network actions may be stacked. `0` optimizes range actions
    /// only.
    pub max_depth: usize,
    /// A candidate must improve the objective by at least this much, in MW, to
    /// be taken.
    pub absolute_min_impact: f64,
    /// …and by at least this fraction of the current cost. Both apply; the
    /// stricter binds.
    pub relative_min_impact: f64,
    /// Options handed to the per-leaf linear optimization.
    pub linear: LinearOptions,
    /// How far a **curative** perimeter must beat the preventive one before it
    /// stops looking.
    ///
    /// Read by [`castor::run`](super::castor::run), not here: this layer only
    /// ever sees the resulting [`stop_at_target`](Self::stop_at_target). The
    /// reference's default is **zero** — a curative perimeter stops the moment
    /// it is better than preventive — and its own test configurations set 500,
    /// 628 or 10000 to move that line.
    pub curative_min_obj_improvement: f64,
    /// Whether a curative perimeter must also end secure before it may stop.
    ///
    /// `objective-function.enforce-curative-security`. Tightens the target to
    /// at least zero, so "better than preventive" alone is not enough.
    pub enforce_curative_security: bool,
    /// Discard candidate actions that are too far, in country boundaries, from
    /// the most limiting element. `None` disables the filter; `Some(0)` keeps
    /// only actions in the same country as the worst CNEC.
    ///
    /// The reference's `skip-actions-far-from-most-limiting-element` together
    /// with `max-number-of-boundaries-for-skipping-actions`. It exists because
    /// an operator in one control area cannot generally be asked to act for an
    /// overload in another, and it changes the answer: without it the search
    /// takes actions the reference never offers itself and reports a better
    /// margin than the problem actually allows.
    ///
    /// Needs [`Network::bus_countries`]; with no countries there is no notion
    /// of far, and the filter passes everything.
    pub skip_far_actions: Option<usize>,
    /// Stop as soon as the objective reaches this value instead of maximizing.
    /// `None` maximizes.
    ///
    /// The reference's `SECURE_FLOW` objective, which
    /// `TreeParameters.buildForPreventivePerimeter` turns into
    /// `AT_TARGET_OBJECTIVE_VALUE` with a target of zero: **secure is enough**.
    /// A network that already has a positive margin gets no remedial action at
    /// all, and a candidate that reaches security is taken whether or not it
    /// clears the minimum-impact thresholds.
    ///
    /// This is not a weaker version of maximizing. Twelve of the reference's
    /// own AC scenarios use it, and one of them asserts that **zero** actions
    /// are used on a network gridoxide would happily have improved — spending
    /// remedial actions to gain margin nobody asked for is a different answer,
    /// not a better one. Zero is expressed in the objective's unit, where it
    /// means the same thing either way.
    pub stop_at_target: Option<f64>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_depth: 2,
            absolute_min_impact: 0.0,
            relative_min_impact: 0.0,
            linear: LinearOptions::default(),
            curative_min_obj_improvement: 0.0,
            enforce_curative_security: false,
            skip_far_actions: None,
            stop_at_target: None,
        }
    }
}

/// What the search settled on.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Indices into [`Crac::network_actions`], in the order they were chosen.
    pub network_actions: Vec<usize>,
    /// Range-action set-points at the winning leaf.
    pub setpoints: Vec<Setpoint>,
    pub initial_margin_mw: f64,
    pub final_margin_mw: f64,
    /// The winning leaf's objective — the minimum margin in
    /// [`ObjectiveUnit`](super::linear::ObjectiveUnit), less what the monitored
    /// CNECs cost.
    ///
    /// Different from [`final_margin_mw`](Self::final_margin_mw) and reported
    /// separately for the same reason the leaf keeps both: a margin is what a
    /// plan is read as, and this is what one perimeter's answer is *compared*
    /// against. [`castor`](super::castor) needs exactly this to set a curative
    /// perimeter's stop criterion, which the reference states relative to the
    /// preventive perimeter's own objective.
    pub final_objective: f64,
    /// Leaves evaluated, root excluded. The cost of the search, and the number
    /// to watch when a case gets big.
    pub leaves: usize,
    /// Depth actually reached.
    pub depth: usize,
    /// Branches the chosen actions leave open, flat indices. Derivable from
    /// `network_actions`, and returned because every consumer needs it and
    /// re-deriving it is where the caller's idea of the winning network drifts
    /// from the searcher's.
    pub open_branches: Vec<usize>,
    /// The buses at the winning leaf, redispatch included. Together with
    /// `transformers` and `open_branches` this is the winning network — and a
    /// redispatch lives *only* here, so a consumer that carries the
    /// transformers and forgets the buses reproduces a network in which no
    /// injection ever moved.
    pub buses: Vec<crate::types::Bus>,
    /// The transformers at the winning leaf, tap movements included.
    ///
    /// Returned rather than applied in place. `optimize` mutates because it
    /// explores one path; a search explores many, and leaving the caller's
    /// network at whichever leaf happened to be evaluated last would be worse
    /// than useless. Together with `open_branches` this is the winning network.
    pub transformers: Vec<crate::types::Transformer>,
}

impl SearchResult {
    pub fn improvement(&self) -> f64 {
        self.final_margin_mw - self.initial_margin_mw
    }

    pub fn is_secure(&self) -> bool {
        self.final_margin_mw >= 0.0
    }
}

/// A network action's effect, reduced to what the evaluator can apply.
///
/// Actions this cannot express are **rejected outright** rather than applied
/// partially. A network action is a set of elementary actions taken *together*
/// — "split this busbar" is one decision, not six — so applying half of one
/// produces a network the CRAC never described, and an answer that references
/// an action whose effect was not what the optimizer measured.
struct Effect {
    /// Branches to open, flat indices.
    open: Vec<usize>,
    /// Branches to close — removed from the open set. A standby circuit the
    /// network file marks out of service is closable precisely because the
    /// importer keeps it.
    close: Vec<usize>,
    /// `(branch, tap position, angle in degrees)` for a phase-shifter
    /// set-point.
    ///
    /// Both the position and the angle are kept because applying one is not the
    /// same as applying the other: the network's own tap changer is the
    /// authority where it has the step, and the angle is the fallback for a
    /// shifter only the CRAC describes. That is the precedence
    /// [`linear::apply`](super::linear) already uses, and splitting it would
    /// let the search evaluate a tap the plan then reports differently.
    taps: Vec<(usize, i32, f64)>,
}

/// Reduce a network action, or refuse it.
fn effect_of(
    action: &NetworkAction,
    crac: &Crac,
    resolution: &Resolution,
    tap_changers: &[Option<crate::types::TapChanger>],
    lines: usize,
) -> Option<Effect> {
    let mut effect = Effect { open: Vec::new(), close: Vec::new(), taps: Vec::new() };
    for elementary in &action.elementary {
        match elementary {
            ElementaryAction::TerminalsConnection { element, connected } => {
                let branch = resolution.branch(element)?;
                if *connected {
                    effect.close.push(branch);
                } else {
                    effect.open.push(branch);
                }
            }
            ElementaryAction::Switch { element, open } => {
                let branch = resolution.branch(element)?;
                if *open {
                    effect.open.push(branch);
                } else {
                    effect.close.push(branch);
                }
            }
            ElementaryAction::PstTapPosition { element, tap } => {
                let branch = resolution.branch(element)?;
                if branch < lines {
                    return None;
                }
                // The tap table comes from the network where it has one, and
                // from a range action describing the same shifter otherwise —
                // the precedence `linear::tap_table` already establishes, and
                // for the same reason: the network's `##R` record is the
                // machine, a CRAC's table is a description of it, and where
                // they disagree the machine wins.
                //
                // Consulting only the range actions is not a smaller version of
                // this rule, it is a different one. A CRAC may declare a PST
                // *set-point network action* and no PST *range action* at all —
                // 5 of the reference's own AC scenarios do — and there the
                // range-action lookup finds nothing and the whole action
                // becomes inexpressible, so the search silently never offers
                // the one thing the scenario is about.
                let declared: Vec<(i32, f64)> = crac
                    .range_actions
                    .iter()
                    .find_map(|r| match &r.kind {
                        super::crac::RangeActionKind::Pst { element: e, tap_to_angle, .. }
                            if e == element =>
                        {
                            Some(tap_to_angle.clone())
                        }
                        _ => None,
                    })
                    .unwrap_or_default();
                let table =
                    super::linear::tap_table(&declared, tap_changers, branch, lines);
                let angle = table.iter().find(|(t, _)| t == tap).map(|(_, a)| *a)?;
                effect.taps.push((branch, *tap, angle));
            }
            // Injection and shunt changes need a mutable bus vector, which this
            // layer does not carry; a switch pair needs a closable switch.
            ElementaryAction::GeneratorSetpoint { .. }
            | ElementaryAction::LoadSetpoint { .. }
            | ElementaryAction::ShuntSection { .. }
            | ElementaryAction::SwitchPair { .. } => return None,
        }
    }
    (!effect.open.is_empty() || !effect.close.is_empty() || !effect.taps.is_empty())
        .then_some(effect)
}

/// Whether two actions can be taken together.
///
/// Two actions touching the same network element conflict: one may open a line
/// the other moves, and the result depends on the order, which a set has none
/// of.
fn compatible(a: &NetworkAction, b: &NetworkAction) -> bool {
    let mine: Vec<&str> = a.elementary.iter().flat_map(ElementaryAction::elements).collect();
    !b.elementary
        .iter()
        .flat_map(ElementaryAction::elements)
        .any(|e| mine.contains(&e))
}

/// Search for a set of network actions, optimizing range actions at every leaf.
///
/// `transformers` is mutated to the winning leaf's state, matching
/// [`optimize`]'s contract: the caller's network ends up where the result says.
///
/// Starts from the network's own out-of-service circuits. A perimeter that
/// begins somewhere else — a curative one, carrying the preventive stage's and
/// the automatons' decisions — wants [`search_with_open`] instead.
pub fn search(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    solver: &mut dyn Solver,
    options: &SearchOptions,
) -> SearchResult {
    search_with_open(crac, network, resolution, perimeter, solver, options, network.initially_open)
}

/// [`search`], starting from a stated open set rather than the network's own.
///
/// # Why this is a parameter and not a pre-modified network
///
/// The obvious way to say "these branches are already open" is to write
/// [`OPEN_BRANCH_Z`](crate::topology::reduction::OPEN_BRANCH_Z) into a copy of
/// the lines before searching. That is what this used to do, and it made a
/// **closing** remedial action structurally impossible: a close is expressed by
/// removing a branch from the open set, so with the set emptied into the
/// impedances there is nothing left to remove and the branch stays at 1e9 Ω
/// whatever the action says.
///
/// The failure is silent in the worst way. The action is not refused — it
/// evaluates as a change that does nothing, loses to every other candidate, and
/// the perimeter reports that no remedial action was worth taking. Against the
/// reference's own suite that was 21 curative closes declined out of 21 offered,
/// while the same actions matched 8 of 8 in preventive and 3 of 3 in auto,
/// which is where the asymmetry gave it away: [`automaton`](super::automaton)
/// carries the set as a list and retains against it, exactly as this now does.
///
/// So the set stays a list all the way down to
/// [`optimize_with_open`], which knows how to apply it — including stating it
/// as outages so the AC path removes those branches from the Y-bus rather than
/// leaving them coupled through a ~1e-9 admittance.
pub fn search_with_open(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    solver: &mut dyn Solver,
    options: &SearchOptions,
    already_open: &[usize],
) -> SearchResult {
    // The root: no network action, range actions untouched.
    let base_open: Vec<usize> = already_open.to_vec();

    // Which CNECs are constrained *here*, before this perimeter acts. A usage
    // rule conditional on a CNEC is answered against this and never
    // re-answered, so an action authorized by an overload keeps its authority
    // even once another action has relieved it — the reference's own scenario
    // 2.4.1.2 is named for that ("no reevaluation"). See `super::usage`.
    let starting = measure_with(crac, network, resolution, &base_open, options.linear.flow_model);
    let branches = crate::linear::btheta::dc_branches(
        network.lines,
        network.transformers,
        crate::linear::DcOptions::default(),
    );
    let constrained = Constrained::from_result(
        &starting,
        options.linear.objective_unit,
        &super::usage::branch_countries(crac, network, resolution, &branches),
    );
    // Every leaf is optimized against the same candidate set, so the filter
    // travels with the options rather than being re-derived per leaf.
    let mut linear = options.linear.clone();
    linear.available = constrained.clone();
    let options = &SearchOptions { linear, ..options.clone() };

    let available: Vec<(usize, Effect)> = crac
        .network_actions
        .iter()
        .filter(|action| constrained.allows(&action.usage_rules, perimeter, crac))
        .filter_map(|action| {
            let index = crac.network_actions.iter().position(|a| a.id == action.id)?;
            let effect =
                effect_of(action, crac, resolution, network.tap_changers, network.lines.len())?;
            Some((index, effect))
        })
        .collect();
    let available = match options.skip_far_actions {
        Some(max) => {
            near_most_limiting(
                crac,
                network,
                resolution,
                perimeter,
                &base_open,
                available,
                max,
                options.linear.flow_model,
            )
        }
        None => available,
    };
    // What the CRAC allows in this perimeter's own instant. A perimeter spans
    // one optimization instant — castor gives the preventive perimeter the
    // preventive state and every curative perimeter a single curative one — so
    // the earliest instant present is that instant, and an outage or
    // pulled-forward state riding along does not bring its own allowance.
    let limits = perimeter
        .iter()
        .map(|s| s.instant)
        .min()
        .map_or_else(Limits::unlimited, |i| Limits::of_instant(crac, i));
    // Each leaf is optimized against what its own network actions leave over,
    // so the budget is rebuilt per candidate rather than shared.
    let budgeted = |chosen: &[usize]| LinearOptions {
        limits: limits.remaining(crac, chosen),
        ..options.linear.clone()
    };

    // Both from the same assessment the availability filter used, rather than
    // from a second one of a network it has already measured — under an AC flow
    // model that is a Newton-Raphson solve per state.
    let initial = worst_of(crac, &starting, perimeter);
    let untouched = objective_of(crac, &starting, perimeter, &options.linear);

    let reached = |objective: f64| options.stop_at_target.is_some_and(|t| objective >= t);

    // The target may be met before anything at all is done, and then the
    // perimeter is left completely alone — not even its range actions are
    // optimized. The reference is explicit about this: `SearchTree.run` tests
    // the stop criterion on the *evaluated* root leaf and returns there,
    // *before* `optimizeLeaf` is called on it. It is the difference between "no
    // network action was worth taking" and "nothing at all was needed", and on
    // scenario 1.3.9.1 it is the difference between two curative remedial
    // actions and none.
    if reached(untouched) {
        return SearchResult {
            network_actions: Vec::new(),
            setpoints: Vec::new(),
            initial_margin_mw: initial,
            final_margin_mw: initial,
            final_objective: untouched,
            leaves: 0,
            depth: 0,
            open_branches: base_open,
            buses: network.buses.to_vec(),
            transformers: network.transformers.to_vec(),
        };
    }

    let root = leaf(
        crac,
        network,
        resolution,
        perimeter,
        solver,
        &budgeted(&[]),
        Applied { open: &base_open, taps: &[] },
    );

    let mut chosen: Vec<usize> = Vec::new();
    let mut best = root;
    let mut leaves = 0usize;
    let mut depth = 0usize;

    for _ in 0..options.max_depth {
        // Already good enough. Checked before the first depth as well as
        // between them, so an untouched network that is already secure is left
        // untouched rather than improved.
        if reached(best.objective) {
            break;
        }
        let mut open_here: Vec<usize> = base_open.clone();
        let mut taps_here: Vec<(usize, i32, f64)> = Vec::new();
        for &index in &chosen {
            if let Some((_, effect)) = available.iter().find(|(i, _)| *i == index) {
                open_here.extend(&effect.open);
                open_here.retain(|b| !effect.close.contains(b));
                taps_here.extend(&effect.taps);
            }
        }

        let mut winner: Option<(usize, Evaluated)> = None;
        // A fixed order, so the search reproduces itself.
        for (index, effect) in &available {
            if chosen.contains(index) {
                continue;
            }
            if chosen.iter().any(|&c| {
                !compatible(&crac.network_actions[c], &crac.network_actions[*index])
            }) {
                continue;
            }
            // More than the plan's allowance permits. Checked before the leaf
            // is evaluated rather than after: a candidate the CRAC forbids is
            // not a worse answer, it is not an answer.
            let mut with_candidate = chosen.clone();
            with_candidate.push(*index);
            if !limits.admits(crac, &with_candidate) {
                continue;
            }
            let mut open = open_here.clone();
            open.extend(&effect.open);
            open.retain(|b| !effect.close.contains(b));
            let mut taps = taps_here.clone();
            taps.extend(&effect.taps);

            let candidate = leaf(
                crac,
                network,
                resolution,
                perimeter,
                solver,
                &budgeted(&with_candidate),
                Applied { open: &open, taps: &taps },
            );
            leaves += 1;

            // Ranked on the objective, not on the reported margin: in an
            // ampere objective those can order two candidates differently.
            let better = match &winner {
                None => true,
                Some((previous_index, previous)) => {
                    candidate.objective > previous.objective + 1e-12
                        || ((candidate.objective - previous.objective).abs() <= 1e-12
                            && crac.network_actions[*index].id
                                < crac.network_actions[*previous_index].id)
                }
            };
            if better {
                winner = Some((*index, candidate));
            }
        }

        let Some((index, candidate)) = winner else { break };
        // A candidate that reaches the target is taken even if it falls short
        // of the minimum-impact thresholds: those exist to stop the search
        // spending an action for nothing, and securing the network is not
        // nothing.
        let enough = improved_enough(best.objective, candidate.objective, options)
            || (candidate.objective > best.objective && reached(candidate.objective));
        if !enough {
            break;
        }
        chosen.push(index);
        best = candidate;
        depth += 1;
    }

    let mut open_branches: Vec<usize> = base_open.clone();
    for &index in &chosen {
        if let Some((_, effect)) = available.iter().find(|(i, _)| *i == index) {
            open_branches.extend(&effect.open);
            open_branches.retain(|b| !effect.close.contains(b));
        }
    }
    open_branches.sort_unstable();
    open_branches.dedup();

    SearchResult {
        network_actions: chosen,
        setpoints: best.setpoints,
        initial_margin_mw: initial,
        // Megawatts, whatever the objective was maximized in — the search
        // reports a margin, it does not report its own scoring function.
        final_margin_mw: best.margin_mw,
        final_objective: best.objective,
        leaves,
        depth,
        open_branches,
        transformers: best.transformers,
        buses: best.buses,
    }
}

/// Whether a candidate beats the incumbent by enough to be worth taking.
///
/// Both thresholds apply and the stricter binds. Without them a search happily
/// spends a remedial action for a rounding error's worth of margin, which is an
/// answer no control room would carry out.
fn improved_enough(current: f64, candidate: f64, options: &SearchOptions) -> bool {
    if candidate <= current + options.absolute_min_impact {
        return false;
    }
    if options.relative_min_impact > 0.0 {
        let scale = current.abs().max(1.0);
        if candidate - current < options.relative_min_impact * scale {
            return false;
        }
    }
    true
}

/// Evaluate one leaf: apply `open`, optimize the range actions, report the
/// margin and the network it left behind.
fn leaf(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    solver: &mut dyn Solver,
    options: &LinearOptions,
    applied: Applied<'_>,
) -> Evaluated {
    let mut transformers = network.transformers.to_vec();
    // A phase-shifter set-point is part of the candidate, so it has to be in
    // force before the leaf is optimized *and* before it is scored. Leaving it
    // out does not make the action fail loudly — it makes it evaluate as a
    // change that does nothing, so the search can never prefer it and would
    // misreport the network if it ever did.
    apply_taps(&mut transformers, network.tap_changers, network.lines.len(), applied.taps);
    // The leaf gets its own copy of the buses as well as the transformers: a
    // redispatch moves injections, and a candidate that is evaluated and
    // discarded must not leave them moved.
    let mut buses = network.buses.to_vec();
    let mut mutable = NetworkMut {
        buses: &mut buses,
        lines: network.lines,
        transformers: &mut transformers,
        branch_ids: network.branch_ids,
        bus_ids: network.bus_ids,
        // The candidate's own open set, not the file's. `optimize_with_open`
        // overrides this whenever the set is non-empty, so the only path this
        // field decides is the empty one — and an empty set is a *result* here,
        // not an absence: a closing remedial action that shuts the network's
        // last out-of-service circuit produces exactly it. Handing the file's
        // list over at that point would re-open the branch the action just
        // closed, and every margin downstream would stay self-consistent while
        // describing a network in which nothing was closed.
        initially_open: applied.open,
        bus_countries: network.bus_countries,
        shunts: network.shunts,
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };
    let (objective, margin_mw, setpoints) =
        optimize_with_open(crac, &mut mutable, resolution, perimeter, solver, options, applied.open);
    Evaluated { objective, margin_mw, setpoints, transformers, buses }
}

/// One leaf's outcome.
///
/// `objective` is what candidates are ranked by and what a minimum-impact
/// threshold is compared against; `margin_mw` is what gets reported. They are
/// the same number only when the objective is already in megawatts.
struct Evaluated {
    objective: f64,
    margin_mw: f64,
    setpoints: Vec<Setpoint>,
    transformers: Vec<crate::types::Transformer>,
    buses: Vec<crate::types::Bus>,
}

/// The network state one candidate puts in force: which branches are open, and
/// where the phase shifters sit.
///
/// The two travel together because they have to be applied together — a leaf
/// optimized with one and scored with the other is measuring a network no
/// decision produced.
#[derive(Clone, Copy)]
struct Applied<'a> {
    open: &'a [usize],
    taps: &'a [(usize, i32, f64)],
}

/// Put phase shifters on the tap positions a candidate names.
///
/// The network's own tap changer is the authority where it has the step —
/// `linear::apply`'s rule, and powsybl's: a `PstRangeAction` converts a
/// set-point to a position and lets the equipment decide the angle. The CRAC's
/// angle is the fallback for a shifter the network does not describe.
fn apply_taps(
    transformers: &mut [crate::types::Transformer],
    tap_changers: &[Option<crate::types::TapChanger>],
    lines: usize,
    taps: &[(usize, i32, f64)],
) {
    for &(branch, tap, angle_deg) in taps {
        let Some(i) = branch.checked_sub(lines) else { continue };
        let from_network = tap_changers.get(i).and_then(|c| c.as_ref()).and_then(|c| c.at(tap));
        if let Some(transformer) = transformers.get_mut(i) {
            // The magnitude is the transformer's own ratio and is not the
            // shifter's to change; only the angle moves.
            let ratio = transformer.tap.norm();
            transformer.tap = match from_network {
                Some(step) => num_complex::Complex::from_polar(ratio, step.arg()),
                None => num_complex::Complex::from_polar(ratio, angle_deg.to_radians()),
            };
        }
    }
}

/// [`optimize`] against a network with branches already open.
///
/// The open set has to reach both the optimization and the measurement, or the
/// leaf is optimized for one network and scored on another.
fn optimize_with_open(
    crac: &Crac,
    network: &mut NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    solver: &mut dyn Solver,
    options: &LinearOptions,
    open: &[usize],
) -> (f64, f64, Vec<Setpoint>) {
    if open.is_empty() {
        let result = optimize(crac, network, resolution, perimeter, solver, options);
        return (result.final_objective, result.final_margin_mw, result.setpoints);
    }
    // Opening a branch is expressed by removing it from the working copy's
    // line list — a line with no admittance carries no flow and contributes
    // nothing, which is exactly what "open" means. Indices are preserved by
    // zeroing rather than deleting.
    let mut lines = network.lines.to_vec();
    let mut transformers = network.transformers.clone();
    for &branch in open {
        if branch < lines.len() {
            lines[branch].r = crate::topology::reduction::OPEN_BRANCH_Z;
            lines[branch].x = crate::topology::reduction::OPEN_BRANCH_Z;
            lines[branch].b_shunt = 0.0;
            lines[branch].g_shunt = 0.0;
        } else if let Some(t) = transformers.get_mut(branch - lines.len()) {
            t.from_status = 0;
            t.to_status = 0;
        }
    }
    let mut inner = NetworkMut {
        buses: network.buses,
        lines: &lines,
        transformers: &mut transformers,
        branch_ids: network.branch_ids,
        bus_ids: network.bus_ids,
        // The *effective* open set, not the file's. Two reasons, and they pull
        // in the same direction.
        //
        // The file's own list would re-open branches a *closing* remedial
        // action has just shut, silently measuring a network in which the
        // automaton never acted. And an empty list is not equivalent to this
        // one either: the zeroing above is what makes a branch non-conducting
        // for the *linear* model, where `OPEN_BRANCH_Z` is genuinely open, but
        // in AC it leaves the bus coupled through a ~1e-9 admittance rather
        // than not at all. That is a nearly singular Jacobian, and the answer
        // it converges to is wrong rather than absent — the FR1-FR2 plus
        // FR1-FR3 combination came out at 234 A against a direct evaluation's
        // 1204. Stating the outages lets the AC path remove them from the
        // Y-bus properly; re-removing an already-zeroed branch costs the DC
        // path nothing, because a branch with no admittance carries no flow to
        // redistribute.
        initially_open: open,
        bus_countries: network.bus_countries,
        shunts: network.shunts,
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };
    let result = optimize(crac, &mut inner, resolution, perimeter, solver, options);
    // Carry any tap movement back to the caller's transformers.
    for (a, b) in network.transformers.iter_mut().zip(transformers.iter()) {
        a.tap = b.tap;
    }
    (result.final_objective, result.final_margin_mw, result.setpoints)
}

/// Evaluate with the given flow model, taking the AC options from the network.
fn measure_with(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    open: &[usize],
    model: super::evaluate::FlowModel,
) -> super::evaluate::SecurityResult {
    let ac = super::evaluate::AcOptions { shunts: network.shunts, ..Default::default() };
    super::evaluate::evaluate_model(crac, network, resolution, open, model, &ac)
}

/// What the search maximizes, read off an assessment already made: the worst
/// optimized margin in the objective's unit, less what the monitored CNECs
/// cost. Mirrors `linear::objective`, which computes the same thing from a
/// network rather than from a result.
fn objective_of(
    crac: &Crac,
    result: &super::evaluate::SecurityResult,
    perimeter: &[State],
    options: &LinearOptions,
) -> f64 {
    let unit = options.objective_unit;
    let margin = result
        .perimeters
        .iter()
        .filter(|p| perimeter.contains(&p.state))
        .flat_map(|p| p.cnecs.iter())
        .filter(|c| crac.flow_cnecs[c.cnec].optimized)
        .map(|c| match unit {
            super::linear::ObjectiveUnit::Megawatt => c.margin_mw,
            super::linear::ObjectiveUnit::Ampere => c.margin_a,
        })
        .fold(f64::INFINITY, f64::min);
    let margin = if margin.is_finite() { margin } else { super::mnec::NO_CNEC_MARGIN };
    margin - options.mnec.cost(crac, result, perimeter, unit)
}

/// The worst optimized margin over `perimeter` in an assessment already made —
/// the "do nothing" baseline the search improves on.
fn worst_of(
    crac: &Crac,
    result: &super::evaluate::SecurityResult,
    perimeter: &[State],
) -> f64 {
    result
        .perimeters
        .iter()
        .filter(|p| perimeter.contains(&p.state))
        .filter_map(|p| p.min_optimized_margin(crac))
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// Actions too far from the most limiting element
// ---------------------------------------------------------------------------

/// The countries a branch touches — one, or two where it crosses a border.
fn branch_countries(
    network: &Network<'_>,
    branch: usize,
    dc: &[crate::linear::btheta::DcBranch],
) -> Vec<String> {
    let Some(b) = dc.iter().find(|b| b.index == branch) else { return Vec::new() };
    [b.from, b.to]
        .iter()
        .filter_map(|&bus| network.bus_countries.get(bus)?.clone())
        .collect()
}

/// Which countries share a border, from the branches that cross one.
///
/// Built from the network rather than from a table, exactly as the reference
/// does: two countries are neighbours if some branch has one end in each. A
/// border no branch crosses is not a border this network has.
fn country_boundaries(
    network: &Network<'_>,
    dc: &[crate::linear::btheta::DcBranch],
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for b in dc {
        let (Some(Some(x)), Some(Some(y))) =
            (network.bus_countries.get(b.from), network.bus_countries.get(b.to))
        else {
            continue;
        };
        if x == y {
            continue;
        }
        let pair = if x < y { (x.clone(), y.clone()) } else { (y.clone(), x.clone()) };
        if !out.contains(&pair) {
            out.push(pair);
        }
    }
    out
}

/// Whether `a` and `b` are within `max_boundaries` borders of each other.
///
/// Zero means "the same country". Breadth-first rather than the reference's
/// recursion, which revisits countries and can walk the same border twice; the
/// answer is the same and the cost is not.
fn within_boundaries(
    a: &str,
    b: &str,
    boundaries: &[(String, String)],
    max_boundaries: usize,
) -> bool {
    if a == b {
        return true;
    }
    let mut seen: Vec<&str> = vec![a];
    let mut frontier: Vec<&str> = vec![a];
    for _ in 0..max_boundaries {
        let mut next: Vec<&str> = Vec::new();
        for country in &frontier {
            for (x, y) in boundaries {
                let other = if x == country {
                    y.as_str()
                } else if y == country {
                    x.as_str()
                } else {
                    continue;
                };
                if other == b {
                    return true;
                }
                if !seen.contains(&other) {
                    seen.push(other);
                    next.push(other);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    false
}

/// Drop candidates too far from the worst CNEC to be plausible remedies.
///
/// Two rules from the reference are load-bearing and both are permissive:
/// an action whose location is unknown is **kept**, and a worst CNEC whose
/// location is unknown filters **nothing**. A filter that silently discarded
/// what it could not place would quietly shrink the search on any network
/// whose importer does not supply countries.
fn near_most_limiting(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    open: &[usize],
    available: Vec<(usize, Effect)>,
    max_boundaries: usize,
    model: super::evaluate::FlowModel,
) -> Vec<(usize, Effect)> {
    if network.bus_countries.is_empty() {
        return available;
    }
    let dc = crate::linear::btheta::dc_branches(
        network.lines,
        network.transformers,
        crate::linear::DcOptions::default(),
    );

    // The most limiting element, measured on the network as it stands and with
    // the same flow model the optimizer scores by — the reference locates it
    // from its own optimization result, not from a second opinion.
    let evaluated = measure_with(crac, network, resolution, open, model);
    let worst = evaluated
        .perimeters
        .iter()
        .filter(|p| perimeter.contains(&p.state))
        .flat_map(|p| p.cnecs.iter())
        .min_by(|a, b| a.margin_mw.total_cmp(&b.margin_mw));
    let Some(worst) = worst else { return available };
    let here = branch_countries(network, worst.branch, &dc);
    if here.is_empty() {
        return available;
    }

    let boundaries = country_boundaries(network, &dc);
    available
        .into_iter()
        .filter(|(_, effect)| {
            let mut touched: Vec<String> = Vec::new();
            for &branch in effect.open.iter().chain(&effect.close) {
                touched.extend(branch_countries(network, branch, &dc));
            }
            for &(branch, _, _) in &effect.taps {
                touched.extend(branch_countries(network, branch, &dc));
            }
            if touched.is_empty() {
                return true;
            }
            touched.iter().any(|t| {
                here.iter().any(|h| within_boundaries(h, t, &boundaries, max_boundaries))
            })
        })
        .collect()
}
