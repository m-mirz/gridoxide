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
use super::evaluate::{evaluate_with, Network, Resolution};
use super::linear::{optimize, LinearOptions, NetworkMut, Setpoint};

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
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_depth: 2,
            absolute_min_impact: 0.0,
            relative_min_impact: 0.0,
            linear: LinearOptions::default(),
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
pub fn search(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    solver: &mut dyn Solver,
    options: &SearchOptions,
) -> SearchResult {
    let available: Vec<(usize, Effect)> = crac
        .network_actions
        .iter()
        .filter(|action| {
            perimeter.iter().any(|s| action.usage_rules.iter().any(|r| r.covers(s)))
        })
        .filter_map(|action| {
            let index = crac.network_actions.iter().position(|a| a.id == action.id)?;
            let effect =
                effect_of(action, crac, resolution, network.tap_changers, network.lines.len())?;
            Some((index, effect))
        })
        .collect();

    // The root: no network action, range actions optimized.
    let base_open: Vec<usize> = network.initially_open.to_vec();
    let (root_margin, root_setpoints, root_transformers, root_buses) =
        leaf(crac, network, resolution, perimeter, solver, &options.linear, &base_open, &[]);
    let initial = margin_of(crac, network, resolution, perimeter, &base_open);

    let mut chosen: Vec<usize> = Vec::new();
    let mut best_margin = root_margin;
    let mut best_setpoints = root_setpoints;
    let mut best_transformers = root_transformers;
    let mut best_buses = root_buses;
    let mut leaves = 0usize;
    let mut depth = 0usize;

    for _ in 0..options.max_depth {
        let mut open_here: Vec<usize> = base_open.clone();
        let mut taps_here: Vec<(usize, i32, f64)> = Vec::new();
        for &index in &chosen {
            if let Some((_, effect)) = available.iter().find(|(i, _)| *i == index) {
                open_here.extend(&effect.open);
                open_here.retain(|b| !effect.close.contains(b));
                taps_here.extend(&effect.taps);
            }
        }

        type Leaf = (usize, f64, Vec<Setpoint>, Vec<crate::types::Transformer>, Vec<crate::types::Bus>);
    let mut winner: Option<Leaf> = None;
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
            let mut open = open_here.clone();
            open.extend(&effect.open);
            open.retain(|b| !effect.close.contains(b));
            let mut taps = taps_here.clone();
            taps.extend(&effect.taps);

            let (margin, setpoints, transformers, buses) = leaf(
                crac, network, resolution, perimeter, solver, &options.linear, &open, &taps,
            );
            leaves += 1;

            let better = match &winner {
                None => true,
                Some((best, previous, _, _, _)) => {
                    margin > *previous + 1e-12
                        || ((margin - *previous).abs() <= 1e-12
                            && crac.network_actions[*index].id < crac.network_actions[*best].id)
                }
            };
            if better {
                winner = Some((*index, margin, setpoints, transformers, buses));
            }
        }

        let Some((index, margin, setpoints, transformers, buses)) = winner else { break };
        if !improved_enough(best_margin, margin, options) {
            break;
        }
        chosen.push(index);
        best_margin = margin;
        best_setpoints = setpoints;
        best_transformers = transformers;
        best_buses = buses;
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
        setpoints: best_setpoints,
        initial_margin_mw: initial,
        final_margin_mw: best_margin,
        leaves,
        depth,
        open_branches,
        transformers: best_transformers,
        buses: best_buses,
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
    open: &[usize],
    taps: &[(usize, i32, f64)],
) -> (f64, Vec<Setpoint>, Vec<crate::types::Transformer>, Vec<crate::types::Bus>) {
    let mut transformers = network.transformers.to_vec();
    // A phase-shifter set-point is part of the candidate, so it has to be in
    // force before the leaf is optimized *and* before it is scored. Leaving it
    // out does not make the action fail loudly — it makes it evaluate as a
    // change that does nothing, so the search can never prefer it and would
    // misreport the network if it ever did.
    apply_taps(&mut transformers, network.tap_changers, network.lines.len(), taps);
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
        initially_open: network.initially_open,
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };
    let result = optimize_with_open(crac, &mut mutable, resolution, perimeter, solver, options, open);
    (result.0, result.1, transformers, buses)
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
) -> (f64, Vec<Setpoint>) {
    if open.is_empty() {
        let result = optimize(crac, network, resolution, perimeter, solver, options);
        return (result.final_margin_mw, result.setpoints);
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
        // Empty, deliberately: the open state is already baked into
        // `lines`/`transformers` above. Leaving the file's own list here
        // would re-open branches a *closing* remedial action has just shut,
        // because `evaluate` derives its open set from this field. That is
        // silent — every margin stays self-consistent and the optimizer
        // simply measures a network in which the automaton never acted.
        initially_open: &[],
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };
    let result = optimize(crac, &mut inner, resolution, perimeter, solver, options);
    // Carry any tap movement back to the caller's transformers.
    for (a, b) in network.transformers.iter_mut().zip(transformers.iter()) {
        a.tap = b.tap;
    }
    (result.final_margin_mw, result.setpoints)
}

/// The perimeter's minimum margin with `open` applied and no range action
/// moved — the "do nothing" baseline the search improves on.
fn margin_of(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    open: &[usize],
) -> f64 {
    evaluate_with(crac, network, resolution, open)
        .perimeters
        .iter()
        .filter(|p| perimeter.contains(&p.state))
        .filter_map(|p| p.min_margin())
        .fold(f64::INFINITY, f64::min)
}
