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
    /// `(branch, angle in degrees)` for a phase-shifter tap position.
    taps: Vec<(usize, f64)>,
}

/// Reduce a network action, or refuse it.
fn effect_of(
    action: &NetworkAction,
    crac: &Crac,
    resolution: &Resolution,
    lines: usize,
) -> Option<Effect> {
    let mut effect = Effect { open: Vec::new(), taps: Vec::new() };
    for elementary in &action.elementary {
        match elementary {
            ElementaryAction::TerminalsConnection { element, connected } => {
                let branch = resolution.branch(element)?;
                if *connected {
                    // Closing a branch that this model already treats as closed
                    // is a no-op, and closing one it treats as *absent* is not
                    // expressible — the importers drop out-of-service branches
                    // rather than carrying them as openable.
                    return None;
                }
                effect.open.push(branch);
            }
            ElementaryAction::Switch { element, open } => {
                let branch = resolution.branch(element)?;
                if !*open {
                    return None;
                }
                effect.open.push(branch);
            }
            ElementaryAction::PstTapPosition { element, tap } => {
                let branch = resolution.branch(element)?;
                if branch < lines {
                    return None;
                }
                // The angle comes from whichever range action describes this
                // phase shifter, since that is where the tap table lives. An
                // action naming a tap on a shifter no range action describes
                // cannot be applied.
                let angle = crac.range_actions.iter().find_map(|r| match &r.kind {
                    super::crac::RangeActionKind::Pst { element: e, .. } if e == element => {
                        r.kind.angle_at(*tap)
                    }
                    _ => None,
                })?;
                effect.taps.push((branch, angle));
            }
            // Injection and shunt changes need a mutable bus vector, which this
            // layer does not carry; a switch pair needs a closable switch.
            ElementaryAction::GeneratorSetpoint { .. }
            | ElementaryAction::LoadSetpoint { .. }
            | ElementaryAction::ShuntSection { .. }
            | ElementaryAction::SwitchPair { .. } => return None,
        }
    }
    (!effect.open.is_empty() || !effect.taps.is_empty()).then_some(effect)
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
            let effect = effect_of(action, crac, resolution, network.lines.len())?;
            Some((index, effect))
        })
        .collect();

    // The root: no network action, range actions optimized.
    let (root_margin, root_setpoints, root_transformers) =
        leaf(crac, network, resolution, perimeter, solver, &options.linear, &[]);
    let initial = margin_of(crac, network, resolution, perimeter, &[]);

    let mut chosen: Vec<usize> = Vec::new();
    let mut best_margin = root_margin;
    let mut best_setpoints = root_setpoints;
    let mut best_transformers = root_transformers;
    let mut leaves = 0usize;
    let mut depth = 0usize;

    for _ in 0..options.max_depth {
        let mut open_here: Vec<usize> = Vec::new();
        for &index in &chosen {
            if let Some((_, effect)) = available.iter().find(|(i, _)| *i == index) {
                open_here.extend(&effect.open);
            }
        }

        let mut winner: Option<(usize, f64, Vec<Setpoint>, Vec<crate::types::Transformer>)> = None;
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

            let (margin, setpoints, transformers) =
                leaf(crac, network, resolution, perimeter, solver, &options.linear, &open);
            leaves += 1;

            let better = match &winner {
                None => true,
                Some((best, previous, _, _)) => {
                    margin > *previous + 1e-12
                        || ((margin - *previous).abs() <= 1e-12
                            && crac.network_actions[*index].id < crac.network_actions[*best].id)
                }
            };
            if better {
                winner = Some((*index, margin, setpoints, transformers));
            }
        }

        let Some((index, margin, setpoints, transformers)) = winner else { break };
        if !improved_enough(best_margin, margin, options) {
            break;
        }
        chosen.push(index);
        best_margin = margin;
        best_setpoints = setpoints;
        best_transformers = transformers;
        depth += 1;
    }

    let mut open_branches: Vec<usize> = Vec::new();
    for &index in &chosen {
        if let Some((_, effect)) = available.iter().find(|(i, _)| *i == index) {
            open_branches.extend(&effect.open);
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
) -> (f64, Vec<Setpoint>, Vec<crate::types::Transformer>) {
    let mut transformers = network.transformers.to_vec();
    let mut mutable = NetworkMut {
        buses: network.buses,
        lines: network.lines,
        transformers: &mut transformers,
        branch_ids: network.branch_ids,
        base_mva: network.base_mva,
    };
    let result = optimize_with_open(crac, &mut mutable, resolution, perimeter, solver, options, open);
    (result.0, result.1, transformers)
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
            lines[branch].r = f64::INFINITY;
            lines[branch].x = f64::INFINITY;
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
