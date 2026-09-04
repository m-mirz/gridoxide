//! The linear optimization of range actions over one perimeter.
//!
//! Given a state, the remedial actions available in it, and the flows that
//! state produces, this chooses set-points for the *continuous* actions —
//! phase-shifter angles and redispatch — to maximize the worst margin.
//!
//! # The formulation
//!
//! Variables, per perimeter:
//!
//! - \\(F(c)\\), the flow on each CNEC, MW.
//! - \\(A(r)\\), the set-point of each range action: degrees for a phase
//!   shifter, MW for a redispatch.
//! - \\(\Delta^{+}(r), \Delta^{-}(r) \ge 0\\), its upward and downward
//!   variation from where it started.
//! - \\(MM\\), the minimum margin.
//!
//! The keystone is the linearization, which is what makes this an LP at all:
//!
//! \\[ F(c) \;=\; f_n(c) \;+\; \sum_r \sigma_n(r, c)\,\bigl[A(r) - \alpha_n(r)\bigr] \\]
//!
//! with \\(f_n\\) the flow at the current operating point and \\(\sigma_n\\)
//! the sensitivity of that flow to the action, both recomputed each outer
//! iteration. Then \\(MM \le f^{+}(c) - F(c)\\) and \\(MM \le F(c) - f^{-}(c)\\)
//! per optimized CNEC, minimizing \\(-MM\\) plus a small penalty on
//! \\(\Delta^{+} + \Delta^{-}\\) so that among equally good answers the one
//! that moves least wins.
//!
//! # One definition of the margin
//!
//! The limits this optimizes against come from
//! [`evaluate`](super::evaluate::evaluate), not from re-reading the CRAC's
//! thresholds here. That is not tidiness. A CRAC states thresholds in MW, in
//! amperes, or as a fraction of a rated current, and each needs a different
//! conversion; doing that conversion twice invites the optimizer to maximize a
//! quantity nobody measures. It did, briefly — the LP read an ampere threshold
//! as MW and so optimized against a limit some 40% adrift of the real one,
//! while the evaluator scored it correctly. Every margin stayed self-consistent
//! and the answer was simply wrong.
//!
//! # Monitored CNECs
//!
//! An MNEC earns a violation column \(V(c) \ge 0\) and the pair of soft
//! bounds in [`super::mnec`], priced into the objective at the configured
//! violation cost. It contributes no \(MM\) row: it is not something to
//! improve. See that module for what "not worse" is defined to mean, and why
//! the rule needs the margins of the *untouched* network to state at all.
//!
//! # Why it iterates
//!
//! The linearization is exact in DC for a *redispatch*, because DC flow is
//! linear in injection. It is **not** exact for a phase shifter: the
//! sensitivity of a flow to a shift depends on the network's susceptances, not
//! on the shift, so in pure DC it is also constant — but the tap-to-angle map
//! is not linear, and a discrete tap lands somewhere the continuous solution
//! did not ask for. So the loop solves, applies, recomputes, and keeps the
//! result only if the true minimum margin improved.
//!
//! # What is not here
//!
//! One perimeter at a time. Chaining a curative action to the preventive one
//! before it — the `relativeToPreviousInstant` range kind — needs several
//! states in one problem, and belongs with the multi-perimeter work.
//!
//! **Network actions are not here.** This layer moves continuous set-points;
//! choosing *which discrete actions to take* is the search tree's job, and the
//! two are meant to interleave — a topological action changes the sensitivities
//! the shifters are optimized against, so optimizing them separately gives a
//! worse answer than optimizing them together.
//!
//! **HVDC range actions are recognised and skipped**: gridoxide models a DC
//! network (`src/dc.rs`) but nothing connects it to a range action yet. A
//! counter trade has no network sensitivity at all, which is why the reference
//! leaves it out of its LP too.

use crate::linear::btheta::{dc_branches, dc_power_flow, DcBranch};
use crate::linear::sensitivity::DcSensitivity;
use crate::linear::DcOptions;
use crate::opf::{LinearProgram, OptStatus, Solver};
use crate::types::Transformer;

use super::crac::{Crac, RangeActionKind, State};
use super::evaluate::{evaluate_model, FlowModel, Network, Resolution};
use super::limits::Budget;
use super::mnec::{Mnec, NO_CNEC_MARGIN};
use super::usage::Constrained;

/// The unit the objective — the minimum margin being maximized — is measured
/// in.
///
/// Not cosmetic. Each CNEC converts between MW and amperes at its own voltage,
/// so the *ordering* of two candidate networks can differ between the two
/// units: a 400 kV CNEC and a 225 kV one with equal MW margins do not have
/// equal ampere margins, and the optimizer will trade one against the other
/// differently depending on which it is told to maximize.
///
/// The reference does not make this a setting. `RaoUtil.getFlowUnit` returns
/// megawatts for a DC load flow and **amperes for an AC one**, so the objective
/// follows the flow model, and every threshold expressed "in the objective's
/// unit" — the minimum-impact thresholds among them — follows with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ObjectiveUnit {
    #[default]
    Megawatt,
    Ampere,
}

/// How the optimizer should treat a phase shifter's taps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TapModel {
    /// Optimize the angle continuously, then round to the nearest tap and
    /// re-evaluate. Needs only an LP, so it runs on the in-house solver.
    #[default]
    Continuous,
    /// Optimize the tap itself as an integer variable. Needs a MIP backend —
    /// [`IpmSolver`](crate::opf::ipm) refuses, by design.
    Discrete,
}

#[derive(Clone, Debug)]
pub struct LinearOptions {
    /// Outer iterations: solve, apply, recompute, keep if better.
    pub max_iterations: usize,
    /// Objective penalty per degree of phase-shifter movement. Small, so it
    /// only breaks ties — but not zero, or the optimizer will happily move
    /// every shifter by a rounding error's worth for no gain.
    pub pst_penalty: f64,
    /// Objective penalty per MW of redispatch.
    pub injection_penalty: f64,
    /// Sensitivities below this are treated as zero, which keeps the problem
    /// sparse. In MW per degree.
    pub sensitivity_threshold: f64,
    pub tap_model: TapModel,
    /// What the minimum margin being maximized is measured in.
    pub objective_unit: ObjectiveUnit,
    /// Which flow model the optimizer *measures* with.
    ///
    /// Sensitivities stay DC either way — they are cheap, exact for the linear
    /// model, and only steer the search. This is what "the truth" means when a
    /// candidate is scored and when the iteration decides whether a move
    /// helped. Choosing AC costs a Newton-Raphson solve per outer iteration and
    /// buys agreement with a reference that does the same.
    pub flow_model: FlowModel,
    /// How monitored CNECs are held. See [`super::mnec`].
    ///
    /// Inert until its baseline has been measured, which
    /// [`castor::run`](super::castor::run) does once on the untouched network.
    pub mnec: Mnec,
    /// Which of the perimeter's CNECs are constrained, so a conditional usage
    /// rule can be answered. See [`super::usage`].
    ///
    /// Per *perimeter* rather than per run, and the second field here that is:
    /// [`search`](super::search::search) fills it in on a copy of these options
    /// before it evaluates any leaf, the same way
    /// [`castor::run`](super::castor::run) fills in the MNEC baseline. Left
    /// unmeasured a conditional rule falls back to its topological half.
    pub available: Constrained,
    /// How many range actions this leaf may still move. See [`super::limits`].
    ///
    /// The third and last of the per-context fields here, and the narrowest:
    /// [`mnec`](Self::mnec) is per run, [`available`](Self::available) is per
    /// perimeter, and this is per **leaf** — two candidates at the same depth
    /// that spend different TSOs' allowances leave different budgets behind.
    /// [`search`](super::search::search) fills all three in; a caller
    /// optimizing a lone perimeter gets the unconstrained default for each.
    pub limits: Budget,
    /// The states whose usage rules decide which range actions are available,
    /// when that is not the same as the states being optimized. See
    /// [`SearchOptions::available_at`](super::search::SearchOptions::available_at)
    /// — second preventive optimizes every CNEC while remaining a preventive
    /// perimeter, and without this it would help itself to every contingency's
    /// curative shifters.
    pub available_at: Option<Vec<State>>,
    /// The tap each phase-shifter range action sat on when this **perimeter**
    /// began, as `(range action index, tap)`.
    ///
    /// The anchor for a `relativeToPreviousInstant` range, and the only place
    /// that information exists: by the time the LP runs, the live tap has moved
    /// — a leaf may have applied a set-point action, and the outer iteration
    /// moves it again every round. Anchoring the box on the live tap would let
    /// it walk, one box-width per iteration, out of the range the CRAC wrote.
    ///
    /// Empty means "the network as imported", which is what a preventive
    /// perimeter starts from and therefore the right answer for a caller
    /// optimizing one on its own.
    pub previous_taps: Vec<(usize, i32)>,
}

impl Default for LinearOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            pst_penalty: 0.01,
            injection_penalty: 0.001,
            sensitivity_threshold: 1e-6,
            tap_model: TapModel::Continuous,
            objective_unit: ObjectiveUnit::Megawatt,
            flow_model: FlowModel::Dc,
            mnec: Mnec::default(),
            available: Constrained::unmeasured(),
            limits: Budget::default(),
            available_at: None,
            previous_taps: Vec::new(),
        }
    }
}

/// What one range action was moved to.
#[derive(Clone, Debug, PartialEq)]
pub struct Setpoint {
    /// Index into [`Crac::range_actions`].
    pub action: usize,
    /// Degrees for a phase shifter, MW for a redispatch.
    pub value: f64,
    /// Where it started, so a caller can see the movement rather than infer it.
    pub initial: f64,
    /// The tap it corresponds to, for a phase shifter.
    pub tap: Option<i32>,
}

impl Setpoint {
    pub fn moved(&self) -> bool {
        (self.value - self.initial).abs() > 1e-9
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearStatus {
    /// An improving set of set-points was found.
    Improved,
    /// The problem solved, but nothing beat the starting point. Not a failure:
    /// it is the answer when the range actions on offer cannot help.
    NoImprovement,
    /// The LP or MIP could not be solved.
    Failed,
}

#[derive(Clone, Debug)]
pub struct LinearResult {
    pub status: LinearStatus,
    pub setpoints: Vec<Setpoint>,
    /// Minimum margin over the perimeter's optimized CNECs, **MW**, before and
    /// after — whatever unit the objective was maximized in.
    pub initial_margin_mw: f64,
    pub final_margin_mw: f64,
    /// The same two in [`LinearOptions::objective_unit`], which is what the
    /// optimizer actually compared and what a minimum-impact threshold is
    /// measured against.
    ///
    /// Equal to the pair above when that unit is megawatts. Kept separate
    /// rather than overloading the `_mw` fields, because a caller reporting a
    /// margin and a caller ranking two candidates want different numbers and
    /// silently giving them the same one mislabels whichever is wrong.
    pub initial_objective: f64,
    pub final_objective: f64,
    /// Outer iterations actually run.
    pub iterations: usize,
}

impl LinearResult {
    pub fn improvement(&self) -> f64 {
        self.final_margin_mw - self.initial_margin_mw
    }
}

/// Measure a network with the requested flow model.
///
/// The shunts the AC path needs travel on the network itself, so this needs no
/// options of its own — which is why they were put there rather than on
/// [`LinearOptions`], where every DC caller would have had to state a value
/// nothing reads.
fn evaluate_in(
    crac: &Crac,
    view: &Network<'_>,
    resolution: &Resolution,
    model: FlowModel,
) -> super::evaluate::SecurityResult {
    let ac = super::evaluate::ac_options(view);
    evaluate_model(crac, view, resolution, view.initially_open, model, &ac)
}

/// A phase shifter's tap-to-angle table, in the **CRAC's** sign convention.
///
/// Prefers the **network's** table, falling back to the one the CRAC declares.
///
/// Angles are negated on the way: gridoxide's transformer angle is the negation
/// of IIDM's, which is what a CRAC is written in.
///
/// The order matters and is the opposite of the obvious one. A CRAC's table is a
/// *description* of a tap changer; the tap changer is equipment. When they
/// disagree the equipment wins, because the equipment is what the flows will
/// actually see — and they do disagree on the vendored material, where one
/// CRAC makes tap 16 6.23 degrees against the network's 9.06 because it was
/// written for an earlier revision. Optimizing against the CRAC's number and
/// applying the network's leaves the two halves modelling different machines.
///
/// The fallback still matters: a CRAC's PST range action frequently omits the
/// table altogether, and without it such an action has no positions to choose
/// between and is skipped without a word.
///
/// Tap *indices* are shared either way, so what gets reported is unaffected.
pub fn tap_table(
    declared: &[(i32, f64)],
    tap_changers: &[Option<crate::types::TapChanger>],
    branch: usize,
    lines: usize,
) -> Vec<(i32, f64)> {
    if let Some(changer) = branch.checked_sub(lines).and_then(|i| tap_changers.get(i)?.as_ref()) {
        let table: Vec<(i32, f64)> = (changer.low..=changer.high())
            .filter_map(|tap| changer.angle_deg(tap).map(|a| (tap, CRAC_ANGLE_SIGN * a)))
            .collect();
        if !table.is_empty() {
            return table;
        }
    }
    let mut table = declared.to_vec();
    table.sort_by_key(|(t, _)| *t);
    table
}

/// The sensitivity of every branch's DC flow to a one-radian phase shift on
/// `branch`, in per-unit power per radian.
///
/// Two terms, and omitting either is a silent error rather than a loud one:
///
/// - **indirect**, through the angles. `btheta` puts a shift on the right-hand
///   side as an injection of `+b·α` at `from` and `−b·α` at `to`, so the
///   response is exactly what those injections produce.
/// - **direct**, on the shifting branch itself. Its flow is
///   `b·(θ_f − θ_t − α)`, so the explicit `−α` contributes `−b` to its own
///   sensitivity and to no other branch's.
pub fn phase_shift_sensitivity(
    sensitivity: &DcSensitivity,
    branches: &[DcBranch],
    branch: usize,
) -> Option<Vec<f64>> {
    let target = branches.iter().find(|b| b.index == branch)?;
    let mut injections = vec![0.0; sensitivity.n_buses()];
    *injections.get_mut(target.from)? += target.b;
    *injections.get_mut(target.to)? -= target.b;
    let (_, mut flows) = sensitivity.response(&injections)?;
    if let Some(direct) = flows.get_mut(branch) {
        *direct -= target.b;
    }
    Some(flows)
}

/// Everything the optimizer needs to know about one range action it can move.
struct Control {
    /// Index into [`Crac::range_actions`].
    action: usize,
    /// Sensitivity of each perimeter CNEC's flow to one unit of this control,
    /// MW per degree (PST) or MW per MW (injection).
    sensitivity: Vec<f64>,
    /// Current set-point, in the control's own unit.
    current: f64,
    lower: f64,
    upper: f64,
    /// Objective penalty per unit moved.
    penalty: f64,
    /// For a phase shifter: the branch it sits on, and its tap table.
    pst: Option<PstControl>,
    /// For a redispatch: `(bus, key)` pairs, and the set-point currently
    /// written into the network. Applying works on the difference, so a
    /// proposal that is tried and reverted leaves the buses exactly as they
    /// were.
    injection: Option<InjectionControl>,
}

struct InjectionControl {
    distribution: Vec<(usize, f64)>,
    applied: f64,
    /// Sum of this action's distribution keys.
    ///
    /// Zero means the action moves power *between* buses and leaves the total
    /// alone. Non-zero means it creates or destroys some, and it may only be
    /// used alongside others that cancel it — which is what the balance row
    /// enforces.
    key_sum: f64,
}

/// Sign conversion between a CRAC's phase-shifter angles and
/// [`Transformer::tap`]'s.
///
/// A CRAC states tap angles in **IIDM's** convention, and gridoxide's complex
/// tap is the MATPOWER one, whose argument is its negation. Both importers
/// already encode that — `iidm.rs` negates `alpha` when building the tap, and
/// `ucte.rs` reverses the transformer for the same reason — so the two are
/// consistently opposite, which is exactly why the conversion can live in one
/// place instead of being decided per network.
///
/// Getting it wrong is almost invisible: the optimizer stays self-consistent
/// and finds the physically correct angle, and only the *tap number* it reports
/// comes out mirrored. On a symmetric phase shifter that means a plan saying
/// "tap +16" for the position an operator knows as −16 — which is worse than a
/// wrong margin, because the margin would have been questioned.
const CRAC_ANGLE_SIGN: f64 = -1.0;

struct PstControl {
    branch: usize,
    /// Ascending by tap.
    tap_to_angle: Vec<(i32, f64)>,
    tap: i32,
}

impl PstControl {
    fn nearest_tap(&self, angle: f64) -> i32 {
        self.tap_to_angle
            .iter()
            .min_by(|a, b| (a.1 - angle).abs().total_cmp(&(b.1 - angle).abs()))
            .map(|(t, _)| *t)
            .unwrap_or(self.tap)
    }

    /// The taps worth trying for a continuous angle: the two that bracket it.
    ///
    /// Rounding to the nearest is not good enough, and the failure is not
    /// subtle. The margin as a function of tap is piecewise linear with a kink
    /// wherever the binding CNEC changes, so its maximum sits *at* a kink — and
    /// the continuous optimum lands between two taps with the better one on
    /// whichever side the kink fell. Nearest-rounding then picks the worse one
    /// about half the time and the iteration converges there, because
    /// relinearizing at that tap proposes the same angle again.
    ///
    /// Observed: an optimum of 27.8 MW at tap 4 reported as 17.9 MW at tap 3,
    /// with the search perfectly convergent and perfectly wrong.
    fn bracketing_taps(&self, angle: f64) -> Vec<i32> {
        let mut below: Option<(i32, f64)> = None;
        let mut above: Option<(i32, f64)> = None;
        for &(tap, a) in &self.tap_to_angle {
            if a <= angle && below.is_none_or(|(_, b)| a > b) {
                below = Some((tap, a));
            }
            if a >= angle && above.is_none_or(|(_, b)| a < b) {
                above = Some((tap, a));
            }
        }
        let mut taps: Vec<i32> = below.into_iter().chain(above).map(|(t, _)| t).collect();
        taps.dedup();
        if taps.is_empty() {
            taps.push(self.nearest_tap(angle));
        }
        taps
    }

    fn angle_at(&self, tap: i32) -> Option<f64> {
        self.tap_to_angle.binary_search_by_key(&tap, |(t, _)| *t).ok().map(|i| self.tap_to_angle[i].1)
    }
}

/// Optimize the range actions available in `state`.
///
/// `transformers` is mutated: the returned set-points are *applied*, so the
/// caller's network reflects the answer. That is deliberate — a search tree
/// evaluates the next candidate against the network this left behind, and
/// returning set-points without applying them invites the two to drift apart.
pub fn optimize(
    crac: &Crac,
    network: &mut NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    solver: &mut dyn Solver,
    options: &LinearOptions,
) -> LinearResult {
    // Snapshotted *before* the first attempt, because that attempt leaves the
    // network wherever its answer says — and the whole point of the retry below
    // is to start each subset from the same place.
    let before = (network.buses.clone(), network.transformers.clone());
    let unconstrained = optimize_within(crac, network, resolution, perimeter, solver, options, None);
    let moved = moved_actions(&unconstrained);
    if options.limits.is_unlimited() || options.limits.admits(crac, &moved) {
        return unconstrained;
    }
    // The answer spends more of the plan's allowance than the CRAC permits, so
    // the question becomes *which* range actions to spend it on. That is a
    // cardinality constraint, which the reference expresses with a binary per
    // range action and a MIP; here the candidate sets are small enough —
    // **four** range actions in the largest vendored CRAC, two in every one
    // that declares a limit — that enumerating the admissible subsets answers
    // the same question exactly, on the LP solver already in hand.
    //
    // Reached only when the free answer is inadmissible, so the ordinary case
    // pays one comparison. A CRAC large enough for this to matter would need
    // the MIP, and `TapModel::Discrete` is where that belongs.
    let usable = usable_range_actions(crac, perimeter, options);
    let mut best: Option<(LinearResult, Vec<crate::types::Bus>, Vec<Transformer>)> = None;
    for subset in admissible_subsets(crac, &usable, &options.limits) {
        network.buses.clone_from(&before.0);
        network.transformers.clone_from(&before.1);
        let result =
            optimize_within(crac, network, resolution, perimeter, solver, options, Some(&subset));
        // The network each attempt leaves behind travels with its result, so
        // the winner can be restored without optimizing it a second time.
        if best.as_ref().is_none_or(|(b, ..)| result.final_objective > b.final_objective) {
            best = Some((result, network.buses.clone(), network.transformers.clone()));
        }
    }
    // The empty subset is admissible under every budget, so it is either
    // maximal itself or contained in one that is: there is always a winner.
    let (result, buses, transformers) = best.expect("doing nothing fits every budget");
    *network.buses = buses;
    *network.transformers = transformers;
    result
}

/// Which range actions a result actually moved, by index into
/// [`Crac::range_actions`].
fn moved_actions(result: &LinearResult) -> Vec<usize> {
    result.setpoints.iter().filter(|s| s.moved()).map(|s| s.action).collect()
}

/// The **maximal** subsets of `usable` the budget admits.
///
/// Maximal, not every subset, and the saving is not the point — the equivalence
/// is. Every one of these caps is a *count*, so admissibility is monotone: any
/// subset of an admissible set is admissible too. Offering the optimizer a
/// maximal set therefore loses nothing, because whatever it chooses to move is
/// a subset of one it was already allowed, and it dominates every smaller set
/// it contains. With one shifter allowed out of two this is two attempts rather
/// than four; with two of four, six rather than sixteen.
///
/// Ordered smallest first so that among equally good answers the one that moves
/// least wins, which is the same tie-break the movement penalty applies inside
/// a single solve.
fn admissible_subsets(crac: &Crac, usable: &[usize], budget: &Budget) -> Vec<Vec<usize>> {
    // Enumeration is exponential and only ever runs on a handful. A CRAC with
    // more range actions than this in one perimeter *and* a usage limit needs
    // the reference's MIP; doing nothing is the conservative answer until
    // `TapModel::Discrete` can give a better one.
    const CEILING: usize = 8;
    if usable.len() > CEILING {
        return vec![Vec::new()];
    }
    let admissible: Vec<Vec<usize>> = (0u32..(1 << usable.len()))
        .map(|mask| {
            usable
                .iter()
                .enumerate()
                .filter(|(k, _)| mask >> k & 1 == 1)
                .map(|(_, &i)| i)
                .collect::<Vec<usize>>()
        })
        .filter(|subset| budget.admits(crac, subset))
        .collect();
    let mut out: Vec<Vec<usize>> = admissible
        .iter()
        .filter(|subset| {
            !usable.iter().any(|extra| {
                if subset.contains(extra) {
                    return false;
                }
                let mut bigger = (*subset).clone();
                bigger.push(*extra);
                budget.admits(crac, &bigger)
            })
        })
        .cloned()
        .collect();
    out.sort_by_key(|s| s.len());
    out
}

/// The range actions this perimeter could move, before any budget is applied.
fn usable_range_actions(crac: &Crac, perimeter: &[State], options: &LinearOptions) -> Vec<usize> {
    crac.range_actions
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            let at = options.available_at.as_deref().unwrap_or(perimeter);
            options.available.allows(&a.usage_rules, at, crac)
        })
        .map(|(i, _)| i)
        .collect()
}

/// The optimization proper, optionally restricted to a subset of the range
/// actions.
fn optimize_within(
    crac: &Crac,
    network: &mut NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    solver: &mut dyn Solver,
    options: &LinearOptions,
    allowed: Option<&[usize]>,
) -> LinearResult {
    let dc_options = DcOptions::default();

    // The CNECs this perimeter optimizes, and the ones it merely monitors.
    //
    // They are kept apart because they earn different rows: an optimized CNEC
    // constrains the minimum margin, a monitored one gets a penalized violation
    // column. A CNEC that is both appears in both lists and gets both, which is
    // what the reference's two fillers between them produce.
    let cnec_indices = perimeter_cnecs(crac, resolution, perimeter, |c| c.optimized);
    let mnec_indices = if options.mnec.active() {
        perimeter_cnecs(crac, resolution, perimeter, |c| c.monitored)
    } else {
        Vec::new()
    };
    // One list for everything that needs a flow and a sensitivity row. The
    // optimized ones come first, so a position below `cnec_indices.len()` is an
    // optimized CNEC and anything after it is a monitored one.
    let lp_indices: Vec<usize> =
        cnec_indices.iter().chain(mnec_indices.iter()).copied().collect();

    let mut best = objective(crac, network, resolution, perimeter, options);
    let initial_objective = best;
    let initial_margin = margin(crac, network, resolution, perimeter, ObjectiveUnit::Megawatt, options.flow_model);
    let mut setpoints: Vec<Setpoint> = Vec::new();
    let mut iterations = 0;

    // Nothing to constrain in either direction. A perimeter with *only*
    // monitored CNECs is not one of these: there the LP has no minimum margin
    // to raise but a violation to remove, and stopping here would leave it
    // unremoved.
    if lp_indices.is_empty() {
        return LinearResult {
            status: LinearStatus::NoImprovement,
            setpoints,
            initial_margin_mw: initial_margin,
            final_margin_mw: initial_margin,
            initial_objective,
            final_objective: best,
            iterations,
        };
    }

    // Where every control started, so `Setpoint::initial` is the pre-optimization
    // value rather than the previous iteration's.
    let mut controls = match build_controls(crac, network, resolution, perimeter, &lp_indices, dc_options, options, allowed) {
        Some(c) if !c.is_empty() => c,
        _ => {
            return LinearResult {
                status: LinearStatus::NoImprovement,
                setpoints,
                initial_margin_mw: initial_margin,
                final_margin_mw: initial_margin,
                initial_objective,
                final_objective: best,
                iterations,
            }
        }
    };
    let starting: Vec<f64> = controls.iter().map(|c| c.current).collect();
    let starting_taps: Vec<Option<i32>> =
        controls.iter().map(|c| c.pst.as_ref().map(|p| p.tap)).collect();

    for _ in 0..options.max_iterations {
        iterations += 1;
        let (flows, limits) =
            perimeter_flows(crac, network, resolution, perimeter, &lp_indices, options.flow_model);
        let program = build_program(
            cnec_indices.len(),
            &mnec_indices,
            &flows,
            &limits,
            &controls,
            options,
        );
        let Ok(solution) = solver.solve(&program) else {
            break;
        };
        if solution.status != OptStatus::Optimal {
            break;
        }

        // Read the new set-points, round a phase shifter to a real tap, and
        // apply. Rounding *before* measuring is the point: the margin that
        // matters is the one at a tap the operator can actually select.
        let previous: Vec<(f64, Option<i32>)> =
            controls.iter().map(|c| (c.current, c.pst.as_ref().map(|p| p.tap))).collect();

        // The LP's answer, rounded to the nearer bracketing tap as a starting
        // point.
        let mut proposed: Vec<(f64, Option<i32>)> = controls
            .iter()
            .enumerate()
            .map(|(k, control)| {
                let value = solution.primal[setpoint_column(k)];
                match &control.pst {
                    Some(pst) => {
                        let tap = pst.nearest_tap(value);
                        (pst.angle_at(tap).unwrap_or(value), Some(tap))
                    }
                    None => (value, None),
                }
            })
            .collect();

        // Then try the *other* bracketing tap for each shifter in turn, keeping
        // it when the true margin improves. This is the step that stops the
        // iteration settling on the wrong side of a kink — see
        // `PstControl::bracketing_taps`.
        apply(crac, network, &mut controls, &proposed);
        let mut best_here = objective(crac, network, resolution, perimeter, options);
        for k in 0..controls.len() {
            let Some(pst) = controls[k].pst.as_ref() else { continue };
            let target = solution.primal[setpoint_column(k)];
            let candidates: Vec<i32> = pst
                .bracketing_taps(target)
                .into_iter()
                .filter(|t| Some(*t) != proposed[k].1)
                .collect();
            for tap in candidates {
                let Some(angle) = controls[k].pst.as_ref().and_then(|p| p.angle_at(tap)) else {
                    continue;
                };
                let mut trial = proposed.clone();
                trial[k] = (angle, Some(tap));
                apply(crac, network, &mut controls, &trial);
                let score = objective(crac, network, resolution, perimeter, options);
                if score > best_here + 1e-9 {
                    best_here = score;
                    proposed = trial;
                } else {
                    apply(crac, network, &mut controls, &proposed);
                }
            }
        }

        let moved = proposed
            .iter()
            .zip(&previous)
            .any(|((value, _), (was, _))| (value - was).abs() > 1e-9);
        if !moved {
            apply(crac, network, &mut controls, &previous);
            break;
        }
        apply(crac, network, &mut controls, &proposed);

        let score = objective(crac, network, resolution, perimeter, options);
        if score > best + 1e-9 {
            best = score;
            setpoints = controls
                .iter()
                .enumerate()
                .map(|(k, c)| Setpoint {
                    action: c.action,
                    value: c.current,
                    initial: starting[k],
                    tap: c.pst.as_ref().map(|p| p.tap),
                })
                .collect();
            // Relinearize around the new point.
            if let Some(rebuilt) =
                build_controls(crac, network, resolution, perimeter, &lp_indices, dc_options, options, allowed)
            {
                if rebuilt.len() == controls.len() {
                    controls = rebuilt;
                }
            }
        } else {
            // Worse or equal: put the network back and stop. Keeping a move
            // that did not help would leave the caller's network somewhere the
            // result does not describe.
            apply(crac, network, &mut controls, &previous);
            break;
        }
    }

    // Nothing kept means the network must be back where it started — the
    // rejection path above restores it, and this asserts the invariant rather
    // than assuming it.
    if setpoints.is_empty() {
        let restore: Vec<(f64, Option<i32>)> =
            starting.iter().copied().zip(starting_taps.iter().copied()).collect();
        apply(crac, network, &mut controls, &restore);
    }

    LinearResult {
        status: if setpoints.iter().any(Setpoint::moved) {
            LinearStatus::Improved
        } else if iterations == 0 {
            LinearStatus::Failed
        } else {
            LinearStatus::NoImprovement
        },
        setpoints,
        initial_margin_mw: initial_margin,
        // Re-measured in megawatts rather than converted from `best`: the two
        // are minima over the *same* CNECs but not necessarily over the same
        // one, so converting the ampere answer would name a margin no CNEC has.
        final_margin_mw: margin(crac, network, resolution, perimeter, ObjectiveUnit::Megawatt, options.flow_model),
        initial_objective,
        final_objective: best,
        iterations,
    }
}

/// The network, mutably, so set-points can be applied.
pub struct NetworkMut<'a> {
    /// Mutable, because a redispatch is applied by moving bus injections.
    /// Without that an injection range action can be optimized and then never
    /// take effect — the measurement sees an unchanged network, finds no
    /// improvement, and the action is silently dropped from every answer.
    pub buses: &'a mut Vec<crate::types::Bus>,
    pub lines: &'a [crate::types::Line],
    pub transformers: &'a mut Vec<Transformer>,
    pub branch_ids: &'a [String],
    pub bus_ids: &'a [String],
    pub initially_open: &'a [usize],
    /// ISO country code per bus; see [`Network::bus_countries`].
    pub bus_countries: &'a [Option<String>],
    /// Shunt admittances; see [`Network::shunts`].
    pub shunts: &'a [crate::network::ShuntAdm],
    /// Per-bus generation; see [`Network::generation`].
    pub generation: &'a [f64],
    pub tap_changers: &'a [Option<crate::types::TapChanger>],
    pub base_mva: f64,
}

impl NetworkMut<'_> {
    fn view(&self) -> Network<'_> {
        Network {
            generation: self.generation,
            buses: self.buses,
            lines: self.lines,
            transformers: self.transformers,
            branch_ids: self.branch_ids,
            bus_ids: self.bus_ids,
            initially_open: self.initially_open,
            bus_countries: self.bus_countries,
            shunts: self.shunts,
            tap_changers: self.tap_changers,
            base_mva: self.base_mva,
        }
    }
}

/// Column layout: `[A(0), Δ⁺(0), Δ⁻(0), A(1), …, MM]`.
fn setpoint_column(k: usize) -> usize {
    3 * k
}
fn up_column(k: usize) -> usize {
    3 * k + 1
}
fn down_column(k: usize) -> usize {
    3 * k + 2
}

fn margin_column(n_controls: usize) -> usize {
    3 * n_controls
}

/// The violation column of the `slot`-th monitored CNEC.
fn violation_column(n_controls: usize, slot: usize) -> usize {
    3 * n_controls + 1 + slot
}

/// The LP.
///
/// Columns are `3·n` control columns (set-point, up, down), then the minimum
/// margin, then one violation column per monitored CNEC.
///
/// `flows` and `limits` cover the optimized CNECs first — `optimized` of them —
/// and the monitored ones after, aligned with `mnecs`.
fn build_program(
    optimized: usize,
    mnecs: &[usize],
    flows: &[f64],
    limits: &[(f64, f64, f64)],
    controls: &[Control],
    options: &LinearOptions,
) -> LinearProgram {
    let n = controls.len();
    let mm = margin_column(n);
    let mut lp = LinearProgram::new(3 * n + 1 + mnecs.len());

    for (k, control) in controls.iter().enumerate() {
        lp.col_lower[setpoint_column(k)] = control.lower;
        lp.col_upper[setpoint_column(k)] = control.upper;
        for column in [up_column(k), down_column(k)] {
            lp.col_lower[column] = 0.0;
            lp.col_upper[column] = f64::INFINITY;
            lp.col_cost[column] = control.penalty;
        }
        // A(r) − Δ⁺(r) + Δ⁻(r) = current
        lp.add_row(
            &[(setpoint_column(k), 1.0), (up_column(k), -1.0), (down_column(k), 1.0)],
            control.current,
            control.current,
        );
    }

    // The network must still balance after any redispatch:
    //
    //     Σ_r (Δ⁺(r) − Δ⁻(r)) · Σ_d key_d(r) = 0
    //
    // An action whose keys sum to zero moves power between buses and is
    // unaffected. One whose keys do not sum to zero creates or destroys power,
    // and this row is what stops it being used alone — it may still be used
    // *alongside* another that cancels it, which is exactly the case a
    // single-action check would forbid and a real CRAC contains.
    //
    // Without this the optimizer happily invents generation and reports a
    // margin no network could achieve.
    let balance: Vec<(usize, f64)> = controls
        .iter()
        .enumerate()
        .filter_map(|(k, c)| {
            let sum = c.injection.as_ref()?.key_sum;
            (sum.abs() > 1e-12).then_some((k, sum))
        })
        .collect();
    if !balance.is_empty() {
        let mut row: Vec<(usize, f64)> = Vec::with_capacity(balance.len() * 2);
        for &(k, sum) in &balance {
            row.push((up_column(k), sum));
            row.push((down_column(k), -sum));
        }
        lp.add_row(&row, 0.0, 0.0);
    }

    // Maximize the minimum margin.
    lp.col_lower[mm] = f64::NEG_INFINITY;
    lp.col_upper[mm] = f64::INFINITY;
    lp.col_cost[mm] = -1.0;

    for position in 0..optimized {
        let reference = flows[position];
        let (lower, upper, amperes_per_mw) = limits[position];
        // Scaling the row is what makes the objective's unit mean something.
        // `MM ≤ (upper − F)` in MW becomes `MM ≤ (upper − F)·k` in amperes, and
        // `k` is this CNEC's own — two CNECs at different voltages convert
        // differently, which is precisely why the two units can rank the same
        // pair of candidate networks in opposite orders.
        let scale = match options.objective_unit {
            ObjectiveUnit::Megawatt => 1.0,
            ObjectiveUnit::Ampere => amperes_per_mw,
        };
        if scale == 0.0 {
            continue;
        }
        // The flow is substituted directly rather than given a variable of its
        // own: `F(c)` appears only in the two margin rows, so eliminating it
        // halves the problem for no loss.
        let terms: Vec<(usize, f64)> = controls
            .iter()
            .enumerate()
            .filter_map(|(k, c)| {
                let s = c.sensitivity.get(position).copied().unwrap_or(0.0);
                (s.abs() >= options.sensitivity_threshold).then_some((setpoint_column(k), s))
            })
            .collect();
        let constant: f64 = reference
            - controls
                .iter()
                .enumerate()
                .map(|(k, c)| {
                    let s = c.sensitivity.get(position).copied().unwrap_or(0.0);
                    if s.abs() >= options.sensitivity_threshold {
                        let _ = k;
                        s * c.current
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();

        // `MM ≤ upper − F` and `MM ≤ F − lower`, with `F = constant + Σ σ·A`.
        //
        // The two bounds are separate rather than one magnitude used twice. A
        // CNEC with only a lower threshold has no upper row at all, and adding
        // one — which is what a symmetric `|F| ≤ limit` does — constrains the
        // flow in a direction the CRAC left free, so the optimizer refuses
        // set-points that are perfectly legal. An infinite bound simply
        // contributes no row.
        for is_upper in [true, false] {
            let limit = if is_upper { upper } else { lower };
            if !limit.is_finite() {
                continue;
            }
            let mut coefficients: Vec<(usize, f64)> = Vec::with_capacity(terms.len() + 1);
            coefficients.push((mm, 1.0));
            let sign = if is_upper { 1.0 } else { -1.0 };
            for &(column, s) in &terms {
                coefficients.push((column, sign * s * scale));
            }
            let bound = scale * if is_upper { limit - constant } else { constant - limit };
            lp.add_row(&coefficients, f64::NEG_INFINITY, bound);
        }
    }

    // Monitored CNECs: a bound the flow may cross only by paying for it.
    //
    //     F(c) − V(c) ≤ max(f⁺(c), f₀(c) + d) − a
    //     F(c) + V(c) ≥ min(f⁻(c), f₀(c) − d) + a
    //
    // The `max`/`min` against the initial flow is what makes this a "do not
    // make it worse" rule rather than a second threshold: an MNEC already past
    // its limit is held to where it was, plus the acceptable decrease `d`, and
    // one comfortably inside its limit is held to the limit. Both are relaxed
    // by `V(c) ≥ 0`, priced into the objective — so the constraint yields when
    // the alternative is worse, which is the whole reason it is soft.
    //
    // `d` and the adjustment `a` are stated in the objective's unit and the LP
    // works in megawatts, so they are divided by this CNEC's own amperes-per-MW
    // on the way in — the same factor, and the same per-CNEC voltage, that
    // scales the margin rows above.
    // Every violation column is non-negative whether or not it ends up in a
    // row. A column left at the default free bounds would be one the solver may
    // move for no reason.
    for slot in 0..mnecs.len() {
        lp.col_lower[violation_column(n, slot)] = 0.0;
        lp.col_upper[violation_column(n, slot)] = f64::INFINITY;
    }
    for (slot, &cnec) in mnecs.iter().enumerate() {
        let position = optimized + slot;
        let violation = violation_column(n, slot);
        let (lower, upper, amperes_per_mw) = limits[position];
        let Some(initial) = options.mnec.baseline.get(cnec) else { continue };
        let to_mw = match options.objective_unit {
            ObjectiveUnit::Megawatt => 1.0,
            // No voltage means no conversion, and a bound converted at a made-up
            // voltage is worse than no bound at all.
            ObjectiveUnit::Ampere if amperes_per_mw > 0.0 => 1.0 / amperes_per_mw,
            ObjectiveUnit::Ampere => continue,
        };
        let decrease = options.mnec.options.acceptable_margin_decrease * to_mw;
        let adjustment = options.mnec.options.constraint_adjustment_coefficient * to_mw;
        // The column is in megawatts and the price is per unit of the
        // objective, so the two are reconciled here rather than by scaling the
        // rows — which would price a violation differently depending on which
        // bound it crossed.
        lp.col_cost[violation] = options.mnec.options.violation_cost / to_mw;

        let terms: Vec<(usize, f64)> = controls
            .iter()
            .enumerate()
            .filter_map(|(k, c)| {
                let s = c.sensitivity.get(position).copied().unwrap_or(0.0);
                (s.abs() >= options.sensitivity_threshold).then_some((k, s))
            })
            .collect();
        let constant: f64 =
            flows[position] - terms.iter().map(|&(k, s)| s * controls[k].current).sum::<f64>();

        if upper.is_finite() {
            let bound = f64::max(upper, initial.flow_mw + decrease) - adjustment;
            let mut coefficients: Vec<(usize, f64)> = Vec::with_capacity(terms.len() + 1);
            coefficients.extend(terms.iter().map(|&(k, s)| (setpoint_column(k), s)));
            coefficients.push((violation, -1.0));
            lp.add_row(&coefficients, f64::NEG_INFINITY, bound - constant);
        }
        if lower.is_finite() {
            let bound = f64::min(lower, initial.flow_mw - decrease) + adjustment;
            let mut coefficients: Vec<(usize, f64)> = Vec::with_capacity(terms.len() + 1);
            coefficients.extend(terms.iter().map(|&(k, s)| (setpoint_column(k), -s)));
            coefficients.push((violation, -1.0));
            lp.add_row(&coefficients, f64::NEG_INFINITY, constant - bound);
        }
    }
    lp
}

/// The CNECs of `perimeter` that `wanted` selects, in CRAC order, skipping any
/// whose network element does not resolve.
fn perimeter_cnecs(
    crac: &Crac,
    resolution: &Resolution,
    perimeter: &[State],
    wanted: impl Fn(&super::crac::FlowCnec) -> bool,
) -> Vec<usize> {
    crac.flow_cnecs
        .iter()
        .enumerate()
        .filter(|(_, c)| perimeter.contains(&c.state) && wanted(c))
        .filter(|(_, c)| resolution.branch(&c.network_element).is_some())
        .map(|(i, _)| i)
        .collect()
}

/// Build one control per usable range action, with its sensitivity to each of
/// `cnecs` — which is the LP's row set, optimized and monitored alike, in the
/// order the rows will be written.
#[allow(clippy::too_many_arguments)]
fn build_controls(
    crac: &Crac,
    network: &NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    cnecs: &[usize],
    dc_options: DcOptions,
    options: &LinearOptions,
    allowed: Option<&[usize]>,
) -> Option<Vec<Control>> {
    // One linearization point **per contingency**, not one for the whole
    // perimeter.
    //
    // A CNEC's sensitivity to a shifter is a property of the network that CNEC
    // lives in, and a post-contingency network is not the intact one. While a
    // perimeter was always a single contingency's — which every ordinary
    // perimeter is — reading them all off the intact network was a small and
    // uniform error. A perimeter spanning every state has no single network,
    // and there the error stops being uniform: preventive columns are traded
    // against curative CNECs whose response to them is the wrong number.
    let points = SensitivityPoints::build(network, crac, resolution, cnecs, dc_options)?;

    let mut controls = Vec::new();
    for (index, action) in crac.range_actions.iter().enumerate() {
        let at = options.available_at.as_deref().unwrap_or(perimeter);
        if !options.available.allows(&action.usage_rules, at, crac) {
            continue;
        }
        // A usage limit already spent on network actions leaves room for only
        // some of these; `allowed` is the subset being tried.
        if allowed.is_some_and(|a| !a.contains(&index)) {
            continue;
        }
        match &action.kind {
            RangeActionKind::Pst { element, initial_tap, tap_to_angle } => {
                let Some(branch) = resolution.branch(element) else { continue };
                let table = tap_table(
                    tap_to_angle,
                    network.tap_changers,
                    branch,
                    network.lines.len(),
                );
                if table.is_empty() {
                    continue;
                }
                let Some(column) = points.phase_shift(branch) else {
                    continue;
                };
                // MW per degree *of the CRAC's angle*: the column is per-unit
                // per radian of gridoxide's, so the sign flips with it.
                let scale =
                    CRAC_ANGLE_SIGN * network.base_mva * std::f64::consts::PI / 180.0;
                let sensitivity_mw: Vec<f64> = column.iter().map(|s| s * scale).collect();

                // The live tap is whatever the transformer's angle is closest
                // to, not the CRAC's `initialTap` — a search tree may have
                // moved it since.
                let live_angle = live_crac_angle(network, branch).unwrap_or_else(|| {
                    table
                        .iter()
                        .find(|(t, _)| t == initial_tap)
                        .map(|(_, a)| *a)
                        .unwrap_or(0.0)
                });
                let tap = tap_nearest(&table, live_angle).unwrap_or(*initial_tap);
                let current = table
                    .iter()
                    .find(|(t, _)| *t == tap)
                    .map(|(_, a)| *a)
                    .unwrap_or(live_angle);

                // Where this perimeter began, which is the anchor a
                // `relativeToPreviousInstant` range is written against. The
                // caller states it because only the caller knows where the
                // perimeter began; falling back to `initial_tap` is the right
                // answer for a preventive perimeter, whose previous instant
                // *is* the network as imported.
                let previous_tap = options
                    .previous_taps
                    .iter()
                    .find(|(i, _)| *i == index)
                    .map_or(*initial_tap, |(_, t)| *t);
                let (lower, upper) = tap_bounds(action, &table, *initial_tap, previous_tap);
                if !starts_inside_its_range(current, lower, upper) {
                    continue;
                }
                controls.push(Control {
                    action: index,
                    sensitivity: sensitivity_mw,
                    current,
                    lower,
                    upper,
                    penalty: options.pst_penalty,
                    pst: Some(PstControl { branch, tap_to_angle: table, tap }),
                    injection: None,
                });
            }
            RangeActionKind::Injection { distribution } => {
                // A redispatch shifts power between buses by its distribution
                // keys. Its sensitivity is the PTDF combination those keys
                // produce, which is exact in DC.
                let mut injections = vec![0.0; network.buses.len()];
                let mut keys: Vec<(usize, f64)> = Vec::new();
                let mut usable = false;
                for (element, key) in distribution {
                    // A redispatch names generators and loads, which are buses.
                    // Resolving them as branches finds nothing and silently
                    // drops the action.
                    let Some(bus) = resolution.bus(element) else { continue };
                    if bus >= injections.len() {
                        continue;
                    }
                    injections[bus] += key / network.base_mva;
                    keys.push((bus, *key));
                    usable = true;
                }
                if !usable {
                    continue;
                }
                let Some(sensitivity_mw) = points.injection(&injections) else { continue };
                let (lower, upper) = standard_bounds(action);
                if !starts_inside_its_range(0.0, lower, upper) {
                    continue;
                }
                controls.push(Control {
                    action: index,
                    sensitivity: sensitivity_mw,
                    current: 0.0,
                    lower,
                    upper,
                    penalty: options.injection_penalty,
                    pst: None,
                    injection: Some(InjectionControl {
                        key_sum: keys.iter().map(|(_, k)| *k).sum(),
                        distribution: keys,
                        applied: 0.0,
                    }),
                });
            }
            // Recognised and skipped: gridoxide models a DC network but nothing
            // connects it to a range action, and a counter trade has no network
            // sensitivity at all.
            RangeActionKind::Hvdc { .. } | RangeActionKind::CounterTrade { .. } => {}
        }
    }
    Some(controls)
}

/// The live phase shift of a branch in the **CRAC's** convention, so it can be
/// looked up in the CRAC's own tap table.
fn live_crac_angle(network: &NetworkMut<'_>, branch: usize) -> Option<f64> {
    let i = branch.checked_sub(network.lines.len())?;
    Some(CRAC_ANGLE_SIGN * network.transformers.get(i)?.tap.arg().to_degrees())
}

/// The position in `table` whose angle is nearest `angle`.
///
/// A search over the table rather than arithmetic on the step size, because a
/// tap-to-angle map is not necessarily linear and is not necessarily monotone.
fn tap_nearest(table: &[(i32, f64)], angle: f64) -> Option<i32> {
    table
        .iter()
        .min_by(|a, b| (a.1 - angle).abs().total_cmp(&(b.1 - angle).abs()))
        .map(|(t, _)| *t)
}

/// The tap each phase-shifter range action sits on in `network`, as
/// [`LinearOptions::previous_taps`] wants it.
///
/// Call this on the network a perimeter is **handed**, before anything in that
/// perimeter has acted: for a preventive perimeter that is the network as
/// imported, and for a curative one it is what the preventive stage and the
/// automatons left, which is precisely the instant a
/// `relativeToPreviousInstant` range is written against.
pub fn taps_now(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
) -> Vec<(usize, i32)> {
    crac.range_actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| {
            let RangeActionKind::Pst { element, initial_tap, tap_to_angle } = &action.kind else {
                return None;
            };
            let branch = resolution.branch(element)?;
            let table =
                tap_table(tap_to_angle, network.tap_changers, branch, network.lines.len());
            if table.is_empty() {
                return None;
            }
            let i = branch.checked_sub(network.lines.len())?;
            let angle =
                CRAC_ANGLE_SIGN * network.transformers.get(i)?.tap.arg().to_degrees();
            Some((index, tap_nearest(&table, angle).unwrap_or(*initial_tap)))
        })
        .collect()
}

/// Angle bounds from a PST range action's tap ranges.
///
/// The three kinds are three different anchors and the ranges are
/// **intersected**, so a CRAC declaring all three — and the reference's own
/// `SL_ep13us5case3` does — is constrained by whichever binds:
///
/// | kind | anchored on |
/// |---|---|
/// | `absolute` | nothing; the bounds are tap positions |
/// | `relativeToInitialNetwork` | `initial_tap`, the network as imported |
/// | `relativeToPreviousInstant` | `previous_tap`, where this perimeter began |
///
/// The last two coincide for a preventive perimeter and part company for a
/// curative one, which is the whole point of having both: "no more than ten
/// taps from where the file had it" and "no more than ten taps from whatever
/// the preventive plan left" are different permissions, and a TSO writes both.
///
/// `previous_tap` is **not** the live tap. By the time this is called the
/// shifter may have been moved by this leaf's own set-point action and will be
/// moved again by every outer iteration; anchoring on that lets the box walk a
/// width per round until the answer bears no relation to what the CRAC allowed.
pub(super) fn tap_bounds(
    action: &super::crac::RangeAction,
    table: &[(i32, f64)],
    initial_tap: i32,
    previous_tap: i32,
) -> (f64, f64) {
    let (mut low, mut high) = (i32::MIN, i32::MAX);
    for range in &action.ranges {
        let anchor = match range.kind {
            super::crac::RangeKind::RelativeToInitialNetwork => initial_tap,
            super::crac::RangeKind::RelativeToPreviousInstant => previous_tap,
            // Absolute bounds anchor on nothing. `relativeToPreviousTimeStep`
            // belongs to the multi-timestamp RAO this does not model, and is
            // read as absolute rather than silently anchored on the wrong
            // instant — there is no previous time step to be relative to.
            super::crac::RangeKind::Absolute
            | super::crac::RangeKind::RelativeToPreviousTimeStep => 0,
        };
        let (min, max) = (
            range.min.map(|m| anchor + m as i32),
            range.max.map(|m| anchor + m as i32),
        );
        if let Some(m) = min {
            low = low.max(m);
        }
        if let Some(m) = max {
            high = high.min(m);
        }
    }
    let angles: Vec<f64> = table
        .iter()
        .filter(|(t, _)| *t >= low && *t <= high)
        .map(|(_, a)| *a)
        .collect();
    if angles.is_empty() {
        let all: Vec<f64> = table.iter().map(|(_, a)| *a).collect();
        return (
            all.iter().copied().fold(f64::INFINITY, f64::min),
            all.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
    }
    (
        angles.iter().copied().fold(f64::INFINITY, f64::min),
        angles.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// One DC linearization per contingency, so every CNEC's sensitivity is read in
/// the network that CNEC actually lives in.
///
/// Built once per `build_controls` call and shared by every control: the
/// expensive part is the factorization, and it does not depend on which range
/// action is being differentiated.
struct SensitivityPoints {
    /// `(contingency, branches, sensitivity)`; `None` is the intact network.
    points: Vec<(Option<usize>, Vec<DcBranch>, DcSensitivity)>,
    /// Which point each CNEC position reads from.
    per_cnec: Vec<usize>,
    /// The branch each CNEC position monitors.
    branches: Vec<usize>,
}

impl SensitivityPoints {
    fn build(
        network: &NetworkMut<'_>,
        crac: &Crac,
        resolution: &Resolution,
        cnecs: &[usize],
        dc_options: DcOptions,
    ) -> Option<Self> {
        let n_branches = network.lines.len() + network.transformers.len();
        // The contingencies this perimeter's CNECs actually live under, so a
        // perimeter spanning one state pays for one factorization exactly as it
        // did before.
        let mut wanted: Vec<Option<usize>> =
            cnecs.iter().map(|&i| crac.flow_cnecs[i].state.contingency).collect();
        wanted.sort_unstable();
        wanted.dedup();

        let mut points = Vec::with_capacity(wanted.len());
        for contingency in wanted {
            let mut lines = network.lines.to_vec();
            let mut transformers = network.transformers.clone();
            if let Some(c) = contingency {
                for element in &crac.contingencies[c].elements {
                    let Some(branch) = resolution.branch(element) else { continue };
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
            }
            let branches = dc_branches(&lines, &transformers, dc_options);
            // A contingency that severs the network has no linearization of its
            // own; those CNECs fall back to the intact point below, which is a
            // better answer than none — and the evaluator reports the state as
            // severed regardless.
            let Some(sensitivity) = DcSensitivity::new(network.buses, &branches, n_branches) else {
                continue;
            };
            points.push((contingency, branches, sensitivity));
        }
        if points.is_empty() {
            return None;
        }
        let per_cnec = cnecs
            .iter()
            .map(|&i| {
                let want = crac.flow_cnecs[i].state.contingency;
                points.iter().position(|(c, _, _)| *c == want).unwrap_or(0)
            })
            .collect();
        let branches = cnecs
            .iter()
            .map(|&i| resolution.branch(&crac.flow_cnecs[i].network_element).unwrap())
            .collect();
        Some(Self { points, per_cnec, branches })
    }

    /// MW per radian of `branch`'s phase shift at each CNEC, each read in its
    /// own network.
    fn phase_shift(&self, branch: usize) -> Option<Vec<f64>> {
        let columns: Vec<Option<Vec<f64>>> = self
            .points
            .iter()
            .map(|(_, branches, s)| phase_shift_sensitivity(s, branches, branch))
            .collect();
        columns.iter().any(Option::is_some).then(|| self.gather(&columns))
    }

    /// The same for an injection pattern.
    fn injection(&self, injections: &[f64]) -> Option<Vec<f64>> {
        let columns: Vec<Option<Vec<f64>>> = self
            .points
            .iter()
            .map(|(_, _, s)| s.response(injections).map(|(_, column)| column))
            .collect();
        columns.iter().any(Option::is_some).then(|| self.gather(&columns))
    }

    fn gather(&self, columns: &[Option<Vec<f64>>]) -> Vec<f64> {
        self.per_cnec
            .iter()
            .zip(&self.branches)
            .map(|(&point, &branch)| {
                columns
                    .get(point)
                    .and_then(|c| c.as_ref())
                    .and_then(|c| c.get(branch))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect()
    }
}

/// Whether a range action can be optimized at all from where the perimeter
/// found it.
///
/// A range action whose **starting** set-point is already outside its own range
/// is not a tightly-constrained lever, it is not a lever: the CRAC is saying
/// this device may only be at positions it is not at, and no movement the
/// optimizer chooses can make that true. The reference drops such an action
/// from the perimeter outright — `doesPrePerimeterSetpointRespectRange` — and
/// so does this.
///
/// It bites when an earlier perimeter has moved the device. The reference's own
/// `SL_ep15us11-3case2_withPstCra` declares four range actions on **one**
/// phase shifter, of which the one named `useless_pst` permits tap 0 and
/// nothing else; by the time the curative perimeter runs, an automaton has put
/// that shifter on tap −8. Kept, it becomes a second control on a device that
/// already has one, pinned to a position the machine is not at and pulling
/// against the action the scenario is about — so the curative perimeter moves
/// nothing and reports that nothing helped.
fn starts_inside_its_range(current: f64, lower: f64, upper: f64) -> bool {
    current >= lower - 1e-6 && current <= upper + 1e-6
}

fn standard_bounds(action: &super::crac::RangeAction) -> (f64, f64) {
    let mut low = f64::NEG_INFINITY;
    let mut high = f64::INFINITY;
    for range in &action.ranges {
        if let Some(m) = range.min {
            low = low.max(m);
        }
        if let Some(m) = range.max {
            high = high.min(m);
        }
    }
    if low > high {
        return (0.0, 0.0);
    }
    (low, high)
}

/// Apply proposed set-points to the network and to the controls.
fn apply(
    _crac: &Crac,
    network: &mut NetworkMut<'_>,
    controls: &mut [Control],
    proposed: &[(f64, Option<i32>)],
) {
    let base_mva = network.base_mva;
    for (control, &(value, tap)) in controls.iter_mut().zip(proposed) {
        control.current = value;
        if let Some(injection) = control.injection.as_mut() {
            // Move the *difference*, so trying a proposal and reverting it
            // returns the buses to exactly where they were rather than
            // accumulating.
            let delta = value - injection.applied;
            if delta != 0.0 {
                for &(bus, key) in &injection.distribution {
                    if let Some(b) = network.buses.get_mut(bus) {
                        b.p_spec += key * delta / base_mva;
                    }
                }
                injection.applied = value;
            }
        }
        if let Some(pst) = control.pst.as_mut() {
            if let Some(tap) = tap {
                pst.tap = tap;
            }
            if let Some(i) = pst.branch.checked_sub(network.lines.len()) {
                // The **network's** own step is what gets applied, at the tap
                // the optimizer chose — not the angle the CRAC's table gives for
                // that tap.
                //
                // The two can disagree, and on the vendored material they do:
                // one CRAC's table makes tap 16 6.23 degrees where the network's
                // `##R` record makes it 9.06, because the CRAC was written
                // against a different revision of the network. A tap changer is
                // physical equipment and a table in a CRAC is a description of
                // it; when they conflict the equipment wins. It is also what
                // powsybl's own `PstRangeAction` does — it converts the
                // set-point to a tap and sets the *position*, letting the
                // network's steps decide the angle.
                let from_network = network
                    .tap_changers
                    .get(i)
                    .and_then(|c| c.as_ref())
                    .and_then(|c| tap.and_then(|t| c.at(t)));
                if let Some(transformer) = network.transformers.get_mut(i) {
                    let ratio = transformer.tap.norm();
                    transformer.tap = match from_network {
                        Some(step) => num_complex::Complex::from_polar(ratio, step.arg()),
                        None => num_complex::Complex::from_polar(
                            ratio,
                            (CRAC_ANGLE_SIGN * value).to_radians(),
                        ),
                    };
                }
            }
        }
    }
}

/// The minimum margin over the perimeter's **optimized** CNECs, in `unit`.
///
/// The filter is the whole point. A monitored CNEC is not something the
/// optimizer improves — it is something it must not ruin — so letting its
/// margin set the minimum makes the optimizer spend actions on a branch nobody
/// asked it to improve, and, worse, makes an MNEC that starts overloaded look
/// like the binding constraint for the entire perimeter. The reference filters
/// the same way and in the same place, in
/// `SumMaxPerTimestampCostEvaluatorResult.getCost`.
///
/// `INFINITY` when the perimeter optimizes nothing at all — see
/// [`objective`] for what that turns into when it has to be a number.
fn margin(
    crac: &Crac,
    network: &NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    unit: ObjectiveUnit,
    model: FlowModel,
) -> f64 {
    let view = network.view();
    let result = evaluate_in(crac, &view, resolution, model);
    margin_of(crac, &result, perimeter, unit)
}

/// The minimum margin over the perimeter's optimized CNECs of an assessment
/// already made.
fn margin_of(
    crac: &Crac,
    result: &super::evaluate::SecurityResult,
    perimeter: &[State],
    unit: ObjectiveUnit,
) -> f64 {
    result
        .perimeters
        .iter()
        .filter(|p| perimeter.contains(&p.state))
        .flat_map(|p| p.cnecs.iter())
        .filter(|c| crac.flow_cnecs[c.cnec].optimized)
        .map(|c| match unit {
            ObjectiveUnit::Megawatt => c.margin_mw,
            ObjectiveUnit::Ampere => c.margin_a,
        })
        .fold(f64::INFINITY, f64::min)
}

/// What the optimizer actually maximizes: the minimum margin, less what the
/// monitored CNECs are costing.
///
/// One evaluation serves both halves, which matters under an AC flow model
/// where an evaluation is a Newton-Raphson solve per state.
///
/// A perimeter with nothing optimized has no minimum margin, and the honest
/// infinity is unusable here: adding a finite penalty to it would lose the
/// penalty and leave a pure-MNEC perimeter unable to tell its candidates apart.
/// [`NO_CNEC_MARGIN`] is the reference's own finite stand-in.
fn objective(
    crac: &Crac,
    network: &NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    options: &LinearOptions,
) -> f64 {
    let view = network.view();
    let result = evaluate_in(crac, &view, resolution, options.flow_model);
    let unit = options.objective_unit;
    let margin = margin_of(crac, &result, perimeter, unit);
    let margin = if margin.is_finite() { margin } else { NO_CNEC_MARGIN };
    margin - options.mnec.cost(crac, &result, perimeter, unit)
}

/// Flows on the perimeter's CNEC branches, MW, at the current operating point.
fn perimeter_flows(
    crac: &Crac,
    network: &NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    cnecs: &[usize],
    model: FlowModel,
) -> (Vec<f64>, Vec<(f64, f64, f64)>) {
    let view = network.view();
    let result = evaluate_in(crac, &view, resolution, model);
    cnecs
        .iter()
        .map(|&i| {
            result
                .perimeters
                .iter()
                .filter(|p| perimeter.contains(&p.state))
                .find_map(|p| p.cnecs.iter().find(|c| c.cnec == i))
                .map(|c| (c.flow_mw, (c.lower_mw, c.upper_mw, c.amperes_per_mw())))
                .unwrap_or((0.0, (f64::NEG_INFINITY, f64::INFINITY, 0.0)))
        })
        .unzip()
}

/// Base-case DC flows, exposed for callers that want the operating point
/// without going through a full evaluation.
pub fn base_flows(network: &Network<'_>) -> Vec<f64> {
    let mut buses = network.buses.to_vec();
    dc_power_flow(&mut buses, network.lines, network.transformers, DcOptions::default()).branch_p
}
