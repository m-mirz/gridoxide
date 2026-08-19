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
//! **MNECs are not constrained.** A monitored CNEC's margin must not get worse,
//! which is a penalized soft constraint rather than something to maximize. This
//! layer currently ignores them entirely: they are excluded from the objective,
//! correctly, but nothing stops an action from degrading one.
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
use super::evaluate::{evaluate, Network, Resolution};

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
}

impl Default for LinearOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            pst_penalty: 0.01,
            injection_penalty: 0.001,
            sensitivity_threshold: 1e-6,
            tap_model: TapModel::Continuous,
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
    /// Minimum margin over the perimeter's optimized CNECs, MW, before and
    /// after.
    pub initial_margin_mw: f64,
    pub final_margin_mw: f64,
    /// Outer iterations actually run.
    pub iterations: usize,
}

impl LinearResult {
    pub fn improvement(&self) -> f64 {
        self.final_margin_mw - self.initial_margin_mw
    }
}

/// A phase shifter's tap-to-angle table, in the **CRAC's** sign convention.
///
/// Prefers the table the CRAC carries and falls back to the network's own,
/// negating it: gridoxide's transformer angle is the negation of IIDM's, which
/// is what a CRAC is written in.
///
/// The fallback is not a nicety. A CRAC's PST range action frequently omits the
/// table — it is a property of the transformer, and a CRAC written against a
/// network that already describes it has no reason to repeat it — and without
/// the fallback such an action has no positions to choose between and is
/// skipped without a word.
pub fn tap_table(
    declared: &[(i32, f64)],
    tap_changers: &[Option<crate::types::TapChanger>],
    branch: usize,
    lines: usize,
) -> Vec<(i32, f64)> {
    if !declared.is_empty() {
        let mut table = declared.to_vec();
        table.sort_by_key(|(t, _)| *t);
        return table;
    }
    let Some(changer) = branch.checked_sub(lines).and_then(|i| tap_changers.get(i)?.as_ref())
    else {
        return Vec::new();
    };
    (changer.low..=changer.high())
        .filter_map(|tap| changer.angle_deg(tap).map(|a| (tap, CRAC_ANGLE_SIGN * a)))
        .collect()
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
    let dc_options = DcOptions::default();

    // The CNECs this perimeter optimizes, and the flows they start from.
    let cnec_indices: Vec<usize> = crac
        .flow_cnecs
        .iter()
        .enumerate()
        .filter(|(_, c)| perimeter.contains(&c.state) && c.optimized)
        .filter(|(_, c)| resolution.branch(&c.network_element).is_some())
        .map(|(i, _)| i)
        .collect();

    let mut best = measure(crac, network, resolution, perimeter);
    let initial_margin = best;
    let mut setpoints: Vec<Setpoint> = Vec::new();
    let mut iterations = 0;

    if cnec_indices.is_empty() {
        return LinearResult {
            status: LinearStatus::NoImprovement,
            setpoints,
            initial_margin_mw: initial_margin,
            final_margin_mw: best,
            iterations,
        };
    }

    // Where every control started, so `Setpoint::initial` is the pre-optimization
    // value rather than the previous iteration's.
    let mut controls = match build_controls(crac, network, resolution, perimeter, dc_options, options) {
        Some(c) if !c.is_empty() => c,
        _ => {
            return LinearResult {
                status: LinearStatus::NoImprovement,
                setpoints,
                initial_margin_mw: initial_margin,
                final_margin_mw: best,
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
            perimeter_flows(crac, network, resolution, perimeter, &cnec_indices);
        let program = build_program(&cnec_indices, &flows, &limits, &controls, options);
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
        let mut best_here = measure(crac, network, resolution, perimeter);
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
                let margin = measure(crac, network, resolution, perimeter);
                if margin > best_here + 1e-9 {
                    best_here = margin;
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

        let margin = measure(crac, network, resolution, perimeter);
        if margin > best + 1e-9 {
            best = margin;
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
                build_controls(crac, network, resolution, perimeter, dc_options, options)
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
        final_margin_mw: best,
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
    pub tap_changers: &'a [Option<crate::types::TapChanger>],
    pub base_mva: f64,
}

impl NetworkMut<'_> {
    fn view(&self) -> Network<'_> {
        Network {
            buses: self.buses,
            lines: self.lines,
            transformers: self.transformers,
            branch_ids: self.branch_ids,
            bus_ids: self.bus_ids,
            initially_open: self.initially_open,
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

fn build_program(
    cnecs: &[usize],
    flows: &[f64],
    limits: &[f64],
    controls: &[Control],
    options: &LinearOptions,
) -> LinearProgram {
    let n = controls.len();
    let mm = margin_column(n);
    let mut lp = LinearProgram::new(3 * n + 1);

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

    for position in 0..cnecs.len() {
        let reference = flows[position];
        let limit = limits[position];
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

        for upper in [true, false] {
            // MM ≤ limit − F  (upper) or MM ≤ F + limit (lower), with
            // F = constant + Σ σ·A. Both directions always, because the margin
            // being maximized is against |F|.
            let mut coefficients: Vec<(usize, f64)> = Vec::with_capacity(terms.len() + 1);
            coefficients.push((mm, 1.0));
            let sign = if upper { 1.0 } else { -1.0 };
            for &(column, s) in &terms {
                coefficients.push((column, sign * s));
            }
            let bound = if upper { limit - constant } else { constant + limit };
            lp.add_row(&coefficients, f64::NEG_INFINITY, bound);
        }
    }
    lp
}

fn build_controls(
    crac: &Crac,
    network: &NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    dc_options: DcOptions,
    options: &LinearOptions,
) -> Option<Vec<Control>> {
    let cnecs: Vec<usize> = crac
        .flow_cnecs
        .iter()
        .enumerate()
        .filter(|(_, c)| perimeter.contains(&c.state) && c.optimized)
        .filter(|(_, c)| resolution.branch(&c.network_element).is_some())
        .map(|(i, _)| i)
        .collect();
    let cnec_branches: Vec<usize> = cnecs
        .iter()
        .map(|&i| resolution.branch(&crac.flow_cnecs[i].network_element).unwrap())
        .collect();

    let branches = dc_branches(network.lines, network.transformers, dc_options);
    let sensitivity =
        DcSensitivity::new(network.buses, &branches, network.lines.len() + network.transformers.len())?;

    let mut controls = Vec::new();
    for (index, action) in crac.range_actions.iter().enumerate() {
        if !perimeter.iter().any(|s| action.usage_rules.iter().any(|r| r.covers(s))) {
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
                let Some(column) = phase_shift_sensitivity(&sensitivity, &branches, branch) else {
                    continue;
                };
                // MW per degree *of the CRAC's angle*: the column is per-unit
                // per radian of gridoxide's, so the sign flips with it.
                let scale =
                    CRAC_ANGLE_SIGN * network.base_mva * std::f64::consts::PI / 180.0;
                let sensitivity_mw: Vec<f64> =
                    cnec_branches.iter().map(|&b| column.get(b).copied().unwrap_or(0.0) * scale).collect();

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
                let tap = table
                    .iter()
                    .min_by(|a, b| (a.1 - live_angle).abs().total_cmp(&(b.1 - live_angle).abs()))
                    .map(|(t, _)| *t)
                    .unwrap_or(*initial_tap);
                let current = table
                    .iter()
                    .find(|(t, _)| *t == tap)
                    .map(|(_, a)| *a)
                    .unwrap_or(live_angle);

                let (lower, upper) = tap_bounds(action, &table, *initial_tap);
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
                let mut injections = vec![0.0; sensitivity.n_buses()];
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
                let Some((_, column)) = sensitivity.response(&injections) else { continue };
                let sensitivity_mw: Vec<f64> =
                    cnec_branches.iter().map(|&b| column.get(b).copied().unwrap_or(0.0)).collect();
                let (lower, upper) = standard_bounds(action);
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

/// Angle bounds from a PST range action's tap ranges.
fn tap_bounds(
    action: &super::crac::RangeAction,
    table: &[(i32, f64)],
    initial_tap: i32,
) -> (f64, f64) {
    let (mut low, mut high) = (i32::MIN, i32::MAX);
    for range in &action.ranges {
        let (min, max) = match range.kind {
            super::crac::RangeKind::RelativeToInitialNetwork => (
                range.min.map(|m| initial_tap + m as i32),
                range.max.map(|m| initial_tap + m as i32),
            ),
            _ => (range.min.map(|m| m as i32), range.max.map(|m| m as i32)),
        };
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
                if let Some(transformer) = network.transformers.get_mut(i) {
                    // Keep the ratio, replace the shift — converting out of the
                    // CRAC's convention on the way.
                    let ratio = transformer.tap.norm();
                    transformer.tap = num_complex::Complex::from_polar(
                        ratio,
                        (CRAC_ANGLE_SIGN * value).to_radians(),
                    );
                }
            }
        }
    }
}

/// The minimum margin over the perimeter's optimized CNECs, MW.
fn measure(
    crac: &Crac,
    network: &NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
) -> f64 {
    let view = network.view();
    let result = evaluate(crac, &view, resolution);
    result
        .perimeters
        .iter()
        .filter(|p| perimeter.contains(&p.state))
        .filter_map(|p| p.min_margin())
        .fold(f64::INFINITY, f64::min)
}

/// Flows on the perimeter's CNEC branches, MW, at the current operating point.
fn perimeter_flows(
    crac: &Crac,
    network: &NetworkMut<'_>,
    resolution: &Resolution,
    perimeter: &[State],
    cnecs: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let view = network.view();
    let result = evaluate(crac, &view, resolution);
    cnecs
        .iter()
        .map(|&i| {
            result
                .perimeters
                .iter()
                .filter(|p| perimeter.contains(&p.state))
                .find_map(|p| p.cnecs.iter().find(|c| c.cnec == i))
                .map(|c| (c.flow_mw, c.limit_mw))
                .unwrap_or((0.0, f64::INFINITY))
        })
        .unzip()
}

/// Base-case DC flows, exposed for callers that want the operating point
/// without going through a full evaluation.
pub fn base_flows(network: &Network<'_>) -> Vec<f64> {
    let mut buses = network.buses.to_vec();
    dc_power_flow(&mut buses, network.lines, network.transformers, DcOptions::default()).branch_p
}
