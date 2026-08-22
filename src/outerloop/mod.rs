//! Outer loops: the controls that sit *around* a Newton-Raphson solve.
//!
//! A Newton solve answers "what is the state, given these injections, these
//! bus types and these tap positions". Several real controls are not
//! expressible that way, because they decide one of those *inputs* from the
//! solved state: a generator that runs out of reactive capability stops
//! holding its voltage, an imbalance is shared across governors rather than
//! dumped on one slack, an on-load tap changer moves until its controlled bus
//! is inside a deadband. Each is a loop around the solve, not a term inside
//! it.
//!
//! # Why this module exists
//!
//! gridoxide grew two such loops as standalone entry points —
//! `newton_raphson_enforcing_q_limits` and
//! `newton_raphson_distributing_slack` — each constructing its own
//! [`PersistentSolver`](crate::solver::PersistentSolver) and running its own
//! pass counter. The consequence was that **they could not be used together**:
//! a caller picked one. Real transmission networks want both, and tap control
//! would have been a third.
//!
//! So the loops are [`OuterLoop`] implementations now, and
//! [`solve_with_loops`] drives an ordered list of them.
//!
//! # Ordering is a physical claim
//!
//! The list is **innermost first**. Each loop is run to its own stability
//! before the next is consulted, and any loop that moves something sends the
//! driver back to the start of the list, so an outer loop's decision is
//! re-examined by the inner ones. [`solve_with_loops`] documents the schedule;
//! [`default_loops`] gives the order powsybl-open-loadflow uses, which encodes
//! that generators respond faster than tap changers.
//!
//! # Invalidation is the driver's job, never a loop's
//!
//! Three things must happen together after a tap move — write
//! [`Transformer::tap`](crate::types::Transformer::tap), restamp the Y-bus
//! entries derived from it, drop the cached Jacobian — and a loop that does
//! one or two of them is a bug that surfaces as a wrong answer on the *next*
//! pass rather than as a failure on this one. A loop therefore declares what
//! it invalidated ([`Invalidates`]) and the driver acts on it.

use crate::network::{build_ybus, stamp_shunts, YBusSparse};
use crate::solver::{IslandReport, IslandStatus, JacobianBackend, PersistentSolver};
use crate::network::ShuntAdm;
use crate::types::{Bus, Line, TapChanger, Transformer};

pub mod qlimits;
pub mod slack;
pub mod taps;

pub use qlimits::{QLimitReport, ReactiveLimits};
pub use slack::{DistributedSlack, SlackDistribution, SlackDistributionReport};
pub use taps::{
    ControllerOutcome, ControllerReport, PhaseControl, RegulationMode, TapControlReport,
    TapRegulation, TransformerVoltageControl,
};

/// What a loop decided after looking at a converged state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OuterLoopStatus {
    /// Nothing to do — this loop's criterion is met.
    Stable,
    /// Something was changed; the network needs re-solving.
    Unstable,
    /// This loop cannot proceed, and the whole run should stop.
    Failed(String),
}

/// What a loop invalidated when it reported [`OuterLoopStatus::Unstable`].
///
/// Returned by [`OuterLoop::invalidates`] and acted on by the driver — see the
/// module docs for why the loop does not act on it itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Invalidates {
    /// Only bus *values* moved (`p_spec`, `q_spec`). The Jacobian's pattern and
    /// its cached admittances are both still valid. Distributed slack.
    Nothing,
    /// `bus_type` moved, so `n_unknowns` and the Jacobian's sparsity pattern
    /// did too. Forces a full [`PersistentSolver::reset`]. Reactive limits.
    Pattern,
    /// [`Transformer::tap`] moved, so the Y-bus *values* did. The pattern
    /// survives, so the symbolic factorization is kept and only the cached
    /// admittances are re-derived — the same position a switch flip is in.
    /// Both tap loops.
    Admittances,
}

/// The mutable network a loop is allowed to act on, plus the last solve's
/// result.
///
/// powsybl passes a network object here; gridoxide has none — the
/// parallel-`Vec` convention is crate-wide — so this bundle of borrows is the
/// honest translation.
pub struct OuterLoopContext<'a, 'n> {
    pub net: &'a mut SolveContext<'n>,
    /// The islands the last inner solve reported. Never empty when a loop's
    /// `check` is called: the driver stops before that if the solve did not
    /// settle.
    pub islands: &'a [IslandReport],
    /// How many times *this* loop has already reported `Unstable`.
    pub iteration: usize,
}

/// The network a solve runs against.
///
/// Constructed once by the caller and passed to [`solve_with_loops`]. The
/// branch and tap fields are optional in the sense that a caller with no tap
/// control simply never supplies them — [`SolveContext::new`] leaves them
/// empty, and only [`Invalidates::Admittances`] loops read them.
pub struct SolveContext<'a> {
    pub buses: &'a mut [Bus],
    pub lines: &'a [Line],
    pub transformers: &'a mut [Transformer],
    pub shunts: &'a [ShuntAdm],
    /// Parallel to `transformers`. Empty when the importer retained no tap
    /// tables, which is what makes tap control silently inapplicable rather
    /// than an error.
    pub tap_changers: &'a mut [Option<TapChanger>],
    /// The regulating controls read from the network document.
    pub regulation: &'a [TapRegulation],
    pub ybus: &'a mut YBusSparse,
}

impl<'a> SolveContext<'a> {
    /// Buses and a Y-bus: everything a loop that never moves a tap needs.
    pub fn new(buses: &'a mut [Bus], ybus: &'a mut YBusSparse) -> Self {
        Self {
            buses,
            lines: &[],
            transformers: &mut [],
            shunts: &[],
            tap_changers: &mut [],
            regulation: &[],
            ybus,
        }
    }

    /// The branch data the Y-bus is built from. Required before any tap loop
    /// can restamp it.
    pub fn with_branches(
        mut self,
        lines: &'a [Line],
        transformers: &'a mut [Transformer],
        shunts: &'a [ShuntAdm],
    ) -> Self {
        self.lines = lines;
        self.transformers = transformers;
        self.shunts = shunts;
        self
    }

    /// The tap tables and the regulating controls that act on them.
    pub fn with_taps(
        mut self,
        tap_changers: &'a mut [Option<TapChanger>],
        regulation: &'a [TapRegulation],
    ) -> Self {
        self.tap_changers = tap_changers;
        self.regulation = regulation;
        self
    }

    /// Rebuild the Y-bus from the current branch data.
    ///
    /// Called by the driver after an [`Invalidates::Admittances`] loop moved a
    /// tap. A tap change alters admittance *values* and not the sparsity
    /// pattern, so this produces a structurally identical matrix and the
    /// symbolic factorization stays valid.
    ///
    /// Does nothing when the caller supplied no branch data, since there is
    /// then nothing to rebuild *from* — a tap loop cannot be configured in
    /// that case either, so this is unreachable rather than lossy.
    pub fn restamp_ybus(&mut self) {
        if self.lines.is_empty() && self.transformers.is_empty() {
            return;
        }
        let n = self.ybus.n();
        let mut y = build_ybus(n, self.lines, self.transformers);
        stamp_shunts(&mut y, self.shunts);
        *self.ybus = y.finish();
    }
}

/// What a loop is allowed to do to a solve.
///
/// Deliberately narrower than powsybl's `OuterLoop`, which is ServiceLoader-
/// discovered so third parties can register their own. That is a
/// Java-ecosystem affordance; here the list is built by the crate, and the
/// abstraction exists to make the loops *compose*, not to be extended from
/// outside.
pub trait OuterLoop {
    fn name(&self) -> &'static str;

    /// What this loop invalidates when it reports
    /// [`OuterLoopStatus::Unstable`]. Read by the driver, which then resets
    /// or restamps as required. See the module docs.
    fn invalidates(&self) -> Invalidates {
        Invalidates::Nothing
    }

    /// Called once, before the first solve, with the network in its initial
    /// state. The place to record starting tap positions or bus types.
    fn initialize(&mut self, _ctx: &mut OuterLoopContext<'_, '_>) {}

    /// Look at the converged state and either accept it or change something.
    fn check(&mut self, ctx: &mut OuterLoopContext<'_, '_>) -> OuterLoopStatus;
}

/// Per-loop accounting from one [`solve_with_loops`] run.
#[derive(Clone, Debug)]
pub struct LoopReport {
    pub name: &'static str,
    /// How many times this loop reported `Unstable`, i.e. how many re-solves
    /// it caused.
    pub iterations: usize,
    /// The status it finished on.
    pub status: OuterLoopStatus,
}

/// What the whole outer-loop run did.
#[derive(Clone, Debug)]
pub struct OuterLoopReport {
    pub loops: Vec<LoopReport>,
    /// Re-solves across every loop.
    pub total_iterations: usize,
    /// Inner Newton solves run, including the first one.
    pub solves: usize,
    /// True when every loop reported `Stable` on the final pass. False when
    /// the budget ran out, a loop failed, or an inner solve stopped
    /// converging.
    pub converged: bool,
    /// Set when the run stopped because `budget` was reached.
    pub budget_exhausted: bool,
}

impl OuterLoopReport {
    /// The report of the named loop, if it ran.
    pub fn loop_named(&self, name: &str) -> Option<&LoopReport> {
        self.loops.iter().find(|l| l.name == name)
    }
}

/// The order powsybl-open-loadflow applies by default, restricted to the loops
/// gridoxide has: distributed slack innermost, then reactive limits, then the
/// two tap controls.
///
/// The ordering is a physical claim rather than a convention — generators
/// respond faster than tap changers, so a tap should be chosen against a
/// reactive dispatch that has already settled, and re-examined when the tap
/// move disturbs it.
/// The loops are passed as `&mut dyn` rather than owned boxes so a caller
/// keeps its concrete values and can read their reports — `DistributedSlack`'s
/// shifts, `ReactiveLimits`' switches, a tap loop's per-controller outcomes —
/// after the run, without downcasting.
pub fn ordered<'a>(
    slack: Option<&'a mut DistributedSlack>,
    qlimits: Option<&'a mut ReactiveLimits>,
    phase: Option<&'a mut PhaseControl>,
    voltage: Option<&'a mut TransformerVoltageControl>,
) -> Vec<&'a mut dyn OuterLoop> {
    let mut loops: Vec<&mut dyn OuterLoop> = Vec::new();
    if let Some(l) = slack {
        loops.push(l);
    }
    if let Some(l) = qlimits {
        loops.push(l);
    }
    if let Some(l) = phase {
        loops.push(l);
    }
    if let Some(l) = voltage {
        loops.push(l);
    }
    loops
}

fn settled(reports: &[IslandReport]) -> bool {
    reports
        .iter()
        .all(|r| !matches!(r.status, IslandStatus::Singular | IslandStatus::MaxIterationsReached))
}

/// Solve, then run an ordered list of outer loops to a fixed point.
///
/// # The schedule
///
/// Transcribed from powsybl-open-loadflow's `AcloadFlowEngine`, because it is
/// not the obvious one:
///
/// - An initial solve runs first. If it does not settle, no loop is consulted
///   at all — there is nothing useful a control can decide from a state that
///   is not a power flow.
/// - The loops are **nested, innermost first**. Each is run to *its own*
///   stability (an inner `check`/re-solve loop) before the next is consulted.
/// - Any loop reporting `Unstable` re-solves and becomes the "last unstable"
///   loop. The driver then continues down the list, and wraps back to the
///   start.
/// - Termination is reaching the last-unstable loop again having changed
///   nothing on the way: the whole list has been walked and every criterion
///   holds simultaneously, which is what a fixed point of the combined
///   controls means.
///
/// `budget` caps total re-solves across every loop, as one shared figure
/// rather than a per-loop count. An empty `loops` makes this exactly one
/// ordinary solve.
pub fn solve_with_loops(
    ctx: &mut SolveContext<'_>,
    tol: f64,
    max_iter: usize,
    backend: JacobianBackend,
    loops: &mut [&mut dyn OuterLoop],
    budget: usize,
) -> (Vec<IslandReport>, OuterLoopReport) {
    let mut solver = PersistentSolver::new(backend);
    let mut counts = vec![0usize; loops.len()];
    let mut statuses = vec![OuterLoopStatus::Stable; loops.len()];

    {
        let mut lc = OuterLoopContext { net: ctx, islands: &[], iteration: 0 };
        for l in loops.iter_mut() {
            l.initialize(&mut lc);
        }
    }

    let mut reports = solver.solve(ctx.buses, ctx.ybus, tol, max_iter);
    let mut solves = 1usize;
    let mut total = 0usize;
    let mut budget_exhausted = false;

    let finish = |reports: Vec<IslandReport>,
                  loops: &[&mut dyn OuterLoop],
                  counts: &[usize],
                  statuses: &[OuterLoopStatus],
                  total: usize,
                  solves: usize,
                  converged: bool,
                  budget_exhausted: bool| {
        let report = OuterLoopReport {
            loops: loops
                .iter()
                .enumerate()
                .map(|(i, l)| LoopReport {
                    name: l.name(),
                    iterations: counts[i],
                    status: statuses[i].clone(),
                })
                .collect(),
            total_iterations: total,
            solves,
            converged,
            budget_exhausted,
        };
        (reports, report)
    };

    if loops.is_empty() || !settled(&reports) {
        let ok = settled(&reports);
        return finish(reports, loops, &counts, &statuses, total, solves, ok, false);
    }

    // Index of the loop that most recently reported `Unstable`. Reaching it
    // again with nothing changed in between is the termination condition.
    let mut last_unstable: Option<usize> = None;
    let mut failed = false;

    'outer: loop {
        let before = total;

        for i in 0..loops.len() {
            if Some(i) == last_unstable {
                // Walked the whole list; nothing else moved. Done.
                break 'outer;
            }
            if total >= budget {
                budget_exhausted = true;
                break 'outer;
            }

            // Run this loop to its own stability before moving on.
            loop {
                let status = {
                    let mut lc = OuterLoopContext {
                        net: ctx,
                        islands: &reports,
                        iteration: counts[i],
                    };
                    loops[i].check(&mut lc)
                };
                statuses[i] = status.clone();

                match status {
                    OuterLoopStatus::Stable => break,
                    OuterLoopStatus::Failed(_) => {
                        failed = true;
                        break 'outer;
                    }
                    OuterLoopStatus::Unstable => {
                        match loops[i].invalidates() {
                            Invalidates::Nothing => {}
                            Invalidates::Pattern => solver.reset(),
                            Invalidates::Admittances => {
                                ctx.restamp_ybus();
                                solver.invalidate_admittances();
                            }
                        }
                        reports = solver.solve(ctx.buses, ctx.ybus, tol, max_iter);
                        solves += 1;
                        counts[i] += 1;
                        total += 1;
                        last_unstable = Some(i);

                        if !settled(&reports) {
                            break 'outer;
                        }
                        if total >= budget {
                            budget_exhausted = true;
                            break 'outer;
                        }
                    }
                }
            }
        }

        // A full pass that changed nothing: every criterion holds.
        if total == before {
            break;
        }
    }

    let converged = !failed
        && !budget_exhausted
        && settled(&reports)
        && statuses.iter().all(|s| *s == OuterLoopStatus::Stable);
    finish(reports, loops, &counts, &statuses, total, solves, converged, budget_exhausted)
}
