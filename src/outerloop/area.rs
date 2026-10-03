//! Area interchange control: holding each control area's net export at its
//! scheduled value.
//!
//! [`DistributedSlack`](super::DistributedSlack) drives one number to one
//! target — the slack's output to its own schedule. Interconnected systems
//! schedule more than that: each control area agrees a **net position** with
//! its neighbours, and the area's generators are dispatched to meet it. An
//! area importing 300 MW when it agreed to import 200 has every tie-line flow
//! around it wrong, for the same reason a single slack absorbing an unscheduled
//! few hundred megawatts does.
//!
//! # The formulation
//!
//! An area's **interchange** is the active power leaving it across its own
//! boundary branches, measured at its own side of each:
//!
//! \\[ X_a = \sum_{b \in \partial a} P_b^{(a\text{-side})} \\]
//!
//! and its mismatch is that against the schedule. The knob is the same one
//! distributed slack uses: the participating generators' `p_spec`.
//!
//! # One area's position is dependent, and it has to be
//!
//! The targets are **not independent**. Each area's side of a tie line is
//! measured at its own terminal, *into* the branch, so both ends of a tie
//! contribute positively and their sum is what the tie dissipates:
//!
//! \\[ \sum_a X_a = +\ell_{\text{tie}} \\]
//!
//! — a quantity nobody knows before the solve. (The sign is easy to get
//! backwards: it is not that one area's export is the other's import, because
//! the loss falls between them and belongs to neither.) Targets that sum to zero, which
//! is what a set of agreed net positions looks like, are therefore unachievable
//! by exactly the tie losses. Counting it out: the knobs are each area's
//! aggregate participant schedule, \\(N\\) of them; the conditions wanted are
//! \\(N\\) interchange targets plus the slack on its own schedule, \\(N+1\\).
//! Over-determined by one, always.
//!
//! So **the area holding the slack is the dependent one**: its own target is
//! not enforced, and its condition is that the slack produces its schedule
//! instead. Every other area's position is met exactly, and the slack's area
//! absorbs the tie losses — which is what a real interconnection does, and what
//! makes one area's position a residual rather than an agreement.
//! [`AreaInterchangeReport::dependent`] names it, and its
//! [`residual`](AreaInterchangeReport::residual) entry reports how far its
//! nominal target was missed, so the number is visible rather than implied.
//!
//! # Deriving the sign, which is the rest of the difficulty
//!
//! For an area that is not the slack's, an export above target means it should
//! generate less, so the mismatch \\(X_a - X_a^{target}\\) is **subtracted**
//! from the participants by weight.
//!
//! For the slack's area, write \\(\delta_s\\) for the slack's excess over its
//! own schedule. Its participants must take that excess up — exactly
//! [`DistributedSlack`](super::DistributedSlack)'s move — so the quantity
//! subtracted is \\(-\delta_s\\).
//!
//! Worth checking the degenerate case, because it is also a test
//! (`area_interchange_with_one_area_is_distributed_slack`): one area covering
//! the whole network *is* the slack's area, so the only condition is
//! \\(-\delta_s\\), and the loop becomes distributed slack exactly. Area
//! interchange **generalizes** it rather than sitting beside it, which is why
//! powsybl-open-loadflow's own `AcAreaInterchangeControlOuterLoop` constructs a
//! `DistributedSlackOuterLoop` as its no-area fallback and why the two must not
//! both be configured.

use crate::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use crate::network::{effective_injection, power_injections};
use crate::types::{Bus, BusType};

use super::{Invalidates, OuterLoop, OuterLoopContext, OuterLoopStatus};

/// Which buses belong to which control area, what each area is scheduled to
/// export, and who adjusts to get there.
#[derive(Clone, Debug)]
pub struct AreaDefinition {
    /// Area index per bus, indexed by [`Bus::idx`]. `None` means the bus is in
    /// no area — its injections are nobody's to schedule, and any branch it
    /// shares with an area counts as that area's boundary.
    pub of_bus: Vec<Option<usize>>,
    /// Scheduled net export per area, per-unit. **Positive is exporting**, the
    /// sign convention ENTSO-E net positions use.
    pub targets: Vec<f64>,
    /// Participation weight per bus. Normalized **within each area**, so only
    /// relative magnitudes matter and raw megawatt headroom can be passed.
    pub factors: Vec<f64>,
    /// Convergence tolerance on an area's remaining mismatch, per-unit.
    pub tolerance: f64,
    /// Cap on outer passes.
    pub max_outer_iter: usize,
}

impl AreaDefinition {
    /// Every generator bus — `Slack` and `PV` — participates equally within its
    /// own area, with every area scheduled for zero net export.
    ///
    /// A zero target is rarely what a real schedule says, but it is the
    /// meaningful default: it asks each area to serve its own load, which is
    /// the reading of "no interchange agreed".
    pub fn uniform(buses: &[Bus], of_bus: Vec<Option<usize>>, n_areas: usize) -> Self {
        let factors = buses
            .iter()
            .map(|b| match b.bus_type {
                BusType::Slack | BusType::PV => 1.0,
                BusType::PQ => 0.0,
            })
            .collect();
        Self {
            of_bus,
            targets: vec![0.0; n_areas],
            factors,
            tolerance: 1e-8,
            max_outer_iter: 30,
        }
    }

    /// Weighted by each bus's own scheduled generation, whatever its bus type.
    ///
    /// # Why this differs from [`uniform`](Self::uniform), and from distributed slack
    ///
    /// [`SlackDistribution::uniform`](super::SlackDistribution::uniform)
    /// participates `Slack` and `PV` buses only, on the grounds that a positive
    /// injection at a `PQ` bus is a fixed schedule rather than a machine under
    /// governor control. That reasoning is about **frequency** response, and it
    /// is right: a machine not on governor control does not pick up imbalance.
    ///
    /// A net position is not frequency response. It is met by **redispatch** —
    /// a scheduling action over the hour — and a generator held at a fixed
    /// active set-point is precisely the machine an operator redispatches.
    /// Excluding it makes whole areas uncontrollable: in
    /// `two_area_case.xiidm`, both of AREA2's generators are
    /// `voltageRegulatorOn="false"` and so arrive as `PQ` buses, leaving that
    /// area with no participant at all under `uniform` and its −400 MW
    /// schedule unreachable.
    ///
    /// So this weights by `p_spec.max(0.0)` at every bus. An area whose
    /// generation is all at fixed set-points is dispatchable; a load bus, with
    /// no positive injection, still is not.
    pub fn by_generation(buses: &[Bus], of_bus: Vec<Option<usize>>, n_areas: usize) -> Self {
        let factors = buses.iter().map(|b| b.p_spec.max(0.0)).collect();
        Self { of_bus, targets: vec![0.0; n_areas], factors, tolerance: 1e-8, max_outer_iter: 30 }
    }

    fn n_areas(&self) -> usize {
        self.targets.len()
    }
}

/// What [`AreaInterchange`] did.
#[derive(Clone, Debug, Default)]
pub struct AreaInterchangeReport {
    /// Per area, its measured net export at exit, per-unit.
    pub interchange: Vec<f64>,
    /// Per area, how far its interchange is from its target at exit. The
    /// [`dependent`](Self::dependent) area's entry is a *reported* residual
    /// rather than a driven one — see the module docs.
    pub residual: Vec<f64>,
    /// The area holding the slack, whose own target is not enforced because the
    /// system is over-determined by one. `None` when the slack is in no area,
    /// in which case nothing drives the slack to its schedule and
    /// [`unbalanced`](Self::unbalanced) says so.
    pub dependent: Option<usize>,
    /// Per bus, how much its active schedule moved, per-unit.
    pub shift: Vec<f64>,
    pub outer_iterations: usize,
    /// False if the pass cap was hit with an area still off its schedule, or
    /// if an inner solve stopped converging.
    pub converged: bool,
    /// Areas left alone, and why — no participating generator, or a boundary
    /// this network does not have.
    pub unbalanced: Vec<(usize, &'static str)>,
}

/// Drives every area's net export to its scheduled value, and the slack to its
/// own schedule, by moving the participating generators.
///
/// Reports [`Invalidates::Nothing`] for the same reason
/// [`DistributedSlack`](super::DistributedSlack) does: only `p_spec` moves, so
/// bus types, `n_unknowns` and the Jacobian's sparsity pattern all hold, and
/// the solver keeps its symbolic factorization across the whole loop.
///
/// **Do not configure this alongside `DistributedSlack`.** It subsumes it (see
/// the module docs), and running both would move the same schedules twice.
#[derive(Clone, Debug)]
pub struct AreaInterchange {
    areas: AreaDefinition,
    report: AreaInterchangeReport,
    exhausted: bool,
}

impl AreaInterchange {
    pub fn new(areas: AreaDefinition) -> Self {
        Self { areas, report: AreaInterchangeReport::default(), exhausted: false }
    }

    pub fn report(&self) -> &AreaInterchangeReport {
        &self.report
    }

    pub fn into_report(self) -> AreaInterchangeReport {
        self.report
    }

    /// Each area's net export at the current state, measured at the area's own
    /// side of every boundary branch.
    ///
    /// Public because it is useful on its own: "what is this area actually
    /// exchanging" is a question worth asking of a solved network without
    /// running a control loop over it.
    pub fn measure(
        buses: &[Bus],
        lines: &[crate::types::Line],
        transformers: &[crate::types::Transformer],
        of_bus: &[Option<usize>],
        n_areas: usize,
    ) -> Vec<f64> {
        let params = branch_params(lines, transformers);
        let v = bus_voltages(buses);
        let mut interchange = vec![0.0; n_areas];
        for bp in &params {
            let (a_from, a_to) = (of_bus[bp.from], of_bus[bp.to]);
            if a_from == a_to {
                // Internal to one area, or outside every area. Either way it
                // crosses no boundary.
                continue;
            }
            // Each side's own terminal flow is the power leaving that side's
            // area into the branch. Taking each independently — rather than one
            // and its negation — is what makes losses on the tie line itself
            // land on neither area, which is the convention a net position
            // uses.
            if let Some(a) = a_from {
                interchange[a] += terminal_flow(bp, Terminal::From, &v).0;
            }
            if let Some(a) = a_to {
                interchange[a] += terminal_flow(bp, Terminal::To, &v).0;
            }
        }
        interchange
    }
}

impl OuterLoop for AreaInterchange {
    fn name(&self) -> &'static str {
        "AreaInterchange"
    }

    fn invalidates(&self) -> Invalidates {
        Invalidates::Nothing
    }

    fn initialize(&mut self, ctx: &mut OuterLoopContext<'_, '_>) {
        let n = ctx.net.buses.len();
        assert_eq!(self.areas.of_bus.len(), n, "area assignment must carry one entry per bus");
        assert_eq!(self.areas.factors.len(), n, "participation factors must carry one weight per bus");
        self.report.shift = vec![0.0; n];
        self.report.interchange = vec![0.0; self.areas.n_areas()];
    }

    fn check(&mut self, ctx: &mut OuterLoopContext<'_, '_>) -> OuterLoopStatus {
        if self.exhausted {
            return OuterLoopStatus::Stable;
        }
        if ctx.net.lines.is_empty() && ctx.net.transformers.is_empty() {
            return OuterLoopStatus::Failed(
                "area interchange needs the branch lists; build the SolveContext with \
                 `with_branches`"
                    .into(),
            );
        }
        self.report.outer_iterations += 1;

        let n_areas = self.areas.n_areas();
        let interchange = Self::measure(
            ctx.net.buses,
            ctx.net.lines,
            ctx.net.transformers,
            &self.areas.of_bus,
            n_areas,
        );

        // The slack's excess over its own schedule, attributed to the area it
        // sits in. powsybl attributes it the same way; a slack outside every
        // area has nowhere to put it, and is reported rather than ignored.
        let (p_calc, _) = power_injections(ctx.net.buses, ctx.net.ybus);
        let mut slack_delta = vec![0.0; n_areas];
        let mut homeless_slack = false;
        // The first area found holding a slack. With several islands there are
        // several slacks; the first is enough, because an island with no
        // reference has nothing to schedule against anyway and one with its own
        // reference is balanced by its own participants.
        let mut dependent: Option<usize> = None;
        for island in ctx.islands {
            let [slack] = island.slack_indices.as_slice() else { continue };
            let delta = p_calc[*slack] - effective_injection(&ctx.net.buses[*slack]).0;
            match self.areas.of_bus[*slack] {
                Some(a) => {
                    slack_delta[a] += delta;
                    dependent = dependent.or(Some(a));
                }
                None => homeless_slack = homeless_slack || delta.abs() > self.areas.tolerance,
            }
        }

        let mut unbalanced = Vec::new();
        if homeless_slack {
            unbalanced.push((usize::MAX, "the slack is in no area; nothing drives it to its schedule"));
        }

        let mut residual = vec![0.0; n_areas];
        let mut worst = 0.0f64;
        // Collected first, applied after, so every area's mismatch is measured
        // against one consistent solved state.
        let mut updates: Vec<(usize, f64)> = Vec::new();

        for a in 0..n_areas {
            // The reported residual is always the same quantity — how far this
            // area's position is from what was asked — whether or not the loop
            // is driving it.
            residual[a] = interchange[a] - self.areas.targets[a];

            // What the loop actually drives. For the slack's area that is the
            // slack's own deviation, not its position: the position is the
            // dependent one, and it absorbs the tie losses. For every other
            // area it is the position. See the module docs for the count that
            // forces this.
            let driven = if dependent == Some(a) { -slack_delta[a] } else { residual[a] };

            let total: f64 = (0..ctx.net.buses.len())
                .filter(|&i| self.areas.of_bus[i] == Some(a))
                .map(|i| self.areas.factors[i].max(0.0))
                .sum();
            if total <= 0.0 {
                if driven.abs() > self.areas.tolerance {
                    unbalanced.push((a, "no participating generator in this area"));
                }
                continue;
            }
            worst = worst.max(driven.abs());
            for i in 0..ctx.net.buses.len() {
                if self.areas.of_bus[i] != Some(a) {
                    continue;
                }
                let weight = self.areas.factors[i].max(0.0);
                if weight > 0.0 {
                    updates.push((i, -weight / total * driven));
                }
            }
        }

        self.report.dependent = dependent;
        self.report.interchange = interchange;
        self.report.residual = residual;
        self.report.unbalanced = unbalanced;

        if worst <= self.areas.tolerance {
            self.report.converged = true;
            return OuterLoopStatus::Stable;
        }

        for (i, amount) in updates {
            ctx.net.buses[i].p_spec += amount;
            self.report.shift[i] += amount;
        }

        // The cap is spent on this redistribution: re-solve once more so the
        // returned state matches the final schedules, then accept it.
        if self.report.outer_iterations >= self.areas.max_outer_iter.max(1) {
            self.exhausted = true;
        }
        OuterLoopStatus::Unstable
    }
}
