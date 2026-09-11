//! The cost objective: `MIN_COST`.
//!
//! Max-min-margin asks *how much room can I buy*. This asks *what is the
//! cheapest plan that leaves no overload*, and the two are different questions
//! rather than the same one measured differently — scenario 3.4.1.2 in the
//! reference's own corpus is built to show it, naming a network action that
//! yields a **smaller** margin than its rival and is chosen anyway because it
//! costs less.
//!
//! \\[
//!   \text{cost} \;=\; p \sum_{c} \max(0,\; \tau - m(c)) \;+\; \sum_{a \in A} k(a)
//! \\]
//!
//! with \\(p\\) the violation penalty, \\(\tau\\) the threshold a margin must
//! clear, \\(m(c)\\) each optimized CNEC's margin, and \\(k(a)\\) what each
//! action in force cost to activate.
//!
//! Two things about that formula drive everything here.
//!
//! **It is a sum, not a min.** [`margin_of`](super::linear) folds with
//! `f64::min`, so relieving an already-comfortable CNEC is worth nothing; here
//! every overloaded CNEC contributes, and relieving any of them helps. That is
//! why this is a separate branch of the objective rather than a penalty added
//! to the existing one.
//!
//! **It depends on which actions were taken**, which no margin does. The
//! evaluation kernel cannot see that — it is handed a network, and a network
//! does not remember how it got that way — so the set has to be carried
//! alongside. [`LinearOptions::activated`](super::linear::LinearOptions) is
//! where it travels.
//!
//! gridoxide maximizes throughout: the search tree, the LP, the stop criteria
//! and `improved_enough` all read higher as better. So a cost is returned
//! **negated** by the objective functions that use this module, and every
//! comparison downstream keeps working unchanged. This module itself deals in
//! honest, positive costs; the sign flip belongs to the caller, and is written
//! once in each of the two places that compute an objective.

use super::crac::{Crac, InstantKind, State};
use super::evaluate::SecurityResult;
use super::linear::ObjectiveUnit;

/// `costly-min-margin-parameters`, the reference's own block.
///
/// Both are in the **objective's unit** — megawatts for a DC run, amperes for
/// an AC one — for the same reason [`MnecOptions`](super::mnec::MnecOptions)
/// is: it is the unit the reference writes the constraint in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostlyOptions {
    /// `shifted-violation-penalty`: what one unit of overload costs. The
    /// vendored configurations all say 1000.0, which is what makes a 250 MW
    /// overload dominate a 10-unit activation and gives 3.4.1.1 its 250000.
    pub violation_penalty: f64,
    /// `shifted-violation-threshold`: the margin a CNEC must clear to count as
    /// unviolated. Zero — the threshold itself — in every `epic92` config; the
    /// `epic93` one says 10, which demands a 10-unit cushion.
    pub violation_threshold: f64,
}

impl Default for CostlyOptions {
    fn default() -> Self {
        Self { violation_penalty: 1000.0, violation_threshold: 0.0 }
    }
}

/// The cost rule, ready to apply.
///
/// Mirrors [`Mnec`](super::mnec::Mnec): the knobs, plus whatever the rule needs
/// that is not on the network. Unlike `Mnec` it needs no baseline — a cost is
/// absolute, where an MNEC violation is measured against where the CNEC started.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Costly {
    pub options: CostlyOptions,
}

impl Costly {
    /// The penalized overload over `perimeter`'s **optimized** CNECs: a sum
    /// within each optimization perimeter, and the **worst** of those.
    ///
    /// Within a perimeter it is a sum. A CNEC clearing the threshold
    /// contributes nothing rather than a negative amount — an unused margin is
    /// not a credit, and treating it as one would let a comfortable CNEC pay
    /// for an overloaded one. That sum is the objective's whole point and is
    /// what the LP minimizes.
    ///
    /// **Across perimeters it is a maximum**, which is the part that is not
    /// obvious. Each perimeter is optimized on its own network by its own
    /// actions, and the reference combines their functional costs the way it
    /// combines margins — a max over the per-perimeter results, the costly
    /// objective reusing the aggregation the max-min-margin one needs. Summing
    /// instead reports a plan as five times worse for having five
    /// contingencies: the reference's 3.4.1.11 spreads 500 units of overload
    /// across nine states and states its cost as **100**, the worst single one.
    ///
    /// A "perimeter" here is the reference's. The base case and its outage
    /// states are optimized together and count as one; every other state is its
    /// own, which is what makes each contingency's auto and curative instants
    /// separate buckets rather than one per contingency.
    ///
    /// One thing the vendored corpus cannot settle: every min-cost scenario in
    /// it has exactly **one** overloaded CNEC per state, so a plain
    /// worst-margin-over-everything rule reproduces all 294 assertions just as
    /// well. The sum-within-perimeter reading is the one chosen, because it is
    /// what the LP and the search demonstrably need and two unrelated
    /// definitions of one cost would be worse than one imperfectly pinned.
    pub fn violation(
        &self,
        crac: &Crac,
        result: &SecurityResult,
        perimeter: &[State],
        unit: ObjectiveUnit,
    ) -> f64 {
        // `None` rather than 0.0 so a perimeter list with no states at all is
        // distinguishable from one whose states are all comfortable. Both
        // answer zero here, but folding a max from 0.0 would silently floor a
        // future threshold that makes a *negative* contribution meaningful.
        let mut preventive: Option<f64> = None;
        let mut worst: Option<f64> = None;
        for p in result.perimeters.iter().filter(|p| perimeter.contains(&p.state)) {
            let total: f64 = p
                .cnecs
                .iter()
                .filter(|c| crac.flow_cnecs[c.cnec].optimized)
                .map(|c| {
                    let margin = match unit {
                        ObjectiveUnit::Megawatt => c.margin_mw,
                        ObjectiveUnit::Ampere => c.margin_a,
                    };
                    (self.options.violation_threshold - margin).max(0.0)
                })
                .sum();
            match crac.instants[p.state.instant].kind {
                InstantKind::Preventive | InstantKind::Outage => {
                    preventive = Some(preventive.unwrap_or(0.0) + total);
                }
                InstantKind::Auto | InstantKind::Curative => {
                    worst = Some(worst.map_or(total, |w: f64| w.max(total)));
                }
            }
        }
        let total = match (preventive, worst) {
            (Some(a), Some(b)) => a.max(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => 0.0,
        };
        self.options.violation_penalty * total
    }

    /// What the actions in force cost: activation once, plus movement by the
    /// distance travelled.
    ///
    /// `activated` indexes [`Crac::network_actions`]. `moved` is
    /// `(range action, distance)` — **in the unit that action's variation cost
    /// is stated in**, which for a phase shifter is *taps*, not degrees. The
    /// reference's 3.4.2.1 pins that: an activation cost of 5 and a variation
    /// cost of 10 make five taps cost 55, where per-degree would make it 24.5.
    ///
    /// A distance of zero still pays the activation: an action chosen and then
    /// left where it stood is a decision someone has to carry out. An action the
    /// CRAC prices at nothing is **free**, not unknown — the field is optional
    /// in the format, and the reference reads its absence as zero rather than
    /// refusing to price the plan.
    pub fn activation(&self, crac: &Crac, activated: &[usize], moved: &[(usize, f64)]) -> f64 {
        let network: f64 = activated
            .iter()
            .filter_map(|&i| crac.network_actions.get(i))
            .filter_map(|a| a.activation_cost)
            .sum();
        let range: f64 = moved
            .iter()
            .filter_map(|&(i, distance)| Some((crac.range_actions.get(i)?, distance)))
            .map(|(a, distance)| {
                // Direction is the sign of the movement; the two prices are
                // separate because a CRAC may state them so.
                let variation = a.variation_cost.map_or(0.0, |c| {
                    if distance >= 0.0 { c.up * distance } else { c.down * -distance }
                });
                a.activation_cost.unwrap_or(0.0) + variation
            })
            .sum();
        network + range
    }

    /// The whole cost. Positive; the caller negates it.
    pub fn cost(
        &self,
        crac: &Crac,
        result: &SecurityResult,
        perimeter: &[State],
        unit: ObjectiveUnit,
        activated: &[usize],
        moved: &[(usize, f64)],
    ) -> f64 {
        self.violation(crac, result, perimeter, unit)
            + self.activation(crac, activated, moved)
    }
}
