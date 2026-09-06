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

use super::crac::{Crac, State};
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
    /// The penalized overload over the perimeter's **optimized** CNECs.
    ///
    /// A sum. A CNEC clearing the threshold contributes nothing rather than a
    /// negative amount — an unused margin is not a credit, and treating it as
    /// one would let a comfortable CNEC pay for an overloaded one.
    pub fn violation(
        &self,
        crac: &Crac,
        result: &SecurityResult,
        perimeter: &[State],
        unit: ObjectiveUnit,
    ) -> f64 {
        let total: f64 = result
            .perimeters
            .iter()
            .filter(|p| perimeter.contains(&p.state))
            .flat_map(|p| p.cnecs.iter())
            .filter(|c| crac.flow_cnecs[c.cnec].optimized)
            .map(|c| {
                let margin = match unit {
                    ObjectiveUnit::Megawatt => c.margin_mw,
                    ObjectiveUnit::Ampere => c.margin_a,
                };
                (self.options.violation_threshold - margin).max(0.0)
            })
            .sum();
        self.options.violation_penalty * total
    }

    /// What the actions in force cost to activate.
    ///
    /// `activated` indexes [`Crac::network_actions`], `moved` indexes
    /// [`Crac::range_actions`]. An action whose CRAC states no cost is **free**,
    /// not unknown: `activationCost` is optional in the format, and the
    /// reference treats its absence as zero rather than refusing to price the
    /// plan.
    pub fn activation(&self, crac: &Crac, activated: &[usize], moved: &[usize]) -> f64 {
        let network: f64 = activated
            .iter()
            .filter_map(|&i| crac.network_actions.get(i))
            .filter_map(|a| a.activation_cost)
            .sum();
        let range: f64 = moved
            .iter()
            .filter_map(|&i| crac.range_actions.get(i))
            .filter_map(|a| a.activation_cost)
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
        moved: &[usize],
    ) -> f64 {
        self.violation(crac, result, perimeter, unit)
            + self.activation(crac, activated, moved)
    }
}
