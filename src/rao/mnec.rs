//! Monitored network elements — CNECs whose margin must not *get worse*.
//!
//! An MNEC is the other half of what a CRAC asks for. An optimized CNEC says
//! "make this better"; a monitored one says "whatever you do elsewhere, do not
//! ruin this". The two cannot both be maximized, because improving one branch
//! generally loads another, so the reference makes the second a **penalized
//! soft constraint** rather than a hard one: an MNEC may be pushed past its
//! threshold, and the optimizer pays for every megawatt (or ampere) of it.
//!
//! # What "not worse" means
//!
//! Not "the margin must stay where it was". The rule the reference implements
//! ([`MnecViolationCostEvaluator`][ref]) is
//!
//! \\[ v(c) \;=\; \max\bigl(0,\; \min(0,\; m_0(c) - d) \;-\; m(c)\bigr) \\]
//!
//! with \\(m_0\\) the margin the untouched network had, \\(m\\) the margin now,
//! and \\(d\\) the *acceptable decrease* — 50 by default, in whatever unit the
//! objective is measured in.
//!
//! The inner `min(0, …)` is the part worth reading twice. It says the floor an
//! MNEC is held to is **zero, or its own initial margin less `d`, whichever is
//! lower** — and the reference wrote one scenario per case:
//!
//! - \(m_0 \ge d\): the floor is zero. An MNEC with room to spare must simply
//!   stay non-negative, and its whole margin is available, not just `d` of it.
//!   (5.2.1.2, initial margin above 50 MW.)
//! - \(0 \le m_0 < d\): the floor is \(m_0 - d\), which is negative. An
//!   MNEC that starts close to its threshold may be pushed *through* it — by
//!   the part of `d` it had not already used up. (5.2.1.4.)
//! - \(m_0 < 0\): the floor is again \(m_0 - d\). An MNEC that starts
//!   overloaded may get `d` worse and no worse. It is not required to be
//!   repaired — that is what makes it monitored rather than optimized — but
//!   neither may the optimizer keep loading it. (5.2.1.3.)
//!
//! Reading the rule as a flat "no more than `d` worse" gets the first case
//! wrong by the entire initial margin; reading it as "never negative" gets the
//! other two wrong by `d`.
//!
//! [ref]: https://github.com/powsybl/powsybl-open-rao
//!
//! # Two places it has to be applied
//!
//! The penalty enters the objective **and** the LP, and both are needed:
//!
//! - The objective ([`Mnec::cost`]) is what ranks two candidate networks in the
//!   search tree, so it is what stops a *topological* action being taken when
//!   the action would wreck an MNEC. Nothing else in the tree looks at MNECs.
//! - The LP constraint ([`super::linear`]'s violation columns) is what stops a
//!   *continuous* action doing the same. Without it the LP would happily
//!   propose a set-point the objective then rejects, and the iteration would
//!   stall at the starting point rather than find the tap that respects the
//!   constraint — which is the difference between tap −9 and tap −16 in
//!   scenario 5.2.1.2.
//!
//! # The baseline is the RAO's starting point, not the perimeter's
//!
//! \\(m_0\\) comes from the network **before any remedial action at all** —
//! preventive ones included. A curative MNEC is judged against the margin it
//! had with nothing applied, which is why [`Baseline::measure`] is called once
//! by [`castor::run`](super::castor::run) on the untouched network and then
//! carried into every perimeter, rather than recomputed per perimeter from
//! whatever the previous stage left behind.

use std::collections::HashMap;

use super::crac::{Crac, State};
use super::evaluate::{evaluate_model, FlowModel, Network, Resolution, SecurityResult};
use super::linear::ObjectiveUnit;

/// The reference's own stand-in for "no margin at all to speak of".
///
/// A perimeter with no optimized CNECs has no minimum margin, and the obvious
/// answer — an infinity — is the wrong one: adding a finite violation cost to
/// an infinite functional cost loses the violation, so a curative perimeter
/// containing nothing but MNECs would rank every candidate identically and act
/// on none of them. The reference returns `-1e9` for that cost
/// (`SumMaxPerTimestampCostEvaluatorResult.COST_LIMIT`), which is dominant
/// against any realistic margin and still lets a penalty be seen.
pub const NO_CNEC_MARGIN: f64 = 1e9;

/// How hard an MNEC's soft constraint pushes back.
///
/// The defaults are the reference's: `mnec-parameters.acceptable-margin-decrease`
/// is 50.0, and the search-tree extension's `violation-cost` is 10.0 with a
/// `constraint-adjustment-coefficient` of 0.0.
///
/// All three are in the **objective's unit** — amperes for an AC run, megawatts
/// for a DC one — because that is the unit the reference's `MnecFiller` writes
/// its constraint in. A configuration stating a decrease of 260 means 260 A
/// when the flow model is AC, and converting it to megawatts is this module's
/// job, not the configuration's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MnecOptions {
    /// Whether monitored CNECs are constrained at all.
    ///
    /// The reference gates this on `mnec-parameters` being present in the
    /// configuration, so a file that never mentions MNECs leaves them
    /// unconstrained even when the CRAC declares them.
    pub enabled: bool,
    /// How far an MNEC that started *overloaded* may be pushed further.
    pub acceptable_margin_decrease: f64,
    /// Objective penalty per unit of violation.
    pub violation_cost: f64,
    /// Tightens the LP's bound by this much, so the solver lands strictly
    /// inside the constraint rather than exactly on it. Zero by default.
    pub constraint_adjustment_coefficient: f64,
}

impl Default for MnecOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            acceptable_margin_decrease: 50.0,
            violation_cost: 10.0,
            constraint_adjustment_coefficient: 0.0,
        }
    }
}

/// What one MNEC looked like before anything was done to the network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Initial {
    /// Signed flow, MW.
    pub flow_mw: f64,
    /// Margin, MW.
    pub margin_mw: f64,
    /// The same margin in amperes, at the voltage its binding threshold names.
    pub margin_a: f64,
}

impl Initial {
    fn margin(&self, unit: ObjectiveUnit) -> f64 {
        match unit {
            ObjectiveUnit::Megawatt => self.margin_mw,
            ObjectiveUnit::Ampere => self.margin_a,
        }
    }
}

/// The margins the untouched network had, by CNEC index.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Baseline {
    initial: HashMap<usize, Initial>,
}

impl Baseline {
    /// Read a baseline out of an already-computed assessment.
    pub fn from_result(result: &SecurityResult) -> Self {
        let mut initial = HashMap::new();
        for perimeter in &result.perimeters {
            for cnec in &perimeter.cnecs {
                initial.insert(
                    cnec.cnec,
                    Initial {
                        flow_mw: cnec.flow_mw,
                        margin_mw: cnec.margin_mw,
                        margin_a: cnec.margin_a,
                    },
                );
            }
        }
        Self { initial }
    }

    /// Measure the untouched network and read a baseline out of it.
    pub fn measure(
        crac: &Crac,
        network: &Network<'_>,
        resolution: &Resolution,
        model: FlowModel,
    ) -> Self {
        let ac = super::evaluate::ac_options(network);
        let result =
            evaluate_model(crac, network, resolution, network.initially_open, model, &ac);
        Self::from_result(&result)
    }

    pub fn get(&self, cnec: usize) -> Option<Initial> {
        self.initial.get(&cnec).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.initial.is_empty()
    }
}

/// The MNEC rule, ready to apply: the knobs plus the baseline they are measured
/// against.
///
/// Carried on [`LinearOptions`](super::linear::LinearOptions) rather than passed
/// separately because every layer that scores a network needs it and none of
/// them wants another argument.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mnec {
    pub options: MnecOptions,
    pub baseline: Baseline,
}

impl Mnec {
    /// Whether this constrains anything at all.
    ///
    /// A baseline that was never measured means nothing is known about where
    /// the MNECs started, and a rule with no baseline is not a weaker rule —
    /// it is an arbitrary one. Better to leave them unconstrained and say so.
    pub fn active(&self) -> bool {
        self.options.enabled && !self.baseline.is_empty()
    }

    /// The floor an MNEC's margin is held to, in `unit`.
    ///
    /// `min(0, m₀ − d)` — zero for an MNEC that started with more than `d` of
    /// margin, and `m₀ − d` for one that did not.
    pub fn floor(&self, initial: Initial, unit: ObjectiveUnit) -> f64 {
        f64::min(0.0, initial.margin(unit) - self.options.acceptable_margin_decrease)
    }

    /// How far below its floor one MNEC has been pushed. Never negative.
    pub fn violation(&self, initial: Initial, margin: f64, unit: ObjectiveUnit) -> f64 {
        (self.floor(initial, unit) - margin).max(0.0)
    }

    /// The virtual cost this assessment incurs over one perimeter, in `unit`.
    ///
    /// Subtracted from the minimum margin to give the objective the search
    /// ranks by. An MNEC with no baseline entry — one whose branch did not
    /// resolve, or that the initial assessment skipped — costs nothing rather
    /// than costing an unknown amount.
    pub fn cost(
        &self,
        crac: &Crac,
        result: &SecurityResult,
        perimeter: &[State],
        unit: ObjectiveUnit,
    ) -> f64 {
        if !self.active() {
            return 0.0;
        }
        let mut total = 0.0;
        for p in result.perimeters.iter().filter(|p| perimeter.contains(&p.state)) {
            for cnec in &p.cnecs {
                if !crac.flow_cnecs[cnec.cnec].monitored {
                    continue;
                }
                let Some(initial) = self.baseline.get(cnec.cnec) else { continue };
                let margin = match unit {
                    ObjectiveUnit::Megawatt => cnec.margin_mw,
                    ObjectiveUnit::Ampere => cnec.margin_a,
                };
                total += self.options.violation_cost * self.violation(initial, margin, unit);
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(decrease: f64) -> Mnec {
        Mnec {
            options: MnecOptions { acceptable_margin_decrease: decrease, ..Default::default() },
            // One entry, so `active` is true; the CNEC index is never read by
            // the two functions under test.
            baseline: Baseline {
                initial: [(0, Initial { flow_mw: 0.0, margin_mw: 0.0, margin_a: 0.0 })]
                    .into_iter()
                    .collect(),
            },
        }
    }

    fn started_at(margin: f64) -> Initial {
        Initial { flow_mw: 0.0, margin_mw: margin, margin_a: margin }
    }

    /// The three cases the reference wrote one scenario each for.
    #[test]
    fn the_floor_is_zero_only_for_an_mnec_with_room_to_spare() {
        let mnec = rule(50.0);
        let mw = ObjectiveUnit::Megawatt;

        // 5.2.1.2: more than the acceptable decrease in hand. Hold it at zero —
        // all of its margin is available, not just 50 of it.
        assert_eq!(mnec.floor(started_at(200.0), mw), 0.0);
        // Exactly at the decrease is still the zero case.
        assert_eq!(mnec.floor(started_at(50.0), mw), 0.0);

        // 5.2.1.4: inside the decrease. It may be pushed through its threshold,
        // but only by the part of the 50 it had not already used.
        assert!((mnec.floor(started_at(42.4), mw) - -7.6).abs() < 1e-9);

        // 5.2.1.3: already overloaded. Fifty worse and no worse — it is not
        // required to be repaired, only not made worse.
        assert!((mnec.floor(started_at(-33.3), mw) - -83.3).abs() < 1e-9);
    }

    /// Reading the rule as a flat "no more than `d` worse" is the tempting
    /// mistake, and it is wrong by the entire initial margin.
    #[test]
    fn a_healthy_mnec_may_not_spend_its_whole_margin_plus_fifty() {
        let mnec = rule(50.0);
        let mw = ObjectiveUnit::Megawatt;
        let initial = started_at(200.0);

        // A flat reading would allow 150. The rule does not.
        assert_eq!(mnec.violation(initial, 150.0, mw), 0.0);
        assert_eq!(mnec.violation(initial, 0.0, mw), 0.0);
        assert_eq!(mnec.violation(initial, -30.0, mw), 30.0);
    }

    /// The unit is the objective's, and the two are not interchangeable.
    #[test]
    fn the_rule_is_read_in_whichever_unit_the_objective_uses() {
        let mnec = rule(50.0);
        // 40 MW of margin is 60 A at this CNEC's voltage. In megawatts it is
        // inside the 50 and gets a negative floor; in amperes it is outside and
        // gets a floor of zero. Same CNEC, same network, different rule —
        // which is why the unit travels with the parameter.
        let initial = Initial { flow_mw: 0.0, margin_mw: 40.0, margin_a: 60.0 };
        assert_eq!(mnec.floor(initial, ObjectiveUnit::Megawatt), -10.0);
        assert_eq!(mnec.floor(initial, ObjectiveUnit::Ampere), 0.0);
    }

    /// A rule with no baseline constrains nothing, rather than constraining
    /// something arbitrary.
    #[test]
    fn an_unmeasured_baseline_leaves_mnecs_alone() {
        let mnec = Mnec::default();
        assert!(!mnec.active(), "the default baseline is empty");
        assert!(MnecOptions::default().enabled, "…which is not the same as being switched off");
    }
}
