//! The half of a usage rule that only a flow result can answer.
//!
//! A CRAC's usage rules say when a remedial action may be used, and they come
//! in two kinds. `onInstant` and `onContingencyState` are **topological**: they
//! can be checked against a state alone, and [`UsageRule::covers`] does that.
//! `onFlowConstraint` and `onFlowConstraintInCountry` are not — they say the
//! action is available *only if some CNEC is actually constrained*, which needs
//! flows.
//!
//! This module is that second half. Without it every conditional action is
//! offered unconditionally, and the search takes actions the reference never
//! puts on the table — on scenario 2.4.1.2 gridoxide used three where the
//! reference uses one, and reported +97 A against its −45. A gate reading only
//! margins would have called that an improvement.
//!
//! # Any rule, not every rule
//!
//! An action is available when **any** of its usage rules is activated
//! (`RaoUtil.canRemedialActionBeUsed`). So a conditional rule is another way
//! *in*, never a restriction: an action carrying both `onInstant: preventive`
//! and `onFlowConstraint` is available in every preventive state regardless of
//! flows. Reading the rules as a conjunction would silence actions the CRAC
//! makes freely available.
//!
//! # Measured once, at the perimeter's starting point
//!
//! The reference builds its list of available actions when it *builds the
//! optimization perimeter*, from the flow result that perimeter starts from,
//! and never revisits it. The scenario titles say so outright — 2.4.1.2 is
//! "onConstraint RAs with a constraint triggered by another preventive RA, **no
//! reevaluation**".
//!
//! That is a decision worth keeping deliberately. Re-deriving availability
//! inside each leaf would let a network action that relieves a CNEC withdraw
//! the very action authorized by it, and the search would then be exploring a
//! candidate set that changes underneath it — a different problem at every
//! depth, and one whose answer depends on the order actions happened to be
//! tried.
//!
//! # "Constrained" means margin ≤ 0, in the objective's unit
//!
//! Not `< 0`: `RaoUtil.isAnyMarginNegative` uses `<= 0`, so a CNEC sitting
//! exactly on its threshold authorizes its action. And the unit matters for the
//! same reason it matters to the objective — a margin that is negative in
//! amperes can be positive in megawatts once the approximations differ — which
//! is why this takes the same [`ObjectiveUnit`] the rest of the run is measured
//! in rather than picking one.

use std::collections::HashMap;

use crate::linear::btheta::DcBranch;

use super::crac::{Crac, State, UsageRule};
use super::evaluate::{Network, Resolution, SecurityResult};
use super::linear::ObjectiveUnit;

/// Which of a perimeter's CNECs are constrained, and where they are.
///
/// [`unmeasured`](Self::unmeasured) means no flows were supplied, and a
/// conditional rule then falls back to its topological half — exactly what
/// [`UsageRule::covers`] answers, and what this layer did before the flow test
/// existed. That is deliberately the permissive reading: a caller who invokes
/// [`optimize`](super::linear::optimize) directly, without a perimeter around
/// it, should not silently lose the actions its CRAC offers.
///
/// It is also why [`search`](super::search::search) measures unconditionally
/// rather than leaving it to a caller. The permissive default is safe only
/// because the one layer that matters never takes it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Constrained {
    /// CNEC index → the countries its branch touches. Present only for CNECs
    /// whose margin is at or below zero.
    at_or_below_zero: HashMap<usize, Vec<String>>,
    measured: bool,
}

impl Constrained {
    /// Nothing measured: conditional rules activate nothing.
    pub fn unmeasured() -> Self {
        Self::default()
    }

    pub fn is_measured(&self) -> bool {
        self.measured
    }

    /// Read the constrained CNECs out of an assessment of the perimeter's
    /// starting point.
    ///
    /// `countries` maps a flat branch index to the countries it touches, which
    /// only [`UsageRule::OnFlowConstraintInCountry`] needs; passing an empty map
    /// leaves that rule unable to activate rather than activating it blindly.
    pub fn from_result(
        result: &SecurityResult,
        unit: ObjectiveUnit,
        countries: &HashMap<usize, Vec<String>>,
    ) -> Self {
        let mut at_or_below_zero = HashMap::new();
        for perimeter in &result.perimeters {
            for cnec in &perimeter.cnecs {
                let margin = match unit {
                    ObjectiveUnit::Megawatt => cnec.margin_mw,
                    ObjectiveUnit::Ampere => cnec.margin_a,
                };
                if margin <= 0.0 {
                    at_or_below_zero
                        .insert(cnec.cnec, countries.get(&cnec.branch).cloned().unwrap_or_default());
                }
            }
        }
        Self { at_or_below_zero, measured: true }
    }

    /// Whether one usage rule is activated in `state`.
    ///
    /// The topological rules defer to [`UsageRule::covers`]; the two
    /// conditional ones add their flow test on top of it.
    pub fn activates(&self, rule: &UsageRule, state: &State, crac: &Crac) -> bool {
        match rule {
            UsageRule::OnInstant { .. } | UsageRule::OnContingencyState { .. } => {
                rule.covers(state)
            }
            _ if !self.measured => rule.covers(state),
            UsageRule::OnConstraint { instant, cnec } => {
                let Some(index) = crac.flow_cnecs.iter().position(|c| c.id == *cnec) else {
                    // A rule naming a CNEC this CRAC does not contain can never
                    // be satisfied, and treating it as free would hand the
                    // search an action nothing authorized.
                    return false;
                };
                let named = &crac.flow_cnecs[index];
                // A curative or outage rule reaches only its own contingency's
                // states. A preventive one reaches all of them, which is what
                // makes "open this line before the outage, because that outage
                // overloads something" expressible at all.
                let preventive = crac.instants[*instant].kind == super::crac::InstantKind::Preventive;
                if !preventive && named.state.contingency != state.contingency {
                    return false;
                }
                state.instant == *instant && self.at_or_below_zero.contains_key(&index)
            }
            UsageRule::OnFlowConstraintInCountry { instant, country, contingency } => {
                if contingency.is_some_and(|c| state.contingency != Some(c)) {
                    return false;
                }
                if state.instant != *instant {
                    return false;
                }
                self.at_or_below_zero.iter().any(|(&index, located)| {
                    let cnec = &crac.flow_cnecs[index];
                    // "Does not come before": a rule at the preventive instant
                    // is authorized by a curative overload too, since acting
                    // early is precisely the point.
                    cnec.state.instant >= *instant
                        && contingency.is_none_or(|c| cnec.state.contingency == Some(c))
                        && located.iter().any(|l| l.eq_ignore_ascii_case(country))
                })
            }
        }
    }

    /// Whether `rules` make an action available anywhere in `perimeter`.
    pub fn allows(&self, rules: &[UsageRule], perimeter: &[State], crac: &Crac) -> bool {
        perimeter.iter().any(|s| rules.iter().any(|r| self.activates(r, s, crac)))
    }
}

/// Which countries each CNEC's branch touches, by flat branch index.
///
/// Derived from the network rather than from the CRAC, because a CRAC does not
/// say where its elements are — the same reason
/// [`search`](mod@super::search)'s far-from-most-limiting filter reads them from
/// the buses.
pub fn branch_countries(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    dc: &[DcBranch],
) -> HashMap<usize, Vec<String>> {
    let mut out: HashMap<usize, Vec<String>> = HashMap::new();
    for cnec in &crac.flow_cnecs {
        let Some(branch) = resolution.branch(&cnec.network_element) else { continue };
        out.entry(branch).or_insert_with(|| {
            let Some(b) = dc.iter().find(|b| b.index == branch) else { return Vec::new() };
            [b.from, b.to]
                .iter()
                .filter_map(|&bus| network.bus_countries.get(bus)?.clone())
                .collect()
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rao::crac::{FlowCnec, Instant, InstantKind, Side, Threshold, Unit};

    fn crac() -> Crac {
        let cnec = |id: &str, contingency: Option<usize>, instant: usize| FlowCnec {
            id: id.into(),
            network_element: id.into(),
            state: State { instant, contingency },
            thresholds: vec![Threshold {
                unit: Unit::Megawatt,
                min: None,
                max: Some(100.0),
                side: Side::One,
            }],
            reliability_margin: 0.0,
            optimized: true,
            monitored: false,
            operator: None,
            i_max: None,
            nominal_v: None,
        };
        Crac {
            instants: vec![
                Instant { id: "preventive".into(), kind: InstantKind::Preventive },
                Instant { id: "curative".into(), kind: InstantKind::Curative },
            ],
            flow_cnecs: vec![cnec("tight", None, 0), cnec("slack", None, 0), cnec("cur", Some(0), 1)],
            ..Default::default()
        }
    }

    /// Only `tight` and `cur` are at or below zero.
    fn constrained() -> Constrained {
        Constrained {
            at_or_below_zero: [(0, vec!["fr".into()]), (2, vec!["be".into()])].into_iter().collect(),
            measured: true,
        }
    }

    const PREVENTIVE: State = State { instant: 0, contingency: None };

    #[test]
    fn a_conditional_rule_needs_its_cnec_to_be_constrained() {
        let crac = crac();
        let c = constrained();
        let on = |cnec: &str| UsageRule::OnConstraint { instant: 0, cnec: cnec.into() };

        assert!(c.activates(&on("tight"), &PREVENTIVE, &crac));
        assert!(!c.activates(&on("slack"), &PREVENTIVE, &crac));
        // A rule naming a CNEC the CRAC does not have authorizes nothing —
        // treating it as unconditional would hand the search a free action.
        assert!(!c.activates(&on("nonesuch"), &PREVENTIVE, &crac));
    }

    /// A preventive rule may be authorized by a *curative* overload. That is the
    /// point of it: act before the outage, because the outage is what hurts.
    #[test]
    fn a_preventive_rule_reaches_across_contingencies() {
        let crac = crac();
        let c = constrained();
        let rule = UsageRule::OnConstraint { instant: 0, cnec: "cur".into() };
        assert!(c.activates(&rule, &PREVENTIVE, &crac));
    }

    /// Availability is "any rule", not "every rule".
    #[test]
    fn an_unconditional_rule_is_not_narrowed_by_a_conditional_one() {
        let crac = crac();
        let c = constrained();
        let rules = vec![
            UsageRule::OnInstant { instant: 0 },
            UsageRule::OnConstraint { instant: 0, cnec: "slack".into() },
        ];
        assert!(
            c.allows(&rules, &[PREVENTIVE], &crac),
            "the free rule alone should make this available"
        );
        assert!(
            !c.allows(&rules[1..], &[PREVENTIVE], &crac),
            "…and without it, the unconstrained CNEC authorizes nothing"
        );
    }

    /// The country rule looks at every constrained CNEC of the perimeter, not
    /// at one named element — and only at those at or after its own instant.
    #[test]
    fn the_country_rule_is_answered_by_any_overload_it_covers() {
        let crac = crac();
        let c = constrained();
        let rule = |instant, country: &str, contingency| UsageRule::OnFlowConstraintInCountry {
            instant,
            country: country.into(),
            contingency,
        };
        // `tight` is preventive and in FR; `cur` is curative and in BE. A
        // preventive rule sees both, because acting early is the point.
        assert!(c.activates(&rule(0, "fr", None), &PREVENTIVE, &crac));
        assert!(c.activates(&rule(0, "be", None), &PREVENTIVE, &crac));
        assert!(!c.activates(&rule(0, "de", None), &PREVENTIVE, &crac));
        // Naming a contingency binds the rule to *that contingency's states*,
        // so it cannot fire in the base case at all — the reference tests
        // `rule.contingency == state.contingency` before it looks at any flow.
        // Note the asymmetry with `OnConstraint` above, which a preventive
        // instant lets reach across contingencies. Both are the reference's.
        assert!(!c.activates(&rule(0, "be", Some(0)), &PREVENTIVE, &crac));
        let curative = State { instant: 1, contingency: Some(0) };
        assert!(c.activates(&rule(1, "be", Some(0)), &curative, &crac));
        // FR's overload is in the base case, which a curative rule cannot see:
        // a preventive CNEC comes *before* the curative instant.
        assert!(!c.activates(&rule(1, "fr", Some(0)), &curative, &crac));
    }

    /// A margin of exactly zero counts. `isAnyMarginNegative` is `<= 0`.
    #[test]
    fn sitting_on_the_threshold_counts_as_constrained() {
        use crate::rao::evaluate::{CnecResult, PerimeterResult, SecurityResult};
        let cnec = |i: usize, margin: f64| CnecResult {
            cnec: i,
            branch: i,
            flow_mw: 0.0,
            margin_mw: margin,
            upper_mw: 0.0,
            lower_mw: 0.0,
            limit_mw: 0.0,
            margin_a: margin,
            conversion_v: 0.0,
            current_a: 0.0,
        };
        let result = SecurityResult {
            perimeters: vec![PerimeterResult {
                state: PREVENTIVE,
                cnecs: vec![cnec(0, 0.0), cnec(1, 1e-9)],
                severed: false,
            }],
            skipped: Vec::new(),
        };
        let c = Constrained::from_result(&result, ObjectiveUnit::Megawatt, &HashMap::new());
        assert!(c.at_or_below_zero.contains_key(&0), "exactly zero is constrained");
        assert!(!c.at_or_below_zero.contains_key(&1), "a hair above zero is not");
    }

    /// Without flows, a conditional rule falls back to its topological half —
    /// which is what this layer did before the flow test existed.
    #[test]
    fn an_unmeasured_perimeter_keeps_the_old_permissive_reading() {
        let crac = crac();
        let c = Constrained::unmeasured();
        let rule = UsageRule::OnConstraint { instant: 0, cnec: "slack".into() };
        assert!(c.activates(&rule, &PREVENTIVE, &crac));
        assert!(!c.is_measured());
    }
}
