//! How much may be done at all.
//!
//! Every other constraint in this layer is about a *branch* — this margin, that
//! threshold. A usage limit is about the **plan**: at most three remedial
//! actions in the curative instant, at most one topological action from any one
//! TSO, at most two PSTs from that one. They are what stops an optimizer
//! returning a coordinated fourteen-action manoeuvre across five countries
//! because it found 20 MW, which is a mathematically better answer and not one
//! any control room would carry out.
//!
//! # A budget, spent by both halves of the search
//!
//! The limits count *remedial actions*, and a phase shifter that moves is a
//! remedial action exactly as much as an opened line is. So the two halves of
//! the search spend one budget between them, and neither can be limited on its
//! own: on the reference's scenario 2.6.1.3 the whole curative allowance is one
//! action, and the reference spends it on the shifter — while gridoxide, with
//! no budget at all, opened a line *and* moved the shifter.
//!
//! The reference splits the enforcement to match that. Its search-tree filters
//! decide which *network-action* combinations may be reached
//! (`MaximumNumberOfRemedialActionsFilter` and friends), and what those leave
//! over becomes an allowance the LP is given for *range* actions
//! (`Leaf.getRaLimitationParameters` feeding `RaUsageLimitsFiller`). This module
//! is both halves of that: [`Limits`] is what the CRAC says, and [`Budget`] is
//! what one particular leaf has left.
//!
//! # `max-tso` is not here, deliberately
//!
//! A CRAC may declare it and the reference **ignores it** — "a max-tso limit can
//! no longer be defined and will be ignored", `RaUsageLimits.java`. Every
//! vendored file that has one sets it to `Integer.MAX_VALUE`. Honouring it would
//! be a constraint the reference does not apply, so the parser keeps the field
//! and nothing reads it.

use std::collections::HashMap;

use super::crac::{Crac, RaUsageLimits};

/// No limit. A CRAC writes `2147483647` for this, which is a bound in name only.
const UNLIMITED: usize = usize::MAX;

fn cap(value: Option<usize>) -> usize {
    match value {
        // `Integer.MAX_VALUE` in a JSON file means "no limit", not a limit of
        // two billion. Reading it as a number would be harmless here and
        // actively wrong in the subtraction below, where a limit is reduced by
        // what is already spent.
        Some(v) if v >= i32::MAX as usize => UNLIMITED,
        Some(v) => v,
        None => UNLIMITED,
    }
}

/// `limit` reduced by what has been used — but "no limit" stays no limit
/// rather than becoming a very large one. `usize::MAX - 3` would admit
/// everything too; it would just stop reporting itself as unlimited, and the
/// fast path that skips the whole check would quietly go unused.
fn spend(limit: usize, used: usize) -> usize {
    if limit == UNLIMITED { UNLIMITED } else { limit.saturating_sub(used) }
}

fn caps(map: &HashMap<String, usize>) -> HashMap<String, usize> {
    map.iter().map(|(k, &v)| (k.clone(), cap(Some(v)))).collect()
}

/// What a CRAC allows in one instant.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Limits {
    pub max_ra: usize,
    pub max_topo_per_tso: HashMap<String, usize>,
    pub max_pst_per_tso: HashMap<String, usize>,
    pub max_ra_per_tso: HashMap<String, usize>,
    pub max_elementary_per_tso: HashMap<String, usize>,
}

impl Limits {
    /// Everything allowed.
    pub fn unlimited() -> Self {
        Self { max_ra: UNLIMITED, ..Default::default() }
    }

    /// The limits a CRAC declares for `instant`, or none.
    pub fn of_instant(crac: &Crac, instant: usize) -> Self {
        crac.usage_limits
            .iter()
            .find(|l| l.instant == instant)
            .map_or_else(Self::unlimited, Self::from_crac)
    }

    fn from_crac(limits: &RaUsageLimits) -> Self {
        Self {
            max_ra: cap(limits.max_ra),
            max_topo_per_tso: caps(&limits.max_topo_per_tso),
            max_pst_per_tso: caps(&limits.max_pst_per_tso),
            max_ra_per_tso: caps(&limits.max_ra_per_tso),
            max_elementary_per_tso: caps(&limits.max_elementary_actions_per_tso),
        }
    }

    pub fn is_unlimited(&self) -> bool {
        self.max_ra == UNLIMITED
            && self.max_topo_per_tso.is_empty()
            && self.max_pst_per_tso.is_empty()
            && self.max_ra_per_tso.is_empty()
            && self.max_elementary_per_tso.is_empty()
    }

    /// Whether a set of network actions is allowed on its own.
    ///
    /// The topological half of the test, applied as the search tree grows. What
    /// this leaves over for the range actions is [`Limits::remaining`].
    pub fn admits(&self, crac: &Crac, chosen: &[usize]) -> bool {
        if chosen.len() > self.max_ra {
            return false;
        }
        let tsos: Vec<&str> = operators(crac, chosen);
        for (tso, &limit) in self.max_topo_per_tso.iter().chain(&self.max_ra_per_tso) {
            if tsos.iter().filter(|t| **t == tso).count() > limit {
                return false;
            }
        }
        for (tso, &limit) in &self.max_elementary_per_tso {
            let used: usize = chosen
                .iter()
                .filter(|&&i| crac.network_actions[i].operator.as_deref() == Some(tso.as_str()))
                .map(|&i| crac.network_actions[i].elementary.len())
                .sum();
            if used > limit {
                return false;
            }
        }
        true
    }

    /// What is left for the range actions once `chosen` network actions are
    /// taken.
    ///
    /// The subtraction is the whole point, and it is asymmetric on purpose:
    /// `max_ra` and `max_ra_per_tso` count everything, so a network action eats
    /// into them, while `max_pst_per_tso` counts only shifters and is untouched
    /// by topology. That is `Leaf.getRaLimitationParameters` exactly.
    pub fn remaining(&self, crac: &Crac, chosen: &[usize]) -> Budget {
        let tsos = operators(crac, chosen);
        let spent = |tso: &str| tsos.iter().filter(|t| **t == tso).count();
        Budget {
            max_ra: spend(self.max_ra, chosen.len()),
            max_pst_per_tso: self.max_pst_per_tso.clone(),
            max_ra_per_tso: self
                .max_ra_per_tso
                .iter()
                .map(|(tso, &limit)| (tso.clone(), spend(limit, spent(tso))))
                .collect(),
        }
    }
}

fn operators<'a>(crac: &'a Crac, chosen: &[usize]) -> Vec<&'a str> {
    chosen.iter().filter_map(|&i| crac.network_actions[i].operator.as_deref()).collect()
}

/// How many range actions one leaf may still move.
///
/// Carried on [`LinearOptions`](super::linear::LinearOptions) because it is
/// per-leaf: two candidates at the same depth that spend different TSOs'
/// allowances leave different budgets behind.
#[derive(Clone, Debug, PartialEq)]
pub struct Budget {
    pub max_ra: usize,
    pub max_pst_per_tso: HashMap<String, usize>,
    pub max_ra_per_tso: HashMap<String, usize>,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_ra: UNLIMITED,
            max_pst_per_tso: HashMap::new(),
            max_ra_per_tso: HashMap::new(),
        }
    }
}

impl Budget {
    pub fn is_unlimited(&self) -> bool {
        self.max_ra == UNLIMITED
            && self.max_pst_per_tso.is_empty()
            && self.max_ra_per_tso.is_empty()
    }

    /// Whether this set of *moved* range actions fits.
    ///
    /// `moved` are indices into [`Crac::range_actions`]. A range action that
    /// stayed where it was is not a remedial action and does not count — which
    /// is why this is asked of the answer rather than of the candidate set.
    pub fn admits(&self, crac: &Crac, moved: &[usize]) -> bool {
        if moved.len() > self.max_ra {
            return false;
        }
        let is = |i: usize, tso: &str| {
            crac.range_actions[i].operator.as_deref() == Some(tso)
        };
        for (tso, &limit) in &self.max_ra_per_tso {
            if moved.iter().filter(|&&i| is(i, tso)).count() > limit {
                return false;
            }
        }
        for (tso, &limit) in &self.max_pst_per_tso {
            let psts = moved
                .iter()
                .filter(|&&i| is(i, tso))
                .filter(|&&i| {
                    matches!(crac.range_actions[i].kind, super::crac::RangeActionKind::Pst { .. })
                })
                .count();
            if psts > limit {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rao::crac::{
        ElementaryAction, NetworkAction, Range, RangeAction, RangeActionKind, RangeKind,
    };

    fn network_action(id: &str, tso: &str, elementary: usize) -> NetworkAction {
        NetworkAction {
            id: id.into(),
            name: None,
            operator: Some(tso.into()),
            speed: None,
            activation_cost: None,
            elementary: (0..elementary)
                .map(|i| ElementaryAction::Switch { element: format!("{id}-{i}"), open: true })
                .collect(),
            usage_rules: Vec::new(),
        }
    }

    fn pst(id: &str, tso: &str) -> RangeAction {
        RangeAction {
            id: id.into(),
            name: None,
            operator: Some(tso.into()),
            speed: None,
            activation_cost: None,
            group: None,
            kind: RangeActionKind::Pst {
                element: id.into(),
                initial_tap: 0,
                tap_to_angle: Vec::new(),
            },
            ranges: vec![Range { kind: RangeKind::Absolute, min: None, max: None }],
            usage_rules: Vec::new(),
        }
    }

    fn crac() -> Crac {
        Crac {
            network_actions: vec![
                network_action("be-topo-1", "be", 1),
                network_action("be-topo-2", "be", 3),
                network_action("fr-topo", "fr", 1),
            ],
            range_actions: vec![pst("be-pst-1", "be"), pst("be-pst-2", "be"), pst("fr-pst", "fr")],
            ..Default::default()
        }
    }

    fn limits() -> Limits {
        Limits {
            max_ra: 3,
            max_topo_per_tso: [("be".into(), 1)].into_iter().collect(),
            max_pst_per_tso: [("be".into(), 1)].into_iter().collect(),
            max_ra_per_tso: [("be".into(), 2)].into_iter().collect(),
            max_elementary_per_tso: HashMap::new(),
        }
    }

    #[test]
    fn a_per_tso_topology_cap_stops_the_second_action_from_that_tso() {
        let crac = crac();
        let limits = limits();
        assert!(limits.admits(&crac, &[0]));
        assert!(!limits.admits(&crac, &[0, 1]), "two BE topological actions, cap is one");
        // The cap is per TSO, so another operator's action is unaffected.
        assert!(limits.admits(&crac, &[0, 2]));
    }

    #[test]
    fn the_total_cap_counts_every_operator_together() {
        let crac = crac();
        let limits = Limits { max_ra: 1, ..Limits::unlimited() };
        assert!(limits.admits(&crac, &[0]));
        assert!(!limits.admits(&crac, &[0, 2]));
    }

    /// A network action eats into the shared allowances and leaves the PST cap
    /// alone. That asymmetry is the reference's, and it is the whole content of
    /// `Leaf.getRaLimitationParameters`.
    #[test]
    fn a_network_action_spends_the_shared_budget_but_not_the_pst_one() {
        let crac = crac();
        let budget = limits().remaining(&crac, &[0]);
        assert_eq!(budget.max_ra, 2, "one of the three is spent");
        assert_eq!(budget.max_ra_per_tso["be"], 1, "BE spent one of its two");
        assert_eq!(budget.max_pst_per_tso["be"], 1, "…but its shifter allowance is untouched");

        // So BE may still move one shifter, and only one.
        assert!(budget.admits(&crac, &[0]));
        assert!(!budget.admits(&crac, &[0, 1]));
        // FR is unconstrained by any of it.
        assert!(budget.admits(&crac, &[0, 2]));
    }

    /// An elementary-action cap counts the *parts*, not the actions.
    #[test]
    fn the_elementary_cap_counts_switches_rather_than_actions() {
        let crac = crac();
        let limits = Limits {
            max_elementary_per_tso: [("be".into(), 2)].into_iter().collect(),
            ..Limits::unlimited()
        };
        assert!(limits.admits(&crac, &[0]), "one switch, cap is two");
        assert!(!limits.admits(&crac, &[1]), "one action but three switches");
    }

    /// Admissibility is monotone — every cap is a count, so dropping an action
    /// can never make a set inadmissible. That is what lets
    /// `linear::admissible_subsets` try only the maximal sets.
    #[test]
    fn dropping_an_action_never_makes_a_set_inadmissible() {
        let crac = crac();
        let budget = limits().remaining(&crac, &[]);
        let all: Vec<usize> = (0..crac.range_actions.len()).collect();
        for mask in 0u32..(1 << all.len()) {
            let subset: Vec<usize> =
                all.iter().copied().filter(|&i| mask >> i & 1 == 1).collect();
            if !budget.admits(&crac, &subset) {
                continue;
            }
            for drop in &subset {
                let smaller: Vec<usize> =
                    subset.iter().copied().filter(|i| i != drop).collect();
                assert!(
                    budget.admits(&crac, &smaller),
                    "{subset:?} is admissible but {smaller:?} is not"
                );
            }
        }
    }

    /// `Integer.MAX_VALUE` in a CRAC means "no limit", and reading it as a
    /// number would survive every test until something subtracted from it.
    #[test]
    fn a_two_billion_limit_is_no_limit() {
        let crac = crac();
        let limits = Limits::from_crac(&RaUsageLimits {
            instant: 0,
            max_ra: Some(i32::MAX as usize),
            max_tso: Some(i32::MAX as usize),
            ..Default::default()
        });
        assert!(limits.is_unlimited());
        assert!(limits.admits(&crac, &[0, 1, 2]));
        assert!(limits.remaining(&crac, &[0, 1, 2]).is_unlimited());
    }
}
