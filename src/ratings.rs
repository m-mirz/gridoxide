//! Branch operating limits — the permanent rating and its temporary
//! overload allowances.
//!
//! Nothing in [`crate::types`] carries a rating: a [`Line`](crate::types::Line)
//! is `r`/`x`/`b_shunt`/`g_shunt` and a
//! [`Transformer`](crate::types::Transformer) is admittances plus a tap. The
//! only rating anywhere before this module was
//! [`opf::model::BranchLimit`](crate::opf::model::BranchLimit) — a single MVA
//! figure per branch, arriving through the companion OPF document, which is
//! enough to ask "is this dispatch feasible" and not enough to ask "is this
//! network secure".
//!
//! Security analysis needs the distinction the grid codes actually draw:
//!
//! - a **PATL** (permanent admissible transmission loading) that may be
//!   carried indefinitely, and
//! - one or more **TATLs** (temporary admissible transmission loading), each
//!   valid only for a stated duration.
//!
//! The gap between them is not a detail — it is the room a curative remedial
//! action is sized to cover. A flow above PATL but below the 20-minute TATL is
//! acceptable *provided* something brings it back down within 20 minutes, so a
//! preventive study judges against the TATL and a curative study against the
//! PATL. Collapsing the two into one number makes that whole distinction
//! inexpressible.
//!
//! Units are **amperes**, matching CGMES `CurrentLimit`, UCTE's `##L`/`##T`
//! current field and IIDM's `currentLimits` — every source gridoxide reads
//! states the permanent rating as a current. A limit set may be absent
//! entirely (`None`), which means *unlimited* and not *zero*; the same
//! convention MATPOWER's `rate = 0` carries and
//! [`opf::model::BranchLimit::rate_pu`](crate::opf::model::BranchLimit::rate_pu)
//! already honours.

/// One temporary rating: a current, and how long it may be carried.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporaryLimit {
    /// Seconds this rating may be carried for. `None` means "no stated
    /// duration", which CGMES spells `OperationalLimitType.isInfiniteDuration`
    /// and which is therefore a *second permanent* limit rather than a
    /// temporary one — kept distinct from `PATL` because a network can carry
    /// several, and taking the tightest is the caller's decision, not the
    /// importer's.
    pub acceptable_duration_s: Option<f64>,
    /// The rating itself, in amperes.
    pub value_a: f64,
}

/// The operating limits of one branch terminal.
///
/// Terminals are kept separate because they genuinely differ: a transformer's
/// two windings sit at different voltages and therefore different current
/// ratings, and CGMES/IIDM both model limits per terminal. A source that
/// states one rating for the whole branch (UCTE does) populates both sides
/// with the same value, which is the honest representation of what it said.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BranchLimits {
    /// Permanent admissible transmission loading, amperes. `None` = unlimited.
    pub patl_a: Option<f64>,
    /// Temporary ratings, in no particular order. Use
    /// [`limit_for`](Self::limit_for) rather than indexing.
    pub tatl: Vec<TemporaryLimit>,
}

impl BranchLimits {
    /// A limit set with a permanent rating and nothing else.
    pub fn permanent(value_a: f64) -> Self {
        Self { patl_a: Some(value_a), tatl: Vec::new() }
    }

    /// True when this terminal has no rating of any kind — the state every
    /// branch is in before an importer that reads limits has run.
    pub fn is_unlimited(&self) -> bool {
        self.patl_a.is_none() && self.tatl.is_empty()
    }

    /// The rating that applies when the overload may last `duration_s`.
    ///
    /// This is the *tightest* temporary rating whose acceptable duration is at
    /// least `duration_s`, falling back to the PATL when no temporary rating
    /// covers that long. Picking the tightest rather than the loosest is
    /// deliberate: several TATLs may nominally cover the requested duration,
    /// and honouring all of them means honouring the smallest.
    ///
    /// `duration_s = 0.0` asks for the most permissive instantaneous rating,
    /// which is what an outage-instant CNEC is judged against; passing
    /// [`f64::INFINITY`] asks for the permanent one.
    pub fn limit_for(&self, duration_s: f64) -> Option<f64> {
        let temporary = self
            .tatl
            .iter()
            .filter(|t| match t.acceptable_duration_s {
                Some(d) => d >= duration_s,
                // An "infinite duration" temporary limit covers any request.
                None => true,
            })
            .map(|t| t.value_a)
            .fold(f64::INFINITY, f64::min);
        match (temporary.is_finite(), self.patl_a) {
            (true, Some(patl)) => Some(temporary.max(patl)),
            (true, None) => Some(temporary),
            (false, patl) => patl,
        }
    }

    /// The single tightest rating this terminal carries, whatever its
    /// duration. The conservative reading, and the right one for a plain
    /// "is anything overloaded" check that has no time axis.
    pub fn tightest(&self) -> Option<f64> {
        let temporary = self.tatl.iter().map(|t| t.value_a).fold(f64::INFINITY, f64::min);
        match (temporary.is_finite(), self.patl_a) {
            (true, Some(patl)) => Some(temporary.min(patl)),
            (true, None) => Some(temporary),
            (false, patl) => patl,
        }
    }
}

/// Convert a current rating in amperes to an apparent-power rating in
/// per-unit, at a stated line-to-line base voltage.
///
/// `S = √3 · U · I`, so `S_pu = √3 · u_rated · i_a / s_base_va`. The `√3` is
/// the whole reason this helper exists rather than being written inline at
/// each call site: dropping it is a 73% error that still looks plausible.
pub fn current_to_power_pu(current_a: f64, u_rated_v: f64, s_base_va: f64) -> f64 {
    3f64.sqrt() * u_rated_v * current_a / s_base_va
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_limit_set_is_unlimited_rather_than_zero() {
        let limits = BranchLimits::default();
        assert!(limits.is_unlimited());
        assert_eq!(limits.tightest(), None);
        assert_eq!(limits.limit_for(0.0), None);
    }

    #[test]
    fn a_temporary_limit_applies_only_while_its_duration_covers_the_request() {
        let limits = BranchLimits {
            patl_a: Some(1000.0),
            tatl: vec![TemporaryLimit { acceptable_duration_s: Some(1200.0), value_a: 1200.0 }],
        };
        // Asking for a 20-minute overload gets the temporary rating ...
        assert_eq!(limits.limit_for(1200.0), Some(1200.0));
        // ... asking for an hour falls back to the permanent one.
        assert_eq!(limits.limit_for(3600.0), Some(1000.0));
        // The unqualified answer is the tightest.
        assert_eq!(limits.tightest(), Some(1000.0));
    }

    #[test]
    fn several_covering_temporary_limits_resolve_to_the_tightest() {
        let limits = BranchLimits {
            patl_a: Some(1000.0),
            tatl: vec![
                TemporaryLimit { acceptable_duration_s: Some(600.0), value_a: 1500.0 },
                TemporaryLimit { acceptable_duration_s: Some(900.0), value_a: 1300.0 },
            ],
        };
        // Both cover 600 s; honouring both means honouring 1300.
        assert_eq!(limits.limit_for(600.0), Some(1300.0));
    }

    #[test]
    fn a_temporary_limit_never_reports_below_the_permanent_one() {
        // Badly-ordered data does occur; a TATL under the PATL would otherwise
        // make a short overload stricter than a permanent one.
        let limits = BranchLimits {
            patl_a: Some(1000.0),
            tatl: vec![TemporaryLimit { acceptable_duration_s: Some(60.0), value_a: 800.0 }],
        };
        assert_eq!(limits.limit_for(60.0), Some(1000.0));
    }

    #[test]
    fn current_converts_to_apparent_power_through_the_root_three() {
        // 1000 A at 380 kV is 658.2 MVA.
        let s = current_to_power_pu(1000.0, 380_000.0, 100e6);
        assert!((s - 6.5818).abs() < 1e-3, "got {s}");
    }
}
