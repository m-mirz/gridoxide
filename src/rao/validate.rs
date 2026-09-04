//! Re-check a finished plan under AC.
//!
//! The search runs on DC: linear, lossless, blind to reactive power and to
//! voltage. That is what makes it fast enough to explore thousands of
//! candidates, and it is also a model of the network that is not quite true.
//! A plan chosen against it can be secure in the model and insecure in the
//! machine.
//!
//! So this is the second stage — ToOp's structure, and the reason its authors
//! separate screening from validation. Every perimeter the plan produced is
//! re-measured with a full AC power flow on the network that perimeter
//! actually leaves behind, and the result is compared against the DC figure
//! the search believed. Three things get a plan rejected:
//!
//! * the AC margin is below [`ValidationOptions::min_margin_mw`];
//! * AC disagrees with DC by more than
//!   [`ValidationOptions::max_regression_mw`], which says the linear model
//!   misled the search rather than merely approximated it;
//! * the AC power flow did not converge, which is not a small margin but an
//!   absence of one.
//!
//! Nothing here changes the plan. A rejected perimeter is reported as
//! rejected, with the numbers that produced the verdict, because the useful
//! output of a validation stage is evidence — silently re-running the search
//! with different parameters would hide exactly the disagreement that is worth
//! seeing.

use super::castor::{PerimeterPlan, Plan};
use super::crac::{Crac, State};
use super::evaluate::{self, AcOptions, Network};
use super::Resolution;

/// Thresholds a perimeter must clear to be accepted.
#[derive(Clone, Copy, Debug)]
pub struct ValidationOptions<'a> {
    /// The AC margin a perimeter must reach, MW. Zero means "secure".
    pub min_margin_mw: f64,
    /// How far AC may fall below the DC figure before the disagreement itself
    /// is treated as a failure, MW.
    ///
    /// This is deliberately separate from `min_margin_mw`. A perimeter that is
    /// comfortably secure in both models but 200 MW apart between them is
    /// still a warning: the same modelling error that cost 200 MW here could
    /// cost more than the margin somewhere the search did not look.
    pub max_regression_mw: f64,
    pub ac: AcOptions<'a>,
}

impl Default for ValidationOptions<'_> {
    fn default() -> Self {
        Self { min_margin_mw: 0.0, max_regression_mw: f64::INFINITY, ac: AcOptions::default() }
    }
}

/// Why a perimeter failed, or that it did not.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Accepted,
    /// The AC power flow did not converge on at least one of the perimeter's
    /// states.
    Diverged,
    /// Secure under DC, insecure under AC.
    Insecure { margin_mw: f64 },
    /// Secure under both, but the two models disagree too much to trust the
    /// search that produced it.
    Regressed { by_mw: f64 },
}

impl Verdict {
    pub fn is_accepted(&self) -> bool {
        matches!(self, Verdict::Accepted)
    }
}

/// One perimeter's verdict and the figures behind it.
#[derive(Clone, Debug)]
pub struct PerimeterValidation {
    pub states: Vec<State>,
    /// What the search believed.
    pub dc_margin_mw: f64,
    /// What AC says.
    pub ac_margin_mw: f64,
    /// The CNEC that bound under AC, if any did.
    pub worst_cnec: Option<usize>,
    pub verdict: Verdict,
}

impl PerimeterValidation {
    /// How much margin the DC model claimed that AC does not grant. Negative
    /// means AC was the more optimistic of the two.
    pub fn regression_mw(&self) -> f64 {
        self.dc_margin_mw - self.ac_margin_mw
    }
}

/// The whole plan's verdict.
#[derive(Clone, Debug)]
pub struct Validation {
    pub perimeters: Vec<PerimeterValidation>,
    /// Worst AC margin anywhere in the plan.
    pub ac_margin_mw: f64,
}

impl Validation {
    /// True when every perimeter cleared its thresholds.
    pub fn is_accepted(&self) -> bool {
        self.perimeters.iter().all(|p| p.verdict.is_accepted())
    }

    pub fn rejected(&self) -> impl Iterator<Item = &PerimeterValidation> {
        self.perimeters.iter().filter(|p| !p.verdict.is_accepted())
    }
}

/// Re-measure every perimeter of `plan` under AC and judge it.
///
/// `network` supplies what the plan does not carry — the line list, the branch
/// ids, the base — while each perimeter's own `buses`, `transformers` and
/// `open_branches` supply the state its decisions left the network in. That
/// split is why this needs no access to the search: a `PerimeterPlan` already
/// describes the network its numbers were measured on.
pub fn validate(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    plan: &Plan,
    options: &ValidationOptions<'_>,
) -> Validation {
    let perimeters: Vec<PerimeterValidation> = plan
        .perimeters()
        .map(|p| validate_perimeter(crac, network, resolution, p, options))
        .collect();
    let ac_margin_mw =
        perimeters.iter().map(|p| p.ac_margin_mw).fold(f64::INFINITY, f64::min);
    Validation { perimeters, ac_margin_mw }
}

fn validate_perimeter(
    crac: &Crac,
    network: &Network<'_>,
    resolution: &Resolution,
    plan: &PerimeterPlan,
    options: &ValidationOptions<'_>,
) -> PerimeterValidation {
    let post = Network {
        generation: network.generation,
        buses: &plan.buses,
        lines: network.lines,
        transformers: &plan.transformers,
        branch_ids: network.branch_ids,
        bus_ids: network.bus_ids,
        // The open set is stated outright rather than re-derived from the
        // network's `initially_open`: a plan may have *closed* something that
        // started open, and re-deriving would silently re-open it.
        initially_open: &[],
        bus_countries: network.bus_countries,
        shunts: network.shunts,
        tap_changers: network.tap_changers,
        base_mva: network.base_mva,
    };

    let result =
        evaluate::evaluate_ac(crac, &post, resolution, &plan.open_branches, &options.ac);

    // Only this perimeter's own states count. The evaluator measures every
    // state the CRAC defines, and the others belong to other perimeters, with
    // their own decisions in force.
    let mut ac_margin_mw = f64::INFINITY;
    let mut worst_cnec = None;
    let mut diverged = false;
    for perimeter in result.perimeters.iter().filter(|p| plan.states.contains(&p.state)) {
        if perimeter.severed {
            diverged = true;
        }
        for cnec in &perimeter.cnecs {
            if cnec.margin_mw < ac_margin_mw {
                ac_margin_mw = cnec.margin_mw;
                worst_cnec = Some(cnec.cnec);
            }
        }
    }

    let dc_margin_mw = plan.final_margin_mw;
    let verdict = if diverged {
        Verdict::Diverged
    } else if ac_margin_mw < options.min_margin_mw {
        Verdict::Insecure { margin_mw: ac_margin_mw }
    } else if dc_margin_mw - ac_margin_mw > options.max_regression_mw {
        Verdict::Regressed { by_mw: dc_margin_mw - ac_margin_mw }
    } else {
        Verdict::Accepted
    };

    PerimeterValidation {
        states: plan.states.clone(),
        dc_margin_mw,
        ac_margin_mw,
        worst_cnec,
        verdict,
    }
}
