//! AC re-validation of a DC-chosen plan — phase 11 of `plans/RAO_PLAN.md`.
//!
//! The search runs on a linear, lossless model because that is what makes it
//! finish. This stage asks whether the plan it produced survives contact with
//! the real one. The interesting property is not that AC and DC agree — they
//! do not, and the whole point of the stage is that they do not — but that the
//! disagreement is measured and reported rather than absorbed.

use std::path::PathBuf;

use gridoxide::opf::ipm::IpmSolver;
use gridoxide::rao::crac::*;
use gridoxide::rao::evaluate::AcOptions;
use gridoxide::rao::validate::{validate, ValidationOptions, Verdict};
use gridoxide::rao::{crac_json, run, Network, Resolution, SearchOptions};
use gridoxide::ucte;

fn ucte_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte").join(name)
}

fn rao_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/rao").join(name)
}

struct Case {
    net: ucte::UcteImport,
    crac: Crac,
}

fn case(crac_name: &str) -> Case {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(rao_fixture(crac_name)).expect("crac");
    Case { net, crac }
}

impl Case {
    fn resolution(&self) -> Resolution {
        Resolution::new(&self.crac, &self.net.branch_ids)
    }

    fn network(&self) -> Network<'_> {
        Network {
            buses: &self.net.buses,
            lines: &self.net.lines,
            transformers: &self.net.transformers,
            branch_ids: &self.net.branch_ids,
            bus_ids: &[],
            initially_open: &[],
            bus_countries: &[],
            tap_changers: &[],
            base_mva: self.net.base_mva,
        }
    }

    fn plan(&self) -> gridoxide::rao::Plan {
        let mut solver = IpmSolver::new();
        run(&self.crac, &self.network(), &self.resolution(), &mut solver, &SearchOptions::default())
    }

    fn options(&self) -> ValidationOptions<'_> {
        ValidationOptions {
            ac: AcOptions { shunts: &self.net.shunts, ..Default::default() },
            ..Default::default()
        }
    }
}

#[test]
fn every_perimeter_of_the_plan_gets_a_verdict() {
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();
    let report = validate(&c.crac, &c.network(), &c.resolution(), &plan, &c.options());

    assert_eq!(
        report.perimeters.len(),
        plan.perimeters().count(),
        "a plan's perimeters and its verdicts have to correspond one to one"
    );
    for (v, p) in report.perimeters.iter().zip(plan.perimeters()) {
        assert_eq!(v.states, p.states);
        assert_eq!(v.dc_margin_mw, p.final_margin_mw, "the DC figure is the search's own");
    }
}

#[test]
fn the_two_models_disagree_and_the_disagreement_is_reported() {
    // If this ever passed with a zero regression everywhere, the AC stage would
    // not be doing an AC solve.
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();
    let report = validate(&c.crac, &c.network(), &c.resolution(), &plan, &c.options());

    let biggest = report
        .perimeters
        .iter()
        .map(|p| p.regression_mw().abs())
        .fold(0.0f64, f64::max);
    assert!(
        biggest > 1e-6,
        "AC reproduced the DC margins exactly, which means it did not run"
    );
    assert!(report.ac_margin_mw.is_finite(), "the plan should have measurable AC margins");
}

#[test]
fn a_threshold_no_plan_can_meet_rejects_it() {
    // The stage's job is to be able to say no. Demanding a gigawatt of margin
    // is not a realistic setting; it is the cheapest way to prove that a
    // failing measurement produces a rejection rather than being rounded into
    // acceptance.
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();
    let options = ValidationOptions { min_margin_mw: 1e6, ..c.options() };
    let report = validate(&c.crac, &c.network(), &c.resolution(), &plan, &options);

    assert!(!report.is_accepted());
    assert!(report.rejected().count() > 0);
    for p in report.rejected() {
        assert!(
            matches!(p.verdict, Verdict::Insecure { .. } | Verdict::Diverged),
            "{:?}",
            p.verdict
        );
    }
}

#[test]
fn a_regression_bound_catches_a_model_disagreement_that_is_still_secure() {
    // Insecurity and disagreement are separate failures. A perimeter can clear
    // its margin in both models and still be a warning, because the modelling
    // error that cost margin here could cost more than the margin elsewhere.
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();

    // Only perimeters that actually solved: a diverged one has the largest
    // apparent regression on this fixture, and divergence is a different
    // failure that legitimately outranks disagreement.
    let lax = validate(&c.crac, &c.network(), &c.resolution(), &plan, &c.options());
    let worst = lax
        .perimeters
        .iter()
        .filter(|p| p.verdict != Verdict::Diverged)
        .map(|p| p.regression_mw())
        .fold(f64::NEG_INFINITY, f64::max);
    if worst <= 0.0 {
        // AC was the more generous model everywhere on this fixture; there is
        // no regression to bound and nothing to assert.
        return;
    }

    let options =
        ValidationOptions { min_margin_mw: f64::NEG_INFINITY, max_regression_mw: worst / 2.0, ..c.options() };
    let strict = validate(&c.crac, &c.network(), &c.resolution(), &plan, &options);
    assert!(
        strict.rejected().any(|p| matches!(p.verdict, Verdict::Regressed { .. })),
        "a bound below the observed regression should reject: {:?}",
        strict.perimeters.iter().map(|p| (&p.verdict, p.regression_mw())).collect::<Vec<_>>()
    );
}

#[test]
fn a_secure_plan_is_accepted_under_the_default_thresholds() {
    // The defaults are "secure, and no bound on disagreement", so a plan the
    // search declared secure should pass unless AC genuinely contradicts it.
    let c = case("crac-for-12nodes.json");
    let plan = c.plan();
    let report = validate(&c.crac, &c.network(), &c.resolution(), &plan, &c.options());

    for (v, p) in report.perimeters.iter().zip(plan.perimeters()) {
        if p.final_margin_mw >= 0.0 && v.ac_margin_mw >= 0.0 {
            assert!(v.verdict.is_accepted(), "{:?} for {:?}", v.verdict, v.states);
        }
    }
}
