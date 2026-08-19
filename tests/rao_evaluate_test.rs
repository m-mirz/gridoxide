//! Evaluating a CRAC against a network — phase 5 of `plans/RAO_PLAN.md`.
//!
//! This is the half of a remedial action optimization that does not optimize,
//! and it is a deliverable on its own: gridoxide could run a contingency
//! analysis before this existed but could not say whether the result was
//! *acceptable*, because nothing told it what the limits were or which elements
//! anyone cared about.
//!
//! The strongest test here is `screened_outage_flows_match_a_full_resolve`. It
//! is independent of the CRAC layer entirely — it holds the Woodbury screening
//! path against a from-scratch re-solve — and it is the one that would catch a
//! wrong answer arriving quickly.

use std::path::PathBuf;

use gridoxide::rao::crac::*;
use gridoxide::rao::{crac_json, evaluate, Network, Resolution};
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

fn case() -> Case {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(rao_fixture("crac-for-12nodes.json")).expect("crac");
    Case { net, crac }
}

impl Case {
    fn network(&self) -> Network<'_> {
        Network {
            buses: &self.net.buses,
            lines: &self.net.lines,
            transformers: &self.net.transformers,
            branch_ids: &self.net.branch_ids,
            bus_ids: &[],
            initially_open: &[],
            tap_changers: &[],
            base_mva: self.net.base_mva,
        }
    }
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

#[test]
fn a_crac_written_for_this_network_resolves_completely() {
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    assert!(
        resolution.is_complete(),
        "unresolved elements: {:?}",
        resolution.unresolved
    );
    assert!(resolution.branch_of.len() >= 10);
}

#[test]
fn ids_match_with_or_without_ucte_column_padding() {
    // UCTE element ids are fixed-width and carry padding; a CRAC written
    // against the same network may or may not preserve it. Matching only
    // exactly would leave every CNEC unresolved on half the corpus.
    let c = case();
    let padded = c.net.branch_ids[0].clone();
    assert!(padded.contains("  "), "fixture should have padded ids: {padded:?}");

    let crac = Crac {
        flow_cnecs: vec![FlowCnec {
            id: "x".into(),
            network_element: padded.trim().to_string(),
            state: State::preventive(0),
            thresholds: vec![],
            reliability_margin: 0.0,
            optimized: true,
            monitored: false,
            operator: None,
            i_max: None,
            nominal_v: None,
        }],
        instants: vec![Instant { id: "preventive".into(), kind: InstantKind::Preventive }],
        ..Default::default()
    };
    let resolution = Resolution::new(&crac, &c.net.branch_ids);
    assert!(resolution.is_complete(), "trimmed id did not match: {:?}", resolution.unresolved);
}

#[test]
fn an_unresolvable_element_is_reported_and_its_cnec_skipped() {
    // A CNEC that silently disappears is a constraint the optimizer never sees,
    // which produces a confident answer to the wrong question.
    let c = case();
    let mut crac = c.crac.clone();
    crac.flow_cnecs[0].network_element = "NO SUCH BRANCH".into();
    let resolution = Resolution::new(&crac, &c.net.branch_ids);
    assert!(!resolution.is_complete());
    assert!(resolution.unresolved.iter().any(|e| e == "NO SUCH BRANCH"));

    let result = evaluate(&crac, &c.network(), &resolution);
    assert_eq!(result.skipped, vec![0]);
}

// ---------------------------------------------------------------------------
// Threshold conversion
// ---------------------------------------------------------------------------

#[test]
fn an_ampere_threshold_converts_through_root_three() {
    // 2165 A converts to 1500 MW — and at **400 kV**, not at the branch's own
    // 380 kV base, because the CRAC states `nominalV: 400.0` and that is the
    // voltage its author wrote the threshold against. A UCTE 380 kV node is
    // routinely operated at 400; converting at the base instead makes every
    // ampere threshold 5% tight, which reads as a slightly more constrained
    // network rather than as an error.
    //
    // Dropping the sqrt(3) would give 866 MW, wrong by 42% and still a
    // plausible-looking line rating.
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate(&c.crac, &c.network(), &resolution);
    let preventive = &result.perimeters[0];
    let limit = preventive
        .cnecs
        .iter()
        .map(|r| r.limit_mw)
        .find(|l| (*l - 1500.0).abs() < 1.0)
        .expect("a 2165 A threshold at 400 kV should convert to about 1500 MW");
    assert!((limit - 1500.0).abs() < 1.0, "got {limit}");
}

/// `percent_imax` is a **fraction**, not a percentage.
///
/// The reference's own `ThresholdAdder` javadoc says so — "the min/max value
/// should be between -1 and 1, where 1 = 100%" — and reading it as a percentage
/// makes every such threshold a hundred times too tight. That does not look
/// like a bug. It looks like a network that is massively overloaded, and it was
/// caught only by noticing that a 380 kV line had come out with a 33 MW limit.
#[test]
fn percent_imax_is_a_fraction_not_a_percentage() {
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate(&c.crac, &c.network(), &resolution);
    for perimeter in &result.perimeters {
        for r in &perimeter.cnecs {
            assert!(
                r.limit_mw > 100.0,
                "cnec `{}` has a {} MW limit on a 380 kV branch, which is a factor-of-100 error",
                c.crac.flow_cnecs[r.cnec].id,
                r.limit_mw
            );
        }
    }
}

#[test]
fn a_reliability_margin_tightens_the_limit() {
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let base = evaluate(&c.crac, &c.network(), &resolution);

    let mut tightened = c.crac.clone();
    for cnec in &mut tightened.flow_cnecs {
        cnec.reliability_margin = 100.0;
    }
    let after = evaluate(&tightened, &c.network(), &resolution);

    let before_limit = base.perimeters[0].cnecs[0].limit_mw;
    let after_limit = after.perimeters[0].cnecs[0].limit_mw;
    assert!(
        (before_limit - after_limit - 100.0).abs() < 1e-9,
        "{before_limit} -> {after_limit}"
    );
}

// ---------------------------------------------------------------------------
// Flows
// ---------------------------------------------------------------------------

#[test]
fn base_case_flows_are_the_dc_solution() {
    // The evaluator must not have its own idea of what the flows are.
    use gridoxide::linear::btheta::dc_power_flow;
    let c = case();
    let mut buses = c.net.buses.clone();
    let expected =
        dc_power_flow(&mut buses, &c.net.lines, &c.net.transformers, Default::default()).branch_p;

    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate(&c.crac, &c.network(), &resolution);
    let preventive = result
        .perimeters
        .iter()
        .find(|p| p.state.is_preventive())
        .expect("a preventive perimeter");
    for r in &preventive.cnecs {
        let want = expected[r.branch] * c.net.base_mva;
        assert!((r.flow_mw - want).abs() < 1e-9, "branch {}: {} vs {want}", r.branch, r.flow_mw);
    }
}

/// The gate that matters: screening must agree with re-solving.
///
/// `multi_outage_flows` answers a contingency with a Woodbury update against a
/// cached factorization instead of a fresh solve. That is where the speed comes
/// from, and it is exactly the kind of shortcut that can be fast and wrong.
/// This holds it against a from-scratch re-solve of the outaged network, and it
/// does not involve the CRAC layer at all.
#[test]
fn screened_outage_flows_match_a_full_resolve() {
    use gridoxide::linear::btheta::{dc_branches, dc_power_flow};
    use gridoxide::linear::sensitivity::DcSensitivity;

    let c = case();
    let options = Default::default();
    let dc = dc_branches(&c.net.lines, &c.net.transformers, options);
    let mut buses = c.net.buses.clone();
    let base = dc_power_flow(&mut buses, &c.net.lines, &c.net.transformers, options).branch_p;
    let sensitivity =
        DcSensitivity::new(&c.net.buses, &dc, c.net.n_branches()).expect("sensitivity");

    let mut compared = 0;
    for outage in 0..c.net.lines.len() {
        if sensitivity.is_breaking_set(&[outage]) {
            continue;
        }
        let Some(screened) = sensitivity.multi_outage_flows(&base, &[outage]) else { continue };

        // Re-solve with that line effectively removed.
        let mut lines = c.net.lines.clone();
        lines[outage].r = 1e9;
        lines[outage].x = 1e9;
        lines[outage].b_shunt = 0.0;
        lines[outage].g_shunt = 0.0;
        let mut buses = c.net.buses.clone();
        let resolved = dc_power_flow(&mut buses, &lines, &c.net.transformers, options).branch_p;

        for b in 0..screened.len() {
            if b == outage {
                continue;
            }
            let diff = (screened[b] - resolved[b]).abs() * c.net.base_mva;
            assert!(diff < 1e-6, "outage {outage}, branch {b}: {diff} MW apart");
        }
        compared += 1;
    }
    assert!(compared >= 10, "only {compared} outages were screenable");
}

#[test]
fn a_contingency_actually_changes_the_flows() {
    // A contingency that resolves but is never applied would produce a
    // perfectly plausible post-contingency perimeter identical to the base
    // case — the quietest possible way to report a network as secure.
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate(&c.crac, &c.network(), &resolution);

    let preventive = result.perimeters.iter().find(|p| p.state.is_preventive()).unwrap();
    let outage = result
        .perimeters
        .iter()
        .find(|p| p.state.contingency.is_some() && !p.severed)
        .expect("a post-contingency perimeter");

    let shared: Vec<(f64, f64)> = outage
        .cnecs
        .iter()
        .filter_map(|a| {
            preventive.cnecs.iter().find(|b| b.branch == a.branch).map(|b| (a.flow_mw, b.flow_mw))
        })
        .collect();
    assert!(!shared.is_empty(), "no branch is monitored in both perimeters");
    assert!(
        shared.iter().any(|(a, b)| (a - b).abs() > 1.0),
        "the contingency moved nothing: {shared:?}"
    );
}

#[test]
fn every_state_the_crac_defines_gets_a_perimeter() {
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate(&c.crac, &c.network(), &resolution);
    for state in c.crac.states() {
        assert!(
            result.perimeters.iter().any(|p| p.state == state),
            "no perimeter for {state:?}"
        );
    }
    // Preventive first, then by instant — the order an optimization visits them.
    assert!(result.perimeters.windows(2).all(|w| w[0].state.instant <= w[1].state.instant));
}

// ---------------------------------------------------------------------------
// Margins
// ---------------------------------------------------------------------------

#[test]
fn a_margin_is_the_limit_less_the_absolute_flow() {
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate(&c.crac, &c.network(), &resolution);
    for perimeter in &result.perimeters {
        for r in &perimeter.cnecs {
            assert!(
                (r.margin_mw - (r.limit_mw - r.flow_mw.abs())).abs() < 1e-9,
                "margin does not agree with limit and flow: {r:?}"
            );
            assert_eq!(r.is_violated(), r.margin_mw < 0.0);
        }
    }
}

#[test]
fn direction_does_not_change_whether_a_branch_is_overloaded() {
    // Flow sign is a labelling convention; an overload is an overload either
    // way. A margin computed on the signed flow would call a heavily loaded
    // branch secure whenever the power happened to run the other way.
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate(&c.crac, &c.network(), &resolution);
    let negative_and_violated = result
        .violations()
        .any(|(_, r)| r.flow_mw < 0.0);
    assert!(
        negative_and_violated,
        "this fixture has a violated branch flowing backwards; none was found"
    );
}

#[test]
fn the_worst_margin_is_reported_across_every_perimeter() {
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate(&c.crac, &c.network(), &resolution);

    let worst = result.min_margin().expect("a margin");
    let by_hand = result
        .perimeters
        .iter()
        .flat_map(|p| p.cnecs.iter().map(|c| c.margin_mw))
        .fold(f64::INFINITY, f64::min);
    assert!((worst - by_hand).abs() < 1e-12);
    assert_eq!(result.is_secure(), worst >= 0.0);
    // This fixture is genuinely insecure, which is what makes it useful.
    assert!(!result.is_secure(), "the fixture should have violations to find");
}

#[test]
fn a_severing_contingency_is_flagged_rather_than_silently_wrong() {
    // Opening a radial branch disconnects part of the network, and the Woodbury
    // update has no answer for it. Falling back to a re-solve is right; not
    // saying so is not.
    let c = case();
    let mut crac = c.crac.clone();
    // Point the contingency at something that cannot be screened away.
    crac.contingencies[0].elements = vec!["NOT A BRANCH".into()];
    let resolution = Resolution::new(&crac, &c.net.branch_ids);
    let result = evaluate(&crac, &c.network(), &resolution);
    assert!(
        result.perimeters.iter().filter(|p| p.state.contingency.is_some()).all(|p| p.severed),
        "an unsimulatable contingency must be flagged"
    );
}
