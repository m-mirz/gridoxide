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
            generation: &self.net.generation,
            buses: &self.net.buses,
            lines: &self.net.lines,
            transformers: &self.net.transformers,
            branch_ids: &self.net.branch_ids,
            bus_ids: &[],
            initially_open: &[],
            bus_countries: &[],
            shunts: &[],
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
fn a_margin_is_the_distance_to_the_nearer_bound() {
    // `min(upper - flow, flow - lower)`, the reference's own
    // `computeMargin`, with an absent bound treated as infinite. For a
    // symmetric CNEC that reduces to `limit - |flow|`; for a one-sided one it
    // does not, and the difference is the whole point — a CNEC with no lower
    // threshold puts no constraint on reverse flow at all.
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate(&c.crac, &c.network(), &resolution);
    let mut one_sided = 0;
    for perimeter in &result.perimeters {
        for r in &perimeter.cnecs {
            let expected = f64::min(r.upper_mw - r.flow_mw, r.flow_mw - r.lower_mw);
            assert!(
                (r.margin_mw - expected).abs() < 1e-9,
                "margin does not agree with its bounds: {r:?}"
            );
            assert_eq!(r.is_violated(), r.margin_mw < 0.0);
            if !r.upper_mw.is_finite() || !r.lower_mw.is_finite() {
                one_sided += 1;
            } else {
                // The symmetric identity still holds where both bounds exist.
                assert!(
                    (r.margin_mw - (r.limit_mw - r.flow_mw.abs())).abs() < 1e-9,
                    "a two-sided CNEC should still be limit - |flow|: {r:?}"
                );
            }
        }
    }
    assert!(one_sided > 0, "this fixture should exercise a one-sided threshold");
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

// ---------------------------------------------------------------------------
// The AC flow model
// ---------------------------------------------------------------------------

/// `(p1, q1, p2, q2)` per branch id, from the vendored pypowsybl solution.
fn reference_flows(name: &str) -> std::collections::HashMap<String, [f64; 4]> {
    let text = std::fs::read_to_string(ucte_fixture(name)).expect("reference");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("json");
    doc["branches"]
        .as_object()
        .expect("branches")
        .iter()
        .map(|(id, v)| {
            let get = |k: &str| v[k].as_f64().unwrap_or(0.0);
            (id.clone(), [get("p1"), get("q1"), get("p2"), get("q2")])
        })
        .collect()
}

/// Solved voltage magnitude, per unit, per node code.
fn reference_voltages(name: &str) -> std::collections::HashMap<String, f64> {
    let text = std::fs::read_to_string(ucte_fixture(name)).expect("reference");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("json");
    doc["buses"]
        .as_object()
        .expect("buses")
        .iter()
        .map(|(id, v)| (id.trim().to_string(), v["v_pu"].as_f64().unwrap_or(1.0)))
        .collect()
}

#[test]
fn ac_currents_match_the_reference_load_flow() {
    // The gate for the AC path. `flow_mw` and `current_a` are recomputed here
    // from pypowsybl's own `(p, q)` rather than from anything gridoxide
    // produced, so a sign convention or a per-unit slip in the evaluator shows
    // up as a mismatch rather than as two implementations agreeing on a shared
    // mistake.
    let c = case();
    let reference = reference_flows("TestCase12Nodes.pypowsybl.json");
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let ac = evaluate::AcOptions { shunts: &c.net.shunts, ..Default::default() };
    let result =
        evaluate::evaluate_ac(&c.crac, &c.network(), &resolution, &[], &ac);

    let base = result
        .perimeters
        .iter()
        .find(|p| p.state.contingency.is_none())
        .expect("a preventive perimeter");
    assert!(!base.severed, "the intact network should solve");
    assert!(!base.cnecs.is_empty(), "the CRAC should monitor something");

    let reference_voltages = reference_voltages("TestCase12Nodes.pypowsybl.json");
    let mut checked = 0;
    for cnec in &base.cnecs {
        let id = &c.net.branch_ids[cnec.branch];
        let Some([p1, q1, ..]) = reference.get(id.trim()).or_else(|| reference.get(id)) else {
            continue;
        };
        assert!(
            (cnec.flow_mw - p1).abs() < 0.5,
            "{id}: flow {} MW against the reference's {p1} MW",
            cnec.flow_mw
        );

        // The evaluator reports current at the *actual* bus voltage, so the
        // expectation has to be built the same way — using nominal here would
        // disagree by however far the solution sits from 1.0 pu, which on this
        // fixture is 5.3%. Both the power and the voltage come from
        // pypowsybl's own solution, so nothing gridoxide computed appears on
        // the expected side.
        let bus = c.net.lines.get(cnec.branch).map(|l| l.from);
        if let Some(bus) = bus {
            let code = c.net.node_codes[bus].trim().to_string();
            let Some(v_pu) = reference_voltages.get(&code) else { continue };
            let u = c.net.buses[bus].u_rated * v_pu;
            let expected = p1.hypot(*q1) * 1e6 / (3f64.sqrt() * u);
            assert!(
                (cnec.current_a - expected).abs() < 2.0,
                "{id}: {} A against {expected} A",
                cnec.current_a
            );
            checked += 1;
        }
    }
    assert!(checked >= 3, "expected several comparable branches, got {checked}");

    // Converting at nominal instead of at the solved voltage would still pass a
    // loose tolerance on a flat network, so assert the fixture is not flat: the
    // rule under test only has teeth where the voltages actually move.
    let spread = reference_voltages.values().map(|v| (v - 1.0).abs()).fold(0.0f64, f64::max);
    assert!(spread > 0.01, "flat voltages would make this test vacuous (spread {spread})");
}

#[test]
fn dc_and_ac_disagree_by_the_reactive_flow() {
    // Not a tolerance check — a check that the two models are genuinely
    // different. If AC evaluation silently fell back to the DC path, every
    // margin would match to the last decimal and this would fail.
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let ac = evaluate::AcOptions { shunts: &c.net.shunts, ..Default::default() };
    let dc = evaluate::evaluate(&c.crac, &c.network(), &resolution);
    let acr = evaluate::evaluate_ac(&c.crac, &c.network(), &resolution, &[], &ac);

    let worst = |r: &evaluate::SecurityResult| {
        r.perimeters
            .iter()
            .flat_map(|p| p.cnecs.iter())
            .map(|c| c.margin_mw)
            .fold(f64::INFINITY, f64::min)
    };
    let (a, b) = (worst(&dc), worst(&acr));
    assert!(a.is_finite() && b.is_finite(), "both models should measure something");
    assert!((a - b).abs() > 1e-6, "AC and DC gave identical margins ({a} vs {b})");

    // The reactive charge only ever consumes headroom, so no AC limit can
    // exceed its DC counterpart on the same CNEC.
    for (p, q) in dc.perimeters.iter().zip(&acr.perimeters) {
        for (x, y) in p.cnecs.iter().zip(&q.cnecs) {
            assert_eq!(x.cnec, y.cnec);
            assert!(
                y.limit_mw <= x.limit_mw + 1e-6,
                "cnec {}: AC limit {} above the DC limit {}",
                x.cnec,
                y.limit_mw,
                x.limit_mw
            );
        }
    }
}

#[test]
fn an_outaged_branch_carries_nothing_in_ac() {
    // `solve_contingencies` removes the branch from the Y-bus, but the branch
    // parameters it was built from are still in the list. Evaluating them
    // would report the current that *would* flow across a branch that is not
    // there — a violation on an element that is out of service.
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let ac = evaluate::AcOptions { shunts: &c.net.shunts, ..Default::default() };

    let monitored: Vec<usize> = c.crac.flow_cnecs
        .iter()
        .filter_map(|cnec| resolution.branch(&cnec.network_element))
        .collect();
    let target = *monitored.first().expect("a monitored branch");

    let result =
        evaluate::evaluate_ac(&c.crac, &c.network(), &resolution, &[target], &ac);
    for p in &result.perimeters {
        for cnec in p.cnecs.iter().filter(|c| c.branch == target) {
            assert_eq!(cnec.flow_mw, 0.0, "an open branch should carry no power");
            assert_eq!(cnec.current_a, 0.0, "an open branch should carry no current");
        }
    }
}

#[test]
fn the_model_selector_dispatches_to_the_two_paths() {
    // `evaluate_model` is the public entry point for "measure this, with that
    // model". If it ever routed both arms to the same place, every AC caller
    // would silently get DC answers.
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let ac = evaluate::AcOptions { shunts: &c.net.shunts, ..Default::default() };

    let via_dc = evaluate::evaluate_model(
        &c.crac, &c.network(), &resolution, &[], evaluate::FlowModel::Dc, &ac,
    );
    let via_ac = evaluate::evaluate_model(
        &c.crac, &c.network(), &resolution, &[], evaluate::FlowModel::Ac, &ac,
    );
    let direct_dc = evaluate::evaluate_with(&c.crac, &c.network(), &resolution, &[]);
    let direct_ac = evaluate::evaluate_ac(&c.crac, &c.network(), &resolution, &[], &ac);

    assert_eq!(via_dc.perimeters, direct_dc.perimeters, "the DC arm must be the DC path");
    assert_eq!(via_ac.perimeters, direct_ac.perimeters, "the AC arm must be the AC path");
    assert_ne!(via_dc.perimeters, via_ac.perimeters, "the two arms must differ");
    assert_eq!(evaluate::FlowModel::default(), evaluate::FlowModel::Dc);
}


#[test]
fn distributing_the_slack_solves_and_agrees_where_losses_are_negligible() {
    // The reference's configurations all set `distributedSlack: true` with
    // `PROPORTIONAL_TO_GENERATION_P`. What it changes is where the loss
    // balance lands, so on a network whose losses are near zero — which the
    // vendored UCTE fixtures are — it should agree with a single slack to
    // well inside any threshold, and that agreement is the check: a
    // distribution that quietly failed to converge would not land here.
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let single = evaluate::AcOptions { shunts: &c.net.shunts, ..Default::default() };
    let spread = evaluate::AcOptions { distribute_slack: true, ..single };

    let a = evaluate::evaluate_ac(&c.crac, &c.network(), &resolution, &[], &single);
    let b = evaluate::evaluate_ac(&c.crac, &c.network(), &resolution, &[], &spread);

    assert_eq!(a.perimeters.len(), b.perimeters.len());
    let mut compared = 0;
    for (p, q) in a.perimeters.iter().zip(&b.perimeters) {
        assert_eq!(p.severed, q.severed, "convergence should not differ on {:?}", p.state);
        for (x, y) in p.cnecs.iter().zip(&q.cnecs) {
            assert!(
                (x.margin_mw - y.margin_mw).abs() < 1.0,
                "cnec {}: {} MW against {} MW",
                x.cnec,
                x.margin_mw,
                y.margin_mw
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "nothing was compared");
}

#[test]
fn side_two_is_the_far_end_of_the_same_flow() {
    // Gated against pypowsybl's own `p2`, not against gridoxide's `p1`, because
    // the question is a *convention* and an internal check cannot settle one.
    //
    // powsybl reports `terminal.getP()` — power **entering** the branch — at
    // both ends, so its `p2` is the negation of what `side_two` reports. Side
    // two is the power *leaving*, measured in the same direction along the
    // branch as `flow_mw`, so that the two differ by the branch's losses rather
    // than by twice the flow. That is what the reference's own per-side flow
    // expectations mean, and reading it the other way agrees on magnitude while
    // being wrong about direction on every one of them.
    let c = case();
    let reference = reference_flows("TestCase12Nodes.pypowsybl.json");
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let ac = evaluate::AcOptions { shunts: &c.net.shunts, ..Default::default() };
    let result = evaluate::evaluate_ac(&c.crac, &c.network(), &resolution, &[], &ac);

    let base = result
        .perimeters
        .iter()
        .find(|p| p.state.contingency.is_none())
        .expect("a preventive perimeter");

    let mut checked = 0;
    for cnec in &base.cnecs {
        let id = &c.net.branch_ids[cnec.branch];
        let Some([p1, _, p2, _]) = reference.get(id.trim()).or_else(|| reference.get(id)) else {
            continue;
        };
        assert!(
            (cnec.side_two.0 - -p2).abs() < 0.5,
            "{id}: side two {} MW against the reference's {} MW",
            cnec.side_two.0,
            -p2
        );
        // The losses, and nothing more. On these near-lossless fixtures that is
        // a fraction of a megawatt, which is exactly the point: a sign error
        // here would show up as twice the flow.
        assert!(
            (cnec.flow_mw - cnec.side_two.0).abs() < 0.05 * p1.abs().max(1.0),
            "{id}: {} MW in, {} MW out — that is not losses",
            cnec.flow_mw,
            cnec.side_two.0
        );
        checked += 1;
    }
    assert!(checked >= 2, "only {checked} branches checked");
}

#[test]
fn a_dc_far_end_carries_exactly_what_the_near_end_took() {
    // DC is lossless, so there is no room for the two sides to differ in power
    // at all — only in current, and then only across a transformer, where the
    // two buses' nominal voltages differ.
    let c = case();
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let result = evaluate::evaluate(&c.crac, &c.network(), &resolution);

    let mut checked = 0;
    for perimeter in &result.perimeters {
        for cnec in &perimeter.cnecs {
            assert!(
                (cnec.flow_mw - cnec.side_two.0).abs() < 1e-9,
                "{}: {} MW in, {} MW out under a lossless model",
                c.net.branch_ids[cnec.branch],
                cnec.flow_mw,
                cnec.side_two.0
            );
            assert!(cnec.side_two.1 >= 0.0, "a current is a magnitude");
            checked += 1;
        }
    }
    assert!(checked > 0, "the CRAC should monitor something");
}

/// An **ampere** margin is a difference of two ampere quantities, not the
/// megawatt margin converted.
///
/// The distinction is invisible for a threshold already written in amperes,
/// where the evaluator's charge has taken reactive flow and the voltage
/// deviation off the megawatt limit and converting back undoes exactly that.
/// For a threshold written in **megawatts** it is not invisible, because
/// `limit − |P|` carries neither effect while the current carries both.
///
/// The reference's `epic5` fixture puts a 2000 MW threshold on a line whose
/// nodes are 380 kV in the file and which runs at about 400. Two independent
/// operating points pin the answer to a tenth of an ampere:
///
/// | actions | P (MW) | I (A) | reference margin |
/// |---|---|---|---|
/// | none | 1499.8 | 2167.1 | 871 A |
/// | one | 1308.0 | 1889.5 | 1149 A |
///
/// Both are `2000 MW at 380 kV − I` = 3038.7 − I. Converting the megawatt
/// margin instead gives 722 and 999 — right to within 0.3 MW on the megawatt
/// assertion in the same scenario, and 150 A adrift here. That 150 A is the
/// **network's** 380 kV against the CRAC's stated `nominalV` of 400: the
/// reference reads a CNEC's nominal voltage off the network, and a megawatt
/// threshold needs no other voltage to be stated in.
#[test]
fn an_ampere_margin_is_measured_in_amperes() {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep5us1.json"),
    )
    .expect("crac");
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        generation: &net.generation,
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &net.shunts,
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let ac = gridoxide::rao::AcOptions { shunts: &net.shunts, ..Default::default() };
    let result = gridoxide::rao::evaluate_model(
        &crac,
        &network,
        &resolution,
        &net.initially_open,
        gridoxide::rao::FlowModel::Ac,
        &ac,
    );

    let cnec = result
        .perimeters
        .iter()
        .flat_map(|p| p.cnecs.iter())
        .find(|c| crac.flow_cnecs[c.cnec].id == "FFR2AA1  DDE3AA1  1 - preventive")
        .expect("the CRAC's preventive CNEC");

    // The megawatt margin is `limit − |P|` and was right all along.
    assert!(
        (cnec.margin_mw - 500.0).abs() < 5.0,
        "megawatt margin {} MW, reference 500",
        cnec.margin_mw
    );
    // The ampere margin is the limit in amperes less the current.
    let limit_a = 2000.0 * 1e6 / (3f64.sqrt() * 380e3);
    assert!(
        (cnec.margin_a - (limit_a - cnec.current_a)).abs() < 0.5,
        "ampere margin {} A is not {} - {}",
        cnec.margin_a,
        limit_a,
        cnec.current_a
    );
    assert!(
        (cnec.margin_a - 871.0).abs() < 5.0,
        "ampere margin {} A, reference 871",
        cnec.margin_a
    );
    // And emphatically not the megawatt margin converted, at either voltage.
    for v in [380e3, 400e3] {
        let converted = cnec.margin_mw * 1e6 / (3f64.sqrt() * v);
        assert!(
            (cnec.margin_a - converted).abs() > 50.0,
            "converting the megawatt margin at {v} V gives {converted} A, which is the \
             reading this test exists to rule out"
        );
    }
}

/// The slack is distributed, and the weights are **generation**.
///
/// Both halves are load-bearing and the second is the one that hides.
///
/// The reference's `epic5` scenario opens both of `FFR1AA1`'s branches — it has
/// only those two — which islands a node carrying 2000 MW of generation against
/// 1000 MW of load. The main component loses 1000 MW of net injection and
/// something has to supply it, and *where* it appears decides the answer:
/// this network's slack is `BBE2AA1`, one end of the Belgium–France tie, so a
/// single slack pushes its entire make-up straight through France and out over
/// the CNEC being measured.
///
/// Against the reference's 1000 MW on `FFR2AA1  DDE3AA1  1`:
///
/// | how the 1000 MW is supplied | flow |
/// |---|---|
/// | single slack | 1165.5 MW |
/// | distributed, weighted by **net injection** | 1160.8 MW |
/// | distributed, weighted by **generation** | 1000.2 MW |
///
/// That middle row is why this looked like an innocent setting for a long time:
/// netting generation against load puts 40% of the make-up back at `BBE2AA1`
/// and 30% at `FFR3AA1` — 70% of it inside or next to the country that lost it
/// — so distributing barely moves the answer and the whole idea looks refuted.
/// It is the weighting that was wrong, and every configuration the reference
/// ships says so outright: `PROPORTIONAL_TO_GENERATION_P`.
#[test]
fn the_slack_is_shared_out_in_proportion_to_generation() {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/rao/features/SL_ep5us1.json"),
    )
    .expect("crac");
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &net.shunts,
        generation: &net.generation,
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };

    // Both of the islanded node's branches.
    let mut open = net.initially_open.clone();
    for name in ["Open tie-line FR1 FR2", "Open tie-line FR1 FR3"] {
        let action = crac.network_actions.iter().find(|a| a.id == name).expect(name);
        for elementary in &action.elementary {
            for element in elementary.elements() {
                if let Some(b) = resolution.branch(element) {
                    open.push(b);
                }
            }
        }
    }

    let flow = |ac: &gridoxide::rao::AcOptions<'_>| -> f64 {
        gridoxide::rao::evaluate_model(
            &crac,
            &network,
            &resolution,
            &open,
            gridoxide::rao::FlowModel::Ac,
            ac,
        )
        .perimeters
        .iter()
        .flat_map(|p| p.cnecs.iter())
        .find(|c| crac.flow_cnecs[c.cnec].id == "FFR2AA1  DDE3AA1  1 - preventive")
        .map(|c| c.flow_mw)
        .expect("the preventive CNEC")
    };

    let shared = flow(&gridoxide::rao::ac_options(&network));
    assert!(
        (shared - 1000.0).abs() < 5.0,
        "generation-weighted, the flow should be the reference's 1000 MW, not {shared}"
    );

    // The weighting, not merely the distributing. Both wrong answers are what
    // this test exists to keep out.
    let netted = flow(&gridoxide::rao::AcOptions {
        shunts: &net.shunts,
        distribute_slack: true,
        slack_weights: &[],
        ..Default::default()
    });
    let single = flow(&gridoxide::rao::AcOptions {
        shunts: &net.shunts,
        distribute_slack: false,
        ..Default::default()
    });
    assert!(
        netted > 1100.0 && single > 1100.0,
        "weighting by net injection ({netted}) and a single slack ({single}) should both \
         still push the make-up through France"
    );
}

/// The UCTE importer keeps generation apart from the load it is netted against.
#[test]
fn generation_is_retained_alongside_the_net_injection() {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    assert_eq!(net.generation.len(), net.buses.len());

    // FFR1AA1: 2000 MW of generation behind 1000 MW of load. The net figure
    // cannot tell it apart from a node generating 1000 behind nothing, which is
    // exactly why it is kept.
    let fr1 = net.node_codes.iter().position(|c| c.trim() == "FFR1AA1").expect("FFR1AA1");
    assert!((net.generation[fr1] * net.base_mva - 2000.0).abs() < 1e-6);
    assert!((net.buses[fr1].p_spec * net.base_mva - 1000.0).abs() < 1e-6);
    assert!(net.generation.iter().all(|g| *g >= 0.0), "generation is stored positive");
}
