//! The CRAC data layer — phase 4 of `plans/RAO_PLAN.md`.
//!
//! Two things are being checked here, and they pull in opposite directions.
//!
//! **Tolerance**, because the OpenRAO JSON format has 24 versions in the wild
//! and renamed things between them. A reader that handles only the newest
//! spelling reads a third of the available material — and, worse, reads the
//! rest *partially*, producing a CRAC with fewer remedial actions than the file
//! declared and no indication that anything went missing.
//!
//! **Strictness about what it could not read**, because that is the failure
//! mode that matters. A dropped remedial action does not make an optimization
//! fail; it makes it report that no action helps, which looks like an answer.

use std::path::PathBuf;

use gridoxide::rao::crac::*;
use gridoxide::rao::crac_json;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/rao").join(name)
}

fn read(name: &str) -> (Crac, crac_json::CracReport) {
    let text = std::fs::read_to_string(fixture(name)).expect("fixture");
    crac_json::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn all_fixtures() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(fixture(""))
        .expect("fixture dir")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "json").then(|| p.file_name()?.to_str().map(str::to_string))?
        })
        .collect();
    names.sort();
    names
}

// ---------------------------------------------------------------------------
// Version tolerance
// ---------------------------------------------------------------------------

#[test]
fn every_vendored_format_version_reads() {
    let names = all_fixtures();
    assert!(names.len() >= 4, "expected several fixtures, found {names:?}");
    let mut versions: Vec<String> = Vec::new();
    for name in &names {
        let (crac, report) = read(name);
        assert!(!crac.instants.is_empty(), "{name}: no instants");
        assert!(!crac.flow_cnecs.is_empty(), "{name}: no CNECs");
        if let Some(v) = report.version.clone() {
            versions.push(v);
        }
    }
    versions.sort();
    versions.dedup();
    assert!(versions.len() >= 4, "expected several format versions, saw {versions:?}");
}

/// OpenRAO writes bare `NaN` for an absent rated current, which is not valid
/// JSON and which every strict parser rejects.
///
/// 98 of the 428 CRACs in the reference checkout contain it. A reader that does
/// not handle it does not read them *partially* — it fails outright, which is at
/// least honest, but it also means nearly a quarter of the corpus is unavailable
/// as test material.
#[test]
fn bare_nan_literals_do_not_stop_the_parse() {
    let raw = std::fs::read_to_string(fixture("crac-v2.6.json")).expect("fixture");
    assert!(raw.contains("NaN"), "this fixture is supposed to contain NaN");
    assert!(
        serde_json::from_str::<serde_json::Value>(&raw).is_err(),
        "if this became valid JSON the test no longer proves anything"
    );
    let (crac, _) = read("crac-v2.6.json");
    assert!(!crac.flow_cnecs.is_empty());
}

/// A `NaN` inside a *string* must survive, or an element legitimately named
/// "NaN" would be silently renamed.
#[test]
fn neutralising_nan_leaves_strings_alone() {
    let doc = r#"{"type":"CRAC","version":"2.6","id":"c",
      "instants":[{"id":"preventive","kind":"PREVENTIVE"}],
      "flowCnecs":[{"id":"NaN cnec","networkElementId":"NaN Substation",
                    "instant":"preventive","iMax":[NaN],
                    "thresholds":[{"unit":"ampere","min":-1.0,"max":1.0,"side":1}]}]}"#;
    let (crac, _) = crac_json::parse(doc).expect("parse");
    assert_eq!(crac.flow_cnecs[0].id, "NaN cnec");
    assert_eq!(crac.flow_cnecs[0].network_element, "NaN Substation");
    assert_eq!(crac.flow_cnecs[0].i_max, Some([None, None]), "NaN should read as absent");
}

#[test]
fn both_generations_of_usage_rule_spelling_are_read() {
    // 1.x says `freeToUseUsageRules` / `onStateUsageRules`, 2.x says
    // `onInstantUsageRules` / `onContingencyStateUsageRules`. A reader that
    // knows one generation makes every action of the other unavailable — which
    // an optimizer reports as "nothing helps".
    let (old, _) = read("crac-for-rao-result-v1.8.json");
    let (new, _) = read("crac-v2.9.json");
    for (label, crac) in [("1.x", &old), ("2.x", &new)] {
        let rules: usize = crac
            .network_actions
            .iter()
            .map(|a| a.usage_rules.len())
            .chain(crac.range_actions.iter().map(|a| a.usage_rules.len()))
            .sum();
        assert!(rules > 0, "{label}: no usage rule was understood");
    }
}

/// The 1.x elementary-action spellings do not end in `Actions` at all.
///
/// `pstSetpoints` and `injectionSetpoints` were invisible both to the reader
/// and to its own unknown-key detector, so 99 network actions across the corpus
/// were dropped while the report said everything was fine. This is the
/// regression test for that.
#[test]
fn the_one_x_setpoint_spellings_are_read() {
    let (crac, report) = read("crac-for-rao-result-v1.8.json");
    let has_pst = crac.network_actions.iter().any(|a| {
        a.elementary.iter().any(|e| matches!(e, ElementaryAction::PstTapPosition { .. }))
    });
    let has_injection = crac.network_actions.iter().any(|a| {
        a.elementary.iter().any(|e| matches!(e, ElementaryAction::GeneratorSetpoint { .. }))
    });
    assert!(has_pst, "no `pstSetpoints` action read");
    assert!(has_injection, "no `injectionSetpoints` action read");
    // And the detector must not report them as unknown any more.
    assert!(!report.unknown_actions.contains_key("pstSetpoints"), "{report:?}");
    assert!(!report.unknown_actions.contains_key("injectionSetpoints"), "{report:?}");
}

/// A PST set-point in 1.x is a **tap position**, not an angle.
#[test]
fn a_pst_setpoint_is_read_as_a_tap_not_an_angle() {
    let (crac, _) = read("crac-for-rao-result-v1.8.json");
    let tap = crac
        .network_actions
        .iter()
        .flat_map(|a| &a.elementary)
        .find_map(|e| match e {
            ElementaryAction::PstTapPosition { tap, .. } => Some(*tap),
            _ => None,
        })
        .expect("a PST action");
    // The fixture says 15 — a tap. Reading it as degrees would put the shifter
    // somewhere no tap changer reaches.
    assert!(tap.abs() <= 100, "tap {tap} looks like an angle, not a position");
}

#[test]
fn an_unrecognised_action_is_counted_rather_than_ignored() {
    // Tolerance is the point; silent tolerance is the danger.
    let mut seen = false;
    for name in all_fixtures() {
        let (_, report) = read(&name);
        if !report.unknown_actions.is_empty() {
            seen = true;
            assert!(report.unknown_actions.values().all(|n| *n > 0));
        }
        for dropped in &report.dropped_actions {
            assert!(!dropped.is_empty(), "{name}: a dropped action had no id");
        }
    }
    assert!(seen, "the fixture set should include at least one unmodelled action kind");
}

// ---------------------------------------------------------------------------
// The model itself
// ---------------------------------------------------------------------------

#[test]
fn several_curative_instants_are_representable() {
    // The reason `Instant` is data and not a four-variant enum. `crac-v2.6`
    // declares two curative instants; an enum could hold only one.
    let (crac, _) = read("crac-v2.6.json");
    let curative = crac.curative_instants();
    assert!(curative.len() >= 2, "expected multi-curative, got {curative:?}");
    // And they stay in the order the file gave, since order is what sequences
    // the optimization.
    assert!(curative.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(crac.instants[crac.preventive_instant().unwrap()].kind, InstantKind::Preventive);
}

#[test]
fn cnecs_resolve_their_instant_and_contingency_to_indices() {
    let (crac, _) = read("crac-v2.6.json");
    for cnec in &crac.flow_cnecs {
        assert!(cnec.state.instant < crac.instants.len(), "{}: bad instant", cnec.id);
        if let Some(c) = cnec.state.contingency {
            assert!(c < crac.contingencies.len(), "{}: bad contingency", cnec.id);
        }
        // An outage-instant CNEC must name a contingency; a preventive one must
        // not. Getting this wrong silently optimizes the wrong perimeter.
        let kind = crac.instants[cnec.state.instant].kind;
        if kind == InstantKind::Preventive {
            assert!(cnec.state.is_preventive(), "{}: preventive CNEC has a contingency", cnec.id);
        } else {
            assert!(cnec.state.contingency.is_some(), "{}: post-contingency CNEC has none", cnec.id);
        }
    }
}

#[test]
fn a_dangling_reference_is_an_error_rather_than_a_silent_drop() {
    let doc = r#"{"type":"CRAC","version":"2.6","id":"c",
      "instants":[{"id":"preventive","kind":"PREVENTIVE"}],
      "flowCnecs":[{"id":"x","networkElementId":"ne","instant":"nonexistent",
                    "thresholds":[{"unit":"ampere","min":-1.0,"side":1}]}]}"#;
    let err = crac_json::parse(doc).unwrap_err();
    assert!(matches!(err, crac_json::CracError::UnknownInstant { .. }), "got {err}");
}

#[test]
fn thresholds_keep_one_sided_limits() {
    // A CNEC limited in one direction only is common and must not be widened
    // into a symmetric pair.
    let mut one_sided = 0;
    for name in all_fixtures() {
        let (crac, _) = read(&name);
        for cnec in &crac.flow_cnecs {
            for t in &cnec.thresholds {
                assert!(t.min.is_some() || t.max.is_some(), "{}: empty threshold", cnec.id);
                if t.min.is_none() || t.max.is_none() {
                    one_sided += 1;
                }
            }
        }
    }
    assert!(one_sided > 0, "the corpus contains one-sided thresholds; none survived");
}

#[test]
fn all_three_side_spellings_are_understood() {
    let doc = |side: &str| {
        format!(
            r#"{{"type":"CRAC","version":"2.6","id":"c",
              "instants":[{{"id":"preventive","kind":"PREVENTIVE"}}],
              "flowCnecs":[{{"id":"x","networkElementId":"ne","instant":"preventive",
                 "thresholds":[{{"unit":"ampere","min":-1.0,"max":1.0{side}}}]}}]}}"#
        )
    };
    for (spelling, expected) in [
        (r#","side":1"#, Side::One),
        (r#","side":2"#, Side::Two),
        (r#","side":"left""#, Side::One),
        (r#","side":"right""#, Side::Two),
        ("", Side::Both),
    ] {
        let (crac, _) = crac_json::parse(&doc(spelling)).expect("parse");
        assert_eq!(crac.flow_cnecs[0].thresholds[0].side, expected, "spelling `{spelling}`");
    }
}

#[test]
fn a_pst_range_action_keeps_its_tap_to_angle_map() {
    // The map is the whole reason a PST range action is not just a bounded
    // scalar: its set-point is an angle, its decision is a tap, and the map
    // between them is nonlinear and cannot be reconstructed from a step size.
    let (crac, _) = read("crac-v2.6.json");
    let map = crac
        .range_actions
        .iter()
        .find_map(|a| match &a.kind {
            RangeActionKind::Pst { tap_to_angle, .. } if !tap_to_angle.is_empty() => {
                Some(tap_to_angle.clone())
            }
            _ => None,
        })
        .expect("a PST range action with a conversion map");
    assert!(map.len() > 2, "a two-entry map is not a tap changer");
    assert!(map.windows(2).all(|w| w[0].0 < w[1].0), "taps are not ascending: {map:?}");
    let angles: Vec<f64> = map.iter().map(|(_, a)| *a).collect();
    assert!(angles.iter().any(|a| *a != angles[0]), "every tap maps to the same angle");

    // The two operations an optimizer actually performs on this table.
    let kind = crac
        .range_actions
        .iter()
        .find_map(|a| matches!(&a.kind, RangeActionKind::Pst { tap_to_angle, .. } if !tap_to_angle.is_empty()).then_some(&a.kind))
        .expect("a PST range action");
    let (tap, angle) = map[map.len() / 2];
    assert_eq!(kind.angle_at(tap), Some(angle));
    assert_eq!(kind.nearest_tap(angle), Some(tap));
}

#[test]
fn range_kinds_survive_and_default_to_absolute() {
    // Older files omit `rangeType` entirely, where every range was absolute.
    // Defaulting to a *relative* kind would silently re-anchor every bound.
    let doc = r#"{"type":"CRAC","version":"1.1","id":"c",
      "pstRangeActions":[{"id":"r","networkElementId":"pst",
        "ranges":[{"min":-5,"max":5}]}]}"#;
    let (crac, _) = crac_json::parse(doc).expect("parse");
    assert_eq!(crac.range_actions[0].ranges[0].kind, RangeKind::Absolute);

    let (rich, _) = read("crac-v2.6.json");
    let kinds: Vec<RangeKind> =
        rich.range_actions.iter().flat_map(|a| a.ranges.iter().map(|r| r.kind)).collect();
    assert!(kinds.contains(&RangeKind::Absolute));
    assert!(
        kinds.contains(&RangeKind::RelativeToInitialNetwork),
        "the corpus uses relative ranges: {kinds:?}"
    );
}

#[test]
fn usage_rules_answer_which_state_they_cover() {
    let (crac, _) = read("crac-v2.6.json");
    let preventive = State::preventive(crac.preventive_instant().expect("preventive"));
    let available = crac.network_actions_for(&preventive);
    assert!(!available.is_empty(), "no network action is available preventively");
    // An OnContingencyState rule must not leak into the preventive state.
    for action in &crac.network_actions {
        for rule in &action.usage_rules {
            if let UsageRule::OnContingencyState { state } = rule {
                assert!(!state.is_preventive(), "{}: contingency-state rule on preventive", action.id);
                assert!(!rule.covers(&preventive));
            }
        }
    }
}

#[test]
fn network_elements_are_collected_once_each() {
    let (crac, _) = read("crac-v2.6.json");
    let ids = crac.network_elements();
    assert!(!ids.is_empty());
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "network_elements returned duplicates");
    // Every CNEC's element must appear — that set is the input to resolution
    // against a real network.
    for cnec in &crac.flow_cnecs {
        assert!(ids.contains(&cnec.network_element.as_str()), "{} missing", cnec.network_element);
    }
}

#[test]
fn states_are_derived_in_chronological_order() {
    let (crac, _) = read("crac-v2.6.json");
    let states = crac.states();
    assert!(states.len() > 1);
    assert!(states.windows(2).all(|w| w[0].instant <= w[1].instant));
    // Every CNEC's state must be among them, or an optimization would skip a
    // perimeter that has constraints in it.
    for cnec in &crac.flow_cnecs {
        assert!(states.contains(&cnec.state), "{}: state not enumerated", cnec.id);
    }
}

// ---------------------------------------------------------------------------
// The native companion document
// ---------------------------------------------------------------------------

#[test]
fn a_crac_round_trips_through_the_native_format_without_loss() {
    // The gate this phase was set: whatever is read from OpenRAO's format must
    // survive being written and read back as gridoxide's own.
    for name in all_fixtures() {
        let (crac, _) = read(&name);
        let text = crac.to_json().expect("serialize");
        let back = Crac::from_json(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(crac, back, "{name} did not survive the round trip");
    }
}

#[test]
fn the_native_document_declares_its_type_and_is_checked_on_the_way_in() {
    let crac = Crac { id: "c".into(), ..Default::default() };
    let text = crac.to_json().expect("serialize");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("json");
    assert_eq!(doc["type"], DOCUMENT_TYPE);
    assert_eq!(doc["version"], DOCUMENT_VERSION);

    // Pointing this at the *other* companion document must fail rather than
    // deserialize into an empty CRAC and report success — a security analysis
    // that finds nothing wrong because it was asked nothing.
    let opf = r#"{"version":"1.0","type":"opf_input","id":"","instants":[],
                  "contingencies":[],"flow_cnecs":[],"network_actions":[],"range_actions":[]}"#;
    assert!(Crac::from_json(opf).is_err(), "an opf_input document was accepted as a CRAC");
}
