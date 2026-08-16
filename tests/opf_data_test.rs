//! The OPF data layer, against the committed pglib-opf fixtures.
//!
//! `src/opf/model.rs`'s own tests cover parsing and unit conversion on
//! hand-written input. What is checked here is that the *real* documents —
//! produced from pglib's `.m` files by `python/gridoxide/matpower.py` — carry
//! what an OPF actually needs, and that the network and OPF documents agree
//! with each other.
//!
//! The last point matters because the two are built differently on purpose:
//! branch limits key into the network document by component id, while
//! generators deliberately do not, since the converter aggregates generators
//! per bus and an OPF needs them individually. A test that did not check the
//! keyed half would not notice the ids drifting apart.

use std::collections::HashSet;
use std::path::PathBuf;

use gridoxide::opf::model::{CostCurve, OpfData};

/// Every case vendored in `tests/data/pglib-opf/`, with the counts pglib's own
/// documentation states for it. Hard-coding these is the point: they come from
/// the upstream description, not from what our converter happened to produce,
/// so a conversion that silently dropped a component fails here.
const CASES: &[(&str, usize, usize)] = &[
    // (case, buses, branches)
    ("pglib_opf_case3_lmbd", 3, 3),
    ("pglib_opf_case5_pjm", 5, 6),
    ("pglib_opf_case14_ieee", 14, 20),
    ("pglib_opf_case30_ieee", 30, 41),
    ("pglib_opf_case118_ieee", 118, 186),
];

fn fixture(name: &str, suffix: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf")
        .join(format!("{name}{suffix}"))
}

fn load(name: &str) -> OpfData {
    let text = std::fs::read_to_string(fixture(name, ".opf.json"))
        .unwrap_or_else(|e| panic!("reading {name}.opf.json: {e}"));
    OpfData::from_json(&text).unwrap_or_else(|e| panic!("parsing {name}.opf.json: {e}"))
}

fn network(name: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(fixture(name, ".json"))
        .unwrap_or_else(|e| panic!("reading {name}.json: {e}"));
    serde_json::from_str(&text).unwrap()
}

#[test]
fn every_case_carries_a_complete_opf_document() {
    for &(name, _buses, branches) in CASES {
        let data = load(name);

        assert_eq!(data.base_mva, 100.0, "{name}: pglib cases are all on a 100 MVA base");
        assert!(!data.generator.is_empty(), "{name}: no generators");
        assert_eq!(
            data.branch_limit.len(),
            branches,
            "{name}: branch limits should cover every branch"
        );

        for g in &data.generator {
            assert!(g.cost.is_some(), "{name}: generator {} has no cost curve", g.index);
            assert!(
                g.p_max >= g.p_min,
                "{name}: generator {} has crossed active limits",
                g.index
            );
        }
    }
}

/// **The reason these fixtures exist.** Every pglib branch is rated, so every
/// congestion constraint can actually bind — unlike the power-flow cases in
/// `benchmark-grids/`, where `case14`, `case118` and `case300` have `rateA = 0`
/// on every single branch and no flow limit is ever active.
///
/// A fixture set that cannot make a constraint bind cannot test an optimizer,
/// so this is a property of the data worth asserting rather than assuming.
#[test]
fn every_branch_can_bind() {
    for &(name, _, branches) in CASES {
        let data = load(name);
        assert_eq!(
            data.binding_capable_branches(),
            branches,
            "{name}: some branches are unlimited, so congestion would go untested"
        );
    }
}

/// Costs must be convex, or the "optimal is provable" argument the whole
/// DC-first plan rests on does not apply.
#[test]
fn every_cost_curve_is_convex() {
    for &(name, _, _) in CASES {
        let data = load(name);
        assert!(data.costs_are_convex(), "{name}: a non-convex cost curve");
    }
}

/// Branch limits key into the network document by component id, so those ids
/// must actually exist there — and cover the lines and transformers exactly.
#[test]
fn branch_limits_key_into_the_network_document() {
    for &(name, _, _) in CASES {
        let data = load(name);
        let net = network(name);

        let mut network_ids = HashSet::new();
        for kind in ["line", "transformer"] {
            if let Some(items) = net["data"][kind].as_array() {
                for item in items {
                    network_ids.insert(item["id"].as_u64().unwrap());
                }
            }
        }

        assert_eq!(
            network_ids.len(),
            data.branch_limit.len(),
            "{name}: branch count differs between the two documents"
        );
        for limit in &data.branch_limit {
            assert!(
                network_ids.contains(&limit.id),
                "{name}: branch limit {} names no component in the network document",
                limit.id
            );
        }
    }
}

/// Generators are *not* keyed into the network document, and this pins why —
/// two independent reasons, each demonstrated on a case that exhibits it.
///
/// **The reference bus's generator has no `sym_gen` at all.** The converter
/// turns it into a `source`, because for a power flow the slack's output is
/// the solve's own result rather than an input. So on every case the OPF's
/// generator count exceeds the network's `sym_gen` count by exactly the number
/// of reference buses. Keying generators by `sym_gen` id would silently drop
/// the slack generator — which is usually the cheapest unit and the one an OPF
/// most wants to dispatch.
///
/// **And generators are aggregated per bus.** `case5_pjm` puts five generators
/// on four buses, so one bus's two units become a single `sym_gen` with their
/// powers summed. Two different cost curves cannot be summed into one, so an
/// id-keyed design loses that unit's curve outright.
#[test]
fn the_network_document_cannot_represent_the_generator_set() {
    for &(name, _, _) in CASES {
        let data = load(name);
        let net = network(name);
        let sym_gens = net["data"]["sym_gen"].as_array().map_or(0, |a| a.len());
        let sources = net["data"]["source"].as_array().map_or(0, |a| a.len());

        // `>=` rather than `==`, and the gap is itself informative: the
        // reference bus accounts for `sources`, and anything beyond that is a
        // bus with more than one generator collapsed into one `sym_gen`.
        assert!(
            data.generator.len() >= sym_gens + sources,
            "{name}: {} OPF generators but {sym_gens} sym_gen + {sources} source — the \
             network document cannot be carrying fewer units than it aggregates",
            data.generator.len()
        );
        assert!(
            sym_gens < data.generator.len(),
            "{name}: the reference bus's generator becomes a `source`, so `sym_gen` must \
             always be short of the full generator set"
        );
    }

    // The aggregation half, on the case that shows it.
    let data = load("pglib_opf_case5_pjm");
    let distinct_nodes: HashSet<u64> = data.generator.iter().map(|g| g.node).collect();
    assert!(
        distinct_nodes.len() < data.generator.len(),
        "case5_pjm should have two generators sharing a bus, got {} across {} nodes",
        data.generator.len(),
        distinct_nodes.len()
    );
}

/// Per-unit conversion, checked against a real document rather than a literal.
#[test]
fn limits_convert_to_per_unit_on_the_documents_own_base() {
    let data = load("pglib_opf_case5_pjm");
    let base = data.base_mva;

    for g in &data.generator {
        let (p_min, p_max) = g.p_limits_pu(base);
        assert!((p_min - g.p_min / base).abs() < 1e-12);
        assert!((p_max - g.p_max / base).abs() < 1e-12);
        assert!(p_max <= 100.0, "a per-unit limit of {p_max} suggests an unconverted MW value");
    }

    for limit in &data.branch_limit {
        let rate = limit.rate_pu(base).expect("every pglib branch is rated");
        assert!((rate - limit.rate_a / base).abs() < 1e-12);
    }
}

/// Marginal cost must be positive across each generator's own operating range
/// — a unit that pays you to produce would make the dispatch unbounded below
/// in the direction of that generator, and it is a cheap way to catch a
/// coefficient ordering that got reversed on the way in.
#[test]
fn marginal_cost_is_positive_across_the_operating_range() {
    for &(name, _, _) in CASES {
        let data = load(name);
        for g in &data.generator {
            let Some(cost) = &g.cost else { continue };
            for fraction in [0.0, 0.5, 1.0] {
                let p = g.p_min + fraction * (g.p_max - g.p_min);
                let marginal = cost.marginal(p);
                assert!(
                    marginal >= 0.0,
                    "{name}: generator {} has marginal cost {marginal} at {p} MW",
                    g.index
                );
            }
        }
    }
}

/// A spot check against the source file, so the whole chain is anchored to
/// something a human can read.
///
/// `pglib_opf_case3_lmbd`'s first generator carries
/// `2 0 0 3 0.11 5.0 0.0` — model 2 (polynomial), 3 coefficients, highest
/// degree first. Ascending, that is `[0.0, 5.0, 0.11]`.
#[test]
fn a_known_cost_row_survives_the_conversion_in_ascending_order() {
    let data = load("pglib_opf_case3_lmbd");
    let first = data.generator.iter().find(|g| g.index == 0).expect("generator 0");
    match first.cost.as_ref().expect("a cost curve") {
        CostCurve::Polynomial { coefficients } => {
            assert_eq!(coefficients.len(), 3, "{coefficients:?}");
            assert!(coefficients[0].abs() < 1e-12, "constant term: {coefficients:?}");
            assert!(
                coefficients[2] < coefficients[1],
                "the quadratic term should be the small one — ordering looks reversed: \
                 {coefficients:?}"
            );
        }
        other => panic!("expected a polynomial, got {other:?}"),
    }
}
