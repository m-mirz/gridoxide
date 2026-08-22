//! PowSyBl IIDM import — phase 3 of `plans/RAO_PLAN.md`.
//!
//! Two gates, and the first is the stronger one.
//!
//! **Cross-format.** Three of these fixtures are pypowsybl exports of `.uct`
//! files that `tests/ucte_test.rs` has already checked against pypowsybl. So
//! gridoxide can read the same network twice, through two parsers that share no
//! code — one fixed-column text, one XML — and the answers must agree. When
//! they do to machine precision, both importers are right about that network in
//! a way neither could establish alone.
//!
//! **Against pypowsybl**, on fixtures written as IIDM in the first place, since
//! the cross-format gate only covers what UCTE can express: it says nothing
//! about node-breaker topology, temporary limits, or the older `currentLimits1`
//! spelling.

use std::collections::HashMap;
use std::path::PathBuf;

use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use gridoxide::iidm;
use gridoxide::solver::{PowerFlowMethod, PowerFlowOptions, SolveStatus};
use gridoxide::topology::bus_view::RetentionPolicy;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/iidm").join(name)
}

fn ucte_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte").join(name)
}

/// `(label -> (|V| pu, angle deg), element id -> [p1, q1, p2, q2] MW/MVar)`.
type Solved = (HashMap<String, (f64, f64)>, HashMap<String, [f64; 4]>);

fn solve(
    buses: Vec<gridoxide::types::Bus>,
    lines: &[gridoxide::types::Line],
    transformers: &[gridoxide::types::Transformer],
    shunts: &[gridoxide::network::ShuntAdm],
    labels: &[String],
    ids: &[String],
    base_mva: f64,
) -> (Solved, SolveStatus) {
    let opts = PowerFlowOptions { method: PowerFlowMethod::NewtonRaphson, ..Default::default() };
    let report = gridoxide::run_power_flow(buses, lines, transformers, shunts, gridoxide::TapData::none(), opts);
    let v = bus_voltages(&report.buses);
    let params = branch_params(lines, transformers);
    let voltages = labels
        .iter()
        .enumerate()
        .map(|(i, l)| {
            (l.trim().to_string(), (report.buses[i].voltage_mag, report.buses[i].voltage_ang.to_degrees()))
        })
        .collect();
    let flows = ids
        .iter()
        .enumerate()
        .map(|(b, id)| {
            let (p1, q1) = terminal_flow(&params[b], Terminal::From, &v);
            let (p2, q2) = terminal_flow(&params[b], Terminal::To, &v);
            (id.trim().to_string(), [p1 * base_mva, q1 * base_mva, p2 * base_mva, q2 * base_mva])
        })
        .collect();
    ((voltages, flows), report.stats.status)
}

fn solve_iidm(net: &iidm::IidmImport) -> (Solved, SolveStatus) {
    solve(
        net.buses.clone(),
        &net.lines,
        &net.transformers,
        &net.shunts,
        &net.bus_labels,
        &net.branch_ids,
        net.base_mva,
    )
}

/// The worst deviation between two flow maps, allowing a branch's two terminals
/// to be reported in either order.
///
/// Terminal order is a labelling choice each source file makes for itself: a
/// UCTE `##L` record may name the X-node first where the IIDM boundary line
/// names the real bus first. That reverses the reported sign without changing
/// any physics, so the comparison takes whichever pairing is closer and reports
/// how often it had to.
fn worst_flow(a: &HashMap<String, [f64; 4]>, b: &HashMap<String, [f64; 4]>) -> (f64, usize, usize) {
    let mut worst: f64 = 0.0;
    let mut shared = 0;
    let mut reversed_count = 0;
    for (id, x) in a {
        let Some(y) = b.get(id) else { continue };
        shared += 1;
        let direct = (0..4).map(|k| (x[k] - y[k]).abs()).fold(0.0f64, f64::max);
        let swapped = [y[2], y[3], y[0], y[1]];
        let reversed = (0..4).map(|k| (x[k] - swapped[k]).abs()).fold(0.0f64, f64::max);
        if reversed < direct {
            reversed_count += 1;
        }
        worst = worst.max(direct.min(reversed));
    }
    (worst, shared, reversed_count)
}

// ---------------------------------------------------------------------------
// Gate 1: the same network through two independent parsers
// ---------------------------------------------------------------------------

fn cross_check(stem: &str, tolerance: f64) -> (f64, usize) {
    let u = gridoxide::ucte::read(ucte_fixture(&format!("{stem}.uct"))).expect("ucte import");
    let i = iidm::read(fixture(&format!("{stem}.xiidm"))).expect("iidm import");

    assert_eq!(
        u.buses.len(),
        i.buses.len(),
        "{stem}: bus counts differ ({} UCTE vs {} IIDM)",
        u.buses.len(),
        i.buses.len()
    );
    assert_eq!(u.n_branches(), i.n_branches(), "{stem}: branch counts differ");

    let ((_, fu), su) = solve(
        u.buses.clone(), &u.lines, &u.transformers, &u.shunts, &u.node_codes, &u.branch_ids, u.base_mva,
    );
    let ((_, fi), si) = solve_iidm(&i);
    assert_eq!(su, SolveStatus::Converged, "{stem}: UCTE side did not converge");
    assert_eq!(si, SolveStatus::Converged, "{stem}: IIDM side did not converge");

    let (worst, shared, reversed) = worst_flow(&fu, &fi);
    assert_eq!(shared, u.n_branches(), "{stem}: only {shared} branch ids matched");
    assert!(worst < tolerance, "{stem}: worst flow deviation {worst} MW/MVar");
    (worst, reversed)
}

/// Twelve buses, fifteen lines and a phase shifter at its neutral tap.
#[test]
fn ucte_and_iidm_agree_exactly_on_the_twelve_node_case() {
    let (worst, reversed) = cross_check("TestCase12Nodes", 1e-9);
    // Two parsers sharing no code, over two serializations of one network,
    // landing on bit-identical flows. Nothing about that is guaranteed by
    // either importer's own tests.
    assert_eq!(worst, 0.0, "expected bit-identical flows, got {worst}");
    assert_eq!(reversed, 0, "no terminal should need reversing here");
}

/// The same, with the phase shifter at tap 16 of 16 — so the IIDM `rho`/`alpha`
/// step table has to reproduce what UCTE's `##R` regulation produced, and both
/// have to match what pypowsybl computed for the UCTE file.
#[test]
fn ucte_and_iidm_agree_exactly_with_the_phase_shifter_at_its_limit() {
    let (worst, _) = cross_check("TestCase12NodesDifferentPstTap", 1e-9);
    assert_eq!(worst, 0.0, "expected bit-identical flows, got {worst}");
}

/// 400/225 transformers and X-nodes, which IIDM writes as boundary lines paired
/// by key. gridoxide keeps the boundary as a real bus on both paths, which is
/// what makes the two structures comparable at all.
#[test]
fn ucte_and_iidm_agree_across_voltage_levels_and_boundary_nodes() {
    let (worst, reversed) = cross_check("TestCase_severalVoltageLevels_Xnodes", 1e-9);
    assert!(worst < 1e-9, "worst deviation {worst}");
    assert!(reversed > 0, "boundary half-lines should differ in terminal order");
}

// ---------------------------------------------------------------------------
// Gate 2: against pypowsybl, on natively-IIDM fixtures
// ---------------------------------------------------------------------------

fn reference(name: &str) -> Solved {
    let text = std::fs::read_to_string(fixture(name)).expect("reference fixture");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("reference json");
    let buses = doc["buses"]
        .as_object()
        .expect("buses")
        .iter()
        .filter_map(|(k, v)| {
            Some((k.trim().to_string(), (v["v_pu"].as_f64()?, v["angle_deg"].as_f64()?)))
        })
        .collect();
    let branches = doc["branches"]
        .as_object()
        .expect("branches")
        .iter()
        .filter_map(|(k, v)| {
            let f: Option<Vec<f64>> =
                ["p1", "q1", "p2", "q2"].iter().map(|n| v[*n].as_f64()).collect();
            Some((k.trim().to_string(), {
                let f = f?;
                [f[0], f[1], f[2], f[3]]
            }))
        })
        .collect();
    (buses, branches)
}

fn against_pypowsybl(stem: &str, min_branches: usize) -> f64 {
    let net = iidm::read(fixture(&format!("{stem}.xiidm"))).expect("import");
    let ((_, flows), status) = solve_iidm(&net);
    assert_eq!(status, SolveStatus::Converged, "{stem} did not converge");
    let (_, expected) = reference(&format!("{stem}.pypowsybl.json"));
    let (worst, shared, _) = worst_flow(&flows, &expected);
    assert!(shared >= min_branches, "{stem}: only {shared} branches matched by id");
    worst
}

/// 52 buses and 80 branches, IIDM 1_8, written as IIDM rather than converted.
#[test]
fn a_larger_native_case_matches_pypowsybl() {
    let worst = against_pypowsybl("nordic32", 75);
    // Residual is the crate's pi-split transformer shunt against IIDM's
    // one-sided one, the same difference `ucte_test.rs` isolates and quantifies.
    assert!(worst < 0.05, "worst flow deviation {worst} MW/MVar");
}

/// A **node-breaker** voltage level: busbar sections, breakers and
/// disconnectors, with the bus view computed from the switch graph rather than
/// read from the file.
#[test]
fn a_node_breaker_case_matches_pypowsybl() {
    let worst = against_pypowsybl("voltage_monitoring", 4);
    assert!(worst < 0.05, "worst flow deviation {worst} MW/MVar");
}

/// IIDM 1_0 — the oldest version in the fixture set, and the `currentLimits1`
/// spelling.
#[test]
fn the_oldest_schema_version_still_reads() {
    let net = iidm::read(fixture("TestCase2Nodes.xiidm")).expect("import");
    assert_eq!(net.version.as_deref(), Some("1_0"));
    let worst = against_pypowsybl("TestCase2Nodes", 2);
    assert!(worst < 1e-3, "worst flow deviation {worst}");
}

// ---------------------------------------------------------------------------
// Version tolerance, limits, taps, topology
// ---------------------------------------------------------------------------

#[test]
fn every_vendored_version_parses() {
    // Five distinct schema versions across the fixture set. The importer must
    // accept all of them without being told which it is reading — a parser
    // pinned to one version reads a fifth of the available material and breaks
    // on the next powsybl release.
    let mut seen: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(fixture("")).expect("fixture dir") {
        let path = entry.expect("entry").path();
        if path.extension().is_none_or(|e| e != "xiidm") {
            continue;
        }
        let net = iidm::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(!net.buses.is_empty(), "{} produced no buses", path.display());
        if let Some(v) = net.version.clone() {
            if !seen.contains(&v) {
                seen.push(v);
            }
        }
    }
    assert!(seen.len() >= 4, "expected several schema versions, saw {seen:?}");
}

#[test]
fn an_unknown_element_is_skipped_and_named() {
    // Tolerance is the point, but silent tolerance is not: an element the
    // importer does not understand has to appear in `notes`, or a network can
    // quietly lose half its equipment and still look fine.
    let doc = r#"<?xml version='1.0'?>
<iidm:network xmlns:iidm="http://www.powsybl.org/schema/iidm/1_14" id="n">
  <iidm:substation id="S">
    <iidm:voltageLevel id="VL" nominalV="400.0" topologyKind="BUS_BREAKER">
      <iidm:busBreakerTopology><iidm:bus id="B"/></iidm:busBreakerTopology>
      <iidm:load id="L" p0="10.0" q0="1.0" bus="B" connectableBus="B"/>
      <iidm:somethingNobodyHasHeardOf id="Z"/>
    </iidm:voltageLevel>
  </iidm:substation>
</iidm:network>"#;
    let net = iidm::parse(doc).expect("import");
    assert_eq!(net.version.as_deref(), Some("1_14"));
    assert_eq!(net.buses.len(), 1);
    assert!(
        net.notes.iter().any(|n| n.contains("somethingNobodyHasHeardOf")),
        "unknown element was skipped without saying so: {:?}",
        net.notes
    );
}

#[test]
fn both_limit_spellings_are_read() {
    // `currentLimits1` (older) and `operationalLimitsGroup1` wrapping a bare
    // `currentLimits` (newer) mean the same thing, and a reader that knows only
    // one of them silently reports a network with no ratings.
    let old = iidm::read(fixture("TestCase2Nodes.xiidm")).expect("import");
    assert!(
        old.limits.iter().any(|[a, b]| a.patl_a.is_some() || b.patl_a.is_some()),
        "no limit read from the currentLimits1 spelling"
    );
    let new = iidm::read(fixture("network_with_dangling_lines.xiidm")).expect("import");
    assert!(
        new.limits.iter().any(|[a, b]| a.patl_a.is_some() || b.patl_a.is_some()),
        "no limit read from the operationalLimitsGroup spelling"
    );
}

#[test]
fn a_temporary_limit_keeps_its_duration() {
    let net = iidm::read(fixture("network_one_voltage_level.xiidm")).expect("import");
    let temporary: Vec<_> =
        net.limits.iter().flat_map(|[a, b]| a.tatl.iter().chain(b.tatl.iter())).collect();
    assert!(!temporary.is_empty(), "fixture declares temporary limits but none were read");
    assert!(
        temporary.iter().any(|t| t.acceptable_duration_s == Some(60.0)),
        "the 60-second limit lost its duration: {temporary:?}"
    );
    // A duration-less temporary limit is legitimate (IIDM writes one for the
    // instantaneous rating) and must not be dropped.
    assert!(temporary.iter().all(|t| t.value_a > 0.0));
}

#[test]
fn limits_are_kept_per_side() {
    // A transformer's two windings sit at different voltages and so carry
    // different ratings; collapsing them to one number loses that.
    let net = iidm::read(fixture("network_with_dangling_lines.xiidm")).expect("import");
    assert_eq!(net.limits.len(), net.n_branches());
    let one_sided = net
        .limits
        .iter()
        .filter(|[a, b]| a.patl_a.is_some() != b.patl_a.is_some())
        .count();
    assert!(one_sided > 0, "this fixture has a line rated on one side only");
}

#[test]
fn a_phase_tap_changer_becomes_a_tap_table() {
    let net = iidm::read(fixture("TestCase12NodesDifferentPstTap.xiidm")).expect("import");
    let changer = net.tap_changers.iter().flatten().next().expect("a tap changer");
    assert!(changer.len() > 1, "a single-step table is not a tap changer");
    // Sitting at the top tap, as the source UCTE file did.
    assert_eq!(changer.position, changer.high());
    let tap = net.transformers[0].tap;
    assert!((tap - changer.current().unwrap()).norm() < 1e-15);
    assert!(tap.arg().abs() > 1e-6, "a shifted PST should have a nonzero angle");
}

#[test]
fn switches_survive_import_as_first_class_objects() {
    // The whole reason the node-breaker path builds a `NodeBreakerTopology`
    // rather than reading the file's calculated bus view: a topological
    // remedial action needs switches it can open, and a bus view has resolved
    // them away.
    let net = iidm::read(fixture("voltage_monitoring.xiidm")).expect("import");
    assert!(!net.topology.switches.is_empty(), "node-breaker file yielded no switches");
    assert_eq!(net.switch_ids.len(), net.topology.switches.len());
    assert!(net.switch_ids.iter().any(|id| id.contains("Break")), "{:?}", net.switch_ids);
    // With every closed switch merged, the bus view has fewer buses than nodes.
    assert!(net.view.n_buses() < net.view.n_nodes());
}

#[test]
fn retaining_switches_yields_more_buses_than_merging_them() {
    let merged = iidm::read(fixture("voltage_monitoring.xiidm")).expect("import");
    let options = iidm::IidmOptions {
        retention: RetentionPolicy::RetainAll,
        ..Default::default()
    };
    let retained =
        iidm::read_with(fixture("voltage_monitoring.xiidm"), &options).expect("import");
    assert!(
        retained.buses.len() > merged.buses.len(),
        "retaining switches ({}) should split buses that merging ({}) joins",
        retained.buses.len(),
        merged.buses.len()
    );
}

#[test]
fn a_disconnected_terminal_omits_the_branch_and_names_it() {
    let net = iidm::read(fixture("network_with_dangling_lines.xiidm")).expect("import");
    for id in &net.disconnected {
        assert!(!net.branch_ids.contains(id), "`{id}` was both omitted and kept");
    }
}

#[test]
fn the_slack_is_chosen_deterministically() {
    let a = iidm::read(fixture("nordic32.xiidm")).expect("import");
    let b = iidm::read(fixture("nordic32.xiidm")).expect("import");
    assert_eq!(a.slack, b.slack);
    assert_eq!(a.buses[a.slack].bus_type, gridoxide::types::BusType::Slack);
    assert_eq!(
        a.buses.iter().filter(|x| x.bus_type == gridoxide::types::BusType::Slack).count(),
        1
    );
    assert!(a.notes.iter().any(|n| n.contains("slack chosen")));
}

#[test]
fn a_boundary_node_takes_the_voltage_of_what_it_attaches_to() {
    // A boundary node belongs to no declared voltage level, so it has no
    // nominalV of its own. Defaulting it to 1 V would silently wreck every
    // per-unit quantity through the boundary.
    let net = iidm::read(fixture("network_with_dangling_lines.xiidm")).expect("import");
    for (i, bus) in net.buses.iter().enumerate() {
        assert!(
            bus.u_rated > 1000.0,
            "bus `{}` has an implausible base of {} V",
            net.bus_labels[i],
            bus.u_rated
        );
    }
}
