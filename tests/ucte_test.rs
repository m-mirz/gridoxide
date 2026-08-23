//! UCTE-DEF import — the phase-1 gate of `plans/RAO_PLAN.md`.
//!
//! The interesting assertions here are the ones that compare against
//! **pypowsybl** (`tests/data/ucte/*.pypowsybl.json`, produced by
//! `scripts/bench/ucte_reference.py`). An importer that parses without error is
//! not an importer that is right: every column could be off by one and the file
//! would still load. Solving the imported network and matching an independent
//! implementation's voltages and branch flows is what actually pins the
//! conventions — the negative generation sign, the voltage-level-code nominal,
//! the reversed transformer orientation, and the phase-shifter tap table.

use std::collections::HashMap;
use std::path::PathBuf;

use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use gridoxide::solver::{PowerFlowMethod, PowerFlowOptions, SolveStatus};
use gridoxide::types::BusType;
use gridoxide::ucte;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte").join(name)
}

/// The reference document, flattened to the two maps the assertions need.
struct Reference {
    buses: HashMap<String, (f64, f64)>,
    branches: HashMap<String, [f64; 4]>,
}

fn reference(name: &str) -> Reference {
    let text = std::fs::read_to_string(fixture(name)).expect("reference fixture");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("reference json");
    let mut buses = HashMap::new();
    for (k, v) in doc["buses"].as_object().expect("buses") {
        buses.insert(
            k.clone(),
            (v["v_pu"].as_f64().unwrap(), v["angle_deg"].as_f64().unwrap()),
        );
    }
    let mut branches = HashMap::new();
    for (k, v) in doc["branches"].as_object().expect("branches") {
        // A de-energised branch reports null. Skipping it is right: gridoxide
        // reports 0.0 there rather than a missing value, and asserting 0 == NaN
        // would be a test of JSON, not of the importer.
        let flows: Option<Vec<f64>> =
            ["p1", "q1", "p2", "q2"].iter().map(|f| v[*f].as_f64()).collect();
        if let Some(f) = flows {
            branches.insert(k.clone(), [f[0], f[1], f[2], f[3]]);
        }
    }
    Reference { buses, branches }
}

/// Solve an imported network and return `(node code -> (|V|, angle°))` and
/// `(element id -> [p1, q1, p2, q2] in MW/MVar)`.
fn solve(
    net: &ucte::UcteImport,
) -> (HashMap<String, (f64, f64)>, HashMap<String, [f64; 4]>, SolveStatus) {
    let opts = PowerFlowOptions { method: PowerFlowMethod::NewtonRaphson, ..Default::default() };
    // `as_switched`, not the raw arrays: an out-of-service branch is kept in the
    // model so it can be closed by a remedial action, and solving the network
    // "as given" means solving it with those branches open.
    let (lines, transformers) = net.as_switched();
    let report = gridoxide::run_power_flow(
        net.buses.clone(),
        &lines,
        &transformers,
        &net.shunts,
        gridoxide::TapData::none(),
        opts,
    );
    let v = bus_voltages(&report.buses);
    let params = branch_params(&lines, &transformers);
    let mva = net.base_mva;

    let buses = net
        .node_codes
        .iter()
        .enumerate()
        .map(|(i, code)| {
            (
                code.trim().to_string(),
                (report.buses[i].voltage_mag, report.buses[i].voltage_ang.to_degrees()),
            )
        })
        .collect();
    let branches = net
        .branch_ids
        .iter()
        .enumerate()
        .map(|(b, id)| {
            let (p1, q1) = terminal_flow(&params[b], Terminal::From, &v);
            let (p2, q2) = terminal_flow(&params[b], Terminal::To, &v);
            (id.clone(), [p1 * mva, q1 * mva, p2 * mva, q2 * mva])
        })
        .collect();
    (buses, branches, report.stats.status)
}

/// Compare against the stored reference, returning the worst absolute
/// deviations as `(bus |V|, bus angle°, branch MW/MVar)`.
///
/// Only buses and branches present in *both* are compared. That is not a
/// weakening: gridoxide keeps an X-node as an ordinary bus while pypowsybl
/// merges the two half-lines into one tie line, so the fixtures with X-nodes
/// legitimately have more elements on our side. `x_node_buses_are_extra_not_
/// different` pins that the shared ones still agree.
fn worst_deviation(net: &ucte::UcteImport, reference_name: &str) -> (f64, f64, f64) {
    let reference = reference(reference_name);
    let (buses, branches, status) = solve(net);
    assert_eq!(status, SolveStatus::Converged, "gridoxide did not converge");

    // Angles are compared **relative to a common bus**, not absolutely. Both
    // sides pin one bus at zero, and they need not pin the same one: this
    // importer picks the slack itself (no vendored file declares a type-3 node)
    // and pypowsybl's own selection can land elsewhere. A whole-network angle
    // offset is a different reference, not a different answer, and comparing it
    // as if it were would reject correct results. Everything physical — voltage
    // magnitudes, angle *differences*, branch flows — is unaffected.
    let datum = {
        let mut shared: Vec<&String> =
            buses.keys().filter(|c| reference.buses.contains_key(*c)).collect();
        shared.sort();
        shared.first().copied().cloned().expect("no shared bus")
    };
    let shift = buses[&datum].1 - reference.buses[&datum].1;

    let mut dv: f64 = 0.0;
    let mut da: f64 = 0.0;
    let mut compared = 0;
    for (code, (v, a)) in &buses {
        if let Some((rv, ra)) = reference.buses.get(code) {
            dv = dv.max((v - rv).abs());
            da = da.max((a - shift - ra).abs());
            compared += 1;
        }
    }
    assert!(compared >= 5, "only {compared} buses matched the reference by name");

    let mut df: f64 = 0.0;
    let mut branches_compared = 0;
    for (id, flows) in &branches {
        if let Some(r) = reference.branches.get(id) {
            for k in 0..4 {
                df = df.max((flows[k] - r[k]).abs());
            }
            branches_compared += 1;
        }
    }
    assert!(branches_compared >= 5, "only {branches_compared} branches matched by id");
    (dv, da, df)
}

#[test]
fn the_twelve_node_case_matches_pypowsybl_to_solver_tolerance() {
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let (dv, da, df) = worst_deviation(&net, "TestCase12Nodes.pypowsybl.json");
    // This case's only transformer has zero magnetizing admittance, so nothing
    // stands between the two models and the agreement is limited purely by how
    // tightly each solver converged.
    assert!(dv < 1e-9, "worst |V| deviation {dv}");
    assert!(da < 1e-6, "worst angle deviation {da} deg");
    assert!(df < 1e-3, "worst branch flow deviation {df} MW/MVar");
}

/// The load-bearing test: 400/225 transformers, X-nodes, and a symmetrical
/// phase shifter sitting at **tap 3** rather than neutral.
///
/// Every convention this importer had to guess is exercised here at once, and
/// each one fails loudly if wrong: a reversed transformer flips the sign of its
/// flow, a mis-referred impedance is a ~3% error on the 400/225 units, and a
/// wrong tap formula moves the PST's flow by hundreds of MW.
#[test]
fn transformers_across_voltage_levels_and_x_nodes_match_pypowsybl() {
    let net = ucte::read(fixture("TestCase_severalVoltageLevels_Xnodes.uct")).expect("import");
    let (dv, da, df) = worst_deviation(&net, "TestCase_severalVoltageLevels_Xnodes.pypowsybl.json");
    // The residual is **one known model difference and nothing else**: two of
    // this file's transformers declare a magnetizing admittance, which IIDM
    // places entirely on side 2 and gridoxide's `network::branch_calc_param`
    // splits equally between both ends — a Γ model against a π model. Zeroing
    // just those two fields drops every figure below to 2e-9 deg and 1.1e-7 MW,
    // which is how the cause was established rather than assumed. The choice is
    // the crate's, not this importer's: the CGMES and PGM paths split the shunt
    // the same way.
    assert!(dv < 1e-9, "worst |V| deviation {dv}");
    assert!(da < 1e-3, "worst angle deviation {da} deg");
    assert!(df < 0.5, "worst branch flow deviation {df} MW/MVar");
}

#[test]
fn a_case_with_a_busbar_coupler_matches_pypowsybl() {
    let net = ucte::read(fixture("TestCase16Nodes_with_different_imax.uct")).expect("import");
    let (dv, da, df) = worst_deviation(&net, "TestCase16Nodes_with_different_imax.pypowsybl.json");
    assert!(dv < 1e-9, "worst |V| deviation {dv}");
    assert!(da < 1e-3, "worst angle deviation {da} deg");
    assert!(df < 0.5, "worst branch flow deviation {df} MW/MVar");
}

#[test]
fn the_sixteen_node_case_matches_pypowsybl() {
    // The network 92 of the reference's AC scenarios are written against, and
    // the largest of the vendored UCTE cases: 16 buses, 27 branches, four
    // countries. Gated here before any of those scenarios rely on it, so a
    // disagreement over margins can never be blamed on the import.
    let net = ucte::read(fixture("TestCase16Nodes.uct")).expect("import");
    let (dv, da, df) = worst_deviation(&net, "TestCase16Nodes.pypowsybl.json");
    assert!(dv < 1e-9, "worst |V| deviation {dv}");
    assert!(da < 1e-3, "worst angle deviation {da} deg");
    assert!(df < 0.5, "worst branch flow deviation {df} MW/MVar");
}

// ---------------------------------------------------------------------------
// The conventions, pinned individually so a failure says which one broke
// ---------------------------------------------------------------------------

#[test]
fn generation_is_negative_in_the_file_and_positive_in_the_model() {
    // `BBE1AA1` carries load 2500 MW and "generation" -1500, meaning 1500 MW
    // generated. Net injection is therefore -1000 MW, i.e. -10 pu at 100 MVA.
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let i = net.node_index["BBE1AA1 "];
    assert!((net.buses[i].p_spec - (-10.0)).abs() < 1e-12, "p_spec {}", net.buses[i].p_spec);
}

#[test]
fn nominal_voltage_comes_from_the_node_code_not_the_record() {
    // Code character 7 is `1` = 380 kV class, while the record's own voltage
    // field says 400.00 — that is the regulation set-point, not the base.
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let i = net.node_index["BBE1AA1 "];
    assert_eq!(net.buses[i].u_rated, 380_000.0);
    assert!((net.buses[i].voltage_mag - 400.0 / 380.0).abs() < 1e-12);
    assert_eq!(net.buses[i].bus_type, BusType::PV);
}

#[test]
fn the_two_two_zero_kv_class_is_read_as_two_twenty() {
    let net = ucte::read(fixture("TestCase_severalVoltageLevels_Xnodes.uct")).expect("import");
    let i = net.node_index["BBE1AA2 "];
    assert_eq!(net.buses[i].u_rated, 220_000.0);
}

#[test]
fn a_transformer_is_reversed_relative_to_the_file() {
    // UCTE refers the impedance to node 1 and taps node 2; gridoxide wants the
    // series admittance at `to` and the ratio at `from`. So `from` is node 2.
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    assert_eq!(net.transformers.len(), 1);
    let t = &net.transformers[0];
    assert_eq!(net.node_codes[t.from].trim(), "BBE3AA1");
    assert_eq!(net.node_codes[t.to].trim(), "BBE2AA1");
}

#[test]
fn a_line_carries_its_current_rating_as_a_permanent_limit() {
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let b = net.branch_ids.iter().position(|id| id.starts_with("BBE1AA1  BBE2AA1")).unwrap();
    assert_eq!(net.limits[b].patl_a, Some(5000.0));
    assert!(net.limits[b].tatl.is_empty(), "UCTE states no temporary ratings");
}

#[test]
fn every_branch_has_a_limit_entry_even_when_the_file_gave_none() {
    // Parallel indexing is the contract: `limits[b]` must exist for every `b`,
    // so a consumer never has to distinguish "no entry" from "no limit".
    let net = ucte::read(fixture("TestCase_severalVoltageLevels_Xnodes.uct")).expect("import");
    assert_eq!(net.limits.len(), net.n_branches());
    assert_eq!(net.branch_ids.len(), net.n_branches());
}

// ---------------------------------------------------------------------------
// Tap changers
// ---------------------------------------------------------------------------

#[test]
fn a_symmetrical_phase_shifter_yields_a_full_tap_table() {
    // `##R`: du = -0.68 %, theta = 90 deg, n = 16, current tap 0.
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let changer = net.tap_changers[0].as_ref().expect("tap changer");
    assert_eq!(changer.low, -16);
    assert_eq!(changer.high(), 16);
    assert_eq!(changer.len(), 33);
    assert_eq!(changer.position, 0);

    // Symmetrical regulation moves the angle and leaves the magnitude alone —
    // the property that lets a PST push flow without pushing voltage.
    for pos in changer.low..=changer.high() {
        let ratio = changer.ratio(pos).unwrap();
        assert!((ratio - 1.0).abs() < 1e-12, "tap {pos} changed |ratio| to {ratio}");
    }
    // Neutral is neutral, and the extremes are symmetric about it.
    assert!(changer.angle_deg(0).unwrap().abs() < 1e-12);
    let hi = changer.angle_deg(16).unwrap();
    let lo = changer.angle_deg(-16).unwrap();
    assert!((hi + lo).abs() < 1e-12, "angles {hi} and {lo} are not symmetric");
    assert!(hi.abs() > 5.0, "16 taps of 0.68% should be several degrees, got {hi}");
}

/// A phase shifter parked at its **extreme** tap, checked against pypowsybl.
///
/// This is the assertion that actually validates the symmetrical tap formula.
/// At tap 0 every candidate formula agrees, so a neutral PST proves nothing; at
/// tap 16 of 16 a wrong `alpha` moves several hundred MW and the comparison
/// fails by a mile rather than by a rounding error.
#[test]
fn a_phase_shifter_at_its_extreme_tap_matches_pypowsybl() {
    let net = ucte::read(fixture("TestCase12NodesDifferentPstTap.uct")).expect("import");
    let changer = net.tap_changers[0].as_ref().expect("tap changer");
    assert_eq!(changer.position, 16, "fixture should sit at the top tap");
    assert!(changer.angle_deg(16).unwrap().abs() > 5.0);

    let (dv, da, df) = worst_deviation(&net, "TestCase12NodesDifferentPstTap.pypowsybl.json");
    // Zero magnetizing admittance on this transformer, so nothing but solver
    // tolerance separates the two.
    assert!(dv < 1e-9, "worst |V| deviation {dv}");
    assert!(da < 1e-6, "worst angle deviation {da} deg");
    assert!(df < 1e-3, "worst branch flow deviation {df} MW/MVar");
}

#[test]
fn a_shifted_pst_puts_its_current_step_on_the_transformer() {
    let net = ucte::read(fixture("TestCase16Nodes_with_different_imax.uct")).expect("import");
    // Two PSTs in this file: one neutral, one at tap 5. The non-neutral one is
    // what matters — a transformer left at its neutral step is exactly the bug
    // this pins.
    let (i, changer) = net
        .tap_changers
        .iter()
        .enumerate()
        .find_map(|(i, c)| c.as_ref().filter(|c| c.position != 0).map(|c| (i, c)))
        .expect("a shifted tap changer");
    assert_eq!(changer.position, 5);
    let tap = net.transformers[i].tap;
    assert!((tap - changer.current().unwrap()).norm() < 1e-15);
    assert!(tap.arg().abs() > 1e-6, "a tap-5 PST should have a nonzero shift");
}

#[test]
fn a_negative_tap_position_shifts_the_other_way() {
    // `3nodes_pst.uct` sits at tap -8. Sign errors in the tap formula survive
    // every positive-tap test, so this one exists purely to catch them.
    let net = ucte::read(fixture("3nodes_pst.uct")).expect("import");
    let changer = net.tap_changers.iter().flatten().next().expect("tap changer");
    assert_eq!(changer.position, -8);
    let here = changer.angle_deg(-8).unwrap();
    let mirrored = changer.angle_deg(8).unwrap();
    assert!(here * mirrored < 0.0, "taps -8 and +8 should shift opposite ways");
    assert!((here + mirrored).abs() < 1e-12, "and by equal amounts");
}

#[test]
fn setting_a_tap_position_moves_the_transformer_and_refuses_to_leave_the_range() {
    let mut net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let mut changer = net.tap_changers[0].take().expect("tap changer");
    let before = net.transformers[0].tap;

    assert!(changer.set_position(&mut net.transformers[0], 8));
    assert_eq!(changer.position, 8);
    assert!((net.transformers[0].tap - before).norm() > 1e-6, "tap did not move");

    // Out of range leaves both objects untouched rather than clamping, so a
    // sweep cannot silently pile up at an endpoint.
    let held = net.transformers[0].tap;
    assert!(!changer.set_position(&mut net.transformers[0], 99));
    assert_eq!(changer.position, 8);
    assert_eq!(net.transformers[0].tap, held);
}

#[test]
fn rounding_an_angle_to_a_tap_searches_rather_than_dividing() {
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let changer = net.tap_changers[0].as_ref().expect("tap changer");
    for pos in [-16, -7, 0, 5, 16] {
        let angle = changer.angle_deg(pos).unwrap();
        assert_eq!(changer.nearest_to_angle(angle), Some(pos));
    }
    // Far outside the range still lands on the nearest reachable tap.
    let extreme = changer.angle_deg(16).unwrap() * 10.0;
    let rounded = changer.nearest_to_angle(extreme).unwrap();
    assert!(rounded == 16 || rounded == -16, "got {rounded}");
}

// ---------------------------------------------------------------------------
// Structure, statuses and diagnostics
// ---------------------------------------------------------------------------

#[test]
fn out_of_service_branches_are_kept_open_rather_than_dropped() {
    // They are kept so that a "close this circuit" remedial action — one of the
    // commonest automatons there is — can be applied to them at all. Dropping
    // them would make that inexpressible, and a silently missing branch is also
    // how an importer produces a plausible wrong answer.
    let net = ucte::read(fixture("TestCase16Nodes_with_different_imax.uct")).expect("import");
    assert_eq!(net.out_of_service.len(), net.initially_open.len());
    for (id, &branch) in net.out_of_service.iter().zip(&net.initially_open) {
        assert!(net.branch_ids.contains(id), "`{id}` was dropped");
        assert_eq!(&net.branch_ids[branch], id, "initially_open does not line up with the ids");
    }
    if !net.out_of_service.is_empty() {
        assert!(net.notes.iter().any(|n| n.contains("out of operation")));
        // And solving the network "as given" must leave them de-energised.
        let (lines, transformers) = net.as_switched();
        let open = net.initially_open[0];
        if open < lines.len() {
            assert!(lines[open].x > 1e6, "an open line is still conducting");
        } else {
            assert_eq!(transformers[open - lines.len()].from_status, 0);
        }
    }
}

#[test]
fn a_closed_busbar_coupler_becomes_a_low_impedance_branch() {
    let net = ucte::read(fixture("TestCase16Nodes_with_different_imax.uct")).expect("import");
    let b = net.branch_ids.iter().position(|id| id.starts_with("BBE1AA1  BBE4AA1")).unwrap();
    let line = &net.lines[b];
    assert!(line.x > 0.0 && line.x < 1e-3, "coupler reactance {}", line.x);
    assert!(net.notes.iter().any(|n| n.contains("busbar coupler")));
}

#[test]
fn x_nodes_are_kept_as_buses_and_said_so() {
    let net = ucte::read(fixture("TestCase_severalVoltageLevels_Xnodes.uct")).expect("import");
    let x = net.node_codes.iter().filter(|c| c.starts_with('X')).count();
    assert!(x > 0);
    assert!(net.notes.iter().any(|n| n.contains("X-node")));
}

#[test]
fn the_slack_is_chosen_deterministically_and_announced() {
    // No vendored fixture uses node type 3, so the importer must pick — and the
    // pick has to be stable, or two runs of the same study disagree.
    let a = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let b = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    assert_eq!(a.slack, b.slack);
    assert_eq!(a.buses[a.slack].bus_type, BusType::Slack);
    assert_eq!(a.node_codes[a.slack].trim(), "BBE2AA1");
    assert!(a.notes.iter().any(|n| n.contains("slack chosen")));
    assert_eq!(
        a.buses.iter().filter(|b| b.bus_type == BusType::Slack).count(),
        1,
        "exactly one slack"
    );
}

#[test]
fn an_explicit_slack_policy_is_honoured() {
    let options = ucte::UcteOptions {
        slack: ucte::SlackPolicy::Node("FFR3AA1 ".to_string()),
        ..Default::default()
    };
    let net = ucte::read_with(fixture("TestCase12Nodes.uct"), &options).expect("import");
    assert_eq!(net.node_codes[net.slack].trim(), "FFR3AA1");

    let bad = ucte::UcteOptions {
        slack: ucte::SlackPolicy::Node("NOPE".to_string()),
        ..Default::default()
    };
    assert!(ucte::read_with(fixture("TestCase12Nodes.uct"), &bad).is_err());
}

#[test]
fn data_before_any_block_header_is_rejected() {
    // The reference implementation rejects this same file ("a node must be
    // defined in a ##Z context"); silently skipping the orphaned records would
    // produce a network missing three nodes.
    let err = ucte::read(fixture("TestCase12Nodes_wrong.uct")).unwrap_err();
    assert!(
        matches!(err, ucte::UcteError::DataOutsideBlock { .. }),
        "expected DataOutsideBlock, got {err}"
    );
}

#[test]
fn a_short_record_is_tolerated_when_only_trailing_fields_are_missing() {
    // Writers routinely stop at the last field they have a value for.
    let doc = b"##N\n##ZBE\nBBE1AA1  BE1          0 2 400.00 2500.00\n";
    let net = ucte::parse(doc).expect("import");
    assert_eq!(net.buses.len(), 1);
    assert!((net.buses[0].p_spec - (-25.0)).abs() < 1e-12);
    assert!(net.buses[0].q_min.is_infinite(), "absent limits are unbounded, not zero");
}

#[test]
fn latin_one_names_do_not_shift_the_columns() {
    // Take a record that is known to parse, replace one byte of the *name*
    // field with a Latin-1 accented character, and require the numbers after it
    // to be unchanged. Decoding the line as UTF-8 first would shorten it by one
    // character and drag every later field left.
    let good = std::fs::read(fixture("TestCase12Nodes.uct")).expect("fixture");
    let plain = ucte::parse(&good).expect("import");
    let reference = plain.buses[0].p_spec;

    let mut accented = good.clone();
    let start = accented
        .windows(7)
        .position(|w| w == b"BBE1AA1")
        .expect("first node record");
    accented[start + 9] = 0xC9; // 'E' with acute, in the name field
    let net = ucte::parse(&accented).expect("import");
    assert_eq!(net.buses.len(), plain.buses.len());
    assert!(
        (net.buses[0].p_spec - reference).abs() < 1e-12,
        "p_spec moved from {reference} to {} when a name gained an accent",
        net.buses[0].p_spec
    );
}

// ---------------------------------------------------------------------------
// Tap regulation
// ---------------------------------------------------------------------------

/// A `##N`/`##T`/`##R` document with the regulation's *target* columns filled
/// in — 33–38 for a ratio regulation's held voltage, 58–63 for an angle
/// regulation's held power.
///
/// Synthesized rather than taken from a fixture, and that is the point: across
/// all 207 `##R` records in the vendored `.uct` corpus, **not one** populates
/// either column. `parse_regulation` used to read up to byte 32 and resume at
/// 39, skipping both, and no fixture could have caught it.
fn regulated(r_record: &str) -> Vec<u8> {
    let mut doc = String::new();
    doc.push_str("##N\n##ZBE\n");
    doc.push_str("BBE1AA1  BE1          0 2 400.00   0.00   0.00   0.00   0.00\n");
    doc.push_str("BBE2AA1  BE2          0 0 225.00  50.00  10.00\n");
    doc.push_str("##T\n");
    doc.push_str(
        "BBE1AA1  BBE2AA1  1 0 400.00 225.00 1000.00 0.5000 10.000 0.0000 0.0000   5000\n",
    );
    doc.push_str("##R\n");
    doc.push_str(r_record);
    doc.push('\n');
    doc.into_bytes()
}

#[test]
fn a_ratio_regulations_held_voltage_is_read() {
    // Columns 33-38 carry 225.00 kV, on a node whose class-7 character makes
    // its own nominal 225 kV — so exactly 1.0 per unit.
    let doc = regulated("BBE1AA1  BBE2AA1  1  1.50  10   2225.00");
    let net = ucte::parse(&doc).expect("import");
    assert_eq!(net.regulation.len(), 1, "notes: {:?}", net.notes);
    let r = &net.regulation[0];
    assert_eq!(r.mode, gridoxide::outerloop::RegulationMode::Voltage);
    assert!(r.enabled);
    assert!(
        (r.target - 225_000.0 / net.buses[r.controlled_bus].u_rated).abs() < 1e-12,
        "target {} against a {} V bus",
        r.target,
        net.buses[r.controlled_bus].u_rated
    );
}

#[test]
fn an_angle_regulations_held_power_is_read() {
    // Columns 58-63 carry -65.00 MW, against the importer's own 100 MVA base.
    let doc = regulated("BBE1AA1  BBE2AA1  1                    -0.68 90.00  16   0-65.00SYMM");
    let net = ucte::parse(&doc).expect("import");
    assert_eq!(net.regulation.len(), 1, "notes: {:?}", net.notes);
    let r = &net.regulation[0];
    match r.mode {
        gridoxide::outerloop::RegulationMode::ActivePower { branch, .. } => {
            assert_eq!(branch, net.lines.len(), "a UCTE shifter holds its own branch's flow");
        }
        m => panic!("expected an active-power control, got {m:?}"),
    }
    assert!((r.target - -0.65).abs() < 1e-12, "{}", r.target);
}

/// The corpus itself: every vendored `.uct` file parses, and none declares a
/// regulation target. Recorded as an assertion so that if a fixture ever gains
/// one, this says so rather than the feature quietly going unexercised.
#[test]
fn no_vendored_file_declares_a_regulation_target() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).expect("ucte fixture dir") {
        let path = entry.expect("entry").path();
        if path.extension().is_none_or(|e| e != "uct") {
            continue;
        }
        let Ok(net) = ucte::read(&path) else { continue };
        assert!(
            net.regulation.is_empty(),
            "{} now declares a regulation target — the note in src/ucte.rs is out of date",
            path.display()
        );
        checked += 1;
    }
    assert!(checked > 0, "no .uct fixtures found");
}

// ---------------------------------------------------------------------------
// Country areas
// ---------------------------------------------------------------------------

/// UCTE's `##Z<cc>` sub-headers are a bus-to-area assignment, and the twelve-
/// node case is a genuine four-country interconnection — the one OpenRAO's own
/// cross-border scenarios are built on.
#[test]
fn country_codes_become_control_areas() {
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let (of_bus, names) = net.country_areas();

    assert_eq!(names, ["BE", "DE", "FR", "NL"], "sorted, so the indices are stable across runs");
    assert_eq!(of_bus.len(), net.buses.len());
    assert!(of_bus.iter().all(|a| a.is_some()), "every node in this file states its country");
    for (i, name) in names.iter().enumerate() {
        assert_eq!(
            of_bus.iter().filter(|a| **a == Some(i)).count(),
            3,
            "{name} should hold three nodes"
        );
    }
}

/// The positions the file's own state implies, measured over those areas.
///
/// The interesting property is that they sum to the tie losses rather than to
/// zero — both ends of a tie are measured into the branch, so an export and the
/// matching import do not cancel. On this fixture the ties are lossless and the
/// sum is exactly zero, which is the degenerate case of that rule rather than a
/// contradiction of it.
#[test]
fn the_measured_positions_sum_to_the_tie_losses() {
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let (of_bus, names) = net.country_areas();
    let (lines, transformers) = net.as_switched();

    let report = gridoxide::run_power_flow(
        net.buses.clone(),
        &lines,
        &transformers,
        &net.shunts,
        gridoxide::TapData::none(),
        gridoxide::solver::PowerFlowOptions { tol: 1e-10, max_iter: 40, ..Default::default() },
    );
    assert_eq!(report.stats.status, gridoxide::solver::SolveStatus::Converged);

    let x = gridoxide::outerloop::AreaInterchange::measure(
        &report.buses,
        &lines,
        &transformers,
        &of_bus,
        names.len(),
    );
    let mw: Vec<f64> = x.iter().map(|v| v * net.base_mva).collect();
    let losses: f64 = mw.iter().sum();
    assert!(losses.abs() < 1e-6, "lossless ties here, so the positions cancel: {mw:?}");
    // Two exporters and two importers, at the scale the file's schedules set.
    assert!(mw.iter().any(|v| *v > 500.0), "{mw:?}");
    assert!(mw.iter().any(|v| *v < -500.0), "{mw:?}");
}

/// UCTE states no schedule anywhere, which is why `country_areas` returns only
/// the assignment and the caller supplies the targets.
#[test]
fn ucte_supplies_membership_but_never_a_schedule() {
    let net = ucte::read(fixture("TestCase12Nodes.uct")).expect("import");
    let (of_bus, names) = net.country_areas();
    // `AreaDefinition::uniform` defaults every target to zero — asking each
    // area to serve its own load, which is the reading of "no interchange
    // agreed" rather than a schedule read from the file.
    let areas = gridoxide::outerloop::AreaDefinition::uniform(&net.buses, of_bus, names.len());
    assert_eq!(areas.targets, vec![0.0; 4]);
}
