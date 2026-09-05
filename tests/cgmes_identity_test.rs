//! CGMES branch and bus identity: what a CRAC names its elements by.
//!
//! The importer produced buses, branches, shunts and tap tables and no way to
//! *name* any of them, so a CRAC — which resolves its network elements by
//! string — matched nothing on a CGMES network however well converted. This is
//! the naming half of making a CGMES-sourced phase shifter usable; the tap
//! table (`cgmes_tap_table_test.rs`) was the modelling half.
//!
//! The identifier is the mRID of the `ConductingEquipment` the branch came
//! from, which is what powsybl gives an IIDM element converted from CGMES and
//! therefore what a CRAC written against such a network names.

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_to_network, load_profiles, CgmesNetwork};

fn config(dir: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(dir);
    path.exists().then_some(path)
}

fn network(dir: &str) -> Option<CgmesNetwork> {
    let dir = config(dir)?;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("configuration directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "xml"))
        .collect();
    paths.sort();
    let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    let ds = load_profiles(&refs).expect("failed to decode CGMES profiles");
    Some(cgmes_to_network(&ds, 100e6).expect("conversion failed"))
}

const CONFIGS: [&str; 6] = [
    "Svedala/Svedala-Merged",
    "FullGrid/FullGrid-Merged",
    "SmallGrid/SmallGrid-Merged",
    "MiniGrid/MiniGrid-Merged",
    "PowerFlow/PowerFlow",
    "PST/PST_PhaseTapChangerLinear_Type1",
];

/// Parallel, complete and unique — the three properties `Resolution` needs.
///
/// Unique matters most: `Resolution::with_buses` keeps the *first* index for a
/// repeated id, so a duplicate would silently make one element unreachable and
/// point every mention of it at another.
#[test]
fn every_branch_and_bus_has_its_own_identifier() {
    let mut checked = 0;
    for name in CONFIGS {
        let Some(net) = network(name) else { continue };
        assert_eq!(
            net.branch_ids.len(),
            net.lines.len() + net.transformers.len(),
            "{name}: branch_ids must be parallel to lines-then-transformers"
        );
        assert_eq!(net.bus_ids.len(), net.buses.len(), "{name}: bus_ids must be parallel to buses");

        for (i, id) in net.branch_ids.iter().enumerate() {
            assert!(!id.is_empty(), "{name}: branch {i} has no identifier");
        }
        for (i, id) in net.bus_ids.iter().enumerate() {
            assert!(!id.is_empty(), "{name}: bus {i} has no identifier");
        }

        let mut seen: Vec<&str> = net.branch_ids.iter().map(String::as_str).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(before, seen.len(), "{name}: two branches share an identifier");

        let mut bus_seen: Vec<&str> = net.bus_ids.iter().map(String::as_str).collect();
        bus_seen.sort_unstable();
        let before = bus_seen.len();
        bus_seen.dedup();
        assert_eq!(before, bus_seen.len(), "{name}: two buses share an identifier");

        checked += net.branch_ids.len();
    }
    assert!(checked > 100, "expected the fixture set to contribute branches, got {checked}");
}

/// The identifiers are the model's own, not invented — every one resolves back
/// to an object the document declares, and a transformer's is its
/// `PowerTransformer`.
///
/// The single exception is a three-winding star point, which no document names.
#[test]
fn identifiers_come_from_the_document() {
    let Some(dir) = config("MiniGrid/MiniGrid-Merged") else {
        eprintln!("skipping: MiniGrid fixture not checked out");
        return;
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("configuration directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "xml"))
        .collect();
    paths.sort();
    let text: String = paths
        .iter()
        .map(|p| std::fs::read_to_string(p).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    let net = network("MiniGrid/MiniGrid-Merged").expect("network");

    // MiniGrid is the fixture with three-winding transformers, so it exercises
    // both the ordinary case and the exception.
    let stars = net.bus_ids.iter().filter(|id| id.ends_with("_star")).count();
    assert!(stars > 0, "MiniGrid should have at least one three-winding star point");

    for id in net.branch_ids.iter().chain(net.bus_ids.iter()) {
        if id.ends_with("_star") {
            continue;
        }
        assert!(
            text.contains(id.as_str()),
            "identifier {id} appears nowhere in the CGMES document"
        );
    }
}

/// The end-to-end property this exists for: a CRAC naming CGMES elements
/// resolves against the imported network.
///
/// Built from the structs rather than from JSON, so the test is about
/// resolution and not about the document format.
#[cfg(feature = "rao")]
#[test]
fn a_crac_resolves_against_an_imported_cgmes_network() {
    use gridoxide::rao::{
        Crac, FlowCnec, Instant, InstantKind, Resolution, Side, State, Threshold, Unit,
    };

    let Some(net) = network("SmallGrid/SmallGrid-Merged") else {
        eprintln!("skipping: SmallGrid fixture not checked out");
        return;
    };
    // Name a line and a transformer out of the imported model, which is what a
    // CRAC author would copy off the document.
    let line = net.branch_ids[0].clone();
    let transformer = net.branch_ids[net.lines.len()].clone();
    let cnec = |id: &str, element: &String| FlowCnec {
        id: id.to_string(),
        network_element: element.clone(),
        state: State::preventive(0),
        thresholds: vec![Threshold {
            unit: Unit::Megawatt,
            min: None,
            max: Some(100.0),
            side: Side::Both,
        }],
        reliability_margin: 0.0,
        optimized: true,
        monitored: false,
        operator: None,
        i_max: None,
        nominal_v: None,
    };
    let crac = Crac {
        id: "t".into(),
        name: None,
        instants: vec![Instant { id: "preventive".into(), kind: InstantKind::Preventive }],
        contingencies: Vec::new(),
        flow_cnecs: vec![cnec("a", &line), cnec("b", &transformer)],
        network_actions: Vec::new(),
        range_actions: Vec::new(),
        usage_limits: Vec::new(),
        angle_cnecs: 0,
        voltage_cnecs: 0,
    };

    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.bus_ids);
    assert!(
        resolution.is_complete(),
        "a CRAC naming imported elements should resolve: {:?}",
        resolution.unresolved
    );
    assert_eq!(resolution.branch(&line), Some(0));
    assert_eq!(resolution.branch(&transformer), Some(net.lines.len()));
}

/// The order is the same on every import.
///
/// `CimDataset::merge` iterates a `HashMap`, so `by_type` lists elements first
/// seen in a merged profile in a randomized order — and a CGMES import is
/// almost always several profiles merged. The transformer loop had imposed its
/// own order for exactly this reason; the line loops had not, and their indices
/// genuinely shuffled, so a branch's flat index differed between two imports of
/// the same files. Nothing asserted on a line index, so it never surfaced. It
/// surfaces the moment a branch has a name and a CRAC resolves to an index.
///
/// Two imports in one process is a real test of it: `RandomState` seeds each
/// `HashMap` differently, so the two datasets iterate differently, and before
/// `sorted_mrids` the two line orders disagreed.
#[test]
fn two_imports_of_the_same_model_agree_on_every_index() {
    for name in CONFIGS {
        let Some(first) = network(name) else { continue };
        let second = network(name).expect("second import");
        assert_eq!(
            first.branch_ids, second.branch_ids,
            "{name}: two imports disagree about which branch is which"
        );
        assert_eq!(
            first.bus_ids, second.bus_ids,
            "{name}: two imports disagree about which bus is which"
        );
        // And the electrical model itself, so the identifiers are not merely
        // consistent with each other but with the network they name.
        assert_eq!(first.lines.len(), second.lines.len(), "{name}: line count");
        for (i, (a, b)) in first.lines.iter().zip(&second.lines).enumerate() {
            assert_eq!((a.from, a.to), (b.from, b.to), "{name}: line {i} moved");
        }
    }
}

/// The whole point: a **phase-shifter range action on a CGMES network**.
///
/// This is what the two halves were for. `cgmes_tap_table_test.rs` gave the
/// shifter every position it can take; this file gave the branch a name a CRAC
/// can call it by; and `run_security` / `run_rao` in `src/main.rs` accept a
/// directory of profiles. With any of the three missing the optimizer either
/// cannot find the element, cannot move it, or is never handed it.
///
/// `PST_PhaseTapChangerLinear_Type1` is the fixture with a real phase shifter —
/// 21 positions spanning ±10°, currently on 6. The assertion is that the
/// optimizer *uses* it: the CNEC is given a threshold tight enough that sitting
/// still is not an answer, and the tap has to move to a position the changer
/// actually has.
#[cfg(feature = "rao")]
#[test]
fn a_phase_shifter_range_action_works_on_a_cgmes_network() {
    use gridoxide::opf::ipm::IpmSolver;
    use gridoxide::rao::{
        run, Crac, FlowCnec, Instant, InstantKind, Network, Range, RangeAction, RangeActionKind,
        RangeKind, Resolution, SearchOptions, Side, State, Threshold, Unit, UsageRule,
    };

    let Some(net) = network("PST/PST_PhaseTapChangerLinear_Type1") else {
        eprintln!("skipping: PST fixture not checked out");
        return;
    };
    // The one phase shifter in the model, found the way a CRAC author would:
    // the branch whose tap table actually changes the angle.
    let (offset, changer) = net
        .tap_changers
        .iter()
        .enumerate()
        .find_map(|(i, c)| {
            let c = c.as_ref()?;
            let span = c.angle_deg(c.high())? - c.angle_deg(c.low)?;
            (span.abs() > 1.0).then_some((i, c))
        })
        .expect("the PST fixture has a phase shifter");
    let element = net.branch_ids[net.lines.len() + offset].clone();
    let table: Vec<(i32, f64)> = (changer.low..=changer.high())
        .filter_map(|t| changer.angle_deg(t).map(|a| (t, a)))
        .collect();

    // A CNEC on the shifter's own branch, with a threshold under the flow it
    // currently carries, so doing nothing is not an answer.
    let view = Network {
        generation: &[],
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.bus_ids,
        initially_open: &[],
        bus_countries: &[],
        shunts: &net.shunts,
        tap_changers: &net.tap_changers,
        base_mva: 100.0,
    };
    let crac = Crac {
        id: "pst".into(),
        name: None,
        instants: vec![Instant { id: "preventive".into(), kind: InstantKind::Preventive }],
        contingencies: Vec::new(),
        flow_cnecs: vec![FlowCnec {
            id: "on the shifter".into(),
            network_element: element.clone(),
            state: State::preventive(0),
            thresholds: vec![Threshold {
                unit: Unit::Megawatt,
                min: Some(-20.0),
                max: Some(20.0),
                side: Side::Both,
            }],
            reliability_margin: 0.0,
            optimized: true,
            monitored: false,
            operator: None,
            i_max: None,
            nominal_v: None,
        }],
        network_actions: Vec::new(),
        range_actions: vec![RangeAction {
            id: "pst".into(),
            name: None,
            operator: None,
            usage_rules: vec![UsageRule::OnInstant { instant: 0 }],
            kind: RangeActionKind::Pst {
                element: element.clone(),
                initial_tap: changer.position,
                tap_to_angle: table,
            },
            ranges: vec![Range {
                kind: RangeKind::Absolute,
                min: Some(changer.low as f64),
                max: Some(changer.high() as f64),
            }],
            speed: None,
            activation_cost: None,
            group: None,
        }],
        usage_limits: Vec::new(),
        angle_cnecs: 0,
        voltage_cnecs: 0,
    };

    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.bus_ids);
    assert!(
        resolution.is_complete(),
        "the range action's element must resolve: {:?}",
        resolution.unresolved
    );

    let mut solver = IpmSolver::new();
    let plan = run(&crac, &view, &resolution, &mut solver, &SearchOptions::default());

    let setpoint = plan
        .preventive
        .setpoints
        .iter()
        .find(|s| s.action == 0)
        .expect("the shifter should be offered a set-point");
    let tap = setpoint.tap.expect("a phase shifter's set-point is a tap");
    assert!(
        (changer.low..=changer.high()).contains(&tap),
        "tap {tap} is outside the changer's own range {}..={}",
        changer.low,
        changer.high()
    );
    assert!(
        setpoint.moved(),
        "the CNEC is overloaded at the starting tap, so the optimizer should move the shifter"
    );
    assert!(
        plan.improvement() > 0.0,
        "moving it should improve the margin, not merely change it ({} MW)",
        plan.improvement()
    );
}
