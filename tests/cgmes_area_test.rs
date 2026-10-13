//! CGMES `ControlArea`: the one place in the tree that states both an area
//! assignment *and* a scheduled net position.
//!
//! `outerloop::AreaInterchange` needs a bus-to-area map and a target per area.
//! CGMES states neither directly — it defines an area by its **boundary**, a
//! `TieFlow` per boundary terminal, and carries the schedule as
//! `ControlArea.netInterchange` in the SSH. `cgmes::cgmes_control_areas` turns
//! that into both.

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_control_areas, cgmes_to_network, load_profiles, CgmesNetwork};
use gridoxide::outerloop::{AreaDefinition, AreaInterchange, OuterLoop, SolveContext};
use gridoxide::solver::{JacobianBackend, PowerFlowOptions, SolveStatus};
use gridoxide::{run_power_flow, TapData};

const S_BASE: f64 = 100e6;

fn fixture(dir: &str) -> Option<(gridoxide::cgmes::CimDataset, CgmesNetwork)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(dir);
    if !dir.exists() {
        return None;
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("configuration directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "xml"))
        .collect();
    paths.sort();
    let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    let ds = load_profiles(&refs).expect("failed to decode CGMES profiles");
    let net = cgmes_to_network(&ds, S_BASE).expect("conversion failed");
    Some((ds, net))
}

/// MicroGrid is the tree's only genuine multi-area fixture: a merged BE + NL
/// model with five tie lines meeting at five boundary X-nodes.
#[test]
fn microgrid_resolves_two_areas_and_their_schedules() {
    let Some((ds, net)) = fixture("MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged") else {
        eprintln!("skipping: MicroGrid fixture not checked out");
        return;
    };
    let areas = cgmes_control_areas(&ds, &net, S_BASE).expect("control areas");

    assert_eq!(areas.report.areas, 2);
    assert_eq!(areas.report.tie_flows, 10, "five ties, named from both sides");
    assert_eq!(areas.report.unresolved_tie_flows, 0);
    assert!(areas.report.without_boundary.is_empty());
    assert!(areas.report.without_target.is_empty());

    // Zero contested is the load-bearing one. A `TieFlow` names the terminal at
    // the *boundary* — `TN_Border_AL11` carries both `NL-Line_1`'s and
    // `BE-Line_3`'s — so seeding membership from the named end puts every
    // area's seed on the same node. That produced five contested buses and one
    // area owning nothing; the fix is to seed from the branch's far end.
    assert_eq!(areas.report.contested_buses, 0);
    assert_eq!(areas.report.unassigned_buses, 5, "the five boundary X-nodes belong to neither");

    let names: Vec<&str> = areas.ids.iter().map(|(_, n)| n.as_str()).collect();
    assert!(names.contains(&"NL") && names.contains(&"BE"), "{names:?}");

    // Every bus that is not a boundary node belongs to exactly one area.
    let claimed = areas.of_bus.iter().filter(|a| a.is_some()).count();
    assert_eq!(claimed + areas.report.unassigned_buses, net.buses.len());
    for (i, (_, name)) in areas.ids.iter().enumerate() {
        let n = areas.of_bus.iter().filter(|a| **a == Some(i)).count();
        assert!(n > 0, "area {name} owns no bus");
    }

    // The schedules are equal and opposite, as a two-area interchange must be.
    let mw: Vec<f64> = areas.targets.iter().map(|t| t * 100.0).collect();
    assert!((mw[0] + mw[1]).abs() < 1e-6, "{mw:?}");
    assert!(mw.iter().any(|v| (v - 236.9771).abs() < 1e-3), "{mw:?}");
}

/// **The sign, checked against the fixture's own solved state.**
///
/// CGMES states `netInterchange` as an *import* — "positive sign means flow in
/// to the area" — while `AreaDefinition::targets` is an export, so the importer
/// negates. Getting that backwards would dispatch every area exactly the wrong
/// way while converging perfectly happily.
///
/// SmallGrid settles the question without a second tool: it declares one area
/// with a net interchange, and its own published state is *already* on that
/// schedule. Measured export and stated schedule agree to 0.3 MW out of 210 —
/// which they could not, with the sign flipped.
#[test]
fn smallgrids_published_state_already_meets_its_own_schedule() {
    let Some((ds, net)) = fixture("SmallGrid/SmallGrid-Merged") else {
        eprintln!("skipping: SmallGrid fixture not checked out");
        return;
    };
    let areas = cgmes_control_areas(&ds, &net, S_BASE).expect("control areas");
    assert_eq!(areas.report.areas, 1);
    assert_eq!(areas.report.contested_buses, 0);

    let solved = run_power_flow(
        net.buses.clone(),
        &net.lines,
        &net.transformers,
        &net.shunts,
        TapData::none(),
        PowerFlowOptions { tol: 1e-8, max_iter: 40, ..Default::default() },
    );
    assert_eq!(solved.stats.status, SolveStatus::Converged);

    let measured = AreaInterchange::measure(
        &solved.buses,
        &net.lines,
        &net.transformers,
        &areas.of_bus,
        areas.ids.len(),
    );
    let (scheduled_mw, actual_mw) = (areas.targets[0] * 100.0, measured[0] * 100.0);
    assert!(
        (scheduled_mw - 210.0).abs() < 1e-6,
        "the fixture declares a 210 MW export, read as {scheduled_mw}"
    );
    assert!(
        (actual_mw - scheduled_mw).abs() < 0.5,
        "the published state exports {actual_mw} MW against a declared {scheduled_mw} — a sign \
         error would make these differ by twice the number, not by a rounding"
    );
}

/// An area with no tie flow has no measurable boundary, so its position can be
/// neither read nor controlled. Reported rather than silently balanced at zero.
#[test]
fn an_area_without_tie_flows_is_reported() {
    let Some((ds, net)) = fixture("Svedala/Svedala-Merged") else {
        eprintln!("skipping: Svedala fixture not checked out");
        return;
    };
    let areas = cgmes_control_areas(&ds, &net, S_BASE).expect("control areas");
    assert_eq!(areas.report.areas, 1);
    assert_eq!(areas.report.tie_flows, 0);
    assert_eq!(areas.report.without_boundary, vec![0]);
    assert!(areas.of_bus.iter().all(|a| a.is_none()), "no boundary means no membership either");
}

/// End to end: the loop moves MicroGrid's two areas onto the schedules its own
/// file declares.
///
/// The published state exports 261 MW from NL against a declared 237, so there
/// is real work to do — this cannot pass by the network already being right.
#[test]
fn the_loop_moves_microgrid_onto_its_declared_schedule() {
    let Some((ds, net)) = fixture("MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged") else {
        eprintln!("skipping: MicroGrid fixture not checked out");
        return;
    };
    let areas = cgmes_control_areas(&ds, &net, S_BASE).expect("control areas");

    let before = AreaInterchange::measure(
        &run_power_flow(
            net.buses.clone(),
            &net.lines,
            &net.transformers,
            &net.shunts,
            TapData::none(),
            PowerFlowOptions { tol: 1e-8, max_iter: 40, ..Default::default() },
        )
        .buses,
        &net.lines,
        &net.transformers,
        &areas.of_bus,
        areas.ids.len(),
    );

    let mut buses = net.buses.clone();
    let mut ybus = {
        let mut y = gridoxide::network::build_ybus(buses.len(), &net.lines, &net.transformers);
        gridoxide::network::stamp_shunts(&mut y, &net.shunts);
        y.finish()
    };
    let definition = AreaDefinition {
        targets: areas.targets.clone(),
        tolerance: 1e-9,
        ..AreaDefinition::uniform(&buses, areas.of_bus.clone(), areas.ids.len())
    };
    let mut transformers = net.transformers.clone();
    let mut changers = Vec::new();
    let mut area_loop = AreaInterchange::new(definition);
    let (islands, report) = {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus)
            .with_branches(&net.lines, &mut transformers, &net.shunts)
            .with_taps(&mut changers, &[]);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut area_loop];
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-9, 40, JacobianBackend::Scalar, &mut list, 60)
    };
    let area_report = area_loop.into_report();

    assert!(
        islands.iter().all(|i| !matches!(
            i.status,
            gridoxide::solver::IslandStatus::Singular
                | gridoxide::solver::IslandStatus::MaxIterationsReached
        )),
        "{islands:?}"
    );
    assert!(report.converged, "{report:?}");

    let after = AreaInterchange::measure(&buses, &net.lines, &transformers, &areas.of_bus, areas.ids.len());
    let dependent = area_report.dependent.expect("one area holds the slack");

    // Every area but the dependent one lands on its declared schedule.
    for a in 0..areas.ids.len() {
        if a == dependent {
            continue;
        }
        assert!(
            (after[a] - areas.targets[a]).abs() < 1e-7,
            "area {} exports {} against a declared {}",
            areas.ids[a].1,
            after[a] * 100.0,
            areas.targets[a] * 100.0
        );
        // And it actually had to move to get there.
        assert!(
            (before[a] - areas.targets[a]).abs() > 0.1,
            "area {} was already on schedule; this test proves nothing",
            areas.ids[a].1
        );
    }
}
