//! Is a fixture's *published* solution inside its own tap-control deadbands?
//!
//! The question that decides whether a tap loop moving a tap on a conformity
//! fixture is a defect or a finding. Compares, at the positions the SSH names:
//! the controlled bus's voltage as gridoxide solves it, and as the fixture's
//! own SV profile publishes it, each against the control's target and deadband.
use cimstructs;
use std::path::{Path, PathBuf};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0");
    for name in std::env::args().skip(1) {
        let dir = root.join(&name);
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "xml"))
            .collect();
        paths.sort();
        let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let ds = gridoxide::cgmes::load_profiles(&refs).expect("load");
        let mut net = gridoxide::cgmes::cgmes_to_network(&ds, 100e6).expect("convert");
        // FullGrid is an HVDC model; without resolving its converters into
        // AC-side injections the AC solve has nothing driving those buses.
        match gridoxide::cgmes::cgmes_resolve_dc_converters(&ds, &mut net.buses, 100e6) {
            Ok(_) => println!("    dc: resolved"),
            Err(e) => println!("    dc: {e}"),
        }
        let index = gridoxide::cgmes::cgmes_topological_node_bus_index(&ds).expect("index");

        // The published SvVoltage per bus, in per unit.
        let mut published = vec![f64::NAN; net.buses.len()];
        for mrid in &ds.by_type["SvVoltage"] {
            let sv: &cimstructs::SvVoltage = ds.entries[mrid]
                .element
                .as_any()
                .downcast_ref()
                .expect("SvVoltage downcast");
            let (Some(tn), Some(v)) = (sv.topological_node.as_ref(), sv.v) else { continue };
            if let Some(&b) = index.get(&tn.mrid) {
                published[b] = v * 1e3 / net.buses[b].u_rated;
            }
        }

        // gridoxide's own solve at the positions the document names.
        let report = gridoxide::run_power_flow(
            net.buses.clone(),
            &net.lines,
            &net.transformers,
            &net.shunts,
            gridoxide::TapData::none(),
            gridoxide::solver::PowerFlowOptions { tol: 1e-8, max_iter: 30, ..Default::default() },
        );
        // And the same, after letting the tap loops run.
        let after = gridoxide::run_power_flow(
            net.buses.clone(),
            &net.lines,
            &net.transformers,
            &net.shunts,
            gridoxide::TapData { changers: &net.tap_changers, regulation: &net.regulation },
            gridoxide::solver::PowerFlowOptions {
                control_taps: true,
                max_outer_iter: 60,
                tol: 1e-8,
                max_iter: 30,
                ..Default::default()
            },
        );
        println!(
            "=== {name}: base solve {:?}, controlled solve {:?}",
            report.stats.status, after.stats.status
        );
        println!(
            "{:>4} {:>9} {:>9} {:>9} {:>9}  {:>9} {:>9}",
            "bus", "target", "half-db", "gridoxide", "|err|", "published", "|err|"
        );
        for r in &net.regulation {
            if r.mode != gridoxide::outerloop::RegulationMode::Voltage {
                continue;
            }
            let mine = report.buses[r.controlled_bus].voltage_mag;
            let theirs = published[r.controlled_bus];
            println!(
                "{:>4} {:>9.5} {:>9.5} {:>9.5} {:>9.5}{} {:>9.5} {:>9.5}{}",
                r.controlled_bus,
                r.target,
                r.deadband / 2.0,
                mine,
                (mine - r.target).abs(),
                if (mine - r.target).abs() > r.deadband / 2.0 { " OUT" } else { "  in" },
                theirs,
                (theirs - r.target).abs(),
                if (theirs - r.target).abs() > r.deadband / 2.0 { " OUT" } else { "  in" },
            );
            let controlled = after.buses[r.controlled_bus].voltage_mag;
            println!(
                "     -> after tap control: {controlled:.5}  |err| {:.5}{}",
                (controlled - r.target).abs(),
                if (controlled - r.target).abs() > r.deadband / 2.0 { " OUT" } else { "  in" },
            );
        }
    }
}
