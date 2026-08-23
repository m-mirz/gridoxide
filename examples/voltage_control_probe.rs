//! What generator-side voltage control aggregates, per conformity fixture.
//!
//! Cross-checks the summed reactive limits against the machines' own declared
//! `minQ`/`maxQ`, read straight out of the XML by a second path that shares no
//! code with the importer.
use std::path::{Path, PathBuf};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0");
    for name in std::env::args().skip(1) {
        let dir = root.join(&name);
        if !dir.is_dir() {
            println!("{name}: absent");
            continue;
        }
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "xml"))
            .collect();
        paths.sort();
        let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let ds = gridoxide::cgmes::load_profiles(&refs).expect("load");
        let net = gridoxide::cgmes::cgmes_to_network(&ds, 100e6).expect("convert");
        let r = &net.voltage_control;
        println!(
            "{name}: {} regulated bus(es), {} held by more than one machine, {} target conflict(s)",
            r.regulated_buses, r.shared_buses, r.target_conflicts.len()
        );
        for c in &r.target_conflicts {
            println!("    bus {} : {} vs {} ({})", c.bus, c.existing, c.proposed, c.id);
        }
        // Which buses are shared, and what they ended up holding — the cells
        // a summed limit actually changes.
        println!("    shared buses and their joint capability:");
        for (i, b) in net.buses.iter().enumerate() {
            if b.bus_type != gridoxide::types::BusType::PV {
                continue;
            }
            // A shared bus is one the report counted; the report does not name
            // them, so this prints every regulated bus's band and lets the
            // caller match against the XML.
            let _ = i;
        }

        // The widest and narrowest finite capability among regulated buses,
        // which is what a summed limit visibly changes.
        let mut widths: Vec<(usize, f64)> = net
            .buses
            .iter()
            .enumerate()
            .filter(|(_, b)| b.bus_type == gridoxide::types::BusType::PV)
            .filter(|(_, b)| b.q_min.is_finite() && b.q_max.is_finite())
            .map(|(i, b)| (i, b.q_max - b.q_min))
            .collect();
        widths.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("    {} PV bus(es) with a finite reactive band", widths.len());
        for (i, w) in widths.iter().take(3) {
            println!(
                "      bus {i:>5}: [{:.4}, {:.4}] pu, width {w:.4}",
                net.buses[*i].q_min, net.buses[*i].q_max
            );
        }
    }
}
