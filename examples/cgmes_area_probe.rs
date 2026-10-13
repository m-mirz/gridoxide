//! CGMES ControlAreas: what the file says each area is scheduled to exchange,
//! and what it is exchanging at the solved state.
use std::path::{Path, PathBuf};
fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0");
    for name in std::env::args().skip(1) {
        let dir = root.join(&name);
        if !dir.is_dir() { println!("{name}: absent"); continue }
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "xml")).collect();
        paths.sort();
        let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let ds = gridoxide::cgmes::load_profiles(&refs).expect("load");
        let net = gridoxide::cgmes::cgmes_to_network(&ds, 100e6).expect("convert");
        let areas = gridoxide::cgmes::cgmes_control_areas(&ds, &net, 100e6).expect("areas");
        println!("=== {name}");
        println!("    {:?}", areas.report);
        let report = gridoxide::run_power_flow(
            net.buses.clone(), &net.lines, &net.transformers, &net.shunts,
            gridoxide::TapData::none(),
            gridoxide::solver::PowerFlowOptions { tol: 1e-8, max_iter: 40, ..Default::default() },
        );
        let x = gridoxide::outerloop::AreaInterchange::measure(
            &report.buses, &net.lines, &net.transformers, &areas.of_bus, areas.ids.len());
        println!("    solve {:?}", report.stats.status);
        // What sits at the buses no area claims — the boundary X-nodes.
        let unassigned: Vec<usize> = areas.of_bus.iter().enumerate()
            .filter(|(_, a)| a.is_none()).map(|(i, _)| i).collect();
        let inj: f64 = unassigned.iter().map(|&i| report.buses[i].p_spec).sum();
        println!("    {} bus(es) in no area, injecting {:.3} MW between them",
            unassigned.len(), inj * 100.0);
        println!("    sum of area positions: {:.3} MW", x.iter().sum::<f64>() * 100.0);
        for (i, (mrid, nm)) in areas.ids.iter().enumerate() {
            let buses = areas.of_bus.iter().filter(|a| **a == Some(i)).count();
            println!("    {:<4} ({}) {:>4} bus(es)  scheduled export {:>10.3} MW  actual {:>10.3} MW",
                nm, &mrid[..8.min(mrid.len())], buses, areas.targets[i] * 100.0, x[i] * 100.0);
        }
    }
}
