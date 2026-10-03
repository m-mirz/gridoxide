//! Runs the tap-control outer loops over a CGMES conformity fixture and prints
//! what each controller decided, against the position the fixture's own SV
//! profile published.
use std::path::{Path, PathBuf};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0");
    for name in std::env::args().skip(1) {
        let dir = root.join(&name);
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "xml"))
            .collect();
        paths.sort();
        let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let ds = gridoxide::cgmes::load_profiles(&refs).expect("load");
        let net = gridoxide::cgmes::cgmes_to_network(&ds, 100e6).expect("convert");

        let before: Vec<Option<i32>> =
            net.tap_changers.iter().map(|c| c.as_ref().map(|c| c.position)).collect();

        let report = gridoxide::run_power_flow(
            net.buses.clone(),
            &net.lines,
            &net.transformers,
            &net.shunts,
            gridoxide::TapData { changers: &net.tap_changers, regulation: &net.regulation },
            gridoxide::solver::PowerFlowOptions {
                control_taps: true,
                enforce_q_limits: true,
                max_outer_iter: 40,
                tol: 1e-8,
                max_iter: 30,
                ..Default::default()
            },
        );
        let outer = report.outer.as_ref().expect("loops ran");
        println!("=== {name}: {:?}", report.stats.status);
        println!("    {:?}", outer.report.as_ref().unwrap());
        for c in &outer.taps {
            println!(
                "    tf {:>3}  {} -> {}  steps {}  reversals {}  {:?}",
                c.transformer, c.initial_position, c.final_position, c.steps_moved,
                c.direction_changes, c.outcome
            );
        }
        let moved: Vec<usize> = outer
            .changers
            .iter()
            .enumerate()
            .filter(|(i, c)| c.as_ref().map(|c| c.position) != before[*i])
            .map(|(i, _)| i)
            .collect();
        println!("    transformers whose tap moved: {moved:?}");
    }
}
