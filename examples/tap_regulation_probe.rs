//! What the CGMES tap-regulation importer finds in a conformity fixture.
//!
//! Not a test — a probe, so the counts in `tests/cgmes_tap_regulation_test.rs`
//! and in `plans/TAP_CONTROL_PLAN.md` are recorded from what the importer
//! actually reads rather than from what the XML appears to say.
use std::path::{Path, PathBuf};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0");
    for name in [
        "Svedala/Svedala-Merged",
        "FullGrid/FullGrid-Merged",
        "SmallGrid/SmallGrid-Merged",
        "PowerFlow/PowerFlow",
        "PST/PST_PhaseTapChangerLinear_Type1",
        "PST/PST_PhaseTapChangerLinear_Type2",
        "PST/PST_PhaseTapChangerTable_Type3",
        "MiniGrid/MiniGrid-Merged",
    ] {
        let dir = root.join(name);
        if !dir.exists() {
            println!("{name}: not checked out");
            continue;
        }
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "xml"))
            .collect();
        paths.sort();
        let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let ds = match gridoxide::cgmes::load_profiles(&refs) {
            Ok(d) => d,
            Err(e) => {
                println!("{name}: load failed: {e}");
                continue;
            }
        };
        match gridoxide::cgmes::cgmes_to_network(&ds, 100e6) {
            Ok(net) => {
                let tables = net.tap_changers.iter().filter(|c| c.is_some()).count();
                println!(
                    "{name}: {} transformers, {tables} tap tables, {} controls, report {:?}",
                    net.transformers.len(),
                    net.regulation.len(),
                    net.tap_report
                );
                for r in &net.regulation {
                    let c = net.tap_changers[r.transformer].as_ref().unwrap();
                    println!(
                        "    transformer {} -> bus {} {:?} target {:.6} deadband {:.6} | pos {} in [{}, {}] neutral {} series {}",
                        r.transformer, r.controlled_bus, r.mode, r.target, r.deadband,
                        c.position, c.low, c.high(), c.neutral, c.series.is_some()
                    );
                }
            }
            Err(e) => println!("{name}: conversion failed: {e}"),
        }
    }
}
