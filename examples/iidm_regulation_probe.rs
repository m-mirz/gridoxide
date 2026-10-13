//! How many regulating tap changers the IIDM importer finds across a tree of
//! `.xiidm` files, and what they hold.
fn main() {
    let root = std::env::args().nth(1).expect("a directory to walk");
    let mut files = 0usize;
    let mut volt = 0usize;
    let mut power = 0usize;
    let mut skipped = 0usize;
    let mut examples = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(root)];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "xiidm") {
                files += 1;
                let Ok(net) = gridoxide::iidm::read(&p) else { continue };
                for r in &net.regulation {
                    match r.mode {
                        gridoxide::outerloop::RegulationMode::Voltage => volt += 1,
                        gridoxide::outerloop::RegulationMode::ActivePower { .. } => power += 1,
                    }
                    if examples.len() < 6 {
                        examples.push(format!(
                            "{}: {} {:?} target {:.5} deadband {:.5}",
                            p.file_name().unwrap().to_string_lossy(),
                            r.id, r.mode, r.target, r.deadband
                        ));
                    }
                }
                skipped += net
                    .notes
                    .iter()
                    .filter(|n| n.contains("hold a current"))
                    .count();
            }
        }
    }
    println!("{files} files: {volt} voltage controls, {power} active-power controls, {skipped} files with current-mode changers dropped");
    for e in examples {
        println!("  {e}");
    }
}
