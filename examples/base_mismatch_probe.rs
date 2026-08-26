//! Probe: branches whose two ends declare different voltage bases.
//!
//! An ACLineSegment cannot change voltage level, so its two ends are the same
//! physical level. If they declare different nominal voltages — 380 kV in
//! Belgium, 400 kV in the Netherlands, the same wire — then per-unitizing the
//! line on one end's base puts the two ends in different per-unit systems, and
//! `|V| = 1.0` at each no longer means the same volts.
//!
//! `cgmes::harmonize_voltage_bases` fixes this at import, so on a current build
//! every fixture should report **zero** spanning lines. What this probe is for
//! is the input side: it also prints what the harmonizer had to merge, which is
//! how a new document's cross-border pairs get noticed.
use gridoxide::cgmes::{cgmes_to_network, load_profiles};
use std::path::{Path, PathBuf};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0");
    let names: Vec<String> = std::env::args().skip(1).collect();
    for name in names {
        let dir = root.join(&name);
        if !dir.is_dir() { println!("{name}: absent"); continue }
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "xml"))
            .filter(|p| { let n = p.file_name().unwrap().to_string_lossy().to_uppercase();
                          ["_EQ", "_SSH", "_TP", "_SV", "EQ_BD", "EQBD"].iter().any(|k| n.contains(k)) })
            .collect();
        paths.sort();
        let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let Ok(ds) = load_profiles(&refs) else { println!("{name}: decode failed"); continue };
        let Ok(net) = cgmes_to_network(&ds, 100e6) else { println!("{name}: convert failed"); continue };

        let mut bad: Vec<(usize, f64, f64)> = Vec::new();
        for (i, l) in net.lines.iter().enumerate() {
            let (a, b) = (net.buses[l.from].u_rated, net.buses[l.to].u_rated);
            if a > 0.0 && b > 0.0 && (a - b).abs() / a.max(b) > 1e-6 {
                bad.push((i, a, b));
            }
        }
        let mut levels: std::collections::BTreeMap<u64, usize> = Default::default();
        for b in &net.buses { *levels.entry((b.u_rated / 1e3).round() as u64).or_default() += 1; }
        println!("{name}: {} line(s), {} spanning two bases after harmonizing (should be 0); \
                  levels {:?}", net.lines.len(), bad.len(), levels);
        let h = &net.base_harmonization;
        if h.buses > 0 {
            println!("   harmonized {} bus(es) over {} group(s): {}", h.buses, h.groups,
                h.merged.iter()
                    .map(|(k, r, c)| format!("{c} bus(es) {r:.0} kV -> {k:.0} kV"))
                    .collect::<Vec<_>>().join(", "));
        }
        for (i, a, b) in bad.iter().take(6) {
            let l = &net.lines[*i];
            println!("   line {i}: bus {} ({:.0} kV) -> bus {} ({:.0} kV)   ratio {:.4}, x = {:.5} pu",
                l.from, a / 1e3, l.to, b / 1e3, a / b, l.x);
        }
    }
}
