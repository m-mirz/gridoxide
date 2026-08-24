//! Probe: per-machine reactive allocation at shared buses.
use gridoxide::dispatch::{allocate, reactive_keys};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::solver::PowerFlowOptions;

fn main() {
    let path = std::env::args().nth(1).expect("path to a CGMES directory");
    let dir = std::path::Path::new(&path);
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir).expect("dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "xml"))
        .collect();
    paths.sort();
    let refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();
    let ds = gridoxide::cgmes::load_profiles(&refs).expect("load");
    let net = gridoxide::cgmes::cgmes_to_network(&ds, 100e6).expect("convert");
    let vc = &net.voltage_control;
    println!("{} regulated bus(es), {} shared, {} regulating machine(s)",
        vc.regulated_buses, vc.shared_buses, vc.machines.len());

    let report = gridoxide::run_power_flow(
        net.buses.clone(), &net.lines, &net.transformers, &net.shunts,
        gridoxide::TapData::none(),
        PowerFlowOptions { enforce_q_limits: true, max_iter: 60, max_outer_iter: 60, ..Default::default() });
    println!("solve: {:?}", report.stats.status);

    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let d = allocate(&vc.machines, &vc.nonregulating_q, &report.buses, &ybus);

    // Group by controlled bus, show the shared ones.
    let mut by_bus: std::collections::BTreeMap<usize, Vec<&gridoxide::dispatch::MachineDispatch>> =
        Default::default();
    for bd in &d { for m in &bd.machines { by_bus.entry(m.controls_bus).or_default().push(m); } }
    let shared: Vec<_> = by_bus.iter().filter(|(_, v)| v.len() > 1).collect();
    println!("{} bus(es) with >1 machine\n", shared.len());

    for (bus, ms) in shared.iter().take(4) {
        let refs: Vec<&gridoxide::types::RegulatingMachine> =
            vc.machines.iter().filter(|m| m.controls_bus == **bus).collect();
        let (_, basis) = reactive_keys(&refs);
        let total: f64 = ms.iter().map(|m| m.q).sum();
        println!("bus {bus}: {} machines, keys from {:?}, total Q = {:+.6} pu", ms.len(), basis, total);
        for m in ms.iter() {
            println!("   {:<22} q={:+.6}  share={:.3}  range=[{:+.4},{:+.4}]{}",
                &m.id[..m.id.len().min(22)], m.q, m.share, m.q_min, m.q_max,
                if m.at_limit { "  AT LIMIT" } else { "" });
        }
        println!();
    }

    // Invariants worth seeing before writing a test around them.
    // Compare against the fixture's own published per-terminal solution.
    let mut published: std::collections::HashMap<String, f64> = Default::default();
    if let Some(mrids) = ds.by_type.get("Terminal") {
        let mut eq_of: std::collections::HashMap<&str, &str> = Default::default();
        for m in mrids {
            let t: &cimstructs::Terminal = ds.entries[m].element.as_any().downcast_ref().unwrap();
            if let Some(e) = &t.conducting_equipment { eq_of.insert(m.as_str(), e.mrid.as_str()); }
        }
        if let Some(fl) = ds.by_type.get("SvPowerFlow") {
            for m in fl {
                let sv: &cimstructs::SvPowerFlow = ds.entries[m].element.as_any().downcast_ref().unwrap();
                let (Some(t), Some(q)) = (&sv.terminal, sv.q) else { continue };
                if let Some(e) = eq_of.get(t.mrid.as_str()) { published.insert((*e).to_string(), -q / 100.0); }
            }
        }
    }
    let mut matched = 0usize; let mut worst_m = 0.0f64; let mut sum_abs = 0.0f64; let mut sum_err = 0.0f64;
    for bd in &d { for m in &bd.machines {
        if let Some(&exp) = published.get(&m.id) {
            matched += 1;
            worst_m = worst_m.max((m.q - exp).abs());
            sum_abs += exp.abs(); sum_err += (m.q - exp).abs();
        }
    }}
    println!("published per-machine Q: matched {matched}/{}  worst |diff| = {worst_m:.4} pu  \
              mean |diff| / mean |published| = {:.1}%",
        d.iter().map(|b| b.machines.len()).sum::<usize>(),
        if sum_abs > 0.0 { 100.0 * sum_err / sum_abs } else { 0.0 });

    // Decompose: does the *bus total* agree (the solve) or the *split* (the rule)?
    let (mut bus_err, mut bus_abs) = (0.0f64, 0.0f64);
    let (mut split_err, mut split_abs) = (0.0f64, 0.0f64);
    let (mut shared_split_err, mut shared_split_abs) = (0.0f64, 0.0f64);
    let mut worst_bus = (0.0f64, 0usize);
    for bd in &d {
        let pub_sum: f64 = bd.machines.iter().filter_map(|m| published.get(&m.id)).sum();
        let n_pub = bd.machines.iter().filter(|m| published.contains_key(&m.id)).count();
        if n_pub != bd.machines.len() { continue }
        bus_err += (bd.attributed - pub_sum).abs();
        bus_abs += pub_sum.abs();
        if (bd.attributed - pub_sum).abs() > worst_bus.0 { worst_bus = ((bd.attributed - pub_sum).abs(), bd.bus); }
        // Split quality, measured against the *published* total so the solve
        // difference is factored out.
        let (keys, _) = reactive_keys(&vc.machines.iter().filter(|m| m.controls_bus == bd.bus).collect::<Vec<_>>());
        for (k, m) in bd.machines.iter().enumerate() {
            let exp = published[&m.id];
            let ours_on_pub_total = pub_sum * keys[k];
            split_err += (ours_on_pub_total - exp).abs();
            split_abs += exp.abs();
            if bd.machines.len() > 1 {
                shared_split_err += (ours_on_pub_total - exp).abs();
                shared_split_abs += exp.abs();
            }
        }
    }
    println!("  bus totals   (the solve): mean rel err {:.1}%   worst {:.4} pu at bus {}",
        100.0 * bus_err / bus_abs.max(1e-12), worst_bus.0, worst_bus.1);
    println!("  the split    (the rule) : mean rel err {:.1}%   (shared buses only: {:.1}%)",
        100.0 * split_err / split_abs.max(1e-12),
        100.0 * shared_split_err / shared_split_abs.max(1e-12));

    // Basis distribution over shared buses, and unattributed at a physical threshold.
    let mut basis_counts: std::collections::BTreeMap<String, usize> = Default::default();
    for bd in &d { if bd.machines.len() > 1 {
        *basis_counts.entry(format!("{:?}", bd.basis)).or_default() += 1; } }
    println!("shared-bus key basis: {basis_counts:?}");
    for thr in [1e-9f64, 1e-7, 1e-6, 1e-5] {
        let n = d.iter().filter(|b| b.unattributed.abs() > thr).count();
        let t: f64 = d.iter().filter(|b| b.unattributed.abs() > thr).map(|b| b.unattributed.abs()).sum();
        println!("  unattributed > {thr:.0e}: {n} bus(es), {t:.4} pu");
    }
    // Which shared buses fall back, and why?
    for bd in d.iter().filter(|b| b.machines.len() > 1 && format!("{:?}", b.basis) != "Capability") {
        println!("  FALLBACK bus {} -> {:?}", bd.bus, bd.basis);
        for m in &bd.machines { println!("     {} range=[{:+.6},{:+.6}] width={:.6}", &m.id[..12], m.q_min, m.q_max, m.q_max - m.q_min); }
    }

    let (_, qall) = gridoxide::network::power_injections(&report.buses, &ybus);
    let mut worst: Vec<(f64, usize, usize, usize, f64, f64)> = Vec::new();
    let mut violations = 0;
    for (bus, ms) in &by_bus {
        let total: f64 = ms.iter().map(|m| m.q).sum();
        let expected = qall[*bus] - vc.nonregulating_q[*bus];
        let pinned = ms.iter().filter(|m| m.at_limit).count();
        worst.push(((total - expected).abs(), *bus, ms.len(), pinned, total, expected));
        for m in ms.iter() {
            if m.q > m.q_max + 1e-9 || m.q < m.q_min - 1e-9 { violations += 1; }
        }
    }
    let n_bad = worst.iter().filter(|w| w.0 > 1e-9).count();
    let total_bad: f64 = worst.iter().map(|w| w.0).sum();
    // How many regulated buses have reactive load co-located with the machine?
    let with_load = by_bus.keys().filter(|b| vc.nonregulating_q[**b].abs() > 1e-9).count();
    println!("regulated buses: {}  with co-located reactive injection: {with_load}  \
              unattributable: {n_bad}  total unattributed = {total_bad:.4} pu", by_bus.len());
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("limit violations = {violations}");
    println!("worst mismatches (err, bus, machines, pinned, sum, expected):");
    for w in worst.iter().take(6) {
        let bt = report.buses[w.1].bus_type;
        println!("  {:.3e}  bus {:4} n={} pinned={} sum={:+.6} expected={:+.6}  type={:?} qlim=[{:+.4},{:+.4}]",
            w.0, w.1, w.2, w.3, w.4, w.5, bt, report.buses[w.1].q_min, report.buses[w.1].q_max);
    }
}
