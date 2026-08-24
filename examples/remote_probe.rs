//! Probe: machines that regulate a bus other than their own — where they are,
//! and what moving the control onto them does.
//!
//! Reports both modes side by side, because the interesting quantity is the
//! difference: which bus the reactive power comes out of, and how the solution
//! compares to the fixture's own published one.
use gridoxide::cgmes::{cgmes_to_network, cgmes_topological_node_bus_index, load_profiles};
use gridoxide::network::{build_ybus, power_injections, stamp_shunts};
use gridoxide::solver::PowerFlowOptions;
use std::path::{Path, PathBuf};

/// Published per-equipment reactive output, per-unit, in injection sign.
fn published_q(ds: &gridoxide::cgmes::CimDataset, base_mva: f64) -> std::collections::HashMap<String, f64> {
    let mut equipment_of: std::collections::HashMap<&str, &str> = Default::default();
    if let Some(ms) = ds.by_type.get("Terminal") {
        for m in ms {
            let t: &cimstructs::Terminal = ds.entries[m].element.as_any().downcast_ref().unwrap();
            if let Some(e) = &t.conducting_equipment {
                equipment_of.insert(m.as_str(), e.mrid.as_str());
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    if let Some(ms) = ds.by_type.get("SvPowerFlow") {
        for m in ms {
            let sv: &cimstructs::SvPowerFlow = ds.entries[m].element.as_any().downcast_ref().unwrap();
            if let (Some(t), Some(q)) = (&sv.terminal, sv.q)
                && let Some(e) = equipment_of.get(t.mrid.as_str())
            {
                out.insert((*e).to_string(), -q / base_mva);
            }
        }
    }
    out
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0");
    let names: Vec<String> = std::env::args().skip(1).collect();
    let names = if names.is_empty() {
        vec![
            "MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged".to_string(),
            "FullGrid/FullGrid-Merged".to_string(),
            "SmallGrid/SmallGrid-Merged".to_string(),
            "RealGrid/RealGrid-Merged".to_string(),
        ]
    } else {
        names
    };

    for name in names {
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
        let Ok(ds) = load_profiles(&refs) else {
            println!("{name}: decode failed");
            continue;
        };
        let Ok(net) = cgmes_to_network(&ds, 100e6) else {
            println!("{name}: convert failed");
            continue;
        };

        let remote: Vec<_> =
            net.voltage_control.machines.iter().filter(|m| m.at_bus != m.controls_bus).collect();
        println!(
            "\n=== {name}: {} bus(es), {} regulating machine(s), {} remote",
            net.buses.len(),
            net.voltage_control.machines.len(),
            remote.len()
        );
        if remote.is_empty() {
            continue;
        }

        for m in &remote {
            let tx = net.transformers.iter().find(|t| {
                (t.from == m.at_bus && t.to == m.controls_bus)
                    || (t.to == m.at_bus && t.from == m.controls_bus)
            });
            let between = match tx {
                Some(t) => format!("one transformer, x = {:.5} pu, tap {:.4}",
                    1.0 / t.y_series.im.abs().max(1e-12), t.tap.norm()),
                None => "not adjacent".to_string(),
            };
            println!(
                "  {} at bus {} ({:.1} kV) holds bus {} ({:.1} kV) at {:.4} — {between}",
                &m.id[..m.id.len().min(14)],
                m.at_bus,
                net.buses[m.at_bus].u_rated / 1e3,
                m.controls_bus,
                net.buses[m.controls_bus].u_rated / 1e3,
                m.target_pu
            );
        }

        let mut y = build_ybus(net.buses.len(), &net.lines, &net.transformers);
        stamp_shunts(&mut y, &net.shunts);
        let ybus = y.finish();
        let idx = cgmes_topological_node_bus_index(&ds).ok();
        let pubq = published_q(&ds, 100.0);

        for on in [false, true] {
            let rep = gridoxide::run_power_flow_with_remote(
                net.buses.clone(),
                &net.lines,
                &net.transformers,
                &net.shunts,
                gridoxide::TapData::none(),
                gridoxide::RemoteControlData { machines: &net.voltage_control.machines },
                PowerFlowOptions {
                    control_remote_voltage: on,
                    enforce_q_limits: true,
                    max_iter: 60,
                    max_outer_iter: 60,
                    ..Default::default()
                },
            );
            println!("  --control-remote-voltage {on:<5}  {:?}", rep.stats.status);
            if rep.stats.status != gridoxide::solver::SolveStatus::Converged {
                continue;
            }
            let (_, q) = power_injections(&rep.buses, &ybus);
            for m in &remote {
                println!(
                    "      Q at the machine {:+.5} pu ({:?}), at the bus it holds {:+.5} pu ({:?}); \
                     |V| held {:.5} vs target {:.5}{}",
                    q[m.at_bus],
                    rep.buses[m.at_bus].bus_type,
                    q[m.controls_bus],
                    rep.buses[m.controls_bus].bus_type,
                    rep.buses[m.controls_bus].voltage_mag,
                    m.target_pu,
                    match pubq.get(&m.id) {
                        Some(p) => format!("   [published machine output {p:+.5} pu]"),
                        None => String::new(),
                    }
                );
            }
            if let (Some(idx), Some(sv)) = (idx.as_ref(), ds.by_type.get("SvVoltage")) {
                let mut errs: Vec<f64> = Vec::new();
                for mrid in sv {
                    let v: &cimstructs::SvVoltage =
                        ds.entries[mrid].element.as_any().downcast_ref().unwrap();
                    let (Some(tn), Some(kv)) = (&v.topological_node, v.v) else { continue };
                    let Some(&b) = idx.get(&tn.mrid) else { continue };
                    let got = rep.buses[b].voltage_mag * net.buses[b].u_rated / 1e3;
                    errs.push((got - kv).abs() / kv.abs().max(1e-9));
                }
                errs.sort_by(f64::total_cmp);
                if !errs.is_empty() {
                    println!(
                        "      voltage vs published: worst {:.4}, median {:.4}, over {} bus(es)",
                        errs[errs.len() - 1],
                        errs[errs.len() / 2],
                        errs.len()
                    );
                }
            }
        }
    }
}
