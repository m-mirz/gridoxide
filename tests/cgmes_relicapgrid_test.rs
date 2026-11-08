//! ENTSO-E's ReliCapGrid, the dataset of the 2026 SV-IOP (`tests/data/relicapgrid`):
//! nine IGMs, their boundary and common data, and an assembled CGM. Skips without
//! the submodule. What each case's data contains, and which of them agree with their
//! SV, is pinned case by case in cimoxide's `scripts/relicapgrid/test_power_flow.py`.

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{
    CimDataset, cgmes_resolve_dc_converters, cgmes_to_buses_and_branches, cgmes_to_network,
    cgmes_topological_node_bus_index, load_profiles,
};
use gridoxide::network::{build_ybus, connected_components, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;
use gridoxide::solver::IslandStatus;
use gridoxide::types::{Bus, BusType};

mod cgmes_common;

fn instance() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/relicapgrid/Instance");
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init tests/data/relicapgrid`",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

fn files(root: &Path, areas: &[&str]) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = areas
        .iter()
        .chain(&["commonData"])
        .flat_map(|area| {
            std::fs::read_dir(root.join(area).join("Grid/cimxml"))
                .expect("area directory")
                .map(|f| f.expect("directory entry").path())
                .collect::<Vec<_>>()
        })
        .collect();
    paths.sort();
    paths
}

fn decode(paths: &[PathBuf]) -> CimDataset {
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    load_profiles(&refs).expect("decoding the profiles")
}

/// Buses, with HVDC converters resolved, solved without reactive limits.
fn solve(ds: &CimDataset) -> (Vec<Bus>, Vec<IslandStatus>) {
    let s_base = 100e6;
    let net = cgmes_to_network(ds, s_base).expect("conversion");
    let mut buses = net.buses;
    cgmes_resolve_dc_converters(ds, &mut buses, s_base).expect("converters");
    let mut ybus = build_ybus(buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus);
    let status = result.islands.iter().map(|i| i.status).collect();
    (result.buses, status)
}

/// Espheim's 10.5 kV busbar has 34 connected terminals on nodes its TP places on no
/// `TopologicalNode`, transformer `T2` among them, whose tap changer no CGM SSH carries.
#[test]
fn equipment_the_topology_leaves_out_stays_off_the_network() {
    let Some(root) = instance() else { return };
    let (buses, status) = solve(&decode(&files(&root, &["Espheim"])));
    assert!(!buses.is_empty());
    assert!(
        status
            .iter()
            .all(|s| matches!(s, IslandStatus::Converged | IslandStatus::NoReferenceBus))
    );
}

#[test]
fn equipment_out_of_service_conducts_nothing() {
    let Some(root) = instance() else { return };
    let ds = decode(&files(&root, &["Espheim"]));
    let segments = &ds.by_type["ACLineSegment"];
    let out = segments
        .iter()
        .filter(|m| ds.entries[*m].get_bool("Equipment.inService") == Some(false))
        .count();
    assert_eq!(out, 5);
    let (_, lines, _, _) = cgmes_to_buses_and_branches(&ds, 100e6).expect("conversion");
    assert_eq!(lines.len(), segments.len() - out);
}

#[test]
fn the_boundary_changes_nothing() {
    let Some(root) = instance() else { return };
    let (alone, _) = solve(&decode(&files(&root, &["Belgovia"])));
    let (merged, _) = solve(&decode(&files(&root, &["Belgovia", "boundaryData"])));
    assert_eq!(alone.len(), merged.len());
    for (a, b) in alone.iter().zip(&merged) {
        assert_eq!(a.voltage_mag, b.voltage_mag);
    }
}

/// The IGMs meet at shared boundary nodes, each with its references and injections.
#[test]
fn merged_igms_hold_one_reference_per_component() {
    let Some(root) = instance() else { return };
    let ds = decode(&files(&root, &["Belgovia", "Svedala", "Galia"]));
    let net = cgmes_to_network(&ds, 100e6).expect("conversion");
    let ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers).finish();
    for component in connected_components(&ybus) {
        let references = component
            .iter()
            .filter(|&&b| net.buses[b].bus_type == BusType::Slack && net.buses[b].voltage_mag > 0.0)
            .count();
        assert!(references <= 1, "{references} references in one component");
    }
    // BP_SD-BO1's two injections are replaced by the tie: nothing else is on the node.
    let index = cgmes_topological_node_bus_index(&ds).expect("index");
    let boundary = ds.by_type["TopologicalNode"]
        .iter()
        .find(|m| ds.entries[*m].get_str("IdentifiedObject.name") == Some("BP_SD-BO1"))
        .expect("the boundary point");
    let bus = &net.buses[index[boundary]];
    assert_eq!((bus.p_spec, bus.q_spec), (0.0, 0.0));

    let (_, status) = solve(&ds);
    assert!(
        status
            .iter()
            .all(|s| matches!(s, IslandStatus::Converged | IslandStatus::NoReferenceBus))
    );
}

/// One injection facing an IGM not loaded still stands in for it.
#[test]
fn an_injection_facing_an_absent_igm_stays() {
    let Some(root) = instance() else { return };
    let ds = decode(&files(&root, &["Belgovia", "Svedala"]));
    let net = cgmes_to_network(&ds, 100e6).expect("conversion");
    let index = cgmes_topological_node_bus_index(&ds).expect("index");
    let boundary = ds.by_type["TopologicalNode"]
        .iter()
        .find(|m| ds.entries[*m].get_str("IdentifiedObject.name") == Some("BP_GA-BO2"))
        .expect("Belgovia's boundary point with Galia");
    assert_ne!(net.buses[index[boundary]].p_spec, 0.0);
}

/// Britheim's `PccTerminal` is on a line ending at the angle reference: injected there,
/// a converter's power vanishes. Its other converter holds 300 kV at its PCC.
#[test]
fn a_converter_injects_at_its_own_terminal_and_holds_at_its_pcc() {
    let Some(root) = instance() else { return };
    let ds = decode(&files(&root, &["Britheim"]));
    let (solved, _) = solve(&ds);
    let index = cgmes_topological_node_bus_index(&ds).expect("index");
    cgmes_common::assert_matches_sv(&solved, &index, &cgmes_common::expected_voltages(&ds), 5e-3);
}

/// Sets `mrid` out of service, as a contingency does.
fn trip(ds: &mut CimDataset, mrid: &str) {
    let out = gridoxide::cgmes::out_of_service(&ds.entries[mrid]);
    ds.entries.insert(mrid.to_string(), out);
}

/// The AC power each bus gets from the HVDC link, per unit: the larger of P and Q.
fn link_injection(ds: &CimDataset) -> Vec<f64> {
    let s_base = 100e6;
    let net = cgmes_to_network(ds, s_base).expect("conversion");
    let mut buses = net.buses.clone();
    cgmes_resolve_dc_converters(ds, &mut buses, s_base).expect("converters");
    buses
        .iter()
        .zip(&net.buses)
        .map(|(with, without)| {
            (with.p_spec - without.p_spec)
                .abs()
                .max((with.q_spec - without.q_spec).abs())
        })
        .collect()
}

/// ReliCapGrid's CO_UC7 and CO_UC8: without its converter or its line, a link transfers
/// nothing at either end, but for the idle loss of a converter left connected (1e-5 MW).
#[test]
fn an_hvdc_link_transfers_nothing_without_a_converter_or_its_line() {
    let Some(root) = instance() else { return };
    let paths = files(&root, &["DC-Espheim-Svedala"]);
    let mut intact = decode(&paths);
    let carried = link_injection(&intact)
        .iter()
        .filter(|&&p| p > 0.05)
        .count();
    assert_eq!(carried, 2, "10 MW at both ends of the intact link");
    for equipment in [
        "_038ce404-30dc-4289-b9db-7076cb870b8e", // CsConverter DC-3P-Convert-1
        "_be09fc02-de3f-49e4-aa84-94803bcc5d76", // DCLineSegment DC-Line-EH_SD1
    ] {
        let original = intact.entries[equipment].clone();
        trip(&mut intact, equipment);
        assert!(
            link_injection(&intact).iter().all(|&p| p < 1e-6),
            "{equipment} out"
        );
        intact.entries.insert(equipment.to_string(), original);
    }
}

/// The CGM as ReliCapGrid's CI assembles it: every IGM's EQ and updated SSH, Jotunheim's
/// topology and state, the boundary and the common data.
fn cgm(root: &Path) -> Vec<PathBuf> {
    let listed = |dir: PathBuf, keep: &dyn Fn(&str) -> bool| -> Vec<PathBuf> {
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("directory")
            .map(|f| f.expect("directory entry").path())
            .filter(|p| keep(p.file_name().and_then(|n| n.to_str()).unwrap_or("")))
            .collect();
        found.sort();
        found
    };
    let mut paths = listed(root.join("Jotunheim/ChangeSet/cimxml"), &|_| true);
    paths.extend(listed(root.join("Jotunheim/Grid/cimxml"), &|n| {
        n.contains("_SV_") || n.contains("_TP_")
    }));
    for area in [
        "Belgovia",
        "Britheim",
        "DC-Espheim-Svedala",
        "DC-Nordheim-Galia",
        "Espheim",
        "Galia",
        "Nordheim",
        "Portheim",
        "Svedala",
    ] {
        paths.extend(listed(root.join(area).join("Grid/cimxml"), &|n| {
            n.contains("_EQ_")
        }));
    }
    paths.extend(listed(root.join("boundaryData/Grid/cimxml"), &|_| true));
    paths.extend(listed(root.join("commonData/Grid/cimxml"), &|_| true));
    paths
}

fn co_files(root: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for area in std::fs::read_dir(root).expect("instances") {
        let dir = area
            .expect("directory entry")
            .path()
            .join("NetworkCode/cimxml");
        if let Ok(files) = std::fs::read_dir(dir) {
            paths.extend(
                files
                    .map(|f| f.expect("directory entry").path())
                    .filter(|p| p.to_string_lossy().contains("_CO")),
            );
        }
    }
    paths.sort();
    paths
}

/// ReliCapGrid's whole security analysis, by who takes the imbalance. BO-TR2_3, behind the CGM's
/// one reference, is what that choice moves most.
#[test]
fn security_analysis_of_the_cgm() {
    use gridoxide::security_analysis::{
        ImbalanceSharing, SecurityOptions, read_contingencies, run,
    };

    let Some(root) = instance() else { return };
    let mut grid = decode(&cgm(&root));
    let contingencies = decode(&co_files(&root));
    assert_eq!(
        read_contingencies(&contingencies)
            .expect("all outOfService")
            .len(),
        325
    );
    for (sharing, solved, stalled, base, after) in [
        (ImbalanceSharing::Capability, 304, 21, 13, 4003),
        (ImbalanceSharing::Reference, 312, 13, 17, 5230),
        (ImbalanceSharing::Equal, 287, 38, 21, 5970),
    ] {
        let options = SecurityOptions {
            sharing,
            ..Default::default()
        };
        let analysis = run(&mut grid, &contingencies, &options).expect("the base case solves");
        let in_base = analysis
            .violations
            .iter()
            .filter(|v| v.contingency.is_none())
            .count();
        assert_eq!(analysis.solved, solved, "{sharing:?}");
        assert_eq!(analysis.stalled.len(), stalled, "{sharing:?}");
        assert_eq!(in_base, base, "{sharing:?}");
        assert_eq!(analysis.violations.len() - in_base, after, "{sharing:?}");
        if sharing == ImbalanceSharing::Capability {
            assert!(analysis.past_limits.is_empty());
        }
    }
}
