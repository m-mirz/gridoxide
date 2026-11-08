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
