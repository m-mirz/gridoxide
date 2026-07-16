use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde_json::Value;

use gridoxide::pgm::{PgmBatchOutput, PgmInput, PgmNodeAsymOutput, PgmNodeOutput, PgmOutput};
use gridoxide::types::Bus;

#[allow(dead_code)]
pub fn load_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[allow(dead_code)]
pub fn load_pgm_input(path: &Path) -> PgmInput {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[allow(dead_code)]
pub fn load_output<N: DeserializeOwned>(path: &Path) -> PgmOutput<N> {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[allow(dead_code)]
pub fn load_batch_output<N: DeserializeOwned>(path: &Path) -> PgmBatchOutput<N> {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

/// Applies one batch-update scenario (as found in `update_batch.json`'s `data[i]`)
/// onto a base PGM input document, overlaying the changed fields of each named
/// component entry (matched by `id`) before deserializing into `PgmInput`.
#[allow(dead_code)]
pub fn apply_batch_scenario(base: &Value, scenario: &Value) -> PgmInput {
    let mut merged = base.clone();
    let data = merged["data"].as_object_mut().unwrap();
    for (component, updates) in scenario.as_object().unwrap() {
        let base_list = data[component].as_array_mut().unwrap();
        for upd in updates.as_array().unwrap() {
            let upd_obj = upd.as_object().unwrap();
            let id = &upd_obj["id"];
            let entry = base_list
                .iter_mut()
                .find(|e| &e["id"] == id)
                .unwrap_or_else(|| panic!("no {component} entry with id {id} in base input"));
            let entry_obj = entry.as_object_mut().unwrap();
            for (k, v) in upd_obj {
                if k == "id" {
                    continue;
                }
                entry_obj.insert(k.clone(), v.clone());
            }
        }
    }
    serde_json::from_value(merged).unwrap()
}

#[allow(dead_code)]
pub fn assert_sym_node(
    result: &[Bus],
    id_to_idx: &HashMap<u64, usize>,
    node_out: &PgmNodeOutput,
    tol: f64,
) {
    let idx = id_to_idx[&node_out.id];
    let bus = &result[idx];
    assert!(
        (bus.voltage_mag - node_out.u_pu).abs() < tol,
        "node {}: voltage_mag = {:.6}, expected u_pu = {:.6}",
        node_out.id, bus.voltage_mag, node_out.u_pu
    );
    assert!(
        (bus.voltage_ang - node_out.u_angle).abs() < tol,
        "node {}: voltage_ang = {:.6}, expected u_angle = {:.6}",
        node_out.id, bus.voltage_ang, node_out.u_angle
    );
}

#[allow(dead_code)]
pub fn assert_asym_node(
    result: &[Bus],
    id_to_idx: &HashMap<u64, usize>,
    node_out: &PgmNodeAsymOutput,
    tol: f64,
) {
    let phys_idx = id_to_idx[&node_out.id];
    for ph in 0..3 {
        let bus = &result[3 * phys_idx + ph];
        assert!(
            (bus.voltage_mag - node_out.u_pu[ph]).abs() < tol,
            "node {} phase {}: voltage_mag = {:.8}, expected u_pu = {:.8}",
            node_out.id, ph, bus.voltage_mag, node_out.u_pu[ph]
        );
        assert!(
            (bus.voltage_ang - node_out.u_angle[ph]).abs() < tol,
            "node {} phase {}: voltage_ang = {:.8}, expected u_angle = {:.8}",
            node_out.id, ph, bus.voltage_ang, node_out.u_angle[ph]
        );
    }
}
