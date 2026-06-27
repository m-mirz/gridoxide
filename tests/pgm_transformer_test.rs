use std::fs;
use std::path::PathBuf;
use serde::Deserialize;
use gridoxide::pgm::{PgmInput, node_id_to_idx, pgm_to_buses_and_branches};
use gridoxide::network::build_ybus;
use gridoxide::run_power_flow_analysis_from_ybus;

#[derive(Deserialize)]
struct BatchOutput {
    data: Vec<BatchEntry>,
}

#[derive(Deserialize)]
struct BatchEntry {
    node: Vec<NodeOut>,
}

#[derive(Deserialize)]
struct NodeOut {
    id: u64,
    u_pu: f64,
    u: f64,
    u_angle: f64,
}

#[test]
fn test_pgm_transformer_power_flow() {
    // Adopted from Power Grid Model tests/data/power_flow/pandapower/components/symmetric/transformer
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/symmetric/transformer");

    let input: PgmInput = serde_json::from_str(
        &fs::read_to_string(base.join("input.json")).unwrap()
    ).unwrap();
    let expected: BatchOutput = serde_json::from_str(
        &fs::read_to_string(base.join("sym_output_batch.json")).unwrap()
    ).unwrap();

    let id_to_idx = node_id_to_idx(&input);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n = buses.len();
    let ybus = build_ybus(n, &lines, &transformers);
    let result = run_power_flow_analysis_from_ybus(buses, ybus);

    // Batch 0 corresponds to tap_pos=−5, which matches the base input.json.
    let tol = 1e-4;
    for node_out in &expected.data[0].node {
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
        let u_phys = bus.voltage_mag * bus.u_rated;
        assert!(
            (u_phys - node_out.u).abs() < tol * bus.u_rated,
            "node {}: u_phys = {:.4}, expected u = {:.4}",
            node_out.id, u_phys, node_out.u
        );
    }
}
