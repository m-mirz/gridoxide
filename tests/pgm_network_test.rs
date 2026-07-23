mod common;

use std::path::PathBuf;
use gridoxide::pgm::{
    node_id_to_idx, pgm_shunts_1ph, pgm_shunts_3ph, pgm_to_buses_and_branches, pgm_to_3ph_network,
    pgm_transformers_3ph, PgmNodeAsymOutput, PgmNodeOutput,
};
use gridoxide::network::{build_ybus, build_ybus_3ph, stamp_shunts, stamp_shunts_3ph, stamp_transformers_3ph};
use gridoxide::run_power_flow_analysis_from_ybus;

fn data_dir(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgm/powerflow").join(rel)
}

#[test]
fn test_pgm_network_symmetric_transmission_case() {
    // Adopted from Power Grid Model tests/data/power_flow/pandapower/networks/symmetric/transmission-case
    let base = data_dir("symmetric/transmission-case");
    let input = common::load_pgm_input(&base.join("input.json"));
    let expected = common::load_output::<PgmNodeOutput>(&base.join("sym_output.json"));

    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let tol = 1e-5;
    for node_out in &expected.data.node {
        common::assert_sym_node(&result, &id_to_idx, node_out, tol);
    }
}

#[test]
fn test_pgm_network_symmetric_distribution_case() {
    // Adopted from Power Grid Model tests/data/power_flow/pandapower/networks/symmetric/distribution-case
    let base = data_dir("symmetric/distribution-case");
    let base_json = common::load_json(&base.join("input.json"));
    let update = common::load_json(&base.join("update_batch.json"));
    let expected = common::load_batch_output::<PgmNodeOutput>(&base.join("sym_output_batch.json"));

    let tol = 1e-5;
    for (scenario, expected_scenario) in update["data"].as_array().unwrap().iter().zip(&expected.data) {
        let input = common::apply_batch_scenario(&base_json, scenario);
        let id_to_idx = node_id_to_idx(&input);
        let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
        let n = buses.len();
        let ybus = build_ybus(n, &lines, &transformers);
        let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

        // Transformer result values aren't validated against pandapower in this
        // fixture (modelling differences, per its README) — only node voltages.
        for node_out in &expected_scenario.node {
            common::assert_sym_node(&result, &id_to_idx, node_out, tol);
        }
    }
}

#[test]
fn test_pgm_network_asymmetric_transmission_case() {
    // Adopted from Power Grid Model tests/data/power_flow/pandapower/networks/asymmetric/transmission-case
    let base = data_dir("asymmetric/transmission-case");
    let input = common::load_pgm_input(&base.join("input.json"));
    let expected = common::load_output::<PgmNodeAsymOutput>(&base.join("asym_output.json"));

    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_3ph(&input, &id_to_idx, 1e6);
    let transformers = pgm_transformers_3ph(&input, &id_to_idx, 1e6);
    let (buses, lines, id_to_idx) = pgm_to_3ph_network(input, 1e6, 50.0);
    let n_total = buses.len() / 3;
    let mut ybus = build_ybus_3ph(n_total, &lines);
    stamp_shunts_3ph(&mut ybus, &shunts);
    stamp_transformers_3ph(&mut ybus, &transformers);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let tol = 1e-5;
    // Shunt results aren't present in this fixture's expected output even
    // though shunts are stamped into the Y-bus for correct node voltages.
    for node_out in &expected.data.node {
        common::assert_asym_node(&result, &id_to_idx, node_out, tol);
    }
}

#[test]
fn test_pgm_network_asymmetric_distribution_case() {
    // Adopted from Power Grid Model tests/data/power_flow/pandapower/networks/asymmetric/distribution-case
    let base = data_dir("asymmetric/distribution-case");
    let base_json = common::load_json(&base.join("input.json"));
    let update = common::load_json(&base.join("update_batch.json"));
    let expected = common::load_batch_output::<PgmNodeAsymOutput>(&base.join("asym_output_batch.json"));

    // Looser tolerance per this fixture's params.json (rtol=atol=1e-3).
    let tol = 1e-3;
    for (scenario, expected_scenario) in update["data"].as_array().unwrap().iter().zip(&expected.data) {
        let input = common::apply_batch_scenario(&base_json, scenario);
        let id_to_idx = node_id_to_idx(&input);
        let transformers = pgm_transformers_3ph(&input, &id_to_idx, 1e6);
        let (buses, lines, id_to_idx) = pgm_to_3ph_network(input, 1e6, 50.0);
        let n_total = buses.len() / 3;
        let mut ybus = build_ybus_3ph(n_total, &lines);
        stamp_transformers_3ph(&mut ybus, &transformers);
        let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

        // Transformer result values aren't present in this fixture's expected
        // output (modelling differences) — only node voltages are validated.
        for node_out in &expected_scenario.node {
            common::assert_asym_node(&result, &id_to_idx, node_out, tol);
        }
    }
}
