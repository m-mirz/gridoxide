mod common;

use std::path::PathBuf;
use gridoxide::pgm::{node_id_to_idx, pgm_to_buses_and_branches, PgmNodeOutput};
use gridoxide::network::build_ybus;
use gridoxide::run_power_flow_analysis_from_ybus;

#[test]
fn test_pgm_three_winding_transformer_batch() {
    // Adopted from Power Grid Model tests/data/power_flow/pandapower/components/symmetric/three_winding_transformer
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/symmetric/three_winding_transformer");

    let base_json = common::load_json(&base.join("input.json"));
    let update = common::load_json(&base.join("update_batch.json"));
    let expected = common::load_batch_output::<PgmNodeOutput>(&base.join("sym_output_batch.json"));

    let tol = 1e-4;
    for (scenario, expected_scenario) in update["data"].as_array().unwrap().iter().zip(&expected.data) {
        let input = common::apply_batch_scenario(&base_json, scenario);
        let id_to_idx = node_id_to_idx(&input);
        let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
        let n = buses.len();
        let ybus = build_ybus(n, &lines, &transformers);
        let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

        for node_out in &expected_scenario.node {
            common::assert_sym_node(&result, &id_to_idx, node_out, tol);
        }
    }
}
