mod common;

use std::path::PathBuf;
use gridoxide::pgm::{node_id_to_idx, pgm_transformers_3ph, pgm_to_3ph_network, PgmNodeAsymOutput};
use gridoxide::network::{build_ybus_3ph, stamp_transformers_3ph};
use gridoxide::run_power_flow_analysis_from_ybus;

#[test]
fn test_pgm_asym_transformer_batch() {
    // Adopted from Power Grid Model tests/data/power_flow/pandapower/components/asymmetric/transformer
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/asymmetric/transformer");

    let base_json = common::load_json(&base.join("input.json"));
    let update = common::load_json(&base.join("update_batch.json"));
    let expected = common::load_batch_output::<PgmNodeAsymOutput>(&base.join("asym_output_batch.json"));

    // Relaxed tolerance per the reference fixture's params.json: pandapower's
    // "T" transformer model vs power-grid-model's "pi" model.
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

        for node_out in &expected_scenario.node {
            common::assert_asym_node(&result, &id_to_idx, node_out, tol);
        }
    }
}
