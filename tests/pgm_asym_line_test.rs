mod common;

use std::path::PathBuf;
use gridoxide::pgm::pgm_to_3ph_network;
use gridoxide::network::build_ybus_3ph;
use gridoxide::run_power_flow_analysis_from_ybus;

#[test]
fn test_pgm_asym_line_power_flow() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/asymmetric/line");

    let input = common::load_pgm_input(&base.join("input.json"));
    let expected = common::load_output(&base.join("asym_output.json"));

    let (buses, lines, id_to_idx) = pgm_to_3ph_network(input, 1e6, 50.0);
    let n_total = buses.len() / 3;
    let ybus = build_ybus_3ph(n_total, &lines);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let tol = 1e-5;
    for node_out in &expected.data.node {
        common::assert_asym_node(&result, &id_to_idx, node_out, tol);
    }
}
