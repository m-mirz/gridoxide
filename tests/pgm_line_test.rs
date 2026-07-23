mod common;

use std::path::PathBuf;
use gridoxide::pgm::{node_id_to_idx, pgm_to_buses_and_branches};
use gridoxide::network::build_ybus;
use gridoxide::run_power_flow_analysis_from_ybus;

#[test]
fn test_pgm_line_power_flow() {
    // Adopted from Power Grid Model tests/data/power_flow/components/symmetric/line
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgm/powerflow/symmetric/line");

    let input = common::load_pgm_input(&base.join("input.json"));
    let expected = common::load_output(&base.join("sym_output.json"));

    let id_to_idx = node_id_to_idx(&input);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n = buses.len();
    let ybus = build_ybus(n, &lines, &transformers);
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let tol = 1e-5;
    for node_out in &expected.data.node {
        common::assert_sym_node(&result, &id_to_idx, node_out, tol);
    }
}
