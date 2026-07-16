mod common;

use std::path::PathBuf;
use gridoxide::pgm::{node_id_to_idx, pgm_to_buses_and_branches, pgm_to_3ph_network};
use gridoxide::network::{build_ybus, build_ybus_3ph};
use gridoxide::run_power_flow_analysis_from_ybus;

fn data_dir(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgm/powerflow").join(rel)
}

#[test]
fn test_pgm_source_sym() {
    let base = data_dir("symmetric/source");
    let input = common::load_pgm_input(&base.join("input.json"));
    let expected = common::load_output(&base.join("sym_output.json"));

    let id_to_idx = node_id_to_idx(&input);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n = buses.len();
    let ybus = build_ybus(n, &lines, &transformers);
    let result = run_power_flow_analysis_from_ybus(buses, ybus);

    let tol = 1e-5;
    for node_out in &expected.data.node {
        common::assert_sym_node(&result, &id_to_idx, node_out, tol);
    }
}

#[test]
fn test_pgm_source_asym() {
    let base = data_dir("asymmetric/source");
    let input = common::load_pgm_input(&base.join("input.json"));
    let expected = common::load_output(&base.join("asym_output.json"));

    let (buses, lines, id_to_idx) = pgm_to_3ph_network(input, 1e6, 50.0);
    let n_total = buses.len() / 3;
    let ybus = build_ybus_3ph(n_total, &lines);
    let result = run_power_flow_analysis_from_ybus(buses, ybus);

    let tol = 1e-5;
    for node_out in &expected.data.node {
        common::assert_asym_node(&result, &id_to_idx, node_out, tol);
    }
}
