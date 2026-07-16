mod common;

use std::path::PathBuf;
use gridoxide::network::{build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches, PgmNodeOutput};
use gridoxide::solver::{newton_raphson_with_backend, JacobianBackend};
use gridoxide::types::Bus;

fn data_dir(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgm/powerflow/symmetric").join(rel)
}

/// Mirrors `run_power_flow_analysis_from_ybus`, but runs the experimental
/// `JacobianBackend::Block` path instead of the default `Scalar` one — the
/// linear initial guess is backend-agnostic (touches only the Y-bus, never
/// the Jacobian) so it's reused unchanged.
fn solve_with_block_backend(mut buses: Vec<Bus>, ybus: gridoxide::network::YBus) -> Vec<Bus> {
    let ybus = ybus.finish();
    linear_initial_guess(&mut buses, &ybus);
    newton_raphson_with_backend(&mut buses, &ybus, 1e-6, 20, JacobianBackend::Block);
    buses
}

fn assert_matches_pgm(base: &PathBuf, tol: f64) {
    let input = common::load_pgm_input(&base.join("input.json"));
    let expected = common::load_output::<PgmNodeOutput>(&base.join("sym_output.json"));

    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n = buses.len();
    let mut ybus = build_ybus(n, &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let result = solve_with_block_backend(buses, ybus);

    for node_out in &expected.data.node {
        common::assert_sym_node(&result, &id_to_idx, node_out, tol);
    }
}

#[test]
fn block_backend_matches_pgm_basic_node() {
    assert_matches_pgm(&data_dir("basic-node"), 1e-5);
}

#[test]
fn block_backend_matches_pgm_node() {
    assert_matches_pgm(&data_dir("node"), 1e-5);
}

#[test]
fn block_backend_matches_pgm_line() {
    assert_matches_pgm(&data_dir("line"), 1e-5);
}

#[test]
fn block_backend_matches_pgm_shunt() {
    assert_matches_pgm(&data_dir("shunt"), 1e-5);
}

#[test]
fn block_backend_matches_pgm_sym_load() {
    assert_matches_pgm(&data_dir("sym_load"), 1e-5);
}

#[test]
fn block_backend_matches_pgm_sym_gen() {
    assert_matches_pgm(&data_dir("sym_gen"), 1e-5);
}

#[test]
fn block_backend_matches_pgm_source() {
    assert_matches_pgm(&data_dir("source"), 1e-5);
}

#[test]
fn block_backend_matches_pgm_transmission_case() {
    // The largest single-scenario symmetric fixture (11 nodes, meshed,
    // transformers + shunts + gens) — a more realistic topology than the
    // small component fixtures above.
    assert_matches_pgm(&data_dir("transmission-case"), 1e-5);
}

#[test]
fn block_backend_matches_pgm_transformer_batch() {
    let base = data_dir("transformer");
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
        let result = solve_with_block_backend(buses, ybus);

        for node_out in &expected_scenario.node {
            common::assert_sym_node(&result, &id_to_idx, node_out, tol);
        }
    }
}

#[test]
fn block_backend_matches_pgm_distribution_case_batch() {
    let base = data_dir("distribution-case");
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
        let result = solve_with_block_backend(buses, ybus);

        for node_out in &expected_scenario.node {
            common::assert_sym_node(&result, &id_to_idx, node_out, tol);
        }
    }
}
