mod common;

use gridoxide::network::{build_ybus, build_ybus_3ph};
use gridoxide::pgm::{
    SourceModel, node_id_to_idx, pgm_to_3ph_network, pgm_to_buses_and_branches, pgm_to_network_with,
};
use gridoxide::run_power_flow_analysis_from_ybus;
use std::path::PathBuf;

fn data_dir(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow")
        .join(rel)
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
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

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
    let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

    let tol = 1e-5;
    for node_out in &expected.data.node {
        common::assert_asym_node(&result, &id_to_idx, node_out, tol);
    }
}

/// An ideal source holds its node at `u_ref` with no bus of its own, and is
/// what power-grid-model's source tends to as its short-circuit power grows.
#[test]
fn an_ideal_source_is_the_limit_of_a_stiff_one() {
    let base = data_dir("symmetric/source");
    let input = common::load_pgm_input(&base.join("input.json"));
    let id_to_idx = node_id_to_idx(&input);
    let src = &input.data.source[0];
    let (source_node, u_ref) = (id_to_idx[&src.node], src.u_ref);

    let ideal = pgm_to_network_with(input.clone(), 1e6, 50.0, SourceModel::Ideal);
    assert_eq!(ideal.buses.len(), input.data.node.len(), "no virtual bus");
    assert!(ideal.source_branch_idx.is_empty(), "no source branch");
    let ybus = build_ybus(ideal.buses.len(), &ideal.lines, &ideal.transformers);
    let ideal = run_power_flow_analysis_from_ybus(ideal.buses, ybus).buses;
    assert_eq!(ideal[source_node].voltage_mag, u_ref);
    assert_eq!(ideal[source_node].voltage_ang, 0.0);

    let mut stiff = input;
    for s in &mut stiff.data.source {
        s.sk = 1e20;
    }
    let (buses, lines, transformers) = pgm_to_buses_and_branches(stiff, 1e6, 50.0);
    let ybus = build_ybus(buses.len(), &lines, &transformers);
    let stiff = run_power_flow_analysis_from_ybus(buses, ybus).buses;
    for &idx in id_to_idx.values() {
        assert!(
            (ideal[idx].voltage_mag - stiff[idx].voltage_mag).abs() < 1e-9,
            "bus {idx}: |V| {} ideal against {} stiff",
            ideal[idx].voltage_mag,
            stiff[idx].voltage_mag
        );
        assert!(
            (ideal[idx].voltage_ang - stiff[idx].voltage_ang).abs() < 1e-9,
            "bus {idx}: angle {} ideal against {} stiff",
            ideal[idx].voltage_ang,
            stiff[idx].voltage_ang
        );
    }
}
