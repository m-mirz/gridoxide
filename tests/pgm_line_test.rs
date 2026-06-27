use std::fs;
use std::path::PathBuf;
use gridoxide::pgm::{PgmInput, PgmOutput, node_id_to_idx, pgm_to_buses_and_branches};
use gridoxide::network::build_ybus;
use gridoxide::run_power_flow_analysis_from_ybus;

#[test]
fn test_pgm_line_power_flow() {
    // Adopted from Power Grid Model tests/data/power_flow/components/symmetric/line
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgm/powerflow/symmetric/line");

    let input: PgmInput = serde_json::from_str(
        &fs::read_to_string(base.join("input.json")).unwrap()
    ).unwrap();
    let expected: PgmOutput = serde_json::from_str(
        &fs::read_to_string(base.join("sym_output.json")).unwrap()
    ).unwrap();

    let id_to_idx = node_id_to_idx(&input);
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n = buses.len();
    let ybus = build_ybus(n, &lines, &transformers);
    let result = run_power_flow_analysis_from_ybus(buses, ybus);

    let tol = 1e-5;
    for node_out in &expected.data.node {
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
}
