use std::fs;
use std::path::PathBuf;
use gridoxide::pgm::{PgmInput, PgmAsymOutput, pgm_to_3ph_network};
use gridoxide::network::build_ybus_3ph;
use gridoxide::run_power_flow_analysis_from_ybus;

#[test]
fn test_pgm_asym_line_power_flow() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/asymmetric/line");

    let input: PgmInput = serde_json::from_str(
        &fs::read_to_string(base.join("input.json")).unwrap()
    ).unwrap();
    let expected: PgmAsymOutput = serde_json::from_str(
        &fs::read_to_string(base.join("asym_output.json")).unwrap()
    ).unwrap();

    let (buses, lines, id_to_idx) = pgm_to_3ph_network(input, 1e6, 50.0);
    let n_total = buses.len() / 3;
    let ybus = build_ybus_3ph(n_total, &lines);
    let result = run_power_flow_analysis_from_ybus(buses, ybus);

    let tol = 1e-5;
    for node_out in &expected.data.node {
        let phys_idx = id_to_idx[&node_out.id];
        for ph in 0..3 {
            let bus = &result[3 * phys_idx + ph];
            assert!(
                (bus.voltage_mag - node_out.u_pu[ph]).abs() < tol,
                "node {} phase {}: voltage_mag = {:.8}, expected u_pu = {:.8}",
                node_out.id, ph, bus.voltage_mag, node_out.u_pu[ph]
            );
            assert!(
                (bus.voltage_ang - node_out.u_angle[ph]).abs() < tol,
                "node {} phase {}: voltage_ang = {:.8}, expected u_angle = {:.8}",
                node_out.id, ph, bus.voltage_ang, node_out.u_angle[ph]
            );
        }
    }
}
