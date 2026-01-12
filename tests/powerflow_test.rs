use std::fs;
use std::path::PathBuf;
use gridoxide::json::NetworkData;
use gridoxide::run_power_flow_analysis;

#[test]
fn test_power_flow_analysis() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let network_json = fs::read_to_string(path).expect("Unable to read network.json");
    let network_data: NetworkData = serde_json::from_str(&network_json).expect("Unable to parse network.json");

    let buses = run_power_flow_analysis(network_data);

    // Expected values would be derived from a known correct power flow solution.
    // For this test, we are doing a snapshot/regression test.
    // These values are from a previous run of the program.
    let expected_voltages = vec![
        (1.06, 0.0),
        (1.04, 0.014349),
        (1.003358, -0.043141)
    ];

    assert_eq!(buses.len(), expected_voltages.len());

    for i in 0..buses.len() {
        assert_eq!(buses[i].idx, i);
        let (expected_mag, expected_ang_rad) = expected_voltages[i];
        assert!((buses[i].voltage_mag - expected_mag).abs() < 1e-5);
        assert!((buses[i].voltage_ang - expected_ang_rad).abs() < 1e-5);
    }
}
