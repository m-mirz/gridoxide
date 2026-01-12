use std::fs;
use std::path::PathBuf;
use gridoxide::json::NetworkData;
use gridoxide::run_power_flow_analysis;

fn main() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let network_json = fs::read_to_string(path).expect("Unable to read network.json");
    let network_data: NetworkData = serde_json::from_str(&network_json).expect("Unable to parse network.json");

    let buses = run_power_flow_analysis(network_data);

    println!("Final voltages:");
    for b in buses.iter() {
        println!(
            "Bus {}: |V| = {:.6}, angle = {:.6} deg",
            b.idx,
            b.voltage_mag,
            b.voltage_ang.to_degrees()
        );
    }
}
