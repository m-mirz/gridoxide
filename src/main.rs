use std::f64::consts::PI;

mod types;
mod network;
mod solver;

use types::{Bus, Line, BusType};
use network::build_ybus;
use solver::newton_raphson;

fn main() {
    // Example 3-bus system: Bus 0 slack, Bus 1 PV, Bus 2 PQ
    // Values are illustrative, not from a standard case
    let mut buses = vec![
        Bus {
            idx: 0,
            bus_type: BusType::Slack,
            voltage_mag: 1.06,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: -999.0,
            q_max: 999.0,
        },
        Bus {
            idx: 1,
            bus_type: BusType::PV,
            voltage_mag: 1.04,
            voltage_ang: 0.0,
            p_spec: 0.5, // generation - load
            q_spec: 0.0, // for PV we use P specified and Vm specified
            q_min: -0.5,
            q_max: 0.5,
        },
        Bus {
            idx: 2,
            bus_type: BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: -0.6, // load of 0.6 p.u.
            q_spec: -0.25,
            q_min: -999.0,
            q_max: 999.0,
        },
    ];

    let lines = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.03 },
        Line { from: 0, to: 2, r: 0.08, x: 0.24, b_shunt: 0.025 },
        Line { from: 1, to: 2, r: 0.06, x: 0.18, b_shunt: 0.02 },
    ];

    let ybus = build_ybus(buses.len(), &lines);

    newton_raphson(&mut buses, &ybus, 1e-6, 20);

    println!("Final voltages:");
    for b in buses.iter() {
        println!(
            "Bus {}: |V| = {:.6}, angle = {:.6} deg",
            b.idx,
            b.voltage_mag,
            b.voltage_ang * 180.0 / PI
        );
    }
}