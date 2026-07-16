//! Ad-hoc runtime benchmark against a PGM-format JSON network, for comparing
//! against power-grid-model on the same input. Not part of the test suite.
//!
//! Usage: cargo run --release --example bench_network -- <path-to-input.json>

use std::env;
use std::fs;
use std::time::Instant;

use gridoxide::network::{build_ybus, linear_initial_guess};
use gridoxide::pgm::pgm_to_buses_and_branches;
use gridoxide::solver::newton_raphson;

fn main() {
    let path = env::args().nth(1).expect("usage: bench_network <input.json>");
    let raw = fs::read_to_string(&path).expect("read input file");

    let t_parse0 = Instant::now();
    let input = serde_json::from_str(&raw).expect("parse PGM input JSON");
    let (mut buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n = buses.len();
    let ybus = build_ybus(n, &lines, &transformers);
    let t_parse = t_parse0.elapsed();

    println!("nodes={} lines={} transformers={}", n, lines.len(), transformers.len());
    println!("parse + build model + Y-bus: {:.3} ms", t_parse.as_secs_f64() * 1e3);

    let t_finish0 = Instant::now();
    let ybus = ybus.finish();
    let t_finish = t_finish0.elapsed();
    println!("Y-bus finish (COO -> sparse): {:.3} ms", t_finish.as_secs_f64() * 1e3);

    let t_guess0 = Instant::now();
    linear_initial_guess(&mut buses, &ybus);
    let t_guess = t_guess0.elapsed();
    println!("linear_initial_guess: {:.3} ms", t_guess.as_secs_f64() * 1e3);

    let t_nr0 = Instant::now();
    newton_raphson(&mut buses, &ybus, 1e-6, 20);
    let t_nr = t_nr0.elapsed();
    println!("newton_raphson: {:.3} ms", t_nr.as_secs_f64() * 1e3);

    let vmin = buses.iter().map(|b| b.voltage_mag).fold(f64::INFINITY, f64::min);
    let vmax = buses.iter().map(|b| b.voltage_mag).fold(f64::NEG_INFINITY, f64::max);
    println!("voltage_mag min/max = {:.6} / {:.6}", vmin, vmax);
    println!("total (guess + NR): {:.3} ms", (t_guess + t_nr).as_secs_f64() * 1e3);
}
