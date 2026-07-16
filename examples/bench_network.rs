//! Ad-hoc runtime benchmark against a PGM-format JSON network, for comparing
//! against power-grid-model on the same input. Not part of the test suite.
//!
//! Usage: cargo run --release --example bench_network -- <path-to-input.json> [repeat-count] [backend]
//!
//! `repeat-count` (default 1) re-runs the linear initial guess + full
//! Newton-Raphson solve that many times from a fresh clone of the
//! post-parse (flat-start) buses each time, so `perf record` gets enough
//! samples to profile — a single solve at realistic network sizes is only
//! tens of milliseconds, too short to sample meaningfully.
//!
//! `backend` (default "scalar") selects `scalar` (the default `faer`-backed
//! path) or `block` (the experimental block-per-bus path, symmetric only)
//! for a head-to-head comparison on the same network.

use std::env;
use std::fs;
use std::time::Instant;

use gridoxide::network::{build_ybus, linear_initial_guess};
use gridoxide::pgm::pgm_to_buses_and_branches;
use gridoxide::solver::{newton_raphson_with_backend, JacobianBackend};

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: bench_network <input.json> [repeat-count] [backend]");
    let repeat: usize = args.next().map(|s| s.parse().expect("repeat-count must be an integer")).unwrap_or(1);
    let backend_arg = args.next().unwrap_or_else(|| "scalar".to_string());
    let backend = match backend_arg.as_str() {
        "scalar" => JacobianBackend::Scalar,
        "block" => JacobianBackend::Block,
        other => panic!("unknown backend '{other}', expected 'scalar' or 'block'"),
    };
    println!("backend={backend_arg}");
    let raw = fs::read_to_string(&path).expect("read input file");

    let t_parse0 = Instant::now();
    let input = serde_json::from_str(&raw).expect("parse PGM input JSON");
    let (buses_template, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n = buses_template.len();
    let ybus = build_ybus(n, &lines, &transformers);
    let t_parse = t_parse0.elapsed();

    println!("nodes={} lines={} transformers={}", n, lines.len(), transformers.len());
    println!("parse + build model + Y-bus: {:.3} ms", t_parse.as_secs_f64() * 1e3);

    let t_finish0 = Instant::now();
    let ybus = ybus.finish();
    let t_finish = t_finish0.elapsed();
    println!("Y-bus finish (COO -> sparse): {:.3} ms", t_finish.as_secs_f64() * 1e3);

    let mut total_guess = std::time::Duration::ZERO;
    let mut total_nr = std::time::Duration::ZERO;
    let mut buses = buses_template.clone();
    for _ in 0..repeat {
        buses = buses_template.clone();

        let t_guess0 = Instant::now();
        linear_initial_guess(&mut buses, &ybus);
        total_guess += t_guess0.elapsed();

        let t_nr0 = Instant::now();
        newton_raphson_with_backend(&mut buses, &ybus, 1e-6, 20, backend);
        total_nr += t_nr0.elapsed();
    }

    println!("linear_initial_guess: {:.3} ms total, {:.3} ms/run over {} run(s)",
        total_guess.as_secs_f64() * 1e3, total_guess.as_secs_f64() * 1e3 / repeat as f64, repeat);
    println!("newton_raphson: {:.3} ms total, {:.3} ms/run over {} run(s)",
        total_nr.as_secs_f64() * 1e3, total_nr.as_secs_f64() * 1e3 / repeat as f64, repeat);

    let vmin = buses.iter().map(|b| b.voltage_mag).fold(f64::INFINITY, f64::min);
    let vmax = buses.iter().map(|b| b.voltage_mag).fold(f64::NEG_INFINITY, f64::max);
    println!("voltage_mag min/max = {:.6} / {:.6}", vmin, vmax);
    println!("total (guess + NR): {:.3} ms", ((total_guess + total_nr).as_secs_f64() * 1e3) / repeat as f64);
}
