//! Where `SwitchTreatment::Regularize` stops working. Not part of the test
//! suite.
//!
//! Usage: cargo run --release --example switch_ceiling
//!
//! `src/cgmes.rs` records that stamping CGMES switches as large-admittance
//! branches was tried and the AC solve diverged on FullGrid with "20-odd" such
//! branches active. This turns that anecdote into a curve, on the friendliest
//! possible network — a single chain of ideal switches from a slack bus to one
//! small load — so whatever ceiling shows up here is an *upper* bound on what
//! real data will tolerate.

use gridoxide::switches::{conditioning_probe, switch_admittance, ProbeShape};
use gridoxide::topology::IDEAL_CONNECTION_Y;

fn main() {
    let counts: Vec<usize> = vec![
        1, 2, 5, 10, 20, 29, 50, 90, 200, 500, 1000, 1266, 1464,
    ];
    for (label, y) in [
        ("inductive (switches.rs)", switch_admittance()),
        ("capacitive (IDEAL_CONNECTION_Y)", IDEAL_CONNECTION_Y),
    ] {
    for shape in [ProbeShape::Chain, ProbeShape::Star, ProbeShape::Ring] {
        println!("\n=== {shape:?}  /  {label}");
        println!("{:>8}  {:>10}  {:>14}  {}", "switches", "iterations", "far-end |V|", "verdict");
        for (n, iters, vm) in conditioning_probe(&counts, shape, y, 1e-6, 30) {
            let verdict = match iters {
                Some(_) if (vm - 1.0).abs() < 0.05 => "ok",
                Some(_) => "converged, implausible |V|",
                None => "DIVERGED",
            };
            println!(
                "{n:>8}  {:>10}  {vm:>14.6}  {verdict}",
                iters.map(|i| i.to_string()).unwrap_or_else(|| "-".into())
            );
        }
    }
    }
}
