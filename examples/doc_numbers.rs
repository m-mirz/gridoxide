//! Prints the DC quantities the book's remedial-action pages derive by hand.
//!
//! `gridoxide security` and `gridoxide rao` already print flows and margins.
//! What they do not print is the *intermediate* arithmetic those pages check —
//! each branch's per-unit susceptance and the phase-shift distribution factors
//! a shifter produces — so this exists to keep those numbers reproducible
//! rather than asserted.
//!
//! ```bash
//! cargo run --release --features ucte,rao --example doc_numbers \
//!     tests/data/ucte/3nodes_pst.uct
//! ```
//!
//! See `docs/examples/README.md`.

use gridoxide::linear::btheta::{dc_branches, dc_power_flow};
use gridoxide::linear::sensitivity::DcSensitivity;
use gridoxide::rao::linear::phase_shift_sensitivity;
use gridoxide::ucte;

/// Radians per degree, the unit every phase-shifter figure in the book uses.
const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.0;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/data/ucte/3nodes_pst.uct".to_string());
    let net = ucte::read(&path).expect("network");
    let name = |i: usize| net.branch_ids.get(i).map(String::as_str).unwrap_or("?");

    println!("{path}  ({} MVA base)\n", net.base_mva);

    println!("--- buses ---");
    for (i, bus) in net.buses.iter().enumerate() {
        let code = net.node_codes.get(i).map(String::as_str).unwrap_or("?");
        println!(
            "{i}: {code:>10}  {:?}  p_spec={:+8.4} pu  u_rated={} V",
            bus.bus_type, bus.p_spec, bus.u_rated
        );
    }

    let options = Default::default();
    let branches = dc_branches(&net.lines, &net.transformers, options);
    println!("\n--- branches ---");
    for br in &branches {
        println!(
            "{:>22}  from={} to={}  b={:>12.4} pu  shift={:+.6} rad",
            name(br.index),
            br.from,
            br.to,
            br.b,
            br.shift
        );
    }

    let mut buses = net.buses.clone();
    let solution = dc_power_flow(&mut buses, &net.lines, &net.transformers, options);
    println!("\n--- DC flows (MW, positive into the `from` terminal) ---");
    for br in &branches {
        println!("{:>22}  {:+10.3}", name(br.index), solution.branch_p[br.index] * net.base_mva);
    }

    let sensitivity =
        DcSensitivity::new(&net.buses, &branches, net.n_branches()).expect("sensitivity");
    for t in 0..net.transformers.len() {
        let branch = net.transformer_branch(t);
        let Some(psdf) = phase_shift_sensitivity(&sensitivity, &branches, branch) else {
            continue;
        };
        println!("\n--- phase-shift sensitivity of transformer {t} (branch {branch}), MW/deg ---");
        for br in &branches {
            let mw_per_degree = psdf[br.index] * net.base_mva * RAD_PER_DEG;
            println!("{:>22}  {:+10.4}", name(br.index), mw_per_degree);
        }
    }
}
