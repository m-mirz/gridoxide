//! Import-time and node-count benchmark for the node-breaker topology reader.
//! Not part of the test suite.
//!
//! Usage: cargo run --release --features cgmes --example bench_node_breaker -- <profile.xml>...
//!
//! Reports how long `cgmes::cgmes_node_breaker_topology` takes against the
//! decode that precedes it, and what each retention policy costs in buses —
//! which is the number that decides whether node-breaker support is usable by
//! default or has to stay opt-in.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use gridoxide::cgmes::{cgmes_node_breaker_topology, load_profiles};
use gridoxide::topology::{bus_view, RetentionPolicy};

fn main() {
    let paths: Vec<PathBuf> = env::args().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        eprintln!("usage: bench_node_breaker <profile.xml>...");
        std::process::exit(2);
    }
    let refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();

    let start = Instant::now();
    let ds = load_profiles(&refs).expect("failed to decode CGMES profiles");
    let decode_ms = start.elapsed().as_secs_f64() * 1e3;

    let start = Instant::now();
    let nb = cgmes_node_breaker_topology(&ds).expect("node-breaker read failed");
    let read_ms = start.elapsed().as_secs_f64() * 1e3;

    println!(
        "{} node(s), {} switch(es) ({} conducting), {} busbar(s)",
        nb.topology.n_nodes,
        nb.topology.switches.len(),
        nb.topology.conducting_count(),
        nb.topology.busbars.len()
    );
    println!("  decode      {decode_ms:8.2} ms");
    println!("  topology    {read_ms:8.2} ms  ({:.1}% of decode)", 100.0 * read_ms / decode_ms);

    for (label, policy) in [
        ("MergeAll", RetentionPolicy::MergeAll),
        ("RetainAdjacentToBusbar", RetentionPolicy::RetainAdjacentToBusbar),
        ("RetainAll", RetentionPolicy::RetainAll),
    ] {
        let start = Instant::now();
        let view = bus_view(&nb.topology, &policy);
        let ms = start.elapsed().as_secs_f64() * 1e3;
        // The AC unknown count the plan's §3.3 argues about: two per non-slack
        // bus, plus two per retained edge under a constrained formulation.
        let unknowns = 2 * view.n_buses() + 2 * view.retained().len();
        println!(
            "  {label:<24} {:>6} buses, {:>5} retained  -> ~{unknowns:>6} AC unknowns  \
             ({ms:.2} ms)",
            view.n_buses(),
            view.retained().len()
        );
    }
}
