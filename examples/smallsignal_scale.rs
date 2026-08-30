//! How the sparse small-signal analysis scales, and what it costs.
//!
//! The dense method (`smallsignal::analyze`) refuses past two thousand states,
//! because forming `A` is `O(n²·n_net)` and decomposing it `O(n³)`. This is the
//! method that does not, measured on the same ring `dynamics_scale` uses.
//!
//! What to watch:
//!
//! - **the factorization**, paid once per shift, which should grow roughly
//!   linearly the way every other sparse factorization in this crate does; and
//! - **the iteration**, whose cost is a number of triangular solves against
//!   that factorization — so the interesting number is not the seconds but
//!   *how many applications* the answer took, which is what a clustered
//!   spectrum drives up.
//!
//! Run with
//! `cargo run --release --features dynamics --example smallsignal_scale`.

#[path = "../tests/dynamics_ring/mod.rs"]
mod ring;

use std::time::Instant;

use gridoxide::dynamics::smallsignal::{self, Method, SmallSignalOptions};

fn main() {
    println!(
        "{:>7}  {:>7}  {:>8}  {:>9}  {:>9}  {:>10}  {:>9}  {:>6}",
        "buses", "units", "states", "build", "analyze", "worst res", "restarts", "part."
    );

    for n_bus in [64usize, 256, 1024, 4096] {
        let document = ring::ring(n_bus, 0.4);
        let started = Instant::now();
        let (system, _) = match document.build() {
            Ok(pair) => pair,
            Err(e) => {
                println!("{n_bus:>7}  build failed: {e}");
                continue;
            }
        };
        let build_time = started.elapsed();
        let n_x = system.n_states();

        let opts = SmallSignalOptions::near_frequency(1.0, 0.05).count(8);
        let started = Instant::now();
        let result = match smallsignal::analyze_near(&system, &opts) {
            Ok(r) => r,
            Err(e) => {
                println!("{n_bus:>7}  analysis failed: {e}");
                continue;
            }
        };
        let analyze_time = started.elapsed();

        // The example is also the gate for the size the test suite cannot
        // afford to build: `dynamics_scale`'s own assertions have the same
        // role. A converged answer at twenty thousand states is the claim this
        // whole method exists to make.
        let worst = result.modes.iter().fold(0.0f64, |m, mode| m.max(mode.residual));
        assert!(result.converged, "{n_bus} buses: unconverged, worst residual {worst:e}");
        assert!(worst < 1e-9, "{n_bus} buses: worst residual {worst:e}");
        assert!(
            result.modes.iter().all(|m| !m.participation.is_empty()),
            "{n_bus} buses: a mode came back with no participation, so the adjoint pass \
             and the forward one did not agree on what they found"
        );
        let restarts = match result.method {
            Method::Sparse { restarts, .. } => restarts,
            Method::Dense => 0,
        };
        let with_participation =
            result.modes.iter().filter(|m| !m.participation.is_empty()).count();

        println!(
            "{n_bus:>7}  {:>7}  {n_x:>8}  {:>8.2?}  {:>8.2?}  {worst:>10.2e}  {restarts:>9}  {:>3}/{:<2}",
            n_x / ring::STATES_PER_UNIT,
            build_time,
            analyze_time,
            with_participation,
            result.modes.len(),
        );

        if n_bus == 4096 {
            // The top participation is a fraction of a per cent, and that is
            // the answer rather than a rounding artefact: an oscillation spread
            // across two thousand machines belongs to all of them a little.
            println!("\nThe modes found at 4096 buses, least damped first:");
            for mode in result.modes.iter().take(8) {
                let top = mode
                    .participation
                    .first()
                    .map(|&(k, p)| format!("{} {:.2}%", result.state_names[k], 100.0 * p))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "  {:+9.5}{:+9.5}j   ζ = {:>7.4}   f = {:>6.3} Hz   {top}",
                    mode.eigenvalue.re, mode.eigenvalue.im, mode.damping, mode.frequency
                );
            }
        }
    }
}
