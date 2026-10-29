//! How much work each step policy does on one sweep, counted on the CPU
//! stand-in: outcomes, Newton steps, and the systems factored and solved —
//! what a masked device call costs in proportion to, if phase 0 confirms it.
//!
//! ```text
//! cargo run --release -p gridoxide-gpu --example batch_counts -- <case.m> [B] [tol]
//! ```
//!
//! The sweep is grid-bench's kind: from the solved base case, scenario k
//! scales every injection by a daily load curve between 0.6 and 1.0, with 5%
//! deterministic noise per bus and scenario.

use gridoxide::network::{build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide::solver::{JacobianBackend, PersistentSolver};
use gridoxide_gpu::{BatchOptions, CpuUniformBatch, Outcome, StepPolicy, solve_batch};

/// A deterministic standard normal: splitmix64 into Box-Muller.
fn normal(state: &mut u64) -> f64 {
    let mut next = || {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let (u, v) = (next().max(1e-300), next());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: batch_counts <case.m> [B] [tol]");
    let batch: usize = args.get(2).map_or(100, |b| b.parse().unwrap());
    let tol: f64 = args.get(3).map_or(1e-6, |t| t.parse().unwrap());

    let net = gridoxide::matpower::read(path).unwrap();
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let mut base = net.buses.clone();
    linear_initial_guess(&mut base, &ybus);
    PersistentSolver::new(JacobianBackend::KluNative).solve(&mut base, &ybus, 1e-10, 30);

    let mut rng = 0x5eed_u64;
    let scenarios: Vec<Scenario> = (0..batch)
        .map(|k| {
            let t = k as f64 / batch as f64;
            let scale = 0.6 + 0.4 * (0.5 - 0.5 * (2.0 * std::f64::consts::PI * t).cos());
            Scenario::new(
                net.buses
                    .iter()
                    .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
                    .map(|b| {
                        let f = scale * (1.0 + 0.05 * normal(&mut rng));
                        BusOverride::new(b.idx).p(b.p_spec * f).q(b.q_spec * f)
                    })
                    .collect(),
            )
        })
        .collect();

    println!(
        "{path}: {} buses, B = {batch}, tol = {tol:e} p.u.",
        net.buses.len()
    );
    println!(
        "{:<26} {:>5} {:>5} {:>5} {:>7} {:>7} {:>9} {:>9} {:>6} {:>5}",
        "policy", "conv", "not", "div", "steps", "max", "factored", "solved", "batch", "cpu"
    );
    let adaptive = BatchOptions {
        tol,
        ..BatchOptions::default()
    };
    let policies: Vec<(String, BatchOptions)> = vec![
        (
            "fixed 4 (gpusim2grid)".into(),
            BatchOptions {
                policy: StepPolicy::Fixed(4),
                ..adaptive
            },
        ),
        ("adaptive".into(), adaptive),
        (
            "adaptive, reuse < 0.1".into(),
            BatchOptions {
                reuse_below: Some(0.1),
                ..adaptive
            },
        ),
        (
            "adaptive, reuse < 0.3".into(),
            BatchOptions {
                reuse_below: Some(0.3),
                ..adaptive
            },
        ),
        (
            format!("adaptive, cpu <= {}", (batch / 20).max(1)),
            BatchOptions {
                cpu_handoff: Some((batch / 20).max(1)),
                ..adaptive
            },
        ),
    ];
    for (name, opts) in policies {
        let report = solve_batch::<CpuUniformBatch>(&base, &ybus, &scenarios, &opts).unwrap();
        let count = |o: Outcome| report.scenarios.iter().filter(|s| s.outcome == o).count();
        let steps: usize = report.scenarios.iter().map(|s| s.newton_steps).sum();
        let max_steps = report
            .scenarios
            .iter()
            .map(|s| s.newton_steps)
            .max()
            .unwrap_or(0);
        println!(
            "{:<26} {:>5} {:>5} {:>5} {:>7} {:>7} {:>9} {:>9} {:>6} {:>5}",
            name,
            count(Outcome::Converged),
            count(Outcome::NotConverged),
            count(Outcome::Diverged),
            steps,
            max_steps,
            report.linear.systems_factored,
            report.linear.systems_solved,
            report.batch_steps,
            report.cpu_handoffs
        );
    }
}
