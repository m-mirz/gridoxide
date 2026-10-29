//! How much work each step policy does on one sweep, counted on the CPU
//! stand-in: outcomes, Newton steps, and the systems factored and solved —
//! what a masked device call costs in proportion to, if phase 0 confirms it.
//!
//! ```text
//! cargo run --release -p gridoxide-gpu --example batch_counts -- <case.m> [B] [tol] [sweep|n1] [chunk]
//! ```
//!
//! The sweep is grid-bench's kind: from the solved base case, scenario k
//! scales every injection by a daily load curve between 0.6 and 1.0, with 5%
//! deterministic noise per bus and scenario. `n1` is B single-branch outages
//! spread over the branch list (B = 0: every branch), from the base case.
//!
//! The scenarios are solved as consecutive batches of at most `chunk`
//! (default 1000), the counts summed over them: the CPU stand-in holds two
//! copies of every scenario's Jacobian values, so case9241pegase's whole N-1
//! list (16049 outages) in one batch would need about 40 GB. A device batch is
//! bounded the same way, by its memory.

use gridoxide::network::{build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::scenario::{BusOverride, Scenario};
use gridoxide::solver::{JacobianBackend, PersistentSolver};
use gridoxide_gpu::{
    BatchOptions, CpuUniformBatch, Outcome, StepPolicy, solve_batch, solve_contingency_batch,
};

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
    let path = args
        .get(1)
        .expect("usage: batch_counts <case.m> [B] [tol] [sweep|n1] [chunk]");
    let batch: usize = args.get(2).map_or(100, |b| b.parse().unwrap());
    let tol: f64 = args.get(3).map_or(1e-6, |t| t.parse().unwrap());
    let n1 = match args.get(4).map(String::as_str) {
        None | Some("sweep") => false,
        Some("n1") => true,
        Some(other) => panic!("unknown kind {other}: sweep or n1"),
    };
    let chunk: usize = args.get(5).map_or(1000, |c| c.parse().unwrap());

    let net = gridoxide::matpower::read(path).unwrap();
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let mut base = net.buses.clone();
    linear_initial_guess(&mut base, &ybus);
    PersistentSolver::new(JacobianBackend::KluNative).solve(&mut base, &ybus, 1e-10, 30);

    let n_branches = net.lines.len() + net.transformers.len();
    let mut rng = 0x5eed_u64;
    let scenarios: Vec<Scenario> = if n1 {
        let count = if batch == 0 { n_branches } else { batch };
        (0..count)
            .map(|k| Scenario {
                bus_overrides: Vec::new(),
                branch_outages: vec![k * n_branches / count],
            })
            .collect()
    } else {
        (0..batch)
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
            .collect()
    };

    println!(
        "{path}: {} buses, {} {} in batches of at most {chunk}, tol = {tol:e} p.u.",
        net.buses.len(),
        scenarios.len(),
        if n1 {
            "single-branch outages"
        } else {
            "sweep scenarios"
        }
    );
    println!(
        "{:<26} {:>5} {:>5} {:>5} {:>5} {:>7} {:>7} {:>9} {:>9} {:>6} {:>5}",
        "policy",
        "conv",
        "not",
        "div",
        "split",
        "steps",
        "max",
        "factored",
        "solved",
        "batch",
        "cpu"
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
            "adaptive, cpu <= 5%".into(),
            BatchOptions {
                cpu_handoff: Some((scenarios.len().min(chunk) / 20).max(1)),
                ..adaptive
            },
        ),
    ];
    for (name, opts) in policies {
        let mut results = Vec::with_capacity(scenarios.len());
        let (mut factored, mut solved, mut batch_steps, mut cpu) = (0, 0, 0, 0);
        for part in scenarios.chunks(chunk) {
            let report = if n1 {
                solve_contingency_batch::<CpuUniformBatch>(
                    &base,
                    &net.lines,
                    &net.transformers,
                    &net.shunts,
                    part,
                    &opts,
                )
            } else {
                solve_batch::<CpuUniformBatch>(&base, &ybus, part, &opts)
            }
            .unwrap();
            factored += report.linear.systems_factored;
            solved += report.linear.systems_solved;
            batch_steps += report.batch_steps;
            cpu += report.cpu_handoffs;
            // Only what the table needs: the voltages are dropped here.
            results.extend(
                report
                    .scenarios
                    .into_iter()
                    .map(|s| (s.outcome, s.newton_steps)),
            );
        }
        let count = |o: Outcome| results.iter().filter(|s| s.0 == o).count();
        let steps: usize = results.iter().map(|s| s.1).sum();
        let max_steps = results.iter().map(|s| s.1).max().unwrap_or(0);
        println!(
            "{:<26} {:>5} {:>5} {:>5} {:>5} {:>7} {:>7} {:>9} {:>9} {:>6} {:>5}",
            name,
            count(Outcome::Converged),
            count(Outcome::NotConverged),
            count(Outcome::Diverged),
            count(Outcome::Splits),
            steps,
            max_steps,
            factored,
            solved,
            batch_steps,
            cpu
        );
    }
}
