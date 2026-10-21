//! Where a warm Newton-Raphson solve spends its time.
//!
//! `cargo run --release --example pf_profile --features matpower,klu -- case.m [repeats]`
//!
//! Times `PersistentSolver::solve` from a flat start the way
//! `PowerFlowModel.solve(init="flat")` drives it — clone the buses, solve on a
//! cached factorization — and then the same Newton loop re-run by hand with a
//! clock on each phase, per backend. The two totals should agree; the phases
//! say which one to work on.

use std::time::{Duration, Instant};

use gridoxide::jacobian::JacobianPattern;
use gridoxide::lu::SparseLu;
use gridoxide::network::{
    build_ybus, connected_components, effective_injection, power_injections, stamp_shunts,
};
use gridoxide::solver::{JacobianBackend, PersistentSolver};
use gridoxide::types::{Bus, BusType};

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

#[derive(Default)]
struct Phases {
    components: Duration,
    injections: Duration,
    mismatch: Duration,
    fill: Duration,
    factor_solve: Duration,
    update: Duration,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: pf_profile <case.m> [repeats]");
    let repeats: usize = args.get(2).map_or(200, |r| r.parse().unwrap());

    let net = gridoxide::matpower::read(path).expect("read");
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();
    let template: Vec<Bus> = net.buses;
    let (tol, max_iter) = (1e-8, 30);
    let nnz: usize = (0..template.len()).map(|i| ybus.row(i).len()).sum();
    println!("{path}: {} buses, {nnz} Y-bus nonzeros", template.len());

    let mut backends = vec![
        ("faer", JacobianBackend::Scalar),
        ("klu_native", JacobianBackend::KluNative),
    ];
    #[cfg(feature = "klu")]
    backends.push(("klu", JacobianBackend::Klu));

    // `PF_BACKEND=klu` runs one backend only, for a profiler.
    let only = std::env::var("PF_BACKEND").ok();
    for (name, backend) in backends {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        // 1. The bindings' path, end to end.
        let mut solver = PersistentSolver::new(backend);
        let mut iterations = 0;
        let mut whole = Vec::with_capacity(repeats);
        for k in 0..=repeats {
            let start = Instant::now();
            let mut buses = template.clone();
            let (_, stats) = solver.solve_with_stats(&mut buses, &ybus, tol, max_iter);
            if k > 0 {
                whole.push(start.elapsed());
            }
            iterations = stats.iterations();
            if k == repeats {
                let trace: Vec<String> = stats
                    .mismatch_history
                    .iter()
                    .map(|m| format!("{m:.1e}"))
                    .collect();
                println!("  {name:10} mismatch by iteration: {}", trace.join(" "));
            }
        }

        // 2. The same loop by hand, phase by phase (no island handling: these
        // cases are one island).
        let mut lu = SparseLu::new(backend.lu_backend());
        let pattern = JacobianPattern::analyze(&template, &ybus);
        let non_slack: Vec<usize> = template
            .iter()
            .filter(|b| b.bus_type != BusType::Slack)
            .map(|b| b.idx)
            .collect();
        let pq: Vec<usize> = template
            .iter()
            .filter(|b| b.bus_type == BusType::PQ)
            .map(|b| b.idx)
            .collect();
        let n = non_slack.len() + pq.len();
        let mut values = Vec::with_capacity(pattern.len());
        let mut p = Phases::default();
        let mut by_hand = Vec::with_capacity(repeats);
        for k in 0..=repeats {
            let start = Instant::now();
            let mut buses = template.clone();
            let t = Instant::now();
            let _ = connected_components(&ybus);
            let components = t.elapsed();
            let mut phases = Phases {
                components,
                ..Default::default()
            };
            for _ in 0..max_iter {
                let t = Instant::now();
                let (p_calc, q_calc) = power_injections(&buses, &ybus);
                phases.injections += t.elapsed();

                let t = Instant::now();
                let mut mismatch = Vec::with_capacity(n);
                mismatch.extend(
                    non_slack
                        .iter()
                        .map(|&i| effective_injection(&buses[i]).0 - p_calc[i]),
                );
                mismatch.extend(
                    pq.iter()
                        .map(|&i| effective_injection(&buses[i]).1 - q_calc[i]),
                );
                let max_mis = mismatch.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
                phases.mismatch += t.elapsed();
                if max_mis < tol {
                    break;
                }

                let t = Instant::now();
                pattern.fill(&buses, &p_calc, &q_calc, &mut values);
                phases.fill += t.elapsed();

                let t = Instant::now();
                if !lu.is_analyzed() {
                    assert!(lu.analyze(n, &pattern.to_triplets(&values)));
                }
                let dx = lu
                    .factor_and_solve_values(&values, &mismatch)
                    .expect("singular");
                phases.factor_solve += t.elapsed();

                let t = Instant::now();
                for (k, &i) in non_slack.iter().enumerate() {
                    buses[i].voltage_ang += dx[k];
                }
                for (k, &i) in pq.iter().enumerate() {
                    buses[i].voltage_mag += dx[non_slack.len() + k];
                }
                phases.update += t.elapsed();
            }
            if k > 0 {
                by_hand.push(start.elapsed());
                p.components += phases.components;
                p.injections += phases.injections;
                p.mismatch += phases.mismatch;
                p.fill += phases.fill;
                p.factor_solve += phases.factor_solve;
                p.update += phases.update;
            }
        }

        let per = |d: Duration| d.as_secs_f64() * 1e6 / repeats as f64;
        let hand_mean: f64 =
            by_hand.iter().map(|d| d.as_secs_f64()).sum::<f64>() * 1e6 / repeats as f64;
        println!(
            "  {name:10} solve median {:8.1} us, {iterations} iterations | by hand mean {:8.1} us: \
             components {:7.1}  injections {:7.1}  mismatch {:6.1}  fill {:7.1}  factor+solve {:8.1}  update {:5.1}",
            median(whole).as_secs_f64() * 1e6,
            hand_mean,
            per(p.components),
            per(p.injections),
            per(p.mismatch),
            per(p.fill),
            per(p.factor_solve),
            per(p.update),
        );
    }
}
