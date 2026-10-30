//! Writes real Newton-Raphson Jacobians for the cuDSS uniform-batch probe
//! (`probes/cudss_ubatch`), with CPU reference solutions to check it against.
//!
//! ```text
//! cargo run --release --example gpu_probe_dump --features matpower -- <case.m> <B> <out_dir>
//! ```
//!
//! Solves the case's base state, then makes `B` scenarios by scaling its loads
//! and generation (factors 0.6 to 1.0, a deterministic spread). For each, from
//! the base solution, it writes two Newton steps' linear systems: the Jacobian
//! and mismatch at the base voltages (step 0), and again after step 0's update
//! (step 1), each with `dx` solved on the CPU. Two value sets are what the
//! probe needs to test refactorization, and a mask that keeps some systems on
//! step 0's factors while others move to step 1's.
//!
//! Files, all little-endian:
//! - `meta.txt`: `n nnz B`
//! - `row_ptr.i32` (n + 1), `col_idx.i32` (nnz): the Jacobian's CSR pattern,
//!   shared by every scenario and step (the pattern is the topology's);
//! - `values{0,1}.f64` (B × nnz, CSR order), `rhs{0,1}.f64` (B × n),
//!   `x{0,1}.f64` (B × n): per step, scenario-major.

use std::io::Write;
use std::path::Path;

use gridoxide::batch::uniform_load_scaling;
use gridoxide::jacobian::JacobianPattern;
use gridoxide::network::{
    build_ybus, effective_injection, linear_initial_guess, power_injections, stamp_shunts,
};
use gridoxide::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use gridoxide::sparse::RealFactorization;
use gridoxide::types::{Bus, BusType};

fn write_f64(dir: &Path, name: &str, data: &[f64]) {
    let mut f = std::fs::File::create(dir.join(name)).unwrap();
    for v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn write_i32(dir: &Path, name: &str, data: &[i32]) {
    let mut f = std::fs::File::create(dir.join(name)).unwrap();
    for v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, batch, out) = match &args[..] {
        [_, path, batch, out] => (path, batch.parse::<usize>().unwrap(), Path::new(out)),
        _ => panic!("usage: gpu_probe_dump <case.m> <B> <out_dir>"),
    };
    std::fs::create_dir_all(out).unwrap();

    let net = gridoxide::matpower::read(path).unwrap();
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let ybus = ybus.finish();

    let mut base = net.buses.clone();
    linear_initial_guess(&mut base, &ybus);
    let (_, stats) = PersistentSolver::new(JacobianBackend::KluNative)
        .solve_with_stats(&mut base, &ybus, 1e-10, 30);
    assert_eq!(
        stats.status,
        SolveStatus::Converged,
        "the base case must solve"
    );

    let pattern = JacobianPattern::analyze(&base, &ybus);
    let n = pattern.n_unknowns;
    let nnz = pattern.len();
    // Pattern order -> CSR order (row-major, columns ascending).
    let mut order: Vec<usize> = (0..nnz).collect();
    order.sort_by_key(|&k| (pattern.rows()[k], pattern.cols()[k]));
    let mut row_ptr = vec![0i32; n + 1];
    for &k in &order {
        row_ptr[pattern.rows()[k] as usize + 1] += 1;
    }
    for r in 0..n {
        row_ptr[r + 1] += row_ptr[r];
    }
    let col_idx: Vec<i32> = order.iter().map(|&k| pattern.cols()[k] as i32).collect();
    for w in order.windows(2) {
        let (a, b) = (w[0], w[1]);
        assert!(
            (pattern.rows()[a], pattern.cols()[a]) != (pattern.rows()[b], pattern.cols()[b]),
            "the Jacobian pattern has duplicate entries"
        );
    }

    let non_slack: Vec<usize> = base
        .iter()
        .filter(|b| b.bus_type != BusType::Slack)
        .map(|b| b.idx)
        .collect();
    let pq: Vec<usize> = base
        .iter()
        .filter(|b| b.bus_type == BusType::PQ)
        .map(|b| b.idx)
        .collect();

    // The linear system of one Newton step at `buses`: CSR values, mismatch, dx.
    let step = |buses: &[Bus]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (p, q) = power_injections(buses, &ybus);
        let mut values = Vec::new();
        pattern.fill(buses, &p, &q, &mut values);
        let mut rhs = Vec::with_capacity(n);
        rhs.extend(
            non_slack
                .iter()
                .map(|&i| effective_injection(&buses[i]).0 - p[i]),
        );
        rhs.extend(pq.iter().map(|&i| effective_injection(&buses[i]).1 - q[i]));
        let x = RealFactorization::new(n, &pattern.to_triplets(&values))
            .and_then(|f| f.solve(&rhs))
            .expect("the Jacobian is nonsingular");
        (order.iter().map(|&k| values[k]).collect(), rhs, x)
    };

    let mut sets: [(Vec<f64>, Vec<f64>, Vec<f64>); 2] = Default::default();
    for k in 0..batch {
        // A spread of operating points, deterministic so a rerun writes the
        // same files.
        let factor = 0.6 + 0.4 * (k as f64 + 0.5) / batch as f64;
        let mut buses = base.clone();
        for ov in uniform_load_scaling(&net.buses, factor).bus_overrides {
            let b = &mut buses[ov.bus];
            b.p_spec = ov.p_spec.unwrap_or(b.p_spec);
            b.q_spec = ov.q_spec.unwrap_or(b.q_spec);
        }
        for (s, set) in sets.iter_mut().enumerate() {
            let (values, rhs, x) = step(&buses);
            set.0.extend(&values);
            set.1.extend(&rhs);
            set.2.extend(&x);
            if s == 0 {
                let (dth, dvm) = x.split_at(non_slack.len());
                for (&i, &d) in non_slack.iter().zip(dth) {
                    buses[i].voltage_ang += d;
                }
                for (&i, &d) in pq.iter().zip(dvm) {
                    buses[i].voltage_mag += d;
                }
            }
        }
    }

    std::fs::write(out.join("meta.txt"), format!("{n} {nnz} {batch}\n")).unwrap();
    write_i32(out, "row_ptr.i32", &row_ptr);
    write_i32(out, "col_idx.i32", &col_idx);
    for (s, (values, rhs, x)) in sets.iter().enumerate() {
        write_f64(out, &format!("values{s}.f64"), values);
        write_f64(out, &format!("rhs{s}.f64"), rhs);
        write_f64(out, &format!("x{s}.f64"), x);
    }
    println!("{path}: n={n} nnz={nnz} B={batch} -> {}", out.display());
}
