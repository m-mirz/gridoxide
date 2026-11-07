//! cuDSS's KKT solver against faer's, on the same matrices: the inertia the
//! interior point's regularization decides on, the solutions, and Cholesky's
//! verdict on positive definiteness.
#![cfg(feature = "cuda")]

use gridoxide_gpu::kkt::{CudssSolver, Factorization};
use gridoxide_sparse::ldlt::{LdltSolver, LltSolver, SymmetricSolver};

/// The KKT-shaped matrix of gridoxide-sparse's own tests: an indefinite `H`
/// (n × n), `A` (m × n), `-δ` on the lower block.
fn kkt(n: usize, m: usize, shift: f64, seed: u64) -> (usize, Vec<(usize, usize, f64)>) {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let mut e = Vec::new();
    for i in 0..n {
        e.push((i, i, next() * 4.0 + shift));
        for j in 0..i {
            if next() > 0.3 {
                let v = next();
                e.push((i, j, v));
                e.push((j, i, v));
            }
        }
    }
    for r in 0..m {
        for j in 0..n {
            if next() > 0.2 || j == r {
                let v = next() + if j == r { 2.0 } else { 0.0 };
                e.push((n + r, j, v));
                e.push((j, n + r, v));
            }
        }
        e.push((n + r, n + r, -1e-8));
    }
    (n + m, e)
}

fn keys(e: &[(usize, usize, f64)]) -> Vec<(usize, usize)> {
    e.iter().map(|&(r, c, _)| (r, c)).collect()
}

fn residual(e: &[(usize, usize, f64)], x: &[f64], b: &[f64]) -> f64 {
    let mut r = b.to_vec();
    for &(i, j, v) in e {
        r[i] -= v * x[j];
    }
    r.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Wherever faer's unpivoted `LDLᵀ` gives an inertia, cuDSS's gives the same,
/// refactorized over one analysis as the shift moves the inertia around.
#[test]
fn ldlt_inertia_matches_faer() {
    let mut compared = 0;
    for &(n, m) in &[(12, 5), (30, 10), (300, 120)] {
        let (dim, e0) = kkt(n, m, 0.0, 1);
        let mut gpu = CudssSolver::new(dim, &keys(&e0), Factorization::Ldlt, true).unwrap();
        let mut cpu = LdltSolver::new(dim, &keys(&e0)).unwrap();
        for shift in [-1.0, 0.0, 0.5, 3.0, 6.0, 10.0] {
            let (_, e) = kkt(n, m, shift, 1);
            assert!(gpu.matches(dim, &e));
            if !cpu.factorize(&e) {
                continue;
            }
            assert!(gpu.factorize(&e), "n {n}, shift {shift}");
            assert_eq!(gpu.inertia(), cpu.inertia(), "n {n}, shift {shift}");
            compared += 1;
        }
    }
    assert!(compared >= 10, "only {compared} comparisons");
}

/// The solution of a quasi-definite system, with the interior point's two
/// rounds of refinement, to the accuracy faer reaches.
#[test]
fn ldlt_solves_like_faer() {
    for &(n, m) in &[(40, 15), (300, 120)] {
        let (dim, e) = kkt(n, m, 10.0, 7);
        let mut gpu = CudssSolver::new(dim, &keys(&e), Factorization::Ldlt, true).unwrap();
        assert!(gpu.factorize(&e));
        assert_eq!(gpu.inertia().positive, n);
        assert_eq!(gpu.inertia().negative, m);
        let b: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
        let mut x = gpu.solve(&b).unwrap();
        for _ in 0..2 {
            let mut r = b.clone();
            for &(i, j, v) in &e {
                r[i] -= v * x[j];
            }
            let dx = gpu.solve(&r).unwrap();
            x.iter_mut().zip(&dx).for_each(|(xi, d)| *xi += d);
        }
        let worst = residual(&e, &x, &b);
        assert!(worst < 1e-12, "n {n}: residual {worst:.1e}");
    }
}

/// Cholesky succeeds on the positive definite Hessian block, solving as
/// faer's does, and fails on the indefinite KKT matrix; and after a failure
/// there is nothing to solve with.
#[test]
fn llt_tells_positive_definite_from_not() {
    for (n_x, size) in [(40, 10.0), (300, 6.0)] {
        let (dim, e) = kkt(n_x, 15, size, 5);
        let block: Vec<(usize, usize, f64)> = e
            .iter()
            .copied()
            .filter(|&(r, c, _)| r < n_x && c < n_x)
            .collect();
        let mut gpu = CudssSolver::new(n_x, &keys(&block), Factorization::Llt, true).unwrap();
        let mut cpu = LltSolver::new(n_x, &keys(&block)).unwrap();
        assert!(gpu.factorize(&block) && cpu.factorize(&block));
        assert_eq!(gpu.inertia(), cpu.inertia());
        let b: Vec<f64> = (0..n_x).map(|i| (i as f64).cos()).collect();
        let (x, y) = (gpu.solve(&b).unwrap(), cpu.solve(&b).unwrap());
        let worst = x
            .iter()
            .zip(&y)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-10, "{worst:e}");

        let mut whole = CudssSolver::new(dim, &keys(&e), Factorization::Llt, true).unwrap();
        assert!(!whole.factorize(&e));
        assert!(whole.solve(&vec![1.0; dim]).is_none());
    }
}

/// A failed factorization does not poison the next: cuDSS keeps its error
/// indicator until it is reset, and an interior point retries the same
/// solver with more regularization after every failure.
#[test]
fn a_failure_does_not_stick() {
    let (dim, e) = kkt(300, 120, 6.0, 3);
    let block = |shift: f64| -> Vec<(usize, usize, f64)> {
        e.iter()
            .copied()
            .filter(|&(r, c, _)| r < 300 && c < 300)
            .map(|(r, c, v)| (r, c, if r == c { v + shift } else { v }))
            .collect()
    };
    let indefinite = block(-20.0);
    let mut llt = CudssSolver::new(300, &keys(&indefinite), Factorization::Llt, true).unwrap();
    assert!(!llt.factorize(&indefinite));
    assert!(llt.factorize(&block(0.0)));
    assert!(llt.solve(&vec![1.0; 300]).is_some());

    let zero: Vec<(usize, usize, f64)> = e.iter().map(|&(r, c, _)| (r, c, 0.0)).collect();
    let mut ldlt = CudssSolver::new(dim, &keys(&e), Factorization::Ldlt, true).unwrap();
    assert!(ldlt.factorize(&e));
    let failed = ldlt.factorize(&zero);
    assert!(
        ldlt.factorize(&e),
        "after a factorization that returned {failed}"
    );
    assert_eq!(ldlt.inertia().positive + ldlt.inertia().negative, dim);
}

/// Deterministic: the same matrix gives the same solution, to the bit, from
/// a fresh solver and from the same one refactorized.
#[test]
fn deterministic_to_the_bit() {
    let (dim, e) = kkt(300, 120, 6.0, 3);
    let b: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
    let run = |solver: &mut CudssSolver| {
        assert!(solver.factorize(&e));
        solver.solve(&b).unwrap()
    };
    let mut one = CudssSolver::new(dim, &keys(&e), Factorization::Ldlt, true).unwrap();
    let mut two = CudssSolver::new(dim, &keys(&e), Factorization::Ldlt, true).unwrap();
    let first = run(&mut one);
    assert_eq!(first, run(&mut one));
    assert_eq!(first, run(&mut two));
}

/// The device's assembly and equilibration are the host's to the bit, on
/// unsorted triplets with duplicates and entries above the diagonal, as the
/// interior point passes them.
#[test]
fn device_assembly_is_the_hosts() {
    use gridoxide_sparse::ldlt::LowerPattern;
    for &(n, m, shift, seed) in &[(12, 5, 0.0, 1), (300, 120, 6.0, 3), (40, 15, -1.0, 9)] {
        let (dim, mut e) = kkt(n, m, shift, seed);
        // Duplicates, spread across the list, and a large spread of scales.
        let extra: Vec<(usize, usize, f64)> = e
            .iter()
            .step_by(7)
            .map(|&(r, c, v)| (r, c, v * 1e6 + 0.25))
            .collect();
        e.extend(extra);
        e.reverse();
        let pattern = LowerPattern::new(dim, &keys(&e));
        let (host_values, host_scale) = pattern.assemble(&e);
        let mut gpu = CudssSolver::new(dim, &keys(&e), Factorization::Ldlt, true).unwrap();
        let (values, scale) = gpu.assemble_on_device(&e).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&values), bits(&host_values), "values, n {n}");
        assert_eq!(bits(&scale), bits(&host_scale), "scale, n {n}");
    }
}

/// `multiply` is `K x` for the matrix of the last factorization, recovered
/// from the equilibrated one on the device, to rounding; and there is none
/// without a factorization.
#[test]
fn multiply_is_the_matrix_product() {
    for &(n, m, shift, seed) in &[(12, 5, 10.0, 1), (300, 120, 6.0, 3)] {
        let (dim, mut e) = kkt(n, m, shift, seed);
        let dup: Vec<(usize, usize, f64)> = e.iter().step_by(5).copied().collect();
        e.extend(dup);
        let mut gpu = CudssSolver::new(dim, &keys(&e), Factorization::Ldlt, true).unwrap();
        let x: Vec<f64> = (0..dim).map(|i| (0.3 * i as f64).cos()).collect();
        assert!(gpu.multiply(&x).is_none());
        assert!(gpu.factorize(&e));
        let y = gpu.multiply(&x).unwrap();
        // K as the solver reads the list: its lower triangle, duplicates
        // summed, mirrored; entries above the diagonal ignored.
        let mut want = vec![0.0; dim];
        for &(r, c, v) in e.iter().filter(|&&(r, c, _)| r >= c) {
            want[r] += v * x[c];
            if r != c {
                want[c] += v * x[r];
            }
        }
        let size = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = y
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-13 * size, "n {n}: {worst:e} of {size:e}");
    }
}

/// A system with no unknowns has an empty solution, on the device as on the
/// host; the mat-vec's buffers hold at least one value, and it declines.
#[test]
fn an_empty_system_solves() {
    let mut gpu = CudssSolver::new(0, &[], Factorization::Ldlt, true).unwrap();
    assert!(gpu.factorize(&[]));
    assert_eq!(gpu.solve(&[]), Some(Vec::new()));
    assert_eq!(gpu.multiply(&[]), None);
}

/// From the values alone, in the keys' order, cuDSS factorizes what it does
/// from the triplets: the same inertia and solution, to the bit.
#[test]
fn values_alone_factorize_as_the_triplets_do() {
    let (dim, e) = kkt(300, 120, 6.0, 3);
    let values: Vec<f64> = e.iter().map(|t| t.2).collect();
    let mut a = CudssSolver::new(dim, &keys(&e), Factorization::Ldlt, true).unwrap();
    let mut b = CudssSolver::new(dim, &keys(&e), Factorization::Ldlt, true).unwrap();
    assert!(a.factorize(&e));
    assert_eq!(b.factorize_values(&values), Some(true));
    assert_eq!(a.inertia(), b.inertia());
    let rhs: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
    assert_eq!(a.solve(&rhs), b.solve(&rhs));
    assert_eq!(b.factorize_values(&values[1..]), Some(false));
}
