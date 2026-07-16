//! Thin internal wrapper around `faer`'s sparse LU, isolating gridoxide's
//! power-flow math from the specific sparse-solver backend. Both a general
//! (non-symmetric, since phase-shifting transformers make the admittance
//! matrix non-Hermitian) complex solve and a real solve are provided, since
//! Newton-Raphson needs the latter for its Jacobian and the linear
//! initial-guess warm start needs the former for the Y-bus.

use num_complex::Complex;
use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat, Triplet};
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::Col;
use faer::prelude::{Reborrow, Solve};

/// A sparse matrix built once from a fixed set of (row, col, value) triplets
/// (duplicates summed), kept around for repeated matrix-vector products —
/// `power_injections` needs one every Newton-Raphson iteration, but the
/// Y-bus itself never changes across iterations, so the conversion from
/// triplets happens once at construction, not on every call.
pub struct SparseMatrix {
    n: usize,
    mat: SparseColMat<usize, Complex<f64>>,
}

impl SparseMatrix {
    pub fn build(n: usize, entries: &[(usize, usize, Complex<f64>)]) -> Option<Self> {
        let triplets: Vec<Triplet<usize, usize, Complex<f64>>> =
            entries.iter().map(|&(r, c, v)| Triplet::new(r, c, v)).collect();
        let mat = SparseColMat::<usize, Complex<f64>>::try_new_from_triplets(n, n, &triplets).ok()?;
        Some(Self { n, mat })
    }

    pub fn mul_vec(&self, v: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let vcol = Col::<Complex<f64>>::from_fn(self.n, |i| v[i]);
        let result = &self.mat * &vcol;
        (0..self.n).map(|i| result[i]).collect()
    }
}

/// Solves a one-shot general complex sparse linear system `A x = b`.
/// Returns `None` if the matrix is singular. Used by `linear_initial_guess`,
/// which only solves once per power-flow run.
///
/// Unlike `nalgebra`'s dense LU, `faer`'s sparse LU does not detect
/// singularity itself — it silently returns NaN-poisoned values instead of
/// an error, so singularity is detected here via an explicit finiteness
/// check on the result, to preserve the `None`-on-singular contract callers
/// rely on.
pub fn solve_complex(
    n: usize,
    entries: &[(usize, usize, Complex<f64>)],
    rhs: &[Complex<f64>],
) -> Option<Vec<Complex<f64>>> {
    let triplets: Vec<Triplet<usize, usize, Complex<f64>>> =
        entries.iter().map(|&(r, c, v)| Triplet::new(r, c, v)).collect();
    let mat = SparseColMat::<usize, Complex<f64>>::try_new_from_triplets(n, n, &triplets).ok()?;
    let lu = mat.sp_lu().ok()?;
    let b = Col::<Complex<f64>>::from_fn(n, |i| rhs[i]);
    let x = lu.solve(&b);
    if (0..n).any(|i| !x[i].re.is_finite() || !x[i].im.is_finite()) {
        return None;
    }
    Some((0..n).map(|i| x[i]).collect())
}

/// A real sparse system whose sparsity *pattern* is fixed across repeated
/// solves (Newton-Raphson's Jacobian: same bus topology every iteration,
/// only numeric values change) — caches both the LU symbolic factorization
/// (ordering + fill-in) and the triplet *argsort* (which sparse-matrix slot
/// each triplet's value lands in) once, and reuses both on each call.
///
/// Reusing the argsort matters on its own, separately from the LU symbolic
/// reuse: without it, every call would still re-sort and re-deduplicate the
/// full triplet list from scratch even though only the *values* differ
/// between calls — profiling with `perf` on a 2,605-node benchmark showed
/// this re-sorting cost was ~10% of total Newton-Raphson time (comparable to
/// the LU factorization itself), since a fresh `try_new_from_triplets` was
/// being paid for on every iteration despite the (row, col) pattern never
/// changing. `new_from_argsort` skips straight to placing values using the
/// cached order.
pub struct RealSparseSystem {
    n: usize,
    symbolic_mat: SymbolicSparseColMat<usize>,
    argsort: Argsort<usize>,
    symbolic_lu: SymbolicLu<usize>,
}

impl RealSparseSystem {
    /// Builds the symbolic factorization and argsort from an initial
    /// sparsity pattern. Subsequent calls to `factor_and_solve` must supply
    /// entries with the exact same (row, col) pairs in the exact same order
    /// (values may differ) — the cached argsort assumes positional
    /// correspondence, not just set equality.
    pub fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        let pairs: Vec<Pair<usize, usize>> =
            entries.iter().map(|&(r, c, _)| Pair { row: r, col: c }).collect();
        let (symbolic_mat, argsort) = SymbolicSparseColMat::try_new_from_indices(n, n, &pairs).ok()?;
        let symbolic_lu = SymbolicLu::try_new(symbolic_mat.rb()).ok()?;
        Some(Self { n, symbolic_mat, argsort, symbolic_lu })
    }

    /// Numeric-only refactorization against the cached symbolic pattern and
    /// argsort, then solves `A x = b`. Returns `None` if the matrix is
    /// singular (see `solve_complex`'s doc comment for why this needs an
    /// explicit check).
    pub fn factor_and_solve(&self, entries: &[(usize, usize, f64)], rhs: &[f64]) -> Option<Vec<f64>> {
        let vals: Vec<f64> = entries.iter().map(|&(_, _, v)| v).collect();
        let mat = SparseColMat::new_from_argsort(self.symbolic_mat.clone(), &self.argsort, &vals).ok()?;
        let lu = Lu::try_new_with_symbolic(self.symbolic_lu.clone(), mat.as_ref()).ok()?;
        let b = Col::<f64>::from_fn(self.n, |i| rhs[i]);
        let x = lu.solve(&b);
        if (0..self.n).any(|i| !x[i].is_finite()) {
            return None;
        }
        Some((0..self.n).map(|i| x[i]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_complex_simple_system() {
        // [[2, 1], [1, 3]] * [x0, x1] = [5, 10]  =>  x0=1, x1=3
        let entries = vec![
            (0, 0, Complex::new(2.0, 0.0)),
            (0, 1, Complex::new(1.0, 0.0)),
            (1, 0, Complex::new(1.0, 0.0)),
            (1, 1, Complex::new(3.0, 0.0)),
        ];
        let rhs = vec![Complex::new(5.0, 0.0), Complex::new(10.0, 0.0)];
        let x = solve_complex(2, &entries, &rhs).unwrap();
        assert!((x[0] - Complex::new(1.0, 0.0)).norm() < 1e-10);
        assert!((x[1] - Complex::new(3.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn solve_complex_duplicate_entries_sum() {
        // (0,0) contributed twice as 1.0+1.0=2.0, matching dense += semantics.
        let entries = vec![
            (0, 0, Complex::new(1.0, 0.0)),
            (0, 0, Complex::new(1.0, 0.0)),
            (0, 1, Complex::new(1.0, 0.0)),
            (1, 0, Complex::new(1.0, 0.0)),
            (1, 1, Complex::new(3.0, 0.0)),
        ];
        let rhs = vec![Complex::new(5.0, 0.0), Complex::new(10.0, 0.0)];
        let x = solve_complex(2, &entries, &rhs).unwrap();
        assert!((x[0] - Complex::new(1.0, 0.0)).norm() < 1e-10);
        assert!((x[1] - Complex::new(3.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn real_sparse_system_refactor_reuses_symbolic() {
        let entries_a = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let sys = RealSparseSystem::new(2, &entries_a).unwrap();
        let x = sys.factor_and_solve(&entries_a, &[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);

        // Same sparsity pattern, different numeric values (as across NR iterations).
        let entries_b = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)];
        // [[4,1],[1,2]] * x = [6, 5] => x0=1, x1=2
        let x2 = sys.factor_and_solve(&entries_b, &[6.0, 5.0]).unwrap();
        assert!((x2[0] - 1.0).abs() < 1e-10);
        assert!((x2[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn singular_matrix_returns_none() {
        let entries = vec![(0, 0, 1.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 2.0)];
        let sys = RealSparseSystem::new(2, &entries);
        // Symbolic analysis may succeed even for a numerically singular matrix
        // (it only depends on the sparsity pattern); the failure surfaces at
        // factor_and_solve time.
        if let Some(sys) = sys {
            assert!(sys.factor_and_solve(&entries, &[1.0, 2.0]).is_none());
        }
    }
}
