//! A sparse symmetric `LDLᵀ` that reports the inertia of the matrix.
//!
//! The factorization an interior point needs for its KKT system. An LU solves
//! the system as well, but says nothing about the *signs* of the matrix's
//! eigenvalues, and the interior point needs exactly that: its Newton step
//! minimizes the barrier model only when the KKT matrix
//!
//! \\[ K = \begin{pmatrix} H + \Sigma + \delta_w I & A^{\mathsf T} \\\\
//!                         A & -\delta_c I \end{pmatrix} \\]
//!
//! has `n` positive and `m` negative eigenvalues (`n` variables, `m`
//! constraints). With any other inertia the step heads for a saddle or a
//! maximum, and the remedy, IPOPT's inertia correction, is to raise
//! \\(\delta_w\\) until the inertia is right.
//!
//! # How the inertia is read
//!
//! By Sylvester's law of inertia, `K = P L D Lᵀ Pᵀ` with `D` diagonal has the
//! inertia of `D`: count its positive, negative and zero entries. That holds
//! for any factorization that exists, pivoted or not.
//!
//! # Why no pivoting
//!
//! This is `faer`'s sparse `LDLᵀ` on an AMD ordering, with no pivoting. An
//! indefinite matrix need not have such a factorization, but a
//! *quasi-definite* one always does, for every symmetric ordering (Vanderbei,
//! 1995): `K` above is quasi-definite once its upper-left block is positive
//! definite and \\(\delta_c > 0\\), which is the state the inertia correction
//! drives it to anyway. Before that, a factorization may fail on a zero pivot
//! or come out with the wrong inertia, and both send the caller to raise
//! \\(\delta_w\\). The same design as QDLDL's (OSQP) and one of MadNLP's
//! options; IPOPT's MUMPS and MA27 pivot instead. Accuracy is recovered by the
//! caller's iterative refinement.
//!
//! The symbolic analysis is redone on every [`SymmetricFactorization::new`].

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, LdltRef, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering,
    factorize_symbolic_cholesky, supernodal::SupernodalLdltRef,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, Mat, Par, Side};

/// Counts of positive, negative and zero eigenvalues.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

/// `LDLᵀ` of a sparse symmetric matrix, and its inertia.
pub struct SymmetricFactorization {
    n: usize,
    symbolic: SymbolicCholesky<usize>,
    values: Vec<f64>,
    inertia: Inertia,
}

impl SymmetricFactorization {
    /// Factorizes the `n × n` symmetric matrix given by `entries`. Entries
    /// above the diagonal are ignored, so either a full symmetric list or its
    /// lower triangle may be passed; duplicates sum. `None` when the
    /// factorization hits a zero pivot (the matrix is singular, or needs
    /// pivoting this does not do; see the module docs).
    pub fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        let mut lower: Vec<Triplet<usize, usize, f64>> = entries
            .iter()
            .filter(|&&(r, c, _)| r >= c)
            .map(|&(r, c, v)| Triplet::new(r, c, v))
            .collect();
        // Every diagonal present, so an empty row still has a pivot slot.
        lower.extend((0..n).map(|k| Triplet::new(k, k, 0.0)));
        let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &lower).ok()?;

        let symbolic = factorize_symbolic_cholesky(
            mat.symbolic(),
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .ok()?;
        let mut values = vec![0.0; symbolic.len_val()];
        let mut mem = MemBuffer::new(
            symbolic.factorize_numeric_ldlt_scratch::<f64>(Par::Seq, Default::default()),
        );
        symbolic
            .factorize_numeric_ldlt(
                &mut values,
                mat.as_ref(),
                Side::Lower,
                LdltRegularization::default(),
                Par::Seq,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .ok()?;

        let inertia = count(&diagonal(&symbolic, &values));
        Some(Self {
            n,
            symbolic,
            values,
            inertia,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn inertia(&self) -> Inertia {
        self.inertia
    }

    /// Solves `K x = b`. `None` if the result is not finite.
    pub fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        debug_assert_eq!(rhs.len(), self.n);
        let mut x = Mat::<f64>::from_fn(self.n, 1, |i, _| rhs[i]);
        let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq));
        LdltRef::new(&self.symbolic, &self.values).solve_in_place_with_conj(
            Conj::No,
            x.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
        );
        let out: Vec<f64> = (0..self.n).map(|i| x[(i, 0)]).collect();
        out.iter().all(|v| v.is_finite()).then_some(out)
    }
}

/// `D`'s entries, in factorization order (the order does not matter for the
/// inertia). `faer` stores `D` on the diagonal of the unit-lower `L`: in the
/// simplicial layout as each column's first entry, in the supernodal one on
/// the diagonal of each supernode's leading square block.
fn diagonal(symbolic: &SymbolicCholesky<usize>, values: &[f64]) -> Vec<f64> {
    match symbolic.raw() {
        SymbolicCholeskyRaw::Simplicial(s) => {
            let col_ptr = s.col_ptr();
            (0..s.nrows()).map(|j| values[col_ptr[j]]).collect()
        }
        SymbolicCholeskyRaw::Supernodal(s) => {
            let factor = SupernodalLdltRef::new(s, values);
            let mut d = Vec::with_capacity(s.nrows());
            for k in 0..s.n_supernodes() {
                let node = factor.supernode(k).val();
                for j in 0..node.ncols() {
                    d.push(node[(j, j)]);
                }
            }
            d
        }
    }
}

fn count(d: &[f64]) -> Inertia {
    let mut inertia = Inertia::default();
    for &v in d {
        if v > 0.0 {
            inertia.positive += 1;
        } else if v < 0.0 {
            inertia.negative += 1;
        } else {
            inertia.zero += 1;
        }
    }
    inertia
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random symmetric KKT-shaped matrix: an
    /// indefinite `H` (n × n, sparse), `A` (m × n), `-δ` on the lower block.
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

    fn dense(dim: usize, e: &[(usize, usize, f64)]) -> Mat<f64> {
        let mut a = Mat::<f64>::zeros(dim, dim);
        for &(r, c, v) in e {
            a[(r, c)] += v;
        }
        a
    }

    fn eigen_inertia(a: &Mat<f64>) -> Inertia {
        let eig = a.self_adjoint_eigenvalues(Side::Lower).unwrap();
        count(
            &eig.iter()
                .map(|&v| if v.abs() < 1e-12 { 0.0 } else { v })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn inertia_matches_the_eigenvalues() {
        for seed in 0..40 {
            for &(n, m, shift) in &[(6, 2, 0.0), (12, 5, -1.0), (20, 8, 3.0), (30, 10, 0.5)] {
                let (dim, e) = kkt(n, m, shift, seed);
                let Some(f) = SymmetricFactorization::new(dim, &e) else {
                    continue; // a zero pivot: the caller regularizes
                };
                assert_eq!(
                    f.inertia(),
                    eigen_inertia(&dense(dim, &e)),
                    "seed {seed}, n {n}, m {m}"
                );
            }
        }
    }

    /// Without pivoting, a constraint row eliminated before its variables
    /// has the pivot -δc = -1e-8, and the growth it causes costs about eight
    /// digits of the solve. The caller's iterative refinement recovers them:
    /// two rounds here reach rounding level, as `opf::nlp` does with its own.
    #[test]
    fn solves_a_quasi_definite_system_with_refinement() {
        let (dim, e) = kkt(40, 15, 10.0, 7);
        let f = SymmetricFactorization::new(dim, &e).unwrap();
        assert_eq!(
            f.inertia(),
            Inertia {
                positive: 40,
                negative: 15,
                zero: 0
            }
        );
        let a = dense(dim, &e);
        let residual = |x: &[f64], b: &[f64]| -> Vec<f64> {
            (0..dim)
                .map(|i| b[i] - (0..dim).map(|j| a[(i, j)] * x[j]).sum::<f64>())
                .collect()
        };
        let b: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
        let mut x = f.solve(&b).unwrap();
        for _ in 0..2 {
            let dx = f.solve(&residual(&x, &b)).unwrap();
            x.iter_mut().zip(&dx).for_each(|(xi, d)| *xi += d);
        }
        let scale = a.norm_max() * x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = residual(&x, &b).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            worst < 1e-14 * scale,
            "residual {worst:.1e}, scale {scale:.1e}"
        );
    }

    /// Large enough for `faer` to choose its supernodal layout, where `D` is
    /// read from a different place.
    #[test]
    fn inertia_in_the_supernodal_layout() {
        let (dim, e) = kkt(300, 120, 6.0, 3);
        let f = SymmetricFactorization::new(dim, &e).unwrap();
        assert!(
            matches!(f.symbolic.raw(), SymbolicCholeskyRaw::Supernodal(_)),
            "test needs the supernodal path"
        );
        assert_eq!(f.inertia(), eigen_inertia(&dense(dim, &e)));
    }
}
