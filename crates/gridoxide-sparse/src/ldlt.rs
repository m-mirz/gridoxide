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
//! The symbolic analysis is done once per pattern ([`SymmetricPattern`]);
//! [`SymmetricFactorization::new`] is the one-off shortcut.
//!
//! # The solver contract
//!
//! [`SymmetricSolver`] is what an interior point needs from its linear
//! solver, independent of where the factor lives: analysis once, then numeric
//! factorization, inertia and solves as the values change. [`LdltSolver`] is
//! this module's implementation; a GPU one (cuDSS) implements the same trait.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, LdltRef, LltRef, SymbolicCholesky, SymbolicCholeskyRaw,
    SymmetricOrdering, factorize_symbolic_cholesky, supernodal::SupernodalLdltRef,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef};
use faer::{Conj, Mat, Par, Side};
use std::rc::Rc;

/// Counts of positive, negative and zero eigenvalues.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

/// The lower triangle of a sparse symmetric pattern in compressed columns,
/// and where each entry of the triplet list it was built from lands: the
/// assembly and equilibration every [`SymmetricSolver`] here shares, CPU or
/// GPU. Compressed lower columns are compressed upper rows, which is how a
/// GPU solver (cuDSS, `CUDSS_MVIEW_UPPER`) takes the same arrays.
pub struct LowerPattern {
    n: usize,
    /// The `(row, col)` of every triplet the pattern was built from, in
    /// order: a matrix matches if its triplets have exactly these.
    keys: Vec<(usize, usize)>,
    /// Per triplet, its slot in `row_idx`, or `None` above the diagonal.
    slot: Vec<Option<usize>>,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
}

impl LowerPattern {
    /// The `n × n` pattern of `keys` (row, column), entries above the
    /// diagonal ignored, duplicates sharing a slot, the diagonal always in.
    pub fn new(n: usize, keys: &[(usize, usize)]) -> Self {
        let mut lower: Vec<(usize, usize)> = keys
            .iter()
            .copied()
            .filter(|&(r, c)| r >= c)
            .chain((0..n).map(|k| (k, k)))
            .collect();
        lower.sort_unstable_by_key(|&(r, c)| (c, r));
        lower.dedup();
        let mut col_ptr = vec![0usize; n + 1];
        for &(_, c) in &lower {
            col_ptr[c + 1] += 1;
        }
        for c in 0..n {
            col_ptr[c + 1] += col_ptr[c];
        }
        let row_idx: Vec<usize> = lower.iter().map(|&(r, _)| r).collect();
        let position = |r: usize, c: usize| -> usize {
            let column = &row_idx[col_ptr[c]..col_ptr[c + 1]];
            col_ptr[c] + column.binary_search(&r).expect("entry in the pattern")
        };
        let slot = keys
            .iter()
            .map(|&(r, c)| (r >= c).then(|| position(r, c)))
            .collect();
        Self {
            n,
            keys: keys.to_vec(),
            slot,
            col_ptr,
            row_idx,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Column pointers of the compressed lower triangle (`n + 1`).
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    /// Row indices of the compressed lower triangle, sorted per column.
    pub fn row_idx(&self) -> &[usize] {
        &self.row_idx
    }

    /// Whether `entries` has exactly this pattern's triplets, in order.
    pub fn matches(&self, n: usize, entries: &[(usize, usize, f64)]) -> bool {
        n == self.n
            && entries.len() == self.keys.len()
            && entries
                .iter()
                .zip(&self.keys)
                .all(|(&(r, c, _), &k)| (r, c) == k)
    }

    /// The compressed lower triangle of `entries` (which must match this
    /// pattern), equilibrated, and the equilibration's scale `D`: the
    /// values are those of `D K D`.
    pub fn assemble(&self, entries: &[(usize, usize, f64)]) -> (Vec<f64>, Vec<f64>) {
        debug_assert_eq!(entries.len(), self.keys.len());
        let mut values = vec![0.0; self.row_idx.len()];
        for (&(_, _, v), slot) in entries.iter().zip(&self.slot) {
            if let Some(s) = *slot {
                values[s] += v;
            }
        }
        let scale = self.equilibrate(&mut values);
        (values, scale)
    }

    /// A symmetric equilibration `D` (Ruiz's iteration, five passes), applied
    /// to `values` in place: every row and column of `D K D` has its largest
    /// entry close to 1. An interior point's KKT matrix spans many orders of
    /// magnitude on one diagonal (the barrier's Σ runs from 1e-8 to 1e11 near
    /// a solution), which costs an unpivoted factorization the accuracy of
    /// the small components; IPOPT's MUMPS scales the same matrix before
    /// factorizing it. `D K D` is congruent to `K`, so the inertia is
    /// unchanged. Returns `D`.
    fn equilibrate(&self, values: &mut [f64]) -> Vec<f64> {
        const PASSES: usize = 5;
        let n = self.n;
        let mut scale = vec![1.0f64; n];
        // Each pass scales by the row maxima, and the sweep that scales also
        // collects the next pass's maxima: six sweeps over the values instead
        // of ten, the same arithmetic in the same order.
        let mut row_max = vec![0.0f64; n];
        for c in 0..n {
            let column = self.col_ptr[c]..self.col_ptr[c + 1];
            for (v, &r) in values[column.clone()].iter().zip(&self.row_idx[column]) {
                let a = v.abs();
                row_max[r] = row_max[r].max(a);
                row_max[c] = row_max[c].max(a);
            }
        }
        let mut factor = vec![1.0f64; n];
        for pass in 0..PASSES {
            for i in 0..n {
                factor[i] = if row_max[i] > 0.0 && row_max[i].is_finite() {
                    1.0 / row_max[i].sqrt()
                } else {
                    1.0
                };
                scale[i] *= factor[i];
                row_max[i] = 0.0;
            }
            let last = pass + 1 == PASSES;
            for c in 0..n {
                for s in self.col_ptr[c]..self.col_ptr[c + 1] {
                    let r = self.row_idx[s];
                    values[s] *= factor[r] * factor[c];
                    if !last {
                        let a = values[s].abs();
                        row_max[r] = row_max[r].max(a);
                        row_max[c] = row_max[c].max(a);
                    }
                }
            }
        }
        scale
    }
}

/// A [`LowerPattern`] with its symbolic `LDLᵀ` (AMD ordering, elimination
/// tree, supernodes), analysed once. Factorizing a matrix of the same
/// pattern again ([`factorize`](Self::factorize)) is then numeric only, as
/// an interior point's KKT matrix is every iteration: its values change, its
/// pattern does not. That split is also what a GPU factorization (cuDSS) is
/// built around: analysis once, refactorization per iteration.
pub struct SymmetricPattern {
    n: usize,
    lower: LowerPattern,
    symbolic: Rc<SymbolicCholesky<usize>>,
}

impl SymmetricPattern {
    /// Analyses the `n × n` pattern of `keys` (row, column), entries above the
    /// diagonal ignored, duplicates sharing a slot. `None` if the symbolic
    /// analysis fails.
    pub fn new(n: usize, keys: &[(usize, usize)]) -> Option<Self> {
        let lower = LowerPattern::new(n, keys);
        let structure = SymbolicSparseColMat::<usize>::new_checked(
            n,
            n,
            lower.col_ptr.clone(),
            None,
            lower.row_idx.clone(),
        );
        let symbolic = factorize_symbolic_cholesky(
            structure.as_ref(),
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .ok()?;
        Some(Self {
            n,
            lower,
            symbolic: Rc::new(symbolic),
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Whether `entries` has exactly this pattern's triplets, in order.
    pub fn matches(&self, n: usize, entries: &[(usize, usize, f64)]) -> bool {
        self.lower.matches(n, entries)
    }

    /// Factorizes the matrix of `entries`, which must match this pattern
    /// (see [`matches`](Self::matches)): the values are summed into their
    /// slots, equilibrated, and factorized numerically. `None` on a zero
    /// pivot (see the module docs).
    pub fn factorize(&self, entries: &[(usize, usize, f64)]) -> Option<SymmetricFactorization> {
        let n = self.n;
        let (values, scale) = self.lower.assemble(entries);
        let structure = SymbolicSparseColMatRef::new_checked(
            n,
            n,
            &self.lower.col_ptr,
            None,
            &self.lower.row_idx,
        );
        let matrix = SparseColMatRef::new(structure, &values);
        let mut factor = vec![0.0; self.symbolic.len_val()];
        let mut mem = MemBuffer::new(
            self.symbolic
                .factorize_numeric_ldlt_scratch::<f64>(Par::Seq, Default::default()),
        );
        self.symbolic
            .factorize_numeric_ldlt(
                &mut factor,
                matrix,
                Side::Lower,
                LdltRegularization::default(),
                Par::Seq,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .ok()?;
        let inertia = count(&diagonal(&self.symbolic, &factor));
        Some(SymmetricFactorization {
            n,
            symbolic: Rc::clone(&self.symbolic),
            values: factor,
            inertia,
            scale,
        })
    }
}

/// A sparse symmetric linear solver for the matrices of one pattern: analysed
/// when made, factorized numerically as often as the values change, and
/// asked for the inertia and for solves in between.
///
/// Stateful (the factorization stays inside) because a GPU backend keeps its
/// factor on the device: the caller passes values in and gets solutions out,
/// and never holds the factor itself.
pub trait SymmetricSolver {
    /// Whether `entries` has the pattern this solver was made for, so that
    /// [`factorize`](Self::factorize) may take it.
    fn matches(&self, n: usize, entries: &[(usize, usize, f64)]) -> bool;

    /// Factorizes the matrix of `entries` (duplicates summed, entries above
    /// the diagonal ignored), replacing the previous factorization. `false`
    /// on a zero pivot; no solve is valid until a factorization succeeds.
    fn factorize(&mut self, entries: &[(usize, usize, f64)]) -> bool;

    /// The inertia of the last factorization.
    fn inertia(&self) -> Inertia;

    /// Solves `K x = b` with the last factorization. `None` if there is none
    /// or the result is not finite.
    fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>>;
}

/// [`SymmetricSolver`] on `faer`'s unpivoted `LDLᵀ`: a [`SymmetricPattern`]
/// and the latest [`SymmetricFactorization`] of it.
pub struct LdltSolver {
    pattern: SymmetricPattern,
    factor: Option<SymmetricFactorization>,
}

impl LdltSolver {
    /// Analyses the `n × n` pattern of `keys` (see [`SymmetricPattern::new`]).
    pub fn new(n: usize, keys: &[(usize, usize)]) -> Option<Self> {
        Some(Self {
            pattern: SymmetricPattern::new(n, keys)?,
            factor: None,
        })
    }
}

impl SymmetricSolver for LdltSolver {
    fn matches(&self, n: usize, entries: &[(usize, usize, f64)]) -> bool {
        self.pattern.matches(n, entries)
    }

    fn factorize(&mut self, entries: &[(usize, usize, f64)]) -> bool {
        self.factor = self.pattern.factorize(entries);
        self.factor.is_some()
    }

    fn inertia(&self) -> Inertia {
        self.factor
            .as_ref()
            .map_or_else(Inertia::default, SymmetricFactorization::inertia)
    }

    fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        self.factor.as_ref()?.solve(rhs)
    }
}

/// [`SymmetricSolver`] on `faer`'s sparse Cholesky `LLᵀ`, for positive
/// definite matrices: a factorization exists exactly when the matrix is
/// positive definite, so success is the inertia `(n, 0, 0)`, and failure
/// says the matrix is not, which is all an interior point on a condensed,
/// positive definite KKT system needs to know. Cheaper than `LDLᵀ` and what
/// GPU solvers do best.
pub struct LltSolver {
    pattern: SymmetricPattern,
    /// The factor's values and the equilibration's scale.
    factor: Option<(Vec<f64>, Vec<f64>)>,
}

impl LltSolver {
    /// Analyses the `n × n` pattern of `keys` (see [`SymmetricPattern::new`]).
    pub fn new(n: usize, keys: &[(usize, usize)]) -> Option<Self> {
        Some(Self {
            pattern: SymmetricPattern::new(n, keys)?,
            factor: None,
        })
    }
}

impl SymmetricSolver for LltSolver {
    fn matches(&self, n: usize, entries: &[(usize, usize, f64)]) -> bool {
        self.pattern.matches(n, entries)
    }

    fn factorize(&mut self, entries: &[(usize, usize, f64)]) -> bool {
        let pattern = &self.pattern;
        let n = pattern.n;
        let (values, scale) = pattern.lower.assemble(entries);
        let structure = SymbolicSparseColMatRef::new_checked(
            n,
            n,
            &pattern.lower.col_ptr,
            None,
            &pattern.lower.row_idx,
        );
        let matrix = SparseColMatRef::new(structure, &values);
        let mut factor = vec![0.0; pattern.symbolic.len_val()];
        let mut mem = MemBuffer::new(
            pattern
                .symbolic
                .factorize_numeric_llt_scratch::<f64>(Par::Seq, Default::default()),
        );
        let ok = pattern
            .symbolic
            .factorize_numeric_llt(
                &mut factor,
                matrix,
                Side::Lower,
                LltRegularization::default(),
                Par::Seq,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .is_ok();
        self.factor = ok.then_some((factor, scale));
        ok
    }

    fn inertia(&self) -> Inertia {
        match self.factor {
            Some(_) => Inertia {
                positive: self.pattern.n,
                negative: 0,
                zero: 0,
            },
            None => Inertia::default(),
        }
    }

    fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        let (values, scale) = self.factor.as_ref()?;
        let n = self.pattern.n;
        let mut x = Mat::<f64>::from_fn(n, 1, |i, _| rhs[i] * scale[i]);
        let mut mem = MemBuffer::new(
            self.pattern
                .symbolic
                .solve_in_place_scratch::<f64>(1, Par::Seq),
        );
        LltRef::new(&self.pattern.symbolic, values).solve_in_place_with_conj(
            Conj::No,
            x.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
        );
        let out: Vec<f64> = (0..n).map(|i| x[(i, 0)] * scale[i]).collect();
        out.iter().all(|v| v.is_finite()).then_some(out)
    }
}

/// `LDLᵀ` of a sparse symmetric matrix, and its inertia.
pub struct SymmetricFactorization {
    n: usize,
    symbolic: Rc<SymbolicCholesky<usize>>,
    values: Vec<f64>,
    inertia: Inertia,
    /// The equilibration `D`: the factorized matrix is `D K D`.
    scale: Vec<f64>,
}

impl SymmetricFactorization {
    /// Factorizes the `n × n` symmetric matrix given by `entries`. Entries
    /// above the diagonal are ignored, so either a full symmetric list or its
    /// lower triangle may be passed; duplicates sum. `None` when the
    /// factorization hits a zero pivot (the matrix is singular, or needs
    /// pivoting this does not do; see the module docs). The analysis is
    /// redone; to reuse it across matrices of one pattern, keep a
    /// [`SymmetricPattern`].
    pub fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        let keys: Vec<(usize, usize)> = entries.iter().map(|&(r, c, _)| (r, c)).collect();
        SymmetricPattern::new(n, &keys)?.factorize(entries)
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
        let mut x = Mat::<f64>::from_fn(self.n, 1, |i, _| rhs[i] * self.scale[i]);
        let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq));
        LdltRef::new(&self.symbolic, &self.values).solve_in_place_with_conj(
            Conj::No,
            x.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
        );
        let out: Vec<f64> = (0..self.n).map(|i| x[(i, 0)] * self.scale[i]).collect();
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

    /// Through the trait: refactorized at new values, the solver gives what a
    /// fresh factorization of those values gives, and a factorization that
    /// fails leaves nothing to solve with.
    #[test]
    fn the_solver_refactorizes_on_its_pattern() {
        let (dim, e) = kkt(40, 15, 10.0, 7);
        let keys: Vec<(usize, usize)> = e.iter().map(|&(r, c, _)| (r, c)).collect();
        let mut solver = LdltSolver::new(dim, &keys).unwrap();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64).cos()).collect();
        for shift in [10.0, 3.0] {
            let (_, values) = kkt(40, 15, shift, 7);
            assert!(solver.matches(dim, &values));
            assert!(solver.factorize(&values));
            let fresh = SymmetricFactorization::new(dim, &values).unwrap();
            assert_eq!(solver.inertia(), fresh.inertia());
            assert_eq!(solver.solve(&b).unwrap(), fresh.solve(&b).unwrap());
        }
        assert!(!solver.matches(dim, &e[1..]));
        let zero: Vec<(usize, usize, f64)> = e.iter().map(|&(r, c, _)| (r, c, 0.0)).collect();
        assert!(!solver.factorize(&zero));
        assert!(solver.solve(&b).is_none());
    }

    /// Cholesky succeeds exactly on a positive definite matrix, and then
    /// solves as `LDLᵀ` does; on an indefinite one it fails. Both layouts.
    #[test]
    fn cholesky_tells_positive_definite_from_not() {
        for (n_x, size) in [(40, 10.0), (300, 6.0)] {
            // The KKT matrix's Hessian block alone is positive definite.
            let (dim, e) = kkt(n_x, 15, size, 5);
            let block: Vec<(usize, usize, f64)> = e
                .iter()
                .copied()
                .filter(|&(r, c, _)| r < n_x && c < n_x)
                .collect();
            let keys: Vec<(usize, usize)> = block.iter().map(|&(r, c, _)| (r, c)).collect();
            let mut llt = LltSolver::new(n_x, &keys).unwrap();
            assert!(llt.factorize(&block));
            assert_eq!(llt.inertia().positive, n_x);
            let b: Vec<f64> = (0..n_x).map(|i| (i as f64).sin()).collect();
            let x = llt.solve(&b).unwrap();
            let y = SymmetricFactorization::new(n_x, &block)
                .unwrap()
                .solve(&b)
                .unwrap();
            let worst = x
                .iter()
                .zip(&y)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(worst < 1e-10, "{worst:e}");
            // The whole KKT matrix is not.
            let keys: Vec<(usize, usize)> = e.iter().map(|&(r, c, _)| (r, c)).collect();
            let mut llt = LltSolver::new(dim, &keys).unwrap();
            assert!(!llt.factorize(&e));
            assert!(llt.solve(&vec![1.0; dim]).is_none());
        }
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
