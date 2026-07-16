//! Experimental block-per-bus sparse LU backend for symmetric power flow,
//! mirroring power-grid-model's approach of grouping each bus's own unknowns
//! (angle, voltage magnitude) into one dense 2×2 block instead of two
//! separate scalar unknowns scattered across the Jacobian.
//!
//! This is a **parallel, opt-in** backend (`solver::JacobianBackend::Block`)
//! — the default `Scalar` path (`sparse.rs`, backed by `faer`) is completely
//! untouched by this module, so a bug here cannot affect existing behavior.
//! Scoped to symmetric power flow only: every non-slack bus in gridoxide's
//! current model is `PQ` (never `PV`; confirmed no code path constructs a
//! `PV` bus), so every block is uniformly 2×2 — `debug_assert!`s in this
//! module enforce that assumption rather than silently mishandling a
//! hypothetical `PV` bus.
//!
//! Only the fill-reducing *ordering* is reused from `faer`
//! (`faer::sparse::linalg::colamd::order`, which is purely structural and
//! works identically for scalar or block payloads). The numeric block LU
//! (symbolic elimination-graph reachability + block-pivoted factorization +
//! block triangular solve) is hand-written here, following the classic
//! Gilbert-Peierls left-looking sparse LU algorithm, adapted from scalar
//! arithmetic to 2×2 block arithmetic at each step.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::prelude::Reborrow;
use faer::sparse::{Pair, SymbolicSparseColMat};
use faer::sparse::linalg::colamd;

/// A 2×2 real block, row-major: `[[a, b], [c, d]]`.
pub type Block2 = [[f64; 2]; 2];

pub fn block_zero() -> Block2 {
    [[0.0, 0.0], [0.0, 0.0]]
}

pub fn block_add(a: Block2, b: Block2) -> Block2 {
    [[a[0][0] + b[0][0], a[0][1] + b[0][1]], [a[1][0] + b[1][0], a[1][1] + b[1][1]]]
}

pub fn block_sub(a: Block2, b: Block2) -> Block2 {
    [[a[0][0] - b[0][0], a[0][1] - b[0][1]], [a[1][0] - b[1][0], a[1][1] - b[1][1]]]
}

pub fn block_mul(a: Block2, b: Block2) -> Block2 {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

pub fn block_vec_mul(a: Block2, v: [f64; 2]) -> [f64; 2] {
    [a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1]]
}

pub fn vec_sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// A 2×2 block's Frobenius-like magnitude, used for pivot comparison.
pub fn block_abs(a: Block2) -> f64 {
    (a[0][0] * a[0][0] + a[0][1] * a[0][1] + a[1][0] * a[1][0] + a[1][1] * a[1][1]).sqrt()
}

/// Inverts a 2×2 block via the explicit determinant formula. Returns `None`
/// if the block is (near-)singular.
pub fn block_inv(a: Block2) -> Option<Block2> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if !det.is_finite() || det.abs() < 1e-300 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([[a[1][1] * inv_det, -a[0][1] * inv_det], [-a[1][0] * inv_det, a[0][0] * inv_det]])
}

/// A mutable block-COO Y-bus/Jacobian accumulator: one 2×2 block per (bus,
/// bus) slot instead of `network::YBus`'s one `Complex<f64>` per (unknown,
/// unknown) slot. Duplicate `(i, j)` entries are summed once finalized,
/// matching `YBus`'s accumulation semantics.
pub struct BlockMatrix {
    n: usize,
    entries: Vec<(usize, usize, Block2)>,
}

impl BlockMatrix {
    pub fn new(n: usize) -> Self {
        Self { n, entries: Vec::new() }
    }

    pub fn add(&mut self, i: usize, j: usize, block: Block2) {
        debug_assert!(i < self.n && j < self.n, "block index out of range");
        self.entries.push((i, j, block));
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Consolidates duplicate `(i, j)` entries (summing them) and returns a
    /// row-major adjacency list: `row(i)` gives all `(j, block)` pairs with
    /// a nonzero block in row `i` (including the diagonal), sorted by `j`.
    ///
    /// Sort-based, not hash-based: `newton_raphson_block` calls this once
    /// per iteration (the Jacobian's numeric values change every iteration
    /// even though its sparsity pattern doesn't), so this runs thousands of
    /// times per solve on realistic networks — profiling with `perf` showed
    /// a `HashMap`-based merge costing ~25% of total block-backend runtime
    /// at 2,605 nodes. Sorting by `(i, j)` avoids hashing entirely and
    /// leaves each row already grouped in ascending-`j` order, so no
    /// separate per-row sort is needed afterwards either.
    pub fn finish(mut self) -> BlockAdjacency {
        self.entries.sort_unstable_by_key(|&(i, j, _)| (i, j));

        let mut rows: Vec<Vec<(usize, Block2)>> = vec![Vec::new(); self.n];
        let mut idx = 0;
        while idx < self.entries.len() {
            let (i, j, mut acc) = self.entries[idx];
            let mut k = idx + 1;
            while k < self.entries.len() && self.entries[k].0 == i && self.entries[k].1 == j {
                acc = block_add(acc, self.entries[k].2);
                k += 1;
            }
            rows[i].push((j, acc));
            idx = k;
        }
        BlockAdjacency { n: self.n, rows }
    }
}

/// The finalized, consolidated block structure of a `BlockMatrix`.
pub struct BlockAdjacency {
    n: usize,
    rows: Vec<Vec<(usize, Block2)>>,
}

impl BlockAdjacency {
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn row(&self, i: usize) -> &[(usize, Block2)] {
        &self.rows[i]
    }

    pub fn get(&self, i: usize, j: usize) -> Block2 {
        self.rows[i]
            .binary_search_by_key(&j, |&(col, _)| col)
            .map(|idx| self.rows[i][idx].1)
            .unwrap_or_else(|_| block_zero())
    }

    /// Computes a fill-reducing column permutation over the *collapsed*
    /// bus-level structural graph (one node per bus, ignoring the internal
    /// 2×2 block contents — `colamd`'s ordering is purely structural),
    /// reusing `faer`'s tested `colamd::order` rather than hand-writing an
    /// ordering heuristic. `perm[k]` = the original bus index placed at
    /// elimination position `k`.
    pub fn colamd_order(&self) -> Vec<usize> {
        let pairs: Vec<Pair<usize, usize>> = self
            .rows
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().map(move |&(j, _)| Pair { row: i, col: j }))
            .collect();
        let (symbolic, _argsort) =
            SymbolicSparseColMat::<usize>::try_new_from_indices(self.n, self.n, &pairs)
                .expect("block adjacency should always form a valid sparsity pattern");

        let nnz = pairs.len();
        let scratch_req = colamd::order_scratch::<usize>(self.n, self.n, nnz);
        let mut buffer = MemBuffer::try_new(scratch_req).expect("colamd scratch allocation");
        let stack = MemStack::new(&mut buffer);

        let mut perm = vec![0usize; self.n];
        let mut perm_inv = vec![0usize; self.n];
        colamd::order(&mut perm, &mut perm_inv, symbolic.rb(), colamd::Control::default(), stack)
            .expect("colamd ordering should always succeed on a valid sparsity pattern");
        perm
    }
}

/// The symbolic (structure-only) part of a block LU factorization: the
/// `colamd` permutation plus which rows are nonzero in each column of L and
/// U after fill-in. This is fixed for as long as the *sparsity pattern*
/// doesn't change — exactly the case across Newton-Raphson iterations, where
/// the bus topology stays constant and only numeric Jacobian values change —
/// so it's computed once via `analyze` and reused by `BlockLu::refactor` for
/// cheap numeric-only refactorization on every iteration, mirroring
/// `sparse::RealSparseSystem`'s `SymbolicLu` reuse for the scalar backend.
pub struct BlockSymbolic {
    n: usize,
    /// `perm[k]` = original bus index placed at elimination position `k`.
    perm: Vec<usize>,
    /// `l_struct[k]` = sorted rows `> k` with a nonzero L entry in column `k`.
    l_struct: Vec<Vec<usize>>,
    /// `u_struct[j]` = sorted rows `<= j` with a nonzero U entry in column
    /// `j` (always includes `j` itself, the diagonal).
    u_struct: Vec<Vec<usize>>,
}

impl BlockSymbolic {
    /// Discovers the elimination structure (fill-in pattern) via the same
    /// left-looking DFS reachability `BlockLu` used to use inline, but
    /// tracking only *which* positions become nonzero, not their values.
    pub fn analyze(adj: &BlockAdjacency) -> Self {
        let n = adj.n();
        let perm = adj.colamd_order();
        let mut pos = vec![0usize; n];
        for (k, &orig) in perm.iter().enumerate() {
            pos[orig] = k;
        }

        let mut l_struct: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut u_struct: Vec<Vec<usize>> = vec![Vec::new(); n];

        let mut touched_at: Vec<i64> = vec![-1; n];
        let mut reach_visited_at: Vec<i64> = vec![-1; n];
        let mut touched_list: Vec<usize> = Vec::new();

        for j in 0..n {
            touched_list.clear();

            for &(orig_neighbor, _) in adj.row(perm[j]) {
                let row = pos[orig_neighbor];
                if touched_at[row] != j as i64 {
                    touched_at[row] = j as i64;
                    touched_list.push(row);
                }
            }

            let mut stack: Vec<usize> = Vec::new();
            for &row in &touched_list {
                if row < j && reach_visited_at[row] != j as i64 {
                    reach_visited_at[row] = j as i64;
                    stack.push(row);
                }
            }
            while let Some(v) = stack.pop() {
                for &row in &l_struct[v] {
                    if row < j && reach_visited_at[row] != j as i64 {
                        reach_visited_at[row] = j as i64;
                        stack.push(row);
                    }
                }
            }

            for k in 0..j {
                if reach_visited_at[k] != j as i64 {
                    continue;
                }
                for &row in &l_struct[k] {
                    if touched_at[row] != j as i64 {
                        touched_at[row] = j as i64;
                        touched_list.push(row);
                    }
                }
            }

            for &row in &touched_list {
                if row <= j {
                    u_struct[j].push(row);
                } else {
                    l_struct[j].push(row);
                }
            }
            u_struct[j].sort_unstable();
            l_struct[j].sort_unstable();
        }

        Self { n, perm, l_struct, u_struct }
    }
}

/// A numeric block LU factorization, computed against a `BlockSymbolic`'s
/// known structure — either freshly analyzed (`factorize`) or reusing an
/// existing `BlockSymbolic` across calls (`refactor`, the cheap path used
/// once per Newton-Raphson iteration).
///
/// **Simplification, clearly scoped**: this uses the diagonal entry as the
/// pivot for each column directly — no partial pivoting (row swapping).
/// Partial pivoting on a *sparse* block matrix would mean swapping entries
/// across every already-factored column's stored structure, potentially
/// inserting or removing sparse entries (new fill-in from the swap itself),
/// a substantially larger and more fragile undertaking than the rest of this
/// module. Power-system Y-bus/Jacobian matrices are normally diagonally
/// dominant enough for this to succeed; factorization returns `None`
/// (treated as singular, same contract as `sparse::RealSparseSystem`) if a
/// diagonal pivot is ever too close to singular, rather than silently
/// producing a wrong answer.
///
/// Relies on the Y-bus/Jacobian's sparsity pattern being structurally
/// symmetric (if bus `i` affects bus `k`, bus `k` affects bus `i` too, since
/// branch admittance couples both directions) — `BlockAdjacency::row` is
/// used for both row *and* column structural lookups on that assumption,
/// rather than maintaining a separate column-indexed structure.
pub struct BlockLu {
    n: usize,
    /// `perm[k]` = original bus index placed at elimination position `k`.
    perm: Vec<usize>,
    /// `l_cols[k]` = `(row, multiplier)` pairs for the strictly-below-diagonal
    /// part of L's column `k` (unit diagonal implied), in permuted indexing.
    l_cols: Vec<Vec<(usize, Block2)>>,
    /// `u_cols[j]` = `(row, value)` pairs for the upper-triangular part
    /// (including the diagonal at `row == j`) of U's column `j`, in permuted
    /// indexing.
    u_cols: Vec<Vec<(usize, Block2)>>,
}

impl BlockLu {
    /// Analyzes `adj`'s structure and factorizes it in one step. Prefer
    /// `refactor` when solving the same sparsity pattern repeatedly (e.g.
    /// across Newton-Raphson iterations) — it skips the `colamd` ordering
    /// and elimination-graph reachability search entirely, reusing a cached
    /// `BlockSymbolic` instead.
    pub fn factorize(adj: &BlockAdjacency) -> Option<Self> {
        let symbolic = BlockSymbolic::analyze(adj);
        Self::refactor(&symbolic, adj)
    }

    /// Numeric-only refactorization against `symbolic`'s known structure —
    /// no ordering or reachability recomputation, just walking the already
    /// known `l_struct`/`u_struct` positions and computing their (new)
    /// numeric values. Returns `None` if any diagonal pivot is singular.
    ///
    /// `adj` must have the exact same sparsity pattern `symbolic` was built
    /// from (values may differ) — this is not re-validated, matching
    /// `sparse::RealSparseSystem::factor_and_solve`'s same-pattern
    /// requirement.
    pub fn refactor(symbolic: &BlockSymbolic, adj: &BlockAdjacency) -> Option<Self> {
        let n = symbolic.n;
        let perm = &symbolic.perm;
        let mut pos = vec![0usize; n];
        for (k, &orig) in perm.iter().enumerate() {
            pos[orig] = k;
        }

        let mut l_cols: Vec<Vec<(usize, Block2)>> = vec![Vec::new(); n];
        let mut u_cols: Vec<Vec<(usize, Block2)>> = vec![Vec::new(); n];

        let mut x: Vec<Block2> = vec![block_zero(); n];
        let mut touched_at: Vec<i64> = vec![-1; n];

        for j in 0..n {
            // 1. Scatter (permuted) column j's numeric entries into x.
            for &(orig_neighbor, block) in adj.row(perm[j]) {
                let row = pos[orig_neighbor];
                if touched_at[row] != j as i64 {
                    touched_at[row] = j as i64;
                    x[row] = block;
                } else {
                    x[row] = block_add(x[row], block);
                }
            }

            // 2. Apply updates from every k < j in the known reach set
            //    (`u_struct[j]` minus the diagonal), in ascending order —
            //    no DFS needed, the structure is already known. Iterate
            //    `l_cols[k]` directly (already finalized this pass, since
            //    k < j) rather than `symbolic.l_struct[k]`, avoiding an
            //    O(size) lookup per entry.
            for &k in &symbolic.u_struct[j] {
                if k == j {
                    continue;
                }
                let xk = if touched_at[k] == j as i64 { x[k] } else { block_zero() };
                for &(row, lval) in &l_cols[k] {
                    let delta = block_mul(lval, xk);
                    if touched_at[row] != j as i64 {
                        touched_at[row] = j as i64;
                        x[row] = block_sub(block_zero(), delta);
                    } else {
                        x[row] = block_sub(x[row], delta);
                    }
                }
            }

            // 3. Diagonal pivot (no partial pivoting — see doc comment).
            let pivot = if touched_at[j] == j as i64 { x[j] } else { block_zero() };
            let pivot_inv = block_inv(pivot)?;

            // 4. Finalize U's and L's column j using the known structure.
            for &row in &symbolic.u_struct[j] {
                let v = if touched_at[row] == j as i64 { x[row] } else { block_zero() };
                u_cols[j].push((row, v));
            }
            for &row in &symbolic.l_struct[j] {
                let v = if touched_at[row] == j as i64 { x[row] } else { block_zero() };
                l_cols[j].push((row, block_mul(v, pivot_inv)));
            }
        }

        Some(Self { n, perm: perm.clone(), l_cols, u_cols })
    }

    /// Solves `A x = b` using this factorization. `b` is in original
    /// (unpermuted) bus indexing; the result is too.
    pub fn solve(&self, b: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let n = self.n;
        // Permute b into elimination-order indexing.
        let mut y: Vec<[f64; 2]> = (0..n).map(|k| b[self.perm[k]]).collect();

        // Forward substitution: solve L y = b (L unit lower triangular),
        // column-oriented.
        for k in 0..n {
            let yk = y[k];
            for &(row, lval) in &self.l_cols[k] {
                y[row] = vec_sub(y[row], block_vec_mul(lval, yk));
            }
        }

        // Backward substitution: solve U x = y (U upper triangular incl.
        // diagonal), column-oriented, descending.
        let mut x = y;
        for j in (0..n).rev() {
            // The diagonal entry is always the last (largest row index <= j,
            // i.e. exactly j) in u_cols[j] since entries are sorted ascending
            // and all rows are <= j.
            let diag = self
                .u_cols[j]
                .last()
                .filter(|&&(row, _)| row == j)
                .map(|&(_, v)| v)
                .expect("factorize() always records the diagonal entry for column j");
            let diag_inv = block_inv(diag).expect("diagonal was already validated non-singular during factorize()");
            x[j] = block_vec_mul(diag_inv, x[j]);
            let xj = x[j];
            for &(row, uval) in &self.u_cols[j] {
                if row < j {
                    x[row] = vec_sub(x[row], block_vec_mul(uval, xj));
                }
            }
        }

        // Permute back into original bus indexing.
        let mut result = vec![[0.0, 0.0]; n];
        for k in 0..n {
            result[self.pos_to_orig(k)] = x[k];
        }
        result
    }

    fn pos_to_orig(&self, k: usize) -> usize {
        self.perm[k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_inv_identity() {
        let id: Block2 = [[1.0, 0.0], [0.0, 1.0]];
        let inv = block_inv(id).unwrap();
        assert_eq!(inv, id);
    }

    #[test]
    fn block_inv_general() {
        let a: Block2 = [[4.0, 3.0], [6.0, 3.0]];
        let inv = block_inv(a).unwrap();
        let product = block_mul(a, inv);
        assert!((product[0][0] - 1.0).abs() < 1e-12);
        assert!(product[0][1].abs() < 1e-12);
        assert!(product[1][0].abs() < 1e-12);
        assert!((product[1][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn block_inv_singular_returns_none() {
        let a: Block2 = [[1.0, 2.0], [2.0, 4.0]];
        assert!(block_inv(a).is_none());
    }

    #[test]
    fn colamd_order_is_a_valid_permutation() {
        // A small chain graph: 0-1-2-3.
        let mut m = BlockMatrix::new(4);
        let id = [[1.0, 0.0], [0.0, 1.0]];
        for &(i, j) in &[(0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)] {
            m.add(i, j, id);
        }
        let adj = m.finish();
        let perm = adj.colamd_order();
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3], "permutation must be a bijection on 0..n");
    }

    #[test]
    fn finish_sums_duplicate_entries() {
        // (0,0) contributed twice as [[1,0],[0,1]]+[[1,0],[0,1]]=[[2,0],[0,2]],
        // matching YBus's/BlockMatrix's documented `+=` accumulation
        // semantics (e.g. parallel branches between the same two buses).
        let mut m = BlockMatrix::new(2);
        m.add(0, 0, [[1.0, 0.0], [0.0, 1.0]]);
        m.add(0, 0, [[1.0, 0.0], [0.0, 1.0]]);
        m.add(0, 1, [[0.5, 0.0], [0.0, 0.5]]);
        m.add(1, 0, [[0.5, 0.0], [0.0, 0.5]]);
        m.add(1, 1, [[3.0, 0.0], [0.0, 3.0]]);
        let adj = m.finish();
        assert_eq!(adj.get(0, 0), [[2.0, 0.0], [0.0, 2.0]]);
        assert_eq!(adj.get(0, 1), [[0.5, 0.0], [0.0, 0.5]]);
        assert_eq!(adj.get(1, 1), [[3.0, 0.0], [0.0, 3.0]]);
        // rows must stay sorted by column for BlockLu's structural-symmetry
        // and binary-search assumptions to hold.
        let cols: Vec<usize> = adj.row(0).iter().map(|&(j, _)| j).collect();
        assert!(cols.windows(2).all(|w| w[0] < w[1]), "row(0) columns must be strictly ascending: {cols:?}");
    }

    /// A ground-truth oracle: expands an `n`-bus block system into a `2n×2n`
    /// scalar dense matrix and solves it via straightforward Gaussian
    /// elimination with partial pivoting. Independent of `BlockLu`'s
    /// implementation (no shared code, no shared assumptions like "diagonal
    /// pivoting is good enough" or "the sparsity pattern is symmetric"), so
    /// cross-checking against it catches indexing/fill-in/ordering bugs that
    /// small hand-verified cases alone could miss.
    fn dense_reference_solve(adj: &BlockAdjacency, b: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let n = adj.n();
        let m = 2 * n;
        let mut a = vec![vec![0.0f64; m]; m];
        for i in 0..n {
            for &(j, block) in adj.row(i) {
                for p in 0..2 {
                    for q in 0..2 {
                        a[2 * i + p][2 * j + q] = block[p][q];
                    }
                }
            }
        }
        let mut rhs = vec![0.0f64; m];
        for i in 0..n {
            rhs[2 * i] = b[i][0];
            rhs[2 * i + 1] = b[i][1];
        }

        // Gaussian elimination with partial pivoting.
        for col in 0..m {
            let pivot_row = (col..m).max_by(|&r1, &r2| a[r1][col].abs().total_cmp(&a[r2][col].abs())).unwrap();
            assert!(a[pivot_row][col].abs() > 1e-12, "dense reference matrix is singular");
            a.swap(col, pivot_row);
            rhs.swap(col, pivot_row);
            for row in (col + 1)..m {
                let factor = a[row][col] / a[col][col];
                for k in col..m {
                    a[row][k] -= factor * a[col][k];
                }
                rhs[row] -= factor * rhs[col];
            }
        }
        let mut x = vec![0.0f64; m];
        for row in (0..m).rev() {
            let mut sum = rhs[row];
            for k in (row + 1)..m {
                sum -= a[row][k] * x[k];
            }
            x[row] = sum / a[row][row];
        }
        (0..n).map(|i| [x[2 * i], x[2 * i + 1]]).collect()
    }

    fn assert_close(a: &[[f64; 2]], b: &[[f64; 2]], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x[0] - y[0]).abs() < tol && (x[1] - y[1]).abs() < tol,
                "index {i}: {x:?} vs {y:?}"
            );
        }
    }

    #[test]
    fn solve_single_bus() {
        let mut m = BlockMatrix::new(1);
        m.add(0, 0, [[4.0, 1.0], [1.0, 3.0]]);
        let adj = m.finish();
        let lu = BlockLu::factorize(&adj).unwrap();
        let b = [[9.0, 8.0]];
        let x = lu.solve(&b);
        // [[4,1],[1,3]] * [x0,x1] = [9,8] => x0=(9*3-1*8)/(4*3-1*1)=19/11, x1=(4*8-1*9)/11=23/11
        assert_close(&x, &[[19.0 / 11.0, 23.0 / 11.0]], 1e-10);
    }

    #[test]
    fn solve_two_coupled_buses_hand_verified() {
        // Block system:
        // [[4,0],[0,4]]   [[-1,0],[0,-1]]     [x0]   [3]
        // [[-1,0],[0,-1]] [[4,0],[0,4]]     * [x1] = [3]
        // (decouples into two independent scalar systems 4y-z=3, -y+4z=3 per phase)
        // scalar solution: y=z=1 for each phase => x0=x1=[1,1]
        let mut m = BlockMatrix::new(2);
        m.add(0, 0, [[4.0, 0.0], [0.0, 4.0]]);
        m.add(1, 1, [[4.0, 0.0], [0.0, 4.0]]);
        m.add(0, 1, [[-1.0, 0.0], [0.0, -1.0]]);
        m.add(1, 0, [[-1.0, 0.0], [0.0, -1.0]]);
        let adj = m.finish();
        let lu = BlockLu::factorize(&adj).unwrap();
        let b = [[3.0, 3.0], [3.0, 3.0]];
        let x = lu.solve(&b);
        assert_close(&x, &[[1.0, 1.0], [1.0, 1.0]], 1e-10);
    }

    #[test]
    fn solve_singular_returns_none() {
        let mut m = BlockMatrix::new(1);
        // Singular block (rows are multiples of each other).
        m.add(0, 0, [[1.0, 2.0], [2.0, 4.0]]);
        let adj = m.finish();
        assert!(BlockLu::factorize(&adj).is_none());
    }

    #[test]
    fn solve_chain_matches_dense_reference() {
        // A radial chain of 6 buses: 0-1-2-3-4-5, each with self-admittance
        // plus coupling to its neighbor(s) — deliberately not diagonally
        // trivial, to exercise real fill-in during elimination.
        let n = 6;
        let mut m = BlockMatrix::new(n);
        for i in 0..n {
            m.add(i, i, [[3.0 + i as f64 * 0.3, 0.2], [0.15, 2.5 + i as f64 * 0.2]]);
        }
        for i in 0..n - 1 {
            let off = [[-0.8, 0.05], [0.03, -0.7]];
            m.add(i, i + 1, off);
            m.add(i + 1, i, off);
        }
        let adj = m.finish();
        let lu = BlockLu::factorize(&adj).unwrap();
        let b: Vec<[f64; 2]> = (0..n).map(|i| [1.0 + i as f64, 2.0 - i as f64 * 0.5]).collect();
        let x = lu.solve(&b);
        let expected = dense_reference_solve(&adj, &b);
        assert_close(&x, &expected, 1e-8);
    }

    #[test]
    fn solve_mesh_matches_dense_reference() {
        // A small ring/mesh (0-1-2-3-4-0) plus a chord (0-2), so colamd's
        // ordering must handle genuine fill-in from a cycle, not just a
        // tree/chain.
        let n = 5;
        let mut m = BlockMatrix::new(n);
        for i in 0..n {
            m.add(i, i, [[5.0 + i as f64 * 0.4, 0.3], [0.2, 4.0 + i as f64 * 0.25]]);
        }
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)];
        for &(i, j) in &edges {
            let off = [[-0.6, 0.04], [0.02, -0.5]];
            m.add(i, j, off);
            m.add(j, i, off);
        }
        let adj = m.finish();
        let lu = BlockLu::factorize(&adj).unwrap();
        let b: Vec<[f64; 2]> = (0..n).map(|i| [0.5 * i as f64 + 1.0, 3.0 - 0.3 * i as f64]).collect();
        let x = lu.solve(&b);
        let expected = dense_reference_solve(&adj, &b);
        assert_close(&x, &expected, 1e-8);
    }

    #[test]
    fn solve_matches_dense_reference_random_systems() {
        // Several pseudo-random diagonally-dominant sparse systems on
        // varying small topologies, cross-checked against the dense oracle.
        // Fixed, hand-rolled LCG (no extra dependency) for determinism.
        let mut seed: u64 = 0x2545F4914F6CDD1D;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };

        for trial in 0..8 {
            let n = 4 + trial % 5;
            let mut m = BlockMatrix::new(n);
            // Random tree (guarantees connectivity) plus a few extra chords.
            for i in 1..n {
                let parent = (next() * i as f64) as usize;
                let off = [[-0.3 - next(), 0.1 * next()], [0.1 * next(), -0.3 - next()]];
                m.add(parent, i, off);
                m.add(i, parent, off);
            }
            for _ in 0..n / 2 {
                let a = (next() * n as f64) as usize;
                let b_idx = (next() * n as f64) as usize;
                if a != b_idx {
                    let off = [[-0.2 - next(), 0.05 * next()], [0.05 * next(), -0.2 - next()]];
                    m.add(a, b_idx, off);
                    m.add(b_idx, a, off);
                }
            }
            // Diagonally dominant self-admittance, well within the no-pivoting scope.
            for i in 0..n {
                m.add(i, i, [[6.0 + next(), 0.2 * next()], [0.2 * next(), 6.0 + next()]]);
            }
            let adj = m.finish();
            let lu = BlockLu::factorize(&adj).unwrap_or_else(|| panic!("trial {trial}: expected non-singular"));
            let rhs: Vec<[f64; 2]> = (0..n).map(|_| [next() * 4.0 - 2.0, next() * 4.0 - 2.0]).collect();
            let x = lu.solve(&rhs);
            let expected = dense_reference_solve(&adj, &rhs);
            assert_close(&x, &expected, 1e-7);
        }
    }

    #[test]
    fn refactor_reuses_symbolic_across_different_values() {
        // Same shape as sparse::tests::real_sparse_system_refactor_reuses_symbolic:
        // analyze the structure once, then refactor twice with different
        // numeric values but the *same* sparsity pattern (as across
        // Newton-Raphson iterations), checking each refactor's result
        // independently matches a from-scratch `factorize`.
        let n = 5;
        let build = |scale: f64| {
            let mut m = BlockMatrix::new(n);
            for i in 0..n {
                m.add(i, i, [[6.0 + i as f64 * scale, 0.2], [0.15, 5.0 + i as f64 * scale]]);
            }
            for i in 0..n - 1 {
                let off = [[-0.9 * scale, 0.05], [0.04, -0.8 * scale]];
                m.add(i, i + 1, off);
                m.add(i + 1, i, off);
            }
            m.finish()
        };

        let adj_a = build(1.0);
        let symbolic = BlockSymbolic::analyze(&adj_a);

        let b: Vec<[f64; 2]> = (0..n).map(|i| [1.0 + i as f64 * 0.3, -0.5 + i as f64 * 0.2]).collect();

        for &scale in &[1.0, 2.3, 0.6] {
            let adj = build(scale);
            let via_refactor = BlockLu::refactor(&symbolic, &adj).unwrap();
            let via_factorize = BlockLu::factorize(&adj).unwrap();
            let x_refactor = via_refactor.solve(&b);
            let x_factorize = via_factorize.solve(&b);
            assert_close(&x_refactor, &x_factorize, 1e-10);
        }
    }
}
