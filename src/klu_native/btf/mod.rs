//! Block-triangular-form preprocessing — ports
//! `vendor/suitesparse/BTF/Source/btf_order.c`'s `btf_order` exactly:
//! maximum transversal (`maxtrans`) to find a matching that maximizes
//! nonzero diagonal entries, completed arbitrarily for any leftover
//! unmatched rows/columns if the matrix is structurally singular, then
//! strongly-connected-components (`strongcomp`) on the matched graph to find
//! the final block-triangular permutation.

mod maxtrans;
mod strongcomp;

use super::types::{flip, EMPTY};

/// Permutes a square `n`-by-`n` sparse matrix (CSC: `col_ptr`/`row_idx`,
/// `col_ptr.len() == n + 1`) to upper block-triangular form. Returns
/// `(p, q, r, nmatch)`:
/// - `p`/`q`: row/column permutation. `A(p, q)` has a zero-free diagonal if
///   `A` has full structural rank (`nmatch == n`); otherwise `q[k]` is
///   "flipped" (see `types::unflip`) for a column with no real nonzero
///   diagonal entry available.
/// - `r`: block boundaries, `r.len() == nblocks + 1`; block `b` spans
///   rows/columns `r[b]..r[b+1]` of the permuted matrix.
/// - `nmatch`: number of nonzeros on the diagonal of `A(p, q)` — equals `n`
///   iff `A` has full structural rank.
///
/// Ports `btf_order.c`'s `btf_order` exactly, including its "complete the
/// permutation" fallback for structurally singular matrices: leftover
/// unmatched columns are assigned arbitrarily to leftover unmatched rows
/// (each flagged via `flip`, so `q` unflipped is still a bijection even
/// though not every diagonal entry is a real nonzero) before the
/// strongly-connected-components step, which needs a full permutation to
/// operate on.
pub fn btf_order(n: usize, col_ptr: &[i64], row_idx: &[i64]) -> (Vec<i64>, Vec<i64>, Vec<i64>, usize) {
    let (mut q, nmatch) = maxtrans::maxtrans(n, n, col_ptr, row_idx);

    if nmatch < n {
        // Flag all matched columns, then list unmatched ones (descending
        // index order, matching btf_order.c's own Work[] build order) so
        // popping from the end assigns lowest-index columns first.
        let mut matched = vec![false; n];
        for &j in q.iter() {
            if j != EMPTY {
                matched[j as usize] = true;
            }
        }
        let mut leftover_cols: Vec<i64> = (0..n as i64).rev().filter(|&j| !matched[j as usize]).collect();
        for qi in q.iter_mut() {
            if *qi == EMPTY && let Some(j) = leftover_cols.pop() {
                *qi = flip(j);
            }
        }
    }

    let (p, r) = strongcomp::strongcomp(n, col_ptr, row_idx, Some(&mut q));
    (p, q, r, nmatch)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "klu")]
    use crate::klu_native::ffi_oracle;
    use crate::klu_native::types::{is_flipped, unflip};

    fn assert_is_bijection(perm: &[i64], n: usize) {
        let mut sorted: Vec<i64> = perm.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n as i64).collect::<Vec<_>>(), "not a bijection of 0..{n}: {perm:?}");
    }

    fn assert_valid_btf(n: usize, col_ptr: &[i64], row_idx: &[i64], p: &[i64], q: &[i64], r: &[i64], nmatch: usize) {
        assert_is_bijection(p, n);
        let q_unflipped: Vec<i64> = q.iter().map(|&x| unflip(x)).collect();
        assert_is_bijection(&q_unflipped, n);
        assert_eq!(r[0], 0);
        assert_eq!(*r.last().unwrap(), n as i64);
        assert!(r.windows(2).all(|w| w[0] < w[1]), "R must be strictly increasing: {r:?}");

        // Build Pinv (original index -> permuted position).
        let mut pinv = vec![0usize; n];
        for (k, &orig) in p.iter().enumerate() {
            pinv[orig as usize] = k;
        }
        // Every (row, col) entry of the permuted matrix A(p,q) must land at
        // or above the diagonal within its own block, or in an earlier
        // block's column range (upper block triangular).
        for (jnew, &qj) in q.iter().enumerate() {
            let jold = unflip(qj) as usize;
            for &row in &row_idx[col_ptr[jold] as usize..col_ptr[jold + 1] as usize] {
                let inew = pinv[row as usize];
                // Find jnew's block.
                let block = r.windows(2).position(|w| jnew >= w[0] as usize && jnew < w[1] as usize).unwrap();
                let (lo, hi) = (r[block] as usize, r[block + 1] as usize);
                assert!(
                    inew < hi,
                    "entry (row {row}, col {jold}) at permuted ({inew},{jnew}) falls below its block \
                     [{lo},{hi}) -- not upper block triangular"
                );
            }
        }
        if nmatch == n {
            for (k, &qk) in q.iter().enumerate() {
                assert!(!is_flipped(qk), "full rank but q[{k}] is flipped");
            }
        }
    }

    #[test]
    fn diagonal_matrix_three_singleton_blocks() {
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [0, 1, 2];
        let (p, q, r, nmatch) = btf_order(3, &col_ptr, &row_idx);
        assert_eq!(nmatch, 3);
        assert_eq!(r, vec![0, 1, 2, 3]);
        assert_valid_btf(3, &col_ptr, &row_idx, &p, &q, &r, nmatch);
    }

    #[test]
    fn needs_full_permutation() {
        // col0 -> row1, col1 -> row0: needs a swap to get a zero-free diagonal.
        let col_ptr = [0, 1, 2];
        let row_idx = [1, 0];
        let (p, q, r, nmatch) = btf_order(2, &col_ptr, &row_idx);
        assert_eq!(nmatch, 2);
        assert_valid_btf(2, &col_ptr, &row_idx, &p, &q, &r, nmatch);
    }

    #[test]
    fn structurally_singular() {
        // Both columns only ever reference row 0: no perfect matching.
        let col_ptr = [0, 1, 2];
        let row_idx = [0, 0];
        let (p, q, r, nmatch) = btf_order(2, &col_ptr, &row_idx);
        assert_eq!(nmatch, 1);
        assert_valid_btf(2, &col_ptr, &row_idx, &p, &q, &r, nmatch);
    }

    #[test]
    fn mesh_with_two_blocks() {
        // Nodes {0,1} form a cycle (block together); node 2 only depends on
        // node 1 (upper triangular addition) -- should yield 2 blocks.
        // col0 -> row1 ; col1 -> row0, row2 ; col2 -> row2
        let col_ptr = [0, 1, 3, 4];
        let row_idx = [1, 0, 2, 2];
        let (p, q, r, nmatch) = btf_order(3, &col_ptr, &row_idx);
        assert_eq!(nmatch, 3);
        assert_valid_btf(3, &col_ptr, &row_idx, &p, &q, &r, nmatch);
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_ffi_oracle_on_curated_matrices() {
        let cases: Vec<(usize, Vec<i32>, Vec<i32>)> = vec![
            (3, vec![0, 1, 2, 3], vec![0, 1, 2]),                 // diagonal
            (2, vec![0, 1, 2], vec![1, 0]),                       // needs swap
            (2, vec![0, 1, 2], vec![0, 0]),                       // structurally singular
            (3, vec![0, 1, 3, 4], vec![1, 0, 2, 2]),              // 2-block mesh
            (5, vec![0, 2, 4, 6, 8, 10], vec![0, 4, 1, 0, 2, 1, 3, 2, 4, 3]), // ring
        ];
        for (n, col_ptr, row_idx) in cases {
            let col_ptr64: Vec<i64> = col_ptr.iter().map(|&x| x as i64).collect();
            let row_idx64: Vec<i64> = row_idx.iter().map(|&x| x as i64).collect();
            let (rp, rq, rr, rnmatch) = btf_order(n, &col_ptr64, &row_idx64);

            let (cp, cq, cr, cnmatch) = ffi_oracle::btf_order_oracle(n, &col_ptr, &row_idx);
            let cp64: Vec<i64> = cp.iter().map(|&x| x as i64).collect();
            let cq64: Vec<i64> = cq.iter().map(|&x| x as i64).collect();
            let cr64: Vec<i64> = cr.iter().map(|&x| x as i64).collect();

            assert_eq!(rnmatch as i32, cnmatch, "nmatch mismatch for n={n}");
            assert_eq!(rr, cr64, "block boundaries R mismatch for n={n}");
            assert_eq!(rp, cp64, "P mismatch for n={n}");
            assert_eq!(rq, cq64, "Q mismatch for n={n}");
        }
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_ffi_oracle_on_random_matrices() {
        // Fixed, hand-rolled LCG (no extra dependency) for determinism --
        // same style as block_sparse.rs's own random-matrix tests.
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut next_f64 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };

        for trial in 0..40 {
            let n = 2 + (trial % 12);
            // Random directed graph: each column gets 1..=3 random row
            // entries (deduplicated), deliberately including some
            // structurally-singular cases (skip the diagonal sometimes) and
            // some fully-connected cases (dense small n) to exercise both
            // BTF's full-matching and partial-matching paths.
            let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
            for (j, col) in cols.iter_mut().enumerate() {
                let mut rows = std::collections::BTreeSet::new();
                if next_f64() > 0.2 {
                    rows.insert(j); // usually include the diagonal
                }
                let degree = 1 + (next_f64() * 3.0) as usize;
                for _ in 0..degree {
                    rows.insert((next_f64() * n as f64) as usize);
                }
                *col = rows.into_iter().collect();
            }
            let mut col_ptr = vec![0i32; n + 1];
            let mut row_idx: Vec<i32> = Vec::new();
            for (j, col) in cols.iter().enumerate() {
                for &r in col {
                    row_idx.push(r as i32);
                }
                col_ptr[j + 1] = row_idx.len() as i32;
            }

            let col_ptr64: Vec<i64> = col_ptr.iter().map(|&x| x as i64).collect();
            let row_idx64: Vec<i64> = row_idx.iter().map(|&x| x as i64).collect();
            let (rp, rq, rr, rnmatch) = btf_order(n, &col_ptr64, &row_idx64);
            assert_valid_btf(n, &col_ptr64, &row_idx64, &rp, &rq, &rr, rnmatch);

            let (cp, cq, cr, cnmatch) = ffi_oracle::btf_order_oracle(n, &col_ptr, &row_idx);
            let cp64: Vec<i64> = cp.iter().map(|&x| x as i64).collect();
            let cq64: Vec<i64> = cq.iter().map(|&x| x as i64).collect();
            let cr64: Vec<i64> = cr.iter().map(|&x| x as i64).collect();

            assert_eq!(rnmatch as i32, cnmatch, "trial {trial} (n={n}): nmatch mismatch");
            assert_eq!(rr, cr64, "trial {trial} (n={n}): block boundaries R mismatch");
            assert_eq!(rp, cp64, "trial {trial} (n={n}): P mismatch");
            assert_eq!(rq, cq64, "trial {trial} (n={n}): Q mismatch");
        }
    }
}
