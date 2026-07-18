//! Multi-block orchestration — ports `vendor/suitesparse/KLU/Source/
//! klu_factor.c`'s `KLU_factor`/`factor2`: for each BTF diagonal block,
//! extract its local sub-matrix (splitting off entries that belong to an
//! earlier block into the off-diagonal accumulator), factor it
//! (`kernel::factor_block`), and compose the block's own numeric pivot
//! order with the symbolic `P`/`Q` into the final global permutation.
//!
//! **Simplification, confirmed correct by tracing the algorithm, not
//! assumed**: `klu_factor.c` hand-optimizes `nk == 1` diagonal blocks with
//! a separate "singleton" fast path (`factor2`'s `if (nk == 1)` branch) —
//! a pure performance optimization, not a behavioral difference. Tracing
//! `kernel::lpivot` by hand for a single-candidate column confirms the
//! general algorithm already produces the exact same result a singleton
//! fast path would (empty `L` column, `Udiag` = the one diagonal value,
//! trivial `p = [0]`) — so this port always calls `kernel::factor_block`
//! uniformly, regardless of block size, and skips the singleton special
//! case entirely.

use super::analyze::Symbolic;
use super::kernel::{factor_block, BlockFactor};
use super::types::flip;

/// The result of factoring the whole matrix: one `BlockFactor` per BTF
/// block (in block order, matching `Symbolic::r`'s boundaries), the final
/// *numeric* global row permutation (`pnum`/`pinv` — composed from each
/// block's own pivot choices with the symbolic `P`), and the off-diagonal
/// part of the BTF-permuted matrix (entries connecting an earlier block's
/// rows to a later block's column — needed for the block-back-substitution
/// `solve.rs` will implement), with its row indices already translated to
/// final numeric pivot order (mirrors `klu_factor.c`'s own late
/// `Offi[p] = Pinv[Offi[p]]` pass, done once all blocks are known).
pub struct Numeric {
    pub blocks: Vec<BlockFactor>,
    pub pnum: Vec<i64>,
    pub pinv: Vec<i64>,
    /// CSC column pointers into `off_i`/`off_x`, size `n + 1`.
    pub off_p: Vec<i64>,
    pub off_i: Vec<i64>,
    pub off_x: Vec<f64>,
}

/// Factors the whole `n`-by-`n` matrix (CSC: `col_ptr`/`row_idx`/`values`)
/// given its symbolic analysis. Returns `None` if any block is
/// structurally or numerically singular (`halt_if_singular` is always true
/// for gridoxide's fixed `Options`, matching `kernel::factor_block`'s own
/// contract).
pub fn factor(n: usize, col_ptr: &[i64], row_idx: &[i64], values: &[f64], sym: &Symbolic, tol: f64) -> Option<Numeric> {
    // Inverse of the symbolic P: pinv_sym[oldrow] = the symbolic pivot
    // *position* oldrow maps to -- used to decide whether an entry falls
    // inside the current block (position >= k1) or belongs to an earlier
    // block (position < k1, off-diagonal), and to compute each block's
    // local row number. Mirrors factor2's own `Pinv[P[k]] = k` (there
    // called `Pinv`, reused later in KLU_kernel's signature as `PSinv`).
    let mut pinv_sym = vec![0i64; n];
    for (k, &orig) in sym.p.iter().enumerate() {
        pinv_sym[orig] = k as i64;
    }

    let nblocks = sym.nblocks();
    let mut blocks = Vec::with_capacity(nblocks);
    let mut pnum = vec![0i64; n];

    let mut off_p = vec![0i64; n + 1];
    let mut off_i: Vec<i64> = Vec::new();
    let mut off_x: Vec<f64> = Vec::new();
    let mut poff: i64 = 0;

    for block in 0..nblocks {
        let k1 = sym.r[block];
        let k2 = sym.r[block + 1];
        let nk = k2 - k1;

        // Extract each local column's entries, splitting off-diagonal ones
        // (position < k1) via the same negative-sentinel encoding
        // kernel::construct_column expects.
        let mut cols: Vec<Vec<(i64, f64)>> = Vec::with_capacity(nk);
        for k in 0..nk {
            let oldcol = sym.q[k + k1];
            let mut entries = Vec::new();
            for p in col_ptr[oldcol] as usize..col_ptr[oldcol + 1] as usize {
                let oldrow = row_idx[p];
                let newrow = pinv_sym[oldrow as usize];
                let value = values[p];
                if (newrow as usize) < k1 {
                    entries.push((flip(oldrow), value));
                } else {
                    entries.push((newrow - k1 as i64, value));
                }
            }
            cols.push(entries);
        }

        let mut off_diag_out: Vec<Vec<(i64, f64)>> = vec![Vec::new(); nk];
        let bf = factor_block(nk, tol, |k| cols[k].clone(), &mut off_diag_out)?;

        for (k, entries) in off_diag_out.into_iter().enumerate() {
            let global_col = k + k1;
            for (oldrow, value) in entries {
                off_i.push(oldrow);
                off_x.push(value);
                poff += 1;
            }
            off_p[global_col + 1] = poff;
        }

        // Combine this block's own numeric pivot order with the symbolic
        // BTF/AMD row order -- mirrors klu_factor.c's own
        // `Pnum[k+k1] = P[Pblock[k]+k1]`.
        for k in 0..nk {
            let pblk_k = bf.p[k] as usize;
            pnum[k + k1] = sym.p[pblk_k + k1] as i64;
        }

        blocks.push(bf);
    }

    let mut pinv = vec![0i64; n];
    for (k, &orig) in pnum.iter().enumerate() {
        pinv[orig as usize] = k as i64;
    }

    // Apply the final numeric pivot row permutation to the off-diagonal
    // entries (their row indices were left as original global rows above,
    // since the final Pinv isn't known until every block's own pivoting is
    // done) -- mirrors klu_factor.c's own late `Offi[p] = Pinv[Offi[p]]`.
    for i in off_i.iter_mut() {
        *i = pinv[*i as usize];
    }

    Some(Numeric { blocks, pnum, pinv, off_p, off_i, off_x })
}

#[cfg(test)]
mod tests {
    use super::super::analyze::analyze;
    use super::*;

    /// Solves `A x = b` given a full `Numeric`. `Symbolic::r`'s block
    /// boundaries describe an *upper* block triangular matrix -- meaning an
    /// *earlier* block's row-equations may reference a *later* block's
    /// columns (`block(row) <= block(col)` required for any nonzero, the
    /// standard upper-triangular condition applied at block granularity),
    /// the same direction `off_p`/`off_i`/`off_x` are built in (entries
    /// found while extracting a block's own columns, at rows belonging to
    /// an *earlier* block). So blocks must be solved in **reverse** order
    /// (last block first, with no unsolved dependencies) and each block's
    /// off-diagonal entries propagated backward into not-yet-solved
    /// *earlier* blocks' right-hand side once that block's own `x` is
    /// known -- the same right-looking update `solve.rs` will formalize;
    /// written directly here purely to validate `factor()`'s output
    /// end-to-end before that phase exists.
    fn solve(num: &Numeric, r: &[usize], q: &[usize], n: usize, b: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; n];
        for k in 0..n {
            y[k] = b[num.pnum[k] as usize];
        }

        let nblocks = r.len() - 1;
        let mut x = vec![0.0; n];
        for block in (0..nblocks).rev() {
            let k1 = r[block];
            let k2 = r[block + 1];
            let nk = k2 - k1;
            let bf = &num.blocks[block];

            let rhs_local: Vec<f64> = y[k1..k2].to_vec();

            // Forward: L y = rhs_local (block-local pivot order).
            let mut yl = rhs_local;
            for k in 0..nk {
                let yk = yl[k];
                for &(row, lij) in &bf.l_cols[k] {
                    yl[row] -= lij * yk;
                }
            }
            // Backward: U x = y.
            let mut xl = yl;
            for k in (0..nk).rev() {
                xl[k] /= bf.udiag[k];
                let xk = xl[k];
                for &(row, uij) in &bf.u_cols[k] {
                    xl[row] -= uij * xk;
                }
            }
            x[k1..(nk + k1)].copy_from_slice(&xl[..nk]);

            // Propagate this block's off-diagonal contribution backward
            // into not-yet-solved earlier blocks' right-hand side.
            for (global_col, &xk) in x.iter().enumerate().take(k2).skip(k1) {
                for p in num.off_p[global_col] as usize..num.off_p[global_col + 1] as usize {
                    let row = num.off_i[p] as usize;
                    y[row] -= num.off_x[p] * xk;
                }
            }
        }
        // x is currently indexed by pivot position (== column position in
        // the Q-permuted system); map back to original variable indices.
        let mut result = vec![0.0; n];
        for k in 0..n {
            result[q[k]] = x[k];
        }
        result
    }

    fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut a: Vec<Vec<f64>> = a.to_vec();
        let mut rhs = b.to_vec();
        for col in 0..n {
            let pivot_row = (col..n).max_by(|&r1, &r2| a[r1][col].abs().total_cmp(&a[r2][col].abs())).unwrap();
            a.swap(col, pivot_row);
            rhs.swap(col, pivot_row);
            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                #[allow(clippy::needless_range_loop)]
                for k in col..n {
                    a[row][k] -= factor * a[col][k];
                }
                rhs[row] -= factor * rhs[col];
            }
        }
        let mut x = vec![0.0; n];
        for row in (0..n).rev() {
            let mut sum = rhs[row];
            for k in (row + 1)..n {
                sum -= a[row][k] * x[k];
            }
            x[row] = sum / a[row][row];
        }
        x
    }

    fn to_csc(n: usize, entries: &[(usize, usize, f64)]) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let mut by_col: Vec<Vec<(i64, f64)>> = vec![Vec::new(); n];
        for &(r, c, v) in entries {
            by_col[c].push((r as i64, v));
        }
        for col in by_col.iter_mut() {
            col.sort_by_key(|&(r, _)| r);
        }
        let mut col_ptr = vec![0i64; n + 1];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for (c, col) in by_col.iter().enumerate() {
            for &(r, v) in col {
                row_idx.push(r);
                values.push(v);
            }
            col_ptr[c + 1] = row_idx.len() as i64;
        }
        (col_ptr, row_idx, values)
    }

    fn dense_from_entries(n: usize, entries: &[(usize, usize, f64)]) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0; n]; n];
        for &(r, c, v) in entries {
            a[r][c] += v;
        }
        a
    }

    #[test]
    fn single_block_matches_dense() {
        let n = 4;
        let entries = vec![
            (0, 0, 6.0),
            (1, 1, 7.0),
            (2, 2, 8.0),
            (3, 3, 5.0),
            (0, 1, -0.9),
            (1, 0, -0.5),
            (1, 2, -0.6),
            (2, 1, -0.3),
            (2, 3, 0.4),
            (3, 2, -0.2),
            (0, 3, 0.1),
            (3, 0, 0.15),
        ];
        let (col_ptr, row_idx, values) = to_csc(n, &entries);
        let sym = analyze(n, &col_ptr, &row_idx);
        let num = factor(n, &col_ptr, &row_idx, &values, &sym, 1e-3).unwrap();

        let b = vec![1.0, 2.0, -1.0, 0.5];
        let x = solve(&num, &sym.r, &sym.q, n, &b);
        let expected = dense_solve(&dense_from_entries(n, &entries), &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-8, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn genuinely_multi_block() {
        // Upper block triangular by construction: block {0,1} (a 2x2 cycle),
        // block {2} (singleton, depends on block 0), block {3,4} (a 2x2
        // cycle depending on both earlier blocks) -- exercises off-diagonal
        // assembly and multi-block composition, not just a single block.
        let n = 5;
        let entries = vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 5.0),
            (2, 2, 3.0),
            (2, 0, 0.5), // off-diagonal: column 0 (block 0) referenced from row 2 (block 1)
            (3, 3, 6.0),
            (3, 4, 1.0),
            (4, 3, 1.0),
            (4, 4, 7.0),
            (3, 0, 0.2), // off-diagonal into block 0
            (4, 2, 0.3), // off-diagonal into block 1 (the singleton)
        ];
        let (col_ptr, row_idx, values) = to_csc(n, &entries);
        let sym = analyze(n, &col_ptr, &row_idx);
        assert!(sym.nblocks() >= 2, "fixture should produce multiple BTF blocks, got {}", sym.nblocks());

        let num = factor(n, &col_ptr, &row_idx, &values, &sym, 1e-3).unwrap();
        let b = vec![1.0, -2.0, 0.5, 3.0, -1.0];
        let x = solve(&num, &sym.r, &sym.q, n, &b);
        let expected = dense_solve(&dense_from_entries(n, &entries), &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-8, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_real_klu_on_random_matrices() {
        let mut seed: u64 = 0x2E1F9C7A5B3D8461;
        let mut next_f64 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };

        for trial in 0..40 {
            let n = 4 + (trial % 14);
            let mut entries: Vec<(usize, usize, f64)> = Vec::new();
            let mut row_sums = vec![0.0f64; n];
            let mut off_entries: Vec<(usize, usize, f64)> = Vec::new();
            #[allow(clippy::needless_range_loop)] // `i` is also compared against `j`, not just an index
            for i in 0..n {
                let mut js: std::collections::BTreeSet<usize> = Default::default();
                let degree = 1 + (next_f64() * 3.0) as usize;
                for _ in 0..degree {
                    let j = (next_f64() * n as f64) as usize;
                    if j != i {
                        js.insert(j);
                    }
                }
                for j in js {
                    let v = next_f64() * 2.0 - 1.0;
                    off_entries.push((i, j, v));
                    row_sums[i] += v.abs();
                }
            }
            for (i, &rs) in row_sums.iter().enumerate() {
                entries.push((i, i, rs + 1.0 + next_f64()));
            }
            entries.extend(off_entries);

            let (col_ptr, row_idx, values) = to_csc(n, &entries);
            let sym = analyze(n, &col_ptr, &row_idx);
            let num = factor(n, &col_ptr, &row_idx, &values, &sym, 1e-3)
                .unwrap_or_else(|| panic!("trial {trial} (n={n}): unexpectedly singular"));

            let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.3).collect();
            let rust_x = solve(&num, &sym.r, &sym.q, n, &b);

            let mut klu_sys = crate::sparse_klu::KluRealSystem::new(n, &entries).unwrap();
            let klu_x = klu_sys.factor_and_solve(&entries, &b).unwrap();

            for i in 0..n {
                assert!(
                    (rust_x[i] - klu_x[i]).abs() < 1e-8,
                    "trial {trial} (n={n}), index {i}: rust={} klu={}",
                    rust_x[i],
                    klu_x[i]
                );
            }
        }
    }
}
