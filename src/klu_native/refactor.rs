//! Cheap numeric-only re-factorization — ports `vendor/suitesparse/KLU/
//! Source/klu_refactor.c`'s `KLU_refactor`: reuse an existing `Numeric`'s
//! BTF/AMD pattern, per-block pivot choices, and L/U sparsity pattern
//! unchanged, only recomputing numeric values against a *new* set of values
//! sharing the *same* sparsity pattern as the matrix `factor::factor`
//! originally saw (`KLU_refactor`'s own precondition — "The pattern of the
//! input matrix (Ap, Ai) must be identical to the pattern given to
//! KLU_factor").
//!
//! No new BTF/AMD ordering or partial pivoting happens here — `analyze`'s
//! `Symbolic` (unchanged) and the previous `factor::Numeric`'s row
//! permutation (`num.pinv`, itself never modified here) drive the same
//! block-local row extraction `factor::factor` does, just feeding
//! `kernel::refactor_block_in_place` instead of `kernel::factor_block` per
//! block.
//!
//! **In place, allocation-light** (see `kernel::refactor_block_in_place`'s
//! own doc comment for the profiling story behind this): `num` is mutated
//! directly rather than rebuilt, and the per-block column-entry scratch
//! (`RefactorScratch::columns`, `RefactorScratch::x`) persists across
//! repeated calls on the same `RefactorScratch` — the caller
//! (`KluNativeSystem`) keeps one alive for the system's whole lifetime
//! rather than creating a fresh one per solve. `num.pnum`/`num.pinv` aren't
//! touched at all (previously needlessly `.clone()`d into a discarded
//! result even though refactor never changes them). `num.off_p`/`off_i`/
//! `off_x` are still rebuilt fresh each call — proportional to `nzoff`
//! (cross-BTF-block nonzeros), which is zero for the common case of a
//! single strongly-connected power system (one BTF block), so this
//! remaining allocation is off the hot path for gridoxide's own real
//! Jacobians; revisit if a topology with genuine BTF structure ever makes
//! it show up in a profile.

use super::analyze::Symbolic;
use super::factor::Numeric;
use super::kernel::refactor_block_in_place;
use super::types::flip;

/// Reusable scratch space for repeated `refactor` calls against the same
/// `n`-by-`n` sparsity pattern — owned by `KluNativeSystem` for its whole
/// lifetime (see `mod.rs`), not recreated per call.
pub struct RefactorScratch {
    /// Dense workspace, reused across every block and every call — see
    /// `kernel::refactor_block_in_place`'s doc comment for why this is
    /// always safely all-zero on entry.
    x: Vec<f64>,
    /// `columns[k]` = global column `k`'s raw entries for the current call,
    /// `.clear()`-then-repopulated each `refactor` call rather than
    /// rebuilt from `Vec::new()` — keeps each column's already-grown
    /// capacity across calls, matching the sparsity pattern being fixed.
    columns: Vec<Vec<(i64, f64)>>,
}

impl RefactorScratch {
    pub fn new(n: usize) -> Self {
        Self { x: vec![0.0; n], columns: vec![Vec::new(); n] }
    }
}

/// Re-factors the whole `n`-by-`n` matrix (CSC: `col_ptr`/`row_idx`/`values`)
/// against a previous `Numeric` from the *same* sparsity pattern and the
/// same `Symbolic` analysis, mutating `num` in place. Returns `false` if any
/// block is numerically singular (`halt_if_singular` is always true for
/// gridoxide's fixed `Options`, matching `kernel::refactor_block_in_place`'s
/// own contract) — `num`'s contents are left partially updated in that case
/// (matching `KLU_refactor`'s own "return FALSE, `Numeric` in an
/// unspecified state" contract; the caller is expected to treat the whole
/// solve as failed, not to keep using `num`).
pub fn refactor(
    n: usize,
    col_ptr: &[i64],
    row_idx: &[i64],
    values: &[f64],
    sym: &Symbolic,
    num: &mut Numeric,
    scratch: &mut RefactorScratch,
) -> bool {
    // Row mapping uses the *final numeric* permutation from the previous
    // factorization (`num.pinv`), not the symbolic-only `pinv_sym`
    // `factor::factor` itself starts from -- by now, block-local row
    // position and global pivotal position coincide, since no further
    // pivoting will move any row. Mirrors `klu_refactor.c`'s own
    // `newrow = Pinv[Ai[p]] - k1` where `Pinv = Numeric->Pinv`.
    let nblocks = sym.nblocks();

    let mut off_p = vec![0i64; n + 1];
    let mut off_i: Vec<i64> = Vec::new();
    let mut off_x: Vec<f64> = Vec::new();
    let mut poff: i64 = 0;

    for block in 0..nblocks {
        let k1 = sym.r[block];
        let k2 = sym.r[block + 1];
        let nk = k2 - k1;

        for k in 0..nk {
            let oldcol = sym.q[k + k1];
            let entries = &mut scratch.columns[k + k1];
            entries.clear();
            for p in col_ptr[oldcol] as usize..col_ptr[oldcol + 1] as usize {
                let oldrow = row_idx[p];
                let newrow = num.pinv[oldrow as usize];
                let value = values[p];
                if (newrow as usize) < k1 {
                    entries.push((flip(oldrow), value));
                } else {
                    entries.push((newrow - k1 as i64, value));
                }
            }
        }

        let mut off_diag_out: Vec<Vec<(i64, f64)>> = vec![Vec::new(); nk];
        let ok = refactor_block_in_place(
            nk,
            &mut num.blocks[block],
            &mut scratch.x[..nk],
            &scratch.columns[k1..k2],
            &mut off_diag_out,
        );
        if !ok {
            return false;
        }

        for (k, entries) in off_diag_out.into_iter().enumerate() {
            let global_col = k + k1;
            for (oldrow, value) in entries {
                off_i.push(oldrow);
                off_x.push(value);
                poff += 1;
            }
            off_p[global_col + 1] = poff;
        }
    }

    // No new pivoting -- num.pnum/num.pinv are exactly what the previous
    // factorization established and are left untouched, matching
    // KLU_refactor's own behavior (Numeric->Pnum/Pinv are read, never
    // rewritten, aside from the pre-existing scale-factor permutation this
    // port hasn't wired in yet -- see scale.rs's module doc comment on
    // Phase 7).
    //
    // Apply the (unchanged) final numeric pivot row permutation to the new
    // off-diagonal entries, exactly as `factor::factor` does for its own.
    for i in off_i.iter_mut() {
        *i = num.pinv[*i as usize];
    }

    num.off_p = off_p;
    num.off_i = off_i;
    num.off_x = off_x;

    true
}

#[cfg(test)]
mod tests {
    use super::super::analyze::analyze;
    use super::super::factor::factor;
    use super::super::solve::solve;
    use super::*;

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

    /// Same sparsity pattern (upper block triangular by construction, as in
    /// `factor.rs`'s own `genuinely_multi_block` fixture), two different
    /// sets of values -- factor once, refactor with the second set, confirm
    /// the refactored solve matches an independent dense solve of the
    /// *second* matrix.
    #[test]
    fn multi_block_refactor_with_new_values_matches_dense() {
        let n = 5;
        let original = vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 5.0),
            (2, 2, 3.0),
            (2, 0, 0.5),
            (3, 3, 6.0),
            (3, 4, 1.0),
            (4, 3, 1.0),
            (4, 4, 7.0),
            (3, 0, 0.2),
            (4, 2, 0.3),
        ];
        let (col_ptr, row_idx, values) = to_csc(n, &original);
        let sym = analyze(n, &col_ptr, &row_idx);
        assert!(sym.nblocks() >= 2, "fixture should produce multiple BTF blocks, got {}", sym.nblocks());
        let mut num = factor(n, &col_ptr, &row_idx, &values, &sym, 1e-3).unwrap();

        let updated = vec![
            (0, 0, 5.5),
            (0, 1, 0.6),
            (1, 0, 0.4),
            (1, 1, 4.2),
            (2, 2, 2.1),
            (2, 0, 0.9),
            (3, 3, 4.4),
            (3, 4, 0.5),
            (4, 3, 0.7),
            (4, 4, 5.6),
            (3, 0, 0.35),
            (4, 2, 0.15),
        ];
        let (col_ptr2, row_idx2, values2) = to_csc(n, &updated);
        assert_eq!(col_ptr, col_ptr2, "fixture must keep the same sparsity pattern");
        assert_eq!(row_idx, row_idx2, "fixture must keep the same sparsity pattern");

        let mut scratch = RefactorScratch::new(n);
        assert!(refactor(n, &col_ptr2, &row_idx2, &values2, &sym, &mut num, &mut scratch));

        let b = vec![1.0, -2.0, 0.5, 3.0, -1.0];
        let x = solve(&sym, &num, None, &b);
        let expected = dense_solve(&dense_from_entries(n, &updated), &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-8, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn refactor_reused_repeatedly_matches_dense_each_time() {
        // Mirrors sparse_klu.rs's own real_sparse_system_refactor_reuses_
        // symbolic precedent: factor once, then refactor several times in a
        // row (as PersistentSolver's own repeated-Newton-iteration use
        // would), confirming each call's result is independently correct
        // rather than only the first refactor after the original factor.
        let n = 4;
        let base = vec![(0, 0, 6.0), (1, 1, 7.0), (2, 2, 8.0), (3, 3, 5.0), (0, 1, -0.9), (1, 0, -0.5), (2, 3, 0.4), (3, 2, -0.2)];
        let (col_ptr, row_idx, values) = to_csc(n, &base);
        let sym = analyze(n, &col_ptr, &row_idx);
        let mut num = factor(n, &col_ptr, &row_idx, &values, &sym, 1e-3).unwrap();
        let mut scratch = RefactorScratch::new(n);

        for trial in 0..5 {
            let scale = 1.0 + trial as f64 * 0.3;
            let updated: Vec<(usize, usize, f64)> = base.iter().map(|&(r, c, v)| (r, c, v * scale)).collect();
            let (cp, ri, vals) = to_csc(n, &updated);
            assert!(refactor(n, &cp, &ri, &vals, &sym, &mut num, &mut scratch), "trial {trial}");

            let b = vec![1.0, 2.0, -1.0, 0.5];
            let x = solve(&sym, &num, None, &b);
            let expected = dense_solve(&dense_from_entries(n, &updated), &b);
            for i in 0..n {
                assert!((x[i] - expected[i]).abs() < 1e-8, "trial {trial}, index {i}: {} vs {}", x[i], expected[i]);
            }
        }
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_real_klu_refactor_on_random_matrices() {
        let mut seed: u64 = 0x7F4A7C15E3B0C9D2;
        let mut next_f64 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };

        for trial in 0..40 {
            let n = 4 + (trial % 14);
            let pattern: Vec<(usize, usize)> = {
                let mut pairs = std::collections::BTreeSet::new();
                for i in 0..n {
                    pairs.insert((i, i));
                    let degree = 1 + (next_f64() * 3.0) as usize;
                    for _ in 0..degree {
                        let j = (next_f64() * n as f64) as usize;
                        if j != i {
                            pairs.insert((i, j));
                        }
                    }
                }
                pairs.into_iter().collect()
            };

            let make_values = |next_f64: &mut dyn FnMut() -> f64| -> Vec<(usize, usize, f64)> {
                let mut row_sums = vec![0.0f64; n];
                let mut off_entries: Vec<(usize, usize, f64)> = Vec::new();
                for &(i, j) in &pattern {
                    if i != j {
                        let v = next_f64() * 2.0 - 1.0;
                        off_entries.push((i, j, v));
                        row_sums[i] += v.abs();
                    }
                }
                let mut entries: Vec<(usize, usize, f64)> =
                    row_sums.iter().enumerate().map(|(i, &rs)| (i, i, rs + 1.0 + next_f64())).collect();
                entries.extend(off_entries);
                entries
            };

            let entries1 = make_values(&mut next_f64);
            let (col_ptr, row_idx, values) = to_csc(n, &entries1);
            let sym = analyze(n, &col_ptr, &row_idx);
            let mut num = factor(n, &col_ptr, &row_idx, &values, &sym, 1e-3)
                .unwrap_or_else(|| panic!("trial {trial} (n={n}): unexpectedly singular (factor)"));
            let mut scratch = RefactorScratch::new(n);

            let entries2 = make_values(&mut next_f64);
            let (col_ptr2, row_idx2, values2) = to_csc(n, &entries2);
            assert!(
                refactor(n, &col_ptr2, &row_idx2, &values2, &sym, &mut num, &mut scratch),
                "trial {trial} (n={n}): unexpectedly singular (refactor)"
            );

            let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.3).collect();
            let rust_x = solve(&sym, &num, None, &b);

            let mut klu_sys = crate::sparse_klu::KluRealSystem::new(n, &entries1).unwrap();
            let _ = klu_sys.factor_and_solve(&entries1, &b).unwrap();
            let klu_x = klu_sys.factor_and_solve(&entries2, &b).unwrap();

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
