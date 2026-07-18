//! Forward/back substitution — ports `vendor/suitesparse/KLU/Source/
//! klu_solve.c`'s `KLU_solve` and `klu.c`'s `KLU_lsolve`/`KLU_usolve`,
//! **`nrhs == 1` only**: gridoxide's own Newton-Raphson Jacobian solve
//! always solves a single right-hand side per iteration (confirmed against
//! every `sparse_klu.rs` call site — see `mod.rs`'s module doc comment), so
//! the `switch (nr) { case 1: ... case 4: ... }` duplication `KLU_lsolve`/
//! `KLU_usolve`/`KLU_solve` itself carry for batched multi-RHS solves is
//! dropped, keeping only the `case 1` body.
//!
//! `X = P*(Rs\B)`, then block-reverse-order `(L*U + Off)\X` (last BTF block
//! first — see `factor.rs`'s own doc comment for why: an *earlier* block's
//! rows may reference a *later* block's columns, so its off-diagonal
//! contribution can only be subtracted out once that later block's own `x`
//! is already known), then `B = Q*X` to map back to original variable
//! order.

use super::analyze::Symbolic;
use super::factor::Numeric;

/// Solves `L x = b` in place: `L` is unit lower triangular (diagonal not
/// stored), `l_cols[k]` gives column `k`'s below-diagonal pattern. Ports
/// `KLU_lsolve`'s `case 1` body.
fn lsolve(l_cols: &[Vec<(usize, f64)>], x: &mut [f64]) {
    for k in 0..l_cols.len() {
        let xk = x[k];
        for &(row, lik) in &l_cols[k] {
            x[row] -= lik * xk;
        }
    }
}

/// Solves `U x = b` in place: `U` is non-unit upper triangular, diagonal
/// entries in `udiag` (not stored in `u_cols`). Ports `KLU_usolve`'s
/// `case 1` body.
fn usolve(u_cols: &[Vec<(usize, f64)>], udiag: &[f64], x: &mut [f64]) {
    for k in (0..u_cols.len()).rev() {
        x[k] /= udiag[k];
        let xk = x[k];
        for &(row, uik) in &u_cols[k] {
            x[row] -= uik * xk;
        }
    }
}

/// Solves `A x = b` given `sym` (from `analyze::analyze`) and `num` (from
/// `factor::factor`/`refactor::refactor`, on the matrix `sym` was computed
/// for). `rs`, if present, is the *already-`Pnum`-permuted* row scale
/// factors (`rs[k]` scales row `k` of the permuted matrix — matches
/// `KLU_solve`'s own precondition that `Numeric->Rs` has already been
/// through the late `Rs[k] = Rs[Pnum[k]]` permutation `factor.c`/
/// `refactor.c` each perform once factorization is complete, *not* the raw,
/// row-indexed `Rs` `scale::scale` itself returns). Ported faithfully now
/// (`KLU_solve`'s own `Rs == NULL` branch) even though `factor::factor`/
/// `refactor::refactor` don't populate scale factors into `Numeric` yet —
/// see `scale.rs`'s module doc comment on Phase 7 wiring this in.
pub fn solve(sym: &Symbolic, num: &Numeric, rs: Option<&[f64]>, b: &[f64]) -> Vec<f64> {
    let n = b.len();

    // X = P*(Rs\B).
    let mut x = vec![0.0; n];
    for k in 0..n {
        let bi = b[num.pnum[k] as usize];
        x[k] = match rs {
            Some(rs) => bi / rs[k],
            None => bi,
        };
    }

    // X = (L*U + Off)\X, block by block in reverse BTF order.
    let nblocks = sym.nblocks();
    for block in (0..nblocks).rev() {
        let k1 = sym.r[block];
        let k2 = sym.r[block + 1];
        let bf = &num.blocks[block];

        let x_block = &mut x[k1..k2];
        lsolve(&bf.l_cols, x_block);
        usolve(&bf.u_cols, &bf.udiag, x_block);

        // Block back-substitution for the off-diagonal entries: subtract
        // this now-solved block's contribution out of not-yet-solved
        // earlier blocks' right-hand side. A no-op for block 0 (no earlier
        // block exists to propagate into), matching `klu_solve.c`'s own
        // `if (block > 0)` guard -- not special-cased here since `off_p`'s
        // range for block 0's own columns is structurally always empty by
        // the same BTF invariant, making the guard purely an optimization.
        for k in k1..k2 {
            let xk = x[k];
            for p in num.off_p[k] as usize..num.off_p[k + 1] as usize {
                let row = num.off_i[p] as usize;
                x[row] -= num.off_x[p] * xk;
            }
        }
    }

    // B = Q*X: map from pivot/column order back to original variable order.
    let mut result = vec![0.0; n];
    for k in 0..n {
        result[sym.q[k]] = x[k];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::super::analyze::analyze;
    use super::super::factor::factor;
    use super::super::refactor::refactor;
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
        let x = solve(&sym, &num, None, &b);
        let expected = dense_solve(&dense_from_entries(n, &entries), &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-8, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn genuinely_multi_block_matches_dense() {
        let n = 5;
        let entries = vec![
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
        let (col_ptr, row_idx, values) = to_csc(n, &entries);
        let sym = analyze(n, &col_ptr, &row_idx);
        assert!(sym.nblocks() >= 2, "fixture should produce multiple BTF blocks, got {}", sym.nblocks());

        let num = factor(n, &col_ptr, &row_idx, &values, &sym, 1e-3).unwrap();
        let b = vec![1.0, -2.0, 0.5, 3.0, -1.0];
        let x = solve(&sym, &num, None, &b);
        let expected = dense_solve(&dense_from_entries(n, &entries), &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-8, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn solve_after_refactor_matches_dense() {
        let n = 4;
        let base =
            vec![(0, 0, 6.0), (1, 1, 7.0), (2, 2, 8.0), (3, 3, 5.0), (0, 1, -0.9), (1, 0, -0.5), (2, 3, 0.4), (3, 2, -0.2)];
        let (col_ptr, row_idx, values) = to_csc(n, &base);
        let sym = analyze(n, &col_ptr, &row_idx);
        let num1 = factor(n, &col_ptr, &row_idx, &values, &sym, 1e-3).unwrap();

        let updated: Vec<(usize, usize, f64)> = base.iter().map(|&(r, c, v)| (r, c, v * 1.7 + 0.1)).collect();
        let (col_ptr2, row_idx2, values2) = to_csc(n, &updated);
        let num2 = refactor(n, &col_ptr2, &row_idx2, &values2, &sym, &num1).unwrap();

        let b = vec![1.0, 2.0, -1.0, 0.5];
        let x = solve(&sym, &num2, None, &b);
        let expected = dense_solve(&dense_from_entries(n, &updated), &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-8, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn rs_scaling_does_not_change_the_solution() {
        // Row scaling is a preconditioning technique: dividing X by a
        // *Pnum-permuted* Rs before the triangular solves and never
        // multiplying back by it is only mathematically consistent if `Rs`
        // was already folded into the L/U values themselves during
        // factorization (real KLU does this in `construct_column` -- see
        // `scale.rs`'s doc comment). This port doesn't wire scaling into
        // `factor`/`refactor` yet (Phase 7), so this test only exercises
        // `solve`'s own `Rs` code path in isolation with `Rs = 1` (a no-op
        // scale), confirming it's wired correctly and ready for Phase 7 to
        // supply a real (factorization-consistent) Rs.
        let n = 3;
        let entries = vec![(0, 0, 4.0), (1, 1, 5.0), (2, 2, 6.0), (0, 1, 0.3), (1, 2, -0.2)];
        let (col_ptr, row_idx, values) = to_csc(n, &entries);
        let sym = analyze(n, &col_ptr, &row_idx);
        let num = factor(n, &col_ptr, &row_idx, &values, &sym, 1e-3).unwrap();

        let b = vec![1.0, 2.0, -1.0];
        let x_unscaled = solve(&sym, &num, None, &b);
        let ones = vec![1.0; n];
        let x_scaled = solve(&sym, &num, Some(&ones), &b);
        for i in 0..n {
            assert_eq!(x_unscaled[i], x_scaled[i], "index {i}: Rs=1 must be a no-op");
        }
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_real_klu_on_random_matrices() {
        let mut seed: u64 = 0xC2B2AE3D27D4EB4F;
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
            #[allow(clippy::needless_range_loop)]
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
            let rust_x = solve(&sym, &num, None, &b);

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
