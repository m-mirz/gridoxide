//! Approximate minimum degree (AMD) ordering — ports
//! `vendor/suitesparse/AMD/Source/amd_order.c`'s `AMD_order` (the
//! user-callable entry point): validate the input, build A+A''s adjacency
//! (`aat::build_symmetric_lists`), then order it (`core::amd_2`).

mod aat;
mod core;
mod postorder;

/// Computes an AMD fill-reducing permutation for a square `n`-by-`n` CSC
/// matrix (`col_ptr`/`row_idx`, diagonal entries allowed but ignored).
/// Returns `perm` (`perm[k]` = the original row/column placed at position
/// `k`), or `None` if the input isn't valid (see `aat::validate` — matches
/// `AMD_order`'s `AMD_INVALID`/`AMD_OK_BUT_JUMBLED` cases, the latter not
/// repaired by this port; see `aat`'s module doc comment for why that's
/// confirmed unreachable for gridoxide's own always-sorted-and-deduped CSC
/// construction).
pub fn amd_order(n: usize, col_ptr: &[i64], row_idx: &[i64]) -> Option<Vec<i64>> {
    if n == 0 {
        return Some(Vec::new());
    }
    if !aat::validate(n, n, col_ptr, row_idx) {
        return None;
    }
    let lists = aat::build_symmetric_lists(n, col_ptr, row_idx);
    let (perm, _pinv) = core::amd_2(n, lists);
    Some(perm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_permutation(p: &[i64], n: usize) -> bool {
        let mut sorted: Vec<i64> = p.to_vec();
        sorted.sort_unstable();
        sorted == (0..n as i64).collect::<Vec<_>>()
    }

    #[test]
    fn diagonal_matrix() {
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [0, 1, 2];
        let perm = amd_order(3, &col_ptr, &row_idx).unwrap();
        assert!(is_permutation(&perm, 3));
    }

    #[test]
    fn rejects_unsorted_input() {
        // 2x2 matrix, column 0 has both rows but unsorted (1 before 0);
        // column 1 is empty.
        let col_ptr = [0, 2, 2];
        let row_idx = [1, 0];
        assert!(amd_order(2, &col_ptr, &row_idx).is_none());
    }

    #[test]
    fn chain_of_five() {
        // Symmetric chain 0-1-2-3-4.
        let col_ptr = [0, 1, 3, 5, 7, 8];
        let row_idx = [1, 0, 2, 1, 3, 2, 4, 3];
        let perm = amd_order(5, &col_ptr, &row_idx).unwrap();
        assert!(is_permutation(&perm, 5));
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_ffi_oracle_on_curated_matrices() {
        use crate::klu_native::ffi_oracle;

        let cases: Vec<(usize, Vec<i32>, Vec<i32>)> = vec![
            (3, vec![0, 1, 2, 3], vec![0, 1, 2]),
            (5, vec![0, 1, 3, 5, 7, 8], vec![1, 0, 2, 1, 3, 2, 4, 3]), // chain
            (5, vec![0, 2, 4, 6, 8, 10], vec![1, 4, 0, 2, 1, 3, 2, 4, 0, 3]), // ring
            (4, vec![0, 3, 6, 9, 12], vec![1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]), // clique
        ];
        for (n, col_ptr, row_idx) in cases {
            let col_ptr64: Vec<i64> = col_ptr.iter().map(|&x| x as i64).collect();
            let row_idx64: Vec<i64> = row_idx.iter().map(|&x| x as i64).collect();
            let rust_perm = amd_order(n, &col_ptr64, &row_idx64).unwrap();

            let c_perm = ffi_oracle::amd_order_oracle(n, &col_ptr, &row_idx).unwrap();
            let c_perm64: Vec<i64> = c_perm.iter().map(|&x| x as i64).collect();

            assert_eq!(rust_perm, c_perm64, "P mismatch for n={n}");
        }
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_ffi_oracle_on_random_matrices() {
        use crate::klu_native::ffi_oracle;

        // Fixed, hand-rolled LCG (no extra dependency) for determinism --
        // same style as block_sparse.rs's and btf::tests's random tests.
        let mut seed: u64 = 0xD1B54A32D192ED03;
        let mut next_f64 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };

        for trial in 0..60 {
            let n = 2 + (trial % 15);
            // Random graph, deliberately asymmetric per-column degree (AMD's
            // own input need not be symmetric -- amd_order symmetrizes it
            // internally via A+A') and including isolated (empty) columns
            // sometimes, to exercise AMD's "empty variable" fast path.
            let mut cols: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); n];
            for (j, col) in cols.iter_mut().enumerate() {
                if next_f64() < 0.15 {
                    continue; // leave this column empty sometimes
                }
                let degree = 1 + (next_f64() * 4.0) as usize;
                for _ in 0..degree {
                    let r = (next_f64() * n as f64) as usize;
                    if r != j {
                        col.insert(r);
                    }
                }
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
            let rust_perm = amd_order(n, &col_ptr64, &row_idx64).unwrap();
            assert!(is_permutation(&rust_perm, n), "trial {trial} (n={n}): not a permutation");

            let c_perm = ffi_oracle::amd_order_oracle(n, &col_ptr, &row_idx).unwrap();
            let c_perm64: Vec<i64> = c_perm.iter().map(|&x| x as i64).collect();

            // Not asserting bit-identical output here: AMD's tie-breaking
            // among equal-degree candidates is genuinely order-sensitive
            // (confirmed directly -- see `core.rs`'s "move first entry to
            // end" comment for a case where this port initially produced a
            // measurably worse ordering before that fix, and a *second*,
            // *harmless* case afterwards where both orderings achieve the
            // same, independently-verified-optimal fill-in). So: verify
            // fill-in quality parity instead, which is what actually matters
            // for a sparse LU's performance -- not bit-identical tie-breaks.
            let rust_fill = symbolic_fill_count(n, &col_ptr64, &row_idx64, &rust_perm);
            let c_fill = symbolic_fill_count(n, &col_ptr64, &row_idx64, &c_perm64);
            assert_eq!(
                rust_fill, c_fill,
                "trial {trial} (n={n}): fill-in differs (rust={rust_fill}, C={c_fill}) -- \
                 rust_perm={rust_perm:?} c_perm={c_perm64:?}"
            );
        }
    }

    /// Simulates symbolic Cholesky/LU elimination of A+A' in the given
    /// pivot order, returning the total fill count (sum of remaining
    /// neighbors eliminated at each step, including original edges) --
    /// independent of both this port's and the FFI oracle's own ordering
    /// code, used purely to judge whether two *different* permutations are
    /// equally good rather than requiring them to be bit-identical.
    fn symbolic_fill_count(n: usize, col_ptr: &[i64], row_idx: &[i64], perm: &[i64]) -> usize {
        let mut adj: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); n];
        for j in 0..n {
            for &i in &row_idx[col_ptr[j] as usize..col_ptr[j + 1] as usize] {
                if i as usize != j {
                    adj[i as usize].insert(j);
                    adj[j].insert(i as usize);
                }
            }
        }
        let mut eliminated = vec![false; n];
        let mut total = 0usize;
        for &node in perm {
            let node = node as usize;
            let neighbors: Vec<usize> = adj[node].iter().copied().filter(|&x| !eliminated[x]).collect();
            total += neighbors.len();
            for &a in &neighbors {
                for &b in &neighbors {
                    if a != b {
                        adj[a].insert(b);
                    }
                }
            }
            eliminated[node] = true;
        }
        total
    }
}
