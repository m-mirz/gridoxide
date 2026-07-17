//! Constructs A+A''s adjacency lists (excluding the diagonal) — the input
//! `AMD_2` orders. Consolidates `vendor/suitesparse/AMD/Source/amd_aat.c`
//! (counts nonzeros per row/column of A+A') and `amd_1.c` (builds the actual
//! structure into a shared, exactly-presized buffer using those counts) into
//! a single merge-scan pass that pushes directly into per-row `Vec<i64>`s.
//!
//! This drops the count-then-fill two-pass structure `amd_aat.c`/`amd_1.c`
//! use — a pure memory-management difference (they need it to size one
//! shared flat buffer up front; `Vec::push`'s amortized growth makes the
//! separate counting pass unnecessary) — and the symmetry/`nzdiag`/`nzboth`
//! statistics `amd_aat.c` also computes (`Info[AMD_SYMMETRY]` etc.), which
//! are diagnostic-only and never consulted by the ordering algorithm itself.
//! The actual adjacency-discovery algorithm (the "merge two sorted lists"
//! scan of A's upper and lower triangular parts per column `k`, using `tp[j]`
//! to remember where the scan of column `j`'s lower part left off) is
//! ported unchanged.
//!
//! Requires sorted, duplicate-free CSC input (diagonal entries allowed but
//! ignored) — matches `AMD_valid`'s "OK" (not "jumbled") case. gridoxide's
//! own CSC construction always guarantees this, so the "jumbled" repair path
//! (`amd_preprocess.c`, which reconstructs a clean copy via `R = A'`) is out
//! of scope here, the same way `types::Options`'s doc comment explains for
//! BTF's dropped `maxwork` bookkeeping. `amd::validate` (ported from
//! `amd_valid.c`) still checks this defensively.

/// Builds A+A''s adjacency lists, one `Vec<i64>` per row/column, containing
/// every column index with a nonzero in that row of A+A' (excluding the
/// diagonal), each edge appearing once from each endpoint's perspective (as
/// `amd_aat.c`/`amd_1.c` themselves produce — this is a symmetric structure
/// by construction, not just conceptually symmetric).
pub fn build_symmetric_lists(n: usize, col_ptr: &[i64], row_idx: &[i64]) -> Vec<Vec<i64>> {
    let mut lists: Vec<Vec<i64>> = vec![Vec::new(); n];
    // Tp[j]: how far the "lower triangular part of column j" scan has
    // progressed so far (shared state across the outer loop over k).
    let mut tp: Vec<i64> = col_ptr[..n].to_vec();

    for k in 0..n {
        let p2 = col_ptr[k + 1];
        let mut p = col_ptr[k];
        loop {
            if p >= p2 {
                break;
            }
            let j = row_idx[p as usize];
            if j < k as i64 {
                // Entry A(j,k) in the strictly upper triangular part: add
                // both A(j,k) and A(k,j) to A+A'.
                lists[j as usize].push(k as i64);
                lists[k].push(j);
                p += 1;
            } else if j == k as i64 {
                p += 1;
                break; // skip the diagonal, done with column k
            } else {
                break; // j > k: first entry below the diagonal
            }

            // Scan the lower triangular part of column j, starting where
            // the last scan left off, until reaching row k.
            let pj2 = col_ptr[j as usize + 1];
            let mut pj = tp[j as usize];
            loop {
                if pj >= pj2 {
                    break;
                }
                let i = row_idx[pj as usize];
                if i < k as i64 {
                    // A(i,j) is only in the lower part, not the upper.
                    lists[i as usize].push(j);
                    lists[j as usize].push(i);
                    pj += 1;
                } else if i == k as i64 {
                    pj += 1;
                    break; // A(k,j) in lower part, A(j,k) already handled above
                } else {
                    break; // consider this entry later, when k advances to i
                }
            }
            tp[j as usize] = pj;
        }
        tp[k] = p;
    }

    // Clean up any remaining mismatched entries (lower-triangular entries
    // whose upper-triangular counterpart's column was never scanned as "k").
    for j in 0..n {
        let pj2 = col_ptr[j + 1];
        let mut pj = tp[j];
        while pj < pj2 {
            let i = row_idx[pj as usize];
            lists[i as usize].push(j as i64);
            lists[j].push(i);
            pj += 1;
        }
    }

    lists
}

/// Ports `AMD/Source/amd_valid.c`'s `AMD_valid` — checks whether a CSC
/// matrix is well-formed. Returns `true` for `AMD_OK` (sorted, no
/// duplicates); `false` for anything else (`AMD_INVALID`, or
/// `AMD_OK_BUT_JUMBLED` — unsorted/duplicate entries, which this port
/// doesn't repair, see this module's doc comment).
pub fn validate(n_row: usize, n_col: usize, col_ptr: &[i64], row_idx: &[i64]) -> bool {
    if col_ptr.is_empty() || col_ptr[0] != 0 {
        return false;
    }
    let nz = col_ptr[n_col];
    if nz < 0 {
        return false;
    }
    for j in 0..n_col {
        let (p1, p2) = (col_ptr[j], col_ptr[j + 1]);
        if p1 > p2 {
            return false;
        }
        let mut ilast: i64 = -1;
        for &i in &row_idx[p1 as usize..p2 as usize] {
            if i < 0 || i as usize >= n_row {
                return false;
            }
            if i <= ilast {
                return false; // unsorted or duplicate
            }
            ilast = i;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_sorted_no_dup() {
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [0, 1, 2];
        assert!(validate(3, 3, &col_ptr, &row_idx));
    }

    #[test]
    fn validate_rejects_unsorted() {
        let col_ptr = [0, 2];
        let row_idx = [1, 0]; // unsorted within column 0
        assert!(!validate(2, 1, &col_ptr, &row_idx));
    }

    #[test]
    fn validate_rejects_duplicate() {
        let col_ptr = [0, 2];
        let row_idx = [0, 0]; // duplicate
        assert!(!validate(2, 1, &col_ptr, &row_idx));
    }

    #[test]
    fn build_symmetric_lists_diagonal_matrix_is_empty() {
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [0, 1, 2];
        let lists = build_symmetric_lists(3, &col_ptr, &row_idx);
        assert!(lists.iter().all(|l| l.is_empty()), "diagonal-only matrix has no off-diagonal A+A' entries");
    }

    #[test]
    fn build_symmetric_lists_is_symmetric() {
        // A: col0 -> row1 (off-diagonal, upper); col1 has nothing new.
        let col_ptr = [0, 1, 1];
        let row_idx = [1];
        let lists = build_symmetric_lists(2, &col_ptr, &row_idx);
        assert_eq!(lists[0], vec![1]);
        assert_eq!(lists[1], vec![0]);
    }
}
