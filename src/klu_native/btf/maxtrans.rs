//! Maximum transversal (bipartite matching) — ports
//! `vendor/suitesparse/BTF/Source/btf_maxtrans.c`'s `augment`/`btf_maxtrans`:
//! a Duff/MC21-style augmenting-path search via a non-recursive DFS with a
//! "cheap assignment" fast path (try each column's still-unscanned entries
//! for an already-unmatched row before falling back to a full DFS).
//!
//! `maxwork`-limiting is dropped outright, not merely defaulted off — see
//! `types::Options`'s doc comment for why this is confirmed dead for
//! gridoxide's fixed configuration (`klu_defaults.c`: `Common->maxwork = 0`,
//! "no limit", never overridden). This port always runs to completion.

use super::super::types::EMPTY;

/// Attempts to find an augmenting path starting at column `k`, extending the
/// existing matching in `match_` in place if successful. Mirrors `augment`
/// in `btf_maxtrans.c` line-for-line (minus the dropped `maxwork` check).
///
/// `cheap[j]` tracks how much of column `j`'s adjacency list is already known
/// to be matched to some row (the "cheap assignment" fast path); `flag[j] ==
/// k as i64` marks column `j` as visited during this call's DFS specifically
/// (reusing one `flag` array across all `n` calls by tagging with the
/// current `k`, avoiding an O(n) reset between calls).
#[allow(clippy::too_many_arguments)]
fn augment(
    k: usize,
    col_ptr: &[i64],
    row_idx: &[i64],
    match_: &mut [i64],
    cheap: &mut [i64],
    flag: &mut [i64],
    istack: &mut [i64],
    jstack: &mut [i64],
    pstack: &mut [i64],
) -> bool {
    let kk = k as i64;
    let mut found = false;
    let mut i: i64 = EMPTY;
    let mut head: i64 = 0;
    jstack[0] = kk;
    debug_assert_ne!(flag[k], kk);

    while head >= 0 {
        let j = jstack[head as usize];
        let pend = col_ptr[(j + 1) as usize];

        if flag[j as usize] != kk {
            // First time j has been visited this call: prework.
            flag[j as usize] = kk;
            let mut p = cheap[j as usize];
            while p < pend && !found {
                i = row_idx[p as usize];
                found = match_[i as usize] == EMPTY;
                p += 1;
            }
            cheap[j as usize] = p;

            if found {
                // End of augmenting path: column j matched with row i.
                istack[head as usize] = i;
                break;
            }
            pstack[head as usize] = col_ptr[j as usize];
        }

        // DFS for nodes adjacent to j: all rows in column j are already
        // matched, so continue the search via each row's current match.
        let pstart = pstack[head as usize];
        let mut p = pstart;
        while p < pend {
            i = row_idx[p as usize];
            let j2 = match_[i as usize];
            debug_assert_ne!(j2, EMPTY);
            if flag[j2 as usize] != kk {
                // Node j2 not yet visited: recurse onto it.
                pstack[head as usize] = p + 1;
                istack[head as usize] = i;
                head += 1;
                jstack[head as usize] = j2;
                break;
            }
            p += 1;
        }

        if p == pend {
            // All adjacent nodes of j already visited: pop and fail here.
            head -= 1;
        }
    }

    if found {
        // Unwind the path and make the corresponding matches.
        let mut p = head;
        while p >= 0 {
            let j = jstack[p as usize];
            let i = istack[p as usize];
            match_[i as usize] = j;
            p -= 1;
        }
    }
    found
}

/// Ports `btf_maxtrans.c`'s `btf_maxtrans`: finds a matching `match_[i] = j`
/// (row `i` matched to column `j`) that maximizes the number of nonzero
/// diagonal entries of the permuted `nrow`-by-`ncol` matrix (`col_ptr`/
/// `row_idx` in CSC form, `col_ptr.len() == ncol + 1`). Returns
/// `(match_, nmatch)`: `match_[i] == EMPTY` for an unmatched row, `nmatch`
/// is the number of matched rows (== number of nonzeros on the diagonal of
/// the permuted matrix).
pub fn maxtrans(nrow: usize, ncol: usize, col_ptr: &[i64], row_idx: &[i64]) -> (Vec<i64>, usize) {
    debug_assert_eq!(col_ptr.len(), ncol + 1);

    let mut cheap: Vec<i64> = col_ptr[..ncol].to_vec();
    let mut flag = vec![EMPTY; ncol];
    let mut istack = vec![0i64; ncol];
    let mut jstack = vec![0i64; ncol];
    let mut pstack = vec![0i64; ncol];
    let mut match_ = vec![EMPTY; nrow];

    let mut nmatch = 0usize;
    for k in 0..ncol {
        if augment(k, col_ptr, row_idx, &mut match_, &mut cheap, &mut flag, &mut istack, &mut jstack, &mut pstack) {
            nmatch += 1;
        }
    }
    (match_, nmatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_matrix_matches_fully() {
        // 3x3 identity pattern: each column already matches its own row.
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [0, 1, 2];
        let (match_, nmatch) = maxtrans(3, 3, &col_ptr, &row_idx);
        assert_eq!(nmatch, 3);
        assert_eq!(match_, vec![0, 1, 2]);
    }

    #[test]
    fn permutation_needed() {
        // Column 0 only has row 1; column 1 only has row 0. Must swap.
        // A = [[0,1],[1,0]] in CSC: col0 -> row1, col1 -> row0.
        let col_ptr = [0, 1, 2];
        let row_idx = [1, 0];
        let (match_, nmatch) = maxtrans(2, 2, &col_ptr, &row_idx);
        assert_eq!(nmatch, 2);
        // match_[i] = j: row 1 matched to col 0, row 0 matched to col 1.
        assert_eq!(match_[1], 0);
        assert_eq!(match_[0], 1);
    }

    #[test]
    fn structurally_singular_partial_match() {
        // 2x2 matrix where both columns only ever point at row 0:
        // no perfect matching possible, nmatch must be < n.
        let col_ptr = [0, 1, 2];
        let row_idx = [0, 0];
        let (match_, nmatch) = maxtrans(2, 2, &col_ptr, &row_idx);
        assert_eq!(nmatch, 1);
        assert!(match_.contains(&EMPTY));
    }
}
