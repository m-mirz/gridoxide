//! Strongly-connected-components (Tarjan) — ports
//! `vendor/suitesparse/BTF/Source/btf_strongcomp.c`'s non-recursive `dfs`/
//! `btf_strongcomp` (the `#ifndef RECURSIVE` default path only — the
//! recursive version in the same C file is documented there as "for
//! illustration only, not for production use", and this port never needs
//! recursion depth beyond what the non-recursive version already handles
//! with an explicit stack).

use super::super::types::{unflip, EMPTY};

const UNVISITED: i64 = -2;
const UNASSIGNED: i64 = -1;

/// Non-recursive DFS rooted at node `j0`, updating `time`/`flag`/`low` and
/// `*nblocks`/`*timestamp` in place. Mirrors `dfs` in `btf_strongcomp.c`
/// exactly, including its `cstack`/`jstack`/`pstack` bookkeeping (kept as
/// separate `Vec`s here rather than aliasing the caller's output arrays the
/// way the C does for memory reuse — a non-load-bearing allocation
/// simplification, not an algorithmic one).
#[allow(clippy::too_many_arguments)]
fn dfs(
    j0: usize,
    col_ptr: &[i64],
    row_idx: &[i64],
    q: Option<&[i64]>,
    time: &mut [i64],
    flag: &mut [i64],
    low: &mut [i64],
    nblocks: &mut i64,
    timestamp: &mut i64,
    cstack: &mut [i64],
    jstack: &mut [i64],
    pstack: &mut [i64],
) {
    let mut chead: i64 = 0;
    let mut jhead: i64 = 0;
    jstack[0] = j0 as i64;
    debug_assert_eq!(flag[j0], UNVISITED);

    while jhead >= 0 {
        let j = jstack[jhead as usize] as usize;
        // Column j of A*Q is column jj of the input matrix A.
        let jj = match q {
            None => j as i64,
            Some(q) => unflip(q[j]),
        } as usize;
        let pend = col_ptr[jj + 1];

        if flag[j] == UNVISITED {
            // First visit to node j: prework.
            chead += 1;
            cstack[chead as usize] = j as i64;
            *timestamp += 1;
            time[j] = *timestamp;
            low[j] = *timestamp;
            flag[j] = UNASSIGNED;
            pstack[jhead as usize] = col_ptr[jj];
        }

        let pend_start = pstack[jhead as usize];
        let mut p = pend_start;
        while p < pend {
            let i = row_idx[p as usize];
            if flag[i as usize] == UNVISITED {
                // Node i not yet visited: recurse onto it.
                pstack[jhead as usize] = p + 1;
                jhead += 1;
                jstack[jhead as usize] = i;
                break;
            } else if flag[i as usize] == UNASSIGNED {
                // Back or cross edge to a visited-but-unassigned node.
                low[j] = low[j].min(time[i as usize]);
            }
            p += 1;
        }

        if p == pend {
            // All adjacent nodes of j already visited: pop and do postwork.
            jhead -= 1;

            if low[j] == time[j] {
                // j is the head of a strongly connected component: pop all
                // its members from cstack.
                loop {
                    let i = cstack[chead as usize];
                    chead -= 1;
                    flag[i as usize] = *nblocks;
                    if i as usize == j {
                        break;
                    }
                }
                *nblocks += 1;
            }
            if jhead >= 0 {
                let parent = jstack[jhead as usize] as usize;
                low[parent] = low[parent].min(low[j]);
            }
        }
    }
}

/// Ports `btf_strongcomp.c`'s `btf_strongcomp` (non-recursive default path).
///
/// `q` is BTF's optional input column permutation from `maxtrans` (may
/// contain "flipped" entries for unmatched columns, per `types::unflip`) —
/// `None` traverses the graph of `A` itself. When `Some`, it's updated in
/// place to `q_new[k] = q_old[p[k]]` (preserving flip bits), exactly as
/// `btf_strongcomp.c` documents, so that the combined permutation is
/// `P*A*Q` rather than just the symmetric `P*(A*Q)*P'` this function alone
/// computes.
///
/// Returns `(p, r)`: `p[k]` gives the original row/column index that becomes
/// row/column `k` of the permuted matrix (natural order preserved within
/// each block), and `r` gives block boundaries — `r.len() == nblocks + 1`,
/// block `b` spans rows/columns `r[b]..r[b+1]` of the permuted matrix. The
/// number of blocks is `r.len() - 1`.
pub fn strongcomp(n: usize, col_ptr: &[i64], row_idx: &[i64], q: Option<&mut [i64]>) -> (Vec<i64>, Vec<i64>) {
    let mut time = vec![EMPTY; n];
    let mut flag = vec![UNVISITED; n];
    let mut low = vec![EMPTY; n];
    let mut jstack = vec![EMPTY; n];
    let mut pstack = vec![EMPTY; n];
    // Sized n+1 to match btf_strongcomp.c's own reuse of the (size n+1)
    // output array R as Cstack workspace — chead is pre-incremented before
    // each push, so it can reach n after n pushes.
    let mut cstack = vec![EMPTY; n + 1];

    let mut timestamp: i64 = 0;
    let mut nblocks: i64 = 0;

    let q_ref = q.as_deref();
    for j in 0..n {
        if flag[j] == UNVISITED {
            dfs(
                j, col_ptr, row_idx, q_ref, &mut time, &mut flag, &mut low, &mut nblocks, &mut timestamp,
                &mut cstack, &mut jstack, &mut pstack,
            );
        }
    }
    debug_assert_eq!(timestamp, n as i64);

    let nblocks = nblocks as usize;

    // Construct the block boundary array R: count nodes per block, then
    // cumulative-sum into block-start offsets.
    let mut r = vec![0i64; nblocks + 1];
    for &f in flag.iter() {
        r[f as usize] += 1;
    }
    let mut cum = vec![0i64; nblocks];
    for b in 1..nblocks {
        cum[b] = cum[b - 1] + r[b - 1];
    }
    r[..nblocks].copy_from_slice(&cum[..nblocks]);
    r[nblocks] = n as i64;

    // Construct P, preserving natural (ascending original-index) order
    // within each block, using `cum` (still holding each block's starting
    // cursor) as a running per-block insertion position.
    let mut p = vec![EMPTY; n];
    let mut next_pos = cum;
    for (j, &f) in flag.iter().enumerate() {
        let b = f as usize;
        p[next_pos[b] as usize] = j as i64;
        next_pos[b] += 1;
    }
    debug_assert!(p.iter().all(|&x| x != EMPTY));

    if let Some(q) = q {
        let new_q: Vec<i64> = (0..n).map(|k| q[p[k] as usize]).collect();
        q.copy_from_slice(&new_q);
    }

    (p, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_matrix_is_all_singleton_blocks() {
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [0, 1, 2];
        let (p, r) = strongcomp(3, &col_ptr, &row_idx, None);
        assert_eq!(r, vec![0, 1, 2, 3]);
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn fully_connected_cycle_is_one_block() {
        // 0 -> 1 -> 2 -> 0 (a 3-cycle): all one strongly connected component.
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [1, 2, 0]; // col0 has row1, col1 has row2, col2 has row0
        let (p, r) = strongcomp(3, &col_ptr, &row_idx, None);
        assert_eq!(r, vec![0, 3], "one block containing all 3 nodes");
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn upper_triangular_chain_is_all_singletons() {
        // Already upper triangular: 0 -> {}, 1 -> {0}, 2 -> {0,1} (col j has
        // entries in rows < j only) -- no cycles, n singleton blocks.
        let col_ptr = [0, 0, 1, 3];
        let row_idx = [0, 0, 1];
        let (_p, r) = strongcomp(3, &col_ptr, &row_idx, None);
        assert_eq!(r, vec![0, 1, 2, 3]);
    }
}
