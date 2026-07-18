//! The Gilbert-Peierls left-looking sparse LU kernel with partial pivoting
//! and Eisenstat-Liu symmetric pruning — ports
//! `vendor/suitesparse/KLU/Source/klu_kernel.c` (`dfs`, `lsolve_symbolic`,
//! `construct_column`, `lsolve_numeric`, `lpivot`, `prune`, `KLU_kernel`),
//! operating on one BTF diagonal block at a time (matching `KLU_kernel`'s
//! own per-block scope — `factor.rs` drives this once per block, handling
//! the BTF off-diagonal bookkeeping and singleton fast path around it).
//!
//! **Storage**: `l_cols`/`u_cols` are `Vec<Vec<(usize, f64)>>` — one growable
//! column each, replacing the packed `Unit *LU` buffer + `Lip`/`Uip`
//! index-into-buffer bookkeeping entirely (see `amd/core.rs`'s module doc
//! comment for the same argument made in more depth for AMD's own packed
//! buffer — the reasoning is identical here: `Lip[k]`/`Uip[k]` only exist to
//! locate column `k`'s data within one shared, `realloc`-grown array;
//! `Vec::push`'s own growth makes that unnecessary, and `Llen[k]`/`Ulen[k]`
//! become simply `l_cols[k].len()`/`u_cols[k].len()`). Everything else —
//! the DFS reachability search, the diagonal-preference partial pivoting
//! (including its `Pinv` `FLIP`-sentinel bookkeeping, kept literal per
//! `types.rs`'s guidance), and Eisenstat-Liu pruning — is ported as
//! literally as this one storage change allows.

use super::types::{flip, EMPTY};

/// Non-recursive DFS for column `k`'s symbolic reachability, starting at
/// (already-pivotal) node `j0`. Ports `dfs` in `klu_kernel.c` exactly,
/// including the Eisenstat-Liu pruning short-circuit via `lpend` — only
/// `l_cols[jnew][..scan_limit]` (not the full column) needs scanning once
/// `jnew` has been pruned. `l_cols` here holds only *already-finalized*
/// columns `0..k`; `lik` accumulates new entries for column `k`'s own
/// (not-yet-finalized) L pattern.
#[allow(clippy::too_many_arguments)]
fn dfs(
    j0: usize,
    k: i64,
    pinv: &[i64],
    l_cols: &[Vec<(usize, f64)>],
    stack: &mut [usize],
    flag: &mut [i64],
    lpend: &[i64],
    mut top: usize,
    lik: &mut Vec<usize>,
    ap_pos: &mut [i64],
) -> usize {
    let mut head: i64 = 0;
    stack[0] = j0;
    debug_assert_ne!(flag[j0], k);

    while head >= 0 {
        let hd = head as usize;
        let j = stack[hd];
        let jnew = pinv[j] as usize;

        if flag[j] != k {
            // First time j has been visited this call.
            flag[j] = k;
            ap_pos[hd] = if lpend[jnew] == EMPTY { l_cols[jnew].len() as i64 } else { lpend[jnew] };
        }

        ap_pos[hd] -= 1;
        let mut pos = ap_pos[hd];
        let mut pushed_child = false;
        while pos >= 0 {
            let i = l_cols[jnew][pos as usize].0;
            if flag[i] != k {
                if pinv[i] >= 0 {
                    // i is pivotal: recurse onto it, remembering where we
                    // left off in j's own scan so we can resume later.
                    ap_pos[hd] = pos;
                    head += 1;
                    stack[head as usize] = i;
                    pushed_child = true;
                    break;
                } else {
                    // i is not pivotal (no outgoing edges): store directly
                    // into L's (still-being-built) pattern for column k.
                    flag[i] = k;
                    lik.push(i);
                }
            }
            pos -= 1;
        }

        if !pushed_child {
            // All adjacent nodes of j already visited: pop j, push onto
            // the output stack.
            head -= 1;
            top -= 1;
            stack[top] = j;
        }
    }

    top
}

/// Finds the symbolic pattern of column `k`'s solve `Lx = b` (`b` = column
/// `k` of `A(p,q)` restricted to this block) — ports `lsolve_symbolic`.
/// `col_entries` gives the block-local row indices with a nonzero in column
/// `k` (already extracted by the caller from the global CSC structure via
/// `q`/`k1`/`pinv_block`, matching `construct_column`'s own row-mapping).
/// Returns `(top, lik)`: `stack[top..n]` holds column `k`'s U pattern (rows
/// that are pivotal ancestors, in the order the DFS finished them), and
/// `lik` holds column `k`'s own new L pattern (non-pivotal neighbors found
/// directly, without needing a DFS).
#[allow(clippy::too_many_arguments)]
fn lsolve_symbolic(
    n: usize,
    k: usize,
    col_entries: &[usize],
    pinv: &[i64],
    l_cols: &[Vec<(usize, f64)>],
    stack: &mut [usize],
    flag: &mut [i64],
    lpend: &[i64],
    ap_pos: &mut [i64],
) -> (usize, Vec<usize>) {
    let mut top = n;
    let mut lik: Vec<usize> = Vec::new();
    let kk = k as i64;

    for &i in col_entries {
        if flag[i] != kk {
            if pinv[i] >= 0 {
                top = dfs(i, kk, pinv, l_cols, stack, flag, lpend, top, &mut lik, ap_pos);
            } else {
                flag[i] = kk;
                lik.push(i);
            }
        }
    }

    (top, lik)
}

/// Scatters column `k`'s numeric values into the dense workspace `x`,
/// splitting off entries that belong to an earlier BTF block (`newrow <
/// 0` after subtracting `k1`, i.e. strictly above this block's row range)
/// into the off-diagonal accumulator. Ports `construct_column`.
///
/// `col_entries_with_row` gives `(block_local_row_or_negative, value)`
/// pairs directly (negative meaning "off-diagonal, original row = -(v)-1"),
/// pre-computed by the caller from the global CSC structure — see
/// `factor::factor_block`'s call site for the exact mapping (`PSinv`/`k1`),
/// which mirrors `construct_column`'s own `PSinv[oldrow] - k1` computation.
/// Row scaling (`Rs`), if present, is applied by the *caller* before this
/// function ever sees the values — see `scale.rs` — since scaling only
/// depends on the original row, not anything this function computes.
fn construct_column(x: &mut [f64], col_entries: &[(i64, f64)], off_diagonal: &mut Vec<(i64, f64)>) {
    for &(row_or_off, value) in col_entries {
        if row_or_off < 0 {
            off_diagonal.push((flip(row_or_off), value)); // recover the real off-diagonal row
        } else {
            x[row_or_off as usize] = value;
        }
    }
}

/// Computes `x` for `Lx = b` via forward substitution against
/// already-finalized L columns, using the symbolic pattern found by
/// `lsolve_symbolic` (`stack[top..n]`, in an order that's already a valid
/// elimination order for this triangular solve). Ports `lsolve_numeric`.
fn lsolve_numeric(pinv: &[i64], l_cols: &[Vec<(usize, f64)>], stack: &[usize], top: usize, x: &mut [f64]) {
    for &j in &stack[top..] {
        let jnew = pinv[j] as usize;
        let xj = x[j];
        for &(row, lij) in &l_cols[jnew] {
            x[row] -= lij * xj;
        }
    }
}

/// Finds a pivot for column `k` via partial pivoting with diagonal
/// preference, and divides `L`'s column by it. Ports `lpivot` exactly.
///
/// `lik` (column `k`'s L pattern, from `lsolve_symbolic`) is consumed by
/// value and returned as the finalized `(row, value)` pairs for `l_cols[k]`
/// (mirroring the C's in-place `Lx[p] = x; ...; DIV(Lx[p], Lx[p], pivot)`).
///
/// Returns `None` if the matrix is structurally or numerically singular
/// (mirrors `lpivot` returning `FALSE`, with `halt_if_singular` — gridoxide
/// only ever uses `Options::default()`'s `halt_if_singular: true`, so this
/// port always halts rather than implementing the "pick lowest-numbered
/// non-pivotal row and continue with a zero pivot" fallback that path is
/// for — confirmed dead for gridoxide's fixed configuration).
/// `(pivrow, pivot, abs_pivot, l_col_k)`.
type LpivotResult = (i64, f64, f64, Vec<(usize, f64)>);

fn lpivot(diagrow: i64, tol: f64, x: &mut [f64], lik: Vec<usize>) -> Option<LpivotResult> {
    if lik.is_empty() {
        // Structurally singular; halt_if_singular is always true here.
        return None;
    }

    let mut l_values: Vec<(usize, f64)> = lik.iter().map(|&i| (i, x[i])).collect();
    for &(row, _) in &l_values {
        x[row] = 0.0;
    }

    let last_idx = l_values.len() - 1;
    let last_row_index = l_values[last_idx].0;
    // Shrink the "candidate" region by one (matching Llen[k] -= 1): the
    // last entry is set aside as a fallback pivot candidate, considered
    // separately below exactly as lpivot.c does.
    let candidates = &l_values[..last_idx];

    let mut pdiag: Option<usize> = None; // index within `candidates`
    let mut ppivrow: Option<usize> = None;
    let mut abs_pivot = -1.0f64; // EMPTY sentinel from lpivot.c is conceptually "unset"; -1 works since abs values are >= 0

    for (p, &(row, value)) in candidates.iter().enumerate() {
        let xabs = value.abs();
        if row as i64 == diagrow {
            pdiag = Some(p);
        }
        if xabs > abs_pivot {
            abs_pivot = xabs;
            ppivrow = Some(p);
        }
    }

    let last_value = l_values[last_idx].1;
    let mut last_xabs = last_value.abs();
    if last_xabs > abs_pivot {
        abs_pivot = last_xabs;
        ppivrow = None; // sentinel for "the last entry is the best candidate"
    }

    if last_row_index as i64 == diagrow {
        if last_xabs >= tol * abs_pivot {
            abs_pivot = last_xabs;
            ppivrow = None;
        }
    } else if let Some(pd) = pdiag {
        let dabs = candidates[pd].1.abs();
        if dabs >= tol * abs_pivot {
            abs_pivot = dabs;
            ppivrow = Some(pd);
        }
    }

    let (pivrow, pivot): (usize, f64) = if let Some(pp) = ppivrow {
        let (prow, pval) = candidates[pp];
        // Overwrite the chosen candidate's slot with the last entry's
        // values (matching Li[ppivrow]=last_row_index; Lx[ppivrow]=X[last]).
        l_values[pp] = (last_row_index, last_value);
        (prow, pval)
    } else {
        (last_row_index, last_value)
    };
    let _ = &mut last_xabs; // silence unused-mut if the compiler considers it otherwise
    l_values.truncate(last_idx); // drop the now-relocated (or duplicate) last slot

    if pivot == 0.0 {
        // Numerically singular; halt_if_singular is always true here.
        return None;
    }

    for v in l_values.iter_mut() {
        v.1 /= pivot;
    }

    Some((pivrow as i64, pivot, abs_pivot, l_values))
}

/// Prunes already-finalized columns of L to reduce future DFS work
/// (Eisenstat-Liu symmetric pruning). Ports `prune` exactly: for every
/// column `j` referenced in the just-finalized column `k` of U, if `j`
/// hasn't been pruned yet and the current pivot row appears in `L`'s
/// column `j`, partition that column so all non-pivotal rows come first
/// (swapping with the tail) and record the split point in `lpend[j]`.
fn prune(lpend: &mut [i64], pinv: &[i64], u_col_k: &[(usize, f64)], pivrow: usize, l_cols: &mut [Vec<(usize, f64)>]) {
    for &(j, _) in u_col_k {
        if lpend[j] == EMPTY {
            let lj = &mut l_cols[j];
            if let Some(found_pos) = lj.iter().position(|&(row, _)| row == pivrow) {
                let _ = found_pos;
                let mut phead = 0usize;
                let mut ptail = lj.len();
                while phead < ptail {
                    let (i, _) = lj[phead];
                    if pinv[i] >= 0 {
                        phead += 1;
                    } else {
                        ptail -= 1;
                        lj.swap(phead, ptail);
                    }
                }
                lpend[j] = ptail as i64;
            }
        }
    }
}

/// The result of factoring one BTF diagonal block: everything `refactor`/
/// `solve` need. `p`/`pinv` are the *numeric* row permutation this block's
/// partial pivoting actually chose (block-local indices) — the caller
/// (`factor.rs`) composes this with the block's own symbolic column order
/// to get the final global permutation, mirroring `KLU_factor`'s own
/// `Pnum[k+k1] = P[Pblock[k]+k1]` composition.
#[derive(Clone)]
pub struct BlockFactor {
    pub l_cols: Vec<Vec<(usize, f64)>>,
    pub u_cols: Vec<Vec<(usize, f64)>>,
    pub udiag: Vec<f64>,
    /// The block-local numeric row permutation this block's partial
    /// pivoting chose. The block-local *inverse* (`pinv`) is deliberately
    /// not stored here — it's only ever needed as a local variable during
    /// this function's own renumbering pass (see below); nothing
    /// downstream (`factor.rs`'s composition into the global permutation,
    /// `refactor.rs`, `solve.rs`) reads a block-local `pinv` instead of `p`.
    pub p: Vec<i64>,
}

/// Factors one `nk`-by-`nk` diagonal block — ports `KLU_kernel`'s main
/// per-column loop (`dfs`/`lsolve_symbolic`/`construct_column`/
/// `lsolve_numeric`/`lpivot`/`prune`, minus the packed-buffer growth
/// bookkeeping `Vec` makes unnecessary — see this module's doc comment).
///
/// `column_entries(k)` must return this block's column `k`'s nonzero
/// entries as `(block_local_row_or_negative_for_off_diagonal, value)`
/// pairs, matching `construct_column`'s own scatter (the caller —
/// `factor::factor_block` — is responsible for the global-to-block-local
/// row mapping via `q`/`PSinv`/`k1`, and for applying row scaling before
/// calling this).
///
/// **No external diagonal-preference input**: unlike column order (`q`,
/// fixed by symbolic analysis), the *row* order this function discovers is
/// entirely self-contained — `KLU_kernel` itself initializes `P[k] = k`
/// (identity) and only ever updates it via its own off-diagonal-pivot
/// swap-forward logic (`P[kbar] = diagrow`), never consulting the symbolic
/// `Symbolic::p` as an input. `Symbolic::p` matters only for *row mapping*
/// into block-local indices (via `PSinv`, already baked into
/// `column_entries`'s block-local row numbers by the caller) and for
/// composing this function's *output* permutation back into global row
/// indices afterward (`factor::factor_block`'s own job, mirroring
/// `klu_factor.c`'s `Pnum[k+k1] = P[Pblock[k]+k1]`) — confirmed by reading
/// `KLU_kernel_factor`'s own call signature, which never passes the
/// symbolic `P` in at all.
///
/// Returns `None` if the block is structurally or numerically singular
/// (`halt_if_singular` is always true for gridoxide's fixed `Options`).
pub fn factor_block(
    nk: usize,
    tol: f64,
    mut column_entries: impl FnMut(usize) -> Vec<(i64, f64)>,
    off_diagonal_out: &mut [Vec<(i64, f64)>], // off_diagonal_out[k] = this column's off-diagonal entries
) -> Option<BlockFactor> {
    let mut l_cols: Vec<Vec<(usize, f64)>> = Vec::with_capacity(nk);
    let mut u_cols: Vec<Vec<(usize, f64)>> = Vec::with_capacity(nk);
    let mut udiag = vec![0.0f64; nk];

    let mut x = vec![0.0f64; nk];
    let mut flag = vec![EMPTY; nk];
    let mut lpend = vec![EMPTY; nk];
    let mut stack = vec![0usize; nk];
    let mut ap_pos = vec![0i64; nk];

    // P[k] = k initially (all rows non-pivotal, mirroring KLU_kernel's own
    // init loop); pinv uses the FLIP(k) sentinel for "not yet pivotal".
    let mut p: Vec<i64> = (0..nk as i64).collect();
    let mut pinv: Vec<i64> = (0..nk as i64).map(flip).collect();

    for k in 0..nk {
        // Symbolic pattern of column k, from the block-local entries with
        // block-local row (negative entries are off-diagonal, handled by
        // construct_column below).
        let raw_entries = column_entries(k);
        let col_rows: Vec<usize> = raw_entries.iter().filter(|&&(r, _)| r >= 0).map(|&(r, _)| r as usize).collect();

        let (top, lik) =
            lsolve_symbolic(nk, k, &col_rows, &pinv, &l_cols, &mut stack, &mut flag, &lpend, &mut ap_pos);

        let mut off_diag = std::mem::take(&mut off_diagonal_out[k]);
        construct_column(&mut x, &raw_entries, &mut off_diag);
        off_diagonal_out[k] = off_diag;

        lsolve_numeric(&pinv, &l_cols, &stack, top, &mut x);

        let diagrow = p[k]; // might already be pivotal, per lpivot.c's own comment
        let (pivrow, pivot, _abs_pivot, l_col_k) = lpivot(diagrow, tol, &mut x, lik)?;

        // Extract U's column k (Stack[top..nk], values from X), clearing X.
        let mut u_col_k: Vec<(usize, f64)> = Vec::with_capacity(nk - top);
        for &j in &stack[top..] {
            let jnew = pinv[j] as usize;
            u_col_k.push((jnew, x[j]));
            x[j] = 0.0;
        }
        udiag[k] = pivot;

        // Log the pivot permutation (diagonal-preference bookkeeping,
        // kept literal -- see types.rs's doc comment on why).
        if pivrow != diagrow && pinv[diagrow as usize] < 0 {
            let kbar = flip(pinv[pivrow as usize]);
            p[kbar as usize] = diagrow;
            pinv[diagrow as usize] = flip(kbar);
        }
        p[k] = pivrow;
        pinv[pivrow as usize] = k as i64;

        prune(&mut lpend, &pinv, &u_col_k, pivrow as usize, &mut l_cols);

        l_cols.push(l_col_k);
        u_cols.push(u_col_k);
    }

    // Renumber L's row indices from original block-local rows to final
    // pivotal positions (mirrors KLU_kernel's own finalization loop:
    // `Li[i] = Pinv[Li[i]]`).
    for col in l_cols.iter_mut() {
        for entry in col.iter_mut() {
            entry.0 = pinv[entry.0] as usize;
        }
    }

    Some(BlockFactor { l_cols, u_cols, udiag, p })
}

/// Re-factors one BTF diagonal block against new numeric values, reusing an
/// *existing* `BlockFactor`'s pattern and pivot choices unchanged — ports
/// `klu_refactor.c`'s per-block loop (its `nk == 1`/`nk > 1` branches
/// collapse into one path here for the same reason `factor.rs`'s own doc
/// comment gives for `factor_block`: tracing the `nk == 1` case by hand
/// through this general loop produces the identical result a dedicated
/// singleton fast path would).
///
/// **No pivoting happens here** (`klu_refactor.c`'s own doc comment: "This
/// routine cannot do any numerical pivoting" — the caller's `columns` must
/// supply the *same sparsity pattern* `factor_block` originally saw, just
/// with new values, matching `KLU_refactor`'s own precondition). So
/// `block.p` (the row permutation) and every column's row *set*
/// (`block.l_cols[k]`/`block.u_cols[k]`, iterated in their already-recorded
/// order) are reused verbatim — only the numeric values change. Critically,
/// `block.u_cols[k]`'s stored order is *already* a valid topological order
/// for this column's Doolittle-style elimination (it came from
/// `lsolve_symbolic`'s DFS, which only ever visits already-pivotal
/// ancestors before their dependents), so no fresh symbolic pass is needed —
/// exactly why `KLU_refactor` itself never calls `dfs`/`lsolve_symbolic` at
/// all, unlike `KLU_kernel`.
///
/// Returns `false` if any diagonal pivot lands on exactly zero (numerically
/// singular — `halt_if_singular` is always true for gridoxide's fixed
/// `Options`, same contract as `factor_block`).
///
/// **In place, allocation-free** (unlike `factor_block`, which only runs
/// once per `PersistentSolver` lifetime): `block.p` and every column's row
/// *pattern* — `block.l_cols[k]`/`block.u_cols[k]`'s existing `.0` (row)
/// fields — are left completely untouched; only the `.1` (value) fields are
/// overwritten, iterating each column's existing entries via `.iter_mut()`
/// instead of building new `Vec`s. This is safe precisely because no
/// pivoting happens here (see above) — the row pattern this function reads
/// (to know *which* entries to overwrite) is exactly the pattern it also
/// writes into, so an in-place pass is a faithful translation of the same
/// math a build-fresh-`Vec`s version would do, just without the two
/// per-column heap allocations (`u_col_k`, `l_col_k`) that version needed.
/// Profiling showed those two allocations, repeated for every column on
/// every Newton iteration's refactor, as the dominant reason `KluNative`
/// ran ~2x slower than the FFI `Klu` backend — real KLU's own
/// `klu_refactor.c` overwrites its one packed buffer in place and
/// allocates nothing at all during a refactor.
///
/// `x` is `nk`-sized reusable scratch (the caller's own persistent buffer —
/// see `refactor::RefactorScratch` — sliced down to this block's size).
/// Like real KLU's own `Numeric->Xwork`, it's guaranteed all-zero on entry
/// and guaranteed all-zero again on return, for the *same reason* `Xwork`
/// is safe to reuse across every block and every call: every position
/// `construct_column` scatters a value into is always later read-and-
/// cleared during this same column's own U/L processing, by construction
/// of the elimination itself (never left dangling for a future call to
/// trip over).
///
/// `columns[k]` gives block-local column `k`'s raw entries (same contract
/// `factor_block`'s `column_entries(k)` closure had, just as a plain slice
/// instead of a closure — the caller already has to build this once into
/// its own reusable buffer to avoid *its* own allocation, so a closure
/// indirection here would add nothing).
pub fn refactor_block_in_place(
    nk: usize,
    block: &mut BlockFactor,
    x: &mut [f64],
    columns: &[Vec<(i64, f64)>],
    off_diagonal_out: &mut [Vec<(i64, f64)>],
) -> bool {
    let BlockFactor { l_cols, u_cols, udiag, .. } = block;

    for k in 0..nk {
        let mut off_diag = std::mem::take(&mut off_diagonal_out[k]);
        construct_column(x, &columns[k], &mut off_diag);
        off_diagonal_out[k] = off_diag;

        // Compute column k of U, updating column k of A in place -- same
        // order as originally recorded, so this is a valid elimination.
        // `j` is always < k (a valid topological order), so `l_cols[j]`
        // here already holds *this* refactor call's freshly-overwritten
        // column j, not a stale one -- matches `klu_refactor.c`, which
        // overwrites the shared LU buffer column-by-column in place, so its
        // own `GET_POINTER(LU, Lip, Llen, Li, Lx, j, ...)` for j < k always
        // reads values already refactored earlier in this same call.
        for entry in u_cols[k].iter_mut() {
            let j = entry.0;
            let ujk = x[j];
            x[j] = 0.0;
            entry.1 = ujk;
            for &(row, lij) in l_cols[j].iter() {
                x[row] -= lij * ujk;
            }
        }

        let ukk = x[k];
        x[k] = 0.0;
        if ukk == 0.0 {
            return false;
        }
        udiag[k] = ukk;

        // Gather and divide by the pivot to overwrite column k of L.
        for entry in l_cols[k].iter_mut() {
            let row = entry.0;
            entry.1 = x[row] / ukk;
            x[row] = 0.0;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense reference solve via Gaussian elimination with partial
    /// pivoting, sharing no code with `factor_block` -- an independent
    /// ground truth, matching `block_sparse.rs`'s own `dense_reference_
    /// solve` precedent.
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
                #[allow(clippy::needless_range_loop)] // needs both a[row] and a[col] simultaneously
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

    /// Solves `A x = b` using a `BlockFactor` (forward + back substitution
    /// through L/U, no BTF off-diagonal blocks -- single-block only).
    fn solve_with_factor(bf: &BlockFactor, b: &[f64]) -> Vec<f64> {
        let n = bf.p.len();
        let mut y = vec![0.0; n];
        for k in 0..n {
            y[k] = b[bf.p[k] as usize];
        }
        // Forward: L y = Pb (unit lower triangular, L stored by pivotal row).
        for k in 0..n {
            let yk = y[k];
            for &(row, lij) in &bf.l_cols[k] {
                y[row] -= lij * yk;
            }
        }
        // Backward: U x = y.
        let mut x = y;
        for k in (0..n).rev() {
            x[k] /= bf.udiag[k];
            let xk = x[k];
            for &(row, uij) in &bf.u_cols[k] {
                x[row] -= uij * xk;
            }
        }
        x
    }

    fn dense_from_entries(n: usize, entries: &[(usize, usize, f64)]) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0; n]; n];
        for &(r, c, v) in entries {
            a[r][c] += v;
        }
        a
    }

    fn factor_dense(n: usize, entries: &[(usize, usize, f64)]) -> BlockFactor {
        let mut by_col: Vec<Vec<(i64, f64)>> = vec![Vec::new(); n];
        for &(r, c, v) in entries {
            by_col[c].push((r as i64, v));
        }
        let mut off = vec![Vec::new(); n];
        factor_block(n, 1e-3, |k| by_col[k].clone(), &mut off).expect("should not be singular")
    }

    #[test]
    fn solve_diagonal_matches_dense() {
        let n = 3;
        let entries = vec![(0, 0, 4.0), (1, 1, 5.0), (2, 2, 6.0)];
        let bf = factor_dense(n, &entries);
        let a = dense_from_entries(n, &entries);
        let b = vec![8.0, 10.0, 12.0];
        let x = solve_with_factor(&bf, &b);
        let expected = dense_solve(&a, &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-10, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn solve_needs_pivoting() {
        // Zero on the diagonal at (0,0): must pivot.
        let n = 2;
        let entries = vec![(0, 1, 2.0), (1, 0, 3.0), (1, 1, 1.0)];
        let bf = factor_dense(n, &entries);
        let a = dense_from_entries(n, &entries);
        let b = vec![4.0, 5.0];
        let x = solve_with_factor(&bf, &b);
        let expected = dense_solve(&a, &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-10, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn solve_chain_matches_dense() {
        let n = 6;
        let mut entries = Vec::new();
        for i in 0..n {
            entries.push((i, i, 3.0 + i as f64 * 0.3));
        }
        for i in 0..n - 1 {
            entries.push((i, i + 1, -0.8));
            entries.push((i + 1, i, -0.7));
        }
        let bf = factor_dense(n, &entries);
        let a = dense_from_entries(n, &entries);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        let x = solve_with_factor(&bf, &b);
        let expected = dense_solve(&a, &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-8, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn solve_asymmetric_matches_dense() {
        // Deliberately non-symmetric off-diagonal values (the real
        // Jacobian's own shape) -- the class of matrix that caught
        // block_sparse.rs's row/col bug this session.
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
        let bf = factor_dense(n, &entries);
        let a = dense_from_entries(n, &entries);
        let b = vec![1.0, 2.0, -1.0, 0.5];
        let x = solve_with_factor(&bf, &b);
        let expected = dense_solve(&a, &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-8, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn structurally_singular_returns_none() {
        let n = 2;
        let entries = vec![(0, 1, 1.0), (1, 1, 1.0)]; // column 0 empty
        let mut by_col: Vec<Vec<(i64, f64)>> = vec![Vec::new(); n];
        for &(r, c, v) in &entries {
            by_col[c].push((r as i64, v));
        }
        let mut off = vec![Vec::new(); n];
        assert!(factor_block(n, 1e-3, |k| by_col[k].clone(), &mut off).is_none());
    }

    fn refactor_dense(n: usize, prev: &BlockFactor, entries: &[(usize, usize, f64)]) -> BlockFactor {
        let mut by_col: Vec<Vec<(i64, f64)>> = vec![Vec::new(); n];
        for &(r, c, v) in entries {
            by_col[c].push((r as i64, v));
        }
        let mut off = vec![Vec::new(); n];
        let mut x = vec![0.0; n];
        let mut block = prev.clone();
        assert!(refactor_block_in_place(n, &mut block, &mut x, &by_col, &mut off), "should not be singular");
        block
    }

    #[test]
    fn refactor_with_identical_values_matches_original() {
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
        let bf = factor_dense(n, &entries);
        let refactored = refactor_dense(n, &bf, &entries);

        // Same pattern (unchanged pivot order) and, since the arithmetic is
        // identical, the exact same values -- not just numerically close.
        assert_eq!(refactored.p, bf.p);
        for k in 0..n {
            assert_eq!(refactored.l_cols[k], bf.l_cols[k], "L column {k}");
            assert_eq!(refactored.u_cols[k], bf.u_cols[k], "U column {k}");
        }
        assert_eq!(refactored.udiag, bf.udiag);
    }

    #[test]
    fn refactor_with_new_values_matches_independent_dense_solve() {
        // Same sparsity pattern as the fixture above, different (still
        // diagonally-dominant, so no singular pivot) values -- mirrors
        // `sparse_klu.rs`'s own `real_sparse_system_refactor_reuses_
        // symbolic` fixture shape. The refactored solve must match an
        // independent dense solve of the *new* matrix -- Ax=b has a unique
        // solution regardless of which pivot order the factorization used,
        // so this doesn't depend on refactor happening to reuse a pivot
        // order a from-scratch factorization would also have chosen.
        let n = 4;
        let original = vec![
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
        let bf = factor_dense(n, &original);

        let updated = vec![
            (0, 0, 9.0),
            (1, 1, 4.0),
            (2, 2, 6.5),
            (3, 3, 3.0),
            (0, 1, -1.2),
            (1, 0, -0.2),
            (1, 2, -0.4),
            (2, 1, -0.1),
            (2, 3, 0.9),
            (3, 2, -0.3),
            (0, 3, 0.05),
            (3, 0, 0.4),
        ];
        let refactored = refactor_dense(n, &bf, &updated);

        let a = dense_from_entries(n, &updated);
        let b = vec![1.0, 2.0, -1.0, 0.5];
        let x = solve_with_factor(&refactored, &b);
        let expected = dense_solve(&a, &b);
        for i in 0..n {
            assert!((x[i] - expected[i]).abs() < 1e-8, "index {i}: {} vs {}", x[i], expected[i]);
        }
    }

    #[test]
    fn refactor_detects_new_singularity() {
        // Same pattern as the diagonal fixture, but this time the new
        // values make the block genuinely singular (a diagonal entry
        // reaches exactly zero during elimination) -- refactor must report
        // that (`None`), matching KLU_refactor's own IS_ZERO(ukk) check
        // rather than silently dividing by zero.
        let n = 2;
        let entries = vec![(0, 0, 4.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 1.0)];
        let bf = factor_dense(n, &entries);

        let singular = vec![(0, 0, 0.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 0.0)];
        let mut by_col: Vec<Vec<(i64, f64)>> = vec![Vec::new(); n];
        for &(r, c, v) in &singular {
            by_col[c].push((r as i64, v));
        }
        let mut off = vec![Vec::new(); n];
        let mut x = vec![0.0; n];
        let mut block = bf.clone();
        assert!(!refactor_block_in_place(n, &mut block, &mut x, &by_col, &mut off));
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_real_klu_on_random_matrices() {
        // Fixed, hand-rolled LCG (no extra dependency), same style as
        // block_sparse.rs's / btf::tests's / amd::tests's random tests.
        let mut seed: u64 = 0xA24BAED4963EE407;
        let mut next_f64 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };

        for trial in 0..40 {
            let n = 3 + (trial % 12);
            // Diagonally-dominant (guarantees no pivoting failure for
            // either implementation) but deliberately asymmetric random
            // matrix, matching a real Jacobian's own shape.
            let mut entries: Vec<(usize, usize, f64)> = Vec::new();
            let mut row_sums = vec![0.0f64; n];
            let mut off_entries: Vec<(usize, usize, f64)> = Vec::new();
            #[allow(clippy::needless_range_loop)] // `i` is also compared against `j`, not just an index
            for i in 0..n {
                // BTreeSet dedups (i, j) pairs -- factor_block's column_entries
                // contract expects already-deduplicated entries per column,
                // matching build_csc_structure's convention elsewhere in
                // gridoxide (KluRealSystem sums duplicates itself, so feeding
                // it raw duplicates would silently produce a *different*,
                // still-valid-looking matrix than what factor_block sees).
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

            let mut by_col: Vec<Vec<(i64, f64)>> = vec![Vec::new(); n];
            for &(r, c, v) in &entries {
                by_col[c].push((r as i64, v));
            }
            let mut off = vec![Vec::new(); n];
            let bf = factor_block(n, 1e-3, |k| by_col[k].clone(), &mut off)
                .unwrap_or_else(|| panic!("trial {trial} (n={n}): unexpectedly singular"));

            let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.3).collect();
            let rust_x = solve_with_factor(&bf, &b);

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
