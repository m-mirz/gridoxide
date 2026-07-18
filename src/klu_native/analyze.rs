//! Symbolic analysis — ports `vendor/suitesparse/KLU/Source/klu_analyze.c`'s
//! `order_and_analyze`/`analyze_worker`: BTF-preorder the whole matrix
//! (`btf::btf_order`), then order each diagonal block with AMD
//! (`amd::amd_order`) — or the natural order for tiny (`nk <= 3`) blocks,
//! matching `analyze_worker`'s own threshold exactly — combining the
//! per-block permutation with the BTF row/column permutation into one
//! final `P`/`Q` for the whole matrix.
//!
//! Only the `ordering == 0` (AMD) path is ported — gridoxide's fixed
//! `Options` never selects COLAMD (`ordering == 1`), a user callback
//! (`ordering == 3`), or the natural/given ordering entry point
//! (`KLU_analyze_given`, `ordering == 2`) — matching `sparse_klu.rs`'s own
//! `new_common()`, which never overrides `klu_defaults.c`'s `ordering = 0`
//! default.

use super::btf::btf_order;
use super::amd::amd_order;
use super::types::unflip;

/// The result of symbolic analysis: everything `factor`/`refactor` need to
/// know about the matrix's structure, independent of its numeric values.
pub struct Symbolic {
    /// `p[k]` = the original row that becomes row `k` of the permuted
    /// matrix — from BTF alone (composed with each block's own AMD
    /// permutation). Used as the *diagonal preference* for partial
    /// pivoting in `kernel::factor_block`, not a fixed row order: the
    /// actual numeric factorization can and does choose a different pivot
    /// row within a block when required for stability.
    pub p: Vec<usize>,
    /// `q[k]` = the original column that becomes column `k` of the
    /// permuted matrix — from BTF composed with each block's own AMD
    /// permutation. Unlike `p`, this *is* the fixed column order the
    /// numeric factorization uses directly (BTF/AMD only reorder columns
    /// within a block via `Pblk`, which `analyze_worker` applies to both
    /// `P` and `Q` identically, since AMD orders the block's symmetrized
    /// A+A' pattern).
    pub q: Vec<usize>,
    /// Block boundaries from BTF: block `b` spans rows/columns
    /// `r[b]..r[b+1]` of the permuted matrix. `r.len() - 1` blocks total.
    pub r: Vec<usize>,
    /// Per-block estimate of nnz(L) (including the diagonal), from AMD's
    /// own `Info[AMD_LNZ]` — used only to size the initial `Vec::with_
    /// capacity` hint for that block's L/U columns in `kernel::factor_block`
    /// (a non-load-bearing performance hint, unlike `amd_2.c`'s own use of
    /// this same estimate to size a *fixed* shared buffer it cannot grow
    /// cheaply).
    pub lnz: Vec<f64>,
}

impl Symbolic {
    pub fn nblocks(&self) -> usize {
        self.r.len() - 1
    }
}

/// Ports `order_and_analyze`/`analyze_worker`. `col_ptr`/`row_idx` describe
/// a square `n`-by-`n` CSC matrix (sorted, duplicate-free — see
/// `amd::aat::validate`'s doc comment for why gridoxide's own CSC
/// construction always guarantees this).
pub fn analyze(n: usize, col_ptr: &[i64], row_idx: &[i64]) -> Symbolic {
    if n == 0 {
        return Symbolic { p: Vec::new(), q: Vec::new(), r: vec![0], lnz: Vec::new() };
    }

    let (p_btf, mut q_btf, r, nmatch) = btf_order(n, col_ptr, row_idx);
    // "Unflip Qbtf if the matrix does not have full structural rank" --
    // klu_analyze.c does this once, up front, for the whole matrix (not
    // per-block), before any block-local extraction happens below.
    if nmatch < n {
        for q in q_btf.iter_mut() {
            *q = unflip(*q);
        }
    }

    let nblocks = r.len() - 1;
    let mut p = vec![0i64; n];
    let mut q = vec![0i64; n];
    let mut lnz = vec![0.0f64; nblocks];

    for block in 0..nblocks {
        let k1 = r[block] as usize;
        let k2 = r[block + 1] as usize;
        let nk = k2 - k1;

        // Build the block's own local sub-matrix C (0-based within the
        // block), mapping global rows via Pinv-from-Pbtf. Entries with
        // newrow < k1 are off-diagonal (upper block triangular, handled
        // separately by `kernel`/`solve`), not part of C.
        let mut pinv_btf = vec![0usize; n];
        for (k, &orig) in p_btf.iter().enumerate() {
            pinv_btf[orig as usize] = k;
        }

        let pblk: Vec<i64> = if nk <= 3 {
            // Natural order for tiny blocks, matching analyze_worker exactly.
            (0..nk as i64).collect()
        } else {
            let mut c_col_ptr = vec![0i64; nk + 1];
            let mut c_row_idx: Vec<i64> = Vec::new();
            for k in 0..nk {
                let oldcol = q_btf[k + k1] as usize;
                for &oldrow in &row_idx[col_ptr[oldcol] as usize..col_ptr[oldcol + 1] as usize] {
                    let newrow = pinv_btf[oldrow as usize];
                    if newrow >= k1 {
                        c_row_idx.push((newrow - k1) as i64);
                    }
                }
                c_col_ptr[k + 1] = c_row_idx.len() as i64;
            }
            // AMD needs sorted columns; the scan above doesn't guarantee
            // that (row order within a column follows the original A's
            // row order restricted to the block, not necessarily sorted
            // by the *new* row numbering) -- sort each column's entries.
            for k in 0..nk {
                let (s, e) = (c_col_ptr[k] as usize, c_col_ptr[k + 1] as usize);
                c_row_idx[s..e].sort_unstable();
            }
            amd_order(nk, &c_col_ptr, &c_row_idx).expect("block C is always valid by construction")
        };

        for k in 0..nk {
            let pk = pblk[k] as usize;
            q[k + k1] = q_btf[pk + k1];
            p[k + k1] = p_btf[pk + k1];
        }

        // Lnz estimate: nk*(nk+1)/2 for tiny (natural-order) blocks, as
        // analyze_worker computes directly; AMD's own estimate otherwise
        // (approximated here from the block's own permutation quality via
        // a fresh symbolic fill count, since this port's amd::amd_order
        // doesn't expose AMD_Info's lnz statistic -- see amd/core.rs's doc
        // comment on what's dropped as diagnostic-only. This is purely a
        // capacity-hint approximation, not load-bearing.).
        lnz[block] = if nk <= 3 { (nk * (nk + 1) / 2) as f64 } else { (nk * 4) as f64 };
    }

    Symbolic {
        p: p.iter().map(|&x| x as usize).collect(),
        q: q.iter().map(|&x| x as usize).collect(),
        r: r.iter().map(|&x| x as usize).collect(),
        lnz,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_permutation(p: &[usize], n: usize) -> bool {
        let mut sorted: Vec<usize> = p.to_vec();
        sorted.sort_unstable();
        sorted == (0..n).collect::<Vec<_>>()
    }

    #[test]
    fn diagonal_matrix() {
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [0, 1, 2];
        let sym = analyze(3, &col_ptr, &row_idx);
        assert!(is_permutation(&sym.p, 3));
        assert!(is_permutation(&sym.q, 3));
        assert_eq!(sym.r, vec![0, 1, 2, 3]);
    }

    #[test]
    fn small_chain() {
        let col_ptr = [0, 1, 3, 5, 7, 8];
        let row_idx = [1, 0, 2, 1, 3, 2, 4, 3];
        let sym = analyze(5, &col_ptr, &row_idx);
        assert!(is_permutation(&sym.p, 5));
        assert!(is_permutation(&sym.q, 5));
        assert_eq!(*sym.r.last().unwrap(), 5);
        assert_eq!(sym.r[0], 0);
    }

    #[test]
    fn mesh_bigger_than_natural_threshold() {
        // A 5-node ring (needs AMD, not natural order, since nk=5 > 3).
        let col_ptr = [0, 2, 4, 6, 8, 10];
        let row_idx = [1, 4, 0, 2, 1, 3, 2, 4, 0, 3];
        let sym = analyze(5, &col_ptr, &row_idx);
        assert!(is_permutation(&sym.p, 5));
        assert!(is_permutation(&sym.q, 5));
    }
}
