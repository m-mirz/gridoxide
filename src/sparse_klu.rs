//! Thin internal wrapper around the vendored SuiteSparse KLU solver
//! (`vendor/suitesparse/`, compiled by `build.rs`), mirroring `sparse.rs`'s
//! shape exactly so it's a drop-in third backend. See the README's "Sparse
//! solver" section for why this exists and `vendor/suitesparse/
//! PROVENANCE.md` for what was vendored and its licensing.
//!
//! All raw-pointer FFI handling is confined to this one file — nothing raw
//! leaks into `solver.rs`/`network.rs`, the same isolation principle
//! `sparse.rs` already applies to `faer` types.
//!
//! KLU represents complex values as interleaved `[re, im, re, im, ...]`
//! `f64` pairs, not a native complex type — `klu_z_factor`/`klu_z_solve`
//! share the exact same `*mut f64` signature as their real counterparts,
//! just interpreting the buffer differently. `pack_complex_values`/the
//! `Vec<f64>` RHS buffers in `klu_solve_complex` do this conversion.

use num_complex::Complex;

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/klu_bindings.rs"));
}
use bindings::{
    klu_analyze, klu_common, klu_defaults, klu_factor, klu_free_numeric, klu_free_symbolic, klu_numeric,
    klu_refactor, klu_solve, klu_symbolic, klu_z_factor, klu_z_solve,
};

const KLU_OK: i32 = 0;

/// Builds a KLU-ready CSC structure (column pointers + sorted row indices)
/// from a set of `(row, col)` index pairs, merging duplicates — the same
/// accumulation semantics used everywhere else in gridoxide's sparse code
/// (e.g. `network::YBus`, parallel branches between the same two buses).
/// Returns `(col_ptr, row_idx, groups)`, where `groups[k]` lists the
/// original `entries` indices contributing to the `k`-th CSC position.
fn build_csc_structure(n: usize, pairs: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>, Vec<Vec<usize>>) {
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&i| (pairs[i].1, pairs[i].0));

    let mut col_ptr = vec![0i32; n + 1];
    let mut row_idx: Vec<i32> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();

    let mut idx = 0;
    while idx < order.len() {
        let (row, col) = pairs[order[idx]];
        let mut group = vec![order[idx]];
        let mut k = idx + 1;
        while k < order.len() && pairs[order[k]] == (row, col) {
            group.push(order[k]);
            k += 1;
        }
        row_idx.push(row as i32);
        groups.push(group);
        col_ptr[col + 1] += 1;
        idx = k;
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    (col_ptr, row_idx, groups)
}

fn pack_real_values(entries: &[(usize, usize, f64)], groups: &[Vec<usize>]) -> Vec<f64> {
    groups.iter().map(|g| g.iter().map(|&i| entries[i].2).sum()).collect()
}

fn pack_complex_values(entries: &[(usize, usize, Complex<f64>)], groups: &[Vec<usize>]) -> Vec<f64> {
    let mut out = Vec::with_capacity(groups.len() * 2);
    for g in groups {
        let sum: Complex<f64> = g.iter().map(|&i| entries[i].2).sum();
        out.push(sum.re);
        out.push(sum.im);
    }
    out
}

fn new_common() -> klu_common {
    let mut common: klu_common = unsafe { std::mem::zeroed() };
    unsafe { klu_defaults(&mut common) };
    common
}

/// A real sparse system whose sparsity *pattern* is fixed across repeated
/// solves — mirrors `sparse::RealSparseSystem`'s role but backed by KLU
/// instead of `faer`. Caches the symbolic analysis (`klu_analyze`) and a
/// numeric factorization object across calls, using `klu_refactor` for
/// cheap numeric-only re-factorization — the same pattern already validated
/// this session for `RealSparseSystem`/`BlockLu::refactor`.
pub struct KluRealSystem {
    n: usize,
    col_ptr: Vec<i32>,
    row_idx: Vec<i32>,
    groups: Vec<Vec<usize>>,
    common: klu_common,
    symbolic: *mut klu_symbolic,
    numeric: *mut klu_numeric,
}

impl KluRealSystem {
    /// Builds the symbolic factorization from an initial sparsity pattern.
    /// Subsequent calls to `factor_and_solve` must supply entries with the
    /// exact same `(row, col)` pairs in the exact same order (values may
    /// differ) — the cached CSC position mapping assumes positional
    /// correspondence, not just set equality.
    pub fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        let pairs: Vec<(usize, usize)> = entries.iter().map(|&(r, c, _)| (r, c)).collect();
        let (mut col_ptr, mut row_idx, groups) = build_csc_structure(n, &pairs);
        let mut values = pack_real_values(entries, &groups);

        let mut common = new_common();
        let symbolic = unsafe { klu_analyze(n as i32, col_ptr.as_mut_ptr(), row_idx.as_mut_ptr(), &mut common) };
        if symbolic.is_null() {
            return None;
        }
        let mut numeric = unsafe {
            klu_factor(col_ptr.as_mut_ptr(), row_idx.as_mut_ptr(), values.as_mut_ptr(), symbolic, &mut common)
        };
        if numeric.is_null() || common.status != KLU_OK {
            let mut symbolic = symbolic;
            unsafe { klu_free_symbolic(&mut symbolic, &mut common) };
            return None;
        }
        // numeric is non-null and owned from here; suppress "unused mut" if
        // the compiler considers it otherwise (kept `mut` since it's freed
        // via `&mut` in Drop).
        let _ = &mut numeric;

        Some(Self { n, col_ptr, row_idx, groups, common, symbolic, numeric })
    }

    /// Numeric-only refactorization against the cached symbolic pattern,
    /// then solves `A x = b`. Returns `None` if the matrix is singular.
    ///
    /// Success/failure is judged from `common.status` alone (`KLU_OK` = 0,
    /// documented as consistent across every KLU function — `klu.h`:
    /// "`status`: `KLU_OK` if OK, `< 0` if error", with `> 0` a warning
    /// including `KLU_SINGULAR`), not from `klu_refactor`/`klu_solve`'s own
    /// return values — confirmed by direct testing that those follow a
    /// *different*, boolean TRUE(1)/FALSE(0) convention (documented for
    /// `klu_refactor` in `klu.h`: "return TRUE if successful, FALSE
    /// otherwise") rather than `klu_factor`'s/`klu_analyze`'s "0 = OK, < 0 =
    /// error" status-code convention. Mixing the two up here previously
    /// caused every successful refactor to be misread as a failure.
    pub fn factor_and_solve(&mut self, entries: &[(usize, usize, f64)], rhs: &[f64]) -> Option<Vec<f64>> {
        let mut values = pack_real_values(entries, &self.groups);
        unsafe {
            klu_refactor(
                self.col_ptr.as_mut_ptr(),
                self.row_idx.as_mut_ptr(),
                values.as_mut_ptr(),
                self.symbolic,
                self.numeric,
                &mut self.common,
            )
        };
        if self.common.status != KLU_OK {
            return None;
        }

        let mut b = rhs.to_vec();
        unsafe { klu_solve(self.symbolic, self.numeric, self.n as i32, 1, b.as_mut_ptr(), &mut self.common) };
        if self.common.status != KLU_OK || b.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(b)
    }
}

impl Drop for KluRealSystem {
    fn drop(&mut self) {
        unsafe {
            klu_free_numeric(&mut self.numeric, &mut self.common);
            klu_free_symbolic(&mut self.symbolic, &mut self.common);
        }
    }
}

/// Solves a one-shot general complex sparse linear system `A x = b` via
/// KLU. Mirrors `sparse::solve_complex`'s role for the `Scalar`/`faer`
/// backend. Returns `None` if the matrix is singular.
pub fn klu_solve_complex(
    n: usize,
    entries: &[(usize, usize, Complex<f64>)],
    rhs: &[Complex<f64>],
) -> Option<Vec<Complex<f64>>> {
    let pairs: Vec<(usize, usize)> = entries.iter().map(|&(r, c, _)| (r, c)).collect();
    let (mut col_ptr, mut row_idx, groups) = build_csc_structure(n, &pairs);
    let mut values = pack_complex_values(entries, &groups);

    let mut common = new_common();
    let mut symbolic = unsafe { klu_analyze(n as i32, col_ptr.as_mut_ptr(), row_idx.as_mut_ptr(), &mut common) };
    if symbolic.is_null() {
        return None;
    }
    let mut numeric = unsafe {
        klu_z_factor(col_ptr.as_mut_ptr(), row_idx.as_mut_ptr(), values.as_mut_ptr(), symbolic, &mut common)
    };

    let result = if numeric.is_null() || common.status != KLU_OK {
        None
    } else {
        let mut b: Vec<f64> = Vec::with_capacity(n * 2);
        for &v in rhs {
            b.push(v.re);
            b.push(v.im);
        }
        unsafe { klu_z_solve(symbolic, numeric, n as i32, 1, b.as_mut_ptr(), &mut common) };
        if common.status != KLU_OK || b.iter().any(|v| !v.is_finite()) {
            None
        } else {
            Some((0..n).map(|i| Complex::new(b[2 * i], b[2 * i + 1])).collect())
        }
    };

    unsafe {
        if !numeric.is_null() {
            klu_free_numeric(&mut numeric, &mut common);
        }
        klu_free_symbolic(&mut symbolic, &mut common);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_complex_simple_system() {
        // [[2, 1], [1, 3]] * [x0, x1] = [5, 10]  =>  x0=1, x1=3
        let entries = vec![
            (0, 0, Complex::new(2.0, 0.0)),
            (0, 1, Complex::new(1.0, 0.0)),
            (1, 0, Complex::new(1.0, 0.0)),
            (1, 1, Complex::new(3.0, 0.0)),
        ];
        let rhs = vec![Complex::new(5.0, 0.0), Complex::new(10.0, 0.0)];
        let x = klu_solve_complex(2, &entries, &rhs).unwrap();
        assert!((x[0] - Complex::new(1.0, 0.0)).norm() < 1e-10);
        assert!((x[1] - Complex::new(3.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn solve_complex_duplicate_entries_sum() {
        // (0,0) contributed twice as 1.0+1.0=2.0, matching dense += semantics.
        let entries = vec![
            (0, 0, Complex::new(1.0, 0.0)),
            (0, 0, Complex::new(1.0, 0.0)),
            (0, 1, Complex::new(1.0, 0.0)),
            (1, 0, Complex::new(1.0, 0.0)),
            (1, 1, Complex::new(3.0, 0.0)),
        ];
        let rhs = vec![Complex::new(5.0, 0.0), Complex::new(10.0, 0.0)];
        let x = klu_solve_complex(2, &entries, &rhs).unwrap();
        assert!((x[0] - Complex::new(1.0, 0.0)).norm() < 1e-10);
        assert!((x[1] - Complex::new(3.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn real_sparse_system_refactor_reuses_symbolic() {
        let entries_a = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let mut sys = KluRealSystem::new(2, &entries_a).unwrap();
        let x = sys.factor_and_solve(&entries_a, &[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);

        // Same sparsity pattern, different numeric values (as across NR iterations).
        let entries_b = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)];
        // [[4,1],[1,2]] * x = [6, 5] => x0=1, x1=2
        let x2 = sys.factor_and_solve(&entries_b, &[6.0, 5.0]).unwrap();
        assert!((x2[0] - 1.0).abs() < 1e-10);
        assert!((x2[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn singular_matrix_returns_none() {
        let entries = vec![(0, 0, 1.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 2.0)];
        let sys = KluRealSystem::new(2, &entries);
        if let Some(mut sys) = sys {
            assert!(sys.factor_and_solve(&entries, &[1.0, 2.0]).is_none());
        }
    }
}
