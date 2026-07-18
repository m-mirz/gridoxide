//! Thin internal wrapper around Intel oneMKL's PARDISO sparse direct solver,
//! mirroring `sparse_klu.rs`'s shape and role — a fifth, opt-in
//! `JacobianBackend` — but linked dynamically against a locally-installed
//! oneMKL instead of a vendored, statically-compiled library, since MKL is
//! proprietary and can't be vendored the way SuiteSparse's KLU can. See the
//! README's "Experimental backends" section and this crate's `pardiso`
//! Cargo feature doc comment for why this exists and how it's linked.
//!
//! All raw-pointer FFI handling is confined to this one file, matching
//! `sparse_klu.rs`'s own isolation principle.
//!
//! PARDISO's C API is a single function, `pardiso(pt, maxfct, mnum, mtype,
//! phase, n, a, ia, ja, perm, nrhs, iparm, msglvl, b, x, error)`, called
//! repeatedly with different `phase` values against a persistent opaque
//! handle `pt` — unlike KLU's separate `klu_analyze`/`klu_factor`/
//! `klu_refactor`/`klu_solve` functions. `phase = 11` is analysis (done
//! once, in `new`), `phase = 22` is numerical factorization (repeated on
//! every `factor_and_solve`, PARDISO's equivalent of `klu_refactor`),
//! `phase = 33` is the solve itself, and `phase = -1` releases `pt`'s
//! internal memory (done in `Drop`). `mtype = 11` selects "real and
//! nonsymmetric" — the correct type for gridoxide's general Jacobian (the
//! same shape KLU already solves). `iparm[34] = 1` switches `ia`/`ja` to
//! 0-based (C-style) indexing, matching gridoxide's existing CSR/CSC
//! construction elsewhere and avoiding an off-by-one translation layer;
//! `iparm[0] = 1` tells PARDISO to use the caller-supplied `iparm` array
//! rather than silently resetting it to all-defaults on every call.
//!
//! Unlike KLU (CSC), PARDISO expects **CSR** — `build_csr_structure` is the
//! row-major analogue of `sparse_klu::build_csc_structure`.

use std::ffi::c_void;

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/pardiso_bindings.rs"));
}
use bindings::{pardiso, pardisoinit};

/// Real and nonsymmetric — gridoxide's Jacobian is a general (not
/// symmetric) real matrix, the same `mtype` KLU is agnostic to.
const MTYPE_REAL_NONSYMMETRIC: i32 = 11;

const PHASE_ANALYSIS: i32 = 11;
const PHASE_NUMERICAL_FACTORIZATION: i32 = 22;
const PHASE_SOLVE: i32 = 33;

/// `iparm[13]` (0-based; `iparm(14)` in Intel's 1-based Fortran-style docs):
/// "Number of perturbed pivots" after a numerical factorization. **Confirmed
/// necessary by direct testing**, not just documentation: PARDISO's default
/// nonsymmetric preprocessing (maximum weighted matching + scaling,
/// `iparm[9]`/`iparm[12]`) *silently perturbs* any pivot smaller than a
/// threshold rather than failing outright — on an exactly row-proportional
/// (perfectly singular) 2×2 test fixture, `error` came back `0` (success!)
/// with a plausible-looking but meaningless solution, while this counter
/// was nonzero. Unlike KLU (whose `common.status` alone reliably flags
/// singularity), PARDISO needs this checked in addition to `error` — a
/// nonzero count here is treated as a solve failure throughout this file,
/// matching gridoxide's other backends never silently returning a
/// perturbed/inaccurate solution for what should be a hard Jacobian-
/// singularity signal.
const IPARM_NUM_PERTURBED_PIVOTS: usize = 13;
const PHASE_RELEASE_MEMORY: i32 = -1;

/// Builds a PARDISO-ready CSR structure (row pointers + sorted column
/// indices) from a set of `(row, col)` index pairs, merging duplicates —
/// mirrors `sparse_klu::build_csc_structure`'s accumulation semantics
/// exactly, just row-major instead of column-major. Returns `(row_ptr,
/// col_idx, groups)`, where `groups[k]` lists the original `entries`
/// indices contributing to the `k`-th CSR position.
fn build_csr_structure(n: usize, pairs: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>, Vec<Vec<usize>>) {
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&i| (pairs[i].0, pairs[i].1));

    let mut row_ptr = vec![0i32; n + 1];
    let mut col_idx: Vec<i32> = Vec::new();
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
        col_idx.push(col as i32);
        groups.push(group);
        row_ptr[row + 1] += 1;
        idx = k;
    }
    for r in 0..n {
        row_ptr[r + 1] += row_ptr[r];
    }
    (row_ptr, col_idx, groups)
}

fn pack_real_values(entries: &[(usize, usize, f64)], groups: &[Vec<usize>]) -> Vec<f64> {
    groups.iter().map(|g| g.iter().map(|&i| entries[i].2).sum()).collect()
}

/// A real sparse system whose sparsity *pattern* is fixed across repeated
/// solves — mirrors `sparse_klu::KluRealSystem`'s role but backed by Intel
/// oneMKL PARDISO instead of vendored SuiteSparse KLU. Caches the symbolic
/// analysis (PARDISO phase 11) and the opaque factorization handle `pt`
/// across calls, using phase 22 for cheap numeric-only refactorization on
/// every `factor_and_solve` — the same reuse pattern already validated for
/// `RealSparseSystem`/`KluRealSystem`.
pub struct PardisoRealSystem {
    n: usize,
    row_ptr: Vec<i32>,
    col_idx: Vec<i32>,
    groups: Vec<Vec<usize>>,
    /// PARDISO's persistent opaque handle — a `void*[64]` array in every
    /// upstream C example. `_MKL_DSS_HANDLE_t` (the type `pardiso`/
    /// `pardisoinit` actually declare their `pt` parameter as) is a plain
    /// `typedef void *`, i.e. the *element* type, not the array — so every
    /// call below passes `self.pt.as_mut_ptr()` (a pointer to the array's
    /// first element), mirroring C's own array-to-pointer decay, not the
    /// array by value.
    pt: [*mut c_void; 64],
    iparm: [i32; 64],
}

impl PardisoRealSystem {
    /// Builds the symbolic analysis (PARDISO phase 11) and an initial
    /// numerical factorization (phase 22) from an initial sparsity pattern.
    /// Subsequent calls to `factor_and_solve` must supply entries with the
    /// exact same `(row, col)` pairs in the exact same order (values may
    /// differ) — the cached CSR position mapping assumes positional
    /// correspondence, not just set equality. Returns `None` if either
    /// phase reports an error (e.g. a structurally or numerically singular
    /// matrix).
    pub fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        let pairs: Vec<(usize, usize)> = entries.iter().map(|&(r, c, _)| (r, c)).collect();
        let (row_ptr, col_idx, groups) = build_csr_structure(n, &pairs);
        let mut values = pack_real_values(entries, &groups);

        let mut pt: [*mut c_void; 64] = [std::ptr::null_mut(); 64];
        let mtype = MTYPE_REAL_NONSYMMETRIC;
        let mut iparm: [i32; 64] = [0; 64];
        unsafe { pardisoinit(pt.as_mut_ptr() as *mut c_void, &mtype, iparm.as_mut_ptr()) };
        iparm[0] = 1; // use our own iparm, not silent per-call defaults
        iparm[34] = 1; // 0-based (C-style) row_ptr/col_idx indexing

        let mut system = Self { n, row_ptr, col_idx, groups, pt, iparm };

        let a = values.as_mut_ptr() as *const c_void;
        if system.call(PHASE_ANALYSIS, a, std::ptr::null_mut(), std::ptr::null_mut()) != 0 {
            // `system`'s `Drop` runs phase -1 to release anything phase 11
            // itself may have already allocated.
            return None;
        }
        if system.call(PHASE_NUMERICAL_FACTORIZATION, a, std::ptr::null_mut(), std::ptr::null_mut()) != 0
            || system.iparm[IPARM_NUM_PERTURBED_PIVOTS] != 0
        {
            return None;
        }

        Some(system)
    }

    /// Numeric-only refactorization (phase 22) against the cached symbolic
    /// pattern (phase 11), then solves `A x = b` (phase 33). Returns `None`
    /// if either phase reports an error, if PARDISO had to perturb a pivot
    /// (see [`IPARM_NUM_PERTURBED_PIVOTS`]'s doc comment), or if the
    /// solution contains a non-finite value.
    pub fn factor_and_solve(&mut self, entries: &[(usize, usize, f64)], rhs: &[f64]) -> Option<Vec<f64>> {
        let mut values = pack_real_values(entries, &self.groups);
        let a = values.as_mut_ptr() as *const c_void;
        if self.call(PHASE_NUMERICAL_FACTORIZATION, a, std::ptr::null_mut(), std::ptr::null_mut()) != 0
            || self.iparm[IPARM_NUM_PERTURBED_PIVOTS] != 0
        {
            return None;
        }

        let mut b = rhs.to_vec();
        let mut x: Vec<f64> = vec![0.0; self.n];
        let error =
            self.call(PHASE_SOLVE, a, b.as_mut_ptr() as *mut c_void, x.as_mut_ptr() as *mut c_void);
        if error != 0 || x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(x)
    }

    /// One `pardiso()` call at the given `phase`, using this system's
    /// cached `row_ptr`/`col_idx`/`pt`/`iparm`. Returns PARDISO's `error`
    /// output (`0` = success).
    fn call(&mut self, phase: i32, a: *const c_void, b: *mut c_void, x: *mut c_void) -> i32 {
        let maxfct: i32 = 1;
        let mnum: i32 = 1;
        let mtype: i32 = MTYPE_REAL_NONSYMMETRIC;
        let n = self.n as i32;
        let nrhs: i32 = 1;
        let msglvl: i32 = 0;
        let mut error: i32 = 0;
        let perm: *mut i32 = std::ptr::null_mut();
        unsafe {
            pardiso(
                self.pt.as_mut_ptr() as *mut c_void,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &n,
                a,
                self.row_ptr.as_ptr(),
                self.col_idx.as_ptr(),
                perm,
                &nrhs,
                self.iparm.as_mut_ptr(),
                &msglvl,
                b,
                x,
                &mut error,
            );
        }
        error
    }
}

impl Drop for PardisoRealSystem {
    fn drop(&mut self) {
        self.call(PHASE_RELEASE_MEMORY, std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut());
    }
}

#[cfg(test)]
mod ffi_smoke_test {
    use super::bindings::{pardiso, pardisoinit};
    use std::ffi::c_void;

    /// Solves a tiny fixed unsymmetric system directly off the raw FFI
    /// bindings, with no `PardisoRealSystem` wrapper involved — proves the
    /// `build.rs` MKL discovery/linking and the `mkl_pardiso.h`-derived
    /// bindgen output are ABI-correct independently of any gridoxide-
    /// specific wrapper logic. Mirrors
    /// `sparse_klu.rs::tests::solve_complex_simple_system`'s role for the
    /// `Klu` backend.
    ///
    /// System: `[[2, 1], [1, 3]] * [x0, x1] = [5, 10]` => `x0=1, x1=3` (the
    /// same fixture `sparse_klu.rs`'s own tests use), in 0-based CSR:
    /// `ia = [0, 2, 4]`, `ja = [0, 1, 0, 1]`, `a = [2, 1, 1, 3]`.
    #[test]
    fn solves_simple_system_via_raw_ffi() {
        let n: i32 = 2;
        let ia: [i32; 3] = [0, 2, 4];
        let ja: [i32; 4] = [0, 1, 0, 1];
        let a: [f64; 4] = [2.0, 1.0, 1.0, 3.0];
        let mut b: [f64; 2] = [5.0, 10.0];
        let mut x: [f64; 2] = [0.0, 0.0];

        let mut pt: [*mut c_void; 64] = [std::ptr::null_mut(); 64];
        let mtype: i32 = 11; // real, nonsymmetric
        let mut iparm: [i32; 64] = [0; 64];

        unsafe { pardisoinit(pt.as_mut_ptr() as *mut c_void, &mtype, iparm.as_mut_ptr()) };
        iparm[0] = 1; // use our own iparm, not silent defaults
        iparm[34] = 1; // 0-based (C-style) ia/ja indexing

        let maxfct: i32 = 1;
        let mnum: i32 = 1;
        let nrhs: i32 = 1;
        let msglvl: i32 = 0;
        let mut error: i32 = 0;
        let perm: *mut i32 = std::ptr::null_mut();

        for phase in [11, 22, 33] {
            unsafe {
                pardiso(
                    pt.as_mut_ptr() as *mut c_void,
                    &maxfct,
                    &mnum,
                    &mtype,
                    &phase,
                    &n,
                    a.as_ptr() as *const c_void,
                    ia.as_ptr(),
                    ja.as_ptr(),
                    perm,
                    &nrhs,
                    iparm.as_mut_ptr(),
                    &msglvl,
                    b.as_mut_ptr() as *mut c_void,
                    x.as_mut_ptr() as *mut c_void,
                    &mut error,
                );
            }
            assert_eq!(error, 0, "PARDISO phase {phase} failed with error {error}");
        }

        // Release pt's internal memory.
        let release_phase: i32 = -1;
        unsafe {
            pardiso(
                pt.as_mut_ptr() as *mut c_void,
                &maxfct,
                &mnum,
                &mtype,
                &release_phase,
                &n,
                std::ptr::null(),
                ia.as_ptr(),
                ja.as_ptr(),
                perm,
                &nrhs,
                iparm.as_mut_ptr(),
                &msglvl,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut error,
            );
        }
        assert_eq!(error, 0, "PARDISO release phase failed with error {error}");

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_simple_system() {
        // [[2, 1], [1, 3]] * [x0, x1] = [5, 10]  =>  x0=1, x1=3
        let entries = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let mut sys = PardisoRealSystem::new(2, &entries).unwrap();
        let x = sys.factor_and_solve(&entries, &[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn duplicate_entries_sum() {
        // (0,0) contributed twice as 1.0+1.0=2.0, matching dense += semantics.
        let entries = vec![
            (0, 0, 1.0),
            (0, 0, 1.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
        ];
        let mut sys = PardisoRealSystem::new(2, &entries).unwrap();
        let x = sys.factor_and_solve(&entries, &[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn refactor_reuses_symbolic() {
        let entries_a = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let mut sys = PardisoRealSystem::new(2, &entries_a).unwrap();
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
        // Exactly row-proportional (row 1 = 2 * row 0), so genuinely
        // singular, not just ill-conditioned. Confirmed by direct testing:
        // PARDISO's `error` output alone does NOT flag this (its default
        // nonsymmetric matching/scaling silently perturbs through it,
        // `error == 0`) — `IPARM_NUM_PERTURBED_PIVOTS` is what actually
        // catches it, see that constant's doc comment.
        let entries = vec![(0, 0, 1.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 2.0)];
        let sys = PardisoRealSystem::new(2, &entries);
        if let Some(mut sys) = sys {
            assert!(sys.factor_and_solve(&entries, &[1.0, 2.0]).is_none());
        }
    }
}
