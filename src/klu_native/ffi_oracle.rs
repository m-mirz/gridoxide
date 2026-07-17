//! Test-only FFI bindings to the vendored `btf_order`/`amd_order` C
//! functions (statically linked into `libklu_vendored` whenever the `klu`
//! Cargo feature is enabled — see `build.rs`), used purely as ground-truth
//! oracles in later phases' differential tests comparing this port's BTF/AMD
//! output against the real vendored implementations. Never compiled into a
//! production binary — this whole module is `#[cfg(all(test, feature =
//! "klu"))]`-gated at the `mod ffi_oracle;` declaration in `mod.rs`, and
//! exists only to validate the port, not to reimplement `sparse_klu.rs`'s
//! own FFI wrapper (which only binds `klu_*`-prefixed symbols).
//!
//! `build.rs`'s `bindgen` invocation allowlists only `klu_.*` functions/types
//! (`sparse_klu.rs`'s needs), so `btf_order`/`amd_order` — both real/int32,
//! non-`klu_`-prefixed — aren't in its generated bindings. Hand-declaring
//! their exact signatures here (confirmed directly against
//! `vendor/suitesparse/BTF/Include/btf.h` and
//! `vendor/suitesparse/AMD/Include/amd.h`) is simpler than expanding
//! `build.rs`'s production bindgen surface for a test-only need — the
//! objects are already compiled and statically linked by the existing `klu`
//! feature, so a plain `extern "C"` declaration is all that's needed to
//! resolve these symbols at link time.
//!
//! `#[allow(dead_code)]` for now: nothing calls these yet (Phase 0 only
//! scaffolds the oracle; Phase 1/2's differential tests are what actually
//! use them) — remove once those phases land.

use std::os::raw::c_int;

#[allow(dead_code)]
unsafe extern "C" {
    /// `vendor/suitesparse/BTF/Include/btf.h`'s `btf_order` (real/int32
    /// variant, not `btf_l_order`).
    fn btf_order(
        n: i32,
        ap: *const i32,
        ai: *const i32,
        maxwork: f64,
        work: *mut f64,
        p: *mut i32,
        q: *mut i32,
        r: *mut i32,
        nmatch: *mut i32,
        work_arr: *mut i32,
    ) -> i32;

    /// `vendor/suitesparse/AMD/Include/amd.h`'s `amd_order` (real/int32
    /// variant, not `amd_l_order`).
    fn amd_order(
        n: i32,
        ap: *const i32,
        ai: *const i32,
        p: *mut i32,
        control: *const f64,
        info: *mut f64,
    ) -> c_int;
}

/// Safe wrapper around the vendored `btf_order`. `n`/`col_ptr`/`row_idx`
/// describe a square `n`-by-`n` CSC matrix (`col_ptr.len() == n + 1`,
/// `row_idx.len() == col_ptr[n]`). No work limit (`maxwork = 0.0`, matching
/// `Options::default()`/`klu_defaults.c`'s own default — see
/// `types::Options`'s doc comment for why this port drops the limit-checking
/// machinery outright rather than merely defaulting it off).
///
/// Returns `(p, q, r, nmatch)` exactly as `btf.h` documents: `p`/`q` are the
/// row/column permutation (`q[k]` may be "flipped" — see `types::unflip` —
/// for an unmatched column when the matrix is structurally singular), `r`
/// gives block boundaries **truncated to `nblocks + 1` entries** (the C
/// function's own return value is `nblocks`; `r.len() == n + 1` is only the
/// oversized buffer it's given to write into — `r[nblocks+1..]` is left
/// undefined by `btf_order` itself, so this wrapper truncates rather than
/// returning that undefined tail), and `nmatch` is the number of nonzeros on
/// the diagonal of the permuted matrix (a separate output parameter, not the
/// same as `nblocks`).
#[allow(dead_code)]
pub fn btf_order_oracle(n: usize, col_ptr: &[i32], row_idx: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>, i32) {
    assert_eq!(col_ptr.len(), n + 1, "col_ptr must have n+1 entries");
    let mut p = vec![0i32; n];
    let mut q = vec![0i32; n];
    let mut r = vec![0i32; n + 1];
    let mut nmatch = 0i32;
    let mut work = 0.0f64;
    let mut work_arr = vec![0i32; 5 * n];
    let nblocks = unsafe {
        btf_order(
            n as i32,
            col_ptr.as_ptr(),
            row_idx.as_ptr(),
            0.0,
            &mut work,
            p.as_mut_ptr(),
            q.as_mut_ptr(),
            r.as_mut_ptr(),
            &mut nmatch,
            work_arr.as_mut_ptr(),
        )
    };
    r.truncate(nblocks as usize + 1);
    (p, q, r, nmatch)
}

/// Safe wrapper around the vendored `amd_order`, using default `Control`
/// (`NULL` — `amd.h`: "Defaults are used if Control is NULL") and discarding
/// `Info`. Returns the permutation `P`, or `None` if AMD reported anything
/// other than `AMD_OK` (0) or `AMD_OK_BUT_JUMBLED` (1) — both successes per
/// `amd.h`; any negative status is a real failure.
#[allow(dead_code)]
pub fn amd_order_oracle(n: usize, col_ptr: &[i32], row_idx: &[i32]) -> Option<Vec<i32>> {
    assert_eq!(col_ptr.len(), n + 1, "col_ptr must have n+1 entries");
    let mut p = vec![0i32; n];
    let status = unsafe {
        amd_order(n as i32, col_ptr.as_ptr(), row_idx.as_ptr(), p.as_mut_ptr(), std::ptr::null(), std::ptr::null_mut())
    };
    (status >= 0).then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn btf_order_oracle_on_identity_pattern() {
        // 3x3 diagonal matrix: already in block-triangular form, 3 singleton blocks.
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [0, 1, 2];
        let (p, q, r, nmatch) = btf_order_oracle(3, &col_ptr, &row_idx);
        assert_eq!(nmatch, 3, "diagonal matrix should have a full matching");
        assert_eq!(r, vec![0, 1, 2, 3], "3 singleton blocks");
        let mut sorted_p = p.clone();
        sorted_p.sort_unstable();
        assert_eq!(sorted_p, vec![0, 1, 2], "P must be a bijection");
        let mut sorted_q: Vec<i32> = q.iter().map(|&x| super::super::types::unflip(x as i64) as i32).collect();
        sorted_q.sort_unstable();
        assert_eq!(sorted_q, vec![0, 1, 2], "Q (unflipped) must be a bijection");
    }

    #[test]
    fn amd_order_oracle_on_identity_pattern() {
        let col_ptr = [0, 1, 2, 3];
        let row_idx = [0, 1, 2];
        let p = amd_order_oracle(3, &col_ptr, &row_idx).expect("should succeed");
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2], "P must be a bijection");
    }
}
