//! A from-scratch Rust port of the vendored SuiteSparse KLU solver
//! (`vendor/suitesparse/{BTF,AMD,KLU}/`, v7.12.2 — see
//! `vendor/suitesparse/PROVENANCE.md` for the upstream tag/commit this was
//! ported from) — a fourth, always-built `JacobianBackend` alongside
//! `Scalar`, `Block`, and the FFI-backed `Klu` (`sparse_klu.rs`).
//!
//! **Why this exists, separately from `sparse_klu.rs`**: `sparse_klu.rs`
//! links the vendored C directly via FFI, which needs a C compiler and
//! `libclang` (for `bindgen`) at build time and is opt-in behind the `klu`
//! Cargo feature specifically to isolate that burden and the LGPL-2.1-or-
//! later license `KLU`/`BTF` carry (see `vendor/suitesparse/PROVENANCE.md`).
//! This module is a faithful *translation* of the same algorithm into Rust
//! instead — no C toolchain needed to build it — but a close translation of
//! LGPL C is still reasonably a derivative work regardless of language, so
//! this module (and everything under it) is itself LGPL-2.1-or-later,
//! **always built** (no feature gate) per a deliberate, confirmed choice —
//! see `Cargo.toml`'s `license` field and the README's licensing section for
//! the crate-wide consequence of that choice.
//!
//! **Scope**: real (`f64`) matrices only, single right-hand side, `int32`-
//! range indices — gridoxide's own Newton-Raphson Jacobian solve never needs
//! anything else (confirmed against every call site in `sparse_klu.rs`).
//! Faithfully ports what real KLU actually does for gridoxide's one fixed
//! configuration (`Options::default()`, mirroring `klu_defaults.c` exactly):
//! BTF block-triangular preprocessing, per-block AMD ordering, a
//! partial-pivoting Gilbert-Peierls left-looking LU kernel with
//! Eisenstat-Liu symmetric pruning, and cheap numeric-only refactorization.
//! Explicitly out of scope (confirmed dead — no gridoxide call path reaches
//! them): COLAMD/user-supplied ordering (`Options`/`klu_common`'s `ordering`
//! is always AMD here), multi-RHS solve, complex arithmetic, and int64
//! (`DLONG`) indices.
//!
//! **Row scaling (`scale.rs`) is ported but not yet wired into
//! `KluNativeSystem` below** — a real, deliberate gap, not a silent one:
//! `scale::scale` is fully implemented and differentially tested against
//! real `klu_scale` in isolation, but `factor`/`refactor` here always run as
//! if `Options.scale` were disabled (`Rs = None` throughout). Row scaling
//! is a numerical-*stability* preconditioning step (which candidate pivots
//! partial pivoting compares), not a correctness requirement — an unscaled
//! factorization of a well-conditioned matrix still produces the exact
//! solution, just with potentially different (still valid) pivot choices
//! than real KLU's default-scaled path. gridoxide's own per-unit power-flow
//! Jacobian is not pathologically ill-scaled, and the differential tests in
//! `solve.rs`/`factor.rs`/`refactor.rs` (unscaled Rust vs. scaled real KLU,
//! same matrices) already confirm the final solved `x` still matches to
//! 1e-8. Wiring `Rs` all the way through `KluNativeSystem` remains future
//! work if a pathological topology ever needs it.
//!
//! **Module layout** (mirrors the phased port plan; each submodule's doc
//! comment names the specific upstream `.c` file(s) it was translated from):
//! `types` (shared sentinel/`Options` plumbing) → `btf` → `amd` → `kernel`/
//! `factor` → `scale` → `refactor` → `solve`, with the public API
//! (`KluNativeSystem`) assembled here.

pub mod types;

mod btf;
mod amd;
mod analyze;
mod kernel;
mod factor;
mod scale;
mod refactor;
mod solve;

#[cfg(all(test, feature = "klu"))]
mod ffi_oracle;

/// Builds a KLU-ready CSC structure (column pointers + sorted row indices)
/// from a set of `(row, col)` index pairs, merging duplicates — the `i64`
/// twin of `sparse_klu.rs::build_csc_structure` (that function's own `i32`
/// arrays are FFI ABI, not reusable here: this module must build without
/// `sparse_klu`/the `klu` feature ever being enabled, per this file's own
/// "always built" licensing posture — see the module doc comment above).
/// Returns `(col_ptr, row_idx, groups)`, where `groups[k]` lists the
/// original `entries` indices contributing to the `k`-th CSC position.
fn build_csc_structure(n: usize, pairs: &[(usize, usize)]) -> (Vec<i64>, Vec<i64>, Vec<Vec<usize>>) {
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&i| (pairs[i].1, pairs[i].0));

    let mut col_ptr = vec![0i64; n + 1];
    let mut row_idx: Vec<i64> = Vec::new();
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
        row_idx.push(row as i64);
        groups.push(group);
        col_ptr[col + 1] += 1;
        idx = k;
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    (col_ptr, row_idx, groups)
}

fn pack_values(entries: &[(usize, usize, f64)], groups: &[Vec<usize>]) -> Vec<f64> {
    groups.iter().map(|g| g.iter().map(|&i| entries[i].2).sum()).collect()
}

/// A real sparse system whose sparsity *pattern* is fixed across repeated
/// solves — the pure-Rust twin of `sparse_klu::KluRealSystem`, same public
/// shape (`new`/`factor_and_solve`) so `solver.rs` can dispatch to either
/// backend mechanically. Caches the symbolic analysis (`analyze::analyze`)
/// and numeric factorization (`factor::Numeric`) across calls, using
/// `refactor::refactor` for cheap numeric-only re-factorization on repeat
/// calls — mirrors `KluRealSystem`'s own doc comment and the same pattern
/// validated for `RealSparseSystem`/`BlockLu::refactor` elsewhere in this
/// crate.
pub struct KluNativeSystem {
    n: usize,
    col_ptr: Vec<i64>,
    row_idx: Vec<i64>,
    groups: Vec<Vec<usize>>,
    sym: analyze::Symbolic,
    num: factor::Numeric,
}

impl KluNativeSystem {
    /// Builds the symbolic factorization from an initial sparsity pattern
    /// and factors it once. Subsequent calls to `factor_and_solve` must
    /// supply entries with the exact same `(row, col)` pairs in the exact
    /// same order (values may differ) — same precondition as
    /// `KluRealSystem::new`, for the same reason (the cached CSC position
    /// mapping in `groups` assumes positional correspondence, not just set
    /// equality). Returns `None` if the matrix is structurally or
    /// numerically singular.
    pub fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        let pairs: Vec<(usize, usize)> = entries.iter().map(|&(r, c, _)| (r, c)).collect();
        let (col_ptr, row_idx, groups) = build_csc_structure(n, &pairs);
        let values = pack_values(entries, &groups);

        let sym = analyze::analyze(n, &col_ptr, &row_idx);
        let num = factor::factor(n, &col_ptr, &row_idx, &values, &sym, types::Options::default().tol)?;

        Some(Self { n, col_ptr, row_idx, groups, sym, num })
    }

    /// Numeric-only refactorization against the cached symbolic pattern,
    /// then solves `A x = b`. Returns `None` if the matrix is singular (or
    /// the solve produces a non-finite result, matching `KluRealSystem`'s
    /// own finite-check — a pure-Rust factorization has no
    /// `common.status`/`KLU_SINGULAR` to check directly, so a non-finite
    /// entry is this port's only symptom of an ill-conditioned refactor
    /// that a fixed pattern still let through).
    pub fn factor_and_solve(&mut self, entries: &[(usize, usize, f64)], rhs: &[f64]) -> Option<Vec<f64>> {
        let values = pack_values(entries, &self.groups);
        let num = refactor::refactor(self.n, &self.col_ptr, &self.row_idx, &values, &self.sym, &self.num)?;
        self.num = num;

        let x = solve::solve(&self.sym, &self.num, None, rhs);
        if x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_sparse_system_refactor_reuses_symbolic() {
        // Mirrors sparse_klu.rs's own test of the same name -- same
        // fixture, confirming KluNativeSystem's public contract matches
        // KluRealSystem's exactly.
        let entries_a = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let mut sys = KluNativeSystem::new(2, &entries_a).unwrap();
        let x = sys.factor_and_solve(&entries_a, &[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);

        let entries_b = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)];
        // [[4,1],[1,2]] * x = [6, 5] => x0=1, x1=2
        let x2 = sys.factor_and_solve(&entries_b, &[6.0, 5.0]).unwrap();
        assert!((x2[0] - 1.0).abs() < 1e-10);
        assert!((x2[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn singular_matrix_returns_none() {
        let entries = vec![(0, 0, 1.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 2.0)];
        assert!(KluNativeSystem::new(2, &entries).is_none());
    }

    #[cfg(feature = "klu")]
    #[test]
    fn matches_real_klu_end_to_end() {
        let mut seed: u64 = 0x9F1D4A2E7C6B0835;
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

            let mut native = KluNativeSystem::new(n, &entries)
                .unwrap_or_else(|| panic!("trial {trial} (n={n}): unexpectedly singular"));
            let mut real = crate::sparse_klu::KluRealSystem::new(n, &entries).unwrap();

            for pass in 0..3 {
                let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.3 + pass as f64).collect();
                let native_x = native.factor_and_solve(&entries, &b).unwrap();
                let real_x = real.factor_and_solve(&entries, &b).unwrap();
                for i in 0..n {
                    assert!(
                        (native_x[i] - real_x[i]).abs() < 1e-8,
                        "trial {trial} pass {pass} (n={n}), index {i}: native={} real={}",
                        native_x[i],
                        real_x[i]
                    );
                }
            }
        }
    }
}
