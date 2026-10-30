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
//! instead — no C toolchain needed to build it.
//!
//! **Licensing**: a close translation of licensed C source is reasonably a
//! derivative work regardless of language, so this module carries forward
//! its upstream licenses — **not uniformly one license**: `amd/` translates
//! `AMD`, which is BSD-3-Clause upstream (confirmed directly from its
//! source headers' `SPDX-License-Identifier`, same as
//! `vendor/suitesparse/PROVENANCE.md`'s own table), while everything else
//! here (`btf/`, `analyze.rs`, `kernel.rs`, `factor.rs`, `scale.rs`,
//! `refactor.rs`, `solve.rs`) translates `BTF`/`KLU`, both
//! LGPL-2.1-or-later. See `PROVENANCE.md` (this directory) for the full
//! per-file mapping. This whole module is **always built** (no feature
//! gate), per a deliberate, confirmed choice — unlike the FFI `klu` feature,
//! there's no C-toolchain burden left to gate against — so a default
//! `cargo build` now always bundles this BSD/LGPL content alongside
//! gridoxide's own Apache-2.0 code. See `Cargo.toml`'s `license` field and
//! `docs/src/reference/provenance.md` for that crate-wide consequence.
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
//! **Rows are scaled by default**, as SuiteSparse's KLU scales them
//! (`KluNativeSystem::new`; `with_scale(.., 0)` turns it off). Scaling divides
//! every row by its largest magnitude before each factorization; it decides
//! which pivots partial pivoting sees as large and how the arithmetic rounds,
//! never what the solution is. Scaled, the port solves as SuiteSparse's KLU
//! does — the same pivots and, on most matrices, the same bits
//! (`tests::scaled_matches_real_klu`).
//!
//! Power-flow Jacobians do not need it: `examples/klu_scaling_stress.rs`
//! runs the worst-scaled networks at hand, with rows spanning up to 24
//! orders of magnitude, and unscaled the backward error stays at machine
//! precision and every Newton solve takes the iterations SuiteSparse's KLU
//! takes. State estimation does: its system weights every measurement by
//! 1/σ², and on the three-phase transmission case with one sensor fifteen σ
//! off (`se_three_phase_test::bad_data_detection_finds_the_corrupted_measurement`)
//! the unscaled port's steps stall at 6e-8 and Gauss-Newton runs out of
//! iterations, where scaled — like SuiteSparse's KLU and faer — it converges
//! in 8. Scaling costs 11–15% of a power-flow solve; a caller that knows its
//! matrices are well scaled can switch it off.
//!
//! **Module layout** (mirrors the phased port plan; each submodule's doc
//! comment names the specific upstream `.c` file(s) it was translated from):
//! `types` (shared sentinel/`Options` plumbing) → `btf` → `amd` → `kernel`/
//! `factor` → `scale` → `refactor` → `solve`, with the public API
//! (`KluNativeSystem`) assembled here.

pub mod types;

mod amd;
mod analyze;
mod btf;
mod factor;
mod kernel;
mod refactor;
mod scale;
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
fn build_csc_structure(
    n: usize,
    pairs: &[(usize, usize)],
) -> (Vec<i64>, Vec<i64>, Vec<Vec<usize>>) {
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
    groups
        .iter()
        .map(|g| g.iter().map(|&i| entries[i].2).sum())
        .collect()
}

/// `pack_values` for callers that already hold just the values, positionally
/// matching the triplets `new` was given — see
/// `sparse::LinearSolver::factor_and_solve_values`.
fn pack_values_slice(values: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    groups
        .iter()
        .map(|g| g.iter().map(|&i| values[i]).sum())
        .collect()
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
    /// Set by [`adopt_storage_order`](Self::adopt_storage_order): values
    /// arrive in CSC order and are refactored as they are.
    in_order: bool,
    sym: analyze::Symbolic,
    num: factor::Numeric,
    /// KLU's `Common->scale`: 2 divides every row by its largest magnitude
    /// before each factorization (KLU's default), 1 by its sum, 0 not at all.
    scale: i32,
    /// The row scale factors of the last factorization, in pivot order
    /// (`rs[k]` for row `pnum[k]`), as `klu_solve.c` reads `Numeric->Rs`;
    /// empty when not scaling.
    rs: Vec<f64>,
    /// The row-indexed scale factors of the last call, kept so the next
    /// reuses the allocation.
    raw: Vec<f64>,
    /// Reused across every `factor_and_solve` call (see
    /// `refactor::RefactorScratch`'s own doc comment) — keeping this alive
    /// for the system's whole lifetime, instead of creating one per call,
    /// is most of what makes repeat solves allocation-light.
    scratch: refactor::RefactorScratch,
}

impl KluNativeSystem {
    /// Builds the symbolic factorization from an initial sparsity pattern
    /// and factors it once. Subsequent calls to `factor_and_solve` must
    /// supply entries with the exact same `(row, col)` pairs in the exact
    /// same order (values may differ) — same precondition as
    /// `KluRealSystem::new`, for the same reason (the cached CSC position
    /// mapping in `groups` assumes positional correspondence, not just set
    /// equality). Scales rows as SuiteSparse's KLU does by default (see the
    /// module documentation). Returns `None` if the matrix is structurally or
    /// numerically singular.
    pub fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        Self::with_scale(n, entries, types::Options::default().scale)
    }

    /// The last factorization's size and work: the nonzeros of L and U
    /// (the diagonal once, BTF's off-diagonal blocks included) and its
    /// floating-point operations, counted as KLU's `klu_flops` does: per
    /// pivot column k of a block, nnz(L_k) divisions and two operations
    /// for each entry of L_k times each entry of row k of U.
    pub fn factor_stats(&self) -> (usize, f64) {
        let mut nnz = self.num.off_i.len();
        let mut flops = 0.0;
        for block in &self.num.blocks {
            let cols = block.l.len();
            nnz += cols;
            let mut u_row = vec![0usize; cols];
            for k in 0..cols {
                let (rows, _) = block.u.col(k);
                nnz += rows.len();
                for &r in rows {
                    u_row[r as usize] += 1;
                }
            }
            for (k, &urow) in u_row.iter().enumerate() {
                let lk = block.l.col(k).0.len();
                nnz += lk;
                flops += (lk * (1 + 2 * urow)) as f64;
            }
        }
        (nnz, flops)
    }

    /// [`new`](Self::new) with KLU's row scaling: `scale` is
    /// `Common->scale` (2 divides each row by its largest magnitude, KLU's
    /// default; 1 by its sum; 0 does not scale). As in `klu_factor.c` and
    /// `klu_refactor.c`, the factors are recomputed from every new set of
    /// values, each value is divided by its row's, and the solve divides the
    /// right-hand side by the same, so the solution is unchanged: scaling
    /// only decides which pivots partial pivoting sees as large, and how the
    /// arithmetic rounds.
    pub fn with_scale(n: usize, entries: &[(usize, usize, f64)], scale: i32) -> Option<Self> {
        let pairs: Vec<(usize, usize)> = entries.iter().map(|&(r, c, _)| (r, c)).collect();
        let (col_ptr, row_idx, groups) = build_csc_structure(n, &pairs);
        let values = pack_values(entries, &groups);

        let sym = analyze::analyze(n, &col_ptr, &row_idx);
        let mut system = Self {
            n,
            col_ptr,
            row_idx,
            groups,
            in_order: false,
            sym,
            num: factor::Numeric::default(),
            scale,
            rs: Vec::new(),
            raw: Vec::new(),
            scratch: refactor::RefactorScratch::new(n),
        };
        // The first factorization validates the pattern as `klu_scale`
        // does; every refactorization after reuses it unchecked.
        let scaled: Vec<f64> = if scale > 0 {
            system.raw = scale::scale(scale, n, &system.col_ptr, &system.row_idx, &values, false)?;
            values
                .iter()
                .zip(&system.row_idx)
                .map(|(&v, &row)| v / system.raw[row as usize])
                .collect()
        } else {
            values
        };
        system.num = factor::factor(
            n,
            &system.col_ptr,
            &system.row_idx,
            &scaled,
            &system.sym,
            types::Options::default().tol,
        )?;
        system.permute_scale();
        Some(system)
    }

    /// Puts the row-indexed factors of the last factorization in pivot
    /// order, `rs[k] = raw[pnum[k]]`, as `klu_factor.c` does at its end.
    fn permute_scale(&mut self) {
        self.rs.clear();
        if self.scale > 0 {
            let raw = &self.raw;
            self.rs
                .extend(self.num.pnum.iter().map(|&row| raw[row as usize]));
        }
    }

    /// Numeric-only refactorization against the cached symbolic pattern,
    /// then solves `A x = b`. Returns `None` if the matrix is singular (or
    /// the solve produces a non-finite result, matching `KluRealSystem`'s
    /// own finite-check — a pure-Rust factorization has no
    /// `common.status`/`KLU_SINGULAR` to check directly, so a non-finite
    /// entry is this port's only symptom of an ill-conditioned refactor
    /// that a fixed pattern still let through).
    pub fn factor_and_solve(
        &mut self,
        entries: &[(usize, usize, f64)],
        rhs: &[f64],
    ) -> Option<Vec<f64>> {
        let values = pack_values(entries, &self.groups);
        self.solve_packed(&values, rhs)
    }

    /// From now on takes values in this system's own storage order and
    /// refactors them without copying: returns `order`, where `order[k]` is
    /// the input index (in the order `new` was given) that position `k`
    /// holds. The caller reorders its values to match, once, and keeps them
    /// so. `None`, changing nothing, if some position sums several inputs.
    pub fn adopt_storage_order(&mut self) -> Option<Vec<usize>> {
        let order: Vec<usize> = self
            .groups
            .iter()
            .map(|g| match g[..] {
                [only] => Some(only),
                _ => None,
            })
            .collect::<Option<_>>()?;
        self.groups = (0..order.len()).map(|k| vec![k]).collect();
        self.in_order = true;
        Some(order)
    }

    /// Shared tail of both `factor_and_solve` paths, taking already-packed
    /// CSC values.
    ///
    /// A refactor keeps the pivot order of the last factorization, and new
    /// values can put an exact zero on one of those pivots — a fault switched
    /// on or off in a dynamic simulation does. Then the matrix is factored
    /// afresh, with new pivots, on the same symbolic analysis, as
    /// lightsim2grid falls back from `klu_refactor` to `klu_factor`; only if
    /// that fails too is the matrix singular.
    fn solve_packed(&mut self, values: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
        if self.scale > 0 {
            scale::row_factors_into(self.scale, self.n, &self.row_idx, values, &mut self.raw);
        }
        let ok = refactor::refactor(
            self.n,
            &self.col_ptr,
            &self.row_idx,
            values,
            &self.sym,
            &mut self.num,
            &mut self.scratch,
            (self.scale > 0).then_some(&self.raw[..]),
        );
        if ok {
            self.permute_scale();
            if let Some(x) = self.solve_factored(rhs) {
                return Some(x);
            }
        }

        let scaled: Vec<f64>;
        let to_factor = if self.scale > 0 {
            scaled = values
                .iter()
                .zip(&self.row_idx)
                .map(|(&v, &row)| v / self.raw[row as usize])
                .collect();
            &scaled
        } else {
            values
        };
        self.num = factor::factor(
            self.n,
            &self.col_ptr,
            &self.row_idx,
            to_factor,
            &self.sym,
            types::Options::default().tol,
        )?;
        // The scatter targets follow the pivot order, which just changed.
        self.scratch.forget_targets();
        self.permute_scale();
        self.solve_factored(rhs)
    }

    /// Solves `A x = rhs` with the factorization as it stands (the last
    /// `new` or `factor_and_solve`), without refactoring: for many right-hand
    /// sides against one matrix. `None` if the result is not finite.
    pub fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        self.solve_factored(rhs)
    }

    /// Solves with the current factorization; `None` if the result is not
    /// finite.
    fn solve_factored(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        let rs = (!self.rs.is_empty()).then_some(&self.rs[..]);
        let x = solve::solve(&self.sym, &self.num, rs, rhs);
        x.iter().all(|v| v.is_finite()).then_some(x)
    }
}

impl crate::sparse::LinearSolver for KluNativeSystem {
    fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        KluNativeSystem::new(n, entries)
    }

    fn factor_and_solve(
        &mut self,
        entries: &[(usize, usize, f64)],
        rhs: &[f64],
    ) -> Option<Vec<f64>> {
        KluNativeSystem::factor_and_solve(self, entries, rhs)
    }

    fn factor_and_solve_values(&mut self, values: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
        if self.in_order {
            return self.solve_packed(values, rhs);
        }
        self.solve_packed(&pack_values_slice(values, &self.groups), rhs)
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

    /// Random matrices whose rows span twelve orders of magnitude and whose
    /// diagonals do not dominate, so partial pivoting has real choices and
    /// row scaling changes them.
    #[cfg(feature = "klu")]
    fn badly_scaled(
        trial: usize,
        next_f64: &mut impl FnMut() -> f64,
    ) -> (usize, Vec<(usize, usize, f64)>) {
        let n = 5 + (trial % 20);
        let row_scale: Vec<f64> = (0..n)
            .map(|_| 10f64.powf(next_f64() * 12.0 - 6.0))
            .collect();
        let mut entries = Vec::new();
        for i in 0..n {
            let mut js: std::collections::BTreeSet<usize> = [i, (i + 1) % n].into();
            for _ in 0..3 {
                js.insert((next_f64() * n as f64) as usize);
            }
            for j in js {
                entries.push((i, j, (next_f64() * 2.0 - 1.0) * row_scale[i]));
            }
        }
        (n, entries)
    }

    /// With KLU's default row scaling the port solves as KLU does, on
    /// matrices where scaling changes the pivots: to rounding (most trials
    /// agree to the last bit; the rest differ in the order a sum is taken).
    #[cfg(feature = "klu")]
    #[test]
    fn scaled_matches_real_klu() {
        let mut seed: u64 = 0x5DEECE66D2F1A3B7;
        let mut next_f64 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut compared = 0;
        for trial in 0..60 {
            let (n, entries) = badly_scaled(trial, &mut next_f64);
            let (Some(mut real), Some(mut native)) = (
                crate::sparse_klu::KluRealSystem::new(n, &entries),
                KluNativeSystem::with_scale(n, &entries, 2),
            ) else {
                continue;
            };
            for pass in 0..2 {
                let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.3 + pass as f64).collect();
                let (Some(real_x), Some(native_x)) = (
                    real.factor_and_solve(&entries, &b),
                    native.factor_and_solve(&entries, &b),
                ) else {
                    continue;
                };
                let size = real_x.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                for (i, (a, b)) in native_x.iter().zip(&real_x).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-12 * size,
                        "trial {trial} pass {pass} (n={n}), index {i}: native={a} real={b}"
                    );
                }
                compared += 1;
            }
        }
        assert!(compared > 60, "only {compared} solves compared");
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
