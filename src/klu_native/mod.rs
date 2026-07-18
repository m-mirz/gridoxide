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
//! Eisenstat-Liu symmetric pruning, row scaling, and cheap numeric-only
//! refactorization. Explicitly out of scope (confirmed dead — no gridoxide
//! call path reaches them): COLAMD/user-supplied ordering
//! (`Options`/`klu_common`'s `ordering` is always AMD here), multi-RHS solve,
//! complex arithmetic, and int64 (`DLONG`) indices.
//!
//! **Module layout** (mirrors the phased port plan; each submodule's doc
//! comment names the specific upstream `.c` file(s) it was translated from):
//! `types` (shared sentinel/`Options` plumbing) → `btf` → `amd` → `kernel`/
//! `factor` → `scale` → `refactor` → `solve`, with the public API assembled
//! here.

// TODO(Phase 7): remove once this module is wired into `solver::
// JacobianBackend` — until then, nothing outside this module's own tests
// calls into it, so cargo's dead-code analysis (correctly) flags the whole
// tree as unreachable from any public entry point.
#![allow(dead_code)]

pub mod types;

mod btf;
mod amd;
mod analyze;
mod kernel;

#[cfg(all(test, feature = "klu"))]
mod ffi_oracle;
