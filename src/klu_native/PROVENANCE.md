# `klu_native`: provenance and licensing

`src/klu_native/` is a from-scratch Rust *translation* of the vendored SuiteSparse `AMD`, `BTF`, and
`KLU` C source under `vendor/suitesparse/` (see `vendor/suitesparse/PROVENANCE.md` for the upstream
tag/commit this was ported from — **`v7.12.2`**, commit `42151688813c45846a597edcb601435a0e38f3dd`,
2026-02-10). Unlike `src/sparse_klu.rs` (which links the vendored C directly via FFI, opt-in behind
the `klu` Cargo feature), this module needs no C toolchain and is **always built**.

A close translation of licensed C source is reasonably a derivative work regardless of implementation
language, so each Rust file below carries forward the license of the specific upstream package it was
translated from — **not uniformly one license**, since (as the table below shows) `AMD` itself is
BSD-3-Clause upstream, while `BTF` and `KLU` are LGPL-2.1-or-later. See "Crate-wide consequence" below
for what this means for a default `cargo build` of gridoxide as a whole.

## File-by-file mapping

| Rust file | Ported from | Upstream license |
|---|---|---|
| `types.rs` | `KLU/Include/klu_internal.h`, `AMD/Include/amd_internal.h`, `BTF/Include/btf.h` (`EMPTY`/`FLIP`/`UNFLIP` sentinel macros — identical across all three headers) and `KLU/Source/klu_defaults.c` (fixed `Options`) | LGPL-2.1-or-later¹ |
| `btf/maxtrans.rs` | `BTF/Source/btf_maxtrans.c` | LGPL-2.1-or-later |
| `btf/strongcomp.rs` | `BTF/Source/btf_strongcomp.c` | LGPL-2.1-or-later |
| `btf/mod.rs` | `BTF/Source/btf_order.c` | LGPL-2.1-or-later |
| `amd/aat.rs` | `AMD/Source/amd_aat.c` (+ `amd_preprocess.c`) | BSD-3-Clause |
| `amd/core.rs` | `AMD/Source/amd_2.c` | BSD-3-Clause |
| `amd/postorder.rs` | `AMD/Source/amd_postorder.c` + `amd_post_tree.c` | BSD-3-Clause |
| `amd/mod.rs` | `AMD/Source/amd_order.c` (+ `amd_valid.c`'s validation gate) | BSD-3-Clause |
| `analyze.rs` | `KLU/Source/klu_analyze.c` (`order_and_analyze`/`analyze_worker`) | LGPL-2.1-or-later |
| `kernel.rs` | `KLU/Source/klu_kernel.c` (`dfs`/`lsolve_symbolic`/`construct_column`/`lsolve_numeric`/`lpivot`/`prune`/`KLU_kernel`) + `klu_refactor.c`'s per-block numeric loop (`refactor_block`, kept in this file since it shares `construct_column` with `factor_block`) | LGPL-2.1-or-later |
| `factor.rs` | `KLU/Source/klu_factor.c` (`KLU_factor`/`factor2`) | LGPL-2.1-or-later |
| `scale.rs` | `KLU/Source/klu_scale.c` | LGPL-2.1-or-later |
| `refactor.rs` | `KLU/Source/klu_refactor.c` (multi-block/BTF orchestration; the per-block numeric update itself is `kernel::refactor_block`) | LGPL-2.1-or-later |
| `solve.rs` | `KLU/Source/klu_solve.c` + `klu.c` (`KLU_lsolve`/`KLU_usolve`, `nrhs == 1` only — see `solve.rs`'s own doc comment) | LGPL-2.1-or-later |
| `mod.rs` (`KluNativeSystem`, `build_csc_structure`, `pack_values`) | Not a direct translation of one upstream file — an original integration mirroring `sparse_klu::KluRealSystem`'s public shape (`new`/`factor_and_solve`) and CSC-construction convention, composed from the ported pieces above | N/A (original) |
| `ffi_oracle.rs` | Test-only `extern "C"` bindings to the real vendored functions, used as differential-testing oracles (`#[cfg(all(test, feature = "klu"))]`) — not shipped in any production build | N/A (test-only, not distributed) |

¹ `types.rs`'s sentinel encoding (`FLIP(i) = -(i)-2`) is byte-identical across `klu_internal.h`,
`amd_internal.h`, and `btf.h` — it draws from all three headers equally, two of which (`KLU`, `BTF`)
are LGPL. Classified as LGPL-2.1-or-later here as the more restrictive of the two licenses actually in
play for this specific file, consistent with how the crate-wide `license` field below is derived.

## Explicitly out of scope (not ported)

Confirmed dead for gridoxide's own fixed usage — see each submodule's own doc comment for the specific
reasoning: `amd_control`/`amd_defaults`/`amd_dump`/`amd_info`/`amd_version`, `klu_analyze_given.c`,
`klu_diagnostics.c`/`klu_dump.c`/`klu_extract.c`/`klu_sort.c`, `klu_tsolve.c` and `KLU_ltsolve`/
`KLU_utsolve` (transpose solve), the `nrhs` 2-4 cases throughout `klu_solve.c`/`klu.c`, and
`maxwork`/work-limit bookkeeping in `btf_maxtrans.c`. Also out of scope: `COLAMD` (never selected —
gridoxide's fixed `Options` always uses AMD ordering), complex (`klu_z_*`) and `int64` (`DLONG`)
variants.

## Crate-wide consequence

Before this module existed, LGPL-2.1-or-later code only entered a gridoxide build via the opt-in `klu`
Cargo feature (`sparse_klu.rs`, FFI to the vendored C) — a plain `cargo build` produced an
Apache-2.0-only artifact. Since `klu_native` is always built, **every default `cargo build` now bundles
BSD-3-Clause (`amd/`) and LGPL-2.1-or-later (everything else in this module) code alongside gridoxide's
own Apache-2.0 code** — see `Cargo.toml`'s `license` field and the README's licensing section for how
that's reflected. This was a deliberate, confirmed choice (not a default the port drifted into) — see
the git history for the `klu_native` port's planning discussion.
