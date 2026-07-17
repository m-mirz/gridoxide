# Vendored SuiteSparse source

This directory contains a partial vendoring of [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse),
used to build the optional `klu` backend (`solver::JacobianBackend::Klu`) — see the "Sparse solver" section of
the top-level README for what that backend is and why it exists.

- **Upstream**: https://github.com/DrTimothyAldenDavis/SuiteSparse
- **Tag**: `v7.12.2`
- **Commit**: `42151688813c45846a597edcb601435a0e38f3dd` (2026-02-10)
- **Vendored**: only the `Source/` and `Include/` subdirectories (plus each package's `Doc/License.txt` and
  `README.txt`) of five packages — `SuiteSparse_config`, `AMD`, `COLAMD`, `BTF`, `KLU` — not the full
  SuiteSparse tree (which also includes CHOLMOD, UMFPACK, GraphBLAS, and many other packages gridoxide doesn't
  use). `Demo/`, `MATLAB/`, `Tcov/`, `build/`, and build-system files (`CMakeLists.txt`, `Makefile`) were
  intentionally not copied — this vendoring only needs to feed `build.rs`'s own `cc::Build` compilation, not
  SuiteSparse's own build system.

## Licensing

Confirmed directly from each package's own `Doc/License.txt` (kept alongside the source in this vendoring, not
stripped) and from `SPDX-License-Identifier` headers in the source files themselves:

| Package | License |
|---|---|
| `AMD` | BSD-3-Clause |
| `COLAMD` | BSD-3-Clause |
| `BTF` | LGPL-2.1-or-later |
| `KLU` | LGPL-2.1-or-later |
| `SuiteSparse_config` | BSD-3-Clause |

`KLU` and `BTF` (one of KLU's own dependencies) being LGPL is why the `klu` Cargo feature that compiles this
vendored source is opt-in (off by default) rather than always built — see the README for the full explanation
and the `klu-dynamic` feature offered as an alternative for anyone who needs strict LGPL relinking compliance
(link against a system-installed `libklu.so` instead of statically linking this vendored copy).

## Updating

To update to a newer SuiteSparse release, re-run the same partial-checkout process against the new tag:

```bash
git clone --filter=blob:none --sparse --depth=1 --branch <new-tag> \
    https://github.com/DrTimothyAldenDavis/SuiteSparse.git /tmp/suitesparse-checkout
cd /tmp/suitesparse-checkout
git sparse-checkout set SuiteSparse_config AMD COLAMD BTF KLU
```

then copy each package's `Source/`, `Include/`, `Doc/License.txt`, and `README.txt` over the corresponding
directory here, and update the tag/commit/date above. Re-run `cargo build --features klu` afterwards and watch
for new undefined-symbol linker errors — SuiteSparse occasionally adds or renames source files between
releases, so `build.rs`'s file list may need adjusting.
