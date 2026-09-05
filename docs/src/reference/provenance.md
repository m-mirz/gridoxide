# Provenance and Licensing

gridoxide's own code is licensed under **Apache-2.0** (`LICENSE`). Several pieces of third-party
code are vendored, translated, or linked, and each carries its own terms. This page is the summary;
the authoritative per-file detail lives in the `PROVENANCE.md` files kept alongside the code they
describe.

## The crate-wide license field

`Cargo.toml`'s `license` field is:

```
Apache-2.0 AND BSD-3-Clause AND LGPL-2.1-or-later
```

That is accurate for **every default `cargo build`**, not just an opt-in one — because
`src/klu_native/` is always built, with no feature gate.

## Always built: `src/klu_native/`

`src/klu_native/` is a from-scratch Rust *translation* of vendored SuiteSparse `AMD`, `BTF`, and
`KLU` C source. A close translation of licensed source is reasonably a derivative work regardless of
implementation language, so it carries forward its upstream license — and that is not one license
here:

| Upstream package | License | Translated into |
|---|---|---|
| `AMD` | BSD-3-Clause | `amd/aat.rs`, `amd/core.rs`, `amd/postorder.rs`, `amd/mod.rs` |
| `BTF` | LGPL-2.1-or-later | `btf/maxtrans.rs`, `btf/strongcomp.rs`, `btf/mod.rs` |
| `KLU` | LGPL-2.1-or-later | `analyze.rs`, `kernel.rs`, `factor.rs`, `scale.rs`, `refactor.rs`, `solve.rs` |

`src/klu_native/PROVENANCE.md` has the exact file-by-file mapping back to the upstream C, including
which specific functions each Rust file was ported from and how shared header material was
classified. See [Inside KLU](../solvers/klu.md) for what the ported algorithm actually does.

## Opt-in: `--features klu`

Building with `cargo build --features klu` additionally compiles the vendored SuiteSparse C itself
into the binary via FFI. `vendor/suitesparse/` is a partial vendoring of
[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) at tag **`v7.12.2`** (commit
`42151688813c45846a597edcb601435a0e38f3dd`, 2026-02-10) — only the `Source/` and `Include/`
subdirectories of five packages (`SuiteSparse_config`, `AMD`, `COLAMD`, `BTF`, `KLU`), each keeping
its own `Doc/License.txt`:

| Package | License |
|---|---|
| `AMD` | BSD-3-Clause |
| `COLAMD` | BSD-3-Clause |
| `BTF` | LGPL-2.1-or-later |
| `KLU` | LGPL-2.1-or-later |
| `SuiteSparse_config` | BSD-3-Clause |

This adds no license beyond what is already listed above, but it does add **LGPL's relinking
obligations** for anyone distributing a binary built with that feature. The `klu-dynamic`
sub-feature exists for that case: it links a system-installed `libklu.so` instead of statically
linking the vendored copy. See `vendor/suitesparse/PROVENANCE.md` for exactly what was and was not
vendored, and how to update to a newer SuiteSparse release.

## Opt-in: `--features pardiso`

A separate case from all of the above. It dynamically links a locally-installed Intel oneMKL
(`libmkl_rt.so`) at build and run time, under Intel's own Simplified Software License — not LGPL,
not OSS, and **not vendored or redistributed by this repo in any form**. No MKL header or source is
copied in; `bindgen` only reads the local install's own `mkl_pardiso.h` at build time to generate FFI
bindings.

Because nothing MKL-derived is ever copied into or shipped by this crate, `Cargo.toml`'s `license`
field does **not** change for this feature. Anyone who builds with `--features pardiso` and
distributes the resulting binary is responsible for their own compliance with Intel's oneMKL
redistribution terms — this project doesn't audit that on their behalf.

## Opt-in: `--features cgmes` — the cimoxide dependency

The optional `cgmes` feature (`src/cgmes.rs`) depends on
[cimoxide](https://github.com/m-mirz/cimoxide)'s decoder and generated-structs crates, a separate
Rust project by the same author providing CGMES RDF/XML decoding into typed CIM structs.

- **Upstream**: <https://github.com/m-mirz/cimoxide>
- **Crates**: [`cimoxide-decoder`](https://crates.io/crates/cimoxide-decoder) and
  [`cimoxide-structs`](https://crates.io/crates/cimoxide-structs), both `0.3.0`

cimoxide is Apache-2.0, matching gridoxide's own license — no licensing mismatch from this
dependency.

### The `package =` rename

`Cargo.toml` pulls both in under their old, shorter names:

```toml
cimdecoder = { package = "cimoxide-decoder", version = "0.3.0", optional = true }
cimstructs = { package = "cimoxide-structs", version = "0.3.0", optional = true }
```

The crates.io names are `cimoxide-`-prefixed, but `src/cgmes.rs` and the CGMES tests were written
against `cimdecoder`/`cimstructs`, so `package =` keeps every existing `use` path valid rather than
renaming 33 references for no behavioural gain.

### Formerly a git dependency

Until cimoxide's crates.io release, these were a git dependency pinned to a `vendor/gridoxide`
branch — `main` plus one commit force-adding the code generator's normally-gitignored output, since
`cimstructs`'s source is produced by `cargo run -p cimgen` from ENTSO-E's RDF/SHACL schemas and so
does not exist on a fresh `main` clone.

### Updating

To pick up a newer cimoxide schema or generator change:

1. Release the new version from the cimoxide repo (`make generate`, then `make build && make test`
   before publishing — `cimstructs`'s generated source is what a schema change moves).
2. Bump the `version =` value in this repo's `Cargo.toml` for the `cimdecoder` and `cimstructs`
   dependencies, and update the version noted above.
3. Run `cargo build --features cgmes` and `cargo test --features cgmes` to confirm the converter
   still matches `tests/cgmes_microgrid_be_test.rs`'s expectations — CGMES field names or shapes
   could in principle change between schema versions. The fixture-backed CGMES tests **silently
   skip** when `tests/data/CGMES-Test-Configurations` isn't checked out, so
   `git submodule update --init tests/data/CGMES-Test-Configurations` first, or the run proves
   nothing beyond "it compiles".

## Test fixtures

The CGMES conformance fixtures under `tests/data/cgmes/` are referenced via a git submodule rather
than committed, because of their own licensing — see `tests/data/cgmes/README.md`.
