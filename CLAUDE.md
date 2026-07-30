# CLAUDE.md

Guidance for Claude Code when working in `gridoxide`.

> The `CLAUDE.md` one directory up describes the **cimlibs** monorepo
> (`cimgo`/`cimoxide`) and does not apply here beyond the shared-submodule
> conventions. `gridoxide` is a separate repository that happens to be checked
> out alongside it.

## What this is

A power flow analysis tool in Rust: Newton-Raphson AC power flow over sparse
Y-bus/Jacobian matrices, with PGM-JSON, MATPOWER and (optionally) CGMES input.
Its value proposition is **agreeing with other solvers to 4+ decimal places** —
pandapower, power-grid-model, lightsim2grid, pypowsybl and MATPOWER are all
cross-validated against in `scripts/bench/`. Any change that trades accuracy for
speed needs that stated explicitly, not assumed.

## Build and test

```bash
cargo build
cargo test                       # expect 170 passed, 0 failed
cargo test --features klu        # expect 198 passed, 0 failed
cargo test --test bde_test       # one test binary
cargo clippy --all-targets
```

CI (`.github/workflows/build.yml`) runs only the default-feature build and test
— every FFI-backed feature below needs something the runner does not have.

## Feature flags, and what each one costs you

| Feature | Needs at build time | Needs at runtime | Notes |
|---|---|---|---|
| *(default)* | nothing | nothing | `faer` sparse LU + the pure-Rust `klu_native` |
| `klu` | C compiler, `libclang` | — | Compiles vendored SuiteSparse. LGPL — see `vendor/suitesparse/PROVENANCE.md` |
| `klu-dynamic` | `libklu.so` | `libklu.so` | Links a system KLU instead, for LGPL relinking |
| `pardiso` | `libclang`, `MKLROOT` | oneMKL | Proprietary; links a system install, vendors nothing |
| `cgmes` | — | — | Pulls `cimdecoder`/`cimstructs` from a pinned git tag |
| `python` | `maturin` | — | **Never** combine with a plain `cargo build`/`cargo test` — see the feature's comment in `Cargo.toml` |
| `gpu` | CUDA toolkit (`nvcc` + headers), `libclang` | NVIDIA GPU | Compiles `cuda/gridoxide_kernels.cu`. `CUDA_ARCH` defaults to `sm_80` |
| `cudss` | cuDSS >= 0.4.0, `libclang` | NVIDIA GPU | Separate NVIDIA download; batch API needs 0.4.0+ |

`gpu` and `cudss` are used together for the batched path. Neither builds on a
machine without the CUDA toolkit — that is expected, not a broken checkout.

## Architecture

- `src/solver.rs` — the Newton loop and `trait LinearSolver`, the seam every
  sparse backend plugs into. `PersistentSolver` caches the symbolic
  factorization across solves (~45% of solve time on a 9,241-bus case).
- `src/jacobian.rs` — `JacobianPattern`: analyze the sparsity pattern once,
  refill values at fixed offsets each iteration. The H/N/M/L formulas here are
  the single source of truth; the CUDA kernel is a transliteration of them, not
  a rederivation.
- `src/network.rs` — Y-bus assembly, `power_injections`, `effective_injection`
  (the ZIP load model), `linear_initial_guess`.
- `src/batch.rs` — `BatchSolver`: one topology, N scenarios, rayon across
  cores. **This is the bar every GPU claim must clear.** Beating a
  single-threaded CPU solver is not a result.
- `src/bde.rs` — block-diagonal embedding. Three entry points; see below.
- `src/sparse*.rs` — one `LinearSolver` impl per backend (`faer`, KLU, PARDISO,
  cuDSS). All raw FFI stays inside its own file.

### The GPU path (branch `gpu-nvidia`)

Read `plans/GPU_PLAN.md`'s amendment first — the original plan's §3 property 2
turned out to be the trap, and the amendment says why. `scripts/GPU_RUNBOOK.md`
Session 3 is the operational checklist for a rented A100 box.

- `src/device_layout.rs` — **deliberately not feature-gated.** Host-side
  flattening (Y-bus CSR, ZIP coefficients, index maps, strides, CSR scatter
  map) with unit tests that run on machines with no GPU. When adding anything
  the kernels index into, put the layout decision *here* with a test against
  the CPU implementation, not inline in `gpu.rs`. This is the main defence
  against burning metered GPU time on layout bugs.
- `cuda/gridoxide_kernels.cu` — five kernels, `extern "C"` launchers taking
  `void* stream`. Each names its CPU original in a comment; keep that true.
- `src/gpu.rs` — raw CUDA FFI. Owns the stream, persistent device buffers, and
  the stable pointers cuDSS binds to.
- `src/sparse_cudss.rs` — `CudssBatchedSystem` (uniform batch API, the path to
  use) and `CudssRealSystem` (stacked, kept as the A/B control).
- `src/bde.rs` — `solve_batch_block_diagonal_batched_device` is the current
  path; the other two are controls. Do not delete them: they are what makes a
  performance claim measurable rather than asserted.

CubeCL was removed at commit `ff92b66`, which still holds the portable
wgpu/ROCm kernel. There is currently **no AMD path**.

## Conventions

- **Comments explain *why*, and stay honest.** This codebase documents
  measured regressions, abandoned approaches and open questions in the code
  itself (see `bde.rs`'s doc comments). Match that. Do not quietly delete a
  comment describing a problem because the problem is inconvenient.
- **Never quote a benchmark number you did not measure on the machine you are
  on.** `scripts/bench/README.md`'s numbers come from a thermally throttled
  laptop APU. Re-measure; say which host.
- **Cross-validation is the point.** New solver behaviour needs to agree with
  the existing 12-case accuracy suite (`scripts/bench/run_case_suite.py`).
- `src/klu_native/` is a close translation of vendored SuiteSparse C. Keep it
  structurally recognisable against the original rather than idiomatic.

## Benchmark data

MATPOWER cases live in the `tests/data/benchmark-grids` git submodule; the
PGM-JSON the benchmarks consume is *generated* into
`scripts/bench/.case-cache/`, which is gitignored. A fresh clone has neither:

```bash
git submodule update --init --recursive
maturin develop --release --features python,klu
pip install numpy scipy          # what the `matpower` extra pulls in
mkdir -p scripts/bench/.case-cache
python3 scripts/bench/matpower_to_pgm.py \
    tests/data/benchmark-grids/matpower/case9241pegase.m \
    scripts/bench/.case-cache/case9241pegase.json
```

Install the deps with plain `pip`, not `pip install -e '.[matpower]'` — that
would re-run the maturin build without `--features python,klu` and replace the
extension you just built.

CGMES test data is a second submodule, `tests/data/CGMES-Test-Configurations`.
