# Backends and Factorization Reuse

The Y-bus admittance matrix and Newton-Raphson Jacobian are stored and factored as **sparse**
matrices, not dense ones. This page covers why that matters, how factorization work is reused across
solves, and what the five interchangeable linear-solver backends are.

## Why sparse

At 2,605 nodes the original dense solver was over 100,000x slower than power-grid-model; the sparse
rewrite closed that to roughly an order of magnitude on a cold, single-shot solve. The underlying
grid is sparse — each bus only connects to a handful of neighbors — so a dense representation was
doing asymptotically unnecessary work regardless of how fast the constant-factor arithmetic was.

Two things make this work, rather than just "swap in a sparse matrix type":

- **Sparse-aware assembly.** Both the Jacobian build (`solver::build_jacobian_triplets`) and the
  linear initial-guess warm start (`network::linear_initial_guess`) walk each bus's actual
  admittance neighbors (via `network::YBusSparse::row`) instead of looping over every possible bus
  pair. The O(n²)/O(m²) assembly cost has to go too, or a sparse *solve* alone doesn't fix the
  bottleneck.
- **Symbolic factorization reuse.** A Newton-Raphson Jacobian has the same sparsity *pattern* every
  iteration — same bus topology, only numeric values change — so `solver::newton_raphson` computes
  the symbolic factorization (fill-reducing ordering) once and reuses it for a cheap numeric-only
  refactorization on each iteration (`sparse::RealSparseSystem`), mirroring what PGM's own solver
  does internally.

[Inside KLU](./klu.md) walks through exactly what "symbolic factorization" and "numeric-only
refactorization" mean, step by step, on a real Jacobian.

## Reusing factorization across repeated solves

A single `newton_raphson`/`newton_raphson_with_backend` call reuses its symbolic factorization
across its own NR iterations, but starts cold on every call — re-deriving the fill-reducing ordering
from scratch. That is fine for a genuinely one-off solve and wasteful for anything that solves the
*same* topology repeatedly: a time series, a batch of scenarios, contingency analysis. In those, only
bus values (`p_spec`, `q_spec`, voltage guess) change between calls, not the topology.

`solver::PersistentSolver` extends the reuse *across* calls. Construct one per topology, then solve
as many times as needed; only the first call pays for symbolic factorization.

```rust
use gridoxide::solver::{JacobianBackend, PersistentSolver};

let mut solver = PersistentSolver::new(JacobianBackend::Klu);
for scenario in scenarios {
    apply_scenario(&mut buses, scenario); // changes p_spec/q_spec only
    solver.solve(&mut buses, &ybus, 1e-6, 20);
}
```

Call `.reset()` (or construct a new solver) if the topology itself changes between solves. This is a
meaningful win on real-world grids, since a cold solve otherwise redoes COLAMD/AMD/BTF ordering from
scratch every call. `examples/bench_network.rs` exposes it as an optional `warm` mode; its default
`cold` mode still measures "N independent flat-start solves with no shared state," a different and
also legitimate number. See [Benchmarking](../reference/benchmarking.md) for the measured
warm-vs-cold figures.

## The backend interface

`src/sparse.rs` is the thin backend wrapper around [`faer`](https://crates.io/crates/faer), and
intentionally the only file that imports `faer` types directly, so a different sparse-solver backend
can be swapped in behind the same interface without touching the rest of the codebase.

`solver::newton_raphson` always uses the default `Scalar` (`faer`-backed) path.
`newton_raphson_with_backend` additionally accepts four alternatives via `solver::JacobianBackend`:

### `Block`

`src/block_sparse.rs`, no extra build requirements. Groups each bus's own (angle,
voltage-magnitude) unknowns into one dense 2×2 block, mirroring power-grid-model's block-per-bus
matrix structure, with a hand-written Gilbert-Peierls sparse block LU (`block_sparse::BlockLu`).
Symmetric power flow only. Consistently faster than `Scalar`.

### `Klu`

`src/sparse_klu.rs`, needs `cargo build --features klu`. The same scalar Jacobian as `Scalar`, solved
by [SuiteSparse's KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse) instead of `faer`,
vendored and compiled from source (`vendor/suitesparse/`) rather than depending on a third-party Rust
wrapper crate. Needs a C compiler and `libclang` (for `bindgen`) at build time.

**KLU and BTF (one of KLU's own dependencies) are LGPL-2.1-or-later**. See [Provenance and Licensing](../reference/provenance.md).

### `KluNative`

`src/klu_native/`, no extra build requirements. A from-scratch Rust *translation* of the same KLU
algorithm `Klu` links over FFI: BTF block-triangular preprocessing, per-block AMD ordering, a
partial-pivoting Gilbert-Peierls LU kernel with Eisenstat-Liu pruning, and cheap numeric-only
refactorization — all faithfully ported, not a simplified reimplementation. No C compiler or
`libclang` needed, so unlike `Klu` it is **always built**.

Validated end-to-end against real KLU on all 13 real MATPOWER benchmark cases (14 to 9,241 buses):
identical iteration counts and identical converged voltages on every case. One known, documented gap
— row scaling (`klu_native::scale`) is ported and independently tested but not yet wired into the
factor/refactor path (see `src/klu_native/mod.rs`'s module doc comment). That is a
numerical-stability preconditioning step, not a correctness one, so it doesn't affect the results
above.

[Inside KLU](./klu.md) is a walkthrough of this port specifically.

### `Pardiso`

`src/sparse_pardiso.rs`, needs `cargo build --features pardiso` and `MKLROOT` set at build time. The
same scalar Jacobian as `Scalar`, solved by
[Intel oneMKL's PARDISO](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/)
direct solver. Unlike `Klu`, nothing is vendored — MKL is proprietary (Intel Simplified Software
License, not OSS), so this only dynamically links a locally-installed oneMKL (`libmkl_rt.so`,
discovered via `MKLROOT`, e.g. `source /opt/intel/oneapi/setvars.sh`) and generates FFI bindings via
`bindgen` against *that install's own* `mkl_pardiso.h`. No MKL header or source is copied into this
repo.

PARDISO's C API is one function called repeatedly with different `phase` values against a persistent
opaque handle, rather than KLU's separate analyze/factor/refactor/solve functions. `mtype = 11`
(real, nonsymmetric) and `iparm[34] = 1` (0-based indexing) are the two settings that matter for
matching gridoxide's CSR/CSC conventions.

**Not built or tested in CI** — no CI runner has MKL installed — so this is a
local/manual-verification-only backend.

## How they compare

All four alternatives are strictly parallel to `Scalar`, not replacements. A bug in any of them
can't affect `newton_raphson`'s default behavior, and every existing test keeps using `Scalar` unless
it explicitly opts into a different backend.

All five backends produce **identical converged voltages at every scale** — these are purely
performance comparisons, not correctness trade-offs. In rough terms:

- `Block`, `Klu`, and `KluNative` are all meaningfully faster than `Scalar`.
- `Klu` and `KluNative` land close to each other, slightly ahead of `Block`.
- `Pardiso` carries a largely size-independent fixed setup cost from its default matching/scaling
  preprocessing, making it the slowest backend at small problem sizes — even behind `Scalar` — though
  it scales better than `Scalar` as node count grows.

PGM is clearly faster than any gridoxide backend on synthetic radial-distribution/LV topology, a
real, standing gap this project hasn't closed. That gap doesn't hold universally, though: on
real-world *transmission*-topology grids, gridoxide's `Klu` backend is frequently faster than
lightsim2grid's own KLU-backed C++ solver. The comparison depends on topology, not just
implementation language.

[Benchmarking](../reference/benchmarking.md) points at the full measured numbers, exact ratios, and
how to reproduce them.
