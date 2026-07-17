# gridoxide

`gridoxide` is a power flow analysis tool written in Rust. It uses the Newton-Raphson method to solve the power flow equations for an electrical grid defined in a JSON file.

## Building

To build the project, you need to have the Rust toolchain installed. You can find instructions on how to install it at [rustup.rs](https://rustup.rs/).

Once you have Rust installed, you can build the project by running:

```bash
cargo build
```

For an optimized release build, use:

```bash
cargo build --release
```

## Running

You can run the program using `cargo run`:

```bash
cargo run
```

If you have built the project, you can also run the executable directly. From the project root:

For a debug build:
```bash
./target/debug/gridoxide
```

For a release build:
```bash
./target/release/gridoxide
```

## Testing

To run the tests for the project, use:

```bash
cargo test
```

## Sparse solver

The Y-bus admittance matrix and Newton-Raphson Jacobian are stored and factored as **sparse** matrices (via
the [`faer`](https://crates.io/crates/faer) crate), not dense ones.

At 2,605 nodes the dense solver was over 100,000x slower than PGM; the sparse rewrite closed that to roughly
an order of magnitude, which is the realistic target — PGM's C++ core has additional tuning (e.g. reusing a
factorization across repeated solves) this implementation doesn't attempt to replicate. The underlying grid
is sparse (each bus only connects to a handful of neighbors), so a dense representation was doing asymptotically
unnecessary work regardless of how fast the constant-factor arithmetic was.

Two things make this work, not just "swap in a sparse matrix type":

- **Sparse-aware assembly.** Both the Jacobian build (`solver::build_jacobian_triplets`) and the
  linear initial-guess warm start (`network::linear_initial_guess`) walk each bus's actual admittance
  neighbors (via `network::YBusSparse::row`) instead of looping over every possible bus pair — the O(n²)/O(m²)
  assembly cost has to go too, or a sparse *solve* alone doesn't fix the bottleneck.
- **Symbolic factorization reuse.** A Newton-Raphson Jacobian has the same sparsity *pattern* every
  iteration (same bus topology, only numeric values change), so `solver::newton_raphson` computes the
  symbolic factorization (fill-reducing ordering) once and reuses it for a cheap numeric-only refactorization
  on every iteration (`sparse::RealSparseSystem`), mirroring what PGM's own solver does internally.

See `src/sparse.rs` for the thin backend wrapper around `faer` — it's intentionally the only file that
imports `faer` types directly, so a different sparse-solver backend can be swapped in behind the same
interface without touching the rest of the codebase. Two such backends exist today, both strictly opt-in
experiments selectable via `solver::JacobianBackend` (`newton_raphson_with_backend`) — see the next section.

## Experimental backends

`solver::newton_raphson` always uses the default `Scalar` (`faer`-backed) path described above.
`newton_raphson_with_backend` additionally accepts:

- **`JacobianBackend::Block`** (`src/block_sparse.rs`, no extra build requirements) — groups each bus's own
  (angle, voltage-magnitude) unknowns into one dense 2×2 block, mirroring power-grid-model's block-per-bus
  matrix structure, with a hand-written Gilbert-Peierls sparse block LU (`block_sparse::BlockLu`). Symmetric
  power flow only. **1.6-3x faster than `Scalar`** at every benchmarked scale.
- **`JacobianBackend::Klu`** (`src/sparse_klu.rs`, needs `cargo build --features klu`) — the same scalar
  Jacobian as `Scalar`, solved by [SuiteSparse's KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse)
  instead of `faer`, vendored and compiled from source (`vendor/suitesparse/`, see
  `vendor/suitesparse/PROVENANCE.md`) rather than depending on a third-party Rust wrapper crate. Needs a C
  compiler and `libclang` (for `bindgen`) at build time. **KLU and BTF (one of KLU's own dependencies) are
  LGPL-2.1-or-later** — this is why the `klu` feature is opt-in rather than always built; a `klu-dynamic`
  sub-feature links a system-installed `libklu.so` instead of compiling the vendored copy statically, for
  anyone who needs strict LGPL relinking compliance. Matches or beats `Block`'s performance at every
  benchmarked scale.

Both are strictly parallel to `Scalar`, not replacements — a bug in either can't affect `newton_raphson`'s
default behavior, and every existing test keeps using `Scalar` unless it explicitly opts into a different
backend (see `tests/block_jacobian_test.rs`, `tests/klu_jacobian_test.rs`).

Measured with `examples/bench_network.rs --backend {scalar,block,klu}` (see `scripts/bench/README.md` for how
to reproduce these numbers, including generating the benchmark grids and running the power-grid-model
comparison from the section above):

| Nodes | Scalar | Block | Klu |
|---|---|---|---|
| 192 | 1.94 ms | 0.74 ms | 0.67 ms |
| 1,003 | 11.70 ms | 3.93 ms | 3.88 ms |
| 2,605 | 15.41 ms | 10.71 ms | 9.27 ms |

All three produce identical converged voltages at every scale — these are purely performance comparisons, not
correctness trade-offs.

A second, separate benchmark compares gridoxide against four other independent solvers — PGM,
[lightsim2grid](https://github.com/m-mirz/lightsim2grid), RTE's
[powsybl-open-loadflow](https://github.com/powsybl/powsybl-open-loadflow) (via `pypowsybl`), and pandapower's
own default solver — on 12 real IEEE/MATPOWER power-system test-case grids (14 to 9,241 buses) — see
`scripts/bench/README.md`'s "Benchmark against real power-system test-case grids" section for the full
results table and methodology. gridoxide and pandapower's own native path are the only two of the five that
converge on all 12; the other three each fail on a subset of the same handful of genuinely hard cases (RTE's
own real production grids), confirmed by cross-checking against `powsybl-open-loadflow` directly, not a
gridoxide gap. This benchmark is also what led to `src/pgm.rs` parsing PGM's `voltage_regulator` component:
real generator PV (voltage-controlled) buses, not just the slack/PQ split described above — see
`types::BusType::PV` and `solver::newton_raphson_scalar`/
`newton_raphson_klu`'s existing PV handling, now actually reachable from PGM JSON input — and to fixing a
real gap in `network::transformer_tap`'s off-nominal tap-ratio clamping (`src/network.rs`).

## Profiling

For profiling with perf, set

    sysctl kernel.perf_event_paranoid=1