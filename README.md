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
imports `faer` types directly, so a different sparse-solver backend (e.g. a KLU binding, if `faer` ever
proves numerically insufficient) could be swapped in behind the same interface without touching the rest of
the codebase.

## Profiling

For profiling with perf, set

    sysctl kernel.perf_event_paranoid=1