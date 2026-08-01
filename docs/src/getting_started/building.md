# Building and Running

## Building

You need the Rust toolchain — see [rustup.rs](https://rustup.rs/) for installation instructions.
Then:

```bash
cargo build
```

For an optimized release build:

```bash
cargo build --release
```

A default build needs no C compiler, no system libraries, and no environment variables. Everything
beyond that — the `klu`, `pardiso`, `cgmes`, and `python` features — is opt-in, and each is
described on the page that covers it
([backends](../solvers/backends.md), [CGMES input](../cgmes/index.md),
[Python bindings](./python.md)).

## Running

```bash
cargo run
```

Or run the built executable directly from the project root:

```bash
./target/debug/gridoxide     # debug build
./target/release/gridoxide   # release build
```

## Testing

```bash
cargo test
```

Note that `cargo test` never compiles the `python` feature — the feature must not be combined with a
plain `cargo` invocation at all (see [Python bindings](./python.md)). Tests for the optional
backends live in their own files (`tests/block_jacobian_test.rs`, `tests/klu_jacobian_test.rs`,
`tests/klu_native_jacobian_test.rs`, `tests/pardiso_jacobian_test.rs`) and only run when the
matching feature is enabled.

## Next steps

To measure rather than just run, see [Benchmarking and Profiling](../reference/benchmarking.md) for
the benchmark harnesses, the `perf` setup, and where the measured numbers are recorded.
