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

A default build needs no C compiler, no system libraries, and no environment variables. The `opf`
feature ([optimal power flow](../opf/index.md)) keeps that property — it is pure Rust and adds no
dependencies. The `capi` feature ([C and C++ API](./c_api.md)) keeps that property too — pure Rust plus a
build-time crate, nothing to install. Everything else — the `klu`, `pardiso`, `opf-highs`,
`opf-ipopt`, `cgmes`, and `python` features — is opt-in and needs something installed; each is
described on the page that covers it
([backends](../solvers/backends.md), [CGMES input](../cgmes/index.md),
[Python bindings](./python.md)).

`parallel`, on by default, gives faer its own threads. WebAssembly builds turn it off
(`--no-default-features`), as `wasm32-unknown-unknown` cannot start threads. There the power flow, CGMES input and
`security_analysis` run on one thread; `BatchSolver` returns an error, as it builds a thread pool.

## Running

The command line is `gridox`, in its own package, `gridoxide-cli`, so a plain `cargo build` at the
root builds only the library. Run it through cargo, as in `cargo run -p gridoxide-cli -- solve
case118.m`, or build it with `cargo build -p gridoxide-cli` and run the executable from the
project root:

```bash
./target/debug/gridox     # debug build
./target/release/gridox   # release build
```

`gridox` has every feature on that needs no system library (`cgmes`, `ucte`, `iidm`, `matpower`,
`opf`, `rao`, `dynamics`); the others it forwards to the library under the same names, as in
`cargo build -p gridoxide-cli --features opf-highs`. `cargo install gridoxide-cli` installs it, and
each GitHub release carries it for Linux, Windows and macOS.

### Subcommands

`gridox --help` prints the full argument list. Each subcommand reads a
[power-grid-model JSON](../cgmes/index.md) document, except `switches`, which reads CGMES profiles.

| Command | What it does |
|---|---|
| `estimate <path> [--iterative-linear]` | [State estimation](../state_estimation/index.md) over a document containing sensors. Newton-Raphson by default; the flag selects the faster linearized method. |
| `dc <path> [--ignore-g] [--ptdf <bus>] [--lodf <branch>]` | [DC power flow](../powerflow/dc.md), optionally printing one sensitivity column. |
| `sensitivity <path> [--dp\|--dq <bus>] [--dk\|--dalpha <branch>] [--watch <branch>] [--terminal from\|to]` | [AC sensitivity](../sensitivity/ac.md). The first four flags each pick one variable and report what responds; `--watch` picks one branch and reports what would move it. |
| `short-circuit <path> [--scaling max\|min]` | [IEC 60909 fault currents](../short_circuit/index.md). `--scaling` picks the voltage factor `c`. |
| `opf <network.json> [--ac] [--data <opf.json>] [--no-shedding] [--shed-price <$/MWh>] [--ignore-r] [--highs] [--no-limits] [--max-iter <n>]` | [DC optimal power flow](../opf/index.md): least-cost dispatch subject to generator limits and branch ratings, reporting dispatch, locational marginal prices and what binds. Costs and limits come from a companion document, defaulting to `<network>.opf.json`. `--ignore-r` selects `b = 1/x` — note this is the *opposite* default from `dc` above, [on purpose](../opf/index.md#which-susceptance--and-why-it-is-not-a-detail). `--highs` swaps the built-in interior-point solver for HiGHS. `--ac` solves the full [AC problem](../opf/index.md#ac-opf) instead — real voltages, reactive power and losses — which is nonconvex, so its answer is a local optimum; `--no-limits` drops branch ratings and `--max-iter` caps the solve; both need `--ac`, and the DC-only flags refuse it. Needs the `opf` feature (`--highs` additionally needs `opf-highs`). |
| `switches <profile.xml>… [--retain …] [--open <mrid>] [--solve]` | [Node-breaker switching devices](../cgmes/node_breaker.md) read from CGMES EQ+SSH. Needs the `cgmes` feature. |

Bus and branch arguments are gridoxide's own 0-based indices, and branches are ordered lines first
then transformers — the tables each command prints say which is which.

```bash
# Least-cost dispatch, and what the network is costing you.
cargo run -p gridoxide-cli -- opf grid.json

# The same question against the full AC equations.
cargo run -p gridoxide-cli -- opf grid.json --ac

# Which injection would relieve an overload on branch 8?
cargo run -p gridoxide-cli -- sensitivity grid.json --watch 8

# The largest three-phase fault current this network can produce.
cargo run -p gridoxide-cli -- short-circuit grid.json --scaling max
```

`gridox --help` lists the commands and `gridox <command> --help` each one's flags; a flag a command
does not have is an error, not ignored. The library — not the binary — is the intended interface for
anything more involved.

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
