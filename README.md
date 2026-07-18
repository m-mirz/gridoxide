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
an order of magnitude on a cold, single-shot solve — the underlying grid is sparse (each bus only connects to
a handful of neighbors), so a dense representation was doing asymptotically unnecessary work regardless of
how fast the constant-factor arithmetic was.

Two things make this work, not just "swap in a sparse matrix type":

- **Sparse-aware assembly.** Both the Jacobian build (`solver::build_jacobian_triplets`) and the
  linear initial-guess warm start (`network::linear_initial_guess`) walk each bus's actual admittance
  neighbors (via `network::YBusSparse::row`) instead of looping over every possible bus pair — the O(n²)/O(m²)
  assembly cost has to go too, or a sparse *solve* alone doesn't fix the bottleneck.
- **Symbolic factorization reuse.** A Newton-Raphson Jacobian has the same sparsity *pattern* every
  iteration (same bus topology, only numeric values change), so `solver::newton_raphson` computes the
  symbolic factorization (fill-reducing ordering) once and reuses it for a cheap numeric-only refactorization
  on every iteration (`sparse::RealSparseSystem`), mirroring what PGM's own solver does internally. This
  reuse now also extends *across* repeated solves, not just the iterations within one — see
  `solver::PersistentSolver` below, PGM's own equivalent tuning this implementation used to not replicate.

## Reusing factorization across repeated solves

A single `newton_raphson`/`newton_raphson_with_backend` call already reuses its symbolic factorization across
its own NR iterations, but starts cold (re-derives the fill-reducing ordering from scratch) on every call —
fine for a genuinely one-off solve, wasteful for anything that solves the *same* topology repeatedly (a time
series, a batch of scenarios, contingency analysis), where only bus values (`p_spec`, `q_spec`, voltage
guess) change between calls, not the topology itself.

`solver::PersistentSolver` fixes this: construct one per topology, call `.solve(&mut buses, &ybus, tol,
max_iter)` as many times as needed, and only the *first* call pays for symbolic factorization — every later
call does a numeric-only refactorization, the same reuse `newton_raphson` already does within one call,
extended across calls. Call `.reset()` (or construct a new one) if the topology itself changes between
solves.

```rust
use gridoxide::solver::{JacobianBackend, PersistentSolver};

let mut solver = PersistentSolver::new(JacobianBackend::Klu);
for scenario in scenarios {
    apply_scenario(&mut buses, scenario); // changes p_spec/q_spec only
    solver.solve(&mut buses, &ybus, 1e-6, 20);
}
```

Measured on a 9,241-bus real-world grid (see the real-test-case benchmark below): reusing factorization
across repeated solves cut per-solve `Klu` time by ~45% (`perf` showed COLAMD/AMD/BTF ordering — the
fill-reducing step a cold solve redoes every time — responsible for roughly a third of total solve time on
that case). `examples/bench_network.rs` exposes this as an optional `warm` mode (its default `cold` mode
still measures "N independent flat-start solves with no shared state," a different, also-legitimate number)
— see `scripts/bench/README.md`.

## Python bindings

`src/python.rs` exposes `PersistentSolver` (and PGM-JSON loading) as a `gridoxide` Python extension module,
built with [maturin](https://www.maturin.rs/) (`maturin develop --release --features python,klu`) — gated
entirely behind the opt-in `python` Cargo feature, compiled by nothing else, so a plain `cargo build`/`cargo
test` never touches it.

```python
import gridoxide

model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="klu")
model.solve()
print(model.voltage_mag(), model.voltage_ang())
```

Repeated `.solve()` calls on the same `PowerFlowModel` reuse cached symbolic factorization exactly like the
Rust `PersistentSolver` API above, since it *is* that API — this is what lets `scripts/bench/` run its whole
comparison in pure Python (`scripts/bench/bench_gridoxide_native.py`), timing gridoxide with the same
`time.perf_counter()`-around-a-persistent-solve-object methodology every other tool there already uses (PGM's
`PowerGridModel`, lightsim2grid's `GridModel`, pandapower's `net`), rather than shelling out to a compiled
Rust binary and parsing its stdout. See `scripts/bench/README.md`'s "Python bindings" section for build
details and the constraint on never combining the `python` feature with a plain `cargo` invocation.

`python/tests/` has a small pytest suite (`scalar`/`block` only — no `klu`, matching what's published, see
below) checked against this project's own committed PGM reference fixtures, run by
`.github/workflows/python.yml` on every push/PR. `.github/workflows/pypi.yml` builds wheels (Linux/Windows/
macOS) plus an sdist and publishes to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/)
on `v*` tags — **the published wheel deliberately doesn't include the `klu` backend** (LGPL-2.1-or-later
vendored SuiteSparse source, and a C compiler + libclang needed on every target platform — see Cargo.toml's
`python`/`klu` feature doc comments); build from source with `--features python,klu` for that backend.

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
- **`JacobianBackend::KluNative`** (`src/klu_native/`, no extra build requirements) — a from-scratch Rust
  *translation* of the same KLU algorithm `Klu` links over FFI: BTF block-triangular preprocessing,
  per-block AMD ordering, a partial-pivoting Gilbert-Peierls LU kernel with Eisenstat-Liu pruning, and
  cheap numeric-only refactorization, all faithfully ported (not a simplified reimplementation — see
  `src/klu_native/PROVENANCE.md` for the file-by-file mapping back to the upstream C). No C compiler or
  `libclang` needed, so — unlike `Klu` — it's always built. Validated end-to-end against real KLU on all
  13 real MATPOWER benchmark cases (`scripts/bench/.case-cache/`, 14 to 9,241 buses): identical iteration
  counts and identical converged voltages on every case. One known, documented gap: row scaling
  (`klu_native::scale`) is ported and independently tested but not yet wired into the factor/refactor
  path (see `src/klu_native/mod.rs`'s module doc comment) — a numerical-stability preconditioning step,
  not a correctness one, so this doesn't affect the results above.
- **`JacobianBackend::Pardiso`** (`src/sparse_pardiso.rs`, needs `cargo build --features pardiso` and
  `MKLROOT` set at build time) — the same scalar Jacobian as `Scalar`, solved by
  [Intel oneMKL's PARDISO](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/) sparse
  direct solver instead of `faer`. Unlike `Klu`, nothing is vendored — MKL is proprietary (Intel Simplified
  Software License, not OSS), so this only dynamically links a locally-installed oneMKL (`libmkl_rt.so`,
  discovered via `MKLROOT`, e.g. `source /opt/intel/oneapi/setvars.sh`) and generates FFI bindings via
  `bindgen` against *that install's own* `mkl_pardiso.h` — no MKL header or source is copied into this repo.
  PARDISO's C API is one function called repeatedly with different `phase` values against a persistent
  opaque handle, rather than KLU's separate analyze/factor/refactor/solve functions; `mtype = 11` (real,
  nonsymmetric) and `iparm[34] = 1` (0-based indexing) are the two settings that matter for matching
  gridoxide's existing CSR/CSC conventions. **Not built or tested in CI** — no CI runner has MKL installed —
  so this is a local/manual-verification-only backend.

All four are strictly parallel to `Scalar`, not replacements — a bug in any of them can't affect
`newton_raphson`'s default behavior, and every existing test keeps using `Scalar` unless it explicitly
opts into a different backend (see `tests/block_jacobian_test.rs`, `tests/klu_jacobian_test.rs`,
`tests/klu_native_jacobian_test.rs`, `tests/pardiso_jacobian_test.rs`).

Measured with `examples/bench_network.rs --backend {scalar,block,klu,klu_native,pardiso} warm` (see
`scripts/bench/README.md` for how to reproduce these numbers, including generating the benchmark grids and
running the power-grid-model comparison from the section above):

| Nodes | Scalar | Block | Klu | KluNative | Pardiso | PGM (warm) |
|---|---|---|---|---|---|---|
| 192 | 1.28 ms | 0.52 ms | 0.44 ms | 0.45 ms | 2.06 ms | 0.42 ms |
| 1,003 | 7.86 ms | 2.80 ms | 2.52 ms | 2.69 ms | 5.06 ms | 0.93 ms |
| 2,605 | 20.97 ms | 6.81 ms | 6.04 ms | 6.56 ms | 11.73 ms | 2.49 ms |

All five gridoxide backends produce identical converged voltages at every scale — these are purely
performance comparisons, not correctness trade-offs. `KluNative` lands close to `Klu` (1.02-1.09x) across
this whole range of scales. `Pardiso` is 2-4.7x slower than `Klu` at every scale, and slower than even
`Scalar` at the smallest size (192 nodes) — its default nonsymmetric preprocessing (maximum weighted
matching + scaling) carries a largely size-independent fixed setup cost per solve that dominates at these
small problem sizes; it does scale somewhat better than the naive `Scalar`/`faer` path as node count grows
(overtaking `Scalar` by 2,605 nodes), but never catches up to `Klu`/`Block`/`KluNative` in this range. This
isn't a gridoxide-specific gap — PARDISO's general-purpose machinery (multi-threading support, iterative
refinement, pivoting robustness for a much broader class of matrices than KLU targets) is understood to pay
off more at larger/denser problems than these small radial-distribution grids, not at this scale.
`perf`-profiling traced that gap to allocator churn, not anything algorithmic: `kernel::refactor_block`
allocated two new `Vec`s per column on *every* Newton iteration's refactor (tens of millions of allocations
across a timed benchmark run on the larger real-world cases below), where real KLU's own `klu_refactor.c`
overwrites one already-allocated buffer in place and allocates nothing at all during a refactor. Rewriting
`refactor_block`/`refactor::refactor` to mutate the existing factorization in place, backed by a
`RefactorScratch` buffer `KluNativeSystem` reuses across every solve, eliminated nearly all of that churn —
see `src/klu_native/kernel.rs`'s `refactor_block_in_place` doc comment and `scripts/bench/README.md`'s
"Benchmark against real power-system test-case grids" section for the full before/after profiling story.
PGM is clearly faster than any gridoxide backend on *this* synthetic radial distribution/LV topology, even
with gridoxide's own factorization reuse now doing the same warm-solve trick PGM's own `PowerGridModel`
does — a real, standing gap, not one this project has closed. (Interestingly, that gap doesn't hold
universally: on the real-world *transmission*-topology grids in the next benchmark, gridoxide's `Klu`
backend is frequently faster than
lightsim2grid's own KLU-backed C++ solver — the comparison depends on topology, not just implementation
language.)

A second, separate benchmark compares gridoxide against four other independent solvers — PGM,
[lightsim2grid](https://github.com/m-mirz/lightsim2grid), RTE's
[powsybl-open-loadflow](https://github.com/powsybl/powsybl-open-loadflow) (via `pypowsybl`), and pandapower's
own default solver — on 12 real IEEE/MATPOWER power-system test-case grids (14 to 9,241 buses) — see
`scripts/bench/README.md`'s "Benchmark against real power-system test-case grids" section for the full
results table and methodology. gridoxide and pandapower's own native path are the only two of the five that
converge on all 12; the other three each fail on a subset of the same handful of genuinely hard cases (RTE's
own real production grids), confirmed by cross-checking against `powsybl-open-loadflow` directly, not a
gridoxide gap. Once compared warm-vs-warm (`PersistentSolver`, see above — the earlier `cold` comparison made
gridoxide look 1.3-1.7x slower across the board, an artifact of redoing symbolic factorization every repeat
that lightsim2grid's own benchmark never does), `Klu` is frequently *faster* than lightsim2grid's own
KLU-backed C++ solver on this real transmission-topology data, tied only on the largest (9,241-bus) case —
even though PGM still clearly beats every gridoxide backend on the synthetic radial-distribution topology
above, so the comparison genuinely depends on grid topology, not just implementation language. This benchmark
is also what led to `src/pgm.rs` parsing PGM's `voltage_regulator` component:
real generator PV (voltage-controlled) buses, not just the slack/PQ split described above — see
`types::BusType::PV` and `solver::newton_raphson_scalar`/
`newton_raphson_klu`'s existing PV handling, now actually reachable from PGM JSON input — and to fixing a
real gap in `network::transformer_tap`'s off-nominal tap-ratio clamping (`src/network.rs`).

`Pardiso` also converges to the same voltages as every other gridoxide backend on all 12 of these real
MATPOWER cases, but with a gap to `Klu` that shrinks sharply with problem size — from ~2.7-3.5x slower on the
two smallest cases down to roughly parity on the largest (`case6515rte`: ~1.01x; `case9241pegase`: ~1.22x) —
see `scripts/bench/README.md`'s results table for the full per-case numbers. This matches the fixed-overhead
pattern already seen on the synthetic grids above: PARDISO's default matching/scaling preprocessing costs
about the same regardless of matrix size, so it matters far less once there's enough real factorization work
to amortize it against. Reproducing this column needs the same local MKL install as everywhere else
`pardiso` is used, on top of the PGM/lightsim2grid/pypowsybl/pandapower Python environments
that benchmark already requires.

## Profiling

For profiling with perf, set

    sysctl kernel.perf_event_paranoid=1

## License

gridoxide's own code is licensed under Apache-2.0 (`LICENSE`). Two pieces of vendored/translated
third-party code are also always part of a default build:

- **`src/klu_native/`** (always built, no feature gate) is a from-scratch Rust translation of vendored
  SuiteSparse `AMD`, `BTF`, and `KLU` C source (see the "Experimental backends" section above). A close
  translation of licensed source carries forward its upstream license, and that's not one license here:
  `AMD` is BSD-3-Clause upstream, while `BTF`/`KLU` are LGPL-2.1-or-later — see
  `src/klu_native/PROVENANCE.md` for the exact file-by-file breakdown.

As a result, `Cargo.toml`'s `license` field is
`"Apache-2.0 AND BSD-3-Clause AND LGPL-2.1-or-later"` — accurate for every default `cargo build`, not
just an opt-in one.

Building with `cargo build --features klu` additionally compiles the vendored SuiteSparse C itself
(`vendor/suitesparse/`, see `vendor/suitesparse/PROVENANCE.md`) into the binary via FFI — already
BSD-3-Clause/LGPL-2.1-or-later per the same table above, so this doesn't add any license beyond what's
already listed, but it does add LGPL's relinking obligations for anyone distributing a binary built with
that feature (a `klu-dynamic` sub-feature exists for that case — see `Cargo.toml`'s feature doc
comments).

Building with `cargo build --features pardiso` is a separate case from all of the above: it dynamically
links a locally-installed Intel oneMKL (`libmkl_rt.so`) at build/run time, under Intel's own Simplified
Software License — not LGPL, not OSS, and not vendored or redistributed by this repo in any form (no MKL
header or source is copied in; `bindgen` only reads the local install's own `mkl_pardiso.h` at build time to
generate FFI bindings). Because nothing MKL-derived is ever copied into or shipped by this crate,
`Cargo.toml`'s `license` field does **not** need to change for this feature. Anyone who builds with
`--features pardiso` and distributes the resulting binary is responsible for their own compliance with
Intel's oneMKL redistribution terms — this project doesn't audit that on their behalf.