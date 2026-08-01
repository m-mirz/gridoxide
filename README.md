# gridoxide

`gridoxide` is an AC power flow analysis tool written in Rust. It solves the power flow equations for
an electrical grid with the Newton-Raphson method, using a sparse Jacobian throughout — assembly,
factorization, and solve.

📖 **[Documentation](https://m-mirz.github.io/gridoxide/)** — the full book covers the method, the
solver backends, the CGMES importer, and the benchmark results. Its source lives in [`docs/`](docs/).

## Quick start

Rust — you need the [Rust toolchain](https://rustup.rs/), nothing else for a default build:

```bash
cargo build --release
cargo test
```

Python:

```bash
pip install gridoxide
```

```python
import gridoxide

model = gridoxide.PowerFlowModel.from_pgm_json("grid.json", backend="klu_native")
model.solve()
print(model.voltage_mag(), model.voltage_ang())
```

See [Building and Running](docs/src/getting_started/building.md) and
[Python Bindings](docs/src/getting_started/python.md).

## What it does

- **Newton-Raphson AC power flow**, symmetric and asymmetric, with a sparse Jacobian and symbolic
  factorization reused across both NR iterations and repeated solves
  (`solver::PersistentSolver`) — see [Backends and Factorization Reuse](docs/src/solvers/backends.md).
- **Five interchangeable linear-solver backends** — `faer` (`Scalar`), a hand-written block LU
  (`Block`), vendored SuiteSparse KLU over FFI (`Klu`), a from-scratch Rust translation of KLU
  (`KluNative`, always built), and Intel oneMKL PARDISO (`Pardiso`). All five produce identical
  converged voltages; the choice is purely performance.
- **Three input formats** — its own JSON, [power-grid-model](https://github.com/PowerGridModel/power-grid-model)
  JSON, and [CGMES RDF/XML](docs/src/cgmes/index.md) (`--features cgmes`) with node-breaker
  reduction, all four phase-tap-changer flavors, 3-winding star resolution, HVDC, and SVC.
- **Modeling features beyond the plain formulation** —
  [reactive power limits](docs/src/powerflow/q_limits.md) (PV→PQ switching),
  [zero-impedance branches](docs/src/powerflow/zero_impedance_branches.md), and
  [multi-island solves](docs/src/powerflow/multi_island.md) with a per-island status report.
- **Batched solving** over one shared topology, parallel across cores via rayon (`batch::BatchSolver`).

[Feature Comparison](docs/src/reference/feature_comparison.md) is a detailed survey against five
other power flow tools, including the gaps gridoxide hasn't closed.

## Benchmarks

`scripts/bench/README.md` is the single source of truth for every benchmark number in this project;
[Benchmarking and Profiling](docs/src/reference/benchmarking.md) is a map into it. In short:
gridoxide is one of two solvers out of six that converge on all 12 real IEEE/MATPOWER test cases, and
its `Klu` backend is frequently faster than lightsim2grid's own KLU-backed C++ solver on real
transmission topology — while power-grid-model remains clearly faster on synthetic radial
distribution grids, a real standing gap.

## License

gridoxide's own code is Apache-2.0 ([`LICENSE`](LICENSE)). A **default** build always includes
`src/klu_native/`, a from-scratch Rust translation of vendored SuiteSparse AMD/BTF/KLU source, so
`Cargo.toml`'s license field is `Apache-2.0 AND BSD-3-Clause AND LGPL-2.1-or-later` — accurate for
every default build, not just an opt-in one.

Building with `--features klu` compiles the vendored SuiteSparse C itself, adding LGPL relinking
obligations for anyone distributing the resulting binary (a `klu-dynamic` sub-feature exists for
that case). Building with `--features pardiso` dynamically links a locally-installed Intel oneMKL
under Intel's own proprietary license; nothing MKL-derived is vendored or shipped by this repo.

See [Provenance and Licensing](docs/src/reference/provenance.md) for the full breakdown, and the
`PROVENANCE.md` files under `src/klu_native/` and `vendor/suitesparse/` for per-file detail.
