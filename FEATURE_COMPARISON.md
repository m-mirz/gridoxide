# Feature comparison: gridoxide vs. reference tools

A survey of the three reference implementations checked out under `references/` (itself
gitignored, hence this file living at the repo root instead) — what each actually supports
(verified against source/docs, not assumed), compared against gridoxide's own current scope —
used to decide what gridoxide tackles next. See each tool's own CLAUDE.md/README for how to
consult them further; this file is a snapshot, not a living document, and will drift as both
gridoxide and the references evolve.

## Summary table

| Feature | lightsim2grid | power-grid-model | powsybl-open-loadflow | **gridoxide (today)** |
|---|---|---|---|---|
| AC power flow (Newton-Raphson) | ✅ | ✅ | ✅ | ✅ |
| DC / linear power flow | ✅ | ✅ ("linear" mode) | ✅ | ⚠️ only as an internal initial guess (`network::linear_initial_guess`), not a standalone mode |
| Gauss-Seidel | ✅ (+ "synch" variant) | ❌ | ❌ | ❌ |
| Fast-decoupled (XB/BX) | ✅ | ❌ | ✅ | ❌ |
| **Q-limit enforcement (PV→PQ switching)** | ❌ explicitly disclaimed | ⚠️ stubbed, "not yet fully implemented" | ✅ `ReactiveLimitsOuterLoop`, incl. capability curves | ✅ `solver::newton_raphson_enforcing_q_limits` (opt-in; plain `newton_raphson` still ignores `q_min`/`q_max`) |
| Distributed slack (multi-bus) | ✅ | ❌ | ✅ + area-interchange control | ❌ single slack only |
| Remote / shared voltage control | ❌ | ❌ | ✅ | ❌ |
| Transformer tap / phase-shifter auto-control | ❌ (fixed at init only) | ✅ `TapChangingStrategy` outer loop | ✅ several outer loops (voltage, reactive power, phase) | ❌ static taps only |
| 3-winding transformers | ❌ absent | ✅ (star-equivalent via 2 legs) | ✅ | ✅ (already have a passing test fixture) |
| Switches / node-breaker topology | ❌ (TODO in source) | ⚠️ implicit via `from_status`/`to_status`, no discrete switch component | ✅ full node-breaker + `NodeBreakerTraverser` | ❌ |
| HVDC | ✅ DC lines | ❌ | ✅ VSC/LCC | ❌ |
| SVC (static var compensator) | ❌ | ❌ | ✅ | ❌ |
| Asymmetric / unbalanced power flow | ❌ symmetric only | ✅ | ✅ (`LfAsym*`) | ✅ (already solving, tested against PGM fixtures) |
| **Contingency / N-1 batch analysis** | ✅ `ContingencyAnalysis`, reuses factorization, ~20x speedup claimed | ❌ | ✅ + Woodbury fast-DC path | ❌ (but `PersistentSolver`'s factorization reuse is the exact prerequisite, already built) |
| Time-series / batch injections | ✅ `TimeSerie`, ~13x speedup claimed | ✅ batch datasets, parallel via `threading` param | — | ❌ (same prerequisite exists) |
| Input validation | ❌ | ✅ `validate_input_data`/`validate_batch_data` | — | ❌ |
| Short-circuit calculation | ❌ | ✅ (IEC 60909) | ❌ | ❌ |
| State estimation | ❌ | ✅ (WLS) | ❌ | ❌ |
| Sensitivity analysis / OPF | ❌ / ❌ | ❌ / ❌ | ✅ / ❌ | ❌ / ❌ |
| Pluggable "outer loop" architecture | ❌ | ⚠️ ad hoc (tap optimizer only) | ✅ extensively (14+ outer loops) | ❌ |
| Sparse solver | KLU/Eigen/NICSLU/CKTSO, pluggable at runtime | hand-rolled 2×2-block LU, pivot perturbation off by default | KLU via JNI (primary path) | faer (`Scalar`) / hand-rolled 2×2-block LU (`Block`, matches PGM's own block granularity) / KLU (`Klu`) |

## Per-tool notes

### lightsim2grid (C++/Python, KLU-backed)

- Solvers: NR (single-slack and distributed-slack variants), Gauss-Seidel (+ "synch"), DC, fast-decoupled (XB/BX). Linear-solver backend is pluggable (Eigen SparseLU, KLU, NICSLU, CKTSO) via `SolverType`.
- Elements: lines, 2-winding transformers (fixed tap ratio + phase-shift angle, changeable only between solves), shunts, loads, static generators, storage, DC lines/HVDC. No 3-winding transformers, no SVC, no switches (explicit TODO in `SubstationContainer`).
- Its own `docs/disclaimer.rst` is refreshingly explicit about what it doesn't do: no Q-limit enforcement, fixed taps mid-solve, steady-state only, symmetric only.
- `ContingencyAnalysis` and `TimeSerie` batch classes reuse Ybus factorization across many solves rather than rebuilding from scratch — same idea as gridoxide's `PersistentSolver`, just applied to a batch-of-scenarios use case rather than only repeated single-topology solves.
- Ingests grids from pandapower and pypowsybl/IIDM directly (`gridmodel/from_pandapower`, `gridmodel/from_pypowsybl`).

### power-grid-model (C++/Python)

- Calculation types: power flow (sym + asym), state estimation (WLS, with observability checks), short-circuit (IEC 60909, phase-domain). No sensitivity/OPF.
- PF solver algorithms: Newton-Raphson (default), iterative-current, linear/linear-current (auto-selected when all loads are constant-impedance).
- No PV bus type in plain power flow ("not supported yet" per its own docs) — PV-like behavior instead comes from the newer `voltage_regulator` component, which fixes `|U|` and solves for Q; `q_min`/`q_max` exist on it but the automatic PV→PQ switching is explicitly flagged as not fully implemented.
- Same hand-rolled block-sparse LU architecture gridoxide's `Block` backend mirrors: per-bus 2×2 real blocks for NR power flow, full pivoting *within* a block only (no cross-block pivoting), pivot perturbation off by default for ordinary power flow (confirmed at the `newton_raphson_pf_solver.hpp` call site — this is what caused the `SparseMatrixError`s investigated earlier this session).
- Batch calculations reuse the prebuilt topology graph and matrix prefactorization across scenarios when only load/gen/source setpoints change (not when topology/tap/shunt status changes) — the same invariant `PersistentSolver::reset()` documents for gridoxide.
- `TapChangingStrategy` outer loop (disabled by default): `any_valid_tap`, `min_voltage_tap`, `max_voltage_tap`, `fast_any_tap`.
- `validate_input_data`/`validate_batch_data` exist but are explicitly *not* run automatically for performance reasons — recommended for debugging, not the hot path.

### powsybl-open-loadflow (Java, RTE)

- Calculation types: AC power flow, DC power flow, sensitivity analysis (AC+DC, incl. post-contingency), security/contingency analysis (N-1/N-k, AC+DC). No short-circuit, no state estimation.
- Solvers: Newton-Raphson (primary), Newton-Krylov, fast-decoupled — all pluggable via `AcSolverFactory` (service-loader based, genuinely extensible). Five voltage-initialization strategies (flat, warm/previous, uniform, DC-angle-based, magnitude-based).
- Most feature-rich of the three on voltage/reactive control: automatic PV→PQ switching with reactive capability curves, remote voltage control (one generator regulating a different bus), shared voltage control among multiple controllers, a priority scheme (generators > transformers > shunts), and even secondary voltage control (research-based).
- Distributed slack: on generators, loads, or "conform" loads; manual or automatic slack-bus selection with multiple strategies (first, largest-generator, most-meshed, named); also area-interchange-based distribution.
- Genuinely modular **outer-loop architecture** — `OuterLoop`/`OuterLoopContext`/`OuterLoopResult` abstractions, extensible via ServiceLoader, with 14+ concrete outer loops (distributed slack, area-interchange, reactive limits, transformer voltage/reactive-power control, phase control, shunt voltage control, secondary voltage control, HVDC AC-emulation limits). This is the architecture responsible for nearly every "extra" feature above the bare NR solve.
- Contingency analysis performance claim is best substantiated for **DC** specifically (Woodbury-formula fast path, `WoodburyEngine`/`WoodburyDcSecurityAnalysis`) — AC contingency/sensitivity analysis is documented as reusing full-resolve-style computation, and its own README's "Contributing" section flags AC performance as an open area, so the tool's reputation for contingency-analysis speed is strongest for DC, not universal.
- Supports asymmetric/unbalanced modeling (`LfAsym*` classes) and full node-breaker topology with connectivity traversal (`NodeBreakerTraverser`).
- Uses `powsybl-math`'s `LUDecomposition`/`MatrixFactory` abstraction; native KLU via JNI is the primary path (same library gridoxide's own `Klu` backend vendors directly).

## Where gridoxide already exceeds or matches

- **Asymmetric power flow**: already solving and tested (matches PGM/powsybl; lightsim2grid doesn't have this at all).
- **3-winding transformers**: already have a passing fixture (matches PGM/powsybl; lightsim2grid doesn't have this at all).
- **Factorization reuse across repeated solves** (`PersistentSolver`): conceptually identical to what lightsim2grid's `ContingencyAnalysis`/`TimeSerie` and PGM's batch-calculation path rely on — gridoxide has the prerequisite infrastructure already built, just not a batch/contingency-analysis API layered on top of it yet.
- **Block-sparse LU backend granularity**: matches PGM's own per-bus 2×2 block design, and (per this session's investigation) gridoxide's `faer`-backed solve handles pivots PGM's own hand-rolled solver refuses (no pivot perturbation) on the same real transmission-scale data.

## Identified gaps, ranked by how often reference tools flag them as important

1. **Q-limit enforcement / PV→PQ switching** — every one of the three either has it, half-has it, or explicitly disclaims *not* having it as a known limitation. gridoxide's `Bus` already carried `q_min`/`q_max`, unused. **Done** — `solver::newton_raphson_enforcing_q_limits` implements the standard MATPOWER-style one-directional PV→PQ switching outer loop, tested in `tests/q_limits_test.rs` across all three Jacobian backends; `PgmVoltageRegulator` now parses PGM's own `q_min`/`q_max` fields. Opt-in: plain `newton_raphson`/`PersistentSolver::solve` are unchanged, so no existing test/benchmark behavior shifted.
2. **Contingency/N-1 batch analysis** — the one place gridoxide is architecturally ahead of schedule; `PersistentSolver` is exactly the prerequisite lightsim2grid and powsybl build this on.
3. **Distributed slack** — 2 of 3 tools have it; real transmission grids often split slack across several generators.
4. **DC power flow as a first-class mode** — cheap, since `linear_initial_guess` is most of the way there already; every reference tool treats this as a basic offering.
5. Everything else in the table (switches/node-breaker, HVDC, SVC, short-circuit, state estimation, sensitivity/OPF, outer-loop architecture as a general extensibility mechanism) — real capabilities elsewhere, but either a materially larger undertaking or outside gridoxide's current scope as a focused AC power-flow library.

## Note on realistic benchmark coverage

`scripts/bench/matpower_to_pgm.py` doesn't currently populate `voltage_regulator.q_min`/`q_max` from
MATPOWER's `gen` matrix `Qmax`/`Qmin` columns, so none of the 12 real benchmark cases exercise
`newton_raphson_enforcing_q_limits` yet (every converted PV bus has unbounded `q_min`/`q_max`,
matching the previous behavior exactly). Wiring that up is a natural, separate follow-up if
real-grid Q-limit enforcement wants exercising end-to-end, not done as part of this change.
