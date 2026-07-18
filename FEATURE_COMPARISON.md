# Feature comparison: gridoxide vs. reference tools

A survey of five independent power-flow implementations — what each actually supports (verified
against source/docs, not assumed), compared against gridoxide's own current scope — used to decide
what gridoxide tackles next. Three (lightsim2grid, power-grid-model, powsybl-open-loadflow) are full
local checkouts under `references/` (itself gitignored, hence this file living at the repo root
instead); see each tool's own CLAUDE.md/README for how to consult them further. The other two,
[VeraGrid](https://github.com/SanPen/VeraGrid) (the GridCal successor) and
[pandapower](https://github.com/e2nIEE/pandapower) — both also used as comparison tools in
`scripts/bench/run_case_suite.py` — aren't checked out under `references/`; they're verified instead
by reading their installed packages' own source directly (`pip install VeraGridEngine pandapower`;
see each package's own directory structure for the file paths cited below). This file is a snapshot,
not a living document, and will drift as gridoxide and all five tools evolve.

## Summary table

| Feature | lightsim2grid | power-grid-model | powsybl-open-loadflow | VeraGrid | pandapower | **gridoxide (today)** |
|---|---|---|---|---|---|---|
| AC power flow (Newton-Raphson) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DC / linear power flow | ✅ | ✅ ("linear" mode) | ✅ | ✅ (`SolverType.Linear`/`LACPF`) | ✅ (`rundcpp`) | ⚠️ only as an internal initial guess (`network::linear_initial_guess`), not a standalone mode |
| Gauss-Seidel | ✅ (+ "synch" variant) | ❌ | ❌ | ✅ (`SolverType.GAUSS`) | ✅ (`algorithm="gs"`) | ❌ |
| Fast-decoupled (XB/BX) | ✅ | ❌ | ✅ | ✅ (`SolverType.FASTDECOUPLED` — one generic variant, not confirmed as a separate XB/BX split) | ✅ explicit `"fdbx"`/`"fdxb"` split (`pypower/fdpf.py`) | ❌ |
| **Q-limit enforcement (PV→PQ switching)** | ❌ explicitly disclaimed | ⚠️ stubbed, "not yet fully implemented" | ✅ `ReactiveLimitsOuterLoop`, incl. capability curves | ✅ `PowerFlowOptions.control_q` | ✅ `enforce_q_lims` (NR algorithm only, per its own docstring) | ✅ `solver::newton_raphson_enforcing_q_limits` (opt-in; plain `newton_raphson` still ignores `q_min`/`q_max`) |
| Distributed slack (multi-bus) | ✅ | ❌ | ✅ + area-interchange control | ✅ `PowerFlowOptions.distributed_slack` | ✅ `distributed_slack` + per-generator `slack_weight` | ❌ single slack only |
| Remote / shared voltage control | ❌ | ❌ | ✅ | ✅ `PowerFlowOptions.control_remote_voltage` | ❌ no built-in equivalent found in `control/` | ❌ |
| Transformer tap / phase-shifter auto-control | ❌ (fixed at init only) | ✅ `TapChangingStrategy` outer loop | ✅ several outer loops (voltage, reactive power, phase) | ✅ `control_taps_modules`/`control_taps_phase` options | ✅ `control.DiscreteTapControl`/`ContinuousTapControl` (`control/trafo_control.py`) | ❌ static taps only |
| 3-winding transformers | ❌ absent | ✅ (star-equivalent via 2 legs) | ✅ | ✅ (`Devices/transformer3w.py`, plus a generic N-winding `transformerNw.py`) | ✅ (`create_transformer3w`) | ✅ (already have a passing test fixture) |
| Switches / node-breaker topology | ❌ (TODO in source) | ⚠️ implicit via `from_status`/`to_status`, no discrete switch component | ✅ full node-breaker + `NodeBreakerTraverser` | ✅ (`Devices/Branches/switch.py`; CIM/IIDM importers also read node-breaker topology directly) | ✅ (`create_switch`/`create_switches` — bus-bus, bus-line, bus-trafo; core, not bolted-on, to pandapower's own topology model) | ❌ |
| HVDC | ✅ DC lines | ❌ | ✅ VSC/LCC | ✅ (`hvdc_line.py`, `vsc.py`) + UPFC (`upfc.py`) | ✅ `create_dcline` (lossy point-to-point) + `create_vsc`/`create_vsc_stacked`/`create_vsc_bipolar` | ❌ |
| SVC (static var compensator) | ❌ | ❌ | ✅ | ✅ (`ControllableShunt`: stepped `Bmin`/`Bmax` regulating a `control_bus`'s voltage to `Vset`) | ✅ `create_svc` + `create_tcsc` (thyristor-controlled series capacitor) + `create_ssc` (static synchronous compensator) — broadest FACTS-device coverage of the six | ❌ |
| Asymmetric / unbalanced power flow | ❌ symmetric only | ✅ | ✅ (`LfAsym*`) | ✅ (dedicated `Simulations/PowerFlow3ph/` driver) | ✅ (`runpp_3ph`) | ✅ (already solving, tested against PGM fixtures) |
| **Contingency / N-1 batch analysis** | ✅ `ContingencyAnalysis`, reuses factorization, ~20x speedup claimed | ❌ | ✅ + Woodbury fast-DC path | ✅ linear *and* nonlinear (full AC) contingency analysis, a HELM-based variant, SRAP support, and a time-series variant | ✅ `contingency` module, with a `run_contingency_ls2g` variant that offloads the actual solves to lightsim2grid for speed | ❌ (but `PersistentSolver`'s factorization reuse is the exact prerequisite, already built) |
| Time-series / batch injections | ✅ `TimeSerie`, ~13x speedup claimed | ✅ batch datasets, parallel via `threading` param | — | ✅ time-series variants of power flow, OPF, linear analysis, *and* contingency analysis | ✅ `timeseries` module (`run_time_series`, pluggable `DataSource`/`OutputWriter`) | ❌ (same prerequisite exists) |
| Input validation | ❌ | ✅ `validate_input_data`/`validate_batch_data` | — | ❌ no generic equivalent found (only format-specific CIM/FMU import validation) | ✅ `diagnostic()` (disconnected elements, implausible values, wrong reference system, ...) | ❌ |
| Short-circuit calculation | ❌ | ✅ (IEC 60909) | ❌ | ✅ (3-phase, LG, LL, LLG fault types — `Simulations/ShortCircuitStudies/`) | ✅ (IEC 60909-style, `shortcircuit` module) | ❌ |
| State estimation | ❌ | ✅ (WLS) | ❌ | ✅ (WLS + observability analysis + pseudo-measurement augmentation) | ✅ (WLS, `estimation` module) | ❌ |
| Sensitivity analysis / OPF | ❌ / ❌ | ❌ / ❌ | ✅ / ❌ | ✅ (PTDF/LODF, `Simulations/LinearFactors/`) / ✅ (linear *and* nonlinear AC OPF, `Simulations/OPF/`) | ✅ (PTDF, `pypower/makePTDF.py`) / ✅ native PDIPM AC+DC OPF (`runopp`/`rundcopp`) *plus* an optional external Julia PandaModels.jl bridge (`runpm.py`) for more advanced formulations | ❌ / ❌ |
| Pluggable "outer loop" architecture | ❌ | ⚠️ ad hoc (tap optimizer only) | ✅ extensively (14+ outer loops) | ⚠️ ad hoc (boolean control flags in `PowerFlowOptions`, not a modular/registry-based architecture like powsybl's) | ✅ genuine `Controller`/`BasicCtrl` base classes (`control/basic_controller.py`) registered on `net.controller` and driven by `run_control` — third-party code can subclass `Controller` directly, closer in spirit to powsybl's extensibility than to VeraGrid's/PGM's fixed flag sets, though not the same formal outer-loop-convergence architecture | ❌ |
| Dynamic / time-domain simulation (EMT, RMS, small-signal stability) | ❌ | ❌ | ❌ | ✅ (`Simulations/EMT/`, `Simulations/Rms/`, `Simulations/SmallSignalStabilityEmt/`+`SmallSignalStabilityRms/` — the only one of the six with this at all) | ❌ | ❌ |
| Sparse solver | KLU/Eigen/NICSLU/CKTSO, pluggable at runtime | hand-rolled 2×2-block LU, pivot perturbation off by default | KLU via JNI (primary path) | SciPy's SuperLU (`scipy.sparse.linalg._dsolve._superlu`), wrapped in a numba-JIT'd custom CSC type (`Utils/Sparse/csc2.py`) — not pluggable | SciPy's `spsolve` (`pypower/newtonpf.py`), with an optional `use_umfpack` flag — UMFPACK is a SuiteSparse sibling of KLU, when `scikit-umfpack` is installed | faer (`Scalar`) / hand-rolled 2×2-block LU (`Block`, matches PGM's own block granularity) / KLU (`Klu`) / from-scratch Rust KLU port (`KluNative`) / Intel oneMKL PARDISO (`Pardiso`) |

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

### VeraGrid (Python, [SanPen/VeraGrid](https://github.com/SanPen/VeraGrid), the GridCal successor)

- By far the broadest *simulation-category* scope of the five — installed as the headless
  `VeraGridEngine` package (not the Qt-GUI-bundled `VeraGrid` package), its `Simulations/` directory alone
  has 25+ top-level categories: beyond power flow, also OPF (linear + nonlinear AC), state estimation,
  short-circuit, contingency analysis, sensitivity (PTDF/LODF), continuation power flow (PV curves),
  stochastic/Monte Carlo analysis, reliability analysis, investment/expansion-planning evaluation,
  net/available transfer capacity (NTC/ATC), topology reduction, and — uniquely among all five — electro-
  magnetic-transient (EMT) and RMS time-domain dynamic simulation with small-signal stability analysis.
  pandapower rivals it in raw feature *count* (see below) but has no equivalent of EMT/RMS dynamic simulation
  at all; lightsim2grid/PGM/powsybl are themselves narrower, purpose-built power-flow-focused engines by
  comparison.
- Solvers: among the most pluggable of the five via `SolverType` — `NR`, Gauss-Seidel, Fast-decoupled,
  Levenberg-Marquardt, Iwamoto-NR, Powell's Dog Leg, HELM (Holomorphic Embedding), Decoupled-LU, plus linear/
  linear-AC modes and dedicated linear/nonlinear OPF solver types, all in one `PowerFlowOptions.solver_type`
  enum — though pandapower's own `algorithm` parameter is comparably broad and additionally offers a
  backward/forward-sweep solver neither VeraGrid nor any of the other four tools here have.
- Voltage/reactive/tap control is a set of independent boolean flags on `PowerFlowOptions`
  (`control_q`, `distributed_slack`, `control_remote_voltage`, `control_taps_modules`,
  `control_taps_phase`, `orthogonalize_controls`) applied inside the NR iteration
  itself, not a modular outer-loop registry the way powsybl's `OuterLoop` abstraction is — closer in spirit
  to power-grid-model's ad hoc tap optimizer than to powsybl's extensible architecture.
- Devices include 3-winding and generic N-winding transformers, switches (with CIM/IIDM node-breaker import),
  HVDC lines, VSC, and UPFC, plus a `ControllableShunt` device (stepped `Bmin`/`Bmax` regulating a bus's
  voltage to a setpoint) filling the SVC role neither lightsim2grid, PGM, nor powsybl have — broad FACTS
  coverage, though pandapower's own dedicated `create_svc`/`create_tcsc`/`create_ssc` set turns out broader
  still (see below).
- MATPOWER import (`parse_matpower_file`) reads each bus's `type` column and each generator's `Vg` setpoint
  directly, so genuine PV-bus modeling comes for free with no PGM-`voltage_regulator`-style conversion step —
  see `scripts/bench/bench_veragrid.py`.
- Its own numerical kernels are numba-JIT-compiled (first call per process pays a multi-second JIT-compilation
  cost unrelated to the power-flow algorithm itself — `scripts/bench/bench_veragrid.py`'s warm-up call
  absorbs this) and its sparse LU solve is SciPy's SuperLU (`scipy.sparse.linalg._dsolve._superlu.gstrf`,
  wrapped in a numba-jitted custom CSC type, `Utils/Sparse/csc2.py`) — not pluggable across multiple sparse
  backends the way lightsim2grid or powsybl are.
- On the 12-case real-MATPOWER benchmark (`scripts/bench/README.md`), converges on 9 of 12 (the same three
  hard RTE cases every tool but gridoxide/pandapower also fails on) and lands roughly on par with pypowsybl —
  markedly slower than the C/Rust-backed solvers here, consistent with being a general-purpose Python
  framework rather than one optimized around raw repeated-solve throughput.

### pandapower (Python, [e2nIEE/pandapower](https://github.com/e2nIEE/pandapower))

- Also very broad in scope, though — unlike VeraGrid's from-scratch simulation engines — pandapower's own
  numerical power-flow/OPF path is largely a thin, numba-accelerated wrapper around PYPOWER
  (`pandapower/pypower/`, itself a Python port of MATPOWER), with pandapower supplying the richer network
  model (switches, controllers, 3-winding transformers, FACTS devices) and everything else (contingency,
  timeseries, estimation, shortcircuit, diagnostic) as sibling top-level packages built on top of that core.
- Solvers (`runpp(algorithm=...)`): `"nr"` (default, PYPOWER's Newton-Raphson, numba-accelerated), Iwamoto-NR
  ("maybe slower... but more robust" per its own docstring), backward/forward sweep (`"bfsw"`, specially
  suited to radial/weakly-meshed networks — a solver category none of the other five tools here offer),
  Gauss-Seidel, and *two* explicitly separate fast-decoupled variants (`"fdbx"`/`"fdxb"`), plus HELM.
- Switches (`create_switch`/`create_switches`, bus-bus/bus-line/bus-trafo) are core to how pandapower
  represents topology at all, not a bolted-on extra the way they are for some other tools here — closest in
  spirit to powsybl's node-breaker model among the tools surveyed.
- FACTS-device coverage (`create_svc`/`create_tcsc`/`create_ssc`/`create_vsc*`) is the broadest of the six
  tools surveyed, including a thyristor-controlled series capacitor (TCSC) none of the others model.
- The generic `Controller`/`BasicCtrl` framework (`control/basic_controller.py`, driven by `run_control=True`)
  is genuinely extensible — any third-party code can subclass `Controller` and register it on `net.controller`
  — closer to powsybl's outer-loop extensibility in spirit than to PGM's/VeraGrid's fixed option flags, even
  though the underlying convergence-loop architecture isn't identical.
- `contingency` module includes a `run_contingency_ls2g` variant that offloads the actual repeated solves to
  lightsim2grid for speed — a real cross-tool dependency between two of the tools surveyed here, not just a
  coincidental feature overlap.
- OPF is two-tiered: a native, no-external-dependency PDIPM-based AC/DC OPF (`runopp`/`rundcopp`, inherited
  from PYPOWER) for standard formulations, plus an optional bridge to Julia's PandaModels.jl (`runpm.py`) for
  more advanced formulations (storage, multi-stage, etc.) when that external toolchain is installed.
- `diagnostic()` (`diagnostic/diagnostic_helpers.py`) is a real, generic input-validation/consistency-check
  function (disconnected elements, implausible parameter values, wrong reference system, ...) — closer to
  PGM's `validate_input_data` than to VeraGrid's format-specific-only import validation.
- Also has a dedicated `protection` package (protection-device/relay-coordination modeling) that none of the
  other five tools here have any equivalent of — outside the scope of this table's rows, but worth noting as
  another area where pandapower's breadth exceeds a pure power-flow-engine comparison.
- This is the same pandapower already used elsewhere in this benchmark suite (`bench_pandapower.py`,
  `bench_lightsim2grid.py`'s and lightsim2grid's own `init_from_pandapower`) — see the top-level README's
  "Experimental backends" section and `scripts/bench/README.md` for its own timing numbers, where it's the
  only tool besides gridoxide to converge on all 12 real MATPOWER cases.

## Where gridoxide already exceeds or matches

- **Asymmetric power flow**: already solving and tested (matches PGM/powsybl/VeraGrid/pandapower; lightsim2grid doesn't have this at all).
- **3-winding transformers**: already have a passing fixture (matches PGM/powsybl/VeraGrid/pandapower; lightsim2grid doesn't have this at all).
- **Factorization reuse across repeated solves** (`PersistentSolver`): conceptually identical to what lightsim2grid's `ContingencyAnalysis`/`TimeSerie` and PGM's batch-calculation path rely on — gridoxide has the prerequisite infrastructure already built, just not a batch/contingency-analysis API layered on top of it yet.
- **Block-sparse LU backend granularity**: matches PGM's own per-bus 2×2 block design, and (per this session's investigation) gridoxide's `faer`-backed solve handles pivots PGM's own hand-rolled solver refuses (no pivot perturbation) on the same real transmission-scale data.
- **Sparse-solver breadth**: four backends (`Scalar`/`Block`/`Klu`/`KluNative`/`Pardiso`) already exceeds VeraGrid's and pandapower's single fixed-solver paths, though it's still short of lightsim2grid's runtime-pluggable KLU/Eigen/NICSLU/CKTSO selection.

## Identified gaps, ranked by how often reference tools flag them as important

1. **Q-limit enforcement / PV→PQ switching** — every one of the five either has it, half-has it, or explicitly disclaims *not* having it as a known limitation. gridoxide's `Bus` already carried `q_min`/`q_max`, unused. **Done** — `solver::newton_raphson_enforcing_q_limits` implements the standard MATPOWER-style one-directional PV→PQ switching outer loop, tested in `tests/q_limits_test.rs` across all three Jacobian backends; `PgmVoltageRegulator` now parses PGM's own `q_min`/`q_max` fields. Opt-in: plain `newton_raphson`/`PersistentSolver::solve` are unchanged, so no existing test/benchmark behavior shifted.
2. **Contingency/N-1 batch analysis** — the one place gridoxide is architecturally ahead of schedule; `PersistentSolver` is exactly the prerequisite lightsim2grid, powsybl, VeraGrid (the most comprehensive, with linear, nonlinear, and HELM-based variants), and pandapower (which even offloads some of its own contingency solves to lightsim2grid for speed) all build this on.
3. **Distributed slack** — 4 of 5 tools have it (only lightsim2grid, powsybl, VeraGrid, and pandapower); real transmission grids often split slack across several generators.
4. **DC power flow as a first-class mode** — cheap, since `linear_initial_guess` is most of the way there already; every reference tool treats this as a basic offering.
5. Everything else in the table (switches/node-breaker, HVDC, SVC/TCSC/SSC, short-circuit, state estimation, sensitivity/OPF, outer-loop/controller architecture as a general extensibility mechanism, VeraGrid's unique EMT/RMS dynamic simulation, pandapower's protection-device modeling) — real capabilities elsewhere, but either a materially larger undertaking or outside gridoxide's current scope as a focused AC power-flow library.

## Note on realistic benchmark coverage

`scripts/bench/matpower_to_pgm.py` now populates `voltage_regulator.q_min`/`q_max` from MATPOWER's
`gen` matrix `Qmax`/`Qmin` columns (summed across every active gen at a bus, matching how
`p_specified` is already summed). Confirmed against all 12 real benchmark cases: 11 of them have at
least one PV bus whose unconstrained Q genuinely exceeds its nameplate limit (from 4 violations on
the smallest case to 166 on `case3120sp`), and `newton_raphson_enforcing_q_limits` converges on
every one of them, including cases needing dozens of simultaneous PV→PQ switches across several
outer iterations. MATPOWER represents "no limit" as literal `+-Inf` on some real cases (e.g.
`case9241pegase`) — the converter omits the key entirely in that case rather than writing a
non-standard `Infinity` JSON token, matching PGM's own "unset means unbounded" convention exactly.
