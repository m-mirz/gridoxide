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

**Scope of the most recent revision.** gridoxide's own column was re-verified against current source
(every cell claiming support names the function or type implementing it), and the new
"CGMES / CIM import" row was checked across all five comparison tools by counting CGMES/CIM-named
files in their installed trees. The other five tools' cells in every *pre-existing* row were **not**
re-surveyed and are carried over from the previous revision — treat them as the older snapshot. The
new "Multi-island" row marks the four tools not checked as `not surveyed` rather than `❌`, since
absence of a survey is not evidence of absence of the feature.

## Summary table

| Feature | lightsim2grid | power-grid-model | powsybl-open-loadflow | VeraGrid | pandapower | **gridoxide (today)** |
|---|---|---|---|---|---|---|
| AC power flow (Newton-Raphson) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DC / linear power flow | ✅ | ✅ ("linear" mode) | ✅ | ✅ (`SolverType.Linear`/`LACPF`) | ✅ (`rundcpp`) | ⚠️ only as an internal initial guess (`network::linear_initial_guess`), not a standalone mode |
| Gauss-Seidel | ✅ (+ "synch" variant) | ❌ | ❌ | ✅ (`SolverType.GAUSS`) | ✅ (`algorithm="gs"`) | ❌ |
| Fast-decoupled (XB/BX) | ✅ | ❌ | ✅ | ✅ (`SolverType.FASTDECOUPLED` — one generic variant, not confirmed as a separate XB/BX split) | ✅ explicit `"fdbx"`/`"fdxb"` split (`pypower/fdpf.py`) | ❌ |
| **Q-limit enforcement (PV→PQ switching)** | ❌ explicitly disclaimed | ⚠️ stubbed, "not yet fully implemented" | ✅ `ReactiveLimitsOuterLoop`, incl. capability curves | ✅ `PowerFlowOptions.control_q` | ✅ `enforce_q_lims` (NR algorithm only, per its own docstring) | ✅ `solver::newton_raphson_enforcing_q_limits` (opt-in; plain `newton_raphson` still ignores `q_min`/`q_max`) |
| Distributed slack (multi-bus) | ✅ | ❌ | ✅ + area-interchange control | ✅ `PowerFlowOptions.distributed_slack` | ✅ `distributed_slack` + per-generator `slack_weight` | ❌ single slack only |
| **Remote** voltage control (controller regulates a *different* bus) | ❌ | ❌ | ✅ `VoltageControl.controlledBus` ≠ controller's own bus | ✅ `control_remote_voltage`: controlled bus → `PQV` mode, controller bus → `P` mode (`Compilers/circuit_to_data.py::set_bus_control_voltage`) | ❌ no built-in equivalent found in `control/` | ⚠️ CGMES import only, and static: `RegulatingControl.Terminal` resolves to the controlled bus, which is pinned to `PV` at the target (`src/cgmes.rs`, both `SynchronousMachine` and `StaticVarCompensator`). No control loop — the assignment happens once at import |
| **Shared** voltage control (several controllers, one controlled bus) | ❌ | ❌ | ✅ genuine reactive dispatch *inside* the Newton system: `DISTR_Q` equations `0 = qPercent_i·Σ_j q_j − q_i`, one per controller, so n controllers add n−1 equations alongside the single `BUS_TARGET_V`. Split keys come from explicit per-generator reactive keys, falling back to Qmax-range-proportional, then uniform (`Control::createReactiveKeys`); recomputed when a controller is disabled, e.g. by the reactive-limits outer loop | ⚠️ not really: `set_bus_control_voltage` tracks `bus_voltage_used` and logs "Different control voltage set points" on conflict. Its `qshare_per_bus` is a per-bus dispatch of that bus's own aggregate Q across its own devices, `(Q_limited − Qmin)/Qrange` — not a cross-bus split among several controllers of one remote bus | ❌ | ❌ last writer wins: each regulating machine overwrites `voltage_mag`/`q_min`/`q_max` on the controlled bus, so two controllers with different targets silently keep only the last, with no Q split and no conflict diagnostic |
| Transformer tap / phase-shifter auto-control | ❌ (fixed at init only) | ✅ `TapChangingStrategy` outer loop | ✅ several outer loops (voltage, reactive power, phase) | ✅ `control_taps_modules`/`control_taps_phase` options | ✅ `control.DiscreteTapControl`/`ContinuousTapControl` (`control/trafo_control.py`) | ❌ static taps only |
| 3-winding transformers | ❌ absent | ✅ (star-equivalent via 2 legs) | ✅ | ✅ (`Devices/transformer3w.py`, plus a generic N-winding `transformerNw.py`) | ✅ (`create_transformer3w`) | ✅ (already have a passing test fixture) |
| Switches / node-breaker topology | ❌ (TODO in source) | ⚠️ implicit via `from_status`/`to_status`, no discrete switch component | ✅ full node-breaker + `NodeBreakerTraverser` | ✅ (`Devices/Branches/switch.py`; CIM/IIDM importers also read node-breaker topology directly) | ✅ (`create_switch`/`create_switches` — bus-bus, bus-line, bus-trafo; core, not bolted-on, to pandapower's own topology model) | ⚠️ consumed, not modeled: `cgmes::merge_closed_switches` union-finds buses across closed `Breaker`/`Disconnector`/`LoadBreakSwitch`/`Fuse`/`Jumper`/`Cut`/`GroundDisconnector`/`DisconnectingCircuitBreaker`, honoring `open` + `inService`. No switch element in gridoxide's own network model, so switching state can't be changed between solves |
| HVDC | ✅ DC lines | ❌ | ✅ VSC/LCC | ✅ (`hvdc_line.py`, `vsc.py`) + UPFC (`upfc.py`) | ✅ `create_dcline` (lossy point-to-point) + `create_vsc`/`create_vsc_stacked`/`create_vsc_bipolar` | ✅ `src/dc.rs`: a real DC-side network (`DcBus`/`DcLine`, `solve_dc_network`) with `VsConverter`/`CsConverter` converters and converter losses, resolved by `cgmes_resolve_dc_converters` into AC-side injections. Only reachable via CGMES import — no HVDC element in the PGM-JSON or native-JSON paths |
| SVC (static var compensator) | ❌ | ❌ | ✅ | ✅ (`ControllableShunt`: stepped `Bmin`/`Bmax` regulating a `control_bus`'s voltage to `Vset`) | ✅ `create_svc` + `create_tcsc` (thyristor-controlled series capacitor) + `create_ssc` (static synchronous compensator) — broadest FACTS-device coverage of the six | ⚠️ CGMES `StaticVarCompensator` only: voltage-regulating (pins the controlled bus, incl. remote) when its `RegulatingControl` is voltage-mode and enabled, else a fixed Q injection. No `Bmin`/`Bmax` susceptance limits, no TCSC/SSC |
| Asymmetric / unbalanced power flow | ❌ symmetric only | ✅ | ✅ (`LfAsym*`) | ✅ (dedicated `Simulations/PowerFlow3ph/` driver) | ✅ (`runpp_3ph`) | ✅ (already solving, tested against PGM fixtures) |
| **CGMES / CIM import** | ❌ (0 CGMES/CIM-named files in its tree) | ❌ (0 CGMES/CIM-named files in its tree) | ✅ native, the reference implementation here | ✅ (48 CGMES/CIM-named files) | ✅ `converter/cim` (54 CGMES/CIM-named files) | ✅ EQ/EQBD/SSH/TP/SV profile merge by mRID, node-breaker reduction, ratio + all four phase-tap-changer flavors, 3-winding star resolution, HVDC, SVC, `ExternalNetworkInjection`, `EquivalentInjection`/`EquivalentBranch`, conform/non-conform loads, linear + nonlinear shunts, `AsynchronousMachine`, `PowerElectronicsConnection`. 14 fixture test files; benchmarked against pypowsybl on 8 conformance configurations (`scripts/bench/README.md` §6) |
| **Multi-island / disconnected components** | not surveyed | not surveyed | ⚠️ `connected_component_mode=MAIN` solves the largest component, drops the rest (verified directly — it is why pypowsybl's bus counts run below gridoxide's on every CGMES fixture) | not surveyed | not surveyed | ✅ every connected component solved in one call with a per-island `IslandReport`/`IslandStatus` (`Converged`/`MaxIterationsReached`/`Singular`/`NoReferenceBus`/`AmbiguousReferenceBus`); sourceless islands get a zero-voltage placeholder rather than an error |
| **Contingency / N-1 batch analysis** | ✅ `ContingencyAnalysis`, reuses factorization, ~20x speedup claimed | ❌ | ✅ + Woodbury fast-DC path | ✅ linear *and* nonlinear (full AC) contingency analysis, a HELM-based variant, SRAP support, and a time-series variant | ✅ `contingency` module, with a `run_contingency_ls2g` variant that offloads the actual solves to lightsim2grid for speed | ❌ — `batch::BatchSolver` solves the batch shape (see the row below), but `Scenario::branch_outages` is a declared seam that returns `BatchError::OutagesUnsupported`: an outage changes the Y-bus, and therefore the one sparsity pattern the batch's shared symbolic factorization is built around |
| Time-series / batch injections | ✅ `TimeSerie`, ~13x speedup claimed | ✅ batch datasets, parallel via `threading` param | — | ✅ time-series variants of power flow, OPF, linear analysis, *and* contingency analysis | ✅ `timeseries` module (`run_time_series`, pluggable `DataSource`/`OutputWriter`) | ⚠️ `batch::BatchSolver` (`src/batch.rs`): many scenarios over one shared topology, parallel across cores via rayon, each worker amortizing one symbolic factorization over its share — 3.5x on 8 physical cores at 256 scenarios (`scripts/bench/README.md` §4b), results identical to a sequential loop and returned in scenario order. Injection overrides only (`BusOverride` deliberately cannot change `bus_type`, since that changes `n_unknowns` and invalidates the shared pattern), and no time-series driver layered on top — no `DataSource`/`OutputWriter` equivalent, no result writer |
| Input validation | ❌ | ✅ `validate_input_data`/`validate_batch_data` | — | ❌ no generic equivalent found (only format-specific CIM/FMU import validation) | ✅ `diagnostic()` (disconnected elements, implausible values, wrong reference system, ...) | ❌ |
| Short-circuit calculation | ❌ | ✅ (IEC 60909) | ❌ | ✅ (3-phase, LG, LL, LLG fault types — `Simulations/ShortCircuitStudies/`) | ✅ (IEC 60909-style, `shortcircuit` module) | ❌ |
| State estimation | ❌ | ✅ (WLS, **sym + asym**, iterative-linear + Newton-Raphson, voltage/power/**current** sensors, **batched** with topology caching and thread-parallelism; no bad-data detection) | ❌ | ✅ (WLS + observability analysis + pseudo-measurement augmentation) | ✅ (WLS, `estimation` module) | ⚠️ **symmetric, single-shot only** (WLS + observability + bad-data detection + zero-injection constraints, both PGM calculation methods; reads asymmetric sensors but reduces them to the symmetric problem) — see the note below |
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
  `bench_lightsim2grid.py`'s and lightsim2grid's own `init_from_pandapower`) — see [Backends and Factorization Reuse](../solvers/backends.md)
  and `scripts/bench/README.md` for its own timing numbers, where it's the
  only tool besides gridoxide to converge on all 12 real MATPOWER cases.

## Where gridoxide already exceeds or matches

- **Asymmetric power flow**: already solving and tested (matches PGM/powsybl/VeraGrid/pandapower; lightsim2grid doesn't have this at all).
- **3-winding transformers**: already have a passing fixture (matches PGM/powsybl/VeraGrid/pandapower; lightsim2grid doesn't have this at all).
- **Factorization reuse across repeated solves** (`PersistentSolver`): conceptually identical to what lightsim2grid's `ContingencyAnalysis`/`TimeSerie` and PGM's batch-calculation path rely on.
- **Batched solving over one topology** (`batch::BatchSolver`): the API layered on top of that reuse, and the shape time-series/QSTS and Monte Carlo runs actually need — thousands of independent scenarios over an unchanging topology, spread across cores on rayon's own pool, one cached symbolic factorization per worker. Matches PGM's batch-calculation path and lightsim2grid's `TimeSerie` on the injection-scenario case; still short of both on contingency, which needs per-scenario topology (see gap 2 below). (`bde::solve_batch_block_diagonal` stacks a batch into one block-diagonal factorization instead, validated bit-exact against independent per-scenario solves in `scripts/bench/README.md` §4d — but it is ~2.7x *slower* on a CPU and exists to validate a future GPU path's architecture, so it is not a batching capability this table should credit.)
- **Block-sparse LU backend granularity**: matches PGM's own per-bus 2×2 block design, and gridoxide's `faer`-backed solve handles pivots PGM's own hand-rolled solver refuses (no pivot perturbation) on the same real transmission-scale data — still true after the converter fixes below, which changed PGM's input but not its failure pattern (same 6 `SparseMatrixError` / 4 `IterationDiverge` cases as before).
- **Sparse-solver breadth**: five backends (`Scalar`/`Block`/`Klu`/`KluNative`/`Pardiso` — the count previously read "four" while listing five) already exceeds VeraGrid's and pandapower's single fixed-solver paths, though it's still short of lightsim2grid's runtime-pluggable KLU/Eigen/NICSLU/CKTSO selection.
- **CGMES import depth**: one of four tools here with any CGMES/CIM import at all, and the only one of those four that is otherwise a focused AC power-flow library rather than a general-purpose framework. On the 8 conformance configurations benchmarked in `scripts/bench/README.md` §6 it is faster than pypowsybl on every fixture where both actually solve, and solves `MicroGrid-Type2-HVDC-MAS`, which pypowsybl declines to attempt (`iteration_count=0`, "Network has no generator with voltage control enabled").
- **Multi-island solving**: solves every connected component with per-island status rather than only the main one.
- **Solution verification tooling**: `scripts/bench/check_matpower_residual.py` checks a solved case against the MATPOWER file's *own* power-flow equations, and `check_cgmes_sv_consistency.py` checks a CGMES fixture's published `SvVoltage` against its own EQ/SSH data. Neither needs a second tool as a reference. This is a benchmark-harness capability, **not** an input-validation feature — it does not close the "Input validation" row above, which is about validating input before a solve (PGM's `validate_input_data`, pandapower's `diagnostic()`).

## Identified gaps, ranked by how often reference tools flag them as important

1. **Q-limit enforcement / PV→PQ switching** — every one of the five either has it, half-has it, or explicitly disclaims *not* having it as a known limitation. gridoxide's `Bus` already carried `q_min`/`q_max`, unused. **Done** — `solver::newton_raphson_enforcing_q_limits` implements the standard MATPOWER-style one-directional PV→PQ switching outer loop, tested in `tests/q_limits_test.rs` across all three Jacobian backends; `PgmVoltageRegulator` now parses PGM's own `q_min`/`q_max` fields. Opt-in: plain `newton_raphson`/`PersistentSolver::solve` are unchanged, so no existing test/benchmark behavior shifted.
2. **Contingency/N-1 batch analysis** — the one gap with most of its machinery already standing: `PersistentSolver`'s factorization reuse and `batch::BatchSolver`'s across-scenario parallelism are exactly what lightsim2grid, powsybl, VeraGrid (the most comprehensive, with linear, nonlinear, and HELM-based variants), and pandapower (which even offloads some of its own contingency solves to lightsim2grid for speed) build this on. What remains is the part the batch fast path deliberately excludes: a branch outage gives each scenario its *own* Y-bus and therefore its own sparsity pattern, so `Scenario::branch_outages` is currently a documented error rather than a solve.
3. **Distributed slack** — 4 of 5 tools have it (only lightsim2grid, powsybl, VeraGrid, and pandapower); real transmission grids often split slack across several generators.
4. **DC power flow as a first-class mode** — cheap, since `linear_initial_guess` is most of the way there already; every reference tool treats this as a basic offering.
5. **Switches, HVDC and SVC as first-class model elements** — no longer absent, but reachable *only* through CGMES import: switching state, converter setpoints and SVC regulation are all fixed at import time, with no element in gridoxide's own network model to change between solves. That is exactly the shape lightsim2grid's disclaimer calls out for its own fixed taps, and it is what stands between the current support and the contingency/time-series work in item 2.
6. Everything else in the table (TCSC/SSC and the wider FACTS set, short-circuit, sensitivity/OPF, outer-loop/controller architecture as a general extensibility mechanism, VeraGrid's unique EMT/RMS dynamic simulation, pandapower's protection-device modeling) — real capabilities elsewhere, but either a materially larger undertaking or outside gridoxide's current scope as a focused AC power-flow library.

## Note on state estimation

**Done for symmetric, single-shot estimation**, and within that scope gridoxide matches the most
capable reference tool and leads it in two places. Outside that scope it trails power-grid-model in
three, listed at the end of this note — the earlier version of this paragraph claimed parity
outright, which overstated it.

`se::nr::estimate` is Gauss-Newton on the normal equations, validated against
power-grid-model's own state-estimation fixtures (committed under `tests/data/pgm/state_estimation/`
with their MPL-2.0 license files): per-unit magnitudes agree to 1.5e-9 on `transmission-case`, and
every sparse backend produces the same answer, since the gain matrix is an ordinary square system.
See the [State Estimation](../state_estimation/index.md) chapter.

Both analyses VeraGrid is credited with above are present. Observability
(`se::observability::analyze`) separates structural from numerical unobservability and names the
buses and quantities involved, rather than only reporting that a factorization failed. Bad-data
detection (`se::bad_data::analyze`) runs the chi-squared test and identifies culprits by largest
normalized residual. Zero injections are enforced as hard equality constraints rather than as
high-weight pseudo-measurements — the approach that avoids the ill-conditioning power-grid-model has
two fixtures named after.

Both of power-grid-model's calculation methods are implemented and agree with each other:
Newton-Raphson (`se::nr`) and the prefactorized `iterative_linear` (`se::iterative`), selectable per
call. `link` is modelled now (stamped as a branch, see the
[zero-impedance](../powerflow/zero_impedance_branches.md) chapter), so the fixtures using one are
reachable. Pseudo-measurement augmentation — filling an
unobservable region with forecast values, which VeraGrid does — is not implemented; gridoxide reports
the unobservable set instead, which is the prerequisite for it.

### The two leads

- **Bad-data detection, which power-grid-model does not have at all.** Checked against its own
  documentation rather than assumed: it reports a per-sensor residual and stops there — no
  chi-squared test, no identification of a culprit. `se::bad_data::analyze` does both.
- **Newton-Raphson robustness.** power-grid-model's Newton-Raphson estimator raises
  `SparseMatrixError` on every benchmark case from 300 buses up, on documents its own
  iterative-linear method estimates from the same sensors without complaint, and that gridoxide's
  Newton-Raphson converges on to 1e-14. See `scripts/bench/README.md` §7.

### The three gaps

Measured against power-grid-model 1.13 (`references/power-grid-model/`), in order of how much they
matter:

1. **Asymmetric state estimation.** power-grid-model has `asym_voltage_sensor`, `asym_power_sensor`
   and `asym_current_sensor`, and estimates three-phase. gridoxide is symmetric-only — no asymmetric
   path anywhere in `src/se/`. This is the largest of the three, because unbalanced LV distribution
   is exactly the setting that needs it, and gridoxide already solves asymmetric *power flow* — so
   the gap is in the estimator, not in the network model underneath it.

   Narrower than it was: `measurement.rs` now reads `asym_voltage_sensor` and `asym_power_sensor`
   and reduces them to the symmetric problem the way power-grid-model's own `sym_calc_param` does —
   positive sequence for a phasor, the mean of the phases otherwise. So an asymmetric *document*
   estimates correctly today; what is missing is estimating the three phases as distinct unknowns.
2. **Current sensors.** power-grid-model 1.13 supports them symmetric and asymmetric, in local-angle
   and global-angle variants, with documented rules for mixing them with power sensors on a terminal.
   `MeasurementKind` has four variants — voltage magnitude, voltage angle, P, Q — and no current at
   all. Not an exotic sensor type: real RTUs frequently report a current magnitude rather than a
   power.
3. **Batch state estimation.** power-grid-model runs many scenarios through the same call with
   topology caching and thread-parallelism. `batch::BatchSolver` (`src/batch.rs`) has no estimation
   entry point at all — it is power-flow only. Production state estimation is a time series, so this
   is the distance between estimating a snapshot and running an estimator.

On speed, the iterative-linear method runs 1.6-2.0x behind power-grid-model's across an order of
magnitude of problem size (`scripts/bench/README.md` §7). Measured rather than inferred, that is
entirely an iteration-count gap: gridoxide's own iterations are 30-40% *cheaper* than
power-grid-model's and it takes about three times as many, and undamped its map does not converge at
all. See `docs/src/state_estimation/iterative.md`.

Two smaller gaps have closed. A sensor on a three-winding transformer side (`measured_terminal_type`
6/7/8) used to return `MeasurementError::UnsupportedTerminalType`; it now maps to the corresponding
leg's `From` terminal, since a three-winding transformer is already resolved into three two-winding
branches around a star bus. And the estimator no longer starts flat: `se::nr::linear_start` carries
the network's structural phase shifts, without which Gauss-Newton converges to a *different*
stationary point on any network containing a phase-shifting transformer — reporting success, with an
objective nine orders of magnitude worse than the true optimum.

One caveat on all of the above that is about evidence rather than features: every benchmark and
fixture here estimates from data that is either perfectly consistent or hand-authored. Nothing in
this repo generates realistically noisy or corrupted measurements, so gridoxide's bad-data
advantage — lead 1 — has never actually been measured against anything. Bad-data behaviour needs a
harness that does not exist yet.

## Note on realistic benchmark coverage

`gridoxide.matpower` (`python/gridoxide/matpower.py` — the conversion logic moved into the pip
package itself; `scripts/bench/matpower_to_pgm.py` is now only a thin CLI wrapper around it)
populates `voltage_regulator.q_min`/`q_max` from MATPOWER's
`gen` matrix `Qmax`/`Qmin` columns (summed across every active gen at a bus, matching how
`p_specified` is already summed). Confirmed against all 12 real benchmark cases: 11 of them have at
least one PV bus whose unconstrained Q genuinely exceeds its nameplate limit (from 4 violations on
the smallest case to 166 on `case3120sp`), and `newton_raphson_enforcing_q_limits` converges on
every one of them, including cases needing dozens of simultaneous PV→PQ switches across several
outer iterations. MATPOWER represents "no limit" as literal `+-Inf` on some real cases (e.g.
`case9241pegase`) — the converter omits the key entirely in that case rather than writing a
non-standard `Infinity` JSON token, matching PGM's own "unset means unbounded" convention exactly.
