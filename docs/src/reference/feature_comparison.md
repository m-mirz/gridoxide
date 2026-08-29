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

**Scope of the most recent revision.** Six rows were **added** for analysis types the table had no
row for at all — remedial action optimization, voltage stability, harmonics, reliability, protection
coordination, and input-format coverage — and the single "dynamic simulation" row was **split into
three**, because RMS, EMT and small-signal analysis are three different machineries and one
checkbox understated the gap. Absence of a row is how a gap goes untracked; that was the point of
the exercise.

For the added rows, the three tools checked out under `references/` (lightsim2grid,
power-grid-model, powsybl-open-loadflow) were searched directly, and each `❌` below names what was
searched for. **VeraGrid and pandapower could not be re-surveyed** — the previous revision read
their installed packages and neither is installed here any more — so they are marked `not surveyed`
rather than `❌`, per the convention the "Multi-island" row introduced: absence of a survey is not
evidence of absence of a feature. Their cells in the three *split* dynamics rows carry forward the
directory names the previous survey recorded, which is evidence rather than inference.

**Earlier revisions.** gridoxide's own column was re-verified against current source (every cell
claiming support names the function or type implementing it), and the "CGMES / CIM import" row was
checked across all five comparison tools by counting CGMES/CIM-named files in their installed trees.
The other five tools' cells in every row predating this revision were **not** re-surveyed — treat
them as the older snapshot.

## Summary table

| Feature | lightsim2grid | power-grid-model | powsybl-open-loadflow | VeraGrid | pandapower | **gridoxide (today)** |
|---|---|---|---|---|---|---|
| AC power flow (Newton-Raphson) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DC (Bθ) power flow | ✅ | ❌ | ✅ (two approximation variants) | ✅ (`SolverType.DC`) | ✅ (`rundcpp`) | ✅ `linear::btheta::dc_power_flow`, both approximation variants |
| Constant-admittance linear power flow | ❌ | ✅ (`CalculationMethod.linear`, + `linear_current`) | ❌ | ✅ (`SolverType.LACPF`) | ❌ | ✅ `linear::impedance::linear_power_flow` |
| Gauss-Seidel | ✅ (+ "synch" variant) | ❌ | ❌ | ✅ (`SolverType.GAUSS`) | ✅ (`algorithm="gs"`) | ❌ |
| Fast-decoupled (XB/BX) | ✅ | ❌ | ✅ | ✅ (`SolverType.FASTDECOUPLED` — one generic variant, not confirmed as a separate XB/BX split) | ✅ explicit `"fdbx"`/`"fdxb"` split (`pypower/fdpf.py`) | ❌ |
| **Q-limit enforcement (PV→PQ switching)** | ❌ explicitly disclaimed | ⚠️ stubbed, "not yet fully implemented" | ✅ `ReactiveLimitsOuterLoop`, incl. capability curves | ✅ `PowerFlowOptions.control_q` | ✅ `enforce_q_lims` (NR algorithm only, per its own docstring) | ✅ `outerloop::ReactiveLimits` (opt-in; plain `newton_raphson` still ignores `q_min`/`q_max`), composable with distributed slack and tap control — see [Outer Loops](../powerflow/outer_loops.md) |
| Distributed slack (multi-bus) | ✅ | ❌ | ✅ + area-interchange control | ✅ `PowerFlowOptions.distributed_slack` | ✅ `distributed_slack` + per-generator `slack_weight` | ✅ `outerloop::DistributedSlack` with `SlackDistribution` — per-bus participation weights (`uniform` over `Slack`+`PV`, or explicit, normalized **per island** so a disconnected network sizes each island's correction by its own generators). Reports the shift per bus, the residual per island, and names any island left on a single slack rather than skipping it silently. Bus types do not change, so `PersistentSolver` keeps its symbolic factorization across the whole outer loop — unlike the Q-limit loop, which must reset. Area-interchange control too: `outerloop::AreaInterchange` holds each control area's net export at a scheduled value, with the slack's own area as the dependent one — the system is over-determined by exactly one, because both ends of a tie are measured into the branch and so sum to its losses rather than cancelling. It generalizes distributed slack (one area, zero target, *is* distributed slack — asserted by running both and comparing shifts) and the two may not both be configured. All three importers supply areas: CGMES from `ControlArea`/`TieFlow` with membership derived by cutting the tie branches, IIDM from `<iidm:area>`'s `<voltageLevelRef>` directly, UCTE from its `##Z` country codes. CGMES and IIDM also state the schedule (`netInterchange`/`interchangeTarget`, both negated — both state an import). `gridoxide solve --area-interchange`. See [Distributed Slack](../powerflow/distributed_slack.md) and [Area Interchange Control](../powerflow/area_interchange.md) |
| **Remote** voltage control (controller regulates a *different* bus) | ❌ | ❌ | ✅ `VoltageControl.controlledBus` ≠ controller's own bus | ✅ `control_remote_voltage`: controlled bus → `PQV` mode, controller bus → `P` mode (`Compilers/circuit_to_data.py::set_bus_control_voltage`) | ❌ no built-in equivalent found in `control/` | ⚠️ CGMES import only, but no longer static: `RegulatingControl.Terminal` resolves to the controlled bus, and `outerloop::RemoteVoltageControl` (`--control-remote-voltage`) then moves the control onto the machine — the controller bus becomes `PV` and its setpoint is driven until the controlled bus reaches target, so the reactive power is produced where the machine is and the machine's *own* capability is what bounds it. Reaches the same fixed point as the exact formulation (fix `|V|` at the controlled bus, free `Q` at the controller) without making the unknown layout depend on more than `BusType`; the test suite checks the fixed point directly. Opt-in, because it changes answers. Without it the old behaviour stands: the controlled bus is pinned and the reactive power appears there instead. Not yet: several machines at *different* buses holding one bus, which needs `DISTR_Q` and which no vendored fixture has |
| **Shared** voltage control (several controllers, one controlled bus) | ❌ | ❌ | ✅ genuine reactive dispatch *inside* the Newton system: `DISTR_Q` equations `0 = qPercent_i·Σ_j q_j − q_i`, one per controller, so n controllers add n−1 equations alongside the single `BUS_TARGET_V`. Split keys come from explicit per-generator reactive keys, falling back to Qmax-range-proportional, then uniform (`Control::createReactiveKeys`); recomputed when a controller is disabled, e.g. by the reactive-limits outer loop | ⚠️ not really: `set_bus_control_voltage` tracks `bus_voltage_used` and logs "Different control voltage set points" on conflict. Its `qshare_per_bus` is a per-bus dispatch of that bus's own aggregate Q across its own devices, `(Q_limited − Qmin)/Qrange` — not a cross-bus split among several controllers of one remote bus | ❌ | ⚠️ **capability and attribution yes, in-solver dispatch no.** Reactive limits **sum** across every machine holding one bus (`cgmes::VoltageControl`), which is what the injections beside them always did — until that was fixed they were overwritten, so a bus held by six machines carried one machine's nameplate. Not rare: 62 of RealGrid's 417 voltage-regulated nodes, 2 of FullGrid's 3. A target two controllers disagree on is reported through `VoltageControlReport::target_conflicts` rather than silently resolved — no vendored fixture has one, which is why the last writer still wins there and any rule would be untested. `src/dispatch.rs` now **attributes** each bus's solved reactive power to the individual machines holding it (`gridoxide solve --dispatch`, `PowerFlowModel.machine_dispatch()`) — by explicit key, else capability-proportional, else uniform, with clamp-and-redistribute against each machine's own range, checked against RealGrid's own published `SvPowerFlow` per terminal (all 496 machines; the split rule lands within 7.4% on the 62 shared buses, against a 7.5% solve-vs-published backdrop). It changes no answer: **every** shared-control case in the vendored corpus is machines at the bus they regulate, which is an ordinary `PV` bus with one injection, so the split is arithmetic on the result. What is genuinely absent is the **in-solver** dispatch: powsybl solves the *remote* case inside the Newton system with n−1 `DISTR_Q` equations, needed when controllers sit at different buses — a configuration no vendored fixture has, so it is deliberately unbuilt (`plans/REACTIVE_DISPATCH_PLAN.md` §2). The attribution also measured a previously invisible simplification: reactive limits bound the bus's *net* injection while the limits describe the machines' *own* capability, so on 6 of RealGrid's 416 regulated buses a co-located load saturates the machine before the clamp fires. The tap-side equivalent *is* built — `outerloop` sizes several tap controllers on one bus against each other — but taps sit outside the Newton system, which is what makes that the cheaper half |
| Transformer tap / phase-shifter auto-control | ❌ (fixed at init only) | ✅ `TapChangingStrategy` outer loop | ✅ several outer loops (voltage, reactive power, phase) | ✅ `control_taps_modules`/`control_taps_phase` options | ✅ `control.DiscreteTapControl`/`ContinuousTapControl` (`control/trafo_control.py`) | ✅ `outerloop::TransformerVoltageControl` and `PhaseControl` — powsybl's *incremental* strategy, stepping the discrete position by \\(\\Delta\\rho = (V^{target} - V_c)/(\\partial V_c/\\partial\\rho)\\) from `ac_sensitivity`, with a deadband, an insensitivity filter and a direction-change budget. Several controllers on one bus (or one regulated flow) are sized against each other rather than each against the whole deviation. CGMES supplies both halves — every position of all four `PhaseTapChanger` flavours plus `RatioTapChangerTable`, and `TapChangerControl` in `voltage` and `activePower` modes; IIDM and UCTE supply theirs. On Svedala all eleven controllers go from 2–5% outside their deadbands to inside. Not the two continuous-then-round strategies, not `reactivePower` mode, not shunt-section control, and not inside a contingency or batch sweep — see [Transformer Tap Control](../powerflow/tap_control.md) |
| 3-winding transformers | ❌ absent | ✅ (star-equivalent via 2 legs) | ✅ | ✅ (`Devices/transformer3w.py`, plus a generic N-winding `transformerNw.py`) | ✅ (`create_transformer3w`) | ✅ (already have a passing test fixture) |
| Switches / node-breaker topology | ❌ (TODO in source) | ⚠️ implicit via `from_status`/`to_status`, no discrete switch component | ✅ full node-breaker + `NodeBreakerTraverser` | ✅ (`Devices/Branches/switch.py`; CIM/IIDM importers also read node-breaker topology directly) | ✅ (`create_switch`/`create_switches` — bus-bus, bus-line, bus-trafo; core, not bolted-on, to pandapower's own topology model) | ✅ `topology::NodeBreakerTopology` over CGMES `ConnectivityNode`s and nine switch classes plus `Junction`, with a per-switch `RetentionPolicy` (merge all / busbar-adjacent / by kind / explicit / retain all) selecting the view. A retained switch is a first-class element: own mRID, own flow, `set_switch_open` between solves, own LODF column (= its bus-split factor). Position is carried as terminal status, so switching does not move the sparsity pattern and `PersistentSolver` keeps its symbolic factorization. Verified on MiniGrid/SmallGrid/Svedala under every policy, up to 1,464 retained switches, at unchanged iteration counts. State estimation cannot yet consume the node-breaker view. See [Node-Breaker Topology](../cgmes/node_breaker.md) |
| HVDC | ✅ DC lines | ❌ | ✅ VSC/LCC | ✅ (`hvdc_line.py`, `vsc.py`) + UPFC (`upfc.py`) | ✅ `create_dcline` (lossy point-to-point) + `create_vsc`/`create_vsc_stacked`/`create_vsc_bipolar` | ✅ `src/dc.rs`: a real DC-side network (`DcBus`/`DcLine`, `solve_dc_network`) with `VsConverter`/`CsConverter` converters and converter losses, resolved by `cgmes_resolve_dc_converters` into AC-side injections. Only reachable via CGMES import — no HVDC element in the PGM-JSON or native-JSON paths |
| SVC (static var compensator) | ❌ | ❌ | ✅ | ✅ (`ControllableShunt`: stepped `Bmin`/`Bmax` regulating a `control_bus`'s voltage to `Vset`) | ✅ `create_svc` + `create_tcsc` (thyristor-controlled series capacitor) + `create_ssc` (static synchronous compensator) — broadest FACTS-device coverage of the six | ⚠️ CGMES `StaticVarCompensator` only: voltage-regulating (pins the controlled bus, incl. remote) when its `RegulatingControl` is voltage-mode and enabled, else a fixed Q injection. No `Bmin`/`Bmax` susceptance limits, no TCSC/SSC |
| Asymmetric / unbalanced power flow | ❌ symmetric only | ✅ | ✅ (`LfAsym*`) | ✅ (dedicated `Simulations/PowerFlow3ph/` driver) | ✅ (`runpp_3ph`) | ✅ (already solving, tested against PGM fixtures) |
| **CGMES / CIM import** | ❌ (0 CGMES/CIM-named files in its tree) | ❌ (0 CGMES/CIM-named files in its tree) | ✅ native, the reference implementation here | ✅ (48 CGMES/CIM-named files) | ✅ `converter/cim` (54 CGMES/CIM-named files) | ✅ EQ/EQBD/SSH/TP/SV profile merge by mRID, node-breaker reduction, ratio + all four phase-tap-changer flavors, 3-winding star resolution, HVDC, SVC, `ExternalNetworkInjection`, `EquivalentInjection`/`EquivalentBranch`, conform/non-conform loads, linear + nonlinear shunts, `AsynchronousMachine`, `PowerElectronicsConnection`. 14 fixture test files; benchmarked against pypowsybl on 8 conformance configurations (`scripts/bench/README.md` §6) |
| **Multi-island / disconnected components** | not surveyed | not surveyed | ⚠️ `connected_component_mode=MAIN` solves the largest component, drops the rest (verified directly — it is why pypowsybl's bus counts run below gridoxide's on every CGMES fixture) | not surveyed | not surveyed | ✅ every connected component solved in one call with a per-island `IslandReport`/`IslandStatus` (`Converged`/`MaxIterationsReached`/`Singular`/`NoReferenceBus`/`AmbiguousReferenceBus`); sourceless islands get a zero-voltage placeholder rather than an error |
| **Contingency / N-1 batch analysis** | ✅ `ContingencyAnalysis`, reuses factorization, ~20x speedup claimed | ❌ | ✅ + Woodbury fast-DC path | ✅ linear *and* nonlinear (full AC) contingency analysis, a HELM-based variant, SRAP support, and a time-series variant | ✅ `contingency` module, with a `run_contingency_ls2g` variant that offloads the actual solves to lightsim2grid for speed | ✅ **DC**: `linear::sensitivity::DcSensitivity::outage_flows` gives post-outage flows from one triangular solve with no refactorization, and `multi_outage_flows` generalizes it to N-k via Woodbury (0.40 ms per pair on `case9241pegase` against a 13.4 ms re-solve), including `is_breaking_set` for outage sets that disconnect the network. ✅ **AC**: `batch::BatchSolver::solve_contingencies`, 2.0x (`case118`) to 2.7x (`case9241pegase`) against independent solves single-threaded — `network::build_ybus_with_outages` takes a branch out while *keeping its structural entries*, so the symbolic factorization carries across a whole N-1 sweep. Contingencies that genuinely sever the network fall back to a full rebuild, which is what lets an islanded contingency report `NoReferenceBus` honestly. Note the plain injection-batch path (`BatchSolver::solve`) still refuses `Scenario::branch_outages` with `BatchError::OutagesUnsupported`; contingencies go through the dedicated method above. See gap 2 below |
| Time-series / batch injections | ✅ `TimeSerie`, ~13x speedup claimed | ✅ batch datasets, parallel via `threading` param | — | ✅ time-series variants of power flow, OPF, linear analysis, *and* contingency analysis | ✅ `timeseries` module (`run_time_series`, pluggable `DataSource`/`OutputWriter`) | ⚠️ `batch::BatchSolver` (`src/batch.rs`): many scenarios over one shared topology, parallel across cores via rayon, each worker amortizing one symbolic factorization over its share — 3.5x on 8 physical cores at 256 scenarios (`scripts/bench/README.md` §4b), results identical to a sequential loop and returned in scenario order. Injection overrides only (`BusOverride` deliberately cannot change `bus_type`, since that changes `n_unknowns` and invalidates the shared pattern), and no time-series driver layered on top — no `DataSource`/`OutputWriter` equivalent, no result writer. The deeper gap is **chronology**: scenarios are independent, so nothing carries state from one step to the next — storage state of charge, tap positions, controller memory — which is what separates a batch from a genuine quasi-static time series. simbench ships the profiles such a driver would be validated against |
| Input validation | ❌ | ✅ `validate_input_data`/`validate_batch_data` | — | ❌ no generic equivalent found (only format-specific CIM/FMU import validation) | ✅ `diagnostic()` (disconnected elements, implausible values, wrong reference system, ...) | ❌ |
| Short-circuit calculation | ❌ | ✅ (IEC 60909) | ❌ | ✅ (3-phase, LG, LL, LLG fault types — `Simulations/ShortCircuitStudies/`) | ✅ (IEC 60909-style, `shortcircuit` module) | ✅ IEC 60909, phase-domain (`src/shortcircuit/`): all four fault types (3-phase, LG, LL, LLG), `c_max`/`c_min` voltage scaling, bolted and impedance faults, multiple simultaneous faults, de-energized-island handling. Results in both bases — phase quantities *and* symmetrical components (the Fortescue view powsybl's API models, which PGM does not offer). Cross-validated against all 15 of power-grid-model's own short-circuit fixtures (`tests/pgm_short_circuit_test.rs`); CLI (`gridoxide short-circuit`) and Python (`gridoxide.short_circuit`). No study types/machine reactances, no derived \\(i_p\\)/\\(I_b\\)/\\(I_{th}\\) — see [The Short-Circuit Problem](../short_circuit/index.md) |
| State estimation | ❌ | ✅ (WLS, **sym + asym**, iterative-linear + Newton-Raphson, voltage/power/**current** sensors, **batched** with topology caching and thread-parallelism; no bad-data detection) | ❌ | ✅ (WLS + observability analysis + pseudo-measurement augmentation) | ✅ (WLS, `estimation` module) | ⚠️ **symmetric only** (WLS + observability + bad-data detection + zero-injection constraints, both PGM calculation methods, **batched** with thread-parallelism and a shared factorization, voltage/power/**current** sensors in both angle frames; reads asymmetric sensors but reduces them to the symmetric problem) — see the note below. The reduction is the binding limit for distribution-level SE, where unbalance is the problem rather than a refinement; [Resources](./resources.md) names simbench, VeraGrid's SE cases and the IEEE/EPRI feeders as the corpus for it |
| Sensitivity analysis / OPF | ❌ / ❌ | ❌ / ❌ | ✅ / ❌ | ✅ (PTDF/LODF, `Simulations/LinearFactors/`) / ✅ (linear *and* nonlinear AC OPF, `Simulations/OPF/`) | ✅ (PTDF, `pypower/makePTDF.py`) / ✅ native PDIPM AC+DC OPF (`runopp`/`rundcopp`) *plus* an optional external Julia PandaModels.jl bridge (`runpm.py`) for more advanced formulations | ✅ **DC**: PTDF/LODF plus N-k outage factors (`linear::sensitivity::DcSensitivity`, exact because DC is linear) — see [DC power flow](../powerflow/dc.md#sensitivity-factors-ptdf-and-lodf). ✅ **AC**: `ac_sensitivity::AcSensitivity` differentiates a converged operating point against active/reactive injection, transformer ratio and phase-shifter angle, for branch P/Q and bus voltage — both forward (one solve per variable) and adjoint (one solve per monitored quantity), against a single Jacobian factorization. Validated by central-difference re-solve (`tests/ac_sensitivity_test.rs`); CLI (`gridoxide sensitivity`) and Python (`AcSensitivityModel`). No AC contingency sensitivities and no differentiation through the outer loops — see [AC Sensitivity Analysis](../sensitivity/ac.md). / ⚠️ **DC-OPF** (`opf::dc`): least-cost dispatch as a convex QP over generator active power and load shedding, subject to generator boxes, branch ratings and the DC balance, reporting dispatch, locational marginal prices and binding limits with shadow prices. Solved through a solver-independent `LinearProgram`/`Solution` boundary with **two backends**: gridoxide's own primal-dual interior-point QP solver (`opf` feature — pure Rust, no system libraries, so it is the default and *is* covered by CI) and HiGHS via gridoxide's own bindgen FFI (`opf-highs`, the reference the two are cross-checked against on the fixtures and on 300 randomized convex QPs). Validated by analytic cases, KKT optimality certificates and pglib-opf's published DC objectives (all five cases inside 0.03%). CLI (`gridoxide opf`) and Python (`dc_opf`). ✅ **AC-OPF** (`opf::ac`): the full nonconvex problem — generator P and Q, voltage magnitudes and angles, apparent-power branch limits — via gridoxide's own nonlinear interior-point method on the injection Hessians of `injection_hessian`. All five pglib cases match the published AC objectives to 0.001%. Locally optimal, as every AC-OPF is; cross-checked against **IPOPT** (opt-in `opf-ipopt`, own bindgen bindings against a system install) to 5e-9 relative on every fixture. CLI and Python. No taps/phase shifters as variables, no unit commitment, no security constraints — see [Optimal Power Flow](../opf/index.md). |
| Pluggable "outer loop" architecture | ❌ | ⚠️ ad hoc (tap optimizer only) | ✅ extensively (14+ outer loops) | ⚠️ ad hoc (boolean control flags in `PowerFlowOptions`, not a modular/registry-based architecture like powsybl's) | ✅ genuine `Controller`/`BasicCtrl` base classes (`control/basic_controller.py`) registered on `net.controller` and driven by `run_control` — third-party code can subclass `Controller` directly, closer in spirit to powsybl's extensibility than to VeraGrid's/PGM's fixed flag sets, though not the same formal outer-loop-convergence architecture | ⚠️ **internal, not extensible**: `outerloop::OuterLoop` plus a driver over an ordered list, with four loops (distributed slack, reactive limits, phase control, transformer voltage control). The schedule is powsybl's — nested, innermost first, each loop run to its own stability before the next is consulted, re-walking on any change and terminating on reaching the last loop that moved something. Deliberately *not* ServiceLoader-style discovery: the list is built by the crate. What it bought is composition, which was the actual defect — before it, `newton_raphson_enforcing_q_limits` and `newton_raphson_distributing_slack` were separate entry points and **a caller could have at most one**. See [Outer Loops](../powerflow/outer_loops.md) |
| **RMS / transient stability** (phasor-domain dynamics) | ❌ | ❌ | ❌ | ✅ `Simulations/Rms/` | ❌ | ✅ `src/dynamics/` (`dynamics` feature): the differential-algebraic system solved **simultaneously** — one Newton per step over device states and network voltages together, in rectangular current balance so the network block is the constant real form of the Y-bus, on the same five sparse backends the power flow uses. Trapezoidal with backward-Euler damping after each discontinuity. Machines classical / transient / salient-pole / subtransient; `SEXS` and a proportional regulator; `TGOV1` and a proportional governor; a washout-plus-lead-lag stabilizer; ZIP loads with a low-voltage cutoff. Bus faults, branch and unit switching, load steps — all **value-only**, so one symbolic factorization serves a whole run — plus **protection relays** on bus voltage, machine speed or angle excursion, whose crossing times are located inside the step rather than rounded to it. Reads gridoxide JSON, PSS/E `.dyr`, and a whole Dynawo case — IIDM network plus `.dyd`/`.par` — in one call; CLI (`gridoxide dynamics`) and Python (`dynamics`). Gated against the equal-area criterion's closed-form critical clearing time (12 µs), a finite-difference Jacobian oracle over every model, and **Dynawo's own published answer** for Kundur's Example 13.2 (`δ₀` to 6e-5 rad, rotor angle through the fault to 9e-4 rad). Non-windup limits on the exciter's field voltage and the governor's valve, carried through from both readers. Not yet: saturation. The classical `ω ≈ 1` stator approximation is the default and costs a measured 0.6% of terminal power per 0.9% of speed deviation; Dynawo's fuller form (speed voltages carried, swing equation in torque) is one flag away and removes 96% of the disagreement with it. See [The RMS Simulation Problem](../dynamics/index.md) |
| **EMT** (electromagnetic transients) | ❌ | ❌ | ❌ | ✅ `Simulations/EMT/` | ❌ | ❌ — sub-cycle three-phase time stepping, switching devices, travelling-wave lines. The furthest of the three |
| **Small-signal / modal** (eigenvalue) analysis | ❌ | ❌ | ❌ | ✅ `Simulations/SmallSignalStabilityRms/` + `SmallSignalStabilityEmt/` | ❌ | ⚠️ **RMS half built** (`src/dynamics/smallsignal.rs`): the DAE linearized about an equilibrium, the algebraic block eliminated (`A = A_x − A_v·C_v⁻¹·C_x`), eigenvalues, damping ratios, modal frequencies, **participation factors** naming the states each mode belongs to, and **mode shapes** saying how the rotors move relative to one another — which is what separates an inter-area mode from a local one. The four blocks are not re-derived — they are the ones `DaePattern::fill` already assembles for every Newton iteration, so the linearization cannot drift from the simulation it describes. Gated against the closed-form swing eigenvalue `±j√(Ω_b·K_s/2H)`, and against the time-domain run's own observed period (2e-3) and peak-to-peak decay (2%), through almost no shared code. CLI (`gridoxide dynamics --modes`) and Python (`small_signal`). Dense, so `O(n³)`: a few thousand states would want sparse Arnoldi, which is not implemented. No EMT half, and no eigenvalue *sensitivities* — see [Small-Signal Analysis](../dynamics/smallsignal.md) |
| **Remedial action optimization** (CRAC / CNEC, preventive + curative) | ❌ | ❌ | ❌ — this is [powsybl-open-rao](https://github.com/powsybl/powsybl-open-rao)'s job, a separate project, and it is the reference gridoxide is gated against | not surveyed (its contingency row mentions SRAP support, which is adjacent) | not surveyed | ✅ `src/rao/`: CRAC model and readers (all 428 vendored CRACs import, 24 format versions), evaluation kernel with Woodbury outage screening, LP over range actions, search tree over network actions, CASTOR perimeter decomposition, automaton simulation, MNEC soft constraints, conditional usage rules, RA usage limits, curative stop criterion, AC re-validation. Gated against the reference's **own** Cucumber suite — 156 scenarios, 138/142 DC and 192/203 + 727/883 AC assertions, 108 scenarios matching completely. Not yet: action combinations beyond the greedy chain, second-preventive, loop flows, relative margins. See [The Remedial Action Problem](../rao/index.md) |
| **Voltage stability / continuation power flow** (P-V and Q-V curves, loadability limit) | ❌ no match for continuation/CPF/loadability in its tree (its `cpf` hits are `dcpf`) | ❌ same | ❌ same | not surveyed | not surveyed | ✅ **both curves and the loadability limit.** `src/qv.rs`: Q-V curves and the per-bus reactive margin (`gridoxide qv`, `PowerFlowModel.qv_curve`) — a fictitious condenser at the bus, its setpoint swept, the minimum interpolated through the three bracketing samples; gated against a **closed-form two-bus Q-V curve** to 1e-5 in Q. Worth knowing, and the opposite of the natural assumption: **Q-V and P-V do not agree on the weakest bus** (on `case14`, one bus in common out of five, and evaluating both at the same loading makes it worse) — they measure a system-wide collapse mode against one bus's local headroom, which is why both exist. See [Q-V Curves](../powerflow/qv.md). And `src/continuation/`: **P-V curves and the loadability limit** — tangent predictor, extended-Newton corrector over the bordered `(n+1)` system, three parametrizations (local by default — pseudo-arclength's dense bordering row measures 92x a sparse one at n = 2449), adaptive step, lower-branch tracing. `case9241pegase` traces to its nose in 10 s. Saddle-node *and* limit-induced bifurcations are distinguished; reactive-limit crossings are located exactly (bisection on arclength, matching a brute-force oracle to `1e-6` in λ against `1e-2`–`1e-1` for step-granularity switching); the weakest-bus ranking is the tangent at the nose. Gated against a **closed-form two-bus nose valid including resistance**, since no vendored reference implements continuation at all. Not yet: Q-V curves, tap control during a trace, bidirectional PV↔PQ switching, ZIP-load derivatives (refused rather than approximated). See [Continuation Power Flow](../powerflow/continuation.md) |
| **Harmonic analysis / frequency scan** | ❌ no match for harmonic/frequency-scan in its tree | ❌ same | ❌ same | not surveyed | not surveyed | ❌ — needs a per-frequency Y-bus, harmonic source models and frequency-dependent branch data |
| **Reliability / adequacy** (Monte Carlo, LOLE/EENS) | ❌ no match for reliability/monte-carlo/LOLE/EENS in its tree | ❌ same | ❌ same | not surveyed | not surveyed | ❌ — the contingency kernel exists; the outage-rate data, sampling driver and indices do not |
| **Protection coordination** (relay curves, selectivity) | ❌ explicitly not simulated — "ignore the protection, that are NOT simulated" in its own examples | ❌ no match in its tree | ❌ same | not surveyed | not surveyed | ❌ — would sit on the short-circuit engine that does exist |
| **Input formats** beyond CGMES | ⚠️ ingests from pandapower and pypowsybl/IIDM rather than reading files itself | ⚠️ own PGM format only | ✅ IIDM native | ✅ Matpower, PSS/E RAW | ✅ Matpower / PYPOWER | ⚠️ PGM-JSON, own native JSON, **UCTE-DEF** (`src/ucte.rs`), **IIDM** (`src/iidm.rs`), CGMES — the last giving each galvanically-connected group of buses one voltage base, since a tie line joining a Belgian 380 kV bus to a Dutch 400 kV one is one conductor and per-unit across it does not otherwise close (`cgmes::harmonize_voltage_bases`; 5 of 13 lines on MicroGrid-Type1 and FullGrid, none elsewhere). MATPOWER `.m` is read only by a *Python* conversion script (`python/gridoxide/matpower.py`) that emits PGM-JSON, not by the Rust core. No PSS/E RAW |
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
- **Remedial action optimization**: the one row where *none* of the five comparison tools competes — the reference is powsybl-open-rao, a separate project. And it is the most externally-validated thing here: the reference publishes its expectations as a Cucumber suite, so 156 of its own scenarios are vendored verbatim and scored assertion by assertion (`tests/rao_cucumber_test.rs`), rather than gridoxide checking its own arithmetic against itself.
- **Solution verification tooling**: `scripts/bench/check_matpower_residual.py` checks a solved case against the MATPOWER file's *own* power-flow equations, and `check_cgmes_sv_consistency.py` checks a CGMES fixture's published `SvVoltage` against its own EQ/SSH data. Neither needs a second tool as a reference. This is a benchmark-harness capability, **not** an input-validation feature — it does not close the "Input validation" row above, which is about validating input before a solve (PGM's `validate_input_data`, pandapower's `diagnostic()`).

## Identified gaps, ranked by how often reference tools flag them as important

1. **Q-limit enforcement / PV→PQ switching** — every one of the five either has it, half-has it, or explicitly disclaims *not* having it as a known limitation. gridoxide's `Bus` already carried `q_min`/`q_max`, unused. **Done** — `outerloop::ReactiveLimits` implements the standard MATPOWER-style one-directional PV→PQ switching outer loop, tested in `tests/q_limits_test.rs` across all three Jacobian backends; `PgmVoltageRegulator` now parses PGM's own `q_min`/`q_max` fields. Opt-in: plain `newton_raphson`/`PersistentSolver::solve` are unchanged, so no existing test/benchmark behavior shifted. It was `solver::newton_raphson_enforcing_q_limits`, a standalone entry point, until the outer-loop layer made it composable with the two below.
2. **Contingency/N-1 batch analysis** — the one gap with most of its machinery already standing: `PersistentSolver`'s factorization reuse and `batch::BatchSolver`'s across-scenario parallelism are exactly what lightsim2grid, powsybl, VeraGrid (the most comprehensive, with linear, nonlinear, and HELM-based variants), and pandapower (which even offloads some of its own contingency solves to lightsim2grid for speed) build this on. **Done for DC**, which is the mode these tools actually screen contingencies in: `linear::sensitivity::DcSensitivity::outage_flows` gives post-outage flows from one triangular solve with no refactorization (and reports radial branches, which have no redistribution factors), while `linear::batch::DcBatchSolver` runs injection scenarios against a single factorization for the whole batch — 9x to 55x per scenario against independent solves, see `scripts/bench/README.md` §8. N-2 and N-k are covered too, via the generalized (Woodbury) formulation in `multi_outage_flows` — 0.40 ms per pair on case9241pegase against a 13.4 ms re-solve — including `is_breaking_set` for outage sets that disconnect the network, which single-branch screening misses by construction. **AC contingency is done as well**, via `batch::BatchSolver::solve_contingencies`. The old objection recorded here — "a branch outage gives each scenario its own Y-bus and therefore its own sparsity pattern" — turned out to be half right: the Y-bus does differ per scenario, but the *pattern* need not, because `network::build_ybus_with_outages` takes a branch out while keeping its structural entries. The symbolic factorization therefore carries across a whole N-1 sweep; only the Jacobian's cached admittances are re-derived, at O(nonzeros). Measured at 2.0x (case118) to 2.7x (case9241pegase) against independent solves *single-threaded*, before any parallelism. Contingencies that genuinely sever the network cannot use that trick — `connected_components` cannot distinguish a zeroed structural entry from a live one — so they are detected up front and fall back to a full rebuild, which is what lets an islanded contingency report `NoReferenceBus` honestly instead of a singular solve.
3. **Distributed slack** — 4 of 5 tools have it (lightsim2grid, powsybl, VeraGrid, pandapower); real transmission grids split slack across several generators, and a single slack absorbing an unscheduled few hundred megawatts distorts every branch flow around it. **Done** — `outerloop::DistributedSlack`, an outer loop in the same shape as the Q-limit one: solve, compare what the slack actually produced against what its own `p_spec` schedules it for, move the difference onto the participating generators by normalized weight, solve again.

   Two things are worth recording. The slack's schedule is its **own `p_spec`**, a field ordinary power flow ignores for a slack bus because the slack's output is an answer rather than an input — so a document that never needed it may leave it at zero, in which case the entire output is redistributed. And the outer loop is cheap in a way the Q-limit loop is not: only `p_spec` changes between passes, so bus types, `n_unknowns` and the Jacobian's sparsity pattern all hold and `PersistentSolver` keeps its symbolic factorization throughout.

   Convergence is linear, not one-and-done. The first-order term cancels exactly in a single pass — distributing \\(\Delta\\) adds \\((1-\alpha_s)\Delta\\) to the other participants while the slack's own schedule rises by \\(\alpha_s\Delta\\) — but the leftover is the *change in transmission losses* from the redistributed flows, which shrinks by a few per cent per pass rather than vanishing. Measured at `tolerance = 1e-8`: `case14_ieee`, `case30_ieee` and `case118_ieee` each take seven passes, moving 2.3, 2.4 and 16.5 per-unit off their slacks. Tested in `tests/distributed_slack_test.rs`, with every check re-deriving injections from `network::power_injections` rather than reading the loop's own bookkeeping.

   `newton_raphson_distributing_slack` is now `outerloop::DistributedSlack`, one loop among the ordered list of gap 11's entry, which is what lets it run *together* with Q-limit enforcement — the two used to be alternatives. Both are reachable from `gridoxide solve --distribute-slack --enforce-q-limits` and from `PowerFlowModel.solve(distribute_slack=True, enforce_q_limits=True)`; the "library-only" note this paragraph used to carry is out of date.

   **Area interchange is done too**, as `outerloop::AreaInterchange`. It is the same mechanism generalized from one balance target to several, which is why powsybl's `AcAreaInterchangeControlOuterLoop` constructs a `DistributedSlackOuterLoop` as its no-area fallback — and here that relationship is an assertion rather than a remark: one area with a zero target reproduces distributed slack's per-bus shifts and solved voltages exactly.

   Building it turned up something the earlier sketch of this paragraph missed. Both ends of a tie line are measured *into* the branch, so the positions sum to the tie's losses rather than cancelling, and a set of agreed net positions — which sums to zero — is unachievable by exactly that amount. The knobs are one aggregate schedule per area, \\(N\\); the conditions wanted are \\(N\\) positions plus the slack on its own schedule, \\(N+1\\). Over-determined by one, always. So the slack's own area is the **dependent** one: its target is not enforced, it absorbs the tie losses, and the report names it and states the residual rather than implying it.

   **Both importers are wired now.** CGMES supplies both halves — `ControlArea.netInterchange` for the schedule, `TieFlow` for the boundary — and membership is derived by cutting the tie branches and taking connected components, because CGMES never lists what is inside an area. MicroGrid is the tree's genuine two-area fixture and the loop puts BE on its declared −236.977 MW exactly. UCTE supplies membership from its `##Z` country codes and no schedule, which is all any UCTE file states; the twelve-node case is a real four-country interconnection. Both reach `gridoxide solve --area-interchange`.

   Two things the fixtures corrected. A `TieFlow` names the terminal at the *boundary* — the X-node two areas' lines meet at — not inside the area, so the seed is the cut branch's far end; seeding from the named end gave five contested buses and one area owning nothing. And the sign: CGMES states an import, `AreaDefinition` wants an export, checked against SmallGrid's own published state sitting at 210.271 MW against its declared 210.

   IIDM is wired too, and is the only importer that states membership *directly* — `<iidm:area>` lists its voltage levels, where CGMES gives only a boundary and UCTE a country per node. The twelve-node case exists in both UCTE and IIDM and their areas agree to 1e-6 (BE +2000 MW, DE −2500, FR +1000, NL −500), by entirely separate code on each side.

   One thing real data forced. `AreaDefinition::uniform` participates `Slack` and `PV` buses only, inherited from distributed slack — right for frequency response, wrong for a net position, which is met by *redispatch*, and a generator on a fixed active set-point is exactly what gets redispatched. powsybl's own `two_area_case.xiidm` has both of AREA2's machines at `voltageRegulatorOn="false"`, so under `uniform` that area had no participant and its −400 MW schedule was unreachable. `AreaDefinition::by_generation` weights by `p_spec` instead and the CLI uses it; AREA2 then reaches −400.000 exactly.

   Still absent: nothing in the tree states a net position for a *UCTE* network, and PGM JSON has no area concept at all. See [Area Interchange Control](../powerflow/area_interchange.md).
4. **DC power flow as a first-class mode** — every reference tool treats this as a basic offering. **Done**, and in the doing it turned out the old single "DC / linear power flow" row was conflating two unrelated algorithms, which is why the summary table above now carries two. The premise that `linear_initial_guess` was "most of the way there" was only true for one of them: that function *is* power-grid-model's `CalculationMethod.linear` (complex, constant-admittance, keeps resistance and produces `|V|`), and it is now exposed as `linear::impedance::linear_power_flow`. What was genuinely missing was real Bθ — the thing lightsim2grid, powsybl-open-loadflow and pandapower's `rundcpp` mean by DC — and that is new code in `linear::btheta`, with both of powsybl's approximation variants, phase-shifter support, per-island factorization, and PTDF/LODF sensitivities on top. See `docs/src/powerflow/dc.md` and `linear_impedance.md`; tested in `tests/dc_powerflow_test.rs` and `tests/dc_sensitivity_test.rs`.
5. **Switches, HVDC and SVC as first-class model elements** — no longer absent, but reachable *only* through CGMES import: switching state, converter setpoints and SVC regulation are all fixed at import time, with no element in gridoxide's own network model to change between solves. That is exactly the shape lightsim2grid's disclaimer calls out for its own fixed taps, and it is what stands between the current support and the contingency/time-series work in item 2.
6. **Short-circuit calculation** — **Done.** This was previously listed in item 6 below as outside gridoxide's scope; it is not any more. `src/shortcircuit/` implements IEC 60909 in the phase domain: all four fault types, both voltage-scaling choices, bolted and impedance faults, several simultaneous faults, and de-energized islands, with results reported in phase quantities *and* symmetrical components. The premise that made it affordable is that gridoxide was already sequence-in, phase-assembled — `Line3Ph` carries `r0`/`x0`, `transformer_seq_params` returns `(y0, y1, y2)`, and `network::fortescue_to_phase` was already building the phase-domain Y-bus from them — so the new code is mostly fault boundary conditions on top of tested assets.

   Building it closed three genuine gaps in the shared three-phase model, each found by a fixture rather than by inspection: `transformer_seq_params` supported only two winding pairs and **panicked** on the rest (it now implements power-grid-model's general zero-sequence algorithm, covering YNd, zigzag and the low-susceptance rule); `Line3Ph` had no conductance field, silently dropping the `tan δ` dielectric-loss term; and the three-phase path dropped `link` components entirely, which quietly de-energizes whatever hangs off one.

   Validated against all 15 of power-grid-model's short-circuit fixtures. Eleven match to its own `1e-8`; four cannot, because power-grid-model changed its zero-sequence regularization in **November 2025** and never regenerated fixtures dating from **2023** — an inconsistency inside the reference set rather than a choice gridoxide can make differently. `tests/data/pgm/short_circuit/README.md` and `KnownDivergence` in the test record which, why, and by how much.

7. **Sensitivity analysis** — **Done for both models**, and it was previously listed in the item below as outside gridoxide's scope. The DC factors (PTDF, LODF, and the Woodbury N-k generalization) were already there under item 2; what is new is `ac_sensitivity::AcSensitivity`, which differentiates a converged AC operating point directly.

   The premise that made it cheap is that the derivative needs no new matrix: \\(dx/dp = -J^{-1}\\,\\partial g/\\partial p\\) uses the *same* Jacobian Newton already assembles, and for an injection the right-hand side is a unit vector, so one triangular solve answers for the whole network. The \\(\\partial f/\\partial x\\) row for a branch flow is `branch_flow::terminal_flow_derivs`, written for the state estimator and reused unchanged. The only genuinely new numerics are the tap derivatives, which reduce to a scaling of each admittance entry by itself.

   Both directions are offered against one factorization — forward (one solve per variable, "what does this move?") and adjoint (one solve per monitored quantity, "what would move this?") — with `sparse::RealFactorization::solve_transpose` reusing the same LU rather than factorizing the transpose. Variables are active and reactive injection, transformer ratio and phase-shifter angle; functions are branch P/Q at either terminal and bus voltage magnitude or angle.

   Validated against a central-difference re-solve of the full nonlinear power flow, which shares no code with the module. One finding worth recording: a tapped branch's *own* flow carries a direct \\(\\partial f/\\partial p\\) term on top of the chain rule, and dropping it flips the sign of that branch's own sensitivity (\\(-0.276\\) to \\(+0.286\\) on `distribution-case`) while leaving every other branch correct — a failure mode that looks entirely plausible in the output.

   Still absent: AC *contingency* sensitivities (an outage is a finite topology change, not a derivative — `BatchSolver::solve_contingencies` re-solves instead) and differentiation through the outer loops.

8. **DC-OPF** (`opf::dc`, `opf` feature) — least-cost dispatch as a convex QP, with locational marginal prices and binding-limit shadow prices, behind a solver-independent boundary. Two backends sit behind it: an in-house interior-point QP solver needing nothing installed, and HiGHS as a cross-checked reference. Validated against analytic cases, KKT certificates, pglib-opf's published DC baseline (all five cases inside 0.03%), and solver-vs-solver agreement.

   **AC-OPF** (`opf::ac`) as well: the full nonconvex problem over generator P and Q, bus voltage magnitudes and angles, with apparent-power branch limits, solved by gridoxide's own nonlinear interior-point method (`opf::nlp` — line search, adaptive regularization, gradual barrier) on the injection Hessians of `injection_hessian`. All five pglib cases match the published **AC** objectives to 0.001% at violations of 1e-9 or better; derivatives are checked against finite differences and prices against a numerical d(cost)/d(load). A KKT point of a nonconvex problem is locally optimal, which is what every AC-OPF tool reports. CLI (`gridoxide opf --ac`) and Python (`ac_opf`). See [Optimal Power Flow](../opf/index.md).

   Still absent: taps and phase shifters as decision variables, unit commitment, and security-constrained OPF.

9. **Remedial action optimization** — **Done**, and it is not a row any of the five comparison tools
   fills: the reference here is powsybl-**open-rao**, a separate project from open-loadflow.
   `src/rao/` carries the CRAC model and readers, the evaluation kernel, an LP over range actions, a
   search tree over network actions, the CASTOR perimeter decomposition, automaton simulation, MNEC
   soft constraints, conditional usage rules, RA usage limits and the curative stop criterion, with
   an AC re-validation stage on top.

   What makes it unusual among the items here is the gate: the reference ships its expectations as a
   Cucumber suite, so 156 of its own scenarios are vendored verbatim and scored — 138/142 DC and
   192/203 plus 727/883 AC assertions, with 108 scenarios matching completely. Nineteen real defects
   were found by it, every one internally consistent and externally wrong. See
   `plans/RAO_PLAN.md` §8.3 for the list and [The Remedial Action Problem](../rao/index.md) for the
   mathematics.

   Still absent: action combinations beyond the greedy chain (the structural one — the reference
   blooms *combinations* at each depth where this search extends a single best chain),
   second-preventive optimization, loop flows and relative margins.

10. **The analysis types with no row until this revision** — voltage stability, harmonics,
    reliability, protection coordination, and the three dynamics rows. None of them is a refinement
    of something already built; each is a different question asked of the network, and they are worth
    ranking rather than lumping together:

    - **Voltage stability / continuation power flow** was the cheapest by a distance, and is now
      built (`src/continuation/`) — a predictor–corrector loop around the Newton solver that already
      handled Q-limits, distributed slack and multi-island, answering the question nothing else in
      the set answers: *how much further can this be loaded?* What remains of the row is Q-V curves.
      The estimate held: the reusable parts were the Jacobian pattern, the reactive-limit outer loop
      and the sparse backends, and the genuinely new numerics are small. What the estimate missed is
      that two of the three outer loops must **not** be run inside the corrector — they move
      `p_spec` as a function of the solved state, which corrupts the tangent — and are folded into
      the loading direction analytically instead. `dynawo-algorithms` offers voltage margin and
      load-increase-to-collapse as an external cross-check.
    - **Quasi-static time series** is next, and half-built: `BatchSolver` already runs many scenarios
      over one topology in parallel. What is missing is chronology — carrying storage state of
      charge, tap positions and controller memory from one step to the next — plus a driver and a
      result writer. simbench ships the profiles.
    - **RMS dynamics** was the largest, and is now built (`src/dynamics/`) — the DAE, its
      integrator, discrete events, the machine/exciter/governor/stabilizer library, three readers,
      and the surfaces. The estimate that the corpus and methodology would come close to free held,
      and better than expected: Dynawo's repository ships its own solver's **reference outputs**
      alongside each example, so the external gate needed no Dynawo install at all — a sparse clone
      of the example files was enough.
      What the estimate missed is where the real risk lay. It was not the integrator, which is a
      hundred lines and gated by closed forms; it was the *conventions*. Published statements of the
      subtransient machine disagree about the sign of `ψ_2q`, and saturation is stated
      incompatibly by PSS/E and Dynawo. Both had to be settled by derivation and by declining to
      guess, not by copying a source. And the one difference no self-consistency check could ever
      have found — the `ω ≈ 1` stator approximation, worth 0.6% of terminal power per 0.9% of speed
      deviation — was found by reading Dynawo's Modelica after its trajectory diverged from ours.
      That is precisely what an external reference is for.
    - **EMT, harmonics, reliability and protection coordination** stay genuinely far off. Each needs
      modelling gridoxide has no foundation for — sub-cycle three-phase stepping, per-frequency
      Y-buses, outage-rate data, relay curves — and none is currently pointed at by any plan.

    See [Resources](./resources.md) for the datasets and reference implementations each would be
    validated against.

11. **Transformer tap control, and the outer-loop layer under it** — **Done.** This entry used to
    lump both in with "materially larger undertaking or outside gridoxide's current scope", and
    singled out static taps as a modelling assumption four of the five comparison tools do not
    make. They are one job, and the second half turned out to be the load-bearing one:
    `newton_raphson_enforcing_q_limits` and `newton_raphson_distributing_slack` were standalone
    entry points, so **a caller could have at most one** — not an architecture nicety but two
    shipped features that could not run together. Both are `outerloop::OuterLoop` implementations
    now, driven over an ordered list by powsybl's own schedule, with
    `TransformerVoltageControl` and `PhaseControl` as the third and fourth.

    Two things the fixtures corrected, both recorded as assertions so they cannot quietly stop
    being true. Svedala's own published solution violates all eleven of its declared deadbands by
    2–5%, so the fixed-point gate the plan proposed — taps must not move, since SSH and SV agree —
    had a false premise, and a loop that left those taps alone would be the defective one. And a
    discrete tap often cannot reach a deadband at all: `PST_PhaseTapChangerLinear_Type2` asks for
    zero flow within 5e-4 pu on a shifter whose closest position leaves 0.11 pu, which needs an
    outcome of its own rather than being reported as success.

    Found on the way: CGMES transformer ordering was nondeterministic — `ends_by_pt` is a
    `HashMap` and Rust randomizes its iteration order per process, so every flat branch index
    derived from the transformer list came out differently on each run of the same program against
    the same file. Nothing asserted on one, so it never failed.

    Still absent: TCSC/SSC and the wider FACTS set; powsybl's two continuous-then-round voltage
    strategies; `reactivePower`-mode tap control; shunt-section control; tap control inside a
    contingency or batch sweep, and in DC. And the outer-loop layer is deliberately *internal* —
    the list is built by the crate, not discovered — so the row above reads `⚠️` rather than `✅`.
    See [Outer Loops](../powerflow/outer_loops.md) and
    [Transformer Tap Control](../powerflow/tap_control.md).

## Note on state estimation

**Done for symmetric estimation — snapshot and batched, with voltage, power and current sensors — and
for asymmetric networks driven by symmetric sensors.** Within that scope gridoxide matches the most
capable reference tool and leads it in two places. What is left of the three gaps this note used to
list is one part of one of them, at the end. An earlier version claimed parity outright, which
overstated it.

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

### The gaps that remain

Measured against power-grid-model 1.13 (`references/power-grid-model/`), in order of how much they
matter:

1. **Asymmetric state estimation.** Substantially done, and what remains is narrower than the
   heading suggests.

   A phase-domain document now estimates end to end. `SeNetwork::from_3ph` builds the measurement
   model for a 3N-bus network, `pgm::pgm_3ph_maps` supplies the object-ID maps a sensor needs to
   resolve against it, and `measurement::measurements_from_pgm_3ph` maps sensors onto phase-expanded
   targets. `tests/se_three_phase_test.rs` estimates power-grid-model's `transmission-case` in the
   phase domain and matches the answer it published for that network solved asymmetrically — all 33
   phase-buses to 1e-6, with angles agreeing up to the single rotation nothing measures.

   Nothing in the estimator changed for it. `Target`, `StateLayout`, the Jacobian, the constraints,
   both methods and the batch solver carry over untouched, because a three-phase branch terminal is a
   six-coefficient `CurrentFunctional` where a scalar one has two. A branch is indexed
   `3·branch + phase` to match the `3·node + phase` bus convention, which is what keeps `Target`
   identical between the two domains.

   The symmetric sensors that fixture carries need no conversion beyond a rotation, and it is worth
   recording why: a voltage reading is line-to-line over `u_rated` in the scalar case and
   line-to-neutral over `u_rated/√3` here, which is the same number for a balanced set, and a power
   reading is a three-phase total over `s_base` against a per-phase value over `s_base/3`, likewise.
   So the value replicates and only the angle rotates by 0/−120/+120 — which is exactly
   power-grid-model's own `ComplexValue<asymmetric_t>` broadcast.

   Asymmetric sensors describe their three phases separately here rather than reducing to the
   symmetric problem: `asym_voltage_sensor`, `asym_power_sensor` and `asym_current_sensor` each
   select their own phase's reading, against a line-to-neutral voltage base and a `s_base/3` power
   base. Checked against `single-node-source-asym-voltage-sensor`, where the sensor determines the
   answer outright and power-grid-model reports back exactly what it read.

   **What is left is one modelling difference, and it is worth stating precisely.** Voltage
   magnitudes alone do not determine a phase relationship. With flows in the set the source
   impedance couples the phases — a flow through it depends on all six of its phasors — but given
   only magnitudes the three per-phase rotations are three separate symmetries where `StateLayout`
   removes one, and the gain matrix is correctly singular. power-grid-model answers such a case
   because its source is a *boundary condition*, a fixed balanced three-phase voltage; gridoxide's
   is an unknown behind a synthesized impedance, the same difference that leaves
   `SeReport::unconstrained` naming a virtual bus per source on the symmetric side.

   The obvious fix is wrong, and it was tried rather than assumed. gridoxide *builds* that virtual
   bus balanced, so constraining its three angles to differ by ±120° looks like free information and
   removes exactly the two directions in question. It also contradicts the data: power-grid-model's
   own `single-node-source-asym-voltage-sensor` reads three phases whose sequence angles are 0.1,
   0.2 and 0.3, on a node with no appliance — zero injection, so zero current through the source
   branch, so `V_virtual = V_node` exactly. The virtual bus is as unbalanced as the measurement says
   the node is, and the constraint moves that fixture's answer from 0.1 to 0.2. The balance is a
   property of the initial state gridoxide synthesizes, not of the equivalent it represents.

   So this is not a missing feature but a real limit: those two directions are undetermined, and
   reporting singular is the correct answer. power-grid-model answers instead because it has no
   source-internal bus to be undetermined about. Both directions are asserted in
   `tests/se_three_phase_test.rs`, so a change that supplies them will announce itself.

   `pgm_3ph_maps` refuses the components the three-phase conversion does not model — `link`,
   `three_winding_transformer`, `voltage_regulator`, and any transformer winding pair outside Dyn
   and YNyn — with a typed error rather than dropping them silently or, in the last case, panicking
   from inside `transformer_seq_params`.

2. ~~**Current sensors.**~~ **Done.** `sym_current_sensor` and `asym_current_sensor` are read in both
   angle frames, on both calculation methods, checked against power-grid-model's own
   `global-current-sensor` and `local-current-sensor` fixtures — which are identical but for the
   frame and converge to visibly different states, so the distinction is genuinely exercised rather
   than nominally supported.

   Stored decomposed into real and imaginary components rather than as a magnitude and an angle,
   following power-grid-model, and for a decisive reason of gridoxide's own: `arg(I)` has a branch
   cut and gridoxide has no `phase_mod_2pi` anywhere, so a polar residual taken near ±π would
   silently chase a 2π error. `|I|` also has an unbounded derivative on an unloaded branch. The
   variance decomposition reproduces power-grid-model's second-order formula exactly.

   Two rules are enforced that power-grid-model checks only in its Python validation layer, its C++
   core accepting and double-counting the mixture: a power sensor and a current sensor may not share
   a terminal, and two current sensors on one terminal may not disagree about the frame. A current
   sensor on a `link` is refused outright — a link's admittance is a regularization constant, so the
   current through one is an artifact of that choice rather than a measurement.

   One divergence worth recording: power-grid-model refuses to run at all when a global-angle sensor
   has no voltage angle to reference, raising `NotObservableError`. gridoxide reports it through
   `ObservabilityReport::global_current_without_angle_reference` instead. The state is fully
   determined there — determined to the *wrong* reference, since `StateLayout` pins a bus the sensor
   contradicts — so calling it unobservable would misname it.
3. ~~**Batch state estimation.**~~ **Done.** `se::batch::SeBatchSolver` (`src/se/batch.rs`) estimates
   many scenarios over one topology and measurement structure, parallel across cores, each worker
   amortizing one symbolic factorization — the same shape `batch::BatchSolver` has for power flow,
   and exactly what `PersistentEstimator`'s already-written cache-validity condition allows.
   `MeasurementOverride` varies values and sigmas and refuses to vary `kind` or `target`, mirroring
   `BusOverride`'s refusal to change `bus_type`. Exposed as
   `StateEstimationModel.solve_batch(scenarios, threads)`. Checked against power-grid-model's own
   `sensor-update-*` and `unbalanced-power-measurements-*` batch fixtures, and asserted bit-for-bit
   identical to a sequential loop at every thread count.

On speed, the iterative-linear method runs 1.6-2.0x behind power-grid-model's across an order of
magnitude of problem size (`scripts/bench/README.md` §7). Measured rather than inferred, that is
entirely an iteration-count gap: gridoxide's own iterations are 30-40% *cheaper* than
power-grid-model's and it takes about three times as many, and undamped its map does not converge at
all. See `docs/src/state_estimation/iterative.md`.

A fourth gap surfaced while closing the third and is now closed too: **de-energized islands**.
power-grid-model reports a node in a component containing no source as `energized: 0` with a state of
exactly zero — topology decides it, and a voltage sensor on such a node is simply ignored. gridoxide
had no equivalent, and the consequence was not a wrong answer but no answer:
`jacobian::mask_untouched` pins a column nothing *structurally* touches, which catches a fully
isolated node, but a de-energized node reached by a zero-injection constraint or by its own sensor is
touched and undetermined, so the gain matrix came back singular. `SeNetwork::energized` now carries
the same topological verdict `solver::PersistentSolver` has always applied on the power-flow side
(`network::connected_components` + `mark_unreferenced_islands`): such a bus contributes no rows and
no constraints, and is reported at zero.

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
the smallest case to 166 on `case3120sp`), and `outerloop::ReactiveLimits` converges on
every one of them, including cases needing dozens of simultaneous PV→PQ switches across several
outer iterations. MATPOWER represents "no limit" as literal `+-Inf` on some real cases (e.g.
`case9241pegase`) — the converter omits the key entirely in that case rather than writing a
non-standard `Infinity` JSON token, matching PGM's own "unset means unbounded" convention exactly.
