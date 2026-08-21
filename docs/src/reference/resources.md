# Power Systems Test Datasets & Tools

A synthesized reference of open datasets and tools, compiled from research into candidate sources for gridoxide/cimoxide testing and validation.

## Transmission-level test cases (steady-state / power flow)

| Source | What it offers | Notes |
|---|---|---|
| [Sienna-Platform/PowerSystemsTestData](https://github.com/Sienna-Platform/PowerSystemsTestData) | 5-Bus PJM, 5-Bus-hydro, 118-Bus PCM (high renewables), TAMU ACTIVSg synthetic grids (10k-bus WECC, 70k-bus EI, 2000-bus ERCOT), Matpower cases | Plain case files, not tied to Julia structs — good for parser testing |
| [Sienna-Platform/PowerSystemCaseBuilder.jl](https://github.com/Sienna-Platform/PowerSystemCaseBuilder.jl) | 200+ cases pre-built into PowerSystems.jl data model | Julia-native; skips parsing step if you're in that ecosystem |
| [SanPen/VeraGrid](https://github.com/SanPen/VeraGrid) — `Grids_and_profiles/grids` | IEEE 39/118/30-bus, Pegase 2869/1354-bus, RTE 1951-bus, PGLib cases, HVDC test cases (2/10/12-bus AC/DC) | Reads Matpower natively; solves all Matpower 8 cases incl. continental-USA scale in ~1s |
| [SanPen/VeraGrid](https://github.com/SanPen/VeraGrid) — `src/tests/StateEstimation` | Reference SE test cases with expected outputs | Directly relevant to DSSE validation |
| [power-grid-lib/pglib-opf](https://github.com/power-grid-lib/pglib-opf) | Curated AC-OPF benchmark cases (incl. TAMU 200/500/2000-bus, ARPA-E GO Competition cases) | Maintained by IEEE PES Task Force; reference results included; supersedes NESTA |
| [power-grid-lib/pglib-opf-hvdc](https://github.com/power-grid-lib/pglib-opf-hvdc) | OPF benchmark cases with HVDC lines (MatACDC format) | Reference impl. in PowerModelsACDC.jl |

## Distribution-level / DSSE-relevant test cases

| Source | What it offers | Notes |
|---|---|---|
| [e2nIEE/simbench](https://github.com/e2nIEE/simbench) | German research-project MV/LV benchmark networks with full time series | Most relevant to your MV DSSE work; some grids underdetermined for SE out-of-box (need synthesized pseudo-measurements) |
| [e2nIEE/pandapower](https://github.com/e2nIEE/pandapower) | Matpower/PYPOWER-compatible cases, plus own SE module | Useful for cross-validating your SE implementation |
| IEEE/EPRI distribution feeders (4/13/34/37/123-bus, 8500-node) | Classic unbalanced 3-phase distribution benchmarks | Distributed via OpenDSS/EPRI, mirrored in various repos (e.g. `dss-extensions` org) rather than one clean source |

## CGMES / CIM data

| Source | What it offers | Notes |
|---|---|---|
| `powsybl-core` — `cgmes-conformity` module | ENTSO-E MicroGrid, SmallGrid, MiniGrid CGMES conformity test configs (v2.0) | Industry-standard CGMES validation suite; extract `cgmes-conformity/src/main/resources`, no need to build the Java project |
| [powsybl/powsybl-cgmes-conformity-assessments](https://github.com/powsybl/powsybl-cgmes-conformity-assessments) | Tracks PowSyBl's own conformity assessment results | Reference for how a mature parser scores |

## Realistic large-scale regional network models

| Source | What it offers | Notes |
|---|---|---|
| [PyPSA/pypsa-eur](https://github.com/PyPSA/pypsa-eur) | Full ENTSO-E-area European transmission network, generator fleet, sector links | Built via Snakemake from open data (Eurostat, JRC-IDEES); realistic scale, not a "test case" |
| [PyPSA/pypsa-usa](https://github.com/PyPSA/pypsa-usa) | US regional network model | Same pattern as pypsa-eur |
| [PyPSA/pypsa-za](https://github.com/PyPSA/pypsa-za) | South Africa network model | Requires separate `.7z` data bundle download |
| [PyPSA/technology-data](https://github.com/PyPSA/technology-data) | Cost/efficiency assumptions by technology and year | Input data for OPF-type work, not topology |
| [PyPSA/PyPSA](https://github.com/PyPSA/PyPSA) — `examples/` | Small toy AC/DC networks | Good for quick unit tests |

## Dynamic simulation (RMS / DAE)

| Source | What it offers | Notes |
|---|---|---|
| [dynawo/dynawo](https://github.com/dynawo/dynawo) | Validated example cases for DynaFlow, DynaWaltz, DynaSwing | RTE-maintained, MPL-2.0; roadmap includes Nordic32 and IEEE dynamic cases |
| [dynawo/dynawo-algorithms](https://github.com/dynawo/dynawo-algorithms) | Contingency analysis, voltage margin, load-increase-to-collapse | Tool, not raw data |
| [dynawo/dynawo-large-scale-validation](https://github.com/dynawo/dynawo-large-scale-validation) | Validation framework vs. legacy simulators (Astre, Hades), extensive contingency runs + Jupyter comparison pipeline | Methodology reference for validating your own RMS/DAE solver |
| [dynawo/dyn-grid-compliance-verification](https://github.com/dynawo) | Grid-code compliance verification for generators/IBR (fault-ride-through etc.) | Relevant to IBR/GFM converter work |
| [dynawo/IEC-WECC-GenericModels-Comparison](https://github.com/dynawo) | WT Type 4 generic model comparison cases | Wind turbine dynamics specifically |

---

# Tools

## Open-source power systems frameworks (readers/solvers)

| Tool | Language | Relevance |
|---|---|---|
| [PowSyBl](https://github.com/powsybl) (powsybl-core) | Java | Strongest open-source CGMES compatibility (per FullGrid evaluation); IIDM network format; CGMES conformity data bundled in |
| [pandapower](https://github.com/e2nIEE/pandapower) | Python | Matpower/PYPOWER-compatible, includes SE module |
| [PyPSA](https://github.com/PyPSA/PyPSA) | Python | Sector-coupled energy system modeling; large realistic regional models via pypsa-eur/usa/za |
| [VeraGrid](https://github.com/SanPen/VeraGrid) | Python | Reads Matpower, PSS/E RAW; has SE test infrastructure; fast large-case AC PF |
| [Dynawo](https://github.com/dynawo) | Modelica/C++ | RMS dynamic simulation (DAE, IDA/SUNDIALS-based); reads IIDM |
| PowerSystems.jl / PowerSimulations.jl (Sienna) | Julia | NREL-origin; paired with PowerSystemCaseBuilder.jl |


## MADSuite — nonlinear optimization solvers and modeling (GPU-oriented)

[madsuite-org](https://github.com/madsuite-org) ([madsuite.org](https://madsuite.org/)) is the
umbrella org for the MadNLP/ExaModels stack — a Julia-centric but increasingly
language-agnostic set of NLP/LP solvers, algebraic modeling with AD, and KKT/linear-algebra
backends, with GPU support as the organizing theme. Directly relevant to gridoxide's
OPF work: it is the closest open-source analogue to the interior-point + sparse-KKT
machinery in the OPF solver, and a source of reference results to check against.

### Solvers

| Repo | Language | Relevance |
|---|---|---|
| [MadNLP.jl](https://github.com/madsuite-org/MadNLP.jl) | Julia | Filter line-search interior-point NLP solver (an IPOPT analogue) with GPU support; modular KKT and linear-solver layers. Reference point for gridoxide's own IPM |
| [MadIPM.jl](https://github.com/madsuite-org/MadIPM.jl) | Julia | Interior-point solver for **linear** programs, GPU-capable |
| [MadNCL.jl](https://github.com/madsuite-org/MadNCL.jl) | Julia | MadNLP extension implementing Algorithm NCL; targets **infeasible or degenerate** NLPs — the failure mode AC-OPF hits on stressed cases |
| [CCOpt.jl](https://github.com/madsuite-org/CCOpt.jl) | Julia | Solver for Mathematical Programs with Complementarity Constraints (MPCCs) — the natural formulation for complementarity-style limit switching (e.g. PV→PQ, tap limits) |
| [HybridKKT.jl](https://github.com/madsuite-org/HybridKKT.jl) | Julia | Golub & Greif condensed-KKT solver inside MadNLP; the "how do I factorize an indefinite KKT system on a GPU" answer |
| [BlockDSS.jl](https://github.com/madsuite-org/BlockDSS.jl) | Julia | Block-structured (esp. block-tridiagonal) linear solvers, CPU / CUDA / ROCm, Float32 and Float64, sequential and batched variants |
| [MadDiff.jl](https://github.com/madsuite-org/MadDiff.jl) | Julia | Forward/reverse **implicit differentiation** through MadNLP/MadIPM/HybridKKT KKT systems — i.e. sensitivities of an optimum w.r.t. parameters. Conceptually adjacent to gridoxide's sensitivity work, but at the OPF level. Explicitly WIP (requires dependency forks) |
| [MadCore.jl](https://github.com/madsuite-org/MadCore.jl) | Julia | Shared core for the Mad\* solvers; no README yet |
| [MadNLPGraph.jl](https://github.com/madsuite-org/MadNLPGraph.jl) | Julia | Graph/structured-decomposition extension for MadNLP; dormant since 2021, MPL-2.0 |

### Modeling and AD

| Repo | Language | Relevance |
|---|---|---|
| [ExaModels.jl](https://github.com/madsuite-org/ExaModels.jl) | Julia | SIMD-abstracted algebraic modeling + automatic differentiation targeting GPUs; the modeling front-end MadNLP is usually paired with |
| [ExaModelsPower.jl](https://github.com/madsuite-org/ExaModelsPower.jl) | Julia | AC-OPF (and multi-period / storage) models built on ExaModels — the most directly comparable artifact to gridoxide's OPF formulations |
| [ExaPowerIO.jl](https://github.com/madsuite-org/ExaPowerIO.jl) | Julia | Minimal Matpower-format reader, deliberately narrower than PowerModels.jl. Useful as a second opinion on Matpower parsing edge cases |
| [ExaModelsAMPL.jl](https://github.com/madsuite-org/ExaModelsAMPL.jl) | Julia | Writes ExaModels models out to AMPL `.nl` — a bridge for feeding models to any `.nl`-reading solver |
| [ExaModelsExamples.jl](https://github.com/madsuite-org/ExaModelsExamples.jl) | Julia | Worked ExaModels examples |
| [DynamicNLPModels.jl](https://github.com/madsuite-org/DynamicNLPModels.jl) | Julia | NLPModels for dynamic optimization / optimal control (MPC-shaped problems) |
| [MPCCModels.jl](https://github.com/madsuite-org/MPCCModels.jl) | Julia | NLPModels.jl extension for MPCCs; the modeling side of CCOpt.jl |

### Cross-language interop (Rust-relevant)

The part of the org most useful to a non-Julia project: a plain-C ABI for handing
nonlinear programs across language boundaries, with both Julia and Python consumers.
It offers a path to expose a gridoxide-built OPF model to MadNLP (or any
NLPModels-compatible solver) without either side embedding the other's runtime.

| Repo | Language | Relevance |
|---|---|---|
| [cnlp-abi](https://github.com/madsuite-org/cnlp-abi) | C | The `cnlp.h` specification: one shared library carries N models, each publishing objective, gradient, constraints, sparse Jacobian and Lagrangian Hessian behind plain C symbols. Evaluation semantics are NLPModels.jl's, transliterated to C. **A Rust `cdylib` can implement this directly** |
| [CNLPModels.jl](https://github.com/madsuite-org/CNLPModels.jl) | Julia | Loads a `cnlp` shared library as an `NLPModels.AbstractNLPModel`, so any Julia solver can solve it |
| [cnlpmodels-py](https://github.com/madsuite-org/cnlpmodels-py) | Python | Same, for Python — ctypes + numpy, no extra runtime; works with `cyipopt` and `scipy.optimize` |
| [examodels-py](https://github.com/madsuite-org/examodels-py) | Python | Python interface to ExaModels.jl (SIMD-parallel modeling + AD, CPU and GPU) |
| [libMad](https://github.com/madsuite-org/libMad) | Julia/CMake | Metapackage compiling MadNLP (and CCOpt) into a **shared library with a C interface**, shipped as standalone release tarballs; loadable as a CasADi ≥3.8 plugin. WIP, but the route to calling MadNLP from Rust |

### Benchmark sets

| Repo | Language | Relevance |
|---|---|---|
| [COPSBenchmark.jl](https://github.com/madsuite-org/COPSBenchmark.jl) | Julia | COPS benchmark suite implemented in JuMP — standard NLP solver stress tests |
| [LuksanVlcekBenchmark.jl](https://github.com/madsuite-org/LuksanVlcekBenchmark.jl) | Julia | Lukšan–Vlček nonlinear test problems |
| [MadNLPBenchmark.jl](https://github.com/madsuite-org/MadNLPBenchmark.jl) | Julia | MadNLP's own benchmark harness |
| [MPCCbenchmark.jl](https://github.com/madsuite-org/MPCCbenchmark.jl) | Julia | MPCC benchmark problems |
| [MIPLIB.jl](https://github.com/madsuite-org/MIPLIB.jl) | Julia | MIPLIB instance access |
| [ExaModelsPower-benchmarking-archive](https://github.com/madsuite-org/ExaModelsPower-benchmarking-archive) | MATLAB | Frozen (2025-10-07) OPF benchmarking data + scripts from ExaModelsPower.jl — **published numbers to compare gridoxide OPF timings against** |

Non-code repos in the org (site, logos, and talk/workshop material — `madsuite-org.github.io`,
`madsuite-logos`, `ifac2026`, `ifac2026-workshop`, `neurips2025-mathprog-on-gpu`,
`slides-informs-2025`, `powertech2025`) are omitted here, though the workshop and slide repos
are worth a look for the current state of GPU-accelerated OPF.

## General power-systems data/format tooling encountered

- **Matpower / PYPOWER case format** — widely supported de facto standard; pandapower, VeraGrid, pglib-opf, PowSyBl-adjacent tools all read it.
- **PSS/E RAW format** — readable by VeraGrid; common utility/vendor export format worth supporting alongside CGMES.
- **UCTE-DEF** — potential fallback path if CGMES export isn't available from a data source.

---

# Books and normative references

Unlike the rest of this file, **this section is not verified against a local checkout** — it is
compiled from general knowledge of the literature, so editions and years may have moved. Titles and
authors are the load-bearing part; treat the year as a hint. Entries are grouped by the chapter of
this book they support, and each says *why it is the one to reach for* rather than simply existing.

## Power flow and general analysis

| Work | Why |
|---|---|
| Kundur, **Power System Stability and Control** (McGraw-Hill, 1994; 2nd ed. with Malik, 2022) | The single most-cited text in the field. Bought for the dynamics half, kept for its modelling chapters — the machine, exciter, governor and load models every other topic assumes |
| Milano, **Power System Modelling and Scripting** (Springer, 2010) | The closest published thing to what `src/` actually is: how to *implement* power flow, continuation, and the DAE formulation, with the Jacobian structure written out rather than waved at. The PSAT author. If one book on this list is worth reading cover to cover for gridoxide's purposes, it is this one |
| Grainger & Stevenson, **Power System Analysis** (McGraw-Hill, 1994) | The standard undergraduate treatment; per-unit, symmetrical components and the classical fault analysis, clearly |
| Expósito, Conejo & Cañizares (eds.), **Electric Energy Systems: Analysis and Operation** (CRC, 2nd ed. 2018) | The best modern single-volume survey — a chapter each by people who work on that chapter's subject |
| Stott, Jardim & Alsaç, *DC Power Flow Revisited*, IEEE Trans. Power Systems 24(3), 2009 | Not a book, but the reference for `linear::btheta` and the reason the summary table distinguishes two DC approximation variants |
| Zimmerman, Murillo-Sánchez & Thomas, *MATPOWER*, IEEE Trans. Power Systems 26(1), 2011, plus the **MATPOWER User's Manual** | The manual is an unusually honest specification of the algorithms, not just a usage guide — worth reading as a spec when a convention is ambiguous |

## Optimal power flow and optimization

| Work | Why |
|---|---|
| Wood, Wollenberg & Sheblé, **Power Generation, Operation, and Control** (Wiley, 3rd ed. 2013) | The classic for economic dispatch, OPF, unit commitment and state estimation in one volume. The reference for the two OPF extensions `plans/OPF_PLAN.md` §10 puts out of scope |
| Nocedal & Wright, **Numerical Optimization** (Springer, 2nd ed. 2006) | The method book behind `opf::nlp` — interior point, line search, filters, and the trust-region alternatives. Where to go when the barrier update or the regularization misbehaves rather than the model |
| Boyd & Vandenberghe, **Convex Optimization** (Cambridge, 2004; free PDF) | For the DC-OPF side: why the QP is convex, what the KKT certificate means, and duality as the source of locational marginal prices |
| Kirschen & Strbac, **Fundamentals of Power System Economics** (Wiley, 2nd ed. 2018) | What the prices the OPF reports actually *are*, and why a congested network produces different ones per bus |
| Conejo & Baringo, **Power System Operations** (Springer, 2018) | Security-constrained dispatch and the operational framing RAO sits inside |

## State estimation

| Work | Why |
|---|---|
| Abur & Expósito, **Power System State Estimation: Theory and Implementation** (CRC, 2004) | The reference, without competition. WLS, observability, bad-data detection by normalized residual, and the constrained formulation for zero injections — which is `src/se/` chapter by chapter |
| Monticelli, **State Estimation in Electric Power Systems: A Generalized Approach** (Kluwer, 1999) | The generalized/Hachtel formulation, and the deeper treatment of network observability |
| Kersting, **Distribution System Modeling and Analysis** (CRC, 4th ed. 2017) | The prerequisite for the unbalanced distribution SE the summary table flags as the binding gap — untransposed lines, phasing, and why the symmetric reduction stops being valid |

## Short circuit and protection

| Work | Why |
|---|---|
| **IEC 60909-0** (Short-circuit currents in three-phase a.c. systems) | The normative document `src/shortcircuit/` implements. There is no substitute: the \\(c\\)-factors, the equivalent voltage source and the impedance corrections are definitions, not derivations |
| Anderson, **Analysis of Faulted Power Systems** (IEEE Press, 1995) | The classical treatment of symmetrical components and fault boundary conditions — the derivation IEC 60909 assumes you already know |
| Das, **Power System Analysis: Short-Circuit Load Flow and Harmonics** (CRC, 3rd ed. 2018) | Covers two of the rows in the summary table at once, and is unusually practical about what the standards leave to judgement |
| Blackburn & Domin, **Protective Relaying: Principles and Applications** (CRC, 4th ed. 2014) | The starting point for the protection-coordination row, which currently has no foundation in `src/` at all |

## Dynamics: RMS, EMT and small-signal

| Work | Why |
|---|---|
| Sauer, Pai & Chow, **Power System Dynamics and Stability** (Wiley-IEEE, 2nd ed. 2017) | The cleanest development of the DAE model and the time-scale separation that makes RMS simulation legitimate. The right first book for the RMS row |
| Machowski, Lubosny, Bialek & Bumby, **Power System Dynamics: Stability and Control** (Wiley, 3rd ed. 2020) | Broader and more modern than Kundur on control and renewables, and better on *why* a phenomenon happens before the equations arrive |
| Milano, Dassios, Liu & Tzounas, **Eigenvalue Problems in Power Systems** (CRC, 2020) | The small-signal/modal row: linearizing the DAE, the generalized eigenproblem, participation factors, and the numerics of doing it at scale |
| Watson & Arrillaga, **Power Systems Electromagnetic Transients Simulation** (IET, 2003) | The EMT row: Dommel's trapezoidal companion-circuit method, travelling-wave lines, and switching |
| Dommel, **EMTP Theory Book** (BPA, 1986) | The foundational EMT document. Every EMT tool is a descendant of it, and it is written as a specification |

## Voltage stability

| Work | Why |
|---|---|
| Van Cutsem & Vournas, **Voltage Stability of Electric Power Systems** (Kluwer, 1998) | The definitive treatment, and careful about the distinction between the static loadability limit and the dynamic collapse mechanism |
| Ajjarapu, **Computational Techniques for Voltage Stability Assessment and Control** (Springer, 2006) | Continuation power flow specifically — predictor, corrector, parametrization and the behaviour at the nose. The method book for the cheapest of the missing analysis types |
| Taylor, **Power System Voltage Stability** (McGraw-Hill, 1994) | The operational view: what actually collapses, and what operators do about it |

## Harmonics and power quality

| Work | Why |
|---|---|
| Arrillaga & Watson, **Power System Harmonics** (Wiley, 2nd ed. 2003) | The harmonics row: per-frequency network models, source characterization and penetration studies |

## Reliability and adequacy

| Work | Why |
|---|---|
| Billinton & Allan, **Reliability Evaluation of Power Systems** (Plenum, 2nd ed. 1996) | Where LOLE, EENS and the rest are defined, and the Monte Carlo and analytical machinery for computing them |

## Sparse numerics

| Work | Why |
|---|---|
| Davis, **Direct Methods for Sparse Linear Systems** (SIAM, 2006) | By KLU's own author, and the book `src/klu_native/` is a translation of the ideas from. AMD ordering, BTF, the symbolic/numeric split, and why factorization reuse is the whole game for repeated solves |
| Davis & Palamadai Natarajan, *Algorithm 907: KLU, a Direct Sparse Solver for Circuit Simulation Problems*, ACM TOMS 37(3), 2010 | The paper for the specific solver; short, and the closest thing to a specification |
| Duff, Erisman & Reid, **Direct Methods for Sparse Matrices** (Oxford, 2nd ed. 2017) | The broader reference — pivoting strategies, fill-reducing orderings and the block methods |
| Tinney & Walker, *Direct Solutions of Sparse Network Equations by Optimally Ordered Triangular Factorization*, Proc. IEEE 55(11), 1967 | The paper that made large-scale power flow possible, and still the clearest statement of why ordering matters more than arithmetic |

## Remedial actions, security and capacity calculation

There is **no textbook** for this one, which is worth knowing before looking for it. The subject is
defined by regulation and by implementation rather than by a literature: the ENTSO-E **CACM**
Regulation and its capacity-calculation methodologies, the CORE and Nordic CCR methodology documents,
and [OpenRAO's own documentation](https://powsybl.readthedocs.io/projects/openrao/) — which is the
reference `plans/RAO_PLAN.md` §8.3 is gated against. Kirschen & Strbac and Conejo & Baringo above
give the economic and operational framing; the mechanics come from the methodologies.

---

## Practical recommendations

- **DSSE/MV work** → simbench (primary), VeraGrid's StateEstimation folder (reference cases), IEEE/EPRI distribution feeders (unbalanced 3-phase)
- **AC-OPF/SCOPF work** → pglib-opf (primary benchmark), pglib-opf-hvdc if MTDC scope continues
- **CGMES parser validation (`cimoxide`)** → ENTSO-E conformity suite via powsybl-core, itesla/CGMES for CIM14 legacy coverage
- **Large-scale solver benchmarking** → pypsa-eur, VeraGrid's Matpower cases (incl. continental-USA scale)
- **RMS/DAE dynamics validation** → Dynawo example cases + dynawo-large-scale-validation methodology
- **Interior-point / OPF solver validation** → MadNLP.jl and ExaModelsPower.jl as the reference implementation to diff against; COPSBenchmark.jl for generic NLP robustness; ExaModelsPower-benchmarking-archive for published timings
- **Handing gridoxide models to external solvers** → implement the `cnlp-abi` C ABI from a Rust `cdylib`, then consume via CNLPModels.jl (Julia) or cnlpmodels-py (Python/cyipopt)
- **Reading, if only one book per topic** → Milano (implementation and DAEs), Abur & Expósito (state
  estimation), Nocedal & Wright (the OPF's interior point), Davis (the sparse solve), Sauer/Pai/Chow
  (RMS dynamics), Ajjarapu (continuation power flow), IEC 60909-0 (short circuit, normative). See
  [Books and normative references](#books-and-normative-references) for why each