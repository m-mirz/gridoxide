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

## Practical recommendations

- **DSSE/MV work** → simbench (primary), VeraGrid's StateEstimation folder (reference cases), IEEE/EPRI distribution feeders (unbalanced 3-phase)
- **AC-OPF/SCOPF work** → pglib-opf (primary benchmark), pglib-opf-hvdc if MTDC scope continues
- **CGMES parser validation (`cimoxide`)** → ENTSO-E conformity suite via powsybl-core, itesla/CGMES for CIM14 legacy coverage
- **Large-scale solver benchmarking** → pypsa-eur, VeraGrid's Matpower cases (incl. continental-USA scale)
- **RMS/DAE dynamics validation** → Dynawo example cases + dynawo-large-scale-validation methodology
- **Interior-point / OPF solver validation** → MadNLP.jl and ExaModelsPower.jl as the reference implementation to diff against; COPSBenchmark.jl for generic NLP robustness; ExaModelsPower-benchmarking-archive for published timings
- **Handing gridoxide models to external solvers** → implement the `cnlp-abi` C ABI from a Rust `cdylib`, then consume via CNLPModels.jl (Julia) or cnlpmodels-py (Python/cyipopt)