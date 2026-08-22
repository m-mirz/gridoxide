//! PyO3 bindings, built into the private `gridoxide._gridoxide` compiled
//! extension module and re-exported by `python/gridoxide/__init__.py` — this
//! is a mixed Rust/Python maturin project (`pyproject.toml`'s
//! `python-source = "python"` + `module-name = "gridoxide._gridoxide"`) so
//! that `python/gridoxide/matpower.py` can ship alongside this compiled
//! extension in the same installed package. Only built via `maturin`
//! (`maturin develop --features python`), never via a plain `cargo
//! build`/`cargo test` — see the `python` feature's doc comment in
//! Cargo.toml for why.
//!
//! Mirrors `solver::PersistentSolver`'s "construct once per topology, call
//! `.solve()` as many times as needed" pattern directly, since that's
//! exactly the shape every other tool in `scripts/bench/` already exposes
//! (PGM's `PowerGridModel.calculate_power_flow`, lightsim2grid's
//! `GridModel.ac_pf`, pandapower's `pp.runpp`) — a Python caller times
//! `solve()` itself with `time.perf_counter()`, the same methodology
//! `bench_pgm.py`/`bench_lightsim2grid.py`/etc. already use, so
//! `scripts/bench/bench_gridoxide_native.py` doesn't need any bespoke
//! protocol on top of this.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::batch::{uniform_load_scaling, BatchSolver, BusOverride, Scenario};
use crate::linear::{
    dc_branches, dc_power_flow, linear_power_flow, DcApproximation, DcIslandStatus, DcOptions,
    DcSensitivity, DcSolution, LinearIslandStatus,
};
use crate::network::{build_ybus, linear_initial_guess, stamp_shunts, ShuntAdm, YBusSparse};
use crate::pgm::PgmInput;
use crate::solver::{IslandStatus, JacobianBackend, PersistentSolver, PowerFlowMethod, SolveStatus};
use crate::types::{Bus, Line, Transformer};
#[cfg(feature = "cgmes")]
use crate::cgmes::{
    cgmes_resolve_dc_converters, cgmes_to_buses_and_branches, cgmes_topological_node_bus_index, load_profiles,
};

fn parse_backend(name: &str) -> PyResult<JacobianBackend> {
    match name {
        "scalar" => Ok(JacobianBackend::Scalar),
        "block" => Ok(JacobianBackend::Block),
        #[cfg(feature = "klu")]
        "klu" => Ok(JacobianBackend::Klu),
        #[cfg(not(feature = "klu"))]
        "klu" => Err(PyValueError::new_err(
            "the 'klu' backend needs the crate's `klu` Cargo feature enabled too \
             (maturin develop --features python,klu)",
        )),
        "klu_native" => Ok(JacobianBackend::KluNative),
        #[cfg(feature = "pardiso")]
        "pardiso" => Ok(JacobianBackend::Pardiso),
        #[cfg(not(feature = "pardiso"))]
        "pardiso" => Err(PyValueError::new_err(
            "the 'pardiso' backend needs the crate's `pardiso` Cargo feature enabled too \
             (maturin develop --features python,pardiso, with MKLROOT set)",
        )),
        other => Err(PyValueError::new_err(format!(
            "unknown backend '{other}', expected 'scalar', 'block', 'klu', 'klu_native', or 'pardiso'"
        ))),
    }
}

fn parse_method(name: &str) -> PyResult<PowerFlowMethod> {
    match name {
        "newton_raphson" => Ok(PowerFlowMethod::NewtonRaphson),
        "dc" => Ok(PowerFlowMethod::Dc),
        "linear_impedance" => Ok(PowerFlowMethod::LinearImpedance),
        other => Err(PyValueError::new_err(format!(
            "unknown method '{other}', expected 'newton_raphson', 'dc', or 'linear_impedance'"
        ))),
    }
}

#[cfg(feature = "cgmes")]
fn parse_retention(name: &str) -> PyResult<crate::topology::RetentionPolicy> {
    use crate::topology::RetentionPolicy;
    match name {
        "none" => Ok(RetentionPolicy::MergeAll),
        "all" => Ok(RetentionPolicy::RetainAll),
        "busbar_adjacent" => Ok(RetentionPolicy::RetainAdjacentToBusbar),
        other => Err(PyValueError::new_err(format!(
            "unknown retain '{other}', expected 'none', 'busbar_adjacent' or 'all'"
        ))),
    }
}

fn parse_dc_approximation(name: &str) -> PyResult<DcApproximation> {
    match name {
        "ignore_r" => Ok(DcApproximation::IgnoreR),
        "ignore_g" => Ok(DcApproximation::IgnoreG),
        other => Err(PyValueError::new_err(format!(
            "unknown dc_approximation '{other}', expected 'ignore_r' (b = 1/x) or \
             'ignore_g' (b = x/(r²+x²))"
        ))),
    }
}

/// One scenario's outcome from [`PowerFlowModel::solve_batch`]. Carries
/// voltages plus the convergence detail `solver::SolveStats` now returns
/// instead of printing.
#[pyclass]
struct BatchResult {
    /// Per-bus voltage magnitude in per-unit, in node order.
    #[pyo3(get)]
    voltage_mag: Vec<f64>,
    /// Per-bus voltage angle in radians, in node order.
    #[pyo3(get)]
    voltage_ang: Vec<f64>,
    /// Newton iterations this scenario used.
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    converged: bool,
    /// Max |mismatch| at the final iteration.
    #[pyo3(get)]
    max_mismatch: f64,
}

/// One scenario's outcome from [`StateEstimationModel::solve_batch`].
#[pyclass]
struct SeBatchOutcome {
    /// Per-bus voltage magnitude in per-unit, in node order.
    #[pyo3(get)]
    voltage_mag: Vec<f64>,
    /// Per-bus voltage angle in radians, in node order.
    #[pyo3(get)]
    voltage_ang: Vec<f64>,
    #[pyo3(get)]
    iterations: usize,
    /// `J(x) = ½ rᵀWr` at the estimate.
    #[pyo3(get)]
    objective: f64,
    #[pyo3(get)]
    converged: bool,
}

/// A power flow model loaded from a PGM-format JSON file, solved via a
/// persistent `solver::PersistentSolver` — repeated `solve()` calls on the
/// same `PowerFlowModel` reuse cached symbolic factorization exactly the
/// way `solver::PersistentSolver` documents.
///
/// `unsendable`: with the `klu` feature, `PersistentSolver` can hold a
/// `sparse_klu::KluRealSystem`, which owns raw `*mut klu_symbolic`/`*mut
/// klu_numeric` pointers into SuiteSparse's own (not thread-safe for
/// concurrent access) state — not `Send`. `unsendable` tells PyO3 this type
/// only ever runs on the thread that created it (true for this project's
/// single-threaded benchmark usage) rather than asserting `Send` ourselves,
/// which would require verifying SuiteSparse's cross-thread-move safety —
/// not something to claim without being sure. Deliberately doesn't touch
/// `sparse_klu.rs`'s own raw-pointer RAII wrapper to stay conservative. With
/// the `pardiso` feature, `PersistentSolver` can likewise hold a
/// `sparse_pardiso::PardisoRealSystem` — PARDISO's own `pt` handle has the
/// same not-safe-for-concurrent-use profile (documented in multiple Intel
/// community threads, including interactions with its internal METIS
/// reordering), so this same `unsendable` already covers it too.
#[pyclass(unsendable)]
struct PowerFlowModel {
    buses_template: Vec<Bus>,
    buses: Vec<Bus>,
    ybus: YBusSparse,
    /// Kept alongside the Y-bus because the DC method is formulated per
    /// branch — it needs each branch's own `x` and `tap`, which the Y-bus has
    /// already summed away — and because the sensitivity factors need them
    /// too. Newton and the constant-admittance method use only `ybus`.
    lines: Vec<Line>,
    transformers: Vec<Transformer>,
    /// Kept for the same reason as the branch lists: a contingency sweep
    /// rebuilds the Y-bus per scenario and has to re-stamp them.
    shunts: Vec<ShuntAdm>,
    solver: PersistentSolver,
    backend: JacobianBackend,
    method: PowerFlowMethod,
    dc_opts: DcOptions,
    /// The last DC solve's result, cleared whenever a non-DC `solve()` runs so
    /// the accessors cannot serve a stale answer from a previous method.
    dc_solution: Option<DcSolution>,
    /// Built on first use and reused, since it holds one factorization per
    /// island. Invalidated by `reset()` along with the Newton factorization.
    sensitivity: Option<DcSensitivity>,
    /// Thread count paired with the `BatchSolver` built for it. Cached so
    /// repeated `solve_batch` calls at one thread count reuse a single rayon
    /// pool instead of respawning workers per call.
    batch: Option<(usize, BatchSolver)>,
    tol: f64,
    max_iter: usize,
    /// The node-breaker network, when the model was built with
    /// `topology="node_breaker"`. Owns the switch-to-branch mapping, which is
    /// what makes a switch addressable by identity rather than by index
    /// arithmetic.
    #[cfg(feature = "cgmes")]
    node_breaker: Option<crate::switches::NodeBreakerNetwork>,
    /// `TopologicalNode` mrid -> bus index, populated by `from_cgmes` (empty
    /// for `from_pgm_json`, which has no mrid concept at all) — lets Python
    /// callers look up a specific bus's solved voltage by mrid to compare
    /// against a CGMES fixture's own published `SvVoltage`, without needing
    /// to separately re-derive `cgmes::cgmes_topological_node_bus_index`'s
    /// own post-switch-merge index remapping themselves.
    tn_bus_index: std::collections::HashMap<String, usize>,
}

#[pymethods]
impl PowerFlowModel {
    /// Loads a PGM-format JSON file (the same format `examples/bench_network.rs`
    /// and `matpower_to_pgm.py`/`convert_pandapower_case.py` produce).
    ///
    /// `backend` is `"scalar"` (default), `"block"`, `"klu"` (needs the
    /// crate's `klu` feature built in too), `"klu_native"`, or `"pardiso"`
    /// (needs the crate's `pardiso` feature built in too, plus `MKLROOT`
    /// set at build time).
    /// `s_base_va`/`freq_hz` match
    /// `pgm::pgm_to_buses_and_branches`'s own defaults used elsewhere in
    /// this project (1e6 VA, 50 Hz) unless overridden.
    ///
    /// `method` is `"newton_raphson"` (default), `"dc"` (real Bθ) or
    /// `"linear_impedance"` (the complex constant-admittance linearization,
    /// power-grid-model's `CalculationMethod.linear`). `tol`/`max_iter`/
    /// `backend` apply to Newton only — the other two are direct solves.
    /// `dc_approximation` is `"ignore_r"` (default, `b = 1/x`) or
    /// `"ignore_g"` (`b = x/(r²+x²)`); it also governs the sensitivity
    /// factors, which are available whatever `method` is set to.
    #[staticmethod]
    #[pyo3(signature = (
        path,
        backend="scalar",
        tol=1e-6,
        max_iter=20,
        s_base_va=1e6,
        freq_hz=50.0,
        method="newton_raphson",
        dc_approximation="ignore_r",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_pgm_json(
        path: &str,
        backend: &str,
        tol: f64,
        max_iter: usize,
        s_base_va: f64,
        freq_hz: f64,
        method: &str,
        dc_approximation: &str,
    ) -> PyResult<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| PyRuntimeError::new_err(format!("reading {path}: {e}")))?;
        let input: PgmInput = serde_json::from_str(&raw)
            .map_err(|e| PyValueError::new_err(format!("parsing {path} as PGM JSON: {e}")))?;
        // `shunt` entries have to be converted before `pgm_to_buses_and_branches`
        // consumes `input`, then stamped onto the Y-bus diagonal — a PGM `shunt`
        // is a self-admittance, not a branch, so `build_ybus` never sees it.
        let id_to_idx = crate::pgm::node_id_to_idx(&input);
        let shunts = crate::pgm::pgm_shunts_1ph(&input, &id_to_idx, s_base_va);
        let (buses_template, lines, transformers) =
            crate::pgm::pgm_to_buses_and_branches(input, s_base_va, freq_hz);
        let mut ybus = build_ybus(buses_template.len(), &lines, &transformers);
        stamp_shunts(&mut ybus, &shunts);
        let ybus = ybus.finish();
        let backend = parse_backend(backend)?;
        Ok(Self {
            buses: buses_template.clone(),
            buses_template,
            ybus,
            lines,
            transformers,
            shunts,
            solver: PersistentSolver::new(backend),
            backend,
            method: parse_method(method)?,
            dc_opts: DcOptions {
                approximation: parse_dc_approximation(dc_approximation)?,
                ..DcOptions::default()
            },
            dc_solution: None,
            sensitivity: None,
            #[cfg(feature = "cgmes")]
            node_breaker: None,
            batch: None,
            tol,
            max_iter,
            tn_bus_index: std::collections::HashMap::new(),
        })
    }

    /// Loads a CGMES profile bundle (any number of EQ/EQBD/SSH/TP/SV files,
    /// in any order — `cgmes::load_profiles` merges them by mRID regardless
    /// of which file each element came from, the same way `references/`'s
    /// own CGMES-capable tool, powsybl, merges a profile set). Resolves any
    /// HVDC converters found (`cgmes_resolve_dc_converters`) into fixed AC
    /// bus injections before the model is built, exactly like every native
    /// CGMES test in `tests/cgmes_*_test.rs` already does — this binding is
    /// a thin wrapper around that same conversion pipeline, not a separate
    /// code path, so results match the Rust-side tests bus-for-bus.
    ///
    /// Only built when the crate's `cgmes` feature is enabled too
    /// (`maturin develop --features python,cgmes`) — see
    /// `docs/src/reference/provenance.md` for why that dependency is opt-in.
    #[cfg(feature = "cgmes")]
    #[staticmethod]
    #[pyo3(signature = (
        paths,
        backend="scalar",
        tol=1e-6,
        max_iter=20,
        s_base_va=100e6,
        method="newton_raphson",
        dc_approximation="ignore_r",
        topology="bus_branch",
        retain="none",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_cgmes(
        paths: Vec<String>,
        backend: &str,
        tol: f64,
        max_iter: usize,
        s_base_va: f64,
        method: &str,
        dc_approximation: &str,
        topology: &str,
        retain: &str,
    ) -> PyResult<Self> {
        let path_refs: Vec<&std::path::Path> = paths.iter().map(std::path::Path::new).collect();
        let ds = load_profiles(&path_refs)
            .map_err(|e| PyRuntimeError::new_err(format!("decoding CGMES profiles: {e}")))?;

        let (buses_template, lines, transformers, shunts, tn_bus_index, node_breaker) =
            match topology {
                "bus_branch" => {
                    if retain != "none" {
                        return Err(PyValueError::new_err(
                            "retain= only applies to topology=\"node_breaker\"; the bus-branch \
                             view has already merged every switch away",
                        ));
                    }
                    let (mut buses, lines, transformers, shunts) =
                        cgmes_to_buses_and_branches(&ds, s_base_va).map_err(|e| {
                            PyRuntimeError::new_err(format!("converting CGMES model: {e}"))
                        })?;
                    cgmes_resolve_dc_converters(&ds, &mut buses, s_base_va).map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "resolving CGMES HVDC converters: {e}"
                        ))
                    })?;
                    let tn = cgmes_topological_node_bus_index(&ds).map_err(|e| {
                        PyRuntimeError::new_err(format!("resolving CGMES bus index: {e}"))
                    })?;
                    (buses, lines, transformers, shunts, tn, None)
                }
                "node_breaker" => {
                    let policy = parse_retention(retain)?;
                    let net = crate::cgmes::cgmes_node_breaker_to_buses_and_branches(
                        &ds,
                        s_base_va,
                        &policy,
                        crate::switches::SwitchTreatment::Regularize,
                    )
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("converting CGMES model: {e}"))
                    })?;
                    // HVDC converters resolve by `TopologicalNode`, which the
                    // node-breaker path indexes too — but its bus numbering is
                    // its own, so the mapping is left empty rather than handing
                    // back indices from the other view.
                    (
                        net.buses.clone(),
                        net.lines.clone(),
                        net.transformers.clone(),
                        net.shunts.clone(),
                        std::collections::HashMap::new(),
                        Some(net),
                    )
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown topology '{other}', expected 'bus_branch' or 'node_breaker'"
                    )))
                }
            };

        let mut ybus = build_ybus(buses_template.len(), &lines, &transformers);
        stamp_shunts(&mut ybus, &shunts);
        let ybus = ybus.finish();
        let backend = parse_backend(backend)?;
        Ok(Self {
            buses: buses_template.clone(),
            buses_template,
            ybus,
            lines,
            transformers,
            shunts,
            solver: PersistentSolver::new(backend),
            backend,
            method: parse_method(method)?,
            dc_opts: DcOptions {
                approximation: parse_dc_approximation(dc_approximation)?,
                ..DcOptions::default()
            },
            dc_solution: None,
            sensitivity: None,
            node_breaker,
            batch: None,
            tol,
            max_iter,
            tn_bus_index,
        })
    }

    /// Number of buses (nodes), including the virtual slack bus each active
    /// `source` adds — matches `bench_network.rs`'s printed `nodes=`.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.buses.len()
    }

    /// The bus index for a given `TopologicalNode` mrid, or `None` if this
    /// model wasn't built via `from_cgmes` or the mrid isn't a bus in it.
    /// Lets a Python caller compare `voltage_mag()[idx]` against a
    /// fixture's own published `SvVoltage` for a specific mrid.
    ///
    /// `mrid` must include the leading `_` CGMES's own `rdf:ID`/`rdf:about`
    /// values always carry (e.g. `"_1234abcd-..."`, not `"1234abcd-..."`) —
    /// `CimElement::mrid()` (what `cgmes_topological_node_bus_index` keys
    /// this map by) returns that raw XML attribute value unmodified, so a
    /// stripped-underscore mrid would silently never match here. A
    /// `TopologicalNode`'s `SvVoltage.TopologicalNode` reference in the SV
    /// profile already carries this same underscore-prefixed form, so a
    /// caller comparing against published values doesn't need to strip or
    /// add one either way.
    fn bus_index_for_mrid(&self, mrid: &str) -> Option<usize> {
        self.tn_bus_index.get(mrid).copied()
    }

    /// Solves from a fresh flat/linear-initial-guess start, reusing this
    /// model's `PersistentSolver` cached factorization from any previous
    /// `solve()` call. Every disconnected component of the network is
    /// solved in this same call, not just the largest one (see
    /// `solver::PersistentSolver::solve`'s own doc comment); a sourceless
    /// component gets a fixed zero-voltage placeholder rather than raising
    /// an error, the same non-error treatment de-energized PGM nodes
    /// already got before this method had any island-level detail at all.
    /// Raises `RuntimeError` only if some component's own Newton-Raphson
    /// genuinely failed — didn't converge within `max_iter` iterations, or
    /// hit a singular Jacobian.
    fn solve(&mut self) -> PyResult<()> {
        match self.method {
            PowerFlowMethod::Dc => return self.solve_dc(),
            PowerFlowMethod::LinearImpedance => return self.solve_linear_impedance(),
            PowerFlowMethod::NewtonRaphson => {}
        }
        self.dc_solution = None;
        self.buses = self.buses_template.clone();
        linear_initial_guess(&mut self.buses, &self.ybus);
        let islands = self.solver.solve(&mut self.buses, &self.ybus, self.tol, self.max_iter);
        for island in &islands {
            match island.status {
                IslandStatus::Converged | IslandStatus::NoReferenceBus | IslandStatus::AmbiguousReferenceBus => {}
                IslandStatus::MaxIterationsReached => return Err(PyRuntimeError::new_err(format!(
                    "power flow did not converge within {} iterations (component with buses {:?})",
                    self.max_iter, island.bus_indices
                ))),
                IslandStatus::Singular => return Err(PyRuntimeError::new_err(format!(
                    "Jacobian is singular (component with buses {:?})", island.bus_indices
                ))),
            }
        }
        Ok(())
    }

    /// Discards cached symbolic factorization — call before the next
    /// `solve()` if the topology (not just bus values) has changed since
    /// this model was constructed or last reset. See
    /// `solver::PersistentSolver::reset`'s doc comment.
    ///
    /// Also drops the cached DC sensitivity factorization, for the same
    /// reason: it is built from the topology and would otherwise answer for
    /// a network that no longer exists.
    fn reset(&mut self) {
        self.solver.reset();
        self.sensitivity = None;
    }

    /// Runs the DC (Bθ) solve. Reached through `solve()` when the model was
    /// built with `method="dc"`; exposed directly so a model built for
    /// Newton can take a DC reading without being rebuilt.
    ///
    /// Raises `RuntimeError` only if some island's reduced susceptance matrix
    /// was singular. An island with no reference bus is *not* an error — its
    /// buses are pinned to zero, the same non-error treatment `solve()` gives
    /// a sourceless component.
    fn solve_dc(&mut self) -> PyResult<()> {
        self.buses = self.buses_template.clone();
        let solution =
            dc_power_flow(&mut self.buses, &self.lines, &self.transformers, self.dc_opts);
        let singular: Vec<&[usize]> = solution
            .islands
            .iter()
            .filter(|i| i.status == DcIslandStatus::Singular)
            .map(|i| i.bus_indices.as_slice())
            .collect();
        if !singular.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "DC susceptance matrix is singular (component(s) with buses {singular:?})"
            )));
        }
        self.dc_solution = Some(solution);
        Ok(())
    }

    /// Runs the constant-admittance linearization (power-grid-model's
    /// `CalculationMethod.linear`). Reached through `solve()` when the model
    /// was built with `method="linear_impedance"`.
    fn solve_linear_impedance(&mut self) -> PyResult<()> {
        self.dc_solution = None;
        self.buses = self.buses_template.clone();
        let report = linear_power_flow(&mut self.buses, &self.ybus);
        let singular: Vec<&[usize]> = report
            .islands
            .iter()
            .filter(|i| i.status == LinearIslandStatus::Singular)
            .map(|i| i.bus_indices.as_slice())
            .collect();
        if !singular.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "linearized system is singular (component(s) with buses {singular:?})"
            )));
        }
        Ok(())
    }

    /// Active power entering each branch at its `from` terminal, per-unit,
    /// indexed by the flat branch index (lines first, then transformers).
    ///
    /// Only available after a DC solve — the other methods produce complex
    /// flows, which `branch_flow::terminal_flow` computes from the solved
    /// voltages rather than returning here.
    fn branch_flow_p(&self) -> PyResult<Vec<f64>> {
        Ok(self.require_dc()?.branch_p.clone())
    }

    /// Per-island `(bus_indices, slack_pickup)` from the last DC solve, with
    /// pickup in per-unit. DC is lossless, so an island's pickup is exactly
    /// the negation of everything else it contains.
    fn dc_slack_pickup(&self) -> PyResult<Vec<(Vec<usize>, f64)>> {
        Ok(self
            .require_dc()?
            .islands
            .iter()
            .map(|i| (i.bus_indices.clone(), i.slack_pickup))
            .collect())
    }

    /// `max |Σ(flows out of bus) − P_bus|` over the last DC solve. Round-off
    /// (~1e-15) on a healthy network; a large value means `B` was
    /// ill-conditioned, most often from clamped zero-impedance branches.
    fn dc_max_residual(&self) -> PyResult<f64> {
        Ok(self.require_dc()?.max_residual)
    }

    /// `∂P_branch/∂P_bus` over every branch, for injection at `bus` with this
    /// island's reference absorbing it. `None` if `bus` sits in an island
    /// with no reference bus.
    ///
    /// Independent of `method`: the factors come from the topology, not from
    /// any particular solve. The factorization behind them is built on first
    /// use and reused until `reset()`.
    fn ptdf_column(&mut self, bus: usize) -> PyResult<Option<Vec<f64>>> {
        if bus >= self.buses.len() {
            return Err(PyValueError::new_err(format!(
                "bus {bus} is out of range (0..{})",
                self.buses.len()
            )));
        }
        Ok(self.require_sensitivity()?.ptdf_column(bus))
    }

    /// `∂P_branch/∂P_bus` over every bus, for one branch — one solve rather
    /// than one per bus, since the reduced susceptance matrix is symmetric.
    fn ptdf_row(&mut self, branch: usize) -> PyResult<Option<Vec<f64>>> {
        self.check_branch(branch)?;
        Ok(self.require_sensitivity()?.ptdf_row(branch))
    }

    /// The fraction of `branch`'s pre-outage flow that lands on each other
    /// branch when it trips. `None` if the branch is radial — removing it
    /// islands the network, so no redistribution factors exist.
    fn lodf_column(&mut self, branch: usize) -> PyResult<Option<Vec<f64>>> {
        self.check_branch(branch)?;
        Ok(self.require_sensitivity()?.lodf_column(branch))
    }

    /// Whether removing `branch` would disconnect the network.
    fn is_radial(&mut self, branch: usize) -> PyResult<bool> {
        self.check_branch(branch)?;
        Ok(self.require_sensitivity()?.is_radial(branch))
    }

    /// Branch flows after `branch` trips, given the flows before it did — the
    /// N-1 screening primitive, one solve and no re-solve of the network.
    ///
    /// `base_flows` defaults to the last DC solve's own flows, so the common
    /// case is `model.solve(); model.outage_flows(7)`. Pass an explicit vector
    /// to screen a contingency against some other operating point.
    ///
    /// `None` if the branch is radial: removing it islands the network, so its
    /// power has nowhere to redistribute to.
    #[pyo3(signature = (branch, base_flows=None))]
    fn outage_flows(
        &mut self,
        branch: usize,
        base_flows: Option<Vec<f64>>,
    ) -> PyResult<Option<Vec<f64>>> {
        self.check_branch(branch)?;
        let base = match base_flows {
            Some(f) => f,
            None => self.require_dc()?.branch_p.clone(),
        };
        let n = self.lines.len() + self.transformers.len();
        if base.len() != n {
            return Err(PyValueError::new_err(format!(
                "base_flows has {} entries, expected one per branch ({n})",
                base.len()
            )));
        }
        Ok(self.require_sensitivity()?.outage_flows(&base, branch))
    }

    /// Branch flows after **every** branch in `branches` trips at once — the
    /// N-2/N-k generalization of `outage_flows`.
    ///
    /// Not obtainable by applying `outage_flows` repeatedly: each single-branch
    /// factor was computed on the intact network, so chaining them ignores how
    /// the outages interact. One `k × k` solve on top of `k` triangular solves
    /// gives the exact answer.
    ///
    /// `None` if removing the whole set would disconnect the network (see
    /// `is_breaking_set`), or if an index is repeated or out of range. An empty
    /// set returns the base flows unchanged.
    #[pyo3(signature = (branches, base_flows=None))]
    fn multi_outage_flows(
        &mut self,
        branches: Vec<usize>,
        base_flows: Option<Vec<f64>>,
    ) -> PyResult<Option<Vec<f64>>> {
        for &b in &branches {
            self.check_branch(b)?;
        }
        let base = match base_flows {
            Some(f) => f,
            None => self.require_dc()?.branch_p.clone(),
        };
        let n = self.lines.len() + self.transformers.len();
        if base.len() != n {
            return Err(PyValueError::new_err(format!(
                "base_flows has {} entries, expected one per branch ({n})",
                base.len()
            )));
        }
        Ok(self.require_sensitivity()?.multi_outage_flows(&base, &branches))
    }

    /// Runs an AC N-1/N-k contingency sweep: one entry in `contingencies` per
    /// scenario, each a list of flat branch indices to take out of service.
    ///
    /// Returns `(status, voltage_mag, voltage_ang)` per scenario, in order.
    /// `status` is `"converged"`, `"max_iterations"` or `"singular"` — a
    /// contingency that leaves an unsolvable network is a screening result, not
    /// an error, so it does not raise.
    ///
    /// The symbolic factorization is shared across every contingency that
    /// leaves the network connected; those that sever it fall back to a full
    /// rebuild, and their orphaned buses come back pinned to zero rather than
    /// as a spurious singular solve.
    #[pyo3(signature = (contingencies, threads=None))]
    fn solve_contingencies(
        &mut self,
        contingencies: Vec<Vec<usize>>,
        threads: Option<usize>,
    ) -> PyResult<Vec<(String, Vec<f64>, Vec<f64>)>> {
        let scenarios: Vec<Scenario> = contingencies
            .into_iter()
            .map(|branch_outages| Scenario { bus_overrides: Vec::new(), branch_outages })
            .collect();
        let batch = match threads {
            Some(t) => BatchSolver::with_threads(self.backend, t.max(1))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            None => BatchSolver::new(self.backend),
        };
        let reports = batch
            .solve_contingencies(
                &self.buses_template,
                &self.lines,
                &self.transformers,
                &self.shunts,
                &scenarios,
                self.tol,
                self.max_iter,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(reports
            .into_iter()
            .map(|r| {
                let status = match r.stats.status {
                    SolveStatus::Converged => "converged",
                    SolveStatus::MaxIterationsReached => "max_iterations",
                    SolveStatus::Singular => "singular",
                };
                let vm = r.buses.iter().map(|b| b.voltage_mag).collect();
                let va = r.buses.iter().map(|b| b.voltage_ang).collect();
                (status.to_string(), vm, va)
            })
            .collect())
    }

    /// Whether removing every branch in `branches` at once would disconnect the
    /// network — the multi-branch analogue of `is_radial`.
    fn is_breaking_set(&mut self, branches: Vec<usize>) -> PyResult<bool> {
        for &b in &branches {
            self.check_branch(b)?;
        }
        Ok(self.require_sensitivity()?.is_breaking_set(&branches))
    }

    /// Branch-flow response to an arbitrary per-bus injection pattern, in
    /// per-unit. The primitive `ptdf_column` is a special case of; within
    /// each island whatever does not sum to zero is absorbed at its
    /// reference.
    fn transfer_factors(&mut self, injections: Vec<f64>) -> PyResult<Vec<f64>> {
        if injections.len() != self.buses.len() {
            return Err(PyValueError::new_err(format!(
                "injections has {} entries, expected one per bus ({})",
                injections.len(),
                self.buses.len()
            )));
        }
        self.require_sensitivity()?
            .transfer_factors(&injections)
            .ok_or_else(|| PyRuntimeError::new_err("the reduced susceptance matrix is singular"))
    }

    /// Every retained switch: `(switch_id, label, kind, bus_from, bus_to,
    /// open, branch_index)`.
    ///
    /// `switch_id` indexes the source topology's own switch list, so it is
    /// stable across retention policies and is what `set_switch` takes.
    /// `branch_index` is the flat branch index the switch occupies, which is
    /// what `lodf_column`, `outage_flows` and a contingency scenario all want —
    /// a switch is an ordinary branch to everything downstream.
    ///
    /// Empty unless the model was built with `topology="node_breaker"`.
    #[cfg(feature = "cgmes")]
    fn switches(&self) -> Vec<(usize, String, String, usize, usize, bool, usize)> {
        let Some(net) = self.node_breaker.as_ref() else { return Vec::new() };
        net.view
            .retained()
            .iter()
            .enumerate()
            .filter_map(|(n, r)| {
                let branch = net.branch_of_retained(n)?;
                let kind = net
                    .switch_kind(r.switch)
                    .map(|k| format!("{k:?}"))
                    .unwrap_or_else(|| "Unknown".to_string());
                Some((
                    r.switch.0,
                    net.switch_label(r.switch),
                    kind,
                    r.buses[0].0,
                    r.buses[1].0,
                    // The *current* position, read from the stamped branch —
                    // `RetainedSwitch::open` is the state the view was built
                    // with and does not follow `set_switch`.
                    net.is_switch_open(r.switch).unwrap_or(r.open),
                    branch,
                ))
            })
            .collect()
    }

    /// Opens or closes a retained switch, rebuilding the admittance matrix.
    ///
    /// The Y-bus *values* change; its sparsity pattern does not, because an
    /// open switch keeps its structural entries at zero. So the symbolic
    /// factorization survives and only the Jacobian's cached admittances are
    /// dropped — which is why this calls `invalidate_admittances` rather than
    /// `reset`, and why a switching campaign costs one symbolic factorization
    /// rather than one per state.
    ///
    /// Raises if the model is not node-breaker, or the switch was not retained
    /// (or was degenerate — both ends already on one bus, so its position
    /// cannot affect connectivity).
    #[cfg(feature = "cgmes")]
    fn set_switch(&mut self, switch: usize, open: bool) -> PyResult<()> {
        let Some(net) = self.node_breaker.as_mut() else {
            return Err(PyRuntimeError::new_err(
                "this model is not node-breaker; build it with topology=\"node_breaker\"",
            ));
        };
        if !net.set_switch_open(crate::topology::SwitchIdx(switch), open) {
            return Err(PyValueError::new_err(format!(
                "switch {switch} is not retained in this view, or is degenerate"
            )));
        }
        self.transformers = net.transformers.clone();

        let mut ybus = build_ybus(self.buses_template.len(), &self.lines, &self.transformers);
        stamp_shunts(&mut ybus, &self.shunts);
        self.ybus = ybus.finish();

        // A full reset, not `invalidate_admittances`, and the distinction is
        // subtle enough to be worth stating: the *Y-bus* pattern really is
        // unchanged — an open switch keeps its structural entries at zero — but
        // the *Jacobian* pattern need not be. Opening a switch can sever part
        // of the network, `mark_unreferenced_islands` then pins those buses to
        // `Slack`, and the unknown count changes with them. A caller that knows
        // its switch cannot island anything may keep the factorization with
        // `invalidate_admittances`; this binding cannot know that.
        self.solver.reset();
        self.sensitivity = None;
        self.dc_solution = None;
        Ok(())
    }

    /// Active power entering each retained switch at its `from` terminal, in
    /// per-unit, in `switches()` order.
    ///
    /// Uses the voltages from the last `solve()`.
    #[cfg(feature = "cgmes")]
    fn switch_flow_p(&self) -> PyResult<Vec<f64>> {
        let Some(net) = self.node_breaker.as_ref() else {
            return Err(PyRuntimeError::new_err(
                "this model is not node-breaker; build it with topology=\"node_breaker\"",
            ));
        };
        let v = crate::branch_flow::bus_voltages(&self.buses);
        Ok(net
            .switch_branches()
            .into_iter()
            .map(|(switch, _)| net.switch_flow(switch, &v).map(|(p, _)| p).unwrap_or(0.0))
            .collect())
    }

    /// Total branch count (lines plus transformers) — the length of every
    /// branch-indexed vector this class returns.
    #[getter]
    fn n_branches(&self) -> usize {
        self.lines.len() + self.transformers.len()
    }

    /// Solves many injection scenarios over this model's topology in
    /// parallel, returning one `BatchResult` per scenario **in scenario
    /// order**. See `batch::BatchSolver`.
    ///
    /// `scenarios` is a list of per-scenario override lists, each entry a
    /// `(bus_index, p_spec, q_spec)` triple in per-unit. Buses not mentioned
    /// keep the model's own loaded values.
    ///
    /// `threads` defaults to rayon's global thread count. The underlying
    /// pool is cached per thread count, so repeated calls at one setting
    /// never respawn workers.
    ///
    /// A scenario that fails to converge is reported via
    /// `BatchResult.converged`, not raised — unlike `solve()`, which raises.
    /// Divergent scenarios are expected in contingency and Monte Carlo
    /// sweeps and must not abort the batch.
    ///
    /// **GIL:** this holds the GIL for the whole batch. `PowerFlowModel` is
    /// `unsendable`, so `Python::allow_threads` (whose closure must be
    /// `Send`) is unavailable. The rayon workers never touch Python, so this
    /// is correct — it only blocks *other* Python threads meanwhile, which
    /// is acceptable for the benchmark/analysis use this binding exists for.
    #[pyo3(signature = (scenarios, threads=None))]
    fn solve_batch(
        &mut self,
        scenarios: Vec<Vec<(usize, f64, f64)>>,
        threads: Option<usize>,
    ) -> PyResult<Vec<BatchResult>> {
        let scenarios: Vec<Scenario> = scenarios
            .into_iter()
            .map(|overrides| {
                Scenario::new(
                    overrides
                        .into_iter()
                        .map(|(bus, p, q)| BusOverride::new(bus).p(p).q(q))
                        .collect(),
                )
            })
            .collect();
        self.run_batch(scenarios, threads)
    }

    /// `solve_batch` for the common time-series/QSTS shape: each entry of
    /// `scales` becomes one scenario with every bus's `p_spec`/`q_spec`
    /// multiplied by that factor. Drives `scripts/bench/bench_batch.py`.
    #[pyo3(signature = (scales, threads=None))]
    fn solve_batch_scaled(
        &mut self,
        scales: Vec<f64>,
        threads: Option<usize>,
    ) -> PyResult<Vec<BatchResult>> {
        let scenarios: Vec<Scenario> = scales
            .into_iter()
            .map(|f| uniform_load_scaling(&self.buses_template, f))
            .collect();
        self.run_batch(scenarios, threads)
    }

    /// Workers the next `solve_batch` call would use with `threads=None`.
    #[staticmethod]
    fn default_threads() -> usize {
        rayon::current_num_threads()
    }

    /// The Y-bus as `(rows, cols, g, b)` triplets, one entry per stored
    /// nonzero, row-major with columns ascending within a row.
    ///
    /// Exists so an external reimplementation — `scripts/bench/jax_oracle.py`
    /// — can consume the *exact* admittance matrix this solver uses rather
    /// than rebuilding one from the same input file. Without that, a
    /// disagreement between the two could equally be a model-conversion
    /// difference (tap ratios, shunt stamping, switch merging) as a solver
    /// difference, and the comparison would prove nothing.
    fn ybus_triplets(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
        let n = self.ybus.n();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut g = Vec::new();
        let mut b = Vec::new();
        for i in 0..n {
            for &(j, y) in self.ybus.row(i) {
                rows.push(i);
                cols.push(j);
                g.push(y.re);
                b.push(y.im);
            }
        }
        (rows, cols, g, b)
    }

    /// Per-bus `(bus_type, p_spec, q_spec)` in node order, where `bus_type`
    /// is 0 = Slack, 1 = PV, 2 = PQ. Injections are net (generation minus
    /// load) in per-unit, matching `types::Bus`.
    ///
    /// `u32` rather than `u8` deliberately: PyO3 maps `Vec<u8>` to Python
    /// `bytes`, not a list of ints, which silently turns `np.asarray(kinds)`
    /// into a 0-d array and makes every downstream mask wrong instead of
    /// raising.
    fn bus_spec(&self) -> (Vec<u32>, Vec<f64>, Vec<f64>) {
        let mut kinds = Vec::with_capacity(self.buses_template.len());
        let mut p = Vec::with_capacity(self.buses_template.len());
        let mut q = Vec::with_capacity(self.buses_template.len());
        for bus in &self.buses_template {
            kinds.push(match bus.bus_type {
                crate::types::BusType::Slack => 0u32,
                crate::types::BusType::PV => 1,
                crate::types::BusType::PQ => 2,
            });
            p.push(bus.p_spec);
            q.push(bus.q_spec);
        }
        (kinds, p, q)
    }

    /// `(voltage_mag, voltage_ang)` after `network::linear_initial_guess`,
    /// i.e. the exact state this model's Newton loop starts its first
    /// iteration from. Lets the oracle begin from the same point rather than
    /// a flat start, so a mismatch cannot be blamed on landing in a different
    /// basin.
    fn initial_guess(&self) -> (Vec<f64>, Vec<f64>) {
        let mut buses = self.buses_template.clone();
        linear_initial_guess(&mut buses, &self.ybus);
        (
            buses.iter().map(|b| b.voltage_mag).collect(),
            buses.iter().map(|b| b.voltage_ang).collect(),
        )
    }

    /// Number of voltage-dependent ZIP terms on each bus.
    ///
    /// The oracle models constant-power injections only. It calls this to
    /// *assert* every bus is pure constant-power rather than silently
    /// producing a wrong answer on a network where `effective_injection`
    /// contributes voltage-dependent terms the oracle does not implement.
    fn zip_term_counts(&self) -> Vec<usize> {
        self.buses_template.iter().map(|b| b.zip_terms.len()).collect()
    }

    /// Per-bus voltage magnitude in per-unit, in node order — `None` before
    /// the first `solve()` call.
    fn voltage_mag(&self) -> Vec<f64> {
        self.buses.iter().map(|b| b.voltage_mag).collect()
    }

    /// Per-bus voltage angle in radians, in node order.
    fn voltage_ang(&self) -> Vec<f64> {
        self.buses.iter().map(|b| b.voltage_ang).collect()
    }

    /// Per-bus voltage magnitude in kV (line-to-line), in node order —
    /// `voltage_mag() * u_rated`, converted to the same unit CGMES's own
    /// `SvVoltage.v` uses, so a caller comparing against a fixture's
    /// published values doesn't need `u_rated` (not itself exposed) at all.
    fn voltage_kv(&self) -> Vec<f64> {
        self.buses.iter().map(|b| b.voltage_mag * b.u_rated / 1e3).collect()
    }
}

/// Not `#[pymethods]` — internal helpers, deliberately not exposed to Python.
impl PowerFlowModel {
    /// The last DC solve's result, or a `RuntimeError` naming what to call.
    fn require_dc(&self) -> PyResult<&DcSolution> {
        self.dc_solution.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "no DC result available — build the model with method=\"dc\" and call solve(), \
                 or call solve_dc() directly",
            )
        })
    }

    /// The sensitivity factors, building and caching them on first use.
    ///
    /// Deliberately independent of any solve: PTDF and LODF are properties of
    /// the topology, so a model built for Newton can be asked for them
    /// without running a DC solve first.
    fn require_sensitivity(&mut self) -> PyResult<&DcSensitivity> {
        if self.sensitivity.is_none() {
            let branches = dc_branches(&self.lines, &self.transformers, self.dc_opts);
            let n_branches = self.lines.len() + self.transformers.len();
            self.sensitivity =
                Some(DcSensitivity::new(&self.buses_template, &branches, n_branches).ok_or_else(
                    || {
                        PyRuntimeError::new_err(
                            "the reduced susceptance matrix is singular, so no sensitivity \
                             factors exist for this network",
                        )
                    },
                )?);
        }
        Ok(self.sensitivity.as_ref().expect("just populated"))
    }

    fn check_branch(&self, branch: usize) -> PyResult<()> {
        let n = self.lines.len() + self.transformers.len();
        if branch >= n {
            return Err(PyValueError::new_err(format!(
                "branch {branch} is out of range (0..{n})"
            )));
        }
        Ok(())
    }

    fn run_batch(
        &mut self,
        scenarios: Vec<Scenario>,
        threads: Option<usize>,
    ) -> PyResult<Vec<BatchResult>> {
        let want = threads.unwrap_or_else(rayon::current_num_threads).max(1);
        if self.batch.as_ref().map(|(n, _)| *n) != Some(want) {
            let solver = BatchSolver::with_threads(self.backend, want)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.batch = Some((want, solver));
        }

        // Disjoint field borrows: `batch` immutably, `buses_template`/`ybus`
        // immutably. Nothing here needs `&mut self`.
        let (_, batch) = self.batch.as_ref().expect("just populated above");
        let reports = batch
            .solve(&self.buses_template, &self.ybus, &scenarios, self.tol, self.max_iter)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(reports
            .into_iter()
            .map(|r| BatchResult {
                voltage_mag: r.buses.iter().map(|b| b.voltage_mag).collect(),
                voltage_ang: r.buses.iter().map(|b| b.voltage_ang).collect(),
                iterations: r.stats.iterations(),
                converged: r.stats.status == SolveStatus::Converged,
                max_mismatch: r.stats.final_mismatch(),
            })
            .collect())
    }
}

/// The extension module's Rust function name must match `pyproject.toml`'s
/// `module-name = "gridoxide._gridoxide"` (its last dotted segment) — maturin
/// links this as `PyInit__gridoxide`, loaded by `python/gridoxide/__init__.py`
/// via `from ._gridoxide import PowerFlowModel`, not imported directly by
/// end users.
/// State estimation over a PGM network and its sensors.
///
/// Deliberately a separate class from `PowerFlowModel` rather than a method on
/// it. The two solve different problems from different inputs: power flow is
/// given injections and computes voltages, while state estimation is given
/// noisy measurements and computes the most likely voltages. They do not even
/// share an unknown count — state estimation has no PV buses, so every bus
/// carries a magnitude.
#[pyclass]
struct StateEstimationModel {
    buses: Vec<Bus>,
    net: crate::pgm::PgmNetwork,
    se_net: crate::se::SeNetwork,
    measurements: Vec<crate::measurement::Measurement>,
    options: crate::se::nr::SeOptions,
    report: Option<crate::se::nr::SeReport>,
    /// Keeps the symbolic factorization between `solve()` calls, the way
    /// `PowerFlowModel` keeps `PersistentSolver`'s. The measurement set on a
    /// model never changes structure — it is fixed at load — so the cache is
    /// valid for the model's whole life.
    estimator: crate::se::nr::PersistentEstimator,
}

#[pymethods]
impl StateEstimationModel {
    /// Loads a PGM-format JSON document containing sensors.
    ///
    /// The document needs `sym_voltage_sensor` and/or `sym_power_sensor`
    /// entries; unlike a power-flow document it does *not* need `p_specified`
    /// on its loads or `u_ref` on its sources, since those are quantities state
    /// estimation solves for rather than inputs it consumes.
    #[staticmethod]
    #[pyo3(signature = (path, backend="scalar", method="newton_raphson", tol=1e-8, max_iter=20, s_base_va=1e6, freq_hz=50.0))]
    fn from_pgm_json(
        path: &str,
        backend: &str,
        method: &str,
        tol: f64,
        max_iter: usize,
        s_base_va: f64,
        freq_hz: f64,
    ) -> PyResult<Self> {
        let method = match method {
            "newton_raphson" => crate::se::nr::SeMethod::NewtonRaphson,
            "iterative_linear" => crate::se::nr::SeMethod::IterativeLinear,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown method {other:?}, expected 'newton_raphson' or 'iterative_linear'"
                )))
            }
        };
        let raw = std::fs::read_to_string(path)
            .map_err(|e| PyRuntimeError::new_err(format!("reading {path}: {e}")))?;
        let input: PgmInput = serde_json::from_str(&raw)
            .map_err(|e| PyValueError::new_err(format!("parsing {path} as PGM JSON: {e}")))?;
        let id_to_idx = crate::pgm::node_id_to_idx(&input);
        let shunts = crate::pgm::pgm_shunts_1ph(&input, &id_to_idx, s_base_va);

        // The conversion consumes its input and the measurement builder needs
        // it, so the document is parsed twice rather than cloned through.
        let net = crate::pgm::pgm_to_network(
            serde_json::from_str(&raw)
                .map_err(|e| PyValueError::new_err(format!("parsing {path} as PGM JSON: {e}")))?,
            s_base_va,
            freq_hz,
        );
        let measurements = crate::measurement::measurements_from_pgm(&input, &net, s_base_va)
            .map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?;
        if measurements.is_empty() {
            return Err(PyValueError::new_err(format!(
                "{path} contains no usable sensors, so there is nothing to estimate"
            )));
        }

        let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
        stamp_shunts(&mut ybus, &shunts);
        let se_net = crate::se::SeNetwork::new(&net, ybus.finish(), &shunts);
        let buses = net.buses.clone();
        // One value shared by the field and the estimator. These used to be two
        // separately-constructed copies, harmless while nothing read the field
        // — `solve_batch` now does.
        let options = crate::se::nr::SeOptions {
            tol,
            max_iter,
            backend: parse_backend(backend)?,
            method,
        };

        Ok(Self {
            buses,
            net,
            se_net,
            measurements,
            options,
            report: None,
            estimator: crate::se::nr::PersistentEstimator::new(options),
        })
    }

    /// Number of buses, including the virtual slack bus gridoxide synthesizes
    /// per active source.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.buses.len()
    }

    /// The loaded value of measurement `i`, per-unit.
    ///
    /// `solve_batch` overrides rows by index, so a caller building scenarios
    /// needs to be able to read the template it is overriding — otherwise
    /// varying one row means re-deriving the whole aggregated set in Python.
    fn measurement_value(&self, i: usize) -> PyResult<f64> {
        self.measurements
            .get(i)
            .map(|m| m.value)
            .ok_or_else(|| PyValueError::new_err(format!("no measurement {i}")))
    }

    /// The loaded standard deviation of measurement `i`, per-unit.
    fn measurement_sigma(&self, i: usize) -> PyResult<f64> {
        self.measurements
            .get(i)
            .map(|m| m.sigma)
            .ok_or_else(|| PyValueError::new_err(format!("no measurement {i}")))
    }

    /// Number of scalar measurements after aggregation — one per `z` entry, so
    /// a power sensor contributes two.
    #[getter]
    fn n_measurements(&self) -> usize {
        self.measurements.len()
    }

    /// Runs the estimate from a linear start. Raises if it does not converge.
    fn solve(&mut self) -> PyResult<()> {
        self.buses = self.net.buses.clone();
        crate::se::nr::linear_start(&mut self.buses, &self.se_net, &self.measurements);
        let report = self
            .estimator
            .estimate(&self.measurements, &mut self.buses, &self.se_net);
        let status = report.status;
        self.report = Some(report);
        match status {
            crate::se::nr::SeStatus::Converged => Ok(()),
            crate::se::nr::SeStatus::MaxIterations => Err(PyRuntimeError::new_err(
                "state estimation did not converge within max_iter",
            )),
            crate::se::nr::SeStatus::Singular => Err(PyRuntimeError::new_err(
                "the gain matrix is singular; the measurements likely leave part of \
                 the state unobservable — call observability() for the detail",
            )),
        }
    }

    /// Estimates many scenarios over this model's topology and measurement
    /// structure, across `threads` workers.
    ///
    /// Each scenario is `[(measurement_index, value, sigma), ...]`, replacing
    /// those rows of the loaded measurement set. Everything not named keeps the
    /// loaded reading. Deliberately index-based rather than document-based:
    /// what may vary between scenarios is exactly values and sigmas, since
    /// anything else would move the gain matrix's sparsity pattern and throw
    /// away the shared factorization batching exists for.
    ///
    /// Returns one `SeBatchOutcome` per scenario, in scenario order regardless
    /// of thread count. A scenario that fails to converge is reported rather
    /// than raised — a divergent scenario must not poison the batch.
    #[pyo3(signature = (scenarios, threads=0))]
    fn solve_batch(
        &mut self,
        scenarios: Vec<Vec<(usize, f64, f64)>>,
        threads: usize,
    ) -> PyResult<Vec<SeBatchOutcome>> {
        use crate::se::batch::{MeasurementOverride, SeBatchSolver, SeScenario};

        let scenarios: Vec<SeScenario> = scenarios
            .into_iter()
            .map(|rows| {
                SeScenario::new(
                    rows.into_iter()
                        .map(|(i, value, sigma)| {
                            MeasurementOverride::new(i).value(value).sigma(sigma)
                        })
                        .collect(),
                )
            })
            .collect();

        let solver = if threads == 0 {
            SeBatchSolver::new(self.options)
        } else {
            SeBatchSolver::with_threads(self.options, threads)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };
        let results = solver
            .estimate(&self.net.buses, &self.se_net, &self.measurements, &scenarios)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| SeBatchOutcome {
                voltage_mag: r.buses.iter().map(|b| b.voltage_mag).collect(),
                voltage_ang: r.buses.iter().map(|b| b.voltage_ang).collect(),
                iterations: r.report.iterations,
                objective: r.report.objective,
                converged: matches!(r.report.status, crate::se::nr::SeStatus::Converged),
            })
            .collect())
    }

    /// Per-bus voltage magnitude in per-unit, in node order.
    fn voltage_mag(&self) -> Vec<f64> {
        self.buses.iter().map(|b| b.voltage_mag).collect()
    }

    /// Per-bus voltage angle in radians, in node order.
    fn voltage_ang(&self) -> Vec<f64> {
        self.buses.iter().map(|b| b.voltage_ang).collect()
    }

    /// `z - h(x)` at the final state, one per measurement.
    fn residuals(&self) -> Vec<f64> {
        self.report.as_ref().map(|r| r.residuals.clone()).unwrap_or_default()
    }

    /// `J(x) = 1/2 r^T W r` at the final state.
    ///
    /// A large value is not a convergence failure: it means the measurements
    /// disagree with each other. `bad_data()` is what interprets it.
    #[getter]
    fn objective(&self) -> f64 {
        self.report.as_ref().map(|r| r.objective).unwrap_or(f64::NAN)
    }

    /// Which buses and quantities the measurements leave undetermined, as a
    /// list of `(bus, "angle" | "magnitude")`.
    ///
    /// Empty means fully observable. Entries beyond the physical node count
    /// refer to gridoxide's synthesized buses, which are expected to appear
    /// whenever a source's own power is unmeasured.
    fn observability(&self) -> Vec<(usize, String)> {
        let layout = crate::se::jacobian::StateLayout::new(
            &self.buses,
            &self.measurements,
            &self.se_net,
        );
        let report = crate::se::observability::analyze(
            &self.measurements,
            &self.buses,
            &self.se_net,
            &layout,
            &crate::se::constraints::Constraints::new(&self.se_net),
        );
        report
            .unobservable
            .iter()
            .chain(&report.structurally_unmeasured)
            .map(|u| {
                let quantity = match u.quantity {
                    crate::se::observability::Quantity::Angle => "angle",
                    crate::se::observability::Quantity::Magnitude => "magnitude",
                };
                (u.bus, quantity.to_string())
            })
            .collect()
    }

    /// Bad-data analysis at the solved state.
    ///
    /// Returns `(chi_squared, degrees_of_freedom, p_value, suspects)`, where
    /// each suspect is `(measurement_index, normalized_residual)` worst first.
    /// A p-value below 0.05 conventionally means the measurements are not
    /// merely noisy; a normalized residual above 3 conventionally identifies
    /// the culprit.
    #[pyo3(signature = (candidates=20))]
    fn bad_data(&self, candidates: usize) -> PyResult<(f64, usize, f64, Vec<(usize, f64)>)> {
        let Some(report) = self.report.as_ref() else {
            return Err(PyRuntimeError::new_err("call solve() before bad_data()"));
        };
        let layout = crate::se::jacobian::StateLayout::new(
            &self.buses,
            &self.measurements,
            &self.se_net,
        );
        let constraints = crate::se::constraints::Constraints::new(&self.se_net);
        let bad = crate::se::bad_data::analyze(
            &self.measurements,
            &report.residuals,
            &self.buses,
            &self.se_net,
            &layout,
            &constraints,
            crate::se::bad_data::Candidates { limit: candidates },
        );
        Ok((
            bad.chi_squared,
            bad.degrees_of_freedom,
            bad.p_value,
            bad.suspects
                .iter()
                .map(|s| (s.measurement, s.normalized_residual))
                .collect(),
        ))
    }
}

/// One node's short-circuit result.
#[pyclass]
struct ShortCircuitNode {
    /// The document's own node id.
    #[pyo3(get)]
    id: u64,
    /// Per-phase voltage magnitude, per unit, as `[a, b, c]`.
    #[pyo3(get)]
    u_pu: [f64; 3],
    /// Per-phase voltage angle, radians.
    #[pyo3(get)]
    u_angle: [f64; 3],
    /// Per-phase voltage magnitude in volts, line-to-neutral.
    #[pyo3(get)]
    u: [f64; 3],
    #[pyo3(get)]
    energized: bool,
    /// Symmetrical-component magnitudes `(zero, positive, negative)` of this
    /// node's voltage — the view IEC 60909 and powsybl's short-circuit API are
    /// both written in. A three-phase fault leaves only the positive
    /// component; a two-phase fault clear of ground has no zero component at
    /// all.
    #[pyo3(get)]
    sequence: (f64, f64, f64),
}

/// One fault's current.
#[pyclass]
struct ShortCircuitFault {
    #[pyo3(get)]
    id: u64,
    /// Per-phase fault-current magnitude, amperes.
    #[pyo3(get)]
    i_f: [f64; 3],
    /// Per-phase fault-current angle, radians.
    #[pyo3(get)]
    i_f_angle: [f64; 3],
}

/// One source's contribution to the fault.
#[pyclass]
struct ShortCircuitSource {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    i: [f64; 3],
    #[pyo3(get)]
    i_angle: [f64; 3],
}

/// The result of [`short_circuit`].
#[pyclass]
struct ShortCircuitResult {
    #[pyo3(get)]
    nodes: Vec<Py<ShortCircuitNode>>,
    #[pyo3(get)]
    faults: Vec<Py<ShortCircuitFault>>,
    #[pyo3(get)]
    sources: Vec<Py<ShortCircuitSource>>,
}

/// Runs an IEC 60909 short-circuit calculation over a PGM-format JSON file.
///
/// Unlike `PowerFlowModel`, this is a plain function rather than a
/// construct-then-solve class: a short-circuit calculation is a single direct
/// solve with nothing worth caching between calls — no factorization is reused,
/// because the fault boundary conditions change the matrix itself.
///
/// `scaling` is `"max"` (default) or `"min"`, choosing the IEC 60909 voltage
/// factor `c`. The maximum is what equipment ratings are sized against; the
/// minimum is the protection-sensitivity study.
#[pyfunction]
#[pyo3(signature = (path, scaling = "max", s_base_va = 1e6, freq_hz = 50.0))]
fn short_circuit(
    py: Python<'_>,
    path: &str,
    scaling: &str,
    s_base_va: f64,
    freq_hz: f64,
) -> PyResult<ShortCircuitResult> {
    use crate::shortcircuit::{
        short_circuit_from_pgm, SequenceValue, ShortCircuitOptions, VoltageScaling,
    };

    let scaling = match scaling {
        "max" | "maximum" => VoltageScaling::Maximum,
        "min" | "minimum" => VoltageScaling::Minimum,
        other => {
            return Err(PyValueError::new_err(format!(
                "scaling must be \"max\" or \"min\", got {other:?}"
            )))
        }
    };

    let raw = std::fs::read_to_string(path)
        .map_err(|e| PyRuntimeError::new_err(format!("reading {path}: {e}")))?;
    let input: PgmInput = serde_json::from_str(&raw)
        .map_err(|e| PyValueError::new_err(format!("parsing {path}: {e}")))?;

    let (net, report) =
        short_circuit_from_pgm(&input, s_base_va, freq_hz, ShortCircuitOptions { scaling })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Node index → the document's own id, so Python sees the ids it wrote
    // rather than gridoxide's internal ordering.
    let mut id_of = vec![0u64; net.n_nodes];
    for (&id, &idx) in &net.node_idx {
        id_of[idx] = id;
    }

    let nodes = report
        .nodes
        .iter()
        .enumerate()
        .map(|(idx, n)| {
            let s = SequenceValue::from_phase(&report.u_bus[idx]);
            Py::new(
                py,
                ShortCircuitNode {
                    id: id_of[idx],
                    u_pu: n.u_pu,
                    u_angle: n.u_angle,
                    u: n.u,
                    energized: n.energized,
                    sequence: (s.zero.norm(), s.positive.norm(), s.negative.norm()),
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;

    let faults = report
        .faults
        .iter()
        .map(|f| {
            Py::new(
                py,
                ShortCircuitFault { id: f.id, i_f: f.i_f, i_f_angle: f.i_f_angle },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;

    let sources = report
        .sources
        .iter()
        .map(|s| {
            Py::new(py, ShortCircuitSource { id: s.id, i: s.i, i_angle: s.i_angle })
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(ShortCircuitResult { nodes, faults, sources })
}

/// How the network responds to one variable — the forward direction.
#[pyclass]
struct SensitivityColumn {
    /// dP/dp for every branch, in flat branch order (lines, then transformers).
    #[pyo3(get)]
    d_branch_active: Vec<f64>,
    /// dQ/dp for every branch.
    #[pyo3(get)]
    d_branch_reactive: Vec<f64>,
    /// d|V|/dp for every bus, per unit.
    #[pyo3(get)]
    d_voltage_magnitude: Vec<f64>,
    /// dθ/dp for every bus, radians.
    #[pyo3(get)]
    d_voltage_angle: Vec<f64>,
}

/// What moves one quantity — the adjoint direction.
#[pyclass]
struct SensitivityRow {
    /// df/dP for every bus.
    #[pyo3(get)]
    d_active_injection: Vec<f64>,
    /// df/dQ for every bus.
    #[pyo3(get)]
    d_reactive_injection: Vec<f64>,
    /// df/dk for every branch — zero for a line.
    #[pyo3(get)]
    d_transformer_ratio: Vec<f64>,
    /// df/dα for every branch, per radian — zero for a line.
    #[pyo3(get)]
    d_phase_shift: Vec<f64>,
}

/// A converged AC operating point, factorized once and ready to differentiate.
///
/// Construct it from a PGM-format file, then ask as many questions as you like:
/// every accessor is a triangular solve against the one factorization taken at
/// construction, never a refactorization. That is why this is a class where
/// `short_circuit` is a plain function — here there is genuinely something
/// worth holding on to between calls.
///
/// ```python
/// s = gridoxide.AcSensitivityModel("network.json")
/// col = s.column(active_injection=2)      # bus 2 ramps — what responds?
/// row = s.row(branch=8)                   # branch 8 is loaded — what moves it?
/// ```
#[pyclass(unsendable)]
struct AcSensitivityModel {
    inner: crate::ac_sensitivity::AcSensitivity,
    n_buses: usize,
    n_branches: usize,
}

fn parse_terminal(name: &str) -> PyResult<crate::branch_flow::Terminal> {
    match name {
        "from" => Ok(crate::branch_flow::Terminal::From),
        "to" => Ok(crate::branch_flow::Terminal::To),
        other => Err(PyValueError::new_err(format!(
            "terminal must be \"from\" or \"to\", got {other:?}"
        ))),
    }
}

#[pymethods]
impl AcSensitivityModel {
    /// Solves the AC power flow in `path` and factorizes its Jacobian.
    ///
    /// Raises if the power flow does not converge: a derivative taken at a
    /// non-converged point is meaningless rather than merely imprecise, and
    /// silently returning one would be the worst of both.
    #[new]
    #[pyo3(signature = (path, s_base_va = 1e6, freq_hz = 50.0, tol = 1e-8, max_iter = 50))]
    fn new(
        path: &str,
        s_base_va: f64,
        freq_hz: f64,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| PyRuntimeError::new_err(format!("reading {path}: {e}")))?;
        let input: PgmInput = serde_json::from_str(&raw)
            .map_err(|e| PyValueError::new_err(format!("parsing {path}: {e}")))?;
        let (buses, lines, transformers) =
            crate::pgm::pgm_to_buses_and_branches(input, s_base_va, freq_hz);

        let opts = crate::solver::PowerFlowOptions { tol, max_iter, ..Default::default() };
        let report = crate::run_power_flow(buses, &lines, &transformers, &[], crate::TapData::none(), opts);
        if report.stats.status != SolveStatus::Converged {
            return Err(PyRuntimeError::new_err(format!(
                "the power flow did not converge ({:?}); there is no operating point to \
                 differentiate",
                report.stats.status
            )));
        }

        let ybus = build_ybus(report.buses.len(), &lines, &transformers).finish();
        let n_buses = report.buses.len();
        let n_branches = lines.len() + transformers.len();
        let inner =
            crate::ac_sensitivity::AcSensitivity::new(&report.buses, &ybus, &lines, &transformers)
                .ok_or_else(|| {
                    PyRuntimeError::new_err("the Jacobian is singular at the solved point")
                })?;
        Ok(Self { inner, n_buses, n_branches })
    }

    #[getter]
    fn n_buses(&self) -> usize {
        self.n_buses
    }

    /// Branch count, in the flat order every returned vector uses: lines first,
    /// then transformers.
    #[getter]
    fn n_branches(&self) -> usize {
        self.n_branches
    }

    /// The forward direction: pick exactly one variable, get the whole
    /// network's response.
    ///
    /// Exactly one of the four keyword arguments must be given. `active_injection`
    /// and `reactive_injection` take a bus index; `transformer_ratio` and
    /// `phase_shift` take a branch index.
    #[pyo3(signature = (*, active_injection = None, reactive_injection = None,
                           transformer_ratio = None, phase_shift = None,
                           terminal = "from"))]
    fn column(
        &self,
        active_injection: Option<usize>,
        reactive_injection: Option<usize>,
        transformer_ratio: Option<usize>,
        phase_shift: Option<usize>,
        terminal: &str,
    ) -> PyResult<SensitivityColumn> {
        use crate::ac_sensitivity::Variable;
        let chosen: Vec<Variable> = [
            active_injection.map(Variable::ActiveInjection),
            reactive_injection.map(Variable::ReactiveInjection),
            transformer_ratio.map(Variable::TransformerRatio),
            phase_shift.map(Variable::PhaseShift),
        ]
        .into_iter()
        .flatten()
        .collect();
        let [variable] = chosen[..] else {
            return Err(PyValueError::new_err(
                "pass exactly one of active_injection, reactive_injection, \
                 transformer_ratio or phase_shift",
            ));
        };

        let terminal = parse_terminal(terminal)?;
        let flows = self
            .inner
            .branch_response(variable, terminal)
            .ok_or_else(|| PyValueError::new_err("index out of range, or a singular solve"))?;
        let state = self
            .inner
            .state_response(variable)
            .ok_or_else(|| PyValueError::new_err("index out of range, or a singular solve"))?;

        Ok(SensitivityColumn {
            d_branch_active: flows.iter().map(|(p, _)| *p).collect(),
            d_branch_reactive: flows.iter().map(|(_, q)| *q).collect(),
            d_voltage_magnitude: state.d_vmag,
            d_voltage_angle: state.d_theta,
        })
    }

    /// The adjoint direction: pick one quantity, get everything that moves it.
    ///
    /// Either a `branch` (with `quantity` "active" or "reactive") or a `bus`
    /// (with `quantity` "magnitude" or "angle").
    #[pyo3(signature = (*, branch = None, bus = None, quantity = "active", terminal = "from"))]
    fn row(
        &self,
        branch: Option<usize>,
        bus: Option<usize>,
        quantity: &str,
        terminal: &str,
    ) -> PyResult<SensitivityRow> {
        use crate::ac_sensitivity::Function;
        let function = match (branch, bus) {
            (Some(branch), None) => {
                let terminal = parse_terminal(terminal)?;
                match quantity {
                    "active" => Function::BranchActivePower { branch, terminal },
                    "reactive" => Function::BranchReactivePower { branch, terminal },
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "for a branch, quantity must be \"active\" or \"reactive\", got {other:?}"
                        )))
                    }
                }
            }
            (None, Some(bus)) => match quantity {
                "magnitude" | "active" => Function::VoltageMagnitude(bus),
                "angle" => Function::VoltageAngle(bus),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "for a bus, quantity must be \"magnitude\" or \"angle\", got {other:?}"
                    )))
                }
            },
            _ => {
                return Err(PyValueError::new_err(
                    "pass exactly one of branch or bus",
                ))
            }
        };

        let row = self
            .inner
            .function_row(function)
            .ok_or_else(|| PyValueError::new_err("index out of range, or a singular solve"))?;
        Ok(SensitivityRow {
            d_active_injection: row.d_active,
            d_reactive_injection: row.d_reactive,
            d_transformer_ratio: row.d_ratio,
            d_phase_shift: row.d_phase,
        })
    }
}

/// A branch at its limit, and what relieving it is worth.
#[cfg(feature = "opf")]
#[pyclass]
struct DcOpfBinding {
    /// Flat branch index — lines first, then transformers.
    #[pyo3(get)]
    branch: usize,
    /// Flow, MW. Signed, so it says which of the two limits is active.
    #[pyo3(get)]
    flow: f64,
    /// The rating it reached, MW.
    #[pyo3(get)]
    rate: f64,
    /// Shadow price, $/MWh — what one more MW of capacity here would save.
    #[pyo3(get)]
    price: f64,
}

/// The result of an AC optimal power flow.
#[cfg(feature = "opf")]
#[pyclass]
struct AcOpfResult {
    /// Total cost, $/h. **A local optimum** — AC-OPF is nonconvex, so no
    /// solver certifies more, and this is what published AC objectives mean.
    #[pyo3(get)]
    objective: f64,
    /// Active dispatch per generator, MW, in the OPF document's order.
    #[pyo3(get)]
    p_gen: Vec<f64>,
    /// Reactive dispatch per generator, MVAr.
    #[pyo3(get)]
    q_gen: Vec<f64>,
    /// The `index` each dispatch entry belongs to.
    #[pyo3(get)]
    generator_index: Vec<usize>,
    /// Bus voltage magnitudes, per-unit.
    #[pyo3(get)]
    magnitudes: Vec<f64>,
    /// Bus voltage angles, radians.
    #[pyo3(get)]
    angles: Vec<f64>,
    /// `(P, Q)` entering each branch at its from-terminal, MW and MVAr.
    #[pyo3(get)]
    flows: Vec<(f64, f64)>,
    /// Active-power locational marginal price per bus, $/MWh.
    #[pyo3(get)]
    lmp_p: Vec<f64>,
    /// Reactive-power price per bus, $/MVArh.
    #[pyo3(get)]
    lmp_q: Vec<f64>,
    #[pyo3(get)]
    iterations: usize,
    /// Largest constraint violation, per-unit. **Read this with the
    /// objective**: on a nonconvex problem a lower cost at an infeasible point
    /// is not a better answer.
    #[pyo3(get)]
    violation: f64,
}

/// The result of a DC optimal power flow.
#[cfg(feature = "opf")]
#[pyclass]
struct DcOpfResult {
    /// Total cost, $/h.
    #[pyo3(get)]
    objective: f64,
    /// Dispatch per generator, MW, in the OPF document's generator order.
    #[pyo3(get)]
    dispatch: Vec<f64>,
    /// The `index` each dispatch entry belongs to, so results can be matched
    /// back to the source case's generator table.
    #[pyo3(get)]
    generator_index: Vec<usize>,
    /// Bus voltage angles, radians.
    #[pyo3(get)]
    angles: Vec<f64>,
    /// Flow per branch, MW.
    #[pyo3(get)]
    flows: Vec<f64>,
    /// **Locational marginal price** per bus, $/MWh — the cost of serving one
    /// more MW there. Uniform when nothing is congested; the spread between
    /// buses *is* the congestion.
    #[pyo3(get)]
    lmp: Vec<f64>,
    /// Load shed, MW, per load in the OPF document. All zero on a case whose
    /// demand can be served.
    #[pyo3(get)]
    shed: Vec<f64>,
    #[pyo3(get)]
    binding: Vec<Py<DcOpfBinding>>,
}

/// Runs a DC optimal power flow: least-cost dispatch subject to generator
/// limits and branch ratings.
///
/// A plain function rather than a class, unlike `AcSensitivityModel` — each
/// solve builds its own program, so there is nothing worth keeping between
/// calls.
///
/// Costs and limits come from the companion OPF document, which defaults to
/// `<path>` with its extension replaced by `.opf.json` — the pair
/// `gridoxide-matpower` writes.
///
/// Raises if no optimal dispatch exists. With `allow_shedding` left on that is
/// rare, since shedding keeps an over-committed case solvable and reports
/// *where* demand could not be served; turning it off makes such a case
/// infeasible instead, which is sometimes the answer wanted.
///
/// `solver` is `"ipm"` (default) — the built-in interior-point method, which
/// needs nothing installed — or `"highs"`, the reference backend, available
/// only when the extension was built with the `opf-highs` feature. The two are
/// cross-checked against each other in `tests/opf_cross_test.rs`, so this
/// picks a dependency rather than an answer.
///
/// `dc_approximation` is `"ignore_g"` (default here, `b = x/(r²+x²)`) or
/// `"ignore_r"` (`b = 1/x`). Note the default is the opposite of
/// `PowerFlowModel.from_pgm_json`'s, and deliberately so: each matches what
/// its own domain's reference implementations compute. `"ignore_g"` is what
/// PowerModels builds its DC model from, and what pglib-opf's published
/// objectives were produced with, while `b = 1/x` is what MATPOWER's
/// `makeBdc` and pandapower use for power flow. The difference is not
/// cosmetic — `1/x` overstates susceptance on resistive branches, which on
/// `case30_ieee` lands on a congested branch and moves the objective 0.4%.
#[cfg(feature = "opf")]
#[pyfunction]
#[pyo3(signature = (path, data_path = None, shed_price = 10_000.0,
                    allow_shedding = true, dc_approximation = "ignore_g",
                    solver = "ipm", freq_hz = 50.0))]
fn dc_opf(
    py: Python<'_>,
    path: &str,
    data_path: Option<&str>,
    shed_price: f64,
    allow_shedding: bool,
    dc_approximation: &str,
    solver: &str,
    freq_hz: f64,
) -> PyResult<DcOpfResult> {
    use crate::opf::dc::{DcOpf, DcOpfNetwork, DcOpfOptions};
    use crate::opf::ipm::IpmSolver;
    use crate::opf::model::OpfData;
    use crate::opf::{OptStatus, Solver};

    let data_path = match data_path {
        Some(p) => std::path::PathBuf::from(p),
        None => std::path::PathBuf::from(path).with_extension("opf.json"),
    };

    let network_text = std::fs::read_to_string(path)
        .map_err(|e| PyRuntimeError::new_err(format!("reading {path}: {e}")))?;
    let data_text = std::fs::read_to_string(&data_path).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "reading {}: {e} — pass data_path if the companion OPF document is elsewhere",
            data_path.display()
        ))
    })?;

    let input: PgmInput = serde_json::from_str(&network_text)
        .map_err(|e| PyValueError::new_err(format!("parsing {path}: {e}")))?;
    let data = OpfData::from_json(&data_text)
        .map_err(|e| PyValueError::new_err(format!("parsing {}: {e}", data_path.display())))?;

    let options = DcOpfOptions { shed_price, allow_shedding };
    let approximation = parse_dc_approximation(dc_approximation)?;
    let network = DcOpfNetwork::from_pgm(input, &data, freq_hz, approximation)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let generator_index: Vec<usize> = network.generators.iter().map(|g| g.index).collect();

    let opf = DcOpf::build(network, options)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let solution = match solver {
        "ipm" => {
            let mut s = IpmSolver::new();
            s.solve(opf.problem()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        }
        "highs" => {
            #[cfg(feature = "opf-highs")]
            {
                let mut s = crate::opf::highs::HighsSolver::new()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                s.solve(opf.problem()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            #[cfg(not(feature = "opf-highs"))]
            {
                return Err(PyValueError::new_err(
                    "solver='highs' needs the extension built with the opf-highs \
                     feature, which links a local HiGHS install",
                ));
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown solver '{other}', expected 'ipm' or 'highs'"
            )))
        }
    };
    let result = opf.interpret(&solution);

    if result.status != OptStatus::Optimal {
        return Err(PyRuntimeError::new_err(format!(
            "no optimal dispatch ({:?}); an infeasible case usually means demand cannot be \
             served within the limits",
            result.status
        )));
    }

    let binding = result
        .binding
        .iter()
        .map(|b| {
            Py::new(
                py,
                DcOpfBinding { branch: b.branch, flow: b.flow, rate: b.rate, price: b.price },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(DcOpfResult {
        objective: result.objective,
        dispatch: result.dispatch,
        generator_index,
        angles: result.angles,
        flows: result.flows,
        lmp: result.lmp,
        shed: result.shed,
        binding,
    })
}

/// Solves an AC optimal power flow.
///
/// The full problem `dc_opf` linearizes: real voltage magnitudes, reactive
/// power and losses, optimizing generator active *and* reactive output
/// together.
///
/// **Nonconvex**, so the answer is a local optimum satisfying the first-order
/// conditions rather than a proven global one. That is the state of the art
/// and what every published AC-OPF objective means — but it makes
/// `result.violation` part of the answer rather than diagnostics: a lower cost
/// at an infeasible point is not better.
///
/// Costs, limits and per-bus voltage bounds come from the companion OPF
/// document, defaulting to `<path>` with its extension replaced by
/// `.opf.json`.
#[cfg(feature = "opf")]
#[pyfunction]
#[pyo3(signature = (path, data_path = None, enforce_limits = true,
                    max_iterations = 300, tolerance = 1e-8, freq_hz = 50.0))]
fn ac_opf(
    py: Python<'_>,
    path: &str,
    data_path: Option<&str>,
    enforce_limits: bool,
    max_iterations: usize,
    tolerance: f64,
    freq_hz: f64,
) -> PyResult<AcOpfResult> {
    use crate::opf::ac::{AcOpf, AcOpfNetwork, AcOpfOptions};
    use crate::opf::model::OpfData;

    let data_path = match data_path {
        Some(p) => std::path::PathBuf::from(p),
        None => std::path::PathBuf::from(path).with_extension("opf.json"),
    };
    let network_text = std::fs::read_to_string(path)
        .map_err(|e| PyRuntimeError::new_err(format!("reading {path}: {e}")))?;
    let data_text = std::fs::read_to_string(&data_path).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "reading {}: {e} — pass data_path if the companion OPF document is elsewhere",
            data_path.display()
        ))
    })?;
    let input: PgmInput = serde_json::from_str(&network_text)
        .map_err(|e| PyValueError::new_err(format!("parsing {path}: {e}")))?;
    let data = OpfData::from_json(&data_text)
        .map_err(|e| PyValueError::new_err(format!("parsing {}: {e}", data_path.display())))?;

    let mut options = AcOpfOptions { enforce_limits, ..AcOpfOptions::default() };
    options.nlp.max_iterations = max_iterations;
    options.nlp.tolerance = tolerance;

    // Released for the duration of the solve: an AC-OPF on a large network is
    // seconds of pure computation touching no Python object, so holding the
    // interpreter lock through it would serialize callers for no reason.
    // (`detach` is pyo3 0.29's name for what was `allow_threads`.)
    let result = py.detach(|| {
        let network = AcOpfNetwork::from_pgm(input, &data, freq_hz, &options)?;
        AcOpf::build(network, options)?.solve()
    });
    let result = result.map_err(|e| PyValueError::new_err(e.to_string()))?;

    if result.status != crate::opf::OptStatus::Optimal {
        return Err(PyRuntimeError::new_err(format!(
            "no optimal dispatch ({:?}), largest constraint violation {:.3e} pu",
            result.status, result.violation
        )));
    }

    Ok(AcOpfResult {
        objective: result.objective,
        p_gen: result.p_gen,
        q_gen: result.q_gen,
        generator_index: result.generator_index,
        magnitudes: result.magnitudes,
        angles: result.angles,
        flows: result.flows,
        lmp_p: result.lmp_p,
        lmp_q: result.lmp_q,
        iterations: result.iterations,
        violation: result.violation,
    })
}

#[pymodule]
fn _gridoxide(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PowerFlowModel>()?;
    m.add_class::<BatchResult>()?;
    m.add_class::<StateEstimationModel>()?;
    m.add_class::<SeBatchOutcome>()?;
    m.add_class::<ShortCircuitResult>()?;
    m.add_class::<ShortCircuitNode>()?;
    m.add_class::<ShortCircuitFault>()?;
    m.add_class::<ShortCircuitSource>()?;
    m.add_function(wrap_pyfunction!(short_circuit, m)?)?;
    m.add_class::<AcSensitivityModel>()?;
    m.add_class::<SensitivityColumn>()?;
    m.add_class::<SensitivityRow>()?;

    // Only present when the optimization layer was built in — see the `opf`
    // feature. `hasattr(gridoxide, "dc_opf")` is the check a caller makes.
    #[cfg(feature = "opf")]
    {
        m.add_class::<DcOpfResult>()?;
        m.add_class::<DcOpfBinding>()?;
        m.add_class::<AcOpfResult>()?;
        m.add_function(wrap_pyfunction!(dc_opf, m)?)?;
        m.add_function(wrap_pyfunction!(ac_opf, m)?)?;
    }
    Ok(())
}
