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
use crate::network::{build_ybus, linear_initial_guess, stamp_shunts, YBusSparse};
use crate::pgm::PgmInput;
use crate::solver::{IslandStatus, JacobianBackend, PersistentSolver, SolveStatus};
use crate::types::Bus;
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
    solver: PersistentSolver,
    backend: JacobianBackend,
    /// Thread count paired with the `BatchSolver` built for it. Cached so
    /// repeated `solve_batch` calls at one thread count reuse a single rayon
    /// pool instead of respawning workers per call.
    batch: Option<(usize, BatchSolver)>,
    tol: f64,
    max_iter: usize,
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
    #[staticmethod]
    #[pyo3(signature = (path, backend="scalar", tol=1e-6, max_iter=20, s_base_va=1e6, freq_hz=50.0))]
    fn from_pgm_json(
        path: &str,
        backend: &str,
        tol: f64,
        max_iter: usize,
        s_base_va: f64,
        freq_hz: f64,
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
            solver: PersistentSolver::new(backend),
            backend,
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
    #[pyo3(signature = (paths, backend="scalar", tol=1e-6, max_iter=20, s_base_va=100e6))]
    fn from_cgmes(
        paths: Vec<String>,
        backend: &str,
        tol: f64,
        max_iter: usize,
        s_base_va: f64,
    ) -> PyResult<Self> {
        let path_refs: Vec<&std::path::Path> = paths.iter().map(std::path::Path::new).collect();
        let ds = load_profiles(&path_refs)
            .map_err(|e| PyRuntimeError::new_err(format!("decoding CGMES profiles: {e}")))?;
        let (mut buses_template, lines, transformers, shunts) = cgmes_to_buses_and_branches(&ds, s_base_va)
            .map_err(|e| PyRuntimeError::new_err(format!("converting CGMES model: {e}")))?;
        cgmes_resolve_dc_converters(&ds, &mut buses_template, s_base_va)
            .map_err(|e| PyRuntimeError::new_err(format!("resolving CGMES HVDC converters: {e}")))?;
        let tn_bus_index = cgmes_topological_node_bus_index(&ds)
            .map_err(|e| PyRuntimeError::new_err(format!("resolving CGMES bus index: {e}")))?;
        let mut ybus = build_ybus(buses_template.len(), &lines, &transformers);
        stamp_shunts(&mut ybus, &shunts);
        let ybus = ybus.finish();
        let backend = parse_backend(backend)?;
        Ok(Self {
            buses: buses_template.clone(),
            buses_template,
            ybus,
            solver: PersistentSolver::new(backend),
            backend,
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
    fn reset(&mut self) {
        self.solver.reset();
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

/// Not `#[pymethods]` — internal helper shared by `solve_batch` and
/// `solve_batch_scaled`, deliberately not exposed to Python.
impl PowerFlowModel {
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
        let constraints = crate::se::constraints::Constraints::new(&self.se_net.constrained_buses());
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

#[pymodule]
fn _gridoxide(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PowerFlowModel>()?;
    m.add_class::<BatchResult>()?;
    m.add_class::<StateEstimationModel>()?;
    m.add_class::<SeBatchOutcome>()?;
    Ok(())
}
