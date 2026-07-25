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

use crate::network::{build_ybus, linear_initial_guess, stamp_shunts, YBusSparse};
use crate::pgm::PgmInput;
use crate::solver::{IslandStatus, JacobianBackend, PersistentSolver};
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
    /// (`maturin develop --features python,cgmes`) — see `CIMOXIDE_PROVENANCE.md`
    /// for why that dependency is opt-in.
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

/// The extension module's Rust function name must match `pyproject.toml`'s
/// `module-name = "gridoxide._gridoxide"` (its last dotted segment) — maturin
/// links this as `PyInit__gridoxide`, loaded by `python/gridoxide/__init__.py`
/// via `from ._gridoxide import PowerFlowModel`, not imported directly by
/// end users.
#[pymodule]
fn _gridoxide(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PowerFlowModel>()?;
    Ok(())
}
