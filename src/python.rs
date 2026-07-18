//! PyO3 bindings — the `gridoxide` Python extension module. Only built via
//! `maturin` (`maturin develop --features python`), never via a plain
//! `cargo build`/`cargo test` — see the `python` feature's doc comment in
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

use crate::network::{build_ybus, linear_initial_guess, YBusSparse};
use crate::pgm::PgmInput;
use crate::solver::{JacobianBackend, PersistentSolver, SolveStatus};
use crate::types::Bus;

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
        let (buses_template, lines, transformers) =
            crate::pgm::pgm_to_buses_and_branches(input, s_base_va, freq_hz);
        let ybus = build_ybus(buses_template.len(), &lines, &transformers).finish();
        let backend = parse_backend(backend)?;
        Ok(Self {
            buses: buses_template.clone(),
            buses_template,
            ybus,
            solver: PersistentSolver::new(backend),
            tol,
            max_iter,
        })
    }

    /// Number of buses (nodes), including the virtual slack bus each active
    /// `source` adds — matches `bench_network.rs`'s printed `nodes=`.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.buses.len()
    }

    /// Solves from a fresh flat/linear-initial-guess start, reusing this
    /// model's `PersistentSolver` cached factorization from any previous
    /// `solve()` call. Raises `RuntimeError` if Newton-Raphson doesn't
    /// converge within `max_iter` iterations.
    fn solve(&mut self) -> PyResult<()> {
        self.buses = self.buses_template.clone();
        linear_initial_guess(&mut self.buses, &self.ybus);
        match self.solver.solve(&mut self.buses, &self.ybus, self.tol, self.max_iter) {
            SolveStatus::Converged => Ok(()),
            SolveStatus::MaxIterationsReached => Err(PyRuntimeError::new_err(format!(
                "power flow did not converge within {} iterations", self.max_iter
            ))),
            SolveStatus::Singular => Err(PyRuntimeError::new_err("Jacobian is singular")),
        }
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
}

#[pymodule]
fn gridoxide(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PowerFlowModel>()?;
    Ok(())
}
