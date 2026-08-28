//! RMS (phasor-domain) dynamic simulation: what the network does over time
//! after a disturbance.
//!
//! Everything else in this crate answers a question about one *instant* — a
//! power flow, a short circuit, a state estimate, an optimum — or, with
//! [`continuation`](crate::continuation), about a curve of instants indexed by
//! loading. None of them can say whether the machines stay in step after a
//! fault, how far the voltage dips, or how fast it recovers. That is a
//! trajectory, and it needs a differential-algebraic system rather than an
//! algebraic one.
//!
//! ```text
//! ẋ = f(x, V)                 machines, exciters, governors — the state
//! Y V − I_inj(x, V) = 0       the network — a constraint, not a state
//! ```
//!
//! The network equations carry no derivative. That is the *algebraic* half,
//! and it is what makes this a DAE and not an ODE: `V` is not integrated, it is
//! whatever the constraint says it is at each instant given `x`. Physically it
//! is the statement that electromagnetic transients in the network settle far
//! faster than the electromechanical ones being simulated, so they are taken as
//! instantaneous. That time-scale separation is exactly what "RMS" means, and
//! it is what an EMT simulation refuses to assume.
//!
//! # How a run is put together
//!
//! 1. Solve an ordinary power flow. This crate's existing solver does it, with
//!    whatever outer loops the case needs.
//! 2. [`build`](init::build) turns the solved case into a [`DynamicSystem`]:
//!    each device's states are chosen so its derivatives are *zero* at that
//!    operating point, the remaining bus injections become constant
//!    admittances, and every device's Norton admittance is stamped into `Y`.
//! 3. [`run_dynamics`] integrates.
//!
//! Step 2 is the part that goes wrong. Its gate is stated in
//! [`init`](init) and is worth repeating here: **a run with no disturbance
//! must produce a flat trajectory.** If initialization is consistent, every
//! derivative is zero at `t = 0` and stays zero. If a sign is flipped or a
//! per-unit base missed, the state drifts immediately. Nearly every mistake in
//! this module is caught by that one check.
//!
//! # Scope
//!
//! Phase 1 of `plans/RMS_PLAN.md`: the DAE, the integrator, and the classical
//! machine, with fixed-voltage buses and constant-impedance loads. Events, the
//! model library (transient and subtransient machines, AVRs, governors, PSS),
//! the readers and the external validation are phases 2 through 6.

pub mod dae;
pub mod events;
pub mod init;
pub mod integrator;
pub mod models;

use num_complex::Complex;

use crate::klu_native::KluNativeSystem;
use crate::network::{build_ybus_with_outages, connected_components, stamp_shunts, ShuntAdm, YBusSparse};
use crate::solver::JacobianBackend;
use crate::sparse::RealSparseSystem;
use crate::types::{Line, Transformer};
#[cfg(feature = "klu")]
use crate::sparse_klu::KluRealSystem;
#[cfg(feature = "pardiso")]
use crate::sparse_pardiso::PardisoRealSystem;

use dae::DaePattern;
use models::DynamicModel;

pub use events::{DynamicsWarning, Event, EventError, EventKind};
pub use init::{build, BuildError, DeviceSpec, SystemSpec};

/// What a run needs to know.
#[derive(Clone, Debug)]
pub struct DynamicsOptions {
    /// Simulated end time, seconds. The run starts at zero.
    pub end_time: f64,
    /// Step size, seconds. Fixed — adaptive stepping is a later phase, and a
    /// fixed step is what gate G3 measures the order of accuracy against.
    pub step: f64,
    /// Infinity-norm convergence tolerance on the residual, per unit.
    pub tol: f64,
    pub max_newton: usize,
    /// Backward-Euler steps taken after each discontinuity, to damp the
    /// trapezoidal rule's ringing. See [`integrator`]. Two is the usual
    /// choice; zero disables the damping and makes the ringing visible, which
    /// is what the test for it does.
    pub damping_steps: usize,
    /// The disturbance schedule. Sorted by time before the run, so the order
    /// given does not matter; events sharing a time are applied in the order
    /// listed, which is what lets a trip and a fault at the same instant be
    /// expressed unambiguously.
    ///
    /// Steps are truncated to land exactly on each event time, so an event
    /// need not fall on a multiple of [`step`](Self::step).
    pub events: Vec<Event>,
    /// Which sparse-LU backend solves each step.
    ///
    /// [`JacobianBackend::Block`] is refused: it assumes a uniform 2×2 block
    /// per bus, and a DAE has variable-size device blocks alongside the
    /// network's. `run_dynamics` falls back to `Scalar` rather than producing
    /// a wrong answer quietly — the same position `continuation` takes.
    pub backend: JacobianBackend,
}

impl Default for DynamicsOptions {
    fn default() -> Self {
        Self {
            end_time: 10.0,
            step: 0.005,
            tol: 1e-9,
            max_newton: 20,
            damping_steps: 2,
            events: Vec::new(),
            backend: JacobianBackend::Scalar,
        }
    }
}

/// How a run ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DynamicsStatus {
    /// Reached `end_time`.
    Completed,
    /// The step's Newton solve hit `max_newton` without converging. Usually
    /// means the step is too large for what the system is doing, not that the
    /// system is unstable — an unstable system integrates perfectly well, it
    /// just diverges physically.
    NewtonFailed { time: f64 },
    /// The Jacobian was singular. An island with no voltage reference does
    /// this, as does a bus with no admittance at all.
    Singular { time: f64 },
}

/// The recorded trajectory: one row of observables per time point.
///
/// Every differential state, then each bus's voltage magnitude and angle.
/// Selecting a subset is a later phase; at phase 1 sizes the whole thing fits
/// comfortably and recording everything makes the gates easier to write.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    pub time: Vec<f64>,
    pub names: Vec<String>,
    /// `rows[k]` is the observable vector at `time[k]`, in `names` order.
    pub rows: Vec<Vec<f64>>,
}

impl Trajectory {
    pub fn new(names: Vec<String>) -> Self {
        Self { time: Vec::new(), names, rows: Vec::new() }
    }

    pub(crate) fn push(&mut self, t: f64, x: &[f64], v: &[Complex<f64>]) {
        let mut row = Vec::with_capacity(self.names.len());
        row.extend_from_slice(x);
        for vi in v {
            row.push(vi.norm());
        }
        for vi in v {
            row.push(vi.arg());
        }
        self.time.push(t);
        self.rows.push(row);
    }

    /// The column index of a named observable, or `None`.
    pub fn column(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// One observable's whole series, by name.
    pub fn series(&self, name: &str) -> Option<Vec<f64>> {
        let col = self.column(name)?;
        Some(self.rows.iter().map(|r| r[col]).collect())
    }
}

/// A run's outcome.
#[derive(Clone, Debug)]
pub struct DynamicsReport {
    pub trajectory: Trajectory,
    pub status: DynamicsStatus,
    /// Steps actually taken.
    pub steps: usize,
    /// Newton iterations summed over every step — the cost measure worth
    /// watching, since each one is a numeric refactorization.
    pub newton_iterations: usize,
    /// Events actually applied. Fewer than were scheduled if the run stopped
    /// early or an event named something that does not exist.
    pub events_applied: usize,
    /// Anything noticed that is not an error — a de-energized island, a
    /// skipped event.
    pub warnings: Vec<DynamicsWarning>,
}

/// An assembled, initialized system, ready to integrate.
///
/// Produced by [`build`]; not constructed directly, because the invariant that
/// makes a run meaningful — every device sitting at an equilibrium consistent
/// with the Y-bus it was stamped into — is established there and cannot be
/// re-established from outside.
pub struct DynamicSystem {
    pub(crate) models: Vec<Box<dyn DynamicModel>>,
    pub(crate) ids: Vec<String>,
    pub(crate) ybus: YBusSparse,
    pub(crate) pattern: DaePattern,
    /// Every bus's voltage as the power flow solved it.
    ///
    /// Used as the held value at fixed buses, and as the reference magnitude
    /// for [`EventKind::LoadStep`]'s power-to-admittance conversion. Both
    /// readings want the same number, which is why there is one field.
    pub(crate) v_fixed: Vec<Complex<f64>>,
    // --- what a topology event needs to reassemble the Y-bus ---
    pub(crate) lines: Vec<Line>,
    pub(crate) transformers: Vec<Transformer>,
    pub(crate) shunts: Vec<ShuntAdm>,
    /// Out-of-service flags, lines then transformers — the index space
    /// `network::build_ybus_with_outages` and [`EventKind::BranchTrip`] share.
    pub(crate) outaged: Vec<bool>,
    /// Per bus: the constant admittance standing in for its load, derived at
    /// the initial voltage and moved by [`EventKind::LoadStep`].
    pub(crate) load_y: Vec<Complex<f64>>,
    /// Per bus: the fault admittance currently applied, zero where none is.
    pub(crate) fault_y: Vec<Complex<f64>>,
    /// The current state. `build` leaves the initial equilibrium here, and
    /// each run advances it, so a run can be continued.
    pub(crate) x0: Vec<f64>,
    pub(crate) v0: Vec<Complex<f64>>,
}

impl std::fmt::Debug for DynamicSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicSystem")
            .field("devices", &self.ids)
            .field("n_bus", &self.v0.len())
            .field("n_states", &self.x0.len())
            .field("nnz", &self.pattern.nnz())
            .finish()
    }
}

impl DynamicSystem {
    pub fn n_bus(&self) -> usize {
        self.v0.len()
    }

    pub fn n_states(&self) -> usize {
        self.x0.len()
    }

    /// The current differential state.
    pub fn state(&self) -> &[f64] {
        &self.x0
    }

    /// The current differential state, mutably.
    ///
    /// Writing here leaves the algebraic constraint unsatisfied — `V` no
    /// longer solves `Y V = I_inj(x, V)` for the new `x` — so a write must be
    /// followed by [`settle`], exactly as an event is. That coupling is the
    /// whole reason the DAE is a DAE, and it is why this returns the states
    /// alone and not the voltages too.
    pub fn state_mut(&mut self) -> &mut [f64] {
        &mut self.x0
    }

    /// The current bus voltages.
    pub fn voltages(&self) -> &[Complex<f64>] {
        &self.v0
    }

    /// Column headings for [`Trajectory`], in row order.
    pub fn observable_names(&self) -> Vec<String> {
        let layout = self.pattern.layout();
        let mut names = Vec::with_capacity(layout.n() );
        for (d, model) in self.models.iter().enumerate() {
            for state in model.state_names() {
                names.push(format!("{}.{}", self.ids[d], state));
            }
        }
        for i in 0..layout.n_bus {
            names.push(format!("bus{i}.vmag"));
        }
        for i in 0..layout.n_bus {
            names.push(format!("bus{i}.vang"));
        }
        names
    }

    /// `f(x, V)` for every device, written into `out`.
    pub(crate) fn derivatives_at(&self, x: &[f64], v: &[Complex<f64>], out: &mut [f64]) {
        let layout = self.pattern.layout();
        for (d, model) in self.models.iter().enumerate() {
            let (off, len, bus) = (layout.dev_offset[d], layout.dev_len[d], layout.dev_bus[d]);
            model.derivatives(&x[off..off + len], v[bus], &mut out[off..off + len]);
        }
    }

    /// Rebuilds the Y-bus from the current switching state, fault
    /// admittances and load admittances.
    ///
    /// Structure-preserving by construction: `build_ybus_with_outages`
    /// re-stamps an out-of-service branch's positions at zero rather than
    /// dropping them, and every bus's diagonal is stamped unconditionally, so
    /// the result always has exactly the sparsity
    /// [`DaePattern`](dae::DaePattern) was analyzed against. That is what makes
    /// a trip a refill rather than a re-analysis, and it is the property to
    /// preserve if anything is ever added here.
    pub(crate) fn reassemble(&mut self) {
        let n = self.v0.len();
        let mut y = build_ybus_with_outages(n, &self.lines, &self.transformers, &self.outaged);
        stamp_shunts(&mut y, &self.shunts);
        for i in 0..n {
            y.add(i, i, self.load_y[i] + self.fault_y[i]);
        }
        let layout = self.pattern.layout();
        for (d, model) in self.models.iter().enumerate() {
            // Read from the model rather than from a cache: a disconnected unit
            // withdraws its stamp, and a cache would have to be kept in step
            // with that.
            if let Some(yn) = model.norton_admittance() {
                let bus = layout.dev_bus[d];
                y.add(bus, bus, yn);
            }
        }
        self.ybus = y.finish();
    }

    /// Whether every device is still connected — used only to decide whether a
    /// dead-island report is warranted, since a disconnected unit does not
    /// energize anything.
    pub(crate) fn connected_devices(&self) -> Vec<bool> {
        self.models.iter().map(|m| m.is_connected()).collect()
    }

    /// Buses in components that hold neither a device nor a fixed-voltage bus.
    ///
    /// Such an island has no voltage reference. Its algebraic equations are
    /// still solvable — the load admittances give every bus a path to ground —
    /// but the answer is a de-energized island, not a dynamic one, and reading
    /// it as a trajectory would be a mistake.
    pub(crate) fn dead_islands(&self) -> Vec<Vec<usize>> {
        let layout = self.pattern.layout();
        let mut alive = vec![false; layout.n_bus];
        let connected = self.connected_devices();
        for (d, &bus) in layout.dev_bus.iter().enumerate() {
            alive[bus] |= connected[d];
        }
        for (i, alive) in alive.iter_mut().enumerate() {
            *alive |= self.pattern.is_fixed(i);
        }
        connected_components(&self.ybus)
            .into_iter()
            .filter(|component| !component.iter().any(|&b| alive[b]))
            .collect()
    }

    /// The largest absolute derivative at the current state — the number gate
    /// G1 watches. At a correct equilibrium it is at machine zero.
    pub fn max_derivative(&self) -> f64 {
        let mut f = vec![0.0; self.x0.len()];
        self.derivatives_at(&self.x0, &self.v0, &mut f);
        f.iter().fold(0.0f64, |m, d| m.max(d.abs()))
    }
}

/// Re-solves the algebraic constraint with every differential state held
/// fixed, and stores the result.
///
/// This is what a discontinuity needs: at an event the algebraic variables
/// jump while the differential ones do not, so the integration rule must not
/// be applied across it. It is also what any direct write through
/// [`DynamicSystem::state_mut`] needs, for the same reason.
pub fn settle(system: &mut DynamicSystem, opts: &DynamicsOptions) -> integrator::StepOutcome {
    match opts.backend {
        JacobianBackend::Scalar | JacobianBackend::Block => {
            integrator::solve_algebraic::<RealSparseSystem>(system, opts.tol, opts.max_newton)
        }
        JacobianBackend::KluNative => {
            integrator::solve_algebraic::<KluNativeSystem>(system, opts.tol, opts.max_newton)
        }
        #[cfg(feature = "klu")]
        JacobianBackend::Klu => {
            integrator::solve_algebraic::<KluRealSystem>(system, opts.tol, opts.max_newton)
        }
        #[cfg(feature = "pardiso")]
        JacobianBackend::Pardiso => {
            integrator::solve_algebraic::<PardisoRealSystem>(system, opts.tol, opts.max_newton)
        }
    }
}

/// Integrates `system` from its current state to `opts.end_time`.
///
/// The system is advanced in place, so a caller can run in segments — which is
/// what event handling will do in phase 2.
pub fn run_dynamics(system: &mut DynamicSystem, opts: &DynamicsOptions) -> DynamicsReport {
    match opts.backend {
        JacobianBackend::Scalar | JacobianBackend::Block => {
            integrator::integrate::<RealSparseSystem>(system, opts)
        }
        JacobianBackend::KluNative => integrator::integrate::<KluNativeSystem>(system, opts),
        #[cfg(feature = "klu")]
        JacobianBackend::Klu => integrator::integrate::<KluRealSystem>(system, opts),
        #[cfg(feature = "pardiso")]
        JacobianBackend::Pardiso => integrator::integrate::<PardisoRealSystem>(system, opts),
    }
}
