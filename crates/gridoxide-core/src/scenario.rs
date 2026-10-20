//! What a batch varies: per-bus overrides and branch outages against a
//! shared template, and the errors a batch reports for them.
//!
//! Shared by the AC [`batch::BatchSolver`](crate::batch::BatchSolver) and the
//! DC [`linear::batch::DcBatchSolver`](crate::linear::batch::DcBatchSolver).
//! Kept apart from either so that `linear` need not depend on the AC batch
//! solver, which itself sits above `linear`; `batch` re-exports everything
//! here, so `gridoxide::batch::Scenario` and friends resolve as before.

use std::fmt;

use crate::types::Bus;

/// A per-bus change applied on top of the batch's shared bus template.
/// `None` fields leave the template's value alone.
///
/// Deliberately cannot change `bus_type`: switching `PV`↔`PQ` changes
/// `n_unknowns`, which invalidates the cached symbolic factorization that
/// makes batching worth doing at all. A caller that needs per-scenario bus
/// types wants separate batches, one per type assignment.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BusOverride {
    pub bus: usize,
    pub p_spec: Option<f64>,
    pub q_spec: Option<f64>,
    /// Voltage setpoint for a `Slack`/`PV` bus. Ignored for `PQ` buses,
    /// whose magnitude is an unknown the solver overwrites.
    pub voltage_mag: Option<f64>,
}

impl BusOverride {
    pub fn new(bus: usize) -> Self {
        Self {
            bus,
            ..Default::default()
        }
    }

    pub fn p(mut self, p_spec: f64) -> Self {
        self.p_spec = Some(p_spec);
        self
    }

    pub fn q(mut self, q_spec: f64) -> Self {
        self.q_spec = Some(q_spec);
        self
    }

    pub fn vm(mut self, voltage_mag: f64) -> Self {
        self.voltage_mag = Some(voltage_mag);
        self
    }
}

/// One scenario in a batch: a set of bus-value overrides against the shared
/// template.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scenario {
    pub bus_overrides: Vec<BusOverride>,
    /// Flat branch indices to take out of service — lines first, then
    /// transformers, the same space `branch_flow::branch_params` uses.
    ///
    /// Honored by
    /// [`BatchSolver::solve_contingencies`](crate::batch::BatchSolver::solve_contingencies),
    /// which takes the branch lists an outage needs.
    /// [`BatchSolver::solve`](crate::batch::BatchSolver::solve) still rejects a non-empty
    /// list with [`BatchError::OutagesUnsupported`] rather than ignoring it: it
    /// is handed a finished Y-bus, in which the branches have already been
    /// summed away and cannot be removed.
    pub branch_outages: Vec<usize>,
}

impl Scenario {
    pub fn new(bus_overrides: Vec<BusOverride>) -> Self {
        Self {
            bus_overrides,
            branch_outages: Vec::new(),
        }
    }
}

/// Builds a scenario scaling every bus's `p_spec`/`q_spec` by `factor` —
/// the standard time-series/QSTS and Monte Carlo shape, and what
/// `scripts/bench/bench_batch.py` drives.
///
/// Buses with zero injection stay zero, so this touches only buses that
/// actually carry load or generation.
pub fn uniform_load_scaling(buses_template: &[Bus], factor: f64) -> Scenario {
    let overrides = buses_template
        .iter()
        .filter(|b| b.p_spec != 0.0 || b.q_spec != 0.0)
        .map(|b| {
            BusOverride::new(b.idx)
                .p(b.p_spec * factor)
                .q(b.q_spec * factor)
        })
        .collect();
    Scenario::new(overrides)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchError {
    /// `Scenario::branch_outages` was non-empty on
    /// [`BatchSolver::solve`](crate::batch::BatchSolver::solve), which cannot
    /// honor it. See that field's doc comment, and use
    /// [`BatchSolver::solve_contingencies`](crate::batch::BatchSolver::solve_contingencies)
    /// instead.
    OutagesUnsupported { scenario: usize },
    /// A `BusOverride::bus` was not a valid index into the bus template.
    BusOutOfRange {
        scenario: usize,
        bus: usize,
        n_buses: usize,
    },
    /// A `Scenario::branch_outages` entry was not a valid flat branch index.
    BranchOutOfRange {
        scenario: usize,
        branch: usize,
        n_branches: usize,
    },
    /// `rayon` refused to build the requested thread pool.
    ThreadPool(String),
    /// The reduced susceptance matrix was singular, so no scenario in a
    /// [`linear::batch::DcBatchSolver`](crate::linear::batch::DcBatchSolver)
    /// batch can be solved.
    ///
    /// Unlike the Newton path — where a scenario that fails is reported *in*
    /// its own result and never poisons the batch — this is a property of the
    /// shared topology rather than of any one scenario, so it fails the whole
    /// call.
    DcSingular,
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchError::OutagesUnsupported { scenario } => write!(
                f,
                "scenario {scenario} requests branch outages, which `solve` cannot honor from a \
                 finished Y-bus — use `BatchSolver::solve_contingencies`, which takes the branch \
                 lists"
            ),
            BatchError::BusOutOfRange {
                scenario,
                bus,
                n_buses,
            } => write!(
                f,
                "scenario {scenario} overrides bus {bus}, but the template has only {n_buses} bus(es)"
            ),
            BatchError::BranchOutOfRange {
                scenario,
                branch,
                n_branches,
            } => write!(
                f,
                "scenario {scenario} outages branch {branch}, but the network has only \
                 {n_branches} branch(es)"
            ),
            BatchError::ThreadPool(e) => write!(f, "building the rayon thread pool failed: {e}"),
            BatchError::DcSingular => write!(
                f,
                "the reduced susceptance matrix is singular, so no DC scenario can be solved"
            ),
        }
    }
}

impl std::error::Error for BatchError {}
