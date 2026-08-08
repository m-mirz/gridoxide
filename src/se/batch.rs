//! Batched state estimation: one topology and one measurement *structure*,
//! many readings, estimated across cores.
//!
//! This is the shape production state estimation actually has. A control room
//! does not estimate one snapshot; it estimates the same grid every few seconds
//! from the same instruments, with only the numbers changing. [`batch::BatchSolver`]
//! already does the equivalent for power flow, and the two share a premise:
//!
//! - Every scenario shares one gain-matrix sparsity pattern, so each worker
//!   keeps one [`PersistentEstimator`] and amortizes a single symbolic
//!   factorization across its whole share of the batch.
//! - Scenarios are independent, so the only coordination is handing out indices.
//!
//! [`PersistentEstimator`]'s cache-validity condition was written down before
//! any of this existed and turns out to be exactly the batch invariant: valid
//! while the topology and the measurement set's *structure* are unchanged, with
//! values and sigmas free to move. [`MeasurementOverride`] is that condition
//! made into a type.
//!
//! [`batch::BatchSolver`]: crate::batch::BatchSolver

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::measurement::{Measurement, MeasurementKind, Target};
use crate::types::Bus;

use super::nr::{linear_start, PersistentEstimator, SeOptions, SeReport};
use super::SeNetwork;

/// A per-measurement change against the batch's shared template.
///
/// Deliberately cannot change `kind` or `target`. Either would move the gain
/// matrix's sparsity pattern and invalidate the cached symbolic factorization
/// that makes batching worth doing at all — the exact analogue of
/// [`BusOverride`](crate::batch::BusOverride) refusing to change `bus_type`. A
/// caller whose sensor set genuinely changes shape wants a second batch.
///
/// A scenario that does not *have* some reading sets its sigma to infinity
/// rather than dropping the row: the row then exists structurally and weighs
/// exactly nothing, which is what keeps one pattern serving every scenario.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeasurementOverride {
    pub measurement: usize,
    pub value: Option<f64>,
    pub sigma: Option<f64>,
}

impl MeasurementOverride {
    pub fn new(measurement: usize) -> Self {
        Self { measurement, value: None, sigma: None }
    }

    pub fn value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }

    /// Marks the row present-but-uninformative, i.e. an infinite sigma.
    pub fn absent(mut self) -> Self {
        self.sigma = Some(f64::INFINITY);
        self
    }
}

/// One scenario: the template's measurements with some of their values and
/// sigmas replaced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeScenario {
    pub overrides: Vec<MeasurementOverride>,
}

impl SeScenario {
    pub fn new(overrides: Vec<MeasurementOverride>) -> Self {
        Self { overrides }
    }
}

/// One scenario's outcome.
#[derive(Clone, Debug)]
pub struct SeBatchResult {
    pub buses: Vec<Bus>,
    pub report: SeReport,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SeBatchError {
    /// An override named a row the template does not have.
    MeasurementOutOfRange { scenario: usize, measurement: usize, n: usize },
    /// A scenario measures a voltage angle where the template does not, or the
    /// reverse.
    ///
    /// That flips whether [`StateLayout`](super::jacobian::StateLayout) pins an
    /// angle reference, which changes the number of unknowns. It is a different
    /// problem, not a different right-hand side, and the shared factorization
    /// cannot serve both — so it is rejected rather than silently answered
    /// wrongly.
    AngleReferenceChanged { scenario: usize },
    /// Two scenarios disagree about which rows exist at all, so no single
    /// template covers them. Only [`align_scenarios`] raises this.
    IncompatibleScenarios,
    /// `rayon` refused to build the requested thread pool.
    ThreadPool(String),
}

impl fmt::Display for SeBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeBatchError::MeasurementOutOfRange { scenario, measurement, n } => write!(
                f,
                "scenario {scenario} overrides measurement {measurement}, but the template has \
                 only {n} measurement(s)"
            ),
            SeBatchError::AngleReferenceChanged { scenario } => write!(
                f,
                "scenario {scenario} changes whether any voltage angle is measured, which changes \
                 the number of unknowns — run it as its own batch"
            ),
            SeBatchError::IncompatibleScenarios => {
                write!(f, "no single measurement template covers every scenario")
            }
            SeBatchError::ThreadPool(e) => write!(f, "building the rayon thread pool failed: {e}"),
        }
    }
}

impl std::error::Error for SeBatchError {}

/// Whether any voltage-angle row carries weight, which is what decides if
/// [`StateLayout`](super::jacobian::StateLayout) pins a reference.
fn angle_is_measured(measurements: &[Measurement]) -> bool {
    measurements
        .iter()
        .any(|m| m.kind == MeasurementKind::VoltageAngle && m.weight() > 0.0)
}

/// Applies a scenario's overrides to a copy of the template.
fn apply(template: &[Measurement], scenario: &SeScenario) -> Vec<Measurement> {
    let mut out = template.to_vec();
    for ov in &scenario.overrides {
        let m = &mut out[ov.measurement];
        if let Some(v) = ov.value {
            m.value = v;
        }
        if let Some(s) = ov.sigma {
            m.sigma = s;
        }
    }
    out
}

/// Builds one template covering every scenario, plus the overrides that
/// reproduce each.
///
/// Written for driving power-grid-model's own batch fixtures, where each
/// scenario arrives as a whole document rather than as a diff, and where the
/// base document may carry no readings at all — `sensor-update-initially-empty`
/// has two valueless voltage sensors that only the scenarios fill in, so a
/// template taken from the base would be empty and cover nothing.
///
/// Rows are keyed by `(kind, target)`, which is precisely the structure
/// [`MeasurementOverride`] refuses to vary. A row missing from some scenario is
/// given an infinite sigma there rather than dropped.
pub fn align_scenarios(
    scenarios: &[Vec<Measurement>],
) -> Result<(Vec<Measurement>, Vec<SeScenario>), SeBatchError> {
    if scenarios.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    // Union in first-seen order, so the template is deterministic and a
    // single-scenario batch reproduces that scenario's own row order exactly.
    let mut index: HashMap<(MeasurementKind, Target), usize> = HashMap::new();
    let mut template: Vec<Measurement> = Vec::new();
    for scenario in scenarios {
        for m in scenario {
            index.entry((m.kind, m.target)).or_insert_with(|| {
                template.push(*m);
                template.len() - 1
            });
        }
    }

    let mut out = Vec::with_capacity(scenarios.len());
    for scenario in scenarios {
        let mut present = vec![false; template.len()];
        let mut overrides = Vec::with_capacity(template.len());
        for m in scenario {
            let i = index[&(m.kind, m.target)];
            if present[i] {
                // The same quantity twice in one scenario means the caller has
                // not aggregated, and merging here would silently invent a
                // rule. `measurements_from_pgm` already does that job.
                return Err(SeBatchError::IncompatibleScenarios);
            }
            present[i] = true;
            overrides.push(MeasurementOverride::new(i).value(m.value).sigma(m.sigma));
        }
        for (i, &seen) in present.iter().enumerate() {
            if !seen {
                overrides.push(MeasurementOverride::new(i).absent());
            }
        }
        overrides.sort_unstable_by_key(|o| o.measurement);
        out.push(SeScenario::new(overrides));
    }
    Ok((template, out))
}

/// Estimates many scenarios over one shared topology and measurement structure.
pub struct SeBatchSolver {
    options: SeOptions,
    /// Built once and reused across `estimate` calls, so repeated batches never
    /// pay thread-creation cost. `None` uses rayon's global pool — which faer
    /// also uses internally, and sharing it is what stops nested parallelism
    /// oversubscribing the machine.
    pool: Option<rayon::ThreadPool>,
}

impl SeBatchSolver {
    /// Uses rayon's global thread pool (honors `RAYON_NUM_THREADS`).
    pub fn new(options: SeOptions) -> Self {
        Self { options, pool: None }
    }

    /// Uses a dedicated pool of exactly `threads` workers.
    pub fn with_threads(options: SeOptions, threads: usize) -> Result<Self, SeBatchError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| SeBatchError::ThreadPool(e.to_string()))?;
        Ok(Self { options, pool: Some(pool) })
    }

    /// Workers this solver will use for a batch.
    pub fn threads(&self) -> usize {
        match &self.pool {
            Some(p) => p.current_num_threads(),
            None => rayon::current_num_threads(),
        }
    }

    /// Estimates every scenario, returning one result each **in scenario order
    /// regardless of thread count**.
    ///
    /// Each scenario starts from `buses_template` with a fresh
    /// [`linear_start`], so scenarios never influence each other's starting
    /// point — results are identical to a sequential loop over
    /// [`PersistentEstimator::estimate`], which `tests/se_batch_test.rs`
    /// asserts bit-for-bit.
    ///
    /// A scenario that fails to converge is *not* an error: its report carries
    /// the status and the rest of the batch is unaffected, matching
    /// [`BatchSolver::solve`](crate::batch::BatchSolver::solve).
    pub fn estimate(
        &self,
        buses_template: &[Bus],
        net: &SeNetwork,
        measurements_template: &[Measurement],
        scenarios: &[SeScenario],
    ) -> Result<Vec<SeBatchResult>, SeBatchError> {
        let n = measurements_template.len();
        let template_has_angle = angle_is_measured(measurements_template);
        for (i, sc) in scenarios.iter().enumerate() {
            for ov in &sc.overrides {
                if ov.measurement >= n {
                    return Err(SeBatchError::MeasurementOutOfRange {
                        scenario: i,
                        measurement: ov.measurement,
                        n,
                    });
                }
            }
            if angle_is_measured(&apply(measurements_template, sc)) != template_has_angle {
                return Err(SeBatchError::AngleReferenceChanged { scenario: i });
            }
        }
        if scenarios.is_empty() {
            return Ok(Vec::new());
        }

        let next = AtomicUsize::new(0);
        let collected: Mutex<Vec<(usize, SeBatchResult)>> =
            Mutex::new(Vec::with_capacity(scenarios.len()));
        let n_workers = self.threads().max(1).min(scenarios.len());
        let options = self.options;

        let run = || {
            rayon::scope(|s| {
                for _ in 0..n_workers {
                    s.spawn(|_| {
                        // Constructed *inside* the spawned closure so it never
                        // crosses a thread boundary, for the same reason
                        // `batch::BatchSolver::solve` does it: a
                        // `PersistentEstimator` can hold a `KluRealSystem`
                        // (raw `*mut klu_symbolic`) or a `PardisoRealSystem`
                        // (`[*mut c_void; 64]`), neither of which is `Send`.
                        // `rayon::scope`/`spawn` require only the *closure* to
                        // be `Send`, and work-stealing never migrates a task
                        // mid-execution, so this local stays on one thread for
                        // its lifetime. `par_iter().map_init` would demand
                        // `T: Send` and silently exclude both backends.
                        let mut estimator = PersistentEstimator::new(options);

                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= scenarios.len() {
                                break;
                            }
                            let measurements = apply(measurements_template, &scenarios[i]);
                            let mut buses = buses_template.to_vec();
                            linear_start(&mut buses, net, &measurements);
                            let report = estimator.estimate(&measurements, &mut buses, net);
                            collected
                                .lock()
                                .expect("batch result mutex poisoned by a panicking worker")
                                .push((i, SeBatchResult { buses, report }));
                        }
                    });
                }
            });
        };

        match &self.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }

        let mut out = collected.into_inner().expect("batch result mutex poisoned");
        out.sort_by_key(|(i, _)| *i);
        Ok(out.into_iter().map(|(_, r)| r).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::Target;

    fn m(kind: MeasurementKind, target: Target, value: f64, sigma: f64) -> Measurement {
        Measurement { kind, target, value, sigma }
    }

    /// The union is taken in first-seen order, and a row a scenario lacks comes
    /// back with an infinite sigma rather than missing.
    #[test]
    fn align_unions_scenarios_and_marks_absent_rows() {
        let a = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 1.0, 0.1),
            m(MeasurementKind::ActivePower, Target::Bus(1), 0.5, 0.2),
        ];
        let b = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 1.1, 0.1),
            m(MeasurementKind::ReactivePower, Target::Bus(1), 0.3, 0.2),
        ];
        let (template, scenarios) = align_scenarios(&[a, b]).unwrap();

        assert_eq!(template.len(), 3, "union of two rows and one shared");
        assert_eq!(template[0].kind, MeasurementKind::VoltageMagnitude);
        assert_eq!(template[1].kind, MeasurementKind::ActivePower);
        assert_eq!(template[2].kind, MeasurementKind::ReactivePower);

        let first = apply(&template, &scenarios[0]);
        assert_eq!(first[0].value, 1.0);
        assert_eq!(first[1].value, 0.5);
        assert!(first[2].sigma.is_infinite(), "scenario 0 has no reactive row");
        assert_eq!(first[2].weight(), 0.0);

        let second = apply(&template, &scenarios[1]);
        assert_eq!(second[0].value, 1.1);
        assert!(second[1].sigma.is_infinite(), "scenario 1 has no active row");
        assert_eq!(second[2].value, 0.3);
    }

    /// A single scenario must reproduce its own row order, so that a batch of
    /// one is indistinguishable from a one-shot estimate.
    #[test]
    fn a_single_scenario_round_trips_exactly() {
        let one = vec![
            m(MeasurementKind::ActivePower, Target::Bus(2), 0.7, 0.3),
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 1.0, 0.1),
        ];
        let (template, scenarios) = align_scenarios(std::slice::from_ref(&one)).unwrap();
        assert_eq!(apply(&template, &scenarios[0]), one);
    }

    /// Two readings of one quantity in a single scenario are a caller error:
    /// merging them here would invent an aggregation rule
    /// `measurements_from_pgm` already owns.
    #[test]
    fn a_duplicated_quantity_in_one_scenario_is_rejected() {
        let dup = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 1.0, 0.1),
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 1.2, 0.1),
        ];
        assert_eq!(align_scenarios(&[dup]), Err(SeBatchError::IncompatibleScenarios));
    }
}
