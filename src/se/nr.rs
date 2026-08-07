//! Weighted-least-squares state estimation by Gauss-Newton.
//!
//! Minimizes `J(x) = ½·(z − h(x))ᵀ W (z − h(x))` by repeatedly solving the
//! normal equations
//!
//! ```text
//! G Δx = HᵀW r,      G = HᵀWH,      r = z − h(x)
//! ```
//!
//! and applying `Δx` to the state. `G` is square and symmetric, so every
//! [`LinearSolver`] backend gridoxide already has for power flow works here
//! unchanged — see [`super::jacobian`] for why that shaped the whole design.
//!
//! # How this differs from the power-flow Newton loop
//!
//! `solver::newton_raphson_cached` drives a *mismatch* to zero: there is an
//! exact solution and it converges onto it. Gauss-Newton has no such target —
//! the measurements are inconsistent by construction, so `r` stays nonzero at
//! the optimum. Convergence is therefore tested on the size of the *step*,
//! not the residual. A converged estimate with a large `J(x)` is not a failure
//! to converge; it means the measurements disagree with each other, which is
//! what bad-data detection is for (phase 6).

use crate::measurement::Measurement;
use crate::solver::{JacobianBackend, LinearSolver};
use crate::sparse::RealSparseSystem;
use crate::types::Bus;

use super::constraints::Constraints;
use super::jacobian::{gain_and_rhs, mask_untouched, measurement_jacobian, StateLayout};
use super::{measurement_functions, SeNetwork};

/// How the estimate finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeStatus {
    /// The step fell below the tolerance.
    Converged,
    /// `max_iter` was reached with the step still above tolerance.
    MaxIterations,
    /// The gain matrix could not be factorized. With WLS this usually means the
    /// system is unobservable — too few independent measurements to pin down
    /// every state variable — rather than a numerical accident.
    ///
    /// [`observability::analyze`](super::observability::analyze) turns this into
    /// a diagnosis, naming the buses and quantities the measurements leave
    /// undetermined. It is not run automatically because it densifies the gain
    /// matrix, which is affordable for a deliberate analysis pass and not for
    /// every solve.
    Singular,
}

/// Which algorithm to estimate with.
///
/// Both minimize the same weighted-least-squares objective and, on a
/// well-conditioned problem, reach the same state — power-grid-model's fixtures
/// accept either method against one expected answer, and gridoxide's tests
/// check that they agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SeMethod {
    /// Gauss-Newton on the nonlinear problem. Slower per iteration — a fresh
    /// Jacobian and factorization each time — and the more accurate of the two.
    #[default]
    NewtonRaphson,
    /// Linearize the measurements into currents, factorize once, iterate on the
    /// right-hand side alone. Faster, at the cost of the approximations
    /// [`super::iterative`] documents.
    IterativeLinear,
}

/// Options for a single estimate.
#[derive(Clone, Copy, Debug)]
pub struct SeOptions {
    /// Convergence threshold on the largest element of `Δx`.
    pub tol: f64,
    pub max_iter: usize,
    pub backend: JacobianBackend,
    /// Which algorithm to use. The backend applies only to
    /// [`SeMethod::NewtonRaphson`]; the iterative-linear method solves a
    /// complex system and has a single path.
    pub method: SeMethod,
}

impl Default for SeOptions {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 20,
            backend: JacobianBackend::Scalar,
            method: SeMethod::default(),
        }
    }
}

/// The outcome of an estimate.
#[derive(Clone, Debug)]
pub struct SeReport {
    pub status: SeStatus,
    pub iterations: usize,
    /// `J(x) = ½·rᵀWr` at the final state — the quantity being minimized, and
    /// the statistic phase 6's chi-squared test is built on.
    pub objective: f64,
    /// Largest `Δx` element of the final step, for diagnosing a run that hit
    /// `max_iter`.
    pub last_step: f64,
    /// Final residuals `z − h(x)`, one per measurement in input order.
    pub residuals: Vec<f64>,
    /// State-vector columns that no measurement constrains at all.
    ///
    /// These are held at their starting value rather than estimated — an
    /// unknown nothing observes cannot be recovered, and moving it would be
    /// inventing information. In practice they are usually the virtual slack
    /// buses gridoxide synthesizes per source: power-grid-model has no such bus
    /// in its state space, so a network it considers fully observable can still
    /// leave gridoxide with surplus unknowns whenever the source's own power is
    /// unmeasured.
    ///
    /// This is *structural* detection only — an all-zero column. A column that
    /// is nonzero but linearly dependent on others is just as unobservable and
    /// is not caught here; that is phase 5's numerical analysis.
    pub unconstrained: Vec<usize>,
}

/// A starting state: every magnitude at 1 p.u. and every angle at zero, with
/// voltage-magnitude measurements applied where they exist.
///
/// The flat start is the standard choice, and seeding it with the voltage
/// measurements costs nothing and starts the buses that *are* measured much
/// closer to their answer. power-grid-model does the same
/// (`se-algorithms.md`'s initialization step).
///
/// Note this leaves every angle at zero, which is wrong by 30° per `clock` step
/// across a phase-shifting transformer — see [`linear_start`], which is the
/// better default wherever a [`SeNetwork`] is available.
pub fn flat_start(buses: &mut [Bus], measurements: &[Measurement]) {
    for b in buses.iter_mut() {
        b.voltage_mag = 1.0;
        b.voltage_ang = 0.0;
    }
    apply_measured_magnitudes(buses, measurements);
}

/// A starting state that carries the network's own structural phase shifts,
/// then applies the voltage-magnitude measurements on top.
///
/// Prefer this to [`flat_start`] wherever a [`SeNetwork`] is at hand. A flat
/// start puts every angle at zero, which is a poor description of any network
/// containing a phase-shifting transformer: a `clock` of 11 offsets one side by
/// 30° (0.52 rad) before any load flows at all. Gauss-Newton started half a
/// radian away from the answer does not merely take longer — on
/// `three_winding_transformer` it converges to a *different* stationary point,
/// reporting an objective of 2.4e2 where the true optimum is 4e-11, with the
/// source node at 0.21 p.u. against its own sensor reading of 1.00.
///
/// [`linear_initial_guess`](crate::network::linear_initial_guess) is the fix
/// gridoxide already had: it solves the linear network with loads as constant
/// admittances, so each transformer's complex tap ratio propagates its phase
/// shift exactly. On a state-estimation document, where `p_specified` is
/// normally unset, it degenerates to a no-load solve — which still carries every
/// phase shift, which is the part that matters here. It is also safe by
/// construction: a singular linear system leaves the flat state untouched rather
/// than failing.
///
/// power-grid-model reaches the same place differently, carrying a per-bus
/// `phase_shift` in its math topology and initializing its voltages with it.
pub fn linear_start(buses: &mut [Bus], net: &SeNetwork, measurements: &[Measurement]) {
    for b in buses.iter_mut() {
        b.voltage_mag = 1.0;
        b.voltage_ang = 0.0;
    }
    crate::network::linear_initial_guess(buses, &net.ybus);
    apply_measured_magnitudes(buses, measurements);
}

/// Overwrites each bus's magnitude with a directly measured one where it exists.
///
/// Shared by [`flat_start`] and [`linear_start`]: a measured magnitude is a
/// better starting value than either produces on its own, and costs nothing.
fn apply_measured_magnitudes(buses: &mut [Bus], measurements: &[Measurement]) {
    for m in measurements {
        if let (crate::measurement::MeasurementKind::VoltageMagnitude,
                crate::measurement::Target::Bus(bus)) = (m.kind, m.target)
        {
            if m.value.is_finite() && m.value > 0.0 {
                buses[bus].voltage_mag = m.value;
            }
        }
    }
}

fn estimate_with<S: LinearSolver>(
    measurements: &[Measurement],
    buses: &mut [Bus],
    net: &SeNetwork,
    layout: &StateLayout,
    constraints: &Constraints,
    options: &SeOptions,
    cache: &mut Option<S>,
) -> SeReport {
    let mut residuals = Vec::new();
    let mut last_step = f64::INFINITY;
    let mut last_unconstrained = Vec::new();

    for iteration in 1..=options.max_iter {
        let h = measurement_functions(measurements, buses, net);
        residuals = measurements
            .iter()
            .zip(&h)
            .map(|(m, &hi)| m.value - hi)
            .collect();

        let rows = measurement_jacobian(measurements, buses, net, layout);
        let (mut triplets, rhs, _) = gain_and_rhs(&rows, measurements, &residuals);

        let n = layout.n_unknowns();
        let mut rhs = rhs;
        rhs.resize(n, 0.0);

        // Zero-injection buses enter as hard constraints rather than as
        // heavily-weighted pseudo-measurements; see `se::constraints` for why
        // that matters for conditioning.
        let (c_values, c_rows) = constraints.evaluate(buses, net, layout);

        let unconstrained = mask_untouched(&mut triplets, &mut rhs, &[&rows, &c_rows], n);

        let (triplets, rhs) =
            super::constraints::augment(triplets, rhs, n, &c_values, &c_rows);
        let n_aug = n + constraints.len();

        // The gain matrix's sparsity pattern is fixed for a topology and a
        // measurement set, so the symbolic factorization is built once and
        // reused across iterations — the same property `PersistentSolver`
        // relies on for power flow.
        if cache.is_none() {
            *cache = S::new(n_aug, &triplets);
        }
        let Some(system) = cache.as_mut() else {
            return SeReport {
                status: SeStatus::Singular,
                iterations: iteration,
                objective: objective(measurements, &residuals),
                last_step,
                residuals,
                unconstrained,
            };
        };
        let Some(dx) = system.factor_and_solve(&triplets, &rhs) else {
            return SeReport {
                status: SeStatus::Singular,
                iterations: iteration,
                objective: objective(measurements, &residuals),
                last_step,
                residuals,
                unconstrained,
            };
        };

        for bus in 0..layout.n_buses {
            if let Some(col) = layout.theta(bus) {
                buses[bus].voltage_ang += dx[col];
            }
            buses[bus].voltage_mag += dx[layout.vmag(bus)];
        }

        last_unconstrained = unconstrained.clone();
        // Only the leading `n` entries are the state step; the tail holds the
        // Lagrange multipliers, which say how hard each constraint is pulling
        // and must not be mistaken for a lack of convergence.
        last_step = dx[..n].iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if last_step < options.tol {
            // Recompute residuals at the state actually returned, so the
            // reported objective describes the answer rather than the
            // second-to-last iterate.
            let h = measurement_functions(measurements, buses, net);
            let residuals: Vec<f64> = measurements
                .iter()
                .zip(&h)
                .map(|(m, &hi)| m.value - hi)
                .collect();
            return SeReport {
                status: SeStatus::Converged,
                iterations: iteration,
                objective: objective(measurements, &residuals),
                last_step,
                residuals,
                unconstrained,
            };
        }
    }

    SeReport {
        status: SeStatus::MaxIterations,
        iterations: options.max_iter,
        objective: objective(measurements, &residuals),
        last_step,
        residuals,
        unconstrained: last_unconstrained,
    }
}

/// `J(x) = ½·rᵀWr`, skipping zero-weight (infinite-sigma) rows.
fn objective(measurements: &[Measurement], residuals: &[f64]) -> f64 {
    measurements
        .iter()
        .zip(residuals)
        .filter(|(m, _)| m.weight().is_finite() && m.weight() != 0.0)
        .map(|(m, &r)| m.weight() * r * r)
        .sum::<f64>()
        / 2.0
}

/// A reusable estimator that keeps its factorization between solves.
///
/// [`estimate`] is the one-shot entry point: it builds the state layout, the
/// constraint set and the solver's symbolic factorization, uses them once, and
/// throws all three away. For a single estimate that is the whole cost of doing
/// business; for a sequence of them — a time series, a scenario sweep, a live
/// feed refreshing the same telemetry — it is repeated work, and measurably so.
/// Power flow solved the identical problem with
/// [`PersistentSolver`](crate::solver::PersistentSolver); this is its
/// counterpart.
///
/// # When a cached factorization stays valid
///
/// The gain matrix's sparsity pattern depends on the topology *and* on the
/// measurement set's structure — which quantities are measured where — but not
/// on their values. So this stays valid while:
///
/// - the network is unchanged, and
/// - the same measurements exist, on the same targets, in the same order.
///
/// New readings on the existing sensors are the case this is built for and need
/// no reset. Sigmas may change too: a weight scales `G`'s values, never its
/// pattern. Adding, removing or reordering a measurement does invalidate it, as
/// does anything that changes whether an angle is measured at all — that flips
/// whether a reference is pinned, and with it the number of unknowns. Call
/// [`reset`](Self::reset) then.
///
/// Nothing here detects a stale cache. The same is true of `PersistentSolver`,
/// and for the same reason: the check would cost as much as the work it saves.
pub struct PersistentEstimator {
    options: SeOptions,
    layout: Option<StateLayout>,
    constraints: Option<Constraints>,
    scalar: Option<RealSparseSystem>,
    klu_native: Option<crate::klu_native::KluNativeSystem>,
    #[cfg(feature = "klu")]
    klu: Option<crate::sparse_klu::KluRealSystem>,
    #[cfg(feature = "pardiso")]
    pardiso: Option<crate::sparse_pardiso::PardisoRealSystem>,
}

impl PersistentEstimator {
    pub fn new(options: SeOptions) -> Self {
        Self {
            options,
            layout: None,
            constraints: None,
            scalar: None,
            klu_native: None,
            #[cfg(feature = "klu")]
            klu: None,
            #[cfg(feature = "pardiso")]
            pardiso: None,
        }
    }

    /// Discards every cached artifact. Call after changing the network or the
    /// structure of the measurement set.
    pub fn reset(&mut self) {
        self.layout = None;
        self.constraints = None;
        self.scalar = None;
        self.klu_native = None;
        #[cfg(feature = "klu")]
        {
            self.klu = None;
        }
        #[cfg(feature = "pardiso")]
        {
            self.pardiso = None;
        }
    }

    /// Estimates, reusing whatever this estimator already holds.
    ///
    /// The iterative-linear method has nothing to reuse *between* calls — it
    /// factorizes once per call already, and its matrix depends on the starting
    /// magnitudes, which move — so it delegates to the one-shot path. The
    /// saving here is Newton-Raphson's, which is also the slower method and so
    /// the one that wanted it.
    pub fn estimate(
        &mut self,
        measurements: &[Measurement],
        buses: &mut [Bus],
        net: &SeNetwork,
    ) -> SeReport {
        if self.options.method == SeMethod::IterativeLinear {
            return super::iterative::estimate(measurements, buses, net, &self.options);
        }

        if self.layout.is_none() {
            self.layout = Some(StateLayout::new(buses, measurements, net));
        }
        if self.constraints.is_none() {
            self.constraints = Some(Constraints::new(&net.zero_injection));
        }
        let layout = self.layout.as_ref().expect("just populated");
        let constraints = self.constraints.as_ref().expect("just populated");

        match self.options.backend {
            JacobianBackend::KluNative => estimate_with(
                measurements, buses, net, layout, constraints, &self.options, &mut self.klu_native,
            ),
            #[cfg(feature = "klu")]
            JacobianBackend::Klu => estimate_with(
                measurements, buses, net, layout, constraints, &self.options, &mut self.klu,
            ),
            #[cfg(feature = "pardiso")]
            JacobianBackend::Pardiso => estimate_with(
                measurements, buses, net, layout, constraints, &self.options, &mut self.pardiso,
            ),
            JacobianBackend::Block | JacobianBackend::Scalar => estimate_with(
                measurements, buses, net, layout, constraints, &self.options, &mut self.scalar,
            ),
        }
    }
}

/// Runs weighted-least-squares state estimation, updating `buses` in place.
///
/// `buses` supplies the starting state; use [`flat_start`] for the standard
/// one. The estimate covers every bus, including the virtual slack buses
/// gridoxide synthesizes for sources — in state estimation a source's voltage
/// is an unknown like any other, not a boundary condition.
pub fn estimate(
    measurements: &[Measurement],
    buses: &mut [Bus],
    net: &SeNetwork,
    options: &SeOptions,
) -> SeReport {
    if options.method == SeMethod::IterativeLinear {
        return super::iterative::estimate(measurements, buses, net, options);
    }
    let layout = StateLayout::new(buses, measurements, net);
    let constraints = Constraints::new(&net.zero_injection);
    match options.backend {
        JacobianBackend::KluNative => estimate_with(
            measurements, buses, net, &layout, &constraints, options,
            &mut None::<crate::klu_native::KluNativeSystem>,
        ),
        #[cfg(feature = "klu")]
        JacobianBackend::Klu => estimate_with(
            measurements, buses, net, &layout, &constraints, options,
            &mut None::<crate::sparse_klu::KluRealSystem>,
        ),
        #[cfg(feature = "pardiso")]
        JacobianBackend::Pardiso => estimate_with(
            measurements, buses, net, &layout, &constraints, options,
            &mut None::<crate::sparse_pardiso::PardisoRealSystem>,
        ),
        // `Block` assumes power flow's two-unknowns-per-bus structure, which the
        // 2N−1 state vector here does not have (the reference bus contributes
        // one unknown, not two), so it falls back to the scalar path rather
        // than silently mis-assembling. Every other backend is a general sparse
        // LU and carries the gain matrix unchanged.
        JacobianBackend::Block | JacobianBackend::Scalar => estimate_with(
            measurements, buses, net, &layout, &constraints, options,
            &mut None::<RealSparseSystem>,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::{MeasurementKind, Target};

    fn m(kind: MeasurementKind, target: Target, value: f64, sigma: f64) -> Measurement {
        Measurement { kind, target, value, sigma }
    }

    /// With enough consistent measurements, the estimate should reproduce the
    /// state they were generated from.
    ///
    /// The measurements here are *exact* readings taken from a known state, so
    /// a correct estimator has to drive the residuals to zero — which makes
    /// this a test of the whole loop (h, H, gain assembly, update) rather than
    /// of any one piece.
    #[test]
    fn recovers_the_state_its_measurements_came_from() {
        let (net, truth) = crate::se::tests::two_bus_net();
        let layout = StateLayout::new(&truth, &[], &net);

        // Read every quantity off the true state, noise-free.
        let h_targets = vec![
            (MeasurementKind::VoltageMagnitude, Target::Bus(0)),
            (MeasurementKind::VoltageMagnitude, Target::Bus(1)),
            (MeasurementKind::ActivePower, Target::Bus(1)),
            (MeasurementKind::ReactivePower, Target::Bus(1)),
            (
                MeasurementKind::ActivePower,
                Target::BranchTerminal { branch: 0, terminal: crate::branch_flow::Terminal::From },
            ),
            (
                MeasurementKind::ReactivePower,
                Target::BranchTerminal { branch: 0, terminal: crate::branch_flow::Terminal::From },
            ),
        ];
        let probe: Vec<Measurement> = h_targets
            .iter()
            .map(|&(kind, target)| m(kind, target, 0.0, 1.0))
            .collect();
        let exact = measurement_functions(&probe, &truth, &net);
        let measurements: Vec<Measurement> = probe
            .iter()
            .zip(&exact)
            .map(|(p, &v)| Measurement { value: v, ..*p })
            .collect();

        let mut buses = truth.clone();
        flat_start(&mut buses, &measurements);
        let report = estimate(&measurements, &mut buses, &net, &SeOptions::default());

        assert_eq!(report.status, SeStatus::Converged, "report={report:?}");
        assert!(report.objective < 1e-16, "objective={}", report.objective);
        for (i, (got, want)) in buses.iter().zip(&truth).enumerate() {
            assert!(
                (got.voltage_mag - want.voltage_mag).abs() < 1e-8,
                "bus {i} magnitude: {} vs {}",
                got.voltage_mag,
                want.voltage_mag
            );
            // Angles are only determined up to the reference, which is pinned.
            let reference = layout.angle_ref.expect("no angle measurements here, so one is pinned");
            let d_ang = (got.voltage_ang - want.voltage_ang)
                - (buses[reference].voltage_ang - truth[reference].voltage_ang);
            assert!(d_ang.abs() < 1e-8, "bus {i} angle off by {d_ang}");
        }
    }

    /// Measurements that leave state variables untouched do not sink the whole
    /// estimate: the unobservable unknowns are pinned and named, and the rest
    /// is solved.
    ///
    /// A single voltage magnitude cannot determine four unknowns, and the naive
    /// result is a singular gain matrix. Reporting that as an outright failure
    /// would be needlessly destructive — the one thing that *is* measured is
    /// perfectly well determined — so the untouched columns are masked out and
    /// listed in `unconstrained` instead. Note this is structural detection
    /// only; a column that is present but linearly dependent still surfaces as
    /// [`SeStatus::Singular`], which is what phase 5 exists to diagnose.
    #[test]
    fn unobservable_unknowns_are_pinned_and_reported() {
        let (net, truth) = crate::se::tests::two_bus_net();
        let measurements = vec![m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 1.03, 0.01)];
        let mut buses = truth.clone();
        flat_start(&mut buses, &measurements);
        let report = estimate(&measurements, &mut buses, &net, &SeOptions::default());

        assert_eq!(report.status, SeStatus::Converged, "report={report:?}");
        // Three of the four unknowns are untouched: the other bus's magnitude
        // and both angles bar the pinned reference.
        assert!(
            !report.unconstrained.is_empty(),
            "a single measurement cannot constrain the whole state: {report:?}"
        );
        // The measured quantity is still recovered exactly.
        assert!((buses[0].voltage_mag - 1.03).abs() < 1e-9, "got {}", buses[0].voltage_mag);
        // And the unconstrained ones kept their starting value rather than
        // drifting somewhere arbitrary.
        assert!((buses[1].voltage_mag - 1.0).abs() < 1e-12, "got {}", buses[1].voltage_mag);
    }

    /// A reused estimator must produce exactly what a fresh one does.
    ///
    /// The whole premise of caching is that the second solve is the same
    /// computation with less setup, so anything else is a bug rather than a
    /// tradeoff. Checked bit-for-bit, not approximately.
    #[test]
    fn reuse_reproduces_a_fresh_estimate_exactly() {
        let (net, truth) = crate::se::tests::two_bus_net();
        let probe = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 0.0, 0.01),
            m(MeasurementKind::VoltageMagnitude, Target::Bus(1), 0.0, 0.01),
            m(MeasurementKind::ActivePower, Target::Bus(1), 0.0, 0.01),
            m(MeasurementKind::ReactivePower, Target::Bus(1), 0.0, 0.01),
        ];
        let exact = measurement_functions(&probe, &truth, &net);
        let measurements: Vec<Measurement> = probe
            .iter()
            .zip(&exact)
            .map(|(p, &v)| Measurement { value: v, ..*p })
            .collect();

        let mut fresh = truth.clone();
        flat_start(&mut fresh, &measurements);
        let one_shot = estimate(&measurements, &mut fresh, &net, &SeOptions::default());

        let mut persistent = PersistentEstimator::new(SeOptions::default());
        let mut reused = Vec::new();
        for _ in 0..3 {
            let mut buses = truth.clone();
            flat_start(&mut buses, &measurements);
            let report = persistent.estimate(&measurements, &mut buses, &net);
            reused.push((report, buses));
        }

        for (i, (report, buses)) in reused.iter().enumerate() {
            assert_eq!(report.status, one_shot.status, "solve {i} status");
            assert_eq!(report.iterations, one_shot.iterations, "solve {i} iterations");
            for (bus, (got, want)) in buses.iter().zip(&fresh).enumerate() {
                assert_eq!(
                    got.voltage_mag, want.voltage_mag,
                    "solve {i} bus {bus}: reuse changed the answer"
                );
                assert_eq!(got.voltage_ang, want.voltage_ang, "solve {i} bus {bus} angle");
            }
        }
    }

    /// New readings on the same sensors are the case the cache is built for:
    /// values may move freely, only the *structure* has to hold still.
    #[test]
    fn reuse_tracks_changed_measurement_values() {
        let (net, truth) = crate::se::tests::two_bus_net();
        let probe = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 0.0, 0.01),
            m(MeasurementKind::VoltageMagnitude, Target::Bus(1), 0.0, 0.01),
            m(MeasurementKind::ActivePower, Target::Bus(1), 0.0, 0.01),
            m(MeasurementKind::ReactivePower, Target::Bus(1), 0.0, 0.01),
        ];
        let exact = measurement_functions(&probe, &truth, &net);
        let first: Vec<Measurement> = probe
            .iter()
            .zip(&exact)
            .map(|(p, &v)| Measurement { value: v, ..*p })
            .collect();
        // A different operating point, same sensors.
        let second: Vec<Measurement> = first
            .iter()
            .map(|m| Measurement { value: m.value * 0.98, ..*m })
            .collect();

        let mut persistent = PersistentEstimator::new(SeOptions::default());
        let mut a = truth.clone();
        flat_start(&mut a, &first);
        persistent.estimate(&first, &mut a, &net);

        let mut b = truth.clone();
        flat_start(&mut b, &second);
        persistent.estimate(&second, &mut b, &net);

        // Against a fresh estimator on the second set.
        let mut reference = truth.clone();
        flat_start(&mut reference, &second);
        estimate(&second, &mut reference, &net, &SeOptions::default());

        for (bus, (got, want)) in b.iter().zip(&reference).enumerate() {
            assert!(
                (got.voltage_mag - want.voltage_mag).abs() < 1e-12,
                "bus {bus}: cached run gave {} against {}",
                got.voltage_mag,
                want.voltage_mag
            );
        }
        assert!(
            (a[1].voltage_mag - b[1].voltage_mag).abs() > 1e-6,
            "the two measurement sets should give visibly different answers, \
             or this test proves nothing"
        );
    }

    /// A noisy measurement set still converges; the leftover objective is the
    /// disagreement, not a convergence failure. Asserting both is what
    /// distinguishes "converged with imperfect data" from "did not converge".
    #[test]
    fn converges_with_inconsistent_measurements() {
        let (net, truth) = crate::se::tests::two_bus_net();
        let probe = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 0.0, 0.01),
            m(MeasurementKind::VoltageMagnitude, Target::Bus(1), 0.0, 0.01),
            m(MeasurementKind::ActivePower, Target::Bus(1), 0.0, 0.01),
            m(MeasurementKind::ReactivePower, Target::Bus(1), 0.0, 0.01),
            m(
                MeasurementKind::ActivePower,
                Target::BranchTerminal { branch: 0, terminal: crate::branch_flow::Terminal::From },
                0.0,
                0.01,
            ),
        ];
        let exact = measurement_functions(&probe, &truth, &net);
        // Perturb each reading by a different amount, so no state explains them all.
        let measurements: Vec<Measurement> = probe
            .iter()
            .zip(&exact)
            .enumerate()
            .map(|(i, (p, &v))| Measurement { value: v + 0.002 * (i as f64 - 2.0), ..*p })
            .collect();

        let mut buses = truth.clone();
        flat_start(&mut buses, &measurements);
        let report = estimate(&measurements, &mut buses, &net, &SeOptions::default());

        assert_eq!(report.status, SeStatus::Converged, "report={report:?}");
        assert!(report.objective > 0.0, "inconsistent data should leave residual disagreement");
        assert!(report.last_step < SeOptions::default().tol);
    }
}
