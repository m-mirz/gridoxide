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
use super::jacobian::{gain_and_rhs, measurement_jacobian, StateLayout};
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
    /// every state variable — rather than a numerical accident. Phase 5's
    /// observability analysis exists to say which part.
    Singular,
}

/// Options for a single estimate.
#[derive(Clone, Copy, Debug)]
pub struct SeOptions {
    /// Convergence threshold on the largest element of `Δx`.
    pub tol: f64,
    pub max_iter: usize,
    pub backend: JacobianBackend,
}

impl Default for SeOptions {
    fn default() -> Self {
        Self { tol: 1e-8, max_iter: 20, backend: JacobianBackend::Scalar }
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
pub fn flat_start(buses: &mut [Bus], measurements: &[Measurement]) {
    for b in buses.iter_mut() {
        b.voltage_mag = 1.0;
        b.voltage_ang = 0.0;
    }
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
) -> SeReport {
    let mut cache: Option<S> = None;
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

        // A state variable no measurement touches leaves an all-zero row and
        // column in G, which makes the whole system singular even though the
        // rest of it is perfectly well determined. Writing an identity into
        // that position with a zero right-hand side pins the variable at its
        // current value and lets the observable part solve — the same masking
        // `bde::mask_scenario` uses for a converged block, and for the same
        // reason: dropping the row instead would change the sparsity pattern
        // and invalidate the cached symbolic factorization.
        // Structural, not numerical: a column counts as constrained if any
        // measurement's row reaches it at all. That keeps the mask — and so the
        // gain matrix's pattern — identical across iterations, which the cached
        // symbolic factorization requires. A column that is present but
        // numerically degenerate is a different problem, and phase 5's job.
        let mut touched = vec![false; n];
        for row in rows.iter().chain(&c_rows) {
            for &(c, _) in row {
                touched[c] = true;
            }
        }
        for (c, &seen) in touched.iter().enumerate() {
            if !seen {
                triplets.push((c, c, 1.0));
                rhs[c] = 0.0;
            }
        }
        let unconstrained: Vec<usize> =
            touched.iter().enumerate().filter(|&(_, &t)| !t).map(|(c, _)| c).collect();

        let (triplets, rhs) =
            super::constraints::augment(triplets, rhs, n, &c_values, &c_rows);
        let n_aug = n + constraints.len();

        // The gain matrix's sparsity pattern is fixed for a topology and a
        // measurement set, so the symbolic factorization is built once and
        // reused across iterations — the same property `PersistentSolver`
        // relies on for power flow.
        if cache.is_none() {
            cache = S::new(n_aug, &triplets);
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
    let layout = StateLayout::new(buses, measurements, net);
    let constraints = Constraints::new(&net.zero_injection);
    match options.backend {
        JacobianBackend::KluNative => {
            estimate_with::<crate::klu_native::KluNativeSystem>(measurements, buses, net, &layout, &constraints, options)
        }
        #[cfg(feature = "klu")]
        JacobianBackend::Klu => {
            estimate_with::<crate::sparse_klu::KluRealSystem>(measurements, buses, net, &layout, &constraints, options)
        }
        #[cfg(feature = "pardiso")]
        JacobianBackend::Pardiso => {
            estimate_with::<crate::sparse_pardiso::PardisoRealSystem>(measurements, buses, net, &layout, &constraints, options)
        }
        // `Block` assumes power flow's two-unknowns-per-bus structure, which the
        // 2N−1 state vector here does not have (the reference bus contributes
        // one unknown, not two), so it falls back to the scalar path rather
        // than silently mis-assembling. Every other backend is a general sparse
        // LU and carries the gain matrix unchanged.
        JacobianBackend::Block | JacobianBackend::Scalar => {
            estimate_with::<RealSparseSystem>(measurements, buses, net, &layout, &constraints, options)
        }
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
