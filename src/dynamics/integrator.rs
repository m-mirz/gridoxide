//! The time step: an implicit one-step rule with a Newton solve per step, and
//! the event schedule that interrupts it.
//!
//! # The rule, and why there are two of them
//!
//! The trapezoidal rule is the default. It is second-order and A-stable, which
//! is what a stiff phasor-domain DAE wants, and it is what every production
//! RMS simulator uses between events.
//!
//! It is **not** L-stable. Its amplification factor tends to `−1` rather than
//! `0` as `h·λ → −∞`, so a step change excites the fastest mode of the system
//! into a numerical oscillation that alternates sign and decays only as slowly
//! as the physical mode does. On a fault application — the one thing an RMS
//! run exists to simulate — that shows up as a visible ringing on the voltages
//! that is entirely an artifact of the integrator.
//!
//! The standard remedy is cheap: take a couple of **backward Euler** steps
//! immediately after each discontinuity, then return to trapezoidal. Backward
//! Euler is L-stable, so it annihilates the mode in one step; it is only
//! first-order, but a handful of steps out of a run of thousands cost nothing
//! measurable. [`DynamicsOptions::damping_steps`](super::DynamicsOptions::damping_steps)
//! is that count.
//!
//! **It has nothing to damp yet.** Ringing needs a mode fast enough for
//! `h·λ` to be large, and the only differential mode in the present model
//! library is a classical machine's swing — order 1 Hz, which a 5 ms step
//! resolves comfortably. The damping is wired in now because the exciters and
//! governors of phase 3 have time constants of 20–50 ms, which is exactly
//! where it starts to matter. `tests/dynamics_events_test.rs` checks only that
//! it does not *distort* the answer here, which is the honest claim to make
//! about it today.
//!
//! Both rules are the same residual with different coefficients — see
//! [`dae`](super::dae) — so nothing here branches on which is active beyond
//! choosing `(a, b)`.
//!
//! # The Newton solve
//!
//! One Newton per step over the whole vector `z = [x; v_re, v_im]`,
//! differential and algebraic together. There is no inner iteration between a
//! device solver and a network solver, so there is no interface error to
//! converge away and no per-step lag between the two halves.
//!
//! The Jacobian's *pattern* is fixed for the lifetime of a topology, and every
//! event in [`events`](super::events) preserves it, so `LinearSolver` analyzes
//! **once for the whole run** — a fault, a clearing and a trip are all numeric
//! refactorizations against the same symbolic factorization.
//!
//! # An event, mechanically
//!
//! The step is truncated to land exactly on the event time. Then the change is
//! applied, the Y-bus reassembled, and the algebraic block re-solved **with
//! every differential state held fixed** — which is done by taking an ordinary
//! step with `h·a = 0`, making each differential row read `x₁ − x₀ = 0` and
//! pinning `x` exactly while leaving the algebraic rows untouched. One
//! assembly, one code path, no separate pattern for the sub-block.
//!
//! Both the pre-event and post-event points are recorded, at the same time
//! value, so the discontinuity is visible in the trajectory rather than
//! smoothed away by whoever plots it.

use num_complex::Complex;

use crate::solver::LinearSolver;

use super::dae::residual;
use super::events::{DynamicsWarning, Event, EventError, EventKind};
use super::models::ModelJacobian;
use super::{DynamicSystem, DynamicsOptions, DynamicsReport, DynamicsStatus, Trajectory};

/// Coefficients `(a, b)` of `x₁ − x₀ − h(a·f₁ + b·f₀) = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    Trapezoidal,
    BackwardEuler,
}

impl Rule {
    pub fn coefficients(self) -> (f64, f64) {
        match self {
            Rule::Trapezoidal => (0.5, 0.5),
            Rule::BackwardEuler => (1.0, 0.0),
        }
    }
}

/// What one step did, so the caller can distinguish "the network came apart"
/// from "the step was too big".
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepOutcome {
    Converged { iterations: usize },
    NotConverged,
    Singular,
}

/// Per-run buffers, allocated once so a step allocates nothing.
struct Workspace {
    scratch: Vec<ModelJacobian>,
    values: Vec<f64>,
    f_scratch: Vec<f64>,
    res: Vec<f64>,
    rhs: Vec<f64>,
    newton_iterations: usize,
}

impl Workspace {
    fn new(system: &DynamicSystem) -> Self {
        let layout = system.pattern.layout();
        Self {
            scratch: (0..layout.n_devices())
                .map(|d| ModelJacobian::zeros(layout.dev_len[d]))
                .collect(),
            values: Vec::with_capacity(system.pattern.nnz()),
            f_scratch: vec![0.0; layout.n_diff],
            res: vec![0.0; layout.n()],
            rhs: vec![0.0; layout.n()],
            newton_iterations: 0,
        }
    }
}

/// One implicit solve of `x₁ − x₀ − h(a·f₁ + b·f₀) = 0` together with the
/// network constraint.
///
/// With `ha = hb = 0` and `pin_states` set this is the algebraic re-solve an
/// event needs: the differential rows collapse to `x₁ − x₀ = 0`, whose
/// Jacobian block is the identity and whose residual is already zero, so `Δx`
/// is exactly zero *in exact arithmetic*.
///
/// It is not exactly zero in floating point. The matrix is block lower
/// triangular — `[I, 0; −∂I/∂x, Y − ∂I/∂V]` — so forward substitution would
/// give zero, but no backend does forward substitution in that order: each
/// applies its own fill-reducing permutation and partial pivoting, and
/// roundoff leaks across the block boundary. Measured at ~1e-10 rad on a
/// faulted network, whose conditioning the large fault admittance dominates.
///
/// Small, but wrong in kind and cumulative over a run with many events: a
/// rotor angle is *defined* to be continuous across a discontinuity, so
/// `pin_states` enforces that rather than approximating it. This costs
/// nothing and removes the leak entirely.
#[allow(clippy::too_many_arguments)]
fn newton<S: LinearSolver>(
    system: &DynamicSystem,
    solver: &mut Option<S>,
    ws: &mut Workspace,
    x_prev: &[f64],
    f0: &[f64],
    x: &mut [f64],
    v: &mut [Complex<f64>],
    ha: f64,
    hb: f64,
    pin_states: bool,
    tol: f64,
    max_newton: usize,
) -> StepOutcome {
    let pattern = &system.pattern;
    let layout = pattern.layout();
    let n = layout.n();

    for iteration in 1..=max_newton {
        residual(
            pattern,
            &system.models,
            x_prev,
            f0,
            x,
            v,
            &system.v_fixed,
            ha,
            hb,
            &system.ybus,
            &mut ws.f_scratch,
            &mut ws.res,
        );
        let norm = ws.res.iter().fold(0.0f64, |m, r| m.max(r.abs()));
        if norm < tol {
            return StepOutcome::Converged { iterations: iteration - 1 };
        }

        pattern.fill(&system.models, x, v, ha, &system.ybus, &mut ws.scratch, &mut ws.values);
        for k in 0..n {
            ws.rhs[k] = -ws.res[k];
        }

        // Analyzed lazily, against real values: `KluNative` and `Klu` factor
        // numerically inside `new` and would refuse a placeholder pattern. Once
        // built it serves the whole run — every event preserves the sparsity.
        if solver.is_none() {
            *solver = S::new(n, &pattern.to_triplets(&ws.values));
        }
        let Some(lu) = solver.as_mut() else {
            return StepOutcome::Singular;
        };
        let Some(dz) = lu.factor_and_solve_values(&ws.values, &ws.rhs) else {
            return StepOutcome::Singular;
        };

        if !pin_states {
            for i in 0..layout.n_diff {
                x[i] += dz[i];
            }
        }
        for i in 0..layout.n_bus {
            v[i] += Complex::new(dz[layout.v_re(i)], dz[layout.v_im(i)]);
        }
        ws.newton_iterations += 1;
    }
    StepOutcome::NotConverged
}

/// Applies one event to the system, leaving the Y-bus to be reassembled by the
/// caller once every event at this instant has been applied.
fn apply(system: &mut DynamicSystem, kind: EventKind) -> Result<(), EventError> {
    let n_bus = system.v0.len();
    let n_branch = system.outaged.len();
    if let Some(bus) = kind.bus().filter(|&b| b >= n_bus) {
        return Err(EventError::BusOutOfRange { bus, n_bus });
    }
    match kind {
        EventKind::BusFault { bus, y } => system.fault_y[bus] = y,
        EventKind::ClearFault { bus } => system.fault_y[bus] = Complex::new(0.0, 0.0),
        EventKind::BranchTrip { branch } | EventKind::BranchClose { branch } => {
            if branch >= n_branch {
                return Err(EventError::BranchOutOfRange { branch, n_branch });
            }
            system.outaged[branch] = matches!(kind, EventKind::BranchTrip { .. });
        }
        EventKind::UnitTrip { unit } | EventKind::UnitClose { unit } => {
            let n_unit = system.models.len();
            if unit >= n_unit {
                return Err(EventError::UnitOutOfRange { unit, n_unit });
            }
            system.models[unit]
                .set_connected(matches!(kind, EventKind::UnitClose { .. }));
        }
        EventKind::LoadStep { bus, ds } => {
            let v_sq = system.v_fixed[bus].norm_sqr();
            if v_sq != 0.0 {
                system.load_y[bus] += -ds.conj() / v_sq;
            }
        }
    }
    Ok(())
}

/// The integration loop, generic over the sparse-LU backend for the same
/// reason `solver::newton_raphson_cached` and `continuation::trace` are: four
/// near-identical copies otherwise.
pub fn integrate<S: LinearSolver>(
    system: &mut DynamicSystem,
    opts: &DynamicsOptions,
) -> DynamicsReport {
    let layout = system.pattern.layout().clone();

    let mut solver: Option<S> = None;
    let mut ws = Workspace::new(system);
    let mut f0 = vec![0.0; layout.n_diff];

    let mut x = system.x0.clone();
    let mut v = system.v0.clone();

    let mut traj = Trajectory::new(system.observable_names());
    let mut warnings: Vec<DynamicsWarning> = Vec::new();
    let mut events: Vec<Event> = opts.events.clone();
    events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    let mut next_event = 0usize;
    let mut events_applied = 0usize;

    // Tolerance for "this step landed on that event". Tied to the step size
    // rather than absolute, and only ever compared against times the loop
    // snapped exactly, so it never has to absorb accumulated drift.
    let eps = opts.step * 1e-9;

    let mut t = 0.0;
    let mut steps = 0usize;
    let mut status = DynamicsStatus::Completed;
    // Starting the run counts as a discontinuity. The initial point is an
    // equilibrium so this costs nothing, and it means a run that *begins* with
    // an event is damped exactly as one that meets an event later is.
    let mut damping_left = opts.damping_steps;

    traj.push(t, &x, &v);

    // Anything scheduled at or before zero happens before the first step.
    let fire = |system: &mut DynamicSystem,
                    events: &[Event],
                    next_event: &mut usize,
                    t: f64,
                    x: &mut Vec<f64>,
                    v: &mut Vec<Complex<f64>>,
                    solver: &mut Option<S>,
                    ws: &mut Workspace,
                    traj: &mut Trajectory,
                    warnings: &mut Vec<DynamicsWarning>,
                    events_applied: &mut usize|
     -> Option<DynamicsStatus> {
        let mut any = false;
        while *next_event < events.len() && events[*next_event].time <= t + eps {
            let event = events[*next_event];
            *next_event += 1;
            match apply(system, event.kind) {
                Ok(()) => {
                    any = true;
                    *events_applied += 1;
                }
                Err(error) => warnings.push(DynamicsWarning::Skipped { time: event.time, error }),
            }
        }
        if !any {
            return None;
        }
        system.reassemble();

        // The algebraic variables jump; the differential ones do not.
        let x_prev = x.clone();
        let zero = vec![0.0; x.len()];
        match newton(
            system, solver, ws, &x_prev, &zero, x, v, 0.0, 0.0, true, opts.tol, opts.max_newton,
        ) {
            StepOutcome::Converged { .. } => {}
            StepOutcome::NotConverged => return Some(DynamicsStatus::NewtonFailed { time: t }),
            StepOutcome::Singular => return Some(DynamicsStatus::Singular { time: t }),
        }
        traj.push(t, x, v);

        for island in system.dead_islands() {
            let warning = DynamicsWarning::DeadIsland { buses: island, time: t };
            if !warnings.contains(&warning) {
                warnings.push(warning);
            }
        }
        None
    };

    if let Some(bad) = fire(
        system, &events, &mut next_event, t, &mut x, &mut v, &mut solver, &mut ws, &mut traj,
        &mut warnings, &mut events_applied,
    ) {
        status = bad;
    } else {
        if events_applied > 0 {
            damping_left = opts.damping_steps;
        }

        while t < opts.end_time - eps {
            let mut h = opts.step.min(opts.end_time - t);
            let mut lands_on_event = false;
            if next_event < events.len() {
                let dt = events[next_event].time - t;
                if dt > eps && dt < h {
                    h = dt;
                    lands_on_event = true;
                }
            }

            let rule = if damping_left > 0 { Rule::BackwardEuler } else { Rule::Trapezoidal };
            let (a, b) = rule.coefficients();

            // Derivatives at the start of the step. Trapezoidal needs them;
            // backward Euler multiplies them by zero, and computing them
            // anyway keeps one code path.
            system.derivatives_at(&x, &v, &mut f0);

            let x_prev = x.clone();
            // Explicit-Euler predictor for the differential half; the algebraic
            // half starts where it was, which already solves the constraint at
            // the previous point.
            for i in 0..layout.n_diff {
                x[i] = x_prev[i] + h * f0[i];
            }

            let outcome = newton(
                system, &mut solver, &mut ws, &x_prev, &f0, &mut x, &mut v, h * a, h * b,
                false, opts.tol, opts.max_newton,
            );
            match outcome {
                StepOutcome::Converged { .. } => {}
                StepOutcome::NotConverged => {
                    status = DynamicsStatus::NewtonFailed { time: t + h };
                    break;
                }
                StepOutcome::Singular => {
                    status = DynamicsStatus::Singular { time: t + h };
                    break;
                }
            }

            // Snapped rather than accumulated, so an event time is hit exactly
            // however many steps preceded it.
            t = if lands_on_event { events[next_event].time } else { t + h };
            steps += 1;
            damping_left = damping_left.saturating_sub(1);
            traj.push(t, &x, &v);

            let before = events_applied;
            if let Some(bad) = fire(
                system, &events, &mut next_event, t, &mut x, &mut v, &mut solver, &mut ws,
                &mut traj, &mut warnings, &mut events_applied,
            ) {
                status = bad;
                break;
            }
            if events_applied > before {
                damping_left = opts.damping_steps;
            }
        }
    }

    system.x0 = x;
    system.v0 = v;
    DynamicsReport {
        trajectory: traj,
        status,
        steps,
        newton_iterations: ws.newton_iterations,
        events_applied,
        warnings,
    }
}

/// Solves the algebraic constraint alone, holding every differential state
/// fixed, and stores the result.
///
/// The same operation an event performs, exposed for a caller that has written
/// through [`DynamicSystem::state_mut`](super::DynamicSystem::state_mut) and
/// needs the network brought back into agreement with the new state.
pub fn solve_algebraic<S: LinearSolver>(
    system: &mut DynamicSystem,
    tol: f64,
    max_newton: usize,
) -> StepOutcome {
    let mut solver: Option<S> = None;
    let mut ws = Workspace::new(system);
    let x_prev = system.x0.clone();
    let mut x = system.x0.clone();
    let mut v = system.v0.clone();
    let zero = vec![0.0; x.len()];

    let outcome = newton(
        system, &mut solver, &mut ws, &x_prev, &zero, &mut x, &mut v, 0.0, 0.0, true, tol,
        max_newton,
    );
    if matches!(outcome, StepOutcome::Converged { .. }) {
        system.x0 = x;
        system.v0 = v;
    }
    outcome
}
