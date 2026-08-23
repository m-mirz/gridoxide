//! Continuation power flow: how much further can this network be loaded?
//!
//! An ordinary Newton solve cannot answer that. At the collapse point the
//! Jacobian is singular, so Newton simply fails to converge — and from the
//! outside that is indistinguishable from a bad initial guess, because
//! `sparse::RealSparseSystem` detects singularity only as a non-finite result
//! vector and no backend in the crate offers a condition estimate. Marching λ
//! upward with ordinary solves therefore brackets the limit from below and
//! never finds it.
//!
//! Continuation reformulates the problem so the singularity is a **regular
//! point** of a bordered system, and walks the curve through it. See
//! [`augmented`] for why that works and `docs/src/powerflow/continuation.md`
//! for the derivation.
//!
//! ```no_run
//! # use gridoxide::continuation::{run_continuation, ContinuationOptions, LoadingDirection};
//! # fn example(buses: Vec<gridoxide::types::Bus>, lines: &[gridoxide::types::Line]) {
//! let direction = LoadingDirection::scale_loads(&buses);
//! let curve = run_continuation(
//!     buses, lines, &[], &[],
//!     ContinuationOptions { direction, ..Default::default() },
//! );
//! if let Some(critical) = &curve.critical {
//!     println!("λ_max = {:.4}, weakest bus {}", critical.lambda_max, critical.weakest[0].bus);
//! }
//! # }
//! ```
//!
//! **λ_max is a property of the direction, not of the network.** Two studies
//! that stress different buses get different noses and neither is wrong.

pub mod augmented;
pub mod corrector;
pub mod direction;
pub mod events;

use crate::network::{build_ybus, stamp_shunts, ShuntAdm, YBusSparse};
use crate::outerloop::{OuterLoop, OuterLoopContext, ReactiveLimits, SolveContext};
use crate::solver::{
    IslandReport, JacobianBackend, LinearSolver, PowerFlowMethod, PowerFlowOptions, SolveStatus,
};
use crate::sparse::RealSparseSystem;
use crate::klu_native::KluNativeSystem;
#[cfg(feature = "klu")]
use crate::sparse_klu::KluRealSystem;
#[cfg(feature = "pardiso")]
use crate::sparse_pardiso::PardisoRealSystem;
use crate::types::{Bus, BusType, Line, Transformer};

use augmented::{dominant_index, Layout, Parametrization};
use corrector::{zip_buses, Constraint, CorrectorStatus, Segment};
pub use direction::{BaseSpec, LoadingDirection};
pub use events::{ContinuationEvent, QLimitKind};
use events::{worst_violation, Illinois};

/// Where to stop walking.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct StopCriterion {
    /// Stop once λ reaches this, landing exactly on it with a `Natural`
    /// corrector. `None` walks to the nose.
    pub target_lambda: Option<f64>,
    /// Keep going past the nose and down the low-voltage branch until λ falls
    /// back below its starting value.
    pub trace_lower_branch: bool,
}

/// What a continuation run needs to know.
#[derive(Clone, Debug)]
pub struct ContinuationOptions {
    /// The λ = 0 base-case solve and the shared knobs — `tol`, `max_iter`,
    /// `backend`, `init`, `enforce_q_limits`. `method` must be
    /// `NewtonRaphson`, and `control_taps` is refused: a tap move is a discrete
    /// change to Y-bus *values*, needing its own event function and a
    /// re-stamping mid-curve, and silently freezing the taps at their base-case
    /// positions would be a wrong answer that looks right.
    pub power_flow: PowerFlowOptions,
    pub direction: LoadingDirection,
    pub parametrization: Parametrization,
    pub step: f64,
    pub step_min: f64,
    pub step_max: f64,
    /// Corrector iterations at or below which the step grows.
    pub target_iterations: usize,
    pub max_steps: usize,
    pub stop: StopCriterion,
    /// Locate the exact λ at which each generator saturates, rather than
    /// switching at whatever λ a step happened to land on. Only meaningful with
    /// `power_flow.enforce_q_limits`.
    pub locate_events: bool,
    pub event_tol: f64,
    pub event_max_iter: usize,
    pub locate_nose: bool,
    pub nose_refine_iters: usize,
    /// **Reporting only** — the crate's internals carry no system base, and
    /// `Bus::u_rated` is a voltage base. Used solely to turn the per-unit
    /// margin into MW.
    pub base_mva: f64,
    /// Proceed despite voltage-dependent load terms the Jacobian does not
    /// differentiate. Off by default: see [`ContinuationError::ZipTermsUnsupported`].
    pub allow_zip: bool,
}

impl Default for ContinuationOptions {
    fn default() -> Self {
        Self {
            power_flow: PowerFlowOptions::default(),
            direction: LoadingDirection::zero(0),
            parametrization: Parametrization::default(),
            step: 0.1,
            step_min: 1e-4,
            step_max: 1.0,
            target_iterations: 3,
            max_steps: 200,
            stop: StopCriterion::default(),
            locate_events: true,
            event_tol: 1e-6,
            event_max_iter: 20,
            locate_nose: true,
            nose_refine_iters: 12,
            base_mva: 100.0,
            allow_zip: false,
        }
    }
}

/// A request continuation will not attempt, stated rather than approximated.
#[derive(Clone, Debug, PartialEq)]
pub enum ContinuationError {
    /// `JacobianBackend::Block` groups each bus's two unknowns into a 2×2
    /// block and has no home for a scalar λ. Refused rather than silently
    /// substituted.
    BackendUnsupported(JacobianBackend),
    /// A tap move is a discrete change to Y-bus values; continuing with the
    /// taps frozen at their base-case positions would be a different question
    /// than the one asked.
    TapControlUnsupported,
    /// The Jacobian carries no `∂s_eff/∂|V|` term for ZIP loads. An ordinary
    /// solve survives that — the mismatch is exact, so only the step direction
    /// is off — but here the Jacobian's singularity *is* the answer, so a
    /// missing term puts the nose in the wrong place while still looking
    /// entirely plausible. Set `allow_zip` to proceed with a warning.
    ZipTermsUnsupported { buses: Vec<usize> },
    MethodUnsupported(PowerFlowMethod),
    /// Nothing moves, so there is no curve to trace.
    EmptyDirection,
    DirectionLengthMismatch { got: usize, want: usize },
}

/// A caveat that does not stop the run.
#[derive(Clone, Debug, PartialEq)]
pub enum ContinuationWarning {
    /// `allow_zip` was set; the tangent and the located nose are approximate.
    ApproximateJacobian { buses: Vec<usize> },
}

/// How the walk ended.
#[derive(Clone, Debug, PartialEq)]
pub enum ContinuationStatus {
    /// The tangent's λ-component changed sign: a fold was found.
    NoseReached,
    /// A generator saturated and the re-seeded tangent pointed downward in λ.
    LimitInducedNose,
    TargetReached,
    /// The lower branch was followed back below the starting λ.
    TracedToZero,
    /// The step floor was hit without any of the above — the curve could not be
    /// continued, and λ_max is a lower bound rather than an answer.
    StepFloor,
    MaxSteps,
    /// Even the bordered matrix was singular, which a simple fold cannot cause.
    Singular,
    /// The λ = 0 solve did not converge; there is no curve to start from.
    BaseCaseFailed(SolveStatus),
    Rejected(ContinuationError),
}

/// Which side of the nose a point sits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveBranch {
    Upper,
    Lower,
}

/// One converged point on the curve.
#[derive(Clone, Debug)]
pub struct CurvePoint {
    pub lambda: f64,
    /// Cumulative arclength — the parameter the walk actually advances in, and
    /// the one that stays meaningful through the nose where λ does not.
    pub arclength: f64,
    pub voltage_mag: Vec<f64>,
    pub voltage_ang: Vec<f64>,
    /// Bus types change along the curve as generators saturate, so each point
    /// carries its own.
    pub bus_type: Vec<BusType>,
    /// `dλ/dσ`. Its sign flip is the nose.
    pub tangent_lambda: f64,
    pub corrector_iterations: usize,
    pub branch: CurveBranch,
}

/// A bus's share of the collapse mode.
#[derive(Clone, Copy, Debug)]
pub struct WeakBus {
    pub bus: usize,
    /// `|dV/dσ|` at the nose, normalized so the weakest bus is 1.0.
    pub participation: f64,
    pub voltage_mag: f64,
}

/// Which kind of critical point was found — and they are genuinely different
/// answers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CriticalPointKind {
    /// `dλ/dσ` passed through zero: the Jacobian is singular. A fold.
    SaddleNode,
    /// A generator hit its reactive limit and the tangent pointed downward in λ
    /// immediately afterwards. λ_max is the event itself, and the Jacobian is
    /// **not** singular there. A continuation that only watches for a smooth
    /// fold marches past this and reports a nonsense λ_max.
    LimitInduced { bus: usize },
}

/// The answer.
#[derive(Clone, Debug)]
pub struct CriticalPoint {
    pub kind: CriticalPointKind,
    pub lambda_max: f64,
    pub point: CurvePoint,
    /// The tangent's voltage-magnitude components, per bus. At a fold this
    /// spans the Jacobian's null space — the collapse mode itself.
    pub tangent_vmag: Vec<f64>,
    pub tangent_ang: Vec<f64>,
    /// Sorted, largest participation first.
    pub weakest: Vec<WeakBus>,
    /// `λ_max · Σ(−Δp)` over net consumers, per-unit.
    pub margin_pu: f64,
    /// `margin_pu · base_mva`. Reporting only.
    pub margin_mw: f64,
    pub p_load_base_pu: f64,
    pub p_load_nose_pu: f64,
}

impl CriticalPoint {
    /// The margin on a different system base than the one the run was told.
    pub fn margin_mva(&self, base_mva: f64) -> f64 {
        self.margin_pu * base_mva
    }

    /// The bus whose voltage collapses.
    pub fn critical_bus(&self) -> Option<usize> {
        self.weakest.first().map(|w| w.bus)
    }
}

/// Everything a run produced.
#[derive(Clone, Debug)]
pub struct ContinuationCurve {
    pub points: Vec<CurvePoint>,
    pub events: Vec<ContinuationEvent>,
    pub critical: Option<CriticalPoint>,
    pub status: ContinuationStatus,
    /// The solved λ = 0 state.
    pub base: Vec<Bus>,
    /// From the base-case solve. With more than one island, λ_max is *the first
    /// island to collapse* — one λ and one parametrization row couple them all.
    pub islands: Vec<IslandReport>,
    pub warnings: Vec<ContinuationWarning>,
    pub steps_attempted: usize,
    pub corrector_iterations: usize,
    /// Bordered solves performed, and symbolic factorizations built. One
    /// factorization per segment; a segment ends only at a bus-type change.
    pub solves: usize,
    pub segments: usize,
    /// Symbolic factorizations rebuilt because the continuation index moved —
    /// see `augmented`'s module docs for why the index is part of the pattern.
    pub reanalyses: usize,
}

impl ContinuationCurve {
    fn rejected(err: ContinuationError, base: Vec<Bus>) -> Self {
        Self {
            points: Vec::new(),
            events: Vec::new(),
            critical: None,
            status: ContinuationStatus::Rejected(err),
            base,
            islands: Vec::new(),
            warnings: Vec::new(),
            steps_attempted: 0,
            corrector_iterations: 0,
            solves: 0,
            segments: 0,
            reanalyses: 0,
        }
    }

    /// λ at the nose, when one was found.
    pub fn lambda_max(&self) -> Option<f64> {
        self.critical.as_ref().map(|c| c.lambda_max)
    }

    /// Every point's λ, in walk order.
    pub fn lambdas(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.lambda).collect()
    }

    /// One bus's voltage magnitude along the whole curve — a P-V curve, ready
    /// to plot.
    pub fn voltage_trace(&self, bus: usize) -> Vec<f64> {
        self.points.iter().filter_map(|p| p.voltage_mag.get(bus).copied()).collect()
    }

    /// Just the machines that saturated, in the order they did, as
    /// `(bus, limit, λ)`.
    ///
    /// [`events`](Self::events) also carries the walk's own bookkeeping —
    /// rejected steps, and so on — which is diagnostic rather than an answer.
    /// Almost every caller wants this instead.
    pub fn q_limit_events(&self) -> Vec<(usize, QLimitKind, f64)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                ContinuationEvent::QLimit { bus, limit, lambda, .. } => {
                    Some((*bus, *limit, *lambda))
                }
                _ => None,
            })
            .collect()
    }
}

/// The bordering row for a tangent solve, as `(columns, values, sign)`.
///
/// The tangent must point the same way as the previous one, or the walk turns
/// round at the nose and retraces the branch it came from. Two ways to get that:
///
/// - border with `e_k`, `k` the previous tangent's largest component, and take
///   the right-hand side's sign from that component. The bottom equation reads
///   `t_k = ±1`, which fixes the direction through the one component least
///   likely to be near zero. **One nonzero**, which is what keeps the
///   factorization cheap.
/// - border with the previous tangent itself, so the bottom equation reads
///   `t·t_prev = 1`. Cleaner in principle, and dense — see `augmented`'s module
///   docs for what that costs.
///
/// The first tangent has no predecessor, so it is seeded with `e_n`: "increase
/// λ". Note that row can only ever return `t_λ > 0`, which is right for the
/// first step and would make the nose undetectable if used throughout.
fn tangent_border(
    param: Parametrization,
    previous: Option<&[f64]>,
    n: usize,
) -> (Vec<usize>, Vec<f64>, f64) {
    match (param, previous) {
        (Parametrization::PseudoArcLength, Some(t)) => ((0..=n).collect(), t[..=n].to_vec(), 1.0),
        (_, Some(t)) => {
            let k = dominant_index(t);
            (vec![k], vec![1.0], t[k].signum())
        }
        (_, None) => (vec![n], vec![1.0], 1.0),
    }
}

fn finished_ybus(
    n: usize,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
) -> YBusSparse {
    let mut y = build_ybus(n, lines, transformers);
    stamp_shunts(&mut y, shunts);
    y.finish()
}

/// Traces the P-V curve from the base case to the nose.
///
/// See the module docs for the shape of the answer and
/// `plans/CONTINUATION_PLAN.md` for the design.
pub fn run_continuation(
    buses: Vec<Bus>,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    opts: ContinuationOptions,
) -> ContinuationCurve {
    if opts.power_flow.method != PowerFlowMethod::NewtonRaphson {
        return ContinuationCurve::rejected(
            ContinuationError::MethodUnsupported(opts.power_flow.method),
            buses,
        );
    }
    if opts.power_flow.control_taps {
        return ContinuationCurve::rejected(ContinuationError::TapControlUnsupported, buses);
    }
    if opts.power_flow.backend == JacobianBackend::Block {
        return ContinuationCurve::rejected(
            ContinuationError::BackendUnsupported(JacobianBackend::Block),
            buses,
        );
    }
    if opts.direction.d_p.len() != buses.len() || opts.direction.d_q.len() != buses.len() {
        return ContinuationCurve::rejected(
            ContinuationError::DirectionLengthMismatch {
                got: opts.direction.d_p.len(),
                want: buses.len(),
            },
            buses,
        );
    }
    if opts.direction.is_empty() {
        return ContinuationCurve::rejected(ContinuationError::EmptyDirection, buses);
    }
    let zip = zip_buses(&buses);
    if !zip.is_empty() && !opts.allow_zip {
        return ContinuationCurve::rejected(
            ContinuationError::ZipTermsUnsupported { buses: zip },
            buses,
        );
    }

    match opts.power_flow.backend {
        JacobianBackend::Scalar => {
            trace::<RealSparseSystem>(buses, lines, transformers, shunts, opts, zip)
        }
        JacobianBackend::KluNative => {
            trace::<KluNativeSystem>(buses, lines, transformers, shunts, opts, zip)
        }
        #[cfg(feature = "klu")]
        JacobianBackend::Klu => {
            trace::<KluRealSystem>(buses, lines, transformers, shunts, opts, zip)
        }
        #[cfg(feature = "pardiso")]
        JacobianBackend::Pardiso => {
            trace::<PardisoRealSystem>(buses, lines, transformers, shunts, opts, zip)
        }
        JacobianBackend::Block => unreachable!("refused above"),
    }
}

/// Runs `ReactiveLimits` exactly once against the current state.
///
/// The **only** place in continuation that flips a bus type. Keeping it a call
/// into `outerloop::qlimits` rather than a reimplementation means the PV → PQ
/// rule — including that it is one-directional, and that `q_spec` is pinned at
/// the limit — is written down once in the crate and cannot drift between the
/// ordinary solve and this one.
fn switch_q_limits(buses: &mut [Bus], ybus: &mut YBusSparse) -> Vec<usize> {
    let mut limits = ReactiveLimits::new();
    {
        let mut ctx = SolveContext::new(buses, ybus);
        let mut lc = OuterLoopContext { net: &mut ctx, islands: &[], iteration: 0 };
        let _ = limits.check(&mut lc);
    }
    limits.switches().to_vec()
}

/// Predict along `tangent` by arclength `s` from `anchor`, then correct.
///
/// Used for the ordinary step *and* for every locator trial, so a bisected
/// point is produced by exactly the same code path as an accepted one — a
/// located event is a genuine point on the curve, not an interpolation between
/// two.
///
/// # Two guards, and why a converged corrector is not enough
///
/// The power-flow equations in polar form have solutions a continuation must
/// not accept, and a corrector that only checks its own mismatch will happily
/// walk onto one:
///
/// - **The mirror.** `(−|V|, θ + π)` satisfies the equations exactly and carries
///   the *same* λ, so it is invisible to any λ-based check. A step that
///   overshoots can land there, and the walk then re-traverses the whole curve
///   on the mirror sheet — converging every time, reporting negative voltage
///   magnitudes, and eventually finding a nose at the right λ for the wrong
///   reason. Observed, not hypothesized: it is what a `step_max`-sized step does
///   on a two-bus case before this guard existed.
/// - **Any other branch.** More generally, the defining property of a
///   *continuation* step is that the corrector lands near the predictor. If the
///   correction moves further than the prediction did, the step did not follow
///   the curve — it found some other solution — and the honest response is to
///   halve the step rather than to accept a point that is on a different branch.
///
/// Both are reported as `None`, which puts the caller on its existing
/// step-halving path.
#[allow(clippy::too_many_arguments)]
fn probe<S: LinearSolver>(
    seg: &mut Segment<S>,
    work: &mut [Bus],
    anchor: &[Bus],
    ybus: &YBusSparse,
    base: &BaseSpec,
    dir: &LoadingDirection,
    tangent: &[f64],
    z0: &[f64],
    lambda0: f64,
    s: f64,
    param: Parametrization,
    k: usize,
    tol: f64,
    max_iter: usize,
) -> Option<(f64, usize)> {
    work.clone_from_slice(anchor);
    let n = seg.layout.n_unknowns;
    for k in 0..n {
        seg.layout.add(work, k, s * tangent[k]);
    }
    let lambda_pred = lambda0 + s * tangent[n];
    let z_pred: Vec<f64> = (0..=n).map(|k| seg.layout.get(work, lambda_pred, k)).collect();
    // `k` is pinned by the caller rather than re-derived here. That matters for
    // the locator, which re-probes the same step at many arclengths: letting
    // each trial re-pick its own continuation index would change the matrix
    // pattern under the search, throwing away the factorization every trial.
    let constraint = match param {
        Parametrization::Natural => Constraint::Natural { target: lambda_pred },
        Parametrization::Local => Constraint::Local { k, target: z0[k] + s * tangent[k] },
        Parametrization::PseudoArcLength => {
            Constraint::Arc { tangent: tangent.to_vec(), z0: z0.to_vec(), step: s }
        }
    };
    let out = seg.correct(work, ybus, base, dir, &constraint, lambda_pred, tol, max_iter);
    if out.status != CorrectorStatus::Converged {
        return None;
    }
    if work.iter().any(|b| b.bus_type == BusType::PQ && b.voltage_mag <= 0.0) {
        return None;
    }
    let moved: f64 = (0..=n)
        .map(|k| {
            let d = seg.layout.get(work, out.lambda, k) - z_pred[k];
            d * d
        })
        .sum::<f64>()
        .sqrt();
    if moved > s {
        return None;
    }
    Some((out.lambda, out.iterations))
}

/// Everything the two locators share: an anchor point, the tangent leaving it,
/// and the step that straddled the crossing.
struct Bracket<'a> {
    anchor: &'a [Bus],
    tangent: &'a [f64],
    z0: &'a [f64],
    lambda0: f64,
    step: f64,
    param: Parametrization,
    /// The step's own continuation index, held fixed for every trial.
    k: usize,
}

/// Narrows `[0, step]` until the scalar `event` crosses zero, leaving `work` at
/// the located point and returning `(λ, arclength, corrector iterations)`.
///
/// `event` is measured at each trial by re-predicting and re-correcting, so
/// every trial is a genuine solution of the power flow rather than an
/// interpolation. That is what "exact event location" means here, and it is
/// what distinguishes the reported λ from "whichever λ the step happened to
/// reach".
///
/// **The point returned is the first one at which the event has *just*
/// happened — `f ≥ 0`, as small as the search could make it — not the closest
/// point to `f = 0`.** The distinction is not cosmetic. The caller reacts to a
/// located reactive-limit event by handing the point to `ReactiveLimits`, which
/// switches a bus only when it is genuinely outside its limit; a point a
/// hair *inside* would leave it unswitched, and the walk would locate the same
/// crossing again on the next step, forever. Seeding the search with the
/// step's own endpoint — known to be past the event, since that is what
/// bracketed it — guarantees such a point always exists.
#[allow(clippy::too_many_arguments)]
fn locate<S: LinearSolver, F>(
    seg: &mut Segment<S>,
    work: &mut Vec<Bus>,
    ybus: &YBusSparse,
    base: &BaseSpec,
    dir: &LoadingDirection,
    br: &Bracket<'_>,
    tol: f64,
    max_iter: usize,
    event_tol: f64,
    max_trials: usize,
    f_lo: f64,
    f_hi: f64,
    endpoint_lambda: f64,
    mut event: F,
) -> Option<(f64, f64, usize)>
where
    F: FnMut(&mut Segment<S>, &[Bus], &YBusSparse) -> f64,
{
    let mut bracket = Illinois { lo: 0.0, hi: br.step, f_lo, f_hi };
    // Seeded with the step's own endpoint, which `work` still holds: the search
    // can only improve on it, and never has to report a point short of the
    // event. Keyed on the value of `f`, not on recency — a bracketing search's
    // last trial is not necessarily its best.
    let mut best: (f64, f64, f64, Vec<Bus>) =
        (endpoint_lambda, br.step, f_hi, work.clone());
    let mut iterations = 0usize;

    for _ in 0..max_trials {
        let s = bracket.next();
        // Every trial uses the *same* parametrization and the *same* pinned
        // continuation index as the step that bracketed the event, so a located
        // point is produced by exactly the code path an accepted one is — and,
        // just as importantly, the matrix pattern does not change under the
        // search. An earlier version forced pseudo-arclength here for
        // robustness; because that parametrization's bordering row is dense,
        // it made every trial cost ~90x a sparse one and dominated the whole
        // run.
        let Some((lambda, iters)) = probe(
            seg, work, br.anchor, ybus, base, dir, br.tangent, br.z0, br.lambda0, s, br.param,
            br.k, tol, max_iter,
        ) else {
            // A failed trial says nothing about the sign, so shrink toward the
            // side already known to be feasible rather than guessing.
            bracket.hi = 0.5 * (bracket.lo + bracket.hi);
            continue;
        };
        iterations += iters;
        let f = event(seg, work, ybus);
        if f >= 0.0 && f < best.2 {
            best = (lambda, s, f, work.clone());
        }
        if (0.0..event_tol).contains(&f) || bracket.width() < 1e-12 * br.step.max(1.0) {
            return Some((lambda, s, iterations));
        }
        bracket.update(s, f);
    }

    work.clone_from_slice(&best.3);
    Some((best.0, best.1, iterations))
}

/// Assembles the answer at a located critical point.
#[allow(clippy::too_many_arguments)]
fn build_critical<S: LinearSolver>(
    kind: CriticalPointKind,
    seg: &Segment<S>,
    buses: &[Bus],
    tangent: &[f64],
    lambda: f64,
    arclength: f64,
    load_per_lambda: f64,
    p_load_base_pu: f64,
    base_mva: f64,
) -> CriticalPoint {
    let n_buses = buses.len();
    let mut tangent_ang = vec![0.0; n_buses];
    let mut tangent_vmag = vec![0.0; n_buses];
    for (row, &i) in seg.layout.non_slack_idx.iter().enumerate() {
        tangent_ang[i] = tangent[row];
    }
    for (row, &i) in seg.layout.pq_idx.iter().enumerate() {
        tangent_vmag[i] = tangent[seg.layout.n_angle + row];
    }

    // At a fold the tangent spans the Jacobian's null space, so its magnitude
    // components *are* the collapse mode — the weakest-bus ranking falls out of
    // the predictor rather than needing an eigensolver.
    let peak = tangent_vmag.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let mut weakest: Vec<WeakBus> = (0..n_buses)
        .filter(|&i| tangent_vmag[i] != 0.0)
        .map(|i| WeakBus {
            bus: i,
            participation: if peak > 0.0 { tangent_vmag[i].abs() / peak } else { 0.0 },
            voltage_mag: buses[i].voltage_mag,
        })
        .collect();
    weakest.sort_by(|a, b| b.participation.total_cmp(&a.participation));

    let margin_pu = lambda * load_per_lambda;
    CriticalPoint {
        kind,
        lambda_max: lambda,
        point: snapshot(buses, lambda, arclength, tangent[seg.n()], 0, CurveBranch::Upper),
        tangent_vmag,
        tangent_ang,
        weakest,
        margin_pu,
        margin_mw: margin_pu * base_mva,
        p_load_base_pu,
        p_load_nose_pu: p_load_base_pu + margin_pu,
    }
}

fn snapshot(
    buses: &[Bus],
    lambda: f64,
    arclength: f64,
    tangent_lambda: f64,
    iterations: usize,
    branch: CurveBranch,
) -> CurvePoint {
    CurvePoint {
        lambda,
        arclength,
        voltage_mag: buses.iter().map(|b| b.voltage_mag).collect(),
        voltage_ang: buses.iter().map(|b| b.voltage_ang).collect(),
        bus_type: buses.iter().map(|b| b.bus_type).collect(),
        tangent_lambda,
        corrector_iterations: iterations,
        branch,
    }
}

/// The walk itself, generic over the sparse-LU backend for the same reason
/// `solver::newton_raphson_cached` is: four near-identical copies otherwise.
fn trace<S: LinearSolver>(
    buses: Vec<Bus>,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    opts: ContinuationOptions,
    zip: Vec<usize>,
) -> ContinuationCurve {
    let n_buses = buses.len();
    let mut ybus = finished_ybus(n_buses, lines, transformers, shunts);
    let (tol, max_iter) = (opts.power_flow.tol, opts.power_flow.max_iter);

    let mut warnings = Vec::new();
    if !zip.is_empty() {
        warnings.push(ContinuationWarning::ApproximateJacobian { buses: zip });
    }

    // The base case goes through the ordinary entry point, so classification,
    // the chosen initializer and — where asked for — reactive limits at λ = 0
    // are exactly what an ordinary solve would have done.
    let base_report = crate::run_power_flow(
        buses,
        lines,
        transformers,
        shunts,
        crate::TapData::none(),
        opts.power_flow.clone(),
    );
    if base_report.stats.status != SolveStatus::Converged
        && !base_report.islands.iter().all(|i| {
            !matches!(
                i.status,
                crate::solver::IslandStatus::Singular
                    | crate::solver::IslandStatus::MaxIterationsReached
            )
        })
    {
        return ContinuationCurve {
            points: Vec::new(),
            events: Vec::new(),
            critical: None,
            status: ContinuationStatus::BaseCaseFailed(base_report.stats.status),
            base: base_report.buses,
            islands: base_report.islands,
            warnings,
            steps_attempted: 0,
            corrector_iterations: 0,
            solves: 0,
            segments: 0,
            reanalyses: 0,
        };
    }

    let mut buses = base_report.buses;
    let islands = base_report.islands;
    let base_switches: Vec<usize> =
        base_report.outer.as_ref().map(|o| o.q_limit_switches.clone()).unwrap_or_default();
    let base_state = buses.clone();

    // Snapshot and direction are taken *after* the base solve, so
    // `mark_unreferenced_islands` has already zeroed any sourceless island and a
    // base-case reactive clamp is already in `q_spec`. Slack entries are dropped
    // outright: a slack bus contributes no equation, so a direction there moves
    // nothing and would only inflate the reported margin.
    let mut base = BaseSpec::capture(&buses);
    let mut dir = opts.direction.clone();
    for b in &buses {
        if b.bus_type == BusType::Slack {
            dir.d_p[b.idx] = 0.0;
            dir.d_q[b.idx] = 0.0;
        }
    }
    // A machine the *base* solve clamped needs its reactive direction frozen
    // exactly as one clamped mid-walk does. Missing this is silent and severe:
    // `ReactiveLimits` pinned `q_spec` at the limit, and every subsequent
    // `q_spec ← q_base + λ·Δq` write would then ramp it straight back off the
    // limit — a reactive limit that is enforced at λ = 0 and quietly
    // un-enforced everywhere after it. The curve still converges at every step,
    // so nothing about it looks wrong; only an independent check of where each
    // machine actually saturates catches it.
    for &b in base_switches.iter() {
        let clamped = buses[b].q_spec;
        dir.freeze_reactive(b, &mut base, clamped);
    }

    let p_load_base_pu: f64 = buses.iter().filter(|b| b.p_spec < 0.0).map(|b| -b.p_spec).sum();
    let load_per_lambda = dir.total_active_load_increase();

    let initial_border = tangent_border(opts.parametrization, None, Layout::analyze(&buses).n_unknowns);
    let mut seg = Segment::<S>::analyze(&buses, &ybus, &dir, initial_border.0.clone());
    let mut segments = 1usize;
    let mut reanalyses = 0usize;
    let mut solves = 0usize;
    let mut events: Vec<ContinuationEvent> = Vec::new();
    let mut points: Vec<CurvePoint> = Vec::new();
    let mut corrector_iterations = 0usize;

    macro_rules! finish {
        ($status:expr, $critical:expr) => {
            return ContinuationCurve {
                points,
                events,
                critical: $critical,
                status: $status,
                base: base_state,
                islands,
                warnings,
                steps_attempted: 0,
                corrector_iterations,
                solves: solves + seg.solves,
                segments,
                reanalyses: reanalyses + seg.reanalyses,
            }
        };
    }

    let Some(mut tangent) =
        seg.tangent(&buses, &ybus, &initial_border.0, &initial_border.1, initial_border.2)
    else {
        finish!(ContinuationStatus::Singular, None);
    };

    let mut lambda = 0.0f64;
    let mut arclength = 0.0f64;
    let mut step = opts.step.clamp(opts.step_min, opts.step_max);
    let mut work = buses.clone();
    let mut critical: Option<CriticalPoint> = None;
    let mut status = ContinuationStatus::MaxSteps;
    let mut steps_attempted = 0usize;

    points.push(snapshot(&buses, lambda, arclength, tangent[seg.n()], 0, CurveBranch::Upper));

    while steps_attempted < opts.max_steps {
        steps_attempted += 1;
        let n = seg.n();
        let anchor = buses.clone();
        let lambda0 = lambda;
        let z0: Vec<f64> = (0..=n).map(|k| seg.layout.get(&anchor, lambda0, k)).collect();
        let t_lambda_before = tangent[n];
        // One continuation index for the whole step — the tangent solve, the
        // corrector, and every locator trial — so the sparsity pattern is
        // stable across all of them and one symbolic factorization serves.
        let k = dominant_index(&tangent);
        let bracket = Bracket {
            anchor: &anchor,
            tangent: &tangent,
            z0: &z0,
            lambda0,
            step,
            param: opts.parametrization,
            k,
        };

        let Some((new_lambda, iterations)) = probe(
            &mut seg, &mut work, &anchor, &ybus, &base, &dir, &tangent, &z0, lambda0, step,
            opts.parametrization, k, tol, max_iter,
        ) else {
            events.push(ContinuationEvent::StepRejected { lambda: lambda0, step });
            step *= 0.5;
            if step < opts.step_min {
                status = ContinuationStatus::StepFloor;
                break;
            }
            continue;
        };
        corrector_iterations += iterations;

        // --- reactive-limit events -------------------------------------------
        // Checked before the point is accepted, because the answer is *where*
        // the machine saturated, not that it had by the time a step landed.
        let mut hit_limit = false;
        let mut event_lambda = new_lambda;
        let mut event_arc = step;
        if opts.power_flow.enforce_q_limits
            && worst_violation(&work, &seg.q_injections(&work, &ybus)) > 0.0
        {
            hit_limit = true;
            let f_hi = worst_violation(&work, &seg.q_injections(&work, &ybus));
            if opts.locate_events {
                let f_lo = worst_violation(&anchor, &seg.q_injections(&anchor, &ybus));
                if f_lo < 0.0 {
                    if let Some((l, s_at, iters)) = locate(
                        &mut seg,
                        &mut work,
                        &ybus,
                        &base,
                        &dir,
                        &bracket,
                        tol,
                        max_iter,
                        opts.event_tol,
                        opts.event_max_iter,
                        f_lo,
                        f_hi,
                        new_lambda,
                        |s, b, y| worst_violation(b, &s.q_injections(b, y)),
                    ) {
                        event_lambda = l;
                        event_arc = s_at;
                        corrector_iterations += iters;
                    }
                }
            }
        }

        // The event path and the ordinary accept path are mutually exclusive,
        // and each commits `buses`, `lambda` and `arclength` exactly once.
        // Letting the event path fall through into the accept path would
        // advance the arclength twice and leave `lambda` describing a different
        // point than `buses` does — a desynchronization that survives every
        // convergence check and corrupts everything downstream of it.
        if hit_limit {
            buses.clone_from_slice(&work);
            lambda = event_lambda;
            arclength += event_arc;
            let switched = switch_q_limits(&mut buses, &mut ybus);
            debug_assert!(
                !switched.is_empty(),
                "the locator must return a point at which a machine is genuinely \
                 outside its limits, or the same crossing is found forever"
            );
            for &b in &switched {
                let clamped = buses[b].q_spec;
                let limit =
                    if clamped >= buses[b].q_max { QLimitKind::Max } else { QLimitKind::Min };
                dir.freeze_reactive(b, &mut base, clamped);
                events.push(ContinuationEvent::QLimit { bus: b, limit, lambda, iterations });
            }

            // A bus type moved, so `n_unknowns` and the sparsity pattern moved
            // with it — the condition `PersistentSolver::reset` documents.
            // Rebuild the segment, re-converge at the located λ (the state is
            // continuous across the switch, so this should take at most an
            // iteration or two) and re-seed the tangent, which belonged to the
            // old Jacobian.
            solves += seg.solves;
            reanalyses += seg.reanalyses;
            let carried_layout = Layout::analyze(&buses);
            let carried = carried_layout.embed(&seg.layout, &tangent);
            let border = tangent_border(opts.parametrization, Some(&carried), carried_layout.n_unknowns);
            seg = Segment::<S>::analyze(&buses, &ybus, &dir, border.0.clone());
            segments += 1;
            let out = seg.correct(
                &mut buses,
                &ybus,
                &base,
                &dir,
                &Constraint::Natural { target: lambda },
                lambda,
                tol,
                max_iter,
            );
            corrector_iterations += out.iterations;
            let Some(t_new) = seg.tangent(&buses, &ybus, &border.0, &border.1, border.2) else {
                status = ContinuationStatus::Singular;
                break;
            };
            tangent = t_new;
            work = buses.clone();
            let t_lambda_after = tangent[seg.n()];
            points.push(snapshot(
                &buses,
                lambda,
                arclength,
                t_lambda_after,
                out.iterations,
                CurveBranch::Upper,
            ));

            // A limit-induced bifurcation: the machine saturating is itself the
            // maximum, and the Jacobian is not singular there. A continuation
            // watching only for a smooth fold marches past this and reports a
            // nonsense λ_max.
            if t_lambda_after < 0.0 {
                critical = Some(build_critical(
                    CriticalPointKind::LimitInduced { bus: switched[0] },
                    &seg,
                    &buses,
                    &tangent,
                    lambda,
                    arclength,
                    load_per_lambda,
                    p_load_base_pu,
                    opts.base_mva,
                ));
                status = ContinuationStatus::LimitInducedNose;
                break;
            }
            step = (step * 0.25).clamp(opts.step_min, opts.step_max);
            continue;
        }

        // --- accept -----------------------------------------------------------
        buses.clone_from_slice(&work);
        lambda = new_lambda;
        arclength += step;

        // Bordered so the new tangent is guaranteed to point the same way as
        // the old one, which leaves `t_λ` free to change sign at the fold.
        let tb = tangent_border(opts.parametrization, Some(&tangent), seg.n());
        let Some(t_new) = seg.tangent(&buses, &ybus, &tb.0, &tb.1, tb.2) else {
            status = ContinuationStatus::Singular;
            break;
        };
        let t_lambda_after = t_new[seg.n()];
        let branch = if t_lambda_after < 0.0 { CurveBranch::Lower } else { CurveBranch::Upper };
        points.push(snapshot(&buses, lambda, arclength, t_lambda_after, iterations, branch));

        // --- the nose ----------------------------------------------------------
        if critical.is_none() && t_lambda_before > 0.0 && t_lambda_after <= 0.0 {
            // `dλ/dσ` changed sign across this step, so the fold is inside it.
            // Refine on arclength with the same predict-correct the step used,
            // measuring `t_λ` at each trial. `−t_λ` is the bracketed scalar so
            // it runs negative-to-positive, as the locator expects.
            let nose_border = tangent_border(opts.parametrization, Some(&tangent), seg.n());
            let refined = if opts.locate_nose {
                locate(
                    &mut seg,
                    &mut work,
                    &ybus,
                    &base,
                    &dir,
                    &bracket,
                    tol,
                    max_iter,
                    1e-9,
                    opts.nose_refine_iters,
                    -t_lambda_before,
                    -t_lambda_after,
                    // `−t_λ` at each trial, so the bracketed scalar runs
                    // negative-to-positive as the locator expects. Bordering
                    // with the step's own tangent keeps every trial pointing
                    // the same way, which is what lets `t_λ` be read as a
                    // signed quantity at all.
                    new_lambda,
                    |s: &mut Segment<S>, b: &[Bus], y: &YBusSparse| {
                        -s.tangent(b, y, &nose_border.0, &nose_border.1, nose_border.2)
                            .map_or(0.0, |t| t[s.n()])
                    },
                )
            } else {
                None
            };
            let (nose_state, nose_lambda) = match refined {
                Some((l, _, _)) => (work.clone(), l),
                None => (buses.clone(), lambda),
            };
            let nose_tangent = seg
                .tangent(&nose_state, &ybus, &nose_border.0, &nose_border.1, nose_border.2)
                .unwrap_or_else(|| t_new.clone());
            critical = Some(build_critical(
                CriticalPointKind::SaddleNode,
                &seg,
                &nose_state,
                &nose_tangent,
                nose_lambda,
                arclength,
                load_per_lambda,
                p_load_base_pu,
                opts.base_mva,
            ));
            status = ContinuationStatus::NoseReached;
            tangent = t_new;
            if !opts.stop.trace_lower_branch {
                break;
            }
            continue;
        }

        tangent = t_new;

        // --- termination --------------------------------------------------------
        if let Some(target) = opts.stop.target_lambda {
            if lambda >= target {
                let out = seg.correct(
                    &mut buses,
                    &ybus,
                    &base,
                    &dir,
                    &Constraint::Natural { target },
                    target,
                    tol,
                    max_iter,
                );
                corrector_iterations += out.iterations;
                if out.status == CorrectorStatus::Converged {
                    lambda = out.lambda;
                    if let Some(last) = points.last_mut() {
                        *last = snapshot(
                            &buses,
                            lambda,
                            arclength,
                            tangent[seg.n()],
                            out.iterations,
                            branch,
                        );
                    }
                }
                status = ContinuationStatus::TargetReached;
                break;
            }
        }
        if opts.stop.trace_lower_branch && critical.is_some() && lambda <= 0.0 {
            status = ContinuationStatus::TracedToZero;
            break;
        }

        // --- step adaptation ------------------------------------------------------
        let scale =
            (opts.target_iterations.max(1) as f64 / iterations.max(1) as f64).clamp(0.5, 2.0);
        step = (step * scale).clamp(opts.step_min, opts.step_max);
    }

    ContinuationCurve {
        points,
        events,
        critical,
        status,
        base: base_state,
        islands,
        warnings,
        steps_attempted,
        corrector_iterations,
        solves: solves + seg.solves,
        segments,
        reanalyses: reanalyses + seg.reanalyses,
    }
}
