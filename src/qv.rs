//! Q-V curves: how much reactive headroom does one bus have?
//!
//! [`continuation`](crate::continuation) asks how much further the *system* can
//! be loaded before it collapses. This asks a narrower and more operational
//! question about a *single* bus: how much reactive support would it take to
//! hold this bus at a given voltage, and how much margin is there before that
//! stops being possible?
//!
//! # The method, and why it needs no continuation
//!
//! Put a fictitious synchronous condenser at the bus — an ideal machine that
//! holds a voltage and produces whatever reactive power that takes — by
//! retyping the bus to `PV`. Then sweep its setpoint downward and read the
//! reactive power it has to supply at each step:
//!
//! ```text
//! Q                     the curve is traced left to right by lowering |V|
//! |    \                 (or right to left; the sweep direction is a choice)
//! |     \___             Q > 0: the bus must be *supported* to hold this high
//! |         \___
//! +-----------\------------------ |V|
//! |            \___
//! |                \__.--          the minimum: dQ/d|V| = 0.
//! |                    ^           Below it no setpoint is reachable with less
//!                    the nose      reactive power — that is the margin.
//! ```
//!
//! Every point is an **ordinary power flow**. Fixing `|V|` is what makes it so:
//! the bus's reactive equation is dropped and its magnitude stops being an
//! unknown, which is exactly a `PV` bus, and the Jacobian stays non-singular
//! all the way through the nose. A P-V curve needs a predictor–corrector
//! precisely because it does *not* fix a magnitude — λ is a free parameter and
//! the Jacobian goes singular at the fold. Here the fold is in `Q`, which is an
//! output rather than an unknown, so nothing goes singular and a sweep is
//! enough.
//!
//! That is a real difference in kind, not an implementation shortcut, and it is
//! why this module reuses [`crate::run_power_flow`] rather than
//! [`continuation`](crate::continuation)'s bordered corrector.
//!
//! # What the margin means
//!
//! The minimum of the curve is the most reactive power the bus can absorb
//! before the relationship inverts. Reported as a positive number of MVAr: a
//! larger margin is a stronger bus. A bus whose curve barely dips below zero is
//! one small contingency away from having no reachable setpoint at all.
//!
//! # What this shares with the P-V side
//!
//! The answer, when both are asked. The bus a P-V trace names critical — the
//! largest component of the tangent at the nose — should be the bus with the
//! smallest Q-V margin, because both are measuring the same weakness by
//! different means. `tests/qv_test.rs` asserts exactly that, and it is the
//! strongest gate here: two independent methods agreeing is worth more than
//! either agreeing with itself.

use crate::network::{build_ybus, effective_injection, power_injections, stamp_shunts, ShuntAdm};
use crate::solver::{PowerFlowOptions, SolveStatus};
use crate::types::{Bus, BusType, Line, Transformer};

/// Everything a Q-V sweep needs to know.
#[derive(Clone, Debug)]
pub struct QvOptions {
    /// The solve at every point. `method` must be `NewtonRaphson`.
    pub pf: PowerFlowOptions,
    /// Highest setpoint to try, per-unit.
    pub v_max: f64,
    /// Lowest. The nose is typically well above 0.4 on a healthy bus; a bus
    /// whose curve is still falling at `v_min` reports `NoseNotReached` rather
    /// than pretending the last sample is a minimum.
    pub v_min: f64,
    /// Setpoint decrement, per-unit.
    pub step: f64,
    /// Refine the minimum by fitting a parabola through the three samples
    /// around it. Exact for a locally-quadratic curve, which this is near a
    /// smooth minimum, and free — no extra solves.
    pub refine: bool,
}

impl Default for QvOptions {
    fn default() -> Self {
        Self {
            pf: PowerFlowOptions { max_iter: 60, ..Default::default() },
            v_max: 1.10,
            v_min: 0.40,
            step: 0.01,
            refine: true,
        }
    }
}

/// One setpoint and the reactive power holding it would take.
#[derive(Clone, Debug, PartialEq)]
pub struct QvPoint {
    /// The setpoint, per-unit.
    pub voltage: f64,
    /// What the fictitious condenser must supply, per-unit. Positive means the
    /// bus has to be *supported* to reach this voltage.
    pub q: f64,
    pub iterations: usize,
}

/// The bottom of the curve: the most reactive power this bus can absorb.
#[derive(Clone, Debug, PartialEq)]
pub struct QvNose {
    pub voltage: f64,
    pub q: f64,
    /// `−q` at the nose, per-unit — positive, and larger is stronger.
    pub margin_pu: f64,
    /// True when the voltage and `q` were interpolated rather than sampled.
    pub refined: bool,
}

impl QvNose {
    /// The margin in MVAr on a stated system base.
    pub fn margin_mvar(&self, base_mva: f64) -> f64 {
        self.margin_pu * base_mva
    }
}

/// How a sweep ended.
#[derive(Clone, Debug, PartialEq)]
pub enum QvStatus {
    /// The curve turned: a minimum was bracketed inside the swept range.
    NoseFound,
    /// Still falling at `v_min`. The lowest sample is a *lower bound* on the
    /// margin, not the margin.
    NoseNotReached,
    /// Too few points converged to say anything.
    Failed,
    /// The request was not attempted.
    Rejected(QvError),
}

/// A request a sweep will not attempt, stated rather than approximated.
#[derive(Clone, Debug, PartialEq)]
pub enum QvError {
    /// A slack bus already fixes its own magnitude and already has a free
    /// reactive injection, so there is no condenser to add and no curve to
    /// trace.
    SlackBus(usize),
    BusOutOfRange { bus: usize, n: usize },
    /// `v_min >= v_max`, or a non-positive step.
    EmptyRange,
    MethodUnsupported(crate::solver::PowerFlowMethod),
}

/// A bus's Q-V curve.
#[derive(Clone, Debug)]
pub struct QvCurve {
    pub bus: usize,
    /// Converged samples, in sweep order (highest setpoint first).
    pub points: Vec<QvPoint>,
    pub nose: Option<QvNose>,
    pub status: QvStatus,
    /// The bus's voltage in the base case, for context: a curve is read
    /// relative to where the bus actually sits.
    pub base_voltage: f64,
    /// Setpoints that did not converge. Worth seeing — a gap in the middle of a
    /// curve is a different thing from a run of failures at the bottom.
    pub failed: Vec<f64>,
    /// The bus was already voltage-controlled, so the curve describes moving an
    /// existing machine's setpoint rather than adding a condenser, and `q`
    /// includes what that machine was already producing.
    pub already_controlled: bool,
}

impl QvCurve {
    fn rejected(bus: usize, why: QvError) -> Self {
        Self {
            bus,
            points: Vec::new(),
            nose: None,
            status: QvStatus::Rejected(why),
            base_voltage: f64::NAN,
            failed: Vec::new(),
            already_controlled: false,
        }
    }

    /// The reactive margin in per-unit, when a nose was found.
    pub fn margin_pu(&self) -> Option<f64> {
        self.nose.as_ref().map(|n| n.margin_pu)
    }
}

/// Traces one bus's Q-V curve.
///
/// See the module docs for what the curve means and why it needs no
/// continuation.
pub fn qv_curve(
    buses: &[Bus],
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    bus: usize,
    opts: QvOptions,
) -> QvCurve {
    if bus >= buses.len() {
        return QvCurve::rejected(bus, QvError::BusOutOfRange { bus, n: buses.len() });
    }
    if buses[bus].bus_type == BusType::Slack {
        return QvCurve::rejected(bus, QvError::SlackBus(bus));
    }
    if opts.pf.method != crate::solver::PowerFlowMethod::NewtonRaphson {
        return QvCurve::rejected(bus, QvError::MethodUnsupported(opts.pf.method));
    }
    // Every case spelled out, so a NaN bound is visibly refused rather than
    // incidentally caught by a negated comparison.
    let range_ok = opts.v_min.is_finite() && opts.v_max.is_finite() && opts.v_min < opts.v_max;
    if !opts.step.is_finite() || opts.step <= 0.0 || !range_ok {
        return QvCurve::rejected(bus, QvError::EmptyRange);
    }

    let mut y = build_ybus(buses.len(), lines, transformers);
    stamp_shunts(&mut y, shunts);
    let ybus = y.finish();

    // Where the bus actually sits, for context.
    let base = crate::run_power_flow(
        buses.to_vec(),
        lines,
        transformers,
        shunts,
        crate::TapData::none(),
        opts.pf.clone(),
    );
    let base_voltage = if base.stats.status == SolveStatus::Converged {
        base.buses[bus].voltage_mag
    } else {
        f64::NAN
    };

    let already_controlled = buses[bus].bus_type == BusType::PV;
    let mut points: Vec<QvPoint> = Vec::new();
    let mut failed: Vec<f64> = Vec::new();

    let steps = ((opts.v_max - opts.v_min) / opts.step).floor() as usize;
    for k in 0..=steps {
        let v = opts.v_max - opts.step * k as f64;

        let mut trial = buses.to_vec();
        // The fictitious condenser. Retyping to `PV` is the whole of it: the
        // bus's reactive equation is dropped, so `Q` becomes free, and its
        // magnitude stops being an unknown.
        trial[bus].bus_type = BusType::PV;
        trial[bus].voltage_mag = v;
        // An unlimited machine, deliberately: the curve measures how much
        // reactive power holding this voltage *would* take, which a limit would
        // truncate rather than answer. Other buses keep their own limits.
        trial[bus].q_min = f64::NEG_INFINITY;
        trial[bus].q_max = f64::INFINITY;

        let report = crate::run_power_flow(
            trial,
            lines,
            transformers,
            shunts,
            crate::TapData::none(),
            opts.pf.clone(),
        );
        if report.stats.status != SolveStatus::Converged {
            failed.push(v);
            continue;
        }

        let (_, q_calc) = power_injections(&report.buses, &ybus);
        // What the condenser supplies is the bus's net injection less whatever
        // the bus injects on its own account. Evaluated through
        // `effective_injection` at the *solved* voltage so a voltage-dependent
        // load is subtracted at the voltage it actually sees, rather than at
        // its nominal.
        let own = Bus { voltage_mag: report.buses[bus].voltage_mag, ..buses[bus].clone() };
        let (_, q_own) = effective_injection(&own);
        points.push(QvPoint {
            voltage: report.buses[bus].voltage_mag,
            q: q_calc[bus] - q_own,
            iterations: report.stats.iterations(),
        });
    }

    let (nose, status) = locate_nose(&points, opts.refine);
    QvCurve { bus, points, nose, status, base_voltage, failed, already_controlled }
}

/// The minimum of a sampled curve, and whether it is genuinely inside the
/// swept range.
///
/// A minimum at either end is not a minimum — it means the sweep stopped before
/// the curve turned, and the sample is a bound rather than an answer. Saying so
/// matters more here than usual: the margin is the headline number, and
/// reporting the last sample of a still-falling curve would understate a bus's
/// weakness in exactly the direction that misleads.
fn locate_nose(points: &[QvPoint], refine: bool) -> (Option<QvNose>, QvStatus) {
    if points.len() < 3 {
        return (None, QvStatus::Failed);
    }
    let mut lowest = 0usize;
    for (i, p) in points.iter().enumerate() {
        if p.q < points[lowest].q {
            lowest = i;
        }
    }
    if lowest == 0 || lowest == points.len() - 1 {
        return (
            Some(QvNose {
                voltage: points[lowest].voltage,
                q: points[lowest].q,
                margin_pu: -points[lowest].q,
                refined: false,
            }),
            QvStatus::NoseNotReached,
        );
    }

    let (a, b, c) = (&points[lowest - 1], &points[lowest], &points[lowest + 1]);
    if !refine {
        return (
            Some(QvNose { voltage: b.voltage, q: b.q, margin_pu: -b.q, refined: false }),
            QvStatus::NoseFound,
        );
    }

    // Parabola through the three samples, minimized. Exact where the curve is
    // locally quadratic, which it is near a smooth minimum.
    let (x1, y1) = (a.voltage, a.q);
    let (x2, y2) = (b.voltage, b.q);
    let (x3, y3) = (c.voltage, c.q);
    let denom = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);
    let nose = if denom.abs() < 1e-15 {
        QvNose { voltage: x2, q: y2, margin_pu: -y2, refined: false }
    } else {
        let vertex = x2 - 0.5 * ((x2 - x1).powi(2) * (y2 - y3) - (x2 - x3).powi(2) * (y2 - y1))
            / ((x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1));
        // Refuse a vertex outside the bracketing samples — that means the three
        // points do not describe a minimum and the parabola is extrapolating.
        if vertex > x1.max(x3) || vertex < x1.min(x3) || !vertex.is_finite() {
            QvNose { voltage: x2, q: y2, margin_pu: -y2, refined: false }
        } else {
            // Lagrange interpolation at the vertex.
            let q = y1 * (vertex - x2) * (vertex - x3) / ((x1 - x2) * (x1 - x3))
                + y2 * (vertex - x1) * (vertex - x3) / ((x2 - x1) * (x2 - x3))
                + y3 * (vertex - x1) * (vertex - x2) / ((x3 - x1) * (x3 - x2));
            QvNose { voltage: vertex, q, margin_pu: -q, refined: true }
        }
    };
    (Some(nose), QvStatus::NoseFound)
}
