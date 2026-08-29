//! Discrete events: faults, clearings, branch switching, load steps.
//!
//! # Why an event is not just another time step
//!
//! At an event the **algebraic** variables jump and the **differential** ones
//! do not. A rotor angle cannot change discontinuously — it is the integral of
//! a speed — but a bus voltage can and does, because the network equations
//! carry no derivative and simply state what `V` must be *right now* given `x`
//! and the present topology.
//!
//! So the integration rule must not be applied across the discontinuity. Doing
//! so would average a pre-fault derivative with a post-fault one over a step
//! in which neither held, which is not an approximation of anything. The
//! handling instead is:
//!
//! 1. step exactly onto the event time;
//! 2. apply the change;
//! 3. hold `x` fixed and re-solve `Y V − I_inj(x, V) = 0` for `V` alone;
//! 4. resume, with a couple of backward-Euler steps to damp the ringing the
//!    trapezoidal rule would otherwise show — see [`integrator`](super::integrator).
//!
//! Step 1 needs no root-finding for a **scheduled** event: it is already a
//! time. A [`Relay`] is different — it is triggered by a *state*, so when it
//! trips has to be found rather than read. See [`Relay`] for why that matters
//! and how it is done.
//!
//! # Value-only versus structural
//!
//! Every event here is **value-only**: it changes numbers in the Y-bus or in a
//! device's inputs, never the set of unknowns. That is deliberate and it is
//! what [`DaePattern`](super::dae::DaePattern) being analyzed from a
//! *topology superset* buys — the pattern is built with every branch in
//! service, and `network::build_ybus_with_outages` re-stamps an out-of-service
//! branch's positions at zero rather than dropping them. A trip is then a
//! refill against a symbolic factorization that is still valid, not a
//! re-analysis.
//!
//! Tripping a whole generating unit *is* structural — its states leave the
//! system — and is deferred to a later phase along with the rest of the model
//! library. See `plans/RMS_PLAN.md` §11.

use num_complex::Complex;
use serde::{Deserialize, Serialize};

/// What happens, and when.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Seconds from the start of the run. Steps are truncated to land exactly
    /// here, so the value need not be a multiple of the step size.
    pub time: f64,
    pub kind: EventKind,
}

impl Event {
    pub fn new(time: f64, kind: EventKind) -> Self {
        Self { time, kind }
    }

    /// A solid three-phase fault at a bus, as an admittance to ground.
    ///
    /// `y` is finite by necessity — an infinite admittance is not a number —
    /// so a "bolted" fault is one large enough that the residual terminal
    /// voltage is negligible. `1e6` per unit against a network whose entries
    /// are order 1 leaves `|V|` around `1e-6`, which is four orders below any
    /// tolerance that matters here and keeps the matrix comfortably
    /// factorable. Going much larger buys nothing and starts to cost
    /// conditioning.
    pub fn bolted_fault(time: f64, bus: usize) -> Self {
        Self::new(time, EventKind::BusFault { bus, y: Complex::new(1e6, 0.0) })
    }
}

/// The kinds of disturbance a run can schedule.
///
/// Tagged by `"kind"`, the same spelling
/// [`EventSpec`](super::json::EventSpec) uses, so a relay's action and a
/// scheduled event read identically in a file. One vocabulary for what can
/// happen, whether a time or a threshold decides when.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// Add a shunt admittance to ground at `bus`. Replaces any fault already
    /// standing there rather than adding to it, so a sequence of faults at one
    /// bus reads as a sequence rather than an accumulation.
    BusFault { bus: usize, y: Complex<f64> },
    /// Remove whatever fault stands at `bus`. A no-op if there is none.
    ClearFault { bus: usize },
    /// Take a branch out of service. Branches are indexed lines-then-
    /// transformers, matching `network::build_ybus_with_outages`.
    BranchTrip { branch: usize },
    /// Put it back.
    BranchClose { branch: usize },
    /// Disconnect a device from the network. Indexed in `SystemSpec::devices`
    /// order.
    ///
    /// Value-only, despite phase 2 of `plans/RMS_PLAN.md` predicting it would
    /// be structural. Removing the unit's states from the DAE would indeed
    /// change the variable layout; **freezing** them does not, and freezing is
    /// also the more defensible model — nothing in the network can observe a
    /// disconnected machine's rotor, so integrating it would be tracking a
    /// quantity no result depends on.
    UnitTrip { unit: usize },
    /// Reconnect one. The unit resumes from the state it was frozen at, which
    /// is only meaningful for a brief disconnection: a real resynchronization
    /// is not modelled.
    UnitClose { unit: usize },
    /// Change a bus's load by `ds` (an *injection*, so a load increase is
    /// negative).
    ///
    /// Applied as a change of **admittance**, not of power, because that is
    /// what the loads in this system are: `Δy = −conj(Δs)/|V₀|²`, evaluated at
    /// the voltage the case was initialized at. The step therefore delivers
    /// exactly `ds` if the voltage happens to be back at `V₀` and less, in
    /// proportion to `|V|²`, if it is not. That is the same modelling choice
    /// [`init`](super::init) already made for the base load, and stating it
    /// here keeps the two from drifting apart.
    LoadStep { bus: usize, ds: Complex<f64> },
}

impl EventKind {
    /// Whether applying this changes the Y-bus and so needs a reassembly.
    /// Everything currently does; the distinction is kept because a reference
    /// step (phase 3, once there are controls to step) will not.
    pub fn touches_network(&self) -> bool {
        true
    }

    /// The bus this event concerns, where it concerns one.
    pub fn bus(&self) -> Option<usize> {
        match *self {
            EventKind::BusFault { bus, .. }
            | EventKind::ClearFault { bus }
            | EventKind::LoadStep { bus, .. } => Some(bus),
            EventKind::BranchTrip { .. }
            | EventKind::BranchClose { .. }
            | EventKind::UnitTrip { .. }
            | EventKind::UnitClose { .. } => None,
        }
    }
}

/// Why an event could not be applied. Reported rather than panicking: an
/// out-of-range index in a schedule is a data error, and a run that names the
/// bad event is more useful than one that dies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventError {
    BusOutOfRange { bus: usize, n_bus: usize },
    BranchOutOfRange { branch: usize, n_branch: usize },
    UnitOutOfRange { unit: usize, n_unit: usize },
}

impl std::fmt::Display for EventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventError::BusOutOfRange { bus, n_bus } => {
                write!(f, "event names bus {bus}, but the network has {n_bus} buses")
            }
            EventError::BranchOutOfRange { branch, n_branch } => {
                write!(f, "event names branch {branch}, but the network has {n_branch} branches")
            }
            EventError::UnitOutOfRange { unit, n_unit } => {
                write!(f, "event names unit {unit}, but the system has {n_unit} devices")
            }
        }
    }
}

impl std::error::Error for EventError {}

/// Something noticed during a run that is not an error.
#[derive(Clone, Debug, PartialEq)]
pub enum DynamicsWarning {
    /// A switching event left a group of buses with no machine and no
    /// fixed-voltage bus in it. Such an island has no voltage reference, so
    /// its voltages collapse to whatever its own admittances imply — usually
    /// zero. That is arguably the right answer for a de-energized island, but
    /// it is not a *dynamic* answer, and reading it as one would be a mistake.
    DeadIsland { buses: Vec<usize>, time: f64 },
    /// An event named something that does not exist; it was skipped.
    Skipped { time: f64, error: EventError },
    /// A relay watching something that does not exist; it was dropped.
    RelayDropped { id: String, detail: String },
}

impl std::fmt::Display for DynamicsWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynamicsWarning::DeadIsland { buses, time } => write!(
                f,
                "at t = {time}: {} bus(es) have no machine and no voltage reference, \
                 starting at bus {}",
                buses.len(),
                buses.first().copied().unwrap_or(0)
            ),
            DynamicsWarning::Skipped { time, error } => {
                write!(f, "at t = {time}: {error}")
            }
            DynamicsWarning::RelayDropped { id, detail } => {
                write!(f, "relay {id}: {detail}; dropped")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// State-triggered events: protection
// ---------------------------------------------------------------------------

/// What a relay watches.
///
/// Deliberately a small set. A relay's *decision logic* is the interesting
/// part, and it is the same whatever the input; adding a quantity is adding one
/// match arm, not a new mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "watch", rename_all = "snake_case")]
pub enum Watch {
    /// A bus's voltage magnitude, per unit.
    BusVoltage { bus: usize },
    /// A machine's rotor speed, per unit — `1.0` is synchronous.
    UnitSpeed { unit: usize },
    /// A machine's rotor angle relative to where it started, radians. Rises
    /// without bound on a machine that has lost synchronism, which is what an
    /// out-of-step relay is for.
    UnitAngleExcursion { unit: usize },
}

/// Which side of a threshold trips the relay.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "when", rename_all = "snake_case")]
pub enum Trigger {
    Below { threshold: f64 },
    Above { threshold: f64 },
}

impl Trigger {
    /// The signed distance into the trip region: positive once tripped.
    ///
    /// A signed residual rather than a boolean, so the exact crossing time can
    /// be *located* rather than rounded up to the end of whichever step
    /// happened to notice — see [`Relay::delay`].
    pub fn residual(&self, value: f64) -> f64 {
        match *self {
            Trigger::Below { threshold } => threshold - value,
            Trigger::Above { threshold } => value - threshold,
        }
    }

    pub fn holds(&self, value: f64) -> bool {
        self.residual(value) > 0.0
    }
}

/// A protection relay: watch a quantity, and act if it stays out of bounds.
///
/// # Why the delay is the whole design
///
/// A relay that acted the instant a threshold were crossed would trip on every
/// fault in the network rather than the ones it is meant to clear. The delay is
/// what makes it selective, and it is why the crossing time has to be *located*
/// rather than rounded to a step boundary: the delay is measured from the
/// crossing, so an error there is an error in when the relay acts.
///
/// The condition must hold **continuously**. If the watched quantity comes back
/// inside its bounds before the delay elapses, the relay resets and forgets —
/// which is exactly what a fault cleared in time should cause, and is the
/// difference between a relay and a stopwatch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Relay {
    pub id: String,
    #[serde(flatten)]
    pub watch: Watch,
    #[serde(flatten)]
    pub trigger: Trigger,
    /// How long the condition must hold before the action fires, in seconds.
    /// Zero acts at the crossing.
    pub delay: f64,
    pub action: EventKind,
    /// Whether the relay may fire more than once. One-shot by default: a trip
    /// is a trip.
    #[serde(default)]
    pub repeating: bool,
}

impl Relay {
    /// An under-voltage trip: act if a bus stays below `threshold` for `delay`.
    pub fn under_voltage(
        id: impl Into<String>,
        bus: usize,
        threshold: f64,
        delay: f64,
        action: EventKind,
    ) -> Self {
        Self {
            id: id.into(),
            watch: Watch::BusVoltage { bus },
            trigger: Trigger::Below { threshold },
            delay,
            action,
            repeating: false,
        }
    }

    /// An over-speed trip.
    pub fn over_speed(
        id: impl Into<String>,
        unit: usize,
        threshold: f64,
        delay: f64,
        action: EventKind,
    ) -> Self {
        Self {
            id: id.into(),
            watch: Watch::UnitSpeed { unit },
            trigger: Trigger::Above { threshold },
            delay,
            action,
            repeating: false,
        }
    }
}

/// What a relay did, and when.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelayAction {
    pub id: String,
    /// When the watched quantity crossed its threshold — **located**, not
    /// rounded to a step boundary.
    pub crossed_at: f64,
    /// When the action fired: `crossed_at + delay`.
    pub fired_at: f64,
    pub action: EventKind,
}
