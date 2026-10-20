//! The measurement model state estimation is fitted against: what was
//! observed, where, and how much it is trusted.
//!
//! This is deliberately the *shared front end* of both estimator methods. Both
//! Newton-Raphson WLS and the iterative-linear method consume the same
//! aggregated [`Measurement`] list; they differ only in what they do with it,
//! so sensor aggregation and per-unit conversion happen exactly once, before
//! either sees the list. For PGM documents that is
//! gridoxide's `pgm::measurement`, which also states the
//! aggregation rules and sign conventions it applies.
//!
//! # Scalar rows, not sensors
//!
//! A [`Measurement`] is one *scalar* observation with its own standard
//! deviation — one row of the estimator's `z` vector and one row of `H`. A PGM
//! power sensor therefore becomes two of them (P and Q), because
//! Newton-Raphson WLS treats `σ_P` and `σ_Q` as independent
//! (`se-algorithms.md`), and a voltage sensor becomes one or two depending on
//! whether it carries an angle.

use crate::branch_flow::Terminal;

/// Which scalar quantity a [`Measurement`] observes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeasurementKind {
    /// Voltage magnitude at a bus, per-unit.
    VoltageMagnitude,
    /// Voltage angle at a bus, radians. Only ever present when a sensor
    /// supplied one — i.e. a phasor/PMU measurement.
    VoltageAngle,
    /// Active power, per-unit.
    ActivePower,
    /// Reactive power, per-unit.
    ReactivePower,
    /// Real part of a measured current, per-unit.
    ///
    /// A current sensor is stored decomposed rather than as a magnitude and an
    /// angle, following power-grid-model. Three reasons, the first decisive:
    /// `arg(I)` has a branch cut, and a residual `z − h(x)` taken near ±π wraps
    /// — gridoxide has no `phase_mod_2pi` anywhere, so a polar row would
    /// silently chase a 2π error. `|I|` also has an unbounded derivative as
    /// `I → 0`, which an unloaded branch reaches. And the iterative-linear
    /// method needs a complex current regardless, so polar rows would be
    /// recombined later and less legibly.
    CurrentReal,
    /// Imaginary part of a measured current, per-unit.
    CurrentImag,
}

/// Which reference a current sensor's angle is measured against.
///
/// The two are not a presentation detail: they measure *different quantities*
/// of the same terminal, and power-grid-model's own `global-current-sensor` and
/// `local-current-sensor` fixtures are identical but for this field and converge
/// to visibly different states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AngleFrame {
    /// The angle is absolute, against the same reference voltage phasors use:
    /// `I = i·e^{j·i_angle}`.
    ///
    /// Only meaningful when something else fixes that reference, so
    /// power-grid-model requires at least one voltage angle measurement
    /// alongside — `se::observability` checks the same.
    Global,
    /// The angle is the shift between the terminal's voltage and its current,
    /// `i_angle = θ_U − θ_I`, which is what a meter reading power factor
    /// produces.
    ///
    /// Carries no absolute phase, so it neither needs nor supplies a reference.
    /// Note the sign convention is the opposite of [`Global`](Self::Global)'s:
    /// `I = conj(i·e^{j·i_angle})·U/|U|`.
    Local,
}

/// What a [`Measurement`] is attached to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// A bus: a voltage measurement, or the injection of the loads and
    /// generators attached to it.
    ///
    /// This is *not* power-grid-model's "node injection" — see
    /// [`Target::NodeInjection`]. gridoxide models sources and shunts as part
    /// of the network rather than as appliances, so they do not appear in this
    /// quantity at all.
    Bus(usize),
    /// One terminal of one branch, indexed as in
    /// [`branch_flow::branch_params`](crate::branch_flow::branch_params).
    BranchTerminal { branch: usize, terminal: Terminal },
    /// The power a `source` delivers into its node.
    ///
    /// power-grid-model treats a source as an appliance at the node, but
    /// gridoxide synthesizes a virtual slack bus behind a source-impedance
    /// branch, so the source's power is that branch's flow rather than any bus
    /// injection. `branch` is the synthesized branch, from
    /// gridoxide's `PgmNetwork::source_branch_idx`.
    SourceInjection { branch: usize },
    /// The injection of the shunts attached to a bus.
    ///
    /// Shunts are stamped into the Y-bus diagonal, so like sources they are
    /// structural here and absent from [`Target::Bus`]. Shunts on one bus share
    /// a single Y-bus entry and cannot be told apart afterwards, so this is
    /// their total.
    ShuntInjection { bus: usize },
    /// power-grid-model's node injection: the sum of *every* appliance at the
    /// bus — loads, generators, sources and shunts alike.
    ///
    /// Equals [`Target::Bus`] plus the source and shunt injections at the same
    /// bus, which is what makes it a distinct measurement function rather than
    /// a relabelling.
    NodeInjection(usize),
    /// A current sensor on a branch terminal.
    ///
    /// Separate from [`Target::BranchTerminal`] rather than a flag on it,
    /// because the frame changes the measurement *function* and because
    /// `se::iterative` keys its row builder on `Target` — which makes
    /// power-grid-model's "no mixing power and current on one terminal" rule
    /// fall out of the type rather than needing to be enforced downstream.
    BranchTerminalCurrent {
        branch: usize,
        terminal: Terminal,
        frame: AngleFrame,
    },
}

/// One scalar observation with its uncertainty, in per-unit.
///
/// `sigma` is a standard deviation, never a variance — the estimator weights
/// rows by `1/σ²`, and storing the deviation keeps this in the same units the
/// input data uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measurement {
    pub kind: MeasurementKind,
    pub target: Target,
    pub value: f64,
    pub sigma: f64,
}

impl Measurement {
    /// The estimator's weight for this row, `1/σ²`.
    pub fn weight(&self) -> f64 {
        1.0 / (self.sigma * self.sigma)
    }
}
