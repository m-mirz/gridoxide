//! The measurement model state estimation is fitted against: what was
//! observed, where, and how much it is trusted.
//!
//! This is deliberately the *shared front end* of both estimator methods. Both
//! Newton-Raphson WLS and the iterative-linear method consume the same
//! aggregated [`Measurement`] list; they differ only in what they do with it,
//! so sensor aggregation and per-unit conversion happen exactly once, here.
//!
//! # Scalar rows, not sensors
//!
//! A [`Measurement`] is one *scalar* observation with its own standard
//! deviation — one row of the estimator's `z` vector and one row of `H`. A PGM
//! power sensor therefore becomes two of them (P and Q), because
//! Newton-Raphson WLS treats `σ_P` and `σ_Q` as independent
//! (`se-algorithms.md`), and a voltage sensor becomes one or two depending on
//! whether it carries an angle.
//!
//! # Aggregation
//!
//! Redundant sensors are merged before estimation rather than being passed
//! through as extra rows, following power-grid-model:
//!
//! - **Same quantity, several sensors** (two voltage sensors on one bus, two
//!   power sensors on one branch terminal) merge by inverse variance:
//!   `z = Σ zₖ/σₖ² / Σ 1/σₖ²`, with the merged variance `1/Σ 1/σₖ²`. A
//!   measurement you have twice is one you know better.
//! - **Several appliances on one bus** sum into a single net bus injection,
//!   with `σ² = Σ σₖ²`. These are *different* quantities being added, so
//!   variances add — the opposite direction from the merge above, and the one
//!   place where more sensors means more absolute uncertainty.
//!
//! # Sign conventions
//!
//! Taken from power-grid-model's reference-direction rules (`data-model.md`),
//! which are not uniform across sensor types:
//!
//! - **Branch terminals** (`branch_from`/`branch_to`): positive means power
//!   flowing *from the node into the branch* — already the convention
//!   [`branch_flow::terminal_flow`](crate::branch_flow::terminal_flow) uses, so
//!   these pass through unchanged.
//! - **Loads and shunts** use the load reference direction: positive means
//!   power flowing *from the node into the appliance*, i.e. consumption. As a
//!   bus injection that is **negated**.
//! - **Sources and generators** use the generator reference direction: positive
//!   means power flowing into the node, which is already a positive injection.
//! - **Node injection sensors** measure the bus injection directly, in the
//!   generator direction.
//!
//! Getting one of these backwards produces an estimate that converges to a
//! confidently wrong answer rather than failing, which is why they are spelled
//! out here and asserted in the tests.

use std::collections::HashMap;

use crate::branch_flow::Terminal;
use crate::pgm::{PgmInput, PgmNetwork};

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
    /// [`PgmNetwork::source_branch_idx`](crate::pgm::PgmNetwork).
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
    BranchTerminalCurrent { branch: usize, terminal: Terminal, frame: AngleFrame },
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

/// Why a sensor could not be turned into a measurement.
#[derive(Clone, Debug, PartialEq)]
pub enum MeasurementError {
    /// The sensor names an object that isn't in the network.
    UnknownObject { sensor: u64, object: u64 },
    /// `measured_terminal_type` is one this conversion doesn't handle.
    UnsupportedTerminalType { sensor: u64, terminal_type: u8 },
    /// A sigma was absent, zero or negative. Zero would mean infinite weight.
    InvalidSigma { sensor: u64, sigma: f64 },
    /// A current sensor was placed on a `link`.
    ///
    /// power-grid-model refuses this outright, and for a reason gridoxide shares
    /// with more force: a link's admittance is a regularization constant, not a
    /// measured property, so the current through one is an artifact of that
    /// choice rather than a physical quantity. gridoxide's own value differs
    /// from power-grid-model's by 500x (`topology::IDEAL_CONNECTION_Y`), which
    /// is exactly how meaningless the reading would be.
    CurrentSensorOnLink { sensor: u64, link: u64 },
    /// Two current sensors on one terminal disagree about which angle frame
    /// they use, so there is no one quantity for them to merge into.
    ConflictingAngleFrame { sensor: u64, object: u64 },
    /// A power sensor and a current sensor share one terminal.
    ///
    /// power-grid-model rejects this in its Python validation layer while its
    /// C++ core would accept and double-count it. gridoxide rejects it here,
    /// which is deliberately the stricter of the two.
    MixedPowerAndCurrent { sensor: u64, object: u64 },
}

impl std::fmt::Display for MeasurementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownObject { sensor, object } => {
                write!(f, "sensor {sensor} measures object {object}, which is not in the network")
            }
            Self::UnsupportedTerminalType { sensor, terminal_type } => {
                write!(f, "sensor {sensor} has unsupported measured_terminal_type {terminal_type}")
            }
            Self::InvalidSigma { sensor, sigma } => {
                write!(f, "sensor {sensor} has non-positive or missing sigma {sigma}")
            }
            Self::CurrentSensorOnLink { sensor, link } => write!(
                f,
                "current sensor {sensor} measures link {link}, whose admittance is a \
                 regularization constant rather than a physical property"
            ),
            Self::ConflictingAngleFrame { sensor, object } => write!(
                f,
                "current sensor {sensor} uses a different angle frame than another sensor on the \
                 same terminal of object {object}"
            ),
            Self::MixedPowerAndCurrent { sensor, object } => write!(
                f,
                "sensor {sensor} puts a current measurement on a terminal of object {object} that \
                 a power sensor already measures"
            ),
        }
    }
}

impl std::error::Error for MeasurementError {}

/// An accumulator for readings of one quantity that merge by inverse variance.
///
/// Kept as the running sums `Σ z/σ²` and `Σ 1/σ²` so merging is order- and
/// count-independent.
#[derive(Clone, Copy, Debug, Default)]
struct Merged {
    weighted_sum: f64,
    weight: f64,
}

impl Merged {
    fn add(&mut self, value: f64, sigma: f64) {
        let w = 1.0 / (sigma * sigma);
        self.weighted_sum += value * w;
        self.weight += w;
    }

    /// `(value, sigma)`, or `None` if nothing was ever added.
    fn finish(self) -> Option<(f64, f64)> {
        (self.weight > 0.0).then(|| (self.weighted_sum / self.weight, (1.0 / self.weight).sqrt()))
    }
}

/// [`Merged`] for a complex quantity: several phasor readings of one voltage.
///
/// Voltage sensors that carry an angle have to merge as *phasors*, not as a
/// magnitude and an angle separately. The two agree only when the readings
/// share an angle. power-grid-model's `sensor-update-initially-empty` is the
/// case that separates them: 11.0∠−0.1 with σ=1 and 9.0∠+0.1 with σ=2 merge to
/// 10.5702∠−0.0662 as phasors, against 10.6 if the magnitudes are merged on
/// their own — the vector sum is shorter than the scalar one whenever the
/// phasors disagree, which is the whole point of merging them as vectors.
#[derive(Clone, Copy, Debug, Default)]
struct MergedPhasor {
    weighted_sum: num_complex::Complex<f64>,
    weight: f64,
}

impl MergedPhasor {
    fn add(&mut self, value: num_complex::Complex<f64>, sigma: f64) {
        let w = 1.0 / (sigma * sigma);
        self.weighted_sum += value * w;
        self.weight += w;
    }

    /// `(phasor, sigma)`, or `None` if nothing was ever added.
    fn finish(self) -> Option<(num_complex::Complex<f64>, f64)> {
        (self.weight > 0.0)
            .then(|| (self.weighted_sum / self.weight, (1.0 / self.weight).sqrt()))
    }
}

/// An accumulator for contributions that *add* rather than merge: several
/// appliances making up one bus injection. Variances add.
#[derive(Clone, Copy, Debug, Default)]
struct Summed {
    value: f64,
    variance: f64,
    any: bool,
}

impl Summed {
    fn add(&mut self, value: f64, sigma: f64) {
        self.value += value;
        self.variance += sigma * sigma;
        self.any = true;
    }

    fn finish(self) -> Option<(f64, f64)> {
        self.any.then(|| (self.value, self.variance.sqrt()))
    }
}

/// Readings of one appliance, merged across however many sensors watch it.
///
/// `sign` converts the appliance's own reference direction into a bus
/// injection, and is fixed by the appliance kind (load/shunt consume,
/// source/generator inject).
#[derive(Clone, Copy, Debug)]
struct ApplianceAccumulator {
    bus: usize,
    sign: f64,
    /// Shunts get their own measurement function, so they are summed into a
    /// separate per-bus total rather than into the load/generator injection.
    is_shunt: bool,
    p: Merged,
    q: Merged,
}

/// Everything a bus can accumulate before becoming [`Measurement`]s.
#[derive(Clone, Copy, Debug, Default)]
struct BusAccumulator {
    /// Magnitude-only readings. A phasor group's own merged magnitude joins
    /// this at flatten time, so a bus carrying both kinds combines them by
    /// inverse variance like any other repeated measurement.
    u_mag: Merged,
    /// Readings that carry an angle, merged as phasors — see [`MergedPhasor`].
    u_phasor: MergedPhasor,
    /// Injection built up from load/generator sensors.
    appliance_p: Summed,
    appliance_q: Summed,
    /// Total shunt injection, summed over the bus's measured shunts.
    shunt_p: Summed,
    shunt_q: Summed,
    /// Injection measured directly by node-injection sensors.
    node_p: Merged,
    node_q: Merged,
}

/// The five accumulators a sensor can feed, gathered so that the symmetric and
/// asymmetric loops can share one routing function.
#[derive(Default)]
struct Accumulators {
    buses: HashMap<usize, BusAccumulator>,
    /// Branch-terminal readings merge per `(branch, terminal)`.
    branch_p: HashMap<(usize, Terminal), Merged>,
    branch_q: HashMap<(usize, Terminal), Merged>,
    /// Appliance readings merge *per appliance* before the per-bus sum: two
    /// sensors on one load are one better-known load, not two loads. Doing this
    /// in one step would double-count the appliance.
    appliances: HashMap<u64, ApplianceAccumulator>,
    /// Source readings, keyed by the synthesized source branch they describe.
    sources: HashMap<usize, (Merged, Merged)>,
    /// Current readings, keyed by the terminal they measure. The frame is part
    /// of the value rather than of the key, so two sensors disagreeing about it
    /// on one terminal are caught rather than silently kept apart.
    currents: HashMap<(usize, Terminal), (AngleFrame, Merged, Merged)>,
}

/// What routing a sensor needs to know about the network it measures.
///
/// Implemented twice: once by [`PgmNetwork`] for the scalar case, and once by a
/// per-phase view of a three-phase network, which answers the same questions
/// with phase-expanded indices. That is what lets one routing function carry
/// both — the sign conventions and terminal semantics are identical, and only
/// the indices differ.
trait Resolver {
    fn resolve_terminal(&self, id: u64, terminal: Terminal) -> Option<(usize, Terminal)>;
    fn source_branch(&self, id: u64) -> Option<usize>;
    fn appliance_bus(&self, id: u64) -> Option<usize>;
    fn node_bus(&self, id: u64) -> Option<usize>;
    fn three_winding_legs(&self, id: u64) -> Option<[usize; 3]>;
}

impl Resolver for PgmNetwork {
    fn resolve_terminal(&self, id: u64, terminal: Terminal) -> Option<(usize, Terminal)> {
        PgmNetwork::resolve_terminal(self, id, terminal)
    }
    fn source_branch(&self, id: u64) -> Option<usize> {
        self.source_branch_idx.get(&id).copied()
    }
    fn appliance_bus(&self, id: u64) -> Option<usize> {
        self.appliance_bus.get(&id).copied()
    }
    fn node_bus(&self, id: u64) -> Option<usize> {
        self.node_idx.get(&id).copied()
    }
    fn three_winding_legs(&self, id: u64) -> Option<[usize; 3]> {
        self.three_winding_branch_idx.get(&id).copied()
    }
}

/// One phase of a three-phase network, seen as if it were a scalar one.
///
/// Bus `k` phase `p` is `3k + p` and branch `b` phase `p` is `3b + p`, the
/// convention `se::SeNetwork::from_3ph` builds its functionals on.
struct PhaseView<'a> {
    net: &'a crate::pgm::PgmNetwork3Ph,
    phase: usize,
}

impl Resolver for PhaseView<'_> {
    fn resolve_terminal(&self, id: u64, terminal: Terminal) -> Option<(usize, Terminal)> {
        let &branch = self.net.branch_idx.get(&id)?;
        match self.net.half_open_terminal.get(&id) {
            Some(&live) if live == terminal => Some((3 * branch + self.phase, Terminal::From)),
            Some(_) => None,
            None => Some((3 * branch + self.phase, terminal)),
        }
    }
    fn source_branch(&self, id: u64) -> Option<usize> {
        self.net.source_branch_idx.get(&id).map(|&b| 3 * b + self.phase)
    }
    fn appliance_bus(&self, id: u64) -> Option<usize> {
        self.net.appliance_bus.get(&id).map(|&k| 3 * k + self.phase)
    }
    fn node_bus(&self, id: u64) -> Option<usize> {
        self.net.node_idx.get(&id).map(|&k| 3 * k + self.phase)
    }
    fn three_winding_legs(&self, _id: u64) -> Option<[usize; 3]> {
        // `pgm_to_3ph_network` does not model three-winding transformers at all,
        // so a sensor on one has nothing to resolve against rather than
        // something to guess at.
        None
    }
}

/// One power reading, already converted to per-unit, as `(value, sigma)` per
/// component. `None` where the document left that component unset.
type PowerReading = (Option<(f64, f64)>, Option<(f64, f64)>);

/// Routes a per-unit power reading to whichever accumulator its
/// `measured_terminal_type` names.
///
/// Shared by `sym_power_sensor` and `asym_power_sensor`; the two differ only in
/// how they arrive at the per-unit `(value, sigma)` pair, never in where it goes.
fn route_power_reading(
    acc: &mut Accumulators,
    net: &dyn Resolver,
    sensor: u64,
    measured_object: u64,
    terminal_type: u8,
    (p, q): PowerReading,
) -> Result<(), MeasurementError> {
    match terminal_type {
        // Branch terminals: same direction convention as `terminal_flow`.
        0 | 1 => {
            let terminal = if terminal_type == 0 { Terminal::From } else { Terminal::To };
            let Some(key) = net.resolve_terminal(measured_object, terminal) else {
                // The branch is absent or this is its open end. PGM reports
                // zero flow there, so the reading carries no information about
                // the state and is dropped rather than fought with.
                return Ok(());
            };
            if let Some((v, s)) = p {
                acc.branch_p.entry(key).or_default().add(v, s);
            }
            if let Some((v, s)) = q {
                acc.branch_q.entry(key).or_default().add(v, s);
            }
        }
        // A source's power is a branch flow here, not a bus injection:
        // gridoxide puts a virtual slack bus and an impedance branch behind
        // every source, so by KCL its power never reaches the node's injection.
        // Generator reference direction, so no sign change.
        2 => {
            let Some(branch) = net.source_branch(measured_object) else {
                // An inactive source has no synthesized branch, so there is
                // nothing for the reading to describe.
                return Ok(());
            };
            let entry = acc.sources.entry(branch).or_default();
            if let Some((v, s)) = p {
                entry.0.add(v, s);
            }
            if let Some((v, s)) = q {
                entry.1.add(v, s);
            }
        }
        // Load/generator appliances, and shunts. `sign` converts the appliance's
        // own reference direction into an injection.
        3 | 4 | 5 => {
            let sign = match terminal_type {
                // load, shunt: load reference direction — consumption.
                3 | 4 => -1.0,
                // generator: generator reference direction.
                _ => 1.0,
            };
            let Some(bus) = net.appliance_bus(measured_object) else {
                return Err(MeasurementError::UnknownObject { sensor, object: measured_object });
            };
            let entry = acc.appliances.entry(measured_object).or_insert(ApplianceAccumulator {
                bus,
                sign,
                is_shunt: terminal_type == 3,
                p: Merged::default(),
                q: Merged::default(),
            });
            if let Some((v, s)) = p {
                entry.p.add(v, s);
            }
            if let Some((v, s)) = q {
                entry.q.add(v, s);
            }
        }
        // 6/7/8 are the three-winding transformer's three sides. gridoxide
        // models one as three two-winding legs running *physical node → star
        // bus* (`pgm_to_network`), so side k's terminal is leg k's `From` end,
        // and its flow already has PGM's sign convention: positive into the
        // transformer.
        //
        // A leg whose own `status_k` is 0 needs no special handling, unlike a
        // half-open *line*. A line with one end open collapses to a self-loop,
        // losing the distinction between its ends — which is what
        // `half_open_terminal` exists to restore. A leg keeps its two distinct
        // endpoints, and `branch_calc_param`'s `(0, 1)` case zeroes both `yff`
        // and `yft`, so the `From` flow evaluates to exactly zero. That is what
        // PGM reports for a disconnected side too, so the row is kept rather
        // than dropped: it contributes nothing to the gain matrix (an all-zero
        // Jacobian row) while still showing up as a residual if the sensor
        // disagrees.
        6 | 7 | 8 => {
            let Some(legs) = net.three_winding_legs(measured_object) else {
                return Err(MeasurementError::UnknownObject { sensor, object: measured_object });
            };
            let key = (legs[(terminal_type - 6) as usize], Terminal::From);
            if let Some((v, s)) = p {
                acc.branch_p.entry(key).or_default().add(v, s);
            }
            if let Some((v, s)) = q {
                acc.branch_q.entry(key).or_default().add(v, s);
            }
        }
        // Node injection: already a bus injection, generator direction.
        9 => {
            let Some(bus) = net.node_bus(measured_object) else {
                return Err(MeasurementError::UnknownObject { sensor, object: measured_object });
            };
            let entry = acc.buses.entry(bus).or_default();
            if let Some((v, s)) = p {
                entry.node_p.add(v, s);
            }
            if let Some((v, s)) = q {
                entry.node_q.add(v, s);
            }
        }
        other => {
            return Err(MeasurementError::UnsupportedTerminalType {
                sensor,
                terminal_type: other,
            })
        }
    }
    Ok(())
}

/// `√3`, the line-to-line / line-to-neutral ratio.
const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// The positive-sequence phasor of a three-phase reading,
/// `(U_a + a·U_b + a²·U_c) / 3` with `a = e^{j2π/3}`.
///
/// This is power-grid-model's `pos_seq` (`three_phase_tensor.hpp`), and it is
/// how an asymmetric phasor measurement becomes a symmetric one.
fn positive_sequence(magnitudes: &[f64; 3], angles: &[f64; 3]) -> num_complex::Complex<f64> {
    let a = num_complex::Complex::from_polar(1.0, std::f64::consts::TAU / 3.0);
    let rotate = [num_complex::Complex::new(1.0, 0.0), a, a * a];
    (0..3)
        .map(|p| rotate[p] * num_complex::Complex::from_polar(magnitudes[p], angles[p]))
        .sum::<num_complex::Complex<f64>>()
        / 3.0
}

/// The per-component standard deviation implied by an *apparent* power sigma.
///
/// power-grid-model splits `power_sigma` across the two components as
/// `Var(P) = Var(Q) = σ_S²/2` (`PowerSensor::sym_calc_param`), so each
/// component's own standard deviation is `σ_S/√2` rather than `σ_S`.
///
/// gridoxide used `σ_S` for both, which is the same answer whenever *every*
/// sensor in a document falls back to `power_sigma` — a uniform scaling of all
/// weights leaves a weighted-least-squares optimum untouched. It stops being
/// the same answer as soon as one sensor supplies `p_sigma`/`q_sigma` and
/// another does not, which is exactly what power-grid-model's own
/// `unbalanced-power-measurements-*` fixtures do.
fn apparent_power_component_sigma(apparent: f64) -> f64 {
    apparent / std::f64::consts::SQRT_2
}

/// The two component sigmas of a symmetric power sensor, per-unit.
///
/// Per-component values win when *both* are given, matching power-grid-model,
/// which tests them jointly rather than falling back one at a time.
fn component_sigmas(
    sensor: u64,
    p_sigma: f64,
    q_sigma: f64,
    power_sigma: f64,
    s_base_va: f64,
) -> Result<(f64, f64), MeasurementError> {
    if p_sigma.is_finite() && q_sigma.is_finite() {
        return Ok((
            checked_sigma(sensor, p_sigma)? / s_base_va,
            checked_sigma(sensor, q_sigma)? / s_base_va,
        ));
    }
    let shared =
        apparent_power_component_sigma(checked_sigma(sensor, power_sigma)? / s_base_va);
    Ok((shared, shared))
}

/// Decomposes a polar reading `(magnitude, angle)` with independent magnitude
/// and angle variances into independent real and imaginary components.
///
/// This is power-grid-model's `compute_decomposed_variance_from_polar`,
/// second-order terms included:
///
/// ```text
/// Var(Re) = σ_i²cos²θ + i²σ_θ²sin²θ + ½i²σ_θ⁴cos²θ + σ_i²σ_θ²sin²θ
/// Var(Im) = σ_i²sin²θ + i²σ_θ²cos²θ + ½i²σ_θ⁴sin²θ + σ_i²σ_θ²cos²θ
/// ```
///
/// The second-order terms are power-grid-model's modelling choice rather than
/// anything forced by the statistics — a first-order propagation would stop
/// after two terms. They are reproduced so its fixtures agree to their own
/// tolerances, and named here so a later reader does not take them for a law.
///
/// Returns `((re, σ_re), (im, σ_im))`.
fn decompose_polar(
    magnitude: f64,
    angle: f64,
    mag_sigma: f64,
    angle_sigma: f64,
) -> ((f64, f64), (f64, f64)) {
    let (sin, cos) = angle.sin_cos();
    let (sin2, cos2) = (sin * sin, cos * cos);
    let mag2 = magnitude * magnitude;
    let mag_var = mag_sigma * mag_sigma;
    let ang_var = angle_sigma * angle_sigma;

    let re_var = mag_var * cos2
        + mag2 * ang_var * sin2
        + 0.5 * mag2 * ang_var * ang_var * cos2
        + mag_var * ang_var * sin2;
    let im_var = mag_var * sin2
        + mag2 * ang_var * cos2
        + 0.5 * mag2 * ang_var * ang_var * sin2
        + mag_var * ang_var * cos2;

    ((magnitude * cos, re_var.sqrt()), (magnitude * sin, im_var.sqrt()))
}

/// A current sensor's terminal, resolved against the network.
struct CurrentTerminal {
    branch: usize,
    terminal: Terminal,
    /// Rated voltage of the node at the measured terminal, which sets the
    /// current base.
    u_rated: f64,
}

/// Resolves a current sensor's `measured_object` and `measured_terminal_type`.
///
/// `Ok(None)` means the terminal carries no flow to measure — an absent branch,
/// or the open end of a half-open one — which is what power-grid-model reports
/// zero for and what the power path already drops.
fn resolve_current_terminal(
    input: &PgmInput,
    net: &PgmNetwork,
    sensor: u64,
    object: u64,
    terminal_type: u8,
) -> Result<Option<CurrentTerminal>, MeasurementError> {
    // A `link`'s admittance is a chosen constant, so the current through one is
    // not a measurable quantity. power-grid-model refuses it too.
    if input.data.link.iter().any(|l| l.id == object) {
        return Err(MeasurementError::CurrentSensorOnLink { sensor, link: object });
    }

    let node_of = |id: u64| net.node_idx.get(&id).map(|&b| net.buses[b].u_rated);

    match terminal_type {
        0 | 1 => {
            let terminal = if terminal_type == 0 { Terminal::From } else { Terminal::To };
            let Some((branch, terminal)) = net.resolve_terminal(object, terminal) else {
                return Ok(None);
            };
            // The current base follows the node at the measured terminal, so a
            // transformer's two sides have different bases. The `current-sensor`
            // fixtures are 10 kV on both sides and would not catch this.
            let node = input
                .data
                .line
                .iter()
                .find(|l| l.id == object)
                .map(|l| if terminal_type == 0 { l.from_node } else { l.to_node })
                .or_else(|| {
                    input
                        .data
                        .transformer
                        .iter()
                        .find(|t| t.id == object)
                        .map(|t| if terminal_type == 0 { t.from_node } else { t.to_node })
                });
            let Some(u_rated) = node.and_then(node_of) else {
                return Err(MeasurementError::UnknownObject { sensor, object });
            };
            Ok(Some(CurrentTerminal { branch, terminal, u_rated }))
        }
        6 | 7 | 8 => {
            let Some(&legs) = net.three_winding_branch_idx.get(&object) else {
                return Err(MeasurementError::UnknownObject { sensor, object });
            };
            let side = (terminal_type - 6) as usize;
            let node = input
                .data
                .three_winding_transformer
                .iter()
                .find(|t| t.id == object)
                .map(|t| [t.node_1, t.node_2, t.node_3][side]);
            let Some(u_rated) = node.and_then(node_of) else {
                return Err(MeasurementError::UnknownObject { sensor, object });
            };
            Ok(Some(CurrentTerminal { branch: legs[side], terminal: Terminal::From, u_rated }))
        }
        // power-grid-model's `CurrentSensor` constructor rejects every other
        // terminal type outright: a current sensor belongs on a branch, and
        // there is no current to speak of at a bus or an appliance.
        other => Err(MeasurementError::UnsupportedTerminalType { sensor, terminal_type: other }),
    }
}

/// Validates a standard deviation.
///
/// An *infinite* sigma is allowed and meaningful: it is power-grid-model's way
/// of saying a measurement exists but carries no information, giving it a
/// weight of exactly zero (`inf-measurement-with-injection` and friends test
/// precisely this). NaN and non-positive values are not — a missing
/// uncertainty has no defensible weight, and zero would mean infinite weight.
fn checked_sigma(sensor: u64, sigma: f64) -> Result<f64, MeasurementError> {
    if sigma.is_nan() || sigma <= 0.0 {
        return Err(MeasurementError::InvalidSigma { sensor, sigma });
    }
    Ok(sigma)
}

/// Builds the aggregated measurement set from a PGM input document and the
/// network it was converted into.
///
/// `net` must come from the *same* document — it supplies the object-ID maps
/// that turn a sensor's `measured_object` into a bus or branch index.
///
/// Sensors whose value is missing (NaN, PGM's "unset") are skipped silently;
/// sensors whose *sigma* is missing are an error, since a measurement with no
/// stated uncertainty has no defensible weight.
pub fn measurements_from_pgm(
    input: &PgmInput,
    net: &PgmNetwork,
    s_base_va: f64,
) -> Result<Vec<Measurement>, MeasurementError> {
    let mut acc = Accumulators::default();

    // ── Voltage sensors ──────────────────────────────────────────────────
    for s in &input.data.sym_voltage_sensor {
        let Some(&bus) = net.node_idx.get(&s.measured_object) else {
            return Err(MeasurementError::UnknownObject {
                sensor: s.id,
                object: s.measured_object,
            });
        };
        // u_measured is a line-to-line voltage in V, as is the node's rating,
        // so their ratio is directly the per-unit magnitude.
        let u_rated = net.buses[bus].u_rated;
        let bus_acc = acc.buses.entry(bus).or_default();

        if !s.u_measured.is_finite() {
            continue;
        }
        let sigma = checked_sigma(s.id, s.u_sigma)? / u_rated;
        let magnitude = s.u_measured / u_rated;
        if s.u_angle_measured.is_finite() {
            // A phasor reading merges as a phasor. An angle sigma is optional
            // even when an angle is given; PGM has no `u_angle_sigma` field at
            // all and weights the whole complex value by `u_sigma`, which is
            // what the phasor merge does here. gridoxide's extra
            // `u_angle_sigma` still overrides it when present.
            let sigma = if s.u_angle_sigma.is_finite() {
                checked_sigma(s.id, s.u_angle_sigma)?
            } else {
                sigma
            };
            bus_acc
                .u_phasor
                .add(num_complex::Complex::from_polar(magnitude, s.u_angle_measured), sigma);
        } else {
            bus_acc.u_mag.add(magnitude, sigma);
        }
    }

    // ── Asymmetric voltage sensors, reduced to the symmetric problem ─────
    //
    // power-grid-model does this itself when a document carrying asymmetric
    // sensors is solved symmetrically (`VoltageSensor<asymmetric_t>::
    // sym_calc_param`), so this is not an approximation gridoxide invents: with
    // angles it takes the positive sequence of the three phasors, without them
    // the mean of the three magnitudes.
    for s in &input.data.asym_voltage_sensor {
        let Some(&bus) = net.node_idx.get(&s.measured_object) else {
            return Err(MeasurementError::UnknownObject {
                sensor: s.id,
                object: s.measured_object,
            });
        };
        if !s.u_measured.iter().all(|v| v.is_finite()) {
            continue;
        }
        // An asymmetric reading is line-to-neutral, so its base is `u_rated/√3`
        // rather than `u_rated` — PGM's `u_scale` (`common.hpp`).
        let u_base = net.buses[bus].u_rated / SQRT_3;
        let sigma = checked_sigma(s.id, s.u_sigma)? / u_base;
        let mag = s.u_measured.map(|v| v / u_base);
        let bus_acc = acc.buses.entry(bus).or_default();

        // PGM requires *every* phase angle to be present before it treats the
        // sensor as a phasor (`has_angle()` is `!isNaN().any()`).
        if s.u_angle_measured.iter().all(|a| a.is_finite()) {
            bus_acc.u_phasor.add(positive_sequence(&mag, &s.u_angle_measured), sigma);
        } else {
            bus_acc.u_mag.add(mag.iter().sum::<f64>() / 3.0, sigma);
        }
    }

    // ── Power sensors ────────────────────────────────────────────────────
    for s in &input.data.sym_power_sensor {
        if !s.p_measured.is_finite() && !s.q_measured.is_finite() {
            continue;
        }
        let (p_sigma, q_sigma) =
            component_sigmas(s.id, s.p_sigma, s.q_sigma, s.power_sigma, s_base_va)?;
        let reading = (
            s.p_measured.is_finite().then(|| (s.p_measured / s_base_va, p_sigma)),
            s.q_measured.is_finite().then(|| (s.q_measured / s_base_va, q_sigma)),
        );
        route_power_reading(
            &mut acc,
            net,
            s.id,
            s.measured_object,
            s.measured_terminal_type,
            reading,
        )?;
    }

    // ── Asymmetric power sensors, reduced to the symmetric problem ───────
    //
    // As with voltage, this is power-grid-model's own reduction
    // (`PowerSensor<asymmetric_t>::sym_calc_param`): the value is the mean of
    // the three per-phase powers and the variance is the mean of the three
    // per-phase variances. Note the *mean* of the variances, not their sum —
    // PGM's approximation, and one that looks like a bug unless you know it is
    // deliberate.
    //
    // The per-unit base differs too: a per-phase power is per-unit against
    // `s_base/3` (PGM's `base_power_1p`), not the three-phase `s_base`.
    for s in &input.data.asym_power_sensor {
        let p_any = s.p_measured.iter().any(|v| v.is_finite());
        let q_any = s.q_measured.iter().any(|v| v.is_finite());
        if !p_any && !q_any {
            continue;
        }
        let s_base_1ph = s_base_va / 3.0;
        let mean = |vs: &[f64; 3]| vs.iter().sum::<f64>() / 3.0 / s_base_1ph;
        // A sigma is per-phase here, unlike the voltage sensor's. Reduce the
        // three variances to their mean and hand back one standard deviation.
        let rms = |vs: &[f64; 3]| {
            (vs.iter().map(|v| (v / s_base_1ph).powi(2)).sum::<f64>() / 3.0).sqrt()
        };
        let per_phase_finite = |vs: &[f64; 3]| vs.iter().all(|v| v.is_finite());

        let (p_sigma, q_sigma) = if per_phase_finite(&s.p_sigma) && per_phase_finite(&s.q_sigma) {
            (checked_sigma(s.id, rms(&s.p_sigma))?, checked_sigma(s.id, rms(&s.q_sigma))?)
        } else {
            let shared = apparent_power_component_sigma(
                checked_sigma(s.id, s.power_sigma)? / s_base_1ph,
            );
            (shared, shared)
        };
        let reading = (
            p_any.then(|| (mean(&s.p_measured), p_sigma)),
            q_any.then(|| (mean(&s.q_measured), q_sigma)),
        );
        route_power_reading(
            &mut acc,
            net,
            s.id,
            s.measured_object,
            s.measured_terminal_type,
            reading,
        )?;
    }

    // ── Current sensors ──────────────────────────────────────────────────
    //
    // Stored decomposed into real and imaginary components rather than as a
    // magnitude and an angle — see `MeasurementKind::CurrentReal` for why, and
    // `decompose_polar` for the variance conversion, which is
    // power-grid-model's including its second-order terms.
    let sym_currents = input.data.sym_current_sensor.iter().map(|s| {
        (
            s.id,
            s.measured_object,
            s.measured_terminal_type,
            s.angle_measurement_type,
            s.i_measured,
            s.i_angle_measured,
            s.i_sigma,
            s.i_angle_sigma,
        )
    });
    // An asymmetric current sensor reduces to the symmetric problem the way its
    // voltage and power counterparts do: the positive sequence when every phase
    // carries an angle, the mean of the magnitudes otherwise.
    let asym_currents = input.data.asym_current_sensor.iter().map(|s| {
        let (magnitude, angle) = if s.i_angle_measured.iter().all(|a| a.is_finite()) {
            let seq = positive_sequence(&s.i_measured, &s.i_angle_measured);
            (seq.norm(), seq.arg())
        } else {
            (s.i_measured.iter().sum::<f64>() / 3.0, f64::NAN)
        };
        (
            s.id,
            s.measured_object,
            s.measured_terminal_type,
            s.angle_measurement_type,
            magnitude,
            angle,
            s.i_sigma,
            s.i_angle_sigma,
        )
    });

    for (id, object, terminal_type, frame_code, i_measured, i_angle, i_sigma, i_angle_sigma) in
        sym_currents.chain(asym_currents)
    {
        if !i_measured.is_finite() || !i_angle.is_finite() {
            // A current without an angle determines nothing linear in the
            // voltages, so unlike a voltage magnitude it cannot be kept as half
            // a measurement.
            continue;
        }
        let Some(t) = resolve_current_terminal(input, net, id, object, terminal_type)? else {
            continue;
        };
        let frame = if frame_code == 0 { AngleFrame::Local } else { AngleFrame::Global };

        // Per-unit against the base current of the node at this terminal:
        // `I_base = S_base / (√3 · u_rated)`.
        let i_base = s_base_va / (SQRT_3 * t.u_rated);
        let magnitude = i_measured / i_base;
        let mag_sigma = checked_sigma(id, i_sigma)? / i_base;
        let angle_sigma = checked_sigma(id, i_angle_sigma)?;
        let ((re, re_sigma), (im, im_sigma)) =
            decompose_polar(magnitude, i_angle, mag_sigma, angle_sigma);

        // Power sensors are read before this loop, so checking one direction
        // catches the mixture whichever sensor the document lists first.
        let key = (t.branch, t.terminal);
        if acc.branch_p.contains_key(&key) || acc.branch_q.contains_key(&key) {
            return Err(MeasurementError::MixedPowerAndCurrent { sensor: id, object });
        }
        let entry = acc
            .currents
            .entry(key)
            .or_insert((frame, Merged::default(), Merged::default()));
        if entry.0 != frame {
            return Err(MeasurementError::ConflictingAngleFrame { sensor: id, object });
        }
        entry.1.add(re, re_sigma);
        entry.2.add(im, im_sigma);
    }

    Ok(flatten(acc))
}


/// Turns the accumulators into measurement rows, in a deterministic order.
///
/// Shared by the scalar and three-phase builders: the two differ in how a
/// sensor reaches an accumulator, never in what an accumulator becomes.
fn flatten(mut acc: Accumulators) -> Vec<Measurement> {
    // Each appliance contributes its merged reading, sign-corrected, to its
    // bus's injection sum. Iterated in id order so the summed variance is
    // built deterministically.
    let mut appliance_ids: Vec<u64> = acc.appliances.keys().copied().collect();
    appliance_ids.sort_unstable();
    for id in appliance_ids {
        let app = acc.appliances[&id];
        let bus = acc.buses.entry(app.bus).or_default();
        let (p_acc, q_acc) = if app.is_shunt {
            (&mut bus.shunt_p, &mut bus.shunt_q)
        } else {
            (&mut bus.appliance_p, &mut bus.appliance_q)
        };
        if let Some((value, sigma)) = app.p.finish() {
            p_acc.add(app.sign * value, sigma);
        }
        if let Some((value, sigma)) = app.q.finish() {
            q_acc.add(app.sign * value, sigma);
        }
    }

    // ── Flatten the accumulators into measurement rows ───────────────────
    let mut out = Vec::new();

    let mut bus_ids: Vec<usize> = acc.buses.keys().copied().collect();
    bus_ids.sort_unstable();
    for bus in bus_ids {
        let b = acc.buses[&bus];
        let target = Target::Bus(bus);
        // The phasor group contributes its magnitude to the magnitude merge and
        // supplies the angle. A bus with only magnitude-only sensors gets no
        // angle row at all, which is what leaves the global phase to
        // `StateLayout`'s pinned reference.
        let mut u_mag = b.u_mag;
        let phasor = b.u_phasor.finish();
        if let Some((value, sigma)) = phasor {
            u_mag.add(value.norm(), sigma);
        }
        if let Some((value, sigma)) = u_mag.finish() {
            out.push(Measurement { kind: MeasurementKind::VoltageMagnitude, target, value, sigma });
        }
        if let Some((value, sigma)) = phasor {
            out.push(Measurement {
                kind: MeasurementKind::VoltageAngle,
                target,
                value: value.arg(),
                sigma,
            });
        }
        // Load/generator injection, shunt injection and node injection are
        // three *different* measurement functions of the same state, not three
        // observations of one quantity, so they stay separate rows. Merging
        // them (as an earlier version did) would assert that a node-injection
        // sensor and a load sensor see the same thing, which is only true when
        // the bus has no source and no shunt.
        for (kind, appliance, shunt, node) in [
            (
                MeasurementKind::ActivePower,
                b.appliance_p.finish(),
                b.shunt_p.finish(),
                b.node_p.finish(),
            ),
            (
                MeasurementKind::ReactivePower,
                b.appliance_q.finish(),
                b.shunt_q.finish(),
                b.node_q.finish(),
            ),
        ] {
            if let Some((value, sigma)) = appliance {
                out.push(Measurement { kind, target, value, sigma });
            }
            if let Some((value, sigma)) = shunt {
                out.push(Measurement {
                    kind,
                    target: Target::ShuntInjection { bus },
                    value,
                    sigma,
                });
            }
            if let Some((value, sigma)) = node {
                out.push(Measurement {
                    kind,
                    target: Target::NodeInjection(bus),
                    value,
                    sigma,
                });
            }
        }
    }

    let mut source_branches: Vec<usize> = acc.sources.keys().copied().collect();
    source_branches.sort_unstable();
    for branch in source_branches {
        let (p, q) = acc.sources[&branch];
        let target = Target::SourceInjection { branch };
        if let Some((value, sigma)) = p.finish() {
            out.push(Measurement { kind: MeasurementKind::ActivePower, target, value, sigma });
        }
        if let Some((value, sigma)) = q.finish() {
            out.push(Measurement { kind: MeasurementKind::ReactivePower, target, value, sigma });
        }
    }

    let mut branch_keys: Vec<(usize, Terminal)> =
        acc.branch_p.keys().chain(acc.branch_q.keys()).copied().collect();
    branch_keys.sort_unstable_by_key(|&(b, t)| (b, t == Terminal::To));
    branch_keys.dedup();
    for key in branch_keys {
        let (branch, terminal) = key;
        let target = Target::BranchTerminal { branch, terminal };
        if let Some((value, sigma)) = acc.branch_p.get(&key).copied().unwrap_or_default().finish() {
            out.push(Measurement { kind: MeasurementKind::ActivePower, target, value, sigma });
        }
        if let Some((value, sigma)) = acc.branch_q.get(&key).copied().unwrap_or_default().finish() {
            out.push(Measurement { kind: MeasurementKind::ReactivePower, target, value, sigma });
        }
    }

    let mut current_keys: Vec<(usize, Terminal)> = acc.currents.keys().copied().collect();
    current_keys.sort_unstable_by_key(|&(b, t)| (b, t == Terminal::To));
    for key in current_keys {
        let (branch, terminal) = key;
        let (frame, re, im) = acc.currents[&key];
        let target = Target::BranchTerminalCurrent { branch, terminal, frame };
        if let Some((value, sigma)) = re.finish() {
            out.push(Measurement { kind: MeasurementKind::CurrentReal, target, value, sigma });
        }
        if let Some((value, sigma)) = im.finish() {
            out.push(Measurement { kind: MeasurementKind::CurrentImag, target, value, sigma });
        }
    }

    out
}

/// The phase angles a balanced positive-sequence quantity takes, per phase.
///
/// power-grid-model broadcasts a scalar to three phases as `x, x·a², x·a` with
/// `a = e^{j2π/3}` (`three_phase_tensor.hpp`), which is this. It is also the
/// flat start `pgm_to_3ph_network` uses, so the two agree by construction.
const PHASE_ANGLE: [f64; 3] = [
    0.0,
    -std::f64::consts::TAU / 3.0,
    std::f64::consts::TAU / 3.0,
];

/// Builds the measurement set for a *three-phase* network.
///
/// Targets carry phase-expanded indices — bus `3k + p`, branch `3b + p` — so
/// everything downstream is the same code the scalar path uses. See
/// [`se::SeNetwork::from_3ph`](crate::se::SeNetwork::from_3ph).
///
/// A **symmetric** sensor describes all three phases at once, and both of its
/// per-unit bases happen to carry over unchanged. A voltage reading is
/// line-to-line over `u_rated` in the scalar case and line-to-neutral over
/// `u_rated/√3` here, which is the same number for a balanced set; a power
/// reading is a three-phase total over `s_base` against a per-phase value over
/// `s_base/3`, likewise. So the value replicates and only the *angle* rotates,
/// by `PHASE_ANGLE`. power-grid-model reaches the same conclusion through its
/// `ComplexValue<asymmetric_t>` broadcast.
///
/// Sigmas replicate rather than divide, which triples a symmetric sensor's total
/// variance relative to the scalar run. That is power-grid-model's choice too,
/// and it is an approximation rather than a derivation — three phase readings
/// from one instrument are not three independent measurements.
pub fn measurements_from_pgm_3ph(
    input: &PgmInput,
    net: &crate::pgm::PgmNetwork3Ph,
    s_base_va: f64,
    u_rated: &dyn Fn(usize) -> f64,
) -> Result<Vec<Measurement>, MeasurementError> {
    let mut acc = Accumulators::default();

    for phase in 0..3 {
        let view = PhaseView { net, phase };

        for s in &input.data.sym_voltage_sensor {
            let Some(bus) = view.node_bus(s.measured_object) else {
                return Err(MeasurementError::UnknownObject {
                    sensor: s.id,
                    object: s.measured_object,
                });
            };
            if !s.u_measured.is_finite() {
                continue;
            }
            // `u_rated` is the *phase bus*'s own rating, which
            // `pgm_to_3ph_network` sets to the node's line-to-line value on all
            // three phases — so the ratio is per-unit on the same base the
            // scalar path uses, and the equivalence above holds.
            let base = u_rated(bus);
            let sigma = checked_sigma(s.id, s.u_sigma)? / base;
            let magnitude = s.u_measured / base;
            let bus_acc = acc.buses.entry(bus).or_default();
            if s.u_angle_measured.is_finite() {
                let sigma = if s.u_angle_sigma.is_finite() {
                    checked_sigma(s.id, s.u_angle_sigma)?
                } else {
                    sigma
                };
                bus_acc.u_phasor.add(
                    num_complex::Complex::from_polar(
                        magnitude,
                        s.u_angle_measured + PHASE_ANGLE[phase],
                    ),
                    sigma,
                );
            } else {
                bus_acc.u_mag.add(magnitude, sigma);
            }
        }

        for s in &input.data.sym_power_sensor {
            if !s.p_measured.is_finite() && !s.q_measured.is_finite() {
                continue;
            }
            let (p_sigma, q_sigma) =
                component_sigmas(s.id, s.p_sigma, s.q_sigma, s.power_sigma, s_base_va)?;
            let reading = (
                s.p_measured.is_finite().then(|| (s.p_measured / s_base_va, p_sigma)),
                s.q_measured.is_finite().then(|| (s.q_measured / s_base_va, q_sigma)),
            );
            route_power_reading(
                &mut acc,
                &view,
                s.id,
                s.measured_object,
                s.measured_terminal_type,
                reading,
            )?;
        }
    }

    Ok(flatten(acc))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two equally-trusted readings of one quantity average, and the result is
    /// `√2` times more certain than either — the standard inverse-variance
    /// result, and the reason redundant sensors are merged rather than passed
    /// through as separate rows.
    #[test]
    fn equal_sensors_merge_to_mean_with_tighter_sigma() {
        let mut m = Merged::default();
        m.add(1.0, 0.1);
        m.add(2.0, 0.1);
        let (value, sigma) = m.finish().unwrap();
        assert!((value - 1.5).abs() < 1e-12);
        assert!((sigma - 0.1 / 2.0_f64.sqrt()).abs() < 1e-12, "sigma={sigma}");
    }

    /// A much noisier second reading barely moves the estimate.
    #[test]
    fn merge_weights_by_inverse_variance() {
        let mut m = Merged::default();
        m.add(1.0, 0.1);
        m.add(5.0, 1.0);
        let (value, sigma) = m.finish().unwrap();
        // Weights are 100 and 1, so the mean sits 1/101 of the way to 5.
        assert!((value - (100.0 + 5.0) / 101.0).abs() < 1e-12, "value={value}");
        assert!(sigma < 0.1, "merging must not lose certainty: {sigma}");
    }

    /// Summing appliances is the opposite case: the total is less certain than
    /// its parts, because independent errors accumulate.
    #[test]
    fn summed_appliances_add_variances() {
        let mut s = Summed::default();
        s.add(1.0, 0.3);
        s.add(2.0, 0.4);
        let (value, sigma) = s.finish().unwrap();
        assert!((value - 3.0).abs() < 1e-12);
        // √(0.09 + 0.16) = 0.5
        assert!((sigma - 0.5).abs() < 1e-12, "sigma={sigma}");
        assert!(sigma > 0.4, "summing must not gain certainty");
    }

    #[test]
    fn empty_accumulators_yield_nothing() {
        assert_eq!(Merged::default().finish(), None);
        assert_eq!(Summed::default().finish(), None);
    }

    #[test]
    fn weight_is_inverse_variance() {
        let m = Measurement {
            kind: MeasurementKind::ActivePower,
            target: Target::Bus(0),
            value: 0.5,
            sigma: 0.25,
        };
        assert!((m.weight() - 16.0).abs() < 1e-12);
    }

    /// A two-node grid with one line, one source and one load, plus whichever
    /// sensors a test wants. Small enough to reason about by hand: node 1 is
    /// the source side, node 2 carries the load, and branch 0 is the line
    /// (branch 1 is the virtual source branch gridoxide synthesizes).
    fn network_with_sensors(sensors: &str) -> (PgmInput, PgmNetwork) {
        let json = format!(
            r#"{{"version":"1.0","type":"input","is_batch":false,"attributes":{{}},"data":{{
                "node":[{{"id":1,"u_rated":10000.0}},{{"id":2,"u_rated":10000.0}}],
                "line":[{{"id":3,"from_node":1,"to_node":2,"from_status":1,"to_status":1,
                          "r1":0.1,"x1":0.2,"c1":0.0,"tan1":0.0,
                          "r0":0.1,"x0":0.2,"c0":0.0}}],
                "source":[{{"id":4,"node":1,"status":1,"u_ref":1.0,"sk":1e10,"rx_ratio":0.1}}],
                "sym_load":[{{"id":5,"node":2,"status":1,"type":0,
                              "p_specified":1000.0,"q_specified":0.0}}],
                {sensors}
            }}}}"#
        );
        let input: PgmInput = serde_json::from_str(&json).expect("test fixture parses");
        // The conversion consumes its input, so parse twice rather than
        // threading a clone through every caller.
        let net = crate::pgm::pgm_to_network(
            serde_json::from_str(&json).expect("test fixture parses"),
            1e6,
            50.0,
        );
        (input, net)
    }

    /// Two phasor sensors on one bus merge as *phasors*, not as a magnitude and
    /// an angle separately.
    ///
    /// The two agree whenever the readings share an angle, which is why every
    /// fixture in this repo agreed with either rule until
    /// `sensor-update-initially-empty` arrived with 11.0∠−0.1 against 9.0∠+0.1.
    /// The vector sum is shorter than the scalar one whenever the phasors
    /// disagree: 1.05703 here, against 1.06 from merging the magnitudes alone.
    #[test]
    fn phasor_sensors_merge_as_vectors_not_as_magnitudes() {
        let (input, net) = network_with_sensors(
            r#""sym_voltage_sensor":[
                {"id":9,"measured_object":1,"u_measured":11000.0,"u_sigma":1000.0,
                 "u_angle_measured":-0.1},
                {"id":10,"measured_object":1,"u_measured":9000.0,"u_sigma":2000.0,
                 "u_angle_measured":0.1}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).unwrap();
        let mag = find(&ms, MeasurementKind::VoltageMagnitude, Target::Bus(0));
        let ang = find(&ms, MeasurementKind::VoltageAngle, Target::Bus(0));

        // (1.1·e^{-0.1i}/0.01 + 0.9·e^{0.1i}/0.04) / (100 + 25). These are the
        // same two phasors `sensor-update-initially-empty` carries at ten times
        // the scale, and power-grid-model publishes exactly
        // `u = 10.570170726436285`, `u_angle = -0.0661620368126038` for them —
        // so this pins the aggregation against PGM's own arithmetic, not just
        // against gridoxide's.
        assert!((mag.value - 1.0570170726436283).abs() < 1e-12, "magnitude={}", mag.value);
        assert!((ang.value + 0.0661620368126038).abs() < 1e-12, "angle={}", ang.value);
        assert!(
            (mag.value - 1.06).abs() > 1e-4,
            "merging the magnitudes alone would give exactly 1.06 — the point is that it does not"
        );
        // Both rows carry the merged phasor's own sigma, 1/√125.
        assert!((mag.sigma - 125.0f64.sqrt().recip()).abs() < 1e-12, "sigma={}", mag.sigma);
        assert!((ang.sigma - 125.0f64.sqrt().recip()).abs() < 1e-12);
    }

    /// A magnitude-only sensor still merges as a scalar, and produces no angle
    /// row at all — which is what leaves the global phase to `StateLayout`'s
    /// pinned reference.
    #[test]
    fn magnitude_only_sensors_produce_no_angle_row() {
        let (input, net) = network_with_sensors(
            r#""sym_voltage_sensor":[
                {"id":9,"measured_object":1,"u_measured":11000.0,"u_sigma":1000.0},
                {"id":10,"measured_object":1,"u_measured":9000.0,"u_sigma":2000.0}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).unwrap();
        let mag = find(&ms, MeasurementKind::VoltageMagnitude, Target::Bus(0));
        assert!((mag.value - 1.06).abs() < 1e-12, "magnitude={}", mag.value);
        assert!(
            !ms.iter().any(|m| m.kind == MeasurementKind::VoltageAngle),
            "no sensor supplied an angle, so no angle row should exist"
        );
    }

    /// A current sensor's per-unit base follows the node at the *measured*
    /// terminal, and its polar reading decomposes with power-grid-model's own
    /// variance formula.
    #[test]
    fn a_current_sensor_converts_to_decomposed_per_unit() {
        // 10 kV node, 1 MVA base: I_base = 1e6/(√3·1e4) = 57.735 A.
        let (input, net) = network_with_sensors(
            r#""sym_current_sensor":[{"id":9,"measured_object":3,"measured_terminal_type":1,
                "angle_measurement_type":1,"i_measured":10.0,"i_angle_measured":0.5,
                "i_sigma":5.0,"i_angle_sigma":0.1}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).expect("measurements");
        let target = Target::BranchTerminalCurrent {
            branch: 0,
            terminal: Terminal::To,
            frame: AngleFrame::Global,
        };
        let re = find(&ms, MeasurementKind::CurrentReal, target);
        let im = find(&ms, MeasurementKind::CurrentImag, target);

        // I_base = 1e6/(√3·1e4) = 57.735 A, so 10 A is 0.173205 p.u.
        let i_pu = 10.0 * SQRT_3 * 10000.0 / 1e6;
        assert!((re.value - i_pu * 0.5f64.cos()).abs() < 1e-12, "Re={}", re.value);
        assert!((im.value - i_pu * 0.5f64.sin()).abs() < 1e-12, "Im={}", im.value);

        // Var(Re) = σ²cos²θ + i²σ_θ²sin²θ + ½i²σ_θ⁴cos²θ + σ²σ_θ²sin²θ
        let sigma = 5.0 * SQRT_3 * 10000.0 / 1e6;
        let (sin, cos) = 0.5f64.sin_cos();
        let (v, a) = (sigma * sigma, 0.01);
        let want_re = (v * cos * cos
            + i_pu * i_pu * a * sin * sin
            + 0.5 * i_pu * i_pu * a * a * cos * cos
            + v * a * sin * sin)
            .sqrt();
        assert!((re.sigma - want_re).abs() < 1e-14, "σ_Re={} want {want_re}", re.sigma);
    }

    /// A current sensor and a power sensor may not share a terminal.
    ///
    /// power-grid-model rejects this in its Python validation layer only — its
    /// C++ core accepts the mixture and double-counts it — so gridoxide is
    /// deliberately the stricter of the two here.
    #[test]
    fn a_power_and_a_current_sensor_may_not_share_a_terminal() {
        let (input, net) = network_with_sensors(
            r#""sym_power_sensor":[{"id":8,"measured_object":3,"measured_terminal_type":1,
                "p_measured":1000.0,"q_measured":0.0,"power_sigma":10.0}],
               "sym_current_sensor":[{"id":9,"measured_object":3,"measured_terminal_type":1,
                "angle_measurement_type":1,"i_measured":10.0,"i_angle_measured":0.5,
                "i_sigma":5.0,"i_angle_sigma":0.1}]"#,
        );
        assert_eq!(
            measurements_from_pgm(&input, &net, 1e6),
            Err(MeasurementError::MixedPowerAndCurrent { sensor: 9, object: 3 })
        );
    }

    /// Two current sensors on one terminal must agree about the angle frame,
    /// since local and global are different quantities.
    #[test]
    fn two_current_sensors_on_one_terminal_must_share_a_frame() {
        let (input, net) = network_with_sensors(
            r#""sym_current_sensor":[
                {"id":9,"measured_object":3,"measured_terminal_type":1,"angle_measurement_type":1,
                 "i_measured":10.0,"i_angle_measured":0.5,"i_sigma":5.0,"i_angle_sigma":0.1},
                {"id":10,"measured_object":3,"measured_terminal_type":1,"angle_measurement_type":0,
                 "i_measured":10.0,"i_angle_measured":0.5,"i_sigma":5.0,"i_angle_sigma":0.1}]"#,
        );
        assert_eq!(
            measurements_from_pgm(&input, &net, 1e6),
            Err(MeasurementError::ConflictingAngleFrame { sensor: 10, object: 3 })
        );
    }

    /// A current sensor on a bus or an appliance is meaningless and rejected,
    /// matching power-grid-model's own constructor.
    #[test]
    fn a_current_sensor_off_a_branch_is_rejected() {
        let (input, net) = network_with_sensors(
            r#""sym_current_sensor":[{"id":9,"measured_object":5,"measured_terminal_type":4,
                "angle_measurement_type":1,"i_measured":10.0,"i_angle_measured":0.5,
                "i_sigma":5.0,"i_angle_sigma":0.1}]"#,
        );
        assert_eq!(
            measurements_from_pgm(&input, &net, 1e6),
            Err(MeasurementError::UnsupportedTerminalType { sensor: 9, terminal_type: 4 })
        );
    }

    /// `measured_terminal_type` 6/7/8 land on the three legs' `From` terminals,
    /// in side order.
    ///
    /// The integration test checks the resulting *estimate* against
    /// power-grid-model's published answer; this pins the mapping itself, which
    /// is the part that would fail silently — a sensor routed to the wrong leg
    /// still produces a plausible-looking measurement set.
    #[test]
    fn three_winding_sides_map_to_their_legs_from_terminal() {
        let json = r#"{"version":"1.0","type":"input","is_batch":false,"attributes":{},"data":{
            "node":[{"id":1,"u_rated":138000.0},{"id":2,"u_rated":69000.0},{"id":3,"u_rated":13800.0}],
            "three_winding_transformer":[{"id":4,"node_1":1,"node_2":2,"node_3":3,
                "status_1":1,"status_2":1,"status_3":1,
                "u1":138000.0,"u2":69000.0,"u3":13800.0,
                "sn_1":60000000.0,"sn_2":50000000.0,"sn_3":10000000.0,
                "uk_12":0.09,"uk_13":0.03,"uk_23":0.06,
                "pk_12":50000.0,"pk_13":5000.0,"pk_23":10000.0,
                "i0":0.1,"p0":50000.0,"winding_1":1,"winding_2":2,"winding_3":2,
                "clock_12":11,"clock_13":11,"tap_side":2,"tap_pos":0,
                "tap_min":-8,"tap_max":10,"tap_nom":0,"tap_size":1380.0}],
            "source":[{"id":7,"node":1,"status":1,"u_ref":1.0,"sk":1e12,"rx_ratio":0.1}],
            "sym_power_sensor":[
                {"id":61,"measured_object":4,"measured_terminal_type":6,
                 "p_measured":1000000.0,"q_measured":0.0,"power_sigma":100000.0},
                {"id":62,"measured_object":4,"measured_terminal_type":7,
                 "p_measured":-600000.0,"q_measured":0.0,"power_sigma":100000.0},
                {"id":63,"measured_object":4,"measured_terminal_type":8,
                 "p_measured":-400000.0,"q_measured":0.0,"power_sigma":100000.0}]
        }}"#;
        let input: PgmInput = serde_json::from_str(json).expect("fixture parses");
        let net = crate::pgm::pgm_to_network(
            serde_json::from_str(json).expect("fixture parses"),
            1e6,
            50.0,
        );
        let legs = net.three_winding_branch_idx[&4];
        let ms = measurements_from_pgm(&input, &net, 1e6).expect("measurements");

        for (side, expected_p) in [(0usize, 1.0), (1, -0.6), (2, -0.4)] {
            let m = find(
                &ms,
                MeasurementKind::ActivePower,
                Target::BranchTerminal { branch: legs[side], terminal: Terminal::From },
            );
            assert!(
                (m.value - expected_p).abs() < 1e-12,
                "side {side} landed on leg {} with value {}",
                legs[side],
                m.value
            );
        }
    }

    /// A terminal type gridoxide genuinely does not model is still an error,
    /// rather than being swallowed by the 6/7/8 arm's neighbours.
    #[test]
    fn an_unknown_terminal_type_is_still_rejected() {
        let (input, net) = network_with_sensors(
            r#""sym_power_sensor":[{"id":9,"measured_object":3,"measured_terminal_type":11,
                "p_measured":1.0,"q_measured":0.0,"power_sigma":1.0}]"#,
        );
        assert_eq!(
            measurements_from_pgm(&input, &net, 1e6),
            Err(MeasurementError::UnsupportedTerminalType { sensor: 9, terminal_type: 11 })
        );
    }

    fn find(ms: &[Measurement], kind: MeasurementKind, target: Target) -> Measurement {
        *ms.iter()
            .find(|m| m.kind == kind && m.target == target)
            .unwrap_or_else(|| panic!("no {kind:?} measurement on {target:?} in {ms:#?}"))
    }

    /// A voltage sensor's volts become per-unit against the node's rating, and
    /// its sigma is scaled the same way.
    #[test]
    fn voltage_sensor_converts_to_per_unit() {
        let (input, net) = network_with_sensors(
            r#""sym_voltage_sensor":[{"id":10,"measured_object":2,
                 "u_measured":10200.0,"u_sigma":100.0}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).unwrap();
        let m = find(&ms, MeasurementKind::VoltageMagnitude, Target::Bus(1));
        assert!((m.value - 1.02).abs() < 1e-12, "value={}", m.value);
        assert!((m.sigma - 0.01).abs() < 1e-12, "sigma={}", m.sigma);
    }

    /// A load uses the *load* reference direction: positive `p_measured` means
    /// consumption, which is a negative bus injection. Getting this backwards
    /// gives a confidently wrong estimate rather than a failure, so it is
    /// asserted directly.
    #[test]
    fn load_sensor_becomes_negative_injection() {
        let (input, net) = network_with_sensors(
            r#""sym_power_sensor":[{"id":11,"measured_object":5,"measured_terminal_type":4,
                 "p_measured":2000.0,"q_measured":500.0,"power_sigma":10.0}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).unwrap();
        let p = find(&ms, MeasurementKind::ActivePower, Target::Bus(1));
        let q = find(&ms, MeasurementKind::ReactivePower, Target::Bus(1));
        assert!((p.value + 0.002).abs() < 1e-12, "p={}", p.value);
        assert!((q.value + 0.0005).abs() < 1e-12, "q={}", q.value);
        // `power_sigma` is an *apparent* power sigma, which power-grid-model
        // splits across the two components as `Var(P) = Var(Q) = σ_S²/2`. So the
        // per-component deviation is `σ_S/√2`, not `σ_S`: 10/1e6/√2.
        assert!(
            (p.sigma - 1e-5 / std::f64::consts::SQRT_2).abs() < 1e-15,
            "sigma={}",
            p.sigma
        );
    }

    /// `p_sigma`/`q_sigma` are per-component already, so they are *not* divided
    /// by √2 the way `power_sigma` is.
    ///
    /// This is the distinction that makes the split matter. When every sensor in
    /// a document falls back to `power_sigma`, scaling them all by the same
    /// constant leaves the weighted-least-squares optimum untouched and the rule
    /// is unobservable. Mixing the two forms in one document is what exposes it,
    /// and power-grid-model's `unbalanced-power-measurements-*` fixtures do
    /// exactly that.
    #[test]
    fn per_component_sigmas_are_not_rescaled() {
        let (input, net) = network_with_sensors(
            r#""sym_power_sensor":[{"id":11,"measured_object":5,"measured_terminal_type":4,
                 "p_measured":2000.0,"q_measured":500.0,
                 "p_sigma":10.0,"q_sigma":20.0}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).unwrap();
        let p = find(&ms, MeasurementKind::ActivePower, Target::Bus(1));
        let q = find(&ms, MeasurementKind::ReactivePower, Target::Bus(1));
        assert!((p.sigma - 1e-5).abs() < 1e-15, "p sigma={}", p.sigma);
        assert!((q.sigma - 2e-5).abs() < 1e-15, "q sigma={}", q.sigma);
    }

    /// A source uses the *generator* reference direction, so the same positive
    /// reading stays positive — the opposite of the load case above.
    ///
    /// It lands on [`Target::SourceInjection`] rather than on the source node's
    /// [`Target::Bus`], and that distinction is the point. gridoxide puts a
    /// virtual slack bus behind an impedance branch for every source, so by KCL
    /// the node's own injection excludes it entirely; treating a source sensor
    /// as a bus injection made the modelled value zero while the sensor read
    /// the full infeed (`tests/measurement_residual_test.rs` caught this at 63
    /// sigma).
    #[test]
    fn source_sensor_targets_the_source_branch_not_the_bus() {
        let (input, net) = network_with_sensors(
            r#""sym_power_sensor":[{"id":12,"measured_object":4,"measured_terminal_type":2,
                 "p_measured":2000.0,"q_measured":0.0,"power_sigma":10.0}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).unwrap();
        let branch = net.source_branch_idx[&4];
        let p = find(
            &ms,
            MeasurementKind::ActivePower,
            Target::SourceInjection { branch },
        );
        assert!((p.value - 0.002).abs() < 1e-12, "p={}", p.value);
        assert!(
            !ms.iter().any(|m| m.target == Target::Bus(0)),
            "a source sensor must not produce a bus-injection row: {ms:#?}"
        );
    }

    /// Branch sensors already use "into the branch" as positive, so they map
    /// straight through — and onto the branch index the id map resolves, not
    /// the sensor's own id.
    #[test]
    fn branch_sensor_maps_to_terminal_without_sign_change() {
        let (input, net) = network_with_sensors(
            r#""sym_power_sensor":[{"id":13,"measured_object":3,"measured_terminal_type":1,
                 "p_measured":-1500.0,"q_measured":0.0,"power_sigma":10.0}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).unwrap();
        let branch = net.branch_idx[&3];
        let target = Target::BranchTerminal { branch, terminal: Terminal::To };
        let p = find(&ms, MeasurementKind::ActivePower, target);
        assert!((p.value + 0.0015).abs() < 1e-12, "p={}", p.value);
    }

    /// Two sensors on the same appliance describe one quantity and merge;
    /// their combined reading is more certain than either.
    #[test]
    fn duplicate_appliance_sensors_merge_rather_than_double() {
        let (input, net) = network_with_sensors(
            r#""sym_power_sensor":[
                 {"id":14,"measured_object":5,"measured_terminal_type":4,
                  "p_measured":1000.0,"q_measured":0.0,"power_sigma":10.0},
                 {"id":15,"measured_object":5,"measured_terminal_type":4,
                  "p_measured":1000.0,"q_measured":0.0,"power_sigma":10.0}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).unwrap();
        let p = find(&ms, MeasurementKind::ActivePower, Target::Bus(1));
        // Two readings of the same appliance: the injection stays -0.001 p.u.
        // rather than summing to -0.002.
        assert!((p.value + 0.001).abs() < 1e-12, "p={}", p.value);
        assert!(p.sigma < 1e-5, "merged sigma should tighten: {}", p.sigma);
    }

    #[test]
    fn unknown_measured_object_is_an_error() {
        let (input, net) = network_with_sensors(
            r#""sym_voltage_sensor":[{"id":16,"measured_object":999,
                 "u_measured":10000.0,"u_sigma":100.0}]"#,
        );
        assert_eq!(
            measurements_from_pgm(&input, &net, 1e6),
            Err(MeasurementError::UnknownObject { sensor: 16, object: 999 })
        );
    }

    #[test]
    fn zero_sigma_is_rejected() {
        assert_eq!(
            checked_sigma(7, 0.0),
            Err(MeasurementError::InvalidSigma { sensor: 7, sigma: 0.0 })
        );
        assert!(checked_sigma(7, f64::NAN).is_err());
        assert!(checked_sigma(7, -1.0).is_err());
        assert_eq!(checked_sigma(7, 2.0), Ok(2.0));
    }
}
