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
}

/// What a [`Measurement`] is attached to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// A bus: a voltage measurement, or a net injection.
    Bus(usize),
    /// One terminal of one branch, indexed as in
    /// [`branch_flow::branch_params`](crate::branch_flow::branch_params).
    BranchTerminal { branch: usize, terminal: Terminal },
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
    p: Merged,
    q: Merged,
}

/// Everything a bus can accumulate before becoming [`Measurement`]s.
#[derive(Clone, Copy, Debug, Default)]
struct BusAccumulator {
    u_mag: Merged,
    u_angle: Merged,
    /// Injection built up from individual appliance sensors.
    appliance_p: Summed,
    appliance_q: Summed,
    /// Injection measured directly by node-injection sensors.
    node_p: Merged,
    node_q: Merged,
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
    let mut buses: HashMap<usize, BusAccumulator> = HashMap::new();
    // Branch-terminal readings merge per (branch, terminal).
    let mut branch_p: HashMap<(usize, Terminal), Merged> = HashMap::new();
    let mut branch_q: HashMap<(usize, Terminal), Merged> = HashMap::new();
    // Appliance readings are merged *per appliance* before the per-bus sum:
    // two sensors on one load are one better-known load, not two loads. Doing
    // this in one step would double-count the appliance.
    let mut appliances: HashMap<u64, ApplianceAccumulator> = HashMap::new();

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
        let acc = buses.entry(bus).or_default();

        if s.u_measured.is_finite() {
            let sigma = checked_sigma(s.id, s.u_sigma)?;
            acc.u_mag.add(s.u_measured / u_rated, sigma / u_rated);
        }
        if s.u_angle_measured.is_finite() {
            // An angle sigma is optional even when an angle is given; PGM
            // reuses u_sigma's relative size in that case, which at |U| ≈ 1 p.u.
            // is the same number in radians.
            let sigma = if s.u_angle_sigma.is_finite() {
                checked_sigma(s.id, s.u_angle_sigma)?
            } else {
                checked_sigma(s.id, s.u_sigma)? / u_rated
            };
            acc.u_angle.add(s.u_angle_measured, sigma);
        }
    }

    // ── Power sensors ────────────────────────────────────────────────────
    for s in &input.data.sym_power_sensor {
        if !s.p_measured.is_finite() && !s.q_measured.is_finite() {
            continue;
        }
        // Per-component sigmas win over the shared one when present.
        let p_sigma = if s.p_sigma.is_finite() { s.p_sigma } else { s.power_sigma };
        let q_sigma = if s.q_sigma.is_finite() { s.q_sigma } else { s.power_sigma };
        let p_sigma = checked_sigma(s.id, p_sigma)? / s_base_va;
        let q_sigma = checked_sigma(s.id, q_sigma)? / s_base_va;
        let p = s.p_measured / s_base_va;
        let q = s.q_measured / s_base_va;

        match s.measured_terminal_type {
            // Branch terminals: same direction convention as `terminal_flow`.
            0 | 1 => {
                let terminal = if s.measured_terminal_type == 0 { Terminal::From } else { Terminal::To };
                let Some((branch, terminal)) = net.resolve_terminal(s.measured_object, terminal) else {
                    // The branch is absent or this is its open end. PGM reports
                    // zero flow there, so the reading carries no information
                    // about the state and is dropped rather than fought with.
                    continue;
                };
                if s.p_measured.is_finite() {
                    branch_p.entry((branch, terminal)).or_default().add(p, p_sigma);
                }
                if s.q_measured.is_finite() {
                    branch_q.entry((branch, terminal)).or_default().add(q, q_sigma);
                }
            }
            // Appliances. `sign` converts the appliance's own reference
            // direction into a bus injection.
            2 | 3 | 4 | 5 => {
                let sign = match s.measured_terminal_type {
                    // load, shunt: load reference direction — consumption.
                    3 | 4 => -1.0,
                    // source, generator: generator reference direction.
                    _ => 1.0,
                };
                let Some(&bus) = net.appliance_bus.get(&s.measured_object) else {
                    return Err(MeasurementError::UnknownObject {
                        sensor: s.id,
                        object: s.measured_object,
                    });
                };
                let acc = appliances.entry(s.measured_object).or_insert(ApplianceAccumulator {
                    bus,
                    sign,
                    p: Merged::default(),
                    q: Merged::default(),
                });
                if s.p_measured.is_finite() {
                    acc.p.add(p, p_sigma);
                }
                if s.q_measured.is_finite() {
                    acc.q.add(q, q_sigma);
                }
            }
            // Node injection: already a bus injection, generator direction.
            9 => {
                let Some(&bus) = net.node_idx.get(&s.measured_object) else {
                    return Err(MeasurementError::UnknownObject {
                        sensor: s.id,
                        object: s.measured_object,
                    });
                };
                let acc = buses.entry(bus).or_default();
                if s.p_measured.is_finite() {
                    acc.node_p.add(p, p_sigma);
                }
                if s.q_measured.is_finite() {
                    acc.node_q.add(q, q_sigma);
                }
            }
            // 6/7/8 are the three-winding transformer sides. gridoxide models
            // one of those as three separate two-winding branches to a star
            // bus, so the mapping exists — but it is not exercised by any
            // symmetric fixture yet, and guessing at it is worse than saying so.
            other => {
                return Err(MeasurementError::UnsupportedTerminalType {
                    sensor: s.id,
                    terminal_type: other,
                })
            }
        }
    }

    // Each appliance contributes its merged reading, sign-corrected, to its
    // bus's injection sum. Iterated in id order so the summed variance is
    // built deterministically.
    let mut appliance_ids: Vec<u64> = appliances.keys().copied().collect();
    appliance_ids.sort_unstable();
    for id in appliance_ids {
        let acc = appliances[&id];
        let bus = buses.entry(acc.bus).or_default();
        if let Some((value, sigma)) = acc.p.finish() {
            bus.appliance_p.add(acc.sign * value, sigma);
        }
        if let Some((value, sigma)) = acc.q.finish() {
            bus.appliance_q.add(acc.sign * value, sigma);
        }
    }

    // ── Flatten the accumulators into measurement rows ───────────────────
    let mut out = Vec::new();

    let mut bus_ids: Vec<usize> = buses.keys().copied().collect();
    bus_ids.sort_unstable();
    for bus in bus_ids {
        let acc = buses[&bus];
        let target = Target::Bus(bus);
        if let Some((value, sigma)) = acc.u_mag.finish() {
            out.push(Measurement { kind: MeasurementKind::VoltageMagnitude, target, value, sigma });
        }
        if let Some((value, sigma)) = acc.u_angle.finish() {
            out.push(Measurement { kind: MeasurementKind::VoltageAngle, target, value, sigma });
        }
        // Appliance-derived and node-derived injections describe the same
        // quantity by different routes, so when a bus has both they merge by
        // inverse variance, same as any other duplicate observation.
        for (kind, appliance, node) in [
            (MeasurementKind::ActivePower, acc.appliance_p.finish(), acc.node_p.finish()),
            (MeasurementKind::ReactivePower, acc.appliance_q.finish(), acc.node_q.finish()),
        ] {
            let mut merged = Merged::default();
            for (value, sigma) in [appliance, node].into_iter().flatten() {
                merged.add(value, sigma);
            }
            if let Some((value, sigma)) = merged.finish() {
                out.push(Measurement { kind, target, value, sigma });
            }
        }
    }

    let mut branch_keys: Vec<(usize, Terminal)> =
        branch_p.keys().chain(branch_q.keys()).copied().collect();
    branch_keys.sort_unstable_by_key(|&(b, t)| (b, t == Terminal::To));
    branch_keys.dedup();
    for key in branch_keys {
        let (branch, terminal) = key;
        let target = Target::BranchTerminal { branch, terminal };
        if let Some((value, sigma)) = branch_p.get(&key).copied().unwrap_or_default().finish() {
            out.push(Measurement { kind: MeasurementKind::ActivePower, target, value, sigma });
        }
        if let Some((value, sigma)) = branch_q.get(&key).copied().unwrap_or_default().finish() {
            out.push(Measurement { kind: MeasurementKind::ReactivePower, target, value, sigma });
        }
    }

    Ok(out)
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
        assert!((p.sigma - 1e-5).abs() < 1e-15, "sigma={}", p.sigma);
    }

    /// A source uses the *generator* reference direction, so the same positive
    /// reading is a positive injection — the opposite of the load case above.
    #[test]
    fn source_sensor_stays_positive_injection() {
        let (input, net) = network_with_sensors(
            r#""sym_power_sensor":[{"id":12,"measured_object":4,"measured_terminal_type":2,
                 "p_measured":2000.0,"q_measured":0.0,"power_sigma":10.0}]"#,
        );
        let ms = measurements_from_pgm(&input, &net, 1e6).unwrap();
        let p = find(&ms, MeasurementKind::ActivePower, Target::Bus(0));
        assert!((p.value - 0.002).abs() < 1e-12, "p={}", p.value);
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
