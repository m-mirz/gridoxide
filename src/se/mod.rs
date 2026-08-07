//! State estimation: recovering the most likely grid state from noisy,
//! redundant, partial measurements.
//!
//! Power flow answers "given the injections, what are the voltages". State
//! estimation answers the operational question instead: given a set of
//! measurements that disagree with each other, none of them exact and not
//! enough of them to determine the state directly, what state best explains
//! them.
//!
//! This module holds the pieces both planned estimator methods share. The
//! measurement *set* comes from [`crate::measurement`]; what lives here is the
//! measurement *model* — the functions `h(x)` that say what each measurement
//! would read in a given state.
//!
//! # Why `h(x)` is not just "read it off the network"
//!
//! Three of power-grid-model's sensor types describe quantities that gridoxide
//! models structurally rather than as appliances, so their measurement
//! functions are not the bus injection despite PGM presenting them that way:
//!
//! | PGM sensor | PGM's model | gridoxide's model | `h(x)` here |
//! |---|---|---|---|
//! | `source` | appliance at the node | virtual slack bus behind an impedance branch | that branch's flow |
//! | `shunt` | appliance at the node | Y-bus diagonal entry | `-|V|²·conj(y_sh)` |
//! | `node` | sum of all appliances | — | bus injection **plus** the two above |
//!
//! Treating any of them as the plain bus injection is wrong by construction,
//! not by a sign: gridoxide's `power_injections` at a source node is zero by
//! KCL, because the source's power arrives through a branch that is part of the
//! network. `tests/measurement_residual_test.rs` is what established this — it
//! reported a 63-sigma disagreement on exactly that quantity.

pub mod bad_data;
pub mod batch;
pub mod constraints;
pub mod functional;
pub mod iterative;
pub mod jacobian;
pub mod observability;
pub mod nr;

use num_complex::Complex;

use crate::branch_flow::{branch_params, BranchParams};
use crate::measurement::{Measurement, MeasurementKind, Target};
use crate::network::YBusSparse;
use crate::types::Bus;

/// Everything `h(x)` needs about the network, beyond the state itself.
///
/// Assembled once per topology. The per-bus shunt admittance and the per-bus
/// list of source branches are kept explicitly because both are otherwise only
/// recoverable by inspecting Y-bus entries, which cannot distinguish a shunt
/// from a line's charging susceptance.
pub struct SeNetwork {
    pub ybus: YBusSparse,
    /// Per branch, the current functional at each terminal: `[0]` is the from
    /// side, `[1]` the to side.
    ///
    /// Stored resolved rather than as branch parameters, because a three-phase
    /// terminal is not a two-bus quantity: phase `p` of one end depends on all
    /// six phasors of the branch, through the 3×3 blocks. A coefficient list
    /// expresses both cases, and the estimator above it cannot tell them apart —
    /// which is what makes the phase domain a matter of longer lists rather than
    /// of a second Jacobian.
    ///
    /// In the phase domain the index is `3·branch + phase`, exactly as a bus
    /// index is `3·node + phase`. That is what keeps
    /// [`Target`](crate::measurement::Target) unchanged between the two.
    pub terminals: Vec<[functional::CurrentFunctional; 2]>,
    /// Total shunt admittance stamped at each bus, or zero.
    pub shunt_y: Vec<Complex<f64>>,
    /// For each bus, the synthesized source branches delivering into it.
    pub source_branches: Vec<Vec<usize>>,
    /// Per-bus zero-injection flags, carried through from the conversion — see
    /// [`PgmNetwork::zero_injection`](crate::pgm::PgmNetwork::zero_injection).
    pub zero_injection: Vec<bool>,
    /// Whether each bus is reachable from any source.
    ///
    /// A connected component containing no source carries no voltage that any
    /// measurement can determine: every quantity in it is identically zero, and
    /// the sensors that sit there — a network can perfectly well have a voltage
    /// sensor on a de-energized node — describe nothing. power-grid-model
    /// reports such nodes as `energized: 0` with a state of exactly zero, and
    /// decides it topologically, ignoring whatever sensors are attached.
    ///
    /// Without this the gain matrix comes back singular rather than merely
    /// uninformative, because those buses' columns are *touched* — by their own
    /// sensors, or by a zero-injection constraint — while being determined by
    /// nothing. `jacobian::mask_untouched` only pins what nothing touches at
    /// all, which catches a fully isolated bus and not this.
    ///
    /// `solver::PersistentSolver` has done the equivalent for power flow all
    /// along, via `network::connected_components` and `mark_unreferenced_islands`.
    pub energized: Vec<bool>,
}

/// One `[from, to]` functional pair per scalar branch.
fn terminals_from_params(params: &[BranchParams]) -> Vec<[functional::CurrentFunctional; 2]> {
    use crate::branch_flow::Terminal;
    params
        .iter()
        .map(|b| {
            [Terminal::From, Terminal::To].map(|t| {
                let (near, far) = b.buses(t);
                let (y_self, y_mut) = b.seen_from(t);
                functional::CurrentFunctional {
                    at: near,
                    coefficients: vec![(near, y_self), (far, y_mut)],
                }
            })
        })
        .collect()
}

impl SeNetwork {
    /// Builds the model from a converted network plus the shunt list that was
    /// stamped into its Y-bus.
    ///
    /// `shunts` must be the same slice passed to
    /// [`network::stamp_shunts`](crate::network::stamp_shunts) — the shunt
    /// measurement function needs the admittance itself, and the Y-bus has
    /// already summed it together with everything else on that diagonal.
    pub fn new(
        net: &crate::pgm::PgmNetwork,
        ybus: YBusSparse,
        shunts: &[crate::network::ShuntAdm],
    ) -> Self {
        let n = ybus.n();
        let mut shunt_y = vec![Complex::new(0.0, 0.0); n];
        for s in shunts {
            shunt_y[s.at] += s.y;
        }

        let terminals = terminals_from_params(&branch_params(&net.lines, &net.transformers));
        let mut source_branches = vec![Vec::new(); n];
        for (&_id, &branch) in &net.source_branch_idx {
            // The synthesized branch runs virtual -> node, so the node it feeds
            // is its `to` end.
            source_branches[terminals[branch][1].at].push(branch);
        }
        for list in &mut source_branches {
            list.sort_unstable();
        }

        // A component is energized if any bus in it is fed by a source. The
        // virtual slack bus behind a source is in that same component, since
        // the synthesized branch connects them.
        let mut energized = vec![false; n];
        for component in crate::network::connected_components(&ybus) {
            if component.iter().any(|&i| !source_branches[i].is_empty()) {
                for &i in &component {
                    energized[i] = true;
                }
            }
        }

        let zero_injection = net.zero_injection.clone();
        Self { ybus, terminals, shunt_y, source_branches, zero_injection, energized }
    }

    /// Builds the model for a three-phase network.
    ///
    /// Everything above this is unchanged by the phase domain. Bus `k` phase `p`
    /// is index `3k + p`, as `lib.rs` documents for power flow, and a branch
    /// terminal is indexed `3b + p` to match — so `Target`, `StateLayout`, the
    /// Jacobian, the constraints and both estimator methods carry over without
    /// knowing there are phases at all. What changes is the length of a
    /// coefficient list: six terms where the scalar case has two, because phase
    /// `p` of one end of a branch depends on all six phasors through the 3×3
    /// blocks.
    ///
    /// `lines`, `transformers` and `shunts` must be the same slices stamped into
    /// `ybus` — this reads their blocks a second time to build the measurement
    /// model, and a disagreement between the two would produce an estimate that
    /// converges confidently to the wrong answer.
    pub fn from_3ph(
        ybus: YBusSparse,
        lines: &[crate::types::Line3Ph],
        transformers: &[crate::types::Transformer3PhSeq],
        shunts: &[crate::network::ShuntAdm3Ph],
        source_branch_idx: &std::collections::HashMap<u64, usize>,
        zero_injection_per_node: &[bool],
    ) -> Self {
        let n = ybus.n();
        let n_nodes = n / 3;

        // One functional per (branch, phase), laid out `3·branch + phase` so the
        // index arithmetic matches the bus one.
        let mut terminals = Vec::with_capacity(3 * (lines.len() + transformers.len()));
        let blocks = lines
            .iter()
            .map(|ln| (ln.from, ln.to, crate::network::line3ph_blocks(ln)))
            .chain(
                transformers
                    .iter()
                    .map(|t| (t.from, t.to, crate::network::transformer3ph_blocks(t))),
            );
        for (from, to, [yff, yft, ytf, ytt]) in blocks {
            for p in 0..3 {
                // From side: `I_p = Σ_q Yff[p][q]·V_from,q + Σ_q Yft[p][q]·V_to,q`.
                let coeffs = |near_block: &[[Complex<f64>; 3]; 3],
                              far_block: &[[Complex<f64>; 3]; 3],
                              near: usize,
                              far: usize| {
                    (0..3)
                        .map(|q| (3 * near + q, near_block[p][q]))
                        .chain((0..3).map(|q| (3 * far + q, far_block[p][q])))
                        .collect::<Vec<_>>()
                };
                terminals.push([
                    functional::CurrentFunctional {
                        at: 3 * from + p,
                        coefficients: coeffs(&yff, &yft, from, to),
                    },
                    functional::CurrentFunctional {
                        at: 3 * to + p,
                        coefficients: coeffs(&ytt, &ytf, to, from),
                    },
                ]);
            }
        }

        let mut shunt_y = vec![Complex::new(0.0, 0.0); n];
        for s in shunts {
            // A three-phase shunt is a 3×3 block, but `shunt_y` is the *diagonal*
            // a shunt-injection measurement reads. The off-diagonal coupling is
            // in the Y-bus, where the bus-injection functional already picks it
            // up; only the standalone shunt target uses this.
            let m = crate::network::seq_to_phase_shunt(s.y1, s.y0);
            for p in 0..3 {
                shunt_y[3 * s.at + p] += m[p][p];
            }
        }

        let mut source_branches = vec![Vec::new(); n];
        for &branch in source_branch_idx.values() {
            for p in 0..3 {
                let phase_branch = 3 * branch + p;
                source_branches[terminals[phase_branch][1].at].push(phase_branch);
            }
        }
        for list in &mut source_branches {
            list.sort_unstable();
        }

        // A node's zero-injection flag applies to all three of its phases: an
        // appliance-free node injects nothing on any of them.
        let mut zero_injection = vec![false; n];
        for (k, &zero) in zero_injection_per_node.iter().enumerate().take(n_nodes) {
            for p in 0..3 {
                zero_injection[3 * k + p] = zero;
            }
        }

        let mut energized = vec![false; n];
        for component in crate::network::connected_components(&ybus) {
            if component.iter().any(|&i| !source_branches[i].is_empty()) {
                for &i in &component {
                    energized[i] = true;
                }
            }
        }

        Self { ybus, terminals, shunt_y, source_branches, zero_injection, energized }
    }

    /// Buses a zero-injection constraint should apply to.
    ///
    /// A de-energized bus's injection is zero too, but asserting it adds a row
    /// that reaches columns nothing determines — which is how a constraint turns
    /// an uninformative island into a singular matrix.
    pub fn constrained_buses(&self) -> Vec<bool> {
        self.zero_injection
            .iter()
            .zip(&self.energized)
            .map(|(&zero, &live)| zero && live)
            .collect()
    }

}

/// Evaluates `h(x)` for every measurement in `measurements`, in order.
///
/// `buses` carries the state; everything else comes from `net`. Two arms, not
/// five: a voltage row reads the state directly, and every power row is its
/// target's [`CurrentFunctional`](functional::CurrentFunctional) evaluated at
/// the state.
pub fn measurement_functions(
    measurements: &[Measurement],
    buses: &[Bus],
    net: &SeNetwork,
) -> Vec<f64> {
    let model = MeasurementModel::new(measurements, net);
    measurement_functions_with(measurements, buses, &model)
}

/// [`measurement_functions`] against an already-resolved model.
///
/// Resolving a target into a functional allocates a coefficient list, and the
/// estimator evaluates `h(x)` once per iteration over an unchanging measurement
/// set — so the resolution belongs outside the loop. `PersistentEstimator`
/// caches one of these next to the state layout.
pub fn measurement_functions_with(
    measurements: &[Measurement],
    buses: &[Bus],
    model: &MeasurementModel,
) -> Vec<f64> {
    let v: Vec<Complex<f64>> = buses
        .iter()
        .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
        .collect();

    measurements
        .iter()
        .enumerate()
        .map(|(i, m)| match (m.kind, m.target) {
            _ if !model.is_live(i) => 0.0,
            (MeasurementKind::VoltageMagnitude, Target::Bus(b)) => buses[b].voltage_mag,
            (MeasurementKind::VoltageAngle, Target::Bus(b)) => buses[b].voltage_ang,
            _ => match model.functional(i) {
                Some(f) => {
                    // Which quantity of the functional this row reads: the
                    // power, the current itself, or the current in the
                    // terminal's own voltage frame.
                    let value = match m.target {
                        Target::BranchTerminalCurrent { frame, .. } => match frame {
                            crate::measurement::AngleFrame::Global => f.current(&v),
                            crate::measurement::AngleFrame::Local => f.local_current(&v),
                        },
                        _ => f.power(&v),
                    };
                    match m.kind {
                        MeasurementKind::ActivePower | MeasurementKind::CurrentReal => value.re,
                        _ => value.im,
                    }
                }
                None => 0.0,
            },
        })
        .collect()
}

/// One resolved [`CurrentFunctional`] per measurement, in input order.
///
/// `None` for a voltage row, which reads the state directly and has no
/// functional, and for a target that does not resolve in this network.
pub struct MeasurementModel {
    functionals: Vec<Option<functional::CurrentFunctional>>,
    /// Whether each measurement describes anything at all.
    ///
    /// False for a measurement whose bus is de-energized. Such a row is dropped
    /// from the system entirely rather than given a zero weight: a zero-weight
    /// row still *touches* its columns, and touching is what stops
    /// [`jacobian::mask_untouched`] pinning them. power-grid-model does the same
    /// by construction, since a de-energized node is not in its math model at
    /// all.
    live: Vec<bool>,
}

impl MeasurementModel {
    /// Resolves every measurement's target once.
    ///
    /// Valid for as long as the network and the measurement set's *structure*
    /// are unchanged — the same condition
    /// [`PersistentEstimator`](nr::PersistentEstimator) already states for its
    /// cached factorization, since both depend on which quantities are measured
    /// where and neither on their values.
    pub fn new(measurements: &[Measurement], net: &SeNetwork) -> Self {
        let functionals: Vec<Option<functional::CurrentFunctional>> = measurements
            .iter()
            .map(|m| match (m.kind, m.target) {
                (MeasurementKind::VoltageMagnitude | MeasurementKind::VoltageAngle, Target::Bus(_)) => None,
                _ => net.functional(m.target),
            })
            .collect();

        // Which bus a measurement lives on: its own for a voltage row, the
        // functional's `at` otherwise. Both ends of a branch are in one
        // component, so either end answers the question.
        let live = measurements
            .iter()
            .zip(&functionals)
            .map(|(m, f)| match (m.target, f) {
                (Target::Bus(b), None) => net.energized[b],
                (_, Some(f)) => net.energized[f.at],
                (_, None) => false,
            })
            .collect();

        Self { functionals, live }
    }

    /// Whether measurement `i` describes anything — see [`live`](Self::live).
    pub fn is_live(&self, measurement: usize) -> bool {
        self.live[measurement]
    }

    pub fn functional(&self, measurement: usize) -> Option<&functional::CurrentFunctional> {
        self.functionals[measurement].as_ref()
    }

    pub fn len(&self) -> usize {
        self.functionals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.functionals.is_empty()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::branch_flow::{terminal_flow, Terminal};
    use crate::network::YBus;
    use crate::types::BusType;

    fn bus(idx: usize, mag: f64, ang: f64) -> Bus {
        Bus {
            idx,
            bus_type: BusType::PQ,
            voltage_mag: mag,
            voltage_ang: ang,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: f64::NEG_INFINITY,
            q_max: f64::INFINITY,
            u_rated: 1.0,
            zip_terms: Vec::new(),
        }
    }

    /// Two buses joined by one branch, with a shunt on bus 1 and the branch
    /// treated as a source feeding bus 1 — enough to exercise all three of the
    /// structural measurement functions against one state.
    /// The scalar branch `two_bus_net` is built from.
    ///
    /// Exposed so the functional tests can check themselves against
    /// `branch_flow`'s own closed forms, which is a genuinely independent
    /// derivation rather than a restatement of the one under test.
    pub(crate) fn two_bus_params() -> BranchParams {
        let y_series = Complex::new(1.0, 0.0) / Complex::new(0.05, 0.2);
        BranchParams { from: 0, to: 1, y: [y_series, -y_series, -y_series, y_series] }
    }

    pub(crate) fn two_bus_net() -> (SeNetwork, Vec<Bus>) {
        let y_series = Complex::new(1.0, 0.0) / Complex::new(0.05, 0.2);
        let y_shunt = Complex::new(0.02, 0.3);

        let mut yb = YBus::new(2);
        yb.add(0, 0, y_series);
        yb.add(1, 1, y_series);
        yb.add(0, 1, -y_series);
        yb.add(1, 0, -y_series);
        yb.add(1, 1, y_shunt);

        let net = SeNetwork {
            ybus: yb.finish(),
            terminals: terminals_from_params(&[two_bus_params()]),
            shunt_y: vec![Complex::new(0.0, 0.0), y_shunt],
            source_branches: vec![Vec::new(), vec![0]],
            zero_injection: vec![false, false],
            // One component, fed by the source on bus 1.
            energized: vec![true, true],
        };
        (net, vec![bus(0, 1.02, 0.03), bus(1, 0.99, -0.02)])
    }

    /// Energization is topological: a component with a source is live, one
    /// without is not, whatever sensors sit on it.
    #[test]
    fn energization_follows_components_not_sensors() {
        // Three buses: 0–1 joined and fed by a source on 1, bus 2 isolated.
        let y = Complex::new(1.0, 0.0) / Complex::new(0.05, 0.2);
        let mut yb = YBus::new(3);
        for (i, j, v) in [(0, 0, y), (1, 1, y), (0, 1, -y), (1, 0, -y)] {
            yb.add(i, j, v);
        }
        // An isolated bus still needs a diagonal to appear in the Y-bus at all.
        yb.add(2, 2, Complex::new(0.0, 0.0));

        let net = SeNetwork {
            ybus: yb.finish(),
            terminals: terminals_from_params(&[BranchParams { from: 0, to: 1, y: [y, -y, -y, y] }]),
            shunt_y: vec![Complex::new(0.0, 0.0); 3],
            source_branches: vec![Vec::new(), vec![0], Vec::new()],
            zero_injection: vec![true, false, true],
            energized: vec![true, true, false],
        };

        let rebuilt: Vec<bool> = crate::network::connected_components(&net.ybus)
            .iter()
            .fold(vec![false; 3], |mut acc, c| {
                if c.iter().any(|&i| !net.source_branches[i].is_empty()) {
                    for &i in c {
                        acc[i] = true;
                    }
                }
                acc
            });
        assert_eq!(rebuilt, vec![true, true, false], "bus 2 has no source in its component");

        // And a zero-injection flag on the de-energized bus does not become a
        // constraint, which is what used to make the gain matrix singular.
        assert_eq!(net.constrained_buses(), vec![true, false, false]);
    }

    /// A shunt *consumes* `|V|²·conj(y)`, so as an injection its active part is
    /// negative for a positive conductance and its reactive part is positive
    /// for a positive susceptance. Sign errors here are invisible in a
    /// magnitude check, so both are asserted.
    #[test]
    fn shunt_injection_is_negated_consumption() {
        let (net, buses) = two_bus_net();
        let v: Vec<Complex<f64>> = buses
            .iter()
            .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
            .collect();
        let s = net
            .functional(Target::ShuntInjection { bus: 1 })
            .expect("shunt functional")
            .power(&v);
        let (p, q) = (s.re, s.im);
        let v2 = 0.99_f64 * 0.99;
        assert!((p + v2 * 0.02).abs() < 1e-12, "p={p}");
        assert!((q - v2 * 0.3).abs() < 1e-12, "q={q}");
        assert!(p < 0.0, "a resistive shunt must consume active power");
    }

    /// A source delivers into its node, which is the negation of the flow the
    /// node pushes into the synthesized branch.
    #[test]
    fn source_injection_is_negated_to_terminal_flow() {
        let (net, buses) = two_bus_net();
        let v: Vec<Complex<f64>> = buses
            .iter()
            .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
            .collect();
        let (p_to, q_to) = terminal_flow(&two_bus_params(), Terminal::To, &v);
        let s_src = net
            .functional(Target::SourceInjection { branch: 0 })
            .expect("source functional")
            .power(&v);
        let (p_src, q_src) = (s_src.re, s_src.im);
        assert!((p_src + p_to).abs() < 1e-12);
        assert!((q_src + q_to).abs() < 1e-12);
    }

    /// The identity that makes `NodeInjection` a separate measurement function
    /// rather than a relabelling of `Bus`: it is the bus injection plus the
    /// contributions gridoxide models structurally.
    #[test]
    fn node_injection_is_bus_plus_source_plus_shunt() {
        let (net, buses) = two_bus_net();
        let ms = |target| Measurement {
            kind: MeasurementKind::ActivePower,
            target,
            value: 0.0,
            sigma: 1.0,
        };
        let measurements = vec![ms(Target::Bus(1)), ms(Target::NodeInjection(1))];
        let h = measurement_functions(&measurements, &buses, &net);

        let v: Vec<Complex<f64>> = buses
            .iter()
            .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
            .collect();
        let power = |t| net.functional(t).expect("functional").power(&v).re;
        let expected = h[0]
            + power(Target::SourceInjection { branch: 0 })
            + power(Target::ShuntInjection { bus: 1 });
        assert!((h[1] - expected).abs() < 1e-12, "node={} expected={expected}", h[1]);
        // And the two are genuinely different quantities, so a test that
        // confused them would not pass by coincidence.
        assert!((h[1] - h[0]).abs() > 1e-6, "bus and node injection should differ here");
    }
}
