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

use num_complex::Complex;

use crate::branch_flow::{branch_params, terminal_flow, BranchParams, Terminal};
use crate::measurement::{Measurement, MeasurementKind, Target};
use crate::network::{power_injections, YBusSparse};
use crate::types::Bus;

/// Everything `h(x)` needs about the network, beyond the state itself.
///
/// Assembled once per topology. The per-bus shunt admittance and the per-bus
/// list of source branches are kept explicitly because both are otherwise only
/// recoverable by inspecting Y-bus entries, which cannot distinguish a shunt
/// from a line's charging susceptance.
pub struct SeNetwork {
    pub ybus: YBusSparse,
    pub branches: Vec<BranchParams>,
    /// Total shunt admittance stamped at each bus, or zero.
    pub shunt_y: Vec<Complex<f64>>,
    /// For each bus, the synthesized source branches delivering into it.
    pub source_branches: Vec<Vec<usize>>,
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

        let branches = branch_params(&net.lines, &net.transformers);
        let mut source_branches = vec![Vec::new(); n];
        for (&_id, &branch) in &net.source_branch_idx {
            // The synthesized branch runs virtual -> node, so the node it feeds
            // is its `to` end.
            source_branches[branches[branch].to].push(branch);
        }
        for list in &mut source_branches {
            list.sort_unstable();
        }

        Self { ybus, branches, shunt_y, source_branches }
    }

    /// The power a source delivers *into* its node, per-unit.
    ///
    /// The synthesized branch runs virtual bus -> node, so the flow gridoxide
    /// computes at the branch's `to` terminal is power leaving the node into
    /// the branch. What the source delivers is its negation.
    pub fn source_injection(&self, branch: usize, v: &[Complex<f64>]) -> (f64, f64) {
        let (p, q) = terminal_flow(&self.branches[branch], Terminal::To, v);
        (-p, -q)
    }

    /// A bus's shunt injection, per-unit.
    ///
    /// A shunt consumes `S = V·conj(y_sh·V) = |V|²·conj(y_sh)`; as an injection
    /// that is negated, giving `(-|V|²g, +|V|²b)`.
    pub fn shunt_injection(&self, bus: usize, v: &[Complex<f64>]) -> (f64, f64) {
        let y = self.shunt_y[bus];
        let v2 = v[bus].norm_sqr();
        (-v2 * y.re, v2 * y.im)
    }
}

/// Evaluates `h(x)` for every measurement in `measurements`, in order.
///
/// `buses` carries the state (magnitudes and angles); everything else is read
/// from `net`. Bus injections are computed once for the whole set rather than
/// per measurement, since [`power_injections`] is a full sparse matrix-vector
/// product.
pub fn measurement_functions(
    measurements: &[Measurement],
    buses: &[Bus],
    net: &SeNetwork,
) -> Vec<f64> {
    let v: Vec<Complex<f64>> = buses
        .iter()
        .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
        .collect();
    let (p_inj, q_inj) = power_injections(buses, &net.ybus);

    measurements
        .iter()
        .map(|m| {
            let active = m.kind == MeasurementKind::ActivePower;
            match m.target {
                Target::Bus(b) => match m.kind {
                    MeasurementKind::VoltageMagnitude => buses[b].voltage_mag,
                    MeasurementKind::VoltageAngle => buses[b].voltage_ang,
                    MeasurementKind::ActivePower => p_inj[b],
                    MeasurementKind::ReactivePower => q_inj[b],
                },
                Target::BranchTerminal { branch, terminal } => {
                    let (p, q) = terminal_flow(&net.branches[branch], terminal, &v);
                    if active { p } else { q }
                }
                Target::SourceInjection { branch } => {
                    let (p, q) = net.source_injection(branch, &v);
                    if active { p } else { q }
                }
                Target::ShuntInjection { bus } => {
                    let (p, q) = net.shunt_injection(bus, &v);
                    if active { p } else { q }
                }
                // power-grid-model's node injection is the sum over every
                // appliance at the bus. gridoxide's bus injection covers the
                // loads and generators; the source and shunt contributions are
                // structural and have to be added back.
                Target::NodeInjection(bus) => {
                    let (mut p, mut q) = (p_inj[bus], q_inj[bus]);
                    for &branch in &net.source_branches[bus] {
                        let (sp, sq) = net.source_injection(branch, &v);
                        p += sp;
                        q += sq;
                    }
                    let (shp, shq) = net.shunt_injection(bus, &v);
                    p += shp;
                    q += shq;
                    if active { p } else { q }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn two_bus_net() -> (SeNetwork, Vec<Bus>) {
        let y_series = Complex::new(1.0, 0.0) / Complex::new(0.05, 0.2);
        let y_shunt = Complex::new(0.02, 0.3);

        let mut yb = YBus::new(2);
        yb.add(0, 0, y_series);
        yb.add(1, 1, y_series);
        yb.add(0, 1, -y_series);
        yb.add(1, 0, -y_series);
        yb.add(1, 1, y_shunt);

        let branches = vec![BranchParams {
            from: 0,
            to: 1,
            y: [y_series, -y_series, -y_series, y_series],
        }];

        let net = SeNetwork {
            ybus: yb.finish(),
            branches,
            shunt_y: vec![Complex::new(0.0, 0.0), y_shunt],
            source_branches: vec![Vec::new(), vec![0]],
        };
        (net, vec![bus(0, 1.02, 0.03), bus(1, 0.99, -0.02)])
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
        let (p, q) = net.shunt_injection(1, &v);
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
        let (p_to, q_to) = terminal_flow(&net.branches[0], Terminal::To, &v);
        let (p_src, q_src) = net.source_injection(0, &v);
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
        let expected = h[0] + net.source_injection(0, &v).0 + net.shunt_injection(1, &v).0;
        assert!((h[1] - expected).abs() < 1e-12, "node={} expected={expected}", h[1]);
        // And the two are genuinely different quantities, so a test that
        // confused them would not pass by coincidence.
        assert!((h[1] - h[0]).abs() > 1e-6, "bus and node injection should differ here");
    }
}
