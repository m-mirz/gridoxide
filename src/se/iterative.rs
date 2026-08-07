//! Iterative-linear state estimation: power-grid-model's default method.
//!
//! [`super::nr`] solves the estimation problem as it really is — nonlinear —
//! and pays for a fresh Jacobian and factorization every iteration. This method
//! makes the problem *linear* by a change of variables, factorizes once, and
//! then only moves the right-hand side.
//!
//! # The trick
//!
//! In terms of the complex voltage vector \\(\underline{U}\\), the awkward part
//! of the measurement model is that power is a *bilinear* function of voltage:
//! `S = U · conj(I)`. But current is perfectly linear in voltage — `I = Y U` —
//! so if the measurements were currents rather than powers, the whole problem
//! would be one linear least-squares solve.
//!
//! So they are converted. A power measurement `S` at a terminal whose voltage
//! is `U` becomes the current measurement `I = conj(S / U)`, using the voltage
//! from the previous iterate. A magnitude-only voltage measurement becomes a
//! phasor by borrowing the previous iterate's angle. Each iteration re-does that
//! conversion with better voltages, and the linearization error vanishes as the
//! angles settle.
//!
//! # Why it is faster, and what it costs
//!
//! The system matrix is built from admittances and measurement weights, neither
//! of which changes between iterations — *provided* the weights are held fixed.
//! They are not exactly constant: converting `S` to `I` divides by `U`, so the
//! current's variance depends on `|U|`. power-grid-model assumes `|U|` constant
//! for this purpose and says so plainly; the resulting matrix is constant and
//! is factorized exactly once, which is where the speed comes from.
//!
//! The cost is accuracy. That assumption, plus the interpretation of power
//! measurements as current measurements, means this method's optimum is not
//! quite the same as the true weighted-least-squares optimum. power-grid-model
//! documents the same caveat and recommends its Newton-Raphson method when
//! precision matters. gridoxide keeps both for the same reason.
//!
//! # Zero-injection buses
//!
//! These become *linear* equality constraints here — a bus with no appliance
//! carries no injected current, so `(Y U)_i = 0` exactly — which is markedly
//! simpler than the nonlinear case [`super::constraints`] handles. They enter
//! the same augmented KKT system, in complex arithmetic.

use num_complex::Complex;

use crate::measurement::{AngleFrame, Measurement, MeasurementKind, Target};
use crate::sparse::ComplexSparseSystem;
use crate::types::Bus;

use super::nr::{SeOptions, SeReport, SeStatus};
use super::SeNetwork;

/// What a linear row measures, and how its measured value becomes the complex
/// current the row's equation is written in.
///
/// Every row states `Σ c_k·U_k = I`, so each variant's job is to produce that
/// `I` from what the sensor actually read, using the previous iterate where the
/// conversion needs a voltage.
#[derive(Clone, Copy, Debug)]
enum RowKind {
    /// A bus voltage. Magnitude-only rows borrow the previous iterate's angle.
    Voltage { bus: usize, magnitude: f64, angle: Option<f64> },
    /// A power, converted as `I = conj(S/U_at)`.
    ///
    /// The only variant whose weight is scaled by `|U_at|²`: dividing by `U`
    /// scales the current's variance by `1/|U|²`, so the weight scales the other
    /// way. A current measurement involves no such division and is *not* scaled
    /// — power-grid-model leaves the variance alone there too.
    Power(Complex<f64>),
    /// A current already in the global frame, used as-is.
    GlobalCurrent(Complex<f64>),
    /// A current in its terminal's own voltage frame, converted as
    /// `I = conj(I_local)·U_at/|U_at|`.
    LocalCurrent(Complex<f64>),
}

/// A measurement rewritten as a linear function of the complex voltages.
struct LinearRow {
    /// Nonzero coefficients: `Σ coeff·U` is the quantity measured.
    coefficients: Vec<(usize, Complex<f64>)>,
    kind: RowKind,
    /// Bus whose voltage converts the reading into a global current.
    reference_bus: usize,
    /// `1/σ²` in the converted units.
    weight: f64,
}

impl RowKind {
    /// The complex current this row's equation is set equal to, at `v`.
    ///
    /// `None` when the conversion needs a voltage that has collapsed to zero,
    /// which leaves the row out of that pass rather than dividing by it.
    fn measured(&self, v: &[Complex<f64>], reference_bus: usize) -> Option<Complex<f64>> {
        match *self {
            RowKind::Voltage { bus, magnitude, angle } => {
                Some(Complex::from_polar(magnitude, angle.unwrap_or_else(|| v[bus].arg())))
            }
            RowKind::Power(s) => {
                let u = v[reference_bus];
                (u.norm() > 0.0).then(|| (s / u).conj())
            }
            RowKind::GlobalCurrent(i) => Some(i),
            RowKind::LocalCurrent(i) => {
                let u = v[reference_bus];
                (u.norm() > 0.0).then(|| i.conj() * u / u.norm())
            }
        }
    }

    /// Factor applied to this row's weight when the matrix is built.
    fn weight_scale(&self, buses: &[Bus], reference_bus: usize) -> f64 {
        match self {
            RowKind::Power(_) => buses[reference_bus].voltage_mag.powi(2),
            _ => 1.0,
        }
    }
}

/// Pairs the scalar measurements into complex linear rows.
///
/// `P` and `Q` on one target become a single complex power measurement, since
/// only the pair determines a current; a magnitude and an angle on one bus
/// become a single phasor. A lone `P` or `Q` is dropped — half a complex power
/// cannot be converted to a current, which is a real limitation of this method
/// rather than of the implementation.
fn build_rows(measurements: &[Measurement], net: &SeNetwork) -> Vec<LinearRow> {
    use std::collections::HashMap;

    let mut by_target: HashMap<Target, Vec<usize>> = HashMap::new();
    let mut order: Vec<Target> = Vec::new();
    for (i, m) in measurements.iter().enumerate() {
        let entry = by_target.entry(m.target).or_default();
        if entry.is_empty() {
            order.push(m.target);
        }
        entry.push(i);
    }

    let mut rows = Vec::new();
    for target in order {
        let indices = &by_target[&target];
        let find = |kind: MeasurementKind| {
            indices
                .iter()
                .copied()
                .find(|&i| measurements[i].kind == kind && measurements[i].weight() > 0.0)
        };

        let magnitude = find(MeasurementKind::VoltageMagnitude);
        let angle = find(MeasurementKind::VoltageAngle);
        if let (Some(mag), Target::Bus(bus)) = (magnitude, target) {
            let m = &measurements[mag];
            rows.push(LinearRow {
                coefficients: vec![(bus, Complex::new(1.0, 0.0))],
                kind: RowKind::Voltage {
                    bus,
                    magnitude: m.value,
                    angle: angle.map(|a| measurements[a].value),
                },
                reference_bus: bus,
                weight: m.weight(),
            });
        }

        // A power pairs P with Q; a current sensor pairs its two components.
        // Either way only the pair determines a current, so a lone half is
        // dropped — a real limitation of this method rather than of the code.
        let (real, imag) = match target {
            Target::BranchTerminalCurrent { .. } => (
                find(MeasurementKind::CurrentReal),
                find(MeasurementKind::CurrentImag),
            ),
            _ => (
                find(MeasurementKind::ActivePower),
                find(MeasurementKind::ReactivePower),
            ),
        };
        let (Some(p), Some(q)) = (real, imag) else {
            continue;
        };
        // One description of what this target measures, shared with the Newton
        // path's Jacobian — including which bus's voltage converts the power
        // into a current, which used to be restated as its own match here.
        let Some(functional) = net.functional(target) else {
            continue;
        };
        // The apparent-power measurement's variance is the sum of the two
        // components', per power-grid-model's aggregation rule.
        let variance = measurements[p].sigma.powi(2) + measurements[q].sigma.powi(2);
        if variance <= 0.0 || !variance.is_finite() {
            continue;
        }
        let value = Complex::new(measurements[p].value, measurements[q].value);
        rows.push(LinearRow {
            coefficients: functional.coefficients,
            kind: match target {
                Target::BranchTerminalCurrent { frame: AngleFrame::Global, .. } => {
                    RowKind::GlobalCurrent(value)
                }
                Target::BranchTerminalCurrent { frame: AngleFrame::Local, .. } => {
                    RowKind::LocalCurrent(value)
                }
                _ => RowKind::Power(value),
            },
            reference_bus: functional.at,
            weight: 1.0 / variance,
        });
    }
    rows
}

/// Runs the iterative-linear estimate, updating `buses` in place.
///
/// The state it produces is the same kind as [`super::nr::estimate`]'s and the
/// report has the same shape, so callers can switch methods without changing
/// how they read the answer. `residuals` are reported against the *original*
/// scalar measurements, not the paired complex rows.
pub fn estimate(
    measurements: &[Measurement],
    buses: &mut [Bus],
    net: &SeNetwork,
    options: &SeOptions,
) -> SeReport {
    let n = buses.len();
    let rows = build_rows(measurements, net);
    if rows.is_empty() {
        return SeReport {
            status: SeStatus::Singular,
            iterations: 0,
            objective: f64::NAN,
            last_step: f64::INFINITY,
            residuals: vec![0.0; measurements.len()],
            unconstrained: Vec::new(),
        };
    }

    // Zero-injection buses: the injected current is exactly zero, a linear
    // constraint in U. Far simpler than the nonlinear case, because no
    // linearization is involved at all.
    let constrained: Vec<usize> = net
        .zero_injection
        .iter()
        .enumerate()
        .filter(|&(_, &z)| z)
        .map(|(i, _)| i)
        .collect();
    let n_aug = n + constrained.len();

    // The normal-equations matrix. Weights are computed once, from the starting
    // magnitudes — the constant-|U| assumption this method rests on — so the
    // matrix never changes and is factorized exactly once.
    let mut triplets: Vec<(usize, usize, Complex<f64>)> = Vec::new();
    let mut row_weights = Vec::with_capacity(rows.len());
    for row in &rows {
        // Converting S to I divides by U, so the current's variance scales by
        // 1/|U|^2 — i.e. the weight scales by |U|^2.
        let w = row.weight * row.kind.weight_scale(buses, row.reference_bus);
        row_weights.push(w);
        for &(i, ci) in &row.coefficients {
            for &(j, cj) in &row.coefficients {
                triplets.push((i, j, ci.conj() * cj * w));
            }
        }
    }
    // Constraint rows, as a complex KKT block.
    for (k, &bus) in constrained.iter().enumerate() {
        let lambda = n + k;
        for &(j, y) in net.ybus.row(bus) {
            triplets.push((lambda, j, y));
            triplets.push((j, lambda, y.conj()));
        }
    }
    // Buses no row reaches would leave an empty pivot; pin them, exactly as the
    // Newton path does.
    let mut touched = vec![false; n];
    for row in &rows {
        for &(i, _) in &row.coefficients {
            touched[i] = true;
        }
    }
    for &bus in &constrained {
        for &(j, _) in net.ybus.row(bus) {
            touched[j] = true;
        }
    }
    let unconstrained: Vec<usize> = (0..n).filter(|&i| !touched[i]).collect();
    for &i in &unconstrained {
        triplets.push((i, i, Complex::new(1.0, 0.0)));
    }

    let Some(system) = ComplexSparseSystem::new(n_aug, &triplets) else {
        return SeReport {
            status: SeStatus::Singular,
            iterations: 0,
            objective: f64::NAN,
            last_step: f64::INFINITY,
            residuals: vec![0.0; measurements.len()],
            unconstrained,
        };
    };

    let phase_is_measured = rows
        .iter()
        .any(|r| matches!(r.kind, RowKind::Voltage { angle: Some(_), .. }));
    let reference = net
        .source_branches
        .iter()
        .position(|feeding| !feeding.is_empty())
        .unwrap_or(0);

    let mut v: Vec<Complex<f64>> = buses
        .iter()
        .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
        .collect();
    let mut last_step = f64::INFINITY;
    let mut status = SeStatus::MaxIterations;
    let mut iterations = options.max_iter;
    // Under-relaxation, engaged only if the plain iteration stops making
    // progress. The linearization `I = conj(S/U)` inverts the voltage, so an
    // estimate that is too low produces a current that is too high and pushes
    // the next estimate too high — a two-cycle the iteration can sit in
    // indefinitely. power-grid-model's `transmission-case` fixture does exactly
    // that here, parking at a step of 5.5e-2 for hundreds of iterations.
    //
    // Damping is safe in a way it would not be for a general solver: it changes
    // the path, never the fixed point. `tests::the_true_state_is_a_fixed_point`
    // pins that down separately, so a damped run converges to the same state an
    // undamped one would if it got there.
    let mut relaxation = 1.0f64;
    let mut previous_step = f64::INFINITY;

    for iteration in 1..=options.max_iter {
        // Re-linearize: voltages borrow the previous angle, powers become
        // currents at the previous voltage.
        let mut rhs = vec![Complex::new(0.0, 0.0); n_aug];
        for (row, &w) in rows.iter().zip(&row_weights) {
            let Some(measured) = row.kind.measured(&v, row.reference_bus) else {
                continue;
            };
            for &(i, ci) in &row.coefficients {
                rhs[i] += ci.conj() * w * measured;
            }
        }

        let Some(solution) = system.solve(&rhs) else {
            return SeReport {
                status: SeStatus::Singular,
                iterations: iteration,
                objective: f64::NAN,
                last_step,
                residuals: vec![0.0; measurements.len()],
                unconstrained,
            };
        };

        let mut next: Vec<Complex<f64>> = solution[..n].to_vec();
        // Without a phasor measurement the estimate is fixed only up to a global
        // rotation, so it is normalized the way power-grid-model normalizes
        // its own: the reference bus's angle to zero.
        if !phase_is_measured {
            let r = next[reference];
            if r.norm() > 0.0 {
                let rotation = Complex::from_polar(1.0, -r.arg());
                for u in next.iter_mut() {
                    *u *= rotation;
                }
            }
        }

        let raw_step = next
            .iter()
            .zip(&v)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        // Stalling, not converging: back off. The floor keeps a badly
        // conditioned case crawling rather than freezing.
        if iteration > 2 && raw_step > previous_step * 0.9 {
            relaxation = (relaxation * 0.5).max(1.0 / 64.0);
        }
        previous_step = raw_step;

        for (u, n) in v.iter_mut().zip(&next) {
            *u += (n - *u) * relaxation;
        }
        last_step = raw_step * relaxation;

        if last_step < options.tol {
            status = SeStatus::Converged;
            iterations = iteration;
            break;
        }
    }

    for (bus, u) in buses.iter_mut().zip(&v) {
        bus.voltage_mag = u.norm();
        bus.voltage_ang = u.arg();
    }

    // Residuals and the objective are reported against the *original* nonlinear
    // measurement functions, not the linearized ones — otherwise they would
    // describe the approximation rather than the answer, and could not be
    // compared with the Newton path's.
    let h = super::measurement_functions(measurements, buses, net);
    let residuals: Vec<f64> = measurements
        .iter()
        .zip(&h)
        .map(|(m, &hi)| m.value - hi)
        .collect();
    let objective = measurements
        .iter()
        .zip(&residuals)
        .filter(|(m, _)| m.weight().is_finite() && m.weight() > 0.0)
        .map(|(m, &r)| m.weight() * r * r)
        .sum::<f64>()
        / 2.0;

    SeReport { status, iterations, objective, last_step, residuals, unconstrained }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch_flow::Terminal;
    use crate::se::measurement_functions;
    use crate::se::nr::flat_start;

    fn m(kind: MeasurementKind, target: Target, sigma: f64) -> Measurement {
        Measurement { kind, target, value: 0.0, sigma }
    }

    /// A lone `P` with no matching `Q` cannot become a current, so it is
    /// dropped rather than half-used. Documenting that in a test because it is
    /// a genuine capability difference from the Newton path, which handles the
    /// two independently.
    #[test]
    fn unpaired_power_measurements_are_dropped() {
        let (net, _) = crate::se::tests::two_bus_net();
        let measurements = vec![m(MeasurementKind::ActivePower, Target::Bus(1), 0.01)];
        assert!(build_rows(&measurements, &net).is_empty());
    }

    /// A magnitude and an angle on one bus pair into a single phasor row.
    #[test]
    fn a_magnitude_and_angle_pair_into_one_row() {
        let (net, _) = crate::se::tests::two_bus_net();
        let measurements = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(1), 0.01),
            m(MeasurementKind::VoltageAngle, Target::Bus(1), 0.01),
        ];
        let rows = build_rows(&measurements, &net);
        assert_eq!(rows.len(), 1);
        assert!(
            matches!(rows[0].kind, RowKind::Voltage { angle: Some(_), .. }),
            "the phasor row should carry the measured angle"
        );
    }

    /// The property that makes under-relaxation safe: damping changes how the
    /// iteration travels, never where it lands.
    ///
    /// Seeded at a state that satisfies the measurements, one iteration must
    /// leave it there. Without this, backing off the step would be a way of
    /// hiding non-convergence rather than curing it.
    ///
    /// "There" means up to a global rotation. With no angle measured the phase
    /// is arbitrary and the estimator normalizes it to the reference bus, so a
    /// state seeded at any other rotation is reported as having *moved* while
    /// still satisfying every measurement exactly. What is invariant — and what
    /// this asserts — is that the residuals stay at zero and the angles shift
    /// by one shared constant.
    #[test]
    fn the_true_state_is_a_fixed_point() {
        let (net, truth) = crate::se::tests::two_bus_net();
        let probe = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 0.001),
            m(MeasurementKind::VoltageMagnitude, Target::Bus(1), 0.001),
            m(MeasurementKind::ActivePower, Target::Bus(1), 0.001),
            m(MeasurementKind::ReactivePower, Target::Bus(1), 0.001),
        ];
        let exact = measurement_functions(&probe, &truth, &net);
        let measurements: Vec<Measurement> = probe
            .iter()
            .zip(&exact)
            .map(|(p, &v)| Measurement { value: v, ..*p })
            .collect();

        let mut buses = truth.clone();
        let report = estimate(
            &measurements,
            &mut buses,
            &net,
            &SeOptions { max_iter: 1, ..SeOptions::default() },
        );

        for (i, r) in report.residuals.iter().enumerate() {
            assert!(r.abs() < 1e-9, "measurement {i} residual {r} should stay at zero");
        }
        for (i, (got, want)) in buses.iter().zip(&truth).enumerate() {
            assert!(
                (got.voltage_mag - want.voltage_mag).abs() < 1e-9,
                "bus {i} magnitude moved: {} vs {}",
                got.voltage_mag,
                want.voltage_mag
            );
        }
        let offsets: Vec<f64> = buses
            .iter()
            .zip(&truth)
            .map(|(g, w)| g.voltage_ang - w.voltage_ang)
            .collect();
        for (i, offset) in offsets.iter().enumerate() {
            assert!(
                (offset - offsets[0]).abs() < 1e-9,
                "bus {i} angle offset {offset} is not the shared rotation {}",
                offsets[0]
            );
        }
    }

    /// The method should recover the state its measurements were read from.
    #[test]
    fn recovers_the_state_its_measurements_came_from() {
        let (net, truth) = crate::se::tests::two_bus_net();
        let probe = vec![
            m(MeasurementKind::VoltageMagnitude, Target::Bus(0), 0.001),
            m(MeasurementKind::VoltageMagnitude, Target::Bus(1), 0.001),
            m(MeasurementKind::ActivePower, Target::Bus(1), 0.001),
            m(MeasurementKind::ReactivePower, Target::Bus(1), 0.001),
            m(
                MeasurementKind::ActivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
                0.001,
            ),
            m(
                MeasurementKind::ReactivePower,
                Target::BranchTerminal { branch: 0, terminal: Terminal::From },
                0.001,
            ),
        ];
        let exact = measurement_functions(&probe, &truth, &net);
        let measurements: Vec<Measurement> = probe
            .iter()
            .zip(&exact)
            .map(|(p, &v)| Measurement { value: v, ..*p })
            .collect();

        let mut buses = truth.clone();
        flat_start(&mut buses, &measurements);
        let report = estimate(&measurements, &mut buses, &net, &SeOptions::default());

        assert_eq!(report.status, SeStatus::Converged, "{report:?}");
        for (i, (got, want)) in buses.iter().zip(&truth).enumerate() {
            assert!(
                (got.voltage_mag - want.voltage_mag).abs() < 1e-6,
                "bus {i}: {} vs {}",
                got.voltage_mag,
                want.voltage_mag
            );
        }
    }
}
