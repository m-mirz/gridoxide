//! Phase-1 gates for `src/dynamics/`: the DAE, the integrator, and the
//! classical machine.
//!
//! Four properties, in the order they are worth failing on:
//!
//! - **G1, equilibrium invariance.** A run with no disturbance is a flat line.
//!   Nearly every mistake in initialization, per-unit conversion or residual
//!   assembly breaks this, and none of the others are meaningful until it
//!   holds.
//! - **G4, the Jacobian oracle.** The analytic Jacobian matches a central
//!   difference. What runs is the analytic one; what proves it is the
//!   numerical one.
//! - **The physics.** A single machine against an infinite bus oscillates at
//!   the frequency the linearized swing equation says it does — a closed form,
//!   with no reference implementation involved.
//! - **G3, order of accuracy.** Halving the step quarters the error, because
//!   the trapezoidal rule is second-order.
//! - **G5, backend agreement.** `Scalar` and `KluNative` produce the same
//!   trajectory.

use num_complex::Complex;

use gridoxide::dynamics::models::machine::{self, GenCls, GenClsParams};
use gridoxide::dynamics::models::{finite_difference, DynamicModel, GeneratingUnit, ModelJacobian};
use gridoxide::dynamics::{
    build, run_dynamics, settle, DeviceSpec, DynamicSystem, DynamicsOptions, DynamicsStatus,
    SystemSpec,
};
use gridoxide::network::{build_ybus, power_injections};
use gridoxide::solver::JacobianBackend;
use gridoxide::types::{Bus, BusType, Line};

const S_BASE: f64 = 100.0;
const F_NOM: f64 = 50.0;

const H: f64 = 5.0;
const XDP: f64 = 0.3;
const X_LINE: f64 = 0.2;
const P_GEN: f64 = 0.8;

fn bus(idx: usize, bus_type: BusType, vmag: f64, p: f64, q: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: vmag,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: q,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: Vec::new(),
    }
}

fn params(d: f64) -> GenClsParams {
    GenClsParams { h: H, d, ra: 0.0, xdp: XDP, mbase: S_BASE }
}

/// A single machine at bus 0, a lossless line, an infinite bus at 1∠0.
///
/// The textbook arrangement the equal-area criterion and the linearized swing
/// equation are both written for, so the closed forms below apply directly.
/// Bus 1 is *fixed*, not merely a slack bus: the analytic results assume an
/// infinite bus, and a fixed-voltage constraint is that exactly, where a
/// large-inertia machine would only approximate it.
fn smib(damping: f64) -> (DynamicSystem, f64, f64) {
    let buses = vec![
        bus(0, BusType::PV, 1.0, P_GEN, 0.0),
        bus(1, BusType::Slack, 1.0, 0.0, 0.0),
    ];
    let lines = vec![Line { from: 0, to: 1, r: 0.0, x: X_LINE, b_shunt: 0.0, g_shunt: 0.0 }];

    let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
        buses,
        lines: lines.clone(),
    });
    assert_eq!(
        report.stats.status,
        gridoxide::solver::SolveStatus::Converged,
        "the SMIB base case must solve"
    );
    let buses = report.buses;

    // The machine's terminal power is whatever the power flow put at bus 0 —
    // read off the solved state rather than from `p_spec`/`q_spec`, which do
    // not carry a PV bus's reactive output.
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();
    let (p_calc, q_calc) = power_injections(&buses, &ybus);
    let s_dev = Complex::new(p_calc[0], q_calc[0]);

    let model = GenCls::new(params(damping), S_BASE, F_NOM).expect("machine parameters are valid");
    let system = build(SystemSpec {
        buses: &buses,
        lines: &lines,
        transformers: &[],
        shunts: &[],
        devices: vec![DeviceSpec {
            id: "G1".to_string(),
            bus: 0,
            s: s_dev,
            model: machine::bare(Box::new(model)),
        }],
        fixed_buses: vec![1],
    })
    .expect("the solved case initializes to an equilibrium");

    // P_max and δ₀ for the closed forms. The whole reactance from the internal
    // EMF to the infinite bus is x'_d + x_line, and the infinite bus is the
    // angle reference at 1∠0.
    let e_mag = {
        let v0 = system.voltages()[0];
        let i = (s_dev / v0).conj();
        (v0 + Complex::new(0.0, XDP) * i).norm()
    };
    let p_max = e_mag * 1.0 / (XDP + X_LINE);
    let delta_0 = system.state()[0];
    (system, p_max, delta_0)
}

fn options(step: f64, end: f64) -> DynamicsOptions {
    DynamicsOptions { end_time: end, step, damping_steps: 0, ..Default::default() }
}

/// G1. No disturbance, so nothing moves — and "nothing" means machine epsilon,
/// not a solver tolerance.
#[test]
fn equilibrium_is_invariant() {
    let (mut system, _, delta_0) = smib(0.0);

    assert!(
        system.max_derivative() < 1e-12,
        "initial derivative should be zero, was {:e}",
        system.max_derivative()
    );
    assert!(
        system.network_residual_norm() < 1e-12,
        "initial network residual should be zero, was {:e}",
        system.network_residual_norm()
    );

    let report = run_dynamics(&mut system, &options(0.01, 10.0));
    assert_eq!(report.status, DynamicsStatus::Completed);

    let delta = report.trajectory.series("G1.delta").expect("delta is recorded");
    let omega = report.trajectory.series("G1.omega").expect("omega is recorded");
    let vmag = report.trajectory.series("bus0.vmag").expect("bus 0 voltage is recorded");

    let d_drift = delta.iter().fold(0.0f64, |m, d| m.max((d - delta_0).abs()));
    let w_drift = omega.iter().fold(0.0f64, |m, w| m.max((w - 1.0).abs()));
    let v_drift = vmag.iter().fold(0.0f64, |m, v| m.max((v - vmag[0]).abs()));

    assert!(d_drift < 1e-12, "rotor angle drifted by {d_drift:e} over 10 s with no disturbance");
    assert!(w_drift < 1e-12, "rotor speed drifted by {w_drift:e}");
    assert!(v_drift < 1e-12, "terminal voltage drifted by {v_drift:e}");
}

/// G4. The analytic Jacobian is what runs; the finite difference is what
/// proves it. Probed away from the equilibrium, where every term is nonzero —
/// at the equilibrium a sign error in `∂P_e/∂δ` could hide.
///
/// Run through [`GeneratingUnit`], not the bare machine, so the composite's
/// chain rule is under the oracle too. With no controls attached the chain
/// rule degenerates to a copy, which is exactly the case worth pinning before
/// anything is attached.
#[test]
fn analytic_jacobian_matches_finite_difference() {
    let mut model = GeneratingUnit::machine_only(Box::new(
        GenCls::new(params(2.0), S_BASE, F_NOM).unwrap(),
    ));
    model
        .initialize(Complex::from_polar(1.02, 0.15), Complex::new(0.8, 0.3))
        .expect("initializes");

    let probes = [
        (0.35, 1.000, Complex::new(1.00, 0.00)),
        (0.35, 1.002, Complex::new(0.98, 0.12)),
        (-0.60, 0.995, Complex::new(1.05, -0.20)),
        (2.10, 1.010, Complex::new(0.60, 0.35)),
        (0.00, 1.000, Complex::new(0.30, 0.00)),
    ];

    for (delta, omega, v) in probes {
        let x = [delta, omega];
        let mut analytic = ModelJacobian::zeros(model.n_states());
        analytic.clear();
        model.jacobian(&x, v, &mut analytic);
        let numeric = finite_difference(&model, &x, v, 1e-6);

        let compare = |name: &str, a: &[f64], b: &[f64]| {
            for (k, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                let scale = av.abs().max(bv.abs()).max(1.0);
                assert!(
                    (av - bv).abs() / scale < 1e-6,
                    "{name}[{k}] at delta={delta}, omega={omega}, v={v}: \
                     analytic {av:e} vs numeric {bv:e}"
                );
            }
        };
        compare("dfdx", &analytic.dfdx, &numeric.dfdx);
        compare("dfdv", &analytic.dfdv, &numeric.dfdv);
        compare("didx", &analytic.didx, &numeric.didx);
        compare("didv", &analytic.didv, &numeric.didv);
    }
}

/// Zero crossings of `delta − delta_0`, linearly interpolated. Two consecutive
/// crossings are half a period apart.
fn crossings(time: &[f64], series: &[f64], about: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for k in 1..series.len() {
        let (a, b) = (series[k - 1] - about, series[k] - about);
        if (a < 0.0) != (b < 0.0) {
            let frac = a / (a - b);
            out.push(time[k - 1] + frac * (time[k] - time[k - 1]));
        }
    }
    out
}

/// The physics gate. Linearizing the swing equation about the equilibrium of
/// an undamped classical machine against an infinite bus gives
///
/// ```text
/// ω_n = sqrt(Ω_b · K_s / (2H)),   K_s = P_max · cos δ₀
/// ```
///
/// so the small-signal oscillation has period `2π/ω_n`. No reference
/// implementation is involved, and the quantity being checked — how fast the
/// rotor swings — is the one the simulator exists to compute.
#[test]
fn small_signal_oscillation_matches_the_closed_form() {
    let (mut system, p_max, delta_0) = smib(0.0);

    // A small nudge, so the linearization is the right prediction. The
    // algebraic constraint must be re-solved after writing the state: `V` no
    // longer satisfies `Y V = I_inj(x, V)` for the new angle.
    let opts = options(0.001, 6.0);
    system.state_mut()[0] += 1e-3;
    settle(&mut system, &opts);
    assert!(
        system.network_residual_norm() < 1e-9,
        "the algebraic re-solve must restore the constraint"
    );

    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);

    let delta = report.trajectory.series("G1.delta").unwrap();
    let zeros = crossings(&report.trajectory.time, &delta, delta_0);
    assert!(zeros.len() >= 5, "expected several oscillations, found {} crossings", zeros.len());
    // Average half-period over the whole record, which is far less sensitive to
    // any one crossing than the first pair would be.
    let measured = 2.0 * (zeros[zeros.len() - 1] - zeros[0]) / (zeros.len() - 1) as f64;

    let omega_b = std::f64::consts::TAU * F_NOM;
    let k_s = p_max * delta_0.cos();
    let expected = std::f64::consts::TAU / (omega_b * k_s / (2.0 * H)).sqrt();

    let rel = (measured - expected).abs() / expected;
    assert!(
        rel < 2e-3,
        "oscillation period {measured:.6} s differs from the closed form {expected:.6} s by {:.3}%",
        rel * 100.0
    );
}

/// Damping makes the swing decay; without it the amplitude is preserved. Both
/// halves matter: an integrator that leaks energy would pass the second check
/// only by accident, and the trapezoidal rule is chosen precisely because it
/// does not.
#[test]
fn damping_decays_and_its_absence_does_not() {
    let run = |d: f64| {
        let (mut system, _, delta_0) = smib(d);
        let opts = options(0.001, 6.0);
        system.state_mut()[0] += 1e-2;
        settle(&mut system, &opts);
        let report = run_dynamics(&mut system, &opts);
        let delta = report.trajectory.series("G1.delta").unwrap();
        let n = delta.len();
        let early = delta[..n / 6].iter().fold(0.0f64, |m, d| m.max((d - delta_0).abs()));
        let late = delta[5 * n / 6..].iter().fold(0.0f64, |m, d| m.max((d - delta_0).abs()));
        (early, late)
    };

    let (early, late) = run(0.0);
    assert!(
        (late - early).abs() / early < 5e-3,
        "an undamped swing should keep its amplitude: {early:e} then {late:e}"
    );

    let (early, late) = run(10.0);
    assert!(late < 0.5 * early, "a damped swing should decay: {early:e} then {late:e}");
}

/// G3. The trapezoidal rule is second-order, so halving the step quarters the
/// error. Measured against a reference run at one sixteenth the coarsest step,
/// and with the backward-Euler damping disabled — two first-order steps at the
/// start of the run would drag the observed order towards one, which is
/// exactly the failure this is here to notice.
#[test]
fn trapezoidal_is_second_order() {
    let final_delta = |step: f64| {
        let (mut system, _, _) = smib(1.0);
        let opts = options(step, 1.0);
        system.state_mut()[0] += 0.2;
        settle(&mut system, &opts);
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        *report.trajectory.series("G1.delta").unwrap().last().unwrap()
    };

    let reference = final_delta(0.02 / 16.0);
    let coarse = (final_delta(0.02) - reference).abs();
    let fine = (final_delta(0.01) - reference).abs();

    let order = (coarse / fine).log2();
    assert!(
        (order - 2.0).abs() < 0.2,
        "observed order of accuracy {order:.3} (errors {coarse:e} then {fine:e})"
    );
}

/// G5. The backend is a performance choice, never an accuracy one.
#[test]
fn backends_agree() {
    let trajectory = |backend: JacobianBackend| {
        let (mut system, _, _) = smib(1.0);
        let opts = DynamicsOptions { backend, ..options(0.005, 2.0) };
        system.state_mut()[0] += 0.1;
        settle(&mut system, &opts);
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        report.trajectory.series("G1.delta").unwrap()
    };

    let scalar = trajectory(JacobianBackend::Scalar);
    let klu = trajectory(JacobianBackend::KluNative);
    assert_eq!(scalar.len(), klu.len());
    let worst = scalar
        .iter()
        .zip(klu.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-9, "Scalar and KluNative disagree by {worst:e}");
}

/// A mis-declared split between a device and its bus's load is **not**
/// caught, and this records why rather than pretending otherwise.
///
/// Declaring half the machine's real output is self-consistent: the machine
/// initializes to an equilibrium at the power it was told it makes, and the
/// half left over becomes part of the bus's constant admittance — a
/// negative-conductance element that injects it. Both the rotor's derivative
/// and the network constraint are then exactly zero, so neither of `build`'s
/// checks can see anything wrong.
///
/// What it produces is a *different machine* — smaller output, smaller
/// internal EMF, smaller rotor angle — swinging against a network that makes
/// up the difference from something inert. The trajectory is plausible and the
/// answer is wrong. The only defence is the caller stating the split
/// correctly, which is why `DeviceSpec::s` is explicit rather than inferred.
#[test]
fn a_mis_declared_split_is_silent_and_changes_the_machine() {
    let split = |s_dev: Complex<f64>| {
        let buses = vec![
            bus(0, BusType::PV, 1.0, P_GEN, 0.0),
            bus(1, BusType::Slack, 1.0, 0.0, 0.0),
        ];
        let lines = vec![Line { from: 0, to: 1, r: 0.0, x: X_LINE, b_shunt: 0.0, g_shunt: 0.0 }];
        let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
            buses,
            lines: lines.clone(),
        });
        let model = GenCls::new(params(0.0), S_BASE, F_NOM).unwrap();
        build(SystemSpec {
            buses: &report.buses,
            lines: &lines,
            transformers: &[],
            shunts: &[],
            devices: vec![DeviceSpec {
                id: "G1".to_string(),
                bus: 0,
                s: s_dev,
                model: machine::bare(Box::new(model)),
            }],
            fixed_buses: vec![1],
        })
        .expect("a self-consistent split always builds")
    };

    let ybus = build_ybus(
        2,
        &[Line { from: 0, to: 1, r: 0.0, x: X_LINE, b_shunt: 0.0, g_shunt: 0.0 }],
        &[],
    )
    .finish();
    let honest = {
        let buses = vec![
            bus(0, BusType::PV, 1.0, P_GEN, 0.0),
            bus(1, BusType::Slack, 1.0, 0.0, 0.0),
        ];
        let lines = vec![Line { from: 0, to: 1, r: 0.0, x: X_LINE, b_shunt: 0.0, g_shunt: 0.0 }];
        let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
            buses,
            lines,
        });
        let (p, q) = power_injections(&report.buses, &ybus);
        Complex::new(p[0], q[0])
    };

    let correct = split(honest);
    let halved = split(Complex::new(honest.re / 2.0, honest.im / 2.0));

    // Both are equilibria — that is the whole point.
    for system in [&correct, &halved] {
        assert!(system.max_derivative() < 1e-12);
        assert!(system.network_residual_norm() < 1e-12);
    }

    // And they are different machines: less power means a smaller rotor angle.
    let (d_correct, d_halved) = (correct.state()[0], halved.state()[0]);
    assert!(
        d_halved < d_correct - 0.05,
        "the mis-declared machine should sit at a visibly smaller angle: \
         {d_correct:.4} rad vs {d_halved:.4} rad"
    );
}
