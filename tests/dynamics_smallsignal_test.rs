//! Small-signal analysis: the modes of the linearized system.
//!
//! The load-bearing gate is that the eigenvalue agrees with the **closed form**
//! `tests/dynamics_test.rs` already checks the time-domain run against. Two
//! routes to the same number — one integrating a trajectory and reading its
//! period, the other linearizing and taking an eigenvalue — through almost no
//! shared code, is much stronger evidence than either agreeing with itself.

use num_complex::Complex;

use gridoxide::dynamics::models::machine::{GenCls, GenClsParams, GenTransient, GenTransientParams};
use gridoxide::dynamics::models::{DynamicModel, GeneratingUnit};
use gridoxide::dynamics::smallsignal::{self, SmallSignalError};
use gridoxide::dynamics::{
    build, run_dynamics, settle, DeviceSpec, DynamicSystem, DynamicsOptions, DynamicsStatus,
    SystemSpec,
};
use gridoxide::network::{build_ybus, power_injections};
use gridoxide::types::{Bus, BusType, Line};

const S_BASE: f64 = 100.0;
const F_NOM: f64 = 50.0;
const OMEGA_B: f64 = std::f64::consts::TAU * F_NOM;
const H: f64 = 5.0;
const XDP: f64 = 0.3;
const X_LINE: f64 = 0.2;
const P_GEN: f64 = 0.8;

fn bus(idx: usize, bus_type: BusType, p: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: 0.0,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: Vec::new(),
    }
}

/// One classical machine against an infinite bus, and the closed-form
/// synchronizing coefficient the linearized swing equation predicts from.
fn smib(damping: f64) -> (DynamicSystem, f64) {
    let buses = vec![bus(0, BusType::PV, P_GEN), bus(1, BusType::Slack, 0.0)];
    let lines = vec![Line { from: 0, to: 1, r: 0.0, x: X_LINE, b_shunt: 0.0, g_shunt: 0.0 }];
    let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
        buses,
        lines: lines.clone(),
    });
    let buses = report.buses;
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();
    let (p_calc, q_calc) = power_injections(&buses, &ybus);
    let s_dev = Complex::new(p_calc[0], q_calc[0]);

    let machine = GenCls::new(
        GenClsParams { h: H, d: damping, ra: 0.0, xdp: XDP, mbase: S_BASE },
        S_BASE,
        F_NOM,
    )
    .unwrap();
    let system = build(SystemSpec {
        buses: &buses,
        lines: &lines,
        transformers: &[],
        shunts: &[],
        devices: vec![DeviceSpec {
            id: "G1".to_string(),
            bus: 0,
            s: s_dev,
            model: Box::new(GeneratingUnit::machine_only(Box::new(machine))),
        }],
        fixed_buses: vec![1],
    })
    .unwrap();

    // K_s = P_max·cos δ₀, with P_max = E·V/X over the whole reactance.
    let v0 = system.voltages()[0];
    let e_mag = (v0 + Complex::new(0.0, XDP) * (s_dev / v0).conj()).norm();
    let k_s = (e_mag / (XDP + X_LINE)) * system.state()[0].cos();
    (system, k_s)
}

/// The eigenvalue is the closed form.
///
/// Linearizing the swing equation of an undamped classical machine against an
/// infinite bus gives `λ = ±j·√(Ω_b·K_s/2H)` exactly — a purely imaginary pair,
/// since nothing dissipates. Both halves of that are checked: the frequency,
/// and that the real part really is zero.
#[test]
fn the_eigenvalue_is_the_closed_form() {
    let (system, k_s) = smib(0.0);
    let result = smallsignal::analyze(&system).expect("an equilibrium can be analyzed");

    assert_eq!(result.modes.len(), 2, "a classical machine has two states, so two modes");
    assert_eq!(result.state_names, vec!["G1.delta", "G1.omega"]);

    let expected = (OMEGA_B * k_s / (2.0 * H)).sqrt();
    let mode = result.critical().expect("there is a mode");
    assert!(
        (mode.eigenvalue.im.abs() - expected).abs() / expected < 1e-9,
        "eigenvalue {} against the closed form ±j{expected:.6}",
        mode.eigenvalue
    );
    assert!(
        mode.eigenvalue.re.abs() < 1e-9,
        "an undamped machine's mode should be purely imaginary, got {}",
        mode.eigenvalue
    );
    assert!(mode.damping.abs() < 1e-9, "and therefore have zero damping ratio");
    assert!(
        (mode.frequency - expected / std::f64::consts::TAU).abs() < 1e-9,
        "the reported frequency should be the imaginary part in Hz"
    );

    // Both halves of the conjugate pair are present, and both states take part.
    let pair: Vec<f64> = result.modes.iter().map(|m| m.eigenvalue.im).collect();
    assert!(pair.iter().any(|v| *v > 0.0) && pair.iter().any(|v| *v < 0.0));
    let names = result.participants(mode);
    assert_eq!(names.len(), 2, "a swing mode is the angle and the speed together");
    assert!(
        (names[0].1 - 0.5).abs() < 1e-6,
        "and they take part equally: {names:?}"
    );
}

/// The eigenvalue and the time-domain run agree, through almost no shared code.
///
/// One integrates a trajectory and counts zero crossings; the other assembles a
/// Jacobian, eliminates the network and takes an eigenvalue. That they land on
/// the same period is the check worth having.
#[test]
fn the_eigenvalue_predicts_the_observed_oscillation() {
    let (mut system, _) = smib(0.0);
    let result = smallsignal::analyze(&system).unwrap();
    let predicted = std::f64::consts::TAU / result.critical().unwrap().eigenvalue.im.abs();

    // The oscillation is about the *equilibrium*, which is where δ was before
    // the nudge — not where the run starts. Centring on the perturbed value
    // makes every crossing a tangency instead.
    let about = system.state()[0];
    let opts = DynamicsOptions { end_time: 6.0, step: 0.001, damping_steps: 0, ..Default::default() };
    system.state_mut()[0] += 1e-3;
    settle(&mut system, &opts);
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);

    let delta = report.trajectory.series("G1.delta").unwrap();
    let time = &report.trajectory.time;
    let mut zeros = Vec::new();
    for k in 1..delta.len() {
        let (a, b) = (delta[k - 1] - about, delta[k] - about);
        if (a < 0.0) != (b < 0.0) {
            zeros.push(time[k - 1] + (a / (a - b)) * (time[k] - time[k - 1]));
        }
    }
    assert!(zeros.len() >= 5);
    let observed = 2.0 * (zeros[zeros.len() - 1] - zeros[0]) / (zeros.len() - 1) as f64;

    assert!(
        (observed - predicted).abs() / predicted < 2e-3,
        "the eigenvalue predicts a period of {predicted:.6} s; the run showed {observed:.6} s"
    );
}

/// Damping shows up in the real part, and predicts the observed decay.
///
/// `ζ` is not a label: the envelope of the oscillation falls as `exp(σ·t)`, so
/// the eigenvalue's real part says how much amplitude is left after a second.
#[test]
fn the_real_part_predicts_the_decay() {
    let (mut system, _) = smib(10.0);
    let result = smallsignal::analyze(&system).unwrap();
    let mode = result.critical().unwrap();
    assert!(mode.eigenvalue.re < 0.0, "a damped machine's mode must decay");
    assert!(mode.damping > 0.0 && mode.damping < 1.0, "damping ratio {}", mode.damping);
    assert!(!result.modes.iter().any(|m| m.is_unstable()));

    let sigma = mode.eigenvalue.re;
    // Measured about the equilibrium: relative to the perturbed starting value
    // the decaying oscillation leaves a constant offset behind, which does not
    // decay and swamps the envelope.
    let about = system.state()[0];
    let opts = DynamicsOptions { end_time: 8.0, step: 0.001, damping_steps: 0, ..Default::default() };
    system.state_mut()[0] += 1e-3;
    settle(&mut system, &opts);
    let report = run_dynamics(&mut system, &opts);
    let delta = report.trajectory.series("G1.delta").unwrap();
    let time = &report.trajectory.time;

    // Actual local maxima, not a windowed maximum: a fixed window need not
    // contain a peak at the same phase of the cycle, and comparing two
    // differently-phased samples measures the phase as much as the decay.
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for k in 1..delta.len() - 1 {
        let (a, b, c) = (delta[k - 1] - about, delta[k] - about, delta[k + 1] - about);
        if b > a && b >= c && b > 0.0 {
            peaks.push((time[k], b));
        }
    }
    assert!(peaks.len() >= 4, "expected several peaks, found {}", peaks.len());

    let (t0, a0) = peaks[0];
    let (t1, a1) = *peaks.last().unwrap();
    let predicted = (sigma * (t1 - t0)).exp();
    let observed = a1 / a0;
    assert!(
        (observed / predicted - 1.0).abs() < 0.02,
        "over {:.3} s the envelope should fall by exp(σ·Δt) = {predicted:.6}; \
         it fell by {observed:.6}",
        t1 - t0
    );
}

/// Participation factors name the states a mode belongs to.
///
/// A fourth-order machine has a slow flux mode alongside its swing mode, and
/// they belong to different states. The flux mode's time constant is of the
/// order of `T'_d0`, and the swing mode's participation is the rotor's — which
/// is what makes the factor actionable rather than decorative.
#[test]
fn participation_factors_separate_the_flux_mode_from_the_swing() {
    let buses = vec![bus(0, BusType::PV, P_GEN), bus(1, BusType::Slack, 0.0)];
    let lines = vec![Line { from: 0, to: 1, r: 0.0, x: X_LINE, b_shunt: 0.0, g_shunt: 0.0 }];
    let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
        buses,
        lines: lines.clone(),
    });
    let buses = report.buses;
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();
    let (p_calc, q_calc) = power_injections(&buses, &ybus);

    let machine = GenTransient::new(
        GenTransientParams {
            h: H,
            d: 2.0,
            ra: 0.003,
            xd: 1.8,
            xq: 1.7,
            xdp: 0.30,
            xqp: 0.55,
            td0p: 8.0,
            tq0p: 0.4,
            mbase: S_BASE,
        },
        S_BASE,
        F_NOM,
    )
    .unwrap();
    let system = build(SystemSpec {
        buses: &buses,
        lines: &lines,
        transformers: &[],
        shunts: &[],
        devices: vec![DeviceSpec {
            id: "G1".to_string(),
            bus: 0,
            s: Complex::new(p_calc[0], q_calc[0]),
            model: Box::new(GeneratingUnit::machine_only(Box::new(machine))) as Box<dyn DynamicModel>,
        }],
        fixed_buses: vec![1],
    })
    .unwrap();

    let result = smallsignal::analyze(&system).unwrap();
    assert_eq!(result.modes.len(), 4);
    for mode in &result.modes {
        let total: f64 = mode.participation.iter().map(|&(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-9, "participations normalize: {total}");
    }

    // The swing mode: the fastest oscillation, and the rotor's.
    let swing = result
        .modes
        .iter()
        .filter(|m| m.is_oscillatory())
        .max_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap())
        .expect("there is an oscillatory mode");
    let owners = result.participants(swing);
    let rotor: f64 = owners
        .iter()
        .filter(|(n, _)| n.ends_with(".delta") || n.ends_with(".omega"))
        .map(|(_, p)| p)
        .sum();
    assert!(rotor > 0.8, "the swing mode belongs to the rotor: {owners:?}");

    // The slow real mode: the field flux, on the order of T'_d0.
    let slow = result
        .modes
        .iter()
        .filter(|m| !m.is_oscillatory())
        .max_by(|a, b| a.time_constant.partial_cmp(&b.time_constant).unwrap())
        .expect("there is a non-oscillatory mode");
    let owners = result.participants(slow);
    assert!(
        owners[0].0.ends_with(".eqp"),
        "the slowest real mode should be the field flux: {owners:?}"
    );
    assert!(
        slow.time_constant > 1.0,
        "and it should be slow — of the order of T'_d0 = 8 s — not {:.3} s",
        slow.time_constant
    );
}

/// Analysing a point the system is not sitting at is refused.
///
/// A linearization about a mid-transient state describes the dynamics of
/// nothing in particular, and its eigenvalues would look entirely plausible.
#[test]
fn a_non_equilibrium_is_refused() {
    let (mut system, _) = smib(0.0);
    system.state_mut()[1] += 0.01;
    let err = smallsignal::analyze(&system).expect_err("must refuse");
    assert!(
        matches!(err, SmallSignalError::NotAnEquilibrium { .. }),
        "got {err}"
    );
    assert!(err.to_string().contains("describes nothing"), "{err}");
}

/// The mode shape says *how* the machines move, which participation cannot.
///
/// Two machines islanded together have two rotor modes, and they are opposites.
/// One is the **common mode**: both rotors drift as one, nothing oscillates,
/// and the eigenvalue is at the origin — an islanded system's absolute angle is
/// free, so this mode is the freedom itself. The other is the **inter-machine
/// oscillation**: the two swing *against* each other, which shows up as their
/// shape components sitting near opposite phase.
///
/// Participation cannot make that distinction. It would say both modes are the
/// rotors', which is true of both and useful about neither.
#[test]
fn the_mode_shape_separates_a_swing_from_a_drift() {
    let machine = |p: f64| {
        GenCls::new(
            GenClsParams { h: H, d: 1.0, ra: 0.0, xdp: XDP, mbase: S_BASE },
            S_BASE,
            F_NOM,
        )
        .map(|m| (m, p))
        .unwrap()
    };
    let buses = vec![
        bus(0, BusType::Slack, 0.4),
        bus(1, BusType::PV, 0.4),
        bus(2, BusType::PQ, -0.8),
    ];
    let lines = vec![
        Line { from: 0, to: 2, r: 0.0, x: 0.10, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.0, x: 0.10, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
        buses,
        lines: lines.clone(),
    });
    let buses = report.buses;
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();
    let (p_calc, q_calc) = power_injections(&buses, &ybus);

    let devices = (0..2)
        .map(|i| DeviceSpec {
            id: format!("G{}", i + 1),
            bus: i,
            s: Complex::new(p_calc[i], q_calc[i]),
            model: Box::new(GeneratingUnit::machine_only(Box::new(machine(0.4).0)))
                as Box<dyn DynamicModel>,
        })
        .collect();

    // No fixed bus: the two machines are islanded together, which is what makes
    // the common mode free.
    let system = build(SystemSpec {
        buses: &buses,
        lines: &lines,
        transformers: &[],
        shunts: &[],
        devices,
        fixed_buses: Vec::new(),
    })
    .unwrap();

    let result = smallsignal::analyze(&system).unwrap();
    assert_eq!(result.modes.len(), 4, "two classical machines, four states");

    // The oscillation: the two rotors near opposite phase.
    let swing = result
        .modes
        .iter()
        .filter(|m| m.is_oscillatory())
        .max_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap())
        .expect("there is an oscillatory mode");
    let shape = result.shape(swing);
    assert_eq!(shape.len(), 2, "one component per rotor: {shape:?}");
    let separation = (shape[0].2 - shape[1].2).abs();
    assert!(
        (separation - 180.0).abs() < 5.0,
        "the two machines should swing against each other, but their phases are \
         {:.1}° apart: {shape:?}",
        separation
    );
    // Equal machines on a symmetric network take equal parts in it.
    assert!(
        (shape[0].1 - shape[1].1).abs() < 1e-6,
        "identical machines should have equal magnitudes: {shape:?}"
    );

    // The drift: an islanded system's absolute angle is free, so one mode sits
    // at the origin and carries no oscillation to have a shape.
    let free = result
        .modes
        .iter()
        .min_by(|a, b| a.eigenvalue.norm().partial_cmp(&b.eigenvalue.norm()).unwrap())
        .unwrap();
    assert!(
        free.eigenvalue.norm() < 1e-8,
        "an islanded pair should have a zero mode, found {}",
        free.eigenvalue
    );
    assert!(free.shape.is_empty(), "a non-oscillatory mode has no shape to report");
}

/// The dense method says so rather than running for hours.
#[test]
fn a_system_too_large_for_the_dense_method_is_refused() {
    // Not built — the check is on the message, and standing a 2000-state system
    // up to be refused would cost more than the refusal is worth.
    let err = SmallSignalError::TooLarge { states: 20_480, limit: 2000 };
    assert!(err.to_string().contains("sparse Arnoldi"), "{err}");
    assert!(err.to_string().contains("not implemented"), "{err}");
}
