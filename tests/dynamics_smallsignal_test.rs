//! Small-signal analysis: the modes of the linearized system.
//!
//! The load-bearing gate is that the eigenvalue agrees with the **closed form**
//! `tests/dynamics_test.rs` already checks the time-domain run against. Two
//! routes to the same number — one integrating a trajectory and reading its
//! period, the other linearizing and taking an eigenvalue — through almost no
//! shared code, is much stronger evidence than either agreeing with itself.

mod dynamics_ring;

use num_complex::Complex;

use gridoxide::dynamics::models::machine::{GenCls, GenClsParams, GenTransient, GenTransientParams};
use gridoxide::dynamics::models::{DynamicModel, GeneratingUnit};
use gridoxide::dynamics::arnoldi::{self, ArnoldiOptions};
use gridoxide::dynamics::shift::{ShiftError, ShiftedDae};
use gridoxide::dynamics::smallsignal::{
    self, Method, ParameterRef, SmallSignalError, SmallSignalOptions,
};
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
    smib_with(H, damping)
}

/// The same, with the inertia named — so a sensitivity can be checked against a
/// difference of two whole analyses.
fn smib_with(h: f64, damping: f64) -> (DynamicSystem, f64) {
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
        GenClsParams { h, d: damping, ra: 0.0, xdp: XDP, mbase: S_BASE },
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

/// The eigenvalue of the swing mode, whichever half of the pair sorts first.
fn swing_eigenvalue(system: &DynamicSystem) -> Complex<f64> {
    smallsignal::analyze(system)
        .unwrap()
        .modes
        .iter()
        .filter(|m| m.eigenvalue.im > 0.0)
        .map(|m| m.eigenvalue)
        .next()
        .expect("the swing pair")
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

/// The same single machine with a *transient* model — four states rather than
/// two, and an `A` with no symmetry to hide a transposition mistake behind.
fn smib_transient() -> DynamicSystem {
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
    build(SystemSpec {
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
    .unwrap()
}

/// Participation factors name the states a mode belongs to.
///
/// A fourth-order machine has a slow flux mode alongside its swing mode, and
/// they belong to different states. The flux mode's time constant is of the
/// order of `T'_d0`, and the swing mode's participation is the rotor's — which
/// is what makes the factor actionable rather than decorative.
#[test]
fn participation_factors_separate_the_flux_mode_from_the_swing() {
    let system = smib_transient();

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

/// Two identical machines islanded together, with no infinite bus.
///
/// Exactly two rotor modes, and they are opposites: the oscillation between the
/// machines, and the free drift of the island's absolute angle. Nothing holds
/// the angle, which is what makes the second one sit at the origin.
fn two_machine_island() -> DynamicSystem {
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

    system
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
    let system = two_machine_island();

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

/// Past the dense limit, `analyze` answers by the other method rather than
/// refusing — and says which one answered.
///
/// This used to be a refusal, and the test used to assert on its wording. The
/// refusal was honest while there was nothing to fall back to; now there is,
/// and what has to be gated instead is that the fallback happens, that the
/// result is labelled, and that a caller can still ask for the dense object
/// specifically and be told why it is not available.
#[test]
fn past_the_dense_limit_the_sparse_method_answers() {
    // 256 units × 10 states = 2560, comfortably past the 2000-state limit and
    // small enough to build in a test.
    let document = dynamics_ring::ring(512, 0.4);
    let (system, _) = document.build().expect("the ring solves and initializes");
    assert!(system.n_states() > 2000, "{} states", system.n_states());

    let result = smallsignal::analyze(&system).expect("answered rather than refused");
    match result.method {
        smallsignal::Method::Sparse { shift, .. } => {
            // The default aim is the electromechanical band.
            assert!((shift.im / std::f64::consts::TAU - 1.0).abs() < 1e-9, "{shift}");
        }
        other => panic!("expected the sparse method past the limit, got {other}"),
    }
    assert!(result.converged, "{:?}", result.modes);
    assert!(!result.modes.is_empty());
    for mode in &result.modes {
        assert!(mode.residual < 1e-6, "{}: residual {}", mode.eigenvalue, mode.residual);
    }

    // The dense object itself is still refused, with a reason that names what
    // to do instead.
    match smallsignal::state_matrix(&system) {
        Err(SmallSignalError::TooLarge { states, limit }) => {
            assert_eq!(states, system.n_states());
            assert_eq!(limit, 2000);
        }
        other => panic!("expected the dense reduction to be refused, got {other:?}"),
    }
}

/// A handful of probe vectors: each unit vector, plus one genuinely complex
/// mixture so a bug that only shows up off the axes has somewhere to show.
fn probes(n: usize) -> Vec<Vec<Complex<f64>>> {
    let mut out: Vec<Vec<Complex<f64>>> = (0..n)
        .map(|k| (0..n).map(|i| Complex::new(if i == k { 1.0 } else { 0.0 }, 0.0)).collect())
        .collect();
    out.push(
        (0..n)
            .map(|i| Complex::new(0.5 + i as f64, -1.5 * (i as f64 + 1.0)))
            .collect(),
    );
    out
}

/// The shifted operator inverts the matrix the dense path forms.
///
/// This is the gate the sparse method rests on. `ShiftedDae::apply` claims to
/// be `(A − σI)⁻¹` without ever building `A`, reaching it through a complex
/// factorization of the *bordered* system and the Schur-complement identity;
/// `smallsignal::state_matrix` builds `A` explicitly by eliminating the network
/// column by column. The two share one thing — the `DaePattern` fill — and
/// nothing else, so checking that `(A − σI)·apply(b)` comes back to `b` tests
/// the whole formulation: the sign flip on the top rows, the `(1 − σ)` on the
/// state diagonal, the zero padding, and the identity itself.
///
/// Checked at three shifts including two well off the real axis, since at a
/// real σ a conjugation mistake is invisible.
#[test]
fn the_shifted_operator_inverts_the_dense_state_matrix() {
    for (name, system) in [("classical", smib(0.4).0), ("transient", smib_transient())] {
        let a = smallsignal::state_matrix(&system).expect("an equilibrium can be reduced");
        let n = a.len();

        for sigma in [
            Complex::new(-0.5, 0.0),
            Complex::new(-0.3, 6.0),
            Complex::new(0.2, -12.5),
        ] {
            let op = ShiftedDae::new(&system, sigma).expect("a shift off the spectrum");
            assert_eq!(op.n_states(), n, "{name}");
            assert_eq!(op.sigma(), sigma, "{name}");

            for b in probes(n) {
                let y = op.apply(&b).expect("the shifted system is nonsingular here");
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut acc = -sigma * y[i];
                    for j in 0..n {
                        acc += a[i][j] * y[j];
                    }
                    worst = worst.max((acc - b[i]).norm());
                }
                assert!(worst < 1e-10, "{name} at σ={sigma}: residual {worst:e}");
            }
        }
    }
}

/// And the adjoint inverts `(A − σI)ᴴ`, off the same factorization.
///
/// Kept separate from the forward gate because it fails differently: a plain
/// transpose where a conjugate transpose belongs passes every real-σ check and
/// every symmetric case, and shows up only here. `A` is real, so `(Aᴴ)ᵢⱼ` is
/// `Aⱼᵢ` — but `σ` is not, and the `σ̄` is the whole test.
#[test]
fn the_adjoint_operator_inverts_the_conjugate_transpose() {
    let system = smib_transient();
    let a = smallsignal::state_matrix(&system).unwrap();
    let n = a.len();

    for sigma in [Complex::new(-0.3, 6.0), Complex::new(0.2, -12.5)] {
        let op = ShiftedDae::new(&system, sigma).unwrap();
        for b in probes(n) {
            let y = op.apply_adjoint(&b).unwrap();
            let mut worst = 0.0f64;
            for i in 0..n {
                let mut acc = -sigma.conj() * y[i];
                for j in 0..n {
                    acc += a[j][i] * y[j];
                }
                worst = worst.max((acc - b[i]).norm());
            }
            assert!(worst < 1e-10, "σ={sigma}: adjoint residual {worst:e}");
        }

        // And it is genuinely the adjoint: at a complex shift the two
        // directions disagree, so a forward solve standing in for this one
        // would be caught.
        let e0 = probes(n).remove(0);
        let forward = op.apply(&e0).unwrap();
        let adjoint = op.apply_adjoint(&e0).unwrap();
        let gap = (0..n).fold(0.0f64, |m, i| m.max((forward[i] - adjoint[i]).norm()));
        assert!(gap > 1e-6, "σ={sigma}: forward and adjoint agree, {gap:e}");
    }
}

/// A shift sitting exactly on an eigenvalue makes the bordered system singular,
/// which is the one value a shift may not take. Reported rather than returned
/// as a NaN-poisoned vector, since `faer` will happily produce one.
#[test]
fn a_shift_on_the_spectrum_is_refused() {
    let (system, k_s) = smib(0.0);
    // The undamped swing pair sits exactly at ±j√(Ω_b·K_s/2H).
    let omega = (OMEGA_B * k_s / (2.0 * H)).sqrt();
    let on_it = Complex::new(0.0, omega);

    match ShiftedDae::new(&system, on_it) {
        Err(ShiftError::Singular { sigma }) => assert_eq!(sigma, on_it),
        // A shift landing *near* rather than *on* the eigenvalue still
        // factorizes; what must not happen is a silent NaN, so the operator is
        // then required to produce a finite answer.
        Ok(op) => {
            let b = probes(system.n_states()).remove(0);
            let y = op.apply(&b).expect("a factorization that succeeded must solve");
            assert!(y.iter().all(|c| c.re.is_finite() && c.im.is_finite()));
        }
        Err(other) => panic!("unexpected refusal: {other}"),
    }
}

/// Every eigenvalue the sparse method reports, given a shift.
fn modes_near(system: &DynamicSystem, sigma: Complex<f64>, count: usize) -> Vec<Complex<f64>> {
    let op = ShiftedDae::new(system, sigma).expect("a shift off the spectrum");
    let opts = ArnoldiOptions { count, ..Default::default() };
    let result = arnoldi::eigenpairs(op.n_states(), |b| op.apply(b), &opts)
        .expect("the operator applies");
    assert!(result.converged(), "unconverged at σ={sigma}: {:?}", result.pairs);
    // λ = σ + 1/θ undoes the shift-invert transform.
    result.pairs.iter().map(|p| sigma + 1.0 / p.theta).collect()
}

/// The distance from `lambda` to the nearest of `candidates`.
fn nearest(lambda: Complex<f64>, candidates: &[Complex<f64>]) -> f64 {
    candidates.iter().fold(f64::INFINITY, |m, c| m.min((c - lambda).norm()))
}

/// The sparse method finds the dense method's eigenvalues.
///
/// The dense path forms `A` and decomposes it whole; the sparse path never
/// forms `A` at all, reaching the same numbers through a Krylov space built out
/// of bordered complex solves. On a case small enough for both, the dense
/// answer is the oracle — and it is a strong one, because every eigenvalue the
/// sparse method returns must be *some* dense eigenvalue, and the ones nearest
/// the shift must be the ones it picked.
#[test]
fn the_sparse_method_finds_the_dense_eigenvalues() {
    for (name, system) in [("classical", smib(0.4).0), ("transient", smib_transient())] {
        let dense: Vec<Complex<f64>> = smallsignal::analyze(&system)
            .unwrap()
            .modes
            .iter()
            .map(|m| m.eigenvalue)
            .collect();

        for sigma in [Complex::new(-0.5, 3.0), Complex::new(-1.0, 0.0), Complex::new(0.0, 8.0)] {
            let sparse = modes_near(&system, sigma, 2);
            for lambda in &sparse {
                assert!(
                    nearest(*lambda, &dense) < 1e-8,
                    "{name} at σ={sigma}: {lambda} is not a dense eigenvalue (dense: {dense:?})"
                );
            }

            // And they are the *nearest* ones, not merely genuine ones. Ranking
            // the dense set by distance to σ must reproduce what came back.
            let mut ranked = dense.clone();
            ranked.sort_by(|a, b| {
                (a - sigma).norm().partial_cmp(&(b - sigma).norm()).unwrap()
            });
            for lambda in &sparse {
                assert!(
                    nearest(*lambda, &ranked[..sparse.len()]) < 1e-8,
                    "{name} at σ={sigma}: {lambda} is not among the {} nearest",
                    sparse.len()
                );
            }
        }
    }
}

/// A mode belongs to the system, not to the shift.
///
/// Three shifts around one eigenvalue — one below it, one above, one off to the
/// side — must all return the same number. This is the check that the transform
/// is being undone correctly rather than consistently wrongly: an error in
/// `λ = σ + 1/θ` that happened to be a constant offset would pass the
/// comparison above at a single shift and fails here.
#[test]
fn the_eigenvalue_does_not_depend_on_the_shift() {
    let (system, k_s) = smib(0.2);
    let omega = (OMEGA_B * k_s / (2.0 * H)).sqrt();

    let mut found: Vec<Complex<f64>> = Vec::new();
    for sigma in [
        Complex::new(-0.1, 0.6 * omega),
        Complex::new(-0.1, 1.4 * omega),
        Complex::new(-2.0, omega),
    ] {
        let modes = modes_near(&system, sigma, 1);
        found.push(modes[0]);
    }
    for pair in found.windows(2) {
        assert!(
            (pair[0] - pair[1]).norm() < 1e-9,
            "the same mode came back differently: {found:?}"
        );
    }
    assert!(found[0].im.abs() > 0.0, "expected the oscillatory mode, got {found:?}");
}

/// The closed form, through the sparse path.
///
/// `tests/dynamics_test.rs` reaches `±j√(Ω_b·K_s/2H)` by integrating and
/// measuring a period; `the_eigenvalue_is_the_closed_form` reaches it by a
/// dense decomposition. This reaches it a third way, through a Krylov space,
/// and the three share almost nothing but the model itself.
#[test]
fn the_sparse_method_recovers_the_closed_form() {
    let (system, k_s) = smib(0.0);
    let expected = (OMEGA_B * k_s / (2.0 * H)).sqrt();

    // Aimed near the pair but deliberately not at it, since a shift sitting on
    // an eigenvalue is the one place the operator does not exist.
    let sigma = Complex::new(-0.05, 0.9 * expected);
    let modes = modes_near(&system, sigma, 2);
    let oscillatory = modes
        .iter()
        .find(|m| m.im.abs() > 1e-6)
        .expect("the swing pair is what sits nearest this shift");

    let relative = (oscillatory.im.abs() - expected).abs() / expected;
    assert!(relative < 1e-9, "got {} against {expected}", oscillatory.im.abs());
    assert!(oscillatory.re.abs() < 1e-8, "undamped, but got σ = {}", oscillatory.re);
}

/// The sparse method on a case big enough for the Krylov space to matter.
///
/// The two fixtures above have two and four states, which is smaller than the
/// Krylov dimension — so Arnoldi runs to completion and is really a dense
/// decomposition of a small Hessenberg wearing a disguise. A 64-bus ring has
/// 320 differential states against a default dimension of 32, so the subspace
/// is a genuine projection and the machinery being tested is the machinery that
/// will run at scale: the orthogonalization, the restart, the selection.
///
/// Still small enough for the dense path, which is what makes it a gate rather
/// than a demonstration.
#[test]
fn the_sparse_method_matches_the_dense_one_on_a_real_subspace() {
    let document = dynamics_ring::ring(64, 0.4);
    let (system, _) = document.build().expect("the ring solves and initializes");
    let n_x = system.n_states();
    assert_eq!(n_x, 32 * dynamics_ring::STATES_PER_UNIT);

    let dense: Vec<Complex<f64>> = smallsignal::analyze(&system)
        .unwrap()
        .modes
        .iter()
        .map(|m| m.eigenvalue)
        .collect();
    assert_eq!(dense.len(), n_x);

    // Aimed at the electromechanical band — around 1 Hz, lightly damped, which
    // is where a stability study looks and where the interesting modes are.
    let sigma = Complex::new(-0.4, std::f64::consts::TAU);
    let op = ShiftedDae::new(&system, sigma).unwrap();
    let opts = ArnoldiOptions { count: 8, ..Default::default() };
    let result = arnoldi::eigenpairs(op.n_states(), |b| op.apply(b), &opts).unwrap();
    assert!(result.converged(), "unconverged: {:?}", result.pairs);
    assert!(
        result.dim < n_x,
        "the Krylov space spanned the whole system, so this proves nothing"
    );

    let mut ranked = dense.clone();
    ranked.sort_by(|a, b| (a - sigma).norm().partial_cmp(&(b - sigma).norm()).unwrap());

    for pair in &result.pairs {
        let lambda = sigma + 1.0 / pair.theta;
        assert!(
            nearest(lambda, &dense) < 1e-7,
            "{lambda} is not an eigenvalue of the dense A"
        );
        assert!(
            nearest(lambda, &ranked[..opts.count]) < 1e-7,
            "{lambda} is not among the {} nearest σ", opts.count
        );
        // The reported residual is in the units of λ, so it must actually bound
        // the error against the oracle.
        assert!(
            pair.residual < 1e-6,
            "{lambda}: residual {} does not look converged", pair.residual
        );
    }
}

/// A degenerate ring, and what the method honestly does with it.
///
/// Identical machines make the ring circulant, and its electromechanical modes
/// come in pairs that coincide to many digits. A Krylov space holds one vector
/// per invariant direction, so Arnoldi finds *one* of each pair and cannot see
/// the multiplicity — a block method is the answer and is not implemented.
///
/// This is gated rather than footnoted, because the failure is silent: the
/// eigenvalues that come back are all genuine, and nothing about them says a
/// second copy went unreported. What must hold is that every returned value is
/// real, converged, and distinct — never a spurious duplicate manufactured by a
/// basis that lost orthogonality, which is the failure mode that would look the
/// same to a caller and mean something entirely different.
#[test]
fn a_degenerate_ring_returns_distinct_converged_modes() {
    let document = dynamics_ring::ring(32, 0.0);
    let (system, _) = document.build().unwrap();

    let dense: Vec<Complex<f64>> = smallsignal::analyze(&system)
        .unwrap()
        .modes
        .iter()
        .map(|m| m.eigenvalue)
        .collect();

    let sigma = Complex::new(-0.4, std::f64::consts::TAU);
    let op = ShiftedDae::new(&system, sigma).unwrap();
    let opts = ArnoldiOptions { count: 6, ..Default::default() };
    let result = arnoldi::eigenpairs(op.n_states(), |b| op.apply(b), &opts).unwrap();

    let found: Vec<Complex<f64>> = result.pairs.iter().map(|p| sigma + 1.0 / p.theta).collect();
    for lambda in &found {
        assert!(nearest(*lambda, &dense) < 1e-6, "{lambda} is not a genuine eigenvalue");
    }
    // Distinct: a duplicated Ritz value here would be an orthogonality failure,
    // not a discovered multiplicity.
    for (i, a) in found.iter().enumerate() {
        for b in &found[i + 1..] {
            assert!(
                (a - b).norm() > 1e-9,
                "the same eigenvalue came back twice: {found:?}"
            );
        }
    }
}

/// A mode is the same object whichever method produced it.
///
/// Not just the eigenvalue — the participation factors and the mode shape too.
/// Those are what a caller actually reads, they are computed from eigenvectors
/// the two methods obtain in entirely different ways (one inverts `U`, the
/// other runs a second Arnoldi pass on the adjoint operator), and a difference
/// in convention between the two would be invisible in an eigenvalue
/// comparison and glaring to anyone using the results.
///
/// Checked on well-separated modes on purpose: where eigenvalues are nearly
/// degenerate the eigenvectors are genuinely not unique, so a disagreement
/// there would say nothing about either method.
#[test]
fn the_two_methods_report_the_same_mode() {
    let system = smib_transient();
    let dense = smallsignal::analyze(&system).unwrap();
    assert_eq!(dense.method, Method::Dense);

    let opts = SmallSignalOptions::near_frequency(1.2, 0.05).count(4);
    let sparse = smallsignal::analyze_near(&system, &opts).unwrap();
    assert!(matches!(sparse.method, Method::Sparse { .. }), "{}", sparse.method);
    assert!(sparse.converged);
    assert_eq!(sparse.state_names, dense.state_names);

    for mode in &sparse.modes {
        let twin = dense
            .modes
            .iter()
            .min_by(|a, b| {
                (a.eigenvalue - mode.eigenvalue)
                    .norm()
                    .partial_cmp(&(b.eigenvalue - mode.eigenvalue).norm())
                    .unwrap()
            })
            .unwrap();
        assert!(
            (twin.eigenvalue - mode.eigenvalue).norm() < 1e-9,
            "{} has no dense twin", mode.eigenvalue
        );

        // Damping, frequency and time constant are derived from the eigenvalue
        // by one shared function, so agreeing here is agreeing that it is
        // shared.
        assert!((twin.damping - mode.damping).abs() < 1e-9);
        assert!((twin.frequency - mode.frequency).abs() < 1e-9);

        // Participation: the same states, in the same order, with the same
        // factors.
        assert_eq!(
            twin.participation.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
            mode.participation.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
            "{}: participation ordering differs", mode.eigenvalue
        );
        for (a, b) in twin.participation.iter().zip(&mode.participation) {
            assert!(
                (a.1 - b.1).abs() < 1e-6,
                "{}: participation {} vs {}", mode.eigenvalue, a.1, b.1
            );
        }

        // Shape: normalized identically, so the components must match as
        // complex numbers and not merely in magnitude.
        assert_eq!(twin.shape.len(), mode.shape.len());
        for (a, b) in twin.shape.iter().zip(&mode.shape) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).norm() < 1e-6, "{}: shape {} vs {}", mode.eigenvalue, a.1, b.1);
        }

        // The eigenvectors: parallel, not equal. An eigenvector has no
        // distinguished phase, so what has to agree is the *direction* — and
        // for unit vectors that is `|⟨u, v⟩| = 1`, which is exactly the
        // statement that they span the same line.
        assert_eq!(twin.eigenvector.len(), mode.eigenvector.len());
        let overlap: Complex<f64> = twin
            .eigenvector
            .iter()
            .zip(&mode.eigenvector)
            .map(|(a, b)| a.conj() * b)
            .sum();
        assert!(
            (overlap.norm() - 1.0).abs() < 1e-6,
            "{}: eigenvectors are not parallel, |⟨u,v⟩| = {}",
            mode.eigenvalue,
            overlap.norm()
        );
    }

    // The dense method carries no residual to report; the sparse one does.
    assert!(dense.modes.iter().all(|m| m.residual == 0.0));
    assert!(sparse.modes.iter().all(|m| m.residual > 0.0 && m.residual < 1e-9));
}

/// The mode shape survives the sparse path, on the case that makes a shape mean
/// something.
///
/// Two islanded machines have exactly two rotor modes and they are opposites.
/// The dense gate above already checks that the shape separates them; this
/// checks the sparse method reaches the same conclusion, since a shape read off
/// a Ritz vector rather than an exact eigenvector is where a normalization
/// mistake would hide.
#[test]
fn the_sparse_method_reads_the_same_mode_shape() {
    let system = two_machine_island();
    let dense = smallsignal::analyze(&system).unwrap();
    let swing = dense
        .modes
        .iter()
        .find(|m| m.is_oscillatory())
        .expect("two islanded machines swing against each other");

    let hz = swing.frequency;
    let opts = SmallSignalOptions::near_frequency(hz, 0.05).count(2);
    let sparse = smallsignal::analyze_near(&system, &opts).unwrap();
    let found = sparse
        .modes
        .iter()
        .find(|m| m.is_oscillatory())
        .expect("aimed at the swing mode");

    assert!((found.eigenvalue - swing.eigenvalue).norm() < 1e-9);
    assert_eq!(found.shape.len(), 2, "one component per rotor");

    let phases: Vec<f64> = sparse.shape(found).iter().map(|&(_, _, phase)| phase).collect();
    let separation = (phases[0] - phases[1]).abs();
    assert!(
        (separation - 180.0).abs() < 5.0,
        "the two rotors should swing against each other, got {separation}°"
    );
}

/// The sparse method's eigenvalue predicts what the ring actually does.
///
/// Every gate above compares the sparse method against the dense one, against a
/// closed form, or against itself. This compares it against a **trajectory** —
/// the same independent evidence §19 used for the dense method, now for the
/// method that runs where the dense one cannot.
///
/// The trick is to excite one mode rather than all of them. A fault on a ring
/// of thirty-two machines rings every mode at once and no period can be read
/// out of the result; displacing the state *along the eigenvector* excites that
/// mode and almost nothing else, and its own period and decay then show up in a
/// single machine's angle. Done that way it holds to the same tolerances the
/// dense method's own trajectory gate uses — `2e-3` on the period and 2% on the
/// decay — which is the point: the sparse method is not being graded on a
/// curve.
#[test]
fn the_sparse_eigenvalue_predicts_the_rings_own_oscillation() {
    let document = dynamics_ring::ring(64, 0.4);
    let (mut system, _) = document.build().unwrap();

    let opts = SmallSignalOptions::near_frequency(1.0, 0.05).count(4);
    let result = smallsignal::analyze_near(&system, &opts).unwrap();
    assert!(result.converged);
    let mode = result
        .modes
        .iter()
        .find(|m| m.is_oscillatory())
        .expect("the electromechanical band is what this shift is aimed at")
        .clone();
    let predicted_period = 1.0 / mode.frequency;
    assert!(mode.shape.len() > 8, "an inter-area mode moves many rotors");

    // Displace the *whole state* along the mode, which is what makes this
    // excite one mode rather than a dozen.
    //
    // Nudging only the rotor angles does not work, and the failures are worth
    // recording because both look like physics: angles alone came out 3.2% off
    // the predicted period, and angles-plus-speeds — with the speeds derived
    // correctly from `δ̇ = Ω_b(ω − 1)` — came out 6.4% off, *worse*. Neither is
    // a point on the mode. A sixth-order machine with an exciter and a governor
    // carries ten states, and an electromechanical mode moves all of them; the
    // flux and control states left sitting at equilibrium are a displacement
    // along every other mode that touches them.
    //
    // The real part of the eigenvector is the displacement at one instant of
    // the oscillation, which is exactly what a starting condition is.
    let about: Vec<f64> = system.state().to_vec();
    let epsilon = 1e-3;
    for (k, component) in mode.eigenvector.iter().enumerate() {
        system.state_mut()[k] += epsilon * component.re;
    }
    let run = DynamicsOptions {
        end_time: 8.0,
        step: 0.002,
        damping_steps: 0,
        ..Default::default()
    };
    settle(&mut system, &run);
    let report = run_dynamics(&mut system, &run);
    assert_eq!(report.status, DynamicsStatus::Completed);

    // Read it at the rotor the mode moves most, which is where the mode is
    // least contaminated by the others.
    let (state, _) = mode
        .shape
        .iter()
        .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
        .unwrap();
    let name = &result.state_names[*state];
    let series = report.trajectory.series(name).unwrap();
    let time = &report.trajectory.time;
    let centre = about[*state];

    let mut zeros = Vec::new();
    for k in 1..series.len() {
        let (a, b) = (series[k - 1] - centre, series[k] - centre);
        if (a < 0.0) != (b < 0.0) {
            zeros.push(time[k - 1] + (a / (a - b)) * (time[k] - time[k - 1]));
        }
    }
    assert!(zeros.len() >= 6, "expected several crossings in 8 s, got {}", zeros.len());
    let observed = 2.0 * (zeros[zeros.len() - 1] - zeros[0]) / (zeros.len() - 1) as f64;
    assert!(
        (observed - predicted_period).abs() / predicted_period < 2e-3,
        "the eigenvalue predicts a period of {predicted_period:.5} s; \
         the run showed {observed:.5} s"
    );

    // And the real part predicts how fast it dies away. Peaks between
    // consecutive crossings, so the envelope is sampled once per half cycle.
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for window in zeros.windows(2) {
        let (from, to) = (window[0], window[1]);
        let mut best = (0.0f64, 0.0f64);
        for (k, &t) in time.iter().enumerate() {
            if t > from && t < to {
                let amplitude = (series[k] - centre).abs();
                if amplitude > best.1 {
                    best = (t, amplitude);
                }
            }
        }
        if best.1 > 0.0 {
            peaks.push(best);
        }
    }
    assert!(peaks.len() >= 4, "expected several peaks, got {}", peaks.len());
    let (t0, a0) = peaks[0];
    let (t1, a1) = peaks[peaks.len() - 1];
    let expected = (mode.eigenvalue.re * (t1 - t0)).exp();
    let observed = a1 / a0;
    assert!(
        (observed - expected).abs() / expected < 0.02,
        "σ = {} predicts the amplitude falls to {expected:.3e} of itself over {:.2} s; \
         the run showed {observed:.3e}",
        mode.eigenvalue.re,
        t1 - t0
    );
}

/// The inertia sensitivity is the closed form.
///
/// For an undamped classical machine the swing eigenvalue is
/// `λ = ±j√(Ω_b·K_s/2H)`, so `dλ/dH = −λ/2H` exactly — the same closed form the
/// eigenvalue itself is gated against, differentiated. `K_s` does not depend on
/// `H`, which is what makes the derivative that clean and is also why `H` is a
/// parameter this method can honestly answer about at all.
#[test]
fn the_inertia_sensitivity_is_the_closed_form() {
    let (mut system, _) = smib(0.0);
    let result = smallsignal::analyze(&system).unwrap();
    let mode = result
        .modes
        .iter()
        .find(|m| m.eigenvalue.im > 0.0)
        .expect("the swing pair")
        .clone();

    let parameters = smallsignal::tunable_parameters(&system);
    assert_eq!(
        parameters,
        vec![
            ParameterRef { device: 0, name: "h".to_string() },
            ParameterRef { device: 0, name: "d".to_string() },
        ],
        "a classical machine's tunable set is its inertia and its damping"
    );

    let sensitivity = smallsignal::sensitivities(&mut system, &mode, &parameters).unwrap();
    let inertia = &sensitivity[0];
    assert_eq!(inertia.value, H);

    let expected = -mode.eigenvalue / (2.0 * H);
    let error = (inertia.d_eigenvalue - expected).norm() / expected.norm();
    assert!(
        error < 1e-6,
        "dλ/dH should be −λ/2H = {expected}, got {}",
        inertia.d_eigenvalue
    );

    // Raising the inertia slows the oscillation, and does not damp it.
    assert!(inertia.d_frequency < 0.0, "{}", inertia.d_frequency);
    assert!(inertia.d_damping.abs() < 1e-9, "nothing dissipates, so nothing damps it");
}

/// And it agrees with a difference of two whole analyses.
///
/// The closed form above checks the formula against theory on the one case that
/// has a theory. This checks it against the *method itself*: build the system
/// twice with `H ± δ`, analyze each from scratch, and difference the
/// eigenvalues. The two routes share only the model — one goes through a
/// quadratic form in the pencil's eigenvectors, the other through two complete
/// decompositions — which is the same shape of evidence
/// `tests/ac_sensitivity_test.rs` uses for the power flow's derivatives.
#[test]
fn the_sensitivity_matches_a_difference_of_two_analyses() {
    let delta = 1e-4;
    let (mut system, _) = smib(1.5);
    let mode = smallsignal::analyze(&system)
        .unwrap()
        .modes
        .iter()
        .find(|m| m.eigenvalue.im > 0.0)
        .unwrap()
        .clone();

    let parameters = smallsignal::tunable_parameters(&system);
    let sensitivity = smallsignal::sensitivities(&mut system, &mode, &parameters).unwrap();

    // Inertia.
    let up = swing_eigenvalue(&smib_with(H + delta, 1.5).0);
    let down = swing_eigenvalue(&smib_with(H - delta, 1.5).0);
    let numeric = (up - down) / (2.0 * delta);
    let error = (sensitivity[0].d_eigenvalue - numeric).norm() / numeric.norm();
    assert!(
        error < 1e-6,
        "dλ/dH: {} against a difference of {numeric}",
        sensitivity[0].d_eigenvalue
    );

    // Damping coefficient. A different row of the Jacobian, and the one whose
    // answer a study actually wants — more `D` is more damping.
    let up = swing_eigenvalue(&smib_with(H, 1.5 + delta).0);
    let down = swing_eigenvalue(&smib_with(H, 1.5 - delta).0);
    let numeric = (up - down) / (2.0 * delta);
    let error = (sensitivity[1].d_eigenvalue - numeric).norm() / numeric.norm();
    assert!(
        error < 1e-6,
        "dλ/dD: {} against a difference of {numeric}",
        sensitivity[1].d_eigenvalue
    );
    assert!(sensitivity[1].d_damping > 0.0, "more damping coefficient, more damping ratio");
}

/// The sparse method's modes carry sensitivities too, and the same ones.
#[test]
fn the_sparse_method_supports_sensitivities() {
    let (mut system, k_s) = smib(1.5);
    let dense = smallsignal::analyze(&system).unwrap();
    let dense_mode = dense.modes.iter().find(|m| m.eigenvalue.im > 0.0).unwrap().clone();

    let hz = (OMEGA_B * k_s / (2.0 * H)).sqrt() / std::f64::consts::TAU;
    let opts = SmallSignalOptions::near_frequency(hz, 0.05).count(2);
    let sparse = smallsignal::analyze_near(&system, &opts).unwrap();
    let sparse_mode = sparse.modes.iter().find(|m| m.eigenvalue.im > 0.0).unwrap().clone();
    assert!((sparse_mode.eigenvalue - dense_mode.eigenvalue).norm() < 1e-9);
    assert!(!sparse_mode.left_eigenvector.is_empty(), "paired, so it has a left vector");

    let parameters = smallsignal::tunable_parameters(&system);
    let from_dense = smallsignal::sensitivities(&mut system, &dense_mode, &parameters).unwrap();
    let from_sparse = smallsignal::sensitivities(&mut system, &sparse_mode, &parameters).unwrap();

    for (a, b) in from_dense.iter().zip(&from_sparse) {
        let error = (a.d_eigenvalue - b.d_eigenvalue).norm() / a.d_eigenvalue.norm();
        assert!(error < 1e-8, "{}: {} vs {}", a.parameter.name, a.d_eigenvalue, b.d_eigenvalue);
    }
}

/// A parameter nothing has, and a device that is not there, are both named.
#[test]
fn an_unknown_parameter_is_refused() {
    let (mut system, _) = smib(1.0);
    let mode = smallsignal::analyze(&system).unwrap().modes[0].clone();

    for bad in [
        ParameterRef { device: 0, name: "xdp".to_string() },
        ParameterRef { device: 7, name: "h".to_string() },
    ] {
        match smallsignal::sensitivities(&mut system, &mode, std::slice::from_ref(&bad)) {
            Err(SmallSignalError::NoSuchParameter(named)) => assert_eq!(named, bad),
            other => panic!("expected a refusal for {bad:?}, got {other:?}"),
        }
    }

    // A reactance is refused on purpose rather than by omission: changing it
    // moves the equilibrium, and a fixed-point sensitivity cannot see that.
    let err = smallsignal::sensitivities(
        &mut system,
        &mode,
        &[ParameterRef { device: 0, name: "xdp".to_string() }],
    )
    .unwrap_err();
    assert!(err.to_string().contains("no tunable parameter"), "{err}");

    // And the system is unharmed by a refused request.
    assert_eq!(smallsignal::analyze(&system).unwrap().modes[0].eigenvalue, mode.eigenvalue);
}
