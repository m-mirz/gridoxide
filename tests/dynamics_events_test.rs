//! Phase-2 gates for `src/dynamics/`: events, the algebraic re-solve at a
//! discontinuity, and the closed form they exist to reproduce.
//!
//! The headline is the **critical clearing time**. It is the question
//! transient stability is asked in practice — how long can a fault stay on
//! before the machine falls out of step — and for a classical machine against
//! an infinite bus, with the fault at the machine terminal, it has a closed
//! form from the equal-area criterion. No reference implementation is
//! involved, and the quantity checked is the one the simulator exists to
//! compute.

use num_complex::Complex;

use gridoxide::dynamics::models::machine::{self, GenCls, GenClsParams};
use gridoxide::dynamics::{
    build, run_dynamics, DeviceSpec, DynamicSystem, DynamicsOptions, DynamicsStatus,
    DynamicsWarning, Event, EventKind, SystemSpec,
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

fn bus(idx: usize, bus_type: BusType, p: f64, q: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: q,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: Vec::new(),
    }
}

/// What the equal-area criterion predicts for this machine.
struct Analytic {
    p_max: f64,
    p_m: f64,
    delta_0: f64,
    t_cc: f64,
}

/// One machine at bus 0, a lossless line, an infinite bus at 1∠0.
fn smib() -> (DynamicSystem, Analytic) {
    let buses = vec![
        bus(0, BusType::PV, P_GEN, 0.0),
        bus(1, BusType::Slack, 0.0, 0.0),
    ];
    let lines = vec![Line { from: 0, to: 1, r: 0.0, x: X_LINE, b_shunt: 0.0, g_shunt: 0.0 }];

    let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
        buses,
        lines: lines.clone(),
    });
    let buses = report.buses;
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();
    let (p_calc, q_calc) = power_injections(&buses, &ybus);
    let s_dev = Complex::new(p_calc[0], q_calc[0]);

    let model = GenCls::new(
        GenClsParams { h: H, d: 0.0, ra: 0.0, xdp: XDP, mbase: S_BASE },
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
            model: machine::bare(Box::new(model)),
        }],
        fixed_buses: vec![1],
    })
    .unwrap();

    // With r_a = 0 there is no stator loss, so the air-gap power the rotor
    // feels is exactly the terminal power the power flow scheduled.
    let p_m = p_calc[0];
    let e_mag = {
        let v0 = system.voltages()[0];
        let i = (s_dev / v0).conj();
        (v0 + Complex::new(0.0, XDP) * i).norm()
    };
    let p_max = e_mag / (XDP + X_LINE);
    let delta_0 = system.state()[0];

    // Equal-area criterion, for a fault that removes the electrical power
    // entirely and a post-fault network identical to the pre-fault one:
    //
    //   A_accel = P_m(δ_cc − δ₀)
    //   A_decel = P_max(cos δ_cc − cos δ_max) − P_m(δ_max − δ_cc)
    //
    // Setting them equal gives cos δ_cc directly. With P_e = 0 during the
    // fault the swing equation integrates exactly, δ(t) = δ₀ + Ω_b·P_m·t²/4H,
    // which inverts for the time.
    let delta_max = std::f64::consts::PI - delta_0;
    let cos_cc = delta_max.cos() + (p_m / p_max) * (delta_max - delta_0);
    let delta_cc = cos_cc.acos();
    let t_cc = (4.0 * H * (delta_cc - delta_0) / (OMEGA_B * p_m)).sqrt();

    (system, Analytic { p_max, p_m, delta_0, t_cc })
}

fn options(step: f64, end: f64, events: Vec<Event>) -> DynamicsOptions {
    DynamicsOptions { end_time: end, step, events, ..Default::default() }
}

const FAULT_AT: f64 = 1.0;

/// Runs the fault-and-clear and reports whether the machine stayed in step.
///
/// A stable machine's angle peaks below `δ_max` (under π here) and swings back;
/// an unstable one passes `δ_max`, never decelerates again, and runs away. `2π`
/// separates the two by a wide margin, so the verdict never depends on where
/// exactly the threshold sits.
fn stays_in_step(clearing: f64, step: f64) -> bool {
    let (mut system, _) = smib();
    let opts = options(
        step,
        FAULT_AT + clearing + 4.0,
        vec![
            Event::bolted_fault(FAULT_AT, 0),
            Event::new(FAULT_AT + clearing, EventKind::ClearFault { bus: 0 }),
        ],
    );
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed, "clearing at {clearing}");
    assert_eq!(report.events_applied, 2);
    let delta = report.trajectory.series("G1.delta").unwrap();
    delta.iter().fold(0.0f64, |m, d| m.max(*d)) < std::f64::consts::TAU
}

/// G2. The headline: the simulated critical clearing time matches the
/// equal-area criterion's closed form.
#[test]
fn critical_clearing_time_matches_the_equal_area_criterion() {
    let (_, analytic) = smib();

    // The setup must be the one the closed form describes: at equilibrium the
    // machine sits on its own power-angle curve.
    let on_curve = analytic.p_max * analytic.delta_0.sin();
    assert!(
        (on_curve - analytic.p_m).abs() < 1e-9,
        "P_max·sin δ₀ = {on_curve:.9} should be P_m = {:.9}",
        analytic.p_m
    );
    assert!(
        analytic.t_cc > 0.2 && analytic.t_cc < 0.5,
        "the closed form should land in a plausible band, got {:.4} s",
        analytic.t_cc
    );

    let step = 0.002;
    let (mut lo, mut hi) = (0.15, 0.45);
    assert!(stays_in_step(lo, step), "the machine must survive a short fault");
    assert!(!stays_in_step(hi, step), "the machine must not survive a long one");

    // 14 halvings of a 0.30 s bracket leaves the answer located to ~18 µs,
    // well inside the 1 ms the gate asks for.
    for _ in 0..14 {
        let mid = 0.5 * (lo + hi);
        if stays_in_step(mid, step) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let measured = 0.5 * (lo + hi);

    assert!(
        (measured - analytic.t_cc).abs() < 1e-3,
        "simulated critical clearing time {measured:.5} s differs from the \
         equal-area criterion's {:.5} s by {:.2} ms",
        analytic.t_cc,
        (measured - analytic.t_cc).abs() * 1e3
    );
}

/// G8. At a discontinuity the algebraic variables jump and the differential
/// ones do not — bit-identically, not approximately. Both points are recorded
/// at the same time value, so the jump is visible rather than smoothed away.
#[test]
fn state_is_continuous_across_an_event_and_voltage_is_not() {
    let (mut system, _) = smib();
    let opts = options(0.005, 2.0, vec![Event::bolted_fault(FAULT_AT, 0)]);
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);

    let at_fault: Vec<usize> = report
        .trajectory
        .time
        .iter()
        .enumerate()
        .filter(|(_, t)| (**t - FAULT_AT).abs() < 1e-12)
        .map(|(k, _)| k)
        .collect();
    assert_eq!(at_fault.len(), 2, "the fault instant should be recorded twice");
    let (before, after) = (at_fault[0], at_fault[1]);

    let col = |name: &str| report.trajectory.column(name).unwrap();
    for state in ["G1.delta", "G1.omega"] {
        let c = col(state);
        assert_eq!(
            report.trajectory.rows[before][c].to_bits(),
            report.trajectory.rows[after][c].to_bits(),
            "{state} must be bit-identical across the jump"
        );
    }

    let vmag = col("bus0.vmag");
    let (v_before, v_after) =
        (report.trajectory.rows[before][vmag], report.trajectory.rows[after][vmag]);
    assert!(v_before > 0.9, "terminal voltage before the fault was {v_before}");
    assert!(v_after < 1e-4, "a bolted fault should collapse the terminal voltage, got {v_after}");

    // And the constraint is satisfied again on the far side, to machine zero
    // rather than to the solver's tolerance.
    assert!(
        system.network_residual_norm() < 1e-9,
        "residual after the run: {:e}",
        system.network_residual_norm()
    );
}

/// During a terminal fault on a machine with no armature resistance the
/// electrical power is zero, which is the assumption the closed form rests on.
/// With `P_e` gone the acceleration is constant, the rotor speed is linear in
/// time, and the angle is exactly quadratic.
///
/// The trapezoidal rule reproduces that **exactly** — it integrates a linear
/// integrand with no error at all — so there is no discretization error to
/// hide behind and every departure has to be accounted for. Two things
/// account for all of it, and separating them is the point of this test,
/// because at the default settings the two happen to be nearly the same size
/// and are easy to mistake for one another:
///
/// - **A fault admittance is finite.** `1e6` per unit leaves a residual
///   terminal voltage near `8e-6` and so a residual `P_e`, which *decelerates*
///   the rotor and grows as `t²`. It scales as `1/y_fault`, which is what
///   identifies it.
/// - **Backward Euler overshoots.** Each damping step evaluates `δ̇` at the end
///   of the step, so on a linearly growing speed it overshoots by exactly
///   `Ω_b·a·h²/2` with `a = P_m/2H`. Two steps leave a constant offset of
///   `Ω_b·a·h²` that never decays, since the speed itself stays exact. It does
///   not depend on the fault at all.
///
/// At `h = 1 ms` and `y = 1e6` those are `2.7e-5` and `2.5e-5` rad
/// respectively, with opposite signs. Measuring only their combination tells
/// you nothing; measuring each against its own closed form pins the
/// integrator, the event handling and the swing equation together.
#[test]
fn a_terminal_fault_removes_the_electrical_power() {
    let departure = |damping_steps: usize, step: f64, y_fault: f64| {
        let (mut system, analytic) = smib();
        let opts = DynamicsOptions {
            damping_steps,
            ..options(
                step,
                FAULT_AT + 0.25,
                vec![Event::new(
                    FAULT_AT,
                    EventKind::BusFault { bus: 0, y: Complex::new(y_fault, 0.0) },
                )],
            )
        };
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);

        let time = &report.trajectory.time;
        let delta = report.trajectory.series("G1.delta").unwrap();
        let accel = OMEGA_B * analytic.p_m / (4.0 * H);

        let mut worst: f64 = 0.0;
        for (k, &t) in time.iter().enumerate() {
            if t <= FAULT_AT + 1e-12 {
                continue;
            }
            let dt = t - FAULT_AT;
            worst = worst.max((delta[k] - (analytic.delta_0 + accel * dt * dt)).abs());
        }
        (worst, analytic.p_m)
    };

    let step = 0.001;

    // With no damping steps, all that is left is the residual electrical power
    // the finite fault admittance lets through — so it must scale with it.
    let (coarse, _) = departure(0, step, 1e6);
    let (stiff, _) = departure(0, step, 1e8);
    let ratio = coarse / stiff;
    assert!(
        (50.0..200.0).contains(&ratio),
        "a hundred-fold stiffer fault should shrink the departure a hundred-fold; \
         got {coarse:e} then {stiff:e}, a ratio of {ratio:.1}"
    );
    assert!(
        stiff < 1e-6,
        "with a stiff fault the trapezoidal rule should be exact here, departed by {stiff:e}"
    );

    // With the damping on and the fault stiff enough that the term above is
    // negligible, what remains is backward Euler's own overshoot.
    let (damped, p_m) = departure(2, step, 1e8);
    let predicted = OMEGA_B * (p_m / (2.0 * H)) * step * step;
    assert!(
        (damped - predicted).abs() / predicted < 0.01,
        "the damping offset should be Ω_b·(P_m/2H)·h² = {predicted:e} rad, measured {damped:e}"
    );

    // And it falls as h², even though backward Euler is a first-order rule,
    // because only a fixed number of steps ever use it.
    let (finer, _) = departure(2, step / 2.0, 1e8);
    let order = (damped / finer).log2();
    assert!(
        (order - 2.0).abs() < 0.05,
        "the damping offset should fall as h², observed order {order:.3}"
    );
}

/// The backward-Euler damping does not distort the answer.
///
/// It cannot yet *improve* it either: ringing needs a mode fast enough that
/// `h·λ` is large, and a classical machine's swing at about 1 Hz is nowhere
/// near that for a 2 ms step. The damping is wired in for the exciters and
/// governors of phase 3, whose time constants are two orders faster. What is
/// worth pinning today is that turning it on and off does not move the
/// critical clearing time.
#[test]
fn damping_steps_do_not_change_the_answer_yet() {
    let clearing = 0.30;
    let peak = |damping_steps: usize| {
        let (mut system, _) = smib();
        let opts = DynamicsOptions {
            damping_steps,
            ..options(
                0.002,
                FAULT_AT + clearing + 2.0,
                vec![
                    Event::bolted_fault(FAULT_AT, 0),
                    Event::new(FAULT_AT + clearing, EventKind::ClearFault { bus: 0 }),
                ],
            )
        };
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        report.trajectory.series("G1.delta").unwrap().iter().fold(0.0f64, |m, d| m.max(*d))
    };

    let (none, damped) = (peak(0), peak(2));
    assert!(
        (none - damped).abs() < 5e-4,
        "peak angle {none:.6} with no damping vs {damped:.6} with two steps"
    );
}

/// An event time need not be a multiple of the step. The step is truncated to
/// land on it exactly, and the recorded time is the event's own — snapped, not
/// accumulated, so it does not drift with the number of steps that preceded it.
#[test]
fn an_event_lands_exactly_on_its_own_time() {
    let (mut system, _) = smib();
    let odd = 0.7071;
    let opts = options(0.005, 1.0, vec![Event::bolted_fault(odd, 0)]);
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.events_applied, 1);

    let hits = report.trajectory.time.iter().filter(|t| **t == odd).count();
    assert_eq!(hits, 2, "the event time should appear exactly twice, bit-exact");
}

/// G9. Switching a network apart leaves an island with no machine and no
/// voltage reference. It still solves — every bus has a path to ground through
/// its own load — but the answer is a de-energized island, not a dynamic one,
/// and the run says so rather than presenting it as a trajectory.
#[test]
fn a_switched_off_island_is_reported() {
    let buses = vec![
        bus(0, BusType::PV, P_GEN, 0.0),
        bus(1, BusType::Slack, 0.0, 0.0),
        bus(2, BusType::PQ, -0.3, -0.1),
    ];
    let lines = vec![
        Line { from: 0, to: 1, r: 0.0, x: X_LINE, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.01, x: 0.1, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
        buses,
        lines: lines.clone(),
    });
    assert_eq!(report.stats.status, gridoxide::solver::SolveStatus::Converged);
    let buses = report.buses;
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();
    let (p_calc, q_calc) = power_injections(&buses, &ybus);

    let model = GenCls::new(
        GenClsParams { h: H, d: 0.0, ra: 0.0, xdp: XDP, mbase: S_BASE },
        S_BASE,
        F_NOM,
    )
    .unwrap();
    let mut system = build(SystemSpec {
        buses: &buses,
        lines: &lines,
        transformers: &[],
        shunts: &[],
        devices: vec![DeviceSpec {
            id: "G1".to_string(),
            bus: 0,
            s: Complex::new(p_calc[0], q_calc[0]),
            model: machine::bare(Box::new(model)),
        }],
        fixed_buses: vec![1],
    })
    .unwrap();

    // Branch 1 is the only thing holding bus 2 on.
    let opts = options(0.005, 1.0, vec![Event::new(0.5, EventKind::BranchTrip { branch: 1 })]);
    let out = run_dynamics(&mut system, &opts);
    assert_eq!(out.status, DynamicsStatus::Completed);
    assert_eq!(out.events_applied, 1);

    let dead: Vec<&DynamicsWarning> = out
        .warnings
        .iter()
        .filter(|w| matches!(w, DynamicsWarning::DeadIsland { .. }))
        .collect();
    assert_eq!(dead.len(), 1, "expected one dead island, got {:?}", out.warnings);
    match dead[0] {
        DynamicsWarning::DeadIsland { buses, time } => {
            assert_eq!(buses, &vec![2]);
            assert_eq!(*time, 0.5);
        }
        other => panic!("unexpected warning {other:?}"),
    }

    // The machine is untouched by a trip on the far side of the infinite bus.
    let delta = out.trajectory.series("G1.delta").unwrap();
    let drift = delta.iter().fold(0.0f64, |m, d| m.max((d - delta[0]).abs()));
    assert!(drift < 1e-9, "the machine should not notice, but moved {drift:e} rad");
}

/// An event naming something that does not exist is a data error, not a crash.
/// It is skipped, named, and the run continues.
#[test]
fn an_impossible_event_is_skipped_and_named() {
    let (mut system, _) = smib();
    let opts = options(
        0.01,
        0.5,
        vec![
            Event::bolted_fault(0.1, 7),
            Event::new(0.2, EventKind::BranchTrip { branch: 9 }),
        ],
    );
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);
    assert_eq!(report.events_applied, 0);
    assert_eq!(report.warnings.len(), 2);
    for warning in &report.warnings {
        assert!(matches!(warning, DynamicsWarning::Skipped { .. }), "got {warning:?}");
    }
}


