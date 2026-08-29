//! Protection relays: actions triggered by a **state** rather than by a time.
//!
//! Two properties carry the design, and neither is obvious from the code:
//!
//! - **The crossing is located, not rounded.** A relay's delay is measured from
//!   the instant its threshold was crossed, so an error there is an error in
//!   when it acts. The step is shortened onto the crossing with the same
//!   Illinois locator `continuation` uses to find a reactive limit along its
//!   curve, and the located time is checked against the trajectory rather than
//!   taken on trust.
//! - **A relay that stops seeing its condition forgets.** That is what
//!   distinguishes it from a stopwatch, and it is what makes a fault cleared in
//!   time not trip anything.

use num_complex::Complex;

use gridoxide::dynamics::models::machine::{GenCls, GenClsParams};
use gridoxide::dynamics::models::GeneratingUnit;
use gridoxide::dynamics::{
    build, run_dynamics, DeviceSpec, DynamicSystem, DynamicsOptions, DynamicsStatus,
    DynamicsWarning, Event, EventKind, Relay, SystemSpec, Trigger, Watch,
};
use gridoxide::network::{build_ybus, power_injections};
use gridoxide::types::{Bus, BusType, Line};

const S_BASE: f64 = 100.0;
const F_NOM: f64 = 50.0;

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

/// A machine, a load bus, and an infinite bus — two lines in from the infinite
/// bus so that one can be tripped without islanding anything.
fn network() -> DynamicSystem {
    let buses = vec![
        bus(0, BusType::PV, 0.8, 0.0),
        bus(1, BusType::PQ, -0.8, -0.25),
        bus(2, BusType::Slack, 0.0, 0.0),
    ];
    let lines = vec![
        Line { from: 0, to: 1, r: 0.005, x: 0.08, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.005, x: 0.08, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.005, x: 0.16, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
        buses,
        lines: lines.clone(),
    });
    assert_eq!(report.stats.status, gridoxide::solver::SolveStatus::Converged);
    let buses = report.buses;
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();
    let (p_calc, q_calc) = power_injections(&buses, &ybus);

    let machine = GenCls::new(
        GenClsParams { h: 5.0, d: 2.0, ra: 0.0, xdp: 0.3, mbase: S_BASE },
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
            model: Box::new(GeneratingUnit::machine_only(Box::new(machine))),
        }],
        fixed_buses: vec![2],
    })
    .unwrap()
}

fn options(end: f64, events: Vec<Event>, relays: Vec<Relay>) -> DynamicsOptions {
    DynamicsOptions { end_time: end, step: 0.005, events, relays, ..Default::default() }
}

/// An under-voltage relay trips a line after its delay, and the delay is
/// measured from the **located** crossing rather than from a step boundary.
#[test]
fn a_relay_fires_a_delay_after_the_located_crossing() {
    let mut system = network();
    let relay = Relay::under_voltage("uv1", 1, 0.85, 0.15, EventKind::BranchTrip { branch: 2 });
    let opts = options(
        4.0,
        vec![
            Event::bolted_fault(1.0, 1),
            Event::new(2.5, EventKind::ClearFault { bus: 1 }),
        ],
        vec![relay],
    );
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);
    assert_eq!(report.relay_actions.len(), 1, "{:?}", report.relay_actions);

    let action = &report.relay_actions[0];
    assert_eq!(action.id, "uv1");
    assert_eq!(action.action, EventKind::BranchTrip { branch: 2 });
    // The fault lands at t = 1.0 and the voltage collapses within the instant,
    // so the crossing is essentially there and the trip 150 ms later.
    assert!(
        (action.crossed_at - 1.0).abs() < 1e-3,
        "crossed at {} rather than at the fault",
        action.crossed_at
    );
    assert!(
        (action.fired_at - action.crossed_at - 0.15).abs() < 1e-9,
        "the delay must be exactly what was asked: {} to {}",
        action.crossed_at,
        action.fired_at
    );

    // And the located crossing is a real crossing: the trajectory is above the
    // threshold just before it and below just after.
    let time = &report.trajectory.time;
    let v = report.trajectory.series("bus1.vmag").unwrap();
    let before = time.iter().rposition(|t| *t < action.crossed_at - 1e-6).unwrap();
    let after = time.iter().position(|t| *t > action.crossed_at + 1e-6).unwrap();
    assert!(v[before] > 0.85, "before the crossing: {}", v[before]);
    assert!(v[after] < 0.85, "after the crossing: {}", v[after]);
}

/// A crossing driven by the machine's own dynamics is located **inside** the
/// step, to far better than a step.
///
/// This is the case a step-boundary detection gets wrong, and it has to be
/// watched on a *differential* state to exist at all: an algebraic quantity
/// like a bus voltage moves discontinuously at an event, so its crossings tend
/// to sit exactly on event times. A rotor speed is integrated, so it crosses a
/// threshold somewhere strictly inside a step, and the located time must
/// reproduce the threshold far more precisely than the step size could.
#[test]
fn a_smooth_crossing_is_located_inside_the_step() {
    let mut system = network();
    let threshold = 1.0015;
    let relay = Relay {
        id: "os1".to_string(),
        watch: Watch::UnitSpeed { unit: 0 },
        trigger: Trigger::Above(threshold),
        delay: 0.0,
        action: EventKind::BranchTrip { branch: 2 },
        repeating: false,
    };
    let step = 0.02;
    let opts = DynamicsOptions {
        step,
        ..options(
            4.0,
            vec![
                Event::bolted_fault(1.0, 1),
                Event::new(1.1, EventKind::ClearFault { bus: 1 }),
            ],
            vec![relay],
        )
    };
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);
    assert_eq!(report.relay_actions.len(), 1, "{:?}", report.relay_actions);
    let action = &report.relay_actions[0];
    assert!((action.fired_at - action.crossed_at).abs() < 1e-12, "a zero delay acts at once");

    // The located time is a recorded point, and the speed there is the
    // threshold — to far better than a 20 ms step could have given.
    let time = &report.trajectory.time;
    let omega = report.trajectory.series("G1.omega").unwrap();
    let at = time
        .iter()
        .position(|t| (t - action.crossed_at).abs() < 1e-12)
        .expect("the located crossing is a recorded point");
    assert!(
        (omega[at] - threshold).abs() < 1e-7,
        "the located crossing should sit on the threshold: ω = {} against {threshold}",
        omega[at]
    );

    // And it is genuinely inside a step rather than on a boundary — which is
    // the whole point of locating it.
    let fraction = (action.crossed_at / step).fract();
    assert!(
        fraction > 1e-3 && fraction < 1.0 - 1e-3,
        "the crossing should fall inside a step, not on a boundary: t = {}",
        action.crossed_at
    );
}

/// A relay whose condition stops holding **forgets**, rather than accumulating
/// time towards a trip it should never make.
///
/// This is what separates selective protection from a stopwatch: a fault
/// cleared inside the relay's delay must leave it having done nothing.
#[test]
fn a_condition_that_clears_in_time_trips_nothing() {
    let run = |fault_duration: f64| {
        let mut system = network();
        let relay =
            Relay::under_voltage("uv1", 1, 0.85, 0.4, EventKind::BranchTrip { branch: 2 });
        let opts = options(
            5.0,
            vec![
                Event::bolted_fault(1.0, 1),
                Event::new(1.0 + fault_duration, EventKind::ClearFault { bus: 1 }),
            ],
            vec![relay],
        );
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        report.relay_actions.len()
    };

    assert_eq!(run(0.2), 0, "a fault cleared inside the delay must trip nothing");
    assert_eq!(run(0.9), 1, "a fault that outlasts the delay must trip");
}

/// An over-speed relay watches a machine rather than a bus, and a one-shot
/// relay fires once.
#[test]
fn a_relay_can_watch_a_machine_and_fires_once() {
    let mut system = network();
    let relay = Relay {
        id: "os1".to_string(),
        watch: Watch::UnitSpeed { unit: 0 },
        trigger: Trigger::Above(1.002),
        delay: 0.05,
        action: EventKind::LoadStep { bus: 1, ds: Complex::new(-0.2, 0.0) },
        repeating: false,
    };
    let opts = options(
        6.0,
        vec![
            Event::bolted_fault(1.0, 1),
            Event::new(1.12, EventKind::ClearFault { bus: 1 }),
        ],
        vec![relay],
    );
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);

    // The fault accelerates the machine well past 1.002, and the speed
    // oscillates back through the threshold afterwards — a repeating relay
    // would fire again, a one-shot must not.
    assert_eq!(report.relay_actions.len(), 1, "{:?}", report.relay_actions);
    let omega = report.trajectory.series("G1.omega").unwrap();
    let crossings = omega
        .windows(2)
        .filter(|w| (w[0] < 1.002) != (w[1] < 1.002))
        .count();
    assert!(crossings > 2, "the speed should cross the threshold more than once: {crossings}");
}

/// A relay naming something that does not exist is dropped and named, not
/// panicked on.
#[test]
fn a_relay_watching_nothing_is_dropped() {
    let mut system = network();
    let relay = Relay::under_voltage("ghost", 9, 0.9, 0.1, EventKind::BranchTrip { branch: 2 });
    let report = run_dynamics(&mut system, &options(1.0, Vec::new(), vec![relay]));
    assert_eq!(report.status, DynamicsStatus::Completed);
    assert!(report.relay_actions.is_empty());
    assert!(
        report
            .warnings
            .iter()
            .any(|w| matches!(w, DynamicsWarning::RelayDropped { id, .. } if id == "ghost")),
        "{:?}",
        report.warnings
    );
}

/// With no relays, nothing about a run changes.
#[test]
fn relays_are_inert_when_there_are_none() {
    let trajectory = |relays: Vec<Relay>| {
        let mut system = network();
        let opts = options(
            3.0,
            vec![
                Event::bolted_fault(1.0, 1),
                Event::new(1.1, EventKind::ClearFault { bus: 1 }),
            ],
            relays,
        );
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        report.trajectory.series("G1.delta").unwrap()
    };

    // A relay that never trips must not perturb the trajectory either — the
    // crossing search must not disturb the step it searches inside.
    let quiet = Relay::under_voltage("never", 1, 0.0, 0.1, EventKind::BranchTrip { branch: 2 });
    let without = trajectory(Vec::new());
    let with = trajectory(vec![quiet]);
    assert_eq!(without.len(), with.len());
    for (a, b) in without.iter().zip(with.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "an inert relay must change nothing");
    }
}

