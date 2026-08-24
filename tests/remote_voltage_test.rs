//! Remote voltage control: a machine holding a bus it is not connected to.
//!
//! The vendored corpus has exactly one usable case of this (MicroGrid-Type1;
//! FullGrid's is in the fixture with the 50 GW shunt), and it cannot referee the
//! numbers — gridoxide's solution of that network already differs from its
//! published one by more than remote control contributes, which is why its
//! voltage assertion tolerates 5%. `cgmes_remote_test.rs` checks what that
//! fixture *can* settle, which is the structure.
//!
//! The numbers are settled here instead, on networks small enough that the right
//! answer is checkable by construction. The strongest gate is a fixed-point
//! one: whatever setpoint the loop lands on, pinning the controller there by
//! hand must put the controlled bus on target — which is precisely what the
//! exact formulation (fix `|V|` at the controlled bus, free `Q` at the
//! controller) asserts, without needing that formulation to be implemented.

use gridoxide::outerloop::RemoteOutcome;
use gridoxide::solver::{PowerFlowOptions, SolveStatus};
use gridoxide::types::{Bus, BusType, Line, RegulatingMachine, Transformer};
use num_complex::Complex;

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

/// Slack — line — load bus — transformer — machine bus.
///
/// The machine sits behind its own step-up transformer and holds the bus on the
/// far side of it, which is the arrangement every real case of this is.
///
/// **Built the way the importer leaves it**, which is the thing under test: the
/// *controlled* bus arrives already pinned to `PV` at the target, because
/// `RegulatingControl.Terminal` resolves to it and that is where the pin has
/// always gone. Starting from a network that already had the control in the
/// right place would test nothing.
fn network(target: f64) -> (Vec<Bus>, Vec<Line>, Vec<Transformer>) {
    let mut held = bus(1, BusType::PV, -0.8, -0.3); // the bus being held, carrying load
    held.voltage_mag = target;
    let buses = vec![
        bus(0, BusType::Slack, 0.0, 0.0),
        held,
        bus(2, BusType::PQ, 0.6, 0.0), // the machine
    ];
    let lines = vec![Line { from: 0, to: 1, r: 0.01, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 }];
    let transformers = vec![Transformer {
        from: 2,
        to: 1,
        from_status: 1,
        to_status: 1,
        y_series: Complex::new(0.0, -1.0 / 0.05),
        y_shunt: Complex::new(0.0, 0.0),
        tap: Complex::new(1.0, 0.0),
    }];
    (buses, lines, transformers)
}

fn machine(at_bus: usize, controls_bus: usize, target: f64, q_min: f64, q_max: f64) -> RegulatingMachine {
    RegulatingMachine {
        id: "m".to_string(),
        at_bus,
        controls_bus,
        target_pu: target,
        q_min,
        q_max,
        q_scheduled: 0.0,
        key: None,
    }
}

fn solve(
    buses: Vec<Bus>,
    lines: &[Line],
    transformers: &[Transformer],
    machines: &[RegulatingMachine],
    on: bool,
) -> gridoxide::PowerFlowReport {
    gridoxide::run_power_flow_with_remote(
        buses,
        lines,
        transformers,
        &[],
        gridoxide::TapData::none(),
        gridoxide::RemoteControlData { machines },
        PowerFlowOptions {
            control_remote_voltage: on,
            enforce_q_limits: true,
            max_iter: 40,
            max_outer_iter: 60,
            ..Default::default()
        },
    )
}

/// **The gate.** The loop finds a genuine fixed point of the exact
/// formulation, checked without implementing that formulation.
///
/// Take the controller setpoint the loop settled on, pin it there by hand as an
/// ordinary `PV` bus, and solve with no loop at all. The controlled bus must
/// land on its target. That is the whole content of "fix `|V|` at the controlled
/// bus, free `Q` at the controller" — reached iteratively rather than in one
/// Newton system, and the difference is iteration, not answer.
#[test]
fn the_setpoint_it_lands_on_is_a_fixed_point_of_the_exact_formulation() {
    for target in [0.98, 1.00, 1.02, 1.05] {
        let (buses, lines, transformers) = network(target);
        let machines = [machine(2, 1, target, -3.0, 3.0)];
        let report = solve(buses.clone(), &lines, &transformers, &machines, true);
        assert_eq!(report.stats.status, SolveStatus::Converged, "target {target}");

        let outer = report.outer.as_ref().expect("the loop ran");
        let r = &outer.remote[0];
        assert_eq!(r.outcome, RemoteOutcome::Held, "target {target}: {r:?}");
        assert!(
            (report.buses[1].voltage_mag - target).abs() < 2e-4,
            "target {target}: controlled bus reached {}",
            report.buses[1].voltage_mag
        );

        // Now the independent half: pin the controller by hand at the setpoint
        // the loop chose, free the bus it was holding, and solve with no loop.
        let settled = report.buses[2].voltage_mag;
        let mut pinned = buses.clone();
        pinned[1].bus_type = BusType::PQ;
        pinned[2].bus_type = BusType::PV;
        pinned[2].voltage_mag = settled;
        let plain = gridoxide::run_power_flow(
            pinned,
            &lines,
            &transformers,
            &[],
            gridoxide::TapData::none(),
            PowerFlowOptions { max_iter: 40, ..Default::default() },
        );
        assert_eq!(plain.stats.status, SolveStatus::Converged);
        // The fixed-point identity: the same setpoint must reproduce the same
        // state. Compared against what the loop reached rather than against the
        // target, because the loop stops inside a deadband and it is the
        // *reproducibility* that is the claim here — the deadband is checked
        // above.
        assert!(
            (plain.buses[1].voltage_mag - report.buses[1].voltage_mag).abs() < 1e-6,
            "target {target}: pinning the controller at {settled} put the controlled bus at {}, \
             where the loop had it at {} — so the loop did not find a fixed point",
            plain.buses[1].voltage_mag,
            report.buses[1].voltage_mag
        );
    }
}

/// The reactive power is produced where the machine is, which is the entire
/// point — and the bus it holds gets none of its own.
#[test]
fn the_reactive_power_is_produced_at_the_machine() {
    use gridoxide::network::{build_ybus, power_injections};

    let (buses, lines, transformers) = network(1.02);
    let machines = [machine(2, 1, 1.02, -3.0, 3.0)];
    let ybus = build_ybus(buses.len(), &lines, &transformers).finish();

    let off = solve(buses.clone(), &lines, &transformers, &machines, false);
    let on = solve(buses.clone(), &lines, &transformers, &machines, true);
    let (_, q_off) = power_injections(&off.buses, &ybus);
    let (_, q_on) = power_injections(&on.buses, &ybus);

    // With the control where the machine is: the machine's bus is PV, so its
    // reactive injection is free and is whatever holding the far bus takes.
    assert_eq!(on.buses[2].bus_type, BusType::PV);
    assert_eq!(on.buses[1].bus_type, BusType::PQ);
    assert!(
        (q_on[1] - buses[1].q_spec).abs() < 1e-6,
        "the held bus should inject only its own load, got {} against a spec of {}",
        q_on[1],
        buses[1].q_spec
    );
    assert!(q_on[2].abs() > 1e-3, "the machine should be producing something, got {}", q_on[2]);

    // Without it, the roles are exactly reversed — which is the defect.
    assert_eq!(off.buses[1].bus_type, BusType::PV);
    assert!(
        (q_off[2] - buses[2].q_spec).abs() < 1e-6,
        "without the loop the machine is pinned at its schedule, not free"
    );
    assert!(
        (q_off[1] - buses[1].q_spec).abs() > 1e-3,
        "without the loop the free reactive power appears at the held bus"
    );
}

/// Local control is left alone. A machine already sitting at the bus it holds
/// is an ordinary `PV` bus and the loop must not touch it — otherwise turning
/// the option on would perturb every network that has no remote control at all.
#[test]
fn local_control_is_untouched() {
    let (buses, lines, transformers) = network(1.02);
    let mut local = buses.clone();
    local[1].bus_type = BusType::PV;
    local[1].voltage_mag = 1.02;
    let machines = [machine(1, 1, 1.02, -3.0, 3.0)];

    let on = solve(local.clone(), &lines, &transformers, &machines, true);
    let off = solve(local, &lines, &transformers, &machines, false);

    assert_eq!(on.stats.status, SolveStatus::Converged);
    let outer = on.outer.as_ref().expect("loops ran");
    assert!(outer.remote.is_empty(), "a local machine is not a remote controller");
    assert!(outer.retyped.is_empty(), "nothing should have been re-typed");
    for (a, b) in on.buses.iter().zip(&off.buses) {
        assert!(
            (a.voltage_mag - b.voltage_mag).abs() < 1e-9
                && (a.voltage_ang - b.voltage_ang).abs() < 1e-9,
            "bus {}: enabling the loop changed a locally-controlled network",
            a.idx
        );
    }
}

/// A machine that runs out of reactive capability stops holding anything, and
/// the limit that bounds it is **its own** — not the limits of a bus it is not
/// connected to.
///
/// This is the half that comes for free from putting the control on the machine:
/// the controller is a `PV` bus carrying the machine's capability, so
/// `ReactiveLimits` clamps the machine.
#[test]
fn a_saturating_machine_is_clamped_at_its_own_limit() {
    use gridoxide::network::{build_ybus, power_injections};

    let (buses, lines, transformers) = network(1.05);
    let ybus = build_ybus(buses.len(), &lines, &transformers).finish();

    // Derive the limit from what holding the target actually takes, rather than
    // guessing a number that might not bind. Solve once unconstrained, read the
    // machine's output, then give it half of that.
    let free = solve(buses.clone(), &lines, &transformers, &[machine(2, 1, 1.05, -9.0, 9.0)], true);
    assert_eq!(free.stats.status, SolveStatus::Converged);
    let (_, q_free) = power_injections(&free.buses, &ybus);
    let needed = q_free[2];
    assert!(needed > 0.05, "the fixture should need real reactive support, got {needed}");

    let cap = needed / 2.0;
    let machines = [machine(2, 1, 1.05, -cap, cap)];
    let report = solve(buses, &lines, &transformers, &machines, true);
    assert_eq!(report.stats.status, SolveStatus::Converged);

    let outer = report.outer.as_ref().expect("the loop ran");
    assert_eq!(outer.remote[0].outcome, RemoteOutcome::AtReactiveLimit, "{:?}", outer.remote[0]);
    assert!(
        outer.q_limit_switches.contains(&2),
        "the *machine's* bus should be the one clamped, got {:?}",
        outer.q_limit_switches
    );
    assert_eq!(report.buses[2].bus_type, BusType::PQ);
    assert!(
        (report.buses[2].q_spec - cap).abs() < 1e-9,
        "pinned at its own q_max ({cap}), got {}",
        report.buses[2].q_spec
    );
    assert!(
        report.buses[1].voltage_mag < 1.05,
        "a saturated machine cannot still be holding its target"
    );
}

/// With no machines, or with the option off, nothing happens at all.
#[test]
fn the_option_is_inert_without_data() {
    let (buses, lines, transformers) = network(1.02);
    let bare = gridoxide::run_power_flow(
        buses.clone(),
        &lines,
        &transformers,
        &[],
        gridoxide::TapData::none(),
        PowerFlowOptions { control_remote_voltage: true, ..Default::default() },
    );
    assert_eq!(bare.stats.status, SolveStatus::Converged);
    assert!(bare.outer.is_none(), "no machines means no loop, not an empty loop");

    let machines = [machine(2, 1, 1.02, -3.0, 3.0)];
    let off = solve(buses, &lines, &transformers, &machines, false);
    assert_eq!(off.buses[1].bus_type, BusType::PV, "off leaves the old behaviour in place");
}
