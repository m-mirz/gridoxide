//! `SwitchTreatment::Constrain`: a switch as an exact equality.
//!
//! `plans/NODE_BREAKER_PLAN.md` phase 3. The oracle throughout is
//! `SwitchTreatment::Regularize`, which is thoroughly covered elsewhere: a
//! closed switch stamped as a very stiff branch and a closed switch enforced as
//! `θ_i = θ_j, V_i = V_j` are two approximations of the same physics, so they
//! must agree to within the stiffness the first one chose. If they disagree by
//! more, one of them is wrong.
//!
//! The comparison is worth more than it looks. The two share no code below the
//! Y-bus: `Regularize` puts a number into the admittance matrix and solves the
//! ordinary Newton system, while `Constrain` adds rows and columns and solves an
//! indefinite KKT system whose diagonal has zeros in it. Agreement to 1e-9 is
//! not two runs of the same arithmetic.

use gridoxide::constrained::{solve_constrained, ConstrainedSwitch};
use gridoxide::network::{build_ybus, stamp_shunts, YBus};
use gridoxide::solver::{newton_raphson, SolveStatus};
use gridoxide::switches::switch_admittance;
use gridoxide::types::{Bus, BusType, Line, Transformer};
use num_complex::Complex;

fn bus(idx: usize, bus_type: BusType, p: f64, q: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: q,
        q_min: -f64::INFINITY,
        q_max: f64::INFINITY,
        u_rated: 400e3,
        zip_terms: Vec::new(),
    }
}

fn line(from: usize, to: usize, r: f64, x: f64) -> Line {
    Line { from, to, r, x, b_shunt: 0.0, g_shunt: 0.0 }
}

fn switch_branch(from: usize, to: usize, open: bool) -> Transformer {
    let status = u8::from(!open);
    Transformer {
        from,
        to,
        from_status: status,
        to_status: status,
        y_series: switch_admittance(),
        y_shunt: Complex::new(0.0, 0.0),
        tap: Complex::new(1.0, 0.0),
    }
}

fn ybus_of(n: usize, lines: &[Line], transformers: &[Transformer]) -> gridoxide::network::YBusSparse {
    let mut y: YBus = build_ybus(n, lines, transformers);
    stamp_shunts(&mut y, &[]);
    y.finish()
}

/// Solves the same network both ways and returns `(regularized, constrained)`
/// bus states plus the constrained solution.
fn both_ways(
    buses: &[Bus],
    lines: &[Line],
    switches: &[ConstrainedSwitch],
) -> (Vec<Bus>, Vec<Bus>, gridoxide::constrained::ConstrainedSolution) {
    let stamped: Vec<Transformer> =
        switches.iter().map(|s| switch_branch(s.from, s.to, s.open)).collect();

    let mut reg = buses.to_vec();
    let y_reg = ybus_of(reg.len(), lines, &stamped);
    let reports = newton_raphson(&mut reg, &y_reg, 1e-10, 30);
    assert!(
        reports.iter().all(|r| matches!(
            r.status,
            gridoxide::solver::IslandStatus::Converged
                | gridoxide::solver::IslandStatus::NoReferenceBus
        )),
        "the regularized reference did not solve: {reports:?}"
    );

    let mut con = buses.to_vec();
    let y_con = ybus_of(con.len(), lines, &[]);
    let solution = solve_constrained(&mut con, &y_con, switches, 1e-10, 30);
    (reg, con, solution)
}

/// How far apart the two treatments are *allowed* to be.
///
/// Not a tuned constant: it is the physics of the difference. A regularized
/// switch is a real impedance, so it drops `|S| · z` in voltage where the
/// constrained one drops exactly nothing, and a path crossing `n` of them
/// accumulates `n` such drops. `switch_reactance()` is ~5e-6 p.u. and the test
/// networks carry ~1 p.u., so this is the drop with a factor of two to spare —
/// and it scales with the switch count rather than being widened by hand when a
/// longer chain fails.
fn agreement_tolerance(n_closed: usize) -> f64 {
    // The floor is ordinary solver noise: with no closed switch the two paths
    // solve the identical system and must agree to round-off.
    1e-9 + 2.0 * n_closed as f64 * gridoxide::switches::switch_reactance()
}

fn assert_states_agree(reg: &[Bus], con: &[Bus], tol: f64) {
    for (i, (r, c)) in reg.iter().zip(con).enumerate() {
        assert!(
            (r.voltage_mag - c.voltage_mag).abs() < tol,
            "bus {i} magnitude: regularized {} vs constrained {}",
            r.voltage_mag,
            c.voltage_mag
        );
        assert!(
            (r.voltage_ang - c.voltage_ang).abs() < tol,
            "bus {i} angle: regularized {} vs constrained {}",
            r.voltage_ang,
            c.voltage_ang
        );
    }
}

/// **Phase 3's gate.** A closed switch carrying load: both treatments agree on
/// the state, and the constrained solve reports the flow through the switch.
///
/// The network is a source, a line, a switch, and a load behind it — so all of
/// the load's power has to pass through the switch, and its flow is not merely
/// determined but known in advance.
#[test]
fn a_closed_switch_carries_the_load_and_both_treatments_agree() {
    let buses = vec![
        bus(0, BusType::Slack, 0.0, 0.0),
        bus(1, BusType::PQ, 0.0, 0.0),
        bus(2, BusType::PQ, -0.8, -0.3),
    ];
    let lines = vec![line(0, 1, 0.01, 0.1)];
    let switches = vec![ConstrainedSwitch { from: 1, to: 2, open: false }];

    let (reg, con, solution) = both_ways(&buses, &lines, &switches);
    assert_eq!(solution.stats.status, SolveStatus::Converged);
    assert!(solution.indeterminate.is_empty(), "nothing here is indeterminate");
    assert_states_agree(&reg, &con, agreement_tolerance(switches.iter().filter(|s| !s.open).count()));

    // The switch's two ends are the same point, exactly.
    assert!((con[1].voltage_mag - con[2].voltage_mag).abs() < 1e-12);
    assert!((con[1].voltage_ang - con[2].voltage_ang).abs() < 1e-12);

    // Everything the load draws passes through the switch, and nothing else
    // reaches bus 2.
    let (p, q) = solution.flows[0];
    assert!((p - 0.8).abs() < 1e-9, "switch carries P = {p}, expected the load's 0.8");
    assert!((q - 0.3).abs() < 1e-9, "switch carries Q = {q}, expected the load's 0.3");
}

/// An open switch is not a connection. Its ends are free to differ, its flow is
/// exactly zero, and the network behind it is de-energized rather than solved.
#[test]
fn an_open_switch_carries_nothing_and_ties_nothing() {
    let buses = vec![
        bus(0, BusType::Slack, 0.0, 0.0),
        bus(1, BusType::PQ, -0.5, -0.2),
        bus(2, BusType::PQ, -0.3, -0.1),
    ];
    let lines = vec![line(0, 1, 0.01, 0.1)];
    let switches = vec![ConstrainedSwitch { from: 1, to: 2, open: true }];

    let (reg, con, solution) = both_ways(&buses, &lines, &switches);
    assert_eq!(solution.stats.status, SolveStatus::Converged);
    assert_eq!(solution.flows[0], (0.0, 0.0), "an open switch carries exactly nothing");
    assert_states_agree(&reg, &con, agreement_tolerance(switches.iter().filter(|s| !s.open).count()));
    // Bus 2 is behind the open switch with no source: de-energized, reported at
    // exactly zero, the same as any unreferenced island.
    assert_eq!(con[2].voltage_mag, 0.0);
}

/// Opening a switch between solves changes the answer, and the two treatments
/// keep agreeing across the change. This is the sequence a switching study runs.
#[test]
fn the_two_treatments_agree_across_a_switching_campaign() {
    // A ring: the load at bus 3 can be fed either way round, and the switch
    // decides which.
    let buses = vec![
        bus(0, BusType::Slack, 0.0, 0.0),
        bus(1, BusType::PQ, -0.2, -0.05),
        bus(2, BusType::PQ, -0.1, -0.02),
        bus(3, BusType::PQ, -0.6, -0.25),
    ];
    let lines = vec![line(0, 1, 0.01, 0.1), line(0, 2, 0.02, 0.15), line(1, 3, 0.015, 0.12)];

    let mut previous: Option<f64> = None;
    for open in [false, true, false] {
        let switches = vec![ConstrainedSwitch { from: 2, to: 3, open }];
        let (reg, con, solution) = both_ways(&buses, &lines, &switches);
        assert_eq!(solution.stats.status, SolveStatus::Converged, "open = {open}");
        assert_states_agree(&reg, &con, agreement_tolerance(switches.iter().filter(|s| !s.open).count()));

        let p = solution.flows[0].0;
        if open {
            assert_eq!(p, 0.0);
        } else {
            assert!(p.abs() > 1e-3, "a closed tie in a ring should carry something, got {p}");
        }
        // Closing and reopening returns to the same state, so the loop is not
        // accumulating anything between runs.
        if !open {
            if let Some(before) = previous {
                assert!((p - before).abs() < 1e-12);
            }
            previous = Some(p);
        }
    }
}

/// A loop of closed switches leaves their flows genuinely undetermined, and this
/// says so instead of inventing a split.
///
/// Two switches in parallel across the same pair of buses: any pair of flows
/// summing to the load is consistent with every equation in the system.
/// `Regularize` picks the even split because its two branches happen to have
/// equal stiffness — a property of the constant it chose, not of the network.
#[test]
fn a_loop_of_closed_switches_is_reported_as_indeterminate() {
    let buses = vec![
        bus(0, BusType::Slack, 0.0, 0.0),
        bus(1, BusType::PQ, 0.0, 0.0),
        bus(2, BusType::PQ, -0.4, -0.15),
    ];
    let lines = vec![line(0, 1, 0.01, 0.1)];
    let switches = vec![
        ConstrainedSwitch { from: 1, to: 2, open: false },
        ConstrainedSwitch { from: 1, to: 2, open: false },
    ];

    let (reg, con, solution) = both_ways(&buses, &lines, &switches);
    assert_eq!(solution.stats.status, SolveStatus::Converged);
    assert_states_agree(&reg, &con, agreement_tolerance(switches.iter().filter(|s| !s.open).count()));

    assert_eq!(solution.indeterminate, vec![1], "the loop-closing switch is the second one");
    assert_eq!(solution.flows[1], (0.0, 0.0), "an indeterminate flow is pinned, not guessed");
    // The determined one carries the whole load, which is the only quantity the
    // network actually fixes.
    let (p, q) = solution.flows[0];
    assert!((p - 0.4).abs() < 1e-9 && (q - 0.15).abs() < 1e-9, "got ({p}, {q})");
}

/// A chain of switches with nothing but switches between source and load — the
/// bay-internal case a node-breaker model is full of. Every intermediate bus is
/// tied to its neighbours exactly, and the whole chain carries the load.
#[test]
fn a_chain_of_switches_ties_every_intermediate_bus() {
    let n = 8;
    let mut buses = vec![bus(0, BusType::Slack, 0.0, 0.0)];
    for i in 1..n {
        buses.push(bus(i, BusType::PQ, 0.0, 0.0));
    }
    buses[n - 1].p_spec = -0.5;
    buses[n - 1].q_spec = -0.2;
    let lines = vec![line(0, 1, 0.01, 0.1)];
    let switches: Vec<ConstrainedSwitch> =
        (1..n - 1).map(|i| ConstrainedSwitch { from: i, to: i + 1, open: false }).collect();

    let (reg, con, solution) = both_ways(&buses, &lines, &switches);
    assert_eq!(solution.stats.status, SolveStatus::Converged);
    assert!(solution.indeterminate.is_empty());
    assert_states_agree(&reg, &con, agreement_tolerance(switches.iter().filter(|s| !s.open).count()));

    for i in 1..n - 1 {
        assert!(
            (con[i].voltage_mag - con[i + 1].voltage_mag).abs() < 1e-12
                && (con[i].voltage_ang - con[i + 1].voltage_ang).abs() < 1e-12,
            "switch {i} did not tie its ends"
        );
    }
    for (k, (p, q)) in solution.flows.iter().enumerate() {
        assert!(
            (p - 0.5).abs() < 1e-9 && (q - 0.2).abs() < 1e-9,
            "switch {k} carries ({p}, {q}), not the load"
        );
    }
}

/// The constrained system is bigger than the unconstrained one but not
/// ill-conditioned: no large number enters the matrix at all, which is the
/// argument `zero_impedance_branches.md` makes for this formulation. A hundred
/// switches in a chain converge in the same handful of iterations as one.
#[test]
fn many_switches_do_not_degrade_convergence() {
    let n = 102;
    let mut buses = vec![bus(0, BusType::Slack, 0.0, 0.0)];
    for i in 1..n {
        buses.push(bus(i, BusType::PQ, 0.0, 0.0));
    }
    buses[n - 1].p_spec = -0.5;
    buses[n - 1].q_spec = -0.2;
    let lines = vec![line(0, 1, 0.01, 0.1)];
    let switches: Vec<ConstrainedSwitch> =
        (1..n - 1).map(|i| ConstrainedSwitch { from: i, to: i + 1, open: false }).collect();

    let mut con = buses.clone();
    let y = ybus_of(n, &lines, &[]);
    let solution = solve_constrained(&mut con, &y, &switches, 1e-10, 30);
    assert_eq!(solution.stats.status, SolveStatus::Converged, "100 switches did not converge");
    assert!(
        solution.stats.iterations() <= 5,
        "took {} iterations",
        solution.stats.iterations()
    );
    for (p, _) in &solution.flows {
        assert!((p - 0.5).abs() < 1e-9);
    }
}

/// A switch between two `Slack` buses constrains nothing that was not already
/// fixed, and its flow is absorbed by whichever reference gets there first. That
/// is reported rather than solved for — an empty constraint row would make the
/// matrix singular.
#[test]
fn a_switch_between_two_references_is_indeterminate_rather_than_singular() {
    let buses = vec![
        bus(0, BusType::Slack, 0.0, 0.0),
        bus(1, BusType::Slack, 0.0, 0.0),
        bus(2, BusType::PQ, -0.3, -0.1),
    ];
    let lines = vec![line(0, 2, 0.01, 0.1)];
    let switches = vec![ConstrainedSwitch { from: 0, to: 1, open: false }];

    let mut con = buses.clone();
    let y = ybus_of(3, &lines, &[]);
    let solution = solve_constrained(&mut con, &y, &switches, 1e-10, 30);
    assert_eq!(solution.stats.status, SolveStatus::Converged);
    assert_eq!(solution.indeterminate, vec![0]);
    assert_eq!(solution.flows[0], (0.0, 0.0));
}

/// A closed switch conducts even though it adds no admittance, so island
/// analysis has to see it. Without that, the far side of every closed switch
/// looks like a component with no reference and is de-energized — which would
/// turn an ordinary network into a dead one.
#[test]
fn islands_are_decided_across_closed_switches() {
    // Bus 1 reaches the source *only* through the switch.
    let buses = vec![
        bus(0, BusType::Slack, 0.0, 0.0),
        bus(1, BusType::PQ, -0.25, -0.1),
    ];
    let lines: Vec<Line> = Vec::new();
    let switches = vec![ConstrainedSwitch { from: 0, to: 1, open: false }];

    let mut con = buses.clone();
    let y = ybus_of(2, &lines, &[]);
    let solution = solve_constrained(&mut con, &y, &switches, 1e-10, 30);
    assert_eq!(solution.stats.status, SolveStatus::Converged);
    assert!(con[1].voltage_mag > 0.5, "bus 1 was de-energized despite a closed switch to the source");
    assert!((con[1].voltage_mag - con[0].voltage_mag).abs() < 1e-12);
    let (p, q) = solution.flows[0];
    assert!((p - 0.25).abs() < 1e-9 && (q - 0.1).abs() < 1e-9, "got ({p}, {q})");
}
