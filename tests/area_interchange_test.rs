//! Area interchange control: holding each control area's net export at its
//! scheduled value.
//!
//! The property under test throughout is the one the loop promises: **after it,
//! each area exports what it agreed to, and the slack produces its own schedule
//! and nothing more.** Every check re-measures the interchange from
//! `branch_flow::terminal_flow` at the returned state rather than reading the
//! loop's own bookkeeping, the same discipline `distributed_slack_test.rs`
//! keeps.

use gridoxide::network::{build_ybus, effective_injection, power_injections, YBusSparse};
use gridoxide::outerloop::{
    AreaDefinition, AreaInterchange, DistributedSlack, OuterLoop, SlackDistribution, SolveContext,
};
use gridoxide::solver::{IslandStatus, JacobianBackend};
use gridoxide::types::{Bus, BusType, Line, Transformer};

fn bus(idx: usize, bus_type: BusType, p_spec: f64, q_spec: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec,
        q_spec,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: Vec::new(),
    }
}

/// Two areas, three buses each, joined by a single tie line — the smallest
/// network where a net position means anything. Resistive lines, so there are
/// real losses for the loop to work against.
///
/// Area 0: slack at bus 0, a generator at 1, load at 2.
/// Area 1: generator at 3, load at 4, generator at 5.
/// The tie is 2 — 3.
fn two_areas() -> (Vec<Bus>, Vec<Line>, YBusSparse, Vec<Option<usize>>) {
    let buses = vec![
        bus(0, BusType::Slack, 0.40, 0.0),
        bus(1, BusType::PV, 0.60, 0.0),
        bus(2, BusType::PQ, -0.70, -0.15),
        bus(3, BusType::PV, 0.50, 0.0),
        bus(4, BusType::PQ, -0.90, -0.20),
        bus(5, BusType::PV, 0.30, 0.0),
    ];
    let lines = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.03, x: 0.08, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 0, to: 2, r: 0.04, x: 0.10, b_shunt: 0.0, g_shunt: 0.0 },
        // the tie
        Line { from: 2, to: 3, r: 0.03, x: 0.09, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 3, to: 4, r: 0.02, x: 0.07, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 4, to: 5, r: 0.03, x: 0.08, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 3, to: 5, r: 0.04, x: 0.11, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let ybus = build_ybus(6, &lines, &Vec::<Transformer>::new()).finish();
    let of_bus = vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)];
    (buses, lines, ybus, of_bus)
}

fn run_area(
    buses: &mut [Bus],
    lines: &[Line],
    ybus: &YBusSparse,
    areas: AreaDefinition,
) -> (Vec<gridoxide::solver::IslandReport>, gridoxide::outerloop::AreaInterchangeReport) {
    let cap = areas.max_outer_iter;
    let mut ybus = ybus.clone();
    let mut transformers: Vec<Transformer> = Vec::new();
    let mut changers = Vec::new();
    let mut loop_ = AreaInterchange::new(areas);
    let islands = {
        let mut ctx = SolveContext::new(buses, &mut ybus)
            .with_branches(lines, &mut transformers, &[])
            .with_taps(&mut changers, &[]);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut loop_];
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-11, 40, JacobianBackend::Scalar, &mut list, cap * 2).0
    };
    (islands, loop_.into_report())
}

/// Measured independently of the loop, from the returned state.
fn interchange(buses: &[Bus], lines: &[Line], of_bus: &[Option<usize>]) -> Vec<f64> {
    AreaInterchange::measure(buses, lines, &[], of_bus, 2)
}

/// **The derivation check.** One area covering the whole network has no
/// boundary branches, so its interchange is identically zero; with a target of
/// zero the mismatch collapses to the slack's own deviation, and the loop
/// becomes distributed slack exactly.
///
/// This is not a curiosity — it is why area interchange *replaces* distributed
/// slack rather than sitting beside it, and powsybl's own
/// `AcAreaInterchangeControlOuterLoop` constructs a `DistributedSlackOuterLoop`
/// as its no-area fallback for the same reason. If the sign in the module's
/// derivation were flipped, this test would diverge instead of agreeing.
#[test]
fn area_interchange_with_one_area_is_distributed_slack() {
    let (buses, lines, ybus, _) = two_areas();

    let mut by_slack = buses.clone();
    let mut ybus_a = ybus.clone();
    let mut slack = DistributedSlack::new(SlackDistribution::uniform(&by_slack));
    {
        let mut ctx = SolveContext::new(&mut by_slack, &mut ybus_a);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut slack];
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-11, 40, JacobianBackend::Scalar, &mut list, 60);
    }
    let slack_report = slack.into_report();
    assert!(slack_report.converged, "{slack_report:?}");

    let mut by_area = buses.clone();
    let one_area = vec![Some(0); 6];
    let areas = AreaDefinition {
        tolerance: 1e-11,
        ..AreaDefinition::uniform(&by_area, one_area, 1)
    };
    let (islands, area_report) = run_area(&mut by_area, &lines, &ybus, areas);

    assert!(islands.iter().all(|i| i.status == IslandStatus::Converged), "{islands:?}");
    assert!(area_report.converged, "{area_report:?}");
    for i in 0..6 {
        assert!(
            (area_report.shift[i] - slack_report.shift[i]).abs() < 1e-9,
            "bus {i}: area shifted {} where distributed slack shifted {}",
            area_report.shift[i],
            slack_report.shift[i]
        );
        assert!(
            (by_area[i].voltage_mag - by_slack[i].voltage_mag).abs() < 1e-9,
            "bus {i} solved to a different voltage"
        );
    }
}

/// **The defining property.** Each area ends up exporting what it was
/// scheduled for, measured from the tie-line flows at the returned state, and
/// the slack ends up producing its own schedule.
#[test]
fn each_area_reaches_its_scheduled_net_position() {
    let (mut buses, lines, ybus, of_bus) = two_areas();

    // Area 0 agrees to export 0.15 pu; area 1 therefore imports about that,
    // less the losses on the tie itself, which belong to neither.
    let areas = AreaDefinition {
        targets: vec![0.15, -0.15],
        tolerance: 1e-10,
        ..AreaDefinition::uniform(&buses, of_bus.clone(), 2)
    };
    let (islands, report) = run_area(&mut buses, &lines, &ybus, areas);

    assert!(islands.iter().all(|i| i.status == IslandStatus::Converged), "{islands:?}");
    assert!(report.converged, "{report:?}");
    assert!(report.unbalanced.is_empty(), "{:?}", report.unbalanced);

    let measured = interchange(&buses, &lines, &of_bus);

    // Area 1 has no slack, so its position is driven and met exactly.
    assert!(
        (measured[1] - -0.15).abs() < 1e-8,
        "area 1 exports {} against a schedule of -0.15",
        measured[1]
    );

    // Area 0 holds the slack, so it is the dependent one: it absorbs the tie
    // losses, which no set of targets can schedule away. Both ends of a tie are
    // measured into the branch, so the two positions sum to what the tie
    // dissipates rather than cancelling.
    assert_eq!(report.dependent, Some(0));
    let tie_loss = measured[0] + measured[1];
    assert!(tie_loss > 0.0, "a resistive tie must lose something: {tie_loss}");
    assert!(
        (measured[0] - (0.15 + tie_loss)).abs() < 1e-8,
        "area 0 exports {} against 0.15 plus {tie_loss} of tie loss",
        measured[0]
    );
    assert!(
        (report.residual[0] - tie_loss).abs() < 1e-8,
        "the dependent area's residual should report exactly the tie loss, got {}",
        report.residual[0]
    );

    // And the slack is on its own schedule, which is the second half of what
    // the loop drives — the half distributed slack does alone.
    let (p_calc, _) = power_injections(&buses, &ybus);
    let scheduled = effective_injection(&buses[0]).0;
    assert!(
        (p_calc[0] - scheduled).abs() < 1e-8,
        "slack produced {} against a schedule of {scheduled}",
        p_calc[0]
    );
}

/// Changing the schedule changes the answer, in the direction and roughly the
/// amount asked for. Without this, the test above could pass on a network that
/// happened to already be at target.
#[test]
fn a_different_schedule_moves_the_flow() {
    let (buses, lines, ybus, of_bus) = two_areas();

    let solve_for = |export: f64| -> f64 {
        let mut b = buses.clone();
        let areas = AreaDefinition {
            targets: vec![export, -export],
            tolerance: 1e-10,
            ..AreaDefinition::uniform(&b, of_bus.clone(), 2)
        };
        let (_, report) = run_area(&mut b, &lines, &ybus, areas);
        assert!(report.converged, "target {export}: {report:?}");
        interchange(&b, &lines, &of_bus)[1]
    };

    // Measured on area 1, whose position the loop drives — area 0 holds the
    // slack and is the dependent one.
    let low = solve_for(-0.10);
    let high = solve_for(0.30);
    assert!((low - 0.10).abs() < 1e-8, "area 1 should import 0.10, got {low}");
    assert!((high - -0.30).abs() < 1e-8, "area 1 should export 0.30, got {high}");
    assert!(low - high > 0.35, "the schedule should move the tie flow: {high} -> {low}");
}

/// An area with nothing to dispatch is named rather than silently left at
/// whatever it happened to be exporting.
#[test]
fn an_area_with_no_generator_is_reported() {
    let (mut buses, lines, ybus, of_bus) = two_areas();
    // Strip area 1's participation, leaving it nothing to move.
    let mut areas = AreaDefinition {
        targets: vec![0.15, -0.15],
        tolerance: 1e-10,
        ..AreaDefinition::uniform(&buses, of_bus.clone(), 2)
    };
    for i in 3..6 {
        areas.factors[i] = 0.0;
    }
    let (_, report) = run_area(&mut buses, &lines, &ybus, areas);

    assert!(
        report.unbalanced.iter().any(|(a, why)| *a == 1 && why.contains("no participating")),
        "{:?}",
        report.unbalanced
    );
    // Area 0 still does its own job — one area's inability to dispatch must
    // not stop the others. Its job is the slack's schedule, since it is the
    // dependent area.
    let (p_calc, _) = power_injections(&buses, &ybus);
    let scheduled = effective_injection(&buses[0]).0;
    assert!(
        (p_calc[0] - scheduled).abs() < 1e-8,
        "slack produced {} against a schedule of {scheduled}",
        p_calc[0]
    );
}

/// A bus in no area is nobody's to schedule: it never moves, and the branches
/// it shares with an area still count as that area's boundary.
#[test]
fn a_bus_in_no_area_is_left_alone() {
    let (mut buses, lines, ybus, mut of_bus) = two_areas();
    of_bus[5] = None;
    let before = buses[5].p_spec;

    let areas = AreaDefinition {
        targets: vec![0.15, -0.15],
        tolerance: 1e-10,
        ..AreaDefinition::uniform(&buses, of_bus.clone(), 2)
    };
    let (_, report) = run_area(&mut buses, &lines, &ybus, areas);

    assert_eq!(report.shift[5], 0.0, "a bus in no area must not be dispatched");
    assert_eq!(buses[5].p_spec, before);
    // Bus 5's two branches (4-5 and 3-5) now cross area 1's boundary, so area
    // 1's interchange counts them — a different number from the all-in-one
    // case, and the loop still reaches its target.
    let measured = interchange(&buses, &lines, &of_bus);
    assert!((measured[1] - -0.15).abs() < 1e-8, "{}", measured[1]);
    assert!(report.converged, "{report:?}");
}

/// Without the branch lists there is no boundary to measure, so the loop says
/// so rather than reporting every area balanced at zero.
#[test]
fn a_context_without_branches_fails_loudly() {
    let (mut buses, _lines, ybus, of_bus) = two_areas();
    let mut ybus = ybus.clone();
    let mut loop_ = AreaInterchange::new(AreaDefinition::uniform(&buses, of_bus, 2));
    let (_, report) = {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut loop_];
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-10, 40, JacobianBackend::Scalar, &mut list, 40)
    };
    assert!(!report.converged);
    let status = report.loop_named("AreaInterchange").map(|l| l.status.clone());
    assert!(
        matches!(status, Some(gridoxide::outerloop::OuterLoopStatus::Failed(ref m)) if m.contains("with_branches")),
        "{status:?}"
    );
}
