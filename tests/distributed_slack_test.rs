//! Distributed slack: sharing the system imbalance across generators instead
//! of dumping it on one bus.
//!
//! The property under test throughout is the same one: **after the loop, the
//! slack produces its own schedule plus its own share, and nothing more.**
//! Everything else — that the shifts sum to the losses, that the answer is
//! still a valid power flow, that islands are handled separately — follows
//! from or supports that.
//!
//! What makes this checkable without a second tool is that a distributed-slack
//! solution is still an ordinary power flow of a *modified* schedule. So every
//! test here re-derives the injections from `network::power_injections` at the
//! returned state and compares against what the schedules say they should be.
//! Nothing is read back out of the solver's own bookkeeping.

use gridoxide::network::{build_ybus, effective_injection, power_injections, YBusSparse};
use gridoxide::outerloop::{
    DistributedSlack, OuterLoop, SlackDistribution, SlackDistributionReport, SolveContext,
};
use gridoxide::solver::{newton_raphson, IslandReport, IslandStatus, JacobianBackend};
use gridoxide::types::{Bus, BusType, Line, Transformer};

/// Distributed slack is one outer loop among the list
/// [`gridoxide::outerloop::solve_with_loops`] drives. This wraps the
/// three-line set-up so the assertions below stay about the physics rather
/// than about the plumbing — it is the whole of what the deleted
/// `newton_raphson_distributing_slack` entry point used to do.
fn distributing_slack(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    tol: f64,
    max_iter: usize,
    backend: JacobianBackend,
    distribution: &SlackDistribution,
) -> (Vec<IslandReport>, SlackDistributionReport) {
    let cap = distribution.max_outer_iter;
    let mut ybus = ybus.clone();
    let mut slack = DistributedSlack::new(distribution.clone());
    let islands = {
        let mut ctx = SolveContext::new(buses, &mut ybus);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut slack];
        gridoxide::outerloop::solve_with_loops(&mut ctx, tol, max_iter, backend, &mut list, cap).0
    };
    (islands, slack.into_report())
}

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

/// Four buses in a ring: a slack and two generators, with load at the fourth.
/// Resistive lines, so there are real losses for the loop to distribute.
fn ring() -> (Vec<Bus>, YBusSparse) {
    let buses = vec![
        bus(0, BusType::Slack, 0.0, 0.0),
        bus(1, BusType::PV, 0.30, 0.0),
        bus(2, BusType::PV, 0.20, 0.0),
        bus(3, BusType::PQ, -1.00, -0.20),
    ];
    let lines = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.03, x: 0.08, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 2, to: 3, r: 0.02, x: 0.07, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 3, to: 0, r: 0.03, x: 0.09, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let ybus = build_ybus(4, &lines, &Vec::<Transformer>::new()).finish();
    (buses, ybus)
}

/// **The defining property.** With every generator sharing equally, the slack
/// must end up producing its schedule plus its own share — not the whole
/// imbalance.
#[test]
fn the_slack_keeps_only_its_own_share() {
    let (mut buses, ybus) = ring();

    // What a single slack does with the same network, for contrast.
    let mut single = buses.clone();
    newton_raphson(&mut single, &ybus, 1e-10, 30);
    let (p_single, _) = power_injections(&single, &ybus);
    let single_slack_output = p_single[0];

    let distribution = SlackDistribution::uniform(&buses);
    let (reports, report) =
        distributing_slack(&mut buses, &ybus, 1e-10, 30, JacobianBackend::Scalar, &distribution);

    assert!(reports.iter().all(|r| r.status == IslandStatus::Converged), "{reports:?}");
    assert!(report.converged, "{report:?}");

    let (p_calc, _) = power_injections(&buses, &ybus);
    let scheduled = effective_injection(&buses[0]).0;
    assert!(
        (p_calc[0] - scheduled).abs() < 1e-8,
        "slack produced {} against a final schedule of {scheduled}",
        p_calc[0]
    );

    // And it genuinely moved: the single-slack answer put the whole imbalance
    // on bus 0, so the two must differ by most of it.
    let total_shift: f64 = report.shift.iter().sum();
    assert!(
        (single_slack_output - p_calc[0]).abs() > 0.3 * total_shift.abs(),
        "the distributed answer ({}) is barely different from the single-slack one ({single_slack_output})",
        p_calc[0]
    );
}

/// The shifts sum to what the slack was carrying — the imbalance is moved, not
/// invented. A formulation that created power would still converge and would
/// still put the slack at its schedule; only this catches it.
#[test]
fn the_shifts_account_for_exactly_the_imbalance() {
    let (mut buses, ybus) = ring();
    let before: Vec<f64> = buses.iter().map(|b| b.p_spec).collect();

    let mut single = buses.clone();
    newton_raphson(&mut single, &ybus, 1e-10, 30);
    let (p_single, _) = power_injections(&single, &ybus);
    let imbalance = p_single[0] - effective_injection(&single[0]).0;

    let distribution = SlackDistribution::uniform(&buses);
    let (_, report) = distributing_slack(
        &mut buses, &ybus, 1e-10, 30, JacobianBackend::Scalar, &distribution,
    );

    let total_shift: f64 = report.shift.iter().sum();
    // Not exactly equal: redistributing changes the flows, and therefore the
    // losses, by a second-order amount. A per-cent bound is generous for that
    // and far tighter than any bookkeeping error would be.
    assert!(
        (total_shift - imbalance).abs() < 0.01 * imbalance.abs(),
        "shifts total {total_shift} against a single-slack imbalance of {imbalance}"
    );

    // And `shift` really is the difference between the schedules.
    for i in 0..buses.len() {
        assert!(
            (buses[i].p_spec - before[i] - report.shift[i]).abs() < 1e-12,
            "bus {i}: p_spec moved {} but shift says {}",
            buses[i].p_spec - before[i],
            report.shift[i]
        );
    }
}

/// Weights are honoured in proportion, and normalized — so passing raw
/// megawatt headroom works without the caller pre-dividing.
#[test]
fn participation_is_proportional_to_the_weights() {
    let (mut buses, ybus) = ring();
    // Bus 2 takes three times bus 1's share; the slack takes none, so it ends
    // up back at its own schedule exactly.
    let distribution = SlackDistribution::from_weights(vec![0.0, 25.0, 75.0, 0.0]);
    let (_, report) = distributing_slack(
        &mut buses, &ybus, 1e-10, 30, JacobianBackend::Scalar, &distribution,
    );
    assert!(report.converged, "{report:?}");

    assert_eq!(report.shift[0], 0.0, "the slack was given no share");
    assert_eq!(report.shift[3], 0.0, "a load bus was given no share");
    assert!(
        (report.shift[2] - 3.0 * report.shift[1]).abs() < 1e-9,
        "bus 2 took {} against bus 1's {}, which is not the 3:1 the weights asked for",
        report.shift[2],
        report.shift[1]
    );
}

/// The returned state is a genuine power flow of the final schedules — every
/// bus's computed injection matches what it is scheduled for.
///
/// This is what stops the loop from "converging" by corrupting the state:
/// re-derived from `power_injections`, sharing nothing with the outer loop's
/// own arithmetic.
#[test]
fn the_answer_is_still_a_valid_power_flow() {
    let (mut buses, ybus) = ring();
    let distribution = SlackDistribution::uniform(&buses);
    distributing_slack(
        &mut buses, &ybus, 1e-10, 30, JacobianBackend::Scalar, &distribution,
    );

    let (p_calc, q_calc) = power_injections(&buses, &ybus);
    for b in &buses {
        let (p_spec, q_spec) = effective_injection(b);
        if b.bus_type != BusType::Slack {
            assert!(
                (p_calc[b.idx] - p_spec).abs() < 1e-8,
                "bus {}: P mismatch {}",
                b.idx,
                p_calc[b.idx] - p_spec
            );
        }
        if b.bus_type == BusType::PQ {
            assert!(
                (q_calc[b.idx] - q_spec).abs() < 1e-8,
                "bus {}: Q mismatch {}",
                b.idx,
                q_calc[b.idx] - q_spec
            );
        }
    }
}

/// Converges in a handful of passes.
///
/// Pinned rather than left implicit, because a count that *grew* would mean
/// the first-order cancellation described on
/// `newton_raphson_distributing_slack` had stopped holding — a correctness
/// signal, not a performance one.
///
/// The bound is a handful rather than the one-or-two the cancellation alone
/// suggests, and the difference is worth recording: the leftover after each
/// pass is the *change in losses* from the redistributed flows, which shrinks
/// geometrically rather than vanishing. All three pglib cases measured take
/// seven passes to 1e-8; this toy ring is smaller and takes fewer.
#[test]
fn the_outer_loop_settles_in_a_handful_of_passes() {
    let (mut buses, ybus) = ring();
    let distribution = SlackDistribution::uniform(&buses);
    let (_, report) = distributing_slack(
        &mut buses, &ybus, 1e-10, 30, JacobianBackend::Scalar, &distribution,
    );
    assert!(report.converged);
    assert!(
        report.outer_iterations <= 10,
        "took {} outer passes; the first-order term should cancel in one",
        report.outer_iterations
    );
}

/// Two islands, each with its own slack, each distributing within itself.
///
/// Normalizing weights globally instead of per island would size one island's
/// correction by the other's generators — so the second island's slack is
/// given a deliberately different imbalance from the first's, which a global
/// normalization gets wrong in both.
#[test]
fn each_island_distributes_its_own_imbalance() {
    let mut buses = vec![
        // Island A: slack + one generator + load.
        bus(0, BusType::Slack, 0.0, 0.0),
        bus(1, BusType::PV, 0.20, 0.0),
        bus(2, BusType::PQ, -0.60, -0.10),
        // Island B: slack + one generator + a much bigger load.
        bus(3, BusType::Slack, 0.0, 0.0),
        bus(4, BusType::PV, 0.50, 0.0),
        bus(5, BusType::PQ, -2.00, -0.30),
    ];
    let lines = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.02, x: 0.07, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 3, to: 4, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 4, to: 5, r: 0.02, x: 0.07, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let ybus = build_ybus(6, &lines, &Vec::<Transformer>::new()).finish();

    let distribution = SlackDistribution::uniform(&buses);
    let (reports, report) = distributing_slack(
        &mut buses, &ybus, 1e-10, 30, JacobianBackend::Scalar, &distribution,
    );

    assert_eq!(reports.len(), 2, "expected two islands");
    assert!(reports.iter().all(|r| r.status == IslandStatus::Converged), "{reports:?}");
    assert!(report.converged, "{report:?}");
    assert!(report.undistributed.is_empty(), "{:?}", report.undistributed);

    let (p_calc, _) = power_injections(&buses, &ybus);
    for slack in [0usize, 3] {
        let scheduled = effective_injection(&buses[slack]).0;
        assert!(
            (p_calc[slack] - scheduled).abs() < 1e-8,
            "island slack {slack} produced {} against schedule {scheduled}",
            p_calc[slack]
        );
    }

    // Island B carries roughly four times island A's load, so its shift must
    // be visibly larger — the evidence that the two were sized separately.
    let shift_a = report.shift[0] + report.shift[1];
    let shift_b = report.shift[3] + report.shift[4];
    assert!(
        shift_b > 2.0 * shift_a,
        "island A shifted {shift_a} and island B {shift_b}; B should dominate"
    );
}

/// An island with no participating generator is left on its single slack and
/// **says so**, rather than silently doing nothing.
///
/// A caller who mis-specifies weights gets an ordinary single-slack answer,
/// which is a perfectly reasonable answer to a different question — so the
/// only thing that distinguishes it from success is this report.
#[test]
fn an_island_with_no_participants_is_reported_not_silently_skipped() {
    let (mut buses, ybus) = ring();
    // Weight only a load bus, which is not a generator.
    let distribution = SlackDistribution::from_weights(vec![0.0, 0.0, 0.0, 0.0]);
    let (reports, report) = distributing_slack(
        &mut buses, &ybus, 1e-10, 30, JacobianBackend::Scalar, &distribution,
    );

    assert!(reports.iter().all(|r| r.status == IslandStatus::Converged));
    assert_eq!(report.undistributed.len(), 1);
    assert_eq!(report.undistributed[0].1, "no participating generator in this island");
    assert!(report.shift.iter().all(|s| *s == 0.0), "nothing should have moved");
    // The untouched deviation is still reported, so a caller can see how much
    // the slack is carrying that nobody took.
    assert!(report.residual[0].abs() > 0.01, "residual {:?}", report.residual);
}

/// Every sparse backend gives the same answer — the outer loop touches
/// schedules, not the linear algebra.
#[test]
fn every_backend_agrees() {
    let mut answers = Vec::new();
    for backend in [JacobianBackend::Scalar, JacobianBackend::Block] {
        let (mut buses, ybus) = ring();
        let distribution = SlackDistribution::uniform(&buses);
        let (_, report) = distributing_slack(
            &mut buses, &ybus, 1e-10, 30, backend, &distribution,
        );
        assert!(report.converged, "{backend:?}: {report:?}");
        answers.push(report.shift);
    }
    for i in 0..answers[0].len() {
        assert!(
            (answers[0][i] - answers[1][i]).abs() < 1e-9,
            "bus {i}: {} vs {}",
            answers[0][i],
            answers[1][i]
        );
    }
}

/// A slack carrying a real schedule keeps it. With `p_spec = 0` the whole
/// output is redistributed; with a schedule set, only the excess is — which is
/// the distinction that makes the field meaningful rather than incidental.
#[test]
fn the_slacks_own_schedule_is_respected() {
    let (mut scheduled, ybus) = ring();
    scheduled[0].p_spec = 0.25;
    let distribution = SlackDistribution::from_weights(vec![0.0, 1.0, 1.0, 0.0]);
    let (_, report) = distributing_slack(
        &mut scheduled, &ybus, 1e-10, 30, JacobianBackend::Scalar, &distribution,
    );
    assert!(report.converged);

    let (p_calc, _) = power_injections(&scheduled, &ybus);
    assert!(
        (p_calc[0] - 0.25).abs() < 1e-8,
        "the slack was scheduled for 0.25 and produced {}",
        p_calc[0]
    );

    // Against an unscheduled slack, which has to be carried entirely by the
    // others: strictly more is shifted.
    let (mut unscheduled, ybus2) = ring();
    let (_, bare) = distributing_slack(
        &mut unscheduled,
        &ybus2,
        1e-10,
        30,
        JacobianBackend::Scalar,
        &SlackDistribution::from_weights(vec![0.0, 1.0, 1.0, 0.0]),
    );
    let with_schedule: f64 = report.shift.iter().sum();
    let without: f64 = bare.shift.iter().sum();
    assert!(
        without > with_schedule + 0.2,
        "unscheduled shifted {without}, scheduled {with_schedule} — the 0.25 should show"
    );
}
