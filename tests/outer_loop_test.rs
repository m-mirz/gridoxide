//! The outer-loop layer, and the capability it exists for.
//!
//! Before it, `newton_raphson_enforcing_q_limits` and
//! `newton_raphson_distributing_slack` were separate entry points, each with
//! its own solver and its own pass counter. A caller picked one. The tests
//! here are about what that made impossible, plus the driver's own contract:
//! ordering, termination, budget and invalidation.

use gridoxide::network::{build_ybus, effective_injection, power_injections, YBusSparse};
use gridoxide::outerloop::{
    DistributedSlack, Invalidates, OuterLoop, OuterLoopContext, OuterLoopStatus, ReactiveLimits,
    SlackDistribution, SolveContext,
};
use gridoxide::solver::{IslandStatus, JacobianBackend};
use gridoxide::types::{Bus, BusType, Line, Transformer};

fn bus(idx: usize, bus_type: BusType, p_spec: f64, q_spec: f64, q_min: f64, q_max: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec,
        q_spec,
        q_min,
        q_max,
        u_rated: 1.0,
        zip_terms: Vec::new(),
    }
}

/// A ring with resistive lines — so there are real losses to distribute — and
/// a generator whose reactive limit genuinely binds. Both controls have
/// something to do here, which is what makes the combination testable.
fn ring() -> (Vec<Bus>, YBusSparse) {
    let buses = vec![
        bus(0, BusType::Slack, 0.0, 0.0, -99.0, 99.0),
        // Tight enough that holding 1.0 pu is out of reach. Unenforced this
        // bus absorbs about -0.16 pu, so it is `q_min` that binds.
        bus(1, BusType::PV, 0.30, 0.0, -0.02, 0.02),
        bus(2, BusType::PV, 0.20, 0.0, -99.0, 99.0),
        bus(3, BusType::PQ, -1.00, -0.30, -99.0, 99.0),
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

/// **The capability this layer exists for.** Both controls in one solve, with
/// the answer satisfying both criteria at once — which neither single-loop
/// entry point could produce, because there was no way to ask for both.
#[test]
fn q_limits_and_distributed_slack_hold_simultaneously() {
    let (mut buses, ybus) = ring();
    let distribution = SlackDistribution::uniform(&buses);
    let mut ybus = ybus;

    let mut slack = DistributedSlack::new(distribution);
    let mut qlim = ReactiveLimits::new();
    let (islands, report) = {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus);
        // Innermost first: the reactive dispatch settles inside the slack
        // sharing, not the other way round.
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut slack, &mut qlim];
        gridoxide::outerloop::solve_with_loops(
            &mut ctx,
            1e-10,
            30,
            JacobianBackend::Scalar,
            &mut list,
            40,
        )
    };

    assert!(islands.iter().all(|i| i.status == IslandStatus::Converged), "{islands:?}");
    assert!(report.converged, "{report:?}");
    assert!(!report.budget_exhausted, "{report:?}");

    // Criterion 1: the reactive limit is respected, and by the mechanism that
    // respects it — the bus is off voltage control and pinned at its limit.
    assert_eq!(buses[1].bus_type, BusType::PQ, "bus 1 should have left voltage control");
    assert_eq!(qlim.switches(), [1], "exactly bus 1 should switch");
    let (p_calc, q_calc) = power_injections(&buses, &ybus);
    assert!(
        (q_calc[1] - buses[1].q_min).abs() < 1e-7,
        "bus 1 should sit on the limit it violated: Q = {}, q_min = {}",
        q_calc[1],
        buses[1].q_min
    );
    // And the switch pinned `q_spec` at that same limit, which is the
    // mechanism rather than a coincidence of the arithmetic.
    assert!((buses[1].q_spec - buses[1].q_min).abs() < 1e-12);

    // Criterion 2: the slack produces its own schedule plus its own share and
    // nothing more. Re-derived from `power_injections` at the returned state
    // rather than read out of the loop's own bookkeeping.
    let scheduled = effective_injection(&buses[0]).0;
    assert!(
        (p_calc[0] - scheduled).abs() < 1e-8,
        "slack produced {} against a schedule of {scheduled}",
        p_calc[0]
    );
    assert!(
        slack.report().shift.iter().sum::<f64>().abs() > 0.0,
        "something should actually have been distributed"
    );

    // Both loops are in the report, and in the order they were given.
    let names: Vec<&str> = report.loops.iter().map(|l| l.name).collect();
    assert_eq!(names, ["DistributedSlack", "ReactiveLimits"]);
}

/// The combination is not what either half produces alone. Without this, the
/// test above could pass with one loop silently doing nothing.
#[test]
fn neither_loop_alone_satisfies_both_criteria() {
    // Q-limits only: the reactive limit holds, the slack still carries
    // everything.
    let (mut buses, mut ybus) = ring();
    let mut qlim = ReactiveLimits::new();
    {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut qlim];
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-10, 30, JacobianBackend::Scalar, &mut list, 40);
    }
    let (p_calc, _) = power_injections(&buses, &ybus);
    let scheduled = effective_injection(&buses[0]).0;
    assert_eq!(buses[1].bus_type, BusType::PQ);
    assert!(
        (p_calc[0] - scheduled).abs() > 0.1,
        "with no slack sharing the slack should be well off its schedule, was {}",
        p_calc[0] - scheduled
    );

    // Distributed slack only: the slack is on its schedule, the reactive limit
    // is violated.
    let (mut buses, mut ybus) = ring();
    let distribution = SlackDistribution::uniform(&buses);
    let mut slack = DistributedSlack::new(distribution);
    {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut slack];
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-10, 30, JacobianBackend::Scalar, &mut list, 40);
    }
    let (p_calc, q_calc) = power_injections(&buses, &ybus);
    let scheduled = effective_injection(&buses[0]).0;
    assert!((p_calc[0] - scheduled).abs() < 1e-8);
    assert_eq!(buses[1].bus_type, BusType::PV, "no Q-limit loop, so no switch");
    assert!(
        q_calc[1] < buses[1].q_min - 1e-3,
        "the fixture must genuinely violate the limit when unenforced: Q = {}, q_min = {}",
        q_calc[1],
        buses[1].q_min
    );
}

/// An empty list is exactly an ordinary solve — same state, one inner solve,
/// nothing reported as having run.
#[test]
fn an_empty_list_is_a_plain_solve() {
    let (mut buses, mut ybus) = ring();
    let mut plain = buses.clone();
    gridoxide::solver::newton_raphson(&mut plain, &ybus, 1e-10, 30);

    let (islands, report) = {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus);
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-10, 30, JacobianBackend::Scalar, &mut [], 40)
    };

    assert!(islands.iter().all(|i| i.status == IslandStatus::Converged));
    assert_eq!(report.solves, 1);
    assert_eq!(report.total_iterations, 0);
    assert!(report.loops.is_empty());
    for (a, b) in buses.iter().zip(&plain) {
        assert_eq!(a.voltage_mag.to_bits(), b.voltage_mag.to_bits(), "bus {}", a.idx);
        assert_eq!(a.voltage_ang.to_bits(), b.voltage_ang.to_bits(), "bus {}", a.idx);
    }
}

// ---------------------------------------------------------------------------
// The driver's own contract, exercised with loops that do nothing physical
// ---------------------------------------------------------------------------

/// Reports `Unstable` a fixed number of times, then `Stable`. Records the
/// order in which the driver consulted it.
struct Counter {
    name: &'static str,
    unstable_for: usize,
    seen: usize,
    log: std::rc::Rc<std::cell::RefCell<Vec<&'static str>>>,
}

impl OuterLoop for Counter {
    fn name(&self) -> &'static str {
        self.name
    }
    fn invalidates(&self) -> Invalidates {
        Invalidates::Nothing
    }
    fn check(&mut self, _ctx: &mut OuterLoopContext<'_, '_>) -> OuterLoopStatus {
        self.log.borrow_mut().push(self.name);
        if self.seen < self.unstable_for {
            self.seen += 1;
            OuterLoopStatus::Unstable
        } else {
            OuterLoopStatus::Stable
        }
    }
}

/// Each loop runs to its *own* stability before the next is consulted, and the
/// whole list is re-walked once anything moves. Both halves matter: the first
/// is why an inner control is not left half-converged, the second is why an
/// outer control's decision is re-examined by the inner ones.
#[test]
fn loops_run_innermost_first_and_the_list_is_re_walked() {
    let (mut buses, mut ybus) = ring();
    let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut inner = Counter { name: "inner", unstable_for: 2, seen: 0, log: log.clone() };
    let mut outer = Counter { name: "outer", unstable_for: 1, seen: 0, log: log.clone() };

    let (_, report) = {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut inner, &mut outer];
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-10, 30, JacobianBackend::Scalar, &mut list, 40)
    };

    assert_eq!(
        *log.borrow(),
        [
            // `inner` is driven to its own stability first: unstable, unstable,
            // stable. Nothing else is consulted while it is still moving.
            "inner", "inner", "inner",
            // then `outer` is, and likewise runs to its own stability:
            // unstable, stable.
            "outer", "outer",
            // `outer` moved something, so the list is walked again from the
            // top — this is what re-exposes the inner loop to the outer one's
            // decision.
            "inner",
            // and the walk ends on reaching `outer`, the last loop that moved
            // anything, with nothing having changed since.
        ],
        "the driver's schedule changed"
    );
    assert!(report.converged, "{report:?}");
    assert_eq!(report.total_iterations, 3, "two from inner, one from outer");
    assert_eq!(report.solves, 4, "the initial solve plus one per move");
}

/// The budget caps total re-solves across every loop, and says so rather than
/// reporting a fixed point it never reached.
#[test]
fn the_budget_is_shared_and_exhaustion_is_reported() {
    let (mut buses, mut ybus) = ring();
    let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut never = Counter { name: "never", unstable_for: usize::MAX, seen: 0, log };

    let (_, report) = {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut never];
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-10, 30, JacobianBackend::Scalar, &mut list, 5)
    };

    assert!(report.budget_exhausted, "{report:?}");
    assert!(!report.converged, "a run that ran out of budget has not converged");
    assert_eq!(report.total_iterations, 5);
}

/// A failing loop stops the run and keeps its message, rather than being
/// retried until the budget runs out.
#[test]
fn a_failing_loop_stops_the_run() {
    struct Fails;
    impl OuterLoop for Fails {
        fn name(&self) -> &'static str {
            "Fails"
        }
        fn check(&mut self, _ctx: &mut OuterLoopContext<'_, '_>) -> OuterLoopStatus {
            OuterLoopStatus::Failed("deliberate".into())
        }
    }

    let (mut buses, mut ybus) = ring();
    let mut fails = Fails;
    let (_, report) = {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut fails];
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-10, 30, JacobianBackend::Scalar, &mut list, 40)
    };

    assert!(!report.converged);
    assert_eq!(report.total_iterations, 0, "a failure re-solves nothing");
    assert_eq!(
        report.loop_named("Fails").map(|l| l.status.clone()),
        Some(OuterLoopStatus::Failed("deliberate".into()))
    );
}

/// No loop is consulted when the initial solve did not settle: there is
/// nothing useful a control can decide from a state that is not a power flow.
///
/// Starving the inner Newton of iterations is the cleanest way to produce
/// that. Note what does *not* qualify: a sourceless island comes back as
/// `NoReferenceBus`, which is a reported verdict rather than a failed solve,
/// and the loops are still consulted for the islands that did converge.
#[test]
fn a_solve_that_does_not_settle_consults_no_loop() {
    let (mut buses, mut ybus) = ring();

    let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut counter = Counter { name: "unused", unstable_for: 3, seen: 0, log: log.clone() };
    let (islands, report) = {
        let mut ctx = SolveContext::new(&mut buses, &mut ybus);
        let mut list: Vec<&mut dyn OuterLoop> = vec![&mut counter];
        // One iteration cannot reach 1e-12 from a flat start.
        gridoxide::outerloop::solve_with_loops(&mut ctx, 1e-12, 1, JacobianBackend::Scalar, &mut list, 40)
    };

    assert!(
        islands.iter().all(|i| i.status == IslandStatus::MaxIterationsReached),
        "fixture assumption broken, this should not settle: {islands:?}"
    );
    assert!(log.borrow().is_empty(), "no loop should have been consulted");
    assert_eq!(report.solves, 1);
}
