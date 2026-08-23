//! Reactive-limit events on the continuation curve.
//!
//! The claim under test is narrow and worth stating precisely: continuation
//! reports the λ at which a generator *actually* saturates, not the λ of
//! whichever step happened to notice it had. The difference is two orders of
//! magnitude or more, and it is the whole reason `events.rs` exists — so the
//! gate here measures it rather than asserting it.
//!
//! The oracle is brute force through code that knows nothing about
//! continuation: bisect λ, solving an ordinary `run_power_flow` with
//! `enforce_q_limits` at each trial, and find where the ordinary
//! `ReactiveLimits` outer loop first clamps that bus. Slow, obviously correct,
//! and completely independent of everything under test.

mod common;

use gridoxide::continuation::{
    run_continuation, ContinuationOptions, CriticalPointKind, LoadingDirection,
};
use gridoxide::network::{build_ybus, power_injections};
use gridoxide::pgm::{pgm_to_buses_and_branches, PgmInput};
use gridoxide::solver::PowerFlowOptions;
use gridoxide::types::{Bus, BusType, Line};

fn fixture(name: &str) -> (Vec<Bus>, Vec<gridoxide::types::Line>, Vec<gridoxide::types::Transformer>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf")
        .join(format!("{name}.json"));
    let raw = std::fs::read_to_string(path).expect("committed pglib fixture");
    let input: PgmInput = serde_json::from_str(&raw).expect("parse");
    pgm_to_buses_and_branches(input, 100e6, 50.0)
}

fn scaled(start: &[Bus], dir: &LoadingDirection, lambda: f64) -> Vec<Bus> {
    start
        .iter()
        .map(|b| Bus {
            p_spec: b.p_spec + lambda * dir.d_p[b.idx],
            q_spec: b.q_spec + lambda * dir.d_q[b.idx],
            ..b.clone()
        })
        .collect()
}

fn q_limits_opts() -> ContinuationOptions {
    ContinuationOptions {
        power_flow: PowerFlowOptions { enforce_q_limits: true, ..Default::default() },
        ..Default::default()
    }
}

/// **The gate.** Every located event must sit where an ordinary
/// `enforce_q_limits` solve first clamps that machine — and turning the locator
/// off must measurably lose that.
///
/// The second half matters as much as the first. Without it the test would pass
/// just as happily if `locate_events` did nothing at all and the steps merely
/// happened to be small.
#[test]
fn located_events_match_a_brute_force_bisection() {
    for name in ["pglib_opf_case5_pjm", "pglib_opf_case14_ieee", "pglib_opf_case30_ieee"] {
        let (start, lines, transformers) = fixture(name);
        let direction = LoadingDirection::scale_loads(&start);

        let clamped_at = |lambda: f64| -> Vec<usize> {
            let report = gridoxide::run_power_flow(
                scaled(&start, &direction, lambda),
                &lines,
                &transformers,
                &[],
                gridoxide::TapData::none(),
                PowerFlowOptions {
                    enforce_q_limits: true,
                    max_iter: 60,
                    max_outer_iter: 60,
                    ..Default::default()
                },
            );
            let mut switches = report.outer.map(|o| o.q_limit_switches).unwrap_or_default();
            switches.sort_unstable();
            switches
        };
        // Machines already saturated in the base case have no crossing to find.
        let at_base = clamped_at(0.0);

        let mut worst = [0.0f64; 2];
        for (slot, locate) in [(0usize, true), (1usize, false)] {
            let curve = run_continuation(
                start.clone(),
                &lines,
                &transformers,
                &[],
                ContinuationOptions {
                    direction: direction.clone(),
                    locate_events: locate,
                    ..q_limits_opts()
                },
            );
            let ceiling = curve.lambda_max().expect("a nose was reported") * 1.05;

            for (bus, _, lambda) in curve.q_limit_events() {
                if at_base.binary_search(&bus).is_ok()
                    || clamped_at(ceiling).binary_search(&bus).is_err()
                {
                    continue;
                }
                let (mut lo, mut hi) = (0.0f64, ceiling);
                for _ in 0..45 {
                    let mid = 0.5 * (lo + hi);
                    if clamped_at(mid).binary_search(&bus).is_ok() {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                worst[slot] = worst[slot].max((lambda - 0.5 * (lo + hi)).abs());
            }
        }

        assert!(
            worst[0] < 1e-5,
            "{name}: located events are off by {:.2e}, which is not 'located'",
            worst[0]
        );
        assert!(
            worst[1] > 20.0 * worst[0].max(1e-9),
            "{name}: switching at step granularity was off by only {:.2e} against the \
             locator's {:.2e} — either the steps happened to be tiny or the locator is \
             not doing anything, and this test would not tell the difference",
            worst[1],
            worst[0]
        );
    }
}

/// The invariant the whole event mechanism exists to keep: no point on the
/// returned curve may have a machine outside its reactive limits, and every
/// machine that was clamped must be pinned *at* its limit for the rest of the
/// trace.
///
/// The second half is the one that bites. A clamped bus has its `q_spec`
/// rewritten to the limit by `ReactiveLimits`; if the loading direction is not
/// frozen there too, every later step ramps it straight back off the limit. The
/// curve still converges at every step, so nothing looks wrong — the reactive
/// limit is simply not enforced any more.
#[test]
fn no_point_on_the_curve_violates_a_reactive_limit() {
    for name in ["pglib_opf_case5_pjm", "pglib_opf_case14_ieee", "pglib_opf_case30_ieee"] {
        let (start, lines, transformers) = fixture(name);
        let direction = LoadingDirection::scale_loads(&start);
        let ybus = build_ybus(start.len(), &lines, &transformers).finish();

        let curve = run_continuation(
            start.clone(),
            &lines,
            &transformers,
            &[],
            ContinuationOptions { direction, ..q_limits_opts() },
        );

        for (n, point) in curve.points.iter().enumerate() {
            let state: Vec<Bus> = (0..start.len())
                .map(|i| Bus {
                    bus_type: point.bus_type[i],
                    voltage_mag: point.voltage_mag[i],
                    voltage_ang: point.voltage_ang[i],
                    ..start[i].clone()
                })
                .collect();
            let (_, q_calc) = power_injections(&state, &ybus);

            for b in &state {
                match b.bus_type {
                    // Still on voltage control: it must be inside its range.
                    BusType::PV => assert!(
                        q_calc[b.idx] <= b.q_max + 1e-5 && q_calc[b.idx] >= b.q_min - 1e-5,
                        "{name} point {n} (lambda {:.6}): bus {} still holds voltage but \
                         needs Q = {:.6}, outside [{:.6}, {:.6}]",
                        point.lambda,
                        b.idx,
                        q_calc[b.idx],
                        b.q_min,
                        b.q_max
                    ),
                    // Clamped: it must be sitting *on* a limit, not drifting off
                    // it. `start`'s type says it was a voltage-controlling
                    // machine to begin with.
                    BusType::PQ if start[b.idx].bus_type == BusType::PV => assert!(
                        (q_calc[b.idx] - b.q_max).abs() < 1e-5
                            || (q_calc[b.idx] - b.q_min).abs() < 1e-5,
                        "{name} point {n} (lambda {:.6}): bus {} was clamped but is now at \
                         Q = {:.6}, neither limit ({:.6}, {:.6}) — the direction ramped it \
                         off its own clamp",
                        point.lambda,
                        b.idx,
                        q_calc[b.idx],
                        b.q_min,
                        b.q_max
                    ),
                    _ => {}
                }
            }
        }
    }
}

/// Reactive limits can only ever shrink the margin. A machine that stops
/// holding its voltage cannot make the system carry *more* load.
#[test]
fn enforcing_reactive_limits_lowers_the_nose() {
    for name in ["pglib_opf_case5_pjm", "pglib_opf_case14_ieee", "pglib_opf_case30_ieee"] {
        let (start, lines, transformers) = fixture(name);
        let direction = LoadingDirection::scale_loads(&start);

        let nose = |enforce: bool| {
            run_continuation(
                start.clone(),
                &lines,
                &transformers,
                &[],
                ContinuationOptions {
                    direction: direction.clone(),
                    power_flow: PowerFlowOptions {
                        enforce_q_limits: enforce,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .lambda_max()
            .unwrap_or_else(|| panic!("{name} (enforce={enforce}) found no nose"))
        };

        let (free, limited) = (nose(false), nose(true));
        assert!(
            limited < free,
            "{name}: reactive limits raised the nose from {free} to {limited}, which is \
             not physically possible"
        );
    }
}

/// A limit-induced bifurcation is a **different answer** from a fold, and it is
/// easy to miss: the Jacobian is not singular there, so a continuation watching
/// only for `dλ/dσ = 0` marches straight past it down the lower branch and
/// reports a λ_max that no operating point can reach.
///
/// `pglib_opf_case118_ieee` has one. What is asserted is the physics rather than
/// the exact λ: the reported maximum must coincide with the last reactive-limit
/// event, and an ordinary solve just past it must fail.
#[test]
fn case118_maximum_is_limit_induced_and_sits_on_its_event() {
    let (start, lines, transformers) = fixture("pglib_opf_case118_ieee");
    let direction = LoadingDirection::scale_loads(&start);
    let curve = run_continuation(
        start.clone(),
        &lines,
        &transformers,
        &[],
        ContinuationOptions { direction: direction.clone(), ..q_limits_opts() },
    );

    let critical = curve.critical.as_ref().expect("a maximum was reported");
    let CriticalPointKind::LimitInduced { bus } = critical.kind else {
        panic!("expected a limit-induced maximum, got {:?}", critical.kind);
    };

    let events = curve.q_limit_events();
    let &(event_bus, _, event_lambda) =
        events.last().expect("a limit-induced maximum implies at least one event");
    assert_eq!(event_bus, bus, "the maximum must be the machine that just saturated");
    assert!(
        (event_lambda - critical.lambda_max).abs() < 1e-9,
        "lambda_max {} should be the event lambda {event_lambda}",
        critical.lambda_max
    );

    let past = gridoxide::run_power_flow(
        scaled(&start, &direction, critical.lambda_max * 1.05),
        &lines,
        &transformers,
        &[],
        gridoxide::TapData::none(),
        PowerFlowOptions {
            enforce_q_limits: true,
            max_iter: 80,
            max_outer_iter: 60,
            ..Default::default()
        },
    );
    assert_ne!(
        past.stats.status,
        gridoxide::solver::SolveStatus::Converged,
        "past a limit-induced maximum there should be no operating point to find"
    );
}

/// A machine with a wide enough range never saturates, so the curve is one
/// segment and the answer is the same as with limits switched off entirely.
#[test]
fn unreachable_limits_leave_the_curve_alone() {
    let buses = vec![
        Bus {
            idx: 0,
            bus_type: BusType::Slack,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: -99.0,
            q_max: 99.0,
            u_rated: 1.0,
            zip_terms: vec![],
        },
        Bus {
            idx: 1,
            bus_type: BusType::PV,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.3,
            q_spec: 0.0,
            q_min: -99.0,
            q_max: 99.0,
            u_rated: 1.0,
            zip_terms: vec![],
        },
        Bus {
            idx: 2,
            bus_type: BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: -0.6,
            q_spec: -0.2,
            q_min: -99.0,
            q_max: 99.0,
            u_rated: 1.0,
            zip_terms: vec![],
        },
    ];
    let lines = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.03, x: 0.09, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 0, to: 2, r: 0.04, x: 0.12, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let direction = LoadingDirection::scale_loads(&buses);

    let with = run_continuation(
        buses.clone(),
        &lines,
        &[],
        &[],
        ContinuationOptions { direction: direction.clone(), ..q_limits_opts() },
    );
    let without = run_continuation(
        buses,
        &lines,
        &[],
        &[],
        ContinuationOptions { direction, ..Default::default() },
    );

    assert!(
        with.q_limit_events().is_empty(),
        "nothing should saturate: {:?}",
        with.q_limit_events()
    );
    assert_eq!(with.segments, 1, "no bus type changed, so there is one segment");
    let (a, b) = (with.lambda_max().unwrap(), without.lambda_max().unwrap());
    assert!((a - b).abs() < 1e-9, "limits that never bind changed the nose: {a} vs {b}");
}
