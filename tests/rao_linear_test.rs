//! The linear optimization of range actions — phase 7 of `plans/RAO_PLAN.md`.
//!
//! The load-bearing test here is `phase_shift_sensitivity_matches_a_finite_
//! difference`. Everything else in this layer is bookkeeping around one
//! quantity — how much a phase shifter moves a flow — and if that quantity is
//! wrong the optimizer will confidently move shifters the wrong way. It is also
//! exactly the sort of derivative that is easy to get almost right: the term
//! that a shift contributes to *its own* branch's flow is separate from the
//! term it contributes through the angles, and dropping either leaves a number
//! that looks plausible.

use std::path::PathBuf;

use gridoxide::linear::btheta::{dc_branches, dc_power_flow};
use gridoxide::linear::sensitivity::DcSensitivity;
use gridoxide::opf::ipm::IpmSolver;
use gridoxide::rao::crac::*;
use gridoxide::rao::linear::{phase_shift_sensitivity, LinearOptions, LinearStatus};
use gridoxide::rao::{crac_json, evaluate, optimize, Network, NetworkMut, Resolution};
use gridoxide::ucte;

fn ucte_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ucte").join(name)
}

fn rao_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/rao").join(name)
}

// ---------------------------------------------------------------------------
// The derivative
// ---------------------------------------------------------------------------

/// The analytic sensitivity of every flow to a phase shift, against a re-solve.
///
/// A shift enters the DC problem twice: as an injection of `+bα` at `from` and
/// `−bα` at `to`, and explicitly in the shifting branch's own flow
/// `b(θf − θt − α)`. So its own branch has a term no other branch has. Omitting
/// it gives a sensitivity that is correct everywhere except on the one branch
/// the operator is actually moving.
#[test]
fn phase_shift_sensitivity_matches_a_finite_difference() {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let options = Default::default();
    let branches = dc_branches(&net.lines, &net.transformers, options);
    let n_branches = net.n_branches();
    let sensitivity =
        DcSensitivity::new(&net.buses, &branches, n_branches).expect("sensitivity");

    // The one transformer in this network is a phase shifter.
    assert_eq!(net.transformers.len(), 1);
    let branch = net.transformer_branch(0);
    let analytic = phase_shift_sensitivity(&sensitivity, &branches, branch).expect("psdf");

    // Central difference on the shift itself, at ±0.001 rad.
    let h = 1e-3;
    let mut flows = Vec::new();
    for step in [-h, h] {
        let mut transformers = net.transformers.clone();
        let ratio = transformers[0].tap.norm();
        let angle = transformers[0].tap.arg() + step;
        transformers[0].tap = num_complex::Complex::from_polar(ratio, angle);
        let mut buses = net.buses.clone();
        flows.push(dc_power_flow(&mut buses, &net.lines, &transformers, options).branch_p);
    }

    let mut compared = 0;
    let mut worst: f64 = 0.0;
    for b in 0..n_branches {
        let numeric = (flows[1][b] - flows[0][b]) / (2.0 * h);
        let error = (analytic[b] - numeric).abs();
        worst = worst.max(error);
        if analytic[b].abs() > 1e-9 || numeric.abs() > 1e-9 {
            compared += 1;
        }
    }
    assert!(compared >= 5, "only {compared} branches respond to the shifter");
    assert!(worst < 1e-6, "worst sensitivity error {worst} pu/rad");

    // And the shifting branch's own sensitivity must be the largest in
    // magnitude — the direct term dominates. If it were dropped this branch
    // would come out with the same small number as its neighbours.
    let own = analytic[branch].abs();
    assert!(
        analytic.iter().all(|s| s.abs() <= own + 1e-9),
        "the shifter's own branch should respond most strongly"
    );
    assert!(own > 1.0, "own sensitivity {own} looks like the direct term is missing");
}

#[test]
fn a_shift_does_not_move_a_radial_branch() {
    // A DC sensitivity has to respect the topology: a branch with no parallel
    // path carries whatever its bus demands regardless of any shifter.
    let net = ucte::read(ucte_fixture("3nodes_pst.uct")).expect("network");
    let options = Default::default();
    let branches = dc_branches(&net.lines, &net.transformers, options);
    let sensitivity =
        DcSensitivity::new(&net.buses, &branches, net.n_branches()).expect("sensitivity");
    let branch = net.transformer_branch(0);
    let analytic = phase_shift_sensitivity(&sensitivity, &branches, branch).expect("psdf");
    for (b, s) in analytic.iter().enumerate() {
        if sensitivity.is_radial(b) {
            assert!(s.abs() < 1e-9, "radial branch {b} responded {s} to a shift");
        }
    }
}

// ---------------------------------------------------------------------------
// The optimization
// ---------------------------------------------------------------------------

struct Case {
    net: ucte::UcteImport,
    crac: Crac,
}

fn case() -> Case {
    let net = ucte::read(ucte_fixture("TestCase12Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(rao_fixture("crac-for-12nodes.json")).expect("crac");
    Case { net, crac }
}

impl Case {
    fn resolution(&self) -> Resolution {
        Resolution::new(&self.crac, &self.net.branch_ids)
    }
}

/// The preventive perimeter of the vendored case is overloaded, and there are
/// two phase shifters available. The optimizer must find a better minimum
/// margin than doing nothing.
#[test]
fn optimizing_a_perimeter_improves_its_worst_margin() {
    let c = case();
    let resolution = c.resolution();
    let preventive = State::preventive(c.crac.preventive_instant().expect("preventive"));

    let mut transformers = c.net.transformers.clone();
    let mut buses = c.net.buses.clone();
    let mut network = NetworkMut {
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let result = optimize(
        &c.crac,
        &mut network,
        &resolution,
        std::slice::from_ref(&preventive),
        &mut solver,
        &LinearOptions::default(),
    );

    assert!(result.initial_margin_mw < 0.0, "the fixture should start overloaded");
    assert_eq!(result.status, LinearStatus::Improved, "{result:?}");
    assert!(
        result.improvement() > 0.0,
        "margin went from {} to {}",
        result.initial_margin_mw,
        result.final_margin_mw
    );
    assert!(result.setpoints.iter().any(|s| s.moved()), "nothing was moved");
    assert!(result.iterations >= 1);
}

/// Every set-point returned must be a real tap, not an angle between two.
#[test]
fn a_phase_shifter_lands_on_a_tap_the_operator_can_select() {
    let c = case();
    let resolution = c.resolution();
    let preventive = State::preventive(c.crac.preventive_instant().unwrap());
    let mut transformers = c.net.transformers.clone();
    let mut buses = c.net.buses.clone();
    let mut network = NetworkMut {
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let result =
        optimize(&c.crac, &mut network, &resolution, std::slice::from_ref(&preventive), &mut solver, &LinearOptions::default());

    for setpoint in &result.setpoints {
        let Some(tap) = setpoint.tap else { continue };
        let RangeActionKind::Pst { tap_to_angle, .. } = &c.crac.range_actions[setpoint.action].kind
        else {
            panic!("a tap on a non-PST action")
        };
        let angle = tap_to_angle
            .iter()
            .find(|(t, _)| *t == tap)
            .map(|(_, a)| *a)
            .unwrap_or_else(|| panic!("tap {tap} is not in the action's table"));
        assert!(
            (setpoint.value - angle).abs() < 1e-9,
            "set-point {} is not the angle of tap {tap} ({angle})",
            setpoint.value
        );
    }
}

/// The network the caller holds must match the answer.
///
/// The optimizer applies set-points as it goes, so a search tree evaluating the
/// next candidate sees the network this left behind. If a rejected move were
/// left applied, or an accepted one not applied, the two would drift apart and
/// every later evaluation would be against a network nobody described.
#[test]
fn the_network_is_left_where_the_result_says_it_is() {
    let c = case();
    let resolution = c.resolution();
    let preventive = State::preventive(c.crac.preventive_instant().unwrap());
    let mut transformers = c.net.transformers.clone();
    let mut buses = c.net.buses.clone();
    let mut network = NetworkMut {
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let result =
        optimize(&c.crac, &mut network, &resolution, std::slice::from_ref(&preventive), &mut solver, &LinearOptions::default());

    // Re-evaluate from scratch against the network as it now stands.
    let view = Network {
        buses: &c.net.buses,
        lines: &c.net.lines,
        transformers: &transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let after = evaluate(&c.crac, &view, &resolution);
    let margin = after
        .perimeters
        .iter()
        .find(|p| p.state == preventive)
        .and_then(|p| p.min_margin())
        .expect("a preventive perimeter");
    assert!(
        (margin - result.final_margin_mw).abs() < 1e-6,
        "result says {} MW, the network says {margin} MW",
        result.final_margin_mw
    );
}

/// A perimeter with nothing available is not a failure.
#[test]
fn a_perimeter_with_no_range_actions_reports_no_improvement() {
    let c = case();
    let mut crac = c.crac.clone();
    crac.range_actions.clear();
    let resolution = Resolution::new(&crac, &c.net.branch_ids);
    let preventive = State::preventive(crac.preventive_instant().unwrap());
    let mut transformers = c.net.transformers.clone();
    let before = transformers.clone();
    let mut buses = c.net.buses.clone();
    let mut network = NetworkMut {
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let result =
        optimize(&crac, &mut network, &resolution, std::slice::from_ref(&preventive), &mut solver, &LinearOptions::default());

    assert_eq!(result.status, LinearStatus::NoImprovement);
    assert!(result.setpoints.is_empty());
    assert_eq!(result.initial_margin_mw, result.final_margin_mw);
    // And nothing was touched.
    for (a, b) in transformers.iter().zip(&before) {
        assert_eq!(a.tap, b.tap);
    }
}

/// The penalty must break ties without distorting the answer.
#[test]
fn the_movement_penalty_prefers_the_smaller_move() {
    let c = case();
    let resolution = c.resolution();
    let preventive = State::preventive(c.crac.preventive_instant().unwrap());

    let run = |penalty: f64| {
        let mut transformers = c.net.transformers.clone();
        let mut buses = c.net.buses.clone();
        let mut network = NetworkMut {
            buses: &mut buses,
            lines: &c.net.lines,
            transformers: &mut transformers,
            branch_ids: &c.net.branch_ids,
            bus_ids: &[],
            initially_open: &[],
            tap_changers: &[],
            base_mva: c.net.base_mva,
        };
        let mut solver = IpmSolver::new();
        let options = LinearOptions { pst_penalty: penalty, ..Default::default() };
        optimize(&c.crac, &mut network, &resolution, std::slice::from_ref(&preventive), &mut solver, &options)
    };

    let cheap = run(0.0);
    let dear = run(50.0);
    // A large penalty must not make the answer *worse* than doing nothing.
    assert!(dear.final_margin_mw >= dear.initial_margin_mw - 1e-9);
    // And it should not move further than the unpenalized run.
    let movement = |r: &gridoxide::rao::LinearResult| -> f64 {
        r.setpoints.iter().map(|s| (s.value - s.initial).abs()).sum()
    };
    assert!(
        movement(&dear) <= movement(&cheap) + 1e-9,
        "penalised run moved {} vs {}",
        movement(&dear),
        movement(&cheap)
    );
}

/// A post-contingency perimeter optimizes against the post-contingency flows.
#[test]
fn a_curative_perimeter_optimizes_its_own_state() {
    let c = case();
    let resolution = c.resolution();
    let curative = c
        .crac
        .states()
        .into_iter()
        .find(|s| {
            s.contingency.is_some()
                && c.crac.instants[s.instant].kind == InstantKind::Curative
        })
        .expect("a curative state");

    let mut transformers = c.net.transformers.clone();
    let mut buses = c.net.buses.clone();
    let mut network = NetworkMut {
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let result =
        optimize(&c.crac, &mut network, &resolution, std::slice::from_ref(&curative), &mut solver, &LinearOptions::default());

    // The starting margin here must be the *post-contingency* one, which
    // differs from the preventive margin — optimizing the wrong perimeter is
    // the mistake this catches.
    let view = Network {
        buses: &c.net.buses,
        lines: &c.net.lines,
        transformers: &c.net.transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let baseline = evaluate(&c.crac, &view, &resolution);
    let expected = baseline
        .perimeters
        .iter()
        .find(|p| p.state == curative)
        .and_then(|p| p.min_margin())
        .expect("curative perimeter");
    assert!(
        (result.initial_margin_mw - expected).abs() < 1e-6,
        "optimizer started from {} MW, the curative perimeter is at {expected} MW",
        result.initial_margin_mw
    );
}

/// A perimeter whose CNECs are all *monitored* has nothing to maximize.
///
/// An MNEC's margin must not get worse; it is not something to spend remedial
/// actions improving. So a perimeter containing only MNECs correctly optimizes
/// nothing, and this pins that as intended rather than as an accident of the
/// filter. The vendored case's auto perimeter is exactly this shape.
#[test]
fn a_perimeter_of_monitored_only_cnecs_optimizes_nothing() {
    let c = case();
    let auto = c
        .crac
        .states()
        .into_iter()
        .find(|s| c.crac.instants[s.instant].kind == InstantKind::Auto)
        .expect("an auto state");
    let here: Vec<&FlowCnec> =
        c.crac.flow_cnecs.iter().filter(|x| x.state == auto).collect();
    assert!(!here.is_empty(), "the auto perimeter should have CNECs");
    assert!(
        here.iter().all(|x| !x.optimized && x.monitored),
        "this fixture's auto perimeter is supposed to be MNEC-only"
    );

    let resolution = c.resolution();
    let mut transformers = c.net.transformers.clone();
    let before = transformers.clone();
    let mut buses = c.net.buses.clone();
    let mut network = NetworkMut {
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let result =
        optimize(&c.crac, &mut network, &resolution, std::slice::from_ref(&auto), &mut solver, &LinearOptions::default());
    assert_eq!(result.status, LinearStatus::NoImprovement);
    assert_eq!(result.iterations, 0, "nothing to optimize means no LP is built");
    for (a, b) in transformers.iter().zip(&before) {
        assert_eq!(a.tap, b.tap, "an MNEC-only perimeter must not move anything");
    }
}

/// A range action is offered only where its usage rules reach.
#[test]
fn usage_rules_decide_which_perimeter_an_action_reaches() {
    // `PRA_PST_BE` is preventive-only and `ARA_PST_DE` is auto-only, so the
    // outage and curative perimeters have no range action at all — they carry
    // only *network* actions, which this layer does not handle.
    let c = case();
    let resolution = c.resolution();
    let mut moved_in = Vec::new();
    for state in c.crac.states() {
        let mut transformers = c.net.transformers.clone();
        let mut buses = c.net.buses.clone();
        let mut network = NetworkMut {
            buses: &mut buses,
            lines: &c.net.lines,
            transformers: &mut transformers,
            branch_ids: &c.net.branch_ids,
            bus_ids: &[],
            initially_open: &[],
            tap_changers: &[],
            base_mva: c.net.base_mva,
        };
        let mut solver = IpmSolver::new();
        let result = optimize(
            &c.crac,
            &mut network,
            &resolution,
            std::slice::from_ref(&state),
            &mut solver,
            &LinearOptions::default(),
        );
        if result.setpoints.iter().any(|s| s.moved()) {
            moved_in.push(c.crac.instants[state.instant].kind);
        }
    }
    assert_eq!(
        moved_in,
        vec![InstantKind::Preventive],
        "only the preventive perimeter has an available range action"
    );
}
