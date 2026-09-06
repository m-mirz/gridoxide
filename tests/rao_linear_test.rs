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
use gridoxide::rao::{crac_json, evaluate, optimize, run, Network, NetworkMut, Resolution, SearchOptions};
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
        generation: &c.net.generation,
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
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
        generation: &c.net.generation,
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
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
        generation: &c.net.generation,
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
        tap_changers: &[],
        base_mva: c.net.base_mva,
    };
    let mut solver = IpmSolver::new();
    let result =
        optimize(&c.crac, &mut network, &resolution, std::slice::from_ref(&preventive), &mut solver, &LinearOptions::default());

    // Re-evaluate from scratch against the network as it now stands.
    let view = Network {
        generation: &c.net.generation,
        buses: &c.net.buses,
        lines: &c.net.lines,
        transformers: &transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
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
        generation: &c.net.generation,
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
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
            generation: &c.net.generation,
            buses: &mut buses,
            lines: &c.net.lines,
            transformers: &mut transformers,
            branch_ids: &c.net.branch_ids,
            bus_ids: &[],
            initially_open: &[],
            bus_countries: &[],
            shunts: &[],
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
        generation: &c.net.generation,
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
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
        generation: &c.net.generation,
        buses: &c.net.buses,
        lines: &c.net.lines,
        transformers: &c.net.transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
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
        generation: &c.net.generation,
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
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
            generation: &c.net.generation,
            buses: &mut buses,
            lines: &c.net.lines,
            transformers: &mut transformers,
            branch_ids: &c.net.branch_ids,
            bus_ids: &[],
            initially_open: &[],
            bus_countries: &[],
            shunts: &[],
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

// ---------------------------------------------------------------------------
// The objective's unit
// ---------------------------------------------------------------------------

fn features_fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/rao/features")
        .join(name)
}

#[test]
fn the_objective_unit_changes_what_the_optimizer_maximizes() {
    // `RaoUtil.getFlowUnit`: megawatts for a DC load flow, amperes for an AC
    // one. It is not cosmetic. Each CNEC converts at its own voltage, so on a
    // network with two voltage levels the same set of margins does not have the
    // same minimum in the two units — and the optimizer trades one CNEC against
    // another differently as a result.
    let net = ucte::read(ucte_fixture("TestCase12Nodes_with_2_voltage_levels_1.uct"))
        .expect("network");
    // This CRAC states 225 kV on some CNECs and 400 kV on others, which is what
    // makes the two units disagree; a CRAC that says 400 everywhere would make
    // the ampere objective a constant rescaling of the megawatt one.
    let (crac, _) =
        crac_json::read(features_fixture("SL_ep15us3case1.json")).expect("crac");
    let resolution = Resolution::new(&crac, &net.branch_ids);

    let view = Network {
        generation: &net.generation,
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };
    let result = evaluate(&crac, &view, &resolution);
    let cnecs: Vec<_> = result.perimeters.iter().flat_map(|p| p.cnecs.iter()).collect();
    assert!(cnecs.len() > 1, "need several CNECs to have an ordering at all");

    let voltages: std::collections::BTreeSet<i64> =
        cnecs.iter().map(|c| c.conversion_v as i64).collect();
    assert!(voltages.len() > 1, "fixture should span two voltage levels: {voltages:?}");

    // The property that matters is not that the *minimum* differs — the same
    // CNEC can be worst in both units, and here it is — but that the ordering
    // is not preserved. Once two CNECs rank differently, a candidate network
    // that helps one at the other's expense is an improvement in one unit and a
    // regression in the other, which is exactly what the optimizer decides on.
    let inverted = cnecs.iter().enumerate().any(|(i, a)| {
        cnecs.iter().skip(i + 1).any(|b| {
            (a.margin_mw < b.margin_mw && a.margin_a > b.margin_a)
                || (a.margin_mw > b.margin_mw && a.margin_a < b.margin_a)
        })
    });
    assert!(
        inverted,
        "no pair of CNECs ranks differently in the two units, so this fixture \
         cannot distinguish them and the test proves nothing"
    );
}

#[test]
fn an_ampere_objective_is_reported_in_megawatts_all_the_same() {
    // The optimizer may rank in amperes, but `final_margin_mw` is a margin in
    // megawatts and has to stay one. Reporting the scoring function under a
    // `_mw` name would be wrong by a factor of the conversion voltage.
    let c = case();
    let mut buses = c.net.buses.clone();
    let mut transformers = c.net.transformers.clone();
    let mut network = NetworkMut {
        generation: &c.net.generation,
        buses: &mut buses,
        lines: &c.net.lines,
        transformers: &mut transformers,
        branch_ids: &c.net.branch_ids,
        bus_ids: &[],
        initially_open: &[],
        bus_countries: &[],
        shunts: &[],
        tap_changers: &c.net.tap_changers,
        base_mva: c.net.base_mva,
    };
    let resolution = Resolution::new(&c.crac, &c.net.branch_ids);
    let perimeter: Vec<State> = c.crac.states();
    let mut solver = IpmSolver::new();
    let options = LinearOptions {
        objective_unit: gridoxide::rao::linear::ObjectiveUnit::Ampere,
        ..Default::default()
    };
    let result =
        optimize(&c.crac, &mut network, &resolution, &perimeter, &mut solver, &options);

    // Amperes are the larger number at these voltages, so the two must not be
    // equal and the MW one must be the smaller.
    assert!(
        result.final_objective.abs() > result.final_margin_mw.abs(),
        "objective {} should be an ampere figure, margin {} a megawatt one",
        result.final_objective,
        result.final_margin_mw
    );
}

/// A redispatch's set-point is **absolute**, and its sensitivity is per
/// megawatt of it.
///
/// Both halves were wrong together, and each hid the other. The control was
/// anchored at zero, so the CRAC's declared range was read as a range on the
/// *shift* — on the reference's 2.3.1.1.a that turned `[−1000, 1000]` into
/// `[0, 2000]` in the reference's own terms. And the sensitivity was the
/// per-unit response to a per-unit pattern, a hundred times too small at a
/// 100 MVA base.
///
/// An LP told a control is a hundred times weaker than it is still moves it the
/// right way and saturates at its bound; `apply` then does the arithmetic in MW
/// and gets the move right, and the iterate-and-relinearize loop keeps it
/// because the true objective improved. On every vendored fixture the answer
/// sits exactly on the bound, so saturating *was* optimal — which is why four
/// scenarios passed for years with a factor of a hundred in the model.
///
/// This asserts the model rather than the answer: two nodes, one line, a
/// generator at each end, and a redispatch whose keys are +1 and −1. The
/// set-point starts at 1000 because that is what the machines are at, and
/// moving it to 0 shuts both down — the reference's own words for this case.
#[test]
fn a_redispatch_starts_at_the_set_point_its_machines_are_already_on() {
    let net = ucte::read(ucte_fixture("2Nodes.uct")).expect("network");
    let (crac, _) = crac_json::read(rao_fixture("features/crac-93-1-1.json")).expect("crac");
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        generation: &net.generation,
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &net.shunts,
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };

    let action = crac
        .range_actions
        .iter()
        .position(|a| a.id == "redispatchingAction")
        .expect("the redispatch action");
    let origin = gridoxide::rao::injection_setpoint(&crac, &network, &resolution, action)
        .expect("both machines agree on where the action sits");
    assert!(
        (origin - 1000.0).abs() < 1e-6,
        "FFR1AA1 generates 1000 MW on a key of 1, so the set-point starts at 1000, not {origin}"
    );

    // And the optimizer takes it to zero, which is the whole scenario: the line
    // carries 1000 MW against a 300 MW threshold, and shutting both machines
    // down is the only thing that helps.
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &network, &resolution, &mut solver, &SearchOptions::default());
    let setpoint = plan
        .preventive
        .setpoints
        .iter()
        .find(|s| s.action == action)
        .expect("the action should be used");
    assert!(
        setpoint.value.abs() < 1e-3,
        "the optimizer should shut both machines down, not move to {}",
        setpoint.value
    );
    assert!(
        (setpoint.initial - 1000.0).abs() < 1e-6,
        "and it should report where it started, not zero ({})",
        setpoint.initial
    );
}

/// The cost objective, against the arithmetic the reference's own scenario
/// spells out.
///
/// 3.4.1.1 is two nodes joined by four parallel lines, three of them open, and
/// three network actions that each close one — priced at 100, 500 and 10. Its
/// comments state the sums: *"Overload penalty (250 * 1000)"* for the untouched
/// network, and *"Activation of closeBeFr4 (10)"* for the answer.
///
/// So this asserts the two numbers the scenario asserts, and one thing it
/// cannot: that the **cheapest** action is chosen. All three close a line and
/// all three relieve the overload, so a margin objective would rank them by the
/// margin each buys and a cost objective ranks them by price — which is the
/// distinction the whole feature exists to draw.
#[test]
fn a_cost_objective_buys_the_cheapest_answer_not_the_roomiest() {
    use gridoxide::rao::{Costly, CostlyOptions};
    use gridoxide::rao::linear::ObjectiveKind;

    let net = ucte::read(ucte_fixture("2Nodes4ParallelLines.uct")).expect("network");
    let (crac, _) = crac_json::read(rao_fixture("features/crac-92-1-1.json")).expect("crac");
    let resolution = Resolution::with_buses(&crac, &net.branch_ids, &net.node_codes);
    let network = Network {
        generation: &net.generation,
        buses: &net.buses,
        lines: &net.lines,
        transformers: &net.transformers,
        branch_ids: &net.branch_ids,
        bus_ids: &net.node_codes,
        initially_open: &net.initially_open,
        bus_countries: &net.bus_countries,
        shunts: &net.shunts,
        tap_changers: &net.tap_changers,
        base_mva: net.base_mva,
    };

    // The three actions and their stated prices, so the assertion below is
    // about *which* is cheapest rather than about a name.
    let priced: Vec<(&str, f64)> = crac
        .network_actions
        .iter()
        .map(|a| (a.id.as_str(), a.activation_cost.expect("every action is priced")))
        .collect();
    assert_eq!(priced.len(), 3, "the fixture declares three actions");
    let cheapest = priced
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(id, _)| id.to_string())
        .expect("a cheapest");
    assert_eq!(cheapest, "closeBeFr4", "10 is the lowest of 100, 500 and 10");

    let mut options = SearchOptions::default();
    options.linear.objective_kind =
        ObjectiveKind::MinCost(Costly { options: CostlyOptions::default() });
    let mut solver = IpmSolver::new();
    let plan = run(&crac, &network, &resolution, &mut solver, &options);

    let used: Vec<&str> = plan
        .preventive
        .network_actions
        .iter()
        .map(|&i| crac.network_actions[i].id.as_str())
        .collect();
    assert_eq!(used, vec![cheapest.as_str()], "the cheapest action, and only it");

    // And the two figures the scenario states. `CostlyOptions::default()` is
    // the vendored configurations' own 1000.0 per unit of overload.
    let costly = Costly { options: CostlyOptions::default() };
    let untouched = evaluate(&crac, &network, &resolution);
    let states = crac.states();
    assert!(
        (costly.violation(&crac, &untouched, &states, gridoxide::rao::ObjectiveUnit::Megawatt)
            - 250_000.0)
            .abs()
            < 500.0,
        "the untouched network is 250 MW overloaded, which at 1000 a unit is 250000"
    );
    assert!(
        (costly.activation(&crac, &plan.preventive.network_actions, &[]) - 10.0).abs() < 1e-9,
        "and the answer costs exactly what closeBeFr4 costs"
    );
}
