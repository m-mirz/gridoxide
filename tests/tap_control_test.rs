//! Transformer tap control: the on-load tap changer as a control rather than a
//! constant.
//!
//! Three kinds of check, in increasing order of how much they depend on
//! anything outside this repository:
//!
//! 1. **Analytic.** A two-bus network whose right position is computable by
//!    hand from the step table. The loop must find it from every start.
//! 2. **Against an exhaustive sweep.** Solve at every position in range,
//!    record the controlled quantity, and check the loop's answer against what
//!    the sweep says. Depends on no reference tool, and it is the check that
//!    catches a sign error in `∂V/∂ρ` — which is otherwise entirely plausible
//!    in the output, since the voltage still moves.
//! 3. **Against the conformity fixtures' own data.** See
//!    `svedalas_published_solution_violates_its_own_deadbands` for why this is
//!    weaker evidence than `plans/TAP_CONTROL_PLAN.md` §8.3 assumed.

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_to_network, load_profiles, CgmesNetwork};
use gridoxide::outerloop::{ControllerOutcome, RegulationMode, TapRegulation};
use gridoxide::solver::{PowerFlowOptions, SolveStatus};
use gridoxide::types::{Bus, BusType, Line, TapChanger, Transformer};
use gridoxide::{run_power_flow, PowerFlowReport, TapData};

use num_complex::Complex;

// ---------------------------------------------------------------------------
// 1. Analytic
// ---------------------------------------------------------------------------

/// Slack — transformer — load. The transformer's ratio changer has eleven
/// positions from 0.90 to 1.10 in steps of 0.02, and the load bus's voltage is
/// monotone in the ratio, so the right position is a lookup.
fn two_bus(position: i32) -> (Vec<Bus>, Vec<Line>, Vec<Transformer>, Vec<Option<TapChanger>>) {
    let buses = vec![
        Bus {
            idx: 0, bus_type: BusType::Slack, voltage_mag: 1.0, voltage_ang: 0.0,
            p_spec: 0.0, q_spec: 0.0, q_min: f64::NEG_INFINITY, q_max: f64::INFINITY,
            u_rated: 1.0, zip_terms: Vec::new(),
        },
        Bus {
            idx: 1, bus_type: BusType::PQ, voltage_mag: 1.0, voltage_ang: 0.0,
            p_spec: -0.8, q_spec: -0.3, q_min: f64::NEG_INFINITY, q_max: f64::INFINITY,
            u_rated: 1.0, zip_terms: Vec::new(),
        },
    ];
    let steps: Vec<Complex<f64>> =
        (0..11).map(|i| Complex::new(0.90 + 0.02 * i as f64, 0.0)).collect();
    let changer = TapChanger { low: 0, position, neutral: 5, steps, series: None };
    let mut transformer = Transformer {
        from: 0,
        to: 1,
        from_status: 1,
        to_status: 1,
        y_series: Complex::new(1.0, 0.0) / Complex::new(0.01, 0.10),
        y_shunt: Complex::new(0.0, 0.0),
        tap: Complex::new(1.0, 0.0),
    };
    transformer.tap = changer.at(position).unwrap();
    (buses, Vec::new(), vec![transformer], vec![Some(changer)])
}

fn voltage_control(target: f64, deadband: f64) -> Vec<TapRegulation> {
    vec![TapRegulation {
        transformer: 0,
        controlled_bus: 1,
        mode: RegulationMode::Voltage,
        target,
        deadband,
        enabled: true,
        id: "analytic".into(),
    }]
}

fn solve(
    buses: Vec<Bus>,
    lines: &[Line],
    transformers: &[Transformer],
    changers: &[Option<TapChanger>],
    regulation: &[TapRegulation],
    control_taps: bool,
) -> PowerFlowReport {
    run_power_flow(
        buses,
        lines,
        transformers,
        &[],
        TapData { changers, regulation },
        PowerFlowOptions {
            control_taps,
            max_outer_iter: 60,
            tol: 1e-10,
            max_iter: 40,
            ..Default::default()
        },
    )
}

/// Solve at every position and report the controlled bus's voltage — the
/// ground truth the loop is checked against, sharing no code with it.
fn sweep(
    buses: &[Bus],
    lines: &[Line],
    transformers: &[Transformer],
    changers: &[Option<TapChanger>],
    which: usize,
    controlled_bus: usize,
) -> Vec<(i32, f64)> {
    let changer = changers[which].as_ref().expect("a changer to sweep");
    let mut out = Vec::new();
    for p in changer.low..=changer.high() {
        let mut t = transformers.to_vec();
        let mut c = changer.clone();
        assert!(c.set_position(&mut t[which], p));
        let report = solve(buses.to_vec(), lines, &t, &[], &[], false);
        if report.stats.status == SolveStatus::Converged {
            out.push((p, report.buses[controlled_bus].voltage_mag));
        }
    }
    out
}

/// From every starting position, the loop reaches the position an exhaustive
/// sweep names — and reaches the *same* one regardless of where it started,
/// which is what makes the answer a property of the network rather than of the
/// path taken to it.
#[test]
fn the_loop_finds_the_position_a_sweep_names() {
    let (buses, lines, transformers, changers) = two_bus(5);
    let table = sweep(&buses, &lines, &transformers, &changers, 0, 1);
    assert_eq!(table.len(), 11, "every position should solve");

    let target = 1.0;
    let best = table
        .iter()
        .min_by(|a, b| (a.1 - target).abs().total_cmp(&(b.1 - target).abs()))
        .expect("a best position")
        .0;

    for start in 0..=10 {
        let (buses, lines, transformers, changers) = two_bus(start);
        // A deadband narrower than one step forces the loop to the argmin
        // rather than letting it stop at the first acceptable position.
        let report = solve(buses, &lines, &transformers, &changers, &voltage_control(target, 1e-6), true);
        let outer = report.outer.as_ref().expect("loops ran");
        let landed = outer.changers[0].as_ref().unwrap().position;
        assert_eq!(
            landed, best,
            "started at {start}, landed at {landed}, sweep says {best}; table {table:?}"
        );
    }
}

/// The deadband is honoured rather than approximated away: a controller whose
/// bus is already inside it does not move, however far the target is from the
/// exact voltage.
#[test]
fn a_bus_inside_its_deadband_does_not_move() {
    let (buses, lines, transformers, changers) = two_bus(5);
    let table = sweep(&buses, &lines, &transformers, &changers, 0, 1);
    let at_five = table.iter().find(|(p, _)| *p == 5).unwrap().1;

    // A deadband wide enough to contain where it already is.
    let report = solve(
        buses,
        &lines,
        &transformers,
        &changers,
        &voltage_control(at_five + 0.01, 0.05),
        true,
    );
    let outer = report.outer.as_ref().expect("loops ran");
    assert_eq!(outer.changers[0].as_ref().unwrap().position, 5, "nothing to do");
    assert_eq!(outer.taps[0].outcome, ControllerOutcome::InDeadband);
    assert_eq!(outer.taps[0].steps_moved, 0);
}

/// A target no position can reach stops at the limit and says so, rather than
/// oscillating until the budget runs out or reporting success at the wrong
/// voltage.
///
/// Which limit is derived from the sweep rather than assumed. `Transformer::tap`
/// scales the *from* side, so on this network the highest ratio gives the
/// lowest controlled voltage — the opposite of the intuitive reading, and
/// exactly the kind of thing an assumed direction gets wrong.
#[test]
fn an_unreachable_target_stops_at_the_limit_and_reports_it() {
    let (buses, lines, transformers, changers) = two_bus(5);
    let table = sweep(&buses, &lines, &transformers, &changers, 0, 1);
    let highest = table.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;

    let report = solve(buses, &lines, &transformers, &changers, &voltage_control(2.0, 1e-6), true);
    let outer = report.outer.as_ref().expect("loops ran");
    let changer = outer.changers[0].as_ref().unwrap();
    assert_eq!(changer.position, highest, "it should have gone as far as it can: {table:?}");
    assert!(
        changer.position == changer.low || changer.position == changer.high(),
        "and that is one end of the range"
    );
    assert_eq!(outer.taps[0].outcome, ControllerOutcome::AtLimit);
    assert!(outer.report.as_ref().unwrap().converged, "stopping at a limit is a settled answer");
}

/// Tap control is opt-in. With `control_taps` off, the same network solves to
/// the same state it always did — bit for bit.
#[test]
fn control_taps_off_changes_nothing() {
    let (buses, lines, transformers, changers) = two_bus(3);
    let regulation = voltage_control(1.0, 1e-6);
    let plain = solve(buses.clone(), &lines, &transformers, &changers, &regulation, false);
    let bare = run_power_flow(
        buses,
        &lines,
        &transformers,
        &[],
        TapData::none(),
        PowerFlowOptions { tol: 1e-10, max_iter: 40, ..Default::default() },
    );
    assert!(plain.outer.is_none(), "no loop was configured, so none should be reported");
    for (a, b) in plain.buses.iter().zip(&bare.buses) {
        assert_eq!(a.voltage_mag.to_bits(), b.voltage_mag.to_bits());
        assert_eq!(a.voltage_ang.to_bits(), b.voltage_ang.to_bits());
    }
}

// ---------------------------------------------------------------------------
// 2 & 3. The conformity fixtures
// ---------------------------------------------------------------------------

fn fixture(dir: &str) -> Option<CgmesNetwork> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(dir);
    if !dir.exists() {
        return None;
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("configuration directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "xml"))
        .collect();
    paths.sort();
    let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    let ds = load_profiles(&refs).expect("failed to decode CGMES profiles");
    Some(cgmes_to_network(&ds, 100e6).expect("conversion failed"))
}

fn solve_fixture(net: &CgmesNetwork, control_taps: bool) -> PowerFlowReport {
    run_power_flow(
        net.buses.clone(),
        &net.lines,
        &net.transformers,
        &net.shunts,
        TapData { changers: &net.tap_changers, regulation: &net.regulation },
        PowerFlowOptions {
            control_taps,
            max_outer_iter: 60,
            tol: 1e-8,
            max_iter: 30,
            ..Default::default()
        },
    )
}

/// **Why the fixed-point gate `plans/TAP_CONTROL_PLAN.md` §8.3 proposed does
/// not apply to Svedala**, recorded as an assertion so it cannot quietly stop
/// being true.
///
/// The plan reasoned that since Svedala's SSH tap positions equal its SV ones,
/// a converged tap loop must leave all eleven where it found them. That
/// premise is wrong: the fixture's *own published solution* sits 2–5% away
/// from the targets its own SSH declares, against half-deadbands of 1–2%. So
/// the published state does not satisfy the controls the document records, and
/// a loop that left those taps alone would be the defective one.
///
/// This says nothing bad about the fixture — a conformity model exists to
/// exercise a profile's classes, not to be a converged operating point — but it
/// does mean Svedala cannot serve as evidence that the loop leaves correct taps
/// alone. `no_control_means_no_movement` carries that instead.
#[test]
fn svedalas_published_solution_violates_its_own_deadbands() {
    let Some(net) = fixture("Svedala/Svedala-Merged") else {
        eprintln!("skipping: Svedala fixture not checked out");
        return;
    };
    let base = solve_fixture(&net, false);
    assert_eq!(base.stats.status, SolveStatus::Converged);

    let mut outside = 0;
    for r in &net.regulation {
        let v = base.buses[r.controlled_bus].voltage_mag;
        if (v - r.target).abs() > r.deadband / 2.0 {
            outside += 1;
        }
    }
    assert_eq!(
        outside,
        net.regulation.len(),
        "every one of Svedala's controls should be violated at the published positions"
    );
}

/// Svedala's eleven voltage controllers all reach their deadbands, and the
/// solve still converges. This is the feature working at real scale: 2–5%
/// away before, inside a 1–2% half-deadband after.
#[test]
fn svedala_brings_every_controller_into_its_deadband() {
    let Some(net) = fixture("Svedala/Svedala-Merged") else {
        eprintln!("skipping: Svedala fixture not checked out");
        return;
    };
    let report = solve_fixture(&net, true);
    assert_eq!(report.stats.status, SolveStatus::Converged);
    let outer = report.outer.as_ref().expect("loops ran");
    assert!(outer.report.as_ref().unwrap().converged, "{:?}", outer.report);

    assert_eq!(outer.taps.len(), 11);
    for c in &outer.taps {
        assert_eq!(c.outcome, ControllerOutcome::InDeadband, "controller {} on {}", c.id, c.transformer);
    }
    // Checked against the *state*, not against the loop's own verdict.
    for r in &net.regulation {
        let v = report.buses[r.controlled_bus].voltage_mag;
        assert!(
            (v - r.target).abs() <= r.deadband / 2.0,
            "bus {} at {v} against target {} ± {}",
            r.controlled_bus,
            r.target,
            r.deadband / 2.0
        );
    }
    // Every position stayed in range, which `set_position` guarantees but is
    // worth asserting at the level a caller reads.
    for (i, c) in outer.changers.iter().enumerate() {
        let Some(c) = c else { continue };
        assert!(c.position >= c.low && c.position <= c.high(), "transformer {i}");
    }
}

/// The sweep check (§8.2) at real scale: at the positions the loop chose, each
/// controller is inside its deadband, and an exhaustive sweep of that
/// controller's own range — every other tap held where the loop left it —
/// confirms the position it picked is one that satisfies the control.
///
/// This is what catches a sign error in `∂V/∂ρ`: a loop that moved the tap the
/// wrong way would still change the voltage, still terminate, and still look
/// plausible — but it would land on a position the sweep says is worse.
#[test]
fn every_chosen_position_is_confirmed_by_a_sweep() {
    let Some(net) = fixture("Svedala/Svedala-Merged") else {
        eprintln!("skipping: Svedala fixture not checked out");
        return;
    };
    let report = solve_fixture(&net, true);
    let outer = report.outer.as_ref().expect("loops ran");
    let settled = &outer.transformers;
    let changers = &outer.changers;

    for r in &net.regulation {
        let chosen = changers[r.transformer].as_ref().expect("a controlled transformer has a table");
        let mut acceptable = Vec::new();
        for p in chosen.low..=chosen.high() {
            let mut t = settled.clone();
            let mut c = chosen.clone();
            assert!(c.set_position(&mut t[r.transformer], p));
            let probe = run_power_flow(
                net.buses.clone(),
                &net.lines,
                &t,
                &net.shunts,
                TapData::none(),
                PowerFlowOptions { tol: 1e-8, max_iter: 30, ..Default::default() },
            );
            if probe.stats.status != SolveStatus::Converged {
                continue;
            }
            let v = probe.buses[r.controlled_bus].voltage_mag;
            if (v - r.target).abs() <= r.deadband / 2.0 {
                acceptable.push(p);
            }
        }
        assert!(
            acceptable.contains(&chosen.position),
            "transformer {} landed on {}, but a sweep says only {acceptable:?} satisfy the control",
            r.transformer,
            chosen.position
        );
    }
}

/// A tap changer with no enabled control must never move, whatever the
/// voltages do. This is the fixed-point evidence Svedala cannot give.
///
/// Two halves, because they fail differently. A fixture with no live control at
/// all configures no tap loop — SmallGrid has ten tables and no controls,
/// PowerFlow has one table whose control is switched off — so the check there
/// is that no loop is even built. A control that exists but is disabled has to
/// be checked against a loop that *is* running, which the analytic case does.
#[test]
fn no_control_means_no_movement() {
    for name in ["SmallGrid/SmallGrid-Merged", "PowerFlow/PowerFlow", "MiniGrid/MiniGrid-Merged"] {
        let Some(net) = fixture(name) else { continue };
        assert!(net.regulation.is_empty(), "{name}: fixture assumption");
        let report = solve_fixture(&net, true);
        assert!(
            report.outer.is_none(),
            "{name}: with nothing to control, no outer loop should be configured at all"
        );
    }

    // A control that is present and switched off, against a loop that ran.
    let (buses, lines, transformers, changers) = two_bus(3);
    let mut regulation = voltage_control(1.0, 1e-6);
    regulation[0].enabled = false;
    let report = solve(buses, &lines, &transformers, &changers, &regulation, true);
    let outer = report.outer.as_ref().expect("control_taps was on with regulation present");
    assert!(outer.taps.is_empty(), "a disabled control must produce no controller");
    assert_eq!(
        outer.changers[0].as_ref().unwrap().position,
        3,
        "a disabled control must not move its tap"
    );
}

/// The phase shifters, and the two honest endings a discrete control has.
///
/// Type 1 starts inside its deadband and must not move — the fixed-point case
/// for the active-power loop. Types 2 and 3 ask for zero flow within 5e-4 pu on
/// a shifter whose closest position leaves 0.11 pu, so **no position satisfies
/// them**: the right answer is to move to the best one and report
/// `Closest`, not to stop somewhere and claim success. An earlier draft
/// relabelled it `InDeadband`, which reported a flow 200 times the tolerance as
/// converged.
#[test]
fn the_pst_fixtures_reach_the_best_position_their_table_offers() {
    for (name, expect_move, expected) in [
        ("PST/PST_PhaseTapChangerLinear_Type1", false, ControllerOutcome::InDeadband),
        ("PST/PST_PhaseTapChangerLinear_Type2", true, ControllerOutcome::Closest),
        ("PST/PST_PhaseTapChangerTable_Type3", true, ControllerOutcome::Closest),
    ] {
        let Some(net) = fixture(name) else { continue };
        assert_eq!(net.regulation.len(), 1, "{name}");
        let before = net.tap_changers[0].as_ref().unwrap().position;

        let report = solve_fixture(&net, true);
        assert_eq!(report.stats.status, SolveStatus::Converged, "{name}");
        let outer = report.outer.as_ref().expect("loops ran");
        let after = outer.changers[0].as_ref().unwrap().position;
        assert_eq!(outer.taps.len(), 1, "{name}");
        assert_eq!(outer.taps[0].outcome, expected, "{name}");
        assert_eq!(after != before, expect_move, "{name}: {before} -> {after}");

        // Whatever it reported, the position it chose must be the one an
        // exhaustive sweep says is closest to the target — measured from the
        // returned state rather than taken from the loop's own verdict.
        let r = &net.regulation[0];
        let RegulationMode::ActivePower { branch, terminal } = r.mode else {
            panic!("{name}: expected an active-power control");
        };
        let flow_at = |p: i32| -> Option<f64> {
            let mut t = net.transformers.clone();
            let mut c = net.tap_changers[0].clone().unwrap();
            c.set_position(&mut t[0], p);
            let probe = run_power_flow(
                net.buses.clone(),
                &net.lines,
                &t,
                &net.shunts,
                TapData::none(),
                PowerFlowOptions { tol: 1e-10, max_iter: 40, ..Default::default() },
            );
            (probe.stats.status == SolveStatus::Converged).then(|| {
                let params = gridoxide::branch_flow::branch_params(&net.lines, &t);
                let v = gridoxide::branch_flow::bus_voltages(&probe.buses);
                gridoxide::branch_flow::terminal_flow(&params[branch], terminal, &v).0
            })
        };
        let changer = net.tap_changers[0].as_ref().unwrap();
        let best = (changer.low..=changer.high())
            .filter_map(|p| flow_at(p).map(|f| (p, (f - r.target).abs())))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("some position solves")
            .0;
        assert_eq!(after, best, "{name}: landed on {after}, sweep says {best}");
    }
}
