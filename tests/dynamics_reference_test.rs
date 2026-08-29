//! Phase-5 gate G7: gridoxide against **Dynawo's own answer**.
//!
//! Every other gate in this module is self-supporting — a closed form, an
//! identity, a reduction, an oracle. This one is the first check against a
//! second implementation, and it is the only kind that can catch a formulation
//! that is internally consistent and physically wrong.
//!
//! The case is Kundur's Example 13.2 as Dynawo ships it: a sixth-order machine
//! with **no regulators**, an infinite bus, two parallel lines, a fixed-ratio
//! transformer, a bolted fault at the line junction and a line trip on
//! clearing. It was chosen because every element is something this library
//! models exactly — no tap changers, no limits, no saturation, nothing to
//! excuse before the comparison starts. See the `PROVENANCE.md` beside the
//! fixtures.
//!
//! Two things the reference file already settles before a single step is taken,
//! and they are checked first because a disagreement there would make the rest
//! meaningless:
//!
//! - `theta` at `t = 0` is where gridoxide's own initialization puts `δ₀`. That
//!   is the rotor-frame convention agreeing with a second implementation.
//! - `PmPu` exceeds the terminal power by exactly the stator copper loss, which
//!   is the statement that Dynawo's mechanical power is the **air-gap** power —
//!   the same choice `GenCls` and `GenRound` document.
//!
//! The comparison is between two approximations, not against truth: Dynawo
//! integrates with IDA at order 2 and a variable step, gridoxide with a fixed
//! trapezoidal step. Tolerances are therefore stated **per variable** and
//! justified, as `plans/RMS_PLAN.md` §7 requires, rather than set to whatever
//! makes the test pass.

use std::collections::HashMap;

use gridoxide::dynamics::dyd;
use num_complex::Complex;
use gridoxide::dynamics::json::{DynamicsData, DynamicsDocument, EventSpec, MachineSpec};
use gridoxide::dynamics::{run_dynamics, DynamicsOptions, DynamicsStatus, Trajectory};
use gridoxide::json::NetworkData;
use gridoxide::types::{Bus, BusType, Line};

const DIR: &str = "tests/data/dynamics/dynawo/kundur13";

/// The network's own base, in MVA. Dynawo's `SnRef`, and what every `…Pu` in
/// the reference file is on.
const S_BASE: f64 = 100.0;
const F_NOM: f64 = 50.0;

/// The infinite bus, the two lines and the transformer, exactly as
/// `KundurExample13.par` states them.
const V_INFINITE: f64 = 0.90081;
const X_LINE_1: f64 = 0.022522;
const X_LINE_2: f64 = 0.04189;
const X_TRANSFORMER: f64 = 0.00675;
/// The operating point the `.par` states, in Dynawo's receptor convention — so
/// the machine *injects* the negation.
const P0_PU: f64 = -19.98;

const FAULT_AT: f64 = 1.0;
const CLEAR_AT: f64 = 1.07;

/// Dynawo's reference curves, by column name.
struct Reference {
    time: Vec<f64>,
    columns: HashMap<String, Vec<f64>>,
}

impl Reference {
    fn read() -> Self {
        let text = std::fs::read_to_string(format!("{DIR}/reference_setpoint.csv"))
            .expect("the reference is committed beside the test");
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().expect("a header").split(';').collect();
        let mut columns: HashMap<String, Vec<f64>> =
            header.iter().map(|h| (h.trim().to_string(), Vec::new())).collect();
        let mut time = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(';').collect();
            for (k, name) in header.iter().enumerate() {
                if let Some(value) = fields.get(k).and_then(|f| f.trim().parse::<f64>().ok()) {
                    columns.get_mut(name.trim()).unwrap().push(value);
                }
            }
            time.push(fields[0].trim().parse().expect("a time"));
        }
        Self { time, columns }
    }

    fn column(&self, name: &str) -> &[f64] {
        self.columns.get(name).unwrap_or_else(|| panic!("no column `{name}`"))
    }
}

/// gridoxide's trajectory sampled at an arbitrary time.
///
/// Both records carry **two rows at each event instant**, one either side of
/// the jump, so a plain "first bracketing pair" search would interpolate across
/// a discontinuity. Taking the *last* interval whose start is at or before the
/// target picks the post-event branch, which is what Dynawo's own sampling does
/// and is the only reading that makes the two comparable at all.
fn sample(trajectory: &Trajectory, column: usize, t: f64) -> f64 {
    let time = &trajectory.time;
    let rows = &trajectory.rows;
    let mut lo = 0usize;
    for i in 0..time.len() - 1 {
        if time[i] <= t {
            lo = i;
        }
    }
    let (t0, t1) = (time[lo], time[lo + 1]);
    let (v0, v1) = (rows[lo][column], rows[lo + 1][column]);
    if t1 <= t0 {
        return v1;
    }
    v0 + (v1 - v0) * ((t - t0) / (t1 - t0))
}

fn bus(idx: usize, bus_type: BusType, vmag: f64, p: f64, q: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: vmag,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: q,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: Vec::new(),
    }
}

/// Builds the case: the machine's parameters come from Dynawo's own `.par`
/// through the reader, and the network from the same file's line and
/// transformer reactances.
fn case() -> DynamicsDocument {
    case_with(false)
}

/// The same case, with the rotor speed carried on the machine's speed-voltage
/// terms and its swing equation written in torque — Dynawo's own form.
fn case_with(speed_voltages: bool) -> DynamicsDocument {
    let dyd = dyd::read_dyd(format!("{DIR}/KundurExample13_SetPoint.dyd")).expect("parses");
    let par = dyd::read_par(format!("{DIR}/KundurExample13.par")).expect("parses");
    // The standalone case has no staticId, so the model's own id identifies it.
    let bus_of = HashMap::from([("SM".to_string(), 2usize)]);
    let (mut units, warnings) = dyd::to_units(&dyd, &par, &bus_of, S_BASE).expect("converts");

    assert_eq!(units.len(), 1, "the SetPoint case has exactly one machine");
    assert!(
        units[0].avr.is_none() && units[0].gov.is_none(),
        "`GeneratorSynchronousFourWindings` carries no regulators — that is why this case \
         was chosen"
    );
    assert!(warnings.is_empty(), "nothing should need excusing here: {warnings:?}");
    units[0].id = "SM".to_string();

    // The bus voltages the network solves to are checked against the file's own
    // stated operating point below, so the network data is under test too.
    let network = NetworkData {
        buses: vec![
            bus(0, BusType::Slack, V_INFINITE, 0.0, 0.0),
            bus(1, BusType::PQ, 1.0, 0.0, 0.0),
            bus(2, BusType::PV, 1.0, -P0_PU, 0.0),
        ],
        lines: vec![
            Line { from: 0, to: 1, r: 0.0, x: X_LINE_1, b_shunt: 0.0, g_shunt: 0.0 },
            Line { from: 0, to: 1, r: 0.0, x: X_LINE_2, b_shunt: 0.0, g_shunt: 0.0 },
            // A fixed-ratio transformer at ratio 1 with no resistance is a
            // series reactance and nothing more, so it needs no separate type.
            Line { from: 1, to: 2, r: 0.0, x: X_TRANSFORMER, b_shunt: 0.0, g_shunt: 0.0 },
        ],
    };

    DynamicsDocument {
        network,
        dynamics: DynamicsData {
            s_base: S_BASE,
            f_nom: F_NOM,
            speed_voltages,
            units,
            loads: Vec::new(),
            fixed_buses: vec![0],
            events: vec![
                EventSpec::BusFault { t: FAULT_AT, bus: 1, y: None },
                EventSpec::ClearFault { t: CLEAR_AT, bus: 1 },
                // Line 2 is branch index 1, in the order the network lists them.
                EventSpec::BranchTrip { t: CLEAR_AT, branch: 1 },
            ],
            relays: Vec::new(),
        },
    }
}

/// The machine's parameters, as read from Dynawo's file, are the ones Kundur's
/// Example 13.2 states.
#[test]
fn the_machine_is_the_one_the_book_describes() {
    let document = case();
    match document.dynamics.units[0].machine {
        MachineSpec::GenRound(p) => {
            assert_eq!(p.h, 3.5);
            assert_eq!(p.d, 0.0);
            assert_eq!(p.ra, 0.003);
            assert_eq!(p.xd, 1.81);
            assert_eq!(p.xq, 1.76);
            assert_eq!(p.xdp, 0.30);
            assert_eq!(p.xqp, 0.65);
            assert_eq!(p.xdpp, 0.23);
            assert_eq!(p.xqpp, 0.25);
            assert_eq!(p.xl, 0.15);
            assert_eq!(p.mbase, 2220.0);
        }
        ref other => panic!("expected a subtransient machine, got {other:?}"),
    }
}

/// The operating point gridoxide's power flow finds is the one Dynawo's file
/// states, and the initialization Dynawo reports.
///
/// This runs before any time stepping, and it is the load-bearing check. `δ₀`
/// is where the whole rotor-frame convention shows up: two implementations
/// deriving the same rotor angle from the same terminal condition, by
/// independent routes, is much stronger evidence than either agreeing with
/// itself.
#[test]
fn the_operating_point_and_the_initial_rotor_angle_agree() {
    let reference = Reference::read();
    let (system, _) = case().build().expect("builds");

    // The network: the power flow must reproduce the terminal angle the file
    // states, which confirms the line and transformer reactances.
    let v_terminal = system.voltages()[2];
    assert!(
        (v_terminal.norm() - 1.0).abs() < 1e-9,
        "the machine terminal should sit at 1.0 pu, found {}",
        v_terminal.norm()
    );
    assert!(
        (v_terminal.arg() - 0.49445).abs() < 1e-4,
        "the terminal angle should be the file's UPhase0 = 0.49445, found {}",
        v_terminal.arg()
    );

    // The initialization: the rotor angle, against Dynawo's own.
    let delta_0 = system.state()[0];
    let dynawo_theta_0 = reference.column("SM_generator_theta")[0];
    assert!(
        (delta_0 - dynawo_theta_0).abs() < 1e-4,
        "δ₀ = {delta_0:.6} against Dynawo's theta₀ = {dynawo_theta_0:.6}"
    );

    // And the equilibrium invariant still holds on a case built from someone
    // else's data, which is not something the earlier gates could show.
    assert!(system.max_derivative() < 1e-10, "drift {:e}", system.max_derivative());
    assert!(system.network_residual_norm() < 1e-10);
}

/// G7. The trajectory, against Dynawo's — and an honest account of where the
/// two part company.
///
/// **Through the fault the two agree to 3e-4 rad**, which is the strongest
/// statement in this whole module: two independent implementations, starting
/// from the same operating point, integrating the same sixth-order machine
/// through a bolted short, staying together to four decimal places.
///
/// After clearing they separate, steadily. The separation was chased down
/// rather than absorbed into a tolerance, and what it is *not* is now known:
///
/// - not step size — gridoxide's answer is converged to `1e-4` rad between
///   `h = 4 ms` and `h = 0.0625 ms`;
/// - not the nominal frequency — at 60 Hz the machine loses synchronism
///   outright, so 50 Hz is right;
/// - not the tripped branch — tripping the other line, or neither, gives a
///   completely different trajectory;
/// - not the sign of the `q`-axis damper coupling that
///   `plans/RMS_PLAN.md` §13 left open — flipping it moves the answer by less
///   than `1e-3` rad, which also means **this case does not settle that
///   question**;
/// - not the operating point, the parameters, or the air-gap power
///   convention, all of which agree exactly.
///
/// What it **is** — settled by reading Dynawo's own Modelica
/// (`Electrical/Machines/OmegaRef/BaseClasses/BaseGeneratorSynchronous.mo`):
///
/// ```text
/// udPu = (Ra + RTfo)·idPu − omegaPu·lambdaqPu       ← ω kept in the stator
/// uqPu = (Ra + RTfo)·iqPu + omegaPu·lambdadPu
/// 2·H·der(omegaPu) = cmPu·… − cePu − DPu·(ω − ωref) ← swing in TORQUE
/// cePu = lambdaqPu·idPu − lambdadPu·iqPu
/// PePu = cePu·omegaPu                               ← power = torque × ω
/// ```
///
/// gridoxide makes the classical RMS approximation in both places: `ω ≈ 1` in
/// the stator equations, and the swing equation written in **power** rather
/// than torque. Dynawo makes neither. The two forms differ by exactly a factor
/// of `ω`, so they agree perfectly at synchronous speed and part company in
/// proportion to the speed deviation — which is precisely what is observed:
/// the discrepancy is nil at `t = 0`, `0.04%` early in the fault where
/// `ω − 1 = 0.0015`, and `0.6%` after clearing where `ω − 1 = 0.009`.
///
/// **Neither form is wrong.** Neglecting the speed-voltage terms is the
/// standard RMS assumption — Kundur §13.3 states it explicitly — and keeping
/// them is what Sauer & Pai do. It is a documented modelling choice, and now a
/// known one rather than an unexplained residual. The next test pins its size.
///
#[test]
fn the_trajectory_matches_dynawo_through_the_fault() {
    let reference = Reference::read();
    let (mut system, events) = case().build().expect("builds");

    let report = run_dynamics(
        &mut system,
        &DynamicsOptions { end_time: 5.0, step: 0.00025, events, ..Default::default() },
    );
    assert_eq!(report.status, DynamicsStatus::Completed);
    assert_eq!(report.events_applied, 3);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);

    let worst = |ours: &str, theirs: &str, until: f64| {
        let column = report.trajectory.column(ours).unwrap();
        let expected = reference.column(theirs);
        let mut worst = 0.0f64;
        for (k, &t) in reference.time.iter().enumerate() {
            if t > until {
                break;
            }
            // The microsecond-spaced samples Dynawo emits while restarting its
            // own integrator after a discontinuity say nothing about either
            // method's answer.
            if (t - FAULT_AT).abs() < 2e-3 || (t - CLEAR_AT).abs() < 2e-3 {
                continue;
            }
            worst = worst.max((sample(&report.trajectory, column, t) - expected[k]).abs());
        }
        worst
    };

    // Through the fault, and for the first 40 ms after clearing.
    let delta_fault = worst("SM.delta", "SM_generator_theta", CLEAR_AT + 0.04);
    let omega_fault = worst("SM.omega", "SM_generator_omegaPu", CLEAR_AT + 0.04);
    assert!(delta_fault < 1e-3, "rotor angle through the fault: {delta_fault:e} rad");
    assert!(omega_fault < 1e-4, "rotor speed through the fault: {omega_fault:e} pu");

    // The terminal voltage is split at the clearing instant, because that is
    // where it tells two different stories. *During* the fault it tracks
    // Dynawo's closely. The moment the post-fault network is in place it steps
    // to 0.57% low — with the rotor angle and the fluxes still agreeing to
    // 3e-4 — which is the same offset the next test pins, showing up in the
    // algebraic solution before any of it has had time to integrate.
    let voltage_during = worst("bus2.vmag", "SM_generator_UPu", CLEAR_AT);
    let voltage_after = worst("bus2.vmag", "SM_generator_UPu", CLEAR_AT + 0.04);
    assert!(voltage_during < 2e-3, "terminal voltage during the fault: {voltage_during:e} pu");
    assert!(
        voltage_after < 8e-3,
        "terminal voltage just after clearing: {voltage_after:e} pu"
    );
    assert!(
        voltage_after > 3.0 * voltage_during,
        "the voltage error should step up at clearing, not grow smoothly: \
         {voltage_during:e} during, {voltage_after:e} after"
    );

    // Over the first swing the separation grows but stays small.
    let delta_swing = worst("SM.delta", "SM_generator_theta", 1.5);
    assert!(
        delta_swing < 0.03,
        "rotor angle over the first swing: {delta_swing:e} rad"
    );

    println!(
        "through-fault worst — delta {delta_fault:.3e} rad, omega {omega_fault:.3e} pu, \
         |V| {voltage_during:.3e} pu during / {voltage_after:.3e} after clearing; \
         first swing delta {delta_swing:.3e} rad"
    );
}

/// The size of the `ω ≈ 1` approximation, pinned.
///
/// Measured where the two rotor angles still agree closely, so that what is
/// compared is the **power-angle relation** and not the accumulated angle
/// difference. Terminal power is computed from the trajectory's own bus
/// voltages across the lossless transformer branch, which is exact.
///
/// The offset is about `−0.6%` at a speed deviation of `0.9%`, which is the
/// order the factor of `ω` predicts. Pinning it means that if gridoxide ever
/// adopts the full form — keeping the speed voltages and writing the swing
/// equation in torque — this number goes to nearly zero and says so, rather
/// than the change passing unnoticed inside a loose bound.
#[test]
fn the_speed_voltage_approximation_costs_six_tenths_of_a_percent() {
    let reference = Reference::read();
    let expected_delta = reference.column("SM_generator_theta");
    let expected_power = reference.column("SM_generator_PGenPu");
    let (mut system, events) = case().build().unwrap();
    let report = run_dynamics(
        &mut system,
        &DynamicsOptions { end_time: 2.0, step: 0.00025, events, ..Default::default() },
    );
    assert_eq!(report.status, DynamicsStatus::Completed);

    let (dm, da) = (
        report.trajectory.column("SM.delta").unwrap(),
        report.trajectory.column("bus1.vang").unwrap(),
    );
    let (m1, m2, a2) = (
        report.trajectory.column("bus1.vmag").unwrap(),
        report.trajectory.column("bus2.vmag").unwrap(),
        report.trajectory.column("bus2.vang").unwrap(),
    );

    let mut offsets = Vec::new();
    for (k, &t) in reference.time.iter().enumerate() {
        if !(CLEAR_AT + 0.01..CLEAR_AT + 0.06).contains(&t) {
            continue;
        }
        // Only where the states still agree, so the comparison is of the
        // relation and not of the divergence.
        let delta = sample(&report.trajectory, dm, t);
        if (delta - expected_delta[k]).abs() > 1e-3 {
            continue;
        }
        let (v1, ang1) = (sample(&report.trajectory, m1, t), sample(&report.trajectory, da, t));
        let (v2, ang2) = (sample(&report.trajectory, m2, t), sample(&report.trajectory, a2, t));
        let ours = v1 * v2 * (ang2 - ang1).sin() / X_TRANSFORMER;
        offsets.push((ours - expected_power[k]) / expected_power[k]);
    }

    assert!(offsets.len() >= 5, "not enough matched samples: {}", offsets.len());
    let mean = offsets.iter().sum::<f64>() / offsets.len() as f64;
    println!("post-fault power offset at matched angle: {:.4}%", mean * 100.0);
    assert!(
        (-0.008..-0.004).contains(&mean),
        "the ω ≈ 1 approximation is expected to cost about −0.6% here; it cost \
         {:.4}%. If it moved, something changed the power-angle relation — which \
         is the whole point of pinning it.",
        mean * 100.0
    );
}

/// The machine loses synchronism if the fault is held long enough, and stays in
/// step if it is not — and the boundary is where Dynawo's own case sits.
///
/// The published case clears at 70 ms and survives. That is a statement about
/// the *case*, and reproducing it is a coarser but entirely independent check
/// on the same physics as the trajectory comparison: it depends only on which
/// side of the critical clearing time 70 ms falls, so it survives any
/// disagreement small enough not to move that boundary.
#[test]
fn the_published_clearing_time_is_survivable_and_a_longer_one_is_not() {
    let stays_in_step = |clear_at: f64| {
        let mut document = case();
        document.dynamics.events = vec![
            EventSpec::BusFault { t: FAULT_AT, bus: 1, y: None },
            EventSpec::ClearFault { t: clear_at, bus: 1 },
            EventSpec::BranchTrip { t: clear_at, branch: 1 },
        ];
        let (mut system, events) = document.build().unwrap();
        let report = run_dynamics(
            &mut system,
            &DynamicsOptions { end_time: 5.0, step: 0.001, events, ..Default::default() },
        );
        assert_eq!(report.status, DynamicsStatus::Completed);
        let delta = report.trajectory.series("SM.delta").unwrap();
        delta.iter().fold(0.0f64, |m, d| m.max(*d)) < std::f64::consts::TAU
    };

    assert!(stays_in_step(CLEAR_AT), "the published 70 ms clearing must be survivable");
    assert!(!stays_in_step(1.0 + 0.5), "half a second of fault must not be");
}


/// **Adopting Dynawo's own form drives the 0.6% offset to nothing.**
///
/// This is what closes the loop on the phase-5 finding. Measuring a
/// discrepancy and attributing it by reading the reference's source is one
/// thing; *acting* on the attribution and watching the number collapse is the
/// proof. If the `ω ≈ 1` approximation were not the cause, switching it off
/// would move the offset somewhere arbitrary rather than to zero.
///
/// The residual after the switch is what remains genuinely unexplained, and it
/// is bounded here so that it stays visible.
#[test]
fn the_full_form_removes_the_offset_against_dynawo() {
    let offset = |speed_voltages: bool| {
        let reference = Reference::read();
        let expected_delta = reference.column("SM_generator_theta");
        let expected_power = reference.column("SM_generator_PGenPu");
        let (mut system, events) = case_with(speed_voltages).build().unwrap();
        let report = run_dynamics(
            &mut system,
            &DynamicsOptions { end_time: 2.0, step: 0.00025, events, ..Default::default() },
        );
        assert_eq!(report.status, DynamicsStatus::Completed);

        let (dm, da) = (
            report.trajectory.column("SM.delta").unwrap(),
            report.trajectory.column("bus1.vang").unwrap(),
        );
        let (m1, m2, a2) = (
            report.trajectory.column("bus1.vmag").unwrap(),
            report.trajectory.column("bus2.vmag").unwrap(),
            report.trajectory.column("bus2.vang").unwrap(),
        );
        let mut offsets = Vec::new();
        for (k, &t) in reference.time.iter().enumerate() {
            if !(CLEAR_AT + 0.01..CLEAR_AT + 0.06).contains(&t) {
                continue;
            }
            let delta = sample(&report.trajectory, dm, t);
            if (delta - expected_delta[k]).abs() > 1e-3 {
                continue;
            }
            let (v1, ang1) =
                (sample(&report.trajectory, m1, t), sample(&report.trajectory, da, t));
            let (v2, ang2) =
                (sample(&report.trajectory, m2, t), sample(&report.trajectory, a2, t));
            let ours = v1 * v2 * (ang2 - ang1).sin() / X_TRANSFORMER;
            offsets.push((ours - expected_power[k]) / expected_power[k]);
        }
        assert!(offsets.len() >= 5, "not enough matched samples: {}", offsets.len());
        offsets.iter().sum::<f64>() / offsets.len() as f64
    };

    let approximate = offset(false);
    let full = offset(true);
    println!(
        "post-fault power offset: {:.4}% with the approximation, {:.4}% without",
        approximate * 100.0,
        full * 100.0
    );

    assert!(
        full.abs() < 0.1 * approximate.abs(),
        "adopting Dynawo's form should remove most of the offset: {:.4}% became {:.4}%",
        approximate * 100.0,
        full * 100.0
    );
    assert!(
        full.abs() < 1e-3,
        "what remains after the switch should be under 0.1%; it was {:.4}%",
        full * 100.0
    );
}

/// And the trajectory agrees far longer, which is the thing a user would
/// actually notice.
#[test]
fn the_full_form_tracks_dynawo_through_the_whole_first_swing() {
    let reference = Reference::read();
    let worst_over = |speed_voltages: bool, until: f64| {
        let (mut system, events) = case_with(speed_voltages).build().unwrap();
        let report = run_dynamics(
            &mut system,
            &DynamicsOptions { end_time: 5.0, step: 0.00025, events, ..Default::default() },
        );
        assert_eq!(report.status, DynamicsStatus::Completed);
        let column = report.trajectory.column("SM.delta").unwrap();
        let expected = reference.column("SM_generator_theta");
        let mut worst = 0.0f64;
        for (k, &t) in reference.time.iter().enumerate() {
            if t > until {
                break;
            }
            if (t - FAULT_AT).abs() < 2e-3 || (t - CLEAR_AT).abs() < 2e-3 {
                continue;
            }
            worst = worst.max((sample(&report.trajectory, column, t) - expected[k]).abs());
        }
        worst
    };

    let (approx_swing, full_swing) = (worst_over(false, 1.5), worst_over(true, 1.5));
    let (approx_run, full_run) = (worst_over(false, 5.0), worst_over(true, 5.0));
    println!(
        "rotor angle vs Dynawo — first swing: {approx_swing:.3e} → {full_swing:.3e} rad; \
         whole 5 s: {approx_run:.3e} → {full_run:.3e} rad"
    );

    assert!(
        full_swing < 0.2 * approx_swing,
        "over the first swing the full form should agree far better: \
         {approx_swing:e} against {full_swing:e}"
    );
    assert!(
        full_run < approx_run,
        "and it should not be worse over the whole run: {approx_run:e} against {full_run:e}"
    );
}

/// The **field-voltage per-unit base** agrees with Dynawo's, exactly.
///
/// This matters beyond a curiosity: a regulator's ceiling is stated in that
/// base, so carrying `voltageRegulator_EfdMaxPu` from a Dynawo file into
/// gridoxide's exciter is only sound if the two mean the same thing by "per
/// unit field voltage". They do — the initialization derives 2.420747 and the
/// reference file's `efdPu` at `t = 0` is 2.420747 — and the `.dyd` reader
/// carries the limits on the strength of it.
///
/// It is also a third independent check on the machine model, alongside the
/// rotor angle and the air-gap power.
#[test]
fn the_field_voltage_base_agrees_with_dynawos() {
    use gridoxide::dynamics::models::machine::{GenRound, Machine};

    let params = match case().dynamics.units[0].machine {
        MachineSpec::GenRound(p) => p,
        ref other => panic!("expected a subtransient machine, got {other:?}"),
    };
    let mut machine = GenRound::new(params, S_BASE, F_NOM).unwrap();
    let init = machine
        .initialize(Complex::from_polar(1.0, 0.49445), Complex::new(-P0_PU, 9.68))
        .expect("initializes");

    let expected = Reference::read().column("SM_generator_efdPu")[0];
    assert!(
        (init.e_fd - expected).abs() < 1e-5,
        "field voltage {:.6} against Dynawo's {expected:.6}",
        init.e_fd
    );
}
