//! Phase-3 gates for `src/dynamics/models/`: the fourth-order machine, the
//! exciter, the governor, the stabilizer, and the composite that joins them.
//!
//! Three kinds of check, in decreasing order of how much they would catch:
//!
//! - **The oracle (G4).** Every model's analytic Jacobian against a central
//!   difference, over every combination of controls. The composite's chain
//!   rule is what is really under test — a machine alone exercises none of it.
//! - **Equilibrium invariance (G1).** Every combination, undisturbed, must be
//!   flat to machine precision. This is what catches an initialization that
//!   latches the wrong reference, which is the single most likely mistake in
//!   this whole module.
//! - **The physics.** The dq round trip reproduces the terminal current
//!   exactly; an exciter returns the voltage; two governors share a
//!   disturbance in inverse proportion to their droops.

use num_complex::Complex;

use gridoxide::dynamics::models::avr::{Sexs, SexsParams};
use gridoxide::dynamics::models::InitError;
use gridoxide::dynamics::models::gov::{Tgov1, Tgov1Params};
use gridoxide::dynamics::models::machine::{
    GenRound, GenRoundParams, GenSalient, GenSalientParams, GenTransient, GenTransientParams,
    Machine,
};
use gridoxide::dynamics::models::load::ZipLoad;
use gridoxide::dynamics::models::pss::{Stab1, Stab1Params};
use gridoxide::dynamics::models::{
    finite_difference, Control, DynamicModel, GeneratingUnit, Limits, ModelJacobian,
};
use gridoxide::dynamics::{
    build, run_dynamics, DeviceSpec, DynamicSystem, DynamicsOptions, DynamicsStatus, Event,
    EventKind, SystemSpec,
};
use gridoxide::network::{build_ybus, power_injections};
use gridoxide::types::{Bus, BusType, Line};

const S_BASE: f64 = 100.0;
const F_NOM: f64 = 50.0;

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

fn machine_params(d: f64) -> GenTransientParams {
    GenTransientParams {
        h: 5.0,
        d,
        ra: 0.003,
        xd: 1.8,
        xq: 1.7,
        xdp: 0.30,
        xqp: 0.55,
        td0p: 8.0,
        tq0p: 0.4,
        mbase: S_BASE,
    }
}

fn new_machine(d: f64) -> GenTransient {
    GenTransient::new(machine_params(d), S_BASE, F_NOM).expect("valid machine parameters")
}

fn new_avr() -> Sexs {
    Sexs::new(SexsParams { k: 200.0, ta: 0.1, tb: 1.0, te: 0.05, limits: Limits::NONE }).unwrap()
}

fn new_gov(r: f64) -> Tgov1 {
    Tgov1::new(Tgov1Params { r, t1: 0.5, t2: 1.0, t3: 5.0, dt: 0.0, limits: Limits::NONE }).unwrap()
}

fn new_pss() -> Stab1 {
    Stab1::new(Stab1Params { k: 5.0, tw: 10.0, t1: 0.15, t2: 0.03, t3: 0.15, t4: 0.03, limits: Limits::NONE }).unwrap()
}

/// Which controls a unit carries, so the gates can sweep every combination.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fit {
    avr: bool,
    gov: bool,
    pss: bool,
}

impl Fit {
    fn label(self) -> String {
        let mut parts = vec!["machine"];
        if self.avr {
            parts.push("avr");
        }
        if self.gov {
            parts.push("gov");
        }
        if self.pss {
            parts.push("pss");
        }
        parts.join("+")
    }
}

/// Every combination. A stabilizer with no exciter to feed is included
/// deliberately: it is legal, it does nothing, and it must still initialize to
/// an equilibrium and differentiate correctly.
const FITS: [Fit; 8] = [
    Fit { avr: false, gov: false, pss: false },
    Fit { avr: true, gov: false, pss: false },
    Fit { avr: false, gov: true, pss: false },
    Fit { avr: false, gov: false, pss: true },
    Fit { avr: true, gov: true, pss: false },
    Fit { avr: true, gov: false, pss: true },
    Fit { avr: false, gov: true, pss: true },
    Fit { avr: true, gov: true, pss: true },
];

fn unit(fit: Fit, damping: f64, droop: f64) -> GeneratingUnit {
    GeneratingUnit::new(
        Box::new(new_machine(damping)),
        fit.avr.then(|| Box::new(new_avr()) as Box<dyn Control>),
        fit.gov.then(|| Box::new(new_gov(droop)) as Box<dyn Control>),
        fit.pss.then(|| Box::new(new_pss()) as Box<dyn Control>),
    )
}

/// G4. The analytic Jacobian of every combination, against a central
/// difference, at points well away from the equilibrium.
#[test]
fn every_combination_matches_the_finite_difference_oracle() {
    for fit in FITS {
        let mut model = unit(fit, 2.0, 0.05);
        model
            .initialize(Complex::from_polar(1.02, 0.15), Complex::new(0.8, 0.3))
            .unwrap_or_else(|e| panic!("{}: {e}", fit.label()));

        let n = model.n_states();
        // Probe states that are nowhere near the equilibrium, so no term is
        // accidentally zero — a Jacobian that is only right at the operating
        // point is right for nothing.
        let probes: [(Vec<f64>, Complex<f64>); 3] = [
            ((0..n).map(|k| 0.4 + 0.13 * k as f64).collect(), Complex::new(1.00, 0.05)),
            ((0..n).map(|k| -0.3 + 0.21 * k as f64).collect(), Complex::new(0.87, -0.24)),
            ((0..n).map(|k| 1.1 - 0.17 * k as f64).collect(), Complex::new(1.06, 0.31)),
        ];

        for (x, v) in probes {
            let mut analytic = ModelJacobian::zeros(n);
            analytic.clear();
            model.jacobian(&x, v, &mut analytic);
            let numeric = finite_difference(&model, &x, v, 1e-6);

            let compare = |name: &str, a: &[f64], b: &[f64]| {
                for (k, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                    let scale = av.abs().max(bv.abs()).max(1.0);
                    assert!(
                        (av - bv).abs() / scale < 1e-6,
                        "{} {name}[{k}]: analytic {av:e} vs numeric {bv:e} (x = {x:?}, v = {v})",
                        fit.label()
                    );
                }
            };
            compare("dfdx", &analytic.dfdx, &numeric.dfdx);
            compare("dfdv", &analytic.dfdv, &numeric.dfdv);
            compare("didx", &analytic.didx, &numeric.didx);
            compare("didv", &analytic.didv, &numeric.didv);
        }
    }
}

/// The dq round trip, checked exactly.
///
/// After initialization the machine must reproduce the very current the power
/// flow gave it. This is the sharpest available check on the rotor-frame
/// conventions — a transposed sine and cosine, a `q` axis defined the other way
/// round, or a sign flipped in the stator equations all survive a plausibility
/// reading and none of them survives this.
///
/// The air-gap power is checked at the same time: it must be the terminal power
/// plus the stator copper loss, which is what makes it the quantity the rotor
/// feels rather than the one the network sees.
#[test]
fn initialization_reproduces_the_terminal_current_and_power() {
    for (v, s) in [
        (Complex::from_polar(1.0, 0.0), Complex::new(0.8, 0.3)),
        (Complex::from_polar(1.05, 0.2), Complex::new(0.5, -0.15)),
        (Complex::from_polar(0.95, -0.1), Complex::new(1.0, 0.6)),
    ] {
        let mut m = new_machine(0.0);
        let init = m.initialize(v, s).unwrap();

        // `injection` returns the machine's own current plus the Norton stamp,
        // so the stamp comes back off to leave the physical current.
        let i_model = m.injection(&init.states, v) - m.norton_admittance() * v;
        let i_expected = (s / v).conj();
        assert!(
            (i_model - i_expected).norm() < 1e-12,
            "terminal current {i_model} should be {i_expected} at v = {v}, s = {s}"
        );

        // P_m is the air-gap power at the equilibrium; it must exceed the
        // terminal power by exactly the copper loss.
        let copper = i_expected.norm_sqr() * machine_params(0.0).ra;
        assert!(
            (init.p_m - (s.re + copper)).abs() < 1e-12,
            "air-gap power {} should be terminal {} plus copper loss {copper:e}",
            init.p_m,
            s.re
        );
    }
}

/// One machine, one line, one load, and **no infinite bus** — so the machine
/// sets the system frequency itself and a governor has something to do.
fn islanded(
    fits: &[(Fit, f64, f64)],
    gen_p: &[f64],
    load: (f64, f64),
) -> (DynamicSystem, Vec<String>) {
    let models: Vec<Box<dyn DynamicModel>> = fits
        .iter()
        .map(|(fit, d, r)| Box::new(unit(*fit, *d, *r)) as Box<dyn DynamicModel>)
        .collect();
    islanded_with(models, gen_p, load)
}

/// As [`islanded`], with the devices supplied directly.
fn islanded_with(
    models: Vec<Box<dyn DynamicModel>>,
    gen_p: &[f64],
    load: (f64, f64),
) -> (DynamicSystem, Vec<String>) {
    let n_gen = models.len();
    let load_bus = n_gen;
    let mut buses = Vec::new();
    for (i, p) in gen_p.iter().enumerate() {
        let kind = if i == 0 { BusType::Slack } else { BusType::PV };
        buses.push(bus(i, kind, 1.0, *p, 0.0));
    }
    buses.push(bus(load_bus, BusType::PQ, 1.0, load.0, load.1));

    let mut lines = Vec::new();
    for i in 0..n_gen {
        lines.push(Line { from: i, to: load_bus, r: 0.005, x: 0.05, b_shunt: 0.0, g_shunt: 0.0 });
    }

    let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
        buses,
        lines: lines.clone(),
    });
    assert_eq!(report.stats.status, gridoxide::solver::SolveStatus::Converged);
    let buses = report.buses;
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();
    let (p_calc, q_calc) = power_injections(&buses, &ybus);

    let mut devices = Vec::new();
    let mut ids = Vec::new();
    for (i, model) in models.into_iter().enumerate() {
        let id = format!("G{}", i + 1);
        ids.push(id.clone());
        devices.push(DeviceSpec { id, bus: i, s: Complex::new(p_calc[i], q_calc[i]), model });
    }

    let system = build(SystemSpec {
        buses: &buses,
        lines: &lines,
        transformers: &[],
        shunts: &[],
        devices,
        fixed_buses: Vec::new(),
    })
    .expect("the solved case initializes to an equilibrium");
    (system, ids)
}

fn options(step: f64, end: f64, events: Vec<Event>) -> DynamicsOptions {
    DynamicsOptions { end_time: end, step, events, ..Default::default() }
}

/// G1, for every combination. Undisturbed, nothing moves — and "nothing" is
/// machine precision, not a solver tolerance.
///
/// This is what catches a control that latched the wrong reference. An exciter
/// initialized to the file's `V_ref` rather than to the field voltage the
/// machine actually needs starts driving immediately, and the trajectory that
/// results is entirely plausible.
#[test]
fn every_combination_is_an_equilibrium() {
    for fit in FITS {
        let (mut system, _) = islanded(&[(fit, 2.0, 0.05)], &[0.8], (-0.8, -0.2));
        assert!(
            system.max_derivative() < 1e-11,
            "{}: initial derivative {:e}",
            fit.label(),
            system.max_derivative()
        );
        assert!(
            system.network_residual_norm() < 1e-11,
            "{}: initial residual {:e}",
            fit.label(),
            system.network_residual_norm()
        );

        let report = run_dynamics(&mut system, &options(0.01, 20.0, Vec::new()));
        assert_eq!(report.status, DynamicsStatus::Completed, "{}", fit.label());

        for name in report.trajectory.names.clone() {
            let series = report.trajectory.series(&name).unwrap();
            // The rotor angle of an islanded machine is free to drift with the
            // system frequency; every other state must stand still.
            if name.ends_with(".delta") || name.ends_with(".vang") {
                continue;
            }
            let drift = series.iter().fold(0.0f64, |m, s| m.max((s - series[0]).abs()));
            assert!(
                drift < 1e-9,
                "{}: {name} drifted by {drift:e} with no disturbance",
                fit.label()
            );
        }
    }
}

/// An exciter returns the terminal voltage; without one it stays where the
/// disturbance left it.
///
/// The steady-state error is not zero and should not be: `SEXS` is a
/// proportional regulator, so it holds `E_fd = K·(V_ref − |V|)` and therefore
/// leaves an offset of `E_fd/K`. With `K = 200` and a field voltage of order
/// two per unit that is about `0.01` — small, real, and the reason a high gain
/// is what makes a machine hold its voltage closely.
#[test]
fn an_exciter_returns_the_terminal_voltage() {
    let recovery = |avr: bool| {
        let fit = Fit { avr, gov: false, pss: false };
        let (mut system, _) = islanded(&[(fit, 2.0, 0.05)], &[0.8], (-0.8, -0.2));
        let opts = options(
            0.005,
            30.0,
            vec![Event::new(1.0, EventKind::LoadStep { bus: 1, ds: Complex::new(-0.25, -0.1) })],
        );
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        let v = report.trajectory.series("bus0.vmag").unwrap();
        (v[0], *v.last().unwrap())
    };

    let (v0_open, v_open) = recovery(false);
    let (v0_avr, v_avr) = recovery(true);

    let sag_open = (v0_open - v_open).abs();
    let sag_avr = (v0_avr - v_avr).abs();
    assert!(
        sag_open > 0.01,
        "the load step should move the voltage without an exciter; it moved {sag_open:e}"
    );
    assert!(
        sag_avr < 0.25 * sag_open,
        "an exciter should recover most of the sag: {sag_avr:.5} left of {sag_open:.5}"
    );
}

/// Two governors share a disturbance in inverse proportion to their droops.
///
/// This is the property droop exists for, and it is a system-level statement
/// that no single model can make on its own: the two machines are rigidly
/// synchronized, so they must settle at the *same* frequency deviation, and
/// each one's steady-state contribution is then `−Δω/R`. A machine with half
/// the droop picks up twice the power.
///
/// The check reads the turbine states straight out of the trajectory, which is
/// what makes it independent of how much power the load step actually
/// delivered — a constant-impedance load draws what the voltage lets it, not
/// what was asked for.
#[test]
fn droop_shares_a_disturbance_in_inverse_proportion() {
    let fit = Fit { avr: true, gov: true, pss: false };
    let (r_a, r_b) = (0.05, 0.10);
    let (mut system, _) =
        islanded(&[(fit, 2.0, r_a), (fit, 2.0, r_b)], &[0.8, 0.8], (-1.6, -0.4));

    let opts = options(
        0.01,
        120.0,
        vec![Event::new(1.0, EventKind::LoadStep { bus: 2, ds: Complex::new(-0.2, 0.0) })],
    );
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);

    let settled = |name: &str| {
        let s = report.trajectory.series(name).unwrap();
        (s[0], *s.last().unwrap())
    };
    let (wa0, wa) = settled("G1.omega");
    let (wb0, wb) = settled("G2.omega");
    let (pa0, pa) = settled("G1.gov_turbine");
    let (pb0, pb) = settled("G2.gov_turbine");

    // Synchronized machines settle at one frequency.
    assert!(
        (wa - wb).abs() < 1e-7,
        "two synchronized machines must settle together: {wa:.9} and {wb:.9}"
    );
    assert!(wa < wa0 && wb < wb0, "more load should lower the frequency");

    let (d_omega, dpa, dpb) = (wa - wa0, pa - pa0, pb - pb0);
    // Each governor's own steady state: x = P_ref − Δω/R.
    for (label, dp, r) in [("G1", dpa, r_a), ("G2", dpb, r_b)] {
        let predicted = -d_omega / r;
        assert!(
            (dp - predicted).abs() < 0.02 * predicted.abs().max(1e-3),
            "{label} should pick up −Δω/R = {predicted:.6}, picked up {dp:.6}"
        );
    }
    // And so the shares are inversely proportional to the droops.
    let ratio = dpa / dpb;
    assert!(
        (ratio - r_b / r_a).abs() < 0.02,
        "the machine with half the droop should pick up twice as much; ratio was {ratio:.4}"
    );
}

/// A stabilizer contributes nothing at steady state and returns to nothing
/// after a disturbance.
///
/// That is the washout's doing, and it is the property that makes a stabilizer
/// safe to add: it cannot shift the voltage setpoint, however it is tuned, so
/// it can only affect the transient. A stabilizer that failed this would be
/// quietly re-regulating the machine.
///
/// **The washout's own state does not go to zero — it goes to the input.**
/// `ẋ₁ = (u − x₁)/T_w`, so at steady state `x₁ = u` and the output `y₁ = u − x₁`
/// is what vanishes. That is precisely the mechanism: the state absorbs the
/// steady component so the output carries only the change. Here there is no
/// governor, so a load increase leaves the frequency permanently low —
/// `Δω = (P_m − P_e)/D` at the machine's own damping — and the washout state
/// settles on exactly that.
#[test]
fn a_stabilizer_leaves_no_steady_signal() {
    let fit = Fit { avr: true, gov: false, pss: true };
    let (mut system, _) = islanded(&[(fit, 2.0, 0.05)], &[0.8], (-0.8, -0.2));
    // Long enough for everything to settle, which takes a surprising while.
    // The washout's own time constant is 10 s and the field flux relaxes with
    // T'_d0 = 8 s, but the loop that closes through the exciter and the network
    // has a much longer tail: the frequency approaches its final
    // `Δω = (P_m − P_e)/D = −0.1` asymptotically as the terminal voltage creeps
    // back to nominal, and at 60 s it is only two thirds of the way there. A
    // shorter run catches the washout faithfully chasing a target that is
    // itself still moving, which looks like a failure and is not.
    let opts = options(
        0.02,
        400.0,
        vec![Event::new(1.0, EventKind::LoadStep { bus: 1, ds: Complex::new(-0.2, 0.0) })],
    );
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);

    let washout = report.trajectory.series("G1.pss_washout").unwrap();
    let lead1 = report.trajectory.series("G1.pss_lead1").unwrap();
    let lead2 = report.trajectory.series("G1.pss_lead2").unwrap();
    let omega = report.trajectory.series("G1.omega").unwrap();

    let peak = lead2.iter().fold(0.0f64, |m, s| m.max(s.abs()));
    assert!(peak > 1e-6, "the stabilizer should respond to the disturbance at all, peaked {peak:e}");

    // The frequency really did stay low — otherwise this proves nothing — and
    // it stayed low by the amount the swing equation's own steady state says:
    // with P_m fixed, `P_m − P_e = D·Δω`, and the 0.2 pu of extra load gives
    // `Δω = −0.2/2.0 = −0.1`.
    let d_omega = omega.last().unwrap() - 1.0;
    assert!(
        (d_omega - -0.1).abs() < 5e-3,
        "with no governor the frequency should settle at (P_m − P_e)/D = −0.1, \
         ended at Δω = {d_omega:e}"
    );

    // The washout state tracks the input — but it tracks it with a **lag of
    // T_w**, so while the input is still creeping the state sits behind it by
    // about `T_w · dΔω/dt`. Asserting a fixed tolerance instead would be
    // asserting that the frequency has completely stopped moving, which on an
    // asymptotic approach it never quite has. Predicting the gap from the
    // observed drift rate is both tighter and an actual statement about what a
    // washout does.
    let t = &report.trajectory.time;
    let last = omega.len() - 1;
    let window = last - last / 20;
    let rate = (omega[last] - omega[window]) / (t[last] - t[window]);
    let expected_lag = 10.0 * rate.abs(); // T_w = 10 s
    let gap = (washout[last] - d_omega).abs();
    assert!(
        gap < 2.0 * expected_lag + 1e-7,
        "the washout should trail the input by about T_w·dΔω/dt = {expected_lag:e}; \
         it trailed by {gap:e}"
    );
    // … which leaves both lead-lag states, and so the output, at zero.
    // Both lead-lag stages have decayed by orders of magnitude from their peak
    // — that is the qualitative claim.
    for (name, series) in [("lead1", &lead1), ("lead2", &lead2)] {
        let stage_peak = series.iter().fold(0.0f64, |m, s| m.max(s.abs()));
        let end = series.last().unwrap().abs();
        assert!(
            end < 0.02 * stage_peak,
            "pss_{name} should decay toward zero: ended at {end:e} against a peak of {stage_peak:e}"
        );
    }

    // And the quantitative one: whatever is *left* is exactly what the
    // washout's residual output feeds forward. The lead-lags are unity at DC,
    // so at any settled point the first stage's state is
    // `K·(1 − T₁/T₂)·y₁`, with `y₁ = u − x₁` the washout's output. Nothing here
    // is drifting on its own — every remaining signal is the same lag, scaled.
    let y1 = d_omega - washout[last];
    let predicted = 5.0 * (1.0 - 0.15 / 0.03) * y1;
    assert!(
        (lead1[last] - predicted).abs() < 0.05 * predicted.abs() + 1e-9,
        "the first lead-lag's residual should be K(1 − T₁/T₂)·y₁ = {predicted:e}, \
         it was {:e}",
        lead1[last]
    );
}


/// The three ZIP parts do what their names say, and the cutoff turns all of
/// them into the same thing.
///
/// Checked on the model directly rather than through a run, because the claim
/// is about the constitutive law and nothing is gained by routing it through an
/// integrator.
#[test]
fn a_zip_load_follows_its_own_exponents() {
    let v0 = Complex::new(1.0, 0.0);
    let s0 = Complex::new(-0.8, -0.3);
    let cutoff = 0.5;

    // Power drawn at a given magnitude, as a multiple of the operating point.
    let ratio = |z: f64, i: f64, p: f64, v_mag: f64| {
        let mut load = ZipLoad::new(z, i, p, cutoff).unwrap();
        load.initialize(v0, s0).unwrap();
        let v = Complex::from_polar(v_mag, 0.35);
        let s = v * load.injection(&[], v).conj();
        s.norm() / s0.norm()
    };

    for v_mag in [0.7, 0.9, 1.1] {
        let (z, i, p) = (ratio(1.0, 0.0, 0.0, v_mag), ratio(0.0, 1.0, 0.0, v_mag), ratio(0.0, 0.0, 1.0, v_mag));
        assert!((z - v_mag * v_mag).abs() < 1e-12, "constant impedance at |V| = {v_mag}: {z}");
        assert!((i - v_mag).abs() < 1e-12, "constant current at |V| = {v_mag}: {i}");
        assert!((p - 1.0).abs() < 1e-12, "constant power at |V| = {v_mag}: {p}");
    }

    // Below the cutoff every part is a constant impedance referred to the
    // cutoff voltage, so all three fall off as |V|² from whatever they were
    // drawing there.
    for v_mag in [0.1, 0.3, 0.49] {
        let scale = (v_mag / cutoff).powi(2);
        for (z, i, p, at_cutoff) in [
            (1.0, 0.0, 0.0, cutoff * cutoff),
            (0.0, 1.0, 0.0, cutoff),
            (0.0, 0.0, 1.0, 1.0),
        ] {
            let got = ratio(z, i, p, v_mag);
            assert!(
                (got - at_cutoff * scale).abs() < 1e-12,
                "below the cutoff, ({z},{i},{p}) at |V| = {v_mag} should be \
                 {} but was {got}",
                at_cutoff * scale
            );
        }
    }

    // Continuity at the cutoff itself: the current does not jump, which is what
    // makes the switch usable inside a Newton step.
    let mut load = ZipLoad::constant_power(cutoff).unwrap();
    load.initialize(v0, s0).unwrap();
    let eps = 1e-9;
    let below = load.injection(&[], Complex::from_polar(cutoff - eps, 0.2));
    let above = load.injection(&[], Complex::from_polar(cutoff + eps, 0.2));
    assert!((below - above).norm() < 1e-7, "the current should be continuous at the cutoff");
}

/// The ZIP load's `∂I/∂V` against the oracle, on both sides of the cutoff.
///
/// A load with no differential states at all is still a device in the DAE — it
/// contributes a nonlinear injection and its derivative — and this is the only
/// model here whose Jacobian is *entirely* the `∂I/∂V` block.
#[test]
fn a_zip_load_matches_the_finite_difference_oracle() {
    let mut load = ZipLoad::new(0.3, 0.3, 0.4, 0.5).unwrap();
    load.initialize(Complex::new(1.0, 0.0), Complex::new(-0.8, -0.3)).unwrap();

    // Never straddling the cutoff: the derivative there is genuinely
    // discontinuous, so a central difference across it is not approximating
    // anything and disagreeing would say nothing.
    for v in [
        Complex::new(1.00, 0.00),
        Complex::new(0.87, -0.24),
        Complex::new(0.62, 0.10),
        Complex::new(0.20, 0.05),
        Complex::new(0.10, -0.03),
    ] {
        let mut analytic = ModelJacobian::zeros(0);
        analytic.clear();
        load.jacobian(&[], v, &mut analytic);
        let numeric = finite_difference(&load, &[], v, 1e-7);
        for k in 0..4 {
            let (a, b) = (analytic.didv[k], numeric.didv[k]);
            let scale = a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() / scale < 1e-6,
                "didv[{k}] at v = {v}: analytic {a:e} vs numeric {b:e}"
            );
        }
    }
}

/// A constant-power load makes a voltage sag worse than a constant-impedance
/// one, which is the whole reason the choice is exposed.
///
/// `init` converts every residual injection to a constant admittance, and that
/// default is optimistic: an impedance sheds power as `|V|²`, so it helps the
/// voltage recover from exactly the disturbance that depressed it. A
/// constant-power load keeps drawing what it drew, which is the pessimistic and
/// often more realistic assumption.
#[test]
fn a_constant_power_load_deepens_a_sag() {
    let sag = |constant_power: bool| {
        let fit = Fit { avr: false, gov: false, pss: false };
        let buses = vec![
            bus(0, BusType::Slack, 1.0, 0.8, 0.0),
            bus(1, BusType::PQ, 1.0, -0.8, -0.3),
        ];
        let lines = vec![Line { from: 0, to: 1, r: 0.01, x: 0.10, b_shunt: 0.0, g_shunt: 0.0 }];
        let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
            buses,
            lines: lines.clone(),
        });
        let buses = report.buses;
        let ybus = build_ybus(buses.len(), &lines, &[]).finish();
        let (p_calc, q_calc) = power_injections(&buses, &ybus);

        let mut devices = vec![DeviceSpec {
            id: "G1".to_string(),
            bus: 0,
            s: Complex::new(p_calc[0], q_calc[0]),
            model: Box::new(unit(fit, 2.0, 0.05)),
        }];
        if constant_power {
            devices.push(DeviceSpec {
                id: "L1".to_string(),
                bus: 1,
                s: Complex::new(p_calc[1], q_calc[1]),
                model: Box::new(ZipLoad::constant_power(0.5).unwrap()),
            });
        }

        let mut system = build(SystemSpec {
            buses: &buses,
            lines: &lines,
            transformers: &[],
            shunts: &[],
            devices,
            fixed_buses: Vec::new(),
        })
        .expect("initializes to an equilibrium either way");

        assert!(system.max_derivative() < 1e-11);
        assert!(system.network_residual_norm() < 1e-11);

        // Weaken the network rather than the load, so the two cases are given
        // exactly the same disturbance.
        let opts = options(
            0.005,
            10.0,
            vec![Event::new(1.0, EventKind::BusFault { bus: 1, y: Complex::new(1.5, 0.0) })],
        );
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        let v = report.trajectory.series("bus1.vmag").unwrap();
        v[0] - v.iter().cloned().fold(f64::INFINITY, f64::min)
    };

    let impedance = sag(false);
    let power = sag(true);
    assert!(
        power > impedance,
        "a constant-power load should sag further than a constant-impedance one: \
         {power:.5} against {impedance:.5}"
    );
}

fn round_params(xdpp: f64, xqpp: f64) -> GenRoundParams {
    GenRoundParams {
        h: 5.0,
        d: 2.0,
        ra: 0.003,
        xd: 1.8,
        xq: 1.7,
        xdp: 0.30,
        xqp: 0.55,
        xdpp,
        xqpp,
        xl: 0.15,
        td0p: 8.0,
        tq0p: 0.4,
        td0pp: 0.03,
        tq0pp: 0.05,
        mbase: S_BASE,
    }
}

/// A machine with no subtransient saliency presents **exactly** its own
/// admittance to the network, so `∂I/∂V` comes out exactly zero.
///
/// This is the sharpest check available on the rotor-frame algebra, and it is
/// exact rather than approximate. `injection` returns the machine's current
/// plus the Norton stamp; when `x''_d = x''_q` the machine really is an
/// impedance in the network frame, the stamp is that impedance, and the two
/// cancel to the last bit. A transposed sine, a `q` axis defined the other way
/// round, or a sign slip anywhere in the 2×2 stator solve all leave a residue.
///
/// It also confirms what the stamp is *for*: it is a conditioning device that
/// changes no answer, and here that claim is visible as an identity.
#[test]
fn a_non_salient_machine_cancels_its_own_norton_stamp() {
    let v = Complex::from_polar(1.02, 0.17);
    let s = Complex::new(0.7, 0.25);

    let mut transient = GenTransient::new(
        GenTransientParams { xqp: 0.30, ..machine_params(0.0) },
        S_BASE,
        F_NOM,
    )
    .unwrap();
    let init = transient.initialize(v, s).unwrap();
    let mut jac = gridoxide::dynamics::models::MachineJacobian::zeros(4);
    jac.clear();
    transient.jacobian(&init.states, v, init.e_fd, init.p_m, &mut jac);
    for (k, value) in jac.didv.iter().enumerate() {
        assert!(value.abs() < 1e-12, "transient didv[{k}] = {value:e}, should be exactly zero");
    }

    let mut round = GenRound::new(round_params(0.22, 0.22), S_BASE, F_NOM).unwrap();
    let init = round.initialize(v, s).unwrap();
    let mut jac = gridoxide::dynamics::models::MachineJacobian::zeros(6);
    jac.clear();
    round.jacobian(&init.states, v, init.e_fd, init.p_m, &mut jac);
    for (k, value) in jac.didv.iter().enumerate() {
        assert!(value.abs() < 1e-12, "round didv[{k}] = {value:e}, should be exactly zero");
    }
}

/// The sixth-order machine's initialization reproduces its terminal current,
/// and its Jacobian matches the oracle.
#[test]
fn the_subtransient_machine_is_self_consistent() {
    for (v, s) in [
        (Complex::from_polar(1.0, 0.0), Complex::new(0.8, 0.3)),
        (Complex::from_polar(1.05, 0.2), Complex::new(0.5, -0.15)),
        (Complex::from_polar(0.95, -0.1), Complex::new(1.0, 0.6)),
    ] {
        let mut m = GenRound::new(round_params(0.22, 0.25), S_BASE, F_NOM).unwrap();
        let init = m.initialize(v, s).unwrap();
        let i_model = m.injection(&init.states, v) - m.norton_admittance() * v;
        let i_expected = (s / v).conj();
        assert!(
            (i_model - i_expected).norm() < 1e-12,
            "terminal current {i_model} should be {i_expected}"
        );
        let copper = i_expected.norm_sqr() * round_params(0.22, 0.25).ra;
        assert!((init.p_m - (s.re + copper)).abs() < 1e-12);
    }

    let mut model = GeneratingUnit::new(
        Box::new(GenRound::new(round_params(0.22, 0.25), S_BASE, F_NOM).unwrap()),
        Some(Box::new(new_avr())),
        Some(Box::new(new_gov(0.05))),
        Some(Box::new(new_pss())),
    );
    model
        .initialize(Complex::from_polar(1.02, 0.15), Complex::new(0.8, 0.3))
        .unwrap();
    let n = model.n_states();
    for scale in [0.13, -0.21, 0.17] {
        let x: Vec<f64> = (0..n).map(|k| 0.4 + scale * k as f64).collect();
        let v = Complex::new(0.97, 0.11);
        let mut analytic = ModelJacobian::zeros(n);
        analytic.clear();
        model.jacobian(&x, v, &mut analytic);
        let numeric = finite_difference(&model, &x, v, 1e-6);
        for (name, a, b) in [
            ("dfdx", &analytic.dfdx, &numeric.dfdx),
            ("dfdv", &analytic.dfdv, &numeric.dfdv),
            ("didx", &analytic.didx, &numeric.didx),
        ] {
            for (k, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                let sc = av.abs().max(bv.abs()).max(1.0);
                assert!(
                    (av - bv).abs() / sc < 1e-6,
                    "{name}[{k}]: analytic {av:e} vs numeric {bv:e}"
                );
            }
        }
    }
}

/// The sixth-order machine reduces to the fourth-order one when its damper
/// windings are made ineffective.
///
/// This is the gate that pins the subtransient sign conventions, which
/// published statements of the model disagree about. Take `x''` up to `x'` on
/// both axes and the interpolation coefficients `b_d`, `b_q` go to zero: the
/// subtransient EMFs collapse onto the transient ones, the damper-coupling
/// corrections vanish, and the subtransient stator equations *become* the
/// transient ones. The two machines must then trace the same trajectory
/// through the same disturbance.
///
/// Because `GenTransient` is independently pinned — by the terminal-current
/// identity and the Norton-cancellation identity above — this transfers that
/// confidence to `GenRound`. What it cannot confirm is the *magnitude* of the
/// damper coupling in the regime where it matters; that waits for the external
/// comparison in phase 5.
#[test]
fn the_subtransient_machine_reduces_to_the_transient_one() {
    let trajectory = |sixth: bool| {
        let model: Box<dyn DynamicModel> = if sixth {
            // As close to the transient reactances as the parameter ordering
            // allows, so the dampers exist but contribute nothing.
            Box::new(GeneratingUnit::machine_only(Box::new(
                GenRound::new(round_params(0.30 - 1e-7, 0.55 - 1e-7), S_BASE, F_NOM).unwrap(),
            )))
        } else {
            Box::new(GeneratingUnit::machine_only(Box::new(new_machine(2.0))))
        };
        let (mut system, _) = islanded_with(vec![model], &[0.8], (-0.8, -0.2));
        let opts = options(
            0.002,
            8.0,
            vec![
                Event::bolted_fault(1.0, 1),
                Event::new(1.08, EventKind::ClearFault { bus: 1 }),
            ],
        );
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        (
            report.trajectory.series("G1.delta").unwrap(),
            report.trajectory.series("G1.omega").unwrap(),
        )
    };

    let (d4, w4) = trajectory(false);
    let (d6, w6) = trajectory(true);
    assert_eq!(d4.len(), d6.len());

    let worst_d = d4.iter().zip(d6.iter()).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let worst_w = w4.iter().zip(w6.iter()).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst_d < 1e-5,
        "with the dampers made ineffective the two orders should agree; \
         the rotor angles differ by {worst_d:e} rad"
    );
    assert!(worst_w < 1e-7, "the speeds differ by {worst_w:e} pu");
}

/// Damper windings make the machine stiffer against a sudden change, which is
/// the whole reason to carry them.
///
/// At the instant a fault arrives, every differential state is frozen, so the
/// machine's response is governed entirely by what it looks like *right then* —
/// a subtransient EMF behind `x''`, not a transient one behind `x'`. A lower
/// reactance means more current, so a sixth-order machine feeds a fault harder
/// than an otherwise identical fourth-order one. Reading the current at the
/// event's own instant is what isolates the effect from everything that
/// happens afterwards.
#[test]
fn dampers_stiffen_the_instantaneous_response() {
    let fault_current = |sixth: bool| {
        let model: Box<dyn DynamicModel> = if sixth {
            Box::new(GeneratingUnit::machine_only(Box::new(
                GenRound::new(round_params(0.20, 0.22), S_BASE, F_NOM).unwrap(),
            )))
        } else {
            Box::new(GeneratingUnit::machine_only(Box::new(new_machine(2.0))))
        };
        let (mut system, _) = islanded_with(vec![model], &[0.8], (-0.8, -0.2));
        let opts = options(
            0.002,
            1.2,
            vec![Event::new(1.0, EventKind::BusFault { bus: 1, y: Complex::new(4.0, 0.0) })],
        );
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        // The two rows recorded at the fault's own time: before, and after the
        // algebraic re-solve.
        let at: Vec<usize> = report
            .trajectory
            .time
            .iter()
            .enumerate()
            .filter(|(_, t)| (**t - 1.0).abs() < 1e-12)
            .map(|(k, _)| k)
            .collect();
        assert_eq!(at.len(), 2);
        let col = report.trajectory.column("bus0.vmag").unwrap();
        let (before, after) =
            (report.trajectory.rows[at[0]][col], report.trajectory.rows[at[1]][col]);
        (before, after)
    };

    let (v4_before, v4_after) = fault_current(false);
    let (v6_before, v6_after) = fault_current(true);
    assert!((v4_before - v6_before).abs() < 1e-6, "the two should start from the same point");
    assert!(
        v6_after > v4_after,
        "a stiffer machine should hold its terminal voltage up better through the \
         first instant: {v6_after:.6} against {v4_after:.6}"
    );
}

/// Tripping a unit freezes it and leaves the rest of the system to cope.
///
/// The freeze is exact, not approximate: the tripped unit's rows become
/// `x₁ − x₀ = 0` and its states hold bit-for-bit. That is what makes a unit
/// trip a value-only event — the sparsity pattern is untouched, so the run's
/// single symbolic factorization still serves and nothing re-analyzes.
#[test]
fn a_tripped_unit_freezes_and_the_rest_picks_up_the_load() {
    let fit = Fit { avr: true, gov: true, pss: false };
    let (mut system, _) =
        islanded(&[(fit, 2.0, 0.05), (fit, 2.0, 0.05)], &[0.8, 0.8], (-1.6, -0.4));

    let opts = options(
        0.005,
        40.0,
        vec![Event::new(1.0, EventKind::UnitTrip { unit: 1 })],
    );
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);
    assert_eq!(report.events_applied, 1);

    // Every state of the tripped unit is bit-identical from the trip onward.
    let after_trip: Vec<usize> = report
        .trajectory
        .time
        .iter()
        .enumerate()
        .filter(|(_, t)| **t >= 1.0)
        .map(|(k, _)| k)
        .collect();
    for name in report.trajectory.names.clone() {
        if !name.starts_with("G2.") {
            continue;
        }
        let col = report.trajectory.column(&name).unwrap();
        let frozen = report.trajectory.rows[after_trip[1]][col];
        for &k in &after_trip[1..] {
            assert_eq!(
                report.trajectory.rows[k][col].to_bits(),
                frozen.to_bits(),
                "{name} should be frozen after the trip"
            );
        }
    }

    // And the survivor takes the load: its governor opens up and the frequency
    // settles low, since one machine's droop now carries what two shared.
    let w = report.trajectory.series("G1.omega").unwrap();
    let p = report.trajectory.series("G1.gov_turbine").unwrap();
    assert!(
        *w.last().unwrap() < w[0] - 1e-3,
        "losing half the generation should depress the frequency, Δω = {:e}",
        w.last().unwrap() - w[0]
    );
    assert!(
        *p.last().unwrap() > p[0] + 0.1,
        "the survivor should pick up load: {:.4} to {:.4}",
        p[0],
        p.last().unwrap()
    );
    // The droop relation still holds for the one machine left.
    let predicted = -(w.last().unwrap() - w[0]) / 0.05;
    let actual = p.last().unwrap() - p[0];
    assert!(
        (actual - predicted).abs() < 0.03 * predicted.abs(),
        "the survivor should pick up −Δω/R = {predicted:.5}, picked up {actual:.5}"
    );
}

/// The full form — rotor speed on the speed-voltage terms, swing equation in
/// torque — is a *different model*, and has to be as well founded as the
/// approximate one.
///
/// Both properties that matter carry over, and neither is automatic. The
/// analytic Jacobian gains a whole column: `ω` now enters the stator solve
/// through the coefficients *and* through its determinant, so its derivative is
/// a product rule rather than the fixed 2×2 inverse every other column goes
/// through. And initialization is untouched, because at synchronous speed the
/// two forms coincide exactly — which is why the equilibrium still holds
/// without a line of new initialization code.
#[test]
fn the_full_form_is_as_well_founded_as_the_approximate_one() {
    // The oracle, over every machine, with the speed voltages on.
    let machines: Vec<Box<dyn Machine>> = vec![
        Box::new(new_machine(2.0).with_speed_voltages(true)),
        Box::new(GenRound::new(round_params(0.22, 0.25), S_BASE, F_NOM)
            .unwrap()
            .with_speed_voltages(true)),
    ];
    for machine in machines {
        let mut model = GeneratingUnit::new(
            machine,
            Some(Box::new(new_avr())),
            Some(Box::new(new_gov(0.05))),
            None,
        );
        model
            .initialize(Complex::from_polar(1.02, 0.15), Complex::new(0.8, 0.3))
            .unwrap();
        let n = model.n_states();
        for scale in [0.13, -0.19] {
            // The speed is probed *away* from 1.0 deliberately: at synchronous
            // speed the new terms vanish and the oracle would be checking the
            // approximate form all over again.
            let mut x: Vec<f64> = (0..n).map(|k| 0.4 + scale * k as f64).collect();
            x[1] = 1.03;
            let v = Complex::new(0.97, 0.11);
            let mut analytic = ModelJacobian::zeros(n);
            analytic.clear();
            model.jacobian(&x, v, &mut analytic);
            let numeric = finite_difference(&model, &x, v, 1e-6);
            for (name, a, b) in [
                ("dfdx", &analytic.dfdx, &numeric.dfdx),
                ("dfdv", &analytic.dfdv, &numeric.dfdv),
                ("didx", &analytic.didx, &numeric.didx),
                ("didv", &analytic.didv.to_vec(), &numeric.didv.to_vec()),
            ] {
                for (k, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                    let sc = av.abs().max(bv.abs()).max(1.0);
                    assert!(
                        (av - bv).abs() / sc < 1e-6,
                        "{name}[{k}]: analytic {av:e} vs numeric {bv:e}"
                    );
                }
            }
        }
    }

    // And the equilibrium, through a whole run.
    let model: Box<dyn DynamicModel> = Box::new(GeneratingUnit::new(
        Box::new(GenRound::new(round_params(0.22, 0.25), S_BASE, F_NOM)
            .unwrap()
            .with_speed_voltages(true)),
        Some(Box::new(new_avr())),
        Some(Box::new(new_gov(0.05))),
        None,
    ));
    let (mut system, _) = islanded_with(vec![model], &[0.8], (-0.8, -0.2));
    assert!(
        system.max_derivative() < 1e-11,
        "the full form must initialize to an equilibrium too, drift {:e}",
        system.max_derivative()
    );
    let report = run_dynamics(&mut system, &options(0.01, 20.0, Vec::new()));
    assert_eq!(report.status, DynamicsStatus::Completed);
    let omega = report.trajectory.series("G1.omega").unwrap();
    let drift = omega.iter().fold(0.0f64, |m, w| m.max((w - omega[0]).abs()));
    assert!(drift < 1e-9, "undisturbed speed drifted by {drift:e}");
}

/// The two forms are **identical** at synchronous speed and differ in
/// proportion to the speed deviation — which is the whole basis of the claim
/// that the difference is the factor of `ω` and not something else.
///
/// The first half is exact and is the sharper statement: with nothing
/// disturbed, `ω` is one, every new term collapses, and the two trajectories
/// must agree bit for bit. A flag that changed anything there would not be the
/// `ω` factor.
///
/// The machine needs something to swing *against*, so this builds a
/// single-machine-infinite-bus case rather than reusing the islanded helper —
/// an islanded machine's rotor angle drifts freely with the system frequency,
/// and comparing two drifts says nothing about either.
#[test]
fn the_two_forms_agree_at_synchronous_speed_and_part_in_proportion_to_deviation() {
    let run = |speed_voltages: bool, events: Vec<Event>| {
        let buses = vec![
            bus(0, BusType::PV, 1.0, 0.8, 0.0),
            bus(1, BusType::Slack, 1.0, 0.0, 0.0),
        ];
        let lines = vec![Line { from: 0, to: 1, r: 0.0, x: 0.20, b_shunt: 0.0, g_shunt: 0.0 }];
        let report = gridoxide::run_power_flow_analysis(gridoxide::json::NetworkData {
            buses,
            lines: lines.clone(),
        });
        let buses = report.buses;
        let ybus = build_ybus(buses.len(), &lines, &[]).finish();
        let (p_calc, q_calc) = power_injections(&buses, &ybus);

        let machine = GenRound::new(round_params(0.22, 0.25), S_BASE, F_NOM)
            .unwrap()
            .with_speed_voltages(speed_voltages);
        let mut system = build(SystemSpec {
            buses: &buses,
            lines: &lines,
            transformers: &[],
            shunts: &[],
            devices: vec![DeviceSpec {
                id: "G1".to_string(),
                bus: 0,
                s: Complex::new(p_calc[0], q_calc[0]),
                model: Box::new(GeneratingUnit::machine_only(Box::new(machine))),
            }],
            fixed_buses: vec![1],
        })
        .expect("both forms initialize to the same equilibrium");

        let report = run_dynamics(&mut system, &options(0.002, 6.0, events));
        assert_eq!(report.status, DynamicsStatus::Completed);
        (
            report.trajectory.series("G1.delta").unwrap(),
            report.trajectory.series("G1.omega").unwrap(),
        )
    };

    // Undisturbed: identical, bit for bit.
    let (still_approx, _) = run(false, Vec::new());
    let (still_full, _) = run(true, Vec::new());
    for (a, b) in still_approx.iter().zip(still_full.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "at synchronous speed the two forms must be the same equations"
        );
    }

    // Disturbed: different, and by the order of the speed deviation.
    let fault = vec![
        Event::bolted_fault(1.0, 0),
        Event::new(1.06, EventKind::ClearFault { bus: 0 }),
    ];
    let (d_approx, w_approx) = run(false, fault.clone());
    let (d_full, _) = run(true, fault);

    let deviation = w_approx.iter().fold(0.0f64, |m, w| m.max((w - 1.0).abs()));
    let gap = d_approx
        .iter()
        .zip(d_full.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let swing = d_approx.iter().fold(0.0f64, |m, d| m.max(*d))
        - d_approx.iter().fold(f64::MAX, |m, d| m.min(*d));

    assert!(deviation > 1e-3, "the case must actually leave synchronous speed");
    assert!(swing < std::f64::consts::TAU, "the machine must stay in step, swing {swing:e}");
    assert!(gap > 1e-5, "the flag must reach the equations, but the gap was {gap:e}");
    assert!(
        gap < 5.0 * deviation * swing,
        "the two forms should differ by the order of the speed deviation: gap {gap:e} \
         against a deviation of {deviation:e} on a swing of {swing:e}"
    );
}

fn limited_avr(limits: Limits) -> Sexs {
    Sexs::new(SexsParams { k: 200.0, ta: 0.1, tb: 1.0, te: 0.05, limits }).unwrap()
}

/// A ceiling the operating point cannot respect has no equilibrium, and is
/// refused by name rather than initialized to a state the model would leave.
///
/// This is what `InitError::OutsideLimits` was reserved for in phase 1 and
/// never used until now.
#[test]
fn an_unreachable_equilibrium_is_refused() {
    // The machine at this operating point needs about 2.3 pu of field voltage.
    let mut avr = limited_avr(Limits::new(-1.0, 1.0));
    let err = avr.initialize(2.3, -1.0).expect_err("a 1.0 ceiling cannot hold 2.3");
    assert!(
        matches!(err, InitError::OutsideLimits { name, value } if name.contains("field") && value == 2.3),
        "got {err}"
    );
    assert!(err.to_string().contains("outside its limits"), "{err}");

    // A ceiling that *can* hold it initializes exactly as an unlimited one.
    let mut limited = limited_avr(Limits::new(-6.0, 6.0));
    let mut free = limited_avr(Limits::NONE);
    assert_eq!(limited.initialize(2.3, -1.0).unwrap(), free.initialize(2.3, -1.0).unwrap());
}

/// The limits are **non-windup**: the state is held at the boundary rather than
/// integrating past it.
///
/// The signature is unmissable once looked for. This exciter has a gain of 200,
/// so during a fault its unlimited state would climb to many times the ceiling;
/// with windup it would then take a visible time to come back down after the
/// fault cleared, pinning the field voltage long after the voltage error
/// reversed. Held at the boundary instead, it starts falling on the next step.
///
/// Two assertions, and the first is the sharper: the state never meaningfully
/// exceeds the ceiling at all.
#[test]
fn a_limit_is_non_windup() {
    let ceiling = 2.6;
    let fit = Fit { avr: true, gov: false, pss: false };
    let unit = GeneratingUnit::new(
        Box::new(new_machine(2.0)),
        Some(Box::new(limited_avr(Limits::new(-6.0, ceiling)))),
        None,
        None,
    );
    let (mut system, _) = islanded_with(vec![Box::new(unit)], &[0.8], (-0.8, -0.2));
    let _ = fit;

    let clear = 1.6;
    // Long enough for the field flux to recover, which is what actually
    // releases the exciter — see below.
    let opts = options(
        0.002,
        30.0,
        vec![
            Event::new(1.0, EventKind::BusFault { bus: 1, y: Complex::new(3.0, 0.0) }),
            Event::new(clear, EventKind::ClearFault { bus: 1 }),
        ],
    );
    let report = run_dynamics(&mut system, &opts);
    assert_eq!(report.status, DynamicsStatus::Completed);

    let time = &report.trajectory.time;
    let efd = report.trajectory.series("G1.efd").unwrap();
    let peak = efd.iter().fold(0.0f64, |m, e| m.max(*e));
    assert!(
        peak > ceiling - 1e-6,
        "the fault should drive the exciter to its ceiling; it reached {peak}"
    );
    // A wound-up state would be far above this — the gain is 200, and the same
    // fault drives an unlimited exciter to 18 pu. Held at the boundary and
    // projected back after each step, it never exceeds it at all.
    assert!(
        peak <= ceiling + 1e-12,
        "a non-windup limit holds the state at the boundary; it reached {peak}"
    );

    // And it does come off the ceiling and settle below it. Note that it stays
    // there for **seconds** after the fault clears, and that is physical rather
    // than windup: the field flux decayed during the fault and recovers with
    // T'_d0 = 8 s, so the voltage error the exciter is answering is genuinely
    // still positive. What windup would add is a further delay proportional to
    // how far the state had travelled past the boundary — which is exactly what
    // the assertion above rules out.
    let settled = *efd.last().unwrap();
    assert!(
        settled < ceiling - 1e-3,
        "the field voltage should settle back below the ceiling, but ended at {settled}"
    );
    let _ = time;
}

/// A ceiling changes the answer, which is the reason to model one.
///
/// A limited exciter cannot support the terminal voltage as hard, so the
/// voltage sits lower through the disturbance. If it made no difference the
/// limit would not be reaching the machine.
#[test]
fn a_ceiling_holds_the_voltage_lower() {
    let recovery = |limits: Limits| {
        let unit = GeneratingUnit::new(
            Box::new(new_machine(2.0)),
            Some(Box::new(limited_avr(limits))),
            None,
            None,
        );
        let (mut system, _) = islanded_with(vec![Box::new(unit)], &[0.8], (-0.8, -0.2));
        let opts = options(
            0.005,
            20.0,
            vec![Event::new(1.0, EventKind::LoadStep { bus: 1, ds: Complex::new(-0.35, -0.15) })],
        );
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        let v = report.trajectory.series("bus0.vmag").unwrap();
        *v.last().unwrap()
    };

    let free = recovery(Limits::NONE);
    let capped = recovery(Limits::new(-6.0, 2.35));
    assert!(
        capped < free - 1e-3,
        "a capped exciter should leave the voltage lower: {capped:.5} against {free:.5}"
    );
}

/// The analytic Jacobian stays exact on each side of a limit.
///
/// The boundary itself is a genuine kink — the derivative does not exist there
/// — so the oracle is asked about points clearly inside and clearly outside,
/// never straddling. That is the same treatment the ZIP load's low-voltage
/// cutoff gets, and for the same reason: disagreeing across a discontinuity
/// would say nothing about either side.
#[test]
fn a_limited_control_matches_the_oracle_on_both_sides() {
    let mut model = GeneratingUnit::new(
        Box::new(new_machine(2.0)),
        Some(Box::new(limited_avr(Limits::new(-6.0, 2.6)))),
        Some(Box::new(
            Tgov1::new(Tgov1Params {
                r: 0.05,
                t1: 0.5,
                t2: 1.0,
                t3: 5.0,
                dt: 0.0,
                limits: Limits::new(0.0, 0.9),
            })
            .unwrap(),
        )),
        None,
    );
    model
        .initialize(Complex::from_polar(1.02, 0.15), Complex::new(0.8, 0.3))
        .unwrap();

    let n = model.n_states();
    // States 4 and 5 are the exciter's; 6 and 7 the governor's. The three
    // probes sit well inside, well above and well below the boundaries.
    for (efd, valve) in [(1.0, 0.5), (4.0, 1.6), (-9.0, -0.6)] {
        let mut x: Vec<f64> = (0..n).map(|k| 0.3 + 0.11 * k as f64).collect();
        x[1] = 1.004;
        x[5] = efd;
        x[6] = valve;
        let v = Complex::new(0.98, 0.09);

        let mut analytic = ModelJacobian::zeros(n);
        analytic.clear();
        model.jacobian(&x, v, &mut analytic);
        let numeric = finite_difference(&model, &x, v, 1e-7);
        for (name, a, b) in [
            ("dfdx", &analytic.dfdx, &numeric.dfdx),
            ("dfdv", &analytic.dfdv, &numeric.dfdv),
            ("didx", &analytic.didx, &numeric.didx),
        ] {
            for (k, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                let sc = av.abs().max(bv.abs()).max(1.0);
                assert!(
                    (av - bv).abs() / sc < 1e-6,
                    "{name}[{k}] at efd = {efd}, valve = {valve}: {av:e} vs {bv:e}"
                );
            }
        }
    }
}


fn salient_params() -> GenSalientParams {
    GenSalientParams {
        h: 5.0,
        d: 1.0,
        ra: 0.003,
        xd: 1.8,
        xq: 1.7,
        xdp: 0.30,
        xdpp: 0.22,
        xqpp: 0.25,
        xl: 0.15,
        td0p: 8.0,
        td0pp: 0.03,
        tq0pp: 0.05,
        mbase: S_BASE,
    }
}

/// The fifth-order machine is self-consistent, and its `q` axis really does
/// carry one winding rather than two.
#[test]
fn the_salient_machine_is_self_consistent() {
    for (v, s) in [
        (Complex::from_polar(1.0, 0.0), Complex::new(0.8, 0.3)),
        (Complex::from_polar(1.05, 0.2), Complex::new(0.5, -0.15)),
        (Complex::from_polar(0.95, -0.1), Complex::new(1.0, 0.6)),
    ] {
        let mut m = GenSalient::new(salient_params(), S_BASE, F_NOM).unwrap();
        let init = m.initialize(v, s).unwrap();
        assert_eq!(init.states.len(), 5, "δ, ω, e'_q, ψ_1d and one q-axis flux");

        let i_model = m.injection(&init.states, v) - m.norton_admittance() * v;
        let i_expected = (s / v).conj();
        assert!(
            (i_model - i_expected).norm() < 1e-12,
            "terminal current {i_model} should be {i_expected}"
        );
        let copper = i_expected.norm_sqr() * salient_params().ra;
        assert!((init.p_m - (s.re + copper)).abs() < 1e-12);
    }

    // The oracle, with controls attached so the chain rule is under test too.
    let mut model = GeneratingUnit::new(
        Box::new(GenSalient::new(salient_params(), S_BASE, F_NOM).unwrap()),
        Some(Box::new(new_avr())),
        Some(Box::new(new_gov(0.05))),
        None,
    );
    model
        .initialize(Complex::from_polar(1.02, 0.15), Complex::new(0.8, 0.3))
        .unwrap();
    let n = model.n_states();
    for scale in [0.13, -0.21] {
        let x: Vec<f64> = (0..n).map(|k| 0.4 + scale * k as f64).collect();
        let v = Complex::new(0.97, 0.11);
        let mut analytic = ModelJacobian::zeros(n);
        analytic.clear();
        model.jacobian(&x, v, &mut analytic);
        let numeric = finite_difference(&model, &x, v, 1e-6);
        for (name, a, b) in [
            ("dfdx", &analytic.dfdx, &numeric.dfdx),
            ("dfdv", &analytic.dfdv, &numeric.dfdv),
            ("didx", &analytic.didx, &numeric.didx),
        ] {
            for (k, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                let sc = av.abs().max(bv.abs()).max(1.0);
                assert!(
                    (av - bv).abs() / sc < 1e-6,
                    "{name}[{k}]: analytic {av:e} vs numeric {bv:e}"
                );
            }
        }
    }
}

/// The sixth-order machine reduces to the fifth-order one when its second
/// `q`-axis winding is made redundant.
///
/// Take `x'_q` down to `x''_q` and `T'_q0` to `T''_q0`, and the round-rotor
/// model's `q` axis has one effective winding rather than two: `b_q` goes to
/// zero, `ψ_2q` decouples, and `e''_d` collapses onto `e'_d` with the
/// subtransient time constant. That is precisely the salient-pole model's `q`
/// axis, and the two must then trace the same trajectory.
///
/// This is the same kind of gate that pinned the subtransient conventions in
/// the first place, and it transfers that confidence one model further along.
#[test]
fn the_round_machine_reduces_to_the_salient_one() {
    let trajectory = |salient: bool| {
        let machine: Box<dyn Machine> = if salient {
            Box::new(GenSalient::new(salient_params(), S_BASE, F_NOM).unwrap())
        } else {
            let p = salient_params();
            Box::new(
                GenRound::new(
                    GenRoundParams {
                        h: p.h,
                        d: p.d,
                        ra: p.ra,
                        xd: p.xd,
                        xq: p.xq,
                        xdp: p.xdp,
                        // One effective q winding: the transient reactance sits
                        // a hair above the subtransient one, so b_q ≈ 0.
                        xqp: p.xqpp + 1e-7,
                        xdpp: p.xdpp,
                        xqpp: p.xqpp,
                        xl: p.xl,
                        td0p: p.td0p,
                        tq0p: p.tq0pp,
                        td0pp: p.td0pp,
                        tq0pp: p.tq0pp,
                        mbase: p.mbase,
                    },
                    S_BASE,
                    F_NOM,
                )
                .unwrap(),
            )
        };
        let model: Box<dyn DynamicModel> =
            Box::new(GeneratingUnit::machine_only(machine));
        let (mut system, _) = islanded_with(vec![model], &[0.8], (-0.8, -0.2));
        let opts = options(
            0.002,
            8.0,
            vec![
                Event::bolted_fault(1.0, 1),
                Event::new(1.08, EventKind::ClearFault { bus: 1 }),
            ],
        );
        let report = run_dynamics(&mut system, &opts);
        assert_eq!(report.status, DynamicsStatus::Completed);
        report.trajectory.series("G1.delta").unwrap()
    };

    let (round, salient) = (trajectory(false), trajectory(true));
    assert_eq!(round.len(), salient.len());
    let worst = round.iter().zip(salient.iter()).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst < 1e-5,
        "with one effective q winding the two orders should agree; \
         the rotor angles differ by {worst:e} rad"
    );
}
