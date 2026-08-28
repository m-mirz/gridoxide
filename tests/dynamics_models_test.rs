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
use gridoxide::dynamics::models::gov::{Tgov1, Tgov1Params};
use gridoxide::dynamics::models::machine::{GenTransient, GenTransientParams, Machine};
use gridoxide::dynamics::models::load::ZipLoad;
use gridoxide::dynamics::models::pss::{Stab1, Stab1Params};
use gridoxide::dynamics::models::{
    finite_difference, Control, DynamicModel, GeneratingUnit, ModelJacobian,
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
    Sexs::new(SexsParams { k: 200.0, ta: 0.1, tb: 1.0, te: 0.05 }).unwrap()
}

fn new_gov(r: f64) -> Tgov1 {
    Tgov1::new(Tgov1Params { r, t1: 0.5, t2: 1.0, t3: 5.0, dt: 0.0 }).unwrap()
}

fn new_pss() -> Stab1 {
    Stab1::new(Stab1Params { k: 5.0, tw: 10.0, t1: 0.15, t2: 0.03, t3: 0.15, t4: 0.03 }).unwrap()
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
    let n_gen = fits.len();
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
    for (i, (fit, damping, droop)) in fits.iter().enumerate() {
        let id = format!("G{}", i + 1);
        ids.push(id.clone());
        devices.push(DeviceSpec {
            id,
            bus: i,
            s: Complex::new(p_calc[i], q_calc[i]),
            model: Box::new(unit(*fit, *damping, *droop)),
        });
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
