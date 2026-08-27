//! Q-V curves and reactive margins.
//!
//! The headline gate is analytic, as it is for the P-V side: on two buses the
//! whole curve is closed-form, so the solver is checked against arithmetic
//! rather than against itself.
//!
//! **What is deliberately not asserted is that Q-V and P-V agree on the weakest
//! bus.** That seemed like the obvious cross-check and it is false — measured on
//! `case14`, continuation ranks [4, 3, 8, 9, 6] and Q-V ranks [7, 13, 5, 9, 11],
//! sharing one bus of five. Re-running the Q-V sweep at the nose loading, on the
//! theory that the two were describing different operating points, makes the
//! agreement *worse* rather than better. The two measure related but distinct
//! things — a system-wide collapse mode along one loading direction, against one
//! bus's local reactive headroom — which is why utilities run both rather than
//! either. A test asserting they agree would have been asserting something
//! untrue about the physics.

use gridoxide::network::{build_ybus, power_injections};
use gridoxide::qv::{qv_curve, QvError, QvOptions, QvStatus};
use gridoxide::solver::PowerFlowOptions;
use gridoxide::types::{Bus, BusType, Line};

fn bus(idx: usize, bus_type: BusType, p: f64, q: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: q,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: Vec::new(),
    }
}

/// Slack at `1∠0`, one lossless line, one constant-power load.
fn two_bus(x: f64, p: f64, q: f64) -> (Vec<Bus>, Vec<Line>) {
    (
        vec![bus(0, BusType::Slack, 0.0, 0.0), bus(1, BusType::PQ, p, q)],
        vec![Line { from: 0, to: 1, r: 0.0, x, b_shunt: 0.0, g_shunt: 0.0 }],
    )
}

fn options(step: f64) -> QvOptions {
    QvOptions {
        pf: PowerFlowOptions { max_iter: 60, ..Default::default() },
        v_max: 1.20,
        v_min: 0.20,
        step,
        refine: true,
    }
}

/// The closed-form Q-V nose of [`two_bus`], with `E = 1` and no resistance.
///
/// Eliminating the angle from the two injection equations gives
/// `u² − (2a + E²)u + (a² + b²) = 0` with `a = xQ`, `b = xP`, `u = |V|²`. Read
/// as a quadratic in `Q` instead, the operating branch is
/// `Q = [u − √(E²u − x²P²)] / x`, and `dQ/du = 0` gives the minimum:
///
/// ```text
/// u_nose = (E⁴ + 4x²P²) / (4E²)      Q_nose = (4x²P² − E⁴) / (4E²x)
/// ```
fn analytic_nose(x: f64, p: f64) -> (f64, f64) {
    let u = (1.0 + 4.0 * x * x * p * p) / 4.0;
    (u.sqrt(), (4.0 * x * x * p * p - 1.0) / (4.0 * x))
}

/// **The gate.** The nose against arithmetic, over a range of line reactances
/// and loadings.
#[test]
fn the_two_bus_nose_matches_the_closed_form() {
    for (x, p) in [(0.10, -1.0), (0.08, -0.8), (0.25, -0.4), (0.05, -1.5)] {
        let (buses, lines) = two_bus(x, p, 0.0);
        let curve = qv_curve(&buses, &lines, &[], &[], 1, options(0.005));
        assert_eq!(curve.status, QvStatus::NoseFound, "x={x} P={p}");
        let nose = curve.nose.as_ref().expect("a nose was found");

        let (v_expected, q_expected) = analytic_nose(x, p);
        assert!(
            (nose.q - q_expected).abs() < 1e-5,
            "x={x} P={p}: nose Q {} against analytic {q_expected}",
            nose.q
        );
        // Held looser than Q, and the asymmetry is the physics rather than
        // slack: at a minimum dQ/d|V| vanishes, so a given error in Q
        // corresponds to a much larger one in the voltage it occurs at.
        assert!(
            (nose.voltage - v_expected).abs() < 1e-3,
            "x={x} P={p}: nose |V| {} against analytic {v_expected}",
            nose.voltage
        );
    }
}

/// Interpolating the minimum is worth doing: it recovers a nose the sampling
/// grid cannot represent.
///
/// Without it the answer is pinned to the nearest sample, so a 0.02 grid can be
/// out by up to 0.01 in voltage by construction. The parabola through the three
/// bracketing samples is exact where the curve is locally quadratic, which it is
/// at a smooth minimum.
#[test]
fn refining_the_minimum_beats_the_nearest_sample() {
    let (x, p) = (0.10, -1.0);
    let (buses, lines) = two_bus(x, p, 0.0);
    let (v_expected, _) = analytic_nose(x, p);

    let coarse = QvOptions { step: 0.02, ..options(0.02) };
    let raw = qv_curve(&buses, &lines, &[], &[], 1, QvOptions { refine: false, ..coarse.clone() });
    let refined = qv_curve(&buses, &lines, &[], &[], 1, coarse);

    let raw_err = (raw.nose.as_ref().unwrap().voltage - v_expected).abs();
    let refined_err = (refined.nose.as_ref().unwrap().voltage - v_expected).abs();
    assert!(!raw.nose.as_ref().unwrap().refined);
    assert!(refined.nose.as_ref().unwrap().refined);
    assert!(
        refined_err < raw_err / 10.0,
        "refining should be an order of magnitude better than the nearest sample, \
         got {refined_err:.2e} against {raw_err:.2e}"
    );
}

/// Every point on the curve is a real power flow at its own setpoint.
///
/// Rebuilt independently — the bus retyped, the network re-solved through the
/// ordinary entry point, the mismatch taken with `power_injections` — so an
/// error in what the sweep reports as "the reactive power needed" cannot hide
/// behind a plausible-looking curve.
#[test]
fn every_point_is_a_power_flow_at_its_own_setpoint() {
    let (buses, lines) = two_bus(0.08, -0.8, -0.2);
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();
    let curve = qv_curve(&buses, &lines, &[], &[], 1, options(0.02));
    assert!(curve.points.len() > 20);

    for p in &curve.points {
        let mut state = buses.clone();
        state[1].bus_type = BusType::PV;
        state[1].voltage_mag = p.voltage;
        state[1].q_min = f64::NEG_INFINITY;
        state[1].q_max = f64::INFINITY;
        let report = gridoxide::run_power_flow(
            state,
            &lines,
            &[],
            &[],
            gridoxide::TapData::none(),
            PowerFlowOptions { max_iter: 60, ..Default::default() },
        );
        assert_eq!(report.stats.status, gridoxide::solver::SolveStatus::Converged);

        // The condenser supplies the bus's net injection less the bus's own.
        let (_, q_calc) = power_injections(&report.buses, &ybus);
        let needed = q_calc[1] - buses[1].q_spec;
        assert!(
            (needed - p.q).abs() < 1e-6,
            "at |V| {}: the curve says {} but an independent solve says {needed}",
            p.voltage,
            p.q
        );
        assert!(
            (report.buses[1].voltage_mag - p.voltage).abs() < 1e-9,
            "the setpoint must be held exactly"
        );
    }
}

/// The curve passes through zero at the voltage the bus already sits at.
///
/// A condenser that holds a bus where the network was going to put it anyway
/// has nothing to do. This is the one point on the curve whose answer is known
/// without solving anything, which makes it a free check that the sweep is
/// measuring what it claims to.
#[test]
fn the_curve_crosses_zero_at_the_buses_own_voltage() {
    let (buses, lines) = two_bus(0.08, -0.8, -0.2);
    let curve = qv_curve(&buses, &lines, &[], &[], 1, options(0.005));
    assert!(curve.base_voltage.is_finite());

    // The two samples straddling zero must straddle the base voltage too.
    let mut crossing = None;
    for w in curve.points.windows(2) {
        if w[0].q.signum() != w[1].q.signum() {
            crossing = Some((w[0].voltage.max(w[1].voltage), w[0].voltage.min(w[1].voltage)));
            break;
        }
    }
    let (hi, lo) = crossing.expect("the curve should cross zero");
    assert!(
        curve.base_voltage <= hi + 1e-9 && curve.base_voltage >= lo - 1e-9,
        "zero crossing is between {lo} and {hi}, but the bus solves at {}",
        curve.base_voltage
    );
}

/// Above its own voltage the bus has to be supported; below it, held down.
#[test]
fn the_sign_of_q_says_which_way_the_bus_is_being_pushed() {
    let (buses, lines) = two_bus(0.08, -0.8, -0.2);
    let curve = qv_curve(&buses, &lines, &[], &[], 1, options(0.01));
    let base = curve.base_voltage;
    for p in &curve.points {
        if p.voltage > base + 0.02 {
            assert!(p.q > 0.0, "holding {} above {base} should need support, got {}", p.voltage, p.q);
        }
        if p.voltage < base - 0.02 && p.voltage > 0.7 {
            assert!(p.q < 0.0, "holding {} below {base} should need absorption, got {}", p.voltage, p.q);
        }
    }
}

/// A weaker connection has less margin. The direction is the physics; the
/// magnitude is whatever the network says.
#[test]
fn a_weaker_line_leaves_a_smaller_margin() {
    let mut previous = f64::INFINITY;
    for x in [0.05, 0.10, 0.20, 0.40] {
        let (buses, lines) = two_bus(x, -0.5, 0.0);
        let curve = qv_curve(&buses, &lines, &[], &[], 1, options(0.005));
        let margin = curve.margin_pu().expect("a nose was found");
        assert!(
            margin < previous,
            "reactance {x} gave margin {margin}, not less than the previous {previous}"
        );
        previous = margin;
    }
}

/// A sweep that stops before the curve turns reports a bound, not a margin.
///
/// The distinction matters more here than it looks: the margin is the headline
/// number, and calling the last sample of a still-falling curve "the margin"
/// would overstate a bus's weakness in exactly the direction that misleads.
#[test]
fn a_sweep_that_stops_too_early_says_so() {
    let (buses, lines) = two_bus(0.10, -1.0, 0.0);
    // The nose is near 0.51; stop well above it.
    let curve = qv_curve(&buses, &lines, &[], &[], 1, QvOptions { v_min: 0.80, ..options(0.01) });
    assert_eq!(curve.status, QvStatus::NoseNotReached);
    let bound = curve.nose.as_ref().expect("a lowest sample is still reported");
    assert!(!bound.refined, "a bound is not an interpolated minimum");

    let full = qv_curve(&buses, &lines, &[], &[], 1, options(0.01));
    assert_eq!(full.status, QvStatus::NoseFound);
    assert!(
        full.margin_pu().unwrap() > bound.margin_pu,
        "the truncated sweep should under-report the margin"
    );
}

/// Requests that cannot mean anything are refused rather than approximated.
#[test]
fn impossible_requests_are_refused() {
    let (buses, lines) = two_bus(0.10, -1.0, 0.0);

    // A slack bus already fixes its magnitude and already has a free reactive
    // injection: there is no condenser to add.
    let slack = qv_curve(&buses, &lines, &[], &[], 0, options(0.01));
    assert_eq!(slack.status, QvStatus::Rejected(QvError::SlackBus(0)));

    let missing = qv_curve(&buses, &lines, &[], &[], 9, options(0.01));
    assert_eq!(
        missing.status,
        QvStatus::Rejected(QvError::BusOutOfRange { bus: 9, n: 2 })
    );

    let backwards =
        qv_curve(&buses, &lines, &[], &[], 1, QvOptions { v_min: 1.2, v_max: 0.8, ..options(0.01) });
    assert_eq!(backwards.status, QvStatus::Rejected(QvError::EmptyRange));
}

/// On a network with real structure, every bus that reports a nose reports a
/// coherent one, and the ranking is stable under a finer sweep.
#[test]
fn case14_margins_are_coherent_and_grid_independent() {
    use gridoxide::pgm::{pgm_to_buses_and_branches, PgmInput};

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf/pglib_opf_case14_ieee.json");
    let raw = std::fs::read_to_string(&path).expect("committed pglib fixture");
    let input: PgmInput = serde_json::from_str(&raw).expect("parse");
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 100e6, 50.0);

    let rank = |step: f64| -> Vec<usize> {
        let mut rows: Vec<(f64, usize)> = Vec::new();
        for b in 0..buses.len() {
            if buses[b].bus_type == BusType::Slack {
                continue;
            }
            let c = qv_curve(
                &buses,
                &lines,
                &transformers,
                &[],
                b,
                QvOptions { v_min: 0.20, step, ..options(step) },
            );
            if c.status == QvStatus::NoseFound {
                let n = c.nose.as_ref().unwrap();
                assert!(n.margin_pu > 0.0, "bus {b}: a margin should be positive, got {}", n.margin_pu);
                assert!(
                    n.voltage > 0.2 && n.voltage < 1.2,
                    "bus {b}: nose at {} is outside the swept range",
                    n.voltage
                );
                rows.push((n.margin_pu, b));
            }
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        rows.into_iter().map(|r| r.1).collect()
    };

    let coarse = rank(0.02);
    let fine = rank(0.005);
    assert!(coarse.len() >= 10, "most buses should yield a curve, got {}", coarse.len());
    assert_eq!(
        &coarse[..3],
        &fine[..3],
        "the three weakest buses should not depend on the sampling grid"
    );
}
