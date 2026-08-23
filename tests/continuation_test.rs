//! Continuation power flow: the curve, the nose, and the tangent.
//!
//! There is no in-tree oracle — none of the six reference implementations under
//! `references/` implements continuation — so the gates here have to be
//! self-supporting. The headline one is analytic: on two buses the solvability
//! boundary is closed-form *including* resistance, so λ_max and the nose voltage
//! are both known exactly and the solver is checked against arithmetic rather
//! than against itself.
//!
//! The second-strongest gate is independence: every point the walk returns is
//! rebuilt from scratch as an ordinary power flow at its own λ and re-checked
//! with `network::power_injections`. That is the house pattern — a fast path
//! against an independently-constructed slow one, as `ac_contingency_test`'s
//! `direct_solve` does.

use gridoxide::continuation::augmented::Parametrization;
use gridoxide::continuation::{
    run_continuation, ContinuationError, ContinuationOptions, ContinuationStatus, CriticalPointKind,
    CurveBranch, LoadingDirection, StopCriterion,
};
use gridoxide::network::{build_ybus, effective_injection, power_injections};
use gridoxide::solver::{JacobianBackend, PowerFlowOptions};
use gridoxide::types::{Bus, BusType, Line};

fn bus(idx: usize, bus_type: BusType, vm: f64, p: f64, q: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: vm,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: q,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: vec![],
    }
}

fn line(from: usize, to: usize, r: f64, x: f64) -> Line {
    Line { from, to, r, x, b_shunt: 0.0, g_shunt: 0.0 }
}

/// Slack at `1∠0`, one line, one constant-power PQ load.
fn two_bus(r: f64, x: f64, p0: f64, q0: f64) -> (Vec<Bus>, Vec<Line>) {
    (
        vec![bus(0, BusType::Slack, 1.0, 0.0, 0.0), bus(1, BusType::PQ, 1.0, p0, q0)],
        vec![line(0, 1, r, x)],
    )
}

/// The closed-form solvability boundary for [`two_bus`].
///
/// With `a = rP + xQ`, `b = xP − rQ` and `u = |V₂|²`, eliminating the angle from
/// the two injection equations gives `u² − (2a + E²)u + (a² + b²) = 0`, whose
/// discriminant is `D = 4aE² + E⁴ − 4b²`. The nose is `D = 0`; substituting
/// `a = μa₀`, `b = μb₀` with `μ = 1 + λ` leaves a quadratic in μ.
///
/// At unity power factor with `r = 0` this reduces to the textbook
/// `P_max = E²/2x`, `|V| = E/√2`.
fn analytic_nose(e: f64, r: f64, x: f64, p0: f64, q0: f64) -> (f64, f64) {
    let a0 = r * p0 + x * q0;
    let b0 = x * p0 - r * q0;
    let mu = e * e * (a0 + (a0 * a0 + b0 * b0).sqrt()) / (2.0 * b0 * b0);
    (mu - 1.0, ((2.0 * mu * a0 + e * e) / 2.0).sqrt())
}

fn options(direction: LoadingDirection) -> ContinuationOptions {
    ContinuationOptions { direction, ..Default::default() }
}

/// **The gate.** λ_max and the nose voltage against arithmetic, over a grid that
/// includes a lossless line, a realistically resistive one, and a capacitive
/// load — so agreement cannot be an artifact of `r = 0` or of a lagging power
/// factor.
///
/// λ is held to 1e-5 and the voltage only to 1e-3. That asymmetry is the
/// physics, not slack: at a fold `dλ/d|V| → 0`, so a given λ error corresponds
/// to a much larger voltage error. Measured, the run does far better than both.
#[test]
fn two_bus_nose_matches_the_closed_form() {
    for (r, x, p0, q0) in [
        (0.0, 0.10, -1.0, 0.0),   // lossless, unity power factor
        (0.0, 0.10, -1.0, -0.3),  // lossless, lagging
        (0.03, 0.10, -1.0, -0.3), // transmission r/x
        (0.05, 0.25, -0.4, -0.13),
        (0.03, 0.10, -1.0, 0.2), // capacitive load
    ] {
        let (buses, lines) = two_bus(r, x, p0, q0);
        let direction = LoadingDirection::scale_loads(&buses);
        let curve = run_continuation(buses, &lines, &[], &[], options(direction));

        assert_eq!(curve.status, ContinuationStatus::NoseReached, "r={r} x={x} q0={q0}");
        let critical = curve.critical.as_ref().expect("a nose was reported");
        assert_eq!(critical.kind, CriticalPointKind::SaddleNode);

        let (lambda_expected, v_expected) = analytic_nose(1.0, r, x, p0, q0);
        assert!(
            (critical.lambda_max - lambda_expected).abs() < 1e-5,
            "r={r} x={x} q0={q0}: lambda_max {} vs analytic {}",
            critical.lambda_max,
            lambda_expected
        );
        assert!(
            (critical.point.voltage_mag[1] - v_expected).abs() < 1e-3,
            "r={r} x={x} q0={q0}: nose voltage {} vs analytic {}",
            critical.point.voltage_mag[1],
            v_expected
        );
        // The definition of the nose, checked directly rather than inferred
        // from λ: the tangent's λ-component vanishes there.
        assert!(
            critical.point.tangent_lambda.abs() < 1e-3,
            "r={r} x={x} q0={q0}: dlambda/dsigma at the nose is {}",
            critical.point.tangent_lambda
        );
    }
}

/// Every point the walk returns must be an ordinary power flow at its own λ.
///
/// Rebuilt independently — injections recomputed from the base spec and the
/// direction, mismatch taken with `power_injections` — so a sign error in the
/// λ column of the augmented Jacobian cannot hide behind a plausible-looking
/// curve.
#[test]
fn every_point_solves_the_power_flow_at_its_own_lambda() {
    let (buses, lines) = two_bus(0.02, 0.08, -0.8, -0.25);
    let direction = LoadingDirection::scale_loads(&buses);
    let base_p: Vec<f64> = buses.iter().map(|b| b.p_spec).collect();
    let base_q: Vec<f64> = buses.iter().map(|b| b.q_spec).collect();
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();

    let curve = run_continuation(buses, &lines, &[], &[], options(direction.clone()));
    assert!(curve.points.len() > 3, "expected a curve, got {} points", curve.points.len());

    for (n, point) in curve.points.iter().enumerate() {
        let rebuilt: Vec<Bus> = (0..point.voltage_mag.len())
            .map(|i| Bus {
                idx: i,
                bus_type: point.bus_type[i],
                voltage_mag: point.voltage_mag[i],
                voltage_ang: point.voltage_ang[i],
                p_spec: base_p[i] + point.lambda * direction.d_p[i],
                q_spec: base_q[i] + point.lambda * direction.d_q[i],
                q_min: f64::NEG_INFINITY,
                q_max: f64::INFINITY,
                u_rated: 1.0,
                zip_terms: vec![],
            })
            .collect();

        let (p_calc, q_calc) = power_injections(&rebuilt, &ybus);
        let mut worst = 0.0f64;
        for b in &rebuilt {
            if b.bus_type == BusType::Slack {
                continue;
            }
            let (p_eff, q_eff) = effective_injection(b);
            worst = worst.max((p_eff - p_calc[b.idx]).abs());
            if b.bus_type == BusType::PQ {
                worst = worst.max((q_eff - q_calc[b.idx]).abs());
            }
        }
        assert!(worst < 1e-6, "point {n} (lambda {}) has mismatch {worst}", point.lambda);
        assert!(
            point.voltage_mag.iter().all(|v| *v > 0.0),
            "point {n} has a non-positive voltage magnitude: {:?}",
            point.voltage_mag
        );
    }
}

/// What λ_max *means*, checked with code that knows nothing about continuation:
/// an ordinary Newton solve converges just below it and fails just above.
#[test]
fn newton_converges_below_the_nose_and_fails_above() {
    let (buses, lines) = two_bus(0.02, 0.08, -0.8, -0.25);
    let direction = LoadingDirection::scale_loads(&buses);
    let base = buses.clone();
    let curve = run_continuation(buses, &lines, &[], &[], options(direction.clone()));
    let lambda_max = curve.lambda_max().expect("a nose was reported");

    let solve_at = |lambda: f64| {
        let scaled: Vec<Bus> = base
            .iter()
            .map(|b| Bus {
                p_spec: b.p_spec + lambda * direction.d_p[b.idx],
                q_spec: b.q_spec + lambda * direction.d_q[b.idx],
                ..b.clone()
            })
            .collect();
        gridoxide::run_power_flow(
            scaled,
            &lines,
            &[],
            &[],
            gridoxide::TapData::none(),
            PowerFlowOptions { max_iter: 60, ..Default::default() },
        )
        .stats
        .status
    };

    assert_eq!(
        solve_at(lambda_max * 0.999),
        gridoxide::solver::SolveStatus::Converged,
        "just below the nose there is a solution and Newton should find it"
    );
    assert_ne!(
        solve_at(lambda_max * 1.01),
        gridoxide::solver::SolveStatus::Converged,
        "past the nose there is no solution at all, so nothing should converge"
    );
}

/// The nose is a property of the network, not of how the walk was configured.
#[test]
fn parametrizations_and_step_sizes_agree_on_the_nose() {
    let (buses, lines) = two_bus(0.02, 0.08, -0.8, -0.25);
    let direction = LoadingDirection::scale_loads(&buses);

    let mut found: Vec<(String, f64)> = Vec::new();
    for param in [Parametrization::PseudoArcLength, Parametrization::Local] {
        for step in [0.02, 0.05, 0.2] {
            let curve = run_continuation(
                buses.clone(),
                &lines,
                &[],
                &[],
                ContinuationOptions {
                    direction: direction.clone(),
                    parametrization: param,
                    step,
                    ..Default::default()
                },
            );
            let lambda = curve
                .lambda_max()
                .unwrap_or_else(|| panic!("{param:?} at step {step} found no nose: {:?}", curve.status));
            found.push((format!("{param:?}/{step}"), lambda));
        }
    }

    let reference = found[0].1;
    for (label, lambda) in &found {
        assert!(
            (lambda - reference).abs() < 1e-4,
            "{label} put the nose at {lambda}, but {} put it at {reference}",
            found[0].0
        );
    }
}

/// Every scalar-triplet backend must agree. `Block` is refused rather than
/// silently substituted: its 2×2-per-bus structure has no home for a scalar λ.
#[test]
fn backends_agree_and_block_is_refused() {
    let (buses, lines) = two_bus(0.02, 0.08, -0.8, -0.25);
    let direction = LoadingDirection::scale_loads(&buses);

    let mut reference: Option<f64> = None;
    for backend in [JacobianBackend::Scalar, JacobianBackend::KluNative] {
        let curve = run_continuation(
            buses.clone(),
            &lines,
            &[],
            &[],
            ContinuationOptions {
                direction: direction.clone(),
                power_flow: PowerFlowOptions { backend, ..Default::default() },
                ..Default::default()
            },
        );
        let lambda = curve.lambda_max().unwrap_or_else(|| panic!("backend {backend:?}: no nose"));
        match reference {
            None => reference = Some(lambda),
            Some(r) => assert!(
                (lambda - r).abs() < 1e-6,
                "backend {backend:?} disagrees: {lambda} vs {r}"
            ),
        }
    }

    let refused = run_continuation(
        buses,
        &lines,
        &[],
        &[],
        ContinuationOptions {
            direction,
            power_flow: PowerFlowOptions {
                backend: JacobianBackend::Block,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    assert_eq!(
        refused.status,
        ContinuationStatus::Rejected(ContinuationError::BackendUnsupported(JacobianBackend::Block))
    );
}

/// Asking for a λ short of the nose must land exactly on it, not near it.
#[test]
fn a_target_lambda_is_landed_on_exactly() {
    let (buses, lines) = two_bus(0.02, 0.08, -0.8, -0.25);
    let direction = LoadingDirection::scale_loads(&buses);
    let curve = run_continuation(
        buses,
        &lines,
        &[],
        &[],
        ContinuationOptions {
            direction,
            stop: StopCriterion { target_lambda: Some(0.4), trace_lower_branch: false },
            ..Default::default()
        },
    );
    assert_eq!(curve.status, ContinuationStatus::TargetReached);
    let last = curve.points.last().expect("points were recorded");
    assert!((last.lambda - 0.4).abs() < 1e-9, "landed at {} instead of 0.4", last.lambda);
}

/// A direction that moves nothing has no curve to trace, and a mis-sized one is
/// a caller mistake. Both are refused rather than approximated.
#[test]
fn a_degenerate_direction_is_refused() {
    let (buses, lines) = two_bus(0.02, 0.08, -0.8, -0.25);

    let empty = run_continuation(
        buses.clone(),
        &lines,
        &[],
        &[],
        options(LoadingDirection::zero(buses.len())),
    );
    assert_eq!(empty.status, ContinuationStatus::Rejected(ContinuationError::EmptyDirection));

    let wrong = run_continuation(
        buses,
        &lines,
        &[],
        &[],
        options(LoadingDirection::explicit(vec![0.0], vec![0.0])),
    );
    assert_eq!(
        wrong.status,
        ContinuationStatus::Rejected(ContinuationError::DirectionLengthMismatch {
            got: 1,
            want: 2
        })
    );
}

/// A voltage-dependent load is refused by default.
///
/// Not fussiness: `jacobian::JacobianPattern` carries no `∂s_eff/∂|V|` term for
/// ZIP loads. An ordinary solve survives that — the mismatch is exact, so only
/// the step direction is off — but here the Jacobian's singularity *is* the
/// answer, so the nose would land in the wrong place while still looking
/// entirely plausible.
#[test]
fn zip_loads_are_refused_unless_explicitly_allowed() {
    let (mut buses, lines) = two_bus(0.02, 0.08, -0.8, -0.25);
    buses[1].zip_terms.push(gridoxide::types::ZipTerm {
        s_const: num_complex::Complex::new(-0.1, -0.02),
        kind: gridoxide::types::ZipKind::ConstImpedance,
    });
    let direction = LoadingDirection::scale_loads(&buses);

    let refused = run_continuation(buses.clone(), &lines, &[], &[], options(direction.clone()));
    assert_eq!(
        refused.status,
        ContinuationStatus::Rejected(ContinuationError::ZipTermsUnsupported { buses: vec![1] })
    );

    let allowed = run_continuation(
        buses,
        &lines,
        &[],
        &[],
        ContinuationOptions { direction, allow_zip: true, ..Default::default() },
    );
    assert!(
        !allowed.warnings.is_empty(),
        "allowing ZIP must still say the tangent is approximate"
    );
}

/// The headline claim about the weakest-bus ranking, checked directly rather
/// than eyeballed: at a fold the reported tangent spans the Jacobian's null
/// space, so `J·t_x` must vanish.
///
/// This is what makes the ranking mean anything. Without it, "the bus with the
/// largest tangent component" is just the largest number in a vector nobody has
/// verified is the collapse mode.
#[test]
fn the_tangent_at_the_nose_is_the_jacobians_null_vector() {
    use gridoxide::jacobian::JacobianPattern;
    use gridoxide::network::power_injections;

    let (buses, lines) = two_bus(0.02, 0.08, -0.8, -0.25);
    let direction = LoadingDirection::scale_loads(&buses);
    let base_p: Vec<f64> = buses.iter().map(|b| b.p_spec).collect();
    let base_q: Vec<f64> = buses.iter().map(|b| b.q_spec).collect();
    let ybus = build_ybus(buses.len(), &lines, &[]).finish();

    let curve = run_continuation(buses, &lines, &[], &[], options(direction.clone()));
    let critical = curve.critical.as_ref().expect("a nose was reported");

    // The state at the nose, rebuilt from the report alone.
    let state: Vec<Bus> = (0..critical.point.voltage_mag.len())
        .map(|i| Bus {
            idx: i,
            bus_type: critical.point.bus_type[i],
            voltage_mag: critical.point.voltage_mag[i],
            voltage_ang: critical.point.voltage_ang[i],
            p_spec: base_p[i] + critical.lambda_max * direction.d_p[i],
            q_spec: base_q[i] + critical.lambda_max * direction.d_q[i],
            q_min: f64::NEG_INFINITY,
            q_max: f64::INFINITY,
            u_rated: 1.0,
            zip_terms: vec![],
        })
        .collect();

    // The Newton unknown layout: non-slack angles in bus order, then PQ
    // magnitudes in bus order.
    let non_slack: Vec<usize> =
        state.iter().filter(|b| b.bus_type != BusType::Slack).map(|b| b.idx).collect();
    let pq: Vec<usize> =
        state.iter().filter(|b| b.bus_type == BusType::PQ).map(|b| b.idx).collect();
    let n_angle = non_slack.len();

    let mut t = vec![0.0; n_angle + pq.len()];
    for (row, &i) in non_slack.iter().enumerate() {
        t[row] = critical.tangent_ang[i];
    }
    for (row, &i) in pq.iter().enumerate() {
        t[n_angle + row] = critical.tangent_vmag[i];
    }

    let (p_calc, q_calc) = power_injections(&state, &ybus);
    let pattern = JacobianPattern::analyze(&state, &ybus);
    let mut values = Vec::new();
    pattern.fill(&state, &p_calc, &q_calc, &mut values);

    let mut jt = vec![0.0; t.len()];
    let mut scale = 0.0f64;
    for (row, col, v) in pattern.to_triplets(&values) {
        jt[row] += v * t[col];
        scale = scale.max(v.abs());
    }

    let residual = jt.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let norm = t.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm > 0.5, "the reported tangent should be a unit vector, got norm {norm}");
    assert!(
        residual < 1e-3 * scale,
        "J·t = {residual} against a largest Jacobian entry of {scale}: the reported \
         collapse mode is not in the null space"
    );

    // On two buses there is only one bus whose voltage can collapse.
    assert_eq!(critical.critical_bus(), Some(1));
}

/// Past the nose the curve does not stop — it turns and comes back as the
/// low-voltage solution branch. Tracing it is the proof that the augmented
/// system really did pass *through* the singularity rather than stopping at it.
#[test]
fn the_lower_branch_is_the_low_voltage_solution() {
    let (buses, lines) = two_bus(0.02, 0.08, -0.8, -0.25);
    let direction = LoadingDirection::scale_loads(&buses);
    let curve = run_continuation(
        buses,
        &lines,
        &[],
        &[],
        ContinuationOptions {
            direction,
            stop: StopCriterion { target_lambda: None, trace_lower_branch: true },
            max_steps: 400,
            ..Default::default()
        },
    );

    assert_eq!(curve.status, ContinuationStatus::TracedToZero);
    let lambda_max = curve.lambda_max().expect("a nose was reported");

    let upper: Vec<_> = curve.points.iter().filter(|p| p.branch == CurveBranch::Upper).collect();
    let lower: Vec<_> = curve.points.iter().filter(|p| p.branch == CurveBranch::Lower).collect();
    assert!(!upper.is_empty() && !lower.is_empty(), "both branches should be walked");

    // No point may exceed the nose, on either branch.
    for p in &curve.points {
        assert!(
            p.lambda <= lambda_max + 1e-6,
            "point at lambda {} is past the reported maximum {lambda_max}",
            p.lambda
        );
    }

    // At a λ both branches reach, the lower one is genuinely the lower-voltage
    // solution — that is what makes it a different operating point rather than
    // the same one walked twice.
    let shared = 0.5 * lambda_max;
    let nearest = |branch: CurveBranch| {
        curve
            .points
            .iter()
            .filter(|p| p.branch == branch)
            .min_by(|a, b| {
                (a.lambda - shared).abs().total_cmp(&(b.lambda - shared).abs())
            })
            .expect("both branches have points")
    };
    let (hi, lo) = (nearest(CurveBranch::Upper), nearest(CurveBranch::Lower));
    assert!(
        lo.voltage_mag[1] < hi.voltage_mag[1] - 0.05,
        "lower branch voltage {} is not below the upper branch's {}",
        lo.voltage_mag[1],
        hi.voltage_mag[1]
    );
    assert!(
        curve.points.iter().all(|p| p.voltage_mag.iter().all(|v| *v > 0.0)),
        "the lower branch is a low-voltage solution, not a negative-voltage one"
    );
}

/// Where the extra generation comes from is part of the scenario, and the
/// pickup constructors have to compose correctly with the rest.
///
/// The first assertion is the load-bearing one. `DistributedSlack` is an outer
/// loop that moves `p_spec` as a function of the *solved* state; that dependence
/// is invisible to `∂g/∂λ`, so running it inside the corrector would corrupt the
/// tangent. Its effect is exactly linear in λ, so it is folded into the
/// direction instead — and the way to know the fold is faithful is that pickup
/// weighted entirely onto the slack reproduces `scale_loads` exactly, since a
/// slack bus has no active-power equation for the share to land in.
#[test]
fn generation_pickup_composes_into_the_direction() {
    let buses = vec![
        bus(0, BusType::Slack, 1.0, 0.0, 0.0),
        bus(1, BusType::PV, 1.0, 0.5, 0.0),
        bus(2, BusType::PQ, 1.0, -0.6, -0.25),
        bus(3, BusType::PQ, 1.0, -0.4, -0.15),
    ];
    let lines = vec![
        line(0, 1, 0.02, 0.06),
        line(1, 2, 0.03, 0.09),
        line(0, 2, 0.04, 0.12),
        line(2, 3, 0.03, 0.10),
    ];

    let plain = LoadingDirection::scale_loads(&buses);
    let on_slack = LoadingDirection::scale_loads_with_pickup(&buses, &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        plain, on_slack,
        "pickup allocated entirely to the slack must reduce to plain load scaling: the \
         slack has no P equation for the share to land in"
    );

    // Shared over the one real generator, the pickup must exactly offset the
    // load increase — otherwise λ moves the system's total balance, which is a
    // different scenario than the one asked for.
    let shared = LoadingDirection::scale_loads_with_pickup(&buses, &[0.0, 1.0, 0.0, 0.0]);
    let net: f64 = shared.d_p.iter().sum();
    assert!(net.abs() < 1e-12, "pickup should balance the load increase exactly, net is {net}");
    assert!(shared.d_p[1] > 0.0, "the generator should pick up, not shed");
    assert_eq!(
        shared.total_active_load_increase(),
        plain.total_active_load_increase(),
        "the margin measures load carried, so pickup must not change it"
    );

    // Both are real scenarios, and the one that holds generation local to the
    // load should carry more of it before collapsing.
    let nose = |d: LoadingDirection| {
        run_continuation(buses.clone(), &lines, &[], &[], options(d))
            .lambda_max()
            .expect("a nose was reported")
    };
    let (slack_nose, shared_nose) = (nose(plain), nose(shared));
    assert!(
        slack_nose > 0.0 && shared_nose > 0.0,
        "both pickup scenarios should have a nose: {slack_nose}, {shared_nose}"
    );
    assert!(
        (slack_nose - shared_nose).abs() > 1e-6,
        "moving where the generation comes from should move the nose, but both gave \
         {slack_nose}"
    );
}

/// λ_max on the committed IEEE fixtures, **recorded rather than targeted**.
///
/// There is no published figure this can be checked against: pglib's
/// `case14_ieee` is not the classic `case14` (its own README says so), the
/// numbers in the literature depend on a loading direction each paper chooses
/// differently, and they depend on how reactive limits are treated. Asserting a
/// band from a paper would be false precision.
///
/// So this pins what the code does today, on a fixture that will not move, and
/// exists to make an unintended change loud. If a deliberate change moves these,
/// update them — but read the diff first.
#[test]
fn ieee_fixtures_have_a_recorded_nose() {
    use gridoxide::pgm::{pgm_to_buses_and_branches, PgmInput};

    for (name, expected_free, expected_limited) in [
        ("pglib_opf_case14_ieee", 2.5361, 0.6018),
        ("pglib_opf_case30_ieee", 1.6945, 0.4664),
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/pglib-opf")
            .join(format!("{name}.json"));
        let raw = std::fs::read_to_string(&path).expect("committed pglib fixture");

        for (enforce, expected) in [(false, expected_free), (true, expected_limited)] {
            let input: PgmInput = serde_json::from_str(&raw).expect("parse");
            let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 100e6, 50.0);
            let direction = LoadingDirection::scale_loads(&buses);
            let curve = run_continuation(
                buses,
                &lines,
                &transformers,
                &[],
                ContinuationOptions {
                    direction,
                    power_flow: PowerFlowOptions {
                        enforce_q_limits: enforce,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            );
            let lambda = curve
                .lambda_max()
                .unwrap_or_else(|| panic!("{name} (q limits {enforce}): {:?}", curve.status));
            assert!(
                (lambda - expected).abs() < 1e-3,
                "{name} (q limits {enforce}): nose moved to {lambda:.4} from the recorded \
                 {expected:.4}"
            );
        }
    }
}

/// The default parametrization is a performance cliff, not a preference, so it
/// is pinned here.
///
/// `PseudoArcLength`'s bordering row *is* the previous tangent, so it is dense.
/// Measured on `case1354pegase` (n = 2449) a dense border row costs 92x a plain
/// Newton solve where a sparse one costs 1.4x — and at 119 buses the same
/// comparison is 1.7x, so no small fixture will ever notice. It reached the CLI
/// and the Python binding once already, by each naming its own default instead
/// of deferring to this one.
#[test]
fn the_default_parametrization_has_a_sparse_border_row() {
    use gridoxide::continuation::augmented::{border_columns, Parametrization};

    let n = 500;
    let cols = border_columns(Parametrization::default(), 17, n);
    assert_eq!(
        cols.len(),
        1,
        "the default parametrization's border row must be a single entry, not {} of them",
        cols.len()
    );
    assert_eq!(Parametrization::default(), Parametrization::Local);

    // And the expensive one really is the dense one, so this test is measuring
    // the thing it claims to.
    assert_eq!(border_columns(Parametrization::PseudoArcLength, 17, n).len(), n + 1);
    assert_eq!(border_columns(Parametrization::Natural, 17, n), vec![n]);
}

/// One symbolic factorization should serve a whole step — the tangent solve,
/// the corrector, and every event-locator trial — because they all pin the same
/// continuation index.
///
/// Without that, the locator re-picks an index per trial (or, worse, switches
/// parametrization) and the pattern changes under the search. That is not a
/// small inefficiency: it was a 36x slowdown on `case1354pegase`, because each
/// change threw away the factorization.
#[test]
fn a_whole_trace_needs_few_reanalyses() {
    use gridoxide::pgm::{pgm_to_buses_and_branches, PgmInput};

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf/pglib_opf_case118_ieee.json");
    let raw = std::fs::read_to_string(&path).expect("committed pglib fixture");
    let input: PgmInput = serde_json::from_str(&raw).expect("parse");
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 100e6, 50.0);
    let direction = LoadingDirection::scale_loads(&buses);

    let curve = run_continuation(
        buses,
        &lines,
        &transformers,
        &[],
        ContinuationOptions {
            direction,
            power_flow: PowerFlowOptions { enforce_q_limits: true, ..Default::default() },
            ..Default::default()
        },
    );

    assert!(curve.solves > 50, "expected a real trace, got {} solves", curve.solves);
    assert!(
        curve.reanalyses * 4 < curve.solves,
        "{} re-analyses against {} solves: the continuation index is churning, so the \
         symbolic factorization is being thrown away instead of reused",
        curve.reanalyses,
        curve.solves
    );
}
