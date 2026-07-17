use gridoxide::network::{build_ybus, linear_initial_guess, power_injections};
use gridoxide::solver::{newton_raphson, newton_raphson_enforcing_q_limits, JacobianBackend, SolveStatus};
use gridoxide::types::{Bus, BusType, Line};

/// The same 3-bus slack/PV/PQ network as `tests/powerflow_test.rs`, except
/// bus 1's `q_min` is tightened to -0.25 (below the -0.317 pu it actually
/// needs to hold 1.04 pu unconstrained — confirmed by
/// `test_unconstrained_violates_q_min` below) so it's guaranteed to trigger
/// PV->PQ switching.
fn three_bus_tight_q_min() -> (Vec<Bus>, Vec<Line>) {
    let buses = vec![
        Bus {
            idx: 0, bus_type: BusType::Slack, voltage_mag: 1.06, voltage_ang: 0.0,
            p_spec: 0.0, q_spec: 0.0, q_min: -999.0, q_max: 999.0, u_rated: 1.0, zip_terms: vec![],
        },
        Bus {
            idx: 1, bus_type: BusType::PV, voltage_mag: 1.04, voltage_ang: 0.0,
            p_spec: 0.5, q_spec: 0.0, q_min: -0.25, q_max: 0.5, u_rated: 1.0, zip_terms: vec![],
        },
        Bus {
            idx: 2, bus_type: BusType::PQ, voltage_mag: 1.0, voltage_ang: 0.0,
            p_spec: -0.6, q_spec: -0.25, q_min: -999.0, q_max: 999.0, u_rated: 1.0, zip_terms: vec![],
        },
    ];
    let lines = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.03 },
        Line { from: 0, to: 2, r: 0.08, x: 0.24, b_shunt: 0.025 },
        Line { from: 1, to: 2, r: 0.06, x: 0.18, b_shunt: 0.02 },
    ];
    (buses, lines)
}

#[test]
fn test_unconstrained_violates_q_min() {
    let (mut buses, lines) = three_bus_tight_q_min();
    let ybus = build_ybus(3, &lines, &[]).finish();
    linear_initial_guess(&mut buses, &ybus);
    newton_raphson(&mut buses, &ybus, 1e-6, 20);

    let (_, q_calc) = power_injections(&buses, &ybus);
    assert_eq!(buses[1].bus_type, BusType::PV, "plain newton_raphson never switches bus types");
    assert!(
        q_calc[1] < buses[1].q_min,
        "fixture assumption broken: bus 1's unconstrained Q ({}) should violate q_min ({})",
        q_calc[1], buses[1].q_min
    );
}

#[test]
fn test_q_limit_enforcement_switches_pv_to_pq() {
    for backend in [
        JacobianBackend::Scalar,
        JacobianBackend::Block,
        #[cfg(feature = "klu")]
        JacobianBackend::Klu,
    ] {
        let (mut buses, lines) = three_bus_tight_q_min();
        let ybus = build_ybus(3, &lines, &[]).finish();
        linear_initial_guess(&mut buses, &ybus);
        let status = newton_raphson_enforcing_q_limits(&mut buses, &ybus, 1e-6, 20, backend, 10);

        assert_eq!(status, SolveStatus::Converged, "backend {backend:?}");
        assert_eq!(buses[1].bus_type, BusType::PQ, "bus 1 should switch PV -> PQ, backend {backend:?}");
        assert!((buses[1].q_spec - (-0.25)).abs() < 1e-12, "q_spec should be pinned at q_min, backend {backend:?}");

        let (p_calc, q_calc) = power_injections(&buses, &ybus);
        // P is unaffected by the switch (same P-mismatch equation whether PV or PQ).
        assert!((p_calc[1] - 0.5).abs() < 1e-6, "backend {backend:?}");
        // Q should now sit exactly at the clamped limit.
        assert!((q_calc[1] - (-0.25)).abs() < 1e-6, "backend {backend:?}");
        // Values pinned from a known-good run (all three backends agree exactly).
        assert!((buses[1].voltage_mag - 1.043469).abs() < 1e-5, "backend {backend:?}");
        assert!((buses[1].voltage_ang - 0.013207).abs() < 1e-5, "backend {backend:?}");
        assert!((buses[2].voltage_mag - 1.005454).abs() < 1e-5, "backend {backend:?}");
        assert!((buses[2].voltage_ang - (-0.043577)).abs() < 1e-5, "backend {backend:?}");
    }
}

#[test]
fn test_q_limit_enforcement_no_violation_matches_unconstrained() {
    // Same network as tests/powerflow_test.rs (q_min = -0.5, well within what
    // bus 1 actually needs) — enforcement shouldn't switch anything or change
    // the converged answer at all.
    let mut buses = vec![
        Bus {
            idx: 0, bus_type: BusType::Slack, voltage_mag: 1.06, voltage_ang: 0.0,
            p_spec: 0.0, q_spec: 0.0, q_min: -999.0, q_max: 999.0, u_rated: 1.0, zip_terms: vec![],
        },
        Bus {
            idx: 1, bus_type: BusType::PV, voltage_mag: 1.04, voltage_ang: 0.0,
            p_spec: 0.5, q_spec: 0.0, q_min: -0.5, q_max: 0.5, u_rated: 1.0, zip_terms: vec![],
        },
        Bus {
            idx: 2, bus_type: BusType::PQ, voltage_mag: 1.0, voltage_ang: 0.0,
            p_spec: -0.6, q_spec: -0.25, q_min: -999.0, q_max: 999.0, u_rated: 1.0, zip_terms: vec![],
        },
    ];
    let lines = vec![
        Line { from: 0, to: 1, r: 0.02, x: 0.06, b_shunt: 0.03 },
        Line { from: 0, to: 2, r: 0.08, x: 0.24, b_shunt: 0.025 },
        Line { from: 1, to: 2, r: 0.06, x: 0.18, b_shunt: 0.02 },
    ];
    let ybus = build_ybus(3, &lines, &[]).finish();
    linear_initial_guess(&mut buses, &ybus);
    let status = newton_raphson_enforcing_q_limits(&mut buses, &ybus, 1e-6, 20, JacobianBackend::Scalar, 10);

    assert_eq!(status, SolveStatus::Converged);
    assert_eq!(buses[1].bus_type, BusType::PV, "no violation, should stay PV");
    // Same expected voltages as tests/powerflow_test.rs's unconstrained run.
    assert!((buses[1].voltage_mag - 1.04).abs() < 1e-5);
    assert!((buses[1].voltage_ang - 0.014349).abs() < 1e-5);
    assert!((buses[2].voltage_mag - 1.003358).abs() < 1e-5);
    assert!((buses[2].voltage_ang - (-0.043141)).abs() < 1e-5);
}
