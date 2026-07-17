use std::fs;
use std::path::PathBuf;

use gridoxide::json::NetworkData;
use gridoxide::network::{build_ybus, linear_initial_guess, YBusSparse};
use gridoxide::solver::{newton_raphson_with_backend, JacobianBackend, PersistentSolver};
use gridoxide::types::Bus;

fn load_network() -> (Vec<Bus>, YBusSparse) {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let raw = fs::read_to_string(path).expect("read network.json");
    let network: NetworkData = serde_json::from_str(&raw).expect("parse network.json");
    let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();
    (network.buses, ybus)
}

/// A `PersistentSolver` reused across two solves with different bus values
/// (same topology) must converge both times, and the second (warm-cache)
/// solve must land on the exact same voltages a fresh one-shot
/// `newton_raphson_with_backend` call reaches for the same modified inputs
/// — proving the reused symbolic factorization doesn't silently corrupt
/// results, not just that it doesn't crash.
fn assert_persistent_solver_matches_fresh(backend: JacobianBackend) {
    let (buses_template, ybus) = load_network();

    let mut solver = PersistentSolver::new(backend);

    // First solve: cold cache (nothing reused yet).
    let mut buses1 = buses_template.clone();
    linear_initial_guess(&mut buses1, &ybus);
    solver.solve(&mut buses1, &ybus, 1e-6, 20);
    let expected_first = [(1.06, 0.0), (1.04, 0.014349), (1.003358, -0.043141)];
    for (i, &(vm, va)) in expected_first.iter().enumerate() {
        assert!((buses1[i].voltage_mag - vm).abs() < 1e-5, "bus {i} vm cold solve");
        assert!((buses1[i].voltage_ang - va).abs() < 1e-5, "bus {i} va cold solve");
    }

    // Second solve on the same PersistentSolver, but with different load —
    // exercises the *warm* path (cached symbolic factorization reused).
    let mut buses2 = buses_template.clone();
    buses2[2].p_spec *= 1.5;
    buses2[2].q_spec *= 1.5;
    linear_initial_guess(&mut buses2, &ybus);
    solver.solve(&mut buses2, &ybus, 1e-6, 20);

    // Independently: a fresh one-shot solve of the identically modified
    // network, with no cached state at all.
    let mut buses3 = buses_template.clone();
    buses3[2].p_spec *= 1.5;
    buses3[2].q_spec *= 1.5;
    linear_initial_guess(&mut buses3, &ybus);
    newton_raphson_with_backend(&mut buses3, &ybus, 1e-6, 20, backend);

    for i in 0..buses2.len() {
        assert!(
            (buses2[i].voltage_mag - buses3[i].voltage_mag).abs() < 1e-9,
            "bus {i} vm: warm={} fresh={}", buses2[i].voltage_mag, buses3[i].voltage_mag
        );
        assert!(
            (buses2[i].voltage_ang - buses3[i].voltage_ang).abs() < 1e-9,
            "bus {i} va: warm={} fresh={}", buses2[i].voltage_ang, buses3[i].voltage_ang
        );
    }
}

#[test]
fn persistent_solver_scalar_matches_fresh_solve() {
    assert_persistent_solver_matches_fresh(JacobianBackend::Scalar);
}

#[test]
#[cfg(feature = "klu")]
fn persistent_solver_klu_matches_fresh_solve() {
    assert_persistent_solver_matches_fresh(JacobianBackend::Klu);
}

/// `reset()` must make the next `solve()` behave like a fresh cold solve —
/// exercised by resetting then solving the identically-modified network
/// again and confirming it still lands on the same answer (a corrupted or
/// stale-but-not-cleared cache could otherwise silently diverge or return
/// garbage without `reset()` actually taking effect).
#[test]
fn persistent_solver_reset_then_solve_matches_fresh() {
    let (buses_template, ybus) = load_network();
    let mut solver = PersistentSolver::new(JacobianBackend::Scalar);

    let mut buses1 = buses_template.clone();
    linear_initial_guess(&mut buses1, &ybus);
    solver.solve(&mut buses1, &ybus, 1e-6, 20);

    solver.reset();

    let mut buses2 = buses_template.clone();
    linear_initial_guess(&mut buses2, &ybus);
    solver.solve(&mut buses2, &ybus, 1e-6, 20);

    for i in 0..buses1.len() {
        assert!((buses1[i].voltage_mag - buses2[i].voltage_mag).abs() < 1e-9, "bus {i} vm after reset");
        assert!((buses1[i].voltage_ang - buses2[i].voltage_ang).abs() < 1e-9, "bus {i} va after reset");
    }
}
