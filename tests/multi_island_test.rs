use gridoxide::network::{build_ybus, linear_initial_guess};
use gridoxide::solver::{IslandStatus, JacobianBackend, PersistentSolver};
use gridoxide::types::{Bus, BusType, Line};
use gridoxide::run_power_flow_analysis_from_ybus;

/// A minimal 2-bus slack+PQ pair, given as its own `(from, to)` bus-index
/// pair so callers can place it anywhere in a larger multi-island fixture.
fn slack_pq_pair(from: usize, to: usize, p_spec: f64, q_spec: f64) -> (Vec<Bus>, Line) {
    let buses = vec![
        Bus {
            idx: from, bus_type: BusType::Slack, voltage_mag: 1.0, voltage_ang: 0.0,
            p_spec: 0.0, q_spec: 0.0, q_min: -f64::INFINITY, q_max: f64::INFINITY, u_rated: 1.0, zip_terms: vec![],
        },
        Bus {
            idx: to, bus_type: BusType::PQ, voltage_mag: 1.0, voltage_ang: 0.0,
            p_spec, q_spec, q_min: -f64::INFINITY, q_max: f64::INFINITY, u_rated: 1.0, zip_terms: vec![],
        },
    ];
    (buses, Line { from, to, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 })
}

fn pq_bus(idx: usize, p_spec: f64, q_spec: f64) -> Bus {
    Bus {
        idx, bus_type: BusType::PQ, voltage_mag: 1.0, voltage_ang: 0.0,
        p_spec, q_spec, q_min: -f64::INFINITY, q_max: f64::INFINITY, u_rated: 1.0, zip_terms: vec![],
    }
}

fn slack_bus(idx: usize) -> Bus {
    Bus {
        idx, bus_type: BusType::Slack, voltage_mag: 1.0, voltage_ang: 0.0,
        p_spec: 0.0, q_spec: 0.0, q_min: -f64::INFINITY, q_max: f64::INFINITY, u_rated: 1.0, zip_terms: vec![],
    }
}

/// Two independent, well-formed slack+PQ islands solved in one shared call.
/// Cross-checks against each island solved entirely on its own (a separate
/// `run_power_flow_analysis_from_ybus` call, single component) to prove the
/// shared multi-island solve reaches the *same* numeric answer, not just
/// "some" plausible one — the mathematical claim this whole feature rests
/// on (disconnected components have no Jacobian coupling, so one combined
/// Newton-Raphson call is equivalent, iteration for iteration, to solving
/// each independently).
#[test]
fn two_well_formed_islands_match_independent_single_island_solves() {
    let (a, line_a) = slack_pq_pair(0, 1, -0.1, -0.05);
    let a_alone = run_power_flow_analysis_from_ybus(a.clone(), build_ybus(2, &[line_a.clone()], &[]));
    assert_eq!(a_alone.islands.len(), 1);
    assert_eq!(a_alone.islands[0].status, IslandStatus::Converged);

    let (b, line_b) = slack_pq_pair(0, 1, -0.2, -0.02);
    let b_alone = run_power_flow_analysis_from_ybus(b.clone(), build_ybus(2, &[line_b.clone()], &[]));
    assert_eq!(b_alone.islands.len(), 1);
    assert_eq!(b_alone.islands[0].status, IslandStatus::Converged);

    let (mut buses, line_a2) = slack_pq_pair(0, 1, -0.1, -0.05);
    let (buses_b, line_b2) = slack_pq_pair(2, 3, -0.2, -0.02);
    buses.extend(buses_b);
    let report = run_power_flow_analysis_from_ybus(buses, build_ybus(4, &[line_a2, line_b2], &[]));

    assert_eq!(report.islands.len(), 2);
    assert!(report.islands.iter().all(|isl| isl.status == IslandStatus::Converged));

    assert!((report.buses[1].voltage_mag - a_alone.buses[1].voltage_mag).abs() < 1e-9);
    assert!((report.buses[1].voltage_ang - a_alone.buses[1].voltage_ang).abs() < 1e-9);
    assert!((report.buses[3].voltage_mag - b_alone.buses[1].voltage_mag).abs() < 1e-9);
    assert!((report.buses[3].voltage_ang - b_alone.buses[1].voltage_ang).abs() < 1e-9);
}

/// The literal repro of the bug this feature fixes: a well-formed island
/// alongside one with no `Slack` bus at all. Before this feature, the
/// no-slack island's structurally singular Jacobian rows made the *entire*
/// solve fail; now the well-formed island still converges normally, and the
/// no-slack one is reported `NoReferenceBus` with placeholder zero values
/// instead of a fabricated reference.
#[test]
fn no_slack_island_reports_placeholder_without_poisoning_the_other_island() {
    let (mut buses, line_a) = slack_pq_pair(0, 1, -0.1, -0.05);
    buses.push(pq_bus(2, -0.2, -0.05));
    buses.push(pq_bus(3, -0.1, -0.02));
    let line_b = Line { from: 2, to: 3, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 };

    let report = run_power_flow_analysis_from_ybus(buses, build_ybus(4, &[line_a, line_b], &[]));
    assert_eq!(report.islands.len(), 2);

    assert_eq!(report.islands[0].bus_indices, vec![0, 1]);
    assert_eq!(report.islands[0].status, IslandStatus::Converged);
    assert!(report.buses[1].voltage_mag > 0.9 && report.buses[1].voltage_mag < 1.0);

    assert_eq!(report.islands[1].bus_indices, vec![2, 3]);
    assert_eq!(report.islands[1].status, IslandStatus::NoReferenceBus);
    assert!(report.islands[1].slack_indices.is_empty());
    assert_eq!(report.buses[2].voltage_mag, 0.0);
    assert_eq!(report.buses[3].voltage_mag, 0.0);
}

/// An island with *two* `Slack` buses (over-determined, not necessarily
/// numerically singular) is reported `AmbiguousReferenceBus` rather than
/// silently picking one arbitrarily — and, just as importantly, doesn't
/// drag a separate well-formed island's own convergence down with it.
#[test]
fn ambiguous_island_does_not_poison_a_well_formed_island_either() {
    let mut buses = vec![slack_bus(0), slack_bus(1), pq_bus(2, -0.05, -0.02)];
    let lines = vec![
        Line { from: 0, to: 2, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 },
        Line { from: 1, to: 2, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 },
    ];
    let (well_formed, line_c) = slack_pq_pair(3, 4, -0.1, -0.05);
    buses.extend(well_formed);
    let all_lines = [lines, vec![line_c]].concat();

    let report = run_power_flow_analysis_from_ybus(buses, build_ybus(5, &all_lines, &[]));
    assert_eq!(report.islands.len(), 2);

    assert_eq!(report.islands[0].bus_indices, vec![0, 1, 2]);
    assert_eq!(report.islands[0].status, IslandStatus::AmbiguousReferenceBus);
    assert_eq!(report.islands[0].slack_indices, vec![0, 1]);

    assert_eq!(report.islands[1].bus_indices, vec![3, 4]);
    assert_eq!(report.islands[1].status, IslandStatus::Converged);
}

/// All three non-trivial outcomes (`Converged`, `NoReferenceBus`,
/// `AmbiguousReferenceBus`) at once, in one call — a smoke test that they
/// coexist correctly, and that `islands` comes back in
/// `connected_components`'s bus-index-ascending discovery order (documented
/// since callers will likely index into it by position).
#[test]
fn three_islands_at_once_report_all_three_statuses_in_ascending_bus_order() {
    let (well_formed, line_a) = slack_pq_pair(0, 1, -0.1, -0.05);
    let mut buses = well_formed;

    buses.push(pq_bus(2, -0.2, -0.05));
    buses.push(pq_bus(3, -0.1, -0.02));
    let line_b = Line { from: 2, to: 3, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 };

    buses.push(slack_bus(4));
    buses.push(slack_bus(5));
    buses.push(pq_bus(6, -0.05, -0.02));
    let line_c1 = Line { from: 4, to: 6, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 };
    let line_c2 = Line { from: 5, to: 6, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 };

    let lines = vec![line_a, line_b, line_c1, line_c2];
    let report = run_power_flow_analysis_from_ybus(buses, build_ybus(7, &lines, &[]));

    assert_eq!(report.islands.len(), 3);
    assert_eq!(report.islands[0].bus_indices, vec![0, 1]);
    assert_eq!(report.islands[0].status, IslandStatus::Converged);
    assert_eq!(report.islands[1].bus_indices, vec![2, 3]);
    assert_eq!(report.islands[1].status, IslandStatus::NoReferenceBus);
    assert_eq!(report.islands[2].bus_indices, vec![4, 5, 6]);
    assert_eq!(report.islands[2].status, IslandStatus::AmbiguousReferenceBus);
}

/// One easy (trivially small loading) island alongside one that's
/// deliberately loaded too heavily to converge within a tiny `max_iter`
/// budget: the easy island's own post-hoc mismatch is independently checked
/// and correctly reported `Converged` even though the hard island alone
/// drags the whole shared solve to `MaxIterationsReached`. Uses
/// `PersistentSolver::solve` directly rather than
/// `run_power_flow_analysis_from_ybus`, since the latter's `tol`/`max_iter`
/// aren't caller-configurable.
#[test]
fn easy_island_converges_independently_of_a_hard_one_under_a_custom_max_iter() {
    let (mut buses, line_easy) = slack_pq_pair(0, 1, -0.01, -0.005); // easy: negligible loading
    buses.push(slack_bus(2));
    buses.push(pq_bus(3, -3.0, -2.5)); // hard: heavily overloaded for this line
    let line_hard = Line { from: 2, to: 3, r: 0.02, x: 0.06, b_shunt: 0.0, g_shunt: 0.0 };

    let ybus = build_ybus(4, &[line_easy, line_hard], &[]).finish();
    linear_initial_guess(&mut buses, &ybus);
    let mut solver = PersistentSolver::new(JacobianBackend::Scalar);
    let islands = solver.solve(&mut buses, &ybus, 1e-6, 2);

    assert_eq!(islands.len(), 2);
    assert_eq!(islands[0].bus_indices, vec![0, 1]);
    assert_eq!(islands[0].status, IslandStatus::Converged, "easy island should have already converged");
    assert_eq!(islands[1].bus_indices, vec![2, 3]);
    assert_eq!(islands[1].status, IslandStatus::MaxIterationsReached, "hard island should still be unconverged after only 2 iterations");
}
