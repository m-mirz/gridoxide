//! State estimation over a CGMES node-breaker network.
//!
//! `plans/NODE_BREAKER_PLAN.md` phase 5. Its stated goal is "a measurement can
//! attach to a breaker"; the real prerequisite, which the plan does not mention,
//! was that `SeNetwork::new` took a `pgm::PgmNetwork`, so no CGMES model of any
//! kind could reach the estimator. `SeNetwork::from_bus_network` closes that,
//! and `NodeBreakerNetwork::se_network` is the bridge.
//!
//! # What is actually different about estimating a node-breaker model
//!
//! Not the measurement model. A retained switch is a branch, so a sensor on a
//! breaker is an ordinary `Target::BranchTerminal` and the estimator never
//! learns that any of its branches is a switch — the same thing that made
//! phases 4 and 6 come free.
//!
//! What differs is the *constraint* set, and it inverts the usual proportions.
//! A bus-branch model has a handful of zero-injection buses. A node-breaker
//! model is mostly zero-injection buses: every internal node of a bay carries
//! switches and nothing else. Under `RetainAdjacentToBusbar`, MiniGrid goes
//! from 15 buses to 45, and all 30 of the added ones inject exactly zero. That
//! is not a nuisance — it is free, exact information, and it is why a
//! node-breaker model can be observable with no more sensors than the
//! bus-branch one needs.
//!
//! Which makes the zero-injection flags load-bearing in a way they never were on
//! PGM data, and is why `cgmes::zero_injection_flags` reads the model's
//! *structure* rather than testing `p_spec`/`q_spec` against zero. A load that
//! happens to sit at zero in one SSH snapshot is not a bus that injects nothing,
//! and a hard equality constraint saying otherwise would bias every estimate
//! around it.
//!
//! # Why every test here starts with `linear_start`
//!
//! `flat_start` does not converge on this data, and the reason is worth
//! recording because it is not the switches. MiniGrid has two de-energized
//! buses — CGMES marks them by leaving them out of every `TopologicalIsland` —
//! and a de-energized bus must *start* at zero, not at 1 p.u. Nothing measures
//! it, so `jacobian::mask_untouched` pins it wherever the start left it, and a
//! bus pinned at 1 p.u. that the true state puts at 0 poisons every measurement
//! that touches it. From flat start the estimate diverges to an objective of
//! 2.2e4; from `linear_start`, which zeroes them, it converges in three
//! iterations to an objective of 4e-10. `linear_start`'s own doc already says
//! to prefer it wherever a `SeNetwork` is at hand; on CGMES data it is not a
//! preference.

use std::path::{Path, PathBuf};

use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use gridoxide::cgmes::{cgmes_node_breaker_to_buses_and_branches, load_profiles};
use gridoxide::measurement::{Measurement, MeasurementKind, Target};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;
use gridoxide::se::constraints::Constraints;
use gridoxide::se::jacobian::StateLayout;
use gridoxide::se::nr::{estimate, SeOptions, SeStatus};
use gridoxide::se::observability;
use gridoxide::solver::SolveStatus;
use gridoxide::switches::{NodeBreakerNetwork, SwitchTreatment};
use gridoxide::topology::RetentionPolicy;
use gridoxide::types::Bus;

fn load_minigrid() -> Option<gridoxide::cgmes::CimDataset> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/MiniGrid/MiniGrid-Merged");
    if !d.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init \
             tests/data/CGMES-Test-Configurations`",
            d.display()
        );
        return None;
    }
    let files: Vec<PathBuf> = ["EQBD", "EQ", "SSH", "TP", "SV"]
        .iter()
        .map(|p| d.join(format!("MiniGrid_{p}.xml")))
        .filter(|p| p.exists())
        .collect();
    let refs: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
    Some(load_profiles(&refs).expect("failed to decode CGMES profiles"))
}

fn node_breaker(policy: RetentionPolicy) -> Option<NodeBreakerNetwork> {
    let ds = load_minigrid()?;
    Some(
        cgmes_node_breaker_to_buses_and_branches(&ds, 100e6, &policy, SwitchTreatment::Regularize)
            .expect("node-breaker conversion failed"),
    )
}

/// The power-flow solution, which every measurement below is generated from.
///
/// Estimating against a state a *different* method produced is the point: if
/// the estimator merely reproduced its own input it would prove nothing.
fn true_state(net: &NodeBreakerNetwork) -> Vec<Bus> {
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    let report = run_power_flow_analysis_from_ybus(net.buses.clone(), ybus);
    assert_eq!(report.stats.status, SolveStatus::Converged, "power flow did not converge");
    report.buses
}

fn ybus_of(net: &NodeBreakerNetwork) -> gridoxide::network::YBusSparse {
    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &net.shunts);
    ybus.finish()
}

/// A sensor set of the kind a real SCADA system provides: voltage magnitude at
/// the energized buses that carry equipment, and P/Q at both ends of every
/// branch that is not a switch.
///
/// Deliberately *nothing* on any bay-internal bus. Those are observable only
/// through the zero-injection constraints, which is exactly the claim under
/// test.
fn scada(net: &NodeBreakerNetwork, state: &[Bus], sigma: f64) -> Vec<Measurement> {
    let v = bus_voltages(state);
    let params = branch_params(&net.lines, &net.transformers);
    let switch_branches: std::collections::HashSet<usize> =
        net.switch_branches().into_iter().map(|(_, b)| b).collect();

    let mut ms = Vec::new();
    for (i, bus) in state.iter().enumerate() {
        if !net.zero_injection[i] && bus.voltage_mag != 0.0 {
            ms.push(Measurement {
                kind: MeasurementKind::VoltageMagnitude,
                target: Target::Bus(i),
                value: bus.voltage_mag,
                sigma,
            });
        }
    }
    for (b, p) in params.iter().enumerate() {
        if switch_branches.contains(&b) {
            continue;
        }
        for terminal in [Terminal::From, Terminal::To] {
            let (pf, qf) = terminal_flow(p, terminal, &v);
            if !pf.is_finite() || !qf.is_finite() {
                continue;
            }
            ms.push(Measurement {
                kind: MeasurementKind::ActivePower,
                target: Target::BranchTerminal { branch: b, terminal },
                value: pf,
                sigma,
            });
            ms.push(Measurement {
                kind: MeasurementKind::ReactivePower,
                target: Target::BranchTerminal { branch: b, terminal },
                value: qf,
                sigma,
            });
        }
    }
    ms
}

fn worst_error(estimated: &[Bus], truth: &[Bus]) -> (f64, f64) {
    let mut dv: f64 = 0.0;
    let mut da: f64 = 0.0;
    for (e, t) in estimated.iter().zip(truth) {
        if t.voltage_mag == 0.0 {
            continue; // de-energized: both report exactly zero
        }
        dv = dv.max((e.voltage_mag - t.voltage_mag).abs());
        da = da.max((e.voltage_ang - t.voltage_ang).abs());
    }
    (dv, da)
}

/// **Phase 5's gate.** A CGMES node-breaker network reaches the estimator at
/// all, and the estimate recovers the state the power flow produced — including
/// on the 30 bay-internal buses no sensor touches.
#[test]
fn minigrid_node_breaker_state_is_recovered_from_scada() {
    let Some(net) = node_breaker(RetentionPolicy::RetainAdjacentToBusbar) else { return };
    let truth = true_state(&net);
    let measurements = scada(&net, &truth, 0.01);

    let se = net.se_network(ybus_of(&net));
    let constrained = se.constrained_buses().iter().filter(|&&c| c).count();
    assert!(
        constrained >= 25,
        "only {constrained} constrained buses — the node-breaker view should be mostly \
         zero-injection"
    );

    let mut buses = net.buses.clone();
    gridoxide::se::nr::linear_start(&mut buses, &se, &measurements);
    let report = estimate(&measurements, &mut buses, &se, &SeOptions::default());
    assert_eq!(report.status, SeStatus::Converged, "estimate did not converge");

    let (dv, da) = worst_error(&buses, &truth);
    assert!(dv < 1e-4, "worst voltage error {dv} — the estimate did not find the true state");
    assert!(da < 1e-4, "worst angle error {da}");
}

/// The bay-internal buses are observable *because of* the constraints, not in
/// spite of the sensor set. Drop the constraints and the same measurements no
/// longer determine the state.
///
/// This is the node-breaker case for `se::observability::analyze` counting
/// constraint rows at all — a fix that was already needed on ordinary PGM data
/// and is unavoidable here.
#[test]
fn the_zero_injection_constraints_are_what_make_the_bays_observable() {
    let Some(net) = node_breaker(RetentionPolicy::RetainAdjacentToBusbar) else { return };
    let truth = true_state(&net);
    let measurements = scada(&net, &truth, 0.01);
    let se = net.se_network(ybus_of(&net));
    let layout = StateLayout::new(&net.buses, &measurements, &se);

    let with = observability::analyze(&measurements, &net.buses, &se, &layout, &Constraints::new(&se));
    let without =
        observability::analyze(&measurements, &net.buses, &se, &layout, &Constraints::from_flags(&[]));

    assert!(with.rank > without.rank, "constraints added no rank: {} vs {}", with.rank, without.rank);
    assert!(
        with.is_observable(),
        "the network should be observable with its constraints: rank {}/{}",
        with.rank, with.n_unknowns
    );
    assert!(
        !without.is_observable(),
        "the sensor set alone should not be enough — nothing measures a bay's interior"
    );
}

/// A measurement can attach to a breaker: phase 5's headline. The sensor is an
/// ordinary branch-terminal measurement at the switch's own branch index, and
/// the estimate reproduces the flow it reported.
#[test]
fn a_measurement_attaches_to_a_switch() {
    let Some(net) = node_breaker(RetentionPolicy::RetainAdjacentToBusbar) else { return };
    let truth = true_state(&net);
    let v = bus_voltages(&truth);

    // The busiest switch in the model — a sensor on a breaker carrying nothing
    // would assert very little.
    let (switch, branch, p_true, q_true) = net
        .switch_branches()
        .into_iter()
        .filter_map(|(s, b)| net.switch_flow(s, &v).map(|(p, q)| (s, b, p, q)))
        .max_by(|a, b| a.2.abs().partial_cmp(&b.2.abs()).unwrap())
        .expect("no switch carries a flow");
    assert!(p_true.abs() > 1e-6, "the busiest switch carries nothing; the fixture changed");

    let mut measurements = scada(&net, &truth, 0.01);
    for (kind, value) in
        [(MeasurementKind::ActivePower, p_true), (MeasurementKind::ReactivePower, q_true)]
    {
        measurements.push(Measurement {
            kind,
            target: Target::BranchTerminal { branch, terminal: Terminal::From },
            value,
            sigma: 0.01,
        });
    }

    let se = net.se_network(ybus_of(&net));
    let mut buses = net.buses.clone();
    gridoxide::se::nr::linear_start(&mut buses, &se, &measurements);
    let report = estimate(&measurements, &mut buses, &se, &SeOptions::default());
    assert_eq!(report.status, SeStatus::Converged);

    let (p_est, q_est) = net
        .switch_flow(switch, &bus_voltages(&buses))
        .expect("the switch still has a branch");
    assert!(
        (p_est - p_true).abs() < 1e-4 && (q_est - q_true).abs() < 1e-4,
        "estimated switch flow ({p_est}, {q_est}) does not match the measured ({p_true}, {q_true})"
    );

    // And the sensor's own residual is small: the estimate honors it rather
    // than discarding it as inconsistent.
    let n = measurements.len();
    for r in &report.residuals[n - 2..] {
        assert!(r.abs() < 1e-3, "switch measurement residual {r} is too large");
    }
}

/// `MergeAll` is the bus-branch view, reached through the same code. It must
/// estimate too — otherwise what the node-breaker path proves is only that
/// zero-injection constraints can carry an under-sensored network, not that
/// CGMES data reaches the estimator.
#[test]
fn the_merged_view_estimates_through_the_same_path() {
    let Some(net) = node_breaker(RetentionPolicy::MergeAll) else { return };
    let truth = true_state(&net);
    let measurements = scada(&net, &truth, 0.01);

    let se = net.se_network(ybus_of(&net));
    let mut buses = net.buses.clone();
    gridoxide::se::nr::linear_start(&mut buses, &se, &measurements);
    let report = estimate(&measurements, &mut buses, &se, &SeOptions::default());
    assert_eq!(report.status, SeStatus::Converged);

    let (dv, da) = worst_error(&buses, &truth);
    assert!(dv < 1e-4 && da < 1e-4, "worst error ({dv}, {da})");
}

/// A bus with a load on it is never constrained, no matter what the snapshot
/// says its power is. The flags come from the model's structure.
#[test]
fn a_bus_with_equipment_is_not_zero_injection() {
    let Some(net) = node_breaker(RetentionPolicy::MergeAll) else { return };
    let mut with_equipment = 0;
    for (i, bus) in net.buses.iter().enumerate() {
        if bus.p_spec != 0.0 || bus.q_spec != 0.0 {
            assert!(!net.zero_injection[i], "bus {i} injects {} but is flagged zero", bus.p_spec);
            with_equipment += 1;
        }
    }
    assert!(with_equipment > 0, "no bus in MiniGrid injects anything; the fixture changed");
    // Shunts are structural rather than an injection in `p_spec`, and are
    // likewise not zero-injection buses.
    for s in &net.shunts {
        assert!(!net.zero_injection[s.at], "bus {} carries a shunt but is flagged zero", s.at);
    }
}
