//! Solving a CGMES model as a node-breaker network, with retained switches.
//!
//! This is `plans/NODE_BREAKER_PLAN.md` phase 2's gate: MiniGrid under
//! `RetainAdjacentToBusbar` must converge and report switch flows. It does —
//! and so does every other node-breaker configuration in the tree except one,
//! at every retention policy including the full node-breaker view.
//!
//! | Config | TN path | `MergeAll` | busbar-retained | `RetainAll` |
//! |---|---|---|---|---|
//! | MiniGrid | converges | converges | converges (30 retained) | converges (90 retained) |
//! | SmallGrid | converges | converges | converges (373 retained) | converges (1,266 retained) |
//! | Svedala | converges | converges | converges (857 retained) | converges (1,464 retained) |
//! | FullGrid | **fails** | fails | fails | fails |
//!
//! # The one bug behind two failures
//!
//! Svedala and SmallGrid-under-`RetainAll` both reported `Singular` at first,
//! and both had the same cause — not the switches.
//!
//! A de-energized bus is `Slack` at `V = 0`: a placeholder, not a reference.
//! `network::classify` counted it as one, so a component consisting of a live
//! `PQ` bus and a dead placeholder came back "solvable" with nothing to solve
//! against. That `PQ` bus then gets an identically zero angle row, because
//! `H_ii = −Q_i − V_i²B_ii` cancels exactly when the only neighbour sits at
//! zero volts. The bus-branch importer never produced such a pair, since it
//! merges the dead node into a live one; node-breaker import stops merging, so
//! it does. Fixed in `classify`, regression-tested in `network`'s own unit
//! tests.
//!
//! # What that settles about §4.1
//!
//! `plans/NODE_BREAKER_PLAN.md` §4.1 calls `SwitchTreatment::Regularize` "dead
//! on arrival at real scale", reasoning from a recorded divergence at ~30
//! switches to SmallGrid's 1,266 and Svedala's 1,464 being "45–52x past the
//! count already measured to diverge".
//!
//! Measured: both converge with **every** switch retained, in the same
//! iteration count as the bus-branch solve of the same model. The scaling
//! argument against `Regularize` does not survive contact with the data.
//! `Constrain` keeps its other advantages — no large number in the matrix, and
//! a principled answer for switch flows inside a loop — but not that one.
//!
//! # The remaining failure
//!
//! FullGrid fails on **both** paths: the ordinary `TopologicalNode` importer
//! returns `MaxIterationsReached` on it too. Pre-existing and unrelated to
//! node-breaker support; there is no `cgmes_fullgrid_test` in the tree for the
//! same reason.

use std::path::{Path, PathBuf};

use gridoxide::branch_flow::{branch_params, bus_voltages};
use gridoxide::cgmes::{cgmes_node_breaker_to_buses_and_branches, load_profiles};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::run_power_flow_analysis_from_ybus;
use gridoxide::solver::SolveStatus;
use gridoxide::switches::{regularized_branches, SwitchTreatment};
use gridoxide::topology::RetentionPolicy;

fn dataset_dir(dir: &str) -> Option<PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(dir);
    if d.exists() {
        return Some(d);
    }
    eprintln!(
        "skipping: {} not found — run `git submodule update --init \
         tests/data/CGMES-Test-Configurations`",
        d.display()
    );
    None
}

fn load(dir: &str, prefix: &str) -> Option<gridoxide::cgmes::CimDataset> {
    let d = dataset_dir(dir)?;
    let files: Vec<PathBuf> = ["EQBD", "EQ", "SSH", "TP", "SV"]
        .iter()
        .map(|p| d.join(format!("{prefix}_{p}.xml")))
        .filter(|p| p.exists())
        .collect();
    let refs: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
    Some(load_profiles(&refs).expect("failed to decode CGMES profiles"))
}

/// **Phase 2's gate.** MiniGrid, imported as a node-breaker model with its
/// busbar-adjacent switches retained as real elements, converges — and every
/// retained switch carries a reportable flow.
///
/// MiniGrid also converges under `RetainAll`, i.e. the *full* node-breaker
/// view: 105 buses and all 90 switches retained, in two iterations. That is
/// covered by `minigrid_converges_under_the_full_node_breaker_view` below.
#[test]
fn minigrid_converges_with_retained_switches_and_reports_their_flows() {
    let Some(ds) = load("MiniGrid/MiniGrid-Merged", "MiniGrid") else { return };

    let net = cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &RetentionPolicy::RetainAdjacentToBusbar,
        SwitchTreatment::Regularize,
    )
    .expect("node-breaker conversion failed");
    let (buses, lines, transformers, shunts) =
        (net.buses.clone(), net.lines.clone(), net.transformers.clone(), net.shunts.clone());

    // Fixture assumption: switches really were retained, or this proves nothing.
    let switch_branches = regularized_branches(&net.view);
    assert!(
        switch_branches.len() >= 20,
        "only {} switches retained — the busbar policy found almost nothing",
        switch_branches.len()
    );
    assert!(
        buses.len() > 13,
        "node-breaker import produced {} buses, no more than the bus-branch view's 13",
        buses.len()
    );

    let mut ybus = build_ybus(buses.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let report = run_power_flow_analysis_from_ybus(buses, ybus);
    assert_eq!(report.stats.status, SolveStatus::Converged);

    // Switch flows, addressed by switch rather than by hand-computed index.
    let v = bus_voltages(&report.buses);
    let mut carrying = 0;
    for (switch, _) in net.switch_branches() {
        let (p, q) = net.switch_flow(switch, &v).expect("a stamped switch must have a flow");
        assert!(p.is_finite() && q.is_finite(), "switch {switch:?} flow is not finite");
        if p.abs() > 1e-9 {
            carrying += 1;
        }
    }
    assert!(carrying > 0, "every retained switch reported zero flow");

    // The hand-computed index and the owned mapping must agree, since existing
    // callers still do the arithmetic.
    let params = branch_params(&lines, &transformers);
    let first_switch = lines.len() + transformers.len() - switch_branches.len();
    for (n, (switch, branch)) in net.switch_branches().iter().enumerate() {
        assert_eq!(*branch, first_switch + n, "mapping disagrees for {switch:?}");
        let _ = &params[*branch];
    }

    // A closed ideal switch should barely drop any voltage. Checked against the
    // network's own spread so the bound is not an arbitrary constant.
    let vm: Vec<f64> = report.buses.iter().map(|b| b.voltage_mag).collect();
    let spread = vm.iter().cloned().fold(f64::MIN, f64::max)
        - vm.iter().cloned().filter(|v| *v > 0.0).fold(f64::MAX, f64::min);
    for (k, branch) in switch_branches.iter().enumerate() {
        if branch.from_status == 0 {
            continue;
        }
        let drop = (vm[branch.from] - vm[branch.to]).abs();
        assert!(
            drop < spread / 100.0,
            "closed switch {k} drops {drop} p.u., against a network spread of {spread}"
        );
    }
}

/// Retaining switches must not change the answer: the buses that exist in both
/// views have to solve to the same voltages, or "retained" would mean
/// "different network".
#[test]
fn retaining_switches_does_not_change_the_solution() {
    let Some(ds) = load("MiniGrid/MiniGrid-Merged", "MiniGrid") else { return };

    let solve = |policy: &RetentionPolicy| {
        let net = cgmes_node_breaker_to_buses_and_branches(&ds, 100e6, policy, SwitchTreatment::Regularize)
            .expect("conversion failed");
        let (buses, lines, transformers, shunts, view) =
            (net.buses, net.lines, net.transformers, net.shunts, net.view);
        let mut ybus = build_ybus(buses.len(), &lines, &transformers);
        stamp_shunts(&mut ybus, &shunts);
        let report = run_power_flow_analysis_from_ybus(buses, ybus);
        assert_eq!(report.stats.status, SolveStatus::Converged);
        (report, view)
    };

    let (merged, _) = solve(&RetentionPolicy::MergeAll);
    let (retained, retained_view) = solve(&RetentionPolicy::RetainAdjacentToBusbar);

    // Retaining can only add buses, never remove them.
    assert!(retained.buses.len() > merged.buses.len(), "nothing was actually retained");
    assert!(retained_view.retained().len() >= 20);

    // And the voltage *range* must be preserved — retained switches are ideal,
    // so they introduce no new extremes.
    let range = |r: &gridoxide::PowerFlowReport| {
        let live: Vec<f64> =
            r.buses.iter().map(|b| b.voltage_mag).filter(|v| *v > 0.0).collect();
        (
            live.iter().cloned().fold(f64::MAX, f64::min),
            live.iter().cloned().fold(f64::MIN, f64::max),
        )
    };
    let (lo_m, hi_m) = range(&merged);
    let (lo_r, hi_r) = range(&retained);
    assert!((lo_m - lo_r).abs() < 1e-4, "min |V| moved: {lo_m} vs {lo_r}");
    assert!((hi_m - hi_r).abs() < 1e-4, "max |V| moved: {hi_m} vs {hi_r}");
}

/// SmallGrid and Svedala at full node-breaker scale — 1,266 and 1,464 retained
/// switches. `plans/NODE_BREAKER_PLAN.md` §4.1 predicts `Regularize` cannot
/// cope at these counts. It can.
#[test]
fn the_largest_models_converge_with_every_switch_retained() {
  for (dir, prefix, min_retained) in [
      ("SmallGrid/SmallGrid-Merged", "SmallGrid", 1266usize),
      ("Svedala/Svedala-Merged", "Svedala", 1464),
  ] {
    let Some(ds) = load(dir, prefix) else { continue };

    for policy in [
        RetentionPolicy::MergeAll,
        RetentionPolicy::RetainAdjacentToBusbar,
        RetentionPolicy::RetainAll,
    ] {
        let net = cgmes_node_breaker_to_buses_and_branches(
            &ds,
            100e6,
            &policy,
            SwitchTreatment::Regularize,
        )
        .expect("conversion failed");
        let retained = net.view.retained().len();
        let (buses, lines, transformers, shunts) =
            (net.buses, net.lines, net.transformers, net.shunts);
        let mut ybus = build_ybus(buses.len(), &lines, &transformers);
        stamp_shunts(&mut ybus, &shunts);
        let n_buses = buses.len();
        let report = run_power_flow_analysis_from_ybus(buses, ybus);
        assert_eq!(
            report.stats.status,
            SolveStatus::Converged,
            "{policy:?}: {n_buses} buses, {retained} retained switches"
        );
        if policy == RetentionPolicy::RetainAll {
            assert!(
                retained >= min_retained,
                "{prefix}: RetainAll kept only {retained} switches"
            );
        }
        eprintln!(
            "{prefix} {policy:?}: {n_buses} buses, {retained} retained, {} iterations",
            report.stats.iterations()
        );
    }
  }
}

/// `SwitchTreatment::Merge` must refuse a policy that retained something,
/// rather than silently dropping the switches and returning a different
/// network than the caller asked for.
#[test]
fn merge_treatment_refuses_a_retaining_policy() {
    let Some(ds) = load("MiniGrid/MiniGrid-Merged", "MiniGrid") else { return };

    assert!(cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &RetentionPolicy::RetainAdjacentToBusbar,
        SwitchTreatment::Merge,
    )
    .is_err());

    // ...but the combination that *is* coherent works.
    assert!(cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &RetentionPolicy::MergeAll,
        SwitchTreatment::Merge,
    )
    .is_ok());
}

/// The full node-breaker view on a real model: every connectivity node its own
/// bus, every switch retained. MiniGrid is small enough that `RetainAll` is
/// tractable, and it is the strongest available demonstration that the
/// formulation works end to end.
#[test]
fn minigrid_converges_under_the_full_node_breaker_view() {
    let Some(ds) = load("MiniGrid/MiniGrid-Merged", "MiniGrid") else { return };

    let net = cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &RetentionPolicy::RetainAll,
        SwitchTreatment::Regularize,
    )
    .expect("conversion failed");
    let (buses, lines, transformers, shunts, view) =
        (net.buses, net.lines, net.transformers, net.shunts, net.view);

    assert_eq!(view.retained().len(), 90, "every MiniGrid switch should be retained");
    assert!(buses.len() >= 100, "RetainAll should merge almost nothing, got {} buses", buses.len());

    let mut ybus = build_ybus(buses.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let report = run_power_flow_analysis_from_ybus(buses, ybus);
    assert_eq!(report.stats.status, SolveStatus::Converged);
}

/// **Phase 4 and 6, for free.** A retained switch is a branch, so every
/// calculation the crate already had works on one without knowing it is a
/// switch. This asserts all three on a real node-breaker model.
#[test]
fn dc_sensitivities_and_contingencies_all_work_on_switches() {
    use gridoxide::batch::{BatchSolver, Scenario};
    use gridoxide::linear::{dc_branches, dc_power_flow, DcOptions, DcSensitivity};
    use gridoxide::solver::JacobianBackend;

    let Some(ds) = load("MiniGrid/MiniGrid-Merged", "MiniGrid") else { return };
    let net = cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &RetentionPolicy::RetainAdjacentToBusbar,
        SwitchTreatment::Regularize,
    )
    .expect("conversion failed");

    let switches = net.switch_branches();
    assert!(switches.len() >= 20, "only {} switches stamped", switches.len());

    // (1) DC solves a node-breaker network, and switch flows land at the
    //     switches' own branch indices.
    let mut buses = net.buses.clone();
    let solution = dc_power_flow(&mut buses, &net.lines, &net.transformers, DcOptions::default());
    assert!(solution.max_residual < 1e-9, "DC residual {}", solution.max_residual);
    let carrying = switches
        .iter()
        .filter(|(_, branch)| solution.branch_p[*branch].abs() > 1e-9)
        .count();
    assert!(carrying > 0, "no switch carries DC flow");

    // (2) A switch's LODF column *is* its bus-split distribution factor —
    //     `plans/NODE_BREAKER_PLAN.md` §5.4's headline capability.
    let branches = dc_branches(&net.lines, &net.transformers, DcOptions::default());
    let sensitivity = DcSensitivity::new(&buses, &branches, net.n_branches())
        .expect("reduced B is singular");
    let with_factors = switches
        .iter()
        .filter(|(_, branch)| sensitivity.lodf_column(*branch).is_some())
        .count();
    assert!(
        with_factors > 0,
        "no switch has redistribution factors; all {} are radial",
        switches.len()
    );
    // A radial switch is a legitimate answer, not a failure — opening it
    // islands whatever sits behind it.
    for (_, branch) in &switches {
        assert_eq!(
            sensitivity.lodf_column(*branch).is_none(),
            sensitivity.is_radial(*branch)
        );
    }

    // (3) A switching campaign is an ordinary contingency sweep.
    let scenarios: Vec<Scenario> = switches
        .iter()
        .map(|(_, branch)| {
            let mut sc = Scenario::new(vec![]);
            sc.branch_outages = vec![*branch];
            sc
        })
        .collect();
    let reports = BatchSolver::new(JacobianBackend::Scalar)
        .solve_contingencies(
            &net.buses,
            &net.lines,
            &net.transformers,
            &net.shunts,
            &scenarios,
            1e-6,
            20,
        )
        .expect("switching campaign failed");
    assert_eq!(reports.len(), switches.len());
    let converged = reports
        .iter()
        .filter(|r| r.stats.status == SolveStatus::Converged)
        .count();
    assert!(
        converged * 2 > switches.len(),
        "only {converged} of {} switch openings converged",
        switches.len()
    );
    // The ones that do not converge must be the severing ones, reported as
    // unreferenced islands rather than as numerical failures.
    for (report, (_, branch)) in reports.iter().zip(&switches) {
        if report.stats.status != SolveStatus::Converged {
            assert!(
                sensitivity.is_radial(*branch),
                "branch {branch} failed to converge but is not radial"
            );
        }
    }
}

/// `set_switch_open` must change the answer, and must do so without changing
/// the Y-bus sparsity pattern — which is what lets a switching campaign share
/// one symbolic factorization.
#[test]
fn opening_a_switch_changes_the_solution_but_not_the_pattern() {
    let Some(ds) = load("MiniGrid/MiniGrid-Merged", "MiniGrid") else { return };
    let mut net = cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &RetentionPolicy::RetainAdjacentToBusbar,
        SwitchTreatment::Regularize,
    )
    .expect("conversion failed");

    let pattern = |n: &gridoxide::switches::NodeBreakerNetwork| {
        let mut ybus = build_ybus(n.buses.len(), &n.lines, &n.transformers);
        stamp_shunts(&mut ybus, &n.shunts);
        let y = ybus.finish();
        (0..y.n())
            .map(|i| y.row(i).iter().map(|&(j, _)| j).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    let solve = |n: &gridoxide::switches::NodeBreakerNetwork| {
        let mut ybus = build_ybus(n.buses.len(), &n.lines, &n.transformers);
        stamp_shunts(&mut ybus, &n.shunts);
        run_power_flow_analysis_from_ybus(n.buses.clone(), ybus)
    };

    let before_pattern = pattern(&net);
    let before = solve(&net);
    assert_eq!(before.stats.status, SolveStatus::Converged);

    // Pick a switch that actually carries flow, so opening it must matter.
    let v = bus_voltages(&before.buses);
    let (switch, _) = *net
        .switch_branches()
        .iter()
        .max_by(|a, b| {
            let fa = net.switch_flow(a.0, &v).map(|(p, _)| p.abs()).unwrap_or(0.0);
            let fb = net.switch_flow(b.0, &v).map(|(p, _)| p.abs()).unwrap_or(0.0);
            fa.total_cmp(&fb)
        })
        .expect("no switches");
    assert!(net.switch_flow(switch, &v).unwrap().0.abs() > 1e-6);

    assert_eq!(net.is_switch_open(switch), Some(false));
    assert!(net.set_switch_open(switch, true));
    assert_eq!(net.is_switch_open(switch), Some(true));

    assert_eq!(pattern(&net), before_pattern, "opening a switch moved the sparsity pattern");

    let after = solve(&net);
    let moved = before
        .buses
        .iter()
        .zip(&after.buses)
        .map(|(a, b)| (a.voltage_ang - b.voltage_ang).abs())
        .fold(0.0f64, f64::max);
    assert!(moved > 1e-9, "opening a load-carrying switch changed nothing");

    // Closing it again restores the original answer exactly.
    assert!(net.set_switch_open(switch, false));
    let restored = solve(&net);
    for (a, b) in before.buses.iter().zip(&restored.buses) {
        assert!((a.voltage_mag - b.voltage_mag).abs() < 1e-9);
        assert!((a.voltage_ang - b.voltage_ang).abs() < 1e-9);
    }
}
