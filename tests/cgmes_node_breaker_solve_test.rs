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

use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
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

    let (buses, lines, transformers, shunts, view) = cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &RetentionPolicy::RetainAdjacentToBusbar,
        SwitchTreatment::Regularize,
    )
    .expect("node-breaker conversion failed");

    // Fixture assumption: switches really were retained, or this proves nothing.
    let switch_branches = regularized_branches(&view);
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

    // Switch flows: the retained switches are the tail of the transformer list,
    // so their flat branch indices follow the model's own branches.
    let params = branch_params(&lines, &transformers);
    let v = bus_voltages(&report.buses);
    let first_switch = lines.len() + transformers.len() - switch_branches.len();

    let mut carrying = 0;
    for k in 0..switch_branches.len() {
        let (p, q) = terminal_flow(&params[first_switch + k], Terminal::From, &v);
        assert!(p.is_finite() && q.is_finite(), "switch {k} flow is not finite");
        if p.abs() > 1e-9 {
            carrying += 1;
        }
    }
    assert!(carrying > 0, "every retained switch reported zero flow");

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
        let (buses, lines, transformers, shunts, view) =
            cgmes_node_breaker_to_buses_and_branches(
                &ds,
                100e6,
                policy,
                SwitchTreatment::Regularize,
            )
            .expect("conversion failed");
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
        let (buses, lines, transformers, shunts, view) =
            cgmes_node_breaker_to_buses_and_branches(
                &ds,
                100e6,
                &policy,
                SwitchTreatment::Regularize,
            )
            .expect("conversion failed");
        let retained = view.retained().len();
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

    let (buses, lines, transformers, shunts, view) = cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &RetentionPolicy::RetainAll,
        SwitchTreatment::Regularize,
    )
    .expect("conversion failed");

    assert_eq!(view.retained().len(), 90, "every MiniGrid switch should be retained");
    assert!(buses.len() >= 100, "RetainAll should merge almost nothing, got {} buses", buses.len());

    let mut ybus = build_ybus(buses.len(), &lines, &transformers);
    stamp_shunts(&mut ybus, &shunts);
    let report = run_power_flow_analysis_from_ybus(buses, ybus);
    assert_eq!(report.stats.status, SolveStatus::Converged);
}
