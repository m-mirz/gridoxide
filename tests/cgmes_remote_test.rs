//! Remote voltage control on the one vendored case that has it.
//!
//! MicroGrid-Type1 has a single machine at 10.5 kV holding the 110 kV side of
//! its own step-up transformer. FullGrid has one too, in the configuration with
//! the 50 GW shunt that does not solve, so this is the only usable case in the
//! corpus.
//!
//! **It can referee the structure and not the numbers**, and the distinction is
//! worth stating rather than papering over. gridoxide's solution of this network
//! already differs from the published one by 4.4% on its 400 kV buses — before
//! any of this, in both modes, and unchanged by it — which is why the fixture's
//! own voltage assertion tolerates 5%. The reactive power remote control moves
//! is smaller than that discrepancy, so matching the published machine output
//! is not available as a gate here. It is settled analytically in
//! `remote_voltage_test.rs` instead.
//!
//! What this fixture *can* settle is that the control ends up in the right
//! place, which is the defect: the reference solution has the machine's reactive
//! output free at the machine and nothing free at the bus it holds, and until
//! now gridoxide had it exactly the other way round.

mod cgmes_common;

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_to_network, load_profiles};
use gridoxide::network::{build_ybus, power_injections, stamp_shunts};
use gridoxide::outerloop::RemoteOutcome;
use gridoxide::solver::{PowerFlowOptions, SolveStatus};
use gridoxide::types::BusType;

const S_BASE_VA: f64 = 100e6;

fn microgrid() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGrid-Type1/MicroGrid-Type1-Merged",
    );
    if !dir.exists() {
        eprintln!(
            "skipping: {} not found — run `git submodule update --init \
             tests/data/CGMES-Test-Configurations`",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

struct Case {
    net: gridoxide::cgmes::CgmesNetwork,
}

fn load() -> Option<Case> {
    let dir = microgrid()?;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "xml"))
        .collect();
    paths.sort();
    let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    let ds = load_profiles(&refs).expect("decode");
    Some(Case { net: cgmes_to_network(&ds, S_BASE_VA).expect("convert") })
}

fn solve(case: &Case, on: bool) -> gridoxide::PowerFlowReport {
    gridoxide::run_power_flow_with_remote(
        case.net.buses.clone(),
        &case.net.lines,
        &case.net.transformers,
        &case.net.shunts,
        gridoxide::TapData::none(),
        gridoxide::RemoteControlData { machines: &case.net.voltage_control.machines },
        PowerFlowOptions {
            control_remote_voltage: on,
            enforce_q_limits: true,
            max_iter: 60,
            max_outer_iter: 60,
            ..Default::default()
        },
    )
}

/// The fixture has exactly one remotely-regulating machine, and it is the
/// arrangement the whole feature is about: a generator holding the far side of
/// its own step-up transformer.
#[test]
fn microgrid_has_one_remote_machine_behind_its_own_transformer() {
    let Some(case) = load() else { return };
    let remote: Vec<_> = case
        .net
        .voltage_control
        .machines
        .iter()
        .filter(|m| m.at_bus != m.controls_bus)
        .collect();
    assert_eq!(remote.len(), 1, "MicroGrid-Type1's single remote controller");

    let m = remote[0];
    let adjacent = case
        .net
        .transformers
        .iter()
        .any(|t| (t.from == m.at_bus && t.to == m.controls_bus) || (t.to == m.at_bus && t.from == m.controls_bus));
    assert!(adjacent, "the two buses should be the ends of one transformer");
    assert!(
        case.net.buses[m.at_bus].u_rated < case.net.buses[m.controls_bus].u_rated,
        "the machine should be on the low-voltage side"
    );
}

/// **The gate this fixture can settle.** The control moves onto the machine, so
/// the reactive power is produced where the machine is.
///
/// Checked as a *reversal*, because that is what the defect was: with the loop
/// off, the bus being held carries a free reactive injection and the machine is
/// pinned at its schedule; with it on, exactly the opposite — which is the
/// structure the published solution has.
#[test]
fn the_control_moves_from_the_held_bus_onto_the_machine() {
    let Some(case) = load() else { return };
    let m = case
        .net
        .voltage_control
        .machines
        .iter()
        .find(|m| m.at_bus != m.controls_bus)
        .expect("one remote machine");

    let mut y = build_ybus(case.net.buses.len(), &case.net.lines, &case.net.transformers);
    stamp_shunts(&mut y, &case.net.shunts);
    let ybus = y.finish();

    let off = solve(&case, false);
    let on = solve(&case, true);
    assert_eq!(off.stats.status, SolveStatus::Converged);
    assert_eq!(on.stats.status, SolveStatus::Converged);

    // Off: the pin is on the bus being held, and the machine is a fixed
    // injection — the defect.
    assert_eq!(off.buses[m.controls_bus].bus_type, BusType::PV);
    assert_eq!(off.buses[m.at_bus].bus_type, BusType::PQ);

    // On: reversed.
    assert_eq!(on.buses[m.controls_bus].bus_type, BusType::PQ);
    assert_eq!(on.buses[m.at_bus].bus_type, BusType::PV);

    let outer = on.outer.as_ref().expect("the loop ran");
    assert_eq!(outer.remote.len(), 1);
    assert_eq!(outer.remote[0].outcome, RemoteOutcome::Held, "{:?}", outer.remote[0]);
    assert_eq!(outer.retyped.len(), 2, "one bus freed, one pinned: {:?}", outer.retyped);

    // The target is still met — moving the control must not lose it.
    assert!(
        (on.buses[m.controls_bus].voltage_mag - m.target_pu).abs() < 2e-4,
        "held bus reached {} against a target of {}",
        on.buses[m.controls_bus].voltage_mag,
        m.target_pu
    );

    // And the reactive power is now where the machine is: the held bus injects
    // only its own load, as the published solution has it.
    let (_, q_on) = power_injections(&on.buses, &ybus);
    let (_, q_off) = power_injections(&off.buses, &ybus);
    let held_spec = case.net.buses[m.controls_bus].q_spec;
    assert!(
        (q_on[m.controls_bus] - held_spec).abs() < 1e-6,
        "with the control moved, the held bus should inject only its own load ({held_spec}), \
         got {}",
        q_on[m.controls_bus]
    );
    assert!(
        (q_off[m.controls_bus] - held_spec).abs() > 1.0,
        "without it, the held bus carries the machine's reactive power — over 1 p.u. of it — \
         which is the defect this fixes"
    );

    // Active power is untouched either way, and matches the published dispatch.
    let (p_on, _) = power_injections(&on.buses, &ybus);
    assert!(
        (p_on[m.at_bus] - case.net.buses[m.at_bus].p_spec).abs() < 1e-6,
        "the machine's active output is a schedule and must not move"
    );
}

/// The 4.4% the fixture's own voltage assertion tolerates is **not** remote
/// control, and this pins that so the two are never confused.
///
/// Every 400 kV bus is off by the same amount in both modes. Only the machine's
/// own bus moves, which is exactly the bus whose voltage the control decides.
#[test]
fn moving_the_control_changes_only_the_machines_own_bus() {
    let Some(case) = load() else { return };
    let m = case
        .net
        .voltage_control
        .machines
        .iter()
        .find(|m| m.at_bus != m.controls_bus)
        .expect("one remote machine");

    let off = solve(&case, false);
    let on = solve(&case, true);

    for i in 0..case.net.buses.len() {
        if i == m.at_bus {
            assert!(
                (on.buses[i].voltage_mag - off.buses[i].voltage_mag).abs() > 1e-3,
                "the machine's own bus is the one the control decides and should move"
            );
            continue;
        }
        assert!(
            (on.buses[i].voltage_mag - off.buses[i].voltage_mag).abs() < 5e-3,
            "bus {i} moved by {} — moving the control should not disturb the rest of the \
             network this much",
            (on.buses[i].voltage_mag - off.buses[i].voltage_mag).abs()
        );
    }
}
