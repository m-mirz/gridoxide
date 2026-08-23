//! ENTSO-E's FullGrid conformance configuration, and why it is not a solve
//! fixture.
//!
//! FullGrid is the richest conformity model in the tree by some distance —
//! both `TapChangerControl` modes, two phase shifters sharing one target, a
//! three-winding transformer, HVDC, a bus held by nine machines. Every one of
//! those makes it valuable for *import* questions, and this crate uses it for
//! exactly that in `cgmes_tap_table_test.rs`, `cgmes_tap_regulation_test.rs`,
//! `cgmes_voltage_control_test.rs` and `cgmes_node_breaker_test.rs`.
//!
//! **It does not converge, and it cannot.** The reason is in the fixture, not
//! in gridoxide: `NonlinearShuntCompensatorPoint._7df4778f` declares
//! `b = 0.99 S` and `g = 0.99 S` for `BE_SHUNT_1` at `nomU = 225 kV`. On a 100
//! MVA base that is
//!
//! ```text
//! z_base = 225000² / 100e6           = 506.25 Ω
//! g_pu   = 0.99 × 506.25             = 501.19 pu
//! P      = g_pu × V² × S_base        = 50,119 MW  at V = 1.0 pu
//! ```
//!
//! — a shunt compensator dissipating **50 GW** on a network whose entire
//! scheduled generation is 485 MW, a factor of 103. The first Newton iteration
//! reports a mismatch of ~503 pu, which is that shunt and almost nothing else.
//!
//! gridoxide's conversion is arithmetically right; the input is not physical. A
//! conformity model exists to exercise a profile's classes, and `0.99` turns up
//! across FullGrid as filler for several unrelated quantities — it is also the
//! SVC's `inductiveRating`/`capacitiveRating`, giving that device a ±511 pu
//! reactive band. These are placeholders, not a modelled network.
//!
//! # Why this file exists
//!
//! To stop the question being reopened. `scripts/bench/README.md` records that
//! `network::dc_angle_guess` was added specifically to make FullGrid converge,
//! did not, and was removed after it broke `case3120sp` — the budget has been
//! spent on this once already. The assertions below pin the fixture's own
//! numbers, so if ENTSO-E ever corrects them this fails and says so; until
//! then, FullGrid is an import fixture and nothing more.
//!
//! Note what this is *not*: the fixture's published `SvVoltage` is consistent
//! with its `EQ`/`SSH` transformer data — `check_cgmes_sv_consistency.py`
//! flags zero of its ten two-winding transformers above 5%. The inconsistency
//! is the shunt, and it is between the fixture and physics rather than between
//! two of its own profiles.

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_to_network, load_profiles, CgmesNetwork};
use gridoxide::solver::{PowerFlowOptions, SolveStatus};
use gridoxide::{run_power_flow, TapData};

fn fullgrid() -> Option<CgmesNetwork> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0/FullGrid/FullGrid-Merged");
    if !dir.exists() {
        eprintln!("skipping: FullGrid fixture not checked out");
        return None;
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("configuration directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "xml"))
        .collect();
    paths.sort();
    let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    let ds = load_profiles(&refs).expect("failed to decode CGMES profiles");
    Some(cgmes_to_network(&ds, 100e6).expect("conversion failed"))
}

/// The 50 GW shunt itself. Pinned so that a corrected fixture announces
/// itself rather than sitting unnoticed behind an `#[ignore]`.
#[test]
fn fullgrid_declares_a_fifty_gigawatt_shunt() {
    let Some(net) = fullgrid() else { return };

    let worst = net
        .shunts
        .iter()
        .max_by(|a, b| a.y.norm().total_cmp(&b.y.norm()))
        .expect("FullGrid has shunts");

    // 0.99 S at 225 kV on a 100 MVA base, in both components.
    let expected = 0.99 * (225e3 * 225e3 / 100e6);
    assert!(
        (worst.y.re - expected).abs() < 1e-6 && (worst.y.im - expected).abs() < 1e-6,
        "the shunt at bus {} is {} pu; expected {expected} in both components. If ENTSO-E has \
         corrected BE_SHUNT_1, delete this file and give FullGrid a real solve test.",
        worst.at,
        worst.y
    );

    // The conductance is the part that makes it unphysical: a compensator is a
    // reactive device, and 50 GW of loss is a hundred times this network.
    let mw = worst.y.re * 100.0;
    assert!(mw > 50_000.0, "{mw} MW");
    let scheduled: f64 = net.buses.iter().map(|b| b.p_spec.max(0.0)).sum::<f64>() * 100.0;
    assert!(
        mw > 50.0 * scheduled,
        "shunt draws {mw:.0} MW against {scheduled:.0} MW of scheduled generation"
    );
}

/// And the consequence: the solve cannot converge, and the first iteration's
/// mismatch is that shunt and almost nothing else.
///
/// Asserting the *failure* is deliberate. A skipped or absent test would let
/// someone spend a week on this again; a failing assertion here would mean the
/// fixture changed, which is the only thing that could make it solvable.
#[test]
fn fullgrid_cannot_converge_and_the_shunt_is_why() {
    let Some(net) = fullgrid() else { return };

    let report = run_power_flow(
        net.buses.clone(),
        &net.lines,
        &net.transformers,
        &net.shunts,
        TapData::none(),
        PowerFlowOptions { tol: 1e-8, max_iter: 40, ..Default::default() },
    );
    assert_eq!(
        report.stats.status,
        SolveStatus::MaxIterationsReached,
        "FullGrid converged. If the fixture was corrected, replace this file with a real \
         solve test; if gridoxide changed, work out which change made an unphysical network \
         solvable before celebrating."
    );

    let first = *report
        .stats
        .mismatch_history
        .first()
        .expect("at least one iteration ran");
    assert!(
        first > 400.0,
        "the first mismatch was {first} pu; the diagnosis in this file's header assumed ~503"
    );
    // Specifically the *conductance*: the largest single mismatch a flat start
    // reports is an active-power one, and 501 pu of `g` draws 501 pu of P at
    // V = 1. The susceptance contributes to the reactive mismatch instead,
    // which is why this compares against `y.re` and not `y.norm()`.
    let g = net.shunts.iter().map(|s| s.y.re).fold(0.0, f64::max);
    assert!(
        (first / g - 1.0).abs() < 0.05,
        "first mismatch {first} pu against a shunt conductance of {g} pu — these should be the \
         same quantity, so the diagnosis in this file's header needs revisiting"
    );
}

/// What FullGrid *is* good for, asserted so this file reads as a scope
/// statement rather than a complaint. These are the properties the import
/// tests rely on, and they are why the fixture stays in the tree.
#[test]
fn fullgrid_remains_the_richest_import_fixture() {
    let Some(net) = fullgrid() else { return };
    assert_eq!(net.tap_report.converted, 3, "both control modes, one disabled set");
    assert_eq!(net.tap_report.disabled, 4);
    assert_eq!(net.voltage_control.shared_buses, 2, "two buses held by several machines");
    assert!(
        net.tap_changers.iter().filter(|c| c.is_some()).count() >= 10,
        "eleven tap tables across all the changer flavours"
    );
    // A three-winding transformer converts to three star legs, so the
    // transformer count exceeds the `PowerTransformer` count and the bus list
    // gains a star point no terminal names.
    assert!(net.transformers.len() >= 13, "{} transformers", net.transformers.len());
    assert!(net.buses.len() > 20, "including a synthesized star point");
}
