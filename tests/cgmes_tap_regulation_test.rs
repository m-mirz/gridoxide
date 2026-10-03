//! CGMES tap-changer regulating controls, resolved onto gridoxide's indices.
//!
//! `src/cgmes.rs` read `RegulatingControl` four times over — for
//! `SynchronousMachine`, `StaticVarCompensator`, `PowerElectronicsConnection`
//! and `ExternalNetworkInjection` — and never for a tap changer, so a CGMES
//! network arrived with tap positions and no statement of what they were
//! holding.
//!
//! What is asserted here is counts, modes and per-unit targets. Deliberately
//! **not** transformer or bus indices: a CGMES import renumbers on every run
//! (bus indices in particular — see the note in this file's last test), so an
//! index assertion would be testing the hash seed.

use std::path::{Path, PathBuf};

use gridoxide::cgmes::{cgmes_to_network, load_profiles, CgmesNetwork, TapRegulationReport};
use gridoxide::outerloop::RegulationMode;

fn network(dir: &str) -> Option<CgmesNetwork> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/CGMES-Test-Configurations/v3.0")
        .join(dir);
    if !dir.exists() {
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

/// Svedala: eleven voltage-mode controls, every one enabled, nothing dropped.
/// The numbers come from the fixture's own SSH — eleven `TapChangerControl`
/// objects, all `enabled=true`, all `RegulatingControlModeKind.voltage`.
#[test]
fn svedala_reads_all_eleven_voltage_controls() {
    let Some(net) = network("Svedala/Svedala-Merged") else {
        eprintln!("skipping: Svedala fixture not checked out");
        return;
    };
    assert_eq!(
        net.tap_report,
        TapRegulationReport { converted: 11, ..Default::default() },
        "everything Svedala declares should convert, with nothing skipped"
    );
    assert_eq!(net.regulation.len(), 11);
    assert!(net.regulation.iter().all(|r| r.mode == RegulationMode::Voltage));
    assert!(net.regulation.iter().all(|r| r.enabled));

    // The four distinct targets the SSH states, converted against each
    // controlled bus's own nominal voltage: 129.5 kV on a 135 kV bus, 130 on
    // 135, 11 on 11, 51.6 on 51.6.
    let mut targets: Vec<String> =
        net.regulation.iter().map(|r| format!("{:.6}", r.target)).collect();
    targets.sort();
    targets.dedup();
    assert_eq!(targets, ["0.959259", "0.962963", "1.000000", "1.032000"]);

    // Deadbands are a full width in the same unit as the target, so they land
    // as a fraction of nominal rather than as the raw kV figure.
    for r in &net.regulation {
        assert!(
            r.deadband > 0.0 && r.deadband < 0.1,
            "deadband {} is not a plausible per-unit width",
            r.deadband
        );
    }
}

/// FullGrid exercises both modes and the disabled case at once: five
/// `TapChangerControl` objects, of which the SSH enables three, across a
/// voltage control and two phase shifters sharing one active-power target.
#[test]
fn fullgrid_reads_both_modes_and_counts_the_disabled() {
    let Some(net) = network("FullGrid/FullGrid-Merged") else {
        eprintln!("skipping: FullGrid fixture not checked out");
        return;
    };
    assert_eq!(
        net.tap_report,
        TapRegulationReport { converted: 3, disabled: 4, ..Default::default() },
        "three enabled controls, four switched off — and nothing unresolved"
    );

    let voltage: Vec<_> =
        net.regulation.iter().filter(|r| r.mode == RegulationMode::Voltage).collect();
    let power: Vec<_> = net
        .regulation
        .iter()
        .filter(|r| matches!(r.mode, RegulationMode::ActivePower { .. }))
        .collect();
    assert_eq!(voltage.len(), 1);
    assert_eq!(power.len(), 2);

    // 10.5 kV on a bus whose own nominal is 10.5 kV, so exactly 1.0 pu — and
    // the deadband, 0.5 kV against the same base, is 0.047619.
    assert!((voltage[0].target - 1.0).abs() < 1e-12, "{}", voltage[0].target);
    assert!((voltage[0].deadband - 0.5 / 10.5).abs() < 1e-12, "{}", voltage[0].deadband);

    // -65 MW on a 100 MVA base, deadband 35 MW. `targetValueUnitMultiplier`
    // is "M" and covers both fields — powsybl's own SSH export writes it that
    // way — so applying 1e6 on top of it, as an earlier draft did, put this
    // target at -650000 pu.
    for r in &power {
        assert!((r.target - -0.65).abs() < 1e-12, "{}", r.target);
        assert!((r.deadband - 0.35).abs() < 1e-12, "{}", r.deadband);
    }
    // Both shifters hold the *same* flow, which is what makes FullGrid the
    // shared-control case the phase loop has to size jointly.
    assert_eq!(power[0].mode, power[1].mode, "both should regulate one branch terminal");
    assert_ne!(power[0].transformer, power[1].transformer, "on two different transformers");
}

/// A control that exists but is switched off is data, not an error. PowerFlow's
/// single `TapChangerControl` has `TapChanger.controlEnabled = false`, so its
/// tap position is an input and the loop must never touch it.
#[test]
fn a_disabled_control_is_counted_not_converted() {
    let Some(net) = network("PowerFlow/PowerFlow") else {
        eprintln!("skipping: PowerFlow fixture not checked out");
        return;
    };
    assert_eq!(net.tap_report, TapRegulationReport { disabled: 1, ..Default::default() });
    assert!(net.regulation.is_empty(), "a disabled control must not become a live one");
    assert_eq!(
        net.tap_changers.iter().filter(|c| c.is_some()).count(),
        1,
        "the table is still retained — only the *control* is off"
    );
}

/// A tap changer with no `TapChangerControl` at all yields no regulation, and
/// must not be confused with a disabled one. SmallGrid's ten changers are all
/// like this, and so is FullGrid's `BE_TR2_HVDC2` — which matters, because the
/// published SV moves that tap and nothing here should try to reproduce it.
#[test]
fn a_changer_with_no_control_yields_nothing_and_is_not_counted() {
    let Some(net) = network("SmallGrid/SmallGrid-Merged") else {
        eprintln!("skipping: SmallGrid fixture not checked out");
        return;
    };
    assert_eq!(net.tap_report, TapRegulationReport::default(), "nothing to report at all");
    assert!(net.regulation.is_empty());
    assert_eq!(net.tap_changers.iter().filter(|c| c.is_some()).count(), 10);
}

/// A phase shifter's active-power target resolves to a branch flow, and the
/// branch index it names is a real branch of this conversion.
#[test]
fn an_active_power_target_names_a_branch_this_conversion_has() {
    for name in [
        "PST/PST_PhaseTapChangerLinear_Type1",
        "PST/PST_PhaseTapChangerLinear_Type2",
        "PST/PST_PhaseTapChangerTable_Type3",
    ] {
        let Some(net) = network(name) else { continue };
        assert_eq!(net.regulation.len(), 1, "{name}");
        let n_branches = net.lines.len() + net.transformers.len();
        match net.regulation[0].mode {
            RegulationMode::ActivePower { branch, .. } => {
                assert!(branch < n_branches, "{name}: branch {branch} of {n_branches}");
                assert!(branch >= net.lines.len(), "{name}: a PST regulates a transformer's flow");
            }
            m => panic!("{name}: expected an active-power control, got {m:?}"),
        }
    }
}

/// Two imports of the same file agree on transformer indices.
///
/// They did not before: `ends_by_pt` was a `HashMap`, whose iteration order
/// Rust randomizes per process, so the transformer list — and every flat branch
/// index derived from it — came out differently on each run. Nothing asserted
/// on a transformer index, so it never failed; it would have surfaced as a
/// `TapRegulation` naming a different element each run.
///
/// Bus indices are a separate matter and still renumber; that is why this
/// compares transformer indices and targets rather than controlled buses.
#[test]
fn two_imports_of_one_file_agree_on_transformer_indices() {
    let Some(a) = network("Svedala/Svedala-Merged") else {
        eprintln!("skipping: Svedala fixture not checked out");
        return;
    };
    let b = network("Svedala/Svedala-Merged").expect("second import");

    let key = |n: &CgmesNetwork| -> Vec<(usize, String)> {
        let mut v: Vec<(usize, String)> = n
            .regulation
            .iter()
            .map(|r| (r.transformer, format!("{:.9}", r.target)))
            .collect();
        v.sort();
        v
    };
    assert_eq!(key(&a), key(&b));

    let positions = |n: &CgmesNetwork| -> Vec<(usize, i32)> {
        n.tap_changers
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_ref().map(|c| (i, c.position)))
            .collect()
    };
    assert_eq!(positions(&a), positions(&b));
}
