//! Phase-4 gates for `src/dynamics/dyd.rs`: Dynawo's `.dyd`/`.par`/`.crv`.
//!
//! The fixtures under `tests/data/dynamics/dynawo/` are copied **verbatim**
//! from Dynawo's own repository — see the `PROVENANCE.md` beside them. That is
//! the point of this suite. A `.par` is addressed by parameter *name*, so the
//! only real risk is looking up a name that no real file uses, and a
//! hand-written fixture would agree with whatever the reader happened to
//! expect. These do not.

use std::collections::HashMap;

use gridoxide::dynamics::dyd::{self, DydError, DydWarning};
use gridoxide::dynamics::json::{AvrSpec, GovSpec, MachineSpec};

const DIR: &str = "tests/data/dynamics/dynawo";

fn documents() -> (dyd::DydDocument, dyd::ParDocument) {
    (
        dyd::read_dyd(format!("{DIR}/IEEE14.dyd")).expect("parses the .dyd"),
        dyd::read_par(format!("{DIR}/IEEE14.par")).expect("parses the .par"),
    )
}

/// Every generator's `staticId` mapped to a bus index, which is what the IIDM
/// half of a real case would supply.
fn bus_of(dyd: &dyd::DydDocument) -> HashMap<String, usize> {
    dyd.models
        .iter()
        .filter(|m| m.lib.starts_with("GeneratorSynchronous"))
        .enumerate()
        .filter_map(|(i, m)| m.static_id.clone().map(|s| (s, i)))
        .collect()
}

/// The architecture file yields the models, their libraries and their
/// references.
#[test]
fn the_architecture_file_yields_its_models() {
    let (dyd, _) = documents();
    assert_eq!(dyd.models.len(), 19, "the vendored case has nineteen black boxes");

    let generators: Vec<_> =
        dyd.models.iter().filter(|m| m.lib.starts_with("GeneratorSynchronous")).collect();
    assert_eq!(generators.len(), 5);

    let g1 = generators.iter().find(|m| m.id == "GEN____1_SM").expect("GEN____1_SM is present");
    assert_eq!(g1.lib, "GeneratorSynchronousFourWindingsProportionalRegulations");
    assert_eq!(g1.par_id.as_deref(), Some("Generator1"));
    assert_eq!(g1.par_file.as_deref(), Some("IEEE14.par"));
    assert_eq!(
        g1.static_id.as_deref(),
        Some("_GEN____1_SM"),
        "the staticId is what ties a dynamic model to the IIDM network"
    );

    // Loads share one parameter set between many models, which is the whole
    // reason parId is separate from id.
    let loads: Vec<_> = dyd.models.iter().filter(|m| m.lib.starts_with("Load")).collect();
    assert!(loads.len() > 5);
    assert!(loads.iter().all(|m| m.par_id.is_some()));
}

/// The parameter sets are found by name, and carry the names this reader looks
/// for.
#[test]
fn the_parameter_file_carries_the_expected_names() {
    let (_, par) = documents();
    assert!(par.sets.contains_key("Generator1"));

    // Spot-check against the file's own literal values. If Dynawo ever renames
    // one of these, this fails here rather than as a machine that silently
    // initializes with a zero reactance.
    for (name, expected) in [
        ("generator_H", 5.4),
        ("generator_RaPu", 0.002796),
        ("generator_XdPu", 2.22),
        ("generator_XpdPu", 0.384),
        ("generator_XppdPu", 0.264),
        ("generator_XqPu", 2.22),
        ("generator_XpqPu", 0.393),
        ("generator_XppqPu", 0.262),
        ("generator_XlPu", 0.202),
        ("generator_Tpd0", 8.094),
        ("generator_Tppd0", 0.08),
        ("generator_Tpq0", 1.572),
        ("generator_Tppq0", 0.084),
        ("generator_SNom", 1211.0),
        ("voltageRegulator_Gain", 20.0),
        ("governor_KGover", 5.0),
        ("governor_PNom", 1090.0),
    ] {
        let got = par
            .number("Generator1", name)
            .unwrap_or_else(|| panic!("Generator1 has no `{name}`"));
        assert_eq!(got, expected, "{name}");
    }
}

/// A four-windings generator maps onto the sixth-order machine, parameter for
/// parameter, with `generator_SNom` as its MVA base.
#[test]
fn a_four_windings_generator_maps_onto_the_subtransient_machine() {
    let (dyd, par) = documents();
    let (units, _) = dyd::to_units(&dyd, &par, &bus_of(&dyd), 100.0).expect("converts");

    let g1 = units.iter().find(|u| u.id == "GEN____1_SM").expect("mapped");
    match g1.machine {
        MachineSpec::GenRound(p) => {
            assert_eq!(p.h, 5.4);
            assert_eq!(p.d, 0.0);
            assert_eq!(p.ra, 0.002796);
            assert_eq!(p.xd, 2.22);
            assert_eq!(p.xq, 2.22);
            assert_eq!(p.xdp, 0.384);
            assert_eq!(p.xqp, 0.393);
            assert_eq!(p.xdpp, 0.264);
            assert_eq!(p.xqpp, 0.262, "unlike GENROU, Dynawo states the two separately");
            assert_eq!(p.xl, 0.202);
            assert_eq!(p.td0p, 8.094);
            assert_eq!(p.tq0pp, 0.084);
            assert_eq!(p.mbase, 1211.0, "SNom is the machine's own base");
        }
        ref other => panic!("expected a subtransient machine, got {other:?}"),
    }

    // The parameters really do satisfy the ordering the model requires, which
    // is a check on the mapping as much as on the data: swapping a transient
    // and a subtransient reactance would break it.
    let params = match g1.machine {
        MachineSpec::GenRound(p) => p,
        _ => unreachable!(),
    };
    assert!(params.xl < params.xdpp);
    assert!(params.xdpp < params.xdp);
    assert!(params.xdp < params.xd);
    assert!(params.xl < params.xqpp);
    assert!(params.xqpp < params.xqp);
    assert!(params.xqp < params.xq);
}

/// Dynawo's proportional regulators map onto this library's proportional
/// regulators exactly, rather than onto lagged ones with invented time
/// constants.
///
/// The governor's gain is the one quantity converted rather than read:
/// `KGover` is on the machine's own `PNom`, and `GoverProportional` wants a
/// gain on the network base.
#[test]
fn proportional_regulators_map_exactly() {
    let (dyd, par) = documents();
    let (units, _) = dyd::to_units(&dyd, &par, &bus_of(&dyd), 100.0).unwrap();
    let g1 = units.iter().find(|u| u.id == "GEN____1_SM").unwrap();

    match g1.avr {
        Some(AvrSpec::VrProportional { k, limits }) => {
            assert_eq!(k, 20.0);
            // Carried straight through: gridoxide's own initialization
            // reproduces Dynawo's `efdPu` exactly, so the two agree on what a
            // per-unit field voltage is.
            assert_eq!(limits.min, Some(-5.0));
            assert_eq!(limits.max, Some(1.44));
        }
        ref other => panic!("expected a proportional regulator, got {other:?}"),
    }
    match g1.gov {
        Some(GovSpec::GoverProportional { k, limits }) => {
            // 5 × 1090 / 100.
            assert!((k - 54.5).abs() < 1e-9, "governor gain was {k}");
            // Stated in MW, so divided by the network base.
            assert_eq!(limits.min, Some(0.0));
            assert!((limits.max.unwrap() - 10.9).abs() < 1e-9);
        }
        ref other => panic!("expected a proportional governor, got {other:?}"),
    }
}

/// Every generator in the case maps: three four-windings onto the sixth-order
/// machine, two three-windings onto the fifth-order one.
///
/// A three-windings parameter set carries no `XpqPu` and no `Tpq0`, because a
/// salient-pole rotor has no `q`-axis transient to have a time constant for.
/// The reader keys on the **library name** rather than on which parameters
/// happen to be present, which is the difference between reading a model and
/// guessing one from its data.
#[test]
fn both_generator_libraries_map() {
    let (dyd, par) = documents();
    let (units, warnings) = dyd::to_units(&dyd, &par, &bus_of(&dyd), 100.0).unwrap();

    assert_eq!(units.len(), 5, "all five generators should map now");
    assert_eq!(
        warnings.iter().filter(|w| matches!(w, DydWarning::UnsupportedLib { .. })).count(),
        0,
        "nothing in this case is unsupported any more"
    );

    let round = units.iter().filter(|u| matches!(u.machine, MachineSpec::GenRound(_))).count();
    let salient = units.iter().filter(|u| matches!(u.machine, MachineSpec::GenSalient(_))).count();
    assert_eq!((round, salient), (3, 2));

    // The fifth-order machine's own values, from the file.
    let g6 = units.iter().find(|u| u.id == "GEN____6_SM").expect("mapped");
    match g6.machine {
        MachineSpec::GenSalient(p) => {
            assert_eq!(p.h, 4.975);
            assert_eq!(p.ra, 0.004);
            assert_eq!(p.xd, 0.75);
            assert_eq!(p.xdp, 0.225);
            assert_eq!(p.xdpp, 0.154);
            assert_eq!(p.xq, 0.45);
            assert_eq!(p.xqpp, 0.2, "the one q-axis reactance there is");
            assert_eq!(p.xl, 0.102);
            assert_eq!(p.td0p, 3.0);
            assert_eq!(p.td0pp, 0.04);
            assert_eq!(p.tq0pp, 0.04, "the one q-axis time constant there is");
            assert_eq!(p.mbase, 80.0);
        }
        ref other => panic!("expected a salient-pole machine, got {other:?}"),
    }
}

/// Saturation is stated as an exponential characteristic here and as a pair of
/// points in PSS/E — two incompatible representations of the same physics, and
/// exactly the divergence `plans/RMS_PLAN.md` §7 predicted. Neither is
/// modelled, and both are reported.
#[test]
fn saturation_and_limits_are_reported() {
    let (dyd, par) = documents();
    let (_, warnings) = dyd::to_units(&dyd, &par, &bus_of(&dyd), 100.0).unwrap();

    let saturation: Vec<&DydWarning> = warnings
        .iter()
        .filter(|w| matches!(w, DydWarning::SaturationIgnored { .. }))
        .collect();
    assert!(!saturation.is_empty(), "the vendored case states nonzero md/mq");
    assert!(saturation[0].to_string().contains("exponential saturation"));

    // Limits are carried, not dropped, so nothing warns about them.
    let limits = warnings.iter().filter(|w| matches!(w, DydWarning::LimitsIgnored { .. })).count();
    assert_eq!(limits, 0, "limits reach the models now");
}

/// A `staticId` with no bus is refused by name — the reader does not invent the
/// correspondence the IIDM half of the case carries.
#[test]
fn an_unmapped_static_id_is_refused() {
    let (dyd, par) = documents();
    let mut partial = bus_of(&dyd);
    partial.remove("_GEN____1_SM");
    let err = dyd::to_units(&dyd, &par, &partial, 100.0).expect_err("must refuse");
    assert!(
        matches!(err, DydError::UnknownStaticId { ref static_id, .. } if static_id == "_GEN____1_SM"),
        "got {err}"
    );
}

/// A missing parameter is named, since in a keyword-addressed format that is
/// the failure mode a wrong lookup produces.
#[test]
fn a_missing_parameter_is_named() {
    let (dyd, mut par) = documents();
    par.sets.get_mut("Generator1").unwrap().remove("generator_XppqPu");
    let err = dyd::to_units(&dyd, &par, &bus_of(&dyd), 100.0).expect_err("must refuse");
    assert!(
        matches!(err, DydError::MissingParameter { ref name, .. } if name == "generator_XppqPu"),
        "got {err}"
    );
}

/// The curve file says what a run should record.
#[test]
fn the_curve_file_yields_its_requests() {
    let curves = dyd::read_crv(format!("{DIR}/IEEE14.crv")).expect("parses");
    assert!(!curves.is_empty());
    assert!(
        curves.iter().any(|c| c.model == "NETWORK" && c.variable.contains("_U_value")),
        "the case asks for bus voltages"
    );
    // This particular case is a tap-changer study, so what it asks for beyond
    // the bus voltages is load and tap-changer state rather than machine
    // speeds. Reading that faithfully is the point — a reader that only
    // recognized generator variables would quietly return a third of the file.
    assert!(
        curves.iter().any(|c| c.variable.contains("tapChanger_tap")),
        "and for the tap-changer states this case is about"
    );
    assert!(curves.iter().any(|c| c.variable == "load_PPu"));
    assert!(
        curves.iter().all(|c| !c.model.is_empty() && !c.variable.is_empty()),
        "every request should carry both halves"
    );
}

/// A whole Dynawo case — the IIDM network and the `.dyd`/`.par` pair — loads in
/// one call and initializes to an equilibrium.
///
/// This is what the two halves were always for. `src/iidm.rs` reads the static
/// network, the reader above reads the dynamic models, and the `staticId` on
/// each `blackBoxModel` is the correspondence between them.
#[test]
fn a_whole_case_loads_from_its_two_halves() {
    let (system, warnings) = dyd::load_case(
        format!("{DIR}/IEEE14.iidm"),
        format!("{DIR}/IEEE14.dyd"),
        format!("{DIR}/IEEE14.par"),
        50.0,
    )
    .expect("the case loads");

    assert_eq!(system.n_bus(), 14);
    // Three sixth-order machines and two fifth-order ones. The proportional
    // regulators contribute no states at all, which is what they are.
    assert_eq!(system.n_states(), 3 * 6 + 2 * 5);

    // The invariant, on somebody else's data and somebody else's network.
    assert!(
        system.max_derivative() < 1e-11,
        "a case assembled from two files must still initialize to an equilibrium, \
         drift {:e}",
        system.max_derivative()
    );
    assert!(system.network_residual_norm() < 1e-11);

    // What it cannot model, it names: saturation on every machine, and the
    // tap-changing load models it treats as static.
    // Four of the five machines state a nonzero saturation characteristic;
    // `Generator8` states `md = 0`, so there is nothing to drop and nothing to
    // warn about. Reporting only what was actually discarded is the point —
    // a warning on a machine with no saturation would be noise a reader learns
    // to ignore.
    let saturation =
        warnings.iter().filter(|w| matches!(w, DydWarning::SaturationIgnored { .. })).count();
    assert_eq!(saturation, 4, "four of the five machines state nonzero saturation");
    let skipped: Vec<&DydWarning> =
        warnings.iter().filter(|w| matches!(w, DydWarning::UnsupportedLib { .. })).collect();
    assert!(
        skipped.iter().any(|w| w.to_string().contains("TapChanger")),
        "the tap-changing loads should be named, not silently treated as static: {skipped:?}"
    );
}

/// Each machine's terminal power is recovered exactly, not apportioned.
///
/// A power flow gives a bus's total; a dynamic study needs the machine's own.
/// For a Dynawo case that is not a guess — the IIDM states every load's `p0`
/// and `q0`, and those are precisely what went into the bus's specification, so
/// subtracting them leaves the machine's share exactly, at a `PV` bus as much
/// as a `PQ` one.
///
/// The check is the equilibrium itself: a wrong split is silent everywhere else
/// (see `plans/RMS_PLAN.md` §11), but here it would have to be *consistent*
/// with a network the file also fixes, and it is not free to be.
#[test]
fn a_machine_sharing_its_bus_with_a_load_still_initializes() {
    let (system, _) = dyd::load_case(
        format!("{DIR}/IEEE14.iidm"),
        format!("{DIR}/IEEE14.dyd"),
        format!("{DIR}/IEEE14.par"),
        50.0,
    )
    .unwrap();

    // Bus 5 carries generator 1; the case puts load on several machine buses,
    // so at least one machine is sharing.
    let network = gridoxide::iidm::read(format!("{DIR}/IEEE14.iidm")).unwrap();
    let generator_buses: Vec<usize> =
        network.injections.iter().filter(|i| i.generator).map(|i| i.bus).collect();
    let shared = network
        .injections
        .iter()
        .filter(|i| !i.generator && generator_buses.contains(&i.bus))
        .count();
    assert!(shared > 0, "the case should have at least one machine sharing a bus with a load");

    assert!(system.max_derivative() < 1e-11);
}
