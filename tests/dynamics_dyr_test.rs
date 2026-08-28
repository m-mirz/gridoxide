//! Phase-4 gates for `src/dynamics/dyr.rs`: PSS/E dynamic records.
//!
//! The interesting risks in a positional format are not "does it parse" but
//! "does field seven mean what the reader thinks it means". So the gates check
//! the *mapping* against the model parameters it produces, and check that the
//! two things a `.dyr` cannot say — the machine's MVA base and its armature
//! resistance, both of which live in the `.raw` — are demanded from the caller
//! rather than invented.

use std::collections::HashMap;

use gridoxide::dynamics::dyr::{self, DyrError, DyrWarning, MachineData};
use gridoxide::dynamics::json::{
    AvrSpec, DynamicsData, DynamicsDocument, GovSpec, MachineSpec,
};
use gridoxide::dynamics::{run_dynamics, DynamicsOptions, DynamicsStatus};
use gridoxide::json::NetworkData;
use gridoxide::types::{Bus, BusType, Line};

const FIXTURE: &str = "tests/data/dynamics/two_machine.dyr";

fn supplied() -> HashMap<(i64, String), (MachineData, f64)> {
    HashMap::from([
        ((1, "1".to_string()), (MachineData { mbase: 100.0, ra: 0.003 }, 0.30)),
        ((2, "1".to_string()), (MachineData { mbase: 100.0, ra: 0.0 }, 0.30)),
    ])
}

fn bus_index() -> HashMap<i64, usize> {
    HashMap::from([(1, 0), (2, 1)])
}

/// Records are found, and a record spanning three lines is one record.
///
/// Only the `/` ends a record in this format, so a reader that works line by
/// line silently produces truncated parameter lists — which then fail an arity
/// check somewhere far from the cause, if at all.
#[test]
fn records_span_lines_and_end_at_the_slash() {
    let doc = dyr::read(FIXTURE).expect("parses");
    assert_eq!(doc.records.len(), 5, "{:#?}", doc.records);

    let genrou = &doc.records[0];
    assert_eq!(genrou.bus, 1);
    assert_eq!(genrou.model, "GENROU");
    assert_eq!(genrou.id, "1", "the quoted identifier should be unquoted and trimmed");
    assert_eq!(genrou.params.len(), 14, "a three-line GENROU record has all 14 parameters");

    assert_eq!(doc.records[3].model, "GENCLS");
    assert_eq!(doc.records[3].bus, 2);
    assert_eq!(doc.records[4].model, "IEEEST");
}

/// The positional mapping, checked field by field against the model it
/// produces.
///
/// `GENROU` gives no `X''q`: a round-rotor machine takes the two subtransient
/// reactances equal, which is what makes it round. And `SEXS`'s first field is
/// the **ratio** `T_a/T_b`, not a time constant — reading it as one produces a
/// plausible exciter that is an order of magnitude too fast.
#[test]
fn the_positional_mapping_is_right() {
    let doc = dyr::read(FIXTURE).unwrap();
    let (units, _) = dyr::to_units(&doc, &bus_index(), &supplied()).expect("converts");
    assert_eq!(units.len(), 2);

    let g1 = &units[0];
    assert_eq!(g1.bus, 0);
    match g1.machine {
        MachineSpec::GenRound(p) => {
            assert_eq!(p.td0p, 8.0);
            assert_eq!(p.td0pp, 0.03);
            assert_eq!(p.tq0p, 0.4);
            assert_eq!(p.tq0pp, 0.05);
            assert_eq!(p.h, 5.0);
            assert_eq!(p.d, 1.0);
            assert_eq!(p.xd, 1.8);
            assert_eq!(p.xq, 1.7);
            assert_eq!(p.xdp, 0.30);
            assert_eq!(p.xqp, 0.55);
            assert_eq!(p.xdpp, 0.22);
            assert_eq!(p.xqpp, 0.22, "GENROU takes X''q = X''d");
            assert_eq!(p.xl, 0.15);
            // Supplied by the caller, since the file cannot say.
            assert_eq!(p.ra, 0.003);
            assert_eq!(p.mbase, 100.0);
        }
        other => panic!("expected a round-rotor machine, got {other:?}"),
    }

    match g1.avr {
        Some(AvrSpec::Sexs(p)) => {
            assert_eq!(p.k, 200.0);
            assert_eq!(p.tb, 1.0);
            assert_eq!(p.te, 0.05);
            assert_eq!(p.ta, 0.1, "T_a is the ratio times T_b, not the field itself");
        }
        ref other => panic!("expected SEXS, got {other:?}"),
    }
    match g1.gov {
        Some(GovSpec::Tgov1(p)) => {
            assert_eq!(p.r, 0.05);
            assert_eq!(p.t1, 0.5);
            assert_eq!(p.t2, 1.0);
            assert_eq!(p.t3, 5.0);
            assert_eq!(p.dt, 0.0);
        }
        ref other => panic!("expected TGOV1, got {other:?}"),
    }

    let g2 = &units[1];
    assert_eq!(g2.bus, 1);
    assert!(matches!(g2.machine, MachineSpec::GenCls(p) if p.h == 4.0 && p.xdp == 0.30));
    assert!(g2.avr.is_none() && g2.gov.is_none());
}

/// What the reader drops, it names.
///
/// Saturation and output limits are both absent from this model library. Both
/// change answers, so a silent drop would show up later as a small unexplained
/// disagreement with a reference — the exact failure mode
/// `plans/RMS_PLAN.md` §7 warns about.
#[test]
fn dropped_data_is_reported() {
    let doc = dyr::read(FIXTURE).unwrap();
    let (_, warnings) = dyr::to_units(&doc, &bus_index(), &supplied()).unwrap();

    let saturation = warnings
        .iter()
        .filter(|w| matches!(w, DyrWarning::SaturationIgnored { .. }))
        .count();
    assert_eq!(saturation, 1, "the GENROU's nonzero S(1.0)/S(1.2) should be reported");

    let limits =
        warnings.iter().filter(|w| matches!(w, DyrWarning::LimitsIgnored { .. })).count();
    assert_eq!(limits, 2, "SEXS and TGOV1 both state limits this library ignores");

    let unsupported: Vec<&DyrWarning> = warnings
        .iter()
        .filter(|w| matches!(w, DyrWarning::UnsupportedModel { .. }))
        .collect();
    assert_eq!(unsupported.len(), 1);
    assert!(
        unsupported[0].to_string().contains("IEEEST"),
        "an unimplemented model should be skipped by name: {}",
        unsupported[0]
    );
}

/// The two things a `.dyr` cannot state are demanded, not invented.
#[test]
fn missing_machine_data_is_refused_by_name() {
    let doc = dyr::read(FIXTURE).unwrap();
    let mut partial = supplied();
    partial.remove(&(2, "1".to_string()));
    let err = dyr::to_units(&doc, &bus_index(), &partial).expect_err("must refuse");
    assert!(
        matches!(err, DyrError::MissingMachineData { bus: 2, ref id } if id == "1"),
        "got {err}"
    );
    assert!(err.to_string().contains("ZSORCE"), "the message should say where they live");

    let mut short = bus_index();
    short.remove(&2);
    let err = dyr::to_units(&doc, &short, &supplied()).expect_err("must refuse");
    assert!(matches!(err, DyrError::UnknownBus { bus: 2 }), "got {err}");
}

/// A record with the wrong number of parameters is caught with its line
/// number, rather than reading field seven as field eight.
#[test]
fn a_short_record_is_caught_with_its_line() {
    let text = "1 'TGOV1' '1 ' 0.05 0.5 1.0 /\n";
    let doc = dyr::parse(text).unwrap();
    let err = dyr::to_units(&doc, &bus_index(), &supplied()).expect_err("must refuse");
    match err {
        DyrError::OrphanedControl { .. } => {}
        other => panic!("a control with no machine should be an orphan first, got {other}"),
    }

    // Same bus as the machine this time, so the orphan check passes and the
    // arity check is what fires.
    let with_machine = "2 'GENCLS' '1 ' 4.0 0.0 /\n2 'TGOV1' '1 ' 0.05 0.5 1.0 /\n".to_string();
    let doc = dyr::parse(&with_machine).unwrap();
    let err = dyr::to_units(&doc, &bus_index(), &supplied()).expect_err("must refuse");
    assert!(
        matches!(err, DyrError::WrongArity { line: 2, expected: 7, found: 3, .. }),
        "got {err}"
    );
}

/// End to end: a `.dyr` drives a run.
///
/// The point is not that the numbers arrive but that a case assembled from them
/// satisfies the same equilibrium invariant a hand-built one does. A misplaced
/// parameter usually produces a machine that still parses and still looks like
/// a machine, and this is the first thing that would notice.
#[test]
fn a_dyr_case_initializes_and_runs() {
    let doc = dyr::read(FIXTURE).unwrap();
    let (units, _) = dyr::to_units(&doc, &bus_index(), &supplied()).unwrap();

    let network = NetworkData {
        buses: vec![
            make_bus(0, BusType::Slack, 0.0, 0.0),
            make_bus(1, BusType::PV, 0.5, 0.0),
            make_bus(2, BusType::PQ, -1.2, -0.4),
        ],
        lines: vec![
            Line { from: 0, to: 2, r: 0.005, x: 0.08, b_shunt: 0.0, g_shunt: 0.0 },
            Line { from: 1, to: 2, r: 0.005, x: 0.08, b_shunt: 0.0, g_shunt: 0.0 },
        ],
    };

    let document = DynamicsDocument {
        network,
        dynamics: DynamicsData {
            s_base: 100.0,
            f_nom: 50.0,
            units,
            loads: Vec::new(),
            fixed_buses: Vec::new(),
            events: Vec::new(),
        },
    };

    let (mut system, _) = document.build().expect("builds");
    assert!(
        system.max_derivative() < 1e-11,
        "a case assembled from a .dyr must initialize to an equilibrium, drift {:e}",
        system.max_derivative()
    );
    assert!(system.network_residual_norm() < 1e-11);
    // Six states for the round rotor, two each for its exciter and governor,
    // two for the classical machine.
    assert_eq!(system.n_states(), 12);

    let report = run_dynamics(
        &mut system,
        &DynamicsOptions { end_time: 5.0, step: 0.005, ..Default::default() },
    );
    assert_eq!(report.status, DynamicsStatus::Completed);
    for name in ["1_1.omega", "2_1.omega"] {
        let series = report.trajectory.series(name).unwrap();
        let drift = series.iter().fold(0.0f64, |m, w| m.max((w - series[0]).abs()));
        assert!(drift < 1e-9, "{name} drifted by {drift:e} with no disturbance");
    }
}

fn make_bus(idx: usize, bus_type: BusType, p: f64, q: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: q,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: Vec::new(),
    }
}
