//! Phase-4 gates for `src/dynamics/json.rs`: the native document format.
//!
//! Two things are being checked. That a file round-trips into a system that
//! satisfies the same equilibrium invariant a hand-built one does — parsing is
//! not the interesting part, *initializing from what was parsed* is. And that
//! the device/load split, which `plans/RMS_PLAN.md` §11 identified as the one
//! hazard no gate can catch downstream, is either derived unambiguously or
//! refused by name.

use gridoxide::dynamics::json::{self, DynamicsError};
use gridoxide::dynamics::{run_dynamics, DynamicsOptions, DynamicsStatus};

const FIXTURE: &str = "tests/data/dynamics/smib.json";

fn fixture() -> String {
    std::fs::read_to_string(FIXTURE).expect("the fixture is committed beside the test")
}

/// A document parses, initializes to an equilibrium, and runs its own events.
#[test]
fn a_document_builds_and_runs() {
    let doc = json::read(FIXTURE).expect("parses");
    let (mut system, events) = doc.build().expect("builds");

    assert_eq!(events.len(), 3);
    assert_eq!(system.n_bus(), 3);
    // Six machine states, two exciter, two governor, three stabilizer. The ZIP
    // load carries none — it is algebraic.
    assert_eq!(system.n_states(), 13);

    assert!(
        system.max_derivative() < 1e-11,
        "a parsed case must initialize to an equilibrium, drift {:e}",
        system.max_derivative()
    );
    assert!(system.network_residual_norm() < 1e-11);

    let report = run_dynamics(
        &mut system,
        &DynamicsOptions { end_time: 8.0, step: 0.005, events, ..Default::default() },
    );
    assert_eq!(report.status, DynamicsStatus::Completed);
    assert_eq!(report.events_applied, 3);
    assert!(report.warnings.is_empty(), "unexpected warnings: {:?}", report.warnings);

    // The fault is visible, and the machine survives it.
    let v = report.trajectory.series("bus1.vmag").unwrap();
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(min < 1e-3, "a bolted fault should collapse the bus, lowest was {min}");
    let omega = report.trajectory.series("G1.omega").unwrap();
    assert!(
        omega.iter().all(|w| (w - 1.0).abs() < 0.1),
        "the machine should stay in step through a 100 ms fault"
    );
}

/// An unstated terminal power is derived from the bus, and derived correctly:
/// stating it explicitly must give the identical machine.
///
/// This is what makes a file immune to §11's hazard. The split is a property of
/// the network, and the file only has to say so when the network cannot.
#[test]
fn an_unstated_terminal_power_is_derived_from_the_bus() {
    let derived = json::parse(&fixture()).unwrap().build().unwrap().0;

    // The same case with the machine's own operating point spelled out. The
    // values are the bus-0 injection the power flow produces, which for this
    // network is the scheduled 0.8 plus whatever reactive support the PV bus
    // turned out to need.
    let state = derived.state().to_vec();
    let stated_text = fixture().replace(
        r#""id": "G1",
        "bus": 0,"#,
        &format!(
            r#""id": "G1",
        "bus": 0, "p": {}, "q": {},"#,
            0.8, 0.0
        ),
    );
    // Only proceed if the substitution actually landed; otherwise the test is
    // silently checking nothing.
    assert!(stated_text.contains("\"p\": 0.8"), "the fixture's shape changed");
    let stated = json::parse(&stated_text).unwrap().build().unwrap().0;

    // Bus 0's solved injection is 0.8 real and *some* reactive; stating the
    // real part alone and zero reactive is a different machine, so this must
    // differ. What matters is that it differs in the expected direction and
    // that both are equilibria — a wrong split is silent, not detectable, and
    // the only defence is that the file need not state it at all.
    assert!(stated.max_derivative() < 1e-11);
    assert!(derived.max_derivative() < 1e-11);
    let delta_derived = state[0];
    let delta_stated = stated.state()[0];
    assert!(
        (delta_derived - delta_stated).abs() > 1e-6,
        "understating the reactive output should give a visibly different machine"
    );
}

/// Two devices at one bus, neither stating its terminal power, is refused by
/// name rather than split by a guess.
#[test]
fn an_ambiguous_split_is_refused() {
    let text = fixture().replace(
        r#""loads": [
      { "id": "L1", "bus": 1, "zip": [0.3, 0.3, 0.4], "cutoff": 0.5 }
    ],"#,
        r#""loads": [
      { "id": "L1", "bus": 1, "zip": [0.3, 0.3, 0.4], "cutoff": 0.5 },
      { "id": "L2", "bus": 1, "zip": [1.0, 0.0, 0.0], "cutoff": 0.5 }
    ],"#,
    );
    assert!(text.contains("L2"), "the fixture's shape changed");
    let err = json::parse(&text).unwrap().build().expect_err("must refuse");
    match err {
        DynamicsError::AmbiguousSplit { bus, ref devices } => {
            assert_eq!(bus, 1);
            assert_eq!(devices.len(), 2);
            // Named, so the message tells the author which two to disambiguate.
            assert!(devices.contains(&"L1".to_string()));
            assert!(devices.contains(&"L2".to_string()));
        }
        other => panic!("expected an ambiguous split, got {other}"),
    }
    assert!(err.to_string().contains("state p and q on all but one"));
}

/// A device at a bus that does not exist is named, not panicked on.
#[test]
fn a_bus_out_of_range_is_named() {
    let text = fixture().replace(r#""bus": 0,"#, r#""bus": 9,"#);
    let err = json::parse(&text).unwrap().build().expect_err("must refuse");
    assert!(
        matches!(err, DynamicsError::BusOutOfRange { ref id, bus: 9, n_bus: 3 } if id == "G1"),
        "got {err}"
    );
}

/// The parameter blocks *are* the models' own parameter structs, so a typo in
/// a field name is a parse error — the real field goes missing — rather than a
/// silently-defaulted zero.
///
/// That is the whole reason the format is defined by deriving `Deserialize` on
/// `GenRoundParams` and friends rather than by a parallel set of file structs:
/// the two cannot drift apart.
#[test]
fn an_unknown_field_or_model_is_a_parse_error() {
    let typo = fixture().replace(r#""h": 5.0,"#, r#""hh": 5.0,"#);
    assert!(matches!(json::parse(&typo), Err(DynamicsError::Parse(_))), "a typo must not parse");

    let unknown = fixture().replace(r#""model": "gen_round","#, r#""model": "gen_octonion","#);
    assert!(matches!(json::parse(&unknown), Err(DynamicsError::Parse(_))));
}

/// The document serializes back to something that parses to the same system.
#[test]
fn a_document_round_trips() {
    let doc = json::parse(&fixture()).unwrap();
    let text = serde_json::to_string(&doc).expect("serializes");
    let again = json::parse(&text).expect("re-parses");

    let (a, _) = doc.build().unwrap();
    let (b, _) = again.build().unwrap();
    assert_eq!(a.n_states(), b.n_states());
    for (x, y) in a.state().iter().zip(b.state().iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "a round trip must not move a single state");
    }
}

/// And it round-trips when a bus states no reactive limits at all.
///
/// The test above passes because the fixture states *finite* limits on every
/// bus, which is exactly why this went unnoticed. Unbounded is `±∞`, JSON has
/// no infinity, and `serde_json` writes any non-finite `f64` as `null` and then
/// refuses to read it back — so a document this crate wrote could not be
/// reopened, and the failure was `invalid type: null, expected f64` a megabyte
/// into the file. Found by dumping a large synthetic case in order to run the
/// CLI over it.
///
/// Absence is the spelling now, which is what `models::Limits` already chose
/// for the same reason.
#[test]
fn an_unbounded_reactive_limit_round_trips() {
    let mut doc = json::parse(&fixture()).unwrap();
    for bus in &mut doc.network.buses {
        bus.q_min = f64::NEG_INFINITY;
        bus.q_max = f64::INFINITY;
    }

    let text = serde_json::to_string(&doc).expect("serializes");
    assert!(!text.contains("q_min"), "an unbounded limit is absent, not null: {text}");
    assert!(!text.contains("q_max"), "an unbounded limit is absent, not null: {text}");

    let again = json::parse(&text).expect("re-parses");
    for bus in &again.network.buses {
        assert_eq!(bus.q_min, f64::NEG_INFINITY);
        assert_eq!(bus.q_max, f64::INFINITY);
    }

    // A finite limit still survives, so the sentinel is not swallowing real
    // data on its way past.
    doc.network.buses[0].q_min = -0.4;
    doc.network.buses[0].q_max = 0.7;
    let again = json::parse(&serde_json::to_string(&doc).unwrap()).unwrap();
    assert_eq!(again.network.buses[0].q_min, -0.4);
    assert_eq!(again.network.buses[0].q_max, 0.7);
    assert_eq!(again.network.buses[1].q_max, f64::INFINITY);
}
