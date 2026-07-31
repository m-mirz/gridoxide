//! Checks the measurement model against power-grid-model's own answers, with
//! no estimator involved.
//!
//! Each state-estimation fixture ships the state PGM converged to. Evaluating
//! every measurement function `h(x)` at *that* state and comparing against the
//! measured value `z` tests the whole modelling chain end to end — branch-flow
//! formulas, terminal resolution, sign conventions, per-unit scaling, sensor
//! aggregation — while the true answer is sitting right there to compare
//! against.
//!
//! This is deliberately done before any estimator exists, because of the
//! failure mode it avoids. A flipped sign or a missing per-unit factor, met
//! later through a solver, looks exactly like a Jacobian bug or an
//! unobservable system; met here, it names the individual measurement that
//! disagrees and by how much.
//!
//! # What "agreement" means
//!
//! Residuals are compared against each measurement's own sigma, not against a
//! fixed tolerance. That is the natural scale: the fixtures' measured values
//! are *noisy* by construction — that is the point of state estimation — so
//! `h(x) == z` is not expected even with a perfect model. What is expected is
//! that the disagreement stays within the uncertainty the sensor itself
//! declares. A modelling error shows up as a residual of many sigma, or as one
//! that scales with the measurement instead of with its noise.

mod common;

use std::path::{Path, PathBuf};

use num_complex::Complex;

use gridoxide::measurement::measurements_from_pgm;
use gridoxide::network::{build_ybus, source_impedance_pu, stamp_shunts};
use gridoxide::pgm::{pgm_shunts_1ph, pgm_to_network, PgmInput, PgmNetwork};
use gridoxide::se::{measurement_functions, SeNetwork};
use gridoxide::types::Bus;

const S_BASE_VA: f64 = 1e6;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/state_estimation")
        .join(name)
}

/// Populates `buses` with the state power-grid-model reports.
///
/// Physical nodes come straight from the `node` output. The virtual slack bus
/// gridoxide synthesizes per source has no PGM counterpart, so it is
/// reconstructed instead: the source's own reported `p`/`q` give the current
/// arriving at its node, and pushing that back through the source impedance
/// gives the voltage behind it, `V_virtual = V_node + I·Z_s`.
///
/// Without that step every bus-injection measurement at a source node would
/// have to be skipped, since its `h(x)` reads the whole Y-bus row — including
/// the virtual neighbour.
fn apply_expected_state(buses: &mut [Bus], net: &PgmNetwork, input: &PgmInput, expected: &serde_json::Value) {
    for node in expected["data"]["node"].as_array().expect("node output") {
        let id = node["id"].as_u64().expect("node id");
        let idx = net.node_idx[&id];
        buses[idx].voltage_mag = node["u_pu"].as_f64().expect("u_pu");
        buses[idx].voltage_ang = node["u_angle"].as_f64().expect("u_angle");
    }

    let source_out = expected["data"]["source"].as_array();
    for src in input.data.source.iter().filter(|s| s.status != 0) {
        let Some(out) = source_out.and_then(|arr| {
            arr.iter().find(|o| o["id"].as_u64() == Some(src.id))
        }) else {
            continue;
        };
        let branch = net.source_branch_idx[&src.id];
        // The synthesized branch runs virtual -> node, so its `from` is the
        // virtual bus.
        let virtual_idx = net.lines[branch].from;
        let node_idx = net.node_idx[&src.node];

        let v_node = Complex::from_polar(buses[node_idx].voltage_mag, buses[node_idx].voltage_ang);
        let s = Complex::new(
            out["p"].as_f64().unwrap_or(0.0) / S_BASE_VA,
            out["q"].as_f64().unwrap_or(0.0) / S_BASE_VA,
        );
        let i = (s / v_node).conj();
        let (r_s, x_s) = source_impedance_pu(src.sk, src.rx_ratio, S_BASE_VA);
        let v_virtual = v_node + i * Complex::new(r_s, x_s);
        buses[virtual_idx].voltage_mag = v_virtual.norm();
        buses[virtual_idx].voltage_ang = v_virtual.arg();
    }
}

/// Loads `name`, evaluates every measurement at PGM's own answer, and requires
/// each residual to sit within `max_sigma` of the value measured.
///
/// Returns how many measurements were checked, so a fixture silently producing
/// none can be caught.
fn assert_residuals_small(name: &str, max_sigma: f64) -> usize {
    let dir = fixture_dir(name);
    let input = common::load_pgm_input(&dir.join("input.json"));
    let expected = common::load_json(&dir.join("sym_output.json"));

    let id_to_idx = gridoxide::pgm::node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, S_BASE_VA);
    let measurements = {
        let net = pgm_to_network(
            common::load_pgm_input(&dir.join("input.json")),
            S_BASE_VA,
            50.0,
        );
        measurements_from_pgm(&input, &net, S_BASE_VA).expect("measurements")
    };

    let net = pgm_to_network(
        common::load_pgm_input(&dir.join("input.json")),
        S_BASE_VA,
        50.0,
    );
    let mut buses = net.buses.clone();
    apply_expected_state(&mut buses, &net, &input, &expected);

    let mut ybus = build_ybus(buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &shunts);
    let se_net = SeNetwork::new(&net, ybus.finish(), &shunts);
    let modelled_all = measurement_functions(&measurements, &buses, &se_net);

    let mut checked = 0;
    for (m, &modelled) in measurements.iter().zip(&modelled_all) {
        // An infinite sigma is power-grid-model's "this measurement carries no
        // information"; there is nothing to agree with.
        if !m.sigma.is_finite() {
            continue;
        }
        let residual = modelled - m.value;
        assert!(
            residual.abs() <= max_sigma * m.sigma,
            "{name}: {:?} on {:?} is {residual:.3e} off ({:.1} sigma) — \
             model says {modelled:.6}, sensor says {:.6}, sigma {:.3e}",
            m.kind,
            m.target,
            residual.abs() / m.sigma,
            m.value,
            m.sigma,
        );
        checked += 1;
    }
    checked
}

/// The fixtures whose networks gridoxide models completely today.
///
/// Excluded, with reasons rather than silently:
///
/// - anything containing a `link` (a zero-impedance branch power-grid-model
///   models as a fixed high admittance): gridoxide does not parse the component
///   at all, so those networks are missing branches and no residual claim about
///   them would be meaningful;
/// - `three_winding_transformer`: its sensor uses `measured_terminal_type` 6,
///   which `measurements_from_pgm` rejects rather than guesses at;
/// - the `single-node-source-asym-voltage-sensor*` pair: their only sensors are
///   asymmetric, and this is the symmetric path, so they yield no measurements
///   to check;
/// - `inf-measurement-with-injection-measured-unmeasured-appliances`: its bus
///   carries both measured and unmeasured appliances, so the measured subset is
///   not the bus injection. `measurements_from_pgm` currently sums appliance
///   sensors into one injection, which is only the same thing when *every*
///   appliance at the bus is measured. Handling the partial case needs
///   power-grid-model's own rule for it, and this fixture is what will verify
///   that rule once implemented;
/// - the fixtures in [`INCONSISTENT_BY_DESIGN`];
/// - fixtures whose `sym_output.json` does not give a complete node state.
///   power-grid-model's validation framework only asserts the fields each case
///   is about, so many of these omit `u_pu`/`u_angle` (or the `node` array
///   entirely). Without the full state there is nothing to evaluate `h(x)` at,
///   and inventing the missing entries would be testing a state nobody
///   published.
const MODELLED_FIXTURES: &[&str] = &[
    "1os2msr",
    "1os2msr-no-angle",
    "inf-measurement-with-injection",
    "transmission-case",
];

/// Fixtures whose measurements deliberately disagree with the state
/// power-grid-model publishes, so "residual within noise" is not a property
/// they have.
///
/// `node-injection-sensor-and-zero-injection` puts a 0.1 p.u. injection sensor
/// on a node with no appliance attached. power-grid-model requires at least one
/// appliance for an injection sensor to mean anything, so it treats the node as
/// a hard zero-injection bus and overrides the reading — its published answer
/// has both buses at exactly 1.0 angle 0, i.e. no flow at all. gridoxide's
/// `h(x)` agrees with that answer exactly; it is the sensor that is 100 sigma
/// out, by construction. The estimator handles it correctly — see
/// `tests/state_estimation_test.rs::zero_injection_constraint_overrides_a_conflicting_sensor`,
/// which reproduces power-grid-model's answer exactly by treating the zero
/// injection as a hard constraint. It is only *this* test's premise, that
/// residuals sit within noise at the published state, that the fixture does not
/// satisfy.
const INCONSISTENT_BY_DESIGN: &[&str] = &["node-injection-sensor-and-zero-injection"];

/// Whether a fixture's expected output pins down every bus voltage, which is
/// what `h(x)` has to be evaluated at.
fn has_complete_state(dir: &Path) -> bool {
    let text = match std::fs::read_to_string(dir.join("sym_output.json")) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return false,
    };
    match v["data"].get("node").and_then(|n| n.as_array()) {
        Some(nodes) if !nodes.is_empty() => {
            nodes.iter().all(|n| n.get("u_pu").is_some() && n.get("u_angle").is_some())
        }
        _ => false,
    }
}

/// Three sigma is the usual "this is noise, not a bug" line, and
/// power-grid-model's own documentation describes a sensor's sigma as its error
/// range divided by three — so a correctly modelled measurement should land
/// inside it essentially always.
#[test]
fn measurements_agree_with_pgm_state_within_noise() {
    let mut total = 0;
    let mut contributing = 0;
    for name in MODELLED_FIXTURES {
        let checked = assert_residuals_small(name, 3.0);
        if checked > 0 {
            contributing += 1;
        }
        total += checked;
    }
    // A fixture can legitimately contribute nothing once the unsettled sensor
    // types are filtered out (`node-injection-sensor-and-zero-injection`
    // measures only node injections), so the floor is on the aggregate rather
    // than per fixture — otherwise this test could quietly become vacuous.
    assert!(contributing >= 3, "only {contributing} fixtures contributed measurements");
    assert!(total > 50, "expected a meaningful number of measurements, got {total}");
}

/// A sanity check on the check: the fixture directory should not have grown
/// cases that silently go untested. Anything not in `MODELLED_FIXTURES` must be
/// one of the documented exclusions.
#[test]
fn every_symmetric_fixture_is_either_modelled_or_excluded() {
    let dir = fixture_dir("");
    let mut unexplained = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("fixture dir") {
        let entry = entry.expect("dir entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if !entry.path().join("sym_output.json").exists() || MODELLED_FIXTURES.contains(&name.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(entry.path().join("input.json")).expect("input");
        let has_link = serde_json::from_str::<serde_json::Value>(&text)
            .expect("input parses")["data"]
            .get("link")
            .is_some();
        let is_asym_only = name.starts_with("single-node-source-asym-voltage-sensor");
        let incomplete = !has_complete_state(&entry.path());
        let partial_appliances = name.ends_with("measured-unmeasured-appliances");
        let inconsistent = INCONSISTENT_BY_DESIGN.contains(&name.as_str());
        if !has_link
            && !is_asym_only
            && !incomplete
            && !partial_appliances
            && !inconsistent
            && name != "three_winding_transformer"
        {
            unexplained.push(name);
        }
    }
    assert!(
        unexplained.is_empty(),
        "these fixtures are neither modelled nor a documented exclusion: {unexplained:?}"
    );
}

/// Guards the exclusion list itself: `link` support is a known gap, and this
/// records how much it is worth. When links are modelled, this count drops and
/// the fixtures move into `MODELLED_FIXTURES`.
#[test]
fn link_gap_is_four_fixtures() {
    let dir: &Path = &fixture_dir("");
    let with_link = std::fs::read_dir(dir)
        .expect("fixture dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("sym_output.json").exists())
        .filter(|e| {
            let text = std::fs::read_to_string(e.path().join("input.json")).unwrap_or_default();
            serde_json::from_str::<serde_json::Value>(&text)
                .map(|v| v["data"].get("link").is_some())
                .unwrap_or(false)
        })
        .count();
    assert_eq!(with_link, 4, "link-using fixtures");
}
