//! A three-winding transformer in the phase domain, against power-grid-model's
//! own asymmetric answer.
//!
//! This is the only asymmetric reference for a three-winding transformer in the
//! vendored corpus. gridoxide models one as three two-winding legs to a
//! synthesized star node — power-grid-model's own star equivalent — and the
//! reference walks ten tap positions with all three sides in service.
//!
//! **What it gates, and what it does not.** The fixture has a source and
//! nothing else: no load, balanced excitation, and therefore no zero-sequence
//! current anywhere in it. So it pins the star equivalent and the *positive*
//! sequence — the uk/pk split across the three legs, each leg's admittance, the
//! tap on each of ten positions — and it is blind to the **winding
//! configurations**, which is precisely what the phase domain adds over the
//! symmetric model.
//!
//! That is not a guess. Swapping `winding_from` and `winding_to` on legs T2 and
//! T3 leaves every number here unchanged, which is why
//! `the_zero_sequence_follows_the_winding_configuration` exists below and
//! asserts the structure directly.

mod common;

use std::path::PathBuf;

use gridoxide::network::{build_ybus_3ph, stamp_transformers_3ph};
use gridoxide::pgm::{node_id_to_idx, pgm_to_3ph_network, pgm_transformers_3ph, PgmNodeAsymOutput};
use gridoxide::run_power_flow_analysis_from_ybus;

#[test]
fn three_winding_transformer_matches_pgms_asymmetric_answer() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/asymmetric/three-winding-transformer");

    let base_json = common::load_json(&base.join("input.json"));
    let update = common::load_json(&base.join("update_batch.json"));
    let expected =
        common::load_batch_output::<PgmNodeAsymOutput>(&base.join("asym_output_batch.json"));

    // The reference's own `params.json`: rtol 1e-4, atol 1e-5.
    let tol = 1e-4;
    let mut scenarios = 0;
    for (scenario, expected_scenario) in
        update["data"].as_array().unwrap().iter().zip(&expected.data)
    {
        let input = common::apply_batch_scenario(&base_json, scenario);
        let id_to_idx = node_id_to_idx(&input);
        let transformers = pgm_transformers_3ph(&input, &id_to_idx, 1e6);
        assert_eq!(transformers.len(), 3, "one three-winding transformer is three legs");

        let (buses, lines, id_to_idx) = pgm_to_3ph_network(input, 1e6, 50.0);
        // Three physical nodes, one star node, one source: five in all.
        assert_eq!(buses.len(), 3 * 5);

        let mut ybus = build_ybus_3ph(buses.len() / 3, &lines);
        stamp_transformers_3ph(&mut ybus, &transformers);
        let result = run_power_flow_analysis_from_ybus(buses, ybus).buses;

        for node_out in &expected_scenario.node {
            common::assert_asym_node(&result, &id_to_idx, node_out, tol);
        }
        scenarios += 1;
    }
    assert_eq!(scenarios, 10, "the reference walks ten tap positions");
}

/// The zero sequence follows the winding configuration.
///
/// The gate above cannot see this: its fixture carries no load, so the
/// excitation is balanced and no zero-sequence current flows whatever the
/// windings are. Verified by mutation — swapping the two sides of legs T2 and
/// T3 changes nothing there.
///
/// So the structure is asserted directly, against
/// `three_winding_transformer.hpp::convert_to_two_winding_transformers`:
///
/// - **T1** is `wye_n`/`wye_n` at clock 0, whatever `winding_1` is, because it
///   is the leg the star node's own base is defined by. Both sides therefore
///   have a zero-sequence path and its two-port is full.
/// - **T2** and **T3** carry their own winding on the *physical* side against
///   `winding_1` on the star side. Here that is `wye_n` into `delta`, which
///   grounds the physical side and not the star: a one-port on `yff` and
///   nothing on `ytt`. Swapping the two sides moves it to `ytt`, which is what
///   this catches.
#[test]
fn the_zero_sequence_follows_the_winding_configuration() {
    use gridoxide::network::{DELTA, WYE_N};

    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/asymmetric/three-winding-transformer");
    let base_json = common::load_json(&base.join("input.json"));
    let update = common::load_json(&base.join("update_batch.json"));
    // The first scenario puts all three sides in service.
    let input =
        common::apply_batch_scenario(&base_json, &update["data"].as_array().unwrap()[0]);

    let t = &input.data.three_winding_transformer[0];
    assert_eq!(t.winding_1, DELTA, "the fixture's side 1 is a delta");
    assert_eq!(t.winding_2, WYE_N);
    assert_eq!(t.winding_3, WYE_N);

    let id_to_idx = node_id_to_idx(&input);
    let legs = pgm_transformers_3ph(&input, &id_to_idx, 1e6);
    assert_eq!(legs.len(), 3);

    let live = |y: num_complex::Complex<f64>| y.norm() > 1e-9;

    // T1: wye_n into wye_n, so the zero sequence is a full two-port — current
    // crosses it, rather than only finding ground on one side.
    let t1 = &legs[0].y0;
    assert!(live(t1[1]) && live(t1[2]), "T1 zero sequence should couple its two ends: {t1:?}");

    // T2 and T3: grounded on the physical side, delta on the star side. A
    // one-port on `yff`, and no path at all on `ytt` beyond the low-susceptance
    // term the delta side picks up.
    for (name, leg) in [("T2", &legs[1].y0), ("T3", &legs[2].y0)] {
        assert!(live(leg[0]), "{name} should ground its physical side: {leg:?}");
        assert!(
            !live(leg[1]) && !live(leg[2]),
            "{name} zero sequence must not cross a delta: {leg:?}"
        );
        assert!(
            leg[3].norm() < leg[0].norm() * 1e-3,
            "{name}'s star side is a delta and should carry no zero-sequence path, \
             got {} against {}",
            leg[3].norm(),
            leg[0].norm()
        );
    }
}
