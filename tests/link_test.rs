//! power-grid-model's `link` component: a zero-impedance connection.
//!
//! The interesting assertion here is not the node voltages — it is the link's
//! **own reported flow**. power-grid-model's output schema carries a `link`
//! record, and these fixtures publish its current. That is precisely why
//! gridoxide stamps a link as a branch instead of merging its endpoints: a
//! merge deletes the branch those numbers describe, and there would be nothing
//! left to report through.
//!
//! A test that only compared node voltages would pass under either treatment
//! and so would not be testing the decision at all. See
//! `docs/src/powerflow/zero_impedance_branches.md` and `src/topology.rs`.

mod common;

use std::path::PathBuf;

use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_network};
use gridoxide::run_power_flow_analysis_from_ybus;

const S_BASE_VA: f64 = 1e6;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/link")
        .join(name)
}

/// Solves `name` and checks its node voltages and every published link current.
///
/// Tolerances come from each fixture's own `params.json` (`rtol`/`atol` of
/// 1e-5), applied against the units power-grid-model publishes: volts for
/// voltages, amperes for currents.
fn assert_link_fixture(name: &str) {
    let dir = fixture_dir(name);
    let input = common::load_pgm_input(&dir.join("input.json"));
    let expected = common::load_json(&dir.join("sym_output.json"));

    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, S_BASE_VA);
    let net = pgm_to_network(
        common::load_pgm_input(&dir.join("input.json")),
        S_BASE_VA,
        50.0,
    );

    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &shunts);
    let report = run_power_flow_analysis_from_ybus(net.buses.clone(), ybus);

    for node in expected["data"]["node"].as_array().expect("node output") {
        let id = node["id"].as_u64().expect("node id");
        let idx = net.node_idx[&id];
        let Some(u) = node["u"].as_f64() else { continue };
        let got = report.buses[idx].voltage_mag * report.buses[idx].u_rated;
        assert!(
            (got - u).abs() < 1e-5 * u.abs().max(1.0),
            "{name} node {id}: |V| = {got} V, PGM says {u}"
        );
    }

    let params = branch_params(&net.lines, &net.transformers);
    let v = bus_voltages(&report.buses);
    let links = expected["data"]["link"].as_array().expect("link output");
    assert!(!links.is_empty(), "{name}: fixture publishes no link output to check");

    for out in links {
        let id = out["id"].as_u64().expect("link id");
        let (branch, terminal) = net
            .resolve_terminal(id, Terminal::From)
            .unwrap_or_else(|| panic!("{name}: link {id} did not survive as a branch"));
        let (p, q) = terminal_flow(&params[branch], terminal, &v);

        let bus = params[branch].from;
        // power-grid-model reports branch current in amperes; converting needs
        // the terminal's own voltage base, `S_base / (u_rated * sqrt(3))`.
        let i_pu = (p * p + q * q).sqrt() / report.buses[bus].voltage_mag;
        let i_a = i_pu * S_BASE_VA / (report.buses[bus].u_rated * 3f64.sqrt());
        let want = out["i_from"].as_f64().expect("i_from");
        assert!(
            (i_a - want).abs() < 1e-5 * want.abs().max(1.0),
            "{name} link {id}: i_from = {i_a} A, PGM says {want}"
        );
    }
}

#[test]
fn link_flow_matches_pgm() {
    assert_link_fixture("dummy-test");
}

/// Same network with `i_n` omitted — an optional-attribute path, which this
/// crate has a history of getting wrong in the strict direction.
#[test]
fn link_flow_matches_pgm_without_i_n() {
    assert_link_fixture("dummy-test-i-n-optional");
}

/// The batch variant's base case. Only the base `sym_output.json` is checked
/// here; the batch scenarios themselves need the update-batch machinery.
#[test]
fn link_flow_matches_pgm_in_the_batch_fixture() {
    assert_link_fixture("dummy-test-batch-shunt");
}

/// A link must reach the solver as a branch, not as a merged bus.
///
/// This is the structural half of the policy, asserted directly rather than
/// inferred from the flows above: the two endpoints stay distinct buses, and
/// the link resolves to a branch index.
#[test]
fn a_link_stays_a_branch_with_distinct_endpoints() {
    let dir = fixture_dir("dummy-test");
    let input = common::load_pgm_input(&dir.join("input.json"));
    let net = pgm_to_network(input, S_BASE_VA, 50.0);

    let raw = common::load_json(&dir.join("input.json"));
    let link = &raw["data"]["link"].as_array().expect("link input")[0];
    let (from, to) = (
        link["from_node"].as_u64().unwrap(),
        link["to_node"].as_u64().unwrap(),
    );

    assert_ne!(
        net.node_idx[&from], net.node_idx[&to],
        "a link's endpoints must remain distinct buses; merging them would \
         delete the branch whose flow the fixture publishes"
    );
    assert!(
        net.branch_idx.contains_key(&link["id"].as_u64().unwrap()),
        "a link must register as a branch so its flow can be reported"
    );
}
