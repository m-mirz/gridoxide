//! Validates `branch_flow` against power-grid-model's own expected branch
//! flows.
//!
//! The existing PGM power-flow fixtures carry a `line` array in their
//! `sym_output.json` that gridoxide has never checked — only the `node` array
//! was ever compared (see `tests/common::assert_sym_node`). Those line entries
//! are exactly PGM's `p_from`/`q_from`/`p_to`/`q_to` for each branch, so they
//! validate the new terminal-flow formulas at no fixture cost.
//!
//! This is the gate for phase 0 of the state-estimation work: branch power
//! measurements are the majority of the sensors in PGM's state-estimation
//! fixtures, and their measurement function is precisely what is checked here.

mod common;

use std::path::PathBuf;

use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use gridoxide::network::{build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_network, PgmLineOutput};
use gridoxide::solver::{JacobianBackend, PersistentSolver};

fn data_dir(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow/symmetric")
        .join(rel)
}

/// Solves `fixture` and asserts every branch flow in its `sym_output.json`
/// matches, in W/var.
///
/// `tol` is absolute and in the fixture's own units, so it is a power
/// tolerance: 1 W on a grid transferring megawatts is ~1e-6 relative, tight
/// enough that a wrong shunt split or a dropped tap term could not pass.
fn assert_branch_flows(fixture: &str, tol: f64) {
    let base = data_dir(fixture);
    let input = common::load_pgm_input(&base.join("input.json"));
    let expected = common::load_json(&base.join("sym_output.json"));
    let s_base_va = 1e6;

    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, s_base_va);
    let net = pgm_to_network(input, s_base_va, 50.0);

    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &shunts);
    let ybus = ybus.finish();

    // Solved to 1e-12 rather than through `run_power_flow_analysis_from_ybus`,
    // whose 1e-6 p.u. mismatch tolerance is 1 W on this 1 MVA base — the same
    // order as the branch-flow differences being asserted. At 1e-6 the `line`
    // fixture misses by 1.4 W purely because Newton stopped, which would make
    // this a test of the solver's stopping point rather than of the flow
    // formulas.
    let mut solved = net.buses.clone();
    linear_initial_guess(&mut solved, &ybus);
    PersistentSolver::new(JacobianBackend::Scalar).solve(&mut solved, &ybus, 1e-12, 50);

    let params = branch_params(&net.lines, &net.transformers);
    let v = bus_voltages(&solved);

    let line_outputs: Vec<PgmLineOutput> =
        serde_json::from_value(expected["data"]["line"].clone()).expect("line output array");
    assert!(!line_outputs.is_empty(), "{fixture}: fixture has no line output to check");

    let mut checked = 0;
    for out in &line_outputs {
        let check = |actual_pu: f64, expected_w: f64, what: &str| {
            let actual_w = actual_pu * s_base_va;
            assert!(
                (actual_w - expected_w).abs() < tol,
                "{fixture} line {}: {what} = {actual_w}, PGM says {expected_w}",
                out.id
            );
        };
        // `resolve_terminal` handles the two degenerate cases: a line with both
        // terminals open produces no branch at all, and the open end of a
        // half-open line has no flow. PGM reports zero for both, so they are
        // still worth asserting rather than skipping.
        let flow = |terminal: Terminal| match net.resolve_terminal(out.id, terminal) {
            Some((b, t)) => terminal_flow(&params[b], t, &v),
            None => (0.0, 0.0),
        };
        let (p_from, q_from) = flow(Terminal::From);
        let (p_to, q_to) = flow(Terminal::To);

        check(p_from, out.p_from, "p_from");
        check(q_from, out.q_from, "q_from");
        check(p_to, out.p_to, "p_to");
        check(q_to, out.q_to, "q_to");
        checked += 1;
    }
    assert!(checked > 0, "{fixture}: no branch was actually compared");
}

#[test]
fn branch_flows_match_pgm_line_fixture() {
    assert_branch_flows("line", 1.0);
}

#[test]
fn branch_flows_match_pgm_node_fixture() {
    assert_branch_flows("node", 1.0);
}

#[test]
fn branch_flows_match_pgm_shunt_fixture() {
    assert_branch_flows("shunt", 1.0);
}

#[test]
fn branch_flows_match_pgm_source_fixture() {
    assert_branch_flows("source", 1.0);
}

#[test]
fn branch_flows_match_pgm_sym_load_fixture() {
    assert_branch_flows("sym_load", 1.0);
}

#[test]
fn branch_flows_match_pgm_sym_gen_fixture() {
    assert_branch_flows("sym_gen", 1.0);
}

/// The one fixture with transformers as well as lines, at transmission scale —
/// the case where an off-nominal tap makes the two terminals genuinely
/// asymmetric. Tolerance is looser in absolute terms only because the powers
/// themselves are ~1e8 W here.
#[test]
fn branch_flows_match_pgm_transmission_case() {
    assert_branch_flows("transmission-case", 100.0);
}
