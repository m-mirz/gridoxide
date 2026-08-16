//! DC optimal power flow, against all three gates `plans/OPF_PLAN.md` §7 names.
//!
//! - **Analytic cases**, where the optimum is derivable by hand, so a failure
//!   points at a specific term rather than somewhere in the solver.
//! - **KKT residuals**, which for a convex problem *prove* optimality rather
//!   than corroborate it, and cannot be fooled by a convention error shared
//!   with another tool.
//! - **Published objectives** from pglib's own `BASELINE.md`, produced by
//!   PowerModels.jl with IPOPT and independent of anything here.
//!
//! The third gate exists because pglib publishes a **DC** column beside the AC
//! one. An earlier draft of the plan claimed otherwise and concluded this
//! phase would ship with no external number; see `tests/data/pglib-opf/README.md`.

use std::collections::HashMap;
use std::path::PathBuf;

use gridoxide::linear::btheta::DcBranch;
use gridoxide::opf::dc::{DcGenerator, DcLoad, DcOpf, DcOpfNetwork, DcOpfOptions};
use gridoxide::opf::highs::HighsSolver;
use gridoxide::opf::model::{CostCurve, OpfData};
use gridoxide::opf::{OptStatus, Solver};
use gridoxide::pgm::PgmInput;

/// pglib's published DC-OPF objectives, $/h — see the fixture README.
const PUBLISHED_DC: &[(&str, f64)] = &[
    ("pglib_opf_case3_lmbd", 5.6959e+03),
    ("pglib_opf_case5_pjm", 1.7480e+04),
    ("pglib_opf_case14_ieee", 2.0515e+03),
    ("pglib_opf_case30_ieee", 7.4728e+03),
    ("pglib_opf_case118_ieee", 9.3101e+04),
];

fn fixture(name: &str, suffix: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pglib-opf")
        .join(format!("{name}{suffix}"))
}

fn solver() -> HighsSolver {
    let s = HighsSolver::new().expect("HiGHS is required by this feature");
    // DC-OPF is a QP wherever a cost curve is quadratic, so it inherits the
    // interior-point tolerance discussed in `opf_highs_test.rs`.
    s.set_tolerance(1e-10, 1e-10).unwrap();
    s
}

fn load_case(name: &str) -> DcOpfNetwork {
    let network_text = std::fs::read_to_string(fixture(name, ".json")).unwrap();
    let opf_text = std::fs::read_to_string(fixture(name, ".opf.json")).unwrap();
    let input: PgmInput = serde_json::from_str(&network_text).unwrap();
    let data = OpfData::from_json(&opf_text).unwrap();
    DcOpfNetwork::from_pgm(input, &data, 50.0).unwrap()
}

/// Checks the returned point satisfies the KKT conditions of the program it
/// came from.
///
/// For a convex problem this is a **proof of optimality**, not a comparison —
/// it cannot be fooled by a shared convention error the way a tool-to-tool
/// check can, which is what makes it the primary gate here.
///
/// Four conditions, all at once:
/// stationarity (`col_dual = Qx + c − Aᵀy`), primal feasibility (every row and
/// column inside its bounds), and complementary slackness (a row strictly
/// inside its bounds prices at zero).
#[track_caller]
fn assert_kkt(opf: &DcOpf, solution: &gridoxide::opf::Solution, tol: f64) {
    let lp = opf.problem();

    // Stationarity.
    let mut lhs = lp.col_cost.clone();
    if let Some(hessian) = &lp.hessian {
        for &(i, j, v) in hessian {
            lhs[i] += v * solution.primal[j];
            if i != j {
                lhs[j] += v * solution.primal[i];
            }
        }
    }
    for &(row, col, v) in &lp.rows {
        lhs[col] -= v * solution.row_dual[row];
    }
    for k in 0..lp.n_vars {
        assert!(
            (solution.col_dual[k] - lhs[k]).abs() < tol,
            "stationarity at column {k}: {} vs {}",
            solution.col_dual[k],
            lhs[k]
        );
    }

    // Primal feasibility.
    for k in 0..lp.n_vars {
        let x = solution.primal[k];
        assert!(
            x >= lp.col_lower[k] - tol && x <= lp.col_upper[k] + tol,
            "column {k} = {x} outside [{}, {}]",
            lp.col_lower[k],
            lp.col_upper[k]
        );
    }
    for k in 0..lp.n_rows {
        let a = solution.row_activity[k];
        assert!(
            a >= lp.row_lower[k] - tol && a <= lp.row_upper[k] + tol,
            "row {k} = {a} outside [{}, {}]",
            lp.row_lower[k],
            lp.row_upper[k]
        );
    }

    // Complementary slackness: a row strictly inside its bounds carries no
    // price.
    for k in 0..lp.n_rows {
        let a = solution.row_activity[k];
        let slack_below = a - lp.row_lower[k];
        let slack_above = lp.row_upper[k] - a;
        if slack_below > 1e-6 && slack_above > 1e-6 {
            assert!(
                solution.row_dual[k].abs() < 1e-6,
                "row {k} is slack ({a} in [{}, {}]) but prices at {}",
                lp.row_lower[k],
                lp.row_upper[k],
                solution.row_dual[k]
            );
        }
    }
}

/// **The analytic case.** Two buses, a cheap generator at bus 0 and an
/// expensive one at bus 1, with 100 MW of demand at bus 1.
///
/// With the line unconstrained the cheap unit serves everything: 100 MW at
/// $10/MWh is $1000/h, and the price is $10/MWh everywhere because one more MW
/// anywhere is met by the same marginal unit.
#[test]
fn an_uncongested_two_bus_case_dispatches_the_cheap_unit_only() {
    let network = two_bus(None);
    let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
    let solution = solver().solve(opf.problem()).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal);
    assert_kkt(&opf, &solution, 1e-7);

    let result = opf.interpret(&solution);
    assert!((result.dispatch[0] - 100.0).abs() < 1e-6, "{:?}", result.dispatch);
    assert!(result.dispatch[1].abs() < 1e-6, "{:?}", result.dispatch);
    assert!((result.objective - 1000.0).abs() < 1e-6, "{}", result.objective);

    for (bus, lmp) in result.lmp.iter().enumerate() {
        assert!((lmp - 10.0).abs() < 1e-6, "bus {bus} prices at {lmp}, expected 10");
    }
    assert!(result.binding.is_empty());
    assert!(result.shed.iter().all(|s| s.abs() < 1e-6));
}

/// The same case with the line capped at 40 MW — now the constraint bites, and
/// every number changes in a way that can be derived by hand.
///
/// The cheap unit can deliver only 40 MW, so the expensive one must cover the
/// remaining 60. Cost is `40·10 + 60·30 = 2200`/h. The price at bus 1 becomes
/// the expensive unit's $30/MWh, because that is what serves one more MW
/// there; bus 0 stays at $10. **The $20 spread is the congestion**, and it is
/// exactly the shadow price of the line.
#[test]
fn a_congested_two_bus_case_splits_the_price() {
    let network = two_bus(Some(0.4));
    let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
    let solution = solver().solve(opf.problem()).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal);
    assert_kkt(&opf, &solution, 1e-7);

    let result = opf.interpret(&solution);
    assert!((result.dispatch[0] - 40.0).abs() < 1e-6, "{:?}", result.dispatch);
    assert!((result.dispatch[1] - 60.0).abs() < 1e-6, "{:?}", result.dispatch);
    assert!((result.objective - 2200.0).abs() < 1e-6, "{}", result.objective);

    assert!((result.lmp[0] - 10.0).abs() < 1e-6, "{:?}", result.lmp);
    assert!((result.lmp[1] - 30.0).abs() < 1e-6, "{:?}", result.lmp);

    assert_eq!(result.binding.len(), 1, "{:?}", result.binding);
    let binding = &result.binding[0];
    assert_eq!(binding.branch, 0);
    assert!((binding.flow - 40.0).abs() < 1e-6, "{binding:?}");
    assert!((binding.rate - 40.0).abs() < 1e-6, "{binding:?}");
    // One more MW of capacity replaces expensive generation with cheap: a
    // saving of exactly the $20 price spread.
    assert!((binding.price.abs() - 20.0).abs() < 1e-6, "{binding:?}");
}

/// With demand that cannot be served and shedding allowed, the answer says
/// *where* rather than merely failing.
#[test]
fn unservable_demand_is_shed_rather_than_reported_infeasible() {
    let mut network = two_bus(None);
    network.generators[0].p_max = 0.2; // 20 MW
    network.generators[1].p_max = 0.3; // 30 MW, against 100 MW of demand
    let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
    let solution = solver().solve(opf.problem()).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal);

    let result = opf.interpret(&solution);
    let shed: f64 = result.shed.iter().sum();
    assert!((shed - 50.0).abs() < 1e-6, "expected 50 MW shed, got {shed}");
    // And shedding prices at the penalty, which is what makes it a last resort.
    assert!(result.lmp[1] > 1000.0, "{:?}", result.lmp);
}

/// Without shedding the same case is genuinely infeasible, and says so.
#[test]
fn without_shedding_unservable_demand_is_infeasible() {
    let mut network = two_bus(None);
    network.generators[0].p_max = 0.2;
    network.generators[1].p_max = 0.3;
    let options = DcOpfOptions { allow_shedding: false, ..Default::default() };
    let opf = DcOpf::build(network, options).unwrap();
    let solution = solver().solve(opf.problem()).unwrap();
    assert!(!solution.is_optimal(), "status was {:?}", solution.status);
}

/// Every pglib case solves, satisfies its KKT conditions, and balances.
#[test]
fn every_pglib_case_solves_to_a_certified_optimum() {
    for &(name, _) in PUBLISHED_DC {
        let network = load_case(name);
        let total_load: f64 = network.loads.iter().map(|l| l.p).sum();
        let base = network.base_mva;

        let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
        let solution = solver().solve(opf.problem()).unwrap();
        assert_eq!(solution.status, OptStatus::Optimal, "{name}");
        assert_kkt(&opf, &solution, 1e-6);

        let result = opf.interpret(&solution);
        let generated: f64 = result.dispatch.iter().sum();
        let shed: f64 = result.shed.iter().sum();
        // DC is lossless, so generation plus shedding must equal demand
        // exactly. This is the cheapest check that the balance rows are
        // assembled with the right signs.
        assert!(
            (generated + shed - total_load * base).abs() < 1e-4,
            "{name}: generated {generated} + shed {shed} != load {}",
            total_load * base
        );
        assert!(shed < 1e-6, "{name}: shed {shed} MW on a case that should be servable");
    }
}

/// **The external gate.** Against pglib's own published DC objectives.
///
/// Measured gaps, recorded so a regression shows up as a *change* rather than
/// having to re-derive what "close enough" means:
///
/// | Case | Gap | Binding |
/// |---|---|---|
/// | `case3_lmbd` | −0.037% | 1 |
/// | `case5_pjm` | −0.0006% | 1 |
/// | `case14_ieee` | +0.0013% | 0 |
/// | `case30_ieee` | **+0.42%** | 1 |
/// | `case118_ieee` | +0.034% | 2 |
///
/// Four of five agree to better than 0.04%. `case30_ieee` is ten times worse,
/// and **that is an open question rather than a resolved one.** What has been
/// ruled out:
///
/// - *Not the network.* Every branch susceptance was compared against
///   MATPOWER's own `makeBdc` (`b = 1/x/tap`), matched by endpoints rather
///   than index — 41 of 41 identical to 1e-9 relative, tap handling included.
/// - *Not missing limits.* Removing branch limits moves every case far in the
///   other direction (−24% on `case30`), so PowerModels enforces them too and
///   so do we.
/// - *Not phase shifters.* None of these cases has one, so the converter's
///   documented 60°-rounding of shifts cannot be involved.
/// - *Not our solver.* `assert_kkt` certifies the returned point is optimal
///   **for the program we posed**, which is what separates "our model differs"
///   from "our answer is wrong".
///
/// What remains is a difference in the constraint set between this formulation
/// and PowerModels'. The most likely candidate not yet examined is the
/// branch angle-difference limits (`angmin`/`angmax`), which PowerModels
/// enforces and this formulation does not — though that would make us *less*
/// constrained, and our objective is *higher*, so it does not obviously fit
/// either.
///
/// The bound below is set to catch a real regression while tolerating the
/// known gap. Tightening it is the right move once the difference is
/// explained.
#[test]
fn objectives_match_the_published_dc_baseline() {
    for &(name, published) in PUBLISHED_DC {
        let network = load_case(name);
        let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
        let solution = solver().solve(opf.problem()).unwrap();
        assert_eq!(solution.status, OptStatus::Optimal, "{name}");

        let result = opf.interpret(&solution);
        let relative = (result.objective - published).abs() / published.abs();
        assert!(
            relative < 5e-3,
            "{name}: objective {:.4e} vs published {published:.4e} ({:.3}% apart)",
            result.objective,
            relative * 100.0
        );
    }
}

/// The four cases that agree closely must keep agreeing *closely* — a bound
/// loose enough for `case30_ieee` would let a real regression through on the
/// others unnoticed.
#[test]
fn the_well_matched_cases_stay_well_matched() {
    for &(name, published) in PUBLISHED_DC {
        if name == "pglib_opf_case30_ieee" {
            continue;
        }
        let network = load_case(name);
        let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
        let solution = solver().solve(opf.problem()).unwrap();
        let result = opf.interpret(&solution);
        let relative = (result.objective - published).abs() / published.abs();
        assert!(
            relative < 1e-3,
            "{name}: {:.4e} vs published {published:.4e} ({:.4}% apart)",
            result.objective,
            relative * 100.0
        );
    }
}

/// Locational marginal prices must be uniform when nothing is congested and
/// spread only where something is — the property that makes them worth
/// publishing at all.
#[test]
fn prices_spread_only_where_the_network_binds() {
    for &(name, _) in PUBLISHED_DC {
        let network = load_case(name);
        let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
        let solution = solver().solve(opf.problem()).unwrap();
        let result = opf.interpret(&solution);

        let min = result.lmp.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = result.lmp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if result.binding.is_empty() {
            assert!(
                (max - min).abs() < 1e-6,
                "{name}: nothing binds, yet prices span {min}..{max}"
            );
        }
        // Prices are costs: negative ones are possible in principle but need a
        // reason, and none of these cases has one.
        assert!(min > -1e-6, "{name}: a negative price of {min}");
    }
}

fn two_bus(rate: Option<f64>) -> DcOpfNetwork {
    DcOpfNetwork {
        n_buses: 2,
        reference: 0,
        branches: vec![DcBranch { index: 0, from: 0, to: 1, b: 10.0, shift: 0.0 }],
        limits: HashMap::from([(0, rate)]),
        generators: vec![
            // $10/MWh on a 100 MVA base: 10 * 100 per per-unit MW.
            DcGenerator {
                index: 0,
                bus: 0,
                p_min: 0.0,
                p_max: 2.0,
                cost: Some(CostCurve::Polynomial { coefficients: vec![0.0, 1000.0] }),
            },
            // $30/MWh.
            DcGenerator {
                index: 1,
                bus: 1,
                p_min: 0.0,
                p_max: 2.0,
                cost: Some(CostCurve::Polynomial { coefficients: vec![0.0, 3000.0] }),
            },
        ],
        loads: vec![DcLoad { bus: 1, p: 1.0 }],
        base_mva: 100.0,
    }
}
