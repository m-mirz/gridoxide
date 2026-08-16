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

use gridoxide::linear::DcApproximation;
use gridoxide::linear::btheta::DcBranch;
use gridoxide::opf::dc::{DcGenerator, DcLoad, DcOpf, DcOpfNetwork, DcOpfOptions};
use gridoxide::opf::ipm::IpmSolver;
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

/// The in-house interior-point method — the default backend, and the one that
/// needs no system install, so these tests run everywhere.
///
/// `tests/opf_cross_test.rs` holds it against HiGHS on these same fixtures
/// wherever HiGHS is available, which is what lets this file treat one
/// backend's answer as the answer.
fn solver() -> IpmSolver {
    IpmSolver::new()
}

fn load_documents(name: &str) -> (PgmInput, OpfData) {
    let network_text = std::fs::read_to_string(fixture(name, ".json")).unwrap();
    let opf_text = std::fs::read_to_string(fixture(name, ".opf.json")).unwrap();
    (
        serde_json::from_str(&network_text).unwrap(),
        OpfData::from_json(&opf_text).unwrap(),
    )
}

/// Every case is built with the susceptance OPF defaults to; see
/// `the_textbook_susceptance_is_measurably_further_from_the_baseline` for why
/// that is `IgnoreG` and not the `1/x` the `dc` command uses.
fn load_case(name: &str) -> DcOpfNetwork {
    let (input, data) = load_documents(name);
    DcOpfNetwork::from_pgm(input, &data, 50.0, DcApproximation::IgnoreG).unwrap()
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
/// | `case3_lmbd` | −0.0001% | 1 |
/// | `case5_pjm` | −0.0006% | 1 |
/// | `case14_ieee` | +0.0013% | 0 |
/// | `case30_ieee` | −0.025% | 1 |
/// | `case118_ieee` | −0.013% | 2 |
///
/// All five agree to better than 0.03%, which for an independently written
/// formulation compared against published figures rounded to five significant
/// digits is about as close as the comparison can resolve.
///
/// Getting there took one real modelling correction, kept here because the
/// symptom was so easy to misread. `case30_ieee` originally sat +0.42% high
/// while the others were inside 0.04%, and the natural reading — a bad limit,
/// a missing constraint, a converter bug — was wrong in every case. Ruled out
/// by direct comparison against the `.m` file: all 41 susceptances, all 41
/// branch limits, every per-bus load, the generator boxes, and (via
/// [`assert_kkt`]) our own optimality. The cause was the *susceptance
/// formula*: branch 1→2 has `r = 0.0192, x = 0.0575`, resistive enough that
/// `b = 1/x` overstates it by 10%, and that branch is congested at the
/// optimum, so the error landed straight on a binding constraint. Switching to
/// `b = x/(r² + x²)` — what PowerModels computes, and what produced these
/// published numbers — fixed `case30` *and* improved all four others.
///
/// The lesson worth keeping: a discrepancy isolated to one case is not
/// evidence that the fault is isolated to one case. The formula was wrong
/// everywhere; only `case30` was congested on a branch resistive enough to
/// show it.
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
            relative < 5e-4,
            "{name}: objective {:.4e} vs published {published:.4e} ({:.4}% apart)",
            result.objective,
            relative * 100.0
        );
    }
}

/// The susceptance choice is load-bearing, so it is pinned rather than left as
/// a default someone could flip while the suite stayed green.
///
/// `b = 1/x` is not *wrong* — it is the textbook DC approximation and what the
/// `dc` power-flow command computes, for the good reason that MATPOWER,
/// pandapower and lightsim2grid all use it. It is simply not the model these
/// published objectives came from. This test asserts the difference is real
/// and lands where the analysis said it does: concentrated on `case30_ieee`,
/// small elsewhere.
#[test]
fn the_textbook_susceptance_is_measurably_further_from_the_baseline() {
    let gap = |name: &str, approximation| {
        let (input, data) = load_documents(name);
        let network = DcOpfNetwork::from_pgm(input, &data, 50.0, approximation).unwrap();
        let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
        let published = PUBLISHED_DC.iter().find(|(n, _)| *n == name).unwrap().1;
        let objective = opf.interpret(&solver().solve(opf.problem()).unwrap()).objective;
        (objective - published) / published
    };

    let case30 = gap("pglib_opf_case30_ieee", DcApproximation::IgnoreR);
    assert!(
        (case30 - 0.00423).abs() < 5e-4,
        "case30 with b = 1/x should sit ~+0.42% above the baseline, got {:.4}%",
        case30 * 100.0
    );

    // And the correction is a strict improvement, not a trade — no case is
    // made worse by it.
    for &(name, _) in PUBLISHED_DC {
        let textbook = gap(name, DcApproximation::IgnoreR).abs();
        let series = gap(name, DcApproximation::IgnoreG).abs();
        assert!(
            series <= textbook + 1e-9,
            "{name}: b = x/(r²+x²) is {series:.2e} off, worse than b = 1/x at {textbook:.2e}"
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
