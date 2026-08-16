//! Piecewise-linear generator cost curves — MATPOWER's cost model 1.
//!
//! # The bug this file exists for
//!
//! Both formulations used to read costs with
//!
//! ```ignore
//! if let Some(CostCurve::Polynomial { coefficients }) = &generator.cost {
//! ```
//!
//! A piecewise-linear curve fell straight through that `if let`, leaving the
//! generator's objective coefficient at zero. The unit then looked **free**,
//! so the optimizer preferred it — a wrong dispatch, a wrong cost, and no
//! error. `a_piecewise_linear_generator_is_not_treated_as_free` pins exactly
//! that.
//!
//! It was reachable from the documented workflow: `gridoxide-matpower`
//! converts model 1 into `{"model": "piecewise_linear", ...}`, and `OpfData`
//! deserializes it happily. What kept it hidden was that every committed
//! fixture used model 2, which `plans/OPF_PLAN.md` §10 had already flagged as
//! the reason the feature could not be claimed.
//!
//! # How it is tested
//!
//! The main fixture is an **exact** rewrite rather than an approximation.
//! Every `case5_pjm` generator has `c2 = 0`, so its cost is already a straight
//! line; `case5_pjm_pwl.m` replaces each with three collinear points. Three,
//! not two, so the multi-segment path runs — and collinear, so the optimum
//! must be *identical* to the original's. That turns "does piecewise-linear
//! work" into a question with a known answer, rather than a comparison against
//! another implementation.

#![cfg(feature = "opf")]

use std::collections::HashMap;
use std::path::PathBuf;

use gridoxide::linear::btheta::DcBranch;
use gridoxide::linear::DcApproximation;
use gridoxide::opf::ac::{AcOpf, AcOpfNetwork, AcOpfOptions};
use gridoxide::opf::dc::{DcGenerator, DcLoad, DcOpf, DcOpfNetwork, DcOpfOptions};
use gridoxide::opf::ipm::IpmSolver;
use gridoxide::opf::model::{CostCurve, OpfData};
use gridoxide::opf::{OptStatus, Solver};
use gridoxide::pgm::PgmInput;

fn documents(name: &str) -> (PgmInput, OpfData) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pglib-opf");
    (
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap())
            .unwrap(),
        OpfData::from_json(&std::fs::read_to_string(dir.join(format!("{name}.opf.json"))).unwrap())
            .unwrap(),
    )
}

fn dc(name: &str) -> gridoxide::opf::dc::DcOpfResult {
    let (input, data) = documents(name);
    let network = DcOpfNetwork::from_pgm(input, &data, 50.0, DcApproximation::IgnoreG).unwrap();
    let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
    opf.interpret(&IpmSolver::new().solve(opf.problem()).unwrap())
}

fn ac(name: &str) -> gridoxide::opf::ac::AcOpfResult {
    let (input, data) = documents(name);
    let options = AcOpfOptions::default();
    let network = AcOpfNetwork::from_pgm(input, &data, 50.0, &options).unwrap();
    AcOpf::build(network, options).unwrap().solve().unwrap()
}

/// The fixture's whole point: an exactly-equivalent rewrite must give an
/// exactly equivalent answer.
#[test]
fn an_exactly_equivalent_piecewise_rewrite_gives_the_same_dc_answer() {
    let polynomial = dc("pglib_opf_case5_pjm");
    let piecewise = dc("case5_pjm_pwl");

    assert_eq!(polynomial.status, OptStatus::Optimal);
    assert_eq!(piecewise.status, OptStatus::Optimal);
    assert!(
        (polynomial.objective - piecewise.objective).abs() < 1e-6,
        "polynomial {} vs piecewise {}",
        polynomial.objective,
        piecewise.objective
    );
    for g in 0..polynomial.dispatch.len() {
        assert!(
            (polynomial.dispatch[g] - piecewise.dispatch[g]).abs() < 1e-5,
            "generator {g}: {} vs {} MW",
            polynomial.dispatch[g],
            piecewise.dispatch[g]
        );
    }
    // Prices too — the epigraph must not disturb the balance rows' duals,
    // which are what the whole formulation is arranged to produce.
    for bus in 0..polynomial.lmp.len() {
        assert!(
            (polynomial.lmp[bus] - piecewise.lmp[bus]).abs() < 1e-6,
            "bus {bus}: {} vs {} $/MWh",
            polynomial.lmp[bus],
            piecewise.lmp[bus]
        );
    }
}

/// The same for AC, where the epigraph earns its keep differently: the
/// objective's *values* were already right there, but a piecewise curve is
/// only C⁰ and the solver's Newton step assumes C². Reformulating makes the
/// objective linear and moves the kinks into constraints.
#[test]
fn an_exactly_equivalent_piecewise_rewrite_gives_the_same_ac_answer() {
    let polynomial = ac("pglib_opf_case5_pjm");
    let piecewise = ac("case5_pjm_pwl");

    assert_eq!(polynomial.status, OptStatus::Optimal);
    assert_eq!(piecewise.status, OptStatus::Optimal);
    assert!(
        (polynomial.objective - piecewise.objective).abs() < 1e-4,
        "polynomial {} vs piecewise {}",
        polynomial.objective,
        piecewise.objective
    );
    assert!(piecewise.violation < 1e-6);
    for bus in 0..polynomial.lmp_p.len() {
        assert!(
            (polynomial.lmp_p[bus] - piecewise.lmp_p[bus]).abs() < 1e-4,
            "bus {bus}: {} vs {} $/MWh",
            polynomial.lmp_p[bus],
            piecewise.lmp_p[bus]
        );
    }
}

/// Two buses, one branch, 100 MW of demand at bus 1.
fn two_bus(cost_0: CostCurve, cost_1: CostCurve) -> DcOpfNetwork {
    DcOpfNetwork {
        n_buses: 2,
        reference: 0,
        branches: vec![DcBranch { index: 0, from: 0, to: 1, b: 10.0, shift: 0.0 }],
        limits: HashMap::new(),
        generators: vec![
            DcGenerator { index: 0, bus: 0, p_min: 0.0, p_max: 1.0, cost: Some(cost_0) },
            DcGenerator { index: 1, bus: 1, p_min: 0.0, p_max: 1.0, cost: Some(cost_1) },
        ],
        loads: vec![DcLoad { bus: 1, p: 1.0 }],
        base_mva: 100.0,
    }
}

fn solve_two_bus(network: DcOpfNetwork) -> gridoxide::opf::dc::DcOpfResult {
    let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
    opf.interpret(&IpmSolver::new().solve(opf.problem()).unwrap())
}

/// **The regression.** A piecewise-linear generator must not read as free.
///
/// The expensive unit is the piecewise-linear one, so the old behaviour is
/// unmistakable: with its cost dropped it looked cheapest and took the whole
/// load. Costing it correctly leaves it idle.
#[test]
fn a_piecewise_linear_generator_is_not_treated_as_free() {
    // Unit 0: $10/MWh, polynomial. Unit 1: $90/MWh, piecewise-linear.
    // Per-unit on a 100 MVA base, so $10/MWh is 1000 $/pu.
    let result = solve_two_bus(two_bus(
        CostCurve::Polynomial { coefficients: vec![0.0, 1000.0] },
        CostCurve::PiecewiseLinear { points: vec![(0.0, 0.0), (1.0, 9000.0)] },
    ));

    assert_eq!(result.status, OptStatus::Optimal);
    assert!(
        (result.dispatch[0] - 100.0).abs() < 1e-4,
        "the cheap unit should serve all 100 MW, but produced {} MW",
        result.dispatch[0]
    );
    assert!(
        result.dispatch[1].abs() < 1e-4,
        "the expensive piecewise-linear unit produced {} MW — it was read as free",
        result.dispatch[1]
    );
    assert!(
        (result.objective - 1000.0).abs() < 1e-4,
        "cost {} should be 100 MW at $10/MWh = $1000/h",
        result.objective
    );
}

/// A genuinely **kinked** convex curve, where the answer is not a straight
/// line's answer.
///
/// Unit 0 charges $10/MWh for its first 50 MW and $30/MWh beyond; unit 1 is
/// flat at $20/MWh. By hand: take 50 MW from unit 0 at $10 (= $500), at which
/// point its marginal cost becomes $30 against unit 1's $20, so unit 1 serves
/// the rest. Total `$500 + 50 × $20 = $1500`, and the marginal unit — hence
/// the price everywhere on a lossless uncongested network — is unit 1's $20.
///
/// This is what the collinear fixture cannot test: with a real kink, the
/// optimum sits exactly *on* a breakpoint, which is both the interesting case
/// and the one a non-smooth objective handles worst.
#[test]
fn a_kinked_convex_curve_is_priced_segment_by_segment() {
    let result = solve_two_bus(two_bus(
        CostCurve::PiecewiseLinear {
            points: vec![(0.0, 0.0), (0.5, 500.0), (1.0, 2000.0)],
        },
        CostCurve::Polynomial { coefficients: vec![0.0, 2000.0] },
    ));

    assert_eq!(result.status, OptStatus::Optimal);
    assert!(
        (result.dispatch[0] - 50.0).abs() < 1e-4,
        "unit 0 should stop at its breakpoint, but produced {} MW",
        result.dispatch[0]
    );
    assert!(
        (result.dispatch[1] - 50.0).abs() < 1e-4,
        "unit 1 should serve the remainder, but produced {} MW",
        result.dispatch[1]
    );
    assert!(
        (result.objective - 1500.0).abs() < 1e-4,
        "cost {} should be $500 + $1000 = $1500/h",
        result.objective
    );
    for (bus, price) in result.lmp.iter().enumerate() {
        assert!(
            (price - 20.0).abs() < 1e-4,
            "bus {bus}: priced at {price} $/MWh, but the marginal unit charges $20"
        );
    }
}

/// Pushing demand past the breakpoint moves onto the second segment, and the
/// price moves with it. Without this, a curve whose optimum never leaves the
/// first segment would pass every check above while its later segments went
/// entirely unexercised.
#[test]
fn crossing_a_breakpoint_changes_the_marginal_price() {
    // Unit 1 is now dearer than unit 0's *second* segment, so unit 0 supplies
    // everything and the price is set on the segment it ends up in.
    let cheap_first = || CostCurve::PiecewiseLinear {
        points: vec![(0.0, 0.0), (0.5, 500.0), (1.0, 2000.0)],
    };
    let expensive = || CostCurve::Polynomial { coefficients: vec![0.0, 9000.0] };

    let mut below = two_bus(cheap_first(), expensive());
    below.loads[0].p = 0.25;
    let below = solve_two_bus(below);

    let mut above = two_bus(cheap_first(), expensive());
    above.loads[0].p = 0.75;
    let above = solve_two_bus(above);

    assert_eq!(below.status, OptStatus::Optimal);
    assert_eq!(above.status, OptStatus::Optimal);
    // First segment: $10/MWh. Second: $30/MWh.
    assert!(
        (below.lmp[0] - 10.0).abs() < 1e-4,
        "below the breakpoint the price should be $10/MWh, got {}",
        below.lmp[0]
    );
    assert!(
        (above.lmp[0] - 30.0).abs() < 1e-4,
        "above the breakpoint the price should be $30/MWh, got {}",
        above.lmp[0]
    );
    // And the costs are the areas under the curve.
    assert!((below.objective - 250.0).abs() < 1e-4, "cost {}", below.objective);
    assert!((above.objective - 1250.0).abs() < 1e-4, "cost {}", above.objective);
}

/// A **non-convex** piecewise curve is refused rather than silently relaxed.
///
/// The epigraph of a non-convex curve is its convex envelope, which charges
/// *less* than the curve does — so accepting one would return a cost the
/// generator would not actually charge. That is precisely the class of silent
/// wrongness this file exists to remove, so it must not be reintroduced by the
/// fix.
#[test]
fn a_non_convex_piecewise_curve_is_rejected() {
    // Decreasing marginal cost: $30/MWh then $10/MWh.
    let concave = CostCurve::PiecewiseLinear {
        points: vec![(0.0, 0.0), (0.5, 1500.0), (1.0, 2000.0)],
    };
    assert!(!concave.is_convex(), "the fixture must actually be non-convex");

    let built = DcOpf::build(
        two_bus(concave, CostCurve::Polynomial { coefficients: vec![0.0, 2000.0] }),
        DcOpfOptions::default(),
    );
    match built {
        Ok(_) => panic!("a non-convex cost must be refused, not relaxed"),
        Err(error) => assert!(
            format!("{error}").contains("non-convex"),
            "the error should say what is wrong: {error}"
        ),
    }
}

/// A one-point curve names a cost without saying how it varies. Treated as a
/// constant, which is the reading that cannot mislead — reading it as free
/// would repeat the original bug in miniature.
#[test]
fn a_single_point_curve_is_a_constant_not_a_free_unit() {
    let result = solve_two_bus(two_bus(
        CostCurve::PiecewiseLinear { points: vec![(0.0, 250.0)] },
        CostCurve::Polynomial { coefficients: vec![0.0, 2000.0] },
    ));

    assert_eq!(result.status, OptStatus::Optimal);
    // Unit 0 carries no *marginal* cost, so it is dispatched — but the
    // constant is still charged.
    assert!(
        (result.dispatch[0] - 100.0).abs() < 1e-4,
        "unit 0 produced {} MW",
        result.dispatch[0]
    );
    assert!(
        (result.objective - 250.0).abs() < 1e-4,
        "cost {} should be the curve's constant",
        result.objective
    );
}
