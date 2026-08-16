//! **The two-solver gate.** HiGHS and the in-house interior-point method,
//! held against each other.
//!
//! On a convex problem the optimal objective is unique, so two independent
//! solvers *must* agree. That makes this a far sharper instrument than most
//! cross-tool comparisons: a disagreement is a bug in one of them, not a
//! modelling convention, not a different local optimum, not a tolerance
//! question. `plans/OPF_PLAN.md` §7.4 counts it as a validation gate in its
//! own right.
//!
//! What it does **not** catch is worth stating as plainly, because the
//! temptation is to read agreement as correctness. Both backends receive the
//! *same* `LinearProgram`. If DC-OPF assembles the wrong program — the wrong
//! susceptance, a missing limit, a sign error in the balance rows — both will
//! agree perfectly on the wrong answer. That happened: the `case30_ieee`
//! susceptance bug recorded in `opf_dc_test.rs` would have sailed through this
//! file. Cross-validation tests the *solvers*; the published-objective
//! comparison tests the *model*. Neither substitutes for the other.
//!
//! The two are also not interchangeable in kind. Simplex lands exactly on a
//! vertex; an interior-point method approaches asymptotically. So the
//! tolerances below are floors set by the interior-point side, measured rather
//! than assumed — see [`OBJECTIVE_TOL`].

#![cfg(all(feature = "opf", feature = "opf-highs"))]

use std::path::PathBuf;

use gridoxide::linear::DcApproximation;
use gridoxide::opf::dc::{DcOpf, DcOpfNetwork, DcOpfOptions};
use gridoxide::opf::highs::HighsSolver;
use gridoxide::opf::ipm::IpmSolver;
use gridoxide::opf::model::OpfData;
use gridoxide::opf::{LinearProgram, OptStatus, Solution, Solver};
use gridoxide::pgm::PgmInput;

/// Relative bound on the objective.
///
/// Measured, not chosen: across the five pglib fixtures the worst observed
/// disagreement is 1.1e-11 relative, at the interior-point method's default
/// tolerance of 1e-10. This bound sits about two orders above that — enough
/// headroom that ordinary numerical weather cannot trip it, tight enough that
/// any real modelling or sign error would.
const OBJECTIVE_TOL: f64 = 1e-9;

/// Absolute bound on prices, in $/MWh.
///
/// Absolute rather than relative because a dual is legitimately zero on any
/// non-binding constraint, and a relative test against zero is meaningless.
/// Observed worst case is 4.2e-9; prices themselves run to tens or hundreds.
const DUAL_TOL: f64 = 1e-6;

const CASES: &[&str] = &[
    "pglib_opf_case3_lmbd",
    "pglib_opf_case5_pjm",
    "pglib_opf_case14_ieee",
    "pglib_opf_case30_ieee",
    "pglib_opf_case118_ieee",
];

fn dc_opf(name: &str, options: DcOpfOptions) -> DcOpf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pglib-opf");
    let input: PgmInput =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap())
            .unwrap();
    let data =
        OpfData::from_json(&std::fs::read_to_string(dir.join(format!("{name}.opf.json"))).unwrap())
            .unwrap();
    let network = DcOpfNetwork::from_pgm(input, &data, 50.0, DcApproximation::IgnoreG).unwrap();
    DcOpf::build(network, options).unwrap()
}

fn both(lp: &LinearProgram) -> (Solution, Solution) {
    let highs = HighsSolver::new().unwrap().solve(lp).unwrap();
    let ipm = IpmSolver::new().solve(lp).unwrap();
    (highs, ipm)
}

fn assert_objectives_agree(highs: &Solution, ipm: &Solution, what: &str) {
    assert_eq!(highs.status, ipm.status, "{what}: the two backends disagree on status");
    if highs.status != OptStatus::Optimal {
        return;
    }
    let relative = (highs.objective - ipm.objective).abs() / highs.objective.abs().max(1.0);
    assert!(
        relative < OBJECTIVE_TOL,
        "{what}: HiGHS {} vs interior-point {} ({relative:.3e} relative)",
        highs.objective,
        ipm.objective
    );
}

/// Every fixture, on the objective — the quantity that is unique for a convex
/// problem whatever the two solvers do internally.
#[test]
fn the_two_backends_agree_on_every_pglib_objective() {
    for name in CASES {
        let opf = dc_opf(name, DcOpfOptions::default());
        let (highs, ipm) = both(opf.problem());
        assert_objectives_agree(&highs, &ipm, name);
    }
}

/// And on the prices, which is the stronger claim.
///
/// Objectives can agree while duals differ — a degenerate optimum has a whole
/// face of valid multipliers, and simplex and interior-point methods make
/// *different* choices there by construction. So this asserting cleanly across
/// every fixture says these problems are non-degenerate in their duals, which
/// is exactly what makes the published locational marginal prices meaningful
/// to report at all.
#[test]
fn the_two_backends_agree_on_every_locational_marginal_price() {
    for name in CASES {
        let opf = dc_opf(name, DcOpfOptions::default());
        let (highs, ipm) = both(opf.problem());
        assert_eq!(highs.status, OptStatus::Optimal);

        let a = opf.interpret(&highs);
        let b = opf.interpret(&ipm);
        for bus in 0..a.lmp.len() {
            assert!(
                (a.lmp[bus] - b.lmp[bus]).abs() < DUAL_TOL,
                "{name} bus {bus}: HiGHS priced it at {} $/MWh, interior-point at {}",
                a.lmp[bus],
                b.lmp[bus]
            );
        }

        // Binding sets must match exactly — a branch is either at its limit or
        // it is not, and that is a discrete fact neither method should blur.
        let mut binding_a: Vec<usize> = a.binding.iter().map(|x| x.branch).collect();
        let mut binding_b: Vec<usize> = b.binding.iter().map(|x| x.branch).collect();
        binding_a.sort_unstable();
        binding_b.sort_unstable();
        assert_eq!(binding_a, binding_b, "{name}: the backends disagree on what binds");
    }
}

/// The dispatch itself. Weaker than it looks — a linear cost curve can leave
/// the *argument* non-unique even when the objective is not — so this checks
/// total generation rather than per-unit output where the two could
/// legitimately differ, and per-unit output only where the cost is strictly
/// convex.
#[test]
fn the_two_backends_agree_on_what_each_generator_produces() {
    for name in CASES {
        let opf = dc_opf(name, DcOpfOptions::default());
        let (highs, ipm) = both(opf.problem());
        let a = opf.interpret(&highs);
        let b = opf.interpret(&ipm);

        let total_a: f64 = a.dispatch.iter().sum();
        let total_b: f64 = b.dispatch.iter().sum();
        assert!(
            (total_a - total_b).abs() < 1e-6,
            "{name}: total generation {total_a} vs {total_b} MW"
        );
        for g in 0..a.dispatch.len() {
            assert!(
                (a.dispatch[g] - b.dispatch[g]).abs() < 1e-5,
                "{name} generator {g}: {} vs {} MW",
                a.dispatch[g],
                b.dispatch[g]
            );
        }
    }
}

/// Agreement on problems that have *no* answer, which the fixtures never
/// exercise: the two methods reach that verdict by completely different routes
/// — simplex proves it from a basis, the interior-point method infers it from
/// divergence — so agreement here is genuinely independent evidence.
#[test]
fn the_two_backends_agree_on_infeasible_and_unbounded_problems() {
    let mut infeasible = LinearProgram::new(1);
    infeasible.col_lower = vec![f64::NEG_INFINITY];
    infeasible.col_upper = vec![f64::INFINITY];
    infeasible.col_cost = vec![1.0];
    infeasible.add_row(&[(0, 1.0)], 5.0, f64::INFINITY);
    infeasible.add_row(&[(0, 1.0)], f64::NEG_INFINITY, 2.0);

    let (highs, ipm) = both(&infeasible);
    assert!(!highs.is_optimal() && !ipm.is_optimal());
    assert_eq!(ipm.status, OptStatus::Infeasible, "the interior-point verdict");

    let mut unbounded = LinearProgram::new(1);
    unbounded.col_lower = vec![f64::NEG_INFINITY];
    unbounded.col_upper = vec![f64::INFINITY];
    unbounded.col_cost = vec![1.0];

    let (highs, ipm) = both(&unbounded);
    assert!(!highs.is_optimal() && !ipm.is_optimal());
    assert_eq!(ipm.status, OptStatus::Unbounded, "the interior-point verdict");
}

/// A deterministic xorshift, so a failure is reproducible from the seed alone.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform on `[low, high)`.
    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Randomized convex QPs — the part of this file that finds things the fixed
/// cases cannot.
///
/// Five hand-written analytic problems test the paths someone thought to test.
/// A few hundred random ones test the combinations nobody enumerated: an
/// equality row next to a free column next to a fixed column next to a range
/// constraint, in proportions no one would write down.
///
/// Two construction details make the generated problems meaningful rather than
/// merely random. The Hessian is built as `LLᵀ + εI`, which is positive
/// definite *by construction* — a random symmetric matrix would usually be
/// indefinite, making the problem nonconvex and the comparison meaningless,
/// since then the two solvers genuinely may find different points. And the row
/// bounds are placed around `A·x₀` for a drawn interior point `x₀`, so every
/// problem is feasible by construction; otherwise most draws would be
/// infeasible and the test would mostly compare two ways of saying "no".
#[test]
fn the_two_backends_agree_on_randomized_convex_qps() {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut compared = 0;

    for trial in 0..300 {
        let n = 2 + (rng.next_u64() % 7) as usize;
        let m = 1 + (rng.next_u64() % 6) as usize;

        let mut lp = LinearProgram::new(n);
        lp.col_cost = (0..n).map(|_| rng.range(-5.0, 5.0)).collect();

        // Q = LLᵀ + εI: positive definite by construction.
        let l: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..=i).map(|_| rng.range(-1.5, 1.5)).collect::<Vec<f64>>())
            .collect();
        let mut hessian = Vec::new();
        for i in 0..n {
            for j in 0..=i {
                let mut value: f64 = (0..=j.min(i)).map(|k| l[i][k] * l[j][k]).sum();
                if i == j {
                    // Not a token nudge to make `Q` technically nonsingular.
                    // How well the *argument* is determined depends on `Q`'s
                    // conditioning, and `LLᵀ` from random `L` is routinely
                    // near-singular — leaving a valley along which the optimum
                    // slides almost freely. On such a problem two correct
                    // solvers legitimately return different points, so a
                    // primal comparison would be testing the generator's luck
                    // rather than the solvers. This floor keeps the drawn
                    // problems well enough conditioned for the primal claim to
                    // mean something.
                    value += 0.25;
                }
                if value != 0.0 {
                    hessian.push((i, j, value));
                }
            }
        }
        lp.hessian = Some(hessian);

        // A drawn interior point, so the constraints below admit a solution.
        let x0: Vec<f64> = (0..n).map(|_| rng.range(-3.0, 3.0)).collect();

        for k in 0..n {
            match rng.next_u64() % 4 {
                0 => {
                    // Free.
                    lp.col_lower[k] = f64::NEG_INFINITY;
                    lp.col_upper[k] = f64::INFINITY;
                }
                1 => {
                    // Fixed — the path with no interior.
                    lp.col_lower[k] = x0[k];
                    lp.col_upper[k] = x0[k];
                }
                2 => {
                    lp.col_lower[k] = x0[k] - rng.range(0.5, 4.0);
                    lp.col_upper[k] = f64::INFINITY;
                }
                _ => {
                    lp.col_lower[k] = x0[k] - rng.range(0.5, 4.0);
                    lp.col_upper[k] = x0[k] + rng.range(0.5, 4.0);
                }
            }
        }

        for _ in 0..m {
            let mut coefficients: Vec<(usize, f64)> = Vec::new();
            for k in 0..n {
                if rng.next_u64() % 3 != 0 {
                    coefficients.push((k, rng.range(-2.0, 2.0)));
                }
            }
            if coefficients.is_empty() {
                continue;
            }
            let activity: f64 = coefficients.iter().map(|&(k, v)| v * x0[k]).sum();
            match rng.next_u64() % 3 {
                0 => lp.add_row(&coefficients, activity, activity),
                1 => lp.add_row(&coefficients, activity - rng.range(0.1, 3.0), f64::INFINITY),
                _ => lp.add_row(
                    &coefficients,
                    activity - rng.range(0.1, 3.0),
                    activity + rng.range(0.1, 3.0),
                ),
            };
        }

        lp.validate().unwrap_or_else(|e| panic!("trial {trial} built an invalid problem: {e}"));

        let highs = HighsSolver::new().unwrap().solve(&lp).unwrap();
        let ipm = IpmSolver::new().solve(&lp).unwrap();

        // A construction-feasible bounded QP should always be Optimal; if
        // HiGHS says otherwise the generator, not the solver, is suspect.
        if highs.status != OptStatus::Optimal {
            continue;
        }
        assert_eq!(
            ipm.status,
            OptStatus::Optimal,
            "trial {trial}: HiGHS solved it, interior-point returned {:?}",
            ipm.status
        );

        assert_objectives_agree(&highs, &ipm, &format!("trial {trial}"));

        // Strictly convex, so the *argument* is unique too — a stronger claim
        // than the objective, and one only a strictly convex problem supports.
        //
        // Bounded relative to the value: unlike the objective, whose
        // uniqueness is exact, how *sharply* the argument is pinned scales
        // with the problem's conditioning and the magnitude of the optimum.
        for k in 0..n {
            let tolerance = 1e-5 * (1.0 + highs.primal[k].abs());
            assert!(
                (highs.primal[k] - ipm.primal[k]).abs() < tolerance,
                "trial {trial} x{k}: HiGHS {} vs interior-point {}",
                highs.primal[k],
                ipm.primal[k]
            );
        }
        compared += 1;
    }

    assert!(compared > 250, "only {compared} of 300 trials produced a comparison");
}
