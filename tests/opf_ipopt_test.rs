//! IPOPT as a reference for the in-house nonlinear solver.
//!
//! The AC counterpart of `opf_cross_test.rs`, and a weaker instrument for a
//! reason worth stating up front. On a *convex* problem the optimum is unique,
//! so two solvers disagreeing means one is wrong. AC-OPF is **nonconvex**: two
//! correct solvers may land on different local optima and neither is at fault.
//! So agreement here is strong evidence and disagreement is a question rather
//! than a verdict.
//!
//! What it does test unambiguously is the **model**. IPOPT consumes the same
//! [`NonlinearProblem`] — the same objective, Jacobian and Hessian — through a
//! completely separate algorithm. Two independent methods reaching the same
//! point corroborates the derivative algebra and the constraint set from
//! outside this crate.
//!
//! It also exercises two conventions that are silent when wrong: the Hessian's
//! sign (`∇²f − Σyᵢ∇²cᵢ` here, `σ∇²f + Σλᵢ∇²cᵢ` there) and its triangle (full
//! symmetric here, lower-only there). Getting either wrong converges to the
//! wrong point rather than failing, so the analytic cases below — whose answers
//! are known on paper — are what actually pin them.
//!
//! # Running these
//!
//! Needs `--features opf-ipopt` and a local IPOPT. On Debian and Ubuntu that
//! package links MUMPS built against OpenMPI, whose load-time transport
//! discovery hangs the process **before `main`** in environments without the
//! facilities it probes for. `.cargo/config.toml` sets `OMPI_MCA_btl=self` to
//! skip that; see the comment there, since nothing inside the binding can work
//! around a hang that happens before any Rust code runs.

#![cfg(feature = "opf-ipopt")]

use std::path::PathBuf;

use gridoxide::opf::ac::{AcOpf, AcOpfNetwork, AcOpfOptions};
use gridoxide::opf::ipopt::{self, IpoptOptions};
use gridoxide::opf::model::OpfData;
use gridoxide::opf::nlp::{self, NlpOptions, NonlinearProblem};
use gridoxide::opf::OptStatus;
use gridoxide::pgm::PgmInput;

const PUBLISHED_AC: &[(&str, f64)] = &[
    ("pglib_opf_case3_lmbd", 5.8126e+03),
    ("pglib_opf_case5_pjm", 1.7552e+04),
    ("pglib_opf_case14_ieee", 2.1781e+03),
    ("pglib_opf_case30_ieee", 8.2085e+03),
    ("pglib_opf_case118_ieee", 9.7214e+04),
];

fn ac_opf(name: &str) -> AcOpf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pglib-opf");
    let input: PgmInput =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap())
            .unwrap();
    let data =
        OpfData::from_json(&std::fs::read_to_string(dir.join(format!("{name}.opf.json"))).unwrap())
            .unwrap();
    let options = AcOpfOptions::default();
    let network = AcOpfNetwork::from_pgm(input, &data, 50.0, &options).unwrap();
    AcOpf::build(network, options).unwrap()
}

/// The linked library identifies itself, which is the cheapest proof the FFI
/// is wired to a real IPOPT rather than to a stub that returns zeros.
#[test]
fn the_linked_library_reports_a_version() {
    let version = ipopt::version();
    let parts: Vec<&str> = version.split('.').collect();
    assert_eq!(parts.len(), 3, "version {version:?} is not major.minor.release");
    assert!(
        parts[0].parse::<u32>().unwrap() >= 3,
        "IPOPT {version} is older than the 3.x C interface this binding targets"
    );
}

/// `min x0 + 2·x1` on the unit circle. The optimum is where the objective's
/// gradient is antiparallel to the constraint's: `x = −(1, 2)/√5`, objective
/// `−√5`.
///
/// The same problem `opf_nlp_test.rs` uses, deliberately. Its answer is known
/// on paper, so it tests IPOPT's view of our model against arithmetic rather
/// than against our solver — which is what makes it capable of catching a
/// Hessian sign or triangle error, where comparing two solvers that shared the
/// mistake could not.
struct OnACircle;

impl NonlinearProblem for OnACircle {
    fn n_vars(&self) -> usize {
        2
    }
    fn n_constraints(&self) -> usize {
        1
    }
    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![-10.0; 2], vec![10.0; 2])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![1.0], vec![1.0])
    }
    fn objective(&self, x: &[f64]) -> f64 {
        x[0] + 2.0 * x[1]
    }
    fn gradient(&self, _x: &[f64]) -> Vec<f64> {
        vec![1.0, 2.0]
    }
    fn constraints(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] + x[1] * x[1]]
    }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 2.0 * x[0]), (0, 1, 2.0 * x[1])]
    }
    fn lagrangian_hessian(&self, _x: &[f64], y: &[f64]) -> Vec<(usize, usize, f64)> {
        let d = -2.0 * y[0];
        vec![(0, 0, d), (1, 1, d)]
    }
    fn initial_point(&self) -> Vec<f64> {
        vec![0.6, -0.4]
    }
}

#[test]
fn ipopt_solves_a_problem_whose_answer_is_known_on_paper() {
    let solution = ipopt::solve(&OnACircle, &IpoptOptions::default()).unwrap();
    assert_eq!(solution.status, OptStatus::Optimal);

    let root5 = 5.0f64.sqrt();
    assert!((solution.objective + root5).abs() < 1e-7, "objective {}", solution.objective);
    assert!((solution.x[0] + 1.0 / root5).abs() < 1e-7, "x0 {}", solution.x[0]);
    assert!((solution.x[1] + 2.0 / root5).abs() < 1e-7, "x1 {}", solution.x[1]);
    assert!(solution.violation < 1e-8, "violation {}", solution.violation);
}

/// **A curved constraint is what makes the Hessian conversion observable.**
///
/// With a *linear* constraint the constraint Hessian is zero, so negating the
/// multipliers on the way to IPOPT changes nothing and a sign error hides
/// completely. `x0² + x1² = 1` curves, so `Σλᵢ∇²cᵢ` is nonzero and the wrong
/// sign gives IPOPT the negated curvature — which on a problem this small is
/// the difference between converging in ten iterations and thrashing.
///
/// So this asserts the *iteration count* stays small, not merely that the
/// answer is right: a mis-signed Hessian still often reaches the optimum
/// eventually, because IPOPT falls back on regularization and its own
/// safeguards. Speed is the symptom that shows first.
#[test]
fn the_hessian_conversion_is_right_and_not_merely_survivable() {
    let mut options = IpoptOptions::default();
    options.max_iterations = 25;

    let solution = ipopt::solve(&OnACircle, &options).unwrap();
    assert_eq!(
        solution.status,
        OptStatus::Optimal,
        "with a correctly signed Hessian this converges well inside 25 iterations"
    );
}

/// Both solvers on every fixture, against each other and against the published
/// objectives.
#[test]
fn ipopt_and_the_in_house_solver_agree_on_every_ac_case() {
    for &(name, published) in PUBLISHED_AC {
        let opf = ac_opf(name);
        let ours = nlp::solve(&opf, &NlpOptions::default()).unwrap();
        let theirs = ipopt::solve(&opf, &IpoptOptions::default()).unwrap();

        assert_eq!(ours.status, OptStatus::Optimal, "{name}: ours");
        assert_eq!(theirs.status, OptStatus::Optimal, "{name}: IPOPT");

        // Both feasible. An objective at an infeasible point is not comparable
        // to anything, so this is checked before the objectives are.
        assert!(ours.violation < 1e-6, "{name}: ours left violation {:.2e}", ours.violation);
        assert!(theirs.violation < 1e-6, "{name}: IPOPT left violation {:.2e}", theirs.violation);

        let between = (ours.objective - theirs.objective).abs() / published;
        assert!(
            between < 1e-6,
            "{name}: ours {} vs IPOPT {} ({between:.2e} relative)",
            ours.objective,
            theirs.objective
        );
        for (label, objective) in [("ours", ours.objective), ("IPOPT", theirs.objective)] {
            let gap = (objective - published).abs() / published;
            assert!(gap < 1e-4, "{name}: {label} {objective} vs published {published}");
        }
    }
}

/// Piecewise-linear costs through IPOPT too.
///
/// The epigraph adds variables and rows that only this cost model produces, so
/// without this the reference backend would never see them — and a mistake in
/// that block would be corroborated by nothing.
#[test]
fn ipopt_agrees_on_a_piecewise_linear_case() {
    let polynomial = ipopt::solve(&ac_opf("pglib_opf_case5_pjm"), &IpoptOptions::default()).unwrap();
    let piecewise = ipopt::solve(&ac_opf("case5_pjm_pwl"), &IpoptOptions::default()).unwrap();

    assert_eq!(polynomial.status, OptStatus::Optimal);
    assert_eq!(piecewise.status, OptStatus::Optimal);
    // `case5_pjm_pwl` is an exactly equivalent rewrite, so the answers must
    // match — see `opf_pwl_test.rs`.
    assert!(
        (polynomial.objective - piecewise.objective).abs() < 1e-3,
        "polynomial {} vs piecewise {}",
        polynomial.objective,
        piecewise.objective
    );
}

/// Prices agree too, which is a sharper test than the objective.
///
/// A multiplier is not determined by the objective value: a degenerate optimum
/// admits a whole face of valid multipliers, and two methods can pick
/// different ones. Agreement across every bus therefore says these problems
/// are non-degenerate in their duals — which is what makes publishing the
/// prices meaningful at all.
#[test]
fn ipopt_and_the_in_house_solver_agree_on_prices() {
    for &(name, _) in PUBLISHED_AC {
        let opf = ac_opf(name);
        let ours = opf.interpret(&nlp::solve(&opf, &NlpOptions::default()).unwrap());
        let theirs = opf.interpret(&ipopt::solve(&opf, &IpoptOptions::default()).unwrap());

        for bus in 0..ours.lmp_p.len() {
            let (a, b) = (ours.lmp_p[bus], theirs.lmp_p[bus]);
            assert!(
                (a - b).abs() < 1e-3 * a.abs().max(1.0),
                "{name} bus {bus}: ours priced it at {a} $/MWh, IPOPT at {b}"
            );
        }
    }
}

/// An impossible iteration budget comes back as a reported status, not a
/// panic, a hang, or a silently wrong answer.
#[test]
fn a_failed_solve_is_reported_rather_than_disguised() {
    let mut options = IpoptOptions::default();
    options.max_iterations = 1;

    let solution = ipopt::solve(&ac_opf("pglib_opf_case118_ieee"), &options).unwrap();
    assert_ne!(solution.status, OptStatus::Optimal);
    // And the violation still describes the point it stopped at, so a caller
    // can tell "gave up while far away" from "gave up while nearly there".
    assert!(solution.violation.is_finite());
}
