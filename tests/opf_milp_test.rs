//! Integrality on the LP boundary — phase 6 of `plans/RAO_PLAN.md`.
//!
//! `LinearProgram` could not express an integer until now, so nothing above the
//! solver boundary could ask a discrete question. The remedial-action work needs
//! to: a phase shifter's tap is an integer, "use at most three actions" is a
//! cardinality constraint, and an activation cost is paid per *decision*, not
//! per MW.
//!
//! The load-bearing property here is not that the MIP solves. It is that a
//! backend which cannot honour integrality **refuses** rather than solving the
//! relaxation. A relaxed answer to a discrete question is a plausible number
//! nobody can act on — a phase shifter cannot sit at tap 4.3 — and returning one
//! silently is worse than returning nothing.

use gridoxide::opf::{LinearProgram, OpfError, OptStatus, Solver};

#[cfg(feature = "opf-highs")]
use gridoxide::opf::highs::HighsSolver;
use gridoxide::opf::ipm::IpmSolver;

/// max 5x + 4y  s.t.  6x + 4y <= 24, x + 2y <= 6, x,y >= 0.
///
/// The continuous optimum is x = 3, y = 1.5 with objective 21; the integer
/// optimum is x = 4, y = 0 with objective 20. Small, but the two answers differ
/// in both the objective *and* the argument, which is the whole point — a
/// backend that quietly solved the relaxation would return 21 at a fractional
/// point and look perfectly reasonable.
fn textbook_program() -> LinearProgram {
    let mut lp = LinearProgram::new(2);
    lp.col_lower = vec![0.0, 0.0];
    lp.col_upper = vec![f64::INFINITY, f64::INFINITY];
    lp.col_cost = vec![-5.0, -4.0]; // minimize the negation
    lp.add_row(&[(0, 6.0), (1, 4.0)], f64::NEG_INFINITY, 24.0);
    lp.add_row(&[(0, 1.0), (1, 2.0)], f64::NEG_INFINITY, 6.0);
    lp
}

#[test]
fn a_program_without_integers_is_unchanged() {
    // The field was added to a type every existing caller constructs. Empty
    // must keep meaning "all continuous", or every OPF built before this
    // becomes a MIP.
    let lp = LinearProgram::new(3);
    assert!(lp.col_integral.is_empty());
    assert!(!lp.has_integers());
    assert_eq!(lp.n_integers(), 0);
    assert!(!lp.is_integral(0));
    lp.validate().expect("a continuous program stays valid");
}

#[test]
fn declaring_a_binary_sets_both_the_type_and_the_bounds() {
    // An "integral but unbounded" activation indicator is not a binary, and
    // big-M constraints written against one are wrong in exactly the cases
    // where the bound would have bound.
    let mut lp = LinearProgram::new(2);
    lp.set_binary(1);
    assert!(lp.is_integral(1));
    assert!(!lp.is_integral(0));
    assert_eq!(lp.col_lower[1], 0.0);
    assert_eq!(lp.col_upper[1], 1.0);
    assert_eq!(lp.n_integers(), 1);
    lp.validate().expect("valid");
}

#[test]
fn the_interior_point_backend_refuses_rather_than_relaxing() {
    let mut lp = textbook_program();
    lp.set_integral(0);
    lp.set_integral(1);

    let mut solver = IpmSolver::new();
    let err = solver.solve(&lp).expect_err("a barrier method cannot honour integrality");
    match err {
        OpfError::IntegralityUnsupported { backend, columns } => {
            assert_eq!(columns, 2);
            assert!(backend.contains("interior"), "{backend}");
        }
        other => panic!("expected IntegralityUnsupported, got {other}"),
    }
    // And the message has to say why, since this is the error a caller meets
    // when it needs a different backend.
    assert!(
        err.to_string().contains("relaxation"),
        "the error should explain the refusal: {err}"
    );
}

#[test]
fn the_same_program_without_integrality_still_solves_on_the_interior_point() {
    let lp = textbook_program();
    let mut solver = IpmSolver::new();
    let solution = solver.solve(&lp).expect("continuous solve");
    assert_eq!(solution.status, OptStatus::Optimal);
    // x = 3, y = 1.5, objective -21.
    assert!((solution.primal[0] - 3.0).abs() < 1e-6, "{:?}", solution.primal);
    assert!((solution.primal[1] - 1.5).abs() < 1e-6, "{:?}", solution.primal);
    assert!((solution.objective + 21.0).abs() < 1e-6, "{}", solution.objective);
}

#[test]
fn a_mixed_integer_quadratic_program_is_rejected_at_the_boundary() {
    // Neither backend solves one, so no caller should be able to build one and
    // receive a silently linearized answer.
    let mut lp = textbook_program();
    lp.set_integral(0);
    lp.hessian = Some(vec![(0, 0, 2.0)]);
    match lp.validate() {
        Err(OpfError::IntegerQuadratic) => {}
        other => panic!("expected IntegerQuadratic, got {other:?}"),
    }
}

#[test]
fn a_wrong_length_integrality_vector_is_a_shape_error() {
    let mut lp = LinearProgram::new(3);
    lp.col_integral = vec![true, false];
    match lp.validate() {
        Err(OpfError::Shape { what, expected, got }) => {
            assert_eq!((what, expected, got), ("col_integral", 3, 2));
        }
        other => panic!("expected a shape error, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The MIP itself. Needs a system HiGHS, so it is behind `opf-highs`.
// ---------------------------------------------------------------------------

#[cfg(feature = "opf-highs")]
#[test]
fn highs_solves_the_integer_problem_and_not_its_relaxation() {
    let mut lp = textbook_program();
    lp.set_integral(0);
    lp.set_integral(1);

    let mut solver = HighsSolver::new().expect("HiGHS instance");
    solver.set_output(false).expect("quiet");
    let solution = solver.solve(&lp).expect("MIP solve");
    assert_eq!(solution.status, OptStatus::Optimal);

    // The integer optimum is 20, the continuous one 21. Getting 21 back would
    // mean the integrality declaration never reached HiGHS.
    assert!(
        (solution.objective + 20.0).abs() < 1e-6,
        "objective {} — 21 would mean the relaxation was solved",
        solution.objective
    );
    assert!((solution.primal[0] - 4.0).abs() < 1e-6, "{:?}", solution.primal);
    assert!(solution.primal[1].abs() < 1e-6, "{:?}", solution.primal);
    for (i, x) in solution.primal.iter().enumerate() {
        assert!((x - x.round()).abs() < 1e-6, "column {i} came back at {x}, which is not integral");
    }
}

#[cfg(feature = "opf-highs")]
#[test]
fn a_binary_choice_is_honoured() {
    // min -3a - 2b  s.t.  a + b <= 1, both binary. Exactly one may be taken and
    // the better one is `a`. A relaxation would split them.
    let mut lp = LinearProgram::new(2);
    lp.col_cost = vec![-3.0, -2.0];
    lp.set_binary(0);
    lp.set_binary(1);
    lp.add_row(&[(0, 1.0), (1, 1.0)], f64::NEG_INFINITY, 1.0);

    let mut solver = HighsSolver::new().expect("HiGHS instance");
    solver.set_output(false).expect("quiet");
    let solution = solver.solve(&lp).expect("MIP solve");
    assert_eq!(solution.status, OptStatus::Optimal);
    assert!((solution.primal[0] - 1.0).abs() < 1e-6, "{:?}", solution.primal);
    assert!(solution.primal[1].abs() < 1e-6, "{:?}", solution.primal);
    assert!((solution.objective + 3.0).abs() < 1e-6);
}

#[cfg(feature = "opf-highs")]
#[test]
fn a_mip_reports_no_duals_rather_than_meaningless_ones() {
    // HiGHS fills the dual arrays for a MIP with values from the final LP
    // relaxation at the winning node. They are not shadow prices of the integer
    // problem, and a caller reading them as such gets a plausible wrong answer.
    let mut lp = textbook_program();
    lp.set_integral(0);
    lp.set_integral(1);

    let mut solver = HighsSolver::new().expect("HiGHS instance");
    solver.set_output(false).expect("quiet");
    let solution = solver.solve(&lp).expect("MIP solve");
    assert!(solution.row_dual.iter().all(|d| *d == 0.0), "{:?}", solution.row_dual);
    assert!(solution.col_dual.iter().all(|d| *d == 0.0), "{:?}", solution.col_dual);
}

#[cfg(feature = "opf-highs")]
#[test]
fn the_continuous_answer_is_unaffected_by_the_new_field() {
    // The regression that matters most: every existing OPF goes through this
    // backend, and none of them declares integrality.
    let lp = textbook_program();
    let mut solver = HighsSolver::new().expect("HiGHS instance");
    solver.set_output(false).expect("quiet");
    let solution = solver.solve(&lp).expect("LP solve");
    assert_eq!(solution.status, OptStatus::Optimal);
    assert!((solution.objective + 21.0).abs() < 1e-6, "{}", solution.objective);
    // And the duals must still come through, since a continuous solve has them.
    assert!(solution.row_dual.iter().any(|d| d.abs() > 1e-9), "{:?}", solution.row_dual);
}

#[cfg(feature = "opf-highs")]
#[test]
fn the_two_backends_still_agree_where_both_apply() {
    // The cross-check `tests/opf_cross_test.rs` performs, restated here for the
    // continuous case of a program that *could* have been a MIP — so that the
    // integrality plumbing is shown not to have perturbed the LP path.
    let lp = textbook_program();
    let mut highs = HighsSolver::new().expect("HiGHS instance");
    highs.set_output(false).expect("quiet");
    let a = highs.solve(&lp).expect("highs");
    let b = IpmSolver::new().solve(&lp).expect("ipm");
    assert!((a.objective - b.objective).abs() < 1e-7, "{} vs {}", a.objective, b.objective);
}
