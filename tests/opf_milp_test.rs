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

// ---------------------------------------------------------------------------
// Branch and bound over the in-house solver — phase 10
// ---------------------------------------------------------------------------

use gridoxide::opf::bnb::BranchAndBound;

/// The point of the whole exercise: a MILP solved with no system library.
#[test]
fn branch_and_bound_solves_the_integer_problem_in_pure_rust() {
    let mut lp = textbook_program();
    lp.set_integral(0);
    lp.set_integral(1);

    let mut solver = BranchAndBound::new();
    let solution = solver.solve(&lp).expect("MILP solve");
    assert_eq!(solution.status, OptStatus::Optimal);
    assert!(
        (solution.objective + 20.0).abs() < 1e-6,
        "objective {} — 21 would mean the relaxation was solved",
        solution.objective
    );
    assert!((solution.primal[0] - 4.0).abs() < 1e-9, "{:?}", solution.primal);
    assert!(solution.primal[1].abs() < 1e-9, "{:?}", solution.primal);
    // Proved, not merely found.
    assert_eq!(solver.gap(), 0.0, "gap {}", solver.gap());
    assert!(solver.nodes() > 0);
}

/// Values come back as exact integers, not as 3.9999999997.
///
/// The relaxation lands within tolerance and no closer. Handing that back makes
/// every downstream `as i32` a rounding decision the caller did not know it was
/// making — and `as i32` truncates, so 3.9999999997 becomes tap 3.
#[test]
fn integral_columns_come_back_exactly_integral() {
    let mut lp = textbook_program();
    lp.set_integral(0);
    lp.set_integral(1);
    let solution = BranchAndBound::new().solve(&lp).expect("solve");
    for (i, x) in solution.primal.iter().enumerate() {
        if lp.is_integral(i) {
            assert_eq!(*x, x.round(), "column {i} came back at {x}");
        }
    }
}

#[test]
fn a_continuous_problem_passes_straight_through() {
    // No integrality means no branching, and the answer must be identical to
    // the inner solver's — this wrapper is not allowed to perturb the LP path.
    let lp = textbook_program();
    let direct = IpmSolver::new().solve(&lp).expect("ipm");
    let mut solver = BranchAndBound::new();
    let wrapped = solver.solve(&lp).expect("bnb");
    assert_eq!(wrapped.status, direct.status);
    assert!((wrapped.objective - direct.objective).abs() < 1e-12);
    assert_eq!(solver.nodes(), 0, "a continuous problem should branch not at all");
    // And its duals survive, since a continuous solve has them.
    assert!(wrapped.row_dual.iter().any(|d| d.abs() > 1e-9));
}

#[test]
fn a_mixed_problem_leaves_its_continuous_columns_alone() {
    // min -5x - 4y, x integral and y not. The optimum has y fractional.
    let mut lp = textbook_program();
    lp.set_integral(0);
    let solution = BranchAndBound::new().solve(&lp).expect("solve");
    assert_eq!(solution.status, OptStatus::Optimal);
    assert_eq!(solution.primal[0], solution.primal[0].round(), "x should be integral");
    // x = 3, y = 1.5 is feasible and better than any all-integer point.
    assert!((solution.objective + 21.0).abs() < 1e-6, "{}", solution.objective);
    assert!((solution.primal[1] - 1.5).abs() < 1e-6, "{:?}", solution.primal);
}

#[test]
fn binaries_are_honoured() {
    let mut lp = LinearProgram::new(3);
    lp.col_cost = vec![-3.0, -2.0, -1.0];
    for c in 0..3 {
        lp.set_binary(c);
    }
    lp.add_row(&[(0, 1.0), (1, 1.0), (2, 1.0)], f64::NEG_INFINITY, 2.0);

    let solution = BranchAndBound::new().solve(&lp).expect("solve");
    assert_eq!(solution.status, OptStatus::Optimal);
    // Take the two most valuable.
    assert!((solution.objective + 5.0).abs() < 1e-6, "{}", solution.objective);
    assert_eq!(solution.primal, vec![1.0, 1.0, 0.0]);
}

#[test]
fn an_infeasible_integer_problem_is_reported_as_infeasible() {
    // 2x = 1 with x integral: the relaxation solves at 0.5 and no integer point
    // exists. Reporting "optimal" at 0 or 1 would be a wrong answer, and
    // reporting a node limit would blame the budget for a genuine result.
    let mut lp = LinearProgram::new(1);
    lp.col_lower = vec![0.0];
    lp.col_upper = vec![1.0];
    lp.col_cost = vec![1.0];
    lp.set_integral(0);
    lp.add_row(&[(0, 2.0)], 1.0, 1.0);

    let solution = BranchAndBound::new().solve(&lp).expect("solve");
    assert_eq!(solution.status, OptStatus::Infeasible, "{solution:?}");
}

#[test]
fn a_node_budget_is_reported_rather_than_silently_passed_off_as_optimal() {
    let mut lp = textbook_program();
    lp.set_integral(0);
    lp.set_integral(1);
    let mut solver = BranchAndBound::new();
    solver.options_mut().max_nodes = 1;
    let solution = solver.solve(&lp).expect("solve");
    // Either it found nothing, or it found something it cannot call optimal.
    // What it must never do is claim optimality it did not prove.
    match solution.status {
        OptStatus::Other(_) => {}
        OptStatus::Optimal => assert_eq!(solver.gap(), 0.0, "claimed optimal with a gap"),
        other => panic!("unexpected status {other:?}"),
    }
    assert!(solver.nodes() <= 1);
}

/// §8.4's cross-check, the third instance of it in this crate.
///
/// For a MILP the comparison is stronger than the nonconvex NLP case and weaker
/// than the convex QP one: the optimal *objective* is unique, so a disagreement
/// there is a bug in one of them — but the optimal *solution* need not be, so
/// which columns took which values proves nothing. Assert on the objective and
/// on feasibility, never on the argmin.
#[cfg(feature = "opf-highs")]
#[test]
fn branch_and_bound_agrees_with_highs_on_randomised_milps() {
    // A tiny deterministic generator: reproducibility matters more than
    // statistical purity, and a failure has to be re-runnable.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut uniform = move || (next() >> 11) as f64 / (1u64 << 53) as f64;

    let mut compared = 0;
    for case in 0..60 {
        let n = 3 + (case % 4);
        let mut lp = LinearProgram::new(n);
        for c in 0..n {
            lp.col_lower[c] = 0.0;
            lp.col_upper[c] = 4.0;
            lp.col_cost[c] = -(1.0 + 4.0 * uniform());
            if c % 2 == 0 {
                lp.set_integral(c);
            }
        }
        for _ in 0..(2 + case % 3) {
            let coefficients: Vec<(usize, f64)> =
                (0..n).map(|c| (c, 0.5 + 2.0 * uniform())).collect();
            lp.add_row(&coefficients, f64::NEG_INFINITY, 4.0 + 6.0 * uniform());
        }

        let mut highs = HighsSolver::new().expect("HiGHS instance");
        highs.set_output(false).expect("quiet");
        let reference = highs.solve(&lp).expect("highs");
        let mut bnb = BranchAndBound::new();
        let mine = bnb.solve(&lp).expect("bnb");

        if reference.status != OptStatus::Optimal {
            continue;
        }
        assert_eq!(mine.status, OptStatus::Optimal, "case {case}: {mine:?}");
        assert!(
            (mine.objective - reference.objective).abs()
                <= 1e-6 * (reference.objective.abs() + 1.0),
            "case {case}: bnb {} vs highs {}",
            mine.objective,
            reference.objective
        );
        // And the answer must actually be integral and feasible, not merely
        // equal in objective.
        for c in 0..n {
            if lp.is_integral(c) {
                assert_eq!(mine.primal[c], mine.primal[c].round(), "case {case} column {c}");
            }
            assert!(mine.primal[c] >= -1e-9 && mine.primal[c] <= 4.0 + 1e-9);
        }
        compared += 1;
    }
    assert!(compared >= 40, "only {compared} cases were comparable");
}
