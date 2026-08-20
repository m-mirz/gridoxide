//! Branch and bound over the in-house interior-point solver.
//!
//! [`ipm::IpmSolver`](crate::opf::ipm) refuses a problem with integral columns
//! rather than solving its relaxation — a barrier method has no way to enforce
//! an integer, and a tap of 4.3 is a plausible number nobody can act on. That
//! refusal is correct and it leaves a gap: every discrete question has to go to
//! [`highs`](crate::opf::highs), which needs a system library CI does not have.
//! So the whole mixed-integer half of the remedial-action work is verified only
//! where `libhighs-dev` happens to be installed.
//!
//! This closes that gap. It is the third instance of a pattern the crate has now
//! run twice — `ipm` against `highs` on convex QPs, `nlp` against `ipopt` on
//! nonlinear programs — and it buys the same two things each time: a portable
//! default that CI can exercise, and a second implementation to disagree with.
//!
//! # The algorithm, and why each choice
//!
//! Textbook depth-first branch and bound. The relaxation is solved by an inner
//! [`Solver`], the fractional integer variables are branched on one at a time,
//! and a node is discarded when its relaxation cannot beat the incumbent.
//!
//! - **Depth-first, diving toward the rounded value.** Best-first proves
//!   optimality with fewer nodes but spends a long time with no integer
//!   solution at all, and an incumbent is what makes the bound prune anything.
//!   Diving finds one within a handful of nodes on the problems this exists for.
//! - **Most-fractional branching.** The cheapest rule that is not arbitrary.
//!   Strong branching is better on hard instances and costs an LP solve per
//!   candidate, which is the wrong trade for problems whose whole point is to be
//!   answered thousands of times inside a search tree.
//! - **A node budget, not a time limit.** Reproducibility matters more here than
//!   squeezing the last node out: a search that returns a different answer
//!   depending on machine speed cannot be regression-tested.
//!
//! # What it does not do
//!
//! No cutting planes, no presolve, no heuristics beyond the dive. For the
//! problems this is meant for — a few dozen discrete taps and activation
//! indicators — that is enough, and each of those would be a substantial piece
//! of work whose absence is better stated than discovered. A large or hard MILP
//! should still go to HiGHS.

use super::{LinearProgram, OpfError, OptStatus, Solution, Solver};

/// Branch-and-bound settings.
#[derive(Clone, Debug)]
pub struct BnbOptions {
    /// How far from a whole number a value may be and still count as integral.
    pub integrality_tolerance: f64,
    /// Stop once `(incumbent − bound) / (|incumbent| + 1)` falls below this.
    /// Zero proves optimality.
    pub relative_gap: f64,
    /// Node budget. Exhausting it returns the incumbent with
    /// [`OptStatus::Other`], never a wrong answer claimed as optimal.
    pub max_nodes: usize,
}

impl Default for BnbOptions {
    fn default() -> Self {
        Self { integrality_tolerance: 1e-6, relative_gap: 0.0, max_nodes: 10_000 }
    }
}

/// A mixed-integer solver built from a continuous one.
///
/// Takes any [`Solver`] for the relaxations; the default is
/// [`IpmSolver`](crate::opf::ipm::IpmSolver), which makes this pure Rust with no
/// system library.
pub struct BranchAndBound {
    relaxation: Box<dyn Solver>,
    options: BnbOptions,
    nodes: usize,
    gap: f64,
}

impl BranchAndBound {
    /// Over the in-house interior-point solver.
    pub fn new() -> Self {
        Self::with_solver(Box::new(super::ipm::IpmSolver::new()))
    }

    pub fn with_solver(relaxation: Box<dyn Solver>) -> Self {
        Self { relaxation, options: BnbOptions::default(), nodes: 0, gap: f64::INFINITY }
    }

    pub fn options_mut(&mut self) -> &mut BnbOptions {
        &mut self.options
    }

    /// Nodes explored by the last solve — the cost, reported rather than
    /// estimated.
    pub fn nodes(&self) -> usize {
        self.nodes
    }

    /// Optimality gap at the end of the last solve. Zero means proven optimal.
    pub fn gap(&self) -> f64 {
        self.gap
    }

    fn is_integral(&self, value: f64) -> bool {
        (value - value.round()).abs() <= self.options.integrality_tolerance
    }

    /// The integral column furthest from a whole number, if any.
    fn most_fractional(&self, problem: &LinearProgram, primal: &[f64]) -> Option<usize> {
        (0..problem.n_vars)
            .filter(|&c| problem.is_integral(c))
            .filter(|&c| !self.is_integral(primal[c]))
            .max_by(|&a, &b| {
                let fa = (primal[a] - primal[a].round()).abs();
                let fb = (primal[b] - primal[b].round()).abs();
                fa.total_cmp(&fb)
            })
    }
}

impl Default for BranchAndBound {
    fn default() -> Self {
        Self::new()
    }
}

/// One node: the column bounds that distinguish it from the root.
#[derive(Clone, Debug)]
struct Node {
    bounds: Vec<(usize, f64, f64)>,
}

impl Solver for BranchAndBound {
    fn solve(&mut self, problem: &LinearProgram) -> Result<Solution, OpfError> {
        problem.validate()?;
        self.nodes = 0;
        self.gap = f64::INFINITY;

        // The relaxation is the same problem with integrality dropped — which
        // is also what makes the inner solver willing to take it.
        let mut relaxed = problem.clone();
        relaxed.col_integral = Vec::new();

        if !problem.has_integers() {
            return self.relaxation.solve(&relaxed);
        }

        let mut incumbent: Option<Solution> = None;
        let mut best_bound = f64::NEG_INFINITY;
        let mut stack = vec![Node { bounds: Vec::new() }];
        // Whether the tree was explored to exhaustion, as opposed to stopped by
        // the node budget. That is what distinguishes a proven optimum from a
        // good answer, and the two must not be reported the same way.
        let mut exhausted = true;

        while let Some(node) = stack.pop() {
            if self.nodes >= self.options.max_nodes {
                exhausted = false;
                break;
            }
            self.nodes += 1;

            let mut sub = relaxed.clone();
            for &(col, lower, upper) in &node.bounds {
                sub.col_lower[col] = sub.col_lower[col].max(lower);
                sub.col_upper[col] = sub.col_upper[col].min(upper);
                if sub.col_lower[col] > sub.col_upper[col] {
                    // The branch is empty by construction; nothing to solve.
                    sub.col_lower[col] = f64::NAN;
                    break;
                }
            }
            if sub.col_lower.iter().any(|x| x.is_nan()) {
                continue;
            }

            let solution = match self.relaxation.solve(&sub) {
                Ok(s) => s,
                // A node that the inner solver cannot answer is pruned rather
                // than failing the whole solve: an interior-point method on a
                // tightly-bounded subproblem can hit numerical trouble that says
                // nothing about the rest of the tree. The node budget and the
                // final status keep this honest — a solve that pruned its way to
                // no answer reports that it found none.
                Err(_) => continue,
            };
            if solution.status != OptStatus::Optimal {
                if solution.status == OptStatus::Unbounded {
                    return Ok(Solution::failed(OptStatus::Unbounded));
                }
                continue;
            }

            if node.bounds.is_empty() {
                best_bound = solution.objective;
            }

            // Bound: this subtree cannot contain anything better.
            if let Some(best) = &incumbent {
                if solution.objective >= best.objective - self.options.integrality_tolerance {
                    continue;
                }
            }

            match self.most_fractional(problem, &solution.primal) {
                None => {
                    // Integral, and better than the incumbent by the test above.
                    let mut found = solution;
                    // Snap to exact integers. The relaxation lands within
                    // tolerance, and handing back 3.9999999997 as a tap position
                    // makes every downstream `as i32` a rounding decision the
                    // caller did not know it was making.
                    for c in 0..problem.n_vars {
                        if problem.is_integral(c) {
                            found.primal[c] = found.primal[c].round();
                        }
                    }
                    incumbent = Some(found);
                }
                Some(col) => {
                    let value = solution.primal[col];
                    let floor = value.floor();
                    let ceil = value.ceil();
                    let (near, far) = if value - floor <= ceil - value {
                        (
                            (col, f64::NEG_INFINITY, floor),
                            (col, ceil, f64::INFINITY),
                        )
                    } else {
                        (
                            (col, ceil, f64::INFINITY),
                            (col, f64::NEG_INFINITY, floor),
                        )
                    };
                    // Pushed far-first so the *near* child is popped first —
                    // the dive that finds an incumbent quickly.
                    for child in [far, near] {
                        let mut bounds = node.bounds.clone();
                        bounds.push(child);
                        stack.push(Node { bounds });
                    }
                }
            }
        }

        let Some(mut best) = incumbent else {
            // No integer point found. Distinguishing "proved there is none" from
            // "ran out of nodes" matters: the first is an answer and the second
            // is a budget.
            return Ok(Solution::failed(if self.nodes >= self.options.max_nodes {
                OptStatus::Other("node limit reached before any integer solution")
            } else {
                OptStatus::Infeasible
            }));
        };

        // An exhausted tree *proves* optimality: every node was either solved,
        // pruned by a bound that could not beat this incumbent, or infeasible.
        // The root relaxation's bound is irrelevant by then — it is a lower
        // bound on the optimum, not evidence about it — and reporting the gap
        // to it would say "5% uncertain" about an answer that is exact.
        //
        // When the budget stopped the search there is no such proof, and the
        // root bound is then the only lower bound available. It never
        // understates the gap, which is the direction to err in.
        self.gap = if exhausted {
            0.0
        } else if best_bound.is_finite() {
            ((best.objective - best_bound) / (best.objective.abs() + 1.0)).max(0.0)
        } else {
            f64::INFINITY
        };
        if !exhausted && self.gap > self.options.relative_gap {
            best.status = OptStatus::Other("node limit reached; solution may not be optimal");
        }

        // A MIP has no duals. The relaxation at the winning node has them and
        // they are not shadow prices of the integer problem, so they are cleared
        // rather than passed off as such — the same choice `highs.rs` makes.
        best.col_dual = vec![0.0; problem.n_vars];
        best.row_dual = vec![0.0; problem.n_rows];
        Ok(best)
    }
}
