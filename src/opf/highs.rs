//! The HiGHS backend for [`super::Solver`].
//!
//! Links a *system* HiGHS install through FFI bindings this crate generates
//! itself — see `mod highs` in `build.rs`. Nothing is vendored and no
//! third-party binding crate is involved, the same position `sparse_pardiso`
//! takes toward oneMKL.
//!
//! # Why generated bindings rather than hand-written `extern "C"`
//!
//! One detail decides it. `HighsInt` is `int64_t` or `int` depending on whether
//! the library was built with `HIGHSINT64`, and only that install's
//! `HConfig.h` knows which. A hand-written `c_int` would compile everywhere,
//! be correct on a default build, and silently misread every index array on a
//! 64-bit-index one. bindgen reads the header and gets it right by
//! construction.
//!
//! # What HiGHS is asked for
//!
//! An interior-point or simplex solve, its choice — the default is simplex for
//! an LP, which additionally yields a *basic* solution and therefore vertex
//! duals. That matters once these duals become locational marginal prices: at
//! a degenerate optimum the dual is not unique, and a vertex dual is the one a
//! market would publish, where an interior-point method returns something in
//! the middle of the optimal face.

use std::ffi::CString;
use std::os::raw::c_void;

use super::{LinearProgram, OpfError, OptStatus, Solution, Solver};

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/highs_bindings.rs"));
}

/// The integer width this HiGHS build uses — see the module docs.
type Int = bindings::HighsInt;

/// A HiGHS instance.
///
/// Owns the opaque `void*` HiGHS hands out and frees it on drop. One instance
/// can solve many problems; each [`solve`](Solver::solve) call passes a fresh
/// model, so nothing carries over between them.
pub struct HighsSolver {
    handle: *mut c_void,
}

impl HighsSolver {
    /// Creates an instance, silent by default.
    ///
    /// HiGHS prints a solve log to stdout unless told otherwise, which would
    /// interleave with gridoxide's own output and, worse, with a test
    /// harness's. Callers who want the log can [`set_output`](Self::set_output).
    pub fn new() -> Result<Self, OpfError> {
        // SAFETY: `Highs_create` takes no arguments and returns either a valid
        // instance pointer or null, which is checked immediately below.
        let handle = unsafe { bindings::Highs_create() };
        if handle.is_null() {
            return Err(OpfError::Backend("Highs_create returned null".to_string()));
        }
        let solver = Self { handle };
        solver.set_output(false)?;
        Ok(solver)
    }

    /// Turns HiGHS's own solve log on or off.
    pub fn set_output(&self, enabled: bool) -> Result<(), OpfError> {
        let option = CString::new("output_flag").expect("literal has no interior nul");
        // SAFETY: `self.handle` is non-null for the lifetime of `self`, and
        // `option` outlives the call.
        let status = unsafe {
            bindings::Highs_setBoolOptionValue(
                self.handle,
                option.as_ptr(),
                Int::from(enabled),
            )
        };
        check(status, "Highs_setBoolOptionValue(output_flag)")
    }

    /// Tightens (or loosens) HiGHS's primal and dual feasibility tolerances.
    ///
    /// Left at HiGHS's own defaults unless a caller asks, because tightening
    /// costs iterations and the default is right for ordinary use. It matters
    /// in two places. An LP solved by simplex lands on an exact vertex, so its
    /// answer is as precise as the arithmetic; a **QP is solved by an
    /// interior-point method and converges asymptotically**, so its answer is
    /// only as precise as the tolerance asked for. And the phase-4
    /// cross-check between this backend and the in-house solver is only as
    /// sharp as the looser of the two.
    ///
    /// HiGHS refuses anything below `1e-10`.
    pub fn set_tolerance(&self, primal: f64, dual: f64) -> Result<(), OpfError> {
        for (name, value) in
            [("primal_feasibility_tolerance", primal), ("dual_feasibility_tolerance", dual)]
        {
            let option = CString::new(name).expect("literal has no interior nul");
            // SAFETY: `self.handle` is non-null for the lifetime of `self`, and
            // `option` outlives the call.
            let status = unsafe {
                bindings::Highs_setDoubleOptionValue(self.handle, option.as_ptr(), value)
            };
            check(status, name)?;
        }
        Ok(())
    }

    /// The linked library's version, as `(major, minor, patch)`.
    ///
    /// Useful in a failure report: the bindings are generated against whatever
    /// header was present at build time, so knowing what is actually loaded
    /// turns "it behaves oddly" into a checkable statement.
    pub fn version() -> (i32, i32, i32) {
        // SAFETY: these take no arguments and only read static data.
        unsafe {
            (
                bindings::Highs_versionMajor() as i32,
                bindings::Highs_versionMinor() as i32,
                bindings::Highs_versionPatch() as i32,
            )
        }
    }
}

impl Drop for HighsSolver {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `Highs_create`, has not been freed
        // elsewhere (nothing else in this type frees it), and is not used
        // again.
        unsafe { bindings::Highs_destroy(self.handle) };
    }
}

impl Solver for HighsSolver {
    fn solve(&mut self, problem: &LinearProgram) -> Result<Solution, OpfError> {
        problem.validate()?;

        let n_col = problem.n_vars;
        let n_row = problem.n_rows;
        let a = Csc::from_triplets(&problem.rows, n_col, n_row);

        // SAFETY for this block: every pointer below is derived from a live
        // local `Vec` whose length matches the count passed alongside it, and
        // HiGHS copies what it is given rather than retaining the pointers.
        let status = unsafe {
            bindings::Highs_passLp(
                self.handle,
                n_col as Int,
                n_row as Int,
                a.values.len() as Int,
                bindings::kHighsMatrixFormatColwise,
                bindings::kHighsObjSenseMinimize,
                problem.offset,
                problem.col_cost.as_ptr(),
                problem.col_lower.as_ptr(),
                problem.col_upper.as_ptr(),
                problem.row_lower.as_ptr(),
                problem.row_upper.as_ptr(),
                a.starts.as_ptr(),
                a.indices.as_ptr(),
                a.values.as_ptr(),
            )
        };
        check(status, "Highs_passLp")?;

        if let Some(hessian) = &problem.hessian {
            // The lower triangle, column-wise — `LinearProgram::hessian`
            // documents that convention and `validate` has already rejected an
            // upper-triangle entry, so no filtering is needed here.
            let q = Csc::from_triplets_transposed(hessian, n_col);
            let status = unsafe {
                bindings::Highs_passHessian(
                    self.handle,
                    n_col as Int,
                    q.values.len() as Int,
                    bindings::kHighsHessianFormatTriangular,
                    q.starts.as_ptr(),
                    q.indices.as_ptr(),
                    q.values.as_ptr(),
                )
            };
            check(status, "Highs_passHessian")?;
        }

        if problem.has_integers() {
            // Declared *after* `Highs_passLp`, which always builds a continuous
            // model — this is the call that turns it into a MIP, and HiGHS then
            // uses its branch-and-cut solver instead of the simplex.
            let integrality: Vec<Int> = (0..n_col)
                .map(|c| {
                    if problem.is_integral(c) {
                        bindings::kHighsVarTypeInteger
                    } else {
                        bindings::kHighsVarTypeContinuous
                    }
                })
                .collect();
            // SAFETY: the array is exactly `n_col` long, which is the range
            // being changed, and HiGHS only reads it.
            let status = unsafe {
                bindings::Highs_changeColsIntegralityByRange(
                    self.handle,
                    0,
                    n_col as Int - 1,
                    integrality.as_ptr(),
                )
            };
            check(status, "Highs_changeColsIntegralityByRange")?;
        }

        // SAFETY: as above; `Highs_run` only mutates HiGHS-owned state.
        let status = unsafe { bindings::Highs_run(self.handle) };
        check(status, "Highs_run")?;

        // SAFETY: reads HiGHS-owned state, no pointers crossed.
        let model_status = unsafe { bindings::Highs_getModelStatus(self.handle) };
        let status = map_status(model_status);
        if status != OptStatus::Optimal {
            return Ok(Solution::failed(status));
        }

        let mut primal = vec![0.0; n_col];
        let mut col_dual = vec![0.0; n_col];
        let mut row_activity = vec![0.0; n_row];
        let mut row_dual = vec![0.0; n_row];
        // SAFETY: each buffer is sized to the dimension HiGHS was given, which
        // is what it writes.
        let get = unsafe {
            bindings::Highs_getSolution(
                self.handle,
                primal.as_mut_ptr(),
                col_dual.as_mut_ptr(),
                row_activity.as_mut_ptr(),
                row_dual.as_mut_ptr(),
            )
        };
        check(get, "Highs_getSolution")?;

        // A MIP has no duals. HiGHS fills the arrays anyway — with values from
        // the final LP relaxation at the winning node, which are *not* shadow
        // prices of the integer problem and mean nothing to a caller reading
        // them as such. Clearing them makes the absence explicit rather than
        // letting a plausible-looking number through.
        let (col_dual, row_dual) = if problem.has_integers() {
            (vec![0.0; n_col], vec![0.0; n_row])
        } else {
            (col_dual, row_dual)
        };

        // SAFETY: reads HiGHS-owned state.
        let objective = unsafe { bindings::Highs_getObjectiveValue(self.handle) };

        Ok(Solution { status, objective, primal, row_activity, col_dual, row_dual })
    }
}

/// Turns a HiGHS return code into an error naming the call that produced it.
fn check(status: Int, call: &str) -> Result<(), OpfError> {
    if status == bindings::kHighsStatusOk {
        return Ok(());
    }
    // `kHighsStatusWarning` is not a failure — HiGHS uses it for things like a
    // recoverable option adjustment, and the solve is still valid.
    if status == bindings::kHighsStatusWarning {
        return Ok(());
    }
    let (major, minor, patch) = HighsSolver::version();
    Err(OpfError::Backend(format!(
        "{call} failed with HiGHS status {status} (library {major}.{minor}.{patch})"
    )))
}

/// Maps HiGHS's model status onto [`OptStatus`].
///
/// `UnboundedOrInfeasible` is deliberately *not* folded into either: HiGHS
/// returns it when presolve proves the problem is one of the two without
/// determining which, and claiming a specific one would be inventing
/// information.
fn map_status(status: Int) -> OptStatus {
    if status == bindings::kHighsModelStatusOptimal {
        OptStatus::Optimal
    } else if status == bindings::kHighsModelStatusInfeasible {
        OptStatus::Infeasible
    } else if status == bindings::kHighsModelStatusUnbounded {
        OptStatus::Unbounded
    } else if status == bindings::kHighsModelStatusUnboundedOrInfeasible {
        OptStatus::Other("unbounded or infeasible (HiGHS did not distinguish)")
    } else if status == bindings::kHighsModelStatusIterationLimit {
        OptStatus::Other("iteration limit")
    } else if status == bindings::kHighsModelStatusTimeLimit {
        OptStatus::Other("time limit")
    } else {
        OptStatus::Other("no optimal solution")
    }
}

/// A matrix in compressed sparse column form, which is what HiGHS takes.
struct Csc {
    starts: Vec<Int>,
    indices: Vec<Int>,
    values: Vec<f64>,
}

impl Csc {
    /// Converts `(row, col, value)` triplets, summing duplicates.
    ///
    /// Summing matches the accumulation semantics `network::YBus` documents
    /// and `sparse::solve_complex` relies on, so a caller that stamps the same
    /// entry twice gets the same answer here as everywhere else in the crate.
    fn from_triplets(triplets: &[(usize, usize, f64)], n_col: usize, _n_row: usize) -> Self {
        Self::build(n_col, triplets.iter().map(|&(row, col, v)| (col, row, v)))
    }

    /// The same, for `(i, j, value)` triplets that are already indexed
    /// `(row, col)` in the *transposed* sense a symmetric matrix's lower
    /// triangle is given in — column `j`, row `i`.
    fn from_triplets_transposed(triplets: &[(usize, usize, f64)], n_col: usize) -> Self {
        Self::build(n_col, triplets.iter().map(|&(i, j, v)| (j, i, v)))
    }

    fn build(n_col: usize, entries: impl Iterator<Item = (usize, usize, f64)>) -> Self {
        let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_col];
        for (col, row, value) in entries {
            by_col[col].push((row, value));
        }

        let mut starts = Vec::with_capacity(n_col + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for column in &mut by_col {
            starts.push(indices.len() as Int);
            column.sort_unstable_by_key(|&(row, _)| row);
            // Sum duplicates rather than emitting them twice: HiGHS rejects a
            // matrix with repeated entries in a column.
            let mut last_row: Option<usize> = None;
            for &(row, value) in column.iter() {
                if last_row == Some(row) {
                    *values.last_mut().expect("last_row implies a value") += value;
                } else {
                    indices.push(row as Int);
                    values.push(value);
                    last_row = Some(row);
                }
            }
        }
        starts.push(indices.len() as Int);

        Self { starts, indices, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csc_orders_by_column_then_row() {
        // Deliberately unordered input.
        let triplets = vec![(1, 1, 5.0), (0, 0, 1.0), (1, 0, 2.0), (0, 1, 4.0)];
        let csc = Csc::from_triplets(&triplets, 2, 2);
        assert_eq!(csc.starts, vec![0, 2, 4]);
        assert_eq!(csc.indices, vec![0, 1, 0, 1]);
        assert_eq!(csc.values, vec![1.0, 2.0, 4.0, 5.0]);
    }

    /// Duplicates sum, matching the rest of the crate — and HiGHS would reject
    /// them if they were passed through unmerged.
    #[test]
    fn csc_sums_duplicate_entries() {
        let triplets = vec![(0, 0, 1.5), (0, 0, 2.5), (1, 0, 1.0)];
        let csc = Csc::from_triplets(&triplets, 1, 2);
        assert_eq!(csc.starts, vec![0, 2]);
        assert_eq!(csc.indices, vec![0, 1]);
        assert_eq!(csc.values, vec![4.0, 1.0]);
    }

    /// An empty column still needs its start entry, or every column after it
    /// is read at the wrong offset.
    #[test]
    fn csc_keeps_empty_columns() {
        let triplets = vec![(0, 2, 7.0)];
        let csc = Csc::from_triplets(&triplets, 3, 1);
        assert_eq!(csc.starts, vec![0, 0, 0, 1]);
        assert_eq!(csc.indices, vec![0]);
    }

    #[test]
    fn the_linked_library_reports_a_plausible_version() {
        let (major, _, _) = HighsSolver::version();
        assert!(major >= 1, "expected HiGHS 1.x or newer, got major {major}");
    }
}
