//! A uniform batch of sparse linear systems: what cuDSS's uniform-batch API
//! (`CUDSS_CONFIG_UBATCH_SIZE`, `CUDSS_DATA_UBATCH_MASK`) offers, as a trait.
//!
//! B systems share one sparsity pattern, analysed once; each has its own
//! values, right-hand side and solution, in contiguous `B × nnz` and `B × n`
//! buffers, system-major. A factorization and a solve each take a mask that
//! selects the systems they process. The rules are the ones phase 0's probe
//! measured on an A100 (`plans/GPU_BATCH_PLAN.md`), and [`CpuUniformBatch`]
//! enforces them so the Newton loop is tested against the device's contract
//! on the CPU:
//!
//! - an unselected system's factors and solution are left exactly as they
//!   were (cuDSS 1.0.0, as NVIDIA documents; 0.8.0, which drops the factors
//!   and computes wrong solutions in a batch, is not supported);
//! - a solve may only select systems some factorization has covered since the
//!   analysis;
//! - a singular system is not an error of the call: its solution comes back
//!   non-finite, which is all a batched device solve can say per system.

use gridoxide_sparse::lu::{LuBackend, SparseLu};

/// The pattern every system of the batch shares, in compressed sparse row
/// form (cuDSS's input format), zero-based.
#[derive(Clone, Debug)]
pub struct CsrPattern {
    pub n: usize,
    /// `n + 1` offsets into `col_idx`.
    pub row_ptr: Vec<usize>,
    /// Column of each stored entry, ascending within a row.
    pub col_idx: Vec<usize>,
}

impl CsrPattern {
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// Values for the analysis, which is of the pattern alone but runs on
    /// numbers (KLU's also factors once; cuDSS's reordering may read them):
    /// a strictly diagonally dominant matrix, which cannot be singular for any
    /// pattern that stores its diagonal, as a power-flow Jacobian does. A real
    /// system that is singular then fails its own solve, not the batch's
    /// analysis.
    pub fn surrogate_values(&self) -> Vec<f64> {
        let mut values = Vec::with_capacity(self.nnz());
        for r in 0..self.n {
            let row = &self.col_idx[self.row_ptr[r]..self.row_ptr[r + 1]];
            values.extend(
                row.iter()
                    .map(|&c| if c == r { row.len() as f64 + 1.0 } else { 1.0 }),
            );
        }
        values
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LinearError {
    /// A buffer or mask whose length is not the batch's.
    Shape(String),
    /// A solve selected a system without factors: none since the analysis,
    /// or, on a backend that drops unselected systems' factors, none in the
    /// latest factorization.
    NotFactored { system: usize },
    /// Anything a backend reports that is not one of the above.
    Backend(String),
}

impl std::fmt::Display for LinearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearError::Shape(what) => write!(f, "{what}"),
            LinearError::NotFactored { system } => {
                write!(
                    f,
                    "system {system} was selected for a solve but has no factors"
                )
            }
            LinearError::Backend(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for LinearError {}

/// What a backend did, in systems processed: the quantity a masked device
/// call should cost in proportion to, and the one the Newton loop's policies
/// try to keep down.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearCounts {
    /// Calls to [`UniformBatch::factor`] and [`UniformBatch::solve`].
    pub factor_calls: usize,
    pub solve_calls: usize,
    /// Systems those calls selected, summed over calls.
    pub systems_factored: usize,
    pub systems_solved: usize,
}

/// A uniform batch of linear systems; see the module documentation.
pub trait UniformBatch: Sized {
    /// Analyses `pattern` once for `batch` systems.
    fn analyze(pattern: &CsrPattern, batch: usize) -> Result<Self, LinearError>;

    /// Factors the selected systems from their values: `values` is
    /// `batch × nnz`, system-major, each system's in `pattern`'s CSR order.
    fn factor(&mut self, values: &[f64], mask: &[bool]) -> Result<(), LinearError>;

    /// Solves the selected systems against their latest factors: `rhs` and
    /// `x` are `batch × n`, system-major. An unselected system's `x` is left
    /// untouched; a singular one's comes back non-finite.
    fn solve(&mut self, rhs: &[f64], x: &mut [f64], mask: &[bool]) -> Result<(), LinearError>;

    fn counts(&self) -> LinearCounts;
}

/// The uniform batch on the CPU, gridoxide's native KLU: one symbolic
/// analysis shared by every system, as on the device.
///
/// For the Newton loop's tests and for counting, not for speed. A
/// factorization records the selected systems' values; a solve refactors each
/// selected system from its recorded values and solves, so a system's solve
/// always uses the values of its latest factorization, whatever has been
/// written into the value buffer since — exactly the device's semantics, at
/// the CPU cost of a refactorization per solve.
pub struct CpuUniformBatch {
    n: usize,
    nnz: usize,
    batch: usize,
    lu: SparseLu,
    /// Each system's values at its latest factorization; empty if never.
    factored: Vec<Vec<f64>>,
    counts: LinearCounts,
}

impl CpuUniformBatch {
    fn check(&self, what: &str, len: usize, per_system: usize) -> Result<(), LinearError> {
        if len == self.batch * per_system {
            Ok(())
        } else {
            Err(LinearError::Shape(format!(
                "{what} has {len} entries, expected {} systems x {per_system}",
                self.batch
            )))
        }
    }

    fn check_mask(&self, mask: &[bool]) -> Result<(), LinearError> {
        self.check("mask", mask.len(), 1)
    }
}

impl UniformBatch for CpuUniformBatch {
    fn analyze(pattern: &CsrPattern, batch: usize) -> Result<Self, LinearError> {
        let rows = (0..pattern.n)
            .flat_map(|r| std::iter::repeat_n(r, pattern.row_ptr[r + 1] - pattern.row_ptr[r]));
        let surrogate: Vec<(usize, usize, f64)> = rows
            .zip(&pattern.col_idx)
            .zip(pattern.surrogate_values())
            .map(|((r, &c), v)| (r, c, v))
            .collect();
        let mut lu = SparseLu::new(LuBackend::KluNative);
        if !lu.analyze(pattern.n, &surrogate) {
            return Err(LinearError::Backend(
                "KLU's analysis failed: the pattern must store its diagonal".into(),
            ));
        }
        Ok(Self {
            n: pattern.n,
            nnz: pattern.nnz(),
            batch,
            lu,
            factored: vec![Vec::new(); batch],
            counts: LinearCounts::default(),
        })
    }

    fn factor(&mut self, values: &[f64], mask: &[bool]) -> Result<(), LinearError> {
        self.check("values", values.len(), self.nnz)?;
        self.check_mask(mask)?;
        self.counts.factor_calls += 1;
        for (s, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
            self.factored[s].clear();
            self.factored[s].extend_from_slice(&values[s * self.nnz..(s + 1) * self.nnz]);
            self.counts.systems_factored += 1;
        }
        Ok(())
    }

    fn solve(&mut self, rhs: &[f64], x: &mut [f64], mask: &[bool]) -> Result<(), LinearError> {
        self.check("rhs", rhs.len(), self.n)?;
        self.check("x", x.len(), self.n)?;
        self.check_mask(mask)?;
        if let Some(system) = (0..self.batch).find(|&s| mask[s] && self.factored[s].is_empty()) {
            return Err(LinearError::NotFactored { system });
        }
        self.counts.solve_calls += 1;
        for s in (0..self.batch).filter(|&s| mask[s]) {
            let values = &self.factored[s];
            let out = &mut x[s * self.n..(s + 1) * self.n];
            match self
                .lu
                .factor_and_solve_values(values, &rhs[s * self.n..(s + 1) * self.n])
            {
                Some(sol) => out.copy_from_slice(&sol),
                None => out.fill(f64::NAN),
            }
            self.counts.systems_solved += 1;
        }
        Ok(())
    }

    fn counts(&self) -> LinearCounts {
        self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[[a, 1], [1, b]]`: a 2×2 pattern with every entry stored.
    fn pattern() -> CsrPattern {
        CsrPattern {
            n: 2,
            row_ptr: vec![0, 2, 4],
            col_idx: vec![0, 1, 0, 1],
        }
    }

    fn values(diag: &[(f64, f64)]) -> Vec<f64> {
        diag.iter().flat_map(|&(a, b)| [a, 1.0, 1.0, b]).collect()
    }

    /// x of [[a, 1], [1, b]] x = [1, 0].
    fn expected(a: f64, b: f64) -> [f64; 2] {
        let det = a * b - 1.0;
        [b / det, -1.0 / det]
    }

    #[test]
    fn a_solve_selecting_an_unfactored_system_is_refused() {
        let mut lin = CpuUniformBatch::analyze(&pattern(), 3).unwrap();
        lin.factor(
            &values(&[(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)]),
            &[true, false, true],
        )
        .unwrap();
        let mut x = vec![0.0; 6];
        assert_eq!(
            lin.solve(&[1.0, 0.0].repeat(3), &mut x, &[false, true, false]),
            Err(LinearError::NotFactored { system: 1 })
        );
    }

    /// An unselected system's solution is left as it was, and a solve uses
    /// each system's values at its own latest factorization, not whatever the
    /// value buffer holds by then.
    #[test]
    fn masks_leave_unselected_systems_as_they_were() {
        let mut lin = CpuUniformBatch::analyze(&pattern(), 3).unwrap();
        lin.factor(
            &values(&[(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)]),
            &[true, true, true],
        )
        .unwrap();
        // System 0 refactors on new values; 1 and 2 keep theirs although the
        // buffer now holds garbage for them.
        lin.factor(
            &values(&[(8.0, 9.0), (1e300, 1e300), (1e300, 1e300)]),
            &[true, false, false],
        )
        .unwrap();
        let mut x = vec![12345.0; 6];
        lin.solve(&[1.0, 0.0].repeat(3), &mut x, &[true, false, true])
            .unwrap();

        let close =
            |got: &[f64], want: [f64; 2]| got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-14);
        assert!(close(&x[0..2], expected(8.0, 9.0)));
        assert_eq!(&x[2..4], &[12345.0, 12345.0]);
        assert!(close(&x[4..6], expected(6.0, 7.0)));
        assert_eq!(
            lin.counts(),
            LinearCounts {
                factor_calls: 2,
                solve_calls: 1,
                systems_factored: 4,
                systems_solved: 2,
            }
        );
    }

    #[test]
    fn a_singular_system_solves_to_non_finite_values_alone() {
        let mut lin = CpuUniformBatch::analyze(&pattern(), 2).unwrap();
        // [[1, 1], [1, 1]] is singular; the other system is not.
        lin.factor(&values(&[(1.0, 1.0), (2.0, 3.0)]), &[true, true])
            .unwrap();
        let mut x = vec![0.0; 4];
        lin.solve(&[1.0, 0.0].repeat(2), &mut x, &[true, true])
            .unwrap();
        assert!(x[0..2].iter().any(|v| !v.is_finite()));
        assert!(x[2..4].iter().all(|v| v.is_finite()));
    }
}
