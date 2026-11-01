//! An interior point's KKT solver on the GPU: [`CudssSolver`], one sparse
//! symmetric system on cuDSS behind gridoxide-sparse's
//! [`SymmetricSolver`], the trait `opf::nlp` factorizes its KKT matrix
//! through. [`cudss_ldlt_solver`] and [`cudss_llt_solver`] are the two
//! `NlpOptions::kkt_solver` hooks:
//!
//! - `LDLᵀ` (`CUDSS_MTYPE_SYMMETRIC`) for the indefinite augmented system,
//!   its inertia read from `CUDSS_DATA_INERTIA`;
//! - Cholesky (`CUDSS_MTYPE_SPD`) for HyKKT's condensed positive definite
//!   one, its success the inertia test (`CUDSS_DATA_INFO`).
//!
//! The assembly and equilibration are the CPU solvers' own
//! ([`LowerPattern`]), so the factorized matrix is the same `D K D` to the
//! last bit: the lower triangle in compressed columns, which cuDSS takes as
//! the upper triangle in compressed rows (`CUDSS_MVIEW_UPPER`). The analysis
//! runs once, at the first factorization, on its values; every later one is
//! numeric only.
//!
//! The matrix and the right-hand side cross the bus every factorization and
//! solve (the triplets arrive on the host); the factor never does.

use std::ptr;

use gridoxide_sparse::ldlt::{Inertia, LowerPattern, SymmetricSolver};

use crate::cuda::{DeviceBuffer, Stream, cuda, ffi};
use crate::cudss::cudss;
use crate::uniform::LinearError;

/// Which factorization a [`CudssSolver`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Factorization {
    /// `LDLᵀ` with cuDSS's default (diagonal) pivoting, for indefinite
    /// matrices; reports the inertia.
    Ldlt,
    /// Cholesky, for positive definite matrices; fails on any other.
    Llt,
}

/// One sparse symmetric system on cuDSS; see the module documentation.
pub struct CudssSolver {
    // First: the handles point into the buffers below and run on `stream`.
    sys: System,
    stream: Stream,
    pattern: LowerPattern,
    kind: Factorization,
    _row_ptr: DeviceBuffer<i32>,
    _col: DeviceBuffer<i32>,
    values: DeviceBuffer<f64>,
    rhs: DeviceBuffer<f64>,
    solution: DeviceBuffer<f64>,
    /// The equilibration of the last factorization: it factorized `D K D`.
    scale: Vec<f64>,
    analysed: bool,
    /// The inertia of the last factorization, `None` if it failed or there
    /// was none.
    inertia: Option<Inertia>,
}

/// cuDSS's handles for the system: matrix, solution and right-hand side
/// descriptors over the solver's device buffers.
struct System {
    stream: ffi::cudaStream_t,
    handle: ffi::cudssHandle_t,
    config: ffi::cudssConfig_t,
    data: ffi::cudssData_t,
    a: ffi::cudssMatrix_t,
    x: ffi::cudssMatrix_t,
    b: ffi::cudssMatrix_t,
}

impl System {
    fn execute(&self, phase: ffi::cudssPhase_t, what: &str) -> Result<(), LinearError> {
        // SAFETY: every handle was made in `CudssSolver::new` over buffers
        // the solver owns.
        cudss(
            unsafe {
                ffi::cudssExecute(
                    self.handle,
                    phase as i32,
                    self.config,
                    self.data,
                    self.a,
                    self.x,
                    self.b,
                )
            },
            what,
        )
    }

    /// A `cudssDataSet` of `T`s from `values`.
    fn set<T: Copy>(
        &self,
        param: ffi::cudssDataParam_t,
        values: &[T],
        what: &str,
    ) -> Result<(), LinearError> {
        // SAFETY: `values` holds `size_of_val(values)` bytes, which cuDSS
        // copies before returning.
        cudss(
            unsafe {
                ffi::cudssDataSet(
                    self.handle,
                    self.data,
                    param,
                    values.as_ptr() as *mut std::ffi::c_void,
                    size_of_val(values),
                )
            },
            what,
        )
    }

    /// A `cudssDataGet` of `T`s into `out`.
    fn get<T: Copy>(
        &self,
        param: ffi::cudssDataParam_t,
        out: &mut [T],
        what: &str,
    ) -> Result<(), LinearError> {
        let mut written = 0usize;
        // SAFETY: `out` holds `size_of_val(out)` bytes.
        cudss(
            unsafe {
                ffi::cudssDataGet(
                    self.handle,
                    self.data,
                    param,
                    out.as_mut_ptr().cast(),
                    size_of_val(out),
                    &mut written,
                )
            },
            what,
        )
    }
}

impl Drop for System {
    fn drop(&mut self) {
        // SAFETY: each handle is destroyed once, if made, after the stream's
        // work that uses it.
        unsafe {
            ffi::cudaStreamSynchronize(self.stream);
            for m in [self.a, self.x, self.b] {
                if !m.is_null() {
                    ffi::cudssMatrixDestroy(m);
                }
            }
            if !self.data.is_null() {
                ffi::cudssDataDestroy(self.handle, self.data);
            }
            if !self.config.is_null() {
                ffi::cudssConfigDestroy(self.config);
            }
            if !self.handle.is_null() {
                ffi::cudssDestroy(self.handle);
            }
        }
    }
}

impl CudssSolver {
    /// A solver for the `n × n` pattern of `keys` (row, column; entries above
    /// the diagonal ignored, duplicates summed), as
    /// [`LowerPattern::new`] reads it.
    ///
    /// `deterministic` sets `CUDSS_CONFIG_DETERMINISTIC_MODE`: the same
    /// matrix factorizes to the same bits every time. Without it, four runs
    /// of case2869_pegase took 60 to 107 interior-point iterations, as
    /// rounding differences steered the regularization; with it, the path is
    /// reproducible, at about 30% more time a factorization there.
    pub fn new(
        n: usize,
        keys: &[(usize, usize)],
        kind: Factorization,
        deterministic: bool,
    ) -> Result<Self, LinearError> {
        let pattern = LowerPattern::new(n, keys);
        let nnz = pattern.row_idx().len();
        let to_i32 = |v: &[usize]| -> Result<Vec<i32>, LinearError> {
            v.iter()
                .map(|&i| {
                    i32::try_from(i).map_err(|_| {
                        LinearError::Shape(format!("index {i} beyond cuDSS's 32-bit indices"))
                    })
                })
                .collect()
        };
        let stream = Stream::new()?;
        let mut row_ptr = DeviceBuffer::new(n + 1)?;
        row_ptr.upload(&to_i32(pattern.col_ptr())?, stream.0)?;
        let mut col = DeviceBuffer::new(nnz.max(1))?;
        if nnz > 0 {
            col.upload(&to_i32(pattern.row_idx())?, stream.0)?;
        }
        let values = DeviceBuffer::<f64>::new(nnz.max(1))?;
        let rhs = DeviceBuffer::<f64>::new(n.max(1))?;
        let solution = DeviceBuffer::<f64>::new(n.max(1))?;
        let mut sys = System {
            stream: stream.0,
            handle: ptr::null_mut(),
            config: ptr::null_mut(),
            data: ptr::null_mut(),
            a: ptr::null_mut(),
            x: ptr::null_mut(),
            b: ptr::null_mut(),
        };
        let mtype = match kind {
            Factorization::Ldlt => ffi::CUDSS_MTYPE_SYMMETRIC,
            Factorization::Llt => ffi::CUDSS_MTYPE_SPD,
        };
        // SAFETY: each call gets valid out-pointers; the descriptors point at
        // the buffers above, which the solver keeps.
        unsafe {
            cudss(ffi::cudssCreate(&mut sys.handle), "cudssCreate")?;
            cudss(ffi::cudssSetStream(sys.handle, stream.0), "cudssSetStream")?;
            cudss(ffi::cudssConfigCreate(&mut sys.config), "cudssConfigCreate")?;
            if deterministic {
                let on: i32 = 1;
                cudss(
                    ffi::cudssConfigSet(
                        sys.config,
                        ffi::CUDSS_CONFIG_DETERMINISTIC_MODE,
                        (&on as *const i32) as *mut std::ffi::c_void,
                        size_of::<i32>(),
                    ),
                    "cudssConfigSet(DETERMINISTIC_MODE)",
                )?;
            }
            cudss(
                ffi::cudssDataCreate(sys.handle, &mut sys.data),
                "cudssDataCreate",
            )?;
            cudss(
                ffi::cudssMatrixCreateCsr(
                    &mut sys.a,
                    n as i64,
                    n as i64,
                    nnz as i64,
                    row_ptr.ptr.cast(),
                    ptr::null_mut(),
                    col.ptr.cast(),
                    values.ptr.cast(),
                    ffi::CUDA_R_32I,
                    ffi::CUDA_R_32I,
                    ffi::CUDA_R_64F,
                    mtype,
                    ffi::CUDSS_MVIEW_UPPER,
                    ffi::CUDSS_BASE_ZERO,
                ),
                "cudssMatrixCreateCsr",
            )?;
            for (m, buf) in [(&mut sys.x, solution.ptr), (&mut sys.b, rhs.ptr)] {
                cudss(
                    ffi::cudssMatrixCreateDn(
                        m,
                        n as i64,
                        1,
                        n as i64,
                        buf.cast(),
                        ffi::CUDA_R_64F,
                        ffi::CUDSS_LAYOUT_COL_MAJOR,
                    ),
                    "cudssMatrixCreateDn",
                )?;
            }
        }
        Ok(Self {
            sys,
            stream,
            pattern,
            kind,
            _row_ptr: row_ptr,
            _col: col,
            values,
            rhs,
            solution,
            scale: Vec::new(),
            analysed: false,
            inertia: None,
        })
    }

    pub fn kind(&self) -> Factorization {
        self.kind
    }

    /// Factorizes; `Ok(None)` when the matrix has no factorization of this
    /// kind (a zero pivot, or not positive definite for Cholesky).
    fn try_factorize(
        &mut self,
        entries: &[(usize, usize, f64)],
    ) -> Result<Option<Inertia>, LinearError> {
        let n = self.pattern.n();
        let (values, scale) = self.pattern.assemble(entries);
        if values.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        self.scale = scale;
        if !values.is_empty() {
            self.values.upload(&values, self.stream.0)?;
        }
        if !self.analysed {
            self.sys.execute(ffi::CUDSS_PHASE_ANALYSIS, "ANALYSIS")?;
            self.analysed = true;
        }
        // `CUDSS_DATA_INFO` is sticky: a failed factorization leaves it set
        // through every later one until it is reset. Unreset, the first
        // non-positive pivot of a Cholesky (case9241_pegase, HyKKT, at
        // iteration 14) failed every retry the interior point's growing δw
        // made, and the run with it.
        self.sys
            .set(ffi::CUDSS_DATA_INFO, &[0i32], "INFO (reset)")?;
        self.sys
            .execute(ffi::CUDSS_PHASE_FACTORIZATION, "FACTORIZATION")?;
        cuda(
            // SAFETY: the stream is the solver's.
            unsafe { ffi::cudaStreamSynchronize(self.stream.0) },
            "cudaStreamSynchronize",
        )?;
        let mut info = [0i32];
        self.sys.get(ffi::CUDSS_DATA_INFO, &mut info, "INFO")?;
        if info[0] != 0 {
            return Ok(None);
        }
        match self.kind {
            Factorization::Llt => Ok(Some(Inertia {
                positive: n,
                negative: 0,
                zero: 0,
            })),
            Factorization::Ldlt => {
                let mut inertia = [0i32; 2];
                self.sys
                    .get(ffi::CUDSS_DATA_INERTIA, &mut inertia, "INERTIA")?;
                let positive = inertia[0].max(0) as usize;
                let negative = inertia[1].max(0) as usize;
                Ok(Some(Inertia {
                    positive,
                    negative,
                    zero: n.saturating_sub(positive + negative),
                }))
            }
        }
    }

    fn try_solve(&self, rhs: &[f64]) -> Result<Option<Vec<f64>>, LinearError> {
        let n = self.pattern.n();
        if self.inertia.is_none() || rhs.len() != n {
            return Ok(None);
        }
        let scaled: Vec<f64> = rhs.iter().zip(&self.scale).map(|(b, d)| b * d).collect();
        // SAFETY: `rhs` is the solver's own buffer of n values; the upload
        // goes through a raw pointer because `solve` takes `&self`.
        unsafe { crate::cuda::copy_to_device(self.rhs.ptr, &scaled, self.stream.0)? };
        self.sys.execute(ffi::CUDSS_PHASE_SOLVE, "SOLVE")?;
        let mut x = vec![0.0; n];
        self.solution.download(&mut x, self.stream.0)?;
        for (xi, d) in x.iter_mut().zip(&self.scale) {
            *xi *= d;
        }
        Ok(x.iter().all(|v| v.is_finite()).then_some(x))
    }
}

impl SymmetricSolver for CudssSolver {
    fn matches(&self, n: usize, entries: &[(usize, usize, f64)]) -> bool {
        self.pattern.matches(n, entries)
    }

    /// A device error counts as a failed factorization: the interior point
    /// then regularizes, and gives up once that stops helping.
    fn factorize(&mut self, entries: &[(usize, usize, f64)]) -> bool {
        self.inertia = self.try_factorize(entries).ok().flatten();
        self.inertia.is_some()
    }

    fn inertia(&self) -> Inertia {
        self.inertia.unwrap_or_default()
    }

    fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        self.try_solve(rhs).ok().flatten()
    }
}

/// `NlpOptions::kkt_solver` for the augmented system: cuDSS's `LDLᵀ`,
/// deterministic.
/// `None` if the device cannot be set up.
pub fn cudss_ldlt_solver(n: usize, keys: &[(usize, usize)]) -> Option<Box<dyn SymmetricSolver>> {
    let solver = CudssSolver::new(n, keys, Factorization::Ldlt, true).ok()?;
    Some(Box::new(solver))
}

/// `NlpOptions::kkt_solver` for `KktForm::Hybrid`: cuDSS's Cholesky,
/// deterministic.
/// `None` if the device cannot be set up.
pub fn cudss_llt_solver(n: usize, keys: &[(usize, usize)]) -> Option<Box<dyn SymmetricSolver>> {
    let solver = CudssSolver::new(n, keys, Factorization::Llt, true).ok()?;
    Some(Box::new(solver))
}
