//! The uniform batch on an NVIDIA GPU, through cuDSS: [`CudssUniformBatch`].
//!
//! gpusim2grid's shape, the one that works (`plans/GPU_BATCH_PLAN.md`): one
//! plain `cudssMatrixCreateCsr` whose value buffer holds all B systems
//! (`B × nnz`), the right-hand side and solution as `n × 1` column-major dense
//! descriptors over `B × n` buffers, `CUDSS_CONFIG_UBATCH_SIZE = B` before the
//! one analysis. Never `cudssMatrixCreateBatchCsr` with it, never
//! `CUDSS_CONFIG_UBATCH_INDEX`: both are dead ends.
//!
//! Masks go to `CUDSS_DATA_UBATCH_MASK`, a host array of one `int` per
//! system, set to all ones before the analysis as the documentation requires
//! and re-set only when it changes.
//!
//! cuDSS 1.0 or later only (build.rs refuses older headers). Phase 0's probe
//! found 0.8.0 returning NaN solutions for some systems of a uniform batch
//! (from index 216 on, case1354pegase, an A100) and when other GPU work runs
//! alongside it, besides dropping the factors of systems a masked
//! factorization leaves out; see `plans/GPU_BATCH_PLAN.md`. In 1.0.0 the
//! mask works as documented: an unselected system keeps its factors and its
//! solution, and `REFACTORIZATION` honours the mask, so it is used whenever
//! every selected system has been factored since the analysis
//! ([`CudssUniformBatch::use_refactorization`] turns it off).
//!
//! Still open: what a singular system's solution looks like. cuDSS perturbs
//! tiny pivots by default, so it may come back finite but wrong rather than
//! non-finite; the Newton loop's divergence test catches either, a step
//! later.
//!
//! Two layers: `CudssBatch`, crate-private, works on device buffers and a
//! stream it does not own, which is how the device loop
//! ([`GpuBatchSolver`](crate::gpu::GpuBatchSolver)) factors and solves the
//! Jacobians its kernels fill in place; [`CudssUniformBatch`] owns its
//! buffers and stream and copies host data in and out, to implement
//! [`UniformBatch`] for the CPU loop and its tests.

use std::ffi::CString;
use std::ptr;

use crate::cuda::{DeviceBuffer, Stream, copy_to_device, ffi};
use crate::uniform::{CsrPattern, LinearCounts, LinearError, UniformBatch};

fn cudss(status: ffi::cudssStatus_t, what: &str) -> Result<(), LinearError> {
    let name = match status {
        ffi::CUDSS_STATUS_SUCCESS => return Ok(()),
        ffi::CUDSS_STATUS_NOT_INITIALIZED => "NOT_INITIALIZED",
        ffi::CUDSS_STATUS_ALLOC_FAILED => "ALLOC_FAILED",
        ffi::CUDSS_STATUS_INVALID_VALUE => "INVALID_VALUE",
        ffi::CUDSS_STATUS_NOT_SUPPORTED => "NOT_SUPPORTED",
        ffi::CUDSS_STATUS_EXECUTION_FAILED => "EXECUTION_FAILED",
        ffi::CUDSS_STATUS_INTERNAL_ERROR => "INTERNAL_ERROR",
        ffi::CUDSS_STATUS_IR_FAILED => "IR_FAILED",
        _ => "UNKNOWN",
    };
    Err(LinearError::Backend(format!("{what}: CUDSS_STATUS_{name}")))
}

/// The cuDSS version the bindings were generated from, as
/// `(major, minor, patch)`.
pub fn version() -> (u32, u32, u32) {
    (
        ffi::CUDSS_VERSION_MAJOR,
        ffi::CUDSS_VERSION_MINOR,
        ffi::CUDSS_VERSION_PATCH,
    )
}

/// cuDSS's uniform batch over a values, a right-hand-side and a solution
/// buffer it does not own, on a stream it does not own: B systems analysed
/// once, then masked (re)factorizations and solves of whatever the buffers
/// hold when they run.
pub(crate) struct CudssBatch {
    batch: usize,
    stream: ffi::cudaStream_t,
    values: *mut f64,
    // Every handle starts null and is destroyed in `drop` if set, so a
    // failure halfway through `analyze` releases what was made.
    handle: ffi::cudssHandle_t,
    config: ffi::cudssConfig_t,
    data: ffi::cudssData_t,
    a: ffi::cudssMatrix_t,
    b: ffi::cudssMatrix_t,
    x: ffi::cudssMatrix_t,
    // Freed after `drop` has destroyed the descriptor that points into them.
    _row_ptr: DeviceBuffer<i32>,
    _col_idx: DeviceBuffer<i32>,
    /// The mask cuDSS holds now, as it was last set.
    mask: Vec<i32>,
    /// Which systems have been factored since the analysis: they may be
    /// solved, and refactored.
    factored: Vec<bool>,
    refactorization: bool,
    counts: LinearCounts,
}

impl CudssBatch {
    /// Analyses `pattern` for `batch` systems over `values` (`batch × nnz`),
    /// `rhs` and `sol` (`batch × n` each), all on `stream`. The analysis
    /// overwrites `values` with surrogate values (see
    /// [`CsrPattern::surrogate_values`]).
    ///
    /// # Safety
    ///
    /// The three pointers must be device allocations of those sizes, and
    /// they and `stream` must outlive the returned batch.
    pub(crate) unsafe fn analyze(
        pattern: &CsrPattern,
        batch: usize,
        stream: ffi::cudaStream_t,
        values: *mut f64,
        rhs: *mut f64,
        sol: *mut f64,
    ) -> Result<Self, LinearError> {
        let (n, nnz) = (pattern.n, pattern.nnz());
        let too_big = |what: &str| LinearError::Shape(format!("{what} does not fit cuDSS's int32"));
        let row_ptr: Vec<i32> = pattern
            .row_ptr
            .iter()
            .map(|&k| i32::try_from(k).map_err(|_| too_big("the pattern")))
            .collect::<Result<_, _>>()?;
        // Columns are below n, checked next.
        let col_idx: Vec<i32> = pattern.col_idx.iter().map(|&c| c as i32).collect();
        i32::try_from(n).map_err(|_| too_big("the pattern"))?;
        let ubatch = i32::try_from(batch).map_err(|_| too_big("the batch size"))?;

        let mut lin = Self {
            batch,
            stream,
            values,
            handle: ptr::null_mut(),
            config: ptr::null_mut(),
            data: ptr::null_mut(),
            a: ptr::null_mut(),
            b: ptr::null_mut(),
            x: ptr::null_mut(),
            _row_ptr: DeviceBuffer::new(n + 1)?,
            _col_idx: DeviceBuffer::new(nnz)?,
            mask: vec![1; batch],
            factored: vec![false; batch],
            refactorization: true,
            counts: LinearCounts::default(),
        };
        let threading_lib = std::env::var("CUDSS_THREADING_LIB")
            .ok()
            .map(|lib| CString::new(lib).expect("CUDSS_THREADING_LIB holds a NUL"));

        // SAFETY: each call gets valid out-pointers.
        unsafe {
            cudss(ffi::cudssCreate(&mut lin.handle), "cudssCreate")?;
            cudss(ffi::cudssSetStream(lin.handle, stream), "cudssSetStream")?;
            // cuDSS's OpenMP host threading, about 3.2x on analysis on the old
            // branch: on when CUDSS_THREADING_LIB names its layer library.
            if let Some(lib) = &threading_lib {
                cudss(
                    ffi::cudssSetThreadingLayer(lin.handle, lib.as_ptr()),
                    "cudssSetThreadingLayer",
                )?;
            }
            cudss(ffi::cudssConfigCreate(&mut lin.config), "cudssConfigCreate")?;
            cudss(
                ffi::cudssDataCreate(lin.handle, &mut lin.data),
                "cudssDataCreate",
            )?;
        }

        lin._row_ptr.upload(&row_ptr, stream)?;
        lin._col_idx.upload(&col_idx, stream)?;
        // Analysis runs on numbers; every system gets the surrogate values,
        // so none is singular at the analysis.
        // SAFETY: `values` holds `batch × nnz` values (the caller's).
        unsafe { copy_to_device(values, &pattern.surrogate_values().repeat(batch), stream)? };

        // SAFETY: the descriptors point at device buffers that outlive `lin`
        // (the caller's, and `lin`'s own pattern).
        unsafe {
            cudss(
                ffi::cudssMatrixCreateCsr(
                    &mut lin.a,
                    n as i64,
                    n as i64,
                    nnz as i64,
                    lin._row_ptr.ptr.cast(),
                    ptr::null_mut(),
                    lin._col_idx.ptr.cast(),
                    values.cast(),
                    ffi::CUDA_R_32I,
                    ffi::CUDA_R_32I,
                    ffi::CUDA_R_64F,
                    ffi::CUDSS_MTYPE_GENERAL,
                    ffi::CUDSS_MVIEW_FULL,
                    ffi::CUDSS_BASE_ZERO,
                ),
                "cudssMatrixCreateCsr",
            )?;
            cudss(
                ffi::cudssMatrixCreateDn(
                    &mut lin.b,
                    n as i64,
                    1,
                    n as i64,
                    rhs.cast(),
                    ffi::CUDA_R_64F,
                    ffi::CUDSS_LAYOUT_COL_MAJOR,
                ),
                "cudssMatrixCreateDn(b)",
            )?;
            cudss(
                ffi::cudssMatrixCreateDn(
                    &mut lin.x,
                    n as i64,
                    1,
                    n as i64,
                    sol.cast(),
                    ffi::CUDA_R_64F,
                    ffi::CUDSS_LAYOUT_COL_MAJOR,
                ),
                "cudssMatrixCreateDn(x)",
            )?;
            cudss(
                ffi::cudssConfigSet(
                    lin.config,
                    ffi::CUDSS_CONFIG_UBATCH_SIZE,
                    (&ubatch as *const i32).cast(),
                    size_of::<i32>(),
                ),
                "cudssConfigSet(UBATCH_SIZE)",
            )?;
        }
        // All ones before the analysis: the mask's device storage is
        // allocated there, and later masks only change its contents.
        lin.push_mask()?;
        lin.execute(ffi::CUDSS_PHASE_ANALYSIS, "ANALYSIS")?;
        Ok(lin)
    }

    pub(crate) fn use_refactorization(&mut self, on: bool) {
        self.refactorization = on;
    }

    /// Factors the selected systems from what the values buffer holds now.
    pub(crate) fn factor(&mut self, mask: &[bool]) -> Result<(), LinearError> {
        self.check_mask(mask)?;
        self.counts.factor_calls += 1;
        let selected = mask.iter().filter(|&&m| m).count();
        if selected == 0 {
            return Ok(());
        }
        // SAFETY: the same device buffer `a` was created over; this tells
        // cuDSS its contents changed, as gpusim2grid and the probe do.
        cudss(
            unsafe { ffi::cudssMatrixSetValues(self.a, self.values.cast()) },
            "cudssMatrixSetValues",
        )?;
        self.set_mask(mask)?;
        let again = mask.iter().zip(&self.factored).all(|(&m, &f)| !m || f);
        if self.refactorization && again {
            self.execute(ffi::CUDSS_PHASE_REFACTORIZATION, "REFACTORIZATION")?;
        } else {
            self.execute(ffi::CUDSS_PHASE_FACTORIZATION, "FACTORIZATION")?;
        }
        for (f, &m) in self.factored.iter_mut().zip(mask) {
            *f |= m;
        }
        self.counts.systems_factored += selected;
        Ok(())
    }

    /// Solves the selected systems from what the right-hand side holds now
    /// into the solution buffer; the others' solutions are left as they were.
    pub(crate) fn solve(&mut self, mask: &[bool]) -> Result<(), LinearError> {
        self.check_mask(mask)?;
        if let Some(system) = (0..self.batch).find(|&s| mask[s] && !self.factored[s]) {
            return Err(LinearError::NotFactored { system });
        }
        self.counts.solve_calls += 1;
        let selected = mask.iter().filter(|&&m| m).count();
        if selected == 0 {
            return Ok(());
        }
        self.set_mask(mask)?;
        self.execute(ffi::CUDSS_PHASE_SOLVE, "SOLVE")?;
        self.counts.systems_solved += selected;
        Ok(())
    }

    pub(crate) fn counts(&self) -> LinearCounts {
        self.counts
    }

    fn check_mask(&self, mask: &[bool]) -> Result<(), LinearError> {
        if mask.len() == self.batch {
            Ok(())
        } else {
            Err(LinearError::Shape(format!(
                "mask has {} entries, expected {} systems",
                mask.len(),
                self.batch
            )))
        }
    }

    fn execute(&mut self, phase: ffi::cudssPhase_t, what: &str) -> Result<(), LinearError> {
        // SAFETY: every handle and descriptor was created in `analyze` and
        // points at buffers that outlive this batch.
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

    fn set_mask(&mut self, mask: &[bool]) -> Result<(), LinearError> {
        if self.mask.iter().zip(mask).all(|(&m, &s)| (m != 0) == s) {
            return Ok(());
        }
        for (m, &s) in self.mask.iter_mut().zip(mask) {
            *m = i32::from(s);
        }
        self.push_mask()
    }

    fn push_mask(&mut self) -> Result<(), LinearError> {
        // SAFETY: cuDSS copies the `batch` ints out of the host array.
        cudss(
            unsafe {
                ffi::cudssDataSet(
                    self.handle,
                    self.data,
                    ffi::CUDSS_DATA_UBATCH_MASK,
                    self.mask.as_ptr().cast(),
                    self.mask.len() * size_of::<i32>(),
                )
            },
            "cudssDataSet(UBATCH_MASK)",
        )
    }
}

impl Drop for CudssBatch {
    fn drop(&mut self) {
        // SAFETY: each handle is destroyed once, if it was made, descriptors
        // before the data and handle they belong to, after the stream has
        // finished the work that uses them.
        unsafe {
            ffi::cudaStreamSynchronize(self.stream);
            for m in [self.a, self.b, self.x] {
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

/// cuDSS's uniform batch with its own buffers, fed from the host; see the
/// module documentation.
pub struct CudssUniformBatch {
    // First, so its descriptors are destroyed before the stream and the
    // buffers they point at.
    lin: CudssBatch,
    stream: Stream,
    n: usize,
    nnz: usize,
    batch: usize,
    values: DeviceBuffer<f64>,
    rhs: DeviceBuffer<f64>,
    sol: DeviceBuffer<f64>,
    /// The device solution, every system, before the selected ones are
    /// copied out: unselected systems' host `x` is left untouched.
    staging: Vec<f64>,
}

impl CudssUniformBatch {
    /// Whether a factorization of systems that have all been factored before
    /// runs as cuDSS's `REFACTORIZATION` (same pivots, cheaper) instead of a
    /// full `FACTORIZATION`. On by default: phase 0's probe showed it honours
    /// the mask.
    pub fn use_refactorization(&mut self, on: bool) {
        self.lin.use_refactorization(on);
    }

    /// The cuDSS version the bindings were generated from, as
    /// `(major, minor, patch)`.
    pub fn version() -> (u32, u32, u32) {
        version()
    }

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
}

impl UniformBatch for CudssUniformBatch {
    fn analyze(pattern: &CsrPattern, batch: usize) -> Result<Self, LinearError> {
        let (n, nnz) = (pattern.n, pattern.nnz());
        let stream = Stream::new()?;
        let values = DeviceBuffer::new(batch * nnz)?;
        let rhs = DeviceBuffer::new(batch * n)?;
        let sol = DeviceBuffer::new(batch * n)?;
        // SAFETY: the buffers are of the sizes `analyze` needs, and the
        // struct below keeps them and the stream alive until `lin`, dropped
        // first, is gone.
        let lin =
            unsafe { CudssBatch::analyze(pattern, batch, stream.0, values.ptr, rhs.ptr, sol.ptr)? };
        Ok(Self {
            lin,
            stream,
            n,
            nnz,
            batch,
            values,
            rhs,
            sol,
            staging: vec![0.0; batch * n],
        })
    }

    fn factor(&mut self, values: &[f64], mask: &[bool]) -> Result<(), LinearError> {
        self.check("values", values.len(), self.nnz)?;
        if mask.iter().any(|&m| m) {
            self.values.upload(values, self.stream.0)?;
        }
        self.lin.factor(mask)
    }

    fn solve(&mut self, rhs: &[f64], x: &mut [f64], mask: &[bool]) -> Result<(), LinearError> {
        self.check("rhs", rhs.len(), self.n)?;
        self.check("x", x.len(), self.n)?;
        let selected = mask.iter().any(|&m| m);
        if selected {
            self.rhs.upload(rhs, self.stream.0)?;
        }
        self.lin.solve(mask)?;
        if selected {
            self.sol.download(&mut self.staging, self.stream.0)?;
            let n = self.n;
            for s in (0..self.batch).filter(|&s| mask[s]) {
                x[s * n..(s + 1) * n].copy_from_slice(&self.staging[s * n..(s + 1) * n]);
            }
        }
        Ok(())
    }

    fn counts(&self) -> LinearCounts {
        self.lin.counts()
    }
}
