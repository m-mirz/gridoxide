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
//! What phase 0's probe measured on an A100, with cuDSS 0.8.0 and 1.0.0:
//!
//! - `REFACTORIZATION` honours the mask in both, so it is used whenever every
//!   selected system has been factored since the analysis
//!   ([`CudssUniformBatch::use_refactorization`] turns it off);
//! - 0.8.0's masked solve writes every system's solution buffer, the
//!   unselected ones from cuDSS's own copy, so only the selected systems are
//!   copied out (1.0.0 leaves the others untouched);
//! - 0.8.0 drops the factors of the systems a factorization does not select
//!   (they solve to zeros), 1.0.0 keeps them, as documented; this follows the
//!   version of the header the bindings were generated from;
//! - 0.8.0 is not safe with two handles at work at once in one process, each
//!   on its own stream: some solutions come back NaN. On 0.8.0 every
//!   `cudssExecute` therefore holds a process-wide lock until its stream has
//!   finished; 1.0.0 runs them concurrently.
//!
//! Still open: what a singular system's solution looks like. cuDSS perturbs
//! tiny pivots by default, so it may come back finite but wrong rather than
//! non-finite; the Newton loop's divergence test catches either, a step
//! later.
//!
//! Host data crosses as plain pageable copies on the solver's stream; the
//! device-resident loop of phase 2 replaces them with kernels.

use std::ffi::{CStr, CString, c_void};
use std::ptr;
use std::sync::Mutex;

use crate::uniform::{CsrPattern, LinearCounts, LinearError, UniformBatch};

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unused_imports,
    unnecessary_transmutes,
    clippy::all
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/cudss_bindings.rs"));
}

fn cuda(err: ffi::cudaError_t, what: &str) -> Result<(), LinearError> {
    if err == ffi::cudaSuccess {
        return Ok(());
    }
    // SAFETY: cudaGetErrorString returns a static, NUL-terminated string for
    // any error code, known or not.
    let msg = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(err)) };
    Err(LinearError::Backend(format!(
        "{what}: {}",
        msg.to_string_lossy()
    )))
}

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

/// A device allocation of `len` values of `T`, freed on drop.
struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> DeviceBuffer<T> {
    fn new(len: usize) -> Result<Self, LinearError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: cudaMalloc writes a device pointer to `ptr` or fails.
        cuda(
            unsafe { ffi::cudaMalloc(&mut ptr, len * size_of::<T>()) },
            "cudaMalloc",
        )?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }

    /// Copies `src` to the device, ordered on `stream`. From pageable memory
    /// the call returns once `src` has been staged, so `src` may be reused.
    fn upload(&mut self, src: &[T], stream: ffi::cudaStream_t) -> Result<(), LinearError> {
        assert_eq!(src.len(), self.len);
        // SAFETY: both ranges are `len` values long.
        cuda(
            unsafe {
                ffi::cudaMemcpyAsync(
                    self.ptr.cast(),
                    src.as_ptr().cast(),
                    self.len * size_of::<T>(),
                    ffi::cudaMemcpyHostToDevice,
                    stream,
                )
            },
            "cudaMemcpyAsync (to device)",
        )
    }

    /// Copies the buffer into `dst` once everything queued on `stream`
    /// before it has run.
    fn download(&self, dst: &mut [T], stream: ffi::cudaStream_t) -> Result<(), LinearError> {
        assert_eq!(dst.len(), self.len);
        // SAFETY: both ranges are `len` values long; the synchronize makes
        // the copy complete before `dst` is read.
        cuda(
            unsafe {
                ffi::cudaMemcpyAsync(
                    dst.as_mut_ptr().cast(),
                    self.ptr.cast(),
                    self.len * size_of::<T>(),
                    ffi::cudaMemcpyDeviceToHost,
                    stream,
                )
            },
            "cudaMemcpyAsync (to host)",
        )?;
        cuda(
            unsafe { ffi::cudaStreamSynchronize(stream) },
            "cudaStreamSynchronize",
        )
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` came from cudaMalloc and is freed once.
            unsafe { ffi::cudaFree(self.ptr.cast()) };
        }
    }
}

/// cuDSS's uniform batch; see the module documentation.
pub struct CudssUniformBatch {
    n: usize,
    nnz: usize,
    batch: usize,
    // Every handle starts null and is destroyed in `drop` if set, so a
    // failure halfway through `analyze` releases what was made.
    stream: ffi::cudaStream_t,
    handle: ffi::cudssHandle_t,
    config: ffi::cudssConfig_t,
    data: ffi::cudssData_t,
    a: ffi::cudssMatrix_t,
    b: ffi::cudssMatrix_t,
    x: ffi::cudssMatrix_t,
    // Dropped after `drop` has destroyed the descriptors that point into
    // them.
    _row_ptr: DeviceBuffer<i32>,
    _col_idx: DeviceBuffer<i32>,
    values: DeviceBuffer<f64>,
    rhs: DeviceBuffer<f64>,
    sol: DeviceBuffer<f64>,
    /// The mask cuDSS holds now, as it was last set.
    mask: Vec<i32>,
    /// The device solution, every system, before the selected ones are
    /// copied out: unselected systems' host `x` is left untouched.
    staging: Vec<f64>,
    /// Which systems have factors now, and which have had any since the
    /// analysis (and so may be refactored).
    factored: Vec<bool>,
    ever_factored: Vec<bool>,
    refactorization: bool,
    counts: LinearCounts,
}

/// Whether the cuDSS these bindings were generated for keeps an unselected
/// system's factors through a masked factorization: 1.0.0 does, 0.8.0 does
/// not.
const KEEPS_UNSELECTED_FACTORS: bool = ffi::CUDSS_VERSION_MAJOR >= 1;

/// Whether concurrent handles must be serialized: 0.8.0 computes NaNs when
/// two are at work at once (see the module documentation).
const SERIALIZE_HANDLES: bool = ffi::CUDSS_VERSION_MAJOR < 1;

static EXECUTE: Mutex<()> = Mutex::new(());

impl CudssUniformBatch {
    /// Whether a factorization of systems that have all been factored before
    /// runs as cuDSS's `REFACTORIZATION` (same pivots, cheaper) instead of a
    /// full `FACTORIZATION`. On by default: phase 0's probe showed it honours
    /// the mask in 0.8.0 and 1.0.0.
    pub fn use_refactorization(&mut self, on: bool) {
        self.refactorization = on;
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

    fn execute(&mut self, phase: ffi::cudssPhase_t, what: &str) -> Result<(), LinearError> {
        // A poisoned lock guards nothing that a panic could have left half
        // written: it only orders device work.
        let _serial = SERIALIZE_HANDLES.then(|| EXECUTE.lock().unwrap_or_else(|e| e.into_inner()));
        // SAFETY: every handle and descriptor was created in `analyze` and
        // points at buffers this struct owns.
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
        )?;
        if SERIALIZE_HANDLES {
            // SAFETY: the stream was created in `analyze`.
            cuda(
                unsafe { ffi::cudaStreamSynchronize(self.stream) },
                "cudaStreamSynchronize",
            )?;
        }
        Ok(())
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

impl UniformBatch for CudssUniformBatch {
    fn analyze(pattern: &CsrPattern, batch: usize) -> Result<Self, LinearError> {
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
        i32::try_from(batch).map_err(|_| too_big("the batch size"))?;

        let mut lin = Self {
            n,
            nnz,
            batch,
            stream: ptr::null_mut(),
            handle: ptr::null_mut(),
            config: ptr::null_mut(),
            data: ptr::null_mut(),
            a: ptr::null_mut(),
            b: ptr::null_mut(),
            x: ptr::null_mut(),
            _row_ptr: DeviceBuffer::new(n + 1)?,
            _col_idx: DeviceBuffer::new(nnz)?,
            values: DeviceBuffer::new(batch * nnz)?,
            rhs: DeviceBuffer::new(batch * n)?,
            sol: DeviceBuffer::new(batch * n)?,
            mask: vec![1; batch],
            staging: vec![0.0; batch * n],
            factored: vec![false; batch],
            ever_factored: vec![false; batch],
            refactorization: true,
            counts: LinearCounts::default(),
        };
        let threading_lib = std::env::var("CUDSS_THREADING_LIB")
            .ok()
            .map(|lib| CString::new(lib).expect("CUDSS_THREADING_LIB holds a NUL"));

        // SAFETY: each call gets valid out-pointers; the descriptors point at
        // device buffers that live as long as `lin`.
        unsafe {
            cuda(ffi::cudaStreamCreate(&mut lin.stream), "cudaStreamCreate")?;
            cudss(ffi::cudssCreate(&mut lin.handle), "cudssCreate")?;
            cudss(
                ffi::cudssSetStream(lin.handle, lin.stream),
                "cudssSetStream",
            )?;
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

        lin._row_ptr.upload(&row_ptr, lin.stream)?;
        lin._col_idx.upload(&col_idx, lin.stream)?;
        // Analysis runs on numbers; every system gets the surrogate values,
        // so none is singular at the analysis.
        lin.values
            .upload(&pattern.surrogate_values().repeat(batch), lin.stream)?;

        let ubatch = batch as i32;
        // SAFETY: as above.
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
                    lin.values.ptr.cast(),
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
                    lin.rhs.ptr.cast(),
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
                    lin.sol.ptr.cast(),
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

    fn factor(&mut self, values: &[f64], mask: &[bool]) -> Result<(), LinearError> {
        self.check("values", values.len(), self.nnz)?;
        self.check("mask", mask.len(), 1)?;
        self.counts.factor_calls += 1;
        let selected = mask.iter().filter(|&&m| m).count();
        if selected == 0 {
            return Ok(());
        }
        self.values.upload(values, self.stream)?;
        // SAFETY: the same device buffer `a` was created over; this tells
        // cuDSS its contents changed, as gpusim2grid and the probe do.
        cudss(
            unsafe { ffi::cudssMatrixSetValues(self.a, self.values.ptr.cast()) },
            "cudssMatrixSetValues",
        )?;
        self.set_mask(mask)?;
        let again = mask.iter().zip(&self.ever_factored).all(|(&m, &f)| !m || f);
        if self.refactorization && again {
            self.execute(ffi::CUDSS_PHASE_REFACTORIZATION, "REFACTORIZATION")?;
        } else {
            self.execute(ffi::CUDSS_PHASE_FACTORIZATION, "FACTORIZATION")?;
        }
        for ((f, ever), &m) in self
            .factored
            .iter_mut()
            .zip(&mut self.ever_factored)
            .zip(mask)
        {
            *f = m || (*f && KEEPS_UNSELECTED_FACTORS);
            *ever |= m;
        }
        self.counts.systems_factored += selected;
        Ok(())
    }

    fn solve(&mut self, rhs: &[f64], x: &mut [f64], mask: &[bool]) -> Result<(), LinearError> {
        self.check("rhs", rhs.len(), self.n)?;
        self.check("x", x.len(), self.n)?;
        self.check("mask", mask.len(), 1)?;
        if let Some(system) = (0..self.batch).find(|&s| mask[s] && !self.factored[s]) {
            return Err(LinearError::NotFactored { system });
        }
        self.counts.solve_calls += 1;
        let selected = mask.iter().filter(|&&m| m).count();
        if selected == 0 {
            return Ok(());
        }
        self.rhs.upload(rhs, self.stream)?;
        self.set_mask(mask)?;
        self.execute(ffi::CUDSS_PHASE_SOLVE, "SOLVE")?;
        self.sol.download(&mut self.staging, self.stream)?;
        let n = self.n;
        for s in (0..self.batch).filter(|&s| mask[s]) {
            x[s * n..(s + 1) * n].copy_from_slice(&self.staging[s * n..(s + 1) * n]);
        }
        self.counts.systems_solved += selected;
        Ok(())
    }

    fn keeps_unselected_factors(&self) -> bool {
        KEEPS_UNSELECTED_FACTORS
    }

    fn counts(&self) -> LinearCounts {
        self.counts
    }
}

impl Drop for CudssUniformBatch {
    fn drop(&mut self) {
        // SAFETY: each handle is destroyed once, if it was made, descriptors
        // before the data and handle they belong to, the stream last; the
        // device buffers are freed after this body.
        unsafe {
            if !self.stream.is_null() {
                ffi::cudaStreamSynchronize(self.stream);
            }
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
            if !self.stream.is_null() {
                ffi::cudaStreamDestroy(self.stream);
            }
        }
    }
}
