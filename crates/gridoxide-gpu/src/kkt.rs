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
//! The assembly and the Ruiz equilibration are the CPU solvers' own
//! ([`LowerPattern`]), run on the device by kernels (`go_kkt_*` in
//! `cuda/kernels.cu`) in the same arithmetic and order, so the factorized
//! matrix is the same `D K D` to the last bit: the lower triangle in
//! compressed columns, which cuDSS takes as the upper triangle in compressed
//! rows (`CUDSS_MVIEW_UPPER`). The host only copies the triplets' values
//! into a pinned buffer, in triplet order; summing duplicates is a gather
//! per slot and every row's maximum a gather over the slots it touches, so
//! no kernel needs an atomic. The analysis runs once, at the first
//! factorization, on its values; every later one is numeric only.
//!
//! On case10000_goc the host's assembly and equilibration took 11 ms a
//! factorization, more than cuDSS's factorization (3.2 ms).
//!
//! The triplets' values and the right-hand side cross the bus every
//! factorization and solve; the matrix, its scale and the factor stay on the
//! device.

use std::ptr;

use gridoxide_sparse::ldlt::{Inertia, LowerPattern, SymmetricSolver};

use gridoxide_sparse::ldlt::RUIZ_PASSES;

use crate::cuda::{DeviceBuffer, PinnedBuffer, Stream, copy_to_device, cuda, ffi};
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
    row_ptr: DeviceBuffer<i32>,
    col: DeviceBuffer<i32>,
    values: DeviceBuffer<f64>,
    rhs: DeviceBuffer<f64>,
    solution: DeviceBuffer<f64>,
    /// The device's half of the assembly and equilibration; see [`Assembly`].
    assembly: Assembly,
    analysed: bool,
    /// The inertia of the last factorization, `None` if it failed or there
    /// was none.
    inertia: Option<Inertia>,
}

/// What the device assembles and equilibrates the matrix from: the
/// triplets' values (staged in pinned memory), where each slot sums them
/// from, the slots' rows and columns, and per row or column the slots in it.
struct Assembly {
    nnz: usize,
    n_trip: usize,
    host_trip: PinnedBuffer<f64>,
    trip: DeviceBuffer<f64>,
    /// Slot `s` sums `trip[src[t]]` for `t` in `slot_ptr[s]..slot_ptr[s+1]`.
    slot_ptr: DeviceBuffer<u32>,
    src: DeviceBuffer<u32>,
    row: DeviceBuffer<u32>,
    col: DeviceBuffer<u32>,
    /// The slots in row or column `i`: `inc[inc_ptr[i]..inc_ptr[i+1]]`.
    inc_ptr: DeviceBuffer<u32>,
    inc: DeviceBuffer<u32>,
    row_max: DeviceBuffer<f64>,
    factor: DeviceBuffer<f64>,
    /// The equilibration of the last factorization: it factorized `D K D`.
    scale: DeviceBuffer<f64>,
    bad: DeviceBuffer<i32>,
    /// The strictly lower part as rows, for the mat-vec: row `r` has columns
    /// `lo_col[k]` and values `values[lo_src[k]]`, `k` in `lo_ptr[r]..`.
    lo_ptr: DeviceBuffer<u32>,
    lo_col: DeviceBuffer<u32>,
    lo_src: DeviceBuffer<u32>,
    /// The mat-vec's vectors.
    mv_x: DeviceBuffer<f64>,
    mv_y: DeviceBuffer<f64>,
}

impl Assembly {
    fn new(pattern: &LowerPattern, stream: ffi::cudaStream_t) -> Result<Self, LinearError> {
        let n = pattern.n();
        let nnz = pattern.row_idx().len();
        let slots = pattern.slots();
        let index = |i: usize| -> Result<u32, LinearError> {
            u32::try_from(i).map_err(|_| LinearError::Shape(format!("index {i} beyond 32 bits")))
        };
        // Per slot, its triplets in triplet order (the CPU's summation order).
        let mut slot_ptr = vec![0u32; nnz + 1];
        for s in slots.iter().flatten() {
            slot_ptr[s + 1] += 1;
        }
        for s in 0..nnz {
            slot_ptr[s + 1] += slot_ptr[s];
        }
        let mut next: Vec<u32> = slot_ptr[..nnz].to_vec();
        let mut src = vec![0u32; slot_ptr[nnz] as usize];
        for (t, s) in slots.iter().enumerate() {
            if let Some(s) = *s {
                src[next[s] as usize] = index(t)?;
                next[s] += 1;
            }
        }
        let row: Vec<u32> = pattern
            .row_idx()
            .iter()
            .map(|&r| index(r))
            .collect::<Result<_, _>>()?;
        let mut col = vec![0u32; nnz];
        let col_ptr = pattern.col_ptr();
        for c in 0..n {
            for slot in &mut col[col_ptr[c]..col_ptr[c + 1]] {
                *slot = index(c)?;
            }
        }
        // Per row or column, the slots in it (a diagonal slot once).
        let mut inc_ptr = vec![0u32; n + 1];
        for s in 0..nnz {
            inc_ptr[row[s] as usize + 1] += 1;
            if col[s] != row[s] {
                inc_ptr[col[s] as usize + 1] += 1;
            }
        }
        for i in 0..n {
            inc_ptr[i + 1] += inc_ptr[i];
        }
        let mut next: Vec<u32> = inc_ptr[..n].to_vec();
        let mut inc = vec![0u32; inc_ptr[n] as usize];
        for s in 0..nnz {
            let (r, c) = (row[s] as usize, col[s] as usize);
            inc[next[r] as usize] = index(s)?;
            next[r] += 1;
            if c != r {
                inc[next[c] as usize] = index(s)?;
                next[c] += 1;
            }
        }
        // The strictly lower part by rows (the transpose of the upper rows
        // cuDSS reads), columns ascending within a row.
        let mut lo_ptr = vec![0u32; n + 1];
        for s in 0..nnz {
            if row[s] != col[s] {
                lo_ptr[row[s] as usize + 1] += 1;
            }
        }
        for i in 0..n {
            lo_ptr[i + 1] += lo_ptr[i];
        }
        let mut next: Vec<u32> = lo_ptr[..n].to_vec();
        let mut lo_col = vec![0u32; lo_ptr[n] as usize];
        let mut lo_src = vec![0u32; lo_ptr[n] as usize];
        for s in 0..nnz {
            let r = row[s] as usize;
            if r != col[s] as usize {
                lo_col[next[r] as usize] = col[s];
                lo_src[next[r] as usize] = index(s)?;
                next[r] += 1;
            }
        }
        let up = |v: &[u32]| -> Result<DeviceBuffer<u32>, LinearError> {
            let mut d = DeviceBuffer::new(v.len().max(1))?;
            if !v.is_empty() {
                d.upload(v, stream)?;
            }
            Ok(d)
        };
        let n_trip = slots.len();
        Ok(Self {
            nnz,
            n_trip,
            host_trip: PinnedBuffer::new(n_trip.max(1))?,
            trip: DeviceBuffer::new(n_trip.max(1))?,
            slot_ptr: up(&slot_ptr)?,
            src: up(&src)?,
            row: up(&row)?,
            col: up(&col)?,
            inc_ptr: up(&inc_ptr)?,
            inc: up(&inc)?,
            row_max: DeviceBuffer::new(n.max(1))?,
            factor: DeviceBuffer::new(n.max(1))?,
            scale: DeviceBuffer::new(n.max(1))?,
            bad: DeviceBuffer::new(1)?,
            lo_ptr: up(&lo_ptr)?,
            lo_col: up(&lo_col)?,
            lo_src: up(&lo_src)?,
            mv_x: DeviceBuffer::new(n.max(1))?,
            mv_y: DeviceBuffer::new(n.max(1))?,
        })
    }

    /// Queues the assembly and equilibration of `entries` into `values` on
    /// `stream`: LowerPattern::assemble's arithmetic, on the device. A
    /// non-finite result sets `bad`.
    fn run(
        &mut self,
        source: Source<'_>,
        values: &mut DeviceBuffer<f64>,
        n: usize,
        stream: ffi::cudaStream_t,
    ) -> Result<(), LinearError> {
        use rayon::prelude::*;
        let nnz = self.nnz as u32;
        let n = n as u32;
        if self.n_trip > 0 {
            // SAFETY: the pinned buffer holds n_trip values, and no copy out
            // of it is in flight: the last factorization synchronized.
            let host =
                unsafe { std::slice::from_raw_parts_mut(self.host_trip.as_mut_ptr(), self.n_trip) };
            // In parallel: 1.2M values on case24464_goc.
            const CHUNK: usize = 1 << 15;
            match source {
                Source::Triplets(entries) => host
                    .par_chunks_mut(CHUNK)
                    .zip(entries.par_chunks(CHUNK))
                    .for_each(|(h, e)| h.iter_mut().zip(e).for_each(|(h, e)| *h = e.2)),
                Source::Values(v) => host
                    .par_chunks_mut(CHUNK)
                    .zip(v.par_chunks(CHUNK))
                    .for_each(|(h, v)| h.copy_from_slice(v)),
            }
            // SAFETY: `trip` holds n_trip values.
            unsafe { copy_to_device(self.trip.ptr, &*host, stream)? };
        }
        // SAFETY: every pointer is a device buffer of the length the kernel
        // reads (made in `new` for this pattern).
        unsafe {
            copy_to_device(self.bad.ptr, &[0i32], stream)?;
            launched(
                ffi::go_kkt_gather(
                    self.slot_ptr.ptr,
                    self.src.ptr,
                    self.trip.ptr,
                    values.ptr,
                    nnz,
                    stream.cast(),
                ),
                "go_kkt_gather",
            )?;
            let row_max = |values: &DeviceBuffer<f64>| {
                launched(
                    ffi::go_kkt_row_max(
                        self.inc_ptr.ptr,
                        self.inc.ptr,
                        values.ptr,
                        self.row_max.ptr,
                        n,
                        stream.cast(),
                    ),
                    "go_kkt_row_max",
                )
            };
            row_max(values)?;
            for pass in 0..RUIZ_PASSES {
                launched(
                    ffi::go_kkt_factor(
                        self.row_max.ptr,
                        self.factor.ptr,
                        self.scale.ptr,
                        (pass == 0) as i32,
                        n,
                        stream.cast(),
                    ),
                    "go_kkt_factor",
                )?;
                let last = pass + 1 == RUIZ_PASSES;
                launched(
                    ffi::go_kkt_scale(
                        self.row.ptr,
                        self.col.ptr,
                        self.factor.ptr,
                        values.ptr,
                        if last { self.bad.ptr } else { ptr::null_mut() },
                        nnz,
                        stream.cast(),
                    ),
                    "go_kkt_scale",
                )?;
                if !last {
                    row_max(values)?;
                }
            }
        }
        Ok(())
    }
}

/// What a factorization's values come from: the triplets, or the values
/// alone in the order of the pattern's keys.
#[derive(Clone, Copy)]
enum Source<'a> {
    Triplets(&'a [(usize, usize, f64)]),
    Values(&'a [f64]),
}

fn launched(err: i32, what: &str) -> Result<(), LinearError> {
    cuda(err as ffi::cudaError_t, what)
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
        let assembly = Assembly::new(&pattern, stream.0)?;
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
        let threading_lib = std::env::var("CUDSS_THREADING_LIB")
            .ok()
            .filter(|lib| !lib.is_empty())
            .and_then(|lib| std::ffi::CString::new(lib).ok());
        // SAFETY: each call gets valid out-pointers; the descriptors point at
        // the buffers above, which the solver keeps.
        unsafe {
            cudss(ffi::cudssCreate(&mut sys.handle), "cudssCreate")?;
            cudss(ffi::cudssSetStream(sys.handle, stream.0), "cudssSetStream")?;
            // cuDSS's OpenMP host threading for the analysis, as the batch
            // solver's (`cudss.rs`): on when CUDSS_THREADING_LIB names its
            // layer library.
            if let Some(lib) = &threading_lib {
                cudss(
                    ffi::cudssSetThreadingLayer(sys.handle, lib.as_ptr()),
                    "cudssSetThreadingLayer",
                )?;
            }
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
            row_ptr,
            col,
            values,
            rhs,
            solution,
            assembly,
            analysed: false,
            inertia: None,
        })
    }

    pub fn kind(&self) -> Factorization {
        self.kind
    }

    /// The matrix of `entries` assembled and equilibrated on the device, as
    /// it would be factorized, and its scale, copied back: what
    /// [`LowerPattern::assemble`] computes on the host, for comparing the
    /// two.
    pub fn assemble_on_device(
        &mut self,
        entries: &[(usize, usize, f64)],
    ) -> Result<(Vec<f64>, Vec<f64>), LinearError> {
        let n = self.pattern.n();
        self.assembly.run(
            Source::Triplets(entries),
            &mut self.values,
            n,
            self.stream.0,
        )?;
        let mut values = vec![0.0; self.values.len()];
        self.values.download(&mut values, self.stream.0)?;
        let mut scale = vec![0.0; n];
        if n > 0 {
            self.assembly.scale.download(&mut scale, self.stream.0)?;
        }
        values.truncate(self.pattern.row_idx().len());
        Ok((values, scale))
    }

    /// Factorizes; `Ok(None)` when the matrix has no factorization of this
    /// kind (a zero pivot, or not positive definite for Cholesky).
    fn try_factorize(&mut self, source: Source<'_>) -> Result<Option<Inertia>, LinearError> {
        let n = self.pattern.n();
        self.assembly
            .run(source, &mut self.values, n, self.stream.0)?;
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
        let mut bad = [0i32];
        self.assembly.bad.download(&mut bad, self.stream.0)?;
        if bad[0] != 0 {
            return Ok(None);
        }
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
        let scale = self.assembly.scale.ptr;
        // SAFETY: `rhs`, `solution` and `scale` are the solver's own buffers
        // of n values; the upload goes through a raw pointer because `solve`
        // takes `&self`.
        unsafe {
            copy_to_device(self.rhs.ptr, rhs, self.stream.0)?;
            launched(
                ffi::go_scale_vec(self.rhs.ptr, scale, n as u32, self.stream.0.cast()),
                "go_scale_vec",
            )?;
        }
        self.sys.execute(ffi::CUDSS_PHASE_SOLVE, "SOLVE")?;
        // SAFETY: as above.
        unsafe {
            launched(
                ffi::go_scale_vec(self.solution.ptr, scale, n as u32, self.stream.0.cast()),
                "go_scale_vec",
            )?;
        }
        let mut x = vec![0.0; n];
        self.solution.download(&mut x, self.stream.0)?;
        Ok(x.iter().all(|v| v.is_finite()).then_some(x))
    }
}

impl CudssSolver {
    /// `K x` on the device: the equilibrated `D K D` it factorized, as
    /// `D⁻¹ (D K D) (D⁻¹ x)`.
    fn try_multiply(&self, x: &[f64]) -> Result<Option<Vec<f64>>, LinearError> {
        let n = self.pattern.n();
        if self.inertia.is_none() || x.len() != n || n == 0 {
            return Ok(None);
        }
        let a = &self.assembly;
        let stream = self.stream.0;
        // SAFETY: every buffer is the solver's own, of n values (the vectors
        // and the scale) or of the pattern's sizes; the upload goes through
        // a raw pointer because `multiply` takes `&self`.
        unsafe {
            copy_to_device(a.mv_x.ptr, x, stream)?;
            launched(
                ffi::go_unscale_vec(a.mv_x.ptr, a.scale.ptr, a.mv_y.ptr, n as u32, stream.cast()),
                "go_unscale_vec",
            )?;
            launched(
                ffi::go_kkt_symv(
                    self.row_ptr.ptr,
                    self.col.ptr,
                    a.lo_ptr.ptr,
                    a.lo_col.ptr,
                    a.lo_src.ptr,
                    self.values.ptr,
                    a.mv_y.ptr,
                    a.mv_x.ptr,
                    n as u32,
                    stream.cast(),
                ),
                "go_kkt_symv",
            )?;
            launched(
                ffi::go_unscale_vec(a.mv_x.ptr, a.scale.ptr, a.mv_y.ptr, n as u32, stream.cast()),
                "go_unscale_vec",
            )?;
        }
        let mut y = vec![0.0; n];
        a.mv_y.download(&mut y, stream)?;
        Ok(y.iter().all(|v| v.is_finite()).then_some(y))
    }
}

impl SymmetricSolver for CudssSolver {
    fn matches(&self, n: usize, entries: &[(usize, usize, f64)]) -> bool {
        self.pattern.matches(n, entries)
    }

    /// A device error counts as a failed factorization: the interior point
    /// then regularizes, and gives up once that stops helping.
    fn factorize(&mut self, entries: &[(usize, usize, f64)]) -> bool {
        self.inertia = (entries.len() == self.assembly.n_trip)
            .then(|| self.try_factorize(Source::Triplets(entries)).ok().flatten())
            .flatten();
        self.inertia.is_some()
    }

    fn factorize_values(&mut self, values: &[f64]) -> Option<bool> {
        self.inertia = (values.len() == self.assembly.n_trip)
            .then(|| self.try_factorize(Source::Values(values)).ok().flatten())
            .flatten();
        Some(self.inertia.is_some())
    }

    fn inertia(&self) -> Inertia {
        self.inertia.unwrap_or_default()
    }

    fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        self.try_solve(rhs).ok().flatten()
    }

    fn multiply(&self, x: &[f64]) -> Option<Vec<f64>> {
        self.try_multiply(x).ok().flatten()
    }
}

/// `NlpOptions::kkt_solver` for the augmented system: cuDSS's `LDLᵀ`,
/// deterministic.
/// An error, which ends the solve, if the device cannot be set up.
pub fn cudss_ldlt_solver(
    n: usize,
    keys: &[(usize, usize)],
) -> Result<Box<dyn SymmetricSolver>, String> {
    let solver = CudssSolver::new(n, keys, Factorization::Ldlt, true).map_err(|e| e.to_string())?;
    Ok(Box::new(solver))
}

/// `NlpOptions::kkt_solver` for `KktForm::Hybrid`: cuDSS's Cholesky,
/// deterministic.
/// An error, which ends the solve, if the device cannot be set up.
pub fn cudss_llt_solver(
    n: usize,
    keys: &[(usize, usize)],
) -> Result<Box<dyn SymmetricSolver>, String> {
    let solver = CudssSolver::new(n, keys, Factorization::Llt, true).map_err(|e| e.to_string())?;
    Ok(Box::new(solver))
}

/// [`cudss_ldlt_solver`] without cuDSS's deterministic mode, for benchmarks:
/// its solves take half the time (case24464_goc: 1.65 against 3.45 ms
/// each, 12 ms an iteration), as MadNLP's cuDSS runs by default, but the
/// path is no longer reproducible from run to run (case2869_pegase took 60
/// to 107 iterations in four).
pub fn cudss_ldlt_solver_fast(
    n: usize,
    keys: &[(usize, usize)],
) -> Result<Box<dyn SymmetricSolver>, String> {
    let solver =
        CudssSolver::new(n, keys, Factorization::Ldlt, false).map_err(|e| e.to_string())?;
    Ok(Box::new(solver))
}

/// [`cudss_llt_solver`] without cuDSS's deterministic mode; see
/// [`cudss_ldlt_solver_fast`].
pub fn cudss_llt_solver_fast(
    n: usize,
    keys: &[(usize, usize)],
) -> Result<Box<dyn SymmetricSolver>, String> {
    let solver = CudssSolver::new(n, keys, Factorization::Llt, false).map_err(|e| e.to_string())?;
    Ok(Box::new(solver))
}
