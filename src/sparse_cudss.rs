//! Thin internal wrapper around NVIDIA cuDSS, gridoxide's device-resident
//! sparse direct solver for `plans/GPU_PLAN.md` Phase 3 — mirrors
//! `sparse_pardiso.rs`'s shape and role (a `LinearSolver` backed by an
//! external FFI library, analysis cached once and reused across repeated
//! numeric refactorizations) but backed by cuDSS on a CUDA GPU instead of
//! oneMKL PARDISO on the CPU.
//!
//! All raw-pointer FFI handling — both cuDSS's own opaque handles and the
//! CUDA runtime calls that manage device memory — is confined to this one
//! file, matching `sparse_klu.rs`/`sparse_pardiso.rs`'s own isolation
//! principle.
//!
//! cuDSS's C API (`cudss.h`) is `cudssExecute(handle, phase, config, data,
//! A, x, b)`, called repeatedly with different `phase` bitflags against a
//! persistent `cudssData_t` handle — structurally the same shape as
//! PARDISO's `phase`-parameterized calls against its persistent `pt`, just
//! split into named phases (`CUDSS_PHASE_ANALYSIS`, `_FACTORIZATION`,
//! `_REFACTORIZATION`, `_SOLVE`) instead of numeric ones (11/22/33). Unlike
//! PARDISO, cuDSS operates on **device** memory: `A`'s CSR arrays and the
//! right-hand-side/solution dense vectors must already live in GPU memory
//! before `cudssExecute` is called, so this module owns the
//! `cudaMalloc`/`cudaMemcpy`/`cudaFree` calls that keep them there —
//! there is no separate crate dependency for this (see `build.rs`'s `cudss`
//! module for why: it links `libcudart` directly alongside `libcudss`).
//!
//! # Two solvers, and which one to use
//!
//! [`CudssBatchedSystem`] is the one that matters. It uses cuDSS's **uniform
//! batch** entry point: `B` independent same-pattern systems, one analysis on
//! a single block, one shared CSR structure, values striding into a buffer the
//! assembly kernel writes directly. Nothing crosses the host boundary after
//! construction.
//!
//! [`CudssRealSystem`] is the earlier design and is kept as the **A/B
//! control**, not for production. It embeds the batch block-diagonally and
//! hands cuDSS one enormous general sparse matrix — the approach
//! `plans/GPU_PLAN.md` §3 property 2 chose because it works on any sparse
//! direct solver, batched or not, which is what kept the AMD path open. That
//! generality turned out to cost ~95% of the runtime (see
//! [`CudssBatchedSystem`]'s own comment for why), and the two paths existing
//! side by side is what makes that claim measurable rather than asserted —
//! `examples/bde_profile.rs` runs both.
//!
//! It also still re-uploads values and the right-hand side on every
//! `factor_and_solve_values` call, which is correct but is a per-iteration
//! host round trip of the batch's whole `nnz`-sized array.
//!
//! `CUDSS_MTYPE_GENERAL`/`CUDSS_MVIEW_FULL` select "general, non-symmetric"
//! storage for both — the same shape KLU and PARDISO are given (gridoxide's
//! Newton Jacobian is never symmetric).

use std::ffi::c_void;
use std::ptr;

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/cudss_bindings.rs"));
}
use bindings::{
    cudaMemcpy, cudaMemcpyKind_cudaMemcpyDeviceToHost as D2H, cudaMemcpyKind_cudaMemcpyHostToDevice as H2D,
    cudaError_t, cudaMalloc, cudaFree, cudaDeviceSynchronize, cudaError_cudaSuccess as cudaSuccess,
    cudaStream_t,
    cudssConfig_t, cudssConfigCreate, cudssConfigDestroy, cudssConfigSet,
    cudssConfigParam_t_CUDSS_CONFIG_DETERMINISTIC_MODE as CUDSS_CONFIG_DETERMINISTIC_MODE,
    cudssCreate, cudssData_t, cudssDataCreate, cudssDataDestroy, cudssDestroy,
    cudssExecute, cudssHandle_t, cudssMatrixCreateBatchCsr, cudssMatrixCreateBatchDn, cudssMatrixCreateCsr,
    cudssMatrixCreateDn, cudssMatrixDestroy, cudssMatrixSetBatchValues, cudssMatrixSetValues, cudssSetStream,
    cudssMatrix_t, cudssStatus_t, cudssStatus_t_CUDSS_STATUS_SUCCESS as CUDSS_SUCCESS,
    cudssDataType_t_CUDSS_R_32I as CUDSS_R_32I, cudssDataType_t_CUDSS_R_64F as CUDSS_R_64F,
    cudssIndexBase_t_CUDSS_BASE_ZERO as CUDSS_BASE_ZERO, cudssLayout_t_CUDSS_LAYOUT_COL_MAJOR as CUDSS_LAYOUT_COL_MAJOR,
    cudssMatrixType_t_CUDSS_MTYPE_GENERAL as CUDSS_MTYPE_GENERAL,
    cudssMatrixViewType_t_CUDSS_MVIEW_FULL as CUDSS_MVIEW_FULL,
    cudssPhase_t_CUDSS_PHASE_ANALYSIS as PHASE_ANALYSIS,
    cudssPhase_t_CUDSS_PHASE_FACTORIZATION as PHASE_FACTORIZATION,
    cudssPhase_t_CUDSS_PHASE_REFACTORIZATION as PHASE_REFACTORIZATION,
    cudssPhase_t_CUDSS_PHASE_SOLVE as PHASE_SOLVE,
};

/// The CSR layout helpers this module used to define now live in
/// [`crate::device_layout`], which is not feature-gated: they are pure
/// host-side index arithmetic, they are shared with the CUDA assembly
/// kernel's scatter map, and getting them wrong breaks a whole batch — so
/// they belong somewhere every `cargo test` exercises them, not only a box
/// with cuDSS installed. Re-exported here because that is where callers of
/// this module expect to find them.
pub use crate::device_layout::{build_csr_structure, csr_scatter_map, pack_values_slice};

fn pack_values(entries: &[(usize, usize, f64)], groups: &[Vec<usize>]) -> Vec<f64> {
    groups.iter().map(|g| g.iter().map(|&i| entries[i].2).sum()).collect()
}

fn cuda_check(err: cudaError_t, what: &str) -> Option<()> {
    if err != cudaSuccess {
        eprintln!("cudss: CUDA call failed ({what}): error code {err}");
        return None;
    }
    Some(())
}

/// Blocks until every previously enqueued CUDA operation (on **every** stream
/// in this process's context) has completed.
///
/// This used to be a per-iteration requirement of the device-resident path:
/// gridoxide's kernels ran on CubeCL's own stream while cuDSS ran on the
/// default one, so without a device-wide barrier cuDSS's refactorization
/// could start reading the Jacobian values buffer before the assembly kernel
/// had finished writing it — not a crash, since both sides are well-formed
/// CUDA operations on valid memory, but a numeric race that perturbed a
/// Newton step's `Δx` enough to change the iteration count without changing
/// the converged answer, which is exactly how it was caught.
///
/// [`CudssBatchedSystem::set_stream`] removes the need for it: gridoxide's
/// kernels and cuDSS now share one stream, so the ordering is implied. Kept
/// for the legacy [`CudssRealSystem`] paths and for tests.
pub fn device_synchronize() -> Option<()> {
    cuda_check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

fn cudss_check(status: cudssStatus_t, what: &str) -> Option<()> {
    if status != CUDSS_SUCCESS {
        eprintln!("cudss: call failed ({what}): status {status}");
        return None;
    }
    Some(())
}

unsafe fn device_alloc(bytes: usize) -> Option<*mut c_void> {
    let mut ptr: *mut c_void = ptr::null_mut();
    cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) }, "cudaMalloc")?;
    Some(ptr)
}

unsafe fn upload<T>(dst: *mut c_void, src: &[T]) -> Option<()> {
    let bytes = std::mem::size_of_val(src);
    cuda_check(unsafe { cudaMemcpy(dst, src.as_ptr() as *const c_void, bytes, H2D) }, "cudaMemcpy H2D")
}

unsafe fn download<T>(dst: &mut [T], src: *mut c_void) -> Option<()> {
    let bytes = std::mem::size_of_val(dst);
    cuda_check(unsafe { cudaMemcpy(dst.as_mut_ptr() as *mut c_void, src, bytes, D2H) }, "cudaMemcpy D2H")
}

/// A real, general (non-symmetric) sparse system solved on-device via cuDSS,
/// with a fixed sparsity pattern reused across repeated
/// [`factor_and_solve`](Self::factor_and_solve) calls — mirrors
/// `sparse_pardiso::PardisoRealSystem`'s role. `new` runs cuDSS's analysis
/// (reordering + symbolic factorization) and an initial numeric
/// factorization once; every subsequent call reuses that analysis via
/// `CUDSS_PHASE_REFACTORIZATION`, cuDSS's equivalent of PARDISO's phase 22 /
/// KLU's `klu_refactor`.
pub struct CudssRealSystem {
    n: usize,
    groups: Vec<Vec<usize>>,

    handle: cudssHandle_t,
    config: cudssConfig_t,
    data: cudssData_t,
    matrix_a: cudssMatrix_t,
    matrix_b: cudssMatrix_t,
    matrix_x: cudssMatrix_t,

    d_row_ptr: *mut c_void,
    d_col_idx: *mut c_void,
    d_values: *mut c_void,
    d_rhs: *mut c_void,
    d_x: *mut c_void,
    /// `false` for [`new_device_resident`](Self::new_device_resident): the
    /// values buffer there is owned by an external caller (e.g.
    /// `gpu::GpuAssembler`, kept alive for at least as long as this
    /// `CudssRealSystem`), so `Drop` must not free it.
    owns_values: bool,
}

/// Where `alloc` gets its values buffer from — see [`CudssRealSystem::alloc`].
enum ValuesSource<'a> {
    /// Allocate a fresh device buffer and upload `values` into it. This
    /// system owns the buffer and frees it on `Drop`.
    Owned(&'a [f64]),
    /// Point directly at an already-live device pointer this system does
    /// not own — the device-resident path (see
    /// [`CudssRealSystem::new_device_resident`]).
    External(u64),
}

impl CudssRealSystem {
    /// Builds the device-resident CSR structure, uploads an initial
    /// numeric pattern, and runs cuDSS's analysis + factorization phases.
    /// Subsequent calls to `factor_and_solve`/`factor_and_solve_values` must
    /// supply the exact same `(row, col)` pairs in the exact same order —
    /// same positional-correspondence precondition as every other
    /// `LinearSolver` backend. Returns `None` if allocation, analysis, or
    /// the initial factorization fails (e.g. a structurally singular
    /// pattern).
    pub fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        let pairs: Vec<(usize, usize)> = entries.iter().map(|&(r, c, _)| (r, c)).collect();
        let (row_ptr, col_idx, groups) = build_csr_structure(n, &pairs);
        let values = pack_values(entries, &groups);
        let nnz = col_idx.len();

        let mut sys = unsafe { Self::alloc(n, nnz, &row_ptr, &col_idx, ValuesSource::Owned(&values)) }?;

        cudss_check(sys.execute(PHASE_ANALYSIS), "analysis")?;
        cudss_check(sys.execute(PHASE_FACTORIZATION), "initial factorization")?;

        sys.groups = groups;
        Some(sys)
    }

    /// Like [`new`](Self::new), but the values buffer is an externally-owned
    /// device pointer — `plans/GPU_PLAN.md` §6 Phase 3's device-resident
    /// path: `values_device_ptr` is typically `gpu::GpuAssembler::values_ptr`,
    /// a persistent buffer that assembler keeps rewriting in place every
    /// Newton iteration, so the Jacobian's numeric values never cross the
    /// host↔device boundary at all — see
    /// [`solve_device_resident`](Self::solve_device_resident).
    ///
    /// `row_ptr`/`col_idx` (CSR structure, from
    /// `sparse_cudss::build_csr_structure`/`csr_scatter_map`'s caller) are
    /// still uploaded and owned here, same as `new`: they're static per
    /// topology and small, so there's no reason to move their ownership.
    ///
    /// **Precondition**: `values_device_ptr` must already hold valid
    /// numeric values for this pattern before this call — cuDSS's
    /// factorization phase (run here, once) reads actual numbers to choose
    /// pivots, not just structure. The caller must have launched at least
    /// one GPU assembly pass into that buffer first.
    pub fn new_device_resident(n: usize, row_ptr: &[i32], col_idx: &[i32], values_device_ptr: u64) -> Option<Self> {
        let nnz = col_idx.len();
        let mut sys = unsafe { Self::alloc(n, nnz, row_ptr, col_idx, ValuesSource::External(values_device_ptr)) }?;

        cudss_check(sys.execute(PHASE_ANALYSIS), "analysis")?;
        cudss_check(sys.execute(PHASE_FACTORIZATION), "initial factorization")?;

        Some(sys)
    }

    /// Allocates device buffers, uploads the fixed CSR structure and an
    /// initial value set, and wraps everything in cuDSS's matrix/handle
    /// objects. Split out from `new` so every fallible step before the
    /// analysis/factorization phases happens behind one `?`-chain, with the
    /// resulting `Self` (and its `Drop`) responsible for cleanup from that
    /// point on regardless of where `new` bails out afterward.
    unsafe fn alloc(n: usize, nnz: usize, row_ptr: &[i32], col_idx: &[i32], values: ValuesSource) -> Option<Self> {
        let d_row_ptr = unsafe { device_alloc((n + 1) * size_of::<i32>()) }?;
        let d_col_idx = unsafe { device_alloc(nnz * size_of::<i32>()) }?;
        let d_rhs = unsafe { device_alloc(n * size_of::<f64>()) }?;
        let d_x = unsafe { device_alloc(n * size_of::<f64>()) }?;

        let (d_values, owns_values) = match values {
            ValuesSource::Owned(initial) => {
                let d = unsafe { device_alloc(nnz * size_of::<f64>()) }?;
                unsafe { upload(d, initial)? };
                (d, true)
            }
            ValuesSource::External(ptr) => (ptr as usize as *mut c_void, false),
        };

        unsafe {
            upload(d_row_ptr, row_ptr)?;
            upload(d_col_idx, col_idx)?;
        }

        let mut handle: cudssHandle_t = ptr::null_mut();
        cudss_check(unsafe { cudssCreate(&mut handle) }, "cudssCreate")?;
        let mut config: cudssConfig_t = ptr::null_mut();
        cudss_check(unsafe { cudssConfigCreate(&mut config) }, "cudssConfigCreate")?;
        // Enabled defensively: this crate's whole value proposition is
        // agreeing with reference implementations, so reproducible
        // factorization is worth the (if any) cost even though it did not
        // measurably change the device-resident iteration-count
        // characteristic documented on
        // `bde::solve_batch_block_diagonal_device_resident`.
        let deterministic: i32 = 1;
        cudss_check(
            unsafe {
                cudssConfigSet(
                    config,
                    CUDSS_CONFIG_DETERMINISTIC_MODE,
                    &deterministic as *const i32 as *const c_void,
                    size_of::<i32>(),
                )
            },
            "cudssConfigSet(DETERMINISTIC_MODE)",
        )?;
        let mut data: cudssData_t = ptr::null_mut();
        cudss_check(unsafe { cudssDataCreate(handle, &mut data) }, "cudssDataCreate")?;

        let mut matrix_a: cudssMatrix_t = ptr::null_mut();
        cudss_check(
            unsafe {
                cudssMatrixCreateCsr(
                    &mut matrix_a,
                    n as i64,
                    n as i64,
                    nnz as i64,
                    d_row_ptr,
                    ptr::null(),
                    d_col_idx,
                    d_values,
                    CUDSS_R_32I,
                    CUDSS_R_32I,
                    CUDSS_R_64F,
                    CUDSS_MTYPE_GENERAL,
                    CUDSS_MVIEW_FULL,
                    CUDSS_BASE_ZERO,
                )
            },
            "cudssMatrixCreateCsr",
        )?;

        let mut matrix_b: cudssMatrix_t = ptr::null_mut();
        cudss_check(
            unsafe { cudssMatrixCreateDn(&mut matrix_b, n as i64, 1, n as i64, d_rhs, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR) },
            "cudssMatrixCreateDn for b",
        )?;
        let mut matrix_x: cudssMatrix_t = ptr::null_mut();
        cudss_check(
            unsafe { cudssMatrixCreateDn(&mut matrix_x, n as i64, 1, n as i64, d_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR) },
            "cudssMatrixCreateDn for x",
        )?;

        Some(Self {
            n,
            groups: Vec::new(),
            handle,
            config,
            data,
            matrix_a,
            matrix_b,
            matrix_x,
            d_row_ptr,
            d_col_idx,
            d_values,
            d_rhs,
            d_x,
            owns_values,
        })
    }

    /// Numeric-only refactorization against the cached symbolic analysis
    /// (`CUDSS_PHASE_REFACTORIZATION`), then solves `A x = b`
    /// (`CUDSS_PHASE_SOLVE`). Uploads `entries`' values and `rhs` to device
    /// memory first — see this module's doc comment on why that round trip
    /// is a correctness-first, not yet a speed-first, implementation.
    /// Returns `None` if either phase fails or the solution contains a
    /// non-finite value.
    pub fn factor_and_solve(&mut self, entries: &[(usize, usize, f64)], rhs: &[f64]) -> Option<Vec<f64>> {
        let values = pack_values(entries, &self.groups);
        self.solve_packed(&values, rhs)
    }

    fn solve_packed(&mut self, values: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
        unsafe {
            upload(self.d_values, values)?;
            upload(self.d_rhs, rhs)?;
        }

        cudss_check(self.execute(PHASE_REFACTORIZATION), "refactorization")?;
        cudss_check(self.execute(PHASE_SOLVE), "solve")?;

        let mut x = vec![0.0f64; self.n];
        unsafe { download(&mut x, self.d_x)? };
        if x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(x)
    }

    /// The device-resident counterpart to `factor_and_solve_values`: the
    /// values buffer itself is **not** uploaded here — whatever the external
    /// owner of `values_device_ptr` (see
    /// [`new_device_resident`](Self::new_device_resident)) last wrote into
    /// that buffer is what gets refactorized and solved; `cudssMatrixSetValues`
    /// below is the documented way to tell cuDSS those values may have
    /// changed (verified harmless and verified not the fix for the
    /// iteration-count characteristic documented on
    /// `bde::solve_batch_block_diagonal_device_resident` — kept anyway since
    /// it is the correct call for this access pattern regardless). Only
    /// `rhs` (small, `O(n)`, not the `O(nnz)` Jacobian) still crosses the
    /// host↔device boundary here, and only the solution comes back — see
    /// `bde::solve_batch_block_diagonal_device_resident`, the caller this
    /// exists for.
    pub fn solve_device_resident(&mut self, rhs: &[f64]) -> Option<Vec<f64>> {
        unsafe { upload(self.d_rhs, rhs)? };
        cudss_check(unsafe { cudssMatrixSetValues(self.matrix_a, self.d_values) }, "cudssMatrixSetValues")?;

        cudss_check(self.execute(PHASE_REFACTORIZATION), "refactorization")?;
        cudss_check(self.execute(PHASE_SOLVE), "solve")?;

        let mut x = vec![0.0f64; self.n];
        unsafe { download(&mut x, self.d_x)? };
        if x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(x)
    }

    fn execute(&mut self, phase: u32) -> cudssStatus_t {
        unsafe {
            cudssExecute(
                self.handle,
                phase as i32,
                self.config,
                self.data,
                self.matrix_a,
                self.matrix_x,
                self.matrix_b,
            )
        }
    }
}

/// cuDSS's **uniform batch** API: `B` independent same-size, same-pattern
/// systems solved as a batch, rather than stacked into one giant matrix.
///
/// This is the change `plans/GPU_PLAN.md` Phase 3 was missing. [`CudssRealSystem`]
/// (still here, as the A/B control) embeds the batch block-diagonally and hands
/// cuDSS *one* general sparse matrix — 10M rows at case1354pegase/4096, ~18M at
/// case9241pegase/1024. §3 property 2 chose that deliberately, because it works
/// on any sparse direct solver, batched API or not, and that is what kept the
/// AMD/rocSOLVER path open. It is also, measured, ~95% of that path's runtime:
/// a multifrontal GPU solver on a 10M-row matrix pays scheduling and bookkeeping
/// proportional to the whole assembly forest (thousands of tiny supernodes per
/// block, times B blocks) and level-schedules a triangular solve over 10M rows,
/// none of which knows the matrix is really B independent 2,450-row problems.
///
/// The uniform batch entry point is told exactly that, and the consequences are
/// structural, not incremental:
///
/// - **Analysis runs on one block.** Ordering and symbolic factorization see a
///   `blk`-row matrix, not a `B * blk`-row one.
/// - **The CSR structure is uploaded once, and shared.** Every block's
///   `rowStart`/`colIndices` pointer is *the same device pointer*
///   ([`device_layout::repeat_device_ptr`](crate::device_layout::repeat_device_ptr)),
///   because block-diagonal embedding makes every block's relative structure
///   identical — the property
///   `device_layout::stacked_scatter_matches_per_block_assumption` pins down.
///   The stacked path's `O(n_total)` row-pointer and `O(nnz_total)` column-index
///   arrays simply cease to exist.
/// - **Only the values stride.** `values[s] = base + s * nnz * 8`, pointing
///   straight into [`crate::gpu::GpuAssembler`]'s persistent output buffer,
///   which already has exactly that layout. Same for the right-hand side and
///   solution, with `ld = blk`.
///
/// Nothing in this struct ever touches host memory after construction:
/// [`refactor_and_solve`](Self::refactor_and_solve) takes no arguments and
/// returns none. The Jacobian, right-hand side and solution are all device
/// buffers owned by [`crate::gpu::GpuBatch`] and written by its kernels.
///
/// # Ownership
///
/// The `Vec`s of pointers and dimensions are **members, not temporaries**:
/// cuDSS's batch API takes host arrays by pointer and reads them again at
/// execute time, so they must outlive the matrix objects. Dropping them early
/// would be a use-after-free that only shows up as wrong numbers.
///
/// # Verify before trusting
///
/// `cudssMatrixCreateBatchCsr`'s parameter order below is written against
/// cuDSS's documented 0.4+ signature. Several parameters are `void*`/`void**`,
/// so a wrong order can compile cleanly and produce silent garbage — which is
/// why [`batched_ffi_smoke_test`](self::batched_ffi_smoke_test) solves a known
/// two-system batch off the raw bindings before any gridoxide logic is
/// involved. Run it first on a new box; if it fails, check the generated
/// `OUT_DIR/cudss_bindings.rs` against the cuDSS install's own batched sample.
pub struct CudssBatchedSystem {
    batch_count: usize,
    blk: usize,
    nnz: usize,

    handle: cudssHandle_t,
    config: cudssConfig_t,
    data: cudssData_t,
    matrix_a: cudssMatrix_t,
    matrix_b: cudssMatrix_t,
    matrix_x: cudssMatrix_t,

    /// The one block's CSR structure, owned here and shared by every block.
    d_row_ptr: *mut c_void,
    d_col_idx: *mut c_void,

    // Host-side descriptor arrays cuDSS keeps reading — see "Ownership".
    nrows: Vec<i32>,
    ncols: Vec<i32>,
    nnz_per_block: Vec<i32>,
    ld: Vec<i32>,
    /// Column count for the dense right-hand-side/solution batch: one column
    /// per system. A separate member rather than an inline `vec![1; n]`
    /// because cuDSS re-reads these host arrays at execute time — a temporary
    /// would dangle, and the symptom would be wrong numbers, not a crash.
    rhs_ncols: Vec<i32>,
    row_start: Vec<*mut c_void>,
    col_indices: Vec<*mut c_void>,
    a_values: Vec<*mut c_void>,
    b_values: Vec<*mut c_void>,
    x_values: Vec<*mut c_void>,
}

impl CudssBatchedSystem {
    /// Binds cuDSS to already-live device buffers and runs analysis plus one
    /// numeric factorization.
    ///
    /// - `row_ptr`/`col_idx` describe **one block** (length `blk + 1` and
    ///   `nnz`), from [`build_csr_structure`] on a single
    ///   [`JacobianPattern`](crate::jacobian::JacobianPattern)'s pairs.
    /// - `values_ptr` is [`crate::gpu::GpuAssembler::values_ptr`]: `B * nnz`
    ///   f64s, scenario-major, in CSR order (the assembly kernel scatters
    ///   there directly via [`csr_scatter_map`]).
    /// - `rhs_ptr`/`x_ptr` are `B * blk` f64s, scenario-major, written and
    ///   read by [`crate::gpu::GpuBatch`]'s mismatch and update kernels.
    /// - `stream` is [`crate::gpu::Stream::as_u64`]. It is bound *before* the
    ///   analysis phase deliberately: that makes even this constructor's own
    ///   factorization stream-ordered after the caller's first assembly
    ///   kernel, so the precondition below needs no `cudaDeviceSynchronize`.
    ///
    /// **Precondition**: `values_ptr` must already hold valid numbers.
    /// cuDSS's factorization phase reads actual values to choose pivots, not
    /// just structure, so the caller must have launched one assembly pass
    /// first.
    pub fn new(
        batch_count: usize,
        blk: usize,
        row_ptr: &[i32],
        col_idx: &[i32],
        values_ptr: u64,
        rhs_ptr: u64,
        x_ptr: u64,
        stream: u64,
    ) -> Option<Self> {
        assert_eq!(row_ptr.len(), blk + 1, "row_ptr must describe exactly one block");
        let nnz = col_idx.len();

        let d_row_ptr = unsafe { device_alloc((blk + 1) * size_of::<i32>()) }?;
        let d_col_idx = unsafe { device_alloc(nnz * size_of::<i32>()) }?;
        unsafe {
            upload(d_row_ptr, row_ptr)?;
            upload(d_col_idx, col_idx)?;
        }

        let as_ptrs = |v: Vec<u64>| -> Vec<*mut c_void> { v.into_iter().map(|p| p as usize as *mut c_void).collect() };
        let mut sys = Self {
            batch_count,
            blk,
            nnz,
            handle: ptr::null_mut(),
            config: ptr::null_mut(),
            data: ptr::null_mut(),
            matrix_a: ptr::null_mut(),
            matrix_b: ptr::null_mut(),
            matrix_x: ptr::null_mut(),
            d_row_ptr,
            d_col_idx,
            nrows: vec![blk as i32; batch_count],
            ncols: vec![blk as i32; batch_count],
            nnz_per_block: vec![nnz as i32; batch_count],
            ld: vec![blk as i32; batch_count],
            rhs_ncols: vec![1i32; batch_count],
            // Uniform batch: one shared structure, `batch_count` aliases of it.
            row_start: as_ptrs(crate::device_layout::repeat_device_ptr(d_row_ptr as usize as u64, batch_count)),
            col_indices: as_ptrs(crate::device_layout::repeat_device_ptr(d_col_idx as usize as u64, batch_count)),
            a_values: as_ptrs(crate::device_layout::strided_device_ptrs(
                values_ptr,
                nnz * size_of::<f64>(),
                batch_count,
            )),
            b_values: as_ptrs(crate::device_layout::strided_device_ptrs(
                rhs_ptr,
                blk * size_of::<f64>(),
                batch_count,
            )),
            x_values: as_ptrs(crate::device_layout::strided_device_ptrs(
                x_ptr,
                blk * size_of::<f64>(),
                batch_count,
            )),
        };

        cudss_check(unsafe { cudssCreate(&mut sys.handle) }, "cudssCreate")?;
        sys.set_stream(stream)?;
        cudss_check(unsafe { cudssConfigCreate(&mut sys.config) }, "cudssConfigCreate")?;
        cudss_check(unsafe { cudssDataCreate(sys.handle, &mut sys.data) }, "cudssDataCreate")?;

        // Deliberately *not* setting CUDSS_CONFIG_DETERMINISTIC_MODE here, in
        // contrast to `CudssRealSystem::alloc`. It was enabled there
        // defensively while chasing the stacked path's iteration-count drift
        // and measurably did not help; on this path it is one of the config
        // knobs `scripts/GPU_RUNBOOK.md`'s sweep step measures rather than
        // assumes. Turn it on via `set_deterministic` if the sweep says it is
        // free.
        cudss_check(
            unsafe {
                cudssMatrixCreateBatchCsr(
                    &mut sys.matrix_a,
                    batch_count as i64,
                    sys.nrows.as_mut_ptr() as *mut c_void,
                    sys.ncols.as_mut_ptr() as *mut c_void,
                    sys.nnz_per_block.as_mut_ptr() as *mut c_void,
                    sys.row_start.as_mut_ptr(),
                    // `rowEnd = NULL` selects the ordinary 3-array CSR form,
                    // where row `r` ends where row `r + 1` starts.
                    ptr::null_mut(),
                    sys.col_indices.as_mut_ptr(),
                    sys.a_values.as_mut_ptr(),
                    CUDSS_R_32I,
                    CUDSS_R_64F,
                    CUDSS_MTYPE_GENERAL,
                    CUDSS_MVIEW_FULL,
                    CUDSS_BASE_ZERO,
                )
            },
            "cudssMatrixCreateBatchCsr",
        )?;

        cudss_check(
            unsafe {
                cudssMatrixCreateBatchDn(
                    &mut sys.matrix_b,
                    batch_count as i64,
                    sys.nrows.as_mut_ptr() as *mut c_void,
                    // One right-hand side per system.
                    sys.rhs_ncols.as_mut_ptr() as *mut c_void,
                    sys.ld.as_mut_ptr() as *mut c_void,
                    sys.b_values.as_mut_ptr(),
                    CUDSS_R_32I,
                    CUDSS_R_64F,
                    CUDSS_LAYOUT_COL_MAJOR,
                )
            },
            "cudssMatrixCreateBatchDn for b",
        )?;
        cudss_check(
            unsafe {
                cudssMatrixCreateBatchDn(
                    &mut sys.matrix_x,
                    batch_count as i64,
                    sys.nrows.as_mut_ptr() as *mut c_void,
                    sys.rhs_ncols.as_mut_ptr() as *mut c_void,
                    sys.ld.as_mut_ptr() as *mut c_void,
                    sys.x_values.as_mut_ptr(),
                    CUDSS_R_32I,
                    CUDSS_R_64F,
                    CUDSS_LAYOUT_COL_MAJOR,
                )
            },
            "cudssMatrixCreateBatchDn for x",
        )?;

        cudss_check(sys.execute(PHASE_ANALYSIS), "batched analysis")?;
        cudss_check(sys.execute(PHASE_FACTORIZATION), "batched initial factorization")?;

        Some(sys)
    }

    /// Runs cuDSS's work on `stream` instead of the default stream.
    ///
    /// This is what lets [`crate::gpu::GpuBatch`]'s kernels and the solve
    /// share one stream, so "assemble, then factorize" is ordered by the
    /// stream rather than by a `cudaDeviceSynchronize` that stalls the whole
    /// device once per Newton iteration. Pass
    /// [`crate::gpu::Stream::as_u64`].
    pub fn set_stream(&mut self, stream: u64) -> Option<()> {
        cudss_check(
            unsafe { cudssSetStream(self.handle, stream as usize as cudaStream_t) },
            "cudssSetStream",
        )
    }

    /// Enables cuDSS's deterministic mode. Off by default here — see
    /// [`new`](Self::new).
    pub fn set_deterministic(&mut self, on: bool) -> Option<()> {
        let flag: i32 = on as i32;
        cudss_check(
            unsafe {
                cudssConfigSet(
                    self.config,
                    CUDSS_CONFIG_DETERMINISTIC_MODE,
                    &flag as *const i32 as *const c_void,
                    size_of::<i32>(),
                )
            },
            "cudssConfigSet(DETERMINISTIC_MODE)",
        )
    }

    /// Numeric refactorization against the cached analysis, then solve — for
    /// every system in the batch, entirely on-device.
    ///
    /// Enqueued on this system's stream and **not** synchronized: the caller's
    /// next kernel (`go_apply_update`) is on the same stream and therefore
    /// ordered after it, and the loop's one synchronization point is the
    /// convergence-norm copy. Nothing crosses the host boundary here at all,
    /// which is why there is nothing to return but success.
    pub fn refactor_and_solve(&mut self) -> Option<()> {
        // The values buffer is rewritten in place by the assembly kernel;
        // this is the documented way to tell cuDSS its contents changed. The
        // pointers themselves never move, so it is cheap.
        cudss_check(
            unsafe { cudssMatrixSetBatchValues(self.matrix_a, self.a_values.as_mut_ptr()) },
            "cudssMatrixSetBatchValues",
        )?;
        cudss_check(self.execute(PHASE_REFACTORIZATION), "batched refactorization")?;
        cudss_check(self.execute(PHASE_SOLVE), "batched solve")
    }

    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    pub fn block_size(&self) -> usize {
        self.blk
    }

    pub fn nnz_per_block(&self) -> usize {
        self.nnz
    }

    fn execute(&mut self, phase: u32) -> cudssStatus_t {
        unsafe {
            cudssExecute(self.handle, phase as i32, self.config, self.data, self.matrix_a, self.matrix_x, self.matrix_b)
        }
    }
}

impl Drop for CudssBatchedSystem {
    fn drop(&mut self) {
        unsafe {
            cudssMatrixDestroy(self.matrix_a);
            cudssMatrixDestroy(self.matrix_b);
            cudssMatrixDestroy(self.matrix_x);
            cudssDataDestroy(self.handle, self.data);
            cudssConfigDestroy(self.config);
            cudssDestroy(self.handle);
            cudaFree(self.d_row_ptr);
            cudaFree(self.d_col_idx);
        }
        // The values/rhs/x device buffers are owned by `gpu::GpuBatch`, which
        // outlives this system — nothing to free for them here.
    }
}

impl crate::solver::LinearSolver for CudssRealSystem {
    fn new(n: usize, entries: &[(usize, usize, f64)]) -> Option<Self> {
        CudssRealSystem::new(n, entries)
    }

    fn factor_and_solve(&mut self, entries: &[(usize, usize, f64)], rhs: &[f64]) -> Option<Vec<f64>> {
        CudssRealSystem::factor_and_solve(self, entries, rhs)
    }

    fn factor_and_solve_values(&mut self, values: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
        let packed = pack_values_slice(values, &self.groups);
        self.solve_packed(&packed, rhs)
    }
}

impl Drop for CudssRealSystem {
    fn drop(&mut self) {
        unsafe {
            cudssMatrixDestroy(self.matrix_a);
            cudssMatrixDestroy(self.matrix_b);
            cudssMatrixDestroy(self.matrix_x);
            cudssDataDestroy(self.handle, self.data);
            cudssConfigDestroy(self.config);
            cudssDestroy(self.handle);
            cudaFree(self.d_row_ptr);
            cudaFree(self.d_col_idx);
            if self.owns_values {
                cudaFree(self.d_values);
            }
            cudaFree(self.d_rhs);
            cudaFree(self.d_x);
        }
    }
}

#[cfg(test)]
mod ffi_smoke_test {
    use super::bindings::*;
    use std::ffi::c_void;
    use std::ptr;

    /// Solves a tiny fixed unsymmetric system directly off the raw FFI
    /// bindings, with no `CudssRealSystem` wrapper involved — proves
    /// `build.rs`'s cuDSS discovery/linking and the `cudss.h`-derived
    /// bindgen output are ABI-correct independently of any gridoxide-
    /// specific wrapper logic. Mirrors `sparse_pardiso.rs::ffi_smoke_test`'s
    /// role for the `Pardiso` backend, and the same fixture: `[[2, 1], [1,
    /// 3]] * [x0, x1] = [5, 10]` => `x0=1, x1=3`, in 0-based CSR: `row_ptr =
    /// [0, 2, 4]`, `col_idx = [0, 1, 0, 1]`, `values = [2, 1, 1, 3]`.
    #[test]
    fn solves_simple_system_via_raw_ffi() {
        let n: i64 = 2;
        let row_ptr: [i32; 3] = [0, 2, 4];
        let col_idx: [i32; 4] = [0, 1, 0, 1];
        let values: [f64; 4] = [2.0, 1.0, 1.0, 3.0];
        let rhs: [f64; 2] = [5.0, 10.0];

        unsafe {
            let mut d_row_ptr: *mut c_void = ptr::null_mut();
            let mut d_col_idx: *mut c_void = ptr::null_mut();
            let mut d_values: *mut c_void = ptr::null_mut();
            let mut d_rhs: *mut c_void = ptr::null_mut();
            let mut d_x: *mut c_void = ptr::null_mut();
            assert_eq!(cudaMalloc(&mut d_row_ptr, 3 * size_of::<i32>()), cudaError_cudaSuccess);
            assert_eq!(cudaMalloc(&mut d_col_idx, 4 * size_of::<i32>()), cudaError_cudaSuccess);
            assert_eq!(cudaMalloc(&mut d_values, 4 * size_of::<f64>()), cudaError_cudaSuccess);
            assert_eq!(cudaMalloc(&mut d_rhs, 2 * size_of::<f64>()), cudaError_cudaSuccess);
            assert_eq!(cudaMalloc(&mut d_x, 2 * size_of::<f64>()), cudaError_cudaSuccess);

            assert_eq!(
                cudaMemcpy(d_row_ptr, row_ptr.as_ptr() as *const c_void, 3 * size_of::<i32>(), cudaMemcpyKind_cudaMemcpyHostToDevice),
                cudaError_cudaSuccess
            );
            assert_eq!(
                cudaMemcpy(d_col_idx, col_idx.as_ptr() as *const c_void, 4 * size_of::<i32>(), cudaMemcpyKind_cudaMemcpyHostToDevice),
                cudaError_cudaSuccess
            );
            assert_eq!(
                cudaMemcpy(d_values, values.as_ptr() as *const c_void, 4 * size_of::<f64>(), cudaMemcpyKind_cudaMemcpyHostToDevice),
                cudaError_cudaSuccess
            );
            assert_eq!(
                cudaMemcpy(d_rhs, rhs.as_ptr() as *const c_void, 2 * size_of::<f64>(), cudaMemcpyKind_cudaMemcpyHostToDevice),
                cudaError_cudaSuccess
            );

            let mut handle: cudssHandle_t = ptr::null_mut();
            assert_eq!(cudssCreate(&mut handle), cudssStatus_t_CUDSS_STATUS_SUCCESS);
            let mut config: cudssConfig_t = ptr::null_mut();
            assert_eq!(cudssConfigCreate(&mut config), cudssStatus_t_CUDSS_STATUS_SUCCESS);
            let mut data: cudssData_t = ptr::null_mut();
            assert_eq!(cudssDataCreate(handle, &mut data), cudssStatus_t_CUDSS_STATUS_SUCCESS);

            let mut a: cudssMatrix_t = ptr::null_mut();
            assert_eq!(
                cudssMatrixCreateCsr(
                    &mut a, n, n, 4, d_row_ptr, ptr::null(), d_col_idx, d_values,
                    cudssDataType_t_CUDSS_R_32I, cudssDataType_t_CUDSS_R_32I, cudssDataType_t_CUDSS_R_64F,
                    cudssMatrixType_t_CUDSS_MTYPE_GENERAL, cudssMatrixViewType_t_CUDSS_MVIEW_FULL,
                    cudssIndexBase_t_CUDSS_BASE_ZERO,
                ),
                cudssStatus_t_CUDSS_STATUS_SUCCESS
            );
            let mut b: cudssMatrix_t = ptr::null_mut();
            assert_eq!(
                cudssMatrixCreateDn(&mut b, n, 1, n, d_rhs, cudssDataType_t_CUDSS_R_64F, cudssLayout_t_CUDSS_LAYOUT_COL_MAJOR),
                cudssStatus_t_CUDSS_STATUS_SUCCESS
            );
            let mut x: cudssMatrix_t = ptr::null_mut();
            assert_eq!(
                cudssMatrixCreateDn(&mut x, n, 1, n, d_x, cudssDataType_t_CUDSS_R_64F, cudssLayout_t_CUDSS_LAYOUT_COL_MAJOR),
                cudssStatus_t_CUDSS_STATUS_SUCCESS
            );

            for phase in [
                cudssPhase_t_CUDSS_PHASE_ANALYSIS,
                cudssPhase_t_CUDSS_PHASE_FACTORIZATION,
                cudssPhase_t_CUDSS_PHASE_SOLVE,
            ] {
                assert_eq!(cudssExecute(handle, phase as i32, config, data, a, x, b), cudssStatus_t_CUDSS_STATUS_SUCCESS);
            }

            let mut x_host = [0.0f64; 2];
            assert_eq!(
                cudaMemcpy(x_host.as_mut_ptr() as *mut c_void, d_x, 2 * size_of::<f64>(), cudaMemcpyKind_cudaMemcpyDeviceToHost),
                cudaError_cudaSuccess
            );

            cudssMatrixDestroy(a);
            cudssMatrixDestroy(b);
            cudssMatrixDestroy(x);
            cudssDataDestroy(handle, data);
            cudssConfigDestroy(config);
            cudssDestroy(handle);
            cudaFree(d_row_ptr);
            cudaFree(d_col_idx);
            cudaFree(d_values);
            cudaFree(d_rhs);
            cudaFree(d_x);

            assert!((x_host[0] - 1.0).abs() < 1e-10);
            assert!((x_host[1] - 3.0).abs() < 1e-10);
        }
    }
}

/// Solves a **two-system batch** off the raw FFI bindings, with no gridoxide
/// wrapper involved.
///
/// Run this first on any new box. `cudssMatrixCreateBatchCsr` and
/// `cudssMatrixCreateBatchDn` take most of their arguments as `void*`/`void**`,
/// so a mistake in parameter order compiles cleanly and produces silent
/// garbage rather than a type error — this is the check that turns that class
/// of bug into an immediate, obvious failure. It also confirms the installed
/// cuDSS actually has the batch API (added in 0.4.0): a missing symbol fails
/// at link time here rather than three hours into a session.
///
/// Fixture: two independent 2x2 systems with different values, so a wrapper
/// that silently solved system 0 twice would fail.
///   [[2,1],[1,3]] x = [5,10]  => x = [1, 3]
///   [[4,1],[1,2]] x = [6, 5]  => x = [1, 2]
/// Both share one CSR structure — row_ptr = [0,2,4], col_idx = [0,1,0,1] —
/// which is the uniform-batch property gridoxide's block-diagonal embedding
/// guarantees, and here every block points at the *same* structure buffers.
#[cfg(test)]
mod batched_ffi_smoke_test {
    use super::bindings::*;
    use std::ffi::c_void;
    use std::ptr;

    #[test]
    fn solves_a_two_system_batch_via_raw_ffi() {
        const B: usize = 2;
        let row_ptr: [i32; 3] = [0, 2, 4];
        let col_idx: [i32; 4] = [0, 1, 0, 1];
        let values: [f64; 8] = [2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0, 2.0];
        let rhs: [f64; 4] = [5.0, 10.0, 6.0, 5.0];

        unsafe {
            let mut d_row_ptr: *mut c_void = ptr::null_mut();
            let mut d_col_idx: *mut c_void = ptr::null_mut();
            let mut d_values: *mut c_void = ptr::null_mut();
            let mut d_rhs: *mut c_void = ptr::null_mut();
            let mut d_x: *mut c_void = ptr::null_mut();
            assert_eq!(cudaMalloc(&mut d_row_ptr, 3 * size_of::<i32>()), cudaError_cudaSuccess);
            assert_eq!(cudaMalloc(&mut d_col_idx, 4 * size_of::<i32>()), cudaError_cudaSuccess);
            assert_eq!(cudaMalloc(&mut d_values, 8 * size_of::<f64>()), cudaError_cudaSuccess);
            assert_eq!(cudaMalloc(&mut d_rhs, 4 * size_of::<f64>()), cudaError_cudaSuccess);
            assert_eq!(cudaMalloc(&mut d_x, 4 * size_of::<f64>()), cudaError_cudaSuccess);

            let h2d = cudaMemcpyKind_cudaMemcpyHostToDevice;
            assert_eq!(cudaMemcpy(d_row_ptr, row_ptr.as_ptr() as *const c_void, 3 * size_of::<i32>(), h2d), cudaError_cudaSuccess);
            assert_eq!(cudaMemcpy(d_col_idx, col_idx.as_ptr() as *const c_void, 4 * size_of::<i32>(), h2d), cudaError_cudaSuccess);
            assert_eq!(cudaMemcpy(d_values, values.as_ptr() as *const c_void, 8 * size_of::<f64>(), h2d), cudaError_cudaSuccess);
            assert_eq!(cudaMemcpy(d_rhs, rhs.as_ptr() as *const c_void, 4 * size_of::<f64>(), h2d), cudaError_cudaSuccess);

            // Uniform batch descriptors. Every block shares one structure;
            // only the values and vectors stride.
            let mut nrows: [i32; B] = [2, 2];
            let mut ncols: [i32; B] = [2, 2];
            let mut nnz: [i32; B] = [4, 4];
            let mut ld: [i32; B] = [2, 2];
            let mut rhs_ncols: [i32; B] = [1, 1];
            let mut row_start: [*mut c_void; B] = [d_row_ptr, d_row_ptr];
            let mut col_indices: [*mut c_void; B] = [d_col_idx, d_col_idx];
            let mut a_values: [*mut c_void; B] =
                [d_values, (d_values as *mut u8).add(4 * size_of::<f64>()) as *mut c_void];
            let mut b_values: [*mut c_void; B] =
                [d_rhs, (d_rhs as *mut u8).add(2 * size_of::<f64>()) as *mut c_void];
            let mut x_values: [*mut c_void; B] =
                [d_x, (d_x as *mut u8).add(2 * size_of::<f64>()) as *mut c_void];

            let mut handle: cudssHandle_t = ptr::null_mut();
            assert_eq!(cudssCreate(&mut handle), cudssStatus_t_CUDSS_STATUS_SUCCESS);
            let mut config: cudssConfig_t = ptr::null_mut();
            assert_eq!(cudssConfigCreate(&mut config), cudssStatus_t_CUDSS_STATUS_SUCCESS);
            let mut data: cudssData_t = ptr::null_mut();
            assert_eq!(cudssDataCreate(handle, &mut data), cudssStatus_t_CUDSS_STATUS_SUCCESS);

            let mut a: cudssMatrix_t = ptr::null_mut();
            assert_eq!(
                cudssMatrixCreateBatchCsr(
                    &mut a,
                    B as i64,
                    nrows.as_mut_ptr() as *mut c_void,
                    ncols.as_mut_ptr() as *mut c_void,
                    nnz.as_mut_ptr() as *mut c_void,
                    row_start.as_mut_ptr(),
                    ptr::null_mut(),
                    col_indices.as_mut_ptr(),
                    a_values.as_mut_ptr(),
                    cudssDataType_t_CUDSS_R_32I,
                    cudssDataType_t_CUDSS_R_64F,
                    cudssMatrixType_t_CUDSS_MTYPE_GENERAL,
                    cudssMatrixViewType_t_CUDSS_MVIEW_FULL,
                    cudssIndexBase_t_CUDSS_BASE_ZERO,
                ),
                cudssStatus_t_CUDSS_STATUS_SUCCESS
            );
            let mut b: cudssMatrix_t = ptr::null_mut();
            assert_eq!(
                cudssMatrixCreateBatchDn(
                    &mut b,
                    B as i64,
                    nrows.as_mut_ptr() as *mut c_void,
                    rhs_ncols.as_mut_ptr() as *mut c_void,
                    ld.as_mut_ptr() as *mut c_void,
                    b_values.as_mut_ptr(),
                    cudssDataType_t_CUDSS_R_32I,
                    cudssDataType_t_CUDSS_R_64F,
                    cudssLayout_t_CUDSS_LAYOUT_COL_MAJOR,
                ),
                cudssStatus_t_CUDSS_STATUS_SUCCESS
            );
            let mut x: cudssMatrix_t = ptr::null_mut();
            assert_eq!(
                cudssMatrixCreateBatchDn(
                    &mut x,
                    B as i64,
                    nrows.as_mut_ptr() as *mut c_void,
                    rhs_ncols.as_mut_ptr() as *mut c_void,
                    ld.as_mut_ptr() as *mut c_void,
                    x_values.as_mut_ptr(),
                    cudssDataType_t_CUDSS_R_32I,
                    cudssDataType_t_CUDSS_R_64F,
                    cudssLayout_t_CUDSS_LAYOUT_COL_MAJOR,
                ),
                cudssStatus_t_CUDSS_STATUS_SUCCESS
            );

            for phase in [
                cudssPhase_t_CUDSS_PHASE_ANALYSIS,
                cudssPhase_t_CUDSS_PHASE_FACTORIZATION,
                cudssPhase_t_CUDSS_PHASE_SOLVE,
            ] {
                assert_eq!(
                    cudssExecute(handle, phase as i32, config, data, a, x, b),
                    cudssStatus_t_CUDSS_STATUS_SUCCESS,
                    "batched phase {phase} failed"
                );
            }

            let mut x_host = [0.0f64; 4];
            assert_eq!(
                cudaMemcpy(
                    x_host.as_mut_ptr() as *mut c_void,
                    d_x,
                    4 * size_of::<f64>(),
                    cudaMemcpyKind_cudaMemcpyDeviceToHost
                ),
                cudaError_cudaSuccess
            );

            cudssMatrixDestroy(a);
            cudssMatrixDestroy(b);
            cudssMatrixDestroy(x);
            cudssDataDestroy(handle, data);
            cudssConfigDestroy(config);
            cudssDestroy(handle);
            cudaFree(d_row_ptr);
            cudaFree(d_col_idx);
            cudaFree(d_values);
            cudaFree(d_rhs);
            cudaFree(d_x);

            // Distinct answers per system: a wrapper that aliased the batch
            // would give [1, 3, 1, 3] or [1, 2, 1, 2] here.
            let want = [1.0, 3.0, 1.0, 2.0];
            for (i, (&got, &w)) in x_host.iter().zip(&want).enumerate() {
                assert!((got - w).abs() < 1e-9, "batched solution[{i}] = {got}, want {w}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_simple_system() {
        // [[2, 1], [1, 3]] * [x0, x1] = [5, 10]  =>  x0=1, x1=3
        let entries = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let mut sys = CudssRealSystem::new(2, &entries).unwrap();
        let x = sys.factor_and_solve(&entries, &[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn duplicate_entries_sum() {
        // (0,0) contributed twice as 1.0+1.0=2.0, matching dense += semantics.
        let entries = vec![
            (0, 0, 1.0),
            (0, 0, 1.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
        ];
        let mut sys = CudssRealSystem::new(2, &entries).unwrap();
        let x = sys.factor_and_solve(&entries, &[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn refactor_reuses_symbolic() {
        let entries_a = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let mut sys = CudssRealSystem::new(2, &entries_a).unwrap();
        let x = sys.factor_and_solve(&entries_a, &[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 3.0).abs() < 1e-9);

        // Same sparsity pattern, different numeric values (as across NR iterations).
        let entries_b = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)];
        // [[4,1],[1,2]] * x = [6, 5] => x0=1, x1=2
        let x2 = sys.factor_and_solve(&entries_b, &[6.0, 5.0]).unwrap();
        assert!((x2[0] - 1.0).abs() < 1e-9);
        assert!((x2[1] - 2.0).abs() < 1e-9);
    }
}
