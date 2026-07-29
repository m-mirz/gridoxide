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
//! **This first implementation re-uploads values and the right-hand side
//! every call** (`factor_and_solve_values` below) — correct, and sufficient
//! to validate the block-diagonal-embedding path end to end
//! (`scripts/GPU_RUNBOOK.md` Phase 3), but not yet the fully device-resident
//! loop `plans/GPU_PLAN.md` §6 Phase 3 describes as the actual payoff. A
//! per-iteration host round trip is exactly the thing that must go away
//! before any GPU speedup claim means anything — see that phase's exit
//! criterion.
//!
//! `CUDSS_MTYPE_GENERAL`/`CUDSS_MVIEW_FULL` select "general, non-symmetric"
//! storage — the same shape KLU and PARDISO are given (gridoxide's Newton
//! Jacobian is never symmetric).

use std::ffi::c_void;
use std::ptr;

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/cudss_bindings.rs"));
}
use bindings::{
    cudaMemcpy, cudaMemcpyKind_cudaMemcpyDeviceToHost as D2H, cudaMemcpyKind_cudaMemcpyHostToDevice as H2D,
    cudaError_t, cudaMalloc, cudaFree, cudaDeviceSynchronize, cudaError_cudaSuccess as cudaSuccess,
    cudssConfig_t, cudssConfigCreate, cudssConfigDestroy, cudssConfigSet,
    cudssConfigParam_t_CUDSS_CONFIG_DETERMINISTIC_MODE as CUDSS_CONFIG_DETERMINISTIC_MODE,
    cudssCreate, cudssData_t, cudssDataCreate, cudssDataDestroy, cudssDestroy, cudssSetStream,
    cudssExecute, cudssHandle_t, cudssMatrixCreateCsr, cudssMatrixCreateDn, cudssMatrixDestroy, cudssMatrixSetValues,
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

/// Builds a cuDSS-ready CSR structure (row pointers + sorted column indices)
/// from a set of `(row, col)` index pairs, merging duplicates — identical
/// accumulation semantics to `sparse_pardiso::build_csr_structure` (row-major,
/// same duplicate-summing convention `sparse_klu`'s CSC builder uses).
/// Returns `(row_ptr, col_idx, groups)`, where `groups[k]` lists the original
/// entry indices contributing to the `k`-th CSR position. `pub(crate)`:
/// `bde::solve_batch_block_diagonal_device_resident` needs this too, to
/// build the stacked block-diagonal CSR structure cuDSS is given directly.
pub(crate) fn build_csr_structure(n: usize, pairs: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>, Vec<Vec<usize>>) {
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&i| (pairs[i].0, pairs[i].1));

    let mut row_ptr = vec![0i32; n + 1];
    let mut col_idx: Vec<i32> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();

    let mut idx = 0;
    while idx < order.len() {
        let (row, col) = pairs[order[idx]];
        let mut group = vec![order[idx]];
        let mut k = idx + 1;
        while k < order.len() && pairs[order[k]] == (row, col) {
            group.push(order[k]);
            k += 1;
        }
        col_idx.push(col as i32);
        groups.push(group);
        row_ptr[row + 1] += 1;
        idx = k;
    }
    for r in 0..n {
        row_ptr[r + 1] += row_ptr[r];
    }
    (row_ptr, col_idx, groups)
}

/// For a **single block's** `(row, col)` pairs in `JacobianPattern` entries
/// order, computes the CSR position each entry maps to — the inverse of
/// `build_csr_structure`'s grouping. This is what lets
/// `gpu::GpuAssembler`'s kernel write its per-entry outputs directly at the
/// CSR position cuDSS expects, in one pass, with no host-side reorder.
///
/// Only defined when every CSR position merges exactly one original entry —
/// true for every gridoxide Jacobian pattern, verified by
/// `csr_permutation_tests::groups_are_all_singletons` and true by
/// construction: `network::YBusSparse` never has a duplicate `(row, col)`
/// neighbor (parallel lines are already summed at Y-bus assembly), so
/// `JacobianPattern`, which walks each unknown row's Y-bus neighbors exactly
/// once, can't produce a duplicate `(row, col)` triplet either. Panics
/// otherwise — a real merge would need atomic adds in the GPU kernel this
/// scatter map feeds, not a plain permutation, and silently dropping a
/// contribution would be a much worse failure than a panic.
///
/// Because block-diagonal embedding gives every scenario's block the exact
/// same relative (row, col) structure, just offset by `s * block_size`
/// (`bde::BlockDiagonal::analyze`), sorting the *whole* stacked matrix's
/// pairs by `(row, col)` reproduces this same single-block permutation
/// inside each scenario's own contiguous `nnz`-sized segment — so one
/// single-block scatter map is all any batch size needs; the caller adds
/// `scenario * nnz` itself.
pub(crate) fn csr_scatter_map(n: usize, pairs: &[(usize, usize)]) -> Vec<u32> {
    let (_, _, groups) = build_csr_structure(n, pairs);
    let mut scatter = vec![0u32; pairs.len()];
    for (csr_pos, group) in groups.iter().enumerate() {
        assert_eq!(
            group.len(),
            1,
            "CSR position {csr_pos} merges {} entries; device-resident scatter assumes a pure permutation",
            group.len()
        );
        scatter[group[0]] = csr_pos as u32;
    }
    scatter
}

fn pack_values(entries: &[(usize, usize, f64)], groups: &[Vec<usize>]) -> Vec<f64> {
    groups.iter().map(|g| g.iter().map(|&i| entries[i].2).sum()).collect()
}

/// `pack_values` for callers that already hold just the values — see
/// `solver::LinearSolver::factor_and_solve_values`.
pub(crate) fn pack_values_slice(values: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    groups.iter().map(|g| g.iter().map(|&i| values[i]).sum()).collect()
}

fn cuda_check(err: cudaError_t, what: &str) -> Option<()> {
    if err != cudaSuccess {
        eprintln!("cudss: CUDA call failed ({what}): error code {err}");
        return None;
    }
    Some(())
}

/// Blocks until every previously enqueued CUDA operation (on **every**
/// stream in this process's context) has completed.
///
/// Needed by the device-resident path
/// (`bde::solve_batch_block_diagonal_device_resident`) specifically: a plain
/// `cudaMemcpy` (as used elsewhere in this file for `rhs`/`x`) only
/// guarantees ordering for buffers *it itself* touches, and CubeCL's CUDA
/// runtime may run its kernels on a stream this module never otherwise
/// synchronizes with. Without an explicit barrier here, cuDSS's
/// refactorization could start reading the Jacobian values buffer before
/// `gpu::GpuAssembler`'s kernel has finished writing it — not a crash, since
/// both sides are well-formed CUDA operations on valid memory, but a subtle
/// numeric race: the Newton step would occasionally read a partially-updated
/// Jacobian, perturbing that iteration's `Δx` enough to change the
/// iteration count without necessarily changing the converged answer. This
/// was caught exactly that way — correct final voltages, but iteration
/// counts that silently disagreed with an independent CPU solve.
pub fn device_synchronize() -> Option<()> {
    cuda_check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

/// Debug-only: reads `len` `f64`s directly from a raw device pointer via
/// `cudaMemcpy`, bypassing cuDSS entirely. Used to check whether a raw CUDA
/// read sees the same buffer content CubeCL's own readback does.
#[cfg(test)]
pub(crate) fn debug_read_f64(ptr: u64, len: usize) -> Option<Vec<f64>> {
    let mut out = vec![0.0f64; len];
    unsafe { download(&mut out, ptr as usize as *mut c_void)? };
    Some(out)
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

#[cfg(test)]
mod csr_permutation_tests {
    use super::*;
    use crate::jacobian::JacobianPattern;
    use crate::json::NetworkData;
    use crate::network::build_ybus;
    use std::fs;
    use std::path::PathBuf;

    /// `csr_scatter_map` (used by the device-resident GPU path,
    /// `bde::solve_batch_block_diagonal_device_resident`) assumes every CSR
    /// position merges exactly one original `JacobianPattern` entry — no
    /// summing, a pure permutation. This holds by construction (Y-bus itself
    /// never has a duplicate `(row, col)` neighbor, so `JacobianPattern`
    /// can't produce one either — see `csr_scatter_map`'s doc comment), and
    /// this pins that invariant down on the committed fixture so a future
    /// change to `JacobianPattern`'s emission order/logic can't silently
    /// break the GPU scatter kernel.
    #[test]
    fn groups_are_all_singletons() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/network.json");
        let raw = fs::read_to_string(path).expect("read network.json");
        let network: NetworkData = serde_json::from_str(&raw).expect("parse network.json");
        let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();
        let pattern = JacobianPattern::analyze(&network.buses, &ybus);

        let pairs: Vec<(usize, usize)> =
            pattern.rows().iter().zip(pattern.cols()).map(|(&r, &c)| (r as usize, c as usize)).collect();
        let (_, _, groups) = build_csr_structure(pattern.n_unknowns, &pairs);
        assert_eq!(pairs.len(), groups.len(), "some CSR position merges >1 original entry");
    }
}

/// Pins down `csr_scatter_map`'s doc-comment claim numerically: one
/// single-block scatter map, offset by `scenario * nnz`, correctly locates
/// every scenario's entries in the *stacked* block-diagonal CSR structure —
/// not just structurally plausible, but checked position-by-position against
/// `build_csr_structure` run directly on the full stacked `(row, col)` pairs.
#[cfg(test)]
mod stacked_scatter_tests {
    use super::*;
    use crate::bde::BlockDiagonal;
    use crate::jacobian::JacobianPattern;
    use crate::json::NetworkData;
    use crate::network::build_ybus;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn stacked_scatter_matches_per_block_assumption() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data/network.json");
        let raw = fs::read_to_string(path).expect("read network.json");
        let network: NetworkData = serde_json::from_str(&raw).expect("parse network.json");
        let ybus = build_ybus(network.buses.len(), &network.lines, &[]).finish();

        let nb = 3usize;
        let bd = BlockDiagonal::analyze(&network.buses, &ybus, nb);
        let base = JacobianPattern::analyze(&network.buses, &ybus);
        let blk = base.n_unknowns;
        let nnz = base.len();

        let block_pairs: Vec<(usize, usize)> =
            base.rows().iter().zip(base.cols()).map(|(&r, &c)| (r as usize, c as usize)).collect();
        let scatter = csr_scatter_map(blk, &block_pairs);
        let (block_row_ptr, block_col_idx, _) = build_csr_structure(blk, &block_pairs);

        let dummy = vec![0.0f64; bd.len()];
        let full_pairs: Vec<(usize, usize)> = bd.to_triplets(&dummy).iter().map(|&(r, c, _)| (r, c)).collect();
        let (full_row_ptr, full_col_idx, full_groups) = build_csr_structure(bd.n_unknowns(), &full_pairs);

        assert_eq!(full_col_idx.len(), nnz * nb);
        assert!(full_groups.iter().all(|g| g.len() == 1));

        for s in 0..nb {
            for e in 0..nnz {
                let expected_col = block_col_idx[scatter[e] as usize] as usize + s * blk;
                let actual_col = full_col_idx[s * nnz + scatter[e] as usize] as usize;
                assert_eq!(expected_col, actual_col, "column mismatch at scenario {s}, entry {e}");
            }
            for r in 0..=blk {
                let expected = block_row_ptr[r] + (s * nnz) as i32;
                let actual = full_row_ptr[s * blk + r];
                assert_eq!(expected, actual, "row_ptr mismatch at scenario {s}, row {r}");
            }
        }
    }
}
