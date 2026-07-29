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
    cudaError_t, cudaMalloc, cudaFree, cudaError_cudaSuccess as cudaSuccess,
    cudssConfig_t, cudssConfigCreate, cudssConfigDestroy,
    cudssCreate, cudssData_t, cudssDataCreate, cudssDataDestroy, cudssDestroy,
    cudssExecute, cudssHandle_t, cudssMatrixCreateCsr, cudssMatrixCreateDn, cudssMatrixDestroy,
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
/// entry indices contributing to the `k`-th CSR position.
fn build_csr_structure(n: usize, pairs: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>, Vec<Vec<usize>>) {
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

fn pack_values(entries: &[(usize, usize, f64)], groups: &[Vec<usize>]) -> Vec<f64> {
    groups.iter().map(|g| g.iter().map(|&i| entries[i].2).sum()).collect()
}

/// `pack_values` for callers that already hold just the values — see
/// `solver::LinearSolver::factor_and_solve_values`.
fn pack_values_slice(values: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    groups.iter().map(|g| g.iter().map(|&i| values[i]).sum()).collect()
}

fn cuda_check(err: cudaError_t, what: &str) -> Option<()> {
    if err != cudaSuccess {
        eprintln!("cudss: CUDA call failed ({what}): error code {err}");
        return None;
    }
    Some(())
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

        let mut sys = unsafe { Self::alloc(n, nnz, &row_ptr, &col_idx, &values) }?;

        cudss_check(sys.execute(PHASE_ANALYSIS), "analysis")?;
        cudss_check(sys.execute(PHASE_FACTORIZATION), "initial factorization")?;

        sys.groups = groups;
        Some(sys)
    }

    /// Allocates device buffers, uploads the fixed CSR structure and an
    /// initial value set, and wraps everything in cuDSS's matrix/handle
    /// objects. Split out from `new` so every fallible step before the
    /// analysis/factorization phases happens behind one `?`-chain, with the
    /// resulting `Self` (and its `Drop`) responsible for cleanup from that
    /// point on regardless of where `new` bails out afterward.
    unsafe fn alloc(n: usize, nnz: usize, row_ptr: &[i32], col_idx: &[i32], values: &[f64]) -> Option<Self> {
        let d_row_ptr = unsafe { device_alloc((n + 1) * size_of::<i32>()) }?;
        let d_col_idx = unsafe { device_alloc(nnz * size_of::<i32>()) }?;
        let d_values = unsafe { device_alloc(nnz * size_of::<f64>()) }?;
        let d_rhs = unsafe { device_alloc(n * size_of::<f64>()) }?;
        let d_x = unsafe { device_alloc(n * size_of::<f64>()) }?;

        unsafe {
            upload(d_row_ptr, row_ptr)?;
            upload(d_col_idx, col_idx)?;
            upload(d_values, values)?;
        }

        let mut handle: cudssHandle_t = ptr::null_mut();
        cudss_check(unsafe { cudssCreate(&mut handle) }, "cudssCreate")?;
        let mut config: cudssConfig_t = ptr::null_mut();
        cudss_check(unsafe { cudssConfigCreate(&mut config) }, "cudssConfigCreate")?;
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
            cudaFree(self.d_values);
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
