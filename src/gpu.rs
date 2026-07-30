//! CUDA batch state for gridoxide's power flow — the Rust half of
//! `cuda/gridoxide_kernels.cu`.
//!
//! This module replaced a CubeCL-based assembler at commit `ff92b66`. CubeCL
//! JIT-compiled one `#[cube]` function to CUDA, ROCm/HIP and WGSL from a
//! single Rust source, which is what `plans/GPU_PLAN.md` §5 built the AMD
//! story on; that portability was given up deliberately, for three things its
//! runtime did not expose and which the batched solver path needs:
//!
//! 1. **Streams.** Work now runs on one owned, non-blocking stream shared with
//!    cuDSS via `cudssSetStream`, so the "assembler must finish writing before
//!    the solver reads" dependency is enforced by stream ordering instead of
//!    the whole-device `cudaDeviceSynchronize` barrier the CubeCL path needed
//!    every Newton iteration.
//! 2. **Persistent allocations.** CubeCL allocated a fresh device buffer per
//!    input per launch; at case1354pegase/4096 that was ~178 MB of
//!    host-gathered upload and five `cudaMalloc`s *per iteration*. Every
//!    buffer here is allocated once and rewritten in place.
//! 3. **Raw device pointers**, stable across launches, for cuDSS's batched
//!    matrix to bind against once and never re-bind.
//!
//! # What lives where
//!
//! [`GpuAssembler`] owns the topology-static assembly recipe (the flat
//! per-entry arrays [`JacobianPattern`] already precomputes), the persistent
//! Jacobian values buffer cuDSS reads, and the per-scenario bus state
//! (`vm`/`va`/`p_calc`/`q_calc`). [`GpuBatch`] wraps it with everything the
//! *rest* of the Newton loop needs on-device — the Y-bus CSR, the ZIP
//! coefficients, the unknown index maps, the right-hand side and solution —
//! so that between iterations the only host↔device traffic is one f64 per
//! scenario down (the convergence norm) and one u32 per scenario up (the
//! active mask).
//!
//! All the host-side flattening these buffers are filled from lives in
//! [`crate::device_layout`], which is **not** feature-gated and is unit-tested
//! against the CPU implementations on machines with no GPU — see that module.
//!
//! # Precision
//!
//! f64 throughout. `plans/GPU_PLAN.md` §4.4 rejected f32/double-single
//! emulation for this workload; §3's mixed-precision idea (f32 solve with f64
//! refinement) is Phase 4 and is not implemented here.

use std::ffi::c_void;
use std::ptr;

use crate::device_layout::{FlatStates, UnknownMaps, YbusCsr, ZipCoeffs};
use crate::jacobian::{EntryKind, JacobianPattern};
use crate::network::YBusSparse;
use crate::types::Bus;

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/cuda_bindings.rs"));
}
use bindings::{
    cudaDeviceSynchronize, cudaError_cudaSuccess as cudaSuccess, cudaError_t, cudaEventCreate,
    cudaEventDestroy, cudaEventElapsedTime, cudaEventRecord, cudaEventSynchronize, cudaEvent_t, cudaFree,
    cudaMalloc, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind_cudaMemcpyDeviceToHost as D2H,
    cudaMemcpyKind_cudaMemcpyHostToDevice as H2D, cudaMemGetInfo, cudaStreamCreateWithFlags,
    cudaStreamDestroy, cudaStreamNonBlocking, cudaStreamSynchronize, cudaStream_t,
};

/// The kernel launchers from `cuda/gridoxide_kernels.cu`.
///
/// Each takes `void* stream` rather than `cudaStream_t` so this declaration
/// stays independent of whatever opaque pointer type `bindgen` produced for
/// the CUDA runtime, and returns `cudaGetLastError()` as an `i32` (0 ==
/// success) — a launch failure surfaces here rather than at the next
/// unrelated synchronization point.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" {
    fn go_assemble_jacobian(
        kinds: *const u32,
        bus_i: *const u32,
        bus_k: *const u32,
        y_re: *const f64,
        y_im: *const f64,
        identity_value: *const f64,
        scatter: *const u32,
        active: *const u32,
        vm: *const f64,
        va: *const f64,
        p_calc: *const f64,
        q_calc: *const f64,
        values: *mut f64,
        nnz: u32,
        n_buses: u32,
        n_scenarios: u32,
        stream: *mut c_void,
    ) -> i32;

    fn go_power_injections(
        row_ptr: *const i32,
        col_idx: *const i32,
        y_re: *const f64,
        y_im: *const f64,
        vm: *const f64,
        va: *const f64,
        p_calc: *mut f64,
        q_calc: *mut f64,
        n_buses: u32,
        n_scenarios: u32,
        stream: *mut c_void,
    ) -> i32;

    fn go_mismatch(
        non_slack: *const u32,
        pq: *const u32,
        p_spec: *const f64,
        q_spec: *const f64,
        zip_p_const: *const f64,
        zip_q_const: *const f64,
        zip_p_curr: *const f64,
        zip_q_curr: *const f64,
        zip_p_imp: *const f64,
        zip_q_imp: *const f64,
        vm: *const f64,
        p_calc: *const f64,
        q_calc: *const f64,
        rhs: *mut f64,
        max_mis: *mut f64,
        n_angle: u32,
        n_pq: u32,
        n_buses: u32,
        n_scenarios: u32,
        stream: *mut c_void,
    ) -> i32;

    fn go_zero_masked_rhs(
        active: *const u32,
        rhs: *mut f64,
        blk: u32,
        n_scenarios: u32,
        stream: *mut c_void,
    ) -> i32;

    fn go_apply_update(
        non_slack: *const u32,
        pq: *const u32,
        active: *const u32,
        dx: *const f64,
        vm: *mut f64,
        va: *mut f64,
        n_angle: u32,
        n_pq: u32,
        n_buses: u32,
        n_scenarios: u32,
        stream: *mut c_void,
    ) -> i32;
}

fn cuda_check(err: cudaError_t, what: &str) -> Option<()> {
    if err != cudaSuccess {
        eprintln!("gpu: CUDA call failed ({what}): error code {err}");
        return None;
    }
    Some(())
}

fn launch_check(err: i32, what: &str) -> Option<()> {
    if err != cudaSuccess as i32 {
        eprintln!("gpu: kernel launch failed ({what}): error code {err}");
        return None;
    }
    Some(())
}

/// Free and total device memory, in bytes — what a caller sizing a batch
/// against a card's capacity needs (`plans/GPU_PLAN.md` Phase 3's chunking
/// question: block-diagonal embedding needs O(batch) device memory, so a
/// year-long QSTS run has to be split into passes that fit).
pub fn device_memory() -> Option<(usize, usize)> {
    let mut free = 0usize;
    let mut total = 0usize;
    cuda_check(unsafe { cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo")?;
    Some((free, total))
}

/// Blocks until every previously enqueued CUDA operation on **every** stream
/// has completed.
///
/// Retained for tests and for the one-shot readback paths. The Newton loop
/// itself no longer needs it: [`GpuBatch`] puts every kernel and cuDSS itself
/// on one stream, so ordering is implied, and its only per-iteration
/// synchronization is [`Stream::synchronize`] after the convergence-norm copy.
pub fn device_synchronize() -> Option<()> {
    cuda_check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

/// An owned CUDA stream. Everything gridoxide enqueues — kernels *and* cuDSS's
/// factorization/solve, via `cudssSetStream` — runs on this one stream, which
/// is what makes the "assembler writes, then solver reads" dependency free
/// rather than a device-wide barrier.
pub struct Stream {
    raw: cudaStream_t,
}

impl Stream {
    pub fn new() -> Option<Self> {
        let mut raw: cudaStream_t = ptr::null_mut();
        cuda_check(
            unsafe { cudaStreamCreateWithFlags(&mut raw, cudaStreamNonBlocking) },
            "cudaStreamCreateWithFlags",
        )?;
        Some(Self { raw })
    }

    /// The raw stream as an integer, for handing to another FFI module
    /// (`sparse_cudss::CudssBatchedSystem::set_stream`) without either side
    /// needing the other's `bindgen`-generated pointer type.
    pub fn as_u64(&self) -> u64 {
        self.raw as usize as u64
    }

    fn as_ptr(&self) -> *mut c_void {
        self.raw as *mut c_void
    }

    pub fn synchronize(&self) -> Option<()> {
        cuda_check(unsafe { cudaStreamSynchronize(self.raw) }, "cudaStreamSynchronize")
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.raw) };
    }
}

/// A pair of CUDA events bracketing one phase, for the per-phase timing
/// `examples/bde_profile.rs` reports.
///
/// Events are used rather than host `Instant`s deliberately: timing a phase
/// with a host clock requires synchronizing after it, which serializes the
/// pipeline and changes the thing being measured. Events are recorded into
/// the stream and read once, at the iteration's existing synchronization
/// point.
pub struct PhaseTimer {
    start: cudaEvent_t,
    stop: cudaEvent_t,
}

impl PhaseTimer {
    pub fn new() -> Option<Self> {
        let mut start: cudaEvent_t = ptr::null_mut();
        let mut stop: cudaEvent_t = ptr::null_mut();
        cuda_check(unsafe { cudaEventCreate(&mut start) }, "cudaEventCreate")?;
        cuda_check(unsafe { cudaEventCreate(&mut stop) }, "cudaEventCreate")?;
        Some(Self { start, stop })
    }

    pub fn begin(&self, stream: &Stream) {
        unsafe { cudaEventRecord(self.start, stream.raw) };
    }

    pub fn end(&self, stream: &Stream) {
        unsafe { cudaEventRecord(self.stop, stream.raw) };
    }

    /// Milliseconds between the last `begin`/`end` pair. Synchronizes on the
    /// stop event, so call it after the iteration's own synchronization point.
    pub fn elapsed_ms(&self) -> Option<f32> {
        cuda_check(unsafe { cudaEventSynchronize(self.stop) }, "cudaEventSynchronize")?;
        let mut ms = 0.0f32;
        cuda_check(unsafe { cudaEventElapsedTime(&mut ms, self.start, self.stop) }, "cudaEventElapsedTime")?;
        Some(ms)
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        unsafe {
            cudaEventDestroy(self.start);
            cudaEventDestroy(self.stop);
        }
    }
}

/// An owned `cudaMalloc` allocation, freed on drop.
///
/// A zero-byte request keeps a null pointer rather than calling `cudaMalloc(0)`
/// (which succeeds but yields a pointer no kernel should dereference); the
/// kernels' own zero-work early-outs make that safe.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Option<Self> {
        if bytes == 0 {
            return Some(Self { ptr: ptr::null_mut(), bytes: 0 });
        }
        let mut ptr: *mut c_void = ptr::null_mut();
        cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) }, "cudaMalloc")?;
        Some(Self { ptr, bytes })
    }

    pub fn from_slice<T>(src: &[T]) -> Option<Self> {
        let mut buf = Self::new(std::mem::size_of_val(src))?;
        buf.upload(src)?;
        Some(buf)
    }

    pub fn upload<T>(&mut self, src: &[T]) -> Option<()> {
        let bytes = std::mem::size_of_val(src);
        if bytes == 0 {
            return Some(());
        }
        assert!(bytes <= self.bytes, "upload of {bytes} bytes into a {}-byte device buffer", self.bytes);
        cuda_check(
            unsafe { cudaMemcpy(self.ptr, src.as_ptr() as *const c_void, bytes, H2D) },
            "cudaMemcpy H2D",
        )
    }

    /// Stream-ordered upload. The source slice must stay alive until the copy
    /// completes; every caller here either synchronizes before returning or
    /// holds the slice for the rest of the iteration.
    pub fn upload_async<T>(&mut self, src: &[T], stream: &Stream) -> Option<()> {
        let bytes = std::mem::size_of_val(src);
        if bytes == 0 {
            return Some(());
        }
        assert!(bytes <= self.bytes, "upload of {bytes} bytes into a {}-byte device buffer", self.bytes);
        cuda_check(
            unsafe { cudaMemcpyAsync(self.ptr, src.as_ptr() as *const c_void, bytes, H2D, stream.raw) },
            "cudaMemcpyAsync H2D",
        )
    }

    pub fn download<T>(&self, dst: &mut [T]) -> Option<()> {
        let bytes = std::mem::size_of_val(dst);
        if bytes == 0 {
            return Some(());
        }
        assert!(bytes <= self.bytes, "download of {bytes} bytes from a {}-byte device buffer", self.bytes);
        cuda_check(
            unsafe { cudaMemcpy(dst.as_mut_ptr() as *mut c_void, self.ptr, bytes, D2H) },
            "cudaMemcpy D2H",
        )
    }

    pub fn download_async<T>(&self, dst: &mut [T], stream: &Stream) -> Option<()> {
        let bytes = std::mem::size_of_val(dst);
        if bytes == 0 {
            return Some(());
        }
        assert!(bytes <= self.bytes, "download of {bytes} bytes from a {}-byte device buffer", self.bytes);
        cuda_check(
            unsafe { cudaMemcpyAsync(dst.as_mut_ptr() as *mut c_void, self.ptr, bytes, D2H, stream.raw) },
            "cudaMemcpyAsync D2H",
        )
    }

    /// The raw device address. Stable for this buffer's lifetime — the
    /// property `sparse_cudss`'s batched matrix binds against.
    pub fn as_u64(&self) -> u64 {
        self.ptr as usize as u64
    }

    fn as_f64(&self) -> *mut f64 {
        self.ptr as *mut f64
    }

    fn as_u32(&self) -> *mut u32 {
        self.ptr as *mut u32
    }

    fn as_i32(&self) -> *mut i32 {
        self.ptr as *mut i32
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFree(self.ptr) };
        }
    }
}

/// Entry-kind discriminants as plain `u32`, for the kernel's eight-way
/// dispatch. Exhaustive over [`EntryKind`] on purpose: adding a variant there
/// fails to compile here rather than silently mis-assembling.
fn kind_code(k: EntryKind) -> u32 {
    match k {
        EntryKind::Hii => 0,
        EntryKind::Nii => 1,
        EntryKind::Hik => 2,
        EntryKind::Nik => 3,
        EntryKind::Mii => 4,
        EntryKind::Lii => 5,
        EntryKind::Mik => 6,
        EntryKind::Lik => 7,
    }
}

/// Device-side Jacobian assembly for one topology across a batch of scenarios.
///
/// The per-entry recipe arrays (`kinds`, `bus_i`, `bus_k`, `y_re`, `y_im`,
/// `identity_value`, `scatter`) depend only on the topology and are uploaded
/// once. The per-scenario buffers (`vm`, `va`, `p_calc`, `q_calc`, `active`)
/// and the Jacobian `values` output are allocated on the first
/// [`resize`](Self::resize) for a given batch size and then rewritten in
/// place, never reallocated — which is what makes [`values_ptr`](Self::values_ptr)
/// a pointer an external consumer can bind to once.
pub struct GpuAssembler {
    stream: Stream,

    // Topology-static.
    kinds: DeviceBuffer,
    bus_i: DeviceBuffer,
    bus_k: DeviceBuffer,
    y_re: DeviceBuffer,
    y_im: DeviceBuffer,
    identity_value: DeviceBuffer,
    scatter: DeviceBuffer,
    nnz: usize,
    n_buses: usize,

    // Per-batch, allocated by `resize`.
    n_scenarios: usize,
    vm: DeviceBuffer,
    va: DeviceBuffer,
    p_calc: DeviceBuffer,
    q_calc: DeviceBuffer,
    active: DeviceBuffer,
    values: DeviceBuffer,
}

impl GpuAssembler {
    /// Uploads a topology's per-entry recipe. `scatter` defaults to the
    /// identity permutation (output in [`JacobianPattern`] entries order);
    /// [`set_scatter`](Self::set_scatter) switches it to the CSR position an
    /// external sparse solver expects (`sparse_cudss::csr_scatter_map`).
    pub fn new(pattern: &JacobianPattern, n_buses: usize) -> Option<Self> {
        let entries = pattern.entries();
        let kinds: Vec<u32> = entries.iter().map(|e| kind_code(e.kind)).collect();
        let bus_i: Vec<u32> = entries.iter().map(|e| e.i).collect();
        let bus_k: Vec<u32> = entries.iter().map(|e| e.k).collect();
        let y_re: Vec<f64> = entries.iter().map(|e| e.y.re).collect();
        let y_im: Vec<f64> = entries.iter().map(|e| e.y.im).collect();
        // Mirrors `JacobianPattern::fill_identity_into`: 1.0 on the matrix
        // diagonal, 0.0 elsewhere — topology-only, so precomputed here rather
        // than branched on per launch.
        let identity_value: Vec<f64> =
            pattern.rows().iter().zip(pattern.cols()).map(|(&r, &c)| if r == c { 1.0 } else { 0.0 }).collect();
        let identity_scatter: Vec<u32> = (0..entries.len() as u32).collect();

        Some(Self {
            stream: Stream::new()?,
            kinds: DeviceBuffer::from_slice(&kinds)?,
            bus_i: DeviceBuffer::from_slice(&bus_i)?,
            bus_k: DeviceBuffer::from_slice(&bus_k)?,
            y_re: DeviceBuffer::from_slice(&y_re)?,
            y_im: DeviceBuffer::from_slice(&y_im)?,
            identity_value: DeviceBuffer::from_slice(&identity_value)?,
            scatter: DeviceBuffer::from_slice(&identity_scatter)?,
            nnz: entries.len(),
            n_buses,
            n_scenarios: 0,
            vm: DeviceBuffer::new(0)?,
            va: DeviceBuffer::new(0)?,
            p_calc: DeviceBuffer::new(0)?,
            q_calc: DeviceBuffer::new(0)?,
            active: DeviceBuffer::new(0)?,
            values: DeviceBuffer::new(0)?,
        })
    }

    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub fn n_buses(&self) -> usize {
        self.n_buses
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Switches each entry's output offset from the default identity (entries
    /// order, matching `JacobianPattern::fill_into`) to a caller-supplied
    /// permutation. `scatter[e]` is entry `e`'s offset *within one scenario's
    /// block*; the kernel adds `scenario * nnz` itself.
    pub fn set_scatter(&mut self, scatter: &[u32]) -> Option<()> {
        assert_eq!(scatter.len(), self.nnz, "scatter map must have one entry per (block-local) Jacobian entry");
        self.scatter = DeviceBuffer::from_slice(scatter)?;
        Some(())
    }

    /// Allocates (or reuses) this assembler's per-batch buffers. Idempotent
    /// for a repeated batch size, which is what keeps `values_ptr` stable.
    pub fn resize(&mut self, n_scenarios: usize) -> Option<()> {
        if self.n_scenarios == n_scenarios {
            return Some(());
        }
        let per_bus = n_scenarios * self.n_buses;
        self.vm = DeviceBuffer::new(per_bus * size_of::<f64>())?;
        self.va = DeviceBuffer::new(per_bus * size_of::<f64>())?;
        self.p_calc = DeviceBuffer::new(per_bus * size_of::<f64>())?;
        self.q_calc = DeviceBuffer::new(per_bus * size_of::<f64>())?;
        self.active = DeviceBuffer::new(n_scenarios * size_of::<u32>())?;
        self.values = DeviceBuffer::new(n_scenarios * self.nnz * size_of::<f64>())?;
        self.n_scenarios = n_scenarios;
        Some(())
    }

    /// The persistent Jacobian values buffer's device address — stable across
    /// launches of matching batch size. `None` before the first
    /// [`resize`](Self::resize).
    pub fn values_ptr(&self) -> Option<u64> {
        (self.n_scenarios > 0).then(|| self.values.as_u64())
    }

    pub fn vm_ptr(&self) -> u64 {
        self.vm.as_u64()
    }

    pub fn va_ptr(&self) -> u64 {
        self.va.as_u64()
    }

    /// Uploads per-scenario voltages. Used once at the start of a solve; the
    /// Newton loop then updates them in place on-device.
    pub fn upload_voltages(&mut self, vm: &[f64], va: &[f64]) -> Option<()> {
        self.vm.upload(vm)?;
        self.va.upload(va)
    }

    pub fn download_voltages(&self, vm: &mut [f64], va: &mut [f64]) -> Option<()> {
        self.vm.download(vm)?;
        self.va.download(va)
    }

    /// Uploads host-computed injections. The device-resident loop does not
    /// use this — [`GpuBatch::power_injections`] writes the same buffers
    /// on-device — but the assembly-only path
    /// (`examples/gpu_assembly_check.rs`, `tests/gpu_assembly_test.rs`) does,
    /// since it checks the assembly kernel in isolation against the CPU.
    pub fn upload_injections(&mut self, p_calc: &[f64], q_calc: &[f64]) -> Option<()> {
        self.p_calc.upload(p_calc)?;
        self.q_calc.upload(q_calc)
    }

    pub fn upload_active(&mut self, active: &[bool]) -> Option<()> {
        let flags: Vec<u32> = active.iter().map(|&a| a as u32).collect();
        self.active.upload(&flags)
    }

    /// Enqueues the assembly kernel. `active[s] == 0` writes an identity block
    /// for scenario `s` instead of the real Newton values — see the kernel's
    /// own comment for why that is a correctness requirement.
    pub fn assemble(&self) -> Option<()> {
        launch_check(
            unsafe {
                go_assemble_jacobian(
                    self.kinds.as_u32(),
                    self.bus_i.as_u32(),
                    self.bus_k.as_u32(),
                    self.y_re.as_f64(),
                    self.y_im.as_f64(),
                    self.identity_value.as_f64(),
                    self.scatter.as_u32(),
                    self.active.as_u32(),
                    self.vm.as_f64(),
                    self.va.as_f64(),
                    self.p_calc.as_f64(),
                    self.q_calc.as_f64(),
                    self.values.as_f64(),
                    self.nnz as u32,
                    self.n_buses as u32,
                    self.n_scenarios as u32,
                    self.stream.as_ptr(),
                )
            },
            "go_assemble_jacobian",
        )
    }

    /// Reads the Jacobian values back to the host. Synchronizes — this is a
    /// test/inspection path, not part of the Newton loop, whose whole point is
    /// that these values never cross the boundary.
    pub fn read_values(&self) -> Option<Vec<f64>> {
        self.stream.synchronize()?;
        let mut out = vec![0.0f64; self.n_scenarios * self.nnz];
        self.values.download(&mut out)?;
        Some(out)
    }

    /// One-shot assembly for the isolation checks: upload state, launch,
    /// read back, every scenario active. Returns the batch's flat value array
    /// scenario-major (`scenario * nnz + scatter[entry]`).
    pub fn assemble_batch(
        &mut self,
        states: &[Vec<Bus>],
        p_calc: &[Vec<f64>],
        q_calc: &[Vec<f64>],
    ) -> Option<Vec<f64>> {
        let active = vec![true; states.len()];
        self.assemble_batch_masked(states, p_calc, q_calc, &active)?;
        self.read_values()
    }

    /// [`assemble_batch`](Self::assemble_batch) with per-scenario masking and
    /// no readback — the result stays in the persistent output buffer for a
    /// device-resident consumer to read via [`values_ptr`](Self::values_ptr).
    pub fn assemble_batch_masked(
        &mut self,
        states: &[Vec<Bus>],
        p_calc: &[Vec<f64>],
        q_calc: &[Vec<f64>],
        active: &[bool],
    ) -> Option<()> {
        let nb = states.len();
        assert!(
            states.iter().all(|s| s.len() == self.n_buses),
            "every scenario's bus array must be n_buses long"
        );
        assert_eq!(active.len(), nb, "one active flag per scenario");

        self.resize(nb)?;

        let flat = FlatStates::from_states(states);
        let p: Vec<f64> = p_calc.iter().flatten().copied().collect();
        let q: Vec<f64> = q_calc.iter().flatten().copied().collect();

        self.upload_voltages(&flat.vm, &flat.va)?;
        self.upload_injections(&p, &q)?;
        self.upload_active(active)?;
        self.assemble()
    }
}

/// The full batched Newton loop's device state: a [`GpuAssembler`] plus every
/// other array the loop touches, so that between iterations only the
/// convergence norm comes down and only the active mask goes up.
///
/// Construct once per (topology, batch), then drive it from
/// [`crate::bde::solve_batch_block_diagonal_batched_device`]. The linear solve
/// itself is not here — it is `sparse_cudss::CudssBatchedSystem`, bound to
/// [`values_ptr`](GpuAssembler::values_ptr), [`rhs_ptr`](Self::rhs_ptr) and
/// [`dx_ptr`](Self::dx_ptr), and sharing this object's stream.
pub struct GpuBatch {
    assembler: GpuAssembler,

    // Topology-static.
    y_row_ptr: DeviceBuffer,
    y_col_idx: DeviceBuffer,
    y_re: DeviceBuffer,
    y_im: DeviceBuffer,
    non_slack: DeviceBuffer,
    pq: DeviceBuffer,
    zip_p_const: DeviceBuffer,
    zip_q_const: DeviceBuffer,
    zip_p_curr: DeviceBuffer,
    zip_q_curr: DeviceBuffer,
    zip_p_imp: DeviceBuffer,
    zip_q_imp: DeviceBuffer,
    n_angle: usize,
    n_pq: usize,

    // Per-batch.
    n_scenarios: usize,
    p_spec: DeviceBuffer,
    q_spec: DeviceBuffer,
    rhs: DeviceBuffer,
    dx: DeviceBuffer,
    max_mis: DeviceBuffer,
}

impl GpuBatch {
    /// Uploads everything the loop needs, from an already-constructed batch of
    /// per-scenario states (post-`linear_initial_guess`, post-overrides).
    ///
    /// Taking the states rather than `&[Scenario]` is deliberate: there is
    /// then exactly one place a `BusOverride` is applied, so the device copy
    /// cannot drift from what the CPU path would have solved. See
    /// [`FlatStates::from_states`].
    pub fn new(
        pattern: &JacobianPattern,
        ybus: &YBusSparse,
        buses_template: &[Bus],
        states: &[Vec<Bus>],
        scatter: &[u32],
    ) -> Option<Self> {
        let n_buses = buses_template.len();
        let nb = states.len();

        let mut assembler = GpuAssembler::new(pattern, n_buses)?;
        assembler.set_scatter(scatter)?;
        assembler.resize(nb)?;

        let csr = YbusCsr::from_ybus(ybus);
        let maps = UnknownMaps::from_buses(buses_template);
        let zip = ZipCoeffs::from_buses(buses_template);
        let flat = FlatStates::from_states(states);
        let blk = maps.block_size();

        assembler.upload_voltages(&flat.vm, &flat.va)?;

        let mut batch = Self {
            y_row_ptr: DeviceBuffer::from_slice(&csr.row_ptr)?,
            y_col_idx: DeviceBuffer::from_slice(&csr.col_idx)?,
            y_re: DeviceBuffer::from_slice(&csr.re)?,
            y_im: DeviceBuffer::from_slice(&csr.im)?,
            non_slack: DeviceBuffer::from_slice(&maps.non_slack)?,
            pq: DeviceBuffer::from_slice(&maps.pq)?,
            zip_p_const: DeviceBuffer::from_slice(&zip.p_const)?,
            zip_q_const: DeviceBuffer::from_slice(&zip.q_const)?,
            zip_p_curr: DeviceBuffer::from_slice(&zip.p_curr)?,
            zip_q_curr: DeviceBuffer::from_slice(&zip.q_curr)?,
            zip_p_imp: DeviceBuffer::from_slice(&zip.p_imp)?,
            zip_q_imp: DeviceBuffer::from_slice(&zip.q_imp)?,
            n_angle: maps.n_angle(),
            n_pq: maps.pq.len(),
            n_scenarios: nb,
            p_spec: DeviceBuffer::from_slice(&flat.p_spec)?,
            q_spec: DeviceBuffer::from_slice(&flat.q_spec)?,
            rhs: DeviceBuffer::new(nb * blk * size_of::<f64>())?,
            dx: DeviceBuffer::new(nb * blk * size_of::<f64>())?,
            max_mis: DeviceBuffer::new(nb * size_of::<f64>())?,
            assembler,
        };
        let all_active = vec![true; nb];
        batch.assembler.upload_active(&all_active)?;
        Some(batch)
    }

    pub fn stream(&self) -> &Stream {
        self.assembler.stream()
    }

    pub fn block_size(&self) -> usize {
        self.n_angle + self.n_pq
    }

    pub fn n_scenarios(&self) -> usize {
        self.n_scenarios
    }

    /// Device address of the Jacobian values buffer cuDSS's batched matrix
    /// binds to. Never `None` here — [`new`](Self::new) always sizes the batch.
    pub fn values_ptr(&self) -> u64 {
        self.assembler.values_ptr().expect("GpuBatch::new always resizes the assembler")
    }

    pub fn rhs_ptr(&self) -> u64 {
        self.rhs.as_u64()
    }

    pub fn dx_ptr(&self) -> u64 {
        self.dx.as_u64()
    }

    /// `p_calc`/`q_calc` for every scenario, on-device — the batched
    /// equivalent of `network::power_injections`, replacing the serial host
    /// loop that ran one `power_injections` call per scenario per iteration.
    pub fn power_injections(&self) -> Option<()> {
        launch_check(
            unsafe {
                go_power_injections(
                    self.y_row_ptr.as_i32(),
                    self.y_col_idx.as_i32(),
                    self.y_re.as_f64(),
                    self.y_im.as_f64(),
                    self.assembler.vm.as_f64(),
                    self.assembler.va.as_f64(),
                    self.assembler.p_calc.as_f64(),
                    self.assembler.q_calc.as_f64(),
                    self.assembler.n_buses as u32,
                    self.n_scenarios as u32,
                    self.stream().as_ptr(),
                )
            },
            "go_power_injections",
        )
    }

    /// Writes every scenario's right-hand side and reduces its max-|mismatch|
    /// into the `max_mis` buffer. Injections are recomputed for masked
    /// scenarios too, exactly as the CPU loop does.
    pub fn mismatch(&self) -> Option<()> {
        launch_check(
            unsafe {
                go_mismatch(
                    self.non_slack.as_u32(),
                    self.pq.as_u32(),
                    self.p_spec.as_f64(),
                    self.q_spec.as_f64(),
                    self.zip_p_const.as_f64(),
                    self.zip_q_const.as_f64(),
                    self.zip_p_curr.as_f64(),
                    self.zip_q_curr.as_f64(),
                    self.zip_p_imp.as_f64(),
                    self.zip_q_imp.as_f64(),
                    self.assembler.vm.as_f64(),
                    self.assembler.p_calc.as_f64(),
                    self.assembler.q_calc.as_f64(),
                    self.rhs.as_f64(),
                    self.max_mis.as_f64(),
                    self.n_angle as u32,
                    self.n_pq as u32,
                    self.assembler.n_buses as u32,
                    self.n_scenarios as u32,
                    self.stream().as_ptr(),
                )
            },
            "go_mismatch",
        )
    }

    /// The loop's **only** per-iteration device-to-host transfer: one f64 per
    /// scenario. Enqueued on the stream, then synchronized — that
    /// synchronization is unavoidable, since whether a scenario has converged
    /// is a host-visible decision.
    pub fn download_max_mismatch(&self, out: &mut [f64]) -> Option<()> {
        assert_eq!(out.len(), self.n_scenarios);
        self.max_mis.download_async(out, self.stream())?;
        self.stream().synchronize()
    }

    /// The loop's only per-iteration host-to-device transfer: one u32 per
    /// scenario.
    pub fn upload_active(&mut self, active: &[bool]) -> Option<()> {
        assert_eq!(active.len(), self.n_scenarios);
        self.assembler.upload_active(active)
    }

    /// Zeroes the right-hand side of every masked scenario, so its identity
    /// block yields Δx = 0. Must run *after* the freshly-updated mask is
    /// uploaded, to catch scenarios that converged this iteration.
    pub fn zero_masked_rhs(&self) -> Option<()> {
        launch_check(
            unsafe {
                go_zero_masked_rhs(
                    self.assembler.active.as_u32(),
                    self.rhs.as_f64(),
                    self.block_size() as u32,
                    self.n_scenarios as u32,
                    self.stream().as_ptr(),
                )
            },
            "go_zero_masked_rhs",
        )
    }

    pub fn assemble(&self) -> Option<()> {
        self.assembler.assemble()
    }

    /// Applies Δx to the device-resident voltages. Masked scenarios are
    /// skipped rather than relying on their Δx being zero.
    pub fn apply_update(&self) -> Option<()> {
        launch_check(
            unsafe {
                go_apply_update(
                    self.non_slack.as_u32(),
                    self.pq.as_u32(),
                    self.assembler.active.as_u32(),
                    self.dx.as_f64(),
                    self.assembler.vm.as_f64(),
                    self.assembler.va.as_f64(),
                    self.n_angle as u32,
                    self.n_pq as u32,
                    self.assembler.n_buses as u32,
                    self.n_scenarios as u32,
                    self.stream().as_ptr(),
                )
            },
            "go_apply_update",
        )
    }

    /// The one readback of the whole solve: every scenario's converged
    /// voltages, written back into the host states.
    pub fn download_voltages_into(&self, states: &mut [Vec<Bus>]) -> Option<()> {
        self.stream().synchronize()?;
        let n_buses = self.assembler.n_buses;
        let total = self.n_scenarios * n_buses;
        let mut flat = FlatStates {
            n_buses,
            n_scenarios: self.n_scenarios,
            vm: vec![0.0; total],
            va: vec![0.0; total],
            p_spec: Vec::new(),
            q_spec: Vec::new(),
        };
        self.assembler.download_voltages(&mut flat.vm, &mut flat.va)?;
        flat.write_voltages_into(states);
        Some(())
    }

    /// Reads the right-hand side back — inspection only, for
    /// `examples/bde_profile.rs`'s cross-check against the CPU mismatch loop.
    pub fn read_rhs(&self) -> Option<Vec<f64>> {
        self.stream().synchronize()?;
        let mut out = vec![0.0f64; self.n_scenarios * self.block_size()];
        self.rhs.download(&mut out)?;
        Some(out)
    }

    /// Reads `p_calc`/`q_calc` back — inspection only, for the GPU-vs-CPU
    /// `power_injections` regression test.
    pub fn read_injections(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        self.stream().synchronize()?;
        let total = self.n_scenarios * self.assembler.n_buses;
        let mut p = vec![0.0f64; total];
        let mut q = vec![0.0f64; total];
        self.assembler.p_calc.download(&mut p)?;
        self.assembler.q_calc.download(&mut q)?;
        Some((p, q))
    }
}
