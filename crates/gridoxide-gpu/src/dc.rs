//! DC N-1 on the device: [`GpuDcSolver`].
//!
//! Phase 2 of `plans/GPU_DC_PLAN.md`: gridoxide-core's
//! `DcSensitivity::contingencies` with its solves and its arithmetic on the
//! device. The reduced susceptance matrix `B` (every non-reference bus of
//! every island with a reference, one block per island) is analysed and
//! factored by cuDSS once per session. For each outage `l` of branch `f → t`,
//! the angle response to a unit transfer from `f` to `t`, `θ⁽ˡ⁾ = B⁻¹(e_f −
//! e_t)`, is one column of a single multi-column solve; from it
//! `d = b_l (θ⁽ˡ⁾_f − θ⁽ˡ⁾_t)`, the transfer `c = p_l / (1 − d)` the removed
//! branch would carry, and
//!
//! ```text
//! θ after l = θ + c θ⁽ˡ⁾,   p after l = p + c ψ⁽ˡ⁾ (ψ⁽ˡ⁾_j = b_j (θ⁽ˡ⁾_from − θ⁽ˡ⁾_to)), p_l = 0
//! ```
//!
//! in two kernels writing the outages' rows. Outages run in chunks of the
//! session's `capacity` columns, pipelined over two slots: while chunk i is
//! computed on one stream, chunk i − 1 is copied to page-locked memory on
//! another and chunk i − 2 from there into the result by a host thread on
//! every core. The output is what costs: case9241pegase's 14384 outages are
//! 2.9 GB of angles and flows. The base case is the CPU's DC solve
//! (`DcSolver`, kept by the session): one sparse solve, a millisecond on
//! case9241pegase.

use std::ptr;

use gridoxide_core::linear::{DcContingencies, DcOptions, DcSolver, dc_branches};
use gridoxide_core::types::{Bus, BusType, Line, Transformer};

use crate::cuda::{DeviceBuffer, Event, PinnedBuffer, Stream, cuda, ffi};
use crate::cudss::cudss;
use crate::newton::BatchError;
use crate::uniform::LinearError;

/// `|1 − d|` below this is a radial branch: gridoxide-core's
/// `LODF_BRIDGE_TOL`.
const BRIDGE_TOL: f64 = 1e-9;

/// The reduced `B` and the per-branch tables of a network, built on the host.
struct Reduced {
    n: usize,
    /// Each bus's row, or -1.
    pos: Vec<i32>,
    row_ptr: Vec<i32>,
    col: Vec<i32>,
    values: Vec<f64>,
    br_from: Vec<u32>,
    br_to: Vec<u32>,
    br_b: Vec<f64>,
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn reduce(buses: &[Bus], lines: &[Line], transformers: &[Transformer], opts: DcOptions) -> Reduced {
    let n_bus = buses.len();
    let n_br = lines.len() + transformers.len();
    let branches = dc_branches(lines, transformers, opts);
    let mut parent: Vec<usize> = (0..n_bus).collect();
    for b in &branches {
        let (x, y) = (find(&mut parent, b.from), find(&mut parent, b.to));
        parent[x] = y;
    }
    let mut referenced = vec![false; n_bus];
    for (i, bus) in buses.iter().enumerate() {
        if bus.bus_type == BusType::Slack {
            let r = find(&mut parent, i);
            referenced[r] = true;
        }
    }
    let in_reference: Vec<bool> = (0..n_bus)
        .map(|i| referenced[find(&mut parent, i)])
        .collect();
    let mut pos = vec![-1i32; n_bus];
    let mut n = 0;
    for (i, bus) in buses.iter().enumerate() {
        if in_reference[i] && bus.bus_type != BusType::Slack {
            pos[i] = n as i32;
            n += 1;
        }
    }

    let (mut br_from, mut br_to, mut br_b) = (vec![0u32; n_br], vec![0u32; n_br], vec![0.0; n_br]);
    let mut rows: Vec<Vec<(i32, f64)>> = vec![Vec::new(); n];
    for b in &branches {
        if !in_reference[b.from] {
            continue;
        }
        br_from[b.index] = b.from as u32;
        br_to[b.index] = b.to as u32;
        br_b[b.index] = b.b;
        let (pf, pt) = (pos[b.from], pos[b.to]);
        for (r, c, v) in [(pf, pf, b.b), (pt, pt, b.b), (pf, pt, -b.b), (pt, pf, -b.b)] {
            if r >= 0 && c >= 0 {
                rows[r as usize].push((c, v));
            }
        }
    }
    let (mut row_ptr, mut col, mut values) = (vec![0i32], Vec::new(), Vec::new());
    for mut row in rows {
        row.sort_by_key(|&(c, _)| c);
        for (c, v) in row {
            if col.len() > row_ptr[row_ptr.len() - 1] as usize && *col.last().unwrap() == c {
                *values.last_mut().unwrap() += v;
            } else {
                col.push(c);
                values.push(v);
            }
        }
        row_ptr.push(col.len() as i32);
    }
    Reduced {
        n,
        pos,
        row_ptr,
        col,
        values,
        br_from,
        br_to,
        br_b,
    }
}

/// cuDSS's handles for the one factored system and its dense right-hand
/// sides and solutions, `n × capacity`.
struct System {
    stream: ffi::cudaStream_t,
    handle: ffi::cudssHandle_t,
    config: ffi::cudssConfig_t,
    data: ffi::cudssData_t,
    a: ffi::cudssMatrix_t,
}

impl System {
    fn execute(
        &self,
        phase: ffi::cudssPhase_t,
        x: ffi::cudssMatrix_t,
        b: ffi::cudssMatrix_t,
        what: &str,
    ) -> Result<(), LinearError> {
        // SAFETY: every handle was created in `GpuDcSolver::new` and points at
        // buffers the session owns.
        cudss(
            unsafe {
                ffi::cudssExecute(
                    self.handle,
                    phase as i32,
                    self.config,
                    self.data,
                    self.a,
                    x,
                    b,
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
            if !self.a.is_null() {
                ffi::cudssMatrixDestroy(self.a);
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

/// A dense `n × k` column-major descriptor over a device buffer, destroyed
/// on drop.
struct Dense(ffi::cudssMatrix_t);

impl Dense {
    fn new(n: usize, k: usize, buf: *mut f64) -> Result<Self, LinearError> {
        let mut m: ffi::cudssMatrix_t = ptr::null_mut();
        // SAFETY: `buf` holds at least n × k values (the caller's).
        cudss(
            unsafe {
                ffi::cudssMatrixCreateDn(
                    &mut m,
                    n as i64,
                    k as i64,
                    n as i64,
                    buf.cast(),
                    ffi::CUDA_R_64F,
                    ffi::CUDSS_LAYOUT_COL_MAJOR,
                )
            },
            "cudssMatrixCreateDn",
        )?;
        Ok(Self(m))
    }
}

impl Drop for Dense {
    fn drop(&mut self) {
        // SAFETY: created in `new`, destroyed once.
        unsafe { ffi::cudssMatrixDestroy(self.0) };
    }
}

/// One chunk's outputs on the device and on their way to the host.
struct Slot {
    c: DeviceBuffer<f64>,
    solved: DeviceBuffer<i32>,
    angles: DeviceBuffer<f64>,
    flows: DeviceBuffer<f64>,
    host_solved: PinnedBuffer<i32>,
    host_angles: PinnedBuffer<f64>,
    host_flows: PinnedBuffer<f64>,
    /// Recorded on the compute stream once the chunk's rows are written.
    computed: Event,
    /// Recorded on the copy stream once they are in the pinned buffers.
    copied: Event,
}

impl Slot {
    fn new(n_bus: usize, n_br: usize, capacity: usize) -> Result<Self, LinearError> {
        Ok(Self {
            c: DeviceBuffer::new(capacity)?,
            solved: DeviceBuffer::new(capacity)?,
            angles: DeviceBuffer::new((n_bus * capacity).max(1))?,
            flows: DeviceBuffer::new((n_br * capacity).max(1))?,
            host_solved: PinnedBuffer::new(capacity)?,
            host_angles: PinnedBuffer::new(n_bus * capacity)?,
            host_flows: PinnedBuffer::new(n_br * capacity)?,
            computed: Event::new()?,
            copied: Event::new()?,
        })
    }
}

/// A session of DC N-1 on one network; see the module documentation.
pub struct GpuDcSolver {
    // First: its descriptors point into the buffers below and it runs on
    // `stream`, so it goes before them; the streams go before the slots,
    // whose buffers their queued work uses.
    sys: System,
    stream: Stream,
    copy_stream: Stream,
    n: usize,
    n_bus: usize,
    n_br: usize,
    capacity: usize,
    base: DcSolver,
    _row_ptr: DeviceBuffer<i32>,
    _col: DeviceBuffer<i32>,
    _values: DeviceBuffer<f64>,
    pos: DeviceBuffer<i32>,
    br_from: DeviceBuffer<u32>,
    br_to: DeviceBuffer<u32>,
    br_b: DeviceBuffer<f64>,
    base_angles: DeviceBuffer<f64>,
    base_flows: DeviceBuffer<f64>,
    out_br: DeviceBuffer<u32>,
    rhs: DeviceBuffer<f64>,
    theta: DeviceBuffer<f64>,
    slots: [Slot; 2],
}

impl GpuDcSolver {
    /// A session for the DC model of `lines` and `transformers` (`opts` as
    /// for `dc_power_flow`), with the reference buses of `buses`, solving
    /// `capacity` outages per chunk.
    pub fn new(
        buses: &[Bus],
        lines: &[Line],
        transformers: &[Transformer],
        opts: DcOptions,
        capacity: usize,
    ) -> Result<Self, BatchError> {
        let capacity = capacity.max(1);
        let r = reduce(buses, lines, transformers, opts);
        let (n, n_bus, n_br) = (r.n, buses.len(), r.br_b.len());
        let stream = Stream::new()?;
        let up = |v: &[f64]| -> Result<DeviceBuffer<f64>, LinearError> {
            let mut d = DeviceBuffer::new(v.len().max(1))?;
            if !v.is_empty() {
                d.upload(v, stream.0)?;
            }
            Ok(d)
        };
        let up_i = |v: &[i32]| -> Result<DeviceBuffer<i32>, LinearError> {
            let mut d = DeviceBuffer::new(v.len().max(1))?;
            if !v.is_empty() {
                d.upload(v, stream.0)?;
            }
            Ok(d)
        };
        let up_u = |v: &[u32]| -> Result<DeviceBuffer<u32>, LinearError> {
            let mut d = DeviceBuffer::new(v.len().max(1))?;
            if !v.is_empty() {
                d.upload(v, stream.0)?;
            }
            Ok(d)
        };
        let row_ptr = up_i(&r.row_ptr)?;
        let col = up_i(&r.col)?;
        let values = up(&r.values)?;
        let mut sys = System {
            stream: stream.0,
            handle: ptr::null_mut(),
            config: ptr::null_mut(),
            data: ptr::null_mut(),
            a: ptr::null_mut(),
        };
        let rhs = DeviceBuffer::new((n * capacity).max(1))?;
        let theta = DeviceBuffer::new((n * capacity).max(1))?;
        // SAFETY: each call gets valid out-pointers; the matrix descriptor
        // points at `row_ptr`, `col` and `values`, kept by the session.
        unsafe {
            cudss(ffi::cudssCreate(&mut sys.handle), "cudssCreate")?;
            cudss(ffi::cudssSetStream(sys.handle, stream.0), "cudssSetStream")?;
            cudss(ffi::cudssConfigCreate(&mut sys.config), "cudssConfigCreate")?;
            cudss(
                ffi::cudssDataCreate(sys.handle, &mut sys.data),
                "cudssDataCreate",
            )?;
            if n > 0 {
                cudss(
                    ffi::cudssMatrixCreateCsr(
                        &mut sys.a,
                        n as i64,
                        n as i64,
                        r.col.len() as i64,
                        row_ptr.ptr.cast(),
                        ptr::null_mut(),
                        col.ptr.cast(),
                        values.ptr.cast(),
                        ffi::CUDA_R_32I,
                        ffi::CUDA_R_32I,
                        ffi::CUDA_R_64F,
                        ffi::CUDSS_MTYPE_GENERAL,
                        ffi::CUDSS_MVIEW_FULL,
                        ffi::CUDSS_BASE_ZERO,
                    ),
                    "cudssMatrixCreateCsr",
                )?;
            }
        }
        if n > 0 {
            let (x, b) = (Dense::new(n, 1, theta.ptr)?, Dense::new(n, 1, rhs.ptr)?);
            sys.execute(ffi::CUDSS_PHASE_ANALYSIS, x.0, b.0, "ANALYSIS")?;
            sys.execute(ffi::CUDSS_PHASE_FACTORIZATION, x.0, b.0, "FACTORIZATION")?;
        }
        Ok(Self {
            n,
            n_bus,
            n_br,
            capacity,
            base: DcSolver::new(buses, lines, transformers, opts),
            pos: up_i(&r.pos)?,
            br_from: up_u(&r.br_from)?,
            br_to: up_u(&r.br_to)?,
            br_b: up(&r.br_b)?,
            base_angles: DeviceBuffer::new(n_bus.max(1))?,
            base_flows: DeviceBuffer::new(n_br.max(1))?,
            out_br: DeviceBuffer::new(capacity)?,
            rhs,
            theta,
            slots: [
                Slot::new(n_bus, n_br, capacity)?,
                Slot::new(n_bus, n_br, capacity)?,
            ],
            copy_stream: Stream::new()?,
            _row_ptr: row_ptr,
            _col: col,
            _values: values,
            sys,
            stream,
        })
    }

    /// Outages per chunk.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The base case of `buses` (their injections; the reference buses the
    /// session was made with) solved on the CPU, then every outage of
    /// `outages` (flat branch indices) on the device: their angles and flows,
    /// as `DcSensitivity::contingencies` returns them.
    pub fn solve_contingencies(
        &mut self,
        buses: &[Bus],
        outages: &[usize],
    ) -> Result<DcContingencies, BatchError> {
        if let Some(&branch) = outages.iter().find(|&&b| b >= self.n_br) {
            return Err(BatchError::BranchOutOfRange {
                scenario: outages.iter().position(|&b| b == branch).unwrap(),
                branch,
                n_branches: self.n_br,
            });
        }
        let mut solved_buses = buses.to_vec();
        let base = self.base.solve(&mut solved_buses);
        let base_angles: Vec<f64> = solved_buses.iter().map(|b| b.voltage_ang).collect();
        if self.n_bus > 0 {
            self.base_angles.upload(&base_angles, self.stream.0)?;
        }
        if self.n_br > 0 {
            self.base_flows.upload(&base.branch_p, self.stream.0)?;
        }

        let total = outages.len();
        let (n_bus, n_br) = (self.n_bus, self.n_br);
        let mut angles = vec![0.0; total * n_bus];
        let mut flows = vec![0.0; total * n_br];
        let mut solved = vec![false; total];
        let cap = self.capacity;
        let this = &*self;
        std::thread::scope(|scope| -> Result<(), BatchError> {
            let mut pending: [Option<std::thread::ScopedJoinHandle<'_, Result<(), LinearError>>>;
                2] = [None, None];
            let rows = angles
                .chunks_mut(cap * n_bus.max(1))
                .zip(flows.chunks_mut(cap * n_br.max(1)))
                .zip(solved.chunks_mut(cap));
            for (i, (chunk, ((dst_angles, dst_flows), dst_solved))) in
                outages.chunks(cap).zip(rows).enumerate()
            {
                let k = chunk.len();
                let slot = &this.slots[i % 2];
                // The slot's pinned buffers are free once chunk i − 2's host
                // copy has run, and its device buffers once that chunk's
                // device-to-host copy has (which the host copy waited for).
                if let Some(h) = pending[i % 2].take() {
                    h.join().expect("a host copy panicked")?;
                }
                let out_br: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
                // SAFETY: every buffer holds `capacity` columns, at least
                // `k`; the slot is free (above), and each stream orders its
                // own work, the events the work between them.
                unsafe {
                    crate::cuda::copy_to_device(this.out_br.ptr, &out_br, this.stream.0)?;
                    this.run_chunk(k, slot)?;
                    cuda(
                        ffi::cudaEventRecord(slot.computed.0, this.stream.0),
                        "cudaEventRecord",
                    )?;
                    cuda(
                        ffi::cudaStreamWaitEvent(this.copy_stream.0, slot.computed.0, 0),
                        "cudaStreamWaitEvent",
                    )?;
                    let cs = this.copy_stream.0;
                    copy_async_to_host(slot.host_solved.as_mut_ptr(), slot.solved.ptr, k, cs)?;
                    copy_async_to_host(
                        slot.host_angles.as_mut_ptr(),
                        slot.angles.ptr,
                        k * n_bus,
                        cs,
                    )?;
                    copy_async_to_host(slot.host_flows.as_mut_ptr(), slot.flows.ptr, k * n_br, cs)?;
                    cuda(ffi::cudaEventRecord(slot.copied.0, cs), "cudaEventRecord")?;
                }
                // Only the host side of the slot crosses to the thread.
                let (copied, host_angles, host_flows, host_solved) = (
                    &slot.copied,
                    &slot.host_angles,
                    &slot.host_flows,
                    &slot.host_solved,
                );
                pending[i % 2] = Some(scope.spawn(move || {
                    // SAFETY: the event is the slot's, live for the scope.
                    cuda(
                        unsafe { ffi::cudaEventSynchronize(copied.0) },
                        "cudaEventSynchronize",
                    )?;
                    parallel_copy(dst_angles, host_angles.prefix(k * n_bus));
                    parallel_copy(dst_flows, host_flows.prefix(k * n_br));
                    for (s, f) in dst_solved.iter_mut().zip(host_solved.prefix(k)) {
                        *s = *f != 0;
                    }
                    Ok(())
                }));
            }
            for h in pending.iter_mut().filter_map(Option::take) {
                h.join().expect("a host copy panicked")?;
            }
            Ok(())
        })?;
        Ok(DcContingencies::from_parts(
            n_bus, n_br, angles, flows, solved,
        ))
    }

    /// The kernels and the solve for the first `k` columns.
    fn run_chunk(&self, k: usize, slot: &Slot) -> Result<(), BatchError> {
        let (n, s) = (self.n, self.stream.0.cast());
        // SAFETY: every buffer is sized for `capacity` columns, at least `k`.
        unsafe {
            if n > 0 {
                launched(
                    ffi::go_dc_unit_rhs(
                        self.out_br.ptr,
                        self.br_from.ptr,
                        self.br_to.ptr,
                        self.pos.ptr,
                        self.rhs.ptr,
                        n as u32,
                        k as u32,
                        s,
                    ),
                    "go_dc_unit_rhs",
                )?;
                let (x, b) = (
                    Dense::new(n, k, self.theta.ptr)?,
                    Dense::new(n, k, self.rhs.ptr)?,
                );
                self.sys
                    .execute(ffi::CUDSS_PHASE_SOLVE, x.0, b.0, "SOLVE")?;
            }
            launched(
                ffi::go_dc_transfer(
                    self.theta.ptr,
                    self.out_br.ptr,
                    self.br_from.ptr,
                    self.br_to.ptr,
                    self.br_b.ptr,
                    self.pos.ptr,
                    self.base_flows.ptr,
                    BRIDGE_TOL,
                    slot.c.ptr,
                    slot.solved.ptr,
                    n as u32,
                    k as u32,
                    s,
                ),
                "go_dc_transfer",
            )?;
            launched(
                ffi::go_dc_angles(
                    self.theta.ptr,
                    self.pos.ptr,
                    self.base_angles.ptr,
                    slot.c.ptr,
                    slot.solved.ptr,
                    slot.angles.ptr,
                    n as u32,
                    self.n_bus as u32,
                    k as u32,
                    s,
                ),
                "go_dc_angles",
            )?;
            launched(
                ffi::go_dc_flows(
                    self.theta.ptr,
                    self.out_br.ptr,
                    self.br_from.ptr,
                    self.br_to.ptr,
                    self.br_b.ptr,
                    self.pos.ptr,
                    self.base_flows.ptr,
                    slot.c.ptr,
                    slot.solved.ptr,
                    slot.flows.ptr,
                    n as u32,
                    self.n_br as u32,
                    k as u32,
                    s,
                ),
                "go_dc_flows",
            )?;
        }
        Ok(())
    }
}

fn launched(err: i32, what: &str) -> Result<(), LinearError> {
    cuda(err as ffi::cudaError_t, what)
}

/// `dst = src` on every core: the destination is freshly allocated, so its
/// page faults are taken in parallel too.
fn parallel_copy(dst: &mut [f64], src: &[f64]) {
    use rayon::prelude::*;
    const CHUNK: usize = 1 << 16;
    dst.par_chunks_mut(CHUNK)
        .zip(src.par_chunks(CHUNK))
        .for_each(|(d, s)| d.copy_from_slice(s));
}

/// Queues a copy of `len` values from the device allocation `src` to the
/// host memory `dst` on `stream`.
///
/// # Safety
///
/// Both must hold at least `len` values, and `dst` must stay untouched until
/// the copy has run.
unsafe fn copy_async_to_host<T: Copy>(
    dst: *mut T,
    src: *const T,
    len: usize,
    stream: ffi::cudaStream_t,
) -> Result<(), LinearError> {
    // SAFETY: the caller's.
    cuda(
        unsafe {
            ffi::cudaMemcpyAsync(
                dst.cast(),
                src.cast(),
                len * size_of::<T>(),
                ffi::cudaMemcpyDeviceToHost,
                stream,
            )
        },
        "cudaMemcpyAsync (to host)",
    )
}
