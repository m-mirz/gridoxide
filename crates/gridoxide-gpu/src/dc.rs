//! DC N-1 and DC sweeps on the device: [`GpuDcSolver`].
//!
//! gridoxide-core's `DcSensitivity::contingencies` with its solves and its
//! arithmetic on the device. The reduced susceptance matrix `B` (every
//! non-reference bus of every island with a reference, one block per
//! island) is analysed and factored by cuDSS once per session. For each outage `l` of branch `f → t`,
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
//!
//! A sweep of injection scenarios ([`GpuDcSolver::solve_sweep`]) is
//! gridoxide-core's `DcBatchSolver` on the same factorization: each
//! scenario's injection change at the unknowns is a column of one
//! multi-column solve, `Δθ = B⁻¹ ΔP`, and its rows are the base case plus
//! that response, through the same two-slot pipeline. Its slack pickups are
//! sums of injections, taken on the host.
//!
//! A single DC power flow ([`GpuDcSolver::solve`]) is the CPU's `DcSolver`
//! with its reduced solve on the device: every island's right-hand side is
//! one column, solved on the session's factorization. The right-hand sides
//! and everything after the angles (flows, residual, pickups) are
//! `DcSolver::solve_with`'s own code, so the two agree to the solve's
//! rounding.

use std::ptr;

use gridoxide_core::linear::{
    DcContingencies, DcOptions, DcSolution, DcSolver, DcSystem, dc_branches,
};
use gridoxide_core::scenario::Scenario;
use gridoxide_core::types::{Bus, BusType, Line, Transformer};

use crate::cuda::{DeviceBuffer, Event, PinnedBuffer, Stream, cuda, ffi};
use crate::cudss::cudss;
use crate::newton::{BatchError, check_buses};
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

/// What [`GpuDcSolver::solve_sweep`] returns: per scenario, what
/// gridoxide-core's `DcBatchResult` holds, in flat scenario-major buffers.
#[derive(Clone, Debug)]
pub struct DcSweep {
    n_scenarios: usize,
    n_buses: usize,
    n_branches: usize,
    n_islands: usize,
    /// Per scenario, one angle per bus (radians).
    pub angles: Vec<f64>,
    /// Per scenario, one from-end flow per flat branch (per unit).
    pub flows: Vec<f64>,
    /// Per scenario, what each island's reference buses supply (per unit),
    /// in the order of the base solve's `DcSolution::islands`, as
    /// `DcBatchResult::slack_pickup`.
    pub slack_pickup: Vec<f64>,
}

impl DcSweep {
    /// Scenarios.
    pub fn len(&self) -> usize {
        self.n_scenarios
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Scenario `k`'s bus angles.
    pub fn angles(&self, k: usize) -> &[f64] {
        &self.angles[k * self.n_buses..(k + 1) * self.n_buses]
    }

    /// Scenario `k`'s branch flows.
    pub fn flows(&self, k: usize) -> &[f64] {
        &self.flows[k * self.n_branches..(k + 1) * self.n_branches]
    }

    /// Scenario `k`'s slack pickups, one per island.
    pub fn slack_pickup(&self, k: usize) -> &[f64] {
        &self.slack_pickup[k * self.n_islands..(k + 1) * self.n_islands]
    }
}

/// A sweep's injection changes at the unknowns, every scenario's after the
/// previous one's: column (scenario), row and value of each.
struct Injections {
    col: Vec<u32>,
    row: Vec<u32>,
    val: Vec<f64>,
    /// Per scenario, where its entries start; one more at the end.
    start: Vec<usize>,
}

/// What [`GpuDcSolver::pipelined`] returns: every column's angles, flows
/// and solved flags.
type Rows = (Vec<f64>, Vec<f64>, Vec<bool>);

/// The device's copy of one chunk's [`Injections`], grown as a sweep needs.
struct InjectionBuffers {
    col: DeviceBuffer<u32>,
    row: DeviceBuffer<u32>,
    val: DeviceBuffer<f64>,
}

impl InjectionBuffers {
    fn new(len: usize) -> Result<Self, LinearError> {
        let len = len.max(1);
        Ok(Self {
            col: DeviceBuffer::new(len)?,
            row: DeviceBuffer::new(len)?,
            val: DeviceBuffer::new(len)?,
        })
    }
}

/// A session of DC N-1 and DC sweeps on one network; see the module
/// documentation.
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
    /// `pos` on the host, for a sweep's right-hand sides.
    host_pos: Vec<i32>,
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
    injections: InjectionBuffers,
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
            host_pos: r.pos.clone(),
            br_from: up_u(&r.br_from)?,
            br_to: up_u(&r.br_to)?,
            br_b: up(&r.br_b)?,
            base_angles: DeviceBuffer::new(n_bus.max(1))?,
            base_flows: DeviceBuffer::new(n_br.max(1))?,
            out_br: DeviceBuffer::new(capacity)?,
            rhs,
            theta,
            injections: InjectionBuffers::new(capacity)?,
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
        self.upload_base(buses)?;
        let (angles, flows, solved) = self.pipelined(outages.len(), true, |first, k, slot| {
            let out_br: Vec<u32> = outages[first..first + k]
                .iter()
                .map(|&b| b as u32)
                .collect();
            // SAFETY: `out_br` holds `capacity` values, at least `k`.
            unsafe { crate::cuda::copy_to_device(self.out_br.ptr, &out_br, self.stream.0)? };
            self.run_chunk(k, slot)
        })?;
        Ok(DcContingencies::from_parts(
            self.n_bus, self.n_br, angles, flows, solved,
        ))
    }

    /// The base case of `buses` (their injections; the reference buses the
    /// session was made with) solved on the CPU, then every scenario of
    /// `scenarios` on the device: its angles, flows and slack pickups, as
    /// gridoxide-core's `DcBatchSolver::solve` returns them for the same
    /// template and scenarios.
    ///
    /// As there, a scenario's `p_spec` overrides replace the template's,
    /// its ZIP terms stay, and `q_spec` and `voltage_mag` are ignored; a
    /// scenario with branch outages is refused (outages are
    /// [`solve_contingencies`](Self::solve_contingencies)).
    pub fn solve_sweep(
        &mut self,
        buses: &[Bus],
        scenarios: &[Scenario],
    ) -> Result<DcSweep, BatchError> {
        if let Some(scenario) = scenarios
            .iter()
            .position(|sc| !sc.branch_outages.is_empty())
        {
            return Err(BatchError::OutagesUnsupported { scenario });
        }
        check_buses(buses, scenarios)?;
        let islands = self.upload_base(buses)?;

        // Each island's non-reference buses, whose injections its references
        // balance (DC is lossless), and their sum in the template.
        let mut island_of = vec![usize::MAX; self.n_bus];
        let mut base_sum = vec![0.0; islands.len()];
        for (i, island) in islands.iter().enumerate() {
            for &bus in &island.bus_indices {
                if !island.slack_indices.contains(&bus) {
                    island_of[bus] = i;
                    base_sum[i] += dc_injection(&buses[bus]);
                }
            }
        }

        // The injection changes, the last override of a bus winning.
        let mut inj = Injections {
            col: Vec::new(),
            row: Vec::new(),
            val: Vec::new(),
            start: Vec::with_capacity(scenarios.len() + 1),
        };
        let mut slack_pickup = Vec::with_capacity(scenarios.len() * islands.len());
        let mut delta = vec![0.0; self.n_bus];
        let mut touched = Vec::new();
        for (s, sc) in scenarios.iter().enumerate() {
            inj.start.push(inj.val.len());
            for ov in &sc.bus_overrides {
                if let Some(p) = ov.p_spec {
                    touched.push(ov.bus);
                    delta[ov.bus] = p - buses[ov.bus].p_spec;
                }
            }
            let mut sum = base_sum.clone();
            for &bus in &touched {
                let d = std::mem::take(&mut delta[bus]);
                if d == 0.0 {
                    continue;
                }
                if island_of[bus] != usize::MAX {
                    sum[island_of[bus]] += d;
                }
                if self.host_pos[bus] >= 0 {
                    inj.col.push(s as u32);
                    inj.row.push(self.host_pos[bus] as u32);
                    inj.val.push(d);
                }
            }
            touched.clear();
            slack_pickup.extend(sum.iter().map(|v| -v));
        }
        inj.start.push(inj.val.len());

        let cap = self.capacity;
        let most = (0..scenarios.len())
            .step_by(cap)
            .map(|first| {
                let last = (first + cap).min(scenarios.len());
                inj.start[last] - inj.start[first]
            })
            .max()
            .unwrap_or(0);
        if most > self.injections.val.len() {
            // The stream's queued work may still read the old buffers.
            cuda(
                // SAFETY: the session's own stream.
                unsafe { ffi::cudaStreamSynchronize(self.stream.0) },
                "cudaStreamSynchronize",
            )?;
            self.injections = InjectionBuffers::new(most)?;
        }

        let (angles, flows, _) = self.pipelined(scenarios.len(), false, |first, k, slot| {
            let (a, b) = (inj.start[first], inj.start[first + k]);
            let col: Vec<u32> = inj.col[a..b].iter().map(|&c| c - first as u32).collect();
            // SAFETY: the buffers hold `most` values, at least `b - a`.
            unsafe {
                crate::cuda::copy_to_device(self.injections.col.ptr, &col, self.stream.0)?;
                crate::cuda::copy_to_device(
                    self.injections.row.ptr,
                    &inj.row[a..b],
                    self.stream.0,
                )?;
                crate::cuda::copy_to_device(
                    self.injections.val.ptr,
                    &inj.val[a..b],
                    self.stream.0,
                )?;
            }
            self.run_sweep_chunk(k, b - a, slot)
        })?;
        Ok(DcSweep {
            n_scenarios: scenarios.len(),
            n_buses: self.n_bus,
            n_branches: self.n_br,
            n_islands: islands.len(),
            angles,
            flows,
            slack_pickup,
        })
    }

    /// A single DC power flow of `buses` (their injections; the reference
    /// buses the session was made with), its reduced solve on the device:
    /// angles written back into `buses` and the rest returned, as
    /// `DcSolver::solve` does on the CPU. An island the CPU cannot factor,
    /// or whose device angles are not finite, is reported
    /// `DcIslandStatus::Singular`.
    pub fn solve(&mut self, buses: &mut [Bus]) -> Result<DcSolution, BatchError> {
        let mut failure = None;
        let this = &*self;
        let solution = this
            .base
            .solve_with(buses, |systems| match this.solve_column(systems) {
                Ok(theta) => theta,
                Err(e) => {
                    failure = Some(e);
                    vec![None; systems.len()]
                }
            });
        failure.map_or(Ok(solution), Err)
    }

    /// Every system's right-hand side in one column, solved on the device;
    /// each system's angles, `None` where it is not factored or not finite.
    fn solve_column(&self, systems: &[DcSystem<'_>]) -> Result<Vec<Option<Vec<f64>>>, BatchError> {
        let n = self.n;
        let row = |bus: usize| -> Result<usize, BatchError> {
            usize::try_from(self.host_pos[bus]).map_err(|_| {
                BatchError::Linear(LinearError::Shape(format!(
                    "bus {bus} is an unknown of the CPU's DC solve but not of the device's"
                )))
            })
        };
        let mut rhs = vec![0.0; n];
        for sys in systems {
            for (&bus, &v) in sys.unknown.iter().zip(&sys.rhs) {
                rhs[row(bus)?] = v;
            }
        }
        let mut theta = vec![0.0; n];
        if n > 0 {
            // SAFETY: `rhs` and `theta` hold `n × capacity` values, at least
            // `n`; the copy back is waited for before `theta` is read.
            unsafe {
                crate::cuda::copy_to_device(self.rhs.ptr, &rhs, self.stream.0)?;
                let (x, b) = (
                    Dense::new(n, 1, self.theta.ptr)?,
                    Dense::new(n, 1, self.rhs.ptr)?,
                );
                self.sys
                    .execute(ffi::CUDSS_PHASE_SOLVE, x.0, b.0, "SOLVE")?;
                copy_async_to_host(theta.as_mut_ptr(), self.theta.ptr, n, self.stream.0)?;
                cuda(
                    ffi::cudaStreamSynchronize(self.stream.0),
                    "cudaStreamSynchronize",
                )?;
            }
        }
        systems
            .iter()
            .map(|sys| {
                if !sys.factored {
                    return Ok(None);
                }
                let angles = sys
                    .unknown
                    .iter()
                    .map(|&bus| row(bus).map(|r| theta[r]))
                    .collect::<Result<Vec<f64>, _>>()?;
                Ok(angles.iter().all(|a| a.is_finite()).then_some(angles))
            })
            .collect()
    }

    /// The base case of `buses` solved on the CPU, its angles and flows on
    /// the device; its islands.
    fn upload_base(
        &mut self,
        buses: &[Bus],
    ) -> Result<Vec<gridoxide_core::linear::DcIslandReport>, BatchError> {
        let mut solved_buses = buses.to_vec();
        let base = self.base.solve(&mut solved_buses);
        let base_angles: Vec<f64> = solved_buses.iter().map(|b| b.voltage_ang).collect();
        if self.n_bus > 0 {
            self.base_angles.upload(&base_angles, self.stream.0)?;
        }
        if self.n_br > 0 {
            self.base_flows.upload(&base.branch_p, self.stream.0)?;
        }
        Ok(base.islands)
    }

    /// `total` columns in chunks of `capacity`, pipelined over the two
    /// slots: `compute(first, k, slot)` queues chunk `first..first + k` on
    /// the session's stream, writing its rows into the slot, and its rows
    /// reach the result while the next chunk is computed. `solved` is
    /// copied back only `with_solved`; otherwise the returned flags are all
    /// false.
    fn pipelined(
        &self,
        total: usize,
        with_solved: bool,
        mut compute: impl FnMut(usize, usize, &Slot) -> Result<(), BatchError>,
    ) -> Result<Rows, BatchError> {
        let (n_bus, n_br) = (self.n_bus, self.n_br);
        let mut angles = vec![0.0; total * n_bus];
        let mut flows = vec![0.0; total * n_br];
        let mut solved = vec![false; total];
        let cap = self.capacity;
        std::thread::scope(|scope| -> Result<(), BatchError> {
            let mut pending: [Option<std::thread::ScopedJoinHandle<'_, Result<(), LinearError>>>;
                2] = [None, None];
            let rows = angles
                .chunks_mut(cap * n_bus.max(1))
                .zip(flows.chunks_mut(cap * n_br.max(1)))
                .zip(solved.chunks_mut(cap));
            for (i, ((dst_angles, dst_flows), dst_solved)) in rows.enumerate() {
                let (first, k) = (i * cap, dst_solved.len());
                let slot = &self.slots[i % 2];
                // The slot's pinned buffers are free once chunk i − 2's host
                // copy has run, and its device buffers once that chunk's
                // device-to-host copy has (which the host copy waited for).
                if let Some(h) = pending[i % 2].take() {
                    h.join().expect("a host copy panicked")?;
                }
                compute(first, k, slot)?;
                // SAFETY: every buffer holds `capacity` columns, at least
                // `k`; the slot is free (above), and each stream orders its
                // own work, the events the work between them.
                unsafe {
                    cuda(
                        ffi::cudaEventRecord(slot.computed.0, self.stream.0),
                        "cudaEventRecord",
                    )?;
                    cuda(
                        ffi::cudaStreamWaitEvent(self.copy_stream.0, slot.computed.0, 0),
                        "cudaStreamWaitEvent",
                    )?;
                    let cs = self.copy_stream.0;
                    if with_solved {
                        copy_async_to_host(slot.host_solved.as_mut_ptr(), slot.solved.ptr, k, cs)?;
                    }
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
                    if with_solved {
                        for (s, f) in dst_solved.iter_mut().zip(host_solved.prefix(k)) {
                            *s = *f != 0;
                        }
                    }
                    Ok(())
                }));
            }
            for h in pending.iter_mut().filter_map(Option::take) {
                h.join().expect("a host copy panicked")?;
            }
            Ok(())
        })?;
        Ok((angles, flows, solved))
    }

    /// A sweep's kernels and solve for the first `k` columns, whose `count`
    /// injection entries are in `self.injections`.
    fn run_sweep_chunk(&self, k: usize, count: usize, slot: &Slot) -> Result<(), BatchError> {
        let (n, s) = (self.n, self.stream.0.cast());
        // SAFETY: every buffer is sized for `capacity` columns, at least `k`,
        // and the injection buffers for at least `count` entries.
        unsafe {
            if n > 0 {
                launched(
                    ffi::go_dc_injection_rhs(
                        self.injections.col.ptr,
                        self.injections.row.ptr,
                        self.injections.val.ptr,
                        count as u32,
                        self.rhs.ptr,
                        n as u32,
                        k as u32,
                        s,
                    ),
                    "go_dc_injection_rhs",
                )?;
                let (x, b) = (
                    Dense::new(n, k, self.theta.ptr)?,
                    Dense::new(n, k, self.rhs.ptr)?,
                );
                self.sys
                    .execute(ffi::CUDSS_PHASE_SOLVE, x.0, b.0, "SOLVE")?;
            }
            launched(
                ffi::go_dc_sweep_angles(
                    self.theta.ptr,
                    self.pos.ptr,
                    self.base_angles.ptr,
                    slot.angles.ptr,
                    n as u32,
                    self.n_bus as u32,
                    k as u32,
                    s,
                ),
                "go_dc_sweep_angles",
            )?;
            launched(
                ffi::go_dc_sweep_flows(
                    self.theta.ptr,
                    self.br_from.ptr,
                    self.br_to.ptr,
                    self.br_b.ptr,
                    self.pos.ptr,
                    self.base_flows.ptr,
                    slot.flows.ptr,
                    n as u32,
                    self.n_br as u32,
                    k as u32,
                    s,
                ),
                "go_dc_sweep_flows",
            )?;
        }
        Ok(())
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

/// A bus's DC injection: `p_spec` and its ZIP terms at `|V| = 1`, as
/// gridoxide-core's DC solve takes it.
fn dc_injection(bus: &Bus) -> f64 {
    bus.p_spec + bus.zip_terms.iter().map(|z| z.s_const.re).sum::<f64>()
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
