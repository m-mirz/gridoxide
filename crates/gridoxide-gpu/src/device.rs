//! A batch's network and Newton state on the device: [`DeviceBatch`].
//!
//! Everything the Newton loop computes per scenario between two linear
//! solves runs here in kernels
//! (`cuda/kernels.cu`): the voltages' sines and cosines, the power
//! injections, the mismatch and its ‖F‖∞, the Jacobian fill in its two
//! passes, and the update. Each transliterates the CPU code the stand-in loop
//! ([`newton`](crate::newton)) runs, so the device takes the same steps.
//!
//! Network data are uploaded once: the Y-bus in CSR form, the bus lists, the
//! ZIP terms summed per kind, and the Jacobian pattern's fill tables
//! ([`csr_jacobian`](crate::newton::csr_jacobian), CSR order). Per scenario,
//! system-major as everywhere in this crate, the device holds the bus state
//! (magnitudes, angles, their sines and cosines, specified and calculated
//! injections), the mismatch and its norm, the fill slots and the Jacobian
//! values, and the update.
//!
//! Every step is masked by [`set_active`](DeviceBatch::set_active): an
//! inactive scenario's buffers are left as they were. The getters copy
//! device buffers back for the tests and for the host side of the loop; the
//! kernels themselves never wait on the host.
//!
//! N-1 ([`contingencies`](DeviceBatch::contingencies)) keeps the shared
//! Y-bus and gives each outage its changed values, as the CPU loop does: two
//! more kernels then recompute only the injections of the rows and the fill
//! slots those values touch, the rows summed in the same column order as
//! the CPU sums its Y-bus changed in place.
//!
//! Not yet here: handing the values and the mismatch to cuDSS on the device
//! (phase 3).

use gridoxide_core::jacobian::JacobianPattern;
use gridoxide_core::network::YBusSparse;
use gridoxide_core::scenario::Scenario;
use gridoxide_core::types::{Bus, BusType, ZipKind};

use std::collections::{BTreeSet, HashMap};

use gridoxide_core::network::ShuntAdm;
use gridoxide_core::types::{Line, Transformer};

use crate::cuda::{DeviceBuffer, Stream, cuda, ffi};
use crate::newton::{BatchError, Contingencies, Delta, check_buses, contingencies, csr_jacobian};
use crate::uniform::{CsrPattern, LinearError};

/// See the module documentation.
pub struct DeviceBatch {
    // First, so it is dropped (synchronized) before the buffers are freed.
    stream: Stream,
    n_bus: usize,
    n_ns: usize,
    n_pq: usize,
    nnz: usize,
    n_diag: usize,
    n_pairs: usize,
    batch: usize,
    pattern: JacobianPattern,
    csr: CsrPattern,

    // The network, shared by every scenario.
    y_row_ptr: DeviceBuffer<u32>,
    y_col: DeviceBuffer<u32>,
    y: DeviceBuffer<f64>,
    non_slack: DeviceBuffer<u32>,
    pq: DeviceBuffer<u32>,
    zip: DeviceBuffer<f64>,
    diag_bus: DeviceBuffer<u32>,
    diag_y: DeviceBuffer<f64>,
    pair_bus: DeviceBuffer<u32>,
    pair_y: DeviceBuffer<f64>,
    src: DeviceBuffer<u32>,

    // N-1's changed Y-bus values and what they touch; empty without outages.
    ovr_ptr: DeviceBuffer<u32>,
    ovr_pos: DeviceBuffer<u32>,
    ovr_y: DeviceBuffer<f64>,
    n_rows: usize,
    row_scn: DeviceBuffer<u32>,
    row_bus: DeviceBuffer<u32>,
    n_slot_items: usize,
    slot_scn: DeviceBuffer<u32>,
    slot_owner: DeviceBuffer<u32>,
    slot_y: DeviceBuffer<f64>,

    // Per scenario.
    active: DeviceBuffer<i32>,
    vm: DeviceBuffer<f64>,
    va: DeviceBuffer<f64>,
    sin_va: DeviceBuffer<f64>,
    cos_va: DeviceBuffer<f64>,
    p_spec: DeviceBuffer<f64>,
    q_spec: DeviceBuffer<f64>,
    p: DeviceBuffer<f64>,
    q: DeviceBuffer<f64>,
    rhs: DeviceBuffer<f64>,
    norm: DeviceBuffer<f64>,
    slots: DeviceBuffer<f64>,
    values: DeviceBuffer<f64>,
    dx: DeviceBuffer<f64>,

    // On the host, for loading scenarios.
    start: Vec<Bus>,
    host_row_ptr: Vec<usize>,
    host_col: Vec<usize>,
    diag_owner: Vec<Option<usize>>,
    pair_owner: HashMap<(u32, u32), usize>,
}

fn u32s(v: impl IntoIterator<Item = usize>, what: &str) -> Result<Vec<u32>, LinearError> {
    v.into_iter()
        .map(|x| {
            u32::try_from(x)
                .map_err(|_| LinearError::Shape(format!("{what} does not fit the kernels' u32")))
        })
        .collect()
}

fn upload<T: Copy>(src: &[T], stream: ffi::cudaStream_t) -> Result<DeviceBuffer<T>, LinearError> {
    let mut buf = DeviceBuffer::new(src.len())?;
    buf.upload(src, stream)?;
    Ok(buf)
}

impl DeviceBatch {
    /// Uploads the network of `start` and `ybus` and one state per scenario:
    /// `start`'s voltages and injections with the scenario's overrides, as
    /// [`solve_batch`](crate::newton::solve_batch) starts them, and refuses
    /// the same scenarios it does. Every scenario starts active.
    pub fn new(
        start: &[Bus],
        ybus: &YBusSparse,
        scenarios: &[Scenario],
    ) -> Result<Self, BatchError> {
        check_buses(start, scenarios)?;
        if let Some(scenario) = scenarios
            .iter()
            .position(|sc| !sc.branch_outages.is_empty())
        {
            return Err(BatchError::OutagesUnsupported { scenario });
        }
        let mut dev = Self::with_capacity(start, ybus, scenarios.len())?;
        dev.load(scenarios, &vec![Delta::default(); scenarios.len()])?;
        Ok(dev)
    }

    /// N-1 as [`solve_contingency_batch`](crate::newton::solve_contingency_batch)
    /// batches it: each scenario's outages taken off the intact network of
    /// `lines`, `transformers` and `shunts`, on top of its overrides. Returns
    /// the batch and, per batch scenario, its index in `scenarios`: an outage
    /// that splits the network is left out.
    pub fn contingencies(
        start: &[Bus],
        lines: &[Line],
        transformers: &[Transformer],
        shunts: &[ShuntAdm],
        scenarios: &[Scenario],
    ) -> Result<(Self, Vec<usize>), BatchError> {
        let Contingencies {
            ybus,
            members,
            deltas,
        } = contingencies(start, lines, transformers, shunts, scenarios)?;
        let batch: Vec<Scenario> = members.iter().map(|&s| scenarios[s].clone()).collect();
        let mut dev = Self::with_capacity(start, &ybus, batch.len())?;
        dev.load(&batch, &deltas)?;
        Ok((dev, members))
    }

    /// The network of `start` and `ybus` and room for `capacity` scenarios,
    /// none loaded: every scenario holds `start`'s state and is inactive.
    pub(crate) fn with_capacity(
        start: &[Bus],
        ybus: &YBusSparse,
        capacity: usize,
    ) -> Result<Self, LinearError> {
        let n_bus = start.len();
        let batch = capacity;
        let (pattern, csr) = csr_jacobian(start, ybus);
        let non_slack: Vec<usize> = start
            .iter()
            .filter(|b| b.bus_type != BusType::Slack)
            .map(|b| b.idx)
            .collect();
        let pq: Vec<usize> = start
            .iter()
            .filter(|b| b.bus_type == BusType::PQ)
            .map(|b| b.idx)
            .collect();
        let nnz = csr.nnz();
        let n_diag = pattern.diag_slots().len();
        let n_pairs = pattern.pair_slots().len();
        u32s(
            [n_bus * batch, nnz * batch, 4 * (n_diag + n_pairs) * batch],
            "the batch",
        )?;

        let mut y_row_ptr = vec![0usize; n_bus + 1];
        let mut y_col = Vec::new();
        let mut y = Vec::new();
        for i in 0..n_bus {
            for &(k, v) in ybus.row(i) {
                y_col.push(k);
                y.extend([v.re, v.im]);
            }
            y_row_ptr[i + 1] = y_col.len();
        }
        let pair_owner: HashMap<(u32, u32), usize> = pattern
            .pair_slots()
            .iter()
            .enumerate()
            .map(|(p, &(i, k, _))| ((i, k), n_diag + p))
            .collect();
        let mut diag_owner = vec![None; n_bus];
        for (d, &(i, _)) in pattern.diag_slots().iter().enumerate() {
            diag_owner[i as usize] = Some(d);
        }
        let mut zip = vec![0.0; 6 * n_bus];
        for b in start {
            for zt in &b.zip_terms {
                let at = 6 * b.idx
                    + match zt.kind {
                        ZipKind::ConstPower => 0,
                        ZipKind::ConstImpedance => 2,
                        ZipKind::ConstCurrent => 4,
                    };
                zip[at] += zt.s_const.re;
                zip[at + 1] += zt.s_const.im;
            }
        }

        let stream = Stream::new()?;
        let n = non_slack.len() + pq.len();
        let n_slots = 4 * (n_diag + n_pairs);
        let s = stream.0;
        let state = |f: fn(&Bus) -> f64| -> Vec<f64> {
            std::iter::repeat_n(start.iter().map(f), batch)
                .flatten()
                .collect()
        };
        Ok(Self {
            n_bus,
            n_ns: non_slack.len(),
            n_pq: pq.len(),
            nnz,
            n_diag,
            n_pairs,
            batch,
            stream,
            y_row_ptr: upload(&u32s(y_row_ptr.iter().copied(), "the Y-bus")?, s)?,
            y_col: upload(&u32s(y_col.iter().copied(), "the Y-bus")?, s)?,
            y: upload(&y, s)?,
            non_slack: upload(&u32s(non_slack, "a bus index")?, s)?,
            pq: upload(&u32s(pq, "a bus index")?, s)?,
            zip: upload(&zip, s)?,
            diag_bus: upload(
                &pattern
                    .diag_slots()
                    .iter()
                    .map(|&(i, _)| i)
                    .collect::<Vec<_>>(),
                s,
            )?,
            diag_y: upload(
                &pattern
                    .diag_slots()
                    .iter()
                    .flat_map(|&(_, y)| [y.re, y.im])
                    .collect::<Vec<_>>(),
                s,
            )?,
            pair_bus: upload(
                &pattern
                    .pair_slots()
                    .iter()
                    .flat_map(|&(i, k, _)| [i, k])
                    .collect::<Vec<_>>(),
                s,
            )?,
            pair_y: upload(
                &pattern
                    .pair_slots()
                    .iter()
                    .flat_map(|&(_, _, y)| [y.re, y.im])
                    .collect::<Vec<_>>(),
                s,
            )?,
            src: upload(pattern.slot_sources(), s)?,
            ovr_ptr: upload(&vec![0u32; batch + 1], s)?,
            ovr_pos: DeviceBuffer::new(0)?,
            ovr_y: DeviceBuffer::new(0)?,
            n_rows: 0,
            row_scn: DeviceBuffer::new(0)?,
            row_bus: DeviceBuffer::new(0)?,
            n_slot_items: 0,
            slot_scn: DeviceBuffer::new(0)?,
            slot_owner: DeviceBuffer::new(0)?,
            slot_y: DeviceBuffer::new(0)?,
            active: upload(&vec![0; batch], s)?,
            vm: upload(&state(|b| b.voltage_mag), s)?,
            va: upload(&state(|b| b.voltage_ang), s)?,
            sin_va: DeviceBuffer::new(n_bus * batch)?,
            cos_va: DeviceBuffer::new(n_bus * batch)?,
            p_spec: upload(&state(|b| b.p_spec), s)?,
            q_spec: upload(&state(|b| b.q_spec), s)?,
            p: DeviceBuffer::new(n_bus * batch)?,
            q: DeviceBuffer::new(n_bus * batch)?,
            rhs: DeviceBuffer::new(n * batch)?,
            norm: DeviceBuffer::new(batch)?,
            slots: DeviceBuffer::new(n_slots * batch)?,
            values: DeviceBuffer::new(nnz * batch)?,
            dx: DeviceBuffer::new(n * batch)?,
            start: start.to_vec(),
            host_row_ptr: y_row_ptr,
            host_col: y_col,
            diag_owner,
            pair_owner,
            pattern,
            csr,
        })
    }

    /// Loads `scenarios` (at most the capacity) into the first slots, each
    /// from `start`'s state with its overrides and, for N-1, its `delta` on
    /// the shared Y-bus; they become the active ones. The slots after them
    /// get `start`'s state back and stay inactive. The caller has checked
    /// the scenarios.
    pub(crate) fn load(
        &mut self,
        scenarios: &[Scenario],
        deltas: &[Delta],
    ) -> Result<(), LinearError> {
        if scenarios.len() > self.batch || deltas.len() != scenarios.len() {
            return Err(LinearError::Shape(format!(
                "{} scenarios with {} deltas for a batch of {}",
                scenarios.len(),
                deltas.len(),
                self.batch
            )));
        }
        let (n_bus, batch, s) = (self.n_bus, self.batch, self.stream.0);
        let mut vm = Vec::with_capacity(n_bus * batch);
        let mut va = Vec::with_capacity(n_bus * batch);
        let mut p_spec = Vec::with_capacity(n_bus * batch);
        let mut q_spec = Vec::with_capacity(n_bus * batch);
        for k in 0..batch {
            let first = vm.len();
            for b in &self.start {
                vm.push(b.voltage_mag);
                va.push(b.voltage_ang);
                p_spec.push(b.p_spec);
                q_spec.push(b.q_spec);
            }
            for ov in scenarios.get(k).map_or(&[][..], |sc| &sc.bus_overrides) {
                let at = first + ov.bus;
                p_spec[at] = ov.p_spec.unwrap_or(p_spec[at]);
                q_spec[at] = ov.q_spec.unwrap_or(q_spec[at]);
                vm[at] = ov.voltage_mag.unwrap_or(vm[at]);
            }
        }
        self.vm.upload(&vm, s)?;
        self.va.upload(&va, s)?;
        self.p_spec.upload(&p_spec, s)?;
        self.q_spec.upload(&q_spec, s)?;

        // Each outage's changed values by Y-bus position, the rows they are
        // in, and the fill-slot owners they change.
        let (mut ovr_ptr, mut ovr_pos, mut ovr_y) = (vec![0usize], Vec::new(), Vec::new());
        let (mut row_scn, mut row_bus) = (Vec::new(), Vec::new());
        let (mut slot_scn, mut slot_owner, mut slot_y) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..batch {
            let mut rows = BTreeSet::new();
            if let Some(delta) = deltas.get(k) {
                for (&(r, c), &v) in delta.at.iter().zip(&delta.outaged) {
                    let row = self.host_row_ptr[r]..self.host_row_ptr[r + 1];
                    let at = self.host_col[row.clone()]
                        .binary_search(&c)
                        .expect("the intact Y-bus stores every branch's entries");
                    ovr_pos.push(row.start + at);
                    ovr_y.extend([v.re, v.im]);
                    rows.insert(r);
                    let owner = if r == c {
                        self.diag_owner[r]
                    } else {
                        self.pair_owner.get(&(r as u32, c as u32)).copied()
                    };
                    if let Some(o) = owner {
                        slot_scn.push(k);
                        slot_owner.push(o);
                        slot_y.extend([v.re, v.im]);
                    }
                }
            }
            ovr_ptr.push(ovr_pos.len());
            row_scn.extend(std::iter::repeat_n(k, rows.len()));
            row_bus.extend(rows);
        }
        self.ovr_ptr.upload(&u32s(ovr_ptr, "N-1's changes")?, s)?;
        self.ovr_pos = upload(&u32s(ovr_pos, "N-1's changes")?, s)?;
        self.ovr_y = upload(&ovr_y, s)?;
        self.n_rows = row_bus.len();
        self.row_scn = upload(&u32s(row_scn, "N-1's changes")?, s)?;
        self.row_bus = upload(&u32s(row_bus, "N-1's changes")?, s)?;
        self.n_slot_items = slot_owner.len();
        self.slot_scn = upload(&u32s(slot_scn, "N-1's changes")?, s)?;
        self.slot_owner = upload(&u32s(slot_owner, "N-1's changes")?, s)?;
        self.slot_y = upload(&slot_y, s)?;

        let active: Vec<bool> = (0..batch).map(|k| k < scenarios.len()).collect();
        self.set_active(&active)
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Unknowns per scenario: the length of its mismatch and update.
    pub fn n(&self) -> usize {
        self.n_ns + self.n_pq
    }

    /// The Jacobian's pattern in the order its values are written.
    pub fn csr(&self) -> &CsrPattern {
        &self.csr
    }

    /// The pattern the fill tables came from, in the same order.
    pub fn jacobian_pattern(&self) -> &JacobianPattern {
        &self.pattern
    }

    /// Selects the scenarios the next steps compute; the others' buffers are
    /// left as they were.
    pub fn set_active(&mut self, active: &[bool]) -> Result<(), LinearError> {
        self.check("active", active.len(), 1)?;
        let flags: Vec<i32> = active.iter().map(|&a| i32::from(a)).collect();
        self.active.upload(&flags, self.stream.0)
    }

    /// The active scenarios' injections, mismatch and ‖F‖∞, from their
    /// present voltages.
    pub fn mismatch(&mut self) -> Result<(), LinearError> {
        let (n_bus, batch) = (self.n_bus as u32, self.batch as u32);
        // SAFETY: every pointer is a device buffer of this struct, sized for
        // the layout of kernels.h at construction.
        unsafe {
            launched(
                ffi::go_voltages(
                    self.va.ptr,
                    self.sin_va.ptr,
                    self.cos_va.ptr,
                    n_bus,
                    batch,
                    self.active.ptr,
                    self.stream.0.cast(),
                ),
                "go_voltages",
            )?;
            launched(
                ffi::go_injections(
                    self.y_row_ptr.ptr,
                    self.y_col.ptr,
                    self.y.ptr,
                    self.vm.ptr,
                    self.sin_va.ptr,
                    self.cos_va.ptr,
                    self.p.ptr,
                    self.q.ptr,
                    n_bus,
                    batch,
                    self.active.ptr,
                    self.stream.0.cast(),
                ),
                "go_injections",
            )?;
            launched(
                ffi::go_injections_delta(
                    self.y_row_ptr.ptr,
                    self.y_col.ptr,
                    self.y.ptr,
                    self.ovr_ptr.ptr,
                    self.ovr_pos.ptr,
                    self.ovr_y.ptr,
                    self.row_scn.ptr,
                    self.row_bus.ptr,
                    self.n_rows as u32,
                    self.vm.ptr,
                    self.sin_va.ptr,
                    self.cos_va.ptr,
                    self.p.ptr,
                    self.q.ptr,
                    n_bus,
                    self.active.ptr,
                    self.stream.0.cast(),
                ),
                "go_injections_delta",
            )?;
            launched(
                ffi::go_mismatch(
                    self.non_slack.ptr,
                    self.pq.ptr,
                    self.n_ns as u32,
                    self.n_pq as u32,
                    self.p_spec.ptr,
                    self.q_spec.ptr,
                    self.zip.ptr,
                    self.vm.ptr,
                    self.p.ptr,
                    self.q.ptr,
                    self.rhs.ptr,
                    self.norm.ptr,
                    n_bus,
                    batch,
                    self.active.ptr,
                    self.stream.0.cast(),
                ),
                "go_mismatch",
            )
        }
    }

    /// The active scenarios' Jacobian values at their present voltages and
    /// the injections of the last [`mismatch`](Self::mismatch).
    pub fn fill_jacobian(&mut self) -> Result<(), LinearError> {
        let batch = self.batch as u32;
        // SAFETY: as in `mismatch`.
        unsafe {
            launched(
                ffi::go_jacobian_slots(
                    self.diag_bus.ptr,
                    self.diag_y.ptr,
                    self.n_diag as u32,
                    self.pair_bus.ptr,
                    self.pair_y.ptr,
                    self.n_pairs as u32,
                    self.vm.ptr,
                    self.sin_va.ptr,
                    self.cos_va.ptr,
                    self.p.ptr,
                    self.q.ptr,
                    self.slots.ptr,
                    self.n_bus as u32,
                    batch,
                    self.active.ptr,
                    self.stream.0.cast(),
                ),
                "go_jacobian_slots",
            )?;
            launched(
                ffi::go_jacobian_slots_delta(
                    self.diag_bus.ptr,
                    self.n_diag as u32,
                    self.pair_bus.ptr,
                    self.n_pairs as u32,
                    self.slot_scn.ptr,
                    self.slot_owner.ptr,
                    self.slot_y.ptr,
                    self.n_slot_items as u32,
                    self.vm.ptr,
                    self.sin_va.ptr,
                    self.cos_va.ptr,
                    self.p.ptr,
                    self.q.ptr,
                    self.slots.ptr,
                    self.n_bus as u32,
                    self.active.ptr,
                    self.stream.0.cast(),
                ),
                "go_jacobian_slots_delta",
            )?;
            launched(
                ffi::go_jacobian_gather(
                    self.src.ptr,
                    self.slots.ptr,
                    self.values.ptr,
                    self.nnz as u32,
                    (4 * (self.n_diag + self.n_pairs)) as u32,
                    batch,
                    self.active.ptr,
                    self.stream.0.cast(),
                ),
                "go_jacobian_gather",
            )
        }
    }

    /// Applies the update `dx` (`batch × n`, from the host) to the active
    /// scenarios' angles and magnitudes.
    pub fn update(&mut self, dx: &[f64]) -> Result<(), LinearError> {
        self.check("dx", dx.len(), self.n())?;
        self.dx.upload(dx, self.stream.0)?;
        self.update_in_place()
    }

    /// The update from what the device's update buffer holds: the solution
    /// cuDSS wrote there ([`linear_buffers`](Self::linear_buffers)).
    pub(crate) fn update_in_place(&mut self) -> Result<(), LinearError> {
        // SAFETY: as in `mismatch`.
        unsafe {
            launched(
                ffi::go_update(
                    self.non_slack.ptr,
                    self.pq.ptr,
                    self.n_ns as u32,
                    self.n_pq as u32,
                    self.dx.ptr,
                    self.vm.ptr,
                    self.va.ptr,
                    self.n_bus as u32,
                    self.batch as u32,
                    self.active.ptr,
                    self.stream.0.cast(),
                ),
                "go_update",
            )
        }
    }

    /// Bus voltage magnitudes and angles, `batch × n_bus` each.
    pub fn voltages(&self) -> Result<(Vec<f64>, Vec<f64>), LinearError> {
        Ok((self.get(&self.vm)?, self.get(&self.va)?))
    }

    /// Calculated injections P and Q of the last mismatch, `batch × n_bus`.
    pub fn injections(&self) -> Result<(Vec<f64>, Vec<f64>), LinearError> {
        Ok((self.get(&self.p)?, self.get(&self.q)?))
    }

    /// The last mismatch, `batch × n`, the right-hand side of the step.
    pub fn rhs(&self) -> Result<Vec<f64>, LinearError> {
        self.get(&self.rhs)
    }

    /// ‖F‖∞ per scenario from the last mismatch.
    pub fn norms(&self) -> Result<Vec<f64>, LinearError> {
        self.get(&self.norm)
    }

    /// The last fill's Jacobian values, `batch × nnz`, in [`csr`](Self::csr)'s
    /// order.
    pub fn jacobian_values(&self) -> Result<Vec<f64>, LinearError> {
        self.get(&self.values)
    }

    /// The stream every kernel runs on, for cuDSS to run on too.
    pub(crate) fn stream(&self) -> ffi::cudaStream_t {
        self.stream.0
    }

    /// The Jacobian values (`batch × nnz`), the mismatch (`batch × n`) and
    /// the update (`batch × n`): what cuDSS factors, solves and writes.
    pub(crate) fn linear_buffers(&self) -> (*mut f64, *mut f64, *mut f64) {
        (self.values.ptr, self.rhs.ptr, self.dx.ptr)
    }

    fn get(&self, buf: &DeviceBuffer<f64>) -> Result<Vec<f64>, LinearError> {
        let mut out = vec![0.0; buf.len()];
        buf.download(&mut out, self.stream.0)?;
        Ok(out)
    }

    fn check(&self, what: &str, len: usize, per_scenario: usize) -> Result<(), LinearError> {
        if len == self.batch * per_scenario {
            Ok(())
        } else {
            Err(LinearError::Shape(format!(
                "{what} has {len} entries, expected {} scenarios x {per_scenario}",
                self.batch
            )))
        }
    }
}

/// A launcher's return, `cudaGetLastError()`, as a result.
fn launched(err: i32, what: &str) -> Result<(), LinearError> {
    cuda(err as ffi::cudaError_t, what)
}
