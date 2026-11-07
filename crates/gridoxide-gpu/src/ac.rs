//! The AC-OPF model on the device: [`DeviceAcModel`] evaluates gridoxide's
//! `opf::ac_kernels` (constraints, Jacobian, Lagrangian Hessian) with the
//! kernels of `cuda/opf.cu`, one thread per element, from the same flat
//! arrays, handed over as [`AcModelData`] (this crate cannot see gridoxide's
//! types; gridoxide builds it, `AcOpf::device_model`).
//!
//! Each element writes its slot values in the CPU's slot order, and the
//! CPU's merge map sums them into the unique entries (a gather per entry,
//! `go_kkt_gather`), so the values come out in the order of the CPU's
//! triplets. They match the CPU's to rounding: the kernels are compiled
//! without FMA contraction, so only the sines and cosines differ (CUDA's and
//! the host's libm round differently in the last place).

use crate::cuda::{DeviceBuffer, Stream, copy_to_device, cuda, ffi};
use crate::uniform::LinearError;

/// Per unique entry of a merged list, its slots in ascending order:
/// `sources[start[u]..start[u + 1]]`, over `slots` slot values.
#[derive(Clone, Debug, Default)]
pub struct Gather {
    pub slots: usize,
    pub start: Vec<usize>,
    pub sources: Vec<usize>,
}

/// The AC-OPF model as `opf::ac_kernels` holds it: see `cuda/opf.h` for the
/// layout. Complex values as `[re, im]`.
#[derive(Clone, Debug, Default)]
pub struct AcModelData {
    pub n: usize,
    pub theta: usize,
    pub vm: usize,
    pub pg: usize,
    pub qg: usize,
    pub near: Vec<usize>,
    pub far: Vec<usize>,
    pub y_self: Vec<[f64; 2]>,
    pub y_mut: Vec<[f64; 2]>,
    pub shunt: Vec<[f64; 2]>,
    pub p_load: Vec<f64>,
    pub q_load: Vec<f64>,
    pub term_ptr: Vec<usize>,
    pub term_idx: Vec<usize>,
    pub gen_ptr: Vec<usize>,
    pub gen_idx: Vec<usize>,
    pub gen_bus: Vec<usize>,
    pub quadratic: Vec<(usize, f64)>,
    pub limit_terminal: Vec<usize>,
    pub angle: Vec<(usize, usize)>,
    pub cost: Vec<(usize, usize, f64)>,
    /// Branch flows: the first flow column and the first definition row.
    pub flow: Option<(usize, usize)>,
    pub n_vars: usize,
    pub n_rows: usize,
    pub jacobian: Gather,
    pub hessian: Gather,
}

/// The model on the device; see the module documentation.
pub struct DeviceAcModel {
    // First: queued work uses the buffers below.
    stream: Stream,
    model: ffi::go_ac_model,
    n_rows: usize,
    n_vars: usize,
    _data: Vec<DeviceBuffer<u32>>,
    _values: Vec<DeviceBuffer<f64>>,
    x: DeviceBuffer<f64>,
    y: DeviceBuffer<f64>,
    flows: DeviceBuffer<f64>,
    rows: DeviceBuffer<f64>,
    jacobian: Merged,
    hessian: Merged,
}

// SAFETY: the model owns its device buffers and stream; CUDA's device
// pointers and streams may be used from any host thread of the process.
// `&mut self` on every evaluation keeps them to one thread at a time.
unsafe impl Send for DeviceAcModel {}

/// A merged list on the device: its slot values, the gather, the values.
struct Merged {
    unique: usize,
    slots: DeviceBuffer<f64>,
    start: DeviceBuffer<u32>,
    sources: DeviceBuffer<u32>,
    values: DeviceBuffer<f64>,
}

fn index(i: usize) -> Result<u32, LinearError> {
    u32::try_from(i).map_err(|_| LinearError::Shape(format!("index {i} beyond 32 bits")))
}

fn launched(err: i32, what: &str) -> Result<(), LinearError> {
    cuda(err as ffi::cudaError_t, what)
}

impl Merged {
    fn new(g: &Gather, stream: ffi::cudaStream_t) -> Result<Self, LinearError> {
        let unique = g.start.len().saturating_sub(1);
        let to = |v: &[usize]| -> Result<Vec<u32>, LinearError> {
            v.iter().map(|&i| index(i)).collect()
        };
        Ok(Self {
            unique,
            slots: DeviceBuffer::new(g.slots.max(1))?,
            start: upload(&to(&g.start)?, stream)?,
            sources: upload(&to(&g.sources)?, stream)?,
            values: DeviceBuffer::new(unique.max(1))?,
        })
    }

    /// Sums the slot values into the unique entries and copies them back.
    fn gather(&mut self, stream: ffi::cudaStream_t) -> Result<Vec<f64>, LinearError> {
        // SAFETY: device buffers of the gather's sizes, made in `new`.
        unsafe {
            launched(
                ffi::go_kkt_gather(
                    self.start.ptr,
                    self.sources.ptr,
                    self.slots.ptr,
                    self.values.ptr,
                    self.unique as u32,
                    stream.cast(),
                ),
                "go_kkt_gather",
            )?;
        }
        let mut out = vec![0.0; self.values.len()];
        self.values.download(&mut out, stream)?;
        out.truncate(self.unique);
        Ok(out)
    }
}

fn upload<T: Copy>(v: &[T], stream: ffi::cudaStream_t) -> Result<DeviceBuffer<T>, LinearError> {
    let d = DeviceBuffer::new(v.len().max(1))?;
    if !v.is_empty() {
        // SAFETY: `d` holds at least v.len() values.
        unsafe { copy_to_device(d.ptr, v, stream)? };
    }
    Ok(d)
}

impl DeviceAcModel {
    pub fn new(data: &AcModelData) -> Result<Self, LinearError> {
        let stream = Stream::new()?;
        let s = stream.0;
        let mut ints: Vec<DeviceBuffer<u32>> = Vec::new();
        let mut reals: Vec<DeviceBuffer<f64>> = Vec::new();
        let mut int = |v: Vec<u32>| -> Result<*const u32, LinearError> {
            let d = upload(&v, s)?;
            let p = d.ptr as *const u32;
            ints.push(d);
            Ok(p)
        };
        let ix = |v: &[usize]| -> Result<Vec<u32>, LinearError> {
            v.iter().map(|&i| index(i)).collect()
        };
        let near = int(ix(&data.near)?)?;
        let far = int(ix(&data.far)?)?;
        let term_ptr = int(ix(&data.term_ptr)?)?;
        let term_idx = int(ix(&data.term_idx)?)?;
        let gen_ptr = int(ix(&data.gen_ptr)?)?;
        let gen_idx = int(ix(&data.gen_idx)?)?;
        let gen_bus = int(ix(&data.gen_bus)?)?;
        let limit_term = int(ix(&data.limit_terminal)?)?;
        let angle_from = int(ix(&data.angle.iter().map(|a| a.0).collect::<Vec<_>>())?)?;
        let angle_to = int(ix(&data.angle.iter().map(|a| a.1).collect::<Vec<_>>())?)?;
        let cost_epi = int(ix(&data.cost.iter().map(|c| c.0).collect::<Vec<_>>())?)?;
        let cost_col = int(ix(&data.cost.iter().map(|c| c.1).collect::<Vec<_>>())?)?;
        let mut real = |v: Vec<f64>| -> Result<*const f64, LinearError> {
            let d = upload(&v, s)?;
            let p = d.ptr as *const f64;
            reals.push(d);
            Ok(p)
        };
        let flat = |v: &[[f64; 2]]| v.iter().flatten().copied().collect::<Vec<f64>>();
        let y_self = real(flat(&data.y_self))?;
        let y_mut = real(flat(&data.y_mut))?;
        let shunt = real(flat(&data.shunt))?;
        let p_load = real(data.p_load.clone())?;
        let q_load = real(data.q_load.clone())?;
        let quad_h = real(data.quadratic.iter().map(|q| q.1).collect())?;
        let cost_slope = real(data.cost.iter().map(|c| c.2).collect())?;
        let (flow_col, flow_row) = data.flow.unwrap_or((0, 0));
        let model = ffi::go_ac_model {
            n: index(data.n)?,
            theta: index(data.theta)?,
            vm: index(data.vm)?,
            pg: index(data.pg)?,
            qg: index(data.qg)?,
            n_term: index(data.near.len())?,
            n_gen: index(data.gen_bus.len())?,
            n_quad: index(data.quadratic.len())?,
            n_limit: index(data.limit_terminal.len())?,
            n_angle: index(data.angle.len())?,
            n_cost: index(data.cost.len())?,
            has_flow: data.flow.is_some() as i32,
            flow_col: index(flow_col)?,
            flow_row: index(flow_row)?,
            near,
            far,
            y_self,
            y_mut,
            shunt,
            p_load,
            q_load,
            term_ptr,
            term_idx,
            gen_ptr,
            gen_idx,
            gen_bus,
            quad_h,
            limit_term,
            angle_from,
            angle_to,
            cost_epi,
            cost_col,
            cost_slope,
        };
        Ok(Self {
            model,
            n_rows: data.n_rows,
            n_vars: data.n_vars,
            _data: ints,
            _values: reals,
            x: DeviceBuffer::new(data.n_vars.max(1))?,
            y: DeviceBuffer::new(data.n_rows.max(1))?,
            flows: DeviceBuffer::new((2 * data.near.len()).max(1))?,
            rows: DeviceBuffer::new(data.n_rows.max(1))?,
            jacobian: Merged::new(&data.jacobian, s)?,
            hessian: Merged::new(&data.hessian, s)?,
            stream,
        })
    }

    fn put_x(&mut self, x: &[f64]) -> Result<(), LinearError> {
        if x.len() != self.n_vars {
            return Err(LinearError::Shape(format!(
                "x has {} entries, the model {}",
                x.len(),
                self.n_vars
            )));
        }
        // SAFETY: `x` on the device holds n_vars values.
        unsafe { copy_to_device(self.x.ptr, x, self.stream.0) }
    }

    /// Every row's value at `x`.
    pub fn constraints(&mut self, x: &[f64]) -> Result<Vec<f64>, LinearError> {
        self.put_x(x)?;
        let s = self.stream.0;
        // SAFETY: the model's pointers are the buffers it owns; the vectors
        // are its own of the model's sizes.
        unsafe {
            launched(
                ffi::go_ac_flows(&self.model, self.x.ptr, self.flows.ptr, s.cast()),
                "go_ac_flows",
            )?;
            launched(
                ffi::go_ac_constraints(
                    &self.model,
                    self.x.ptr,
                    self.flows.ptr,
                    self.rows.ptr,
                    s.cast(),
                ),
                "go_ac_constraints",
            )?;
        }
        let mut out = vec![0.0; self.rows.len()];
        self.rows.download(&mut out, s)?;
        out.truncate(self.n_rows);
        Ok(out)
    }

    /// The Jacobian's values at `x`, in the order of the CPU's triplets.
    pub fn jacobian_values(&mut self, x: &[f64]) -> Result<Vec<f64>, LinearError> {
        self.put_x(x)?;
        let s = self.stream.0;
        // SAFETY: as in `constraints`; the slot buffer holds the Jacobian's
        // slot count.
        unsafe {
            launched(
                ffi::go_ac_jacobian_slots(
                    &self.model,
                    self.x.ptr,
                    self.jacobian.slots.ptr,
                    s.cast(),
                ),
                "go_ac_jacobian_slots",
            )?;
        }
        self.jacobian.gather(s)
    }

    /// The Lagrangian Hessian's values at `(x, y)`, in the order of the
    /// CPU's triplets.
    pub fn hessian_values(&mut self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, LinearError> {
        self.put_x(x)?;
        if y.len() != self.n_rows {
            return Err(LinearError::Shape(format!(
                "y has {} entries, the model {}",
                y.len(),
                self.n_rows
            )));
        }
        let s = self.stream.0;
        // SAFETY: as in `jacobian_values`; `y` on the device holds n_rows
        // values.
        unsafe {
            copy_to_device(self.y.ptr, y, s)?;
            launched(
                ffi::go_ac_hessian_slots(
                    &self.model,
                    self.x.ptr,
                    self.y.ptr,
                    self.hessian.slots.ptr,
                    s.cast(),
                ),
                "go_ac_hessian_slots",
            )?;
        }
        self.hessian.gather(s)
    }
}
