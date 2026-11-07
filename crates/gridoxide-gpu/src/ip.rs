//! `opf::nlp`'s interior point with its vectors on the device:
//! [`DeviceInteriorPoint`] is the arithmetic of `opf::nlp`'s `HostEngine`,
//! kernel for loop (`cuda/ip.cu`), on [`DeviceAcModel`]'s evaluations and a
//! [`CudssSolver`]. `opf::nlp` keeps every decision, which it takes from the
//! scalars the methods here return; the iterate, its multipliers, the
//! Jacobian, the Hessian and the steps never reach the host. All that crosses
//! the bus during an iteration is the scalars, and `x` and the gradient,
//! because the objective stays the host's.
//!
//! Sums run in a reduction's order, not the host's, and in the same order
//! every run. Everything else rounds as the host's does, but for the
//! logarithms of φ and the model's sines and cosines.

use gridoxide_sparse::ldlt::Inertia;

use crate::ac::DeviceAcModel;
use crate::cuda::{DeviceBuffer, PinnedBuffer, cuda, ffi};
use crate::kkt::{CudssSolver, Factorization};
use crate::uniform::LinearError;

/// A problem in `opf::nlp`'s internal form, for [`DeviceInteriorPoint::new`]:
/// `n_x` variables and their slacks, `rows` rows of the problem and the rows
/// that pin its fixed variables after them.
#[derive(Clone, Debug, Default)]
pub struct InteriorPointData {
    pub n_x: usize,
    pub rows: usize,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    /// Per problem row, its lower bound, and its slack column if any.
    pub c_lower: Vec<f64>,
    pub slack_of: Vec<Option<usize>>,
    /// The pinning rows, as `(variable, value)`.
    pub pinned: Vec<(usize, f64)>,
    /// The internal Jacobian's `(row, column)`: the problem's entries, in the
    /// model's order, `n_jacobian` of them, then its constant ones.
    pub jacobian: Vec<(usize, usize)>,
    pub n_jacobian: usize,
    /// The constant entries' values, after the first `n_jacobian`.
    pub constants: Vec<f64>,
    /// The Hessian's `(row, column)`, in the model's order.
    pub hessian: Vec<(usize, usize)>,
    /// The scale of each problem row and of the objective.
    pub row_scale: Vec<f64>,
    pub objective_scale: f64,
    /// Whether cuDSS runs in its deterministic mode.
    pub deterministic: bool,
}

/// The iterate at a trial point, for the host to finish: `x`, and φ less
/// the objective's value and θ.
#[derive(Clone, Copy, Debug)]
pub struct Trial<'a> {
    pub x: &'a [f64],
    pub barrier: f64,
    pub theta: f64,
}

/// See the module documentation.
pub struct DeviceInteriorPoint {
    // First: its stream is what everything here queues on, so dropping it
    // finishes the queued work before the buffers below go.
    solver: CudssSolver,
    data: ffi::go_ip,
    n_x: usize,
    n: usize,
    m: usize,
    _ints: Vec<DeviceBuffer<u32>>,
    _signed: DeviceBuffer<i32>,
    _reals: Vec<DeviceBuffer<f64>>,
    u: DeviceBuffer<f64>,
    y: DeviceBuffer<f64>,
    z_l: DeviceBuffer<f64>,
    z_u: DeviceBuffer<f64>,
    grad: DeviceBuffer<f64>,
    r_p: DeviceBuffer<f64>,
    r_d: DeviceBuffer<f64>,
    y_model: DeviceBuffer<f64>,
    trial_u: DeviceBuffer<f64>,
    trial_r: DeviceBuffer<f64>,
    c_soc: DeviceBuffer<f64>,
    w: DeviceBuffer<f64>,
    rhs: DeviceBuffer<f64>,
    /// The Newton step and the last correction: `n` variables, `m` rows.
    newton: DeviceBuffer<f64>,
    correction: DeviceBuffer<f64>,
    /// Refinement's working solution, residual and correction.
    work: DeviceBuffer<f64>,
    residual: DeviceBuffer<f64>,
    refinement: DeviceBuffer<f64>,
    /// The two products of a residual, and their arguments.
    primal_in: DeviceBuffer<f64>,
    dual_in: DeviceBuffer<f64>,
    primal_out: DeviceBuffer<f64>,
    dual_out: DeviceBuffer<f64>,
    saved: [DeviceBuffer<f64>; 4],
    partial: DeviceBuffer<f64>,
    scalars: DeviceBuffer<f64>,
    host_scalars: PinnedBuffer<f64>,
    host_x: PinnedBuffer<f64>,
}

fn launched(err: i32, what: &str) -> Result<(), LinearError> {
    cuda(err as ffi::cudaError_t, what)
}

fn index(i: usize) -> Result<u32, LinearError> {
    u32::try_from(i).map_err(|_| LinearError::Shape(format!("index {i} beyond 32 bits")))
}

macro_rules! go {
    ($name:ident ( $($arg:expr),* $(,)? ), $stream:expr) => {
        launched(ffi::$name($($arg,)* $stream.cast()), stringify!($name))
    };
}

/// The most values a reduction's result slot holds in one fetch.
const SCALARS: usize = 8;

impl DeviceInteriorPoint {
    /// Its buffers and the KKT solver for `data`'s pattern, evaluating on
    /// `model`. cuDSS's analysis waits for the first factorization.
    pub fn new(data: &InteriorPointData, model: &DeviceAcModel) -> Result<Self, LinearError> {
        let (n_x, rows) = (data.n_x, data.rows);
        let n = data.lower.len();
        let m = rows + data.pinned.len();
        if model.n_rows() != rows || data.upper.len() != n || data.slack_of.len() != rows {
            return Err(LinearError::Shape(
                "the interior point's data do not fit the model".into(),
            ));
        }
        // The KKT matrix: the Hessian's lower triangle on the variables, the
        // diagonal, the Jacobian on the variables, the rows' diagonal.
        let h_index: Vec<usize> = (0..data.hessian.len())
            .filter(|&e| data.hessian[e].0 >= data.hessian[e].1 && data.hessian[e].0 < n_x)
            .collect();
        let a_index: Vec<usize> = (0..data.jacobian.len())
            .filter(|&e| data.jacobian[e].1 < n_x)
            .collect();
        let mut keys = Vec::with_capacity(h_index.len() + n_x + a_index.len() + m);
        keys.extend(h_index.iter().map(|&e| data.hessian[e]));
        keys.extend((0..n_x).map(|k| (k, k)));
        keys.extend(a_index.iter().map(|&e| {
            let (r, k) = data.jacobian[e];
            (n_x + r, k)
        }));
        keys.extend((0..m).map(|r| (n_x + r, n_x + r)));
        let solver = CudssSolver::new(n_x + m, &keys, Factorization::Ldlt, data.deterministic)?;
        let s = solver.stream();

        let mut ints = Vec::new();
        let mut int = |v: &[usize]| -> Result<*const u32, LinearError> {
            let v: Vec<u32> = v.iter().map(|&i| index(i)).collect::<Result<_, _>>()?;
            let mut d = DeviceBuffer::new(v.len().max(1))?;
            if !v.is_empty() {
                upload(&mut d, &v, s)?;
            }
            let p = d.ptr as *const u32;
            ints.push(d);
            Ok(p)
        };
        // Column by column, each column's entries ascending: `Transpose`.
        let mut a_start = vec![0usize; n + 1];
        for &(_, c) in &data.jacobian {
            a_start[c + 1] += 1;
        }
        for c in 0..n {
            a_start[c + 1] += a_start[c];
        }
        let mut next = a_start[..n].to_vec();
        let mut a_entries = vec![0usize; data.jacobian.len()];
        for (e, &(_, c)) in data.jacobian.iter().enumerate() {
            a_entries[next[c]] = e;
            next[c] += 1;
        }
        let mut slack_row = vec![0usize; n - n_x];
        for (r, s) in data.slack_of.iter().enumerate() {
            if let Some(k) = *s {
                slack_row[k - n_x] = r;
            }
        }
        let a_row: Vec<usize> = data.jacobian.iter().map(|e| e.0).collect();
        let pin_col: Vec<usize> = data.pinned.iter().map(|p| p.0).collect();
        let a_row_ptr = int(&a_row)?;
        let a_start_ptr = int(&a_start)?;
        let a_entries_ptr = int(&a_entries)?;
        let h_index_ptr = int(&h_index)?;
        let a_index_ptr = int(&a_index)?;
        let slack_row_ptr = int(&slack_row)?;
        let pin_col_ptr = int(&pin_col)?;

        let slack_of: Vec<i32> = data
            .slack_of
            .iter()
            .map(|s| s.map_or(Ok(-1), |k| index(k).map(|k| k as i32)))
            .collect::<Result<_, _>>()?;
        let mut signed = DeviceBuffer::new(rows.max(1))?;
        if rows > 0 {
            upload(&mut signed, &slack_of, s)?;
        }

        let mut reals = Vec::new();
        let mut real = |v: &[f64]| -> Result<*const f64, LinearError> {
            let mut d = DeviceBuffer::new(v.len().max(1))?;
            if !v.is_empty() {
                upload(&mut d, v, s)?;
            }
            let p = d.ptr as *const f64;
            reals.push(d);
            Ok(p)
        };
        let lower = real(&data.lower)?;
        let upper = real(&data.upper)?;
        let c_lower = real(&data.c_lower)?;
        let row_scale = real(&data.row_scale)?;
        let pin_value = real(&data.pinned.iter().map(|p| p.1).collect::<Vec<_>>())?;

        let vector = |len: usize| DeviceBuffer::<f64>::new(len.max(1));
        let dim = n + m;
        let u = vector(n)?;
        let y = vector(m)?;
        let z_l = vector(n)?;
        let z_u = vector(n)?;
        let g = vector(n)?;
        let t = vector(n)?;
        let grad = vector(n)?;
        let r_d = vector(n)?;
        let r_p = vector(m)?;
        let sigma = vector(n)?;
        let diag = vector(n)?;
        let a_val = vector(data.jacobian.len())?;
        let y_model = vector(rows)?;
        let trial_u = vector(n)?;
        let trial_r = vector(m)?;
        let c_soc = vector(m)?;
        let dz_l = vector(n)?;
        let dz_u = vector(n)?;
        let w = vector(n.max(m))?;
        if data.jacobian.len() > data.n_jacobian {
            // SAFETY: `a_val` holds the internal Jacobian's entries.
            unsafe {
                crate::cuda::copy_to_device(a_val.ptr.add(data.n_jacobian), &data.constants, s)?;
            }
        }
        let values = ffi::go_ip {
            n_x: index(n_x)?,
            n: index(n)?,
            m: index(m)?,
            rows: index(rows)?,
            n_jac: index(data.n_jacobian)?,
            n_a: index(data.jacobian.len())?,
            h_kept: index(h_index.len())?,
            a_kept: index(a_index.len())?,
            objective: data.objective_scale,
            lower,
            upper,
            c_lower,
            row_scale,
            pin_value,
            pin_col: pin_col_ptr,
            slack_of: signed.ptr,
            slack_row: slack_row_ptr,
            a_row: a_row_ptr,
            a_start: a_start_ptr,
            a_entries: a_entries_ptr,
            h_index: h_index_ptr,
            a_index: a_index_ptr,
            c: model.rows_ptr(),
            jac: model.jacobian_ptr(),
            hess: model.hessian_ptr(),
            trip: solver.triplet_values(),
            u: u.ptr,
            y: y.ptr,
            z_l: z_l.ptr,
            z_u: z_u.ptr,
            g: g.ptr,
            t: t.ptr,
            grad: grad.ptr,
            r_d: r_d.ptr,
            r_p: r_p.ptr,
            sigma: sigma.ptr,
            diag: diag.ptr,
            a_val: a_val.ptr,
            y_model: y_model.ptr,
            trial_u: trial_u.ptr,
            trial_r: trial_r.ptr,
            c_soc: c_soc.ptr,
            dz_l: dz_l.ptr,
            dz_u: dz_u.ptr,
            w: w.ptr,
        };
        // SAFETY: `grad` holds n values; its slack columns stay zero.
        unsafe { go!(go_ip_zero(grad.ptr, n as u32), s)? };
        reals.extend([g, t, sigma, diag, a_val, dz_l, dz_u]);
        Ok(Self {
            data: values,
            n_x,
            n,
            m,
            _ints: ints,
            _signed: signed,
            _reals: reals,
            u,
            y,
            z_l,
            z_u,
            grad,
            r_p,
            r_d,
            y_model,
            trial_u,
            trial_r,
            c_soc,
            w,
            rhs: vector(dim)?,
            newton: vector(dim)?,
            correction: vector(dim)?,
            work: vector(dim)?,
            residual: vector(dim)?,
            refinement: vector(dim)?,
            primal_in: vector(n_x + m)?,
            dual_in: vector(n_x + m)?,
            primal_out: vector(n_x + m)?,
            dual_out: vector(n_x + m)?,
            saved: [vector(n)?, vector(m)?, vector(n)?, vector(n)?],
            partial: vector(ffi::GO_IP_PARTIALS as usize)?,
            scalars: vector(SCALARS)?,
            host_scalars: PinnedBuffer::new(SCALARS)?,
            host_x: PinnedBuffer::new(n_x.max(1))?,
            solver,
        })
    }

    fn stream(&self) -> ffi::cudaStream_t {
        self.solver.stream()
    }

    /// Queues a reduction of `count` values at `x` into `slot`.
    fn reduce(
        &self,
        kind: u32,
        x: *const f64,
        count: usize,
        slot: usize,
    ) -> Result<(), LinearError> {
        assert!(slot < SCALARS);
        // SAFETY: `x` holds `count` values, `partial` GO_IP_PARTIALS and
        // `scalars` SCALARS.
        unsafe {
            go!(
                go_ip_reduce(
                    kind as i32,
                    x,
                    count as u32,
                    self.partial.ptr,
                    self.scalars.ptr.add(slot),
                ),
                self.stream()
            )
        }
    }

    /// The first `N` slots, once everything queued has run.
    fn fetch<const N: usize>(&self) -> Result<[f64; N], LinearError> {
        let s = self.stream();
        // SAFETY: the pinned buffer and `scalars` hold SCALARS ≥ N values.
        unsafe {
            cuda(
                ffi::cudaMemcpyAsync(
                    self.host_scalars.as_mut_ptr().cast(),
                    self.scalars.ptr.cast(),
                    N * size_of::<f64>(),
                    ffi::cudaMemcpyDeviceToHost,
                    s,
                ),
                "cudaMemcpyAsync (scalars)",
            )?;
            cuda(ffi::cudaStreamSynchronize(s), "cudaStreamSynchronize")?;
        }
        let mut out = [0.0; N];
        out.copy_from_slice(self.host_scalars.prefix(N));
        Ok(out)
    }

    fn device_copy(&self, dst: *mut f64, src: *const f64, count: usize) -> Result<(), LinearError> {
        // SAFETY: the callers' buffers hold `count` values.
        cuda(
            unsafe {
                ffi::cudaMemcpyAsync(
                    dst.cast(),
                    src.cast(),
                    count * size_of::<f64>(),
                    ffi::cudaMemcpyDeviceToDevice,
                    self.stream(),
                )
            },
            "cudaMemcpyAsync (on device)",
        )
    }

    /// Queues the copy of `x` from the device vector `u` to the host.
    fn queue_x(&self, u: *const f64) -> Result<(), LinearError> {
        // SAFETY: `u` holds at least n_x values, the pinned buffer n_x.
        cuda(
            unsafe {
                ffi::cudaMemcpyAsync(
                    self.host_x.as_mut_ptr().cast(),
                    u.cast(),
                    self.n_x * size_of::<f64>(),
                    ffi::cudaMemcpyDeviceToHost,
                    self.stream(),
                )
            },
            "cudaMemcpyAsync (x)",
        )
    }

    fn step(&self, correction: bool) -> *const f64 {
        if correction {
            self.correction.ptr
        } else {
            self.newton.ptr
        }
    }

    /// Sets the iterate and its multipliers.
    pub fn set_iterate(
        &mut self,
        u: &[f64],
        y: &[f64],
        z_l: &[f64],
        z_u: &[f64],
    ) -> Result<(), LinearError> {
        let s = self.stream();
        upload(&mut self.u, u, s)?;
        upload(&mut self.y, y, s)?;
        upload(&mut self.z_l, z_l, s)?;
        upload(&mut self.z_u, z_u, s)
    }

    /// The iterate and its multipliers on the host, in `set_iterate`'s order.
    #[allow(clippy::type_complexity)]
    pub fn iterate(&self) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), LinearError> {
        let s = self.stream();
        let read = |b: &DeviceBuffer<f64>, len: usize| -> Result<Vec<f64>, LinearError> {
            let mut out = vec![0.0; b.len()];
            b.download(&mut out, s)?;
            out.truncate(len);
            Ok(out)
        };
        Ok((
            read(&self.u, self.n)?,
            read(&self.y, self.m)?,
            read(&self.z_l, self.n)?,
            read(&self.z_u, self.n)?,
        ))
    }

    /// The iterate's `x`, for the objective.
    pub fn x(&mut self) -> Result<&[f64], LinearError> {
        self.queue_x(self.u.ptr)?;
        // SAFETY: the stream is the solver's.
        cuda(
            unsafe { ffi::cudaStreamSynchronize(self.stream()) },
            "cudaStreamSynchronize",
        )?;
        Ok(self.host_x.prefix(self.n_x))
    }

    /// The gradient, and the model's rows and Jacobian at the iterate.
    pub fn evaluate(&mut self, model: &DeviceAcModel, grad: &[f64]) -> Result<(), LinearError> {
        let s = self.stream();
        // SAFETY: `grad` on the device holds n ≥ n_x values; `u` holds the
        // model's variables; `r_p` m values; `self.data` is this solver's.
        unsafe {
            crate::cuda::copy_to_device(self.grad.ptr, grad, s)?;
            model.queue_constraints(self.u.ptr, s)?;
            model.queue_jacobian(self.u.ptr, s)?;
            go!(go_ip_jacobian(&self.data), s)?;
            go!(go_ip_primal(&self.data, self.u.ptr, self.r_p.ptr), s)
        }
    }

    /// The least-squares multipliers into `y`, if the solve succeeds and they
    /// are within `cap`.
    pub fn start_multipliers(&mut self, delta_c: f64, cap: f64) -> Result<(), LinearError> {
        let s = self.stream();
        // SAFETY: `self.data` is this solver's; `rhs` holds n + m values.
        unsafe {
            go!(go_ip_identity(&self.data), s)?;
            go!(go_ip_kkt(&self.data, 0, delta_c), s)?;
        }
        if self.solver.factorize_on_device()?.is_none() {
            return Ok(());
        }
        // SAFETY: as above.
        unsafe { go!(go_ip_least_squares_rhs(&self.data, self.rhs.ptr), s)? };
        self.queue_full_solve(self.rhs.ptr, self.newton.ptr)?;
        let dim = self.n + self.m;
        self.reduce(ffi::GO_IP_NONFINITE, self.newton.ptr, dim, 0)?;
        // SAFETY: `newton` holds n + m values.
        let rows = unsafe { self.newton.ptr.add(self.n) };
        self.reduce(ffi::GO_IP_MAX_ABS, rows, self.m, 1)?;
        let [bad, size] = self.fetch::<2>()?;
        if bad == 0.0 && size <= cap {
            self.device_copy(self.y.ptr, rows, self.m)?;
        }
        Ok(())
    }

    /// `[‖r_d‖∞, ‖r_p‖∞, ‖r_p‖₁, ‖y‖₁, ‖z_L‖₁, ‖z_U‖₁]`.
    pub fn residuals(&mut self) -> Result<[f64; 6], LinearError> {
        let s = self.stream();
        // SAFETY: `self.data` is this solver's.
        unsafe {
            go!(go_ip_slacks(&self.data), s)?;
            go!(go_ip_dual(&self.data), s)?;
        }
        self.reduce(ffi::GO_IP_MAX_ABS, self.r_d.ptr, self.n, 0)?;
        self.reduce(ffi::GO_IP_MAX_ABS, self.r_p.ptr, self.m, 1)?;
        self.reduce(ffi::GO_IP_SUM_ABS, self.r_p.ptr, self.m, 2)?;
        self.reduce(ffi::GO_IP_SUM_ABS, self.y.ptr, self.m, 3)?;
        self.reduce(ffi::GO_IP_SUM_ABS, self.z_l.ptr, self.n, 4)?;
        self.reduce(ffi::GO_IP_SUM_ABS, self.z_u.ptr, self.n, 5)?;
        self.fetch::<6>()
    }

    pub fn complementarity(&mut self, target: f64) -> Result<f64, LinearError> {
        // SAFETY: `self.data` is this solver's.
        unsafe { go!(go_ip_complementarity(&self.data, target), self.stream())? };
        self.reduce(ffi::GO_IP_MAX_ABS, self.w.ptr, self.n, 0)?;
        Ok(self.fetch::<1>()?[0])
    }

    /// The Hessian and Σ at the iterate.
    pub fn prepare(&mut self, model: &DeviceAcModel) -> Result<(), LinearError> {
        let s = self.stream();
        // SAFETY: `u` and `y_model` hold the model's variables and rows.
        unsafe {
            go!(go_ip_sigma(&self.data), s)?;
            go!(go_ip_model_y(&self.data), s)?;
            model.queue_hessian(self.u.ptr, self.y_model.ptr, s)
        }
    }

    /// Factorizes with `dw`; the inertia, or `None` if a slack's diagonal is
    /// not positive or the factorization failed.
    pub fn factorize(&mut self, dw: f64, delta_c: f64) -> Result<Option<Inertia>, LinearError> {
        let s = self.stream();
        // SAFETY: `self.data` is this solver's.
        unsafe {
            go!(go_ip_diagonal(&self.data, dw), s)?;
            go!(go_ip_bad_slacks(&self.data), s)?;
        }
        self.reduce(ffi::GO_IP_MAX_ABS, self.w.ptr, self.data.rows as usize, 0)?;
        if self.fetch::<1>()?[0] != 0.0 {
            return Ok(None);
        }
        // SAFETY: as above.
        unsafe { go!(go_ip_kkt(&self.data, 1, delta_c), s)? };
        self.solver.factorize_on_device()
    }

    /// The expected inertia of the KKT matrix.
    pub fn minimizing(&self) -> Inertia {
        Inertia {
            positive: self.n_x,
            negative: self.m,
            zero: 0,
        }
    }

    /// Queues `Augmented::solve`: the full system through the condensed one.
    fn queue_full_solve(&self, rhs: *const f64, d: *mut f64) -> Result<(), LinearError> {
        let s = self.stream();
        // SAFETY: `rhs` and `d` hold n + m values, the solver's vectors
        // n_x + m.
        unsafe {
            go!(go_ip_condense(&self.data, rhs, self.solver.rhs_ptr()), s)?;
            self.solver.queue_solve()?;
            go!(
                go_ip_expand(&self.data, rhs, self.solver.solution_ptr(), d),
                s
            )
        }
    }

    /// `HostEngine::solve`, from `rhs` to `out`: `None` if the solution is
    /// not finite, else its residual relative to the right-hand side.
    fn solve(&mut self, out: *mut f64, rounds: usize) -> Result<Option<f64>, LinearError> {
        let dim = self.n + self.m;
        let (s, rhs) = (self.stream(), self.rhs.ptr as *const f64);
        let (work, residual, refinement) = (self.work.ptr, self.residual.ptr, self.refinement.ptr);
        self.queue_full_solve(rhs, work)?;
        self.reduce(ffi::GO_IP_NONFINITE, work, dim, 0)?;
        self.reduce(ffi::GO_IP_MAX_ABS, rhs, dim, 1)?;
        let [bad, rhs_size] = self.fetch::<2>()?;
        if bad != 0.0 {
            return Ok(None);
        }
        let target = 1e-12 * rhs_size;
        let mut last = f64::INFINITY;
        let mut best = f64::INFINITY;
        self.device_copy(out, work, dim)?;
        for round in 0..rounds {
            // SAFETY: the vectors hold n + m and n_x + m values.
            unsafe {
                go!(
                    go_ip_split(&self.data, work, self.primal_in.ptr, self.dual_in.ptr),
                    s
                )?;
                self.solver
                    .queue_multiply(self.primal_in.ptr, self.primal_out.ptr)?;
                self.solver
                    .queue_multiply(self.dual_in.ptr, self.dual_out.ptr)?;
                go!(
                    go_ip_residual(
                        &self.data,
                        rhs,
                        work,
                        self.primal_out.ptr,
                        self.dual_out.ptr,
                        residual,
                    ),
                    s
                )?;
            }
            self.reduce(ffi::GO_IP_MAX_ABS, residual, dim, 0)?;
            self.reduce(ffi::GO_IP_NONFINITE, residual, dim, 1)?;
            let [size, bad] = self.fetch::<2>()?;
            let size = if bad != 0.0 { f64::INFINITY } else { size };
            if size < best {
                best = size;
                self.device_copy(out, work, dim)?;
            }
            if round > 0 && (size <= target || size > 0.5 * last) {
                break;
            }
            last = size;
            self.queue_full_solve(residual, refinement)?;
            self.reduce(ffi::GO_IP_NONFINITE, refinement, dim, 0)?;
            if self.fetch::<1>()?[0] != 0.0 {
                break;
            }
            // SAFETY: both hold n + m values.
            unsafe { go!(go_ip_add(work, refinement, dim as u32), s)? };
        }
        Ok(Some(best / rhs_size.max(f64::MIN_POSITIVE)))
    }

    /// The Newton step on the last factorization; false if the solve failed.
    pub fn direction(&mut self, mu: f64, rounds: usize) -> Result<bool, LinearError> {
        // SAFETY: `rhs` holds n + m values.
        unsafe { go!(go_ip_rhs(&self.data, mu, self.rhs.ptr), self.stream())? };
        Ok(self.solve(self.newton.ptr, rounds)?.is_some())
    }

    fn queue_boundary(&self, du: *const f64, slot: usize) -> Result<(), LinearError> {
        // SAFETY: `du` holds at least n values.
        unsafe { go!(go_ip_boundary(&self.data, du), self.stream())? };
        self.reduce(ffi::GO_IP_MIN, self.w.ptr, self.n, slot)
    }

    /// `[φ less the objective, dφ, the blocking step, ‖Δu‖∞, ‖u‖∞]`.
    pub fn search_start(&mut self, mu: f64) -> Result<[f64; 5], LinearError> {
        let (s, du) = (self.stream(), self.newton.ptr as *const f64);
        // SAFETY: `u` and `du` hold n values.
        unsafe { go!(go_ip_barrier(&self.data, self.u.ptr, mu), s)? };
        self.reduce(ffi::GO_IP_SUM, self.w.ptr, self.n, 0)?;
        // SAFETY: as above.
        unsafe { go!(go_ip_slope(&self.data, du, mu), s)? };
        self.reduce(ffi::GO_IP_SUM, self.w.ptr, self.n, 1)?;
        self.queue_boundary(du, 2)?;
        self.reduce(ffi::GO_IP_MAX_ABS, du, self.n, 3)?;
        self.reduce(ffi::GO_IP_MAX_ABS, self.u.ptr, self.n, 4)?;
        self.fetch::<5>()
    }

    /// The trial point `alpha` along a step: the model's rows there, its
    /// residual, and θ and φ's barrier terms.
    pub fn trial(
        &mut self,
        model: &DeviceAcModel,
        correction: bool,
        alpha: f64,
        mu: f64,
    ) -> Result<Trial<'_>, LinearError> {
        let s = self.stream();
        let du = self.step(correction);
        // SAFETY: `du`, `trial_u` hold n values, `trial_r` m.
        unsafe {
            go!(go_ip_trial(&self.data, du, alpha), s)?;
            self.queue_x(self.trial_u.ptr)?;
            model.queue_constraints(self.trial_u.ptr, s)?;
            go!(
                go_ip_primal(&self.data, self.trial_u.ptr, self.trial_r.ptr),
                s
            )?;
            go!(go_ip_barrier(&self.data, self.trial_u.ptr, mu), s)?;
        }
        self.reduce(ffi::GO_IP_SUM, self.w.ptr, self.n, 0)?;
        self.reduce(ffi::GO_IP_SUM_ABS, self.trial_r.ptr, self.m, 1)?;
        let [barrier, theta] = self.fetch::<2>()?;
        Ok(Trial {
            x: self.host_x.prefix(self.n_x),
            barrier,
            theta,
        })
    }

    pub fn start_correction(&mut self) -> Result<(), LinearError> {
        self.device_copy(self.c_soc.ptr, self.r_p.ptr, self.m)
    }

    /// The next correction, from the last one's step `alpha`: its relative
    /// residual and blocking step, or `None` if the solve failed.
    pub fn correct(&mut self, alpha: f64, rounds: usize) -> Result<Option<[f64; 2]>, LinearError> {
        // SAFETY: `rhs` holds n + m values.
        unsafe {
            go!(
                go_ip_correct(&self.data, alpha, self.rhs.ptr),
                self.stream()
            )?
        };
        let Some(residual) = self.solve(self.correction.ptr, rounds)? else {
            return Ok(None);
        };
        self.queue_boundary(self.correction.ptr, 0)?;
        Ok(Some([residual, self.fetch::<1>()?[0]]))
    }

    pub fn save(&mut self) -> Result<(), LinearError> {
        let (n, m) = (self.n, self.m);
        self.device_copy(self.saved[0].ptr, self.u.ptr, n)?;
        self.device_copy(self.saved[1].ptr, self.y.ptr, m)?;
        self.device_copy(self.saved[2].ptr, self.z_l.ptr, n)?;
        self.device_copy(self.saved[3].ptr, self.z_u.ptr, n)
    }

    pub fn restore(&mut self) -> Result<(), LinearError> {
        let (n, m) = (self.n, self.m);
        self.device_copy(self.u.ptr, self.saved[0].ptr, n)?;
        self.device_copy(self.y.ptr, self.saved[1].ptr, m)?;
        self.device_copy(self.z_l.ptr, self.saved[2].ptr, n)?;
        self.device_copy(self.z_u.ptr, self.saved[3].ptr, n)
    }

    /// The bound multipliers' step and the blocking fraction of it, before
    /// the cap of 1.
    pub fn multiplier_step(
        &mut self,
        correction: bool,
        mu: f64,
        tau: f64,
    ) -> Result<f64, LinearError> {
        let du = self.step(correction);
        // SAFETY: `du` holds n values.
        unsafe { go!(go_ip_multipliers(&self.data, du, mu, tau), self.stream())? };
        self.reduce(ffi::GO_IP_MIN, self.w.ptr, self.n, 0)?;
        Ok(self.fetch::<1>()?[0])
    }

    pub fn step_norm(&mut self, correction: bool) -> Result<f64, LinearError> {
        self.reduce(ffi::GO_IP_MAX_ABS, self.step(correction), self.n, 0)?;
        Ok(self.fetch::<1>()?[0])
    }

    pub fn update(
        &mut self,
        correction: bool,
        alpha: f64,
        alpha_z: f64,
        mu: f64,
    ) -> Result<(), LinearError> {
        let (s, d) = (self.stream(), self.step(correction));
        // SAFETY: `d` holds n + m values.
        unsafe {
            go!(go_ip_update(&self.data, d, alpha, alpha_z), s)?;
            go!(go_ip_band(&self.data, mu), s)
        }
    }

    /// The whole iterate, on the host.
    pub fn point(&self) -> Result<Vec<f64>, LinearError> {
        let mut out = vec![0.0; self.u.len()];
        self.u.download(&mut out, self.stream())?;
        out.truncate(self.n);
        Ok(out)
    }

    /// Continues from `u`, `y` at zero, and the bound multipliers at one if
    /// any exceeds `cap`.
    pub fn restored(&mut self, u: &[f64], cap: f64) -> Result<(), LinearError> {
        let s = self.stream();
        upload(&mut self.u, u, s)?;
        // SAFETY: `y` holds m values.
        unsafe { go!(go_ip_zero(self.y.ptr, self.m as u32), s)? };
        self.reduce(ffi::GO_IP_MAX_ABS, self.z_l.ptr, self.n, 0)?;
        self.reduce(ffi::GO_IP_MAX_ABS, self.z_u.ptr, self.n, 1)?;
        let [lower, upper] = self.fetch::<2>()?;
        if lower.max(upper) > cap {
            // SAFETY: `self.data` is this solver's.
            unsafe { go!(go_ip_reset_bounds(&self.data), s)? };
        }
        Ok(())
    }
}

/// `d`'s first `src.len()` values from the host. Unlike
/// [`DeviceBuffer::upload`], `d` may be longer: a buffer of an empty vector
/// holds one value.
fn upload<T: Copy>(
    d: &mut DeviceBuffer<T>,
    src: &[T],
    s: ffi::cudaStream_t,
) -> Result<(), LinearError> {
    assert!(src.len() <= d.len());
    if src.is_empty() {
        return Ok(());
    }
    // SAFETY: `d` holds at least src.len() values.
    unsafe { crate::cuda::copy_to_device(d.ptr, src, s) }
}
