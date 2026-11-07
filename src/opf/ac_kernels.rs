//! The AC-OPF model's evaluation as flat loops over plain arrays: the shape
//! of GPU kernels, run on the CPU.
//!
//! Every row of the model is a sum of *element* terms (branch terminals, bus
//! shunts, generators, limit, angle and cost rows), and each term is a pure
//! function of a handful of entries of `x`. So:
//!
//! - **One function per element type** returns the element's Jacobian or
//!   Hessian entries as a fixed-size array of `(row, column, value)`. Rows
//!   and columns depend on indices only, never on `x`, so the pattern is
//!   fixed; on a GPU the function is the kernel body, element `e` writing
//!   its `K` values at `base + K·e`. Two elements may share a `(row,
//!   column)` (two terminals at one bus): the consumer sums duplicates, as
//!   every factorization does, so no element waits on another and nothing
//!   needs an atomic.
//! - **The balance rows**, the only sums over elements, gather per bus over
//!   the bus's incident terminals and generators (CSR lists), again without
//!   atomics.
//! - **Duplicates summed once.** Element entries are slots; [`Merge`] maps
//!   each slot to its unique `(row, column)`, computed at construction, and
//!   the evaluation sums into those (a gather per unique entry on a GPU).
//!   Unmerged, the per-terminal entries double the triplet lists, and every
//!   pass the interior point makes over them (scaling, assembly,
//!   refinement) slowed by a third on case1354_pegase.
//! - **No Y-bus.** A bus injection is the sum of its terminal flows and its
//!   shunt, the identity the module docs of [`ac`](super) already use for the
//!   limit rows; both formulations share the terminal kernel
//!   ([`terminal`]).
//!
//! Layout, as in [`AcOpf`](super::AcOpf): columns `[θ][|V|][P_g][Q_g]
//! [epigraph][flows]`, rows `[P balance][Q balance][limits][angles][costs]
//! [flow definitions]`. Terminal `t` is branch `t / 2`'s from end (`t` even)
//! or to end; its flow variables (branch flows) are columns `flow.0 + 2t`
//! and `+1`, its definition rows `flow.1 + 2t` and `+1`.

use num_complex::Complex;

use crate::branch_flow::BranchParams;

/// A terminal flow and its derivatives in the four local variables
/// `[θ_near, θ_far, |V|_near, |V|_far]`.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TerminalDerivatives {
    pub p: f64,
    pub q: f64,
    pub dp: [f64; 4],
    pub dq: [f64; 4],
    /// Full symmetric 4×4 blocks.
    pub d2p: [[f64; 4]; 4],
    pub d2q: [[f64; 4]; 4],
}

const TH_NEAR: usize = 0;
const TH_FAR: usize = 1;
const V_NEAR: usize = 2;
const V_FAR: usize = 3;

/// The terminal kernel: the flow out of a branch end,
/// \\(P = |V_n|^2 g_s + |V_n||V_f| (g_m \cos\theta + b_m \sin\theta)\\) and its
/// `Q`, with \\(\theta = \theta_n - \theta_f\\), and their first and second
/// derivatives. Six numbers in, nothing else read.
pub(super) fn terminal(
    y_self: Complex<f64>,
    y_mut: Complex<f64>,
    v_n: f64,
    v_f: f64,
    angle: f64,
) -> TerminalDerivatives {
    let (sin, cos) = angle.sin_cos();
    let c = y_mut.re * cos + y_mut.im * sin;
    let s = y_mut.re * sin - y_mut.im * cos;

    let mut d = TerminalDerivatives {
        p: v_n * v_n * y_self.re + v_n * v_f * c,
        q: -v_n * v_n * y_self.im + v_n * v_f * s,
        ..Default::default()
    };

    d.dp[TH_NEAR] = -v_n * v_f * s;
    d.dp[TH_FAR] = v_n * v_f * s;
    d.dp[V_NEAR] = 2.0 * v_n * y_self.re + v_f * c;
    d.dp[V_FAR] = v_n * c;

    d.dq[TH_NEAR] = v_n * v_f * c;
    d.dq[TH_FAR] = -v_n * v_f * c;
    d.dq[V_NEAR] = -2.0 * v_n * y_self.im + v_f * s;
    d.dq[V_FAR] = v_n * s;

    // θθ: both flows depend on the angle *difference*, so the 2×2 angle block
    // is one value with alternating sign.
    let pc = v_n * v_f * c;
    let ps = v_n * v_f * s;
    d.d2p[TH_NEAR][TH_NEAR] = -pc;
    d.d2p[TH_NEAR][TH_FAR] = pc;
    d.d2p[TH_FAR][TH_NEAR] = pc;
    d.d2p[TH_FAR][TH_FAR] = -pc;
    d.d2q[TH_NEAR][TH_NEAR] = -ps;
    d.d2q[TH_NEAR][TH_FAR] = ps;
    d.d2q[TH_FAR][TH_NEAR] = ps;
    d.d2q[TH_FAR][TH_FAR] = -ps;

    // |V||V|: quadratic in the near magnitude, bilinear across the pair.
    d.d2p[V_NEAR][V_NEAR] = 2.0 * y_self.re;
    d.d2p[V_NEAR][V_FAR] = c;
    d.d2p[V_FAR][V_NEAR] = c;
    d.d2q[V_NEAR][V_NEAR] = -2.0 * y_self.im;
    d.d2q[V_NEAR][V_FAR] = s;
    d.d2q[V_FAR][V_NEAR] = s;

    // Mixed, in both orderings since mixed partials commute.
    for (v_index, scale) in [(V_NEAR, v_f), (V_FAR, v_n)] {
        d.d2p[v_index][TH_NEAR] = -scale * s;
        d.d2p[TH_NEAR][v_index] = -scale * s;
        d.d2p[v_index][TH_FAR] = scale * s;
        d.d2p[TH_FAR][v_index] = scale * s;
        d.d2q[v_index][TH_NEAR] = scale * c;
        d.d2q[TH_NEAR][v_index] = scale * c;
        d.d2q[v_index][TH_FAR] = -scale * c;
        d.d2q[TH_FAR][v_index] = -scale * c;
    }
    d
}

/// Compressed lists: `idx[ptr[i]..ptr[i + 1]]` belong to `i`.
pub(super) struct Csr {
    ptr: Vec<usize>,
    idx: Vec<usize>,
}

impl Csr {
    fn new(n: usize, owner: impl Iterator<Item = usize>) -> Self {
        let owner: Vec<usize> = owner.collect();
        let mut ptr = vec![0usize; n + 1];
        for &o in &owner {
            ptr[o + 1] += 1;
        }
        for i in 0..n {
            ptr[i + 1] += ptr[i];
        }
        let mut fill = ptr.clone();
        let mut idx = vec![0usize; owner.len()];
        for (e, &o) in owner.iter().enumerate() {
            idx[fill[o]] = e;
            fill[o] += 1;
        }
        Self { ptr, idx }
    }

    fn of(&self, i: usize) -> &[usize] {
        &self.idx[self.ptr[i]..self.ptr[i + 1]]
    }
}

/// The model's data as flat arrays, and its evaluation by element.
pub(super) struct AcKernels {
    pub n: usize,
    pub theta: usize,
    pub vm: usize,
    pub pg: usize,
    pub qg: usize,
    /// Per terminal: its bus, the far bus, and its self and mutual
    /// admittances.
    pub near: Vec<usize>,
    pub far: Vec<usize>,
    pub y_self: Vec<Complex<f64>>,
    pub y_mut: Vec<Complex<f64>>,
    /// Per bus.
    pub shunt: Vec<Complex<f64>>,
    pub p_load: Vec<f64>,
    pub q_load: Vec<f64>,
    pub terminals_at: Csr,
    pub generators_at: Csr,
    /// Per generator.
    pub gen_bus: Vec<usize>,
    /// `(column, ∂²cost)` for each generator with a quadratic cost term.
    pub quadratic: Vec<(usize, f64)>,
    /// Per limit row, its terminal.
    pub limit_terminal: Vec<usize>,
    /// Per angle row, `(from bus, to bus)`.
    pub angle: Vec<(usize, usize)>,
    /// Per cost row, `(epigraph column, generator column, slope)`.
    pub cost: Vec<(usize, usize, f64)>,
    /// Branch flows: the first flow column and the first definition row.
    pub flow: Option<(usize, usize)>,
    /// The model's rows, counted once, in `AcOpf::build`.
    n_rows: usize,
    jacobian_merge: Merge,
    hessian_merge: Merge,
}

/// Element slots onto unique `(row, column)` entries, computed once.
#[derive(Default)]
struct Merge {
    unique: Vec<(usize, usize)>,
    /// Per slot, its unique entry.
    slot: Vec<usize>,
    /// Per unique entry, its slots in ascending order:
    /// `sources[start[u]..start[u + 1]]`, for `apply`'s gather.
    start: Vec<usize>,
    sources: Vec<usize>,
}

impl Merge {
    fn new(slots: &[(usize, usize, f64)]) -> Self {
        let mut index = std::collections::HashMap::new();
        let mut unique = Vec::new();
        let slot = slots
            .iter()
            .map(|&(r, c, _)| {
                *index.entry((r, c)).or_insert_with(|| {
                    unique.push((r, c));
                    unique.len() - 1
                })
            })
            .collect::<Vec<usize>>();
        let mut start = vec![0usize; unique.len() + 1];
        for &u in &slot {
            start[u + 1] += 1;
        }
        for u in 0..unique.len() {
            start[u + 1] += start[u];
        }
        let mut next = start[..unique.len()].to_vec();
        let mut sources = vec![0usize; slot.len()];
        for (s, &u) in slot.iter().enumerate() {
            sources[next[u]] = s;
            next[u] += 1;
        }
        Self {
            unique,
            slot,
            start,
            sources,
        }
    }

    /// The slots' values summed into their unique entries, as triplets: a
    /// gather per entry over its slots in ascending order, in parallel, so
    /// the sums are those of a sequential pass over the slots.
    fn apply(&self, slots: &[(usize, usize, f64)]) -> Vec<(usize, usize, f64)> {
        use rayon::prelude::*;
        debug_assert_eq!(slots.len(), self.slot.len());
        self.unique
            .par_iter()
            .enumerate()
            .with_min_len(4096)
            .map(|(u, &(r, c))| {
                let mut v = 0.0;
                for &s in &self.sources[self.start[u]..self.start[u + 1]] {
                    v += slots[s].2;
                }
                (r, c, v)
            })
            .collect()
    }
}

/// What [`AcKernels::new`] takes besides the branches.
pub(super) struct Layout {
    pub n: usize,
    pub theta: usize,
    pub vm: usize,
    pub pg: usize,
    pub qg: usize,
    pub shunt: Vec<Complex<f64>>,
    pub p_load: Vec<f64>,
    pub q_load: Vec<f64>,
    pub gen_bus: Vec<usize>,
    pub quadratic: Vec<(usize, f64)>,
    /// `(branch, from end?)` per limit row.
    pub limits: Vec<(usize, bool)>,
    pub angle: Vec<(usize, usize)>,
    pub cost: Vec<(usize, usize, f64)>,
    pub flow: Option<(usize, usize)>,
    /// The model's sizes, for computing the patterns at construction.
    pub n_vars: usize,
    pub n_rows: usize,
}

impl AcKernels {
    pub fn new(params: &[BranchParams], layout: Layout) -> Self {
        let mut near = Vec::with_capacity(2 * params.len());
        let mut far = Vec::with_capacity(2 * params.len());
        let mut y_self = Vec::with_capacity(2 * params.len());
        let mut y_mut = Vec::with_capacity(2 * params.len());
        for b in params {
            near.extend([b.from, b.to]);
            far.extend([b.to, b.from]);
            y_self.extend([b.y[0], b.y[3]]);
            y_mut.extend([b.y[1], b.y[2]]);
        }
        let n = layout.n;
        let mut kernels = Self {
            terminals_at: Csr::new(n, near.iter().copied()),
            generators_at: Csr::new(n, layout.gen_bus.iter().copied()),
            near,
            far,
            y_self,
            y_mut,
            n,
            theta: layout.theta,
            vm: layout.vm,
            pg: layout.pg,
            qg: layout.qg,
            shunt: layout.shunt,
            p_load: layout.p_load,
            q_load: layout.q_load,
            gen_bus: layout.gen_bus,
            quadratic: layout.quadratic,
            limit_terminal: layout
                .limits
                .iter()
                .map(|&(b, from)| 2 * b + usize::from(!from))
                .collect(),
            angle: layout.angle,
            cost: layout.cost,
            flow: layout.flow,
            n_rows: layout.n_rows,
            jacobian_merge: Merge::default(),
            hessian_merge: Merge::default(),
        };
        // The patterns depend on indices only, so any point gives them.
        let (x, y) = (vec![0.0; layout.n_vars], vec![0.0; layout.n_rows]);
        kernels.jacobian_merge = Merge::new(&kernels.jacobian_slots(&x));
        kernels.hessian_merge = Merge::new(&kernels.hessian_slots(&x, &y));
        kernels
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn terminal_count(&self) -> usize {
        self.near.len()
    }

    /// Terminal `t`'s flow and derivatives at `x`.
    fn evaluate(&self, t: usize, x: &[f64]) -> TerminalDerivatives {
        let (i, j) = (self.near[t], self.far[t]);
        terminal(
            self.y_self[t],
            self.y_mut[t],
            x[self.vm + i],
            x[self.vm + j],
            x[self.theta + i] - x[self.theta + j],
        )
    }

    /// The columns of terminal `t`'s local variables.
    fn columns(&self, t: usize) -> [usize; 4] {
        let (i, j) = (self.near[t], self.far[t]);
        [self.theta + i, self.theta + j, self.vm + i, self.vm + j]
    }

    fn limit_row(&self, r: usize) -> usize {
        2 * self.n + r
    }

    fn angle_row(&self, r: usize) -> usize {
        2 * self.n + self.limit_terminal.len() + r
    }

    fn cost_row(&self, r: usize) -> usize {
        2 * self.n + self.limit_terminal.len() + self.angle.len() + r
    }

    /// Terminal `t`'s flow columns and definition rows (branch flows).
    fn flow_of(&self, t: usize) -> Option<((usize, usize), (usize, usize))> {
        self.flow.map(|(column, row)| {
            (
                (column + 2 * t, column + 2 * t + 1),
                (row + 2 * t, row + 2 * t + 1),
            )
        })
    }

    /// Every row's value, into `out` (the model's `n_constraints` long).
    pub fn constraints(&self, x: &[f64], out: &mut [f64]) {
        let n = self.n;
        // Kernel 1, per terminal: its flow.
        let flows: Vec<(f64, f64)> = (0..self.terminal_count())
            .map(|t| {
                let (i, j) = (self.near[t], self.far[t]);
                let (v_n, v_f) = (x[self.vm + i], x[self.vm + j]);
                let (sin, cos) = (x[self.theta + i] - x[self.theta + j]).sin_cos();
                let (ys, ym) = (self.y_self[t], self.y_mut[t]);
                let c = ym.re * cos + ym.im * sin;
                let s = ym.re * sin - ym.im * cos;
                (
                    v_n * v_n * ys.re + v_n * v_f * c,
                    -v_n * v_n * ys.im + v_n * v_f * s,
                )
            })
            .collect();
        // Kernel 2, per bus: gather. With branch flows, the bus sums the flow
        // variables instead of the flows the voltages imply.
        for i in 0..n {
            let v2 = x[self.vm + i] * x[self.vm + i];
            // S = V conj(y V) = |V|² conj(y) for a shunt.
            let mut p = v2 * self.shunt[i].re + self.p_load[i];
            let mut q = -v2 * self.shunt[i].im + self.q_load[i];
            for &t in self.terminals_at.of(i) {
                match self.flow_of(t) {
                    Some(((pc, qc), _)) => {
                        p += x[pc];
                        q += x[qc];
                    }
                    None => {
                        p += flows[t].0;
                        q += flows[t].1;
                    }
                }
            }
            for &k in self.generators_at.of(i) {
                p -= x[self.pg + k];
                q -= x[self.qg + k];
            }
            out[i] = p;
            out[n + i] = q;
        }
        // Per row.
        for (r, &t) in self.limit_terminal.iter().enumerate() {
            let (p, q) = match self.flow_of(t) {
                Some(((pc, qc), _)) => (x[pc], x[qc]),
                None => flows[t],
            };
            out[self.limit_row(r)] = p * p + q * q;
        }
        for (r, &(from, to)) in self.angle.iter().enumerate() {
            out[self.angle_row(r)] = x[self.theta + from] - x[self.theta + to];
        }
        for (r, &(epigraph, column, slope)) in self.cost.iter().enumerate() {
            out[self.cost_row(r)] = x[epigraph] - slope * x[column];
        }
        for (t, &(p, q)) in flows.iter().enumerate() {
            if let Some(((pc, qc), (rp, rq))) = self.flow_of(t) {
                out[rp] = x[pc] - p;
                out[rq] = x[qc] - q;
            }
        }
    }

    // ---- Jacobian, by element -------------------------------------------

    /// Bus injection: terminal `t`'s part of its bus's balance rows.
    fn jacobian_terminal(&self, t: usize, x: &[f64]) -> [(usize, usize, f64); 8] {
        let d = self.evaluate(t, x);
        let c = self.columns(t);
        let (rp, rq) = (self.near[t], self.n + self.near[t]);
        std::array::from_fn(|k| {
            if k < 4 {
                (rp, c[k], d.dp[k])
            } else {
                (rq, c[k - 4], d.dq[k - 4])
            }
        })
    }

    /// Branch flows: terminal `t`'s flow variables in its bus's balance
    /// rows, and its two definition rows.
    fn jacobian_terminal_flow(&self, t: usize, x: &[f64]) -> [(usize, usize, f64); 12] {
        let ((pc, qc), (rp, rq)) = self.flow_of(t).expect("branch flows");
        let d = self.evaluate(t, x);
        let c = self.columns(t);
        let i = self.near[t];
        let mut out = [(0, 0, 0.0); 12];
        out[0] = (i, pc, 1.0);
        out[1] = (self.n + i, qc, 1.0);
        out[2] = (rp, pc, 1.0);
        out[3] = (rq, qc, 1.0);
        for k in 0..4 {
            out[4 + k] = (rp, c[k], -d.dp[k]);
            out[8 + k] = (rq, c[k], -d.dq[k]);
        }
        out
    }

    /// Bus `i`'s shunt in its balance rows.
    fn jacobian_shunt(&self, i: usize, x: &[f64]) -> [(usize, usize, f64); 2] {
        let v = x[self.vm + i];
        [
            (i, self.vm + i, 2.0 * v * self.shunt[i].re),
            (self.n + i, self.vm + i, -2.0 * v * self.shunt[i].im),
        ]
    }

    fn jacobian_generator(&self, k: usize) -> [(usize, usize, f64); 2] {
        let i = self.gen_bus[k];
        [(i, self.pg + k, -1.0), (self.n + i, self.qg + k, -1.0)]
    }

    /// Bus injection: `∇(P² + Q²) = 2P∇P + 2Q∇Q`.
    fn jacobian_limit(&self, r: usize, x: &[f64]) -> [(usize, usize, f64); 4] {
        let t = self.limit_terminal[r];
        let d = self.evaluate(t, x);
        let c = self.columns(t);
        let row = self.limit_row(r);
        std::array::from_fn(|k| (row, c[k], 2.0 * (d.p * d.dp[k] + d.q * d.dq[k])))
    }

    /// Branch flows: the limit on the flow variables.
    fn jacobian_limit_flow(&self, r: usize, x: &[f64]) -> [(usize, usize, f64); 2] {
        let ((pc, qc), _) = self.flow_of(self.limit_terminal[r]).expect("branch flows");
        let row = self.limit_row(r);
        [(row, pc, 2.0 * x[pc]), (row, qc, 2.0 * x[qc])]
    }

    fn jacobian_angle(&self, r: usize) -> [(usize, usize, f64); 2] {
        let (from, to) = self.angle[r];
        let row = self.angle_row(r);
        [(row, self.theta + from, 1.0), (row, self.theta + to, -1.0)]
    }

    fn jacobian_cost(&self, r: usize) -> [(usize, usize, f64); 2] {
        let (epigraph, column, slope) = self.cost[r];
        let row = self.cost_row(r);
        [(row, epigraph, 1.0), (row, column, -slope)]
    }

    /// The Jacobian's `(row, column)` entries, in the order `jacobian`
    /// returns them.
    #[cfg(feature = "gpu")]
    pub fn jacobian_pattern(&self) -> Vec<(usize, usize)> {
        self.jacobian_merge.unique.clone()
    }

    /// The Hessian's, likewise.
    #[cfg(feature = "gpu")]
    pub fn hessian_pattern(&self) -> Vec<(usize, usize)> {
        self.hessian_merge.unique.clone()
    }

    /// The Jacobian, duplicates summed.
    pub fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_merge.apply(&self.jacobian_slots(x))
    }

    /// The Jacobian's slots, element block by element block, each block
    /// written by its own element: in parallel on the host, as one thread
    /// an element would on a GPU.
    fn jacobian_slots(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut out = vec![(0, 0, 0.0); self.jacobian_len()];
        let rest = if self.flow.is_some() {
            fill(&mut out, self.terminal_count(), |t| {
                self.jacobian_terminal_flow(t, x)
            })
        } else {
            fill(&mut out, self.terminal_count(), |t| {
                self.jacobian_terminal(t, x)
            })
        };
        let rest = fill(rest, self.n, |i| self.jacobian_shunt(i, x));
        let rest = fill(rest, self.gen_bus.len(), |k| self.jacobian_generator(k));
        let rest = if self.flow.is_some() {
            fill(rest, self.limit_terminal.len(), |r| {
                self.jacobian_limit_flow(r, x)
            })
        } else {
            fill(rest, self.limit_terminal.len(), |r| {
                self.jacobian_limit(r, x)
            })
        };
        let rest = fill(rest, self.angle.len(), |r| self.jacobian_angle(r));
        let rest = fill(rest, self.cost.len(), |r| self.jacobian_cost(r));
        debug_assert!(rest.is_empty());
        out
    }

    /// The Jacobian's slot count: the blocks' sizes above.
    fn jacobian_len(&self) -> usize {
        let flows = self.flow.is_some();
        self.terminal_count() * if flows { 12 } else { 8 }
            + self.n * 2
            + self.gen_bus.len() * 2
            + self.limit_terminal.len() * if flows { 2 } else { 4 }
            + self.angle.len() * 2
            + self.cost.len() * 2
    }

    // ---- Hessian of the constraints, −Σ y ∇²c, by element -----------------

    /// Bus injection: terminal `t`'s part of its bus's balance rows.
    fn hessian_terminal(&self, t: usize, x: &[f64], y: &[f64]) -> [(usize, usize, f64); 16] {
        let d = self.evaluate(t, x);
        let c = self.columns(t);
        let (wp, wq) = (-y[self.near[t]], -y[self.n + self.near[t]]);
        std::array::from_fn(|k| {
            let (a, b) = (k / 4, k % 4);
            (c[a], c[b], wp * d.d2p[a][b] + wq * d.d2q[a][b])
        })
    }

    /// Branch flows: terminal `t`'s definition rows, `c = flow − P(V)`, so
    /// `−y∇²c = +y∇²P(V)`.
    fn hessian_terminal_flow(&self, t: usize, x: &[f64], y: &[f64]) -> [(usize, usize, f64); 16] {
        let (_, (rp, rq)) = self.flow_of(t).expect("branch flows");
        let d = self.evaluate(t, x);
        let c = self.columns(t);
        let (wp, wq) = (y[rp], y[rq]);
        std::array::from_fn(|k| {
            let (a, b) = (k / 4, k % 4);
            (c[a], c[b], wp * d.d2p[a][b] + wq * d.d2q[a][b])
        })
    }

    /// Bus `i`'s shunt: `∂²(|V|² g)/∂|V|² = 2g`, and `−2b` for `Q`.
    fn hessian_shunt(&self, i: usize, y: &[f64]) -> [(usize, usize, f64); 1] {
        let value = -y[i] * 2.0 * self.shunt[i].re + y[self.n + i] * 2.0 * self.shunt[i].im;
        [(self.vm + i, self.vm + i, value)]
    }

    /// Bus injection: `∇²(P² + Q²) = 2(∇P∇Pᵀ + P∇²P + ∇Q∇Qᵀ + Q∇²Q)`.
    fn hessian_limit(&self, r: usize, x: &[f64], y: &[f64]) -> [(usize, usize, f64); 16] {
        let t = self.limit_terminal[r];
        let d = self.evaluate(t, x);
        let c = self.columns(t);
        let weight = -y[self.limit_row(r)];
        std::array::from_fn(|k| {
            let (a, b) = (k / 4, k % 4);
            let value = 2.0
                * (d.dp[a] * d.dp[b] + d.p * d.d2p[a][b] + d.dq[a] * d.dq[b] + d.q * d.d2q[a][b]);
            (c[a], c[b], weight * value)
        })
    }

    /// Branch flows: `∇²(p² + q²) = 2` on each flow variable.
    fn hessian_limit_flow(&self, r: usize, y: &[f64]) -> [(usize, usize, f64); 2] {
        let ((pc, qc), _) = self.flow_of(self.limit_terminal[r]).expect("branch flows");
        let weight = -2.0 * y[self.limit_row(r)];
        [(pc, pc, weight), (qc, qc, weight)]
    }

    /// The Lagrangian's Hessian, `∇²f − Σ y ∇²c`, duplicates summed, every
    /// entry present whatever its value (the pattern does not depend on the
    /// point).
    pub fn lagrangian_hessian(&self, x: &[f64], y: &[f64]) -> Vec<(usize, usize, f64)> {
        self.hessian_merge.apply(&self.hessian_slots(x, y))
    }

    /// The Hessian's slots, element block by element block, each written
    /// by its own element, in parallel (see `jacobian_slots`). On
    /// case24464_goc the blocks took 44 ms an iteration on one core.
    fn hessian_slots(&self, x: &[f64], y: &[f64]) -> Vec<(usize, usize, f64)> {
        let flows = self.flow.is_some();
        let len = self.quadratic.len()
            + self.terminal_count() * 16
            + self.n
            + self.limit_terminal.len() * if flows { 2 } else { 16 };
        let mut out = vec![(0, 0, 0.0); len];
        let rest = fill(&mut out, self.quadratic.len(), |k| {
            let (column, h) = self.quadratic[k];
            [(column, column, h)]
        });
        let rest = if flows {
            fill(rest, self.terminal_count(), |t| {
                self.hessian_terminal_flow(t, x, y)
            })
        } else {
            fill(rest, self.terminal_count(), |t| {
                self.hessian_terminal(t, x, y)
            })
        };
        let rest = fill(rest, self.n, |i| self.hessian_shunt(i, y));
        let rest = if flows {
            fill(rest, self.limit_terminal.len(), |r| {
                self.hessian_limit_flow(r, y)
            })
        } else {
            fill(rest, self.limit_terminal.len(), |r| {
                self.hessian_limit(r, x, y)
            })
        };
        debug_assert!(rest.is_empty());
        out
    }
}

#[cfg(feature = "gpu")]
impl Merge {
    fn gather(&self) -> gridoxide_gpu::ac::Gather {
        gridoxide_gpu::ac::Gather {
            slots: self.slot.len(),
            start: self.start.clone(),
            sources: self.sources.clone(),
        }
    }
}

#[cfg(feature = "gpu")]
impl AcKernels {
    /// The model's arrays and merge maps for the device
    /// (`gridoxide_gpu::ac::DeviceAcModel`), whose kernels are this
    /// module's functions in CUDA.
    pub fn device_data(&self, n_vars: usize) -> gridoxide_gpu::ac::AcModelData {
        let pair = |c: &Complex<f64>| [c.re, c.im];
        gridoxide_gpu::ac::AcModelData {
            n: self.n,
            theta: self.theta,
            vm: self.vm,
            pg: self.pg,
            qg: self.qg,
            near: self.near.clone(),
            far: self.far.clone(),
            y_self: self.y_self.iter().map(pair).collect(),
            y_mut: self.y_mut.iter().map(pair).collect(),
            shunt: self.shunt.iter().map(pair).collect(),
            p_load: self.p_load.clone(),
            q_load: self.q_load.clone(),
            term_ptr: self.terminals_at.ptr.clone(),
            term_idx: self.terminals_at.idx.clone(),
            gen_ptr: self.generators_at.ptr.clone(),
            gen_idx: self.generators_at.idx.clone(),
            gen_bus: self.gen_bus.clone(),
            quadratic: self.quadratic.clone(),
            limit_terminal: self.limit_terminal.clone(),
            angle: self.angle.clone(),
            cost: self.cost.clone(),
            flow: self.flow,
            n_vars,
            n_rows: self.n_rows,
            jacobian: self.jacobian_merge.gather(),
            hessian: self.hessian_merge.gather(),
        }
    }
}

/// Writes `count` elements' blocks of `K` entries, element `e`'s from
/// `block(e)`, at the front of `out`, and returns the rest. In parallel
/// above a few thousand elements, where it pays (case14's whole Hessian is
/// a few microseconds); every block is computed alone, so the values are
/// those of a sequential pass.
fn fill<'a, const K: usize>(
    out: &'a mut [(usize, usize, f64)],
    count: usize,
    block: impl Fn(usize) -> [(usize, usize, f64); K] + Sync,
) -> &'a mut [(usize, usize, f64)] {
    use rayon::prelude::*;
    let (head, rest) = out.split_at_mut(count * K);
    head.par_chunks_mut(K.max(1))
        .enumerate()
        .with_min_len(2048)
        .for_each(|(e, chunk)| chunk.copy_from_slice(&block(e)));
    rest
}
