//! AC optimal power flow.
//!
//! The real problem, of which [`dc`](super::dc) is the linearization: least-cost
//! dispatch against the *full* power flow equations, with reactive power,
//! voltage magnitudes and losses all present.
//!
//! # What changes from DC
//!
//! Everything that made DC-OPF a convex QP goes away. The balance equations
//! are trigonometric rather than linear, so the feasible set is curved and
//! generally nonconvex; a solution is therefore a **local** optimum satisfying
//! the first-order conditions, not a proven global one. That is not a
//! limitation of this implementation — it is the state of the art, and it is
//! what every published AC-OPF objective means.
//!
//! In exchange, two control families DC cannot express arrive: generator
//! reactive power, and bus voltage magnitude.
//!
//! # Variables
//!
//! \\[ x = [\;\theta_0 \dots \theta_{n-1},\;
//!          |V|_0 \dots |V|_{n-1},\;
//!          P_{g_0} \dots,\; Q_{g_0} \dots\;] \\]
//!
//! Generator outputs are variables in their own right rather than being
//! eliminated into the balance equations. That costs \\(2G\\) unknowns and buys
//! two things: a generator's box constraint becomes a simple variable bound,
//! and several generators can sit on one bus without the model having to
//! decide how to split their output — which the optimizer is precisely there
//! to decide.
//!
//! # Constraints
//!
//! - **Power balance** at every bus, both `P` and `Q`: an equality per bus per
//!   quantity, \\(2n\\) rows. These are the power flow equations.
//! - **Branch apparent-power limits**, two per rated branch (one per
//!   terminal), written \\(P^2 + Q^2 \le \text{rate}^2\\). Squared, so that no
//!   square root — whose derivative is undefined at zero flow — ever enters
//!   the model.
//! - **Voltage magnitudes** and **generator boxes** as variable bounds, which
//!   the interior-point method handles directly through its barrier rather
//!   than as constraint rows.
//! - **The reference angle**, pinned by equal bounds.
//!
//! # Where the derivatives come from
//!
//! The balance rows' first and second derivatives are exactly
//! [`injection_hessian`](crate::injection_hessian)'s — that module exists for
//! this. The branch limits need their own, and the structure turns out to be
//! the same: a terminal flow
//!
//! \\[ P_{ft} = |V_f|^2 g_{self} + |V_f||V_t| (g_{mut}\cos\theta_{ft}
//!     + b_{mut}\sin\theta_{ft}) \\]
//!
//! is algebraically **a bus injection with exactly one neighbour**, with
//! \\(y_{ff}\\) playing the role of the self-admittance and \\(y_{ft}\\) the
//! mutual one. So the same \\(c\\)/\\(s\\) pair and the same derivative
//! rotations apply, which is why [`branch_flow`](crate::branch_flow) and
//! `injection_hessian` share their shape.

use std::collections::HashMap;

use num_complex::Complex;

use crate::branch_flow::{branch_params, BranchParams, Terminal};
use crate::injection_hessian::{injection_jacobian, weighted_injection_hessian};
use crate::network::{power_injections, YBusSparse};
use crate::opf::model::{CostCurve, OpfData};
use crate::opf::nlp::{NonlinearProblem, NlpOptions, NlpSolution};
use crate::opf::{OpfError, OptStatus};
use crate::types::{Bus, BusType, Line, Transformer};

/// A generator, resolved onto a bus, in per-unit.
#[derive(Clone, Debug)]
pub struct AcGenerator {
    pub index: usize,
    pub bus: usize,
    pub p_min: f64,
    pub p_max: f64,
    pub q_min: f64,
    pub q_max: f64,
    pub cost: Option<CostCurve>,
}

/// Everything AC-OPF needs, per-unit on [`base_mva`](Self::base_mva).
#[derive(Clone, Debug)]
pub struct AcOpfNetwork {
    pub n_buses: usize,
    pub reference: usize,
    pub lines: Vec<Line>,
    pub transformers: Vec<Transformer>,
    pub generators: Vec<AcGenerator>,
    /// Per-bus constant-power demand, per-unit.
    pub p_load: Vec<f64>,
    pub q_load: Vec<f64>,
    pub v_min: Vec<f64>,
    pub v_max: Vec<f64>,
    /// Flat branch index → rating in per-unit; absent or `None` is unlimited.
    pub limits: HashMap<usize, Option<f64>>,
    /// Bus shunt admittances, per-unit, stamped onto the Y-bus diagonal.
    pub shunts: Vec<(usize, Complex<f64>)>,
    pub base_mva: f64,
    /// Initial voltage magnitude. `None` starts flat at 1.0.
    pub v_start: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct AcOpfOptions {
    /// Bounds applied to every bus magnitude when the network carries none.
    pub default_v_min: f64,
    pub default_v_max: f64,
    /// Whether branch apparent-power limits are enforced at all. Turning them
    /// off is the cleanest way to tell a *binding limit* apart from an
    /// infeasible network when a case will not solve.
    pub enforce_limits: bool,
    pub nlp: NlpOptions,
}

impl Default for AcOpfOptions {
    fn default() -> Self {
        Self {
            default_v_min: 0.9,
            default_v_max: 1.1,
            enforce_limits: true,
            nlp: NlpOptions::default(),
        }
    }
}

impl AcOpfNetwork {
    /// Builds from a PGM document and its companion OPF document.
    ///
    /// The same three traps [`DcOpfNetwork::from_pgm`](super::dc::DcOpfNetwork::from_pgm)
    /// documents apply unchanged — the per-unit base must be the OPF
    /// document's, synthesized source buses must be dropped, and branch limits
    /// are keyed by component id — so they are not repeated here.
    pub fn from_pgm(
        input: crate::pgm::PgmInput,
        opf: &OpfData,
        freq_hz: f64,
        options: &AcOpfOptions,
    ) -> Result<Self, OpfError> {
        let base_mva = opf.base_mva;
        let s_base_va = base_mva * 1e6;

        let loads_raw: Vec<(u64, f64, f64)> = input
            .data
            .sym_load
            .iter()
            .filter(|l| l.status != 0)
            .map(|l| (l.node, l.p_specified, l.q_specified))
            .collect();
        let n_physical = input.data.node.len();

        // Bus shunts, captured before `pgm_to_network` consumes the document.
        //
        // Dropping these is not a rounding error. A shunt capacitor supplies
        // reactive power for free; without it the generators have to, and the
        // dispatch gets more expensive. On the pglib fixtures exactly the
        // three cases carrying shunts were the three that disagreed with the
        // published objectives, and the two with none matched to 0.001%.
        let id_to_idx = crate::pgm::node_id_to_idx(&input);
        let shunts_raw: Vec<(usize, num_complex::Complex<f64>)> =
            crate::pgm::pgm_shunts_1ph(&input, &id_to_idx, s_base_va)
                .into_iter()
                .filter(|s| s.at < n_physical)
                .map(|s| (s.at, s.y))
                .collect();

        let net = crate::pgm::pgm_to_network(input, s_base_va, freq_hz);

        let lines: Vec<Line> = net
            .lines
            .iter()
            .enumerate()
            .filter(|(flat, l)| {
                l.from < n_physical
                    && l.to < n_physical
                    && !net.source_branch_idx.values().any(|v| v == flat)
            })
            .map(|(_, l)| l.clone())
            .collect();
        let transformers: Vec<Transformer> = net
            .transformers
            .iter()
            .filter(|t| t.from < n_physical && t.to < n_physical)
            .cloned()
            .collect();

        let reference = net
            .source_branch_idx
            .values()
            .filter_map(|&flat| net.lines.get(flat))
            .map(|line| line.to.min(line.from))
            .find(|&bus| bus < n_physical)
            .unwrap_or(0);

        let mut p_load = vec![0.0; n_physical];
        let mut q_load = vec![0.0; n_physical];
        for (node, p, q) in loads_raw {
            if let Some(&bus) = net.node_idx.get(&node) {
                if bus < n_physical {
                    p_load[bus] += p / s_base_va;
                    q_load[bus] += q / s_base_va;
                }
            }
        }

        let mut generators = Vec::with_capacity(opf.generator.len());
        for g in &opf.generator {
            let Some(&bus) = net.node_idx.get(&g.node) else {
                return Err(OpfError::Backend(format!(
                    "generator {} sits on node {}, which is not in the network document",
                    g.index, g.node
                )));
            };
            let (p_min, p_max) = g.p_limits_pu(base_mva);
            let (q_min, q_max) = g.q_limits_pu(base_mva);
            generators.push(AcGenerator {
                index: g.index,
                bus,
                p_min,
                p_max,
                q_min,
                q_max,
                cost: g.cost.as_ref().map(|c| c.to_per_unit(base_mva)),
            });
        }

        // Branch limits are keyed on the *filtered* branch order, which is
        // lines-then-transformers over the physical branches only.
        let mut limits = HashMap::new();
        for limit in &opf.branch_limit {
            if let Some(&flat) = net.branch_idx.get(&limit.id) {
                if flat < lines.len() + transformers.len() {
                    limits.insert(flat, limit.rate_pu(base_mva));
                }
            }
        }

        // Per-bus limits where the document supplies them, the option
        // defaults where it does not. A document written before `bus_voltage`
        // existed carries none, and falling back is right there — but falling
        // back *per bus* would be wrong, since a partial list would silently
        // mix two conventions. The list is all-or-nothing by construction:
        // `build_opf_data` emits an entry for every bus or none at all.
        let mut v_min = vec![options.default_v_min; n_physical];
        let mut v_max = vec![options.default_v_max; n_physical];
        for entry in &opf.bus_voltage {
            if let Some(&bus) = net.node_idx.get(&entry.node) {
                if bus < n_physical {
                    v_min[bus] = entry.v_min;
                    v_max[bus] = entry.v_max;
                }
            }
        }

        Ok(Self {
            n_buses: n_physical,
            reference,
            lines,
            transformers,
            generators,
            p_load,
            q_load,
            v_min,
            v_max,
            limits,
            shunts: shunts_raw,
            base_mva,
            v_start: None,
        })
    }
}

/// Value, gradient and Hessian of one terminal's `(P, Q)`, in the four local
/// state variables `[θ_near, θ_far, |V|_near, |V|_far]`.
///
/// A terminal flow is a one-neighbour bus injection — see the module docs — so
/// these are the injection formulas with the sum over neighbours removed.
#[derive(Clone, Copy, Debug, Default)]
struct TerminalDerivatives {
    p: f64,
    q: f64,
    dp: [f64; 4],
    dq: [f64; 4],
    /// Full symmetric 4×4 blocks.
    d2p: [[f64; 4]; 4],
    d2q: [[f64; 4]; 4],
}

const TH_NEAR: usize = 0;
const TH_FAR: usize = 1;
const V_NEAR: usize = 2;
const V_FAR: usize = 3;

fn terminal_derivatives(
    branch: &BranchParams,
    terminal: Terminal,
    theta: &[f64],
    vm: &[f64],
) -> TerminalDerivatives {
    let (near, far) = match terminal {
        Terminal::From => (branch.from, branch.to),
        Terminal::To => (branch.to, branch.from),
    };
    let (y_self, y_mut) = match terminal {
        Terminal::From => (branch.y[0], branch.y[1]),
        Terminal::To => (branch.y[3], branch.y[2]),
    };
    let (v_n, v_f) = (vm[near], vm[far]);
    let (sin, cos) = (theta[near] - theta[far]).sin_cos();
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

    // Mixed, emitted in both orderings since mixed partials commute.
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

/// One enforced branch limit.
#[derive(Clone, Copy, Debug)]
struct LimitRow {
    branch: usize,
    terminal: Terminal,
    rate_squared: f64,
}

/// The AC-OPF as a [`NonlinearProblem`].
pub struct AcOpf {
    network: AcOpfNetwork,
    options: AcOpfOptions,
    ybus: YBusSparse,
    params: Vec<BranchParams>,
    limit_rows: Vec<LimitRow>,
    /// Cached bus template, so evaluations can reuse it.
    template: Vec<Bus>,
    n_vars: usize,
    theta: usize,
    vm: usize,
    pg: usize,
    qg: usize,
}

impl AcOpf {
    pub fn build(network: AcOpfNetwork, options: AcOpfOptions) -> Result<Self, OpfError> {
        let n = network.n_buses;
        let g = network.generators.len();
        let mut ybus_builder =
            crate::network::build_ybus(n, &network.lines, &network.transformers);
        for &(at, y) in &network.shunts {
            ybus_builder.add(at, at, y);
        }
        let ybus = ybus_builder.finish();
        let params = branch_params(&network.lines, &network.transformers);

        let mut limit_rows = Vec::new();
        if options.enforce_limits {
            for (flat, rate) in network.limits.iter() {
                let Some(rate) = rate else { continue };
                if *flat >= params.len() {
                    continue;
                }
                for terminal in [Terminal::From, Terminal::To] {
                    limit_rows.push(LimitRow {
                        branch: *flat,
                        terminal,
                        rate_squared: rate * rate,
                    });
                }
            }
            limit_rows.sort_by_key(|l| (l.branch, matches!(l.terminal, Terminal::To)));
        }

        let template: Vec<Bus> = (0..n)
            .map(|i| Bus {
                idx: i,
                bus_type: if i == network.reference { BusType::Slack } else { BusType::PQ },
                voltage_mag: 1.0,
                voltage_ang: 0.0,
                p_spec: 0.0,
                q_spec: 0.0,
                q_min: f64::NEG_INFINITY,
                q_max: f64::INFINITY,
                u_rated: 1.0,
                zip_terms: Vec::new(),
            })
            .collect();

        Ok(Self {
            options,
            ybus,
            params,
            limit_rows,
            template,
            n_vars: 2 * n + 2 * g,
            theta: 0,
            vm: n,
            pg: 2 * n,
            qg: 2 * n + g,
            network,
        })
    }

    pub fn network(&self) -> &AcOpfNetwork {
        &self.network
    }

    /// The bus vector at a given state, for tests that want to re-derive the
    /// power flow independently of the solver's own residual.
    #[doc(hidden)]
    pub fn buses_for_test(&self, angles: &[f64], magnitudes: &[f64]) -> Vec<Bus> {
        let mut buses = self.template.clone();
        for i in 0..self.network.n_buses {
            buses[i].voltage_ang = angles[i];
            buses[i].voltage_mag = magnitudes[i];
        }
        buses
    }

    /// The assembled Y-bus, shunts included.
    #[doc(hidden)]
    pub fn ybus_for_test(&self) -> &YBusSparse {
        &self.ybus
    }

    fn buses_at(&self, x: &[f64]) -> Vec<Bus> {
        let n = self.network.n_buses;
        let mut buses = self.template.clone();
        for i in 0..n {
            buses[i].voltage_ang = x[self.theta + i];
            buses[i].voltage_mag = x[self.vm + i];
        }
        buses
    }

    /// Solves, and interprets the result in physical units.
    pub fn solve(&self) -> Result<AcOpfResult, OpfError> {
        let solution = crate::opf::nlp::solve(self, &self.options.nlp)?;
        Ok(self.interpret(&solution))
    }

    pub fn interpret(&self, solution: &NlpSolution) -> AcOpfResult {
        let n = self.network.n_buses;
        let g = self.network.generators.len();
        let base = self.network.base_mva;
        let x = &solution.x;

        let angles: Vec<f64> = (0..n).map(|i| x[self.theta + i]).collect();
        let magnitudes: Vec<f64> = (0..n).map(|i| x[self.vm + i]).collect();
        let p_gen: Vec<f64> = (0..g).map(|k| x[self.pg + k] * base).collect();
        let q_gen: Vec<f64> = (0..g).map(|k| x[self.qg + k] * base).collect();

        let v: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::from_polar(magnitudes[i], angles[i]))
            .collect();
        let mut flows = Vec::with_capacity(self.params.len());
        for branch in &self.params {
            let (p, q) = crate::branch_flow::terminal_flow(branch, Terminal::From, &v);
            flows.push((p * base, q * base));
        }

        // **The price sign.** The balance row is written
        //
        //     c_i = P_inj(θ,|V|) + P_load − Σ P_g = 0
        //
        // so raising demand by δ shifts the row by −δ, and with the
        // Lagrangian written `L = f − yᵀc` the multiplier is ∂f/∂c. One more
        // MW of demand therefore costs **−y**, not `y`.
        //
        // Worth the arithmetic rather than a guess: the unnegated version
        // produced prices of exactly the right magnitude with the wrong sign
        // on every bus of every case — the failure mode `dc`'s module docs
        // warn about, plausible everywhere and wrong everywhere.
        // `opf_ac_test.rs` pins it against a numerical ∂cost/∂load rather than
        // against this reasoning.
        let lmp_p: Vec<f64> = (0..n).map(|i| -solution.y[i] / base).collect();
        let lmp_q: Vec<f64> = (0..n).map(|i| -solution.y[n + i] / base).collect();

        AcOpfResult {
            status: solution.status,
            objective: solution.objective,
            angles,
            magnitudes,
            p_gen,
            q_gen,
            generator_index: self.network.generators.iter().map(|x| x.index).collect(),
            flows,
            lmp_p,
            lmp_q,
            iterations: solution.iterations,
            violation: solution.violation,
        }
    }
}

/// A solved AC-OPF, in physical units.
#[derive(Clone, Debug)]
pub struct AcOpfResult {
    /// [`OptStatus::Optimal`] here means the first-order conditions hold — a
    /// **local** optimum. AC-OPF is nonconvex and no solver certifies more.
    pub status: OptStatus,
    /// Total cost, \\$/h.
    pub objective: f64,
    pub angles: Vec<f64>,
    pub magnitudes: Vec<f64>,
    pub p_gen: Vec<f64>,
    pub q_gen: Vec<f64>,
    pub generator_index: Vec<usize>,
    /// `(P, Q)` entering each branch at its from-terminal, MW and MVAr.
    pub flows: Vec<(f64, f64)>,
    /// Active-power locational marginal price per bus, \\$/MWh.
    pub lmp_p: Vec<f64>,
    /// Reactive-power price per bus, \\$/MVArh.
    pub lmp_q: Vec<f64>,
    pub iterations: usize,
    /// Largest constraint violation, per-unit. **Read this alongside the
    /// objective**: on a nonconvex problem a lower cost at an infeasible point
    /// is not a better answer, so a comparison that reports only the objective
    /// can be badly misleading.
    pub violation: f64,
}

impl NonlinearProblem for AcOpf {
    fn n_vars(&self) -> usize {
        self.n_vars
    }

    fn n_constraints(&self) -> usize {
        2 * self.network.n_buses + self.limit_rows.len()
    }

    fn var_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.network.n_buses;
        let g = self.network.generators.len();
        let mut lower = vec![f64::NEG_INFINITY; self.n_vars];
        let mut upper = vec![f64::INFINITY; self.n_vars];

        // The reference angle is pinned by equal bounds, which `nlp` rewrites
        // into an equality row rather than trying to put a barrier on it.
        lower[self.theta + self.network.reference] = 0.0;
        upper[self.theta + self.network.reference] = 0.0;

        for i in 0..n {
            lower[self.vm + i] = self.network.v_min[i];
            upper[self.vm + i] = self.network.v_max[i];
        }
        for (k, unit) in self.network.generators.iter().enumerate() {
            lower[self.pg + k] = unit.p_min;
            upper[self.pg + k] = unit.p_max;
            lower[self.qg + k] = unit.q_min;
            upper[self.qg + k] = unit.q_max;
        }
        let _ = g;
        (lower, upper)
    }

    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.network.n_buses;
        let mut lower = vec![0.0; 2 * n];
        let mut upper = vec![0.0; 2 * n];
        for limit in &self.limit_rows {
            lower.push(f64::NEG_INFINITY);
            upper.push(limit.rate_squared);
        }
        (lower, upper)
    }

    fn objective(&self, x: &[f64]) -> f64 {
        self.network
            .generators
            .iter()
            .enumerate()
            .map(|(k, unit)| match &unit.cost {
                Some(curve) => curve.evaluate(x[self.pg + k]),
                None => 0.0,
            })
            .sum()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_vars];
        for (k, unit) in self.network.generators.iter().enumerate() {
            if let Some(curve) = &unit.cost {
                out[self.pg + k] = curve.marginal(x[self.pg + k]);
            }
        }
        out
    }

    fn constraints(&self, x: &[f64]) -> Vec<f64> {
        let n = self.network.n_buses;
        let buses = self.buses_at(x);
        let (p_inj, q_inj) = power_injections(&buses, &self.ybus);

        let mut out = vec![0.0; 2 * n + self.limit_rows.len()];
        for i in 0..n {
            out[i] = p_inj[i] + self.network.p_load[i];
            out[n + i] = q_inj[i] + self.network.q_load[i];
        }
        for (k, unit) in self.network.generators.iter().enumerate() {
            out[unit.bus] -= x[self.pg + k];
            out[n + unit.bus] -= x[self.qg + k];
        }

        let theta: Vec<f64> = (0..n).map(|i| x[self.theta + i]).collect();
        let vm: Vec<f64> = (0..n).map(|i| x[self.vm + i]).collect();
        for (r, limit) in self.limit_rows.iter().enumerate() {
            let d = terminal_derivatives(
                &self.params[limit.branch],
                limit.terminal,
                &theta,
                &vm,
            );
            out[2 * n + r] = d.p * d.p + d.q * d.q;
        }
        out
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.network.n_buses;
        let buses = self.buses_at(x);
        let mut out = injection_jacobian(&buses, &self.ybus);

        for (k, unit) in self.network.generators.iter().enumerate() {
            out.push((unit.bus, self.pg + k, -1.0));
            out.push((n + unit.bus, self.qg + k, -1.0));
        }

        let theta: Vec<f64> = (0..n).map(|i| x[self.theta + i]).collect();
        let vm: Vec<f64> = (0..n).map(|i| x[self.vm + i]).collect();
        for (r, limit) in self.limit_rows.iter().enumerate() {
            let branch = &self.params[limit.branch];
            let d = terminal_derivatives(branch, limit.terminal, &theta, &vm);
            let columns = self.limit_columns(branch, limit.terminal);
            // ∇(P² + Q²) = 2P∇P + 2Q∇Q
            for local in 0..4 {
                out.push((
                    2 * n + r,
                    columns[local],
                    2.0 * (d.p * d.dp[local] + d.q * d.dq[local]),
                ));
            }
        }
        out
    }

    fn lagrangian_hessian(&self, x: &[f64], y: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.network.n_buses;

        // The objective's own curvature: quadratic cost curves only, so the
        // block is diagonal on the P_g columns.
        let mut out = Vec::new();
        for (k, unit) in self.network.generators.iter().enumerate() {
            if let Some(CostCurve::Polynomial { coefficients }) = &unit.cost {
                if coefficients.len() > 2 {
                    out.push((self.pg + k, self.pg + k, 2.0 * coefficients[2]));
                }
            }
        }

        // −Σ yᵢ ∇²cᵢ for the balance rows. The sign is the module's
        // convention (L = f − yᵀc), so the multipliers are negated on the way
        // in rather than the result negated on the way out.
        let buses = self.buses_at(x);
        let lambda: Vec<f64> = (0..n).map(|i| -y[i]).collect();
        let mu: Vec<f64> = (0..n).map(|i| -y[n + i]).collect();
        out.extend(weighted_injection_hessian(&buses, &self.ybus, &lambda, &mu));

        // −Σ y_r ∇²(P² + Q²) for the limit rows, where
        //   ∇²(P² + Q²) = 2(∇P∇Pᵀ + P∇²P + ∇Q∇Qᵀ + Q∇²Q).
        // The outer products are what a limit contributes beyond the
        // injections, and they stay inside the branch's own 2-bus block — so
        // limits add no fill either.
        let theta: Vec<f64> = (0..n).map(|i| x[self.theta + i]).collect();
        let vm: Vec<f64> = (0..n).map(|i| x[self.vm + i]).collect();
        for (r, limit) in self.limit_rows.iter().enumerate() {
            let weight = -y[2 * n + r];
            if weight == 0.0 {
                continue;
            }
            let branch = &self.params[limit.branch];
            let d = terminal_derivatives(branch, limit.terminal, &theta, &vm);
            let columns = self.limit_columns(branch, limit.terminal);
            for a in 0..4 {
                for b in 0..4 {
                    let value = 2.0
                        * (d.dp[a] * d.dp[b]
                            + d.p * d.d2p[a][b]
                            + d.dq[a] * d.dq[b]
                            + d.q * d.d2q[a][b]);
                    if value != 0.0 {
                        out.push((columns[a], columns[b], weight * value));
                    }
                }
            }
        }

        out
    }

    /// A flat voltage profile with the dispatch already roughly meeting
    /// demand.
    ///
    /// Starting each generator at the midpoint of its box is the obvious
    /// choice and a bad one. On pglib's cases `p_min` is usually zero, so
    /// midpoints sum to about half of total `p_max` — some 23% short of load
    /// on every fixture here. The barrier method then opens on a point
    /// violating the balance equations by hundreds of megawatts, and spends
    /// its early iterations dragging the dispatch across the feasible set
    /// rather than optimizing over it. On `case5_pjm` and `case118_ieee` it
    /// never recovered, stalling at the iteration limit with violations of
    /// 1.3 and 5.5 per-unit while smaller cases converged.
    ///
    /// Distributing demand proportionally across the generators' ranges costs
    /// nothing and starts the solve near the balance manifold instead.
    /// Nothing about correctness depends on it — a nonconvex solver's starting
    /// point selects *which* local optimum is found, not whether the answer is
    /// one — but on these cases it is the difference between converging and
    /// not.
    fn initial_point(&self) -> Vec<f64> {
        let n = self.network.n_buses;
        let mut x = vec![0.0; self.n_vars];
        for i in 0..n {
            x[self.vm + i] = self.network.v_start.unwrap_or(1.0)
                .clamp(self.network.v_min[i], self.network.v_max[i]);
        }

        let demand: f64 = self.network.p_load.iter().sum();
        let floor: f64 = self.network.generators.iter().map(|u| u.p_min.max(0.0)).sum();
        let headroom: f64 = self
            .network
            .generators
            .iter()
            .map(|u| (u.p_max - u.p_min.max(0.0)).max(0.0))
            .sum();
        // Losses are a few percent of demand and are pure headroom here, so
        // aiming slightly high beats aiming exactly at load.
        let share = if headroom > 0.0 {
            ((demand * 1.05 - floor) / headroom).clamp(0.0, 1.0)
        } else {
            0.0
        };

        for (k, unit) in self.network.generators.iter().enumerate() {
            let low = unit.p_min.max(0.0);
            x[self.pg + k] = if unit.p_max > low {
                low + share * (unit.p_max - low)
            } else {
                midpoint(unit.p_min, unit.p_max)
            };
            // No comparable heuristic for reactive power: how much a network
            // needs depends on the voltage profile being solved for, so the
            // box midpoint is as good a guess as any.
            x[self.qg + k] = midpoint(unit.q_min, unit.q_max);
        }
        x
    }
}

impl AcOpf {
    /// Global column indices for a terminal's four local state variables.
    fn limit_columns(&self, branch: &BranchParams, terminal: Terminal) -> [usize; 4] {
        let (near, far) = match terminal {
            Terminal::From => (branch.from, branch.to),
            Terminal::To => (branch.to, branch.from),
        };
        [self.theta + near, self.theta + far, self.vm + near, self.vm + far]
    }
}

fn midpoint(low: f64, high: f64) -> f64 {
    match (low.is_finite(), high.is_finite()) {
        (true, true) => 0.5 * (low + high),
        (true, false) => low + 1.0,
        (false, true) => high - 1.0,
        (false, false) => 0.0,
    }
}
