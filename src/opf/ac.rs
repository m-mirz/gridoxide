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
//! - **Angle-difference limits**, \\(\theta_{min} \le \theta_f - \theta_t \le
//!   \theta_{max}\\) per limited branch: linear rows, so a constant Jacobian
//!   and nothing in the Hessian. MATPOWER's `ANGMIN`/`ANGMAX`.
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

use crate::branch_flow::{BranchParams, Terminal, branch_params};
use crate::injection_hessian::{injection_jacobian, weighted_injection_hessian};
use crate::network::{YBusSparse, power_injections};
use crate::opf::model::CostCurve;
use crate::opf::nlp::{NlpOptions, NlpSolution, NonlinearProblem};
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
    /// Bounds on the angle difference across a branch, enforced whatever
    /// [`AcOpfOptions::enforce_limits`] says: that switch is about apparent
    /// power.
    pub angle_limits: Vec<AngleLimit>,
}

/// `min ≤ θ_from − θ_to ≤ max`, radians; either side may be infinite.
#[derive(Clone, Copy, Debug)]
pub struct AngleLimit {
    pub from: usize,
    pub to: usize,
    pub min: f64,
    pub max: f64,
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
    /// Where every solve starts; see [`AcOpfStart`].
    pub start: AcOpfStart,
    pub nlp: NlpOptions,
}

impl Default for AcOpfOptions {
    fn default() -> Self {
        Self {
            default_v_min: 0.9,
            default_v_max: 1.1,
            enforce_limits: true,
            start: AcOpfStart::default(),
            nlp: NlpOptions::default(),
        }
    }
}

/// The starting point of a solve. Both start every bus at
/// [`AcOpfNetwork::v_start`] (1 p.u. unless set), clamped into its limits,
/// and zero angle, and every generator's reactive output at the middle of its
/// range; they differ in active output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AcOpfStart {
    /// Demand plus 5%, spread over the generators in proportion to their
    /// ranges above `max(p_min, 0)`. See `initial_point` for why.
    #[default]
    Proportional,
    /// Every generator at the middle of its active-power range: MATPOWER's
    /// and PowerModels' flat start, for comparing solvers from one point.
    Flat,
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
    /// Generator index → its epigraph column, for those that have one. No
    /// block-start field beside it: unlike `theta`/`vm`/`pg`/`qg`, nothing
    /// indexes this block positionally, because a generator without a
    /// piecewise cost has no column in it at all.
    cost_col: Vec<Option<usize>>,
    /// `(epigraph column, generator column, slope, intercept)` per segment,
    /// one constraint row each.
    cost_rows: Vec<(usize, usize, f64, f64)>,
}

impl AcOpf {
    pub fn build(network: AcOpfNetwork, options: AcOpfOptions) -> Result<Self, OpfError> {
        let n = network.n_buses;
        let g = network.generators.len();
        let mut ybus_builder = crate::network::build_ybus(n, &network.lines, &network.transformers);
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
                bus_type: if i == network.reference {
                    BusType::Slack
                } else {
                    BusType::PQ
                },
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

        // **Piecewise-linear costs enter through an epigraph**, exactly as in
        // `dc`: `min f(p)` becomes `min z` subject to `z ≥ (segment i)(p)`.
        //
        // Here it buys something DC does not need. `CostCurve::evaluate` and
        // `marginal` handle a piecewise curve correctly, so feeding it to the
        // objective directly gives the right *values* — but the objective is
        // then only C⁰, and a Newton method assumes C². The gradient jumps at
        // every breakpoint, and an optimum very often sits exactly on one,
        // since a breakpoint is where marginal cost changes. The epigraph
        // makes the objective linear in `z` and pushes the kinks into
        // constraints, which the solver handles exactly.
        let mut cost_col = vec![None; g];
        let mut cost_rows = Vec::new();
        let mut n_cost = 0;
        for (k, unit) in network.generators.iter().enumerate() {
            if let Some(CostCurve::PiecewiseLinear { points }) = &unit.cost {
                if !unit.cost.as_ref().is_some_and(|c| c.is_convex()) {
                    return Err(OpfError::Backend(format!(
                        "generator {} has a non-convex piecewise-linear cost; its epigraph \
                         describes the convex envelope instead, which charges less than the \
                         curve does",
                        unit.index
                    )));
                }
                let column = 2 * n + 2 * g + n_cost;
                cost_col[k] = Some(column);
                n_cost += 1;
                for pair in points.windows(2) {
                    let ((p0, c0), (p1, c1)) = (pair[0], pair[1]);
                    if p1 == p0 {
                        continue;
                    }
                    let slope = (c1 - c0) / (p1 - p0);
                    cost_rows.push((column, 2 * n + k, slope, c0 - slope * p0));
                }
            }
        }

        Ok(Self {
            options,
            ybus,
            params,
            limit_rows,
            template,
            n_vars: 2 * n + 2 * g + n_cost,
            theta: 0,
            vm: n,
            pg: 2 * n,
            qg: 2 * n + g,
            cost_col,
            cost_rows,
            network,
        })
    }

    pub fn network(&self) -> &AcOpfNetwork {
        &self.network
    }

    /// Changes where later solves start; the model is otherwise unchanged.
    pub fn set_start(&mut self, start: AcOpfStart) {
        self.options.start = start;
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
        2 * self.network.n_buses
            + self.limit_rows.len()
            + self.network.angle_limits.len()
            + self.cost_rows.len()
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
        // Epigraph variables are free; the segment rows bound them from below
        // and minimizing drives them down to the curve.
        for column in self.cost_col.iter().flatten() {
            lower[*column] = f64::NEG_INFINITY;
            upper[*column] = f64::INFINITY;
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
        for angle in &self.network.angle_limits {
            lower.push(angle.min);
            upper.push(angle.max);
        }
        // `z − m·p ≥ k`, one row per segment.
        for &(_, _, _, intercept) in &self.cost_rows {
            lower.push(intercept);
            upper.push(f64::INFINITY);
        }
        (lower, upper)
    }

    fn objective(&self, x: &[f64]) -> f64 {
        self.network
            .generators
            .iter()
            .enumerate()
            .map(|(k, unit)| match &unit.cost {
                // A piecewise-linear unit's cost is carried by its epigraph
                // variable, not evaluated on the curve — that is what keeps
                // the objective smooth.
                Some(CostCurve::PiecewiseLinear { .. }) => {
                    x[self.cost_col[k].expect("assigned in build")]
                }
                Some(curve) => curve.evaluate(x[self.pg + k]),
                None => 0.0,
            })
            .sum()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_vars];
        for (k, unit) in self.network.generators.iter().enumerate() {
            match &unit.cost {
                Some(CostCurve::PiecewiseLinear { .. }) => {
                    out[self.cost_col[k].expect("assigned in build")] = 1.0;
                }
                Some(curve) => out[self.pg + k] = curve.marginal(x[self.pg + k]),
                None => {}
            }
        }
        out
    }

    fn constraints(&self, x: &[f64]) -> Vec<f64> {
        let n = self.network.n_buses;
        let buses = self.buses_at(x);
        let (p_inj, q_inj) = power_injections(&buses, &self.ybus);

        let mut out = vec![0.0; self.n_constraints()];
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
            let d = terminal_derivatives(&self.params[limit.branch], limit.terminal, &theta, &vm);
            out[2 * n + r] = d.p * d.p + d.q * d.q;
        }
        let base = 2 * n + self.limit_rows.len();
        for (r, angle) in self.network.angle_limits.iter().enumerate() {
            out[base + r] = x[self.theta + angle.from] - x[self.theta + angle.to];
        }
        let base = base + self.network.angle_limits.len();
        for (r, &(epigraph, column, slope, _)) in self.cost_rows.iter().enumerate() {
            out[base + r] = x[epigraph] - slope * x[column];
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
        // The angle and epigraph rows are linear, so their Jacobian is
        // constant — and for the same reason they contribute nothing to the
        // Hessian.
        let base = 2 * n + self.limit_rows.len();
        for (r, angle) in self.network.angle_limits.iter().enumerate() {
            out.push((base + r, self.theta + angle.from, 1.0));
            out.push((base + r, self.theta + angle.to, -1.0));
        }
        let base = base + self.network.angle_limits.len();
        for (r, &(epigraph, column, slope, _)) in self.cost_rows.iter().enumerate() {
            out.push((base + r, epigraph, 1.0));
            out.push((base + r, column, -slope));
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
                    // Kept when zero: IPOPT fixes the Hessian's pattern at
                    // the start, so it must not depend on the point.
                    out.push((columns[a], columns[b], weight * value));
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
    /// not. [`AcOpfStart::Flat`] takes the midpoints anyway, for a caller
    /// comparing solvers from the start they share.
    fn initial_point(&self) -> Vec<f64> {
        let n = self.network.n_buses;
        let mut x = vec![0.0; self.n_vars];
        for i in 0..n {
            x[self.vm + i] = self
                .network
                .v_start
                .unwrap_or(1.0)
                .clamp(self.network.v_min[i], self.network.v_max[i]);
        }

        let demand: f64 = self.network.p_load.iter().sum();
        let floor: f64 = self
            .network
            .generators
            .iter()
            .map(|u| u.p_min.max(0.0))
            .sum();
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
            x[self.pg + k] = if self.options.start == AcOpfStart::Flat {
                midpoint(unit.p_min, unit.p_max)
            } else if unit.p_max > low {
                low + share * (unit.p_max - low)
            } else {
                midpoint(unit.p_min, unit.p_max)
            };
            // No comparable heuristic for reactive power: how much a network
            // needs depends on the voltage profile being solved for, so the
            // box midpoint is as good a guess as any.
            x[self.qg + k] = midpoint(unit.q_min, unit.q_max);
        }
        // Seed each epigraph variable on its own curve, so the segment rows
        // start satisfied rather than needing to be dragged into feasibility.
        for (k, unit) in self.network.generators.iter().enumerate() {
            if let (Some(column), Some(curve)) = (self.cost_col[k], unit.cost.as_ref()) {
                x[column] = curve.evaluate(x[self.pg + k]);
            }
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
        [
            self.theta + near,
            self.theta + far,
            self.vm + near,
            self.vm + far,
        ]
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
