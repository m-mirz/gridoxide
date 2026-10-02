//! DC optimal power flow.
//!
//! *What should each generator produce, so that demand is met at least cost
//! without overloading anything?* — under the DC approximation, which makes
//! the answer a convex quadratic program and therefore provably optimal rather
//! than merely converged.
//!
//! # The problem
//!
//! \\[ \min_{P_g,\ \theta,\ s}\ \sum_g \left(c_{2g}P_g^2 + c_{1g}P_g +
//!     c_{0g}\right) + \sum_d \pi\, s_d \\]
//!
//! subject to, at every bus \\(i\\),
//!
//! \\[ \sum_{g \in i} P_g + s_i - (B\theta)_i = P^{load}_i - c_i \\]
//!
//! plus the generation box \\(P^{min}_g \le P_g \le P^{max}_g\\), the shedding
//! box \\(0 \le s_d \le P^{load}_d\\), branch limits
//! \\(-\text{rate} \le b(\theta_f - \theta_t - \alpha) \le \text{rate}\\), and
//! \\(\theta_{ref} = 0\\).
//!
//! **The balance row is written generation-minus-outflow equals load**, rather
//! than the other way round, and that is a deliberate choice about the duals
//! rather than cosmetics. With load on the right-hand side, the row's dual is
//! \\(\partial(\text{cost})/\partial(\text{load})\\) directly — which *is* the
//! locational marginal price, positive, in the sign anyone reading it expects.
//! The opposite orientation gives the negated quantity and an easy sign error
//! in exactly the output most likely to be published.
//!
//! # Why θ and not PTDF
//!
//! Branch limits could be written against injections through a PTDF matrix,
//! eliminating the angles. That matrix is dense — 1.19 GB on `case9241pegase`,
//! as `docs/src/powerflow/dc.md` records — where the \\(\theta\\) formulation
//! keeps the sparsity the rest of this crate is built around.
//! [`DcSensitivity`](crate::linear::sensitivity::DcSensitivity) stays a
//! screening tool rather than becoming a constraint builder.
//!
//! # Which controls DC admits
//!
//! Generator active power, load shedding, and phase-shifter angles are all
//! linear here. **Transformer tap ratio is not**: DC uses \\(b/k\\), so making
//! \\(k\\) a variable is nonconvex and would defeat the reason for doing DC
//! first. Reactive controls have no meaning at all, since DC has no reactive
//! power and holds \\(|V| = 1\\). Both arrive with AC-OPF.
//!
//! Phase shifters are read from the network but held at their present values
//! rather than optimized — they enter \\(c_i\\) as a constant. Making them
//! variables is a small extension this module is shaped for but does not yet
//! take.

use std::collections::HashMap;

use crate::linear::btheta::DcBranch;
use crate::opf::model::{CostCurve, OpfData};
use crate::opf::{LinearProgram, OpfError, OptStatus, Solution};

/// A generator, resolved onto a bus index with everything in per-unit.
#[derive(Clone, Debug)]
pub struct DcGenerator {
    /// Its `index` in the OPF document, carried through so results can be
    /// matched back.
    pub index: usize,
    pub bus: usize,
    pub p_min: f64,
    pub p_max: f64,
    /// Cost in per-unit argument — see [`CostCurve::to_per_unit`]. `None`
    /// means this unit is free, which makes the problem degenerate but is not
    /// an error.
    pub cost: Option<CostCurve>,
}

/// A load, resolved onto a bus index, per-unit.
#[derive(Clone, Debug)]
pub struct DcLoad {
    pub bus: usize,
    pub p: f64,
}

/// Everything DC-OPF needs, with every quantity already in per-unit on
/// [`base_mva`](Self::base_mva).
#[derive(Clone, Debug)]
pub struct DcOpfNetwork {
    pub n_buses: usize,
    /// The bus whose angle is pinned to zero.
    pub reference: usize,
    pub branches: Vec<DcBranch>,
    /// Flat branch index → rating in per-unit. A branch absent from this map,
    /// or mapped to `None`, is unlimited.
    pub limits: HashMap<usize, Option<f64>>,
    pub generators: Vec<DcGenerator>,
    pub loads: Vec<DcLoad>,
    pub base_mva: f64,
}

impl DcOpfNetwork {
    /// Builds the network from a PGM document and its companion OPF document.
    ///
    /// Three things need care, and each is a place a naive assembly goes
    /// wrong:
    ///
    /// - **The per-unit base must be the OPF document's.** `pgm_to_network`
    ///   takes `s_base_va`, and passing anything other than
    ///   `base_mva · 1e6` would put the network's per-unit powers on a
    ///   different base from the cost curves', which is invisible until the
    ///   dispatch comes out scaled.
    /// - **The synthesized source branches and buses are dropped.** Converting
    ///   a PGM document appends a virtual slack bus per `source`, joined by an
    ///   impedance branch. That is right for a power flow, where the source's
    ///   output is an unknown; here the source's generator is a decision
    ///   variable listed in the OPF document, so the virtual pair would double
    ///   it. The physical end of a source branch is the reference bus.
    /// - **Branch limits are keyed by PGM component id, not by flat index.**
    ///   `PgmNetwork::branch_idx` is the translation.
    ///
    /// `approximation` picks how branch susceptance is formed, and it is not a
    /// cosmetic choice — see [`DcOpfOptions`] for the measured effect on the
    /// published baseline. [`DcApproximation::IgnoreG`] is what OPF should
    /// normally use.
    ///
    /// [`DcApproximation::IgnoreG`]: crate::linear::DcApproximation::IgnoreG
    pub fn from_pgm(
        input: crate::pgm::PgmInput,
        opf: &OpfData,
        freq_hz: f64,
        approximation: crate::linear::DcApproximation,
    ) -> Result<Self, OpfError> {
        let base_mva = opf.base_mva;
        let s_base_va = base_mva * 1e6;

        let loads_raw: Vec<(u64, f64)> = input
            .data
            .sym_load
            .iter()
            .filter(|l| l.status != 0)
            .map(|l| (l.node, l.p_specified))
            .collect();
        let n_physical = input.data.node.len();

        let net = crate::pgm::pgm_to_network(input, s_base_va, freq_hz);
        let branches: Vec<DcBranch> = crate::linear::btheta::dc_branches(
            &net.lines,
            &net.transformers,
            crate::linear::DcOptions { approximation, ..crate::linear::DcOptions::default() },
        )
        .into_iter()
        // A branch touching a virtual bus is a synthesized source connection.
        .filter(|b| b.from < n_physical && b.to < n_physical)
        .collect();

        // The reference is the physical bus a source attaches to. Falling back
        // to bus 0 keeps a source-less document solvable rather than erroring
        // on something the OPF itself does not need.
        let reference = net
            .source_branch_idx
            .values()
            .filter_map(|&flat| net.lines.get(flat))
            .map(|line| line.to.min(line.from))
            .find(|&bus| bus < n_physical)
            .unwrap_or(0);

        let mut limits = HashMap::new();
        for limit in &opf.branch_limit {
            let Some(&flat) = net.branch_idx.get(&limit.id) else { continue };
            limits.insert(flat, limit.rate_pu(base_mva));
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
            generators.push(DcGenerator {
                index: g.index,
                bus,
                p_min,
                p_max,
                cost: g.cost.as_ref().map(|c| c.to_per_unit(base_mva)),
            });
        }

        let mut loads = Vec::new();
        for (node, p_specified) in loads_raw {
            let Some(&bus) = net.node_idx.get(&node) else { continue };
            loads.push(DcLoad { bus, p: p_specified / s_base_va });
        }

        Ok(Self {
            n_buses: n_physical,
            reference,
            branches,
            limits,
            generators,
            loads,
            base_mva,
        })
    }
}

/// Options for a DC-OPF.
///
/// # Which susceptance the branches get
///
/// Not held here — it is fixed when the network is built, by the
/// `approximation` argument to [`DcOpfNetwork::from_pgm`] — but this is where
/// a caller will look for it, so the reasoning is recorded here.
///
/// DC has two defensible susceptances, and OPF is far more sensitive to the
/// choice than DC power flow is, because it moves the *constraint set* and not
/// just the reported flows. Against pglib-opf's published DC objectives:
///
/// | case | `IgnoreR` (`b = 1/x`) | `IgnoreG` (`b = x/(r²+x²)`) |
/// |---|---|---|
/// | `case3_lmbd`   | −0.037% | −0.0001% |
/// | `case5_pjm`    | −0.001% | −0.001% |
/// | `case14_ieee`  | +0.001% | +0.001% |
/// | `case30_ieee`  | **+0.423%** | −0.025% |
/// | `case118_ieee` | +0.034% | −0.013% |
///
/// `case30_ieee` is the case that separates them, and it is worth
/// understanding rather than tuning around: its branch 1→2 has
/// `r = 0.0192, x = 0.0575`, an r/x ratio high enough that `1/x` overstates
/// the susceptance by 10%. That branch is congested at the optimum, so the
/// overstatement lands directly on a binding constraint and lifts the cost.
///
/// So [`IgnoreG`] is the right default *for OPF*, and it is the same
/// principle that makes [`IgnoreR`] the right default for the `dc` power flow
/// command: each defaults to what the reference implementations in its own
/// domain compute. PowerModels — which produced these published numbers —
/// builds its DC model from the full series admittance; MATPOWER's `makeBdc`,
/// pandapower and lightsim2grid all use `1/x` for power flow.
///
/// [`IgnoreG`]: crate::linear::DcApproximation::IgnoreG
/// [`IgnoreR`]: crate::linear::DcApproximation::IgnoreR
#[derive(Clone, Copy, Debug)]
pub struct DcOpfOptions {
    /// Price of shedding load, \\$/MWh. Deliberately far above any plausible
    /// generator marginal cost so that shedding is a last resort rather than
    /// an economic choice — but finite, so that an otherwise-infeasible case
    /// still returns an answer that says *where* it could not be served.
    pub shed_price: f64,
    /// Whether load shedding is offered at all. With it off, a case that
    /// cannot serve its demand comes back infeasible, which is sometimes the
    /// answer wanted.
    pub allow_shedding: bool,
}

impl Default for DcOpfOptions {
    fn default() -> Self {
        Self { shed_price: 10_000.0, allow_shedding: true }
    }
}

/// Where each variable and row lives in the assembled program.
///
/// Kept so a [`Solution`] can be read back without re-deriving the layout, and
/// so the mapping is stated once rather than implied by construction order in
/// two places.
#[derive(Clone, Debug)]
struct Layout {
    theta: usize,
    generator: usize,
    shed: usize,
    n_shed: usize,
    /// Generator index → its epigraph column, for those that have one. There
    /// is no block-start field beside it: unlike the other blocks nothing
    /// indexes this one positionally, since a generator without a piecewise
    /// cost has no column here at all.
    cost_col: Vec<Option<usize>>,
    balance_row: usize,
    limit_rows: Vec<(usize, usize)>,
}

/// A built DC-OPF, ready to hand to a solver.
pub struct DcOpf {
    network: DcOpfNetwork,
    options: DcOpfOptions,
    layout: Layout,
    problem: LinearProgram,
}

/// One constraint that is actually binding at the optimum.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    /// Flat branch index.
    pub branch: usize,
    /// Its flow, MW — signed, so the sign says which limit is active.
    pub flow: f64,
    /// The rating it reached, MW.
    pub rate: f64,
    /// Shadow price, \\$/MWh: what one more MW of capacity on this branch
    /// would save.
    pub price: f64,
}

/// A solved DC-OPF, in physical units.
#[derive(Clone, Debug)]
pub struct DcOpfResult {
    pub status: OptStatus,
    /// Total cost, \\$/h.
    pub objective: f64,
    /// Dispatch per generator, MW, in the order [`DcOpfNetwork::generators`]
    /// lists them.
    pub dispatch: Vec<f64>,
    /// Bus voltage angles, radians.
    pub angles: Vec<f64>,
    /// Flow per branch, MW, indexed by flat branch index.
    pub flows: Vec<f64>,
    /// **Locational marginal price** per bus, \\$/MWh — the cost of serving
    /// one more MW of demand there. Equal at every bus when nothing is
    /// congested; the spread between buses is the congestion.
    pub lmp: Vec<f64>,
    /// Load shed per entry of [`DcOpfNetwork::loads`], MW. All zero on a case
    /// that can serve its demand.
    pub shed: Vec<f64>,
    /// Branches at their limit, with what relieving them is worth.
    pub binding: Vec<Binding>,
}

impl DcOpf {
    /// Assembles the program.
    ///
    /// Fails on a non-convex cost curve rather than solving it: a convex
    /// solver would return *a* point, and nothing would mark it as merely
    /// stationary rather than optimal.
    pub fn build(network: DcOpfNetwork, options: DcOpfOptions) -> Result<Self, OpfError> {
        for g in &network.generators {
            if let Some(cost) = &g.cost {
                if !cost.is_convex() {
                    return Err(OpfError::Backend(format!(
                        "generator {} has a non-convex cost curve; the optimum of a \
                         non-convex problem cannot be certified by the checks this \
                         formulation relies on",
                        g.index
                    )));
                }
            }
        }

        let n_buses = network.n_buses;
        let n_gen = network.generators.len();
        let n_shed = if options.allow_shedding { network.loads.len() } else { 0 };

        // **Piecewise-linear costs enter through an epigraph.** A convex
        // piecewise-linear function is the upper envelope of its segments, so
        // `min f(p)` is `min z` subject to `z ≥ (line of segment i)(p)` for
        // every segment — turning a curve the objective cannot express into
        // ordinary rows the objective can. That is exactly why §5.2 of
        // `plans/OPF_PLAN.md` insisted on an LP-capable solver.
        //
        // Exactness needs convexity, which `is_convex` has already required
        // above: on a non-convex curve the same rows describe the convex
        // envelope instead, which is a *relaxation* — a lower cost than the
        // curve actually charges. Silently returning that would be the same
        // class of failure this block exists to fix.
        let mut cost_col = vec![None; n_gen];
        let mut n_cost = 0;
        for (g, generator) in network.generators.iter().enumerate() {
            if matches!(generator.cost, Some(CostCurve::PiecewiseLinear { .. })) {
                cost_col[g] = Some(n_buses + n_gen + n_shed + n_cost);
                n_cost += 1;
            }
        }

        let layout = Layout {
            theta: 0,
            generator: n_buses,
            shed: n_buses + n_gen,
            n_shed,
            cost_col,
            balance_row: 0,
            limit_rows: Vec::new(),
        };
        let n_vars = n_buses + n_gen + n_shed + n_cost;
        let mut lp = LinearProgram::new(n_vars);

        // ── Variables ────────────────────────────────────────────────────
        // Angles are free, except the reference.
        for i in 0..n_buses {
            lp.col_lower[layout.theta + i] = f64::NEG_INFINITY;
            lp.col_upper[layout.theta + i] = f64::INFINITY;
        }
        lp.col_lower[layout.theta + network.reference] = 0.0;
        lp.col_upper[layout.theta + network.reference] = 0.0;

        let mut hessian: Vec<(usize, usize, f64)> = Vec::new();
        for (g, generator) in network.generators.iter().enumerate() {
            let col = layout.generator + g;
            lp.col_lower[col] = generator.p_min;
            lp.col_upper[col] = generator.p_max;

            // Matched exhaustively rather than with `if let`. The `if let`
            // this replaces skipped a piecewise-linear curve in silence,
            // leaving the generator's objective coefficient at zero — so it
            // looked *free* and the optimizer preferred it, producing a wrong
            // dispatch with no error. An exhaustive match makes a future
            // variant a compile error instead.
            match &generator.cost {
                Some(CostCurve::Polynomial { coefficients }) => {
                    // The objective is `½ xᵀQx + cᵀx`, and the cost is
                    // `c₂p² + c₁p + c₀`, so the diagonal Hessian entry is
                    // **twice** the quadratic coefficient. Dropping that factor
                    // of two is silent — it still produces a plausible
                    // dispatch, just the wrong one.
                    if let Some(&c2) = coefficients.get(2) {
                        if c2 != 0.0 {
                            hessian.push((col, col, 2.0 * c2));
                        }
                    }
                    lp.col_cost[col] = coefficients.get(1).copied().unwrap_or(0.0);
                    lp.offset += coefficients.first().copied().unwrap_or(0.0);
                }
                Some(CostCurve::PiecewiseLinear { .. }) => {
                    // The cost is carried by the epigraph variable; the rows
                    // tying it to `p` are added below, once the row block is
                    // being built.
                    let epigraph = layout.cost_col[g].expect("counted above");
                    lp.col_cost[epigraph] = 1.0;
                    lp.col_lower[epigraph] = f64::NEG_INFINITY;
                    lp.col_upper[epigraph] = f64::INFINITY;
                }
                None => {}
            }
        }

        // Shedding is priced per unit of power, converted from $/MWh the same
        // way a linear cost coefficient is.
        for d in 0..n_shed {
            let col = layout.shed + d;
            lp.col_lower[col] = 0.0;
            lp.col_upper[col] = network.loads[d].p.max(0.0);
            lp.col_cost[col] = options.shed_price * network.base_mva;
        }

        if !hessian.is_empty() {
            lp.hessian = Some(hessian);
        }

        // ── Nodal balance ────────────────────────────────────────────────
        // One equality per bus: generation plus shedding, minus net outflow,
        // equals load. See the module docs for why this orientation.
        let mut b_matrix: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n_buses];
        let mut shift_injection = vec![0.0; n_buses];
        for branch in &network.branches {
            let (f, t, b) = (branch.from, branch.to, branch.b);
            *b_matrix[f].entry(f).or_insert(0.0) += b;
            *b_matrix[f].entry(t).or_insert(0.0) -= b;
            *b_matrix[t].entry(t).or_insert(0.0) += b;
            *b_matrix[t].entry(f).or_insert(0.0) -= b;
            // A phase shift is a constant injection, not a variable — it moves
            // to the right-hand side.
            shift_injection[f] += b * branch.shift;
            shift_injection[t] -= b * branch.shift;
        }

        let mut load_at = vec![0.0; n_buses];
        for load in &network.loads {
            load_at[load.bus] += load.p;
        }

        let mut gens_at: Vec<Vec<usize>> = vec![Vec::new(); n_buses];
        for (g, generator) in network.generators.iter().enumerate() {
            gens_at[generator.bus].push(g);
        }
        let mut sheds_at: Vec<Vec<usize>> = vec![Vec::new(); n_buses];
        for d in 0..n_shed {
            sheds_at[network.loads[d].bus].push(d);
        }

        for i in 0..n_buses {
            let mut coefficients: Vec<(usize, f64)> = Vec::new();
            for (&j, &value) in &b_matrix[i] {
                if value != 0.0 {
                    coefficients.push((layout.theta + j, -value));
                }
            }
            for &g in &gens_at[i] {
                coefficients.push((layout.generator + g, 1.0));
            }
            for &d in &sheds_at[i] {
                coefficients.push((layout.shed + d, 1.0));
            }
            let rhs = load_at[i] - shift_injection[i];
            lp.add_row(&coefficients, rhs, rhs);
        }

        // ── Branch limits ────────────────────────────────────────────────
        let mut limit_rows = Vec::new();
        for branch in &network.branches {
            let Some(Some(rate)) = network.limits.get(&branch.index).copied() else {
                continue;
            };
            // `b·θ_f − b·θ_t ∈ [−rate + b·α, rate + b·α]` — the shift moves to
            // the bounds for the same reason it moves to the balance
            // right-hand side.
            let offset = branch.b * branch.shift;
            let row = lp.add_row(
                &[(layout.theta + branch.from, branch.b), (layout.theta + branch.to, -branch.b)],
                -rate + offset,
                rate + offset,
            );
            limit_rows.push((row, branch.index));
        }

        // ── Epigraph rows for piecewise-linear costs ─────────────────────
        // One row per segment: `z ≥ m·p + k`, written `z − m·p ≥ k` so the
        // variables stay on the left. With `z` minimized, the binding row at
        // any `p` is the segment lying highest there — which for a convex
        // curve is the segment `p` actually sits on, so `z` equals the true
        // cost rather than merely bounding it.
        for (g, generator) in network.generators.iter().enumerate() {
            let Some(CostCurve::PiecewiseLinear { points }) = &generator.cost else {
                continue;
            };
            let epigraph = layout.cost_col[g].expect("assigned above");
            let column = layout.generator + g;

            if points.len() < 2 {
                // A single point fixes the cost but says nothing about how it
                // varies, so there is no segment to write. Treated as a
                // constant rather than as free, which is the reading that
                // cannot mislead.
                lp.col_cost[epigraph] = 0.0;
                lp.offset += points.first().map(|p| p.1).unwrap_or(0.0);
                continue;
            }

            for pair in points.windows(2) {
                let ((p0, c0), (p1, c1)) = (pair[0], pair[1]);
                if p1 == p0 {
                    // A vertical segment carries no slope; skipping it is
                    // right, since the neighbouring segments already bound `z`
                    // there.
                    continue;
                }
                let slope = (c1 - c0) / (p1 - p0);
                let intercept = c0 - slope * p0;
                lp.add_row(
                    &[(epigraph, 1.0), (column, -slope)],
                    intercept,
                    f64::INFINITY,
                );
            }
        }

        let layout = Layout { limit_rows, ..layout };
        lp.validate()?;
        Ok(Self { network, options, layout, problem: lp })
    }

    /// The assembled program, for handing to a [`Solver`](crate::opf::Solver).
    pub fn problem(&self) -> &LinearProgram {
        &self.problem
    }

    pub fn network(&self) -> &DcOpfNetwork {
        &self.network
    }

    pub fn options(&self) -> &DcOpfOptions {
        &self.options
    }

    /// Reads a solution back into physical units.
    pub fn interpret(&self, solution: &Solution) -> DcOpfResult {
        let base = self.network.base_mva;
        if !solution.is_optimal() {
            return DcOpfResult {
                status: solution.status,
                objective: f64::NAN,
                dispatch: Vec::new(),
                angles: Vec::new(),
                flows: Vec::new(),
                lmp: Vec::new(),
                shed: Vec::new(),
                binding: Vec::new(),
            };
        }

        let angles: Vec<f64> = (0..self.network.n_buses)
            .map(|i| solution.primal[self.layout.theta + i])
            .collect();
        let dispatch: Vec<f64> = (0..self.network.generators.len())
            .map(|g| solution.primal[self.layout.generator + g] * base)
            .collect();
        let shed: Vec<f64> = (0..self.layout.n_shed)
            .map(|d| solution.primal[self.layout.shed + d] * base)
            .collect();

        let n_branches =
            self.network.branches.iter().map(|b| b.index + 1).max().unwrap_or(0);
        let mut flows = vec![0.0; n_branches];
        for branch in &self.network.branches {
            let flow = branch.b * (angles[branch.from] - angles[branch.to] - branch.shift);
            flows[branch.index] = flow * base;
        }

        // The balance row's dual is ∂cost/∂load in $/h per per-unit; dividing
        // by the base turns it into $/MWh.
        let lmp: Vec<f64> = (0..self.network.n_buses)
            .map(|i| solution.row_dual[self.layout.balance_row + i] / base)
            .collect();

        let binding = self
            .layout
            .limit_rows
            .iter()
            .filter_map(|&(row, branch_index)| {
                let price = solution.row_dual[row];
                let rate = self.network.limits.get(&branch_index).copied().flatten()?;

                // Binding is decided on the **flow**, not on the dual being
                // exactly zero. That distinction is not pedantry: a simplex
                // solve sets inactive duals to a hard 0.0, but an interior-
                // point method only drives them toward it, leaving ~1e-12 on
                // every inactive row. Testing `price == 0.0` therefore reports
                // *every* limit as binding under one backend and the right
                // ones under the other — which is exactly how the two-solver
                // cross-check found this.
                //
                // The flow reaching its rating is also the definition a
                // dispatcher means by "binding", and it is a primal fact both
                // methods agree on to eight digits, so it is the more robust
                // criterion as well as the more honest one. A branch at its
                // limit whose dual is genuinely zero is weakly binding — real,
                // and reported, with the zero price saying relieving it saves
                // nothing.
                let flow = flows[branch_index].abs();
                let rate_mw = rate * base;
                if (flow - rate_mw).abs() > 1e-6 * rate_mw.max(1.0) {
                    return None;
                }
                Some(Binding {
                    branch: branch_index,
                    flow: flows[branch_index],
                    rate: rate_mw,
                    // A limit's dual has the opposite sense to a balance
                    // row's: relaxing it *reduces* cost, so the saving per MW
                    // of extra capacity is the negated dual.
                    price: -price.abs() / base,
                })
            })
            .collect();

        DcOpfResult {
            status: solution.status,
            objective: solution.objective,
            dispatch,
            angles,
            flows,
            lmp,
            shed,
            binding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two buses, one line, one generator at each — the smallest case with a
    /// real dispatch decision.
    fn two_bus(rate: Option<f64>) -> DcOpfNetwork {
        DcOpfNetwork {
            n_buses: 2,
            reference: 0,
            branches: vec![DcBranch { index: 0, from: 0, to: 1, b: 10.0, shift: 0.0 }],
            limits: HashMap::from([(0, rate)]),
            generators: vec![
                DcGenerator {
                    index: 0,
                    bus: 0,
                    p_min: 0.0,
                    p_max: 2.0,
                    // $10/MWh, linear, already per-unit on a 100 MVA base.
                    cost: Some(CostCurve::Polynomial { coefficients: vec![0.0, 1000.0] }),
                },
                DcGenerator {
                    index: 1,
                    bus: 1,
                    p_min: 0.0,
                    p_max: 2.0,
                    // $30/MWh.
                    cost: Some(CostCurve::Polynomial { coefficients: vec![0.0, 3000.0] }),
                },
            ],
            loads: vec![DcLoad { bus: 1, p: 1.0 }],
            base_mva: 100.0,
        }
    }

    #[test]
    fn the_program_has_the_expected_shape() {
        let opf = DcOpf::build(two_bus(Some(2.0)), DcOpfOptions::default()).unwrap();
        let lp = opf.problem();
        // 2 angles + 2 generators + 1 sheddable load.
        assert_eq!(lp.n_vars, 5);
        // 2 balance rows + 1 branch limit.
        assert_eq!(lp.n_rows, 3);
        // The reference angle is pinned.
        assert_eq!(lp.col_lower[0], 0.0);
        assert_eq!(lp.col_upper[0], 0.0);
    }

    /// An unlimited branch contributes no row at all — the cheapest check that
    /// MATPOWER's "rate 0 means unlimited" survives into the formulation.
    #[test]
    fn an_unlimited_branch_adds_no_constraint() {
        let opf = DcOpf::build(two_bus(None), DcOpfOptions::default()).unwrap();
        assert_eq!(opf.problem().n_rows, 2, "only the two balance rows");
    }

    #[test]
    fn shedding_can_be_switched_off() {
        let options = DcOpfOptions { allow_shedding: false, ..Default::default() };
        let opf = DcOpf::build(two_bus(Some(2.0)), options).unwrap();
        assert_eq!(opf.problem().n_vars, 4, "no shedding column");
    }

    /// A quadratic cost must reach the Hessian as **twice** its coefficient,
    /// because the objective is `½xᵀQx`.
    #[test]
    fn a_quadratic_cost_is_doubled_into_the_hessian() {
        let mut network = two_bus(None);
        network.generators[0].cost =
            Some(CostCurve::Polynomial { coefficients: vec![5.0, 1000.0, 7.0] });
        let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();

        let hessian = opf.problem().hessian.as_ref().expect("a quadratic term");
        assert_eq!(hessian.len(), 1);
        let (row, col, value) = hessian[0];
        assert_eq!((row, col), (2, 2), "generator 0 is the third column");
        assert_eq!(value, 14.0, "2 * 7.0");
        // The constant term lands in the offset, where it changes the reported
        // cost but not the optimum.
        assert_eq!(opf.problem().offset, 5.0);
    }

    #[test]
    fn a_non_convex_cost_is_refused_rather_than_solved() {
        let mut network = two_bus(None);
        network.generators[0].cost =
            Some(CostCurve::Polynomial { coefficients: vec![0.0, 1000.0, -1.0] });
        assert!(DcOpf::build(network, DcOpfOptions::default()).is_err());
    }

    /// A phase shift is a constant, so it belongs on the right-hand side of
    /// the balance rows and in the limit bounds — not among the variables.
    #[test]
    fn a_phase_shift_moves_to_the_right_hand_side() {
        let mut network = two_bus(Some(2.0));
        network.branches[0].shift = 0.1;
        let opf = DcOpf::build(network, DcOpfOptions::default()).unwrap();
        let lp = opf.problem();

        // Bus 0's balance row: rhs = load(0) - shift_injection(0) = 0 - 10*0.1.
        assert!((lp.row_lower[0] - (-1.0)).abs() < 1e-12, "{:?}", lp.row_lower);
        // Bus 1's: rhs = 1.0 - (-10*0.1).
        assert!((lp.row_lower[1] - 2.0).abs() < 1e-12, "{:?}", lp.row_lower);
        // And the limit row's bounds pick up `b·α` on both sides.
        assert!((lp.row_lower[2] - (-2.0 + 1.0)).abs() < 1e-12);
        assert!((lp.row_upper[2] - (2.0 + 1.0)).abs() < 1e-12);
        assert_eq!(lp.n_vars, 5, "the shift adds no variable");
    }
}
