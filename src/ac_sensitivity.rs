//! AC sensitivity analysis: how a converged operating point responds to a
//! small change.
//!
//! Answers the questions that sit between a single power flow and a full
//! contingency study. *If this generator ramps 10 MW, which line picks it up?
//! Which bus voltage moves, and by how much? Which injection would relieve
//! this overload fastest?* Each is a partial derivative of the solved state
//! with respect to an input, evaluated at the operating point.
//!
//! # The formulation
//!
//! A converged power flow satisfies
//!
//! \\[ g(x, p) = s_{calc}(x) - s_{spec}(p) = 0 \\]
//!
//! with state \\(x = [\theta_{non\text{-}slack};\ |V|_{PQ}]\\) — exactly the
//! unknown vector [`solver::newton_raphson`](crate::solver) iterates on.
//! Differentiating and rearranging:
//!
//! \\[ \frac{dx}{dp} = -J^{-1} \frac{\partial g}{\partial p}
//!                   = J^{-1} \frac{\partial s_{spec}}{\partial p} \\]
//!
//! where \\(J = \partial g / \partial x\\) is **the same Jacobian Newton
//! already builds** — `jacobian::JacobianPattern`, filled at the converged
//! state. And because a specified injection enters exactly one equation, its
//! \\(\partial s_{spec} / \partial p\\) is a unit vector. So the response of
//! the entire network to one injection is a single triangular solve with a
//! unit right-hand side. That is the whole method.
//!
//! Any quantity of interest \\(f(x)\\) then follows by the chain rule,
//! \\(df/dp = (\partial f/\partial x)^{T} \, dx/dp\\), and for a branch flow
//! that row is precisely what
//! [`branch_flow::terminal_flow_derivs`](crate::branch_flow) already computes
//! for the state estimator.
//!
//! # Two directions, and when each is the cheap one
//!
//! The expression above can be bracketed either way, and the two groupings
//! have genuinely different costs:
//!
//! - **Forward** ([`state_response`](AcSensitivity::state_response)) — solve
//!   \\(J\,z = e_i\\) once per *variable*, then every function is a dot
//!   product. Use when few things move and many are watched: *one generator
//!   ramps — what happens to all 5,000 branches?*
//! - **Adjoint** ([`function_row`](AcSensitivity::function_row)) — solve
//!   \\(J^{T} w = \partial f/\partial x\\) once per *function*, then every
//!   variable is a lookup. Use when one thing is watched and many could move:
//!   *this line is overloaded — which injection relieves it?*
//!
//! Both run against one factorization, computed once in [`new`](
//! AcSensitivity::new); [`sparse::RealFactorization::solve_transpose`](
//! crate::sparse::RealFactorization::solve_transpose) reuses it for the
//! transposed direction rather than factorizing again. This mirrors the
//! `ptdf_column`/`ptdf_row` pair in
//! [`linear::sensitivity`](crate::linear::sensitivity), which is the same
//! duality on the DC side.
//!
//! # Relationship to the DC sensitivities
//!
//! [`DcSensitivity`](crate::linear::sensitivity::DcSensitivity) is *exact* for
//! the model it describes, because the DC model is linear — a PTDF is not an
//! approximation of DC, it **is** DC. These are different: the AC problem is
//! nonlinear, so these are first-order derivatives, accurate for small
//! perturbations and degrading as the step grows. What they buy in exchange is
//! everything DC discards — voltage magnitudes, reactive flows, losses.
//!
//! Use DC to screen thousands of cases, AC to answer precisely about the few
//! that matter.
//!
//! # Transformer taps and phase shifters
//!
//! These are variables too, but of a different shape: an injection enters
//! \\(\partial g/\partial p\\) as a unit vector, whereas a tap changes the
//! branch's admittance and therefore the *calculated* injections at both its
//! ends. Two consequences follow, and the second is easy to miss.
//!
//! First the sign. For an injection, \\(p\\) appears in \\(s_{spec}\\) and
//! \\(\partial g/\partial p = -\partial s_{spec}/\partial p\\); for a tap it
//! appears in \\(s_{calc}\\) and \\(\partial g/\partial p = +\partial
//! s_{calc}/\partial p\\). The response picks up the opposite sign.
//!
//! Second, and the part a naive chain rule drops: a tapped branch's *own* flow
//! depends on the tap **directly**, not only through the state. The full
//! derivative is
//!
//! \\[ \frac{df}{dp} = \left(\frac{\partial f}{\partial x}\right)^{T}
//!                     \frac{dx}{dp} + \left.\frac{\partial f}{\partial
//!                     p}\right|_{x} \\]
//!
//! and that second term is zero for every branch except the tapped one, where
//! it is large. Omitting it gives an answer that looks plausible everywhere
//! and is badly wrong exactly where it is most likely to be read.
//!
//! The admittance derivatives themselves are unusually tidy. With
//! \\(tap = k\,e^{j\alpha}\\), and given that `yff` always carries \\(1/k^2\\),
//! `yft`/`ytf` always carry \\(1/k\\), and `ytt` never depends on the tap at
//! all:
//!
//! \\[ \frac{\partial y_{ff}}{\partial k} = \frac{-2\,y_{ff}}{k}, \quad
//!     \frac{\partial y_{ft}}{\partial k} = \frac{-y_{ft}}{k}, \quad
//!     \frac{\partial y_{tf}}{\partial k} = \frac{-y_{tf}}{k}, \quad
//!     \frac{\partial y_{tt}}{\partial k} = 0 \\]
//!
//! \\[ \frac{\partial y_{ft}}{\partial \alpha} = j\,y_{ft}, \quad
//!     \frac{\partial y_{tf}}{\partial \alpha} = -j\,y_{tf}, \quad
//!     \frac{\partial y_{ff}}{\partial \alpha} =
//!     \frac{\partial y_{tt}}{\partial \alpha} = 0 \\]
//!
//! Each is a scaling of the entry itself, so no admittance has to be rebuilt
//! from nameplate data — and the relations hold for the half-open terminal
//! states too, where the affected entries are simply zero.
//!
//! # What is *not* here
//!
//! No contingency (outage) sensitivities. An outage is a finite change in
//! topology, not an infinitesimal change in an input, so it is not a
//! derivative at all — the DC side handles it with a Woodbury update
//! (`DcSensitivity::outage_flows`) that has no equally cheap AC analogue.
//! `batch::BatchSolver::solve_contingencies` is the AC answer, and it re-solves.

use crate::branch_flow::{branch_params, terminal_flow_derivs, BranchParams, Terminal};
use crate::jacobian::JacobianPattern;
use crate::network::{power_injections, YBusSparse};
use crate::sparse::RealFactorization;
use crate::types::{Bus, BusType, Line, Transformer};

use num_complex::Complex;

const ZERO: Complex<f64> = Complex::new(0.0, 0.0);

/// Marks a bus whose quantity is not a free unknown — a slack bus's angle, or
/// a PV or slack bus's voltage magnitude. The same sentinel
/// `linear::sensitivity` uses, for the same reason.
const NOT_UNKNOWN: usize = usize::MAX;

/// An input the operating point can be differentiated with respect to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variable {
    /// Active power injected at a bus, per unit.
    ///
    /// Undefined at a slack bus, whose injection is not an input but the
    /// solve's own result — the response there is identically zero, exactly as
    /// a DC PTDF column is zero at its reference.
    ActiveInjection(usize),
    /// Reactive power injected at a bus, per unit.
    ///
    /// Only meaningful at a `PQ` bus. At a `PV` or slack bus the voltage
    /// magnitude is held, so reactive power is an output — the response is
    /// identically zero.
    ReactiveInjection(usize),
    /// A transformer's off-nominal voltage-magnitude ratio `k`, by flat branch
    /// index (lines first, then transformers).
    ///
    /// Zero for a line, which has no tap to move.
    TransformerRatio(usize),
    /// A phase-shifting transformer's angle `α`, radians, by flat branch index.
    ///
    /// This is the argument of the complex tap — the quantity a phase shifter
    /// actually controls, and the one that steers active power.
    PhaseShift(usize),
}

/// Which of the two tap quantities a variable names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TapKind {
    Ratio,
    Phase,
}

/// A quantity whose sensitivity to every variable is wanted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Function {
    /// Active power flowing into `branch` at `terminal`.
    BranchActivePower { branch: usize, terminal: Terminal },
    /// Reactive power flowing into `branch` at `terminal`.
    BranchReactivePower { branch: usize, terminal: Terminal },
    /// A bus's voltage magnitude, per unit.
    VoltageMagnitude(usize),
    /// A bus's voltage angle, radians.
    VoltageAngle(usize),
}

/// How the whole state moves in response to one variable.
///
/// Both vectors are indexed by bus and are always full length. A bus whose
/// quantity is not free carries zero — a slack bus's angle does not move
/// because it is the reference, and a `PV` bus's magnitude does not move
/// because its controller holds it. Those zeros are answers, not gaps.
#[derive(Clone, Debug)]
pub struct StateResponse {
    /// dθ/dp per bus, radians per per-unit.
    pub d_theta: Vec<f64>,
    /// d|V|/dp per bus, per-unit per per-unit.
    pub d_vmag: Vec<f64>,
}

/// How one function responds to injection at every bus.
#[derive(Clone, Debug)]
pub struct FunctionRow {
    /// df/dP per bus. Zero at a slack bus.
    pub d_active: Vec<f64>,
    /// df/dQ per bus. Zero anywhere the voltage magnitude is held.
    pub d_reactive: Vec<f64>,
    /// df/dk per branch — the response to each transformer's ratio. Zero for a
    /// line.
    ///
    /// One adjoint solve yields this alongside the injection sensitivities, so
    /// "which tap relieves this overload?" costs nothing beyond "which
    /// injection does".
    pub d_ratio: Vec<f64>,
    /// df/dα per branch — the response to each phase shifter's angle, in
    /// radians. Zero for a line.
    pub d_phase: Vec<f64>,
}

/// A converged operating point, factorized once and ready to be differentiated.
///
/// Holds one factorization of the power-flow Jacobian at that point. Every
/// accessor is a triangular solve against it, never a refactorization — the
/// same contract [`DcSensitivity`](crate::linear::sensitivity::DcSensitivity)
/// offers, and the reason both own a
/// [`RealFactorization`](crate::sparse::RealFactorization) rather than
/// borrowing the Newton path's solver.
pub struct AcSensitivity {
    n_buses: usize,
    /// The branch list this was built against, in the flat order
    /// `branch_flow::branch_params` produces: lines first, then transformers.
    branches: Vec<BranchParams>,
    /// Per flat branch, its complex tap `k·e^{jα}` — `None` for a line, which
    /// has no tap to differentiate with respect to.
    taps: Vec<Option<Complex<f64>>>,
    /// Complex bus voltages at the linearization point.
    v: Vec<Complex<f64>>,
    /// Bus → its angle unknown's row, or [`NOT_UNKNOWN`] for a slack bus.
    theta_pos: Vec<usize>,
    /// Bus → its magnitude unknown's row, or [`NOT_UNKNOWN`] unless `PQ`.
    vmag_pos: Vec<usize>,
    n_unknowns: usize,
    factorization: RealFactorization,
}

impl AcSensitivity {
    /// Factorizes the Jacobian at the operating point `buses` describes.
    ///
    /// `buses` must be a **converged** solution — this linearizes about
    /// whatever state it is handed and cannot tell a solved network from an
    /// unsolved one, so the derivatives of a non-converged state are
    /// meaningless rather than wrong-looking. `ybus`, `lines` and
    /// `transformers` must be the same network that solution came from.
    ///
    /// Takes the branch lists rather than a finished `Vec<BranchParams>` for
    /// the same reason [`crate::run_power_flow`] does: the tap sensitivities
    /// need each transformer's own `tap`, which `BranchParams` has already
    /// folded into its four admittance entries. Deriving both here also means
    /// a caller cannot pass a branch list that disagrees with the Y-bus.
    ///
    /// Returns `None` if the Jacobian is singular there.
    ///
    /// Unlike the DC side there is no per-island bookkeeping, and none is
    /// needed: every island carries its own slack, so the assembled Jacobian
    /// is block-diagonal across islands and non-singular as a whole. A
    /// sourceless island contributes no rows at all, because
    /// `network::mark_unreferenced_islands` has already pinned its buses to
    /// `Slack` — so it is excluded rather than making the system singular.
    pub fn new(
        buses: &[Bus],
        ybus: &YBusSparse,
        lines: &[Line],
        transformers: &[Transformer],
    ) -> Option<Self> {
        let n = buses.len();
        let branches = branch_params(lines, transformers);
        let taps: Vec<Option<Complex<f64>>> = std::iter::repeat_n(None, lines.len())
            .chain(transformers.iter().map(|t| Some(t.tap)))
            .collect();

        // The same layout `solver::newton_raphson` uses: every non-slack bus
        // contributes an angle unknown, every PQ bus a magnitude unknown, in
        // bus order, angles first. The Jacobian's rows follow the same
        // ordering (P equations then Q equations), so a bus's P-equation row
        // and its angle-unknown column share an index — which is what lets a
        // variable's right-hand side be a plain unit vector.
        let mut theta_pos = vec![NOT_UNKNOWN; n];
        let mut vmag_pos = vec![NOT_UNKNOWN; n];
        let mut next = 0;
        for b in buses {
            if b.bus_type != BusType::Slack {
                theta_pos[b.idx] = next;
                next += 1;
            }
        }
        for b in buses {
            if b.bus_type == BusType::PQ {
                vmag_pos[b.idx] = next;
                next += 1;
            }
        }
        let n_unknowns = next;

        let (p_calc, q_calc) = power_injections(buses, ybus);
        let pattern = JacobianPattern::analyze(buses, ybus);
        let mut values = Vec::with_capacity(pattern.len());
        pattern.fill(buses, &p_calc, &q_calc, &mut values);
        let factorization =
            RealFactorization::new(n_unknowns, &pattern.to_triplets(&values))?;

        Some(Self {
            n_buses: n,
            branches,
            taps,
            v: crate::branch_flow::bus_voltages(buses),
            theta_pos,
            vmag_pos,
            n_unknowns,
            factorization,
        })
    }

    pub fn n_buses(&self) -> usize {
        self.n_buses
    }

    pub fn n_branches(&self) -> usize {
        self.branches.len()
    }

    /// The row of `J` an injection variable perturbs, or `None` when it is not
    /// a free input at that bus.
    fn variable_row(&self, variable: Variable) -> Option<usize> {
        let row = match variable {
            Variable::ActiveInjection(bus) => *self.theta_pos.get(bus)?,
            Variable::ReactiveInjection(bus) => *self.vmag_pos.get(bus)?,
            _ => return None,
        };
        (row != NOT_UNKNOWN).then_some(row)
    }

    /// The derivative of a branch's four admittance entries with respect to one
    /// of its tap quantities — see the module docs for the derivation.
    ///
    /// Every entry is a scaling of the entry itself, which is what makes this
    /// exact without reconstructing the branch from nameplate data, and what
    /// makes it hold unchanged for the half-open terminal states (where the
    /// affected entries are already zero).
    fn admittance_derivative(
        &self,
        branch: usize,
        kind: TapKind,
    ) -> Option<[Complex<f64>; 4]> {
        let tap = (*self.taps.get(branch)?)?;
        let y = &self.branches[branch].y;
        Some(match kind {
            TapKind::Ratio => {
                let k = tap.norm();
                [-2.0 * y[0] / k, -y[1] / k, -y[2] / k, ZERO]
            }
            TapKind::Phase => {
                let j = Complex::new(0.0, 1.0);
                [ZERO, j * y[1], -j * y[2], ZERO]
            }
        })
    }

    /// `∂s_calc/∂p` at the branch's two ends, holding the state fixed.
    ///
    /// These are the *only* injections a tap touches: a branch's admittance
    /// appears in the nodal balance of its own two buses and nowhere else.
    fn tap_injection_derivative(
        &self,
        branch: usize,
        kind: TapKind,
    ) -> Option<(Complex<f64>, Complex<f64>)> {
        let dy = self.admittance_derivative(branch, kind)?;
        let (f, t) = (self.branches[branch].from, self.branches[branch].to);
        let (vf, vt) = (self.v[f], self.v[t]);
        Some((
            vf * (dy[0] * vf + dy[1] * vt).conj(),
            vt * (dy[2] * vf + dy[3] * vt).conj(),
        ))
    }

    /// `∂g/∂p` in the unknown layout, for a tap variable.
    fn tap_residual_derivative(&self, branch: usize, kind: TapKind) -> Option<Vec<f64>> {
        let (ds_from, ds_to) = self.tap_injection_derivative(branch, kind)?;
        let (f, t) = (self.branches[branch].from, self.branches[branch].to);

        let mut dg = vec![0.0; self.n_unknowns];
        let mut place = |pos: usize, value: f64| {
            if pos != NOT_UNKNOWN {
                dg[pos] += value;
            }
        };
        // A bus's P equation sits at its angle unknown's index and its Q
        // equation at its magnitude unknown's — the same coincidence that lets
        // an injection's right-hand side be a unit vector.
        place(self.theta_pos[f], ds_from.re);
        place(self.vmag_pos[f], ds_from.im);
        place(self.theta_pos[t], ds_to.re);
        place(self.vmag_pos[t], ds_to.im);
        Some(dg)
    }

    /// Splits a variable into its tap branch and kind, if it is one.
    fn as_tap(variable: Variable) -> Option<(usize, TapKind)> {
        match variable {
            Variable::TransformerRatio(b) => Some((b, TapKind::Ratio)),
            Variable::PhaseShift(b) => Some((b, TapKind::Phase)),
            _ => None,
        }
    }

    /// How the whole state responds to one variable — the **forward**
    /// direction, one solve.
    ///
    /// Returns an all-zero response for a variable that is not a free input
    /// (active injection at a slack bus, reactive injection where the voltage
    /// is held). That is the correct derivative, not a failure: perturbing a
    /// quantity the solve itself determines moves nothing.
    ///
    /// `None` only for an out-of-range bus, or a singular solve.
    pub fn state_response(&self, variable: Variable) -> Option<StateResponse> {
        let zero_response = || StateResponse {
            d_theta: vec![0.0; self.n_buses],
            d_vmag: vec![0.0; self.n_buses],
        };

        if let Some((branch, kind)) = Self::as_tap(variable) {
            if branch >= self.branches.len() {
                return None;
            }
            // A line has no tap, so moving one moves nothing.
            let Some(dg) = self.tap_residual_derivative(branch, kind) else {
                return Some(zero_response());
            };
            // `dx/dp = -J⁻¹ ∂g/∂p`, and here `g` depends on `p` through
            // `s_calc` — hence the negation, which the injection case does not
            // carry because there `p` enters through `s_spec` instead.
            let mut dx = self.factorization.solve(&dg)?;
            for value in &mut dx {
                *value = -*value;
            }
            return Some(self.scatter(&dx));
        }

        let bus = match variable {
            Variable::ActiveInjection(b) | Variable::ReactiveInjection(b) => b,
            _ => unreachable!("tap variables returned above"),
        };
        if bus >= self.n_buses {
            return None;
        }

        let Some(row) = self.variable_row(variable) else {
            return Some(zero_response());
        };

        let mut rhs = vec![0.0; self.n_unknowns];
        rhs[row] = 1.0;
        let dx = self.factorization.solve(&rhs)?;
        Some(self.scatter(&dx))
    }

    /// Spreads a solved unknown vector back over the buses it came from.
    fn scatter(&self, dx: &[f64]) -> StateResponse {
        let mut d_theta = vec![0.0; self.n_buses];
        let mut d_vmag = vec![0.0; self.n_buses];
        for i in 0..self.n_buses {
            if self.theta_pos[i] != NOT_UNKNOWN {
                d_theta[i] = dx[self.theta_pos[i]];
            }
            if self.vmag_pos[i] != NOT_UNKNOWN {
                d_vmag[i] = dx[self.vmag_pos[i]];
            }
        }
        StateResponse { d_theta, d_vmag }
    }

    /// `(dP, dQ)` at every branch's `terminal`, for one variable.
    ///
    /// The AC counterpart of a PTDF column, and the reason the forward
    /// direction exists: one solve, then a dot product per branch.
    pub fn branch_response(
        &self,
        variable: Variable,
        terminal: Terminal,
    ) -> Option<Vec<(f64, f64)>> {
        let state = self.state_response(variable)?;

        // The direct term. A tapped branch's own flow depends on its tap
        // without the state moving at all, so that branch — and only that
        // branch — picks up an extra `∂f/∂p|_x`. Leaving it out yields an
        // answer that is right everywhere except on the branch being adjusted,
        // which is the one most likely to be read.
        let direct = Self::as_tap(variable)
            .and_then(|(branch, kind)| Some((branch, self.tap_flow_derivative(branch, kind, terminal)?)));

        Some(
            self.branches
                .iter()
                .enumerate()
                .map(|(i, branch)| {
                    let d = terminal_flow_derivs(branch, terminal, &self.v);
                    let (near, far) = branch.buses(terminal);
                    let mut dp = d.dp_dtheta_near * state.d_theta[near]
                        + d.dp_dtheta_far * state.d_theta[far]
                        + d.dp_dv_near * state.d_vmag[near]
                        + d.dp_dv_far * state.d_vmag[far];
                    let mut dq = d.dq_dtheta_near * state.d_theta[near]
                        + d.dq_dtheta_far * state.d_theta[far]
                        + d.dq_dv_near * state.d_vmag[near]
                        + d.dq_dv_far * state.d_vmag[far];
                    if let Some((tapped, s)) = direct {
                        if tapped == i {
                            dp += s.re;
                            dq += s.im;
                        }
                    }
                    (dp, dq)
                })
                .collect(),
        )
    }

    /// `∂f/∂p|_x` for a tapped branch's own terminal flow — the direct term.
    ///
    /// Same shape as [`tap_injection_derivative`](Self::tap_injection_derivative),
    /// but read at the requested terminal rather than at both ends.
    fn tap_flow_derivative(
        &self,
        branch: usize,
        kind: TapKind,
        terminal: Terminal,
    ) -> Option<Complex<f64>> {
        let (ds_from, ds_to) = self.tap_injection_derivative(branch, kind)?;
        Some(match terminal {
            Terminal::From => ds_from,
            Terminal::To => ds_to,
        })
    }

    /// `∂f/∂x` in the unknown layout — the gradient of a function with respect
    /// to the state.
    fn function_gradient(&self, function: Function) -> Option<Vec<f64>> {
        let mut grad = vec![0.0; self.n_unknowns];
        let mut place = |pos: usize, value: f64| {
            if pos != NOT_UNKNOWN {
                grad[pos] += value;
            }
        };

        match function {
            Function::BranchActivePower { branch, terminal }
            | Function::BranchReactivePower { branch, terminal } => {
                let params = self.branches.get(branch)?;
                let d = terminal_flow_derivs(params, terminal, &self.v);
                let (near, far) = params.buses(terminal);
                let active = matches!(function, Function::BranchActivePower { .. });
                let (dth_near, dth_far, dv_near, dv_far) = if active {
                    (d.dp_dtheta_near, d.dp_dtheta_far, d.dp_dv_near, d.dp_dv_far)
                } else {
                    (d.dq_dtheta_near, d.dq_dtheta_far, d.dq_dv_near, d.dq_dv_far)
                };
                place(self.theta_pos[near], dth_near);
                place(self.theta_pos[far], dth_far);
                place(self.vmag_pos[near], dv_near);
                place(self.vmag_pos[far], dv_far);
            }
            Function::VoltageMagnitude(bus) => {
                place(*self.vmag_pos.get(bus)?, 1.0);
            }
            Function::VoltageAngle(bus) => {
                place(*self.theta_pos.get(bus)?, 1.0);
            }
        }
        Some(grad)
    }

    /// How one function responds to injection at *every* bus — the **adjoint**
    /// direction, one solve.
    ///
    /// The AC counterpart of a PTDF row. Where
    /// [`branch_response`](Self::branch_response) answers "one variable moves,
    /// what happens everywhere", this answers "one thing is watched, what
    /// moves it" — which is the question an overload actually poses.
    pub fn function_row(&self, function: Function) -> Option<FunctionRow> {
        let grad = self.function_gradient(function)?;
        // `w = J⁻ᵀ ∂f/∂x`, so `wᵀ e_i = (∂f/∂x)ᵀ J⁻¹ e_i` — the same scalar the
        // forward direction computes, obtained for every `i` at once.
        let w = self.factorization.solve_transpose(&grad)?;

        let mut d_active = vec![0.0; self.n_buses];
        let mut d_reactive = vec![0.0; self.n_buses];
        for i in 0..self.n_buses {
            if self.theta_pos[i] != NOT_UNKNOWN {
                d_active[i] = w[self.theta_pos[i]];
            }
            if self.vmag_pos[i] != NOT_UNKNOWN {
                d_reactive[i] = w[self.vmag_pos[i]];
            }
        }

        // Taps come out of the *same* adjoint vector, at no extra solve:
        // `df/dp = -wᵀ ∂g/∂p + ∂f/∂p|_x`. Only the sign and the direct term
        // differ from the injection case, and the direct term is nonzero only
        // when the function is the tapped branch's own flow.
        let mut d_ratio = vec![0.0; self.branches.len()];
        let mut d_phase = vec![0.0; self.branches.len()];
        for branch in 0..self.branches.len() {
            for (kind, out) in
                [(TapKind::Ratio, &mut d_ratio), (TapKind::Phase, &mut d_phase)]
            {
                let Some(dg) = self.tap_residual_derivative(branch, kind) else {
                    continue;
                };
                let mut value: f64 = -dg.iter().zip(&w).map(|(g, wi)| g * wi).sum::<f64>();
                if let Function::BranchActivePower { branch: b, terminal }
                | Function::BranchReactivePower { branch: b, terminal } = function
                {
                    if b == branch {
                        let s = self
                            .tap_flow_derivative(branch, kind, terminal)
                            .expect("the tap exists, it was read just above");
                        value += if matches!(function, Function::BranchActivePower { .. }) {
                            s.re
                        } else {
                            s.im
                        };
                    }
                }
                out[branch] = value;
            }
        }

        Some(FunctionRow { d_active, d_reactive, d_ratio, d_phase })
    }
}
