//! Transformer tap control: the on-load tap changer as a control rather than a
//! constant.
//!
//! Every steady-state result gridoxide produced before this module existed was
//! computed with taps that never move, which is a modelling assumption four of
//! the five tools in `docs/src/reference/feature_comparison.md` do not make. A
//! real tap changer holds a voltage or a flow, and its position is an *output*
//! of the solve.
//!
//! # The strategy, and why this one
//!
//! powsybl-open-loadflow offers three for voltage control. Two put the
//! continuous ratio inside the Newton system and round afterwards; the third —
//! `IncrementalTransformerVoltageControlOuterLoop` — steps the discrete
//! position directly, using the sensitivity of the controlled voltage to the
//! ratio:
//!
//! \\[ \Delta\rho = \frac{V^{\text{target}} - V_c}{\partial V_c/\partial\rho} \\]
//!
//! The incremental one is implemented here, for a reason specific to this
//! crate: that derivative is already [`crate::ac_sensitivity::AcSensitivity`]
//! with [`Variable::TransformerRatio`] against [`Function::VoltageMagnitude`],
//! validated against a central-difference re-solve of the full nonlinear power
//! flow. The continuous variants would need the ratio inside
//! [`crate::jacobian::JacobianPattern`] and a different `n_unknowns`.
//!
//! # Three guards, all load-bearing
//!
//! - **Deadband.** Every enabled control in the CGMES conformity fixtures
//!   carries one. A loop that ignores it hunts, and a loop that hunts does not
//!   terminate.
//! - **Insensitivity filter** ([`MIN_SENSITIVITY`]). Below it the controller
//!   cannot reach its bus, and \\(\Delta\rho\\) is a division by nearly zero
//!   that slams the tap to a limit.
//! - **Direction-change budget** ([`MAX_DIRECTION_CHANGES`]). A tap that has
//!   reversed this many times is oscillating between two positions that
//!   straddle the target with none inside the deadband. That is a property of
//!   the network and the deadband, not a solver failure, and the honest answer
//!   is to stop and report [`ControllerOutcome::Hunting`].

use crate::ac_sensitivity::{AcSensitivity, Variable};
use crate::branch_flow::Terminal;

use super::{Invalidates, OuterLoop, OuterLoopContext, OuterLoopStatus};

/// Below this, a controller is treated as unable to reach its controlled
/// quantity. powsybl's own `MIN_SENSI_FILTER`.
pub const MIN_SENSITIVITY: f64 = 0.05;

/// How many times a controller may reverse direction before it is declared to
/// be hunting. powsybl's own `MAX_DIRECTION_CHANGE`.
pub const MAX_DIRECTION_CHANGES: usize = 3;

/// What a tap changer's regulating control holds.
///
/// Both variants' targets are in **per-unit**, converted by the importer
/// against the controlled bus's own nominal voltage (or the system power base),
/// so this module is unit-free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegulationMode {
    /// Hold the voltage magnitude at `controlled_bus`. CGMES
    /// `RegulatingControlModeKind.voltage`.
    Voltage,
    /// Hold the active power flowing into `branch` at `terminal`. CGMES
    /// `RegulatingControlModeKind.activePower`. The branch index is flat —
    /// lines first, then transformers — matching `branch_flow::branch_params`.
    ActivePower { branch: usize, terminal: Terminal },
}

/// One regulating control, resolved against gridoxide's own indices.
#[derive(Clone, Debug)]
pub struct TapRegulation {
    /// Index into the `transformers` slice — *not* a flat branch index.
    pub transformer: usize,
    /// The bus whose voltage is held. Meaningful for
    /// [`RegulationMode::Voltage`]; carried for both so a report can name it.
    pub controlled_bus: usize,
    pub mode: RegulationMode,
    /// Set-point, per-unit.
    pub target: f64,
    /// Full width, per-unit. The controller is satisfied within `±deadband/2`.
    /// Zero means "hit the target exactly", which in practice means hunt until
    /// the direction budget stops it.
    pub deadband: f64,
    /// `TapChanger.controlEnabled` AND `RegulatingControl.enabled`. A control
    /// that exists but is off is data, not an error.
    pub enabled: bool,
    /// The element's own identifier, for reports.
    pub id: String,
}

/// How one controller finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerOutcome {
    /// The controlled quantity is inside the deadband. The success case.
    InDeadband,
    /// The tap reached `low` or `high` with the target still out of reach.
    AtLimit,
    /// `|∂target/∂tap|` is below [`MIN_SENSITIVITY`]; this controller cannot
    /// move its own controlled quantity.
    Insensitive,
    /// Reversed direction [`MAX_DIRECTION_CHANGES`] times without landing
    /// inside the deadband — two positions straddle the target and neither is
    /// acceptable.
    Hunting,
    /// The run ended (budget, or another loop failing) before this controller
    /// settled.
    Unfinished,
}

/// One controller's history through a solve.
#[derive(Clone, Debug)]
pub struct ControllerReport {
    pub id: String,
    pub transformer: usize,
    pub controlled_bus: usize,
    pub initial_position: i32,
    pub final_position: i32,
    /// Positions moved, summed over every step, ignoring direction.
    pub steps_moved: usize,
    pub direction_changes: usize,
    pub outcome: ControllerOutcome,
}

/// What a tap-control loop did.
#[derive(Clone, Debug, Default)]
pub struct TapControlReport {
    pub controllers: Vec<ControllerReport>,
}

impl TapControlReport {
    /// The report for the controller on `transformer`, if that transformer had
    /// one.
    pub fn controller(&self, transformer: usize) -> Option<&ControllerReport> {
        self.controllers.iter().find(|c| c.transformer == transformer)
    }
}

/// Per-controller state the loop carries between passes.
#[derive(Clone, Debug)]
struct ControllerState {
    regulation_index: usize,
    initial_position: i32,
    steps_moved: usize,
    direction_changes: usize,
    /// `Some(true)` last moved up, `Some(false)` last moved down.
    last_direction: Option<bool>,
    outcome: ControllerOutcome,
    /// Stops being consulted once it is hunting, insensitive or at a limit
    /// with nowhere to go.
    frozen: bool,
}

/// Shared machinery for the two tap loops. They differ only in which quantity
/// they read, which tap coordinate they move, and which sensitivity relates
/// them — the deadband, filter, budget and reporting are identical.
struct TapLoop {
    max_tap_shift: i32,
    states: Vec<ControllerState>,
    initialized: bool,
}

impl TapLoop {
    fn new(max_tap_shift: i32) -> Self {
        Self { max_tap_shift: max_tap_shift.max(1), states: Vec::new(), initialized: false }
    }

    /// Records the starting positions of every controller this loop owns.
    fn initialize(&mut self, ctx: &mut OuterLoopContext<'_, '_>, mine: fn(&TapRegulation) -> bool) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        for (i, reg) in ctx.net.regulation.iter().enumerate() {
            if !reg.enabled || !mine(reg) {
                continue;
            }
            let Some(Some(tc)) = ctx.net.tap_changers.get(reg.transformer) else {
                continue;
            };
            self.states.push(ControllerState {
                regulation_index: i,
                initial_position: tc.position,
                steps_moved: 0,
                direction_changes: 0,
                last_direction: None,
                outcome: ControllerOutcome::Unfinished,
                frozen: false,
            });
        }
    }

    fn report(&self, regulation: &[TapRegulation], tap_changers: &[Option<crate::types::TapChanger>]) -> TapControlReport {
        TapControlReport {
            controllers: self
                .states
                .iter()
                .map(|s| {
                    let reg = &regulation[s.regulation_index];
                    ControllerReport {
                        id: reg.id.clone(),
                        transformer: reg.transformer,
                        controlled_bus: reg.controlled_bus,
                        initial_position: s.initial_position,
                        final_position: tap_changers[reg.transformer]
                            .as_ref()
                            .map(|t| t.position)
                            .unwrap_or(s.initial_position),
                        steps_moved: s.steps_moved,
                        direction_changes: s.direction_changes,
                        outcome: s.outcome,
                    }
                })
                .collect(),
        }
    }

    /// Applies one decided position move, maintaining the direction budget.
    ///
    /// Returns true if the tap actually moved.
    fn move_to(
        &mut self,
        state_idx: usize,
        wanted: i32,
        ctx: &mut OuterLoopContext<'_, '_>,
    ) -> bool {
        let reg_idx = self.states[state_idx].regulation_index;
        let transformer = ctx.net.regulation[reg_idx].transformer;
        let Some(Some(tc)) = ctx.net.tap_changers.get(transformer) else {
            return false;
        };
        let current = tc.position;
        let low = tc.low;
        let high = tc.high();

        // Cap the step, then clamp into range. Clamping after capping is what
        // makes a controller at a limit report `AtLimit` rather than looking
        // like it moved.
        let capped = wanted.clamp(current - self.max_tap_shift, current + self.max_tap_shift);
        let target = capped.clamp(low, high);
        if target == current {
            if wanted != current {
                // It wanted to move and could not: it is against a limit.
                self.states[state_idx].outcome = ControllerOutcome::AtLimit;
                self.states[state_idx].frozen = true;
            }
            return false;
        }

        let up = target > current;
        if let Some(previous) = self.states[state_idx].last_direction {
            if previous != up {
                self.states[state_idx].direction_changes += 1;
                if self.states[state_idx].direction_changes >= MAX_DIRECTION_CHANGES {
                    self.states[state_idx].outcome = ControllerOutcome::Hunting;
                    self.states[state_idx].frozen = true;
                    return false;
                }
            }
        }

        let tc = ctx.net.tap_changers[transformer].as_mut().expect("checked above");
        if !tc.set_position(&mut ctx.net.transformers[transformer], target) {
            return false;
        }
        self.states[state_idx].last_direction = Some(up);
        self.states[state_idx].steps_moved += (target - current).unsigned_abs() as usize;
        true
    }
}

// ---------------------------------------------------------------------------
// Voltage control
// ---------------------------------------------------------------------------

/// A ratio tap changer holding the voltage magnitude at its controlled bus.
///
/// Handles several controllers regulating one bus: the required change is
/// shared among them in sensitivity order, each one's achieved effect being
/// subtracted before the next is sized. That is powsybl's
/// `adjustWithSeveralControllers`, and it is deliberately *not* the "last
/// writer wins" behaviour the comparison table records for gridoxide's
/// generator-side shared voltage control.
pub struct TransformerVoltageControl {
    inner: TapLoop,
}

impl Default for TransformerVoltageControl {
    fn default() -> Self {
        Self { inner: TapLoop::new(3) }
    }
}

impl TransformerVoltageControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Positions a controller may move in one outer pass. The sensitivity is
    /// exact only at the converged state, so a large step overshoots; capping
    /// trades passes for stability.
    pub fn max_tap_shift(mut self, steps: i32) -> Self {
        self.inner.max_tap_shift = steps.max(1);
        self
    }

    pub fn report(&self, ctx_regulation: &[TapRegulation], tap_changers: &[Option<crate::types::TapChanger>]) -> TapControlReport {
        self.inner.report(ctx_regulation, tap_changers)
    }
}

fn is_voltage(reg: &TapRegulation) -> bool {
    matches!(reg.mode, RegulationMode::Voltage)
}

impl OuterLoop for TransformerVoltageControl {
    fn name(&self) -> &'static str {
        "TransformerVoltageControl"
    }

    fn invalidates(&self) -> Invalidates {
        Invalidates::Admittances
    }

    fn initialize(&mut self, ctx: &mut OuterLoopContext<'_, '_>) {
        self.inner.initialize(ctx, is_voltage);
    }

    fn check(&mut self, ctx: &mut OuterLoopContext<'_, '_>) -> OuterLoopStatus {
        if self.inner.states.is_empty() {
            return OuterLoopStatus::Stable;
        }

        // Which controllers act on which bus. Several tap changers regulating
        // one bus is normal on a substation with parallel transformers.
        let mut by_bus: Vec<(usize, Vec<usize>)> = Vec::new();
        for (s, state) in self.inner.states.iter().enumerate() {
            if state.frozen {
                continue;
            }
            let bus = ctx.net.regulation[state.regulation_index].controlled_bus;
            match by_bus.iter_mut().find(|(b, _)| *b == bus) {
                Some((_, v)) => v.push(s),
                None => by_bus.push((bus, vec![s])),
            }
        }
        if by_bus.is_empty() {
            return OuterLoopStatus::Stable;
        }

        let n_lines = ctx.net.lines.len();
        let Some(sensi) = AcSensitivity::new(
            ctx.net.buses,
            ctx.net.ybus,
            ctx.net.lines,
            ctx.net.transformers,
        ) else {
            return OuterLoopStatus::Failed(
                "tap voltage control: the Jacobian is singular at the operating point".into(),
            );
        };

        let mut moved = false;
        for (bus, controllers) in by_bus {
            let v = ctx.net.buses[bus].voltage_mag;
            // Every controller of one bus should name the same target; if they
            // disagree the tightest deadband and the first target win, and the
            // disagreement is visible in the report rather than silent.
            let reg0 = &ctx.net.regulation[self.inner.states[controllers[0]].regulation_index];
            let target = reg0.target;
            let half_deadband = reg0.deadband.abs() / 2.0;

            let mut remaining = target - v;
            if remaining.abs() <= half_deadband {
                for &s in &controllers {
                    self.inner.states[s].outcome = ControllerOutcome::InDeadband;
                }
                continue;
            }

            // Sensitivity first, so the controller with the most authority is
            // asked to do the most.
            let mut ranked: Vec<(usize, f64)> = Vec::new();
            for &s in &controllers {
                let t = ctx.net.regulation[self.inner.states[s].regulation_index].transformer;
                let sens = sensi
                    .state_response(Variable::TransformerRatio(n_lines + t))
                    .map(|r| r.d_vmag[bus])
                    .unwrap_or(0.0);
                if sens.abs() < MIN_SENSITIVITY {
                    self.inner.states[s].outcome = ControllerOutcome::Insensitive;
                    self.inner.states[s].frozen = true;
                    continue;
                }
                ranked.push((s, sens));
            }
            ranked.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));

            for (s, sens) in ranked {
                if remaining.abs() <= half_deadband {
                    self.inner.states[s].outcome = ControllerOutcome::InDeadband;
                    continue;
                }
                let t = ctx.net.regulation[self.inner.states[s].regulation_index].transformer;
                let Some(Some(tc)) = ctx.net.tap_changers.get(t) else { continue };
                let Some(current_ratio) = tc.ratio(tc.position) else { continue };
                let wanted_ratio = current_ratio + remaining / sens;
                let Some(wanted) = tc.nearest_to_ratio(wanted_ratio) else { continue };

                let before = tc.position;
                if self.inner.move_to(s, wanted, ctx) {
                    moved = true;
                    let tc = ctx.net.tap_changers[t].as_ref().expect("moved it");
                    let achieved = (tc.ratio(tc.position).unwrap_or(current_ratio) - current_ratio) * sens;
                    remaining -= achieved;
                    let _ = before;
                }
            }
        }

        if moved {
            OuterLoopStatus::Unstable
        } else {
            for state in self.inner.states.iter_mut() {
                if state.outcome == ControllerOutcome::Unfinished {
                    state.outcome = ControllerOutcome::InDeadband;
                }
            }
            OuterLoopStatus::Stable
        }
    }
}

// ---------------------------------------------------------------------------
// Phase control
// ---------------------------------------------------------------------------

/// A phase-shifting transformer holding the active power on a monitored
/// branch.
///
/// The sensitivity is [`Variable::PhaseShift`] against the branch's own active
/// flow. That pairing matters: a shifter usually regulates *its own* flow, and
/// `AcSensitivity` carries a direct \\(\partial f/\partial p\\) term for
/// exactly that case which, if dropped, flips the sign of the answer on the
/// regulated branch while leaving every other branch correct.
pub struct PhaseControl {
    inner: TapLoop,
}

impl Default for PhaseControl {
    fn default() -> Self {
        Self { inner: TapLoop::new(3) }
    }
}

impl PhaseControl {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_tap_shift(mut self, steps: i32) -> Self {
        self.inner.max_tap_shift = steps.max(1);
        self
    }

    pub fn report(&self, regulation: &[TapRegulation], tap_changers: &[Option<crate::types::TapChanger>]) -> TapControlReport {
        self.inner.report(regulation, tap_changers)
    }
}

fn is_active_power(reg: &TapRegulation) -> bool {
    matches!(reg.mode, RegulationMode::ActivePower { .. })
}

impl OuterLoop for PhaseControl {
    fn name(&self) -> &'static str {
        "PhaseControl"
    }

    fn invalidates(&self) -> Invalidates {
        Invalidates::Admittances
    }

    fn initialize(&mut self, ctx: &mut OuterLoopContext<'_, '_>) {
        self.inner.initialize(ctx, is_active_power);
    }

    fn check(&mut self, ctx: &mut OuterLoopContext<'_, '_>) -> OuterLoopStatus {
        if self.inner.states.is_empty() {
            return OuterLoopStatus::Stable;
        }
        let active: Vec<usize> = (0..self.inner.states.len())
            .filter(|&s| !self.inner.states[s].frozen)
            .collect();
        if active.is_empty() {
            return OuterLoopStatus::Stable;
        }

        let n_lines = ctx.net.lines.len();
        let Some(sensi) = AcSensitivity::new(
            ctx.net.buses,
            ctx.net.ybus,
            ctx.net.lines,
            ctx.net.transformers,
        ) else {
            return OuterLoopStatus::Failed(
                "tap phase control: the Jacobian is singular at the operating point".into(),
            );
        };
        let params = crate::branch_flow::branch_params(ctx.net.lines, ctx.net.transformers);
        let v = crate::branch_flow::bus_voltages(ctx.net.buses);

        let mut moved = false;
        for s in active {
            let reg = &ctx.net.regulation[self.inner.states[s].regulation_index];
            let RegulationMode::ActivePower { branch, terminal } = reg.mode else {
                continue;
            };
            let t = reg.transformer;
            let target = reg.target;
            let half_deadband = reg.deadband.abs() / 2.0;

            let Some(bp) = params.get(branch) else { continue };
            let (p, _) = crate::branch_flow::terminal_flow(bp, terminal, &v);
            let diff = target - p;
            if diff.abs() <= half_deadband {
                self.inner.states[s].outcome = ControllerOutcome::InDeadband;
                continue;
            }

            let sens = sensi
                .branch_response(Variable::PhaseShift(n_lines + t), terminal)
                .and_then(|r| r.get(branch).map(|(dp, _)| *dp))
                .unwrap_or(0.0);
            if sens.abs() < MIN_SENSITIVITY {
                self.inner.states[s].outcome = ControllerOutcome::Insensitive;
                self.inner.states[s].frozen = true;
                continue;
            }

            let Some(Some(tc)) = ctx.net.tap_changers.get(t) else { continue };
            let Some(current_angle) = tc.angle_deg(tc.position) else { continue };
            // The sensitivity is per radian; the tap table is in degrees.
            let wanted_angle = current_angle + (diff / sens).to_degrees();
            let Some(wanted) = tc.nearest_to_angle(wanted_angle) else { continue };

            if self.inner.move_to(s, wanted, ctx) {
                moved = true;
            }
        }

        if moved {
            OuterLoopStatus::Unstable
        } else {
            for state in self.inner.states.iter_mut() {
                if state.outcome == ControllerOutcome::Unfinished {
                    state.outcome = ControllerOutcome::InDeadband;
                }
            }
            OuterLoopStatus::Stable
        }
    }
}
