//! Remote voltage control: holding a bus the machine is not connected to.
//!
//! A generator regulating the far side of its own step-up transformer is
//! ordinary, and gridoxide has always imported it — `RegulatingControl.Terminal`
//! resolves to the controlled bus, which is then pinned to `PV` at the target.
//! That gets the target right and the reactive power **in the wrong place**: it
//! appears at the bus being held rather than at the machine, so the reactive
//! flow on the path between them is missing and the machine's own bus sits at
//! whatever the network makes it, rather than at whatever holding the far bus
//! requires.
//!
//! # The formulation, and why this one
//!
//! Exactly, the problem is a small generalization of the `PV` bus. A `PV` bus
//! today means two things at once — *this bus's magnitude is fixed* and *this
//! bus's reactive injection is free* — and remote control simply separates
//! them: fix `|V|` at the controlled bus, free `Q` at the controller. The
//! unknown count still balances, one magnitude removed against one reactive
//! equation removed.
//!
//! Implementing it that way means the unknown layout stops being derivable from
//! `BusType`, which is the assumption `jacobian::JacobianPattern`,
//! `solver::newton_raphson_cached`, `ac_sensitivity`, `bde`, `block_sparse` and
//! `continuation::augmented` all share. That is a large change to the most
//! load-bearing code in the crate, for a configuration two vendored fixtures
//! have.
//!
//! This loop reaches the **same fixed point** without touching any of it. Make
//! the *controller* bus `PV` — which is what makes its reactive power free, and
//! free at the right bus — and drive its own setpoint until the controlled bus
//! reaches the target. At convergence `|V|` at the controlled bus is the target
//! and `Q` at the controller is whatever holding it takes, which is precisely
//! what the exact formulation asserts. The difference is iteration, not answer,
//! and iterating a control to a fixed point is what this module is for.
//!
//! It also inherits the reactive limits for free: the controller bus is a `PV`
//! bus carrying the machine's own capability, so
//! [`ReactiveLimits`](super::ReactiveLimits) clamps the machine rather than the
//! bus it happens to be holding — which is the more correct bound as well as
//! the easier one.
//!
//! # The step
//!
//! `Δ|V|_controller = gain · (target − |V|_controlled)`, with `gain` starting at
//! 1.0 and refined by secant from the response actually observed.
//!
//! The initial 1.0 is not arbitrary: a machine holding the far side of its own
//! transformer moves that bus nearly one-for-one in per-unit, so the first step
//! lands close and the secant corrects the rest. A derived sensitivity
//! (`ac_sensitivity` would need a voltage-setpoint variable, which it has no
//! other caller for) is the better answer once a second fixture asks for it;
//! deriving it for one case would be building the general thing on a sample of
//! one, which §2 of `plans/REACTIVE_DISPATCH_PLAN.md` is a warning about.

use crate::types::{Bus, BusType, RegulatingMachine};

use super::{Invalidates, OuterLoop, OuterLoopContext, OuterLoopStatus};

/// Below this response the controller cannot move its target and says so,
/// rather than dividing by it and stepping to infinity. Mirrors
/// `taps::MIN_SENSITIVITY`'s role.
const MIN_RESPONSE: f64 = 1e-4;

/// How far the setpoint may move in one pass, per-unit. The secant gain is
/// exact only locally, so an early large step overshoots and oscillates;
/// capping trades passes for stability, exactly as `max_tap_shift` does.
const MAX_STEP: f64 = 0.1;

/// What one remote controller did.
#[derive(Clone, Debug, PartialEq)]
pub struct RemoteControlReport {
    pub machine: String,
    pub controller_bus: usize,
    pub controlled_bus: usize,
    pub target: f64,
    /// The controlled bus's voltage as the run left it.
    pub reached: f64,
    /// The controller's own setpoint as the run left it.
    pub setpoint: f64,
    pub moves: usize,
    pub outcome: RemoteOutcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteOutcome {
    /// Inside the deadband. The success case.
    Held,
    /// The controlled bus barely responds to this machine, so no setpoint
    /// reaches the target. A property of the network, not a solver failure.
    Insensitive,
    /// The budget ran out before the deadband was reached.
    Unfinished,
    /// The machine ran out of reactive capability on the way, so it stopped
    /// holding anything — `ReactiveLimits` switched its bus to `PQ` and this
    /// controller is no longer in control.
    AtReactiveLimit,
}

struct Controller {
    machine: String,
    controller_bus: usize,
    controlled_bus: usize,
    target: f64,
    moves: usize,
    /// The previous (setpoint, reached) pair, for the secant.
    previous: Option<(f64, f64)>,
    gain: f64,
    outcome: RemoteOutcome,
}

/// Drives each remotely-regulating machine's own voltage until the bus it holds
/// reaches its target.
///
/// See the module docs for why this is a loop rather than an equation.
pub struct RemoteVoltageControl {
    controllers: Vec<Controller>,
    /// Every regulating machine, not just the remote ones — the capability a
    /// bus ends up with is the sum of everything regulating *at* it, and a
    /// controller bus may host locally-regulating machines too.
    machines: Vec<RegulatingMachine>,
    /// Full width, per-unit; a controller is satisfied within `±deadband/2`.
    deadband: f64,
    /// Buses this loop re-typed at `initialize`, so a caller can see that the
    /// network it solved is not quite the one it handed over.
    retyped: Vec<(usize, BusType, BusType)>,
    /// Machines whose controlled bus is also held locally by something else —
    /// reported rather than resolved, since two controls on one bus disagreeing
    /// is a statement about the document, not a case to arbitrate.
    contested: Vec<String>,
}

impl RemoteVoltageControl {
    /// Builds a loop for every machine in `machines` that regulates a bus other
    /// than its own. Machines under local control are left entirely alone —
    /// they are already `PV` at the bus they hold, which is correct.
    pub fn new(machines: &[RegulatingMachine]) -> Self {
        let controllers = machines
            .iter()
            .filter(|m| m.at_bus != m.controls_bus)
            .map(|m| Controller {
                machine: m.id.clone(),
                controller_bus: m.at_bus,
                controlled_bus: m.controls_bus,
                target: m.target_pu,
                moves: 0,
                previous: None,
                gain: 1.0,
                outcome: RemoteOutcome::Unfinished,
            })
            .collect();
        Self {
            controllers,
            machines: machines.to_vec(),
            deadband: 2e-4,
            retyped: Vec::new(),
            contested: Vec::new(),
        }
    }

    /// Full deadband width, per-unit. Default `2e-4`.
    pub fn deadband(mut self, width: f64) -> Self {
        self.deadband = width.abs().max(1e-9);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.controllers.is_empty()
    }

    /// Buses whose type this loop changed, as `(bus, from, to)`.
    pub fn retyped(&self) -> &[(usize, BusType, BusType)] {
        &self.retyped
    }

    /// Machines whose controlled bus is held locally as well. Their remote
    /// control is dropped rather than fought over.
    pub fn contested(&self) -> &[String] {
        &self.contested
    }

    pub fn reports(&self) -> Vec<RemoteControlReport> {
        self.controllers
            .iter()
            .map(|c| RemoteControlReport {
                machine: c.machine.clone(),
                controller_bus: c.controller_bus,
                controlled_bus: c.controlled_bus,
                target: c.target,
                reached: f64::NAN,
                setpoint: f64::NAN,
                moves: c.moves,
                outcome: c.outcome,
            })
            .collect()
    }

    /// As [`reports`](Self::reports), filled in from the state the run ended at.
    pub fn report_against(&self, buses: &[Bus]) -> Vec<RemoteControlReport> {
        self.controllers
            .iter()
            .map(|c| RemoteControlReport {
                machine: c.machine.clone(),
                controller_bus: c.controller_bus,
                controlled_bus: c.controlled_bus,
                target: c.target,
                reached: buses[c.controlled_bus].voltage_mag,
                setpoint: buses[c.controller_bus].voltage_mag,
                moves: c.moves,
                outcome: c.outcome,
            })
            .collect()
    }
}

impl OuterLoop for RemoteVoltageControl {
    fn name(&self) -> &'static str {
        "RemoteVoltageControl"
    }

    fn invalidates(&self) -> Invalidates {
        // Only the controller bus's own magnitude setpoint moves once running.
        // The bus-type changes all happen in `initialize`, before any
        // factorization exists to invalidate.
        Invalidates::Nothing
    }

    /// Moves the control from the bus being held to the machine holding it.
    ///
    /// The importer pins the *controlled* bus to `PV`, which is where the
    /// reactive power wrongly ends up. Here that is undone and the machine's own
    /// bus is pinned instead, carrying the machine's own capability so the
    /// reactive-limit clamp applies to the machine rather than to a bus it is
    /// not connected to.
    fn initialize(&mut self, ctx: &mut OuterLoopContext<'_, '_>) {
        let buses = &mut *ctx.net.buses;

        // A controlled bus that some *other* machine holds locally is already
        // legitimately `PV`, and un-pinning it would break that control to
        // serve this one. Neither answer is right, so the remote control is
        // dropped and named.
        let mut keep: Vec<bool> = vec![true; self.controllers.len()];
        for (i, c) in self.controllers.iter().enumerate() {
            let held_locally = self
                .controllers
                .iter()
                .any(|o| o.controller_bus == c.controlled_bus && o.controlled_bus == c.controlled_bus);
            if held_locally {
                keep[i] = false;
                self.contested.push(c.machine.clone());
            }
        }
        let mut idx = 0;
        self.controllers.retain(|_| {
            idx += 1;
            keep[idx - 1]
        });

        // The capability has to move with the control, or the machine is
        // clamped against a limit belonging to a bus it is not connected to —
        // which for a bus with nothing else holding it means no limit at all.
        // Summed over everything regulating *at* each bus, because a controller
        // bus may host locally-regulating machines as well.
        let capability = |bus: usize, machines: &[RegulatingMachine]| {
            let here: Vec<&RegulatingMachine> =
                machines.iter().filter(|m| m.at_bus == bus).collect();
            if here.is_empty() {
                return None;
            }
            Some((
                here.iter().map(|m| m.q_min).sum::<f64>(),
                here.iter().map(|m| m.q_max).sum::<f64>(),
            ))
        };

        for c in &self.controllers {
            let was = buses[c.controlled_bus].bus_type;
            if was == BusType::PV {
                buses[c.controlled_bus].bus_type = BusType::PQ;
                // Whatever still regulates *at* this bus keeps its own
                // capability; a bus left with nothing holding it is unlimited,
                // which is "no constraint", not "no capability".
                let (lo, hi) = capability(c.controlled_bus, &self.machines)
                    .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
                buses[c.controlled_bus].q_min = lo;
                buses[c.controlled_bus].q_max = hi;
                self.retyped.push((c.controlled_bus, was, BusType::PQ));
            }
        }
        for c in &self.controllers {
            let was = buses[c.controller_bus].bus_type;
            if was == BusType::PQ {
                buses[c.controller_bus].bus_type = BusType::PV;
                self.retyped.push((c.controller_bus, was, BusType::PV));
            }
            if was == BusType::Slack {
                // A slack bus already fixes its own magnitude and already has a
                // free reactive injection, so there is nothing for this loop to
                // move; its target is simply unreachable through this machine.
                continue;
            }
            if let Some((lo, hi)) = capability(c.controller_bus, &self.machines) {
                buses[c.controller_bus].q_min = lo;
                buses[c.controller_bus].q_max = hi;
            }
            // Start where the target is. In per-unit the two buses track each
            // other closely across a transformer, so this lands near the answer
            // and leaves the loop a short correction rather than a search.
            buses[c.controller_bus].voltage_mag = c.target;
        }
    }

    fn check(&mut self, ctx: &mut OuterLoopContext<'_, '_>) -> OuterLoopStatus {
        let half = self.deadband / 2.0;
        let mut moved = false;

        for c in &mut self.controllers {
            // A machine that ran out of reactive capability is no longer
            // holding anything, and `ReactiveLimits` has already said so by
            // switching its bus. Continuing to move a setpoint nothing reads
            // would spend the budget to no effect.
            if ctx.net.buses[c.controller_bus].bus_type != BusType::PV {
                c.outcome = RemoteOutcome::AtReactiveLimit;
                continue;
            }

            let reached = ctx.net.buses[c.controlled_bus].voltage_mag;
            let error = c.target - reached;
            if error.abs() <= half {
                c.outcome = RemoteOutcome::Held;
                continue;
            }

            let setpoint = ctx.net.buses[c.controller_bus].voltage_mag;
            if let Some((prev_set, prev_reached)) = c.previous {
                let d_set = setpoint - prev_set;
                let d_reached = reached - prev_reached;
                if d_set.abs() > 1e-12 {
                    let response = d_reached / d_set;
                    if response.abs() < MIN_RESPONSE {
                        c.outcome = RemoteOutcome::Insensitive;
                        continue;
                    }
                    c.gain = 1.0 / response;
                }
            }

            let step = (c.gain * error).clamp(-MAX_STEP, MAX_STEP);
            c.previous = Some((setpoint, reached));
            ctx.net.buses[c.controller_bus].voltage_mag = setpoint + step;
            c.moves += 1;
            c.outcome = RemoteOutcome::Unfinished;
            moved = true;
        }

        if moved {
            OuterLoopStatus::Unstable
        } else {
            OuterLoopStatus::Stable
        }
    }
}
