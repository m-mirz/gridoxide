//! Turning a solved power flow into an initialized dynamic system.
//!
//! # The rule
//!
//! Every device is initialized **backwards**: from its terminal voltage and
//! power, choose the states that make its derivatives *zero*, and latch
//! whatever internal reference that implies. A classical machine latches its
//! internal EMF magnitude and its mechanical power; an exciter (phase 3) will
//! latch `V_ref` so that its output is exactly the field voltage the machine
//! turned out to need. Nothing reads a setpoint from the input file — the
//! setpoint is *derived* from the operating point, because a file's stated
//! `V_ref` and its stated power flow are two independent claims and they will
//! not agree to machine precision.
//!
//! # The invariant this establishes, and the gate that checks it
//!
//! After [`build`], at `t = 0`:
//!
//! - every device's derivative vector is zero, and
//! - the network residual `Y V − I_inj(x, V)` is zero,
//!
//! both to machine precision, not merely to a solver tolerance. That is not an
//! aspiration; it falls out algebraically. Each machine's Norton current is
//! `E·y` with `E = V + z·I` for the very `I = conj(S/V)` the power flow
//! produced, so `Y V − I_inj` at its bus reduces to the power-flow equation
//! that was already satisfied. Each remaining injection is converted to an
//! admittance `y = −conj(R)/|V|²`, which reproduces `R` exactly at the voltage
//! it was derived at.
//!
//! [`build`] therefore *checks* both, and refuses to hand back a system that
//! fails either. A drift here is never a rounding artifact; it is a sign error
//! in a model's derivatives, a missed per-unit conversion, a Norton stamp that
//! disagrees with the impedance the same model initialized against, or a
//! mistake in the residual assembly. All of those are silent otherwise, and
//! all of them produce a trajectory that looks entirely plausible.
//!
//! **What the check cannot see is a wrong `DeviceSpec::s`.** Declaring that a
//! machine makes half of what it really makes is *self-consistent*: the
//! machine initializes to an equilibrium at the power it was told, and the
//! remainder becomes part of the bus's admittance, so both residuals are still
//! exactly zero. The result is a different machine — smaller output, smaller
//! internal EMF, smaller rotor angle — swinging against a network that makes
//! the difference up from something inert. This is why
//! [`DeviceSpec::s`](DeviceSpec::s) is stated by the caller rather than
//! guessed here: nothing downstream can recover the split, and nothing
//! downstream can detect that it was wrong.
//! `tests/dynamics_test.rs::a_mis_declared_split_is_silent_and_changes_the_machine`
//! pins that behaviour so it stays a known limitation rather than a surprise.
//!
//! # What loads become
//!
//! Constant impedance, derived at the solved voltage. This is the standard
//! phase-1 load model and it is exact at `t = 0` by construction. It is also a
//! real modelling choice with real consequences — a constant-impedance load
//! sheds power as the voltage dips, so it is *optimistic* about voltage
//! recovery compared to a constant-power load. Voltage-dependent and dynamic
//! load models are phase 3.

use num_complex::Complex;

use crate::network::{build_ybus, power_injections, stamp_shunts, ShuntAdm};
use crate::types::{Bus, Line, Transformer};

use super::dae::{residual, DaeLayout, DaePattern};
use super::models::{DynamicModel, InitError};
use super::DynamicSystem;

/// One device to place on the network.
pub struct DeviceSpec {
    /// The source document's own identifier, used for trajectory column names
    /// and for error messages. Not interpreted.
    pub id: String,
    pub bus: usize,
    /// The complex power this device injects into its bus at the initial
    /// operating point, per unit on the network base.
    ///
    /// Stated explicitly rather than inferred, because a bus may carry several
    /// machines and a load at once, and only the caller knows the split. What
    /// is left after every device at a bus is accounted for becomes that bus's
    /// constant-admittance load.
    pub s: Complex<f64>,
    pub model: Box<dyn DynamicModel>,
}

impl std::fmt::Debug for DeviceSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceSpec")
            .field("id", &self.id)
            .field("bus", &self.bus)
            .field("s", &self.s)
            .field("model", &self.model)
            .finish()
    }
}

/// Everything [`build`] needs: a solved network, its devices, and any buses
/// held at constant voltage.
#[derive(Debug)]
pub struct SystemSpec<'a> {
    /// Buses as the power flow **solved** them — `voltage_mag`/`voltage_ang`
    /// are read, and must be the converged values.
    pub buses: &'a [Bus],
    pub lines: &'a [Line],
    pub transformers: &'a [Transformer],
    pub shunts: &'a [ShuntAdm],
    pub devices: Vec<DeviceSpec>,
    /// Buses whose voltage is held constant for the whole run — an infinite
    /// bus. Their two algebraic equations become `V = V₀`.
    ///
    /// Exact, unlike the usual approximation of a machine with a very large
    /// inertia or a very small source impedance, and that exactness is what
    /// the closed-form equal-area gate needs. Optional: a system with no fixed
    /// bus is perfectly well posed, since every machine's rotor angle is an
    /// absolute state.
    pub fixed_buses: Vec<usize>,
}

/// Why a system could not be assembled.
#[derive(Clone, Debug, PartialEq)]
pub enum BuildError {
    BusOutOfRange { id: String, bus: usize, n_bus: usize },
    /// A device at a bus whose voltage is held constant. Refused rather than
    /// handled: the constraint would absorb whatever the device injected, so
    /// the device's own dynamics would be decoupled from the network and its
    /// trajectory would be meaningless.
    DeviceAtFixedBus { id: String, bus: usize },
    ModelInit { id: String, source: InitError },
    /// A bus with no admittance to anything, so its two algebraic rows are
    /// empty and the block is structurally singular.
    BusWithoutAdmittance { bus: usize },
    /// Some device is not at an equilibrium. See the module doc: this is
    /// always a real internal inconsistency, never rounding — but note that a
    /// mis-declared device/load split is *not* one of the things it catches.
    NotAnEquilibrium { max_derivative: f64 },
    /// The network constraint is not satisfied at the initial point.
    NetworkResidual { norm: f64 },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::BusOutOfRange { id, bus, n_bus } => {
                write!(f, "device {id} is at bus {bus}, but the network has {n_bus} buses")
            }
            BuildError::DeviceAtFixedBus { id, bus } => {
                write!(f, "device {id} is at bus {bus}, whose voltage is held constant")
            }
            BuildError::ModelInit { id, source } => write!(f, "device {id}: {source}"),
            BuildError::BusWithoutAdmittance { bus } => {
                write!(f, "bus {bus} has no admittance to anything, so the network block is singular")
            }
            BuildError::NotAnEquilibrium { max_derivative } => write!(
                f,
                "initial point is not an equilibrium: largest state derivative is {max_derivative:.3e}"
            ),
            BuildError::NetworkResidual { norm } => write!(
                f,
                "network constraint is not satisfied at the initial point: residual {norm:.3e}"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

/// Tolerance for the two equilibrium checks. Generous by three orders of
/// magnitude relative to what a correct build actually achieves (`~1e-16`), so
/// this fires on mistakes and never on arithmetic.
const EQUILIBRIUM_TOL: f64 = 1e-8;

/// Assembles and initializes a [`DynamicSystem`] from a solved case.
pub fn build(spec: SystemSpec<'_>) -> Result<DynamicSystem, BuildError> {
    let n_bus = spec.buses.len();

    let mut is_fixed = vec![false; n_bus];
    for &b in &spec.fixed_buses {
        if b < n_bus {
            is_fixed[b] = true;
        }
    }

    for dev in &spec.devices {
        if dev.bus >= n_bus {
            return Err(BuildError::BusOutOfRange {
                id: dev.id.clone(),
                bus: dev.bus,
                n_bus,
            });
        }
        if is_fixed[dev.bus] {
            return Err(BuildError::DeviceAtFixedBus { id: dev.id.clone(), bus: dev.bus });
        }
    }

    let v: Vec<Complex<f64>> = spec
        .buses
        .iter()
        .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
        .collect();

    // The network as the power flow saw it: branches and switched shunts only.
    // Its injections are read off *this* matrix, not off `p_spec`/`q_spec`,
    // because at a slack or PV bus those are not the solved values — the solve
    // decided them. `power_injections` is exact for every bus type alike.
    let mut base = build_ybus(n_bus, spec.lines, spec.transformers);
    stamp_shunts(&mut base, spec.shunts);
    let (p_calc, q_calc) = power_injections(spec.buses, &base.finish());

    // Assembled a second time rather than cloned: `YBus` is a builder, not a
    // matrix, and the two differ by every stamp added below. Rebuilding says
    // that plainly and costs one pass over the branch list.
    let mut ybus = build_ybus(n_bus, spec.lines, spec.transformers);
    stamp_shunts(&mut ybus, spec.shunts);

    // What each bus injects beyond its devices becomes a constant admittance.
    let mut remaining: Vec<Complex<f64>> =
        (0..n_bus).map(|i| Complex::new(p_calc[i], q_calc[i])).collect();
    for dev in &spec.devices {
        remaining[dev.bus] -= dev.s;
    }
    // An element of admittance y injects S = −|V|²·conj(y), so reproducing an
    // injection R at this voltage takes y = −conj(R)/|V|².
    //
    // Stamped for **every** bus, including the zeros: the entry has to exist in
    // the pattern before a fault can be applied there later, and a diagonal
    // that is structurally present but numerically zero costs one nonzero.
    // This is the topology-superset property `events` relies on.
    let mut load_y = vec![Complex::new(0.0, 0.0); n_bus];
    for i in 0..n_bus {
        let v_sq = v[i].norm_sqr();
        if !is_fixed[i] && v_sq != 0.0 {
            load_y[i] = -remaining[i].conj() / v_sq;
        }
        ybus.add(i, i, load_y[i]);
    }

    // Each device's Norton admittance, constant for the topology's lifetime.
    let norton: Vec<Option<Complex<f64>>> =
        spec.devices.iter().map(|d| d.model.norton_admittance()).collect();
    for (dev, y) in spec.devices.iter().zip(norton.iter()) {
        if let Some(y) = y {
            ybus.add(dev.bus, dev.bus, *y);
        }
    }
    let ybus = ybus.finish();

    // Initialize every device to its own equilibrium.
    let mut ids = Vec::with_capacity(spec.devices.len());
    let mut models: Vec<Box<dyn DynamicModel>> = Vec::with_capacity(spec.devices.len());
    let mut dev_bus = Vec::with_capacity(spec.devices.len());
    let mut dev_len = Vec::with_capacity(spec.devices.len());
    let mut x0: Vec<f64> = Vec::new();

    for dev in spec.devices {
        let DeviceSpec { id, bus, s, mut model } = dev;
        let states = model
            .initialize(v[bus], s)
            .map_err(|source| BuildError::ModelInit { id: id.clone(), source })?;
        dev_len.push(states.len());
        dev_bus.push(bus);
        x0.extend_from_slice(&states);
        ids.push(id);
        models.push(model);
    }

    let layout = DaeLayout::new(&dev_len, &dev_bus, n_bus);
    let pattern = DaePattern::analyze(layout, &ybus, &spec.fixed_buses);
    if let Some(bus) = pattern.bus_without_diagonal() {
        return Err(BuildError::BusWithoutAdmittance { bus });
    }

    let system = DynamicSystem {
        models,
        ids,
        ybus,
        pattern,
        v_fixed: v.clone(),
        lines: spec.lines.to_vec(),
        transformers: spec.transformers.to_vec(),
        shunts: spec.shunts.to_vec(),
        outaged: vec![false; spec.lines.len() + spec.transformers.len()],
        load_y,
        fault_y: vec![Complex::new(0.0, 0.0); n_bus],
        norton,
        x0,
        v0: v,
    };

    let drift = system.max_derivative();
    if drift > EQUILIBRIUM_TOL {
        return Err(BuildError::NotAnEquilibrium { max_derivative: drift });
    }
    let norm = system.network_residual_norm();
    if norm > EQUILIBRIUM_TOL {
        return Err(BuildError::NetworkResidual { norm });
    }

    Ok(system)
}

impl DynamicSystem {
    /// The infinity norm of the network constraint at the current state — the
    /// second half of the equilibrium invariant, and the quantity an event's
    /// algebraic re-solve drives back to zero.
    pub fn network_residual_norm(&self) -> f64 {
        let layout = self.pattern.layout();
        let n = layout.n();
        let f0 = vec![0.0; layout.n_diff];
        let mut f_scratch = vec![0.0; layout.n_diff];
        let mut out = vec![0.0; n];
        residual(
            &self.pattern,
            &self.models,
            &self.x0,
            &f0,
            &self.x0,
            &self.v0,
            &self.v_fixed,
            0.0,
            0.0,
            &self.ybus,
            &mut f_scratch,
            &mut out,
        );
        out[layout.n_diff..].iter().fold(0.0f64, |m, r| m.max(r.abs()))
    }
}
