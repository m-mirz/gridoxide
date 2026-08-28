//! The device model boundary: what every dynamic device must be able to say
//! about itself so [`dae`](crate::dynamics::dae) can assemble the DAE.
//!
//! # The contract
//!
//! A device owns a contiguous block of differential states `x`, sits at one
//! bus, and answers four questions:
//!
//! - what its states' derivatives are, given `x` and the terminal voltage;
//! - what current it injects into the network, given the same;
//! - what the four partial-derivative blocks of those two answers are;
//! - what `x` must be for the derivatives to be *zero* at a given terminal
//!   condition — the initialization in reverse, see [`init`](super::init).
//!
//! Everything else — the integration rule, the Newton loop, the sparsity
//! pattern, the event schedule — is the caller's, and no model needs to know
//! which of them is in use.
//!
//! # Why a machine and its controls are one device
//!
//! An exciter writes `E_fd` into its machine, a governor writes `P_m`, a PSS
//! writes `v_s` into the exciter, and all three read machine states or the
//! terminal voltage. Modelled as separate devices those couplings would be
//! cross-device Jacobian entries: real, dense-ish, and a second sparsity
//! concern on top of the network's.
//!
//! [`GeneratingUnit`] instead composes them into **one** device with one
//! contiguous state block, so every coupling is an ordinary partial derivative
//! *inside* `dfdx` and the pattern sees one diagonal block per unit.
//!
//! The couplings are not written out per combination — that would be
//! combinatorial. Each part declares its own derivatives with respect to its
//! own states and its own scalar input, and [`unit`] composes them by the
//! chain rule over the signal graph, which is small and acyclic:
//!
//! ```text
//!            |V|  ────────────────┐
//!   ω ──► PSS ──► v_s ──► AVR ──► E_fd ──┐
//!   ω ──► governor ─────────────► P_m ───┤──► machine ──► I, ω, δ
//!                                 V  ────┘
//! ```
//!
//! No block's output depends on its own input through another block, so one
//! forward pass evaluates everything and one sweep of the chain rule
//! differentiates it.
//!
//! # Why the Norton admittance is separate
//!
//! A machine is a source behind an impedance. Writing the whole thing as a
//! current injection would make `∂I/∂V` large and would leave the Y-bus with
//! nothing on the generator diagonals; instead each device declares a
//! **constant** [`norton_admittance`](DynamicModel::norton_admittance) that is
//! stamped into `Y` once at build time, and [`injection`](
//! DynamicModel::injection) returns only the source half `E·y`. `Y` stays
//! constant for the lifetime of a topology — the property the whole
//! formulation rests on — and stays diagonally dominant at generator buses.
//!
//! A salient machine (`x'_q ≠ x'_d`) still leaves a `δ`-dependent term in the
//! injection. That is expected: it is handled in the model's own analytic
//! Jacobian, never by touching `Y`.

pub mod avr;
pub mod gov;
pub mod load;
pub mod machine;
pub mod pss;
pub mod unit;

use num_complex::Complex;

pub use avr::Sexs;
pub use gov::Tgov1;
pub use load::ZipLoad;
pub use machine::{GenCls, GenRound, GenTransient, Machine, MachineInit, MachineJacobian};
pub use pss::Stab1;
pub use unit::GeneratingUnit;

/// The four partial-derivative blocks a model contributes to the DAE Jacobian.
///
/// Sized by the model at construction and then refilled in place every Newton
/// iteration, so a step allocates nothing. All four are row-major and dense —
/// a device block is small (2 to ~12 states) and dense within itself, so
/// sparsity here would cost more than it saves.
#[derive(Clone, Debug)]
pub struct ModelJacobian {
    /// `∂f/∂x`, `n_states × n_states`.
    pub dfdx: Vec<f64>,
    /// `∂f/∂V`, `n_states × 2`. Columns are `(v_re, v_im)`.
    pub dfdv: Vec<f64>,
    /// `∂I/∂x`, `2 × n_states`. Rows are `(i_re, i_im)`.
    pub didx: Vec<f64>,
    /// `∂I/∂V`, `2 × 2`, row-major: `[∂i_re/∂v_re, ∂i_re/∂v_im,
    /// ∂i_im/∂v_re, ∂i_im/∂v_im]`.
    pub didv: [f64; 4],
}

impl ModelJacobian {
    pub fn zeros(n_states: usize) -> Self {
        Self {
            dfdx: vec![0.0; n_states * n_states],
            dfdv: vec![0.0; n_states * 2],
            didx: vec![0.0; 2 * n_states],
            didv: [0.0; 4],
        }
    }

    /// Zeroes every block without reallocating. Models fill only the entries
    /// they actually have, so the reset is theirs to rely on.
    pub fn clear(&mut self) {
        self.dfdx.fill(0.0);
        self.dfdv.fill(0.0);
        self.didx.fill(0.0);
        self.didv = [0.0; 4];
    }
}

/// Why a device could not be initialized to an equilibrium.
#[derive(Clone, Debug, PartialEq)]
pub enum InitError {
    /// The terminal voltage was zero, so the terminal current is undefined.
    /// Which device it was is the caller's to report — a model does not know
    /// its own bus, and deliberately so.
    ZeroTerminalVoltage,
    /// A parameter that must be positive was not — inertia, a time constant.
    NonPositiveParameter { name: &'static str, value: f64 },
    /// The equilibrium exists but lies outside a limit the model enforces.
    /// Not reachable in phase 1 (no model has limits yet); present so that
    /// adding one does not widen this enum in a breaking way later.
    OutsideLimits { name: &'static str, value: f64 },
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::ZeroTerminalVoltage => {
                write!(f, "terminal voltage is zero, so no terminal current is defined")
            }
            InitError::NonPositiveParameter { name, value } => {
                write!(f, "parameter {name} must be positive, got {value}")
            }
            InitError::OutsideLimits { name, value } => {
                write!(f, "equilibrium value of {name} ({value}) lies outside its limits")
            }
        }
    }
}

impl std::error::Error for InitError {}

/// One dynamic device.
///
/// Every method takes the terminal voltage as a `Complex<f64>` in the **system
/// reference frame**, per unit on the network's own base. A model that works
/// internally in its rotor frame does the rotation itself; nothing outside
/// knows about `dq` axes.
///
/// Machine parameters are conventionally given on the machine's own MVA
/// rating. Conversion to the network base happens **once**, at construction,
/// so every method here is already on the network base and no per-step code
/// has to remember which is which. See `plans/RMS_PLAN.md` §9.5 for why that
/// placement matters.
pub trait DynamicModel: std::fmt::Debug {
    fn n_states(&self) -> usize;

    /// One label per state, in state order. Used for trajectory column
    /// headings and for the finite-difference oracle's failure messages.
    fn state_names(&self) -> &[&'static str];

    /// The constant admittance stamped into `Y` at this device's bus, once, at
    /// build time. `None` for a device that is a pure current source.
    fn norton_admittance(&self) -> Option<Complex<f64>>;

    /// `dx/dt`, written into `out` (length `n_states`).
    fn derivatives(&self, x: &[f64], v: Complex<f64>, out: &mut [f64]);

    /// The current this device injects into its bus, in the system frame —
    /// the source half only, since [`norton_admittance`](Self::norton_admittance)
    /// already accounts for the rest.
    fn injection(&self, x: &[f64], v: Complex<f64>) -> Complex<f64>;

    /// The four blocks of §2's Jacobian, analytically. `out` arrives cleared.
    ///
    /// Every implementation of this is checked against
    /// [`finite_difference`] by `tests/dynamics_jacobian_test.rs` — gate G4.
    /// An analytic Jacobian is worth having (a numerical one costs
    /// `2·n_states` extra residual evaluations per iteration and loses four
    /// digits), but it is also the easiest thing in the crate to get subtly
    /// wrong, so it is never trusted without the oracle.
    fn jacobian(&self, x: &[f64], v: Complex<f64>, out: &mut ModelJacobian);

    /// Disconnect or reconnect this device.
    ///
    /// A disconnected device contributes no current, no admittance and no
    /// derivatives, so its states freeze where they were and the network stops
    /// seeing it. That is enough to make a unit trip a **value-only** event:
    /// the device's rows stay in the DAE, its Jacobian block becomes the
    /// identity the implicit rule contributes, and the sparsity pattern is
    /// untouched — so no re-analysis, and the run's one symbolic factorization
    /// still serves.
    ///
    /// Freezing is a modelling choice and worth naming. A real tripped machine
    /// keeps spinning and accelerates, having lost its load; nothing in the
    /// network can observe that, and reconnecting it would need
    /// synchronization, which is not modelled. So the states are held rather
    /// than integrated, and a reconnected unit resumes from where it stopped.
    ///
    /// The default ignores it — a device with no meaningful disconnected state
    /// need not implement this.
    fn set_connected(&mut self, _connected: bool) {}

    fn is_connected(&self) -> bool {
        true
    }

    /// Choose `x` so that `derivatives(x, v) == 0` at this terminal condition,
    /// and latch whatever internal references that implies (`P_m`, `V_ref`,
    /// the constant EMF magnitude).
    ///
    /// `s` is the complex power the device delivers **into the network** at
    /// its terminal, per unit on the network base, as the power flow solved
    /// it. `&mut self` because the latched references are the model's own.
    fn initialize(&mut self, v: Complex<f64>, s: Complex<f64>) -> Result<Vec<f64>, InitError>;
}

/// A scalar-input, scalar-output dynamic block inside a generating unit: an
/// exciter, a governor, a stabilizer.
///
/// One input and one output is not a simplification of these devices, it is
/// what they are — an exciter sees a voltage error and produces a field
/// voltage, a governor sees a speed deviation and produces a mechanical power.
/// Keeping the interface that narrow is what lets
/// [`GeneratingUnit`](unit::GeneratingUnit) compose any combination of them by
/// the chain rule instead of enumerating combinations.
///
/// The block's own **reference** — an exciter's `V_ref`, a governor's `P_ref` —
/// is latched inside it by [`initialize`](Control::initialize) and never
/// appears in the input. That is deliberate: a control should not have to know
/// what it is regulating, and the reference is derived from the operating point
/// rather than read from a file. See [`init`](crate::dynamics::init).
pub trait Control: std::fmt::Debug {
    fn n_states(&self) -> usize;
    fn state_names(&self) -> &[&'static str];

    /// `dx/dt`, written into `out`.
    fn derivatives(&self, x: &[f64], u: f64, out: &mut [f64]);

    /// `∂f/∂x` (`n × n` row-major) into `dfdx`, `∂f/∂u` (length `n`) into
    /// `dfdu`. Both arrive cleared.
    fn jacobian(&self, x: &[f64], u: f64, dfdx: &mut [f64], dfdu: &mut [f64]);

    fn output(&self, x: &[f64], u: f64) -> f64;

    /// `∂y/∂x` into `dydx` (cleared on arrival); returns `∂y/∂u`.
    ///
    /// A direct input-to-output path is normal here — a washout and a lead-lag
    /// both have one — and it is what makes the chain rule in
    /// [`unit`](unit) more than a block-diagonal copy.
    fn output_jacobian(&self, x: &[f64], u: f64, dydx: &mut [f64]) -> f64;

    /// Choose states, and latch whatever reference that implies, so that
    /// `output(x, u) == y` **and** every derivative is zero.
    ///
    /// Both conditions matter. Reproducing the output without sitting at an
    /// equilibrium gives a control that starts moving at `t = 0` for no
    /// physical reason.
    fn initialize(&mut self, y: f64, u: f64) -> Result<Vec<f64>, InitError>;
}

/// A central-difference oracle for [`DynamicModel::jacobian`].
///
/// Returns the same four blocks computed numerically. This exists for the same
/// reason `klu_native::ffi_oracle` does: the analytic version is what runs, and
/// the numerical version is what proves it. Not used in production paths.
pub fn finite_difference(model: &dyn DynamicModel, x: &[f64], v: Complex<f64>, h: f64) -> ModelJacobian {
    let n = model.n_states();
    let mut out = ModelJacobian::zeros(n);

    let mut f_plus = vec![0.0; n];
    let mut f_minus = vec![0.0; n];
    let mut probe = x.to_vec();

    // df/dx and dI/dx: perturb each state.
    for j in 0..n {
        let step = h * x[j].abs().max(1.0);
        probe.copy_from_slice(x);
        probe[j] = x[j] + step;
        model.derivatives(&probe, v, &mut f_plus);
        let i_plus = model.injection(&probe, v);
        probe[j] = x[j] - step;
        model.derivatives(&probe, v, &mut f_minus);
        let i_minus = model.injection(&probe, v);

        for i in 0..n {
            out.dfdx[i * n + j] = (f_plus[i] - f_minus[i]) / (2.0 * step);
        }
        out.didx[j] = (i_plus.re - i_minus.re) / (2.0 * step);
        out.didx[n + j] = (i_plus.im - i_minus.im) / (2.0 * step);
    }

    // df/dV and dI/dV: perturb each voltage component.
    for (col, unit) in [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)].iter().enumerate() {
        let step = h * v.norm().max(1.0);
        let v_plus = v + unit * step;
        let v_minus = v - unit * step;
        model.derivatives(x, v_plus, &mut f_plus);
        let i_plus = model.injection(x, v_plus);
        model.derivatives(x, v_minus, &mut f_minus);
        let i_minus = model.injection(x, v_minus);

        for i in 0..n {
            out.dfdv[i * 2 + col] = (f_plus[i] - f_minus[i]) / (2.0 * step);
        }
        out.didv[col] = (i_plus.re - i_minus.re) / (2.0 * step);
        out.didv[2 + col] = (i_plus.im - i_minus.im) / (2.0 * step);
    }

    out
}
