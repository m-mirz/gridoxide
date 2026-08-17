//! C-layout mirrors of the network primitives, for callers who already have a
//! grid model in memory and would rather not serialize it to a file.
//!
//! Each is a plain-old-data struct: no pointers, no lengths, nothing owned.
//! That is what lets a caller build a `std::vector<GridoxideBus>` and hand over
//! `.data()` with no marshalling step at all.
//!
//! # Why these are not just `#[repr(C)]` on the real types
//!
//! Because [`Bus`](crate::types::Bus) carries a `Vec<ZipTerm>`, which has no C
//! layout. Mirroring lets that one field be handled deliberately (see
//! [`GridoxideBus`]) instead of forcing the whole type into a shape it does not
//! have — and it decouples the ABI from internal refactors, which matters more:
//! a field reordered in `types::Bus` would otherwise silently change the
//! meaning of every struct a compiled C++ binary passes in.

use num_complex::Complex;

use crate::network::ShuntAdm;
use crate::types::{Bus, BusType, Line, Transformer};

/// A complex number, as C sees it. Layout-compatible with `double[2]`,
/// `std::complex<double>` and `double _Complex`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GridoxideComplex {
    pub re: f64,
    pub im: f64,
}

impl From<GridoxideComplex> for Complex<f64> {
    fn from(value: GridoxideComplex) -> Self {
        Complex::new(value.re, value.im)
    }
}

/// What a bus is for the solve. Values are fixed by the ABI, not by the
/// discriminants of [`BusType`](crate::types::BusType).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridoxideBusType {
    /// Angle and magnitude fixed; active power is the answer.
    Slack = 0,
    /// Magnitude and active power fixed; reactive power is the answer.
    Pv = 1,
    /// Both powers fixed; the voltage is the answer.
    Pq = 2,
}

/// A bus.
///
/// # What is missing, and why
///
/// **ZIP terms.** [`Bus`](crate::types::Bus) can carry voltage-dependent load
/// components, which are a `Vec` and so have no place in a POD struct. A
/// network needing them must come in through the PGM document path, where they
/// are read as a matter of course.
///
/// This is a real limitation and is stated in the header rather than left to be
/// discovered: a caller who has ZIP loads and uses this path gets them silently
/// treated as constant power, which converges perfectly well to the wrong
/// answer.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GridoxideBus {
    pub bus_type: GridoxideBusType,
    /// Per-unit. For `Slack` and `Pv` this is the setpoint; elsewhere the
    /// starting guess.
    pub voltage_magnitude: f64,
    /// Radians. Only meaningful for `Slack`.
    pub voltage_angle: f64,
    /// Per-unit specified injection, **generation minus load** — so a load bus
    /// is negative.
    pub p_spec: f64,
    pub q_spec: f64,
    /// Per-unit reactive limits, used only by the Q-limit-enforcing solve.
    /// Pass `-INFINITY` / `INFINITY` for an unlimited machine.
    pub q_min: f64,
    pub q_max: f64,
    /// Rated line-to-line voltage in volts, used to report `voltage_kv`.
    /// Zero if you do not need it.
    pub u_rated: f64,
}

impl GridoxideBus {
    pub(crate) fn to_bus(self, idx: usize) -> Bus {
        Bus {
            idx,
            bus_type: match self.bus_type {
                GridoxideBusType::Slack => BusType::Slack,
                GridoxideBusType::Pv => BusType::PV,
                GridoxideBusType::Pq => BusType::PQ,
            },
            voltage_mag: self.voltage_magnitude,
            voltage_ang: self.voltage_angle,
            p_spec: self.p_spec,
            q_spec: self.q_spec,
            q_min: self.q_min,
            q_max: self.q_max,
            u_rated: self.u_rated,
            zip_terms: Vec::new(),
        }
    }
}

/// A line, in per-unit on the system base.
///
/// `b_shunt`/`g_shunt` are the **total** charging of the π-model, split evenly
/// across the two terminals — not the per-terminal half.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GridoxideLine {
    pub from: usize,
    pub to: usize,
    pub r: f64,
    pub x: f64,
    pub b_shunt: f64,
    pub g_shunt: f64,
}

impl From<GridoxideLine> for Line {
    fn from(value: GridoxideLine) -> Self {
        Line {
            from: value.from,
            to: value.to,
            r: value.r,
            x: value.x,
            b_shunt: value.b_shunt,
            g_shunt: value.g_shunt,
        }
    }
}

/// A two-winding transformer, in per-unit on the system base and the *to*-side
/// voltage base.
///
/// `tap` is complex: its magnitude is the off-nominal ratio `k`, its argument
/// the vector-group phase shift. A plain ratio with no shift is
/// `{ re: k, im: 0 }`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GridoxideTransformer {
    pub from: usize,
    pub to: usize,
    /// Nonzero if the winding is connected. Both must be nonzero for the
    /// branch to carry power.
    pub from_status: u8,
    pub to_status: u8,
    pub y_series: GridoxideComplex,
    pub y_shunt: GridoxideComplex,
    pub tap: GridoxideComplex,
}

impl From<GridoxideTransformer> for Transformer {
    fn from(value: GridoxideTransformer) -> Self {
        Transformer {
            from: value.from,
            to: value.to,
            from_status: value.from_status,
            to_status: value.to_status,
            y_series: value.y_series.into(),
            y_shunt: value.y_shunt.into(),
            tap: value.tap.into(),
        }
    }
}

/// A shunt admittance at a bus, per-unit.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GridoxideShunt {
    pub at: usize,
    pub y: GridoxideComplex,
}

impl From<GridoxideShunt> for ShuntAdm {
    fn from(value: GridoxideShunt) -> Self {
        ShuntAdm { at: value.at, y: value.y.into() }
    }
}

/// Which sparse-LU backend factorizes the Jacobian.
///
/// The two beyond `Scalar` and `Block` need their cargo features compiled in;
/// asking for one that is not produces
/// [`Status::InvalidArgument`](super::Status::InvalidArgument) rather than a
/// silent downgrade, because a caller selecting a backend usually has a reason.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridoxideBackend {
    /// faer's sparse LU. Always available; the default.
    Scalar = 0,
    /// Per-bus 2×2 real blocks. Always available.
    Block = 1,
    /// SuiteSparse KLU. Needs the `klu` feature.
    Klu = 2,
    /// The pure-Rust KLU port. Needs no feature.
    KluNative = 3,
    /// Intel oneMKL PARDISO. Needs the `pardiso` feature.
    Pardiso = 4,
}

/// Everything that shapes a solve, in one struct.
///
/// A struct rather than eight positional parameters because the equivalent
/// Python entry points already carry `#[allow(clippy::too_many_arguments)]`
/// twice — and because adding a field here is source-compatible for a C++
/// caller who zero-initializes, where adding a parameter is not.
///
/// Zero-initializing gives an unusable configuration (`tol = 0`,
/// `max_iter = 0`); call
/// [`gridoxide_options_default`](gridoxide_options_default) first and then
/// override what you care about.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GridoxideOptions {
    pub backend: GridoxideBackend,
    /// Convergence tolerance on the largest power mismatch, per-unit.
    pub tolerance: f64,
    pub max_iterations: usize,
    /// System power base in VA, used when reading a PGM document. Ignored by
    /// the in-memory path, which is per-unit already.
    pub s_base_va: f64,
    /// Nominal frequency in Hz, used when reading a PGM document.
    pub frequency_hz: f64,
}

impl Default for GridoxideOptions {
    fn default() -> Self {
        Self {
            backend: GridoxideBackend::Scalar,
            tolerance: 1e-8,
            max_iterations: 20,
            s_base_va: 1e6,
            frequency_hz: 50.0,
        }
    }
}

/// Fills `out` with the defaults. Call this before overriding fields, so a
/// later-added field does not leave you with a zero in it.
///
/// # Safety
///
/// `out` must point to a writable [`GridoxideOptions`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_options_default(out: *mut GridoxideOptions) {
    if out.is_null() {
        return;
    }
    // SAFETY: non-null and writable by contract.
    unsafe { *out = GridoxideOptions::default() };
}
