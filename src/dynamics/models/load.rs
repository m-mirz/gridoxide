//! Voltage-dependent loads.
//!
//! # Why a load needs a model at all
//!
//! [`init`](crate::dynamics::init) turns every bus's residual injection into a
//! constant admittance, which is exact at the operating point and is the right
//! default. It is also *optimistic*: a constant-impedance load sheds power as
//! the square of the voltage, so during a fault it draws far less than a real
//! load would and the voltage recovers more readily than it should.
//!
//! A constant-*power* load is the pessimistic end of the same axis, and real
//! loads sit somewhere between. [`ZipLoad`] puts the choice in the caller's
//! hands, as three fractions summing to one.
//!
//! # The low-voltage problem, and the cutoff
//!
//! A constant-power load is singular at zero voltage: `I = conj(S)/conj(V)`
//! diverges as `V → 0`, and a nearby fault drives it there. Left alone it makes
//! the step's Newton solve diverge, and the physical claim it encodes —
//! that a load draws unbounded current from a collapsed bus — is false anyway.
//!
//! Every RMS simulator therefore applies a cutoff, and this one does it by
//! writing all three ZIP parts in one form:
//!
//! ```text
//! I = conj(S_z)·V/V₀²  +  conj(S_i)·V/(V₀·m)  +  conj(S_p)·V/m²
//! ```
//!
//! with `m = max(|V|, V_cut)`. Above the cutoff those are exactly the constant
//! impedance, constant current and constant power terms. Below it, `m` freezes
//! and every term becomes proportional to `V` — that is, the whole load becomes
//! a constant impedance, the one that reproduces its own current at the cutoff
//! voltage.
//!
//! The current is continuous at the cutoff; its derivative is not. That kink is
//! standard and survivable — a Newton step lands on it rarely — but it is a
//! real property of the model and worth knowing about, not an implementation
//! artifact.
//!
//! **The cutoff changes answers.** It is a modelling parameter, not a
//! numerical tolerance, and it should be reported alongside a result that
//! depended on it.

use num_complex::Complex;

use super::{DynamicModel, InitError, ModelJacobian};

/// A static ZIP load: constant impedance, constant current and constant power
/// in stated proportions, with a low-voltage cutoff.
///
/// It carries **no differential states**. It is a device in the DAE all the
/// same, because it contributes a nonlinear current injection and its
/// `∂I/∂V` — which is exactly what an algebraic device is.
#[derive(Clone, Debug)]
pub struct ZipLoad {
    /// Fractions of the operating-point injection, in `(z, i, p)` order.
    fractions: [f64; 3],
    cutoff: f64,
    /// Latched at initialization: the reference voltage magnitude and the
    /// conjugated injection of each part.
    v0: f64,
    s_conj: [Complex<f64>; 3],
}

impl ZipLoad {
    /// Fractions must sum to one; they are the shares of the load's *own*
    /// operating-point injection, not of the bus's.
    ///
    /// `cutoff` is the voltage below which the whole load becomes constant
    /// impedance. `0.5` per unit is a common choice and is what Dynawo's own
    /// default is near; a value of zero is refused, since it would restore the
    /// singularity the cutoff exists to remove.
    pub fn new(z: f64, i: f64, p: f64, cutoff: f64) -> Result<Self, InitError> {
        let sum = z + i + p;
        if (sum - 1.0).abs() > 1e-9 {
            return Err(InitError::OutsideLimits { name: "zip fractions sum", value: sum });
        }
        if cutoff <= 0.0 {
            return Err(InitError::NonPositiveParameter { name: "zip cutoff", value: cutoff });
        }
        Ok(Self {
            fractions: [z, i, p],
            cutoff,
            v0: 1.0,
            s_conj: [Complex::new(0.0, 0.0); 3],
        })
    }

    /// A pure constant-power load, with the usual cutoff. The pessimistic end
    /// of the range, and the one that makes voltage recovery hardest.
    pub fn constant_power(cutoff: f64) -> Result<Self, InitError> {
        Self::new(0.0, 0.0, 1.0, cutoff)
    }

    /// The per-part weights `w` and their derivatives with respect to `|V|`,
    /// at this terminal voltage.
    fn weights(&self, v_mag: f64) -> ([f64; 3], [f64; 3]) {
        let m = v_mag.max(self.cutoff);
        let clamped = v_mag <= self.cutoff;
        let w = [1.0 / (self.v0 * self.v0), 1.0 / (self.v0 * m), 1.0 / (m * m)];
        let dw = if clamped {
            [0.0, 0.0, 0.0]
        } else {
            [0.0, -1.0 / (self.v0 * m * m), -2.0 / (m * m * m)]
        };
        (w, dw)
    }
}

const NO_STATES: [&str; 0] = [];

impl DynamicModel for ZipLoad {
    fn n_states(&self) -> usize {
        0
    }

    fn state_names(&self) -> &[&'static str] {
        &NO_STATES
    }

    fn norton_admittance(&self) -> Option<Complex<f64>> {
        // None. Stamping the constant-impedance part into Y and subtracting it
        // back out would be exactly equivalent and one more thing to keep
        // consistent; the whole load is carried in the injection.
        None
    }

    fn derivatives(&self, _x: &[f64], _v: Complex<f64>, _out: &mut [f64]) {}

    fn injection(&self, _x: &[f64], v: Complex<f64>) -> Complex<f64> {
        let (w, _) = self.weights(v.norm());
        (0..3).map(|k| self.s_conj[k] * v * w[k]).sum()
    }

    fn jacobian(&self, _x: &[f64], v: Complex<f64>, out: &mut ModelJacobian) {
        let v_mag = v.norm();
        let (w, dw) = self.weights(v_mag);
        // ∂|V|/∂V, undefined at the origin; the cutoff means the weights are
        // constant there anyway, so zero is both finite and correct.
        let (dm_re, dm_im) =
            if v_mag > 0.0 { (v.re / v_mag, v.im / v_mag) } else { (0.0, 0.0) };

        let mut d_re = Complex::new(0.0, 0.0);
        let mut d_im = Complex::new(0.0, 0.0);
        for k in 0..3 {
            let c = self.s_conj[k];
            // I_k = c·V·w(|V|), so ∂I/∂v = c·(∂V/∂v·w + V·w'·∂|V|/∂v).
            d_re += c * (Complex::new(w[k], 0.0) + v * dw[k] * dm_re);
            d_im += c * (Complex::new(0.0, w[k]) + v * dw[k] * dm_im);
        }
        out.didv = [d_re.re, d_im.re, d_re.im, d_im.im];
    }

    fn initialize(&mut self, v: Complex<f64>, s: Complex<f64>) -> Result<Vec<f64>, InitError> {
        let v_mag = v.norm();
        if v_mag == 0.0 {
            return Err(InitError::ZeroTerminalVoltage);
        }
        self.v0 = v_mag;
        // Split the operating-point injection by the stated fractions, and
        // store each part already conjugated — the form every use of it wants.
        for k in 0..3 {
            self.s_conj[k] = (s * self.fractions[k]).conj();
        }
        Ok(Vec::new())
    }
}
