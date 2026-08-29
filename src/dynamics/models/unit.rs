//! [`GeneratingUnit`]: a machine and its controls as one device.
//!
//! # Why they are one device and not four
//!
//! An exciter writes `E_fd` into its machine, a governor writes `P_m`, a
//! stabilizer writes `v_s` into the exciter, and all three read a machine state
//! or the terminal voltage. As separate devices those couplings would be
//! cross-device Jacobian entries — real, dense-ish, and a second sparsity
//! problem on top of the network's, for a set of blocks that are physically
//! one machine.
//!
//! Composed here they are ordinary partial derivatives inside one diagonal
//! block, and [`DaePattern`](crate::dynamics::dae::DaePattern) sees one block
//! per unit.
//!
//! # The signal graph
//!
//! ```text
//!   Δω ──► PSS ────► v_s ──┐
//!                          ├──► AVR ──► E_fd ──┐
//!   |V| ────────────(−)────┘                   │
//!                                              ├──► machine ──► I, δ, ω
//!   Δω ──► governor ─────────────► P_m ────────┤
//!   V   ───────────────────────────────────────┘
//! ```
//!
//! It is acyclic, and no block's output depends on its own input through
//! another block. That is what makes one forward pass enough to evaluate and
//! one sweep of the chain rule enough to differentiate: there is no algebraic
//! loop to iterate.
//!
//! The governor's direct feedthrough from `Δω` to `P_m` is not a loop either.
//! `P_m` enters `ω̇`, and `ω̇` does not enter `P_m` — only `ω` does.
//!
//! # State layout
//!
//! `[machine | avr | governor | pss]`, each part's states contiguous and in
//! that order. The order is fixed rather than sorted so that a trajectory's
//! column names stay stable when a control is added to a unit.
//!
//! # What is latched, and what is read
//!
//! A unit with no governor holds `P_m` at whatever the power flow implied; a
//! unit with no exciter holds `E_fd` likewise. A unit *with* them latches each
//! control's own reference — `V_ref`, `P_ref` — so that its output at `t = 0`
//! is exactly that same value. Nothing reads a setpoint from a file: the file's
//! stated `V_ref` and its stated power flow are two independent claims and will
//! not agree to machine precision. See [`init`](crate::dynamics::init).

use num_complex::Complex;

use super::machine::{Machine, MachineJacobian};
use super::{Control, DynamicModel, InitError, ModelJacobian};

/// A machine, optionally with an exciter, a governor and a stabilizer.
#[derive(Debug)]
pub struct GeneratingUnit {
    machine: Box<dyn Machine>,
    avr: Option<Box<dyn Control>>,
    gov: Option<Box<dyn Control>>,
    pss: Option<Box<dyn Control>>,
    /// Field voltage held constant when there is no exciter.
    e_fd0: f64,
    /// Mechanical power held constant when there is no governor.
    p_m0: f64,
    names: Vec<&'static str>,
    n_m: usize,
    n_a: usize,
    n_g: usize,
    n_p: usize,
    off_a: usize,
    off_g: usize,
    off_p: usize,
    n_states: usize,
    omega_col: usize,
    connected: bool,
    scratch: std::cell::RefCell<Scratch>,
}

/// Per-call buffers, sized once so a Jacobian evaluation allocates nothing.
#[derive(Debug)]
struct Scratch {
    machine: MachineJacobian,
    avr_dfdx: Vec<f64>,
    avr_dfdu: Vec<f64>,
    avr_dydx: Vec<f64>,
    gov_dfdx: Vec<f64>,
    gov_dfdu: Vec<f64>,
    gov_dydx: Vec<f64>,
    pss_dfdx: Vec<f64>,
    pss_dfdu: Vec<f64>,
    pss_dydx: Vec<f64>,
}

/// The signals one forward pass over the graph produces.
struct Signals {
    d_omega: f64,
    v_t: f64,
    /// The exciter's input, `v_s − |V|`. Its own `V_ref` is latched inside it.
    u_avr: f64,
    e_fd: f64,
    p_m: f64,
}

impl GeneratingUnit {
    /// A machine with no controls: constant field voltage and constant
    /// mechanical power, both latched from the operating point.
    pub fn machine_only(machine: Box<dyn Machine>) -> Self {
        Self::new(machine, None, None, None)
    }

    pub fn new(
        machine: Box<dyn Machine>,
        avr: Option<Box<dyn Control>>,
        gov: Option<Box<dyn Control>>,
        pss: Option<Box<dyn Control>>,
    ) -> Self {
        let n_m = machine.n_states();
        let n_a = avr.as_ref().map_or(0, |c| c.n_states());
        let n_g = gov.as_ref().map_or(0, |c| c.n_states());
        let n_p = pss.as_ref().map_or(0, |c| c.n_states());

        let mut names: Vec<&'static str> = machine.state_names().to_vec();
        for control in [&avr, &gov, &pss].into_iter().flatten() {
            names.extend_from_slice(control.state_names());
        }

        let omega_col = machine.omega_index();
        Self {
            machine,
            avr,
            gov,
            pss,
            e_fd0: 0.0,
            p_m0: 0.0,
            names,
            n_m,
            n_a,
            n_g,
            n_p,
            off_a: n_m,
            off_g: n_m + n_a,
            off_p: n_m + n_a + n_g,
            n_states: n_m + n_a + n_g + n_p,
            omega_col,
            connected: true,
            scratch: std::cell::RefCell::new(Scratch {
                machine: MachineJacobian::zeros(n_m),
                avr_dfdx: vec![0.0; n_a * n_a],
                avr_dfdu: vec![0.0; n_a],
                avr_dydx: vec![0.0; n_a],
                gov_dfdx: vec![0.0; n_g * n_g],
                gov_dfdu: vec![0.0; n_g],
                gov_dydx: vec![0.0; n_g],
                pss_dfdx: vec![0.0; n_p * n_p],
                pss_dfdu: vec![0.0; n_p],
                pss_dydx: vec![0.0; n_p],
            }),
        }
    }

    /// One forward pass over the signal graph. Acyclic, so one pass suffices.
    fn signals(&self, x: &[f64], v: Complex<f64>) -> Signals {
        let d_omega = x[self.omega_col] - 1.0;
        let v_t = v.norm();

        let v_s = match &self.pss {
            Some(c) => c.output(&x[self.off_p..self.off_p + self.n_p], d_omega),
            None => 0.0,
        };
        let u_avr = v_s - v_t;
        let e_fd = match &self.avr {
            Some(c) => c.output(&x[self.off_a..self.off_a + self.n_a], u_avr),
            None => self.e_fd0,
        };
        let p_m = match &self.gov {
            Some(c) => c.output(&x[self.off_g..self.off_g + self.n_g], d_omega),
            None => self.p_m0,
        };
        Signals { d_omega, v_t, u_avr, e_fd, p_m }
    }

    /// The machine, for a caller that needs a model-specific accessor — an
    /// analytic gate reading a latched internal EMF, say.
    pub fn machine(&self) -> &dyn Machine {
        self.machine.as_ref()
    }
}

impl DynamicModel for GeneratingUnit {
    fn n_states(&self) -> usize {
        self.n_states
    }

    fn state_names(&self) -> &[&'static str] {
        &self.names
    }

    fn norton_admittance(&self) -> Option<Complex<f64>> {
        self.connected.then(|| self.machine.norton_admittance())
    }

    fn set_connected(&mut self, connected: bool) {
        self.connected = connected;
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn derivatives(&self, x: &[f64], v: Complex<f64>, out: &mut [f64]) {
        if !self.connected {
            out.fill(0.0);
            return;
        }
        let sig = self.signals(x, v);
        self.machine.derivatives(&x[..self.n_m], v, sig.e_fd, sig.p_m, &mut out[..self.n_m]);
        if let Some(c) = &self.avr {
            let (o, n) = (self.off_a, self.n_a);
            c.derivatives(&x[o..o + n], sig.u_avr, &mut out[o..o + n]);
        }
        if let Some(c) = &self.gov {
            let (o, n) = (self.off_g, self.n_g);
            c.derivatives(&x[o..o + n], sig.d_omega, &mut out[o..o + n]);
        }
        if let Some(c) = &self.pss {
            let (o, n) = (self.off_p, self.n_p);
            c.derivatives(&x[o..o + n], sig.d_omega, &mut out[o..o + n]);
        }
    }

    fn injection(&self, x: &[f64], v: Complex<f64>) -> Complex<f64> {
        if !self.connected {
            return Complex::new(0.0, 0.0);
        }
        self.machine.injection(&x[..self.n_m], v)
    }

    fn jacobian(&self, x: &[f64], v: Complex<f64>, out: &mut ModelJacobian) {
        if !self.connected {
            // Everything is zero, which the caller already cleared it to. The
            // implicit rule then contributes the identity on the diagonal, so
            // the frozen rows read `x₁ − x₀ = 0` and the states hold exactly.
            return;
        }
        let n = self.n_states;
        let sig = self.signals(x, v);
        let mut s = self.scratch.borrow_mut();

        s.machine.clear();
        self.machine.jacobian(&x[..self.n_m], v, sig.e_fd, sig.p_m, &mut s.machine);

        // ∂|V|/∂V is the unit vector along V. At |V| = 0 it is undefined; zero
        // is the only finite choice, and a machine terminal at exactly zero
        // volts is already outside what any of these models describe.
        let (dvt_re, dvt_im) =
            if sig.v_t > 0.0 { (v.re / sig.v_t, v.im / sig.v_t) } else { (0.0, 0.0) };

        // --- each control's own blocks and output sensitivities ---
        let mut dvs_du = 0.0;
        if let Some(c) = &self.pss {
            let xs = &x[self.off_p..self.off_p + self.n_p];
            s.pss_dfdx.fill(0.0);
            s.pss_dfdu.fill(0.0);
            s.pss_dydx.fill(0.0);
            let (mut fx, mut fu) = (std::mem::take(&mut s.pss_dfdx), std::mem::take(&mut s.pss_dfdu));
            c.jacobian(xs, sig.d_omega, &mut fx, &mut fu);
            s.pss_dfdx = fx;
            s.pss_dfdu = fu;
            let mut yx = std::mem::take(&mut s.pss_dydx);
            dvs_du = c.output_jacobian(xs, sig.d_omega, &mut yx);
            s.pss_dydx = yx;
        }
        let mut defd_du = 0.0;
        if let Some(c) = &self.avr {
            let xs = &x[self.off_a..self.off_a + self.n_a];
            s.avr_dfdx.fill(0.0);
            s.avr_dfdu.fill(0.0);
            s.avr_dydx.fill(0.0);
            let (mut fx, mut fu) = (std::mem::take(&mut s.avr_dfdx), std::mem::take(&mut s.avr_dfdu));
            c.jacobian(xs, sig.u_avr, &mut fx, &mut fu);
            s.avr_dfdx = fx;
            s.avr_dfdu = fu;
            let mut yx = std::mem::take(&mut s.avr_dydx);
            defd_du = c.output_jacobian(xs, sig.u_avr, &mut yx);
            s.avr_dydx = yx;
        }
        let mut dpm_du = 0.0;
        if let Some(c) = &self.gov {
            let xs = &x[self.off_g..self.off_g + self.n_g];
            s.gov_dfdx.fill(0.0);
            s.gov_dfdu.fill(0.0);
            s.gov_dydx.fill(0.0);
            let (mut fx, mut fu) = (std::mem::take(&mut s.gov_dfdx), std::mem::take(&mut s.gov_dfdu));
            c.jacobian(xs, sig.d_omega, &mut fx, &mut fu);
            s.gov_dfdx = fx;
            s.gov_dfdu = fu;
            let mut yx = std::mem::take(&mut s.gov_dydx);
            dpm_du = c.output_jacobian(xs, sig.d_omega, &mut yx);
            s.gov_dydx = yx;
        }

        // The exciter's input is v_s − |V|, so E_fd reaches ω through the
        // stabilizer and V through the terminal magnitude.
        let defd_dw = defd_du * dvs_du;

        // --- machine rows ---
        for i in 0..self.n_m {
            for j in 0..self.n_m {
                out.dfdx[i * n + j] = s.machine.dfdx[i * self.n_m + j];
            }
            let (fe, fp) = (s.machine.dfde[i], s.machine.dfdp[i]);
            out.dfdx[i * n + self.omega_col] += fe * defd_dw + fp * dpm_du;
            for j in 0..self.n_a {
                out.dfdx[i * n + self.off_a + j] = fe * s.avr_dydx[j];
            }
            for j in 0..self.n_g {
                out.dfdx[i * n + self.off_g + j] = fp * s.gov_dydx[j];
            }
            for j in 0..self.n_p {
                out.dfdx[i * n + self.off_p + j] = fe * defd_du * s.pss_dydx[j];
            }
            out.dfdv[i * 2] = s.machine.dfdv[i * 2] - fe * defd_du * dvt_re;
            out.dfdv[i * 2 + 1] = s.machine.dfdv[i * 2 + 1] - fe * defd_du * dvt_im;
        }

        // --- exciter rows ---
        for i in 0..self.n_a {
            let row = self.off_a + i;
            for j in 0..self.n_a {
                out.dfdx[row * n + self.off_a + j] = s.avr_dfdx[i * self.n_a + j];
            }
            let fu = s.avr_dfdu[i];
            out.dfdx[row * n + self.omega_col] += fu * dvs_du;
            for j in 0..self.n_p {
                out.dfdx[row * n + self.off_p + j] += fu * s.pss_dydx[j];
            }
            out.dfdv[row * 2] = -fu * dvt_re;
            out.dfdv[row * 2 + 1] = -fu * dvt_im;
        }

        // --- governor rows ---
        for i in 0..self.n_g {
            let row = self.off_g + i;
            for j in 0..self.n_g {
                out.dfdx[row * n + self.off_g + j] = s.gov_dfdx[i * self.n_g + j];
            }
            out.dfdx[row * n + self.omega_col] += s.gov_dfdu[i];
        }

        // --- stabilizer rows ---
        for i in 0..self.n_p {
            let row = self.off_p + i;
            for j in 0..self.n_p {
                out.dfdx[row * n + self.off_p + j] = s.pss_dfdx[i * self.n_p + j];
            }
            out.dfdx[row * n + self.omega_col] += s.pss_dfdu[i];
        }

        // --- the network coupling: only the machine carries current ---
        for j in 0..self.n_m {
            out.didx[j] = s.machine.didx[j];
            out.didx[n + j] = s.machine.didx[self.n_m + j];
        }
        out.didv = s.machine.didv;
    }

    fn latch(&self, x: &[f64], v: Complex<f64>) {
        let sig = self.signals(x, v);
        if let Some(c) = &self.avr {
            c.latch(&x[self.off_a..self.off_a + self.n_a], sig.u_avr);
        }
        if let Some(c) = &self.gov {
            c.latch(&x[self.off_g..self.off_g + self.n_g], sig.d_omega);
        }
        if let Some(c) = &self.pss {
            c.latch(&x[self.off_p..self.off_p + self.n_p], sig.d_omega);
        }
    }

    fn project(&self, x: &mut [f64]) {
        if let Some(c) = &self.avr {
            c.project(&mut x[self.off_a..self.off_a + self.n_a]);
        }
        if let Some(c) = &self.gov {
            c.project(&mut x[self.off_g..self.off_g + self.n_g]);
        }
        if let Some(c) = &self.pss {
            c.project(&mut x[self.off_p..self.off_p + self.n_p]);
        }
    }

    fn initialize(&mut self, v: Complex<f64>, s: Complex<f64>) -> Result<Vec<f64>, InitError> {
        let init = self.machine.initialize(v, s)?;
        self.e_fd0 = init.e_fd;
        self.p_m0 = init.p_m;
        let mut x = init.states;

        // Each control is initialized to *reproduce* what the machine turned
        // out to need, which is what fixes its reference. The order follows the
        // signal graph: the stabilizer's output is an input to the exciter, and
        // it is zero at equilibrium by construction — every stabilizer here
        // begins with a washout, which passes nothing at steady state — so the
        // exciter's input is known once the terminal voltage is.
        let mut pss_states = Vec::new();
        if let Some(c) = &mut self.pss {
            pss_states = c.initialize(0.0, 0.0)?;
        }
        let mut avr_states = Vec::new();
        if let Some(c) = &mut self.avr {
            avr_states = c.initialize(init.e_fd, -v.norm())?;
        }
        let mut gov_states = Vec::new();
        if let Some(c) = &mut self.gov {
            gov_states = c.initialize(init.p_m, 0.0)?;
        }
        x.extend_from_slice(&avr_states);
        x.extend_from_slice(&gov_states);
        x.extend_from_slice(&pss_states);
        Ok(x)
    }
}
