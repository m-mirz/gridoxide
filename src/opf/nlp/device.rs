//! [`Engine`] on the GPU: the interior point's vectors stay on the device
//! (`gridoxide_gpu::ip`), its decisions in [`drive`], the host's.
//!
//! The objective and its gradient stay the host's: they are cheap, and
//! general where the model's kernels are not, so `x` crosses the bus once an
//! evaluation and the gradient back. The restoration phase runs on
//! [`HostEngine`], as NCL and HyKKT do: it is rare, and its problem is not
//! the device model's. [`drive`] takes the iterate off the device for it and
//! puts the restored point back.

use std::sync::Mutex;

use gridoxide_gpu::LinearError;
use gridoxide_gpu::ac::DeviceAcModel;
use gridoxide_gpu::ip::{DeviceInteriorPoint, InteriorPointData};

use super::{
    Barrier, CONSTR_MULT_INIT_MAX, Control, Direction, Engine, Internal, NlpOptions, NlpSolution,
    NonlinearProblem, Residuals, SearchStart, State, conclude, drive,
};
use crate::opf::OpfError;

/// A problem's model on the device, and the patterns its values fill;
/// [`NonlinearProblem::on_device`].
#[derive(Clone, Copy)]
pub struct DeviceProblem<'a> {
    pub model: &'a Mutex<DeviceAcModel>,
    pub jacobian: &'a [(usize, usize)],
    pub hessian: &'a [(usize, usize)],
}

fn backend(e: LinearError) -> OpfError {
    OpfError::Backend(e.to_string())
}

fn correction(direction: Direction) -> bool {
    direction == Direction::Correction
}

struct DeviceEngine<'a> {
    ip: DeviceInteriorPoint,
    problem: &'a dyn NonlinearProblem,
    model: &'a Mutex<DeviceAcModel>,
    options: &'a NlpOptions,
}

/// Solves `problem` with its vectors on the device. `scale` is the row and
/// objective scale of a `Scaled` problem, which the device applies itself to
/// the model's values.
pub(super) fn solve(
    problem: &dyn NonlinearProblem,
    device: DeviceProblem<'_>,
    scale: Option<(&[f64], f64)>,
    options: &NlpOptions,
    deterministic: bool,
) -> Result<NlpSolution, OpfError> {
    let p = Internal::build(problem);
    let mut state = State::start(problem, &p, options);
    let mut ip = interior_point(problem, &p, device, scale, deterministic)?;
    ip.set_iterate(&state.u, &state.y, &state.z_l, &state.z_u)
        .map_err(backend)?;
    let mut engine = DeviceEngine {
        ip,
        problem,
        model: device.model,
        options,
    };
    let status = drive(
        &mut engine,
        problem,
        &p,
        &mut state,
        options,
        Barrier::Schedule,
        &Control::SOLVE,
    )?;
    (state.u, state.y, state.z_l, state.z_u) = engine.ip.iterate().map_err(backend)?;
    Ok(conclude(problem, &p, &state, status))
}

/// The device's half of `problem` in the internal form `p`.
fn interior_point(
    problem: &dyn NonlinearProblem,
    p: &Internal,
    device: DeviceProblem<'_>,
    scale: Option<(&[f64], f64)>,
    deterministic: bool,
) -> Result<DeviceInteriorPoint, OpfError> {
    let rows = problem.n_constraints();
    let base: Vec<(usize, usize, f64)> =
        device.jacobian.iter().map(|&(r, c)| (r, c, 0.0)).collect();
    let internal = p.jacobian(&base);
    let data = InteriorPointData {
        n_x: p.n_x,
        rows,
        lower: p.lower.clone(),
        upper: p.upper.clone(),
        c_lower: p.c_lower.clone(),
        slack_of: p.slack_of.clone(),
        pinned: p.pinned.iter().map(|&(_, k, v)| (k, v)).collect(),
        jacobian: internal.iter().map(|&(r, c, _)| (r, c)).collect(),
        n_jacobian: base.len(),
        constants: internal[base.len()..].iter().map(|e| e.2).collect(),
        hessian: device.hessian.to_vec(),
        row_scale: scale.map_or_else(|| vec![1.0; rows], |(rows, _)| rows.to_vec()),
        objective_scale: scale.map_or(1.0, |(_, objective)| objective),
        deterministic,
    };
    DeviceInteriorPoint::new(&data, &device.model.lock().unwrap()).map_err(backend)
}

impl Engine for DeviceEngine<'_> {
    fn evaluate(&mut self) -> Result<f64, OpfError> {
        let x = self.ip.x().map_err(backend)?;
        let f = self.problem.objective(x);
        let grad = self.problem.gradient(x);
        let model = self.model.lock().unwrap();
        self.ip.evaluate(&model, &grad).map_err(backend)?;
        Ok(f)
    }

    fn start_multipliers(&mut self) -> Result<(), OpfError> {
        self.ip
            .start_multipliers(self.options.dual_regularization, CONSTR_MULT_INIT_MAX)
            .map_err(backend)
    }

    fn residuals(&mut self) -> Result<Residuals, OpfError> {
        let [dual, primal, theta, y_sum, z_l, z_u] = self.ip.residuals().map_err(backend)?;
        Ok(Residuals {
            dual,
            primal,
            theta,
            y_sum,
            z_sum: z_l + z_u,
        })
    }

    fn complementarity(&mut self, target: f64) -> Result<f64, OpfError> {
        self.ip.complementarity(target).map_err(backend)
    }

    fn prepare(&mut self) -> Result<(), OpfError> {
        let model = self.model.lock().unwrap();
        self.ip.prepare(&model).map_err(backend)
    }

    fn factorize(&mut self, dw: f64) -> Result<bool, OpfError> {
        let inertia = self
            .ip
            .factorize(dw, self.options.dual_regularization)
            .map_err(backend)?;
        Ok(inertia == Some(self.ip.minimizing()))
    }

    fn direction(&mut self, mu: f64) -> Result<bool, OpfError> {
        self.ip
            .direction(mu, self.options.refinement_rounds)
            .map_err(backend)
    }

    fn search_start(&mut self, f: f64, mu: f64, tau: f64) -> Result<SearchStart, OpfError> {
        let [barrier, dphi, blocking, step, point] = self.ip.search_start(mu).map_err(backend)?;
        Ok(SearchStart {
            phi: f + barrier,
            dphi,
            alpha_max: (tau * blocking).min(1.0),
            step,
            point,
        })
    }

    fn trial(
        &mut self,
        direction: Direction,
        alpha: f64,
        mu: f64,
    ) -> Result<Option<(f64, f64)>, OpfError> {
        let model = self.model.lock().unwrap();
        let trial = self
            .ip
            .trial(&model, correction(direction), alpha, mu)
            .map_err(backend)?;
        let phi = self.problem.objective(trial.x) + trial.barrier;
        Ok((phi.is_finite() && trial.theta.is_finite()).then_some((trial.theta, phi)))
    }

    fn start_correction(&mut self) -> Result<(), OpfError> {
        self.ip.start_correction().map_err(backend)
    }

    fn correct(&mut self, alpha: f64, tau: f64) -> Result<Option<f64>, OpfError> {
        let corrected = self
            .ip
            .correct(alpha, self.options.refinement_rounds)
            .map_err(backend)?;
        // As `HostEngine::correct`: only an accurate solve is worth a trial.
        Ok(corrected
            .filter(|&[residual, _]| residual <= 1e-5)
            .map(|[_, blocking]| (tau * blocking).min(1.0)))
    }

    fn save(&mut self) -> Result<(), OpfError> {
        self.ip.save().map_err(backend)
    }

    fn restore(&mut self) -> Result<(), OpfError> {
        self.ip.restore().map_err(backend)
    }

    fn multiplier_step(
        &mut self,
        direction: Direction,
        mu: f64,
        tau: f64,
    ) -> Result<f64, OpfError> {
        let blocking = self
            .ip
            .multiplier_step(correction(direction), mu, tau)
            .map_err(backend)?;
        Ok(blocking.min(1.0))
    }

    fn step_norm(&mut self, direction: Direction) -> Result<f64, OpfError> {
        self.ip.step_norm(correction(direction)).map_err(backend)
    }

    fn update(
        &mut self,
        direction: Direction,
        alpha: f64,
        alpha_z: f64,
        mu: f64,
    ) -> Result<(), OpfError> {
        self.ip
            .update(correction(direction), alpha, alpha_z, mu)
            .map_err(backend)
    }

    fn point(&mut self) -> Result<Vec<f64>, OpfError> {
        self.ip.point().map_err(backend)
    }

    fn restored(&mut self, u: &[f64]) -> Result<(), OpfError> {
        self.ip
            .restored(u, super::BOUND_MULT_RESET_THRESHOLD)
            .map_err(backend)
    }
}

#[cfg(all(test, feature = "matpower"))]
mod tests {
    use super::*;
    use crate::matpower;
    use crate::opf::ac::{AcOpf, AcOpfOptions};

    /// The watchdog's way back, which no PGLib run here has taken on the
    /// device: every vector of the saved point returns, each to its own place.
    #[test]
    fn restore_returns_the_saved_point() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/benchmark-grids/pglib/pglib_opf_case14_ieee.m"
        );
        let case = matpower::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
        let network = matpower::ac_opf_network(&case).unwrap().network;
        let mut opf = AcOpf::build(network, AcOpfOptions::default()).unwrap();
        opf.evaluate_on_device().unwrap();
        let device = opf.on_device().unwrap();
        let p = Internal::build(&opf);
        let mut ip = interior_point(&opf, &p, device, None, true).unwrap();

        let vector =
            |len: usize, start: f64| -> Vec<f64> { (0..len).map(|k| start + k as f64).collect() };
        let (n, m) = (p.n_total, p.m_eq);
        let saved = (
            vector(n, 1.0),
            vector(m, 1e3),
            vector(n, 1e4),
            vector(n, 1e5),
        );
        ip.set_iterate(&saved.0, &saved.1, &saved.2, &saved.3)
            .unwrap();
        ip.save().unwrap();
        let other = vector(n.max(m), -1.0);
        ip.set_iterate(&other[..n], &other[..m], &other[..n], &other[..n])
            .unwrap();
        assert_ne!(ip.iterate().unwrap().0, saved.0);
        ip.restore().unwrap();
        assert_eq!(ip.iterate().unwrap(), saved);
    }
}
