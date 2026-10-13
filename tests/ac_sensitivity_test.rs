//! AC sensitivities, checked against the thing they claim to predict.
//!
//! Unlike the DC case, a finite difference here is *not* exact: the AC problem
//! is nonlinear, so a re-solve at a finite step differs from the derivative by
//! O(h). A **central** difference makes that error O(h²), which at h = 1e-5 on
//! a well-conditioned case leaves ~1e-8 — comfortably below the tolerances
//! asserted here, and far below any error a sign or indexing mistake would
//! produce.
//!
//! The oracle borrows nothing from `ac_sensitivity`: it perturbs an injection,
//! runs the ordinary Newton solver to convergence, and reads the answer off the
//! solved state. A shared error cannot hide in both.

use std::path::PathBuf;

use num_complex::Complex;

use gridoxide::ac_sensitivity::{AcSensitivity, Function, Variable};
use gridoxide::branch_flow::{branch_params, bus_voltages, terminal_flow, Terminal};
use gridoxide::network::build_ybus;
use gridoxide::pgm::pgm_to_buses_and_branches;
use gridoxide::run_power_flow;
use gridoxide::solver::{PowerFlowOptions, SolveStatus};
use gridoxide::types::{Bus, BusType, Line, Transformer};

mod common;

/// Step and solve tolerance for the central differences. These two are chosen
/// together, because they trade against each other:
///
/// - the solve's residual noise is divided by `2H`, so a *smaller* step
///   amplifies it — at `SOLVE_TOL = 1e-10` and `H = 1e-4` that floor is ~5e-8;
/// - the central difference's own truncation error is O(H²), so a *larger*
///   step grows it — at `H = 1e-4` that is ~1e-8 on a derivative of order one.
///
/// Both land near 1e-7, comfortably under the tolerances asserted below and
/// orders of magnitude under anything a sign or indexing error would produce.
/// `SOLVE_TOL` cannot usefully go tighter: this case's residual floors at
/// ~1.2e-11 in double precision, so asking for 1e-12 simply never converges.
const H: f64 = 1e-4;
const SOLVE_TOL: f64 = 1e-10;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow")
        .join(rel)
        .join("input.json")
}

struct Case {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    transformers: Vec<Transformer>,
}

impl Case {
    fn load(rel: &str) -> Self {
        let input = common::load_pgm_input(&fixture(rel));
        let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
        Self { buses, lines, transformers }
    }

    /// Solves the AC power flow with the given per-bus injection offsets
    /// applied, and returns the converged buses.
    fn solve_with(&self, dp: &[f64], dq: &[f64]) -> Vec<Bus> {
        let mut buses = self.buses.clone();
        for (i, b) in buses.iter_mut().enumerate() {
            b.p_spec += dp[i];
            b.q_spec += dq[i];
        }
        let opts = PowerFlowOptions { tol: SOLVE_TOL, max_iter: 50, ..Default::default() };
        let report = run_power_flow(buses, &self.lines, &self.transformers, &[], gridoxide::TapData::none(), opts);
        assert_eq!(
            report.stats.status,
            SolveStatus::Converged,
            "the oracle's own solve must converge, last mismatch {:?}",
            report.stats.mismatch_history.last()
        );
        report.buses
    }

    fn base(&self) -> Vec<Bus> {
        let zero = vec![0.0; self.buses.len()];
        self.solve_with(&zero, &zero)
    }

    fn sensitivity(&self, base: &[Bus]) -> AcSensitivity {
        let ybus = build_ybus(base.len(), &self.lines, &self.transformers).finish();
        AcSensitivity::new(base, &ybus, &self.lines, &self.transformers)
            .expect("the Jacobian at a converged point should be nonsingular")
    }

    fn branch_flows(&self, buses: &[Bus], terminal: Terminal) -> Vec<(f64, f64)> {
        let v = bus_voltages(buses);
        branch_params(&self.lines, &self.transformers)
            .iter()
            .map(|b| terminal_flow(b, terminal, &v))
            .collect()
    }

    /// A bus that is a genuine free variable for active injection.
    fn a_non_slack_bus(&self) -> usize {
        self.buses
            .iter()
            .find(|b| b.bus_type != BusType::Slack)
            .expect("the fixture must have a non-slack bus")
            .idx
    }

    fn a_pq_bus(&self) -> usize {
        self.buses
            .iter()
            .find(|b| b.bus_type == BusType::PQ)
            .expect("the fixture must have a PQ bus")
            .idx
    }
}

/// Central difference of the solved state with respect to active injection at
/// `bus`.
fn state_finite_difference(case: &Case, bus: usize, reactive: bool) -> (Vec<f64>, Vec<f64>) {
    let n = case.buses.len();
    let mut up = (vec![0.0; n], vec![0.0; n]);
    let mut dn = (vec![0.0; n], vec![0.0; n]);
    if reactive {
        up.1[bus] = H;
        dn.1[bus] = -H;
    } else {
        up.0[bus] = H;
        dn.0[bus] = -H;
    }

    let plus = case.solve_with(&up.0, &up.1);
    let minus = case.solve_with(&dn.0, &dn.1);

    let d_theta = (0..n)
        .map(|i| (plus[i].voltage_ang - minus[i].voltage_ang) / (2.0 * H))
        .collect();
    let d_vmag = (0..n)
        .map(|i| (plus[i].voltage_mag - minus[i].voltage_mag) / (2.0 * H))
        .collect();
    (d_theta, d_vmag)
}

#[track_caller]
fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:.9}, want {want:.9} (diff {:.3e})",
        (got - want).abs()
    );
}

/// The state response to an active injection, against a re-solve. This is the
/// core claim — everything else in the module is a chain rule on top of it.
#[test]
fn state_response_matches_a_finite_difference_of_the_ac_solve() {
    let case = Case::load("symmetric/transmission-case");
    let base = case.base();
    let sens = case.sensitivity(&base);
    let bus = case.a_non_slack_bus();

    let analytic = sens.state_response(Variable::ActiveInjection(bus)).unwrap();
    let (fd_theta, fd_vmag) = state_finite_difference(&case, bus, false);

    for i in 0..case.buses.len() {
        close(analytic.d_theta[i], fd_theta[i], 1e-6, &format!("dtheta[{i}]/dP[{bus}]"));
        close(analytic.d_vmag[i], fd_vmag[i], 1e-6, &format!("dvmag[{i}]/dP[{bus}]"));
    }
    // Not a trivially-zero answer.
    assert!(
        analytic.d_theta.iter().any(|d| d.abs() > 1e-4),
        "the response should be visible: {:?}",
        analytic.d_theta
    );
}

/// The same, for reactive injection — which exercises the `|V|` half of the
/// unknown layout rather than the `θ` half.
#[test]
fn reactive_state_response_matches_a_finite_difference() {
    let case = Case::load("symmetric/transmission-case");
    let base = case.base();
    let sens = case.sensitivity(&base);
    let bus = case.a_pq_bus();

    let analytic = sens.state_response(Variable::ReactiveInjection(bus)).unwrap();
    let (fd_theta, fd_vmag) = state_finite_difference(&case, bus, true);

    for i in 0..case.buses.len() {
        close(analytic.d_theta[i], fd_theta[i], 1e-6, &format!("dtheta[{i}]/dQ[{bus}]"));
        close(analytic.d_vmag[i], fd_vmag[i], 1e-6, &format!("dvmag[{i}]/dQ[{bus}]"));
    }
    assert!(analytic.d_vmag.iter().any(|d| d.abs() > 1e-4));
}

/// Branch flow sensitivities — the chain rule on top of the state response,
/// and the quantity a user actually asks for.
#[test]
fn branch_response_matches_a_finite_difference_of_the_flows() {
    let case = Case::load("symmetric/transmission-case");
    let base = case.base();
    let sens = case.sensitivity(&base);
    let bus = case.a_non_slack_bus();
    let n = case.buses.len();

    for terminal in [Terminal::From, Terminal::To] {
        let analytic = sens.branch_response(Variable::ActiveInjection(bus), terminal).unwrap();

        let mut up = vec![0.0; n];
        let mut dn = vec![0.0; n];
        up[bus] = H;
        dn[bus] = -H;
        let zero = vec![0.0; n];
        let plus = case.branch_flows(&case.solve_with(&up, &zero), terminal);
        let minus = case.branch_flows(&case.solve_with(&dn, &zero), terminal);

        for b in 0..analytic.len() {
            let fd_p = (plus[b].0 - minus[b].0) / (2.0 * H);
            let fd_q = (plus[b].1 - minus[b].1) / (2.0 * H);
            close(analytic[b].0, fd_p, 1e-5, &format!("dP_branch[{b}]/dP[{bus}] ({terminal:?})"));
            close(analytic[b].1, fd_q, 1e-5, &format!("dQ_branch[{b}]/dP[{bus}] ({terminal:?})"));
        }
    }
}

/// The adjoint direction must agree with the forward one exactly — it is the
/// same scalar bracketed the other way, so any disagreement is a transpose or
/// indexing error rather than a modelling choice.
#[test]
fn adjoint_rows_agree_with_forward_columns() {
    let case = Case::load("symmetric/transmission-case");
    let base = case.base();
    let sens = case.sensitivity(&base);
    let n = case.buses.len();

    for branch in 0..sens.n_branches() {
        for terminal in [Terminal::From, Terminal::To] {
            let row = sens
                .function_row(Function::BranchActivePower { branch, terminal })
                .unwrap();
            for bus in 0..n {
                let forward =
                    sens.branch_response(Variable::ActiveInjection(bus), terminal).unwrap()[branch].0;
                close(
                    row.d_active[bus],
                    forward,
                    1e-9,
                    &format!("dP_branch[{branch}]/dP[{bus}] ({terminal:?}) adjoint vs forward"),
                );
            }
        }
    }
}

/// The adjoint's reactive half, against the forward direction's own reactive
/// variable — the pairing the previous test does not reach.
#[test]
fn adjoint_reactive_column_agrees_with_the_forward_direction() {
    let case = Case::load("symmetric/transmission-case");
    let base = case.base();
    let sens = case.sensitivity(&base);

    let row = sens
        .function_row(Function::BranchReactivePower { branch: 0, terminal: Terminal::From })
        .unwrap();
    for bus in 0..case.buses.len() {
        let forward = sens
            .branch_response(Variable::ReactiveInjection(bus), Terminal::From)
            .unwrap()[0]
            .1;
        close(row.d_reactive[bus], forward, 1e-9, &format!("dQ_branch[0]/dQ[{bus}]"));
    }
}

/// A voltage-magnitude function is a unit row, so its adjoint must reproduce
/// the forward state response exactly. This is the cheapest possible check
/// that the two directions share an index space.
#[test]
fn voltage_functions_agree_between_directions() {
    let case = Case::load("symmetric/transmission-case");
    let base = case.base();
    let sens = case.sensitivity(&base);
    let watched = case.a_pq_bus();

    let row = sens.function_row(Function::VoltageMagnitude(watched)).unwrap();
    for bus in 0..case.buses.len() {
        let forward = sens.state_response(Variable::ActiveInjection(bus)).unwrap().d_vmag[watched];
        close(row.d_active[bus], forward, 1e-9, &format!("dV[{watched}]/dP[{bus}]"));
    }

    let row = sens.function_row(Function::VoltageAngle(watched)).unwrap();
    for bus in 0..case.buses.len() {
        let forward = sens.state_response(Variable::ActiveInjection(bus)).unwrap().d_theta[watched];
        close(row.d_active[bus], forward, 1e-9, &format!("dtheta[{watched}]/dP[{bus}]"));
    }
}

/// A slack bus's injection is an output of the solve, not an input, so its
/// column is identically zero — the same contract a DC PTDF column has at its
/// reference bus. Likewise reactive injection where the magnitude is held.
#[test]
fn quantities_the_solve_determines_have_no_sensitivity() {
    let case = Case::load("symmetric/transmission-case");
    let base = case.base();
    let sens = case.sensitivity(&base);

    let slack = base
        .iter()
        .find(|b| b.bus_type == BusType::Slack)
        .expect("a slack bus")
        .idx;

    let response = sens.state_response(Variable::ActiveInjection(slack)).unwrap();
    assert!(response.d_theta.iter().all(|d| *d == 0.0), "{:?}", response.d_theta);
    assert!(response.d_vmag.iter().all(|d| *d == 0.0), "{:?}", response.d_vmag);

    let response = sens.state_response(Variable::ReactiveInjection(slack)).unwrap();
    assert!(response.d_vmag.iter().all(|d| *d == 0.0));

    // And an out-of-range bus is an error, not a silent zero.
    assert!(sens.state_response(Variable::ActiveInjection(base.len())).is_none());
}

/// The same core claim on a structurally different network — the distribution
/// case carries transformers, where `transmission-case` is lines only, so the
/// off-nominal-ratio and phase-shift terms in the Jacobian and in
/// `terminal_flow_derivs` actually participate.
#[test]
fn sensitivities_hold_on_a_network_with_transformers() {
    let case = Case::load("symmetric/distribution-case");
    let base = case.base();
    let sens = case.sensitivity(&base);
    let bus = case.a_pq_bus();
    let n = case.buses.len();

    assert!(
        !case.transformers.is_empty(),
        "this fixture is chosen for its transformers"
    );

    let analytic = sens.state_response(Variable::ActiveInjection(bus)).unwrap();
    let (fd_theta, fd_vmag) = state_finite_difference(&case, bus, false);
    for i in 0..n {
        close(analytic.d_theta[i], fd_theta[i], 1e-5, &format!("dtheta[{i}]/dP[{bus}]"));
        close(analytic.d_vmag[i], fd_vmag[i], 1e-5, &format!("dvmag[{i}]/dP[{bus}]"));
    }

    let analytic = sens.branch_response(Variable::ActiveInjection(bus), Terminal::From).unwrap();
    let mut up = vec![0.0; n];
    let mut dn = vec![0.0; n];
    up[bus] = H;
    dn[bus] = -H;
    let zero = vec![0.0; n];
    let plus = case.branch_flows(&case.solve_with(&up, &zero), Terminal::From);
    let minus = case.branch_flows(&case.solve_with(&dn, &zero), Terminal::From);
    for b in 0..analytic.len() {
        let fd_p = (plus[b].0 - minus[b].0) / (2.0 * H);
        close(analytic[b].0, fd_p, 1e-4, &format!("dP_branch[{b}]/dP[{bus}]"));
    }
}

/// Perturbs one transformer's tap and re-solves, returning the branch flows.
///
/// The perturbation is applied to the `Transformer` itself, so the whole
/// downstream chain — `branch_calc_param`, the Y-bus, the Jacobian — sees it
/// exactly as it would see a real tap change. Nothing here borrows the
/// derivative formulas from `ac_sensitivity`.
fn flows_with_tap(
    case: &Case,
    transformer: usize,
    kind: TapPerturbation,
    delta: f64,
    terminal: Terminal,
) -> (Vec<(f64, f64)>, Vec<Bus>) {
    let mut transformers = case.transformers.clone();
    let tap = transformers[transformer].tap;
    transformers[transformer].tap = match kind {
        TapPerturbation::Ratio => {
            Complex::from_polar(tap.norm() + delta, tap.arg())
        }
        TapPerturbation::Phase => Complex::from_polar(tap.norm(), tap.arg() + delta),
    };

    let mut buses = case.buses.clone();
    let opts = PowerFlowOptions { tol: SOLVE_TOL, max_iter: 50, ..Default::default() };
    let report = run_power_flow(std::mem::take(&mut buses), &case.lines, &transformers, &[], gridoxide::TapData::none(), opts);
    assert_eq!(report.stats.status, SolveStatus::Converged);

    let v = bus_voltages(&report.buses);
    let flows = branch_params(&case.lines, &transformers)
        .iter()
        .map(|b| terminal_flow(b, terminal, &v))
        .collect();
    (flows, report.buses)
}

#[derive(Clone, Copy)]
enum TapPerturbation {
    Ratio,
    Phase,
}

/// Tap-ratio sensitivity against an actual tap change and re-solve.
///
/// The branch being tapped is checked along with every other, and that is the
/// point: its flow carries a direct `∂f/∂p` term that no other branch has, so
/// a chain rule that forgets it passes everywhere except here.
#[test]
fn transformer_ratio_sensitivity_matches_a_re_solve() {
    let case = Case::load("symmetric/distribution-case");
    let base = case.base();
    let sens = case.sensitivity(&base);
    assert!(!case.transformers.is_empty());

    // Flat branch index of the first transformer.
    let branch = case.lines.len();

    for terminal in [Terminal::From, Terminal::To] {
        let analytic = sens
            .branch_response(Variable::TransformerRatio(branch), terminal)
            .unwrap();

        let (plus, _) = flows_with_tap(&case, 0, TapPerturbation::Ratio, H, terminal);
        let (minus, _) = flows_with_tap(&case, 0, TapPerturbation::Ratio, -H, terminal);

        let mut saw_direct_term = false;
        for b in 0..analytic.len() {
            let fd_p = (plus[b].0 - minus[b].0) / (2.0 * H);
            let fd_q = (plus[b].1 - minus[b].1) / (2.0 * H);
            close(analytic[b].0, fd_p, 1e-3, &format!("dP[{b}]/dk ({terminal:?})"));
            close(analytic[b].1, fd_q, 1e-3, &format!("dQ[{b}]/dk ({terminal:?})"));
            if b == branch && analytic[b].0.abs() > 1e-3 {
                saw_direct_term = true;
            }
        }
        assert!(
            saw_direct_term,
            "the tapped branch's own response should be substantial, or this test \
             is not exercising the direct term"
        );
    }
}

/// The same for the phase-shifter angle, which steers active power where the
/// ratio steers reactive.
#[test]
fn phase_shift_sensitivity_matches_a_re_solve() {
    let case = Case::load("symmetric/distribution-case");
    let base = case.base();
    let sens = case.sensitivity(&base);
    let branch = case.lines.len();

    let analytic = sens
        .branch_response(Variable::PhaseShift(branch), Terminal::From)
        .unwrap();
    let (plus, _) = flows_with_tap(&case, 0, TapPerturbation::Phase, H, Terminal::From);
    let (minus, _) = flows_with_tap(&case, 0, TapPerturbation::Phase, -H, Terminal::From);

    for b in 0..analytic.len() {
        let fd_p = (plus[b].0 - minus[b].0) / (2.0 * H);
        let fd_q = (plus[b].1 - minus[b].1) / (2.0 * H);
        close(analytic[b].0, fd_p, 1e-3, &format!("dP[{b}]/dalpha"));
        close(analytic[b].1, fd_q, 1e-3, &format!("dQ[{b}]/dalpha"));
    }
    assert!(analytic.iter().any(|(p, _)| p.abs() > 1e-3), "{analytic:?}");
}

/// The adjoint's tap columns must reproduce the forward direction's, including
/// the direct term on the tapped branch itself.
#[test]
fn adjoint_tap_columns_agree_with_the_forward_direction() {
    let case = Case::load("symmetric/distribution-case");
    let base = case.base();
    let sens = case.sensitivity(&base);

    for watched in 0..sens.n_branches() {
        for terminal in [Terminal::From, Terminal::To] {
            let row = sens
                .function_row(Function::BranchActivePower { branch: watched, terminal })
                .unwrap();
            for branch in 0..sens.n_branches() {
                let ratio = sens
                    .branch_response(Variable::TransformerRatio(branch), terminal)
                    .unwrap()[watched]
                    .0;
                close(
                    row.d_ratio[branch],
                    ratio,
                    1e-9,
                    &format!("dP[{watched}]/dk[{branch}] ({terminal:?})"),
                );
                let phase = sens
                    .branch_response(Variable::PhaseShift(branch), terminal)
                    .unwrap()[watched]
                    .0;
                close(
                    row.d_phase[branch],
                    phase,
                    1e-9,
                    &format!("dP[{watched}]/dalpha[{branch}] ({terminal:?})"),
                );
            }
        }
    }
}

/// A line has no tap, so its tap sensitivities are zero rather than an error —
/// and an out-of-range branch is an error rather than a silent zero.
#[test]
fn lines_have_no_tap_sensitivity() {
    let case = Case::load("symmetric/distribution-case");
    let base = case.base();
    let sens = case.sensitivity(&base);
    assert!(!case.lines.is_empty());

    let response = sens.state_response(Variable::TransformerRatio(0)).unwrap();
    assert!(response.d_theta.iter().all(|d| *d == 0.0));
    assert!(response.d_vmag.iter().all(|d| *d == 0.0));

    assert!(sens.state_response(Variable::PhaseShift(sens.n_branches())).is_none());

    let row = sens
        .function_row(Function::BranchActivePower { branch: 0, terminal: Terminal::From })
        .unwrap();
    for line in 0..case.lines.len() {
        assert_eq!(row.d_ratio[line], 0.0);
        assert_eq!(row.d_phase[line], 0.0);
    }
}
