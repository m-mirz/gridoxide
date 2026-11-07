//! The AC-OPF model on the GPU (`AcOpf::device_model`) against the CPU's
//! own evaluation, on PGLib cases in both formulations: the same
//! constraints, Jacobian and Hessian to rounding (only the device's sines
//! and cosines round differently), in the order of the CPU's triplets. And
//! the interior point with its vectors on the device
//! (`InteriorPoint::Device`) against the same interior point on the host.

use std::path::Path;

use gridoxide::gpu::cudss_ldlt_solver;
use gridoxide::matpower;
use gridoxide::opf::OptStatus;
use gridoxide::opf::ac::{AcOpf, AcOpfFormulation, AcOpfOptions};
use gridoxide::opf::nlp::{self, InteriorPoint, KktForm, NlpOptions, NonlinearProblem};

fn build(name: &str, formulation: AcOpfFormulation) -> AcOpf {
    build_with(name, formulation, NlpOptions::default())
}

fn build_with(name: &str, formulation: AcOpfFormulation, nlp: NlpOptions) -> AcOpf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/benchmark-grids/pglib")
        .join(format!("{name}.m"));
    let case = matpower::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let opf = matpower::ac_opf_network(&case).unwrap();
    let options = AcOpfOptions {
        formulation,
        nlp,
        ..AcOpfOptions::default()
    };
    AcOpf::build(opf.network, options).unwrap()
}

/// `name` with the model on the device and `nlp` for its interior point, on
/// cuDSS in its deterministic mode.
fn on_device(name: &str, formulation: AcOpfFormulation, nlp: NlpOptions) -> nlp::NlpSolution {
    let nlp = NlpOptions {
        kkt_solver: cudss_ldlt_solver,
        ..nlp
    };
    let mut opf = build_with(name, formulation, nlp);
    opf.evaluate_on_device().unwrap();
    nlp::solve(&opf, &nlp).unwrap()
}

const DEVICE: InteriorPoint = InteriorPoint::Device {
    deterministic: true,
};

/// Both solved, near the same iterations, to the same point: the paths part
/// only by the order of a sum.
fn assert_same_solve(device: &nlp::NlpSolution, host: &nlp::NlpSolution, what: &str) {
    assert_eq!(device.status, OptStatus::Optimal, "{what}");
    assert_eq!(host.status, OptStatus::Optimal, "{what}");
    assert!(
        device.iterations.abs_diff(host.iterations) <= 2,
        "{what}: {} iterations against {}",
        device.iterations,
        host.iterations
    );
    assert!(
        (device.objective - host.objective).abs() <= 1e-8 * host.objective.abs(),
        "{what}: objective {} against {}",
        device.objective,
        host.objective
    );
    assert!(gap(&device.x, &host.x) < 1e-5, "{what}: x");
    assert!(device.violation < 1e-7, "{what}: {}", device.violation);
}

/// Largest difference relative to the largest value.
fn gap(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let size = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
        / size
}

#[test]
fn the_device_model_evaluates_as_the_cpus() {
    for name in ["pglib_opf_case118_ieee", "pglib_opf_case1354_pegase"] {
        for formulation in [AcOpfFormulation::BusInjection, AcOpfFormulation::BranchFlow] {
            let opf = build(name, formulation);
            // A point off the start and multipliers of both signs, so every
            // term counts.
            let x: Vec<f64> = opf
                .initial_point()
                .iter()
                .enumerate()
                .map(|(k, v)| v + 0.05 * (0.37 * k as f64).sin())
                .collect();
            let y: Vec<f64> = (0..opf.n_constraints())
                .map(|r| (0.71 * r as f64).cos())
                .collect();
            let mut device = opf.device_model().unwrap();
            let what = format!("{name}, {formulation:?}");

            let c = device.constraints(&x).unwrap();
            assert!(gap(&c, &opf.constraints(&x)) < 1e-14, "constraints, {what}");

            let j: Vec<f64> = opf.jacobian(&x).iter().map(|e| e.2).collect();
            assert!(
                gap(&device.jacobian_values(&x).unwrap(), &j) < 1e-14,
                "Jacobian, {what}"
            );

            let h: Vec<f64> = opf.lagrangian_hessian(&x, &y).iter().map(|e| e.2).collect();
            assert!(
                gap(&device.hessian_values(&x, &y).unwrap(), &h) < 1e-14,
                "Hessian, {what}"
            );
        }
    }
}

#[test]
fn the_interior_point_on_the_device_solves_as_on_the_host() {
    for name in [
        "pglib_opf_case3_lmbd",
        // Piecewise costs: free epigraph variables, and rows with a lower
        // bound alone.
        "pglib_opf_case5_pjm",
        "pglib_opf_case14_ieee",
        "pglib_opf_case30_ieee",
        "pglib_opf_case118_ieee",
        "pglib_opf_case300_ieee",
    ] {
        for formulation in [AcOpfFormulation::BusInjection, AcOpfFormulation::BranchFlow] {
            let device = on_device(
                name,
                formulation,
                NlpOptions {
                    interior_point: DEVICE,
                    ..NlpOptions::default()
                },
            );
            let host = on_device(name, formulation, NlpOptions::default());
            assert_same_solve(&device, &host, &format!("{name}, {formulation:?}"));
        }
    }
}

/// Without `Scaled`, the device takes the model's values as they are.
#[test]
fn the_interior_point_on_the_device_solves_unscaled() {
    let options = NlpOptions {
        scaling: false,
        ..NlpOptions::default()
    };
    let device = on_device(
        "pglib_opf_case14_ieee",
        AcOpfFormulation::BusInjection,
        NlpOptions {
            interior_point: DEVICE,
            ..options
        },
    );
    let host = on_device(
        "pglib_opf_case14_ieee",
        AcOpfFormulation::BusInjection,
        options,
    );
    assert_same_solve(&device, &host, "unscaled");
}

/// What the device does not run stays on the host: the same solve as without
/// asking for it.
#[test]
fn hykkt_and_ncl_stay_on_the_host() {
    let name = "pglib_opf_case14_ieee";
    let hybrid = NlpOptions {
        kkt: KktForm::Hybrid { gamma: 1e8 },
        kkt_solver: nlp::llt_solver,
        ..NlpOptions::default()
    };
    let ncl = NlpOptions {
        ncl: true,
        ..NlpOptions::default()
    };
    for options in [hybrid, ncl] {
        let solve = |interior_point| {
            let nlp = NlpOptions {
                interior_point,
                ..options
            };
            let mut opf = build_with(name, AcOpfFormulation::BusInjection, nlp);
            opf.evaluate_on_device().unwrap();
            nlp::solve(&opf, &nlp).unwrap()
        };
        let (device, host) = (solve(DEVICE), solve(InteriorPoint::Host));
        assert_eq!(device.iterations, host.iterations);
        assert_eq!(device.objective, host.objective);
    }
}

/// A problem without a model on the device stays on the host too.
#[test]
fn a_problem_off_the_device_stays_on_the_host() {
    let nlp = NlpOptions {
        interior_point: DEVICE,
        ..NlpOptions::default()
    };
    let opf = build_with("pglib_opf_case14_ieee", AcOpfFormulation::BusInjection, nlp);
    assert_eq!(nlp::solve(&opf, &nlp).unwrap().status, OptStatus::Optimal);
}
