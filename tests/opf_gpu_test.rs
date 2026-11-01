//! The AC-OPF model on the GPU (`AcOpf::device_model`) against the CPU's
//! own evaluation, on PGLib cases in both formulations: the same
//! constraints, Jacobian and Hessian to rounding (only the device's sines
//! and cosines round differently), in the order of the CPU's triplets.

use std::path::Path;

use gridoxide::matpower;
use gridoxide::opf::ac::{AcOpf, AcOpfFormulation, AcOpfOptions};
use gridoxide::opf::nlp::NonlinearProblem;

fn build(name: &str, formulation: AcOpfFormulation) -> AcOpf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/benchmark-grids/pglib")
        .join(format!("{name}.m"));
    let case = matpower::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let opf = matpower::ac_opf_network(&case).unwrap();
    let options = AcOpfOptions {
        formulation,
        ..AcOpfOptions::default()
    };
    AcOpf::build(opf.network, options).unwrap()
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
