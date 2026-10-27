//! AC-OPF on a PGLib-OPF case read directly (`matpower::ac_opf_network`).
//!
//! PGLib publishes each case's AC objective, a local optimum to five
//! significant digits, and `pglib/BASELINE.md` lists it. A solve that lands
//! within that rounding with every constraint met reads the case as PGLib's
//! own solvers do: cost, generator rows, limits and the angle-difference rows
//! the `__sad` variants exist for.
//!
//! IPOPT's half needs `--features opf-ipopt`; see `opf_ipopt_test.rs`.

use std::path::{Path, PathBuf};

use gridoxide::matpower::{self, MatpowerOpf};
use gridoxide::opf::OptStatus;
use gridoxide::opf::ac::{AcOpf, AcOpfOptions, AcOpfResult};

/// PGLib v23.07's AC objectives, from `BASELINE.md`.
const PUBLISHED: &[(&str, f64)] = &[
    ("pglib_opf_case14_ieee", 2.1781e+03),
    ("pglib_opf_case14_ieee__api", 5.9994e+03),
    ("pglib_opf_case14_ieee__sad", 2.7768e+03),
    ("pglib_opf_case118_ieee", 9.7214e+04),
    ("pglib_opf_case118_ieee__api", 2.4961e+05),
    ("pglib_opf_case118_ieee__sad", 1.0516e+05),
    ("pglib_opf_case300_ieee", 5.6522e+05),
];

/// Not reached by `opf::nlp` within its iteration limit; IPOPT solves it.
const NLP_UNSOLVED: &[&str] = &["pglib_opf_case300_ieee"];

/// Five significant digits leave half a unit of the fifth: 5e-5 relative.
const GAP: f64 = 1e-4;

fn case_path(name: &str) -> PathBuf {
    let variant = name.rsplit_once("__").map_or("", |(_, v)| v);
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/benchmark-grids/pglib")
        .join(variant)
        .join(format!("{name}.m"));
    assert!(
        path.exists(),
        "{} not found — run `git submodule update --init tests/data/benchmark-grids`",
        path.display()
    );
    path
}

fn read(name: &str) -> MatpowerOpf {
    let text = std::fs::read_to_string(case_path(name)).unwrap();
    matpower::ac_opf_network(&matpower::parse(&text).unwrap()).unwrap()
}

fn build(name: &str) -> AcOpf {
    AcOpf::build(read(name).network, AcOpfOptions::default()).unwrap()
}

fn check(name: &str, published: f64, result: &AcOpfResult) {
    assert_eq!(result.status, OptStatus::Optimal, "{name}");
    assert!(
        result.violation < 1e-6,
        "{name}: violation {}",
        result.violation
    );
    let gap = (result.objective - published) / published;
    assert!(
        gap.abs() < GAP,
        "{name}: objective {} against PGLib's {published} (gap {gap:.2e})",
        result.objective
    );
}

#[test]
fn the_in_house_solver_reaches_pglibs_objectives() {
    for &(name, published) in PUBLISHED {
        if !NLP_UNSOLVED.contains(&name) {
            check(name, published, &build(name).solve().unwrap());
        }
    }
}

#[cfg(feature = "opf-ipopt")]
#[test]
fn ipopt_reaches_pglibs_objectives() {
    use gridoxide::opf::ipopt::{self, IpoptOptions};

    for &(name, published) in PUBLISHED {
        let opf = build(name);
        let solution = ipopt::solve(&opf, &IpoptOptions::default()).unwrap();
        let result = opf.interpret(&solution);
        check(name, published, &result);
        assert!(result.iterations > 0, "{name}: iterations not reported");
    }
}

/// On `__sad` the angle-difference limits are what makes the case: dropped,
/// the optimum is cheaper and breaks them.
#[test]
fn small_angle_difference_limits_bind() {
    let opf = read("pglib_opf_case14_ieee__sad");
    let limits = opf.network.angle_limits.clone();
    assert_eq!(limits.len(), 20, "one per in-service branch");

    let result = AcOpf::build(opf.network.clone(), AcOpfOptions::default())
        .unwrap()
        .solve()
        .unwrap();
    let widest = limits
        .iter()
        .map(|l| {
            let d = result.angles[l.from] - result.angles[l.to];
            (d - l.max).max(l.min - d)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(widest <= 1e-6, "a limit is exceeded by {widest} rad");
    assert!(widest > -1e-4, "no angle limit binds on __sad");

    let mut free = opf.network;
    free.angle_limits.clear();
    let unlimited = AcOpf::build(free, AcOpfOptions::default())
        .unwrap()
        .solve()
        .unwrap();
    assert!(unlimited.objective < result.objective * (1.0 - 1e-3));
}

#[test]
fn generators_keep_their_row_and_isolated_buses_are_dropped() {
    let mut case =
        matpower::parse(&std::fs::read_to_string(case_path("pglib_opf_case14_ieee")).unwrap())
            .unwrap();
    // Take generator row 1 out of service and isolate bus 8 with its
    // generator (row 4) and its one branch.
    case.gens[1][7] = 0.0;
    case.bus[7][1] = 4.0;
    let opf = matpower::ac_opf_network(&case).unwrap();

    assert_eq!(opf.network.n_buses, 13);
    assert!(!opf.bus_numbers.contains(&8));
    let rows: Vec<usize> = opf.network.generators.iter().map(|g| g.index).collect();
    assert_eq!(rows, vec![0, 2, 3]);
}

#[test]
fn a_case_without_costs_is_refused() {
    let mut case =
        matpower::parse(&std::fs::read_to_string(case_path("pglib_opf_case14_ieee")).unwrap())
            .unwrap();
    case.gencost = None;
    let err = matpower::ac_opf_network(&case).unwrap_err();
    assert!(err.to_string().contains("gencost"), "{err}");
}
