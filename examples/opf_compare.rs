//! `opf::nlp` against IPOPT on PGLib-OPF cases, from the same flat start.
//!
//! The measuring stick for making the in-house interior point robust: each
//! case is solved by both on the same `AcOpf` model, and the table says who
//! reached PGLib's published objective, in how many iterations and how long.
//!
//!     cargo run --release --example opf_compare --features opf,matpower[,opf-ipopt] -- [--log] [case ...]
//!
//! With no case named, grid-bench's default OPF set (typical 14 to 2869, and
//! the `__api` and `__sad` variants of 14, 118 and 300). `--log` prints
//! `opf::nlp`'s per-iteration log to stderr. Without `opf-ipopt`, only the
//! in-house solver runs. `PGLIB_DIR` points at another copy of PGLib's
//! cases (this repository's submodule stops at case9241).
//!
//! Debian's IPOPT links MUMPS against OpenMPI, whose hwloc probes the X11
//! display sockets at load and can block there for good (seen with a second
//! display that accepts no connection). `HWLOC_COMPONENTS=-gl` skips that
//! probe. `IPOPT_PRINT=5` prints IPOPT's own log. `OPF_START=proportional`
//! starts from `AcOpfStart::Proportional` instead of the flat start, and
//! `OPF_FORM=branch-flow` solves `AcOpfFormulation::BranchFlow`.
use std::path::{Path, PathBuf};
use std::time::Instant;

use gridoxide::matpower;
use gridoxide::opf::OptStatus;
use gridoxide::opf::ac::{AcOpf, AcOpfFormulation, AcOpfOptions, AcOpfResult, AcOpfStart};

/// PGLib v23.07's AC objectives (`pglib/BASELINE.md`), five significant digits.
const PUBLISHED: &[(&str, f64)] = &[
    ("pglib_opf_case14_ieee", 2.1781e+03),
    ("pglib_opf_case118_ieee", 9.7214e+04),
    ("pglib_opf_case300_ieee", 5.6522e+05),
    ("pglib_opf_case1354_pegase", 1.2588e+06),
    ("pglib_opf_case2869_pegase", 2.4628e+06),
    ("pglib_opf_case14_ieee__api", 5.9994e+03),
    ("pglib_opf_case118_ieee__api", 2.4961e+05),
    ("pglib_opf_case300_ieee__api", 6.8604e+05),
    ("pglib_opf_case14_ieee__sad", 2.7768e+03),
    ("pglib_opf_case118_ieee__sad", 1.0516e+05),
    ("pglib_opf_case300_ieee__sad", 5.6570e+05),
    ("pglib_opf_case9241_pegase", 6.2431e+06),
    ("pglib_opf_case10000_goc", 1.3540e+06),
    ("pglib_opf_case10480_goc", 2.3146e+06),
    ("pglib_opf_case13659_pegase", 8.9480e+06),
    ("pglib_opf_case19402_goc", 1.9778e+06),
    ("pglib_opf_case24464_goc", 2.6295e+06),
    ("pglib_opf_case30000_goc", 1.1423e+06),
];

/// The cases grid-bench runs by default.
const DEFAULT: usize = 11;

/// Half a unit in the fifth digit of the reference, and then some.
const GAP: f64 = 1e-4;

fn case_path(name: &str) -> PathBuf {
    let variant = name.rsplit_once("__").map_or("", |(_, v)| v);
    std::env::var_os("PGLIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/benchmark-grids/pglib")
        })
        .join(variant)
        .join(format!("{name}.m"))
}

fn build(name: &str, log: bool) -> AcOpf {
    let text = std::fs::read_to_string(case_path(name)).expect("case file");
    let opf = matpower::ac_opf_network(&matpower::parse(&text).unwrap()).unwrap();
    let start = match std::env::var("OPF_START").as_deref() {
        Ok("proportional") => AcOpfStart::Proportional,
        _ => AcOpfStart::Flat,
    };
    let formulation = match std::env::var("OPF_FORM").as_deref() {
        Ok("branch-flow") => AcOpfFormulation::BranchFlow,
        _ => AcOpfFormulation::BusInjection,
    };
    let mut options = AcOpfOptions {
        start,
        formulation,
        ..AcOpfOptions::default()
    };
    options.nlp.log = log;
    options.nlp.ncl = std::env::var_os("NLP_NCL").is_some();
    AcOpf::build(opf.network, options).unwrap()
}

fn verdict(result: &AcOpfResult, published: f64) -> (bool, f64) {
    let gap = (result.objective - published) / published;
    let ok = result.status == OptStatus::Optimal && result.violation < 1e-5 && gap.abs() < GAP;
    (ok, gap)
}

fn row(solver: &str, name: &str, published: f64, result: &AcOpfResult, seconds: f64) -> bool {
    let (ok, gap) = verdict(result, published);
    println!(
        "{name:30} {solver:6} {:4} {:>5} it {:9.3} s  gap {gap:+.1e}  viol {:.1e}  {:?}",
        if ok { "ok" } else { "FAIL" },
        result.iterations,
        seconds,
        result.violation,
        result.status,
    );
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let log = args.iter().any(|a| a == "--log");
    let named: Vec<&str> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .collect();
    let cases: Vec<(&str, f64)> = if named.is_empty() {
        PUBLISHED[..DEFAULT].to_vec()
    } else {
        named
            .iter()
            .map(|n| {
                *PUBLISHED
                    .iter()
                    .find(|(p, _)| p == n)
                    .unwrap_or_else(|| panic!("unknown case {n}"))
            })
            .collect()
    };

    #[allow(unused_mut)] // `ipopt_ok` counts only with `opf-ipopt`
    let (mut nlp_ok, mut ipopt_ok) = (0, 0);
    for (name, published) in &cases {
        let opf = build(name, log);
        let t0 = Instant::now();
        let result = opf.solve().unwrap();
        nlp_ok += row("nlp", name, *published, &result, t0.elapsed().as_secs_f64()) as usize;

        #[cfg(feature = "opf-ipopt")]
        {
            use gridoxide::opf::ipopt::{self, IpoptOptions};
            let options = IpoptOptions {
                tolerance: gridoxide::opf::nlp::NlpOptions::default().tolerance,
                print_level: std::env::var("IPOPT_PRINT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                ..IpoptOptions::default()
            };
            let t0 = Instant::now();
            let solution = ipopt::solve(&opf, &options).unwrap();
            let result = opf.interpret(&solution);
            ipopt_ok += row(
                "ipopt",
                name,
                *published,
                &result,
                t0.elapsed().as_secs_f64(),
            ) as usize;
        }
    }
    println!(
        "\nnlp {nlp_ok}/{n} ok, ipopt {ipopt_ok}/{n} ok",
        n = cases.len()
    );
}
