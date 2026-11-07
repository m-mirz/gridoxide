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
//! `OPF_KKT=hybrid[:γ]` forms HyKKT's condensed system (`KktForm::Hybrid`,
//! γ = 1e8 by default), and `OPF_LINEAR=llt` factorizes it by Cholesky.
//! `OPF_TOL` sets the tolerance (1e-8 by default; grid-bench runs 1e-6), and
//! `OPF_REPEAT=n` times the median of `n` solves after an untimed warm-up,
//! as grid-bench does. With `--features gpu`, `OPF_LINEAR=cudss` factorizes the KKT system on
//! the GPU: cuDSS's `LDLᵀ` for the augmented system, its Cholesky for
//! HyKKT's; `OPF_LINEAR=cudss-fast` without cuDSS's deterministic mode,
//! as benchmarks run it. `CUDSS_THREADING_LIB` (cuDSS's `libcudss_mtlayer_gomp.so`)
//! runs its analysis on every core. `OPF_MODEL=gpu` evaluates the model
//! there too (`AcOpf::evaluate_on_device`), and the interior point's vectors
//! with it (`InteriorPoint::Device`), unless `OPF_IP=host`.
use std::path::{Path, PathBuf};
use std::time::Instant;

use gridoxide::matpower;
use gridoxide::opf::OptStatus;
use gridoxide::opf::ac::{AcOpf, AcOpfFormulation, AcOpfOptions, AcOpfResult, AcOpfStart};
use gridoxide::opf::nlp::{self, KktForm};

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
    if let Ok(tol) = std::env::var("OPF_TOL") {
        options.nlp.tolerance = tol.parse().expect("OPF_TOL");
    }
    if let Ok(kkt) = std::env::var("OPF_KKT")
        && let Some(rest) = kkt.strip_prefix("hybrid")
    {
        let gamma = rest
            .strip_prefix(':')
            .map_or(1e8, |g| g.parse().expect("gamma"));
        options.nlp.kkt = KktForm::Hybrid { gamma };
    }
    match std::env::var("OPF_LINEAR").as_deref() {
        Ok("llt") => options.nlp.kkt_solver = nlp::llt_solver,
        #[cfg(feature = "gpu")]
        Ok("cudss") => {
            options.nlp.kkt_solver = match options.nlp.kkt {
                KktForm::Augmented => gridoxide::gpu::cudss_ldlt_solver,
                KktForm::Hybrid { .. } => gridoxide::gpu::cudss_llt_solver,
            }
        }
        #[cfg(feature = "gpu")]
        Ok("cudss-fast") => {
            options.nlp.kkt_solver = match options.nlp.kkt {
                KktForm::Augmented => gridoxide::gpu::cudss_ldlt_solver_fast,
                KktForm::Hybrid { .. } => gridoxide::gpu::cudss_llt_solver_fast,
            }
        }
        Ok(other) if !other.is_empty() => panic!("unknown OPF_LINEAR {other}"),
        _ => {}
    }
    options.nlp.ncl = std::env::var_os("NLP_NCL").is_some();
    // With the model on the device, the interior point's vectors too.
    if std::env::var("OPF_MODEL").as_deref() == Ok("gpu")
        && std::env::var("OPF_IP").as_deref() != Ok("host")
    {
        let linear = std::env::var("OPF_LINEAR");
        options.nlp.interior_point = nlp::InteriorPoint::Device {
            deterministic: linear.as_deref() != Ok("cudss-fast"),
        };
    }
    #[allow(unused_mut)]
    let mut opf = AcOpf::build(opf.network, options).unwrap();
    #[cfg(feature = "gpu")]
    if std::env::var("OPF_MODEL").as_deref() == Ok("gpu") {
        opf.evaluate_on_device().unwrap();
    }
    opf
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
        let repeat: usize = std::env::var("OPF_REPEAT")
            .ok()
            .map_or(0, |n| n.parse().expect("OPF_REPEAT"));
        if repeat > 0 {
            opf.solve().unwrap();
        }
        let mut times = Vec::new();
        let mut result = None;
        for _ in 0..repeat.max(1) {
            let t0 = Instant::now();
            result = Some(opf.solve().unwrap());
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        let seconds = times[times.len() / 2];
        nlp_ok += row("nlp", name, *published, &result.unwrap(), seconds) as usize;

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
