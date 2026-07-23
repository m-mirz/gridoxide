//! Ad-hoc runtime benchmark against a PGM-format JSON network, for comparing
//! against power-grid-model on the same input. Not part of the test suite.
//!
//! Usage: cargo run --release --example bench_network -- <path-to-input.json> [repeat-count] [backend] [mode]
//!
//! `repeat-count` (default 1) re-runs the linear initial guess + full
//! Newton-Raphson solve that many times from a fresh clone of the
//! post-parse (flat-start) buses each time, so `perf record` gets enough
//! samples to profile — a single solve at realistic network sizes is only
//! tens of milliseconds, too short to sample meaningfully.
//!
//! `backend` (default "scalar") selects `scalar` (the default `faer`-backed
//! path), `block` (the experimental block-per-bus path, symmetric only),
//! `klu` (the experimental vendored-KLU-via-FFI path, only available when
//! built with `--features klu`), `klu_native` (the experimental
//! from-scratch Rust port of the same KLU algorithm, no feature flag needed),
//! or `pardiso` (Intel oneMKL PARDISO, linked dynamically, only available
//! when built with `--features pardiso` and `MKLROOT` set) for a
//! head-to-head comparison on the same network.
//!
//! `mode` (default "cold") selects `cold` (each repeat calls
//! `newton_raphson_with_backend` fresh — no state carries over between
//! repeats, so every repeat redoes symbolic factorization, i.e. fill-
//! reducing ordering, from scratch) or `warm` (one `solver::PersistentSolver`
//! is reused across all repeats, so only the *first* repeat pays for
//! symbolic factorization — every later repeat only does a numeric
//! refactorization). `warm` is the fair comparison against tools that keep
//! one persistent solver/model object across repeated solves on unchanged
//! topology (lightsim2grid's `ac_pf`, PGM's `PowerGridModel`) — `cold`
//! measures "N independent flat-start solves with no shared state," a
//! different, also-legitimate scenario. Confirmed on a 9,241-bus case:
//! `warm` cut per-repeat `klu` time by ~45%.

use std::env;
use std::fs;
use std::time::Instant;

use gridoxide::network::{build_ybus, linear_initial_guess};
use gridoxide::pgm::pgm_to_buses_and_branches;
use gridoxide::solver::{newton_raphson_with_backend, JacobianBackend, PersistentSolver};

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: bench_network <input.json> [repeat-count] [backend] [mode]");
    let repeat: usize = args.next().map(|s| s.parse().expect("repeat-count must be an integer")).unwrap_or(1);
    let backend_arg = args.next().unwrap_or_else(|| "scalar".to_string());
    let backend = match backend_arg.as_str() {
        "scalar" => JacobianBackend::Scalar,
        "block" => JacobianBackend::Block,
        #[cfg(feature = "klu")]
        "klu" => JacobianBackend::Klu,
        #[cfg(not(feature = "klu"))]
        "klu" => panic!("the 'klu' backend needs `cargo run --features klu ...` (see the README's \"Sparse solver\" section)"),
        "klu_native" => JacobianBackend::KluNative,
        #[cfg(feature = "pardiso")]
        "pardiso" => JacobianBackend::Pardiso,
        #[cfg(not(feature = "pardiso"))]
        "pardiso" => panic!(
            "the 'pardiso' backend needs `cargo run --features pardiso ...` with MKLROOT set \
             (see the README's \"Experimental backends\" section)"
        ),
        other => panic!("unknown backend '{other}', expected 'scalar', 'block', 'klu', 'klu_native', or 'pardiso'"),
    };
    let mode_arg = args.next().unwrap_or_else(|| "cold".to_string());
    let warm = match mode_arg.as_str() {
        "cold" => false,
        "warm" => true,
        other => panic!("unknown mode '{other}', expected 'cold' or 'warm'"),
    };
    println!("backend={backend_arg} mode={mode_arg}");
    let raw = fs::read_to_string(&path).expect("read input file");

    let t_parse0 = Instant::now();
    let input = serde_json::from_str(&raw).expect("parse PGM input JSON");
    let (buses_template, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
    let n = buses_template.len();
    let ybus = build_ybus(n, &lines, &transformers);
    let t_parse = t_parse0.elapsed();

    println!("nodes={} lines={} transformers={}", n, lines.len(), transformers.len());
    println!("parse + build model + Y-bus: {:.3} ms", t_parse.as_secs_f64() * 1e3);

    let t_finish0 = Instant::now();
    let ybus = ybus.finish();
    let t_finish = t_finish0.elapsed();
    println!("Y-bus finish (COO -> sparse): {:.3} ms", t_finish.as_secs_f64() * 1e3);

    let mut total_guess = std::time::Duration::ZERO;
    let mut total_nr = std::time::Duration::ZERO;
    let mut buses = buses_template.clone();
    let mut persistent_solver = warm.then(|| PersistentSolver::new(backend));
    for _ in 0..repeat {
        buses = buses_template.clone();

        let t_guess0 = Instant::now();
        linear_initial_guess(&mut buses, &ybus);
        total_guess += t_guess0.elapsed();

        let t_nr0 = Instant::now();
        match persistent_solver.as_mut() {
            Some(solver) => { solver.solve(&mut buses, &ybus, 1e-6, 20); }
            None => { newton_raphson_with_backend(&mut buses, &ybus, 1e-6, 20, backend); }
        }
        total_nr += t_nr0.elapsed();
    }

    println!("linear_initial_guess: {:.3} ms total, {:.3} ms/run over {} run(s)",
        total_guess.as_secs_f64() * 1e3, total_guess.as_secs_f64() * 1e3 / repeat as f64, repeat);
    println!("newton_raphson: {:.3} ms total, {:.3} ms/run over {} run(s)",
        total_nr.as_secs_f64() * 1e3, total_nr.as_secs_f64() * 1e3 / repeat as f64, repeat);

    let vmin = buses.iter().map(|b| b.voltage_mag).fold(f64::INFINITY, f64::min);
    let vmax = buses.iter().map(|b| b.voltage_mag).fold(f64::NEG_INFINITY, f64::max);
    println!("voltage_mag min/max = {:.6} / {:.6}", vmin, vmax);
    println!("total (guess + NR): {:.3} ms", ((total_guess + total_nr).as_secs_f64() * 1e3) / repeat as f64);
}
