//! Continuation on the pglib fixtures: the loadability limit, the
//! reactive-limit events on the way, and the weakest buses at the end.
//!
//! The numbers `tests/continuation_test.rs` records as baselines come from
//! here. With no arguments the cases are converted from the `benchmark-grids`
//! submodule by `tests/data/pglib_opf.py`, the same as the tests do, which needs
//! a Python with numpy (`python3` unless `PYTHON` names another). Run with paths
//! to trace other networks:
//! `cargo run --release --example cpf_probe -- case1354pegase.json`
use gridoxide::PowerFlowOptions;
use gridoxide::continuation::{ContinuationOptions, LoadingDirection, run_continuation};
use gridoxide::pgm::{PgmInput, pgm_to_buses_and_branches};
use std::time::Instant;

/// Converts the pglib cases into a scratch directory and returns it. Examples
/// get no `CARGO_TARGET_TMPDIR`, so this is the system temp directory rather
/// than the per-binary one `tests/pglib` uses.
fn convert_pglib() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("gridoxide-cpf-probe");
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pglib_opf.py");
    let out = std::process::Command::new(&python)
        .arg(&script)
        .arg(&dir)
        .output()
        .unwrap_or_else(|e| panic!("running {python}: {e}; set PYTHON to a Python with numpy"));
    assert!(
        out.status.success(),
        "{} failed (needs numpy and the benchmark-grids submodule):\n{}",
        script.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    dir
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let paths: Vec<String> = if args.is_empty() {
        let base = convert_pglib();
        [
            "pglib_opf_case5_pjm",
            "pglib_opf_case14_ieee",
            "pglib_opf_case30_ieee",
            "pglib_opf_case118_ieee",
        ]
        .iter()
        .map(|n| {
            base.join(format!("{n}.json"))
                .to_string_lossy()
                .into_owned()
        })
        .collect()
    } else {
        args
    };

    for path in paths {
        let name = std::path::Path::new(&path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let raw = match std::fs::read_to_string(&path) {
            Ok(r) => r,
            Err(e) => {
                println!("{name}: {e}");
                continue;
            }
        };
        let input: PgmInput = serde_json::from_str(&raw).expect("PGM JSON");
        let (template, lines, transformers) = pgm_to_buses_and_branches(input, 100e6, 50.0);
        let n = template.len();

        for q_limits in [false, true] {
            let direction = LoadingDirection::scale_loads(&template);
            let started = Instant::now();
            let curve = run_continuation(
                template.clone(),
                &lines,
                &transformers,
                &[],
                ContinuationOptions {
                    direction,
                    power_flow: PowerFlowOptions {
                        enforce_q_limits: q_limits,
                        max_iter: 60,
                        ..Default::default()
                    },
                    base_mva: 100.0,
                    ..Default::default()
                },
            );
            let elapsed = started.elapsed();

            match &curve.critical {
                Some(c) => {
                    let weak: Vec<String> = c
                        .weakest
                        .iter()
                        .take(3)
                        .map(|w| format!("{}({:.2})", w.bus, w.participation))
                        .collect();
                    println!(
                        "{name:24} n={n:5} qlim={q_limits:5}  lambda_max={:8.4}  margin={:9.1} MW  \
                         {:?}  weakest=[{}]",
                        c.lambda_max,
                        c.margin_mw,
                        c.kind,
                        weak.join(" ")
                    );
                    println!(
                        "{:24} {:>11}  points={:<4} events={:<3} segments={:<3} solves={:<5} \
                         reanalyses={:<3} {:?}",
                        "",
                        format!("{elapsed:.2?}"),
                        curve.points.len(),
                        curve.q_limit_events().len(),
                        curve.segments,
                        curve.solves,
                        curve.reanalyses,
                        curve.status
                    );
                }
                None => println!(
                    "{name:24} n={n:5} qlim={q_limits:5}  no collapse point: {:?} after {} \
                     point(s), last lambda {:?}",
                    curve.status,
                    curve.points.len(),
                    curve.points.last().map(|p| p.lambda)
                ),
            }
        }
    }
}
