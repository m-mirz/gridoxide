//! Does the native KLU need KLU's row scaling? A stress run on the worst-scaled
//! networks at hand.
//!
//! `cargo run --release --example klu_scaling_stress --features matpower,cgmes,klu`
//!
//! For each network, two measurements:
//!
//! - **One linear solve**: the Newton Jacobian and mismatch at the
//!   constant-admittance start, solved by SuiteSparse's KLU (which scales),
//!   the native port unscaled and the native port scaled. Reported as the
//!   backward error ‖J·x − r‖∞ / (‖J‖∞‖x‖∞ + ‖r‖∞) — what partial pivoting is
//!   meant to keep near machine precision — and the difference from KLU's x.
//! - **The whole Newton solve**, per backend (SuiteSparse's KLU and the native
//!   port as the solver runs it, unscaled): status, iterations and the final
//!   mismatch, from both starts.

use std::path::{Path, PathBuf};

use gridoxide::jacobian::JacobianPattern;
use gridoxide::klu_native::KluNativeSystem;
use gridoxide::network::{
    ShuntAdm, YBusSparse, build_ybus, effective_injection, linear_initial_guess, power_injections,
    stamp_shunts,
};
use gridoxide::solver::JacobianBackend;
use gridoxide::sparse_klu::KluRealSystem;
use gridoxide::types::{Bus, BusType, Line, Transformer};
use gridoxide::{PowerFlowInit, PowerFlowOptions, TapData, run_power_flow};

struct Network {
    name: String,
    buses: Vec<Bus>,
    lines: Vec<Line>,
    transformers: Vec<Transformer>,
    shunts: Vec<ShuntAdm>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn matpower(name: &str) -> Option<Network> {
    let path = root().join(format!("benchmark-grids/matpower/{name}.m"));
    let n = gridoxide::matpower::read(&path).ok()?;
    Some(Network {
        name: name.into(),
        buses: n.buses,
        lines: n.lines,
        transformers: n.transformers,
        shunts: n.shunts,
    })
}

fn cgmes_realgrid() -> Option<Network> {
    let d = root().join("CGMES-Test-Configurations/v3.0/RealGrid/RealGrid-Merged");
    let files: Vec<PathBuf> = ["EQ", "SSH", "TP", "SV"]
        .iter()
        .map(|p| d.join(format!("RealGrid_{p}.xml")))
        .collect();
    let refs: Vec<&Path> = files.iter().map(PathBuf::as_path).collect();
    let ds = gridoxide::cgmes::load_profiles(&refs).ok()?;
    let (buses, lines, transformers, shunts) =
        gridoxide::cgmes::cgmes_to_buses_and_branches(&ds, 100e6).ok()?;
    Some(Network {
        name: "RealGrid (CGMES)".into(),
        buses,
        lines,
        transformers,
        shunts,
    })
}

/// Every switch retained as a stiff branch: the Jacobian's rows then span
/// the switch admittance and the ordinary ones.
fn cgmes_node_breaker(dir: &str, prefix: &str) -> Option<Network> {
    use gridoxide::switches::SwitchTreatment;
    use gridoxide::topology::RetentionPolicy;
    let d = root().join("CGMES-Test-Configurations/v3.0").join(dir);
    let files: Vec<PathBuf> = ["EQBD", "EQ", "SSH", "TP", "SV"]
        .iter()
        .map(|p| d.join(format!("{prefix}_{p}.xml")))
        .filter(|p| p.exists())
        .collect();
    let refs: Vec<&Path> = files.iter().map(PathBuf::as_path).collect();
    let ds = gridoxide::cgmes::load_profiles(&refs).ok()?;
    let net = gridoxide::cgmes::cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &RetentionPolicy::RetainAll,
        SwitchTreatment::Regularize,
    )
    .ok()?;
    Some(Network {
        name: format!("{prefix} node-breaker, all switches"),
        buses: net.buses,
        lines: net.lines,
        transformers: net.transformers,
        shunts: net.shunts,
    })
}

/// A PGM fixture with its source made `sk` VA stiff: the source branch's
/// admittance grows with it, far past everything else in the network.
fn pgm_stiff_source(case: &str, sk: f64) -> Option<Network> {
    let path = root().join(format!("pgm/powerflow/symmetric/{case}/input.json"));
    let mut input: gridoxide::pgm::PgmInput =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    for s in &mut input.data.source {
        s.sk = sk;
    }
    let idx = gridoxide::pgm::node_id_to_idx(&input);
    let shunts = gridoxide::pgm::pgm_shunts_1ph(&input, &idx, 1e6);
    let (buses, lines, transformers) = gridoxide::pgm::pgm_to_buses_and_branches(input, 1e6, 50.0);
    Some(Network {
        name: format!("{case} (PGM), sk = {sk:.0e}"),
        buses,
        lines,
        transformers,
        shunts,
    })
}

fn ybus_of(n: &Network) -> YBusSparse {
    let mut y = build_ybus(n.buses.len(), &n.lines, &n.transformers);
    stamp_shunts(&mut y, &n.shunts);
    y.finish()
}

/// The Jacobian and mismatch at the constant-admittance start.
fn jacobian(n: &Network, ybus: &YBusSparse) -> (usize, Vec<(usize, usize, f64)>, Vec<f64>) {
    let mut buses = n.buses.clone();
    linear_initial_guess(&mut buses, ybus);
    let (p, q) = power_injections(&buses, ybus);
    let pattern = JacobianPattern::analyze(&buses, ybus);
    let mut values = Vec::new();
    pattern.fill(&buses, &p, &q, &mut values);
    let mut r: Vec<f64> = buses
        .iter()
        .filter(|b| b.bus_type != BusType::Slack)
        .map(|b| effective_injection(b).0 - p[b.idx])
        .collect();
    r.extend(
        buses
            .iter()
            .filter(|b| b.bus_type == BusType::PQ)
            .map(|b| effective_injection(b).1 - q[b.idx]),
    );
    (pattern.n_unknowns, pattern.to_triplets(&values), r)
}

fn backward_error(j: &[(usize, usize, f64)], x: &[f64], r: &[f64]) -> f64 {
    let n = r.len();
    let mut jx = vec![0.0; n];
    let mut row_sum = vec![0.0f64; n];
    for &(i, k, v) in j {
        jx[i] += v * x[k];
        row_sum[i] += v.abs();
    }
    let inf = |v: &[f64]| v.iter().fold(0.0f64, |m, a| m.max(a.abs()));
    let resid: Vec<f64> = jx.iter().zip(r).map(|(a, b)| a - b).collect();
    inf(&resid) / (inf(&row_sum) * inf(x) + inf(r))
}

fn main() {
    let networks: Vec<Network> = [
        matpower("case9241pegase"),
        matpower("case1888rte"),
        matpower("case2848rte"),
        matpower("case6495rte"),
        matpower("case6515rte"),
        cgmes_realgrid(),
        cgmes_node_breaker("MiniGrid/MiniGrid-Merged", "MiniGrid"),
        cgmes_node_breaker("SmallGrid/SmallGrid-Merged", "SmallGrid"),
        pgm_stiff_source("transmission-case", 1e10),
        pgm_stiff_source("transmission-case", 1e20),
        pgm_stiff_source("transmission-case", 1e30),
        pgm_stiff_source("distribution-case", 1e30),
    ]
    .into_iter()
    .flatten()
    .collect();

    for net in &networks {
        let ybus = ybus_of(net);
        let (n, j, r) = jacobian(net, &ybus);
        let span = {
            let mut row_max = vec![0.0f64; n];
            for &(i, _, v) in &j {
                row_max[i] = row_max[i].max(v.abs());
            }
            let (lo, hi) = row_max
                .iter()
                .filter(|&&m| m > 0.0)
                .fold((f64::MAX, 0.0f64), |(lo, hi), &m| (lo.min(m), hi.max(m)));
            hi / lo
        };
        println!(
            "\n{} — {} unknowns, row magnitudes span {:.0e}",
            net.name, n, span
        );

        let c = KluRealSystem::new(n, &j).and_then(|mut s| s.factor_and_solve(&j, &r));
        let report = |label: &str, x: Option<Vec<f64>>| match x {
            None => println!("  {label:22} singular"),
            Some(x) => {
                let diff = c.as_ref().map_or(f64::NAN, |c| {
                    let size = c.iter().fold(0.0f64, |m, a| m.max(a.abs()));
                    x.iter()
                        .zip(c)
                        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
                        / size
                });
                println!(
                    "  {label:22} backward error {:9.1e}   vs SuiteSparse x {:9.1e}",
                    backward_error(&j, &x, &r),
                    diff
                );
            }
        };
        report("SuiteSparse KLU", c.clone());
        for (label, scale) in [("native, unscaled", 0), ("native, scaled", 2)] {
            let x = KluNativeSystem::with_scale(n, &j, scale)
                .and_then(|mut s| s.factor_and_solve(&j, &r));
            report(label, x);
        }

        for (label, backend) in [
            ("SuiteSparse KLU", JacobianBackend::Klu),
            ("native, unscaled", JacobianBackend::KluNative),
        ] {
            for init in [PowerFlowInit::LinearImpedance, PowerFlowInit::Flat] {
                let report = run_power_flow(
                    net.buses.clone(),
                    &net.lines,
                    &net.transformers,
                    &net.shunts,
                    TapData::none(),
                    PowerFlowOptions {
                        backend,
                        init,
                        tol: 1e-8,
                        max_iter: 30,
                        ..Default::default()
                    },
                );
                let s = &report.stats;
                println!(
                    "  Newton {label:17} from {:16} {:22} {:2} iterations, final mismatch {:.1e}",
                    format!("{init:?}"),
                    format!("{:?}", s.status),
                    s.iterations(),
                    s.final_mismatch()
                );
            }
        }
    }
}
