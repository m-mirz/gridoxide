//! Another solver's AC-OPF solution evaluated in gridoxide's model of the
//! same case: which rows and bounds it violates, and by how much. Tells a
//! model that differs from the case (rows violated at the other solver's
//! optimum) from one the solver cannot reach (none violated): on
//! case13659_pegase, ExaModelsPower's optimum is feasible here to 2e-7.
//!
//!     PGLIB_DIR=... cargo run --release --example opf_point_probe --features opf,matpower -- <case> <solution.json>
//!
//! `solution.json`: grid-bench's dump (`vm`, `va_deg` per bus number; `pg_mw`,
//! `qg_mvar` per gen row).
use std::collections::HashMap;
use std::path::PathBuf;

use gridoxide::matpower;
use gridoxide::opf::ac::{AcOpf, AcOpfOptions};
use gridoxide::opf::nlp::NonlinearProblem;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = PathBuf::from(std::env::var("PGLIB_DIR").unwrap()).join(format!("{}.m", args[0]));
    let sol: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let mp = matpower::ac_opf_network(
        &matpower::parse(&std::fs::read_to_string(path).unwrap()).unwrap(),
    )
    .unwrap();
    let base = mp.network.base_mva;
    let n = mp.network.n_buses;
    let gens: Vec<(usize, usize)> = mp
        .network
        .generators
        .iter()
        .map(|g| (g.index, g.bus))
        .collect();
    let reference = mp.network.reference;
    let opf = AcOpf::build(mp.network, AcOpfOptions::default()).unwrap();

    let num = |v: &serde_json::Value| v.as_f64().unwrap();
    let va: HashMap<u64, f64> = sol["va_deg"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.parse().unwrap(), num(v)))
        .collect();
    let vm: HashMap<u64, f64> = sol["vm"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.parse().unwrap(), num(v)))
        .collect();
    let ref_va = va[&mp.bus_numbers[reference]];
    let g = gens.len();
    let mut x = vec![0.0; opf.n_vars()];
    for i in 0..n {
        let b = mp.bus_numbers[i];
        x[i] = (va[&b] - ref_va).to_radians();
        x[n + i] = vm[&b];
    }
    for (k, &(row, _)) in gens.iter().enumerate() {
        x[2 * n + k] = num(&sol["pg_mw"][row.to_string()]) / base;
        x[2 * n + g + k] = num(&sol["qg_mvar"][row.to_string()]) / base;
    }
    let (xl, xu) = opf.var_bounds();
    let worst_bound = (0..x.len())
        .map(|k| (k, (xl[k] - x[k]).max(x[k] - xu[k])))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    println!(
        "objective {:.6e} (PGLib reference in the json's tool)",
        opf.objective(&x)
    );
    println!(
        "worst variable bound violation {:.3e} at x[{}] (n {n}, g {g})",
        worst_bound.1, worst_bound.0
    );

    let c = opf.constraints(&x);
    let (lo, hi) = opf.constraint_bounds();
    let mut by_kind: HashMap<&str, (usize, f64, usize)> = HashMap::new();
    let mut rows: Vec<(usize, f64)> = Vec::new();
    for r in 0..c.len() {
        let v = (lo[r] - c[r]).max(c[r] - hi[r]).max(0.0);
        let kind = if r < n {
            "P balance"
        } else if r < 2 * n {
            "Q balance"
        } else if lo[r] == f64::NEG_INFINITY {
            "branch limit"
        } else if hi[r] == f64::INFINITY {
            "cost segment"
        } else {
            "angle difference"
        };
        let e = by_kind.entry(kind).or_insert((0, 0.0, 0));
        e.0 += 1;
        if v > e.1 {
            e.1 = v;
            e.2 = r;
        }
        if v > 1e-6 {
            rows.push((r, v));
        }
    }
    for (kind, (count, worst, at)) in &by_kind {
        println!("{kind:18} {count:6} rows, worst violation {worst:.3e} at row {at}");
    }
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("{} rows violated by more than 1e-6; worst:", rows.len());
    for &(r, v) in rows.iter().take(12) {
        let bus = if r < 2 * n {
            Some(mp.bus_numbers[r % n])
        } else {
            None
        };
        println!(
            "  row {r:6} by {v:.3e}  value {:.6e} bounds [{:e}, {:e}] bus {bus:?}",
            c[r], lo[r], hi[r]
        );
    }
}
