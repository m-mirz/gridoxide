//! Reactive margins across a network, and how that ranking compares to the
//! P-V one.
//!
//! The comparison is the point. Both are voltage-stability measures and it is
//! natural to expect them to agree on which bus is weakest; they do not, and
//! this is what shows it. See `docs/src/powerflow/qv.md`.
use gridoxide::continuation::{run_continuation, ContinuationOptions, LoadingDirection};
use gridoxide::pgm::{pgm_to_buses_and_branches, PgmInput};
use gridoxide::qv::{qv_curve, QvOptions, QvStatus};
use gridoxide::solver::PowerFlowOptions;
use gridoxide::types::{Bus, BusType};

/// Every bus's reactive margin, weakest first.
fn margins(
    buses: &[Bus],
    lines: &[gridoxide::types::Line],
    transformers: &[gridoxide::types::Transformer],
) -> Vec<(f64, usize, f64)> {
    let mut rows = Vec::new();
    for b in 0..buses.len() {
        if buses[b].bus_type == BusType::Slack {
            continue;
        }
        let curve = qv_curve(
            buses,
            lines,
            transformers,
            &[],
            b,
            QvOptions {
                pf: PowerFlowOptions { max_iter: 60, ..Default::default() },
                v_min: 0.20,
                ..Default::default()
            },
        );
        if curve.status == QvStatus::NoseFound {
            let n = curve.nose.as_ref().expect("a nose was found");
            rows.push((n.margin_pu, b, n.voltage));
        }
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    rows
}

fn main() {
    let names: Vec<String> = std::env::args().skip(1).collect();
    let names = if names.is_empty() {
        vec!["pglib_opf_case14_ieee".into(), "pglib_opf_case30_ieee".into()]
    } else {
        names
    };

    for name in names {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/pglib-opf")
            .join(format!("{name}.json"));
        let Ok(raw) = std::fs::read_to_string(&path) else {
            println!("{name}: absent");
            continue;
        };
        let input: PgmInput = serde_json::from_str(&raw).expect("PGM JSON");
        let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 100e6, 50.0);

        println!("\n=== {name}: {} buses", buses.len());
        let rows = margins(&buses, &lines, &transformers);
        println!("  weakest by Q-V reactive margin:");
        for (m, bus, v) in rows.iter().take(5) {
            println!("     bus {bus:3}  {:8.2} MVAr   nose at |V| {v:.4}", m * 100.0);
        }

        // The same networks, ranked by the P-V collapse mode instead.
        let direction = LoadingDirection::scale_loads(&buses);
        let curve = run_continuation(
            buses.clone(),
            &lines,
            &transformers,
            &[],
            ContinuationOptions { direction, ..Default::default() },
        );
        let pv: Vec<usize> = curve
            .critical
            .as_ref()
            .map(|c| c.weakest.iter().take(5).map(|w| w.bus).collect())
            .unwrap_or_default();
        let qv: Vec<usize> = rows.iter().take(5).map(|r| r.1).collect();
        let shared = qv.iter().filter(|b| pv.contains(b)).count();
        println!("  P-V (system collapse mode): {pv:?}");
        println!("  Q-V (local reactive headroom): {qv:?}");
        println!("  {shared} of 5 in common — the two measure different things");
    }
}
