//! Where a Newton-Raphson state-estimation iteration actually spends its time.
use std::path::PathBuf;
use std::time::Instant;
use gridoxide::measurement::Measurement;
use gridoxide::network::{build_ybus, linear_initial_guess, stamp_shunts};
use gridoxide::pgm::{node_id_to_idx, pgm_shunts_1ph, pgm_to_network, PgmInput};
use gridoxide::se::constraints::Constraints;
use gridoxide::se::jacobian::{gain_and_rhs, measurement_jacobian, StateLayout};
use gridoxide::se::nr::flat_start;
use gridoxide::se::{measurement_functions, SeNetwork};
use gridoxide::solver::{JacobianBackend, LinearSolver, PersistentSolver};
use gridoxide::sparse::RealSparseSystem;

fn main() {
    let case = std::env::args().nth(1).unwrap_or_else(|| "case1354pegase".into());
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts/bench/.case-cache").join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path).unwrap();
    let input: PgmInput = serde_json::from_str(&text).unwrap();
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, 1e6);
    let net = pgm_to_network(serde_json::from_str(&text).unwrap(), 1e6, 50.0);
    let mut yb = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut yb, &shunts);
    let ybus = yb.finish();
    let mut truth = net.buses.clone();
    linear_initial_guess(&mut truth, &ybus);
    PersistentSolver::new(JacobianBackend::Scalar).solve(&mut truth, &ybus, 1e-10, 40);
    let se_net = SeNetwork::new(&net, ybus, &shunts);

    // Same synthesis as bench_se.rs, inlined.
    let ms: Vec<Measurement> = {
        use gridoxide::branch_flow::*;
        use gridoxide::measurement::{MeasurementKind, Target};
        let v = bus_voltages(&truth);
        let params = branch_params(&net.lines, &net.transformers);
        let mut out = Vec::new();
        for (bus, b) in truth.iter().enumerate() {
            out.push(Measurement { kind: MeasurementKind::VoltageMagnitude,
                target: Target::Bus(bus), value: b.voltage_mag, sigma: 1e-3 });
        }
        for (branch, p) in params.iter().enumerate() {
            if p.from == p.to { continue }
            for t in [Terminal::From, Terminal::To] {
                let (pf, qf) = terminal_flow(p, t, &v);
                let target = Target::BranchTerminal { branch, terminal: t };
                out.push(Measurement { kind: MeasurementKind::ActivePower, target, value: pf, sigma: 1e-2 });
                out.push(Measurement { kind: MeasurementKind::ReactivePower, target, value: qf, sigma: 1e-2 });
            }
        }
        out
    };

    let mut buses = net.buses.clone();
    flat_start(&mut buses, &ms);
    let layout = StateLayout::new(&buses, &ms, &se_net);
    let constraints = Constraints::new(&se_net.constrained_buses());
    let n = layout.n_unknowns();

    let reps = 6; // one estimate's worth of iterations
    let t = |f: &mut dyn FnMut()| { let s = Instant::now(); for _ in 0..reps { f(); } s.elapsed().as_secs_f64() * 1e3 };

    let mut h_ms = 0.0;
    let mut jac_ms = 0.0; let mut gain_ms = 0.0; let mut solve_ms = 0.0;
    h_ms += t(&mut || { measurement_functions(&ms, &buses, &se_net); });
    jac_ms += t(&mut || { measurement_jacobian(&ms, &buses, &se_net, &layout); });

    let rows = measurement_jacobian(&ms, &buses, &se_net, &layout);
    let resid: Vec<f64> = ms.iter().map(|m| m.value).collect();
    gain_ms += t(&mut || { gain_and_rhs(&rows, &ms, &resid); });

    let (triplets, mut rhs, _) = gain_and_rhs(&rows, &ms, &resid);
    rhs.resize(n, 0.0);
    let mut trip = triplets.clone();
    let (c_values, c_rows) = constraints.evaluate(&buses, &se_net, &layout);
    gridoxide::se::jacobian::mask_untouched(&mut trip, &mut rhs, &[&rows, &c_rows], n);
    let (trip, rhs) = gridoxide::se::constraints::augment(trip, rhs, n, &c_values, &c_rows);
    let mut sys = RealSparseSystem::new(n + constraints.len(), &trip).unwrap();
    solve_ms += t(&mut || { sys.factor_and_solve(&trip, &rhs); });

    let total = h_ms + jac_ms + gain_ms + solve_ms;
    println!("{case}: {} measurements, {n} unknowns, {reps} iterations", ms.len());
    for (name, v) in [("h(x)", h_ms), ("H assembly", jac_ms), ("gain assembly", gain_ms), ("factor+solve", solve_ms)] {
        println!("   {name:<16} {v:7.1} ms  {:5.1}%", 100.0 * v / total);
    }
    println!("   {:<16} {total:7.1} ms", "total");
}
