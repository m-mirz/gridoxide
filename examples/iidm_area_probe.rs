//! IIDM control areas: what a file declares, and what each area exchanges.
fn main() {
    for path in std::env::args().skip(1) {
        let net = match gridoxide::iidm::read(&path) {
            Ok(n) => n,
            Err(e) => { println!("{path}: {e}"); continue }
        };
        let a = &net.areas;
        println!("=== {path}");
        println!("    {:?}", a.report);
        if a.ids.is_empty() { continue }
        let report = gridoxide::run_power_flow(
            net.buses.clone(), &net.lines, &net.transformers, &net.shunts,
            gridoxide::TapData::none(),
            gridoxide::solver::PowerFlowOptions { tol: 1e-10, max_iter: 40, ..Default::default() },
        );
        let x = gridoxide::outerloop::AreaInterchange::measure(
            &report.buses, &net.lines, &net.transformers, &a.of_bus, a.ids.len());
        println!("    solve {:?}", report.stats.status);
        for (i, (id, name)) in a.ids.iter().enumerate() {
            let n = a.of_bus.iter().filter(|z| **z == Some(i)).count();
            println!("    {id:<6} {name:<28} {n:>3} bus(es)  scheduled export {:>9.3} MW  actual {:>9.3} MW",
                a.targets[i] * net.base_mva, x[i] * net.base_mva);
        }
    }
}
