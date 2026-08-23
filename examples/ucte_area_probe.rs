//! Country areas from a UCTE file, and what each is currently exchanging.
fn main() {
    for path in std::env::args().skip(1) {
        let net = gridoxide::ucte::read(&path).expect("import");
        let (of_bus, names) = net.country_areas();
        let (lines, transformers) = net.as_switched();
        let report = gridoxide::run_power_flow(
            net.buses.clone(), &lines, &transformers, &net.shunts,
            gridoxide::TapData::none(),
            gridoxide::solver::PowerFlowOptions { tol: 1e-10, max_iter: 40, ..Default::default() },
        );
        let x = gridoxide::outerloop::AreaInterchange::measure(
            &report.buses, &lines, &transformers, &of_bus, names.len());
        println!("{path}: {} areas {names:?}, solve {:?}", names.len(), report.stats.status);
        for (i, n) in names.iter().enumerate() {
            let buses = of_bus.iter().filter(|a| **a == Some(i)).count();
            println!("    {n}: {buses:>3} bus(es), exporting {:>9.3} MW", x[i] * net.base_mva);
        }
        println!("    unassigned buses: {}", of_bus.iter().filter(|a| a.is_none()).count());
        println!("    sum of positions (= tie losses): {:.3} MW",
            x.iter().sum::<f64>() * net.base_mva);
    }
}
