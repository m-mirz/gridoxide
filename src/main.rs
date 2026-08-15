//! gridoxide's command-line front end.
//!
//! Invoked with no arguments it runs the bundled power-flow demo, which is what
//! it has always done. `estimate <path>` runs state estimation over a PGM
//! document containing sensors, and `dc <path>` runs a DC (Bθ) power flow over
//! one.
//!
//! Argument handling is deliberately hand-rolled: a few modes and one path do
//! not justify a dependency, and the crate otherwise has none for this.

use std::fs;
use std::path::PathBuf;

use gridoxide::json::NetworkData;
use gridoxide::linear::{
    dc_branches, dc_power_flow, DcApproximation, DcOptions, DcSensitivity,
};
use gridoxide::measurement::measurements_from_pgm;
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{
    node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches, pgm_to_network, PgmInput,
};
use gridoxide::run_power_flow_analysis;
use gridoxide::se::bad_data::{self, Candidates};
use gridoxide::se::constraints::Constraints;
use gridoxide::se::jacobian::StateLayout;
use gridoxide::se::nr::{estimate, linear_start, SeMethod, SeOptions, SeStatus};
use gridoxide::se::observability;
use gridoxide::se::SeNetwork;
use gridoxide::shortcircuit::{
    short_circuit_from_pgm, SequenceValue, ShortCircuitOptions, VoltageScaling,
};
use gridoxide::solver::SolveStatus;

const USAGE: &str = "\
usage:
  gridoxide                     run the bundled power-flow demo
  gridoxide estimate <path> [--iterative-linear]
                                run state estimation over a PGM JSON document
                                containing sym_voltage_sensor/sym_power_sensor.
                                The default method is Newton-Raphson; the flag
                                selects the faster, less exact linearized one.
  gridoxide switches <profile.xml>... [--retain none|busbar_adjacent|all]
                     [--open <mrid>] [--solve]
                                list a CGMES model's switching devices, read
                                from EQ+SSH rather than merged away at import.
                                --retain chooses which survive as elements
                                (default busbar_adjacent); --open flips one
                                before solving; --solve runs a power flow and
                                reports each switch's own flow.
                                Needs the `cgmes` feature.
  gridoxide short-circuit <path> [--scaling max|min]
                                run an IEC 60909 short-circuit calculation over
                                a PGM JSON document containing `fault` entries,
                                printing per-node voltages, per-fault currents
                                and each source's contribution.
                                --scaling picks the voltage factor c (default
                                max, for the largest current; min is the
                                sensitivity study).
  gridoxide dc <path> [--ignore-g] [--ptdf <bus>] [--lodf <branch>]
                                run a DC (Bθ) power flow over a PGM JSON
                                document, printing bus angles, branch flows and
                                per-island slack pickup.
                                --ignore-g uses b = x/(r²+x²) instead of the
                                default b = 1/x; --ptdf/--lodf additionally
                                print one sensitivity column. Bus and branch
                                arguments are gridoxide's own 0-based indices,
                                as printed by the tables above them.
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => power_flow_demo(),
        Some("estimate") => match args.get(1) {
            Some(path) => {
                let method = if args.iter().any(|a| a == "--iterative-linear") {
                    SeMethod::IterativeLinear
                } else {
                    SeMethod::NewtonRaphson
                };
                if let Err(message) = run_estimate(path, method) {
                    eprintln!("error: {message}");
                    std::process::exit(1);
                }
            }
            None => {
                eprintln!("error: estimate needs a path\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("dc") => match args.get(1) {
            Some(path) if !path.starts_with("--") => {
                if let Err(message) = run_dc(path, &args[2..]) {
                    eprintln!("error: {message}");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("error: dc needs a path\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("short-circuit") => match args.get(1) {
            Some(path) if !path.starts_with("--") => {
                if let Err(message) = run_short_circuit(path, &args[2..]) {
                    eprintln!("error: {message}");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("error: short-circuit needs a path\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("switches") => {
            if let Err(message) = run_switches(&args[1..]) {
                eprintln!("error: {message}");
                std::process::exit(1);
            }
        }
        Some("-h") | Some("--help") | Some("help") => print!("{USAGE}"),
        Some(other) => {
            eprintln!("error: unknown command {other:?}\n\n{USAGE}");
            std::process::exit(2);
        }
    }
}

fn power_flow_demo() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/data/network.json");
    let network_json = fs::read_to_string(path).expect("Unable to read network.json");
    let network_data: NetworkData =
        serde_json::from_str(&network_json).expect("Unable to parse network.json");

    let report = run_power_flow_analysis(network_data);

    // The solver itself is silent so it can be driven from many threads at
    // once (see `batch::BatchSolver`); this reconstructs the progress output
    // it used to print, from `SolveStats::mismatch_history`.
    for (i, max_mis) in report.stats.mismatch_history.iter().enumerate() {
        println!("iter {}: max mismatch = {:.6e}", i + 1, max_mis);
    }
    match report.stats.status {
        SolveStatus::Converged => println!("Converged in {} iterations", report.stats.iterations()),
        SolveStatus::MaxIterationsReached => {
            println!("Failed to converge in {} iterations", report.stats.iterations())
        }
        SolveStatus::Singular => println!("Jacobian is singular. Failed to solve."),
    }

    println!("Final voltages:");
    for b in report.buses.iter() {
        println!(
            "Bus {}: |V| = {:.6}, angle = {:.6} deg",
            b.idx,
            b.voltage_mag,
            b.voltage_ang.to_degrees()
        );
    }

    println!("\nIslands: {} connected component(s)", report.islands.len());
    for (i, island) in report.islands.iter().enumerate() {
        println!("  island {i}: {} bus(es), status = {:?}", island.bus_indices.len(), island.status);
    }
}

/// Estimates the state of the grid in `path` and prints it, along with the two
/// analyses that say whether the answer should be trusted.
fn run_estimate(path: &str, method: SeMethod) -> Result<(), String> {
    let s_base_va = 1e6;
    let raw = fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let input: PgmInput =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?;
    let id_to_idx = node_id_to_idx(&input);
    let shunts = pgm_shunts_1ph(&input, &id_to_idx, s_base_va);
    let net = pgm_to_network(
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?,
        s_base_va,
        50.0,
    );
    let measurements =
        measurements_from_pgm(&input, &net, s_base_va).map_err(|e| format!("{path}: {e}"))?;
    if measurements.is_empty() {
        return Err(format!("{path} contains no usable sensors, so there is nothing to estimate"));
    }

    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &shunts);
    let se_net = SeNetwork::new(&net, ybus.finish(), &shunts);

    let mut buses = net.buses.clone();
    linear_start(&mut buses, &se_net, &measurements);
    // The linearized method converges linearly rather than quadratically, so it
    // wants a larger budget for the same tolerance — that trade, a cheaper
    // iteration for more of them, is the point of it.
    let max_iter = match method {
        SeMethod::IterativeLinear => 100,
        SeMethod::NewtonRaphson => 20,
    };
    let report = estimate(
        &measurements,
        &mut buses,
        &se_net,
        &SeOptions { method, max_iter, ..SeOptions::default() },
    );

    println!(
        "{} bus(es), {} measurement(s) after aggregation",
        buses.len(),
        measurements.len()
    );
    match report.status {
        SeStatus::Converged => {
            println!("Converged in {} iteration(s)", report.iterations)
        }
        SeStatus::MaxIterations => println!(
            "Did not converge in {} iteration(s); last step {:.3e}",
            report.iterations, report.last_step
        ),
        SeStatus::Singular => println!(
            "Gain matrix is singular after {} iteration(s) — see the observability \
             report below",
            report.iterations
        ),
    }
    println!("Objective J(x) = {:.6e}", report.objective);

    println!("\nEstimated voltages:");
    let mut ids: Vec<(&u64, &usize)> = net.node_idx.iter().collect();
    ids.sort();
    for (id, &idx) in ids {
        println!(
            "  node {id}: |V| = {:.6} p.u., angle = {:.6} deg",
            buses[idx].voltage_mag,
            buses[idx].voltage_ang.to_degrees()
        );
    }

    let layout = StateLayout::new(&buses, &measurements, &se_net);
    // The same constraints the estimator itself enforces — a state a
    // zero-injection constraint determines is observable, and reporting it
    // otherwise would send the user hunting for a sensor they do not need.
    let obs = observability::analyze(
        &measurements,
        &buses,
        &se_net,
        &layout,
        &Constraints::new(&se_net),
    );
    println!(
        "\nObservability: rank {} of {} unknown(s)",
        obs.rank, obs.n_unknowns
    );
    if obs.skipped_numerical {
        println!("  (too large for the dense rank check; structural analysis only)");
    }
    // Buses beyond the physical node count are gridoxide's own synthesized
    // ones — a virtual slack bus per source, a star point per three-winding
    // transformer — and are expected to be unobservable when the source's own
    // power is unmeasured. Saying which is which avoids a false alarm.
    let n_physical = net.node_idx.len();
    // The two lists overlap by design — a structurally unmeasured column is
    // also rank-deficient — so they are merged before printing rather than
    // reported twice.
    let mut undetermined: Vec<_> =
        obs.unobservable.iter().chain(&obs.structurally_unmeasured).copied().collect();
    undetermined.sort_by_key(|u| (u.bus, format!("{:?}", u.quantity)));
    undetermined.dedup();
    for unknown in undetermined {
        let kind = if unknown.bus >= n_physical { "synthesized" } else { "physical" };
        println!("  undetermined: {kind} bus {} ({:?})", unknown.bus, unknown.quantity);
    }
    if obs.is_observable() {
        println!("  fully observable");
    }

    let constraints = Constraints::new(&se_net);
    let bad = bad_data::analyze(
        &measurements,
        &report.residuals,
        &buses,
        &se_net,
        &layout,
        &constraints,
        Candidates::default(),
    );
    println!(
        "\nBad data: chi-squared {:.4e} on {} dof, p = {:.4e}",
        bad.chi_squared, bad.degrees_of_freedom, bad.p_value
    );
    if bad.rejects_at(0.05) {
        println!("  REJECTED at 5%: the measurements are not merely noisy");
        for suspect in bad.suspects.iter().take(5) {
            let m = &measurements[suspect.measurement];
            println!(
                "  suspect: measurement {} ({:?} on {:?}), normalized residual {:.2}",
                suspect.measurement, m.kind, m.target, suspect.normalized_residual
            );
        }
    } else {
        println!("  not rejected at 5%");
    }

    Ok(())
}

/// Parses `--flag <value>` out of the trailing argument list.
///
/// Returns `Ok(None)` when the flag is absent, so a missing flag and a
/// malformed one stay distinguishable.
fn flag_value(args: &[String], flag: &str) -> Result<Option<String>, String> {
    match args.iter().position(|a| a == flag) {
        None => Ok(None),
        Some(i) => args
            .get(i + 1)
            .cloned()
            .ok_or_else(|| format!("{flag} needs a value"))
            .map(Some),
    }
}

fn parse_index(raw: &str, flag: &str, limit: usize) -> Result<usize, String> {
    let value: usize =
        raw.parse().map_err(|_| format!("{flag}: {raw:?} is not a non-negative integer"))?;
    if value >= limit {
        return Err(format!("{flag}: {value} is out of range (0..{limit})"));
    }
    Ok(value)
}

/// Runs an IEC 60909 short-circuit calculation over the PGM document at `path`.
///
/// Prints the three things a protection study actually reads off: what each
/// fault draws, what each source contributes to it, and how far the voltage
/// collapses across the rest of the network while it does.
fn run_short_circuit(path: &str, flags: &[String]) -> Result<(), String> {
    let s_base_va = 1e6;
    let raw = fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let input: PgmInput =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?;

    let scaling = match flag_value(flags, "--scaling")?.as_deref() {
        None | Some("max") | Some("maximum") => VoltageScaling::Maximum,
        Some("min") | Some("minimum") => VoltageScaling::Minimum,
        Some(other) => return Err(format!("--scaling: expected max or min, got {other:?}")),
    };
    let opts = ShortCircuitOptions { scaling };

    let (net, report) = short_circuit_from_pgm(&input, s_base_va, 50.0, opts)
        .map_err(|e| format!("{e}"))?;

    if report.faults.is_empty() {
        println!(
            "note: this document declares no active `fault`, so what follows is the \
             pre-fault state under the scaled source voltages"
        );
    }

    println!(
        "{} node(s), {} fault(s), {} source(s), voltage scaling = {}",
        net.n_nodes,
        report.faults.len(),
        report.sources.len(),
        match scaling {
            VoltageScaling::Maximum => "c_max (largest current)",
            VoltageScaling::Minimum => "c_min (smallest current)",
        }
    );

    // Node ids, not gridoxide's indices: a fault is reported against the
    // document the user wrote.
    let mut by_index: Vec<(u64, usize)> =
        net.node_idx.iter().map(|(&id, &idx)| (id, idx)).collect();
    by_index.sort();

    if !report.faults.is_empty() {
        println!("\nfault currents (A):");
        for fault in &report.faults {
            println!(
                "  fault {:>6}: a = {:>12.2}, b = {:>12.2}, c = {:>12.2}",
                fault.id, fault.i_f[0], fault.i_f[1], fault.i_f[2]
            );
        }
    }

    println!("\nsource contributions (A):");
    for source in &report.sources {
        println!(
            "  source {:>5}: a = {:>12.2}, b = {:>12.2}, c = {:>12.2}",
            source.id, source.i[0], source.i[1], source.i[2]
        );
    }

    println!("\nnode voltages (p.u.):");
    for (id, idx) in &by_index {
        let node = &report.nodes[*idx];
        let mark = if node.energized { ' ' } else { '*' };
        println!(
            "  node {:>6}{mark}: a = {:>8.5}, b = {:>8.5}, c = {:>8.5}",
            id, node.u_pu[0], node.u_pu[1], node.u_pu[2]
        );
    }
    if report.nodes.iter().any(|n| !n.energized) {
        println!("  (* = de-energized: no path to any source, pinned at zero)");
    }

    // The sequence view, which is where a fault type's signature is legible:
    // a three-phase fault is pure positive sequence, a two-phase fault clear
    // of ground has no zero-sequence component at all, and anything involving
    // ground has one.
    println!("\nsymmetrical components of node voltage (p.u.):");
    for (id, idx) in &by_index {
        let s = SequenceValue::from_phase(&report.u_bus[*idx]);
        println!(
            "  node {:>6}: zero = {:>8.5}, positive = {:>8.5}, negative = {:>8.5}",
            id,
            s.zero.norm(),
            s.positive.norm(),
            s.negative.norm()
        );
    }

    Ok(())
}

/// Runs a DC power flow over the PGM document at `path` and prints it.
///
/// Reports the same three things the solver produces — angles, branch flows,
/// per-island slack pickup — plus the residual, which on a healthy network is
/// round-off and on an ill-conditioned one is the first sign of it.
fn run_dc(path: &str, flags: &[String]) -> Result<(), String> {
    let s_base_va = 1e6;
    let raw = fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let input: PgmInput =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?;
    let (mut buses, lines, transformers) = pgm_to_buses_and_branches(input, s_base_va, 50.0);

    let approximation = if flags.iter().any(|a| a == "--ignore-g") {
        DcApproximation::IgnoreG
    } else {
        DcApproximation::IgnoreR
    };
    let opts = DcOptions { approximation, ..DcOptions::default() };

    let n_branches = lines.len() + transformers.len();
    let solution = dc_power_flow(&mut buses, &lines, &transformers, opts);

    println!(
        "{} bus(es), {} branch(es), b = {}",
        buses.len(),
        n_branches,
        match approximation {
            DcApproximation::IgnoreR => "1/x",
            DcApproximation::IgnoreG => "x/(r²+x²)",
        }
    );

    for (i, island) in solution.islands.iter().enumerate() {
        println!(
            "  island {i}: {} bus(es), status = {:?}, slack pickup = {:.6} p.u. ({:.3} MW)",
            island.bus_indices.len(),
            island.status,
            island.slack_pickup,
            island.slack_pickup * s_base_va / 1e6,
        );
    }
    if !solution.negative_reactance_branches.is_empty() {
        println!(
            "  note: {} branch(es) have negative susceptance (series capacitors); B is indefinite",
            solution.negative_reactance_branches.len()
        );
    }
    if !solution.ignored_branches.is_empty() {
        println!(
            "  note: {} branch(es) carry no DC flow (open terminal, self-loop, or no susceptance)",
            solution.ignored_branches.len()
        );
    }
    println!("  max residual = {:.3e} p.u.", solution.max_residual);

    println!("\nbus angles:");
    for bus in &buses {
        println!(
            "  Bus {:>4}: {:>10.5} rad = {:>9.4}°",
            bus.idx,
            bus.voltage_ang,
            bus.voltage_ang.to_degrees()
        );
    }

    println!("\nbranch flows (at the from terminal):");
    for (i, p) in solution.branch_p.iter().enumerate() {
        println!("  Branch {i:>4}: {:>12.6} p.u. = {:>10.3} MW", p, p * s_base_va / 1e6);
    }

    let ptdf = flag_value(flags, "--ptdf")?;
    let lodf = flag_value(flags, "--lodf")?;
    if ptdf.is_none() && lodf.is_none() {
        return Ok(());
    }

    let branches = dc_branches(&lines, &transformers, opts);
    let sensitivity = DcSensitivity::new(&buses, &branches, n_branches)
        .ok_or("the reduced susceptance matrix is singular, so no sensitivities exist")?;

    if let Some(raw) = ptdf {
        let bus = parse_index(&raw, "--ptdf", buses.len())?;
        match sensitivity.ptdf_column(bus) {
            Some(column) => {
                println!("\nPTDF column for bus {bus} (∂P_branch / ∂P_bus):");
                for (k, v) in column.iter().enumerate() {
                    println!("  Branch {k:>4}: {v:>12.6}");
                }
            }
            None => println!("\nbus {bus} is in an island with no reference, so it has no PTDF"),
        }
    }

    if let Some(raw) = lodf {
        let branch = parse_index(&raw, "--lodf", n_branches)?;
        match sensitivity.lodf_column(branch) {
            Some(column) => {
                println!("\nLODF column for branch {branch} (fraction of its flow picked up):");
                for (k, v) in column.iter().enumerate() {
                    println!("  Branch {k:>4}: {v:>12.6}");
                }
            }
            None => println!(
                "\nbranch {branch} is radial: removing it islands the network, so no \
                 redistribution factors exist"
            ),
        }
    }

    Ok(())
}


/// Lists a CGMES model's switching devices, and optionally solves with them
/// retained as real elements.
///
/// This is the one command that reads a CGMES model at all — every other mode
/// takes power-grid-model JSON. It exists because a switch is only visible in
/// the node-breaker view, and that view is what `--retain` selects.
#[cfg(feature = "cgmes")]
fn run_switches(args: &[String]) -> Result<(), String> {
    use gridoxide::cgmes::{cgmes_node_breaker_to_buses_and_branches, load_profiles};
    use gridoxide::switches::SwitchTreatment;
    use gridoxide::topology::{RetentionPolicy, SwitchIdx};

    let paths: Vec<&String> = args.iter().take_while(|a| !a.starts_with("--")).collect();
    if paths.is_empty() {
        return Err("switches needs at least one CGMES profile file".to_string());
    }
    let flags = &args[paths.len()..];

    let retain = flag_value(flags, "--retain")?.unwrap_or_else(|| "busbar_adjacent".to_string());
    let policy = match retain.as_str() {
        "none" => RetentionPolicy::MergeAll,
        "busbar_adjacent" => RetentionPolicy::RetainAdjacentToBusbar,
        "all" => RetentionPolicy::RetainAll,
        other => {
            return Err(format!(
                "--retain: unknown policy {other:?}, expected none, busbar_adjacent or all"
            ))
        }
    };
    let open = flag_value(flags, "--open")?;
    let solve = flags.iter().any(|a| a == "--solve");

    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| std::path::Path::new(p.as_str())).collect();
    let ds = load_profiles(&path_refs).map_err(|e| format!("decoding CGMES profiles: {e}"))?;
    let mut net = cgmes_node_breaker_to_buses_and_branches(
        &ds,
        100e6,
        &policy,
        SwitchTreatment::Regularize,
    )
    .map_err(|e| format!("converting CGMES model: {e}"))?;

    if let Some(wanted) = &open {
        let target = net
            .view
            .retained()
            .iter()
            .map(|r| r.switch)
            .find(|s| &net.switch_label(*s) == wanted || s.0.to_string() == *wanted)
            .ok_or_else(|| {
                format!("--open: no retained switch matches {wanted:?} (try --retain all)")
            })?;
        if !net.set_switch_open(target, true) {
            return Err(format!("--open: switch {wanted} is degenerate, so its position is moot"));
        }
        println!("opened {}\n", net.switch_label(target));
    }

    println!(
        "{} bus(es) from {} connectivity node(s); retain = {retain}",
        net.buses.len(),
        net.view.n_nodes()
    );
    let stamped = net.switch_branches();
    let degenerate = gridoxide::switches::degenerate_switches(&net.view).len();
    println!(
        "{} switch(es) in the model, {} retained, {stamped_len} stamped as branches{}",
        net.topology.switches.len(),
        net.view.retained().len(),
        if degenerate > 0 {
            format!(" ({degenerate} degenerate — both ends already one bus)")
        } else {
            String::new()
        },
        stamped_len = stamped.len(),
    );

    let flows = if solve {
        let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
        stamp_shunts(&mut ybus, &net.shunts);
        let report = gridoxide::run_power_flow_analysis_from_ybus(net.buses.clone(), ybus);
        println!("\npower flow: {:?} in {} iteration(s)", report.stats.status, report.stats.iterations());
        let v = gridoxide::branch_flow::bus_voltages(&report.buses);
        Some(
            stamped
                .iter()
                .map(|(s, _)| net.switch_flow(*s, &v).unwrap_or((0.0, 0.0)))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    println!();
    print!("{:<40} {:<28} {:>6} {:>6} {:>7} {:>7}", "mrid", "kind", "bus", "bus", "state", "branch");
    if flows.is_some() {
        print!(" {:>12}", "P (MW)");
    }
    println!();
    for (n, (switch, branch)) in stamped.iter().enumerate() {
        let r = net.view.retained().iter().find(|r| r.switch == *switch).expect("stamped");
        print!(
            "{:<40} {:<28} {:>6} {:>6} {:>7} {:>7}",
            net.switch_label(*switch),
            net.switch_kind(*switch).map(|k| format!("{k:?}")).unwrap_or_default(),
            r.buses[0].0,
            r.buses[1].0,
            if net.is_switch_open(*switch).unwrap_or(false) { "open" } else { "closed" },
            branch,
        );
        if let Some(flows) = &flows {
            print!(" {:>12.3}", flows[n].0 * 100.0);
        }
        println!();
    }

    let _ = SwitchIdx(0);
    Ok(())
}

#[cfg(not(feature = "cgmes"))]
fn run_switches(_args: &[String]) -> Result<(), String> {
    Err("this build has no CGMES support; rebuild with `cargo build --features cgmes`".to_string())
}
