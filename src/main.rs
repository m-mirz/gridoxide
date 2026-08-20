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

use gridoxide::ac_sensitivity::{AcSensitivity, Function, Variable};
use gridoxide::branch_flow::Terminal;
use gridoxide::json::NetworkData;
use gridoxide::linear::{
    dc_branches, dc_power_flow, DcApproximation, DcOptions, DcSensitivity,
};
use gridoxide::measurement::measurements_from_pgm;
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{
    node_id_to_idx, pgm_shunts_1ph, pgm_to_buses_and_branches, pgm_to_network, PgmInput,
};
use gridoxide::{run_power_flow, run_power_flow_analysis};
use gridoxide::se::bad_data::{self, Candidates};
use gridoxide::se::constraints::Constraints;
use gridoxide::se::jacobian::StateLayout;
use gridoxide::se::nr::{estimate, linear_start, SeMethod, SeOptions, SeStatus};
use gridoxide::se::observability;
use gridoxide::se::SeNetwork;
use gridoxide::shortcircuit::{
    short_circuit_from_pgm, SequenceValue, ShortCircuitOptions, VoltageScaling,
};
use gridoxide::solver::{PowerFlowOptions, SolveStatus};

const USAGE: &str = "\
usage:
  gridoxide                     run the bundled power-flow demo
  gridoxide estimate <path> [--iterative-linear]
                                run state estimation over a PGM JSON document
                                containing sym_voltage_sensor/sym_power_sensor.
                                The default method is Newton-Raphson; the flag
                                selects the faster, less exact linearized one.
  gridoxide security <network> --crac <crac.json> [--json]
                                assess a network against a CRAC: which critical
                                elements are overloaded, in which state, and by
                                how much. The network may be UCTE-DEF (.uct) or
                                IIDM (.xiidm); the CRAC may be OpenRAO JSON or
                                gridoxide's own <network>.rao.json companion.
                                Exits 0 when every margin is non-negative and 1
                                when any is not, so it can gate a pipeline.
  gridoxide rao <network> --crac <crac.json> [--depth N] [--validate-ac] [--json]
                                optimize remedial actions: search over the
                                network actions the CRAC permits, re-optimizing
                                the range actions at every candidate, and report
                                what to do in each state. --depth bounds how
                                many network actions may be stacked (default 2).
                                Exits 0 when every perimeter ends secure.
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
  gridoxide opf <network.json> [--ac] [--data <opf.json>] [--no-shedding]
                     [--shed-price <$/MWh>] [--ignore-r] [--highs]
                     [--no-limits] [--max-iter <n>]
                                run a DC optimal power flow: least-cost dispatch
                                subject to generator limits and branch ratings,
                                printing the dispatch, locational marginal
                                prices and whatever is binding.
                                Costs and limits come from a companion document,
                                defaulting to <network>.opf.json — the pair
                                `gridoxide-matpower` writes. Needs the `opf`
                                feature; --highs selects the reference HiGHS
                                backend instead of the built-in interior-point
                                one, and additionally needs `opf-highs`.
                                --ignore-r uses b = 1/x instead of the default
                                b = x/(r²+x²). Note this is the opposite default
                                from `dc` above, on purpose: each matches what
                                its own field's reference tools compute.
                                --ac solves the full AC problem instead: real
                                voltages, reactive power and losses, optimizing
                                generator P and Q together. Nonconvex, so the
                                answer is a local optimum. --no-limits drops the
                                branch ratings and --max-iter caps the solve;
                                --no-shedding/--shed-price/--ignore-r/--highs
                                are DC-only.
  gridoxide sensitivity <path> [--dp <bus>] [--dq <bus>] [--dk <branch>]
                     [--dalpha <branch>] [--watch <branch>] [--terminal from|to]
                                solve an AC power flow and differentiate it.
                                --dp/--dq/--dk/--dalpha each pick one variable
                                (active/reactive injection, transformer ratio,
                                phase-shifter angle) and print how every branch
                                flow and bus voltage responds to it.
                                --watch picks one branch instead and prints the
                                reverse: what every injection and every tap
                                would do to *its* flow. At least one is needed.
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
        Some("security") => match args.get(1) {
            Some(path) if !path.starts_with("--") => match run_security(path, &args[2..]) {
                Ok(secure) => {
                    if !secure {
                        std::process::exit(1);
                    }
                }
                Err(message) => {
                    eprintln!("error: {message}");
                    std::process::exit(2);
                }
            },
            _ => {
                eprintln!("error: security needs a network path\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("rao") => match args.get(1) {
            Some(path) if !path.starts_with("--") => match run_rao(path, &args[2..]) {
                Ok(secure) => {
                    if !secure {
                        std::process::exit(1);
                    }
                }
                Err(message) => {
                    eprintln!("error: {message}");
                    std::process::exit(2);
                }
            },
            _ => {
                eprintln!("error: rao needs a network path\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("opf") => match args.get(1) {
            Some(path) if !path.starts_with("--") => {
                if let Err(message) = run_opf(path, &args[2..]) {
                    eprintln!("error: {message}");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("error: opf needs a network path\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("sensitivity") => match args.get(1) {
            Some(path) if !path.starts_with("--") => {
                if let Err(message) = run_sensitivity(path, &args[2..]) {
                    eprintln!("error: {message}");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("error: sensitivity needs a path\n\n{USAGE}");
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

/// Runs a DC optimal power flow over the PGM document at `path` and its
/// companion OPF document.
///
/// Prints what a dispatcher reads off one: what each unit should produce, what
/// it costs, what a marginal megawatt is worth at each bus, and which limits
/// are the reason the answer is not simply "run the cheapest unit".
#[cfg(feature = "opf")]
fn run_opf(path: &str, flags: &[String]) -> Result<(), String> {
    if flags.iter().any(|f| f == "--ac") {
        return run_ac_opf(path, flags);
    }
    use gridoxide::opf::dc::{DcOpf, DcOpfNetwork, DcOpfOptions};
    use gridoxide::opf::model::OpfData;
    use gridoxide::opf::{OptStatus, Solver};

    // The converter writes `<network>.opf.json` beside the network document,
    // so defaulting to that makes the common case a single argument.
    let data_path = match flag_value(flags, "--data")? {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(path).with_extension("opf.json"),
    };

    let network_text = fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let data_text = fs::read_to_string(&data_path).map_err(|e| {
        format!(
            "reading {}: {e} — pass --data if the companion OPF document is elsewhere",
            data_path.display()
        )
    })?;

    let input: PgmInput =
        serde_json::from_str(&network_text).map_err(|e| format!("parsing {path}: {e}"))?;
    let data = OpfData::from_json(&data_text)
        .map_err(|e| format!("parsing {}: {e}", data_path.display()))?;

    let mut options = DcOpfOptions {
        allow_shedding: !flags.iter().any(|a| a == "--no-shedding"),
        ..Default::default()
    };
    if let Some(raw) = flag_value(flags, "--shed-price")? {
        options.shed_price = raw
            .parse()
            .map_err(|_| format!("--shed-price: {raw:?} is not a number"))?;
    }

    // DC-OPF defaults to the true series susceptance rather than the `1/x`
    // the `dc` command uses; see `DcOpfOptions` for why the two differ.
    let approximation = if flags.iter().any(|f| f == "--ignore-r") {
        gridoxide::linear::DcApproximation::IgnoreR
    } else {
        gridoxide::linear::DcApproximation::IgnoreG
    };
    let network =
        DcOpfNetwork::from_pgm(input, &data, 50.0, approximation).map_err(|e| e.to_string())?;
    let n_buses = network.n_buses;
    let generators = network.generators.clone();
    let total_load: f64 = network.loads.iter().map(|l| l.p).sum::<f64>() * network.base_mva;

    let opf = DcOpf::build(network, options).map_err(|e| e.to_string())?;

    // The in-house interior-point method is the default because it needs no
    // system install; `--highs` reaches the reference backend where one is
    // available. The two are cross-checked in `tests/opf_cross_test.rs`, so
    // this is a choice about dependencies rather than about the answer.
    let solution = if flags.iter().any(|f| f == "--highs") {
        #[cfg(feature = "opf-highs")]
        {
            let mut solver = gridoxide::opf::highs::HighsSolver::new()
                .map_err(|e| e.to_string())?;
            solver.solve(opf.problem()).map_err(|e| e.to_string())?
        }
        #[cfg(not(feature = "opf-highs"))]
        {
            return Err("--highs needs the `opf-highs` feature, which links a local \
                        HiGHS install (on Debian/Ubuntu, `apt install libhighs-dev`)"
                .to_string());
        }
    } else {
        let mut solver = gridoxide::opf::ipm::IpmSolver::new();
        solver.solve(opf.problem()).map_err(|e| e.to_string())?
    };
    let result = opf.interpret(&solution);

    if result.status != OptStatus::Optimal {
        return Err(format!(
            "no optimal dispatch ({:?}). An infeasible case usually means demand cannot be \
             served within the limits — try without --no-shedding to see where",
            result.status
        ));
    }

    println!(
        "{n_buses} bus(es), {} generator(s), {:.1} MW of demand",
        generators.len(),
        total_load
    );
    println!("total cost: {:.2} $/h", result.objective);

    println!("\ndispatch (MW):");
    for (g, generator) in generators.iter().enumerate() {
        let p = result.dispatch[g];
        // A unit at a limit is the interesting one — it is where the answer is
        // being shaped by something other than price.
        let at = if (p - generator.p_max * opf.network().base_mva).abs() < 1e-6 {
            " (at max)"
        } else if (p - generator.p_min * opf.network().base_mva).abs() < 1e-6 {
            " (at min)"
        } else {
            ""
        };
        println!("  generator {:>4}: {:>10.3}{at}", generator.index, p);
    }

    println!("\nlocational marginal price ($/MWh):");
    let min = result.lmp.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = result.lmp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-9 {
        println!("  uniform at {min:.4} — nothing is congested");
    } else {
        for (bus, price) in result.lmp.iter().enumerate() {
            println!("  bus {bus:>4}: {price:>10.4}");
        }
        println!("  spread: {:.4} (the congestion)", max - min);
    }

    if result.binding.is_empty() {
        println!("\nno binding branch limits");
    } else {
        println!("\nbinding branch limits:");
        for b in &result.binding {
            println!(
                "  branch {:>4}: flow {:>10.3} of {:>9.3} MW, worth {:>9.4} $/MWh to relieve",
                b.branch,
                b.flow,
                b.rate,
                b.price.abs()
            );
        }
    }

    let shed: f64 = result.shed.iter().sum();
    if shed > 1e-6 {
        println!("\nload shed: {shed:.3} MW — demand could not be served in full");
        for (d, amount) in result.shed.iter().enumerate() {
            if *amount > 1e-6 {
                println!("  load {d:>4}: {amount:>10.3} MW");
            }
        }
    }

    Ok(())
}

/// AC optimal power flow.
///
/// Prints the same things the DC form does plus what only AC has: voltage
/// magnitudes, reactive dispatch, and the reactive price. The reported
/// constraint violation is not decoration — on a nonconvex problem an
/// objective at an infeasible point is not an answer, so the two belong
/// together.
#[cfg(feature = "opf")]
fn run_ac_opf(path: &str, flags: &[String]) -> Result<(), String> {
    use gridoxide::opf::ac::{AcOpf, AcOpfNetwork, AcOpfOptions};
    use gridoxide::opf::model::OpfData;
    use gridoxide::opf::OptStatus;

    let data_path = match flag_value(flags, "--data")? {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(path).with_extension("opf.json"),
    };
    let network_text =
        fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let data_text = fs::read_to_string(&data_path).map_err(|e| {
        format!(
            "reading {}: {e} — pass --data if the companion OPF document is elsewhere",
            data_path.display()
        )
    })?;
    let input: PgmInput =
        serde_json::from_str(&network_text).map_err(|e| format!("parsing {path}: {e}"))?;
    let data = OpfData::from_json(&data_text)
        .map_err(|e| format!("parsing {}: {e}", data_path.display()))?;

    let mut options = AcOpfOptions {
        enforce_limits: !flags.iter().any(|f| f == "--no-limits"),
        ..AcOpfOptions::default()
    };
    if let Some(raw) = flag_value(flags, "--max-iter")? {
        options.nlp.max_iterations =
            raw.parse().map_err(|_| format!("--max-iter: {raw:?} is not a number"))?;
    }

    let network = AcOpfNetwork::from_pgm(input, &data, 50.0, &options)
        .map_err(|e| e.to_string())?;
    let n_buses = network.n_buses;
    let demand: f64 = network.p_load.iter().sum::<f64>() * network.base_mva;
    let opf = AcOpf::build(network, options).map_err(|e| e.to_string())?;
    let result = opf.solve().map_err(|e| e.to_string())?;

    if result.status != OptStatus::Optimal {
        return Err(format!(
            "no optimal dispatch ({:?}), largest constraint violation {:.3e} pu. AC-OPF is \
             nonconvex, so this means the method could not find a first-order point from \
             its starting guess — not that none exists",
            result.status, result.violation
        ));
    }

    let network = opf.network();
    println!(
        "{n_buses} bus(es), {} generator(s), {:.1} MW of demand",
        network.generators.len(),
        demand
    );
    println!("total cost: {:.2} $/h", result.objective);
    println!(
        "converged in {} iterations, largest constraint violation {:.2e} pu",
        result.iterations, result.violation
    );
    // Said plainly, because it is the difference between this and the DC form.
    println!("this is a local optimum — AC-OPF is nonconvex and no solver certifies more");

    println!("\ndispatch (MW, MVAr):");
    for (k, unit) in network.generators.iter().enumerate() {
        println!(
            "  generator {:>4}: P {:>10.3}   Q {:>10.3}",
            unit.index, result.p_gen[k], result.q_gen[k]
        );
    }

    println!("\nvoltage magnitudes (pu):");
    let lowest = result
        .magnitudes
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, v)| (i, *v));
    let highest = result
        .magnitudes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, v)| (i, *v));
    if let (Some((lo_bus, lo)), Some((hi_bus, hi))) = (lowest, highest) {
        println!("  lowest  bus {lo_bus:>4}: {lo:.4}");
        println!("  highest bus {hi_bus:>4}: {hi:.4}");
    }

    println!("\nlocational marginal price ($/MWh):");
    let min = result.lmp_p.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = result.lmp_p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-6 {
        println!("  uniform at {min:.4} — nothing is congested");
    } else {
        for (bus, price) in result.lmp_p.iter().enumerate() {
            println!("  bus {bus:>4}: {price:>10.4}");
        }
        println!("  spread: {:.4} (the congestion)", max - min);
    }

    let base = network.base_mva;
    let mut loaded: Vec<(usize, f64, f64)> = Vec::new();
    for (flat, rate) in network.limits.iter() {
        let Some(rate) = rate else { continue };
        let Some(&(p, q)) = result.flows.get(*flat) else { continue };
        let apparent = (p * p + q * q).sqrt() / base;
        if apparent > 0.99 * rate {
            loaded.push((*flat, apparent * base, rate * base));
        }
    }
    loaded.sort_by(|a, b| (b.1 / b.2).total_cmp(&(a.1 / a.2)));
    if loaded.is_empty() {
        println!("\nno branch above 99% of its rating");
    } else {
        println!("\nbranches at or near their rating:");
        for (branch, flow, rate) in loaded {
            println!(
                "  branch {branch:>4}: |S| {flow:>10.3} of {rate:>9.3} MVA ({:.1}%)",
                flow / rate * 100.0
            );
        }
    }

    Ok(())
}

#[cfg(not(feature = "opf"))]
fn run_opf(_path: &str, _flags: &[String]) -> Result<(), String> {
    Err("this build has no OPF solver; rebuild with `cargo build --features opf`".to_string())
}

/// Solves an AC power flow over the PGM document at `path` and differentiates
/// it.
///
/// Offers both directions the library does, because they answer different
/// questions: `--dp` and friends ask "this moves — what responds?", `--watch`
/// asks "this is overloaded — what would relieve it?".
fn run_sensitivity(path: &str, flags: &[String]) -> Result<(), String> {
    let s_base_va = 1e6;
    let raw = fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let input: PgmInput =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?;
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, s_base_va, 50.0);

    let terminal = match flag_value(flags, "--terminal")?.as_deref() {
        None | Some("from") => Terminal::From,
        Some("to") => Terminal::To,
        Some(other) => return Err(format!("--terminal: expected from or to, got {other:?}")),
    };

    // The linearization is only as meaningful as the point it is taken at, so
    // this refuses to differentiate a solve that did not converge rather than
    // printing derivatives of nothing.
    let report = run_power_flow(buses, &lines, &transformers, &[], PowerFlowOptions::default());
    if report.stats.status != SolveStatus::Converged {
        return Err(format!(
            "the power flow did not converge ({:?}), so there is no operating point to \
             differentiate",
            report.stats.status
        ));
    }
    let buses = report.buses;
    let n_branches = lines.len() + transformers.len();

    let ybus = {
        let mut y = build_ybus(buses.len(), &lines, &transformers);
        stamp_shunts(&mut y, &[]);
        y.finish()
    };
    let sensitivity = AcSensitivity::new(&buses, &ybus, &lines, &transformers)
        .ok_or("the Jacobian is singular at the solved point")?;

    println!(
        "{} bus(es), {} branch(es); differentiated at the converged point ({} iteration(s))",
        buses.len(),
        n_branches,
        report.stats.mismatch_history.len()
    );
    println!("branch indices are lines first ({}), then transformers", lines.len());

    let mut did_something = false;

    for (flag, make) in [
        ("--dp", &Variable::ActiveInjection as &dyn Fn(usize) -> Variable),
        ("--dq", &Variable::ReactiveInjection),
    ] {
        if let Some(raw) = flag_value(flags, flag)? {
            let bus = parse_index(&raw, flag, buses.len())?;
            print_forward(&sensitivity, make(bus), terminal, flag, bus, &buses)?;
            did_something = true;
        }
    }
    for (flag, make) in [
        ("--dk", &Variable::TransformerRatio as &dyn Fn(usize) -> Variable),
        ("--dalpha", &Variable::PhaseShift),
    ] {
        if let Some(raw) = flag_value(flags, flag)? {
            let branch = parse_index(&raw, flag, n_branches)?;
            print_forward(&sensitivity, make(branch), terminal, flag, branch, &buses)?;
            did_something = true;
        }
    }

    if let Some(raw) = flag_value(flags, "--watch")? {
        let branch = parse_index(&raw, "--watch", n_branches)?;
        let row = sensitivity
            .function_row(Function::BranchActivePower { branch, terminal })
            .ok_or("the adjoint solve is singular")?;

        println!("\nwhat moves the active flow on branch {branch} ({terminal:?} terminal):");
        println!("  by injection (p.u. per p.u.):");
        for bus in 0..buses.len() {
            if row.d_active[bus].abs() < 1e-9 && row.d_reactive[bus].abs() < 1e-9 {
                continue;
            }
            println!(
                "    bus {bus:>4}: dP {:>10.5}, dQ {:>10.5}",
                row.d_active[bus], row.d_reactive[bus]
            );
        }
        let taps: Vec<usize> = (0..n_branches)
            .filter(|&b| row.d_ratio[b].abs() > 1e-9 || row.d_phase[b].abs() > 1e-9)
            .collect();
        if taps.is_empty() {
            println!("  by tap: none (no transformer affects this flow)");
        } else {
            println!("  by tap (p.u. per unit ratio, p.u. per radian):");
            for b in taps {
                println!(
                    "    branch {b:>4}: dk {:>10.5}, dalpha {:>10.5}",
                    row.d_ratio[b], row.d_phase[b]
                );
            }
        }
        did_something = true;
    }

    if !did_something {
        return Err(
            "nothing to do — pass at least one of --dp, --dq, --dk, --dalpha or --watch"
                .to_string(),
        );
    }
    Ok(())
}

/// Prints one forward column: how every branch flow and bus voltage responds to
/// a single variable.
fn print_forward(
    sensitivity: &AcSensitivity,
    variable: Variable,
    terminal: Terminal,
    flag: &str,
    index: usize,
    buses: &[gridoxide::types::Bus],
) -> Result<(), String> {
    let flows = sensitivity
        .branch_response(variable, terminal)
        .ok_or_else(|| format!("{flag} {index}: the solve is singular"))?;
    let state = sensitivity
        .state_response(variable)
        .ok_or_else(|| format!("{flag} {index}: the solve is singular"))?;

    println!("\n{flag} {index}: branch flow response ({terminal:?} terminal, p.u. per unit):");
    for (b, (dp, dq)) in flows.iter().enumerate() {
        if dp.abs() < 1e-9 && dq.abs() < 1e-9 {
            continue;
        }
        println!("  branch {b:>4}: dP {dp:>10.5}, dQ {dq:>10.5}");
    }

    println!("{flag} {index}: bus voltage response:");
    for bus in 0..buses.len() {
        if state.d_theta[bus].abs() < 1e-9 && state.d_vmag[bus].abs() < 1e-9 {
            continue;
        }
        println!(
            "  bus {bus:>4}: d|V| {:>10.5} p.u., dtheta {:>10.5} rad",
            state.d_vmag[bus], state.d_theta[bus]
        );
    }
    Ok(())
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

/// `gridoxide security <network> --crac <crac.json> [--json]`
///
/// Returns `Ok(true)` when every margin is non-negative. The caller turns that
/// into an exit code, so this can gate a pipeline: a secure network exits 0, an
/// insecure one exits 1, and a *broken invocation* exits 2. Conflating the last
/// two is how a study silently passes because the CRAC failed to load.
#[cfg(all(feature = "rao", any(feature = "ucte", feature = "iidm")))]
fn run_security(path: &str, flags: &[String]) -> Result<bool, String> {
    use gridoxide::rao::{crac_json, evaluate, Network, Resolution};

    let crac_path = flag_value(flags, "--crac")?
        .ok_or("security needs --crac <crac.json>")?;
    let as_json = flags.iter().any(|f| f == "--json");

    let network = load_network_for_security(path)?;
    let text = std::fs::read_to_string(&crac_path)
        .map_err(|e| format!("reading {crac_path}: {e}"))?;
    // gridoxide's own companion document first, then OpenRAO's format. Trying
    // ours first means a malformed companion reports its own error rather than
    // the less helpful "document is not a CRAC".
    let (crac, report) = match gridoxide::rao::Crac::from_json(&text) {
        Ok(crac) => (crac, crac_json::CracReport::default()),
        Err(_) => crac_json::parse(&text).map_err(|e| e.to_string())?,
    };

    let resolution =
        Resolution::with_buses(&crac, &network.branch_ids, &network.bus_ids);
    let view = Network {
        buses: &network.buses,
        lines: &network.lines,
        transformers: &network.transformers,
        branch_ids: &network.branch_ids,
        bus_ids: &network.bus_ids,
        initially_open: &network.initially_open,
        bus_countries: &network.bus_countries,
        tap_changers: &network.tap_changers,
        base_mva: network.base_mva,
    };
    let result = evaluate(&crac, &view, &resolution);

    if as_json {
        println!("{}", security_json(&crac, &result, &resolution));
        return Ok(result.is_secure());
    }

    for note in &network.notes {
        println!("note: {note}");
    }
    if let Some(version) = &report.version {
        println!("crac format version {version}");
    }
    if !resolution.is_complete() {
        println!(
            "warning: {} element(s) not found in the network: {}",
            resolution.unresolved.len(),
            resolution.unresolved.join(", ")
        );
    }
    if !result.skipped.is_empty() {
        println!("warning: {} CNEC(s) skipped for want of their element", result.skipped.len());
    }

    for perimeter in &result.perimeters {
        let instant = &crac.instants[perimeter.state.instant].id;
        let contingency = perimeter
            .state
            .contingency
            .map(|c| crac.contingencies[c].id.as_str())
            .unwrap_or("base case");
        let severed = if perimeter.severed { "  [not screenable; re-solved]" } else { "" };
        println!(
            "\n{instant} / {contingency}: {} monitored, worst margin {:.1} MW{severed}",
            perimeter.cnecs.len(),
            perimeter.min_margin().unwrap_or(f64::NAN)
        );
        for violation in perimeter.violations() {
            println!(
                "  OVERLOAD {:<28} flow {:>9.1}  limit {:>9.1}  by {:>8.1} MW",
                crac.flow_cnecs[violation.cnec].id,
                violation.flow_mw,
                violation.limit_mw,
                -violation.margin_mw
            );
        }
    }

    let violations = result.violations().count();
    println!(
        "\n{}: {violations} overload(s) across {} perimeter(s)",
        if result.is_secure() { "SECURE" } else { "INSECURE" },
        result.perimeters.len()
    );
    Ok(result.is_secure())
}

/// The subset of an import `run_security` needs, so the two importers can be
/// handled without a trait.
#[cfg(all(feature = "rao", any(feature = "ucte", feature = "iidm")))]
struct SecurityNetwork {
    buses: Vec<gridoxide::types::Bus>,
    lines: Vec<gridoxide::types::Line>,
    transformers: Vec<gridoxide::types::Transformer>,
    branch_ids: Vec<String>,
    /// Bus labels, so a redispatch's generators and loads resolve.
    bus_ids: Vec<String>,
    /// Branches the file says are out of service.
    initially_open: Vec<usize>,
    /// ISO country per bus, for the search's "skip actions far from the most
    /// limiting element" filter. Empty when the importer does not state them.
    bus_countries: Vec<Option<String>>,
    /// Tap changers, so a CRAC that omits its own tap table still works.
    tap_changers: Vec<Option<gridoxide::types::TapChanger>>,
    /// Shunt admittances. Only the AC re-validation stage reads them; the DC
    /// search has no use for them.
    shunts: Vec<gridoxide::network::ShuntAdm>,
    base_mva: f64,
    notes: Vec<String>,
}

#[cfg(all(feature = "rao", any(feature = "ucte", feature = "iidm")))]
fn load_network_for_security(path: &str) -> Result<SecurityNetwork, String> {
    let lower = path.to_ascii_lowercase();
    #[cfg(feature = "iidm")]
    if lower.ends_with(".xiidm") || lower.ends_with(".xml") {
        let n = gridoxide::iidm::read(path).map_err(|e| e.to_string())?;
        return Ok(SecurityNetwork {
            buses: n.buses,
            lines: n.lines,
            transformers: n.transformers,
            branch_ids: n.branch_ids,
            bus_ids: n.bus_labels,
            // The IIDM importer omits disconnected branches rather than keeping
            // them openable, so there is nothing to seed here yet.
            initially_open: Vec::new(),
            // IIDM states countries on substations, which `iidm.rs` skips, so
            // the filter that reads this stays off for IIDM networks.
            bus_countries: Vec::new(),
            tap_changers: n.tap_changers,
            shunts: n.shunts,
            base_mva: n.base_mva,
            notes: n.notes,
        });
    }
    #[cfg(feature = "ucte")]
    if lower.ends_with(".uct") || lower.ends_with(".ucte") {
        let n = gridoxide::ucte::read(path).map_err(|e| e.to_string())?;
        return Ok(SecurityNetwork {
            buses: n.buses,
            lines: n.lines,
            transformers: n.transformers,
            branch_ids: n.branch_ids,
            bus_ids: n.node_codes,
            initially_open: n.initially_open,
            bus_countries: n.bus_countries,
            tap_changers: n.tap_changers,
            shunts: n.shunts,
            base_mva: n.base_mva,
            notes: n.notes,
        });
    }
    Err(format!(
        "cannot tell what `{path}` is; expected a .uct or .xiidm file"
    ))
}

#[cfg(all(feature = "rao", any(feature = "ucte", feature = "iidm")))]
fn security_json(
    crac: &gridoxide::rao::Crac,
    result: &gridoxide::rao::SecurityResult,
    resolution: &gridoxide::rao::Resolution,
) -> String {
    let mut perimeters = String::new();
    for (i, p) in result.perimeters.iter().enumerate() {
        if i > 0 {
            perimeters.push(',');
        }
        let mut cnecs = String::new();
        for (j, c) in p.cnecs.iter().enumerate() {
            if j > 0 {
                cnecs.push(',');
            }
            cnecs.push_str(&format!(
                "\n    {{\"cnec\": {:?}, \"flow_mw\": {}, \"limit_mw\": {}, \"margin_mw\": {}}}",
                crac.flow_cnecs[c.cnec].id, c.flow_mw, c.limit_mw, c.margin_mw
            ));
        }
        perimeters.push_str(&format!(
            "\n  {{\"instant\": {:?}, \"contingency\": {}, \"severed\": {}, \"cnecs\": [{cnecs}]}}",
            crac.instants[p.state.instant].id,
            match p.state.contingency {
                Some(c) => format!("{:?}", crac.contingencies[c].id),
                None => "null".to_string(),
            },
            p.severed
        ));
    }
    format!(
        "{{\n \"secure\": {},\n \"min_margin_mw\": {},\n \"unresolved\": {},\n \"skipped_cnecs\": {},\n \"perimeters\": [{perimeters}\n ]\n}}",
        result.is_secure(),
        result.min_margin().map(|m| m.to_string()).unwrap_or("null".into()),
        resolution.unresolved.len(),
        result.skipped.len()
    )
}

#[cfg(not(all(feature = "rao", any(feature = "ucte", feature = "iidm"))))]
fn run_security(_path: &str, _flags: &[String]) -> Result<bool, String> {
    Err("security needs the `rao` feature and an importer (`ucte` or `iidm`); \
         rebuild with `--features rao,ucte`"
        .to_string())
}

/// `gridoxide rao <network> --crac <crac.json> [--depth N] [--json]`
///
/// Each state the CRAC defines is optimized independently: the actions
/// available there are searched, and the range actions re-optimized under each
/// candidate. States are *not* chained — a curative perimeter here is solved as
/// if the preventive one had done nothing, which is the multi-perimeter work
/// still to come. The output says so rather than leaving it to be assumed.
#[cfg(all(feature = "rao", any(feature = "ucte", feature = "iidm")))]
fn run_rao(path: &str, flags: &[String]) -> Result<bool, String> {
    use gridoxide::opf::ipm::IpmSolver;
    use gridoxide::rao::{crac_json, Network, Resolution, SearchOptions};

    let crac_path = flag_value(flags, "--crac")?.ok_or("rao needs --crac <crac.json>")?;
    let depth = match flag_value(flags, "--depth")? {
        Some(value) => value.parse::<usize>().map_err(|_| format!("bad --depth `{value}`"))?,
        None => 2,
    };
    let as_json = flags.iter().any(|f| f == "--json");
    let validate_ac = flags.iter().any(|f| f == "--validate-ac");

    let network = load_network_for_security(path)?;
    let text = std::fs::read_to_string(&crac_path)
        .map_err(|e| format!("reading {crac_path}: {e}"))?;
    let crac = match gridoxide::rao::Crac::from_json(&text) {
        Ok(crac) => crac,
        Err(_) => crac_json::parse(&text).map_err(|e| e.to_string())?.0,
    };

    let resolution =
        Resolution::with_buses(&crac, &network.branch_ids, &network.bus_ids);
    let view = Network {
        buses: &network.buses,
        lines: &network.lines,
        transformers: &network.transformers,
        branch_ids: &network.branch_ids,
        bus_ids: &network.bus_ids,
        initially_open: &network.initially_open,
        bus_countries: &network.bus_countries,
        tap_changers: &network.tap_changers,
        base_mva: network.base_mva,
    };
    let options = SearchOptions { max_depth: depth, ..Default::default() };

    let mut solver = IpmSolver::new();
    let plan = gridoxide::rao::run(&crac, &view, &resolution, &mut solver, &options);

    // The search ran on DC. Asking AC whether it agrees is a separate stage
    // that never changes the plan — it only decides whether to believe it.
    let validation = validate_ac.then(|| {
        use gridoxide::rao::evaluate::AcOptions;
        use gridoxide::rao::validate::{validate, ValidationOptions};
        let options = ValidationOptions {
            ac: AcOptions { shunts: &network.shunts, ..Default::default() },
            ..Default::default()
        };
        validate(&crac, &view, &resolution, &plan, &options)
    });

    if as_json {
        println!("{}", rao_json(&crac, &plan, validation.as_ref()));
        return Ok(plan.is_secure() && validation.as_ref().is_none_or(|v| v.is_accepted()));
    }

    if !resolution.is_complete() {
        println!(
            "warning: {} element(s) not found in the network: {}",
            resolution.unresolved.len(),
            resolution.unresolved.join(", ")
        );
    }
    if !plan.pulled_forward.is_empty() {
        // This changes the answer rather than just organising the work, so it
        // is reported rather than left to be inferred from the perimeter list.
        println!(
            "note: {} curative CNEC(s) have no curative action and are secured preventively",
            plan.pulled_forward.len()
        );
    }

    println!(
        "\npreventive perimeter ({} state(s)): {:.1} -> {:.1} MW ({:+.1}), {} leaf/leaves",
        plan.preventive.states.len(),
        plan.preventive.initial_margin_mw,
        plan.preventive.final_margin_mw,
        plan.preventive.improvement(),
        plan.preventive.leaves
    );
    print_actions(&crac, &plan.preventive);

    for scenario in &plan.scenarios {
        for perimeter in &scenario.perimeters {
            let instant = perimeter
                .states
                .first()
                .map(|s| crac.instants[s.instant].id.as_str())
                .unwrap_or("?");
            println!(
                "\n{} after {}: {:.1} -> {:.1} MW ({:+.1}), {} leaf/leaves",
                instant,
                crac.contingencies[scenario.contingency].id,
                perimeter.initial_margin_mw,
                perimeter.final_margin_mw,
                perimeter.improvement(),
                perimeter.leaves
            );
            print_actions(&crac, perimeter);
        }
    }

    println!(
        "\n{}: worst margin {:.1} MW (was {:.1})",
        if plan.is_secure() { "SECURE" } else { "INSECURE" },
        plan.final_margin_mw,
        plan.initial_margin_mw
    );

    let Some(validation) = validation else { return Ok(plan.is_secure()) };

    use gridoxide::rao::validate::Verdict;
    println!("\nAC re-validation: worst margin {:.1} MW", validation.ac_margin_mw);
    for p in &validation.perimeters {
        let instant = p
            .states
            .first()
            .map(|s| crac.instants[s.instant].id.as_str())
            .unwrap_or("?");
        let verdict = match &p.verdict {
            Verdict::Accepted => "ok".to_string(),
            Verdict::Diverged => "DIVERGED".to_string(),
            Verdict::Insecure { margin_mw } => format!("INSECURE ({margin_mw:.1} MW)"),
            Verdict::Regressed { by_mw } => format!("REGRESSED ({by_mw:.1} MW)"),
        };
        println!(
            "  {instant:<12} dc {:>8.1} MW   ac {:>8.1} MW   {verdict}",
            p.dc_margin_mw, p.ac_margin_mw
        );
    }
    if !validation.is_accepted() {
        println!("\nREJECTED: the AC check does not support the plan");
    }
    Ok(plan.is_secure() && validation.is_accepted())
}

#[cfg(all(feature = "rao", any(feature = "ucte", feature = "iidm")))]
fn print_actions(crac: &gridoxide::rao::Crac, perimeter: &gridoxide::rao::PerimeterPlan) {
    for &action in &perimeter.network_actions {
        println!("  APPLY  {}", crac.network_actions[action].id);
    }
    for setpoint in perimeter.setpoints.iter().filter(|s| s.moved()) {
        match setpoint.tap {
            Some(tap) => println!(
                "  SET    {} to tap {tap} ({:.3} deg, was {:.3})",
                crac.range_actions[setpoint.action].id, setpoint.value, setpoint.initial
            ),
            None => println!(
                "  SET    {} to {:.1} MW (was {:.1})",
                crac.range_actions[setpoint.action].id, setpoint.value, setpoint.initial
            ),
        }
    }
    if perimeter.network_actions.is_empty() && !perimeter.setpoints.iter().any(|s| s.moved()) {
        println!("  (nothing available helps)");
    }
}

#[cfg(all(feature = "rao", any(feature = "ucte", feature = "iidm")))]
fn rao_json(
    crac: &gridoxide::rao::Crac,
    plan: &gridoxide::rao::Plan,
    validation: Option<&gridoxide::rao::validate::Validation>,
) -> String {
    let one = |perimeter: &gridoxide::rao::PerimeterPlan, contingency: Option<usize>| -> String {
        let actions: Vec<String> = perimeter
            .network_actions
            .iter()
            .map(|&a| format!("{:?}", crac.network_actions[a].id))
            .collect();
        let setpoints: Vec<String> = perimeter
            .setpoints
            .iter()
            .filter(|s| s.moved())
            .map(|s| {
                format!(
                    "{{\"action\": {:?}, \"value\": {}, \"tap\": {}}}",
                    crac.range_actions[s.action].id,
                    s.value,
                    s.tap.map(|t| t.to_string()).unwrap_or("null".into())
                )
            })
            .collect();
        let instants: Vec<String> = perimeter
            .states
            .iter()
            .map(|s| format!("{:?}", crac.instants[s.instant].id))
            .collect();
        format!(
            "{{\"instants\": [{}], \"contingency\": {}, \"initial_margin_mw\": {},              \"final_margin_mw\": {}, \"leaves\": {}, \"network_actions\": [{}],              \"setpoints\": [{}]}}",
            instants.join(", "),
            match contingency {
                Some(c) => format!("{:?}", crac.contingencies[c].id),
                None => "null".to_string(),
            },
            perimeter.initial_margin_mw,
            perimeter.final_margin_mw,
            perimeter.leaves,
            actions.join(", "),
            setpoints.join(", ")
        )
    };

    let mut body = vec![format!("\n  {}", one(&plan.preventive, None))];
    for scenario in &plan.scenarios {
        for perimeter in &scenario.perimeters {
            body.push(format!("\n  {}", one(perimeter, Some(scenario.contingency))));
        }
    }
    // Absent unless asked for, rather than present and null: a consumer that
    // never passed `--validate-ac` should not have to distinguish "AC said
    // nothing" from "AC was never run".
    let ac = match validation {
        None => String::new(),
        Some(v) => {
            use gridoxide::rao::validate::Verdict;
            let rows: Vec<String> = v
                .perimeters
                .iter()
                .map(|p| {
                    let verdict = match &p.verdict {
                        Verdict::Accepted => "accepted".to_string(),
                        Verdict::Diverged => "diverged".to_string(),
                        Verdict::Insecure { .. } => "insecure".to_string(),
                        Verdict::Regressed { .. } => "regressed".to_string(),
                    };
                    format!(
                        "\n   {{\"instants\": [{}], \"dc_margin_mw\": {}, \"ac_margin_mw\": {}, \"verdict\": {:?}}}",
                        p.states
                            .iter()
                            .map(|s| format!("{:?}", crac.instants[s.instant].id))
                            .collect::<Vec<_>>()
                            .join(", "),
                        p.dc_margin_mw,
                        p.ac_margin_mw,
                        verdict
                    )
                })
                .collect();
            format!(
                ",\n \"ac_validation\": {{\"accepted\": {}, \"ac_margin_mw\": {}, \"perimeters\": [{}\n  ]}}",
                v.is_accepted(),
                v.ac_margin_mw,
                rows.join(",")
            )
        }
    };

    format!(
        "{{\n \"secure\": {},\n \"initial_margin_mw\": {},\n \"final_margin_mw\": {},\n          \"pulled_forward\": {},\n \"perimeters\": [{}\n ]{}\n}}",
        plan.is_secure(),
        plan.initial_margin_mw,
        plan.final_margin_mw,
        plan.pulled_forward.len(),
        body.join(","),
        ac
    )
}

#[cfg(not(all(feature = "rao", any(feature = "ucte", feature = "iidm"))))]
fn run_rao(_path: &str, _flags: &[String]) -> Result<bool, String> {
    Err("rao needs the `rao` feature and an importer (`ucte` or `iidm`); \
         rebuild with `--features rao,ucte`"
        .to_string())
}
