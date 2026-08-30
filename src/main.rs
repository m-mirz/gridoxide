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
use gridoxide::network::{build_ybus, stamp_shunts};
use gridoxide::pgm::{pgm_to_buses_and_branches, PgmInput};
use gridoxide::{run_power_flow, run_power_flow_analysis};
use gridoxide::se::bad_data::{self, Candidates};
use gridoxide::se::constraints::Constraints;
use gridoxide::se::jacobian::StateLayout;
use gridoxide::se::nr::{estimate, linear_start, SeMethod, SeOptions, SeStatus};
use gridoxide::se::observability;
use gridoxide::shortcircuit::{
    short_circuit_from_pgm, SequenceValue, ShortCircuitOptions, VoltageScaling,
};
use gridoxide::solver::{PowerFlowOptions, SolveStatus};

const USAGE: &str = "\
usage:
  gridoxide                     run the bundled power-flow demo
  gridoxide solve <network> [--control-taps] [--enforce-q-limits] [--dispatch]
                  [--control-remote-voltage]
                  [--distribute-slack | --area-interchange]
                  [--max-outer <n>] [--max-tap-shift <n>]
                  [--tol <t>] [--max-iter <n>]
                                run an AC power flow, optionally with the outer
                                loops that make a control a control rather than
                                a constant: on-load tap changers holding a
                                voltage or a flow, generators respecting their
                                reactive limits, and the system imbalance shared
                                across generators instead of dumped on one
                                slack. All three compose; without any of them
                                this is an ordinary Newton solve.
                                --control-remote-voltage moves a machine that
                                regulates a bus other than its own onto that
                                machine: without it the reactive power appears
                                at the bus being held instead of at the machine,
                                so the reactive flow between them is missing. It
                                changes answers, which is why it is opt-in.
                                CGMES only.
                                --dispatch additionally attributes each
                                voltage-controlled bus's reactive power to the
                                individual machines holding it, and names any
                                machine that is at its own limit while the bus
                                as a whole is not. CGMES only — no other format
                                states per-machine reactive capability.
                                --area-interchange stands in distributed
                                slack's place — it generalizes it — and holds
                                each control area's net export at what the file
                                says it agreed to. CGMES supplies both the areas
                                (ControlArea/TieFlow) and the schedule
                                (netInterchange); UCTE supplies areas from its
                                ##Z country codes and states no schedule, so
                                every target is zero there.
                                The network may be CGMES (a directory, or the
                                profile .xml files), UCTE-DEF (.uct) or IIDM
                                (.xiidm). Needs the matching importer feature.
  gridoxide estimate <path> [--iterative-linear] [--asymmetric]
                                run state estimation over a PGM JSON document
                                containing voltage, power or current sensors.
                                The default method is Newton-Raphson; the flag
                                selects the faster, less exact linearized one.
                                --asymmetric estimates in the *phase domain*:
                                three buses per node rather than one, with each
                                asym sensor reading its own phase instead of a
                                balanced total. That is what makes an unbalanced
                                network estimable rather than approximable, and
                                on a distribution feeder the unbalance is the
                                question rather than a refinement. It refuses,
                                by name, the components the three-phase model
                                does not carry.
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
  gridoxide qv <path> [--bus <i> | --weakest <n>] [--v-max <v>] [--v-min <v>]
               [--step <s>] [--no-refine] [--base-mva <m>] [--curve]
                                trace a bus's Q-V curve: how much reactive
                                support holding it at a given voltage takes, and
                                how much margin there is before no setpoint is
                                reachable. The minimum of the curve is the
                                reactive margin, in MVAr.
                                --weakest ranks every bus by that margin, which
                                is the usual way to ask the question; --bus
                                traces one and prints it.
                                Complements `continuation`, which asks how much
                                further the whole system can be loaded. The two
                                measure related but different things and are
                                known not to agree on which bus is weakest —
                                that is why both exist.
                                Reads a PGM JSON document.
  gridoxide continuation <path> [--target-lambda <l> | --lower-branch]
                     [--step <s>] [--max-steps <n>] [--enforce-q-limits]
                     [--parametrization local|natural|arclength]
                     [--pickup <i,j,k>] [--buses <i,j,k>]
                     [--base-mva <m>] [--tol <t>] [--max-iter <n>] [--curve]
                                trace the P-V curve to the point of voltage
                                collapse: how much further this network can be
                                loaded, which bus gives way first, and where
                                each generator runs out of reactive capability
                                on the way. Reports the loadability limit as a
                                multiple of the base load (lambda) and as MW.
                                The answer is a property of the *direction*
                                loading grows in, not of the network alone;
                                the default grows every net consumer at constant
                                power factor and leaves the slack to pick it up.
                                --pickup shares that pickup over the named buses
                                instead; --buses stresses only the named ones.
                                --enforce-q-limits is worth having: reactive
                                limits usually decide where the nose is, and
                                the exact lambda each machine saturates at is
                                located rather than rounded to a step.
                                Reads a PGM JSON document.
  gridoxide dynamics <path> --modes <n> [--modes-freq <Hz>]
                     [--modes-damping <zeta>] [--modes-near <re>,<im>]
                     [--modes-sensitivity]
                                report the n least-damped modes of the
                                linearized system instead of running it: which
                                oscillations exist, how fast each decays, and
                                which machines take part in it. Answers a
                                question no single run can, and is the only
                                reliable way to find a *negatively* damped mode
                                — a run finds one only if the disturbance
                                happened to excite it.
                                Below two thousand states this computes *every*
                                mode. Past that it switches to a sparse method
                                that computes only the n nearest a point in the
                                complex plane, and says so — name that point
                                with --modes-freq (a frequency in Hz, with
                                --modes-damping as a ratio) or with
                                --modes-near. Naming any of them selects the
                                sparse method at any size, which is how to ask
                                about one band of a large system.
                                --modes-sensitivity adds, for the least-damped
                                mode, how far it moves per unit of each machine
                                parameter — the question behind deciding what
                                to change. Only parameters the equilibrium does
                                not depend on are offered; see the docs for why
                                that is a restriction and not an omission.
  gridoxide dynamics <path> [--stop <t>] [--step <h>] [--tol <t>]
                     [--damping <n>] [--backend scalar|klu-native]
                     [--speed-voltages | --no-speed-voltages]
                     [--csv <out>] [--observe <substring>]
                                run an RMS (phasor-domain) dynamic simulation:
                                what the network does over *time* after a
                                disturbance, rather than at one instant. Reads a
                                gridoxide JSON document carrying a `dynamics`
                                section — the machines, their exciters,
                                governors and stabilizers, and the schedule of
                                faults, trips and load steps to apply.
                                Prints how each machine fared and what the
                                voltages did; --csv writes the whole trajectory,
                                and --observe narrows it to the columns whose
                                names contain a substring.
                                Everything starts from a solved power flow, so
                                the case must be one that solves; every device
                                is then initialized so that nothing moves until
                                the first event does.
                                --speed-voltages carries the rotor speed on the
                                machines' speed-voltage terms and writes their
                                swing equations in torque, which is what Dynawo
                                and Sauer & Pai do. Off by default: the omega
                                approximately one assumption is what makes the
                                phasor formulation coherent, and it is what the
                                closed-form gates are derived from. It is worth
                                0.6% of terminal power at a 0.9% speed
                                deviation — turning it on moves the answer.
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
        Some("solve") => match args.get(1) {
            Some(path) if !path.starts_with("--") => {
                if let Err(message) = run_solve(path, &args[2..]) {
                    eprintln!("error: {message}");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("error: solve needs a network path\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("estimate") => match args.get(1) {
            Some(path) => {
                let method = if args.iter().any(|a| a == "--iterative-linear") {
                    SeMethod::IterativeLinear
                } else {
                    SeMethod::NewtonRaphson
                };
                let phases = if args.iter().any(|a| a == "--asymmetric") { 3 } else { 1 };
                if let Err(message) = run_estimate(path, method, phases) {
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
        Some("qv") => match args.get(1) {
            Some(path) if !path.starts_with("--") => {
                if let Err(message) = run_qv_cli(path, &args[2..]) {
                    eprintln!("{message}");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("usage: gridoxide qv <path> [--bus <i> | --weakest <n>]");
                std::process::exit(1);
            }
        },
        Some("continuation") => match args.get(1) {
            Some(path) if !path.starts_with("--") => {
                if let Err(message) = run_continuation_cli(path, &args[2..]) {
                    eprintln!("{message}");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("usage: gridoxide continuation <path> [options]");
                std::process::exit(1);
            }
        },
        Some("dynamics") => match args.get(1) {
            Some(path) if !path.starts_with("--") => {
                if let Err(message) = run_dynamics_cli(path, &args[2..]) {
                    eprintln!("error: {message}");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("error: dynamics needs a document path\n\n{USAGE}");
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
fn run_estimate(path: &str, method: SeMethod, phases: usize) -> Result<(), String> {
    use gridoxide::se::case::{SeCase, PHASE_NAMES};

    let s_base_va = 1e6;
    let raw = fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let input: PgmInput =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?;

    let case = if phases == 3 {
        SeCase::from_pgm_3ph(&input, s_base_va, 50.0)
    } else {
        SeCase::from_pgm(&input, s_base_va, 50.0)
    }
    .map_err(|e| format!("{path}: {e}"))?;

    let nodes = case.nodes();
    let n_physical = case.phases * nodes.len();
    let SeCase { network: se_net, mut buses, measurements, .. } = case;

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
        "{} bus(es) ({}), {} measurement(s) after aggregation",
        buses.len(),
        if phases == 3 { "phase domain, three per node" } else { "symmetric, one per node" },
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
    for (id, idx) in &nodes {
        for phase in 0..phases {
            // A node owns one bus symmetrically and three in the phase domain,
            // laid out `3*node + phase`, which is the same arithmetic every
            // functional in the model uses.
            let bus = phases * idx + phase;
            let label = if phases == 3 {
                format!("node {id} phase {}", PHASE_NAMES[phase])
            } else {
                format!("node {id}")
            };
            println!(
                "  {label}: |V| = {:.6} p.u., angle = {:.6} deg",
                buses[bus].voltage_mag,
                buses[bus].voltage_ang.to_degrees()
            );
        }
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
    // Buses beyond the physical count are gridoxide's own synthesized ones — a
    // virtual slack bus per source, a star point per three-winding transformer
    // — and are expected to be unobservable when the source's own power is
    // unmeasured. Saying which is which avoids a false alarm. In the phase
    // domain the physical count is three per node, for the same reason.

    // The two lists overlap by design — a structurally unmeasured column is
    // also rank-deficient — so they are merged before printing rather than
    // reported twice.
    let mut undetermined: Vec<_> =
        obs.unobservable.iter().chain(&obs.structurally_unmeasured).copied().collect();
    undetermined.sort_by_key(|u| (u.bus, format!("{:?}", u.quantity)));
    undetermined.dedup();
    for unknown in undetermined {
        let kind = if unknown.bus >= n_physical { "synthesized" } else { "physical" };
        let where_ = if phases == 3 && unknown.bus < n_physical {
            format!(
                "bus {} (node index {}, phase {})",
                unknown.bus,
                unknown.bus / 3,
                PHASE_NAMES[unknown.bus % 3]
            )
        } else {
            format!("bus {}", unknown.bus)
        };
        println!("  undetermined: {kind} {where_} ({:?})", unknown.quantity);
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
/// Parses a comma-separated bus-index list, e.g. `--pickup 3,7,12`.
fn parse_index_list(raw: &str, flag: &str, limit: usize) -> Result<Vec<usize>, String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| parse_index(s, flag, limit))
        .collect()
}

fn parse_f64_flag(flags: &[String], flag: &str) -> Result<Option<f64>, String> {
    match flag_value(flags, flag)? {
        None => Ok(None),
        Some(raw) => raw
            .parse::<f64>()
            .map(Some)
            .map_err(|_| format!("{flag}: expected a number, got {raw:?}")),
    }
}

/// `gridoxide dynamics` — what the network does over time.
///
/// Everything else this binary offers answers a question about one instant.
/// This one integrates: the machines' rotors and fluxes, their controls, and
/// the network constraint, all as one implicit system.
///
/// The summary is deliberately about *machines* rather than about states. A
/// trajectory has hundreds of columns and almost none of them is the answer to
/// anything; whether each machine stayed in step, how far the frequency went
/// and how deep the voltage dipped are.
#[cfg(feature = "dynamics")]
fn run_dynamics_cli(path: &str, flags: &[String]) -> Result<(), String> {
    use gridoxide::dynamics::{json, run_dynamics, DynamicsOptions, DynamicsStatus};
    use gridoxide::solver::JacobianBackend;

    let mut document = json::read(path).map_err(|e| e.to_string())?;
    // The document states this; the flag overrides it, because it is a
    // modelling question a reader may want to ask both ways of the same case.
    if flags.iter().any(|f| f == "--speed-voltages") {
        document.dynamics.speed_voltages = true;
    }
    if flags.iter().any(|f| f == "--no-speed-voltages") {
        document.dynamics.speed_voltages = false;
    }
    let (mut system, events, relays) =
        document.build_with_relays().map_err(|e| e.to_string())?;

    // `--modes` asks a different question from a run, so it answers that one
    // and stops: not "what happens after this disturbance" but "what dynamic
    // character does this system have at all".
    if let Some(count) = flag_value(flags, "--modes")? {
        let count: usize = count
            .parse()
            .map_err(|_| format!("--modes: expected a count, got {count:?}"))?;
        return report_modes(&mut system, count, flags);
    }

    let backend = match flag_value(flags, "--backend")?.as_deref() {
        None | Some("scalar") => JacobianBackend::Scalar,
        Some("klu-native") => JacobianBackend::KluNative,
        Some(other) => {
            return Err(format!("--backend: expected scalar or klu-native, got {other:?}"))
        }
    };
    let options = DynamicsOptions {
        end_time: parse_f64_flag(flags, "--stop")?.unwrap_or(10.0),
        step: parse_f64_flag(flags, "--step")?.unwrap_or(0.005),
        tol: parse_f64_flag(flags, "--tol")?.unwrap_or(1e-9),
        damping_steps: match flag_value(flags, "--damping")? {
            None => 2,
            Some(raw) => raw
                .parse()
                .map_err(|_| format!("--damping: expected a count, got {raw:?}"))?,
        },
        events,
        relays,
        backend,
        ..Default::default()
    };

    println!(
        "{} bus(es), {} differential state(s), {} scheduled event(s)",
        system.n_bus(),
        system.n_states(),
        options.events.len()
    );
    // Worth printing: it is the number that says the case was initialized
    // consistently, and a nonzero one invalidates everything after it.
    println!("initial state derivative: {:.2e}", system.max_derivative());

    let started = std::time::Instant::now();
    let report = run_dynamics(&mut system, &options);
    let elapsed = started.elapsed();

    match report.status {
        DynamicsStatus::Completed => {
            println!(
                "\nran to {:.3} s in {} step(s), {} Newton iteration(s), {:.2?}",
                options.end_time, report.steps, report.newton_iterations, elapsed
            );
        }
        DynamicsStatus::NewtonFailed { time } => println!(
            "\nstopped at {time:.4} s: the step did not converge. A smaller --step usually \
             fixes this; an unstable system integrates perfectly well and simply diverges."
        ),
        DynamicsStatus::Singular { time } => println!(
            "\nstopped at {time:.4} s: the Jacobian was singular. An island with no machine \
             and no voltage reference does this."
        ),
    }
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
    for action in &report.relay_actions {
        println!(
            "relay {} saw its threshold at {:.4} s and acted at {:.4} s: {:?}",
            action.id, action.crossed_at, action.fired_at, action.action
        );
    }

    // Per machine: did it stay in step, and where did its speed go?
    let trajectory = &report.trajectory;
    let machines: Vec<String> = trajectory
        .names
        .iter()
        .filter(|n| n.ends_with(".omega"))
        .map(|n| n.trim_end_matches(".omega").to_string())
        .collect();
    if !machines.is_empty() {
        println!("\n{:<14}  {:>12}  {:>12}  {:>12}", "machine", "min speed", "max speed", "swing");
        for id in &machines {
            let omega = trajectory.series(&format!("{id}.omega")).unwrap_or_default();
            let delta = trajectory.series(&format!("{id}.delta")).unwrap_or_default();
            let (lo, hi) = omega.iter().fold((f64::MAX, f64::MIN), |(l, h), w| (l.min(*w), h.max(*w)));
            let swing = delta
                .iter()
                .fold((f64::MAX, f64::MIN), |(l, h), d| (l.min(*d), h.max(*d)));
            println!(
                "{id:<14}  {lo:>12.6}  {hi:>12.6}  {:>11.4} rad",
                swing.1 - swing.0
            );
        }
    }

    let voltages: Vec<&String> = trajectory.names.iter().filter(|n| n.ends_with(".vmag")).collect();
    if !voltages.is_empty() {
        let mut lowest = (f64::MAX, String::new(), 0.0f64);
        for name in &voltages {
            let series = trajectory.series(name).unwrap_or_default();
            for (k, v) in series.iter().enumerate() {
                if *v < lowest.0 {
                    lowest = (*v, (*name).clone(), trajectory.time[k]);
                }
            }
        }
        println!(
            "\nlowest voltage {:.4} pu at {}, t = {:.4} s",
            lowest.0, lowest.1, lowest.2
        );
    }

    if let Some(out) = flag_value(flags, "--csv")? {
        let filter = flag_value(flags, "--observe")?;
        let keep: Vec<usize> = (0..trajectory.names.len())
            .filter(|&k| match &filter {
                None => true,
                Some(pattern) => trajectory.names[k].contains(pattern.as_str()),
            })
            .collect();
        if keep.is_empty() {
            return Err(format!(
                "--observe {:?} matched none of the {} columns",
                filter.unwrap_or_default(),
                trajectory.names.len()
            ));
        }
        let mut text = String::from("time");
        for &k in &keep {
            text.push(',');
            text.push_str(&trajectory.names[k]);
        }
        text.push('\n');
        for (row, t) in trajectory.rows.iter().zip(trajectory.time.iter()) {
            text.push_str(&format!("{t:.6}"));
            for &k in &keep {
                text.push_str(&format!(",{:.9}", row[k]));
            }
            text.push('\n');
        }
        fs::write(&out, text).map_err(|e| format!("writing {out}: {e}"))?;
        println!("wrote {} column(s) x {} row(s) to {out}", keep.len(), trajectory.rows.len());
    }

    Ok(())
}

#[cfg(not(feature = "dynamics"))]
fn run_dynamics_cli(_path: &str, _flags: &[String]) -> Result<(), String> {
    Err("this build has no RMS dynamics; rebuild with `cargo build --features dynamics`"
        .to_string())
}

/// `gridoxide dynamics --modes` — the modes of the linearized system.
#[cfg(feature = "dynamics")]
fn report_modes(
    system: &mut gridoxide::dynamics::DynamicSystem,
    count: usize,
    flags: &[String],
) -> Result<(), String> {
    use gridoxide::dynamics::smallsignal::{self, Method, SmallSignalOptions};

    // Naming *where* to look selects the sparse method, whatever the size:
    // asking about one band is a different question from asking for everything,
    // and a caller who asks it should get it answered rather than overruled.
    let aimed = parse_shift(flags)?;
    let result = match aimed {
        Some(shift) => {
            let opts = SmallSignalOptions { shift, count, ..SmallSignalOptions::default() };
            smallsignal::analyze_near(system, &opts).map_err(|e| e.to_string())?
        }
        None => smallsignal::analyze(system).map_err(|e| e.to_string())?,
    };

    println!(
        "{} mode(s) over {} differential state(s)",
        result.modes.len(),
        result.state_names.len()
    );
    match result.method {
        Method::Dense => println!("every mode, by a dense decomposition of the state matrix"),
        Method::Sparse { shift, restarts } => {
            println!(
                "the {} nearest {:+.4}{:+.4}j ({:.3} Hz) — sparse Arnoldi, {restarts} restart(s). \
                 NOT every mode the system has.",
                result.modes.len(),
                shift.re,
                shift.im,
                shift.im / std::f64::consts::TAU
            );
            if !result.converged {
                println!(
                    "WARNING: not every mode converged — read the residual column before the rest"
                );
            }
        }
    }
    println!();
    println!(
        "{:>26}  {:>9}  {:>10}  {:>9}  {:>9}   participation",
        "eigenvalue", "damping", "freq (Hz)", "tau (s)", "residual"
    );
    for mode in result.modes.iter().take(count) {
        let who: Vec<String> = result
            .participants(mode)
            .into_iter()
            .take(3)
            .map(|(name, p)| format!("{name} {}", percentage(p)))
            .collect();
        let tau = if mode.time_constant.is_finite() {
            format!("{:.3}", mode.time_constant)
        } else {
            "inf".to_string()
        };
        // A mode judged non-oscillatory has a frequency of zero, so its
        // imaginary part is printed as zero too rather than as the `-1e-17` the
        // arithmetic left behind — the two columns should not contradict each
        // other over a rounding artefact.
        let omega = if mode.is_oscillatory() { mode.eigenvalue.im } else { 0.0 };
        println!(
            "{:>+12.5} {:>+12.5}j  {:>9.4}  {:>10.4}  {tau:>9}  {:>9.1e}   {}",
            mode.eigenvalue.re,
            omega,
            mode.damping,
            mode.frequency,
            mode.residual,
            who.join(", ")
        );
        // For an oscillation, *how* the rotors move is the part that separates
        // an inter-area mode from a local one, and participation cannot say it.
        //
        // Printed for one half of a conjugate pair rather than both, since the
        // shapes are conjugates and saying it twice says nothing. Which half is
        // whichever sorts first, so the test is on the sign of the imaginary
        // part of *this* mode against its own twin, not on it being positive.
        if mode.is_oscillatory() && mode.shape.len() > 1 && mode.eigenvalue.im > 0.0 {
            // Sorted by magnitude, because the components that move most are
            // the ones the mode is about. Taking the first four in *state*
            // order instead says nothing on a system with two thousand rotors:
            // every one of them printed `0.00∠…`, since the shape is scaled so
            // the largest is 1 and G0 was not it.
            let mut shape = result.shape(mode);
            shape.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let shape: Vec<String> = shape
                .into_iter()
                .take(4)
                .map(|(name, magnitude, phase)| {
                    format!("{name} {magnitude:.2}∠{phase:+.0}°")
                })
                .collect();
            println!("{:>73}   shape: {}", "", shape.join(", "));
        }
    }

    // Which knob moves the worst mode, when asked. The critical mode only:
    // that is the question a stability study asks, and a sensitivity per mode
    // per parameter is a table nobody reads.
    if flags.iter().any(|f| f == "--modes-sensitivity") {
        report_sensitivities(system, &result)?;
    }

    let unstable = result.unstable();
    if unstable.is_empty() {
        println!("\nevery mode decays");
    } else {
        println!("\n{} mode(s) GROW rather than decay:", unstable.len());
        for mode in unstable {
            println!(
                "  {:+.5}{:+.5}j at {:.4} Hz — {}",
                mode.eigenvalue.re,
                mode.eigenvalue.im,
                mode.frequency,
                result
                    .participants(mode)
                    .into_iter()
                    .take(2)
                    .map(|(n, p)| format!("{n} {}", percentage(p)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    Ok(())
}

/// A participation factor, with enough digits to still say something.
///
/// Whole per cent for anything above one, and two decimals below. An
/// oscillation spread across two thousand machines belongs to each of them at
/// about 0.14%, and rounding that to `0%` reads as *no* participation when the
/// truth is the opposite — that the mode is everyone's.
#[cfg(feature = "dynamics")]
fn percentage(p: f64) -> String {
    if p >= 0.01 {
        format!("{:.0}%", p * 100.0)
    } else {
        format!("{:.2}%", p * 100.0)
    }
}

/// `--modes-sensitivity` — which parameter moves the least-damped mode.
#[cfg(feature = "dynamics")]
fn report_sensitivities(
    system: &mut gridoxide::dynamics::DynamicSystem,
    result: &gridoxide::dynamics::smallsignal::SmallSignal,
) -> Result<(), String> {
    use gridoxide::dynamics::smallsignal;

    let Some(mode) = result.critical().cloned() else {
        return Ok(());
    };
    let parameters = smallsignal::tunable_parameters(system);
    if parameters.is_empty() {
        println!("\nno device carries a parameter a sensitivity can be taken with respect to");
        return Ok(());
    }
    let mut sensitivities = smallsignal::sensitivities(system, &mode, &parameters)
        .map_err(|e| e.to_string())?;

    // Ranked by what a study is actually choosing between: how much damping a
    // change buys. Relative to the parameter's own size, since an inertia of
    // 5 s and a damping coefficient of 1 are not comparable per unit.
    sensitivities.sort_by(|a, b| {
        let (x, y) = (a.d_damping * a.value, b.d_damping * b.value);
        y.abs().partial_cmp(&x.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "\nsensitivity of the least-damped mode ({:+.5}{:+.5}j, ζ = {:.4}):",
        mode.eigenvalue.re, mode.eigenvalue.im, mode.damping
    );
    println!(
        "{:>18}  {:>10}  {:>26}  {:>10}  {:>12}",
        "parameter", "value", "dlambda/dp", "dzeta/dp", "dzeta per 1%"
    );
    for s in sensitivities.iter().take(10) {
        let name = format!("{}.{}", device_label(result, s.parameter.device), s.parameter.name);
        println!(
            "{name:>18}  {:>10.4}  {:>+12.5} {:>+12.5}j  {:>10.3e}  {:>12.3e}",
            s.value,
            s.d_eigenvalue.re,
            s.d_eigenvalue.im,
            s.d_damping,
            s.d_damping * s.value * 0.01,
        );
    }
    Ok(())
}

/// A device's name, taken from the states it owns.
///
/// The state names are `unit.state`, and every state of one device shares the
/// prefix, so the device's own name is recoverable without a second list.
#[cfg(feature = "dynamics")]
fn device_label(
    result: &gridoxide::dynamics::smallsignal::SmallSignal,
    device: usize,
) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for name in &result.state_names {
        let unit = name.split('.').next().unwrap_or(name);
        if !seen.contains(&unit) {
            seen.push(unit);
        }
    }
    seen.get(device).map(|s| (*s).to_string()).unwrap_or_else(|| format!("device{device}"))
}

/// Where `--modes` should look, if the caller said.
///
/// `--modes-freq` is the form the question comes in — "the modes near 1 Hz" —
/// and `--modes-near` is the literal complex number for anyone who wants to
/// name it directly. Giving neither means the dense method, which needs no aim
/// because it computes everything.
#[cfg(feature = "dynamics")]
fn parse_shift(flags: &[String]) -> Result<Option<num_complex::Complex<f64>>, String> {
    use gridoxide::dynamics::smallsignal::SmallSignalOptions;

    let freq = parse_f64_flag(flags, "--modes-freq")?;
    let damping = parse_f64_flag(flags, "--modes-damping")?;
    let near = flag_value(flags, "--modes-near")?;

    if let Some(near) = near {
        if freq.is_some() || damping.is_some() {
            return Err("--modes-near names the shift outright; \
                        do not also give --modes-freq or --modes-damping"
                .to_string());
        }
        let (re, im) = near.split_once(',').ok_or_else(|| {
            format!("--modes-near: expected <re>,<im>, got {near:?}")
        })?;
        let re: f64 = re
            .trim()
            .parse()
            .map_err(|_| format!("--modes-near: {re:?} is not a number"))?;
        let im: f64 = im
            .trim()
            .parse()
            .map_err(|_| format!("--modes-near: {im:?} is not a number"))?;
        return Ok(Some(num_complex::Complex::new(re, im)));
    }

    match (freq, damping) {
        (None, None) => Ok(None),
        (None, Some(_)) => Err(
            "--modes-damping only says how far off the axis to aim; \
             give --modes-freq to say where"
                .to_string(),
        ),
        (Some(hz), zeta) => {
            if hz <= 0.0 {
                return Err(format!("--modes-freq: expected a positive frequency, got {hz}"));
            }
            Ok(Some(SmallSignalOptions::near_frequency(hz, zeta.unwrap_or(0.05)).shift))
        }
    }
}

/// `gridoxide qv` — a bus's reactive margin.
fn run_qv_cli(path: &str, flags: &[String]) -> Result<(), String> {
    use gridoxide::qv::{qv_curve, QvOptions, QvStatus};

    let base_mva = parse_f64_flag(flags, "--base-mva")?.unwrap_or(100.0);
    let raw = fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let input: PgmInput =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?;
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, base_mva * 1e6, 50.0);
    let n = buses.len();

    let opts = QvOptions {
        pf: PowerFlowOptions { max_iter: 60, ..Default::default() },
        v_max: parse_f64_flag(flags, "--v-max")?.unwrap_or(1.10),
        v_min: parse_f64_flag(flags, "--v-min")?.unwrap_or(0.40),
        step: parse_f64_flag(flags, "--step")?.unwrap_or(0.01),
        refine: !flags.iter().any(|f| f == "--no-refine"),
    };

    let one = match flag_value(flags, "--bus")? {
        Some(raw) => Some(parse_index(&raw, "--bus", n)?),
        None => None,
    };
    let weakest = match flag_value(flags, "--weakest")? {
        Some(raw) => Some(
            raw.parse::<usize>()
                .map_err(|_| format!("--weakest: expected a count, got {raw:?}"))?,
        ),
        None => None,
    };
    if one.is_some() && weakest.is_some() {
        return Err("--bus and --weakest ask for different things".into());
    }

    println!(
        "{n} bus(es); sweeping |V| from {:.3} to {:.3} in steps of {:.3}{}",
        opts.v_max,
        opts.v_min,
        opts.step,
        if opts.refine { ", minimum interpolated" } else { "" }
    );

    if let Some(bus) = one {
        let curve = qv_curve(&buses, &lines, &transformers, &[], bus, opts);
        if let QvStatus::Rejected(why) = &curve.status {
            return Err(format!("bus {bus}: {why:?}"));
        }
        println!(
            "\nbus {bus}: solves at |V| {:.5} in the base case{}",
            curve.base_voltage,
            if curve.already_controlled {
                " — already voltage-controlled, so this moves an existing machine's setpoint \
                 rather than adding a condenser"
            } else {
                ""
            }
        );
        if flags.iter().any(|f| f == "--curve") {
            println!("\n     |V|      Q needed (MVAr)   iterations");
            for p in &curve.points {
                println!("  {:7.4}   {:15.2}   {:5}", p.voltage, p.q * base_mva, p.iterations);
            }
        }
        if !curve.failed.is_empty() {
            println!("\n  {} setpoint(s) did not converge: {:?}", curve.failed.len(), curve.failed);
        }
        match (&curve.nose, &curve.status) {
            (Some(nose), QvStatus::NoseFound) => {
                println!("\nreactive margin");
                println!("  {:.2} MVAr, at |V| = {:.5}{}",
                    nose.margin_mvar(base_mva), nose.voltage,
                    if nose.refined { " (interpolated)" } else { "" });
            }
            (Some(bound), QvStatus::NoseNotReached) => {
                println!(
                    "\nthe curve was still falling at |V| = {:.4}, so {:.2} MVAr is a LOWER BOUND \
                     on the margin, not the margin — lower --v-min to find it",
                    bound.voltage,
                    bound.margin_mvar(base_mva)
                );
            }
            _ => println!("\nno curve: {:?}", curve.status),
        }
        return Ok(());
    }

    // The ranking.
    let mut rows: Vec<(f64, usize, f64, bool)> = Vec::new();
    let mut skipped = 0usize;
    for bus in 0..n {
        let curve = qv_curve(&buses, &lines, &transformers, &[], bus, opts.clone());
        match (&curve.nose, &curve.status) {
            (Some(nose), QvStatus::NoseFound) => {
                rows.push((nose.margin_pu, bus, nose.voltage, true))
            }
            (Some(bound), QvStatus::NoseNotReached) => {
                rows.push((bound.margin_pu, bus, bound.voltage, false))
            }
            _ => skipped += 1,
        }
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));

    let show = weakest.unwrap_or(10).min(rows.len());
    println!("\nweakest {show} bus(es) by reactive margin:");
    println!("     bus   margin (MVAr)   nose at |V|");
    for (m, bus, v, found) in rows.iter().take(show) {
        println!(
            "  {bus:6}   {:13.2}   {v:11.5}{}",
            m * base_mva,
            if *found { "" } else { "   (lower bound — curve still falling)" }
        );
    }
    if skipped > 0 {
        println!("\n  {skipped} bus(es) yielded no curve (slack, or too few converged points)");
    }
    println!(
        "\nthis is a per-bus measure at the base operating point. `gridoxide continuation` asks \
         the system-wide question, and the two are known not to agree on which bus is weakest."
    );
    Ok(())
}

/// `gridoxide continuation` — trace the P-V curve to the point of collapse.
fn run_continuation_cli(path: &str, flags: &[String]) -> Result<(), String> {
    use gridoxide::continuation::augmented::Parametrization;
    use gridoxide::continuation::{
        run_continuation, ContinuationOptions, ContinuationStatus, CriticalPointKind,
        CurveBranch, LoadingDirection, QLimitKind, StopCriterion,
    };

    let base_mva = parse_f64_flag(flags, "--base-mva")?.unwrap_or(100.0);
    let s_base_va = base_mva * 1e6;
    let raw = fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let input: PgmInput =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {path}: {e}"))?;
    let (buses, lines, transformers) = pgm_to_buses_and_branches(input, s_base_va, 50.0);
    let n = buses.len();

    let direction = match flag_value(flags, "--buses")? {
        Some(list) => LoadingDirection::scale_buses(&buses, &parse_index_list(&list, "--buses", n)?),
        None => LoadingDirection::scale_loads(&buses),
    };
    let direction = match flag_value(flags, "--pickup")? {
        None => direction,
        Some(list) => {
            let mut weights = vec![0.0; n];
            for i in parse_index_list(&list, "--pickup", n)? {
                weights[i] = 1.0;
            }
            LoadingDirection::scale_loads_with_pickup(&buses, &weights)
        }
    };

    let parametrization = match flag_value(flags, "--parametrization")?.as_deref() {
        // Deliberately `default()` rather than a name repeated here: the choice
        // is a performance cliff (pseudo-arclength's bordering row is dense, and
        // costs ~90x a sparse one at a few thousand buses), so the CLI must not
        // be able to drift away from the library's default.
        None => Parametrization::default(),
        Some("local") => Parametrization::Local,
        Some("natural") => Parametrization::Natural,
        Some("arclength") => Parametrization::PseudoArcLength,
        Some(other) => {
            return Err(format!(
                "--parametrization: expected arclength, local or natural, got {other:?}"
            ))
        }
    };
    let target_lambda = parse_f64_flag(flags, "--target-lambda")?;
    let lower_branch = flags.iter().any(|f| f == "--lower-branch");
    if target_lambda.is_some() && lower_branch {
        return Err("--target-lambda and --lower-branch ask for different walks".into());
    }
    let enforce_q_limits = flags.iter().any(|f| f == "--enforce-q-limits");

    let opts = ContinuationOptions {
        direction: direction.clone(),
        parametrization,
        stop: StopCriterion { target_lambda, trace_lower_branch: lower_branch },
        step: parse_f64_flag(flags, "--step")?.unwrap_or(0.1),
        max_steps: match flag_value(flags, "--max-steps")? {
            None => 200,
            Some(raw) => raw
                .parse()
                .map_err(|_| format!("--max-steps: expected a count, got {raw:?}"))?,
        },
        power_flow: PowerFlowOptions {
            enforce_q_limits,
            tol: parse_f64_flag(flags, "--tol")?.unwrap_or(1e-8),
            max_iter: match flag_value(flags, "--max-iter")? {
                None => 30,
                Some(raw) => raw
                    .parse()
                    .map_err(|_| format!("--max-iter: expected a count, got {raw:?}"))?,
            },
            ..Default::default()
        },
        base_mva,
        ..Default::default()
    };

    let curve = run_continuation(buses, &lines, &transformers, &[], opts);
    if let ContinuationStatus::Rejected(why) = &curve.status {
        return Err(format!("continuation was not attempted: {why:?}"));
    }
    if let ContinuationStatus::BaseCaseFailed(status) = &curve.status {
        return Err(format!(
            "the base case did not converge ({status:?}), so there is no curve to trace from"
        ));
    }
    for warning in &curve.warnings {
        println!("warning: {warning:?}");
    }

    println!(
        "{} point(s) over {} segment(s), {} bordered solve(s); stopped: {:?}",
        curve.points.len(),
        curve.segments,
        curve.solves,
        curve.status
    );
    println!(
        "loading grows along a fixed direction ({:.1} MW per unit lambda) — the limit below \
         is a property of that direction, not of the network alone",
        direction.total_active_load_increase() * base_mva
    );

    if flags.iter().any(|f| f == "--curve") {
        println!("\n  lambda   arclength   min |V|  at bus   dlambda/dsigma   its  branch");
        for p in &curve.points {
            let (bus, vmin) = p
                .voltage_mag
                .iter()
                .enumerate()
                .filter(|(_, v)| **v > 0.0)
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, v)| (i, *v))
                .unwrap_or((0, 0.0));
            println!(
                "  {:+7.4}   {:9.4}   {:7.4}  {:6}   {:+14.6}   {:3}  {}",
                p.lambda,
                p.arclength,
                vmin,
                bus,
                p.tangent_lambda,
                p.corrector_iterations,
                if p.branch == CurveBranch::Upper { "upper" } else { "lower" }
            );
        }
    }

    let events = curve.q_limit_events();
    if !events.is_empty() {
        println!("\nreactive limits reached, in order:");
        for (bus, limit, lambda) in &events {
            println!(
                "  lambda {lambda:.6}: bus {bus} reached q_{} and stopped holding its voltage",
                if *limit == QLimitKind::Max { "max" } else { "min" }
            );
        }
    }

    match &curve.critical {
        None => {
            println!(
                "\nno collapse point was found — the walk stopped at lambda {:.4} for another \
                 reason (see the status above), so this is a lower bound on the limit, not the \
                 limit",
                curve.points.last().map(|p| p.lambda).unwrap_or(0.0)
            );
        }
        Some(c) => {
            println!("\nloadability limit");
            match c.kind {
                CriticalPointKind::SaddleNode => println!(
                    "  a saddle-node bifurcation: the power-flow Jacobian is singular there"
                ),
                CriticalPointKind::LimitInduced { bus } => println!(
                    "  limit-induced by bus {bus}: the limit is the point that machine \
                     saturates, not a fold"
                ),
            }
            println!("  lambda_max      {:.6}  ({:.4}x the base load)", c.lambda_max, 1.0 + c.lambda_max);
            println!("  margin          {:.1} MW  ({:.4} p.u.)", c.margin_mw, c.margin_pu);
            println!("  load at limit   {:.1} MW (from {:.1} MW)",
                c.p_load_nose_pu * base_mva, c.p_load_base_pu * base_mva);
            println!("  weakest buses (share of the collapse mode):");
            for w in c.weakest.iter().take(5) {
                println!("    bus {:5}  {:.3}   |V| = {:.4}", w.bus, w.participation, w.voltage_mag);
            }
        }
    }
    Ok(())
}

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
    let report = run_power_flow(buses, &lines, &transformers, &[], gridoxide::TapData::none(), PowerFlowOptions::default());
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
        shunts: &network.shunts,
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
                "\n    {{\"cnec\": {:?}, \"flow_mw\": {}, \"limit_mw\": {}, \"upper_mw\": {}, \
                 \"lower_mw\": {}, \"margin_mw\": {}, \"margin_a\": {}}}",
                crac.flow_cnecs[c.cnec].id,
                c.flow_mw,
                c.limit_mw,
                // A one-sided CNEC has an infinite bound, which is not JSON.
                // `null` says "no bound" where `Infinity` would say nothing at
                // all, and both beat emitting a document no parser accepts.
                json_number(c.upper_mw),
                json_number(c.lower_mw),
                c.margin_mw,
                c.margin_a
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
        shunts: &network.shunts,
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

/// A finite float, or `null` for an infinity JSON cannot represent.
#[cfg(all(feature = "rao", any(feature = "ucte", feature = "iidm")))]
fn json_number(value: f64) -> String {
    if value.is_finite() { value.to_string() } else { "null".to_string() }
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

// ---------------------------------------------------------------------------
// `gridoxide solve`
// ---------------------------------------------------------------------------

/// A network loaded for a solve, whichever importer produced it.
///
/// Deliberately not `SecurityNetwork`: that one is gated behind the `rao`
/// feature and carries CRAC-shaped fields (countries, initially-open branches)
/// a power flow has no use for, and it cannot load CGMES — which is the only
/// format whose fixtures declare tap controls.
#[cfg(any(feature = "ucte", feature = "iidm", feature = "cgmes"))]
struct SolveNetwork {
    buses: Vec<gridoxide::types::Bus>,
    lines: Vec<gridoxide::types::Line>,
    transformers: Vec<gridoxide::types::Transformer>,
    shunts: Vec<gridoxide::network::ShuntAdm>,
    tap_changers: Vec<Option<gridoxide::types::TapChanger>>,
    regulation: Vec<gridoxide::outerloop::TapRegulation>,
    /// Area index per bus, and what each area is called. Empty when the format
    /// says nothing about areas.
    areas: (Vec<Option<usize>>, Vec<String>),
    /// Scheduled net export per area, per-unit. Empty unless the format states
    /// one — only CGMES does.
    area_targets: Vec<f64>,
    labels: Vec<String>,
    notes: Vec<String>,
    s_base_va: f64,
    /// The regulating machines the document declared, where it declares any.
    /// Only CGMES states per-machine reactive capability; the other importers
    /// leave this empty and `--dispatch` then has nothing to attribute.
    machines: Vec<gridoxide::types::RegulatingMachine>,
    /// Per-bus reactive injection that is not a regulating machine's.
    nonregulating_q: Vec<f64>,
}

#[cfg(any(feature = "ucte", feature = "iidm", feature = "cgmes"))]
fn load_network_for_solve(path: &str) -> Result<SolveNetwork, String> {
    // Only the two extension-dispatched importers look at it; a cgmes-only
    // build reaches the directory branch without ever asking.
    #[cfg(any(feature = "ucte", feature = "iidm"))]
    let lower = path.to_ascii_lowercase();

    #[cfg(feature = "ucte")]
    if lower.ends_with(".uct") || lower.ends_with(".ucte") {
        let n = gridoxide::ucte::read(path).map_err(|e| e.to_string())?;
        let areas = n.country_areas();
        return Ok(SolveNetwork {
            buses: n.buses,
            lines: n.lines,
            transformers: n.transformers,
            shunts: n.shunts,
            tap_changers: n.tap_changers,
            regulation: n.regulation,
            areas,
            // UCTE states no scheduled net position anywhere; `##Z<cc>` gives
            // the membership and nothing else.
            area_targets: Vec::new(),
            labels: n.node_codes,
            notes: n.notes,
            s_base_va: n.base_mva * 1e6,
            // UCTE-DEF and IIDM state no per-machine reactive capability, so
            // there is nothing to attribute and `--dispatch` says so rather
            // than printing an empty table.
            machines: Vec::new(),
            nonregulating_q: Vec::new(),
        });
    }

    #[cfg(feature = "iidm")]
    if lower.ends_with(".xiidm") {
        let n = gridoxide::iidm::read(path).map_err(|e| e.to_string())?;
        let mut notes = n.notes.clone();
        if n.areas.report.areas > 0 || n.areas.report.other_types > 0 {
            let r = &n.areas.report;
            notes.push(format!(
                "{} control area(s): {} of another areaType, {} without a schedule, \
                 {} unknown voltage level(s), {} contested bus(es), {} bus(es) in no area",
                r.areas,
                r.other_types,
                r.without_target.len(),
                r.unknown_voltage_levels,
                r.contested_buses,
                r.unassigned_buses
            ));
        }
        return Ok(SolveNetwork {
            buses: n.buses,
            lines: n.lines,
            transformers: n.transformers,
            shunts: n.shunts,
            areas: (
                n.areas.of_bus.clone(),
                n.areas.ids.iter().map(|(id, name)| {
                    if name.trim().is_empty() { id.clone() } else { name.trim().to_string() }
                }).collect(),
            ),
            area_targets: n.areas.targets.clone(),
            tap_changers: n.tap_changers,
            regulation: n.regulation,
            labels: n.bus_labels,
            notes,
            s_base_va: n.base_mva * 1e6,
            // See the UCTE branch above.
            machines: Vec::new(),
            nonregulating_q: Vec::new(),
        });
    }

    #[cfg(feature = "cgmes")]
    {
        // A directory of profiles, or one profile file whose siblings are the
        // rest of the set — which is how every conformity configuration is
        // laid out, and how a user actually has them on disk.
        let dir = std::path::Path::new(path);
        let dir = if dir.is_dir() { dir.to_path_buf() } else { dir.parent().unwrap_or(dir).to_path_buf() };
        let mut profiles: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| format!("reading {}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "xml"))
            .collect();
        profiles.sort();
        if profiles.is_empty() {
            return Err(format!("no CGMES profile .xml files in {}", dir.display()));
        }
        let refs: Vec<&std::path::Path> = profiles.iter().map(|p| p.as_path()).collect();
        let ds = gridoxide::cgmes::load_profiles(&refs).map_err(|e| e.to_string())?;
        let s_base_va = 100e6;
        let net = gridoxide::cgmes::cgmes_to_network(&ds, s_base_va).map_err(|e| e.to_string())?;
        let mut notes = Vec::new();
        let r = &net.tap_report;
        notes.push(format!(
            "{} CGMES profile file(s); tap controls: {} read, {} disabled, {} unattached, \
             {} without a target, {} in an unmodelled mode, {} with an unresolvable flow",
            profiles.len(),
            r.converted,
            r.disabled,
            r.unattached,
            r.without_target,
            r.unsupported_mode,
            r.unresolved_flow
        ));
        let areas = gridoxide::cgmes::cgmes_control_areas(&ds, &net, s_base_va)
            .map_err(|e| e.to_string())?;
        if areas.report.areas > 0 {
            notes.push(format!(
                "{} control area(s), {} tie flow(s): {} unresolved, {} without a boundary, \
                 {} without a schedule, {} contested bus(es), {} bus(es) in no area",
                areas.report.areas,
                areas.report.tie_flows,
                areas.report.unresolved_tie_flows,
                areas.report.without_boundary.len(),
                areas.report.without_target.len(),
                areas.report.contested_buses,
                areas.report.unassigned_buses
            ));
        }
        let labels = (0..net.buses.len()).map(|i| format!("bus {i}")).collect();
        let machines = net.voltage_control.machines;
        let nonregulating_q = net.voltage_control.nonregulating_q;
        return Ok(SolveNetwork {
            areas: (areas.of_bus, areas.ids.into_iter().map(|(_, n)| n).collect()),
            area_targets: areas.targets,
            buses: net.buses,
            lines: net.lines,
            transformers: net.transformers,
            shunts: net.shunts,
            tap_changers: net.tap_changers,
            regulation: net.regulation,
            labels,
            notes,
            s_base_va,
            machines,
            nonregulating_q,
        });
    }

    #[allow(unreachable_code)]
    Err(format!(
        "cannot tell what `{path}` is; expected a .uct, .xiidm, or a directory of CGMES profiles"
    ))
}

#[cfg(any(feature = "ucte", feature = "iidm", feature = "cgmes"))]
fn run_solve(path: &str, flags: &[String]) -> Result<(), String> {
    use gridoxide::outerloop::ControllerOutcome;
    use gridoxide::solver::{PowerFlowOptions, SolveStatus};

    let number = |flag: &str, default: f64| -> Result<f64, String> {
        match flag_value(flags, flag)? {
            None => Ok(default),
            Some(raw) => raw.parse().map_err(|_| format!("{flag}: {raw:?} is not a number")),
        }
    };
    let count = |flag: &str, default: usize| -> Result<usize, String> {
        match flag_value(flags, flag)? {
            None => Ok(default),
            Some(raw) => raw.parse().map_err(|_| format!("{flag}: {raw:?} is not an integer")),
        }
    };

    let control_taps = flags.iter().any(|f| f == "--control-taps");
    let enforce_q_limits = flags.iter().any(|f| f == "--enforce-q-limits");
    let distribute = flags.iter().any(|f| f == "--distribute-slack");
    let area_control = flags.iter().any(|f| f == "--area-interchange");
    if distribute && area_control {
        return Err("--distribute-slack and --area-interchange cannot both be given: area \
                    interchange subsumes distributed slack (one area with a zero target *is* \
                    distributed slack)"
            .to_string());
    }
    let tol = number("--tol", 1e-8)?;
    let max_iter = count("--max-iter", 30)?;
    let max_outer = count("--max-outer", 40)?;
    let max_tap_shift = count("--max-tap-shift", 3)? as i32;
    let dispatch = flags.iter().any(|f| f == "--dispatch");
    let remote_voltage = flags.iter().any(|f| f == "--control-remote-voltage");

    let net = load_network_for_solve(path)?;
    for note in &net.notes {
        println!("note: {note}");
    }

    let distribution = distribute
        .then(|| gridoxide::outerloop::SlackDistribution::uniform(&net.buses));
    let (of_bus, area_names) = net.areas.clone();
    let area_definition = if area_control {
        if area_names.is_empty() {
            return Err(format!(
                "--area-interchange needs an area assignment, and `{path}` states none. CGMES \
                 supplies one through ControlArea/TieFlow and UCTE through its ##Z country codes"
            ));
        }
        // Weighted by generation rather than by bus type: a net position is met
        // by redispatch, and a generator held at a fixed active set-point — a
        // `PQ` bus here — is exactly the machine an operator redispatches. The
        // `uniform` policy distributed slack uses would leave such an area with
        // no participant at all.
        let mut d = gridoxide::outerloop::AreaDefinition::by_generation(
            &net.buses,
            of_bus,
            area_names.len(),
        );
        if !net.area_targets.is_empty() {
            d.targets = net.area_targets.clone();
        }
        d.max_outer_iter = max_outer;
        Some(d)
    } else {
        None
    };
    let report = gridoxide::run_power_flow_with_remote(
        net.buses.clone(),
        &net.lines,
        &net.transformers,
        &net.shunts,
        gridoxide::TapData { changers: &net.tap_changers, regulation: &net.regulation },
        gridoxide::RemoteControlData { machines: &net.machines },
        PowerFlowOptions {
            control_taps,
            control_remote_voltage: remote_voltage,
            enforce_q_limits,
            distribute_slack: distribution,
            area_interchange: area_definition,
            tap_max_shift: max_tap_shift,
            max_outer_iter: max_outer,
            tol,
            max_iter,
            ..Default::default()
        },
    );

    println!(
        "\n{} bus(es), {} line(s), {} transformer(s); {} tap table(s), {} regulating control(s)",
        net.buses.len(),
        net.lines.len(),
        net.transformers.len(),
        net.tap_changers.iter().filter(|c| c.is_some()).count(),
        net.regulation.len()
    );

    // Real networks come with a long tail of de-energized one-bus islands —
    // Svedala alone has 78 — so the interesting ones are listed and the rest
    // counted. A `NoReferenceBus` island is a reported verdict, not a failure.
    use gridoxide::solver::IslandStatus;
    let mut trivial = 0usize;
    let mut trivial_buses = 0usize;
    for (i, island) in report.islands.iter().enumerate() {
        if island.status == IslandStatus::NoReferenceBus && island.bus_indices.len() <= 2 {
            trivial += 1;
            trivial_buses += island.bus_indices.len();
            continue;
        }
        println!("  island {i}: {} bus(es), {:?}", island.bus_indices.len(), island.status);
    }
    if trivial > 0 {
        println!("  ... and {trivial} de-energized island(s) covering {trivial_buses} bus(es)");
    }

    if dispatch {
        if net.machines.is_empty() {
            println!(
                "\n--dispatch: this format states no per-machine reactive capability, so there \
                 is nothing to attribute (only CGMES does)"
            );
        } else {
            let mut y = gridoxide::network::build_ybus(net.buses.len(), &net.lines, &net.transformers);
            gridoxide::network::stamp_shunts(&mut y, &net.shunts);
            let ybus = y.finish();
            let split = gridoxide::dispatch::allocate(
                &net.machines, &net.nonregulating_q, &report.buses, &ybus);

            let base_mva = net.s_base_va / 1e6;
            let shared: Vec<_> = split.iter().filter(|b| b.machines.len() > 1).collect();
            let unattributed: Vec<_> =
                split.iter().filter(|b| b.unattributed.abs() > 1e-6).collect();
            println!(
                "\nreactive dispatch: {} machine(s) over {} regulated bus(es), {} of them held \
                 by more than one",
                net.machines.len(),
                split.len(),
                shared.len()
            );

            for bd in shared.iter().take(12) {
                println!(
                    "  bus {:<6} {:>9.2} MVAr over {} machine(s), split by {:?}",
                    bd.bus,
                    bd.attributed * base_mva,
                    bd.machines.len(),
                    bd.basis
                );
                for m in &bd.machines {
                    println!(
                        "      {:<40} {:>9.2} MVAr  ({:.0}% of the bus){}",
                        m.id,
                        m.q * base_mva,
                        m.share * 100.0,
                        if m.at_limit { "  at its own limit" } else { "" }
                    );
                }
            }
            if shared.len() > 12 {
                println!("  ... and {} more shared bus(es)", shared.len() - 12);
            }

            // Worth surfacing rather than hiding: it means the solve put the
            // bus outside what its own machines can produce, which happens
            // because the reactive-limit clamp bounds the bus's *net* injection
            // while the limits describe the machines' own capability. See
            // `dispatch::BusDispatch::unattributed`.
            if !unattributed.is_empty() {
                let total: f64 = unattributed.iter().map(|b| b.unattributed.abs()).sum();
                println!(
                    "\n  {} bus(es) need {:.2} MVAr their own machines cannot produce — a \
                     reactive load shares the bus, so the machine saturates before the bus's \
                     net injection reaches its bound",
                    unattributed.len(),
                    total * base_mva
                );
                for bd in unattributed.iter().take(5) {
                    println!(
                        "      bus {:<6} short by {:>8.2} MVAr",
                        bd.bus,
                        bd.unattributed * base_mva
                    );
                }
            }
        }
    }

    if let Some(outer) = report.outer.as_ref() {
        if let Some(r) = outer.report.as_ref() {
            println!(
                "\nouter loops: {} re-solve(s) over {} inner solve(s), converged = {}{}",
                r.total_iterations,
                r.solves,
                r.converged,
                if r.budget_exhausted { " (budget exhausted)" } else { "" }
            );
            for l in &r.loops {
                println!("  {:<26} {:>3} iteration(s)  {:?}", l.name, l.iterations, l.status);
            }
        }
        if !outer.remote.is_empty() {
            println!("\nremote voltage control: {} machine(s)", outer.remote.len());
            for r in &outer.remote {
                println!(
                    "  {:<40} bus {} holds bus {}: reached {:.5} against {:.5} in {} move(s), {:?}",
                    r.machine, r.controller_bus, r.controlled_bus, r.reached, r.target, r.moves,
                    r.outcome
                );
            }
            // The network solved is not quite the one handed over, which is
            // worth saying rather than leaving to be noticed.
            for (bus, from, to) in &outer.retyped {
                println!("  bus {bus} re-typed {from:?} -> {to:?} to put the control on the machine");
            }
        }
        if !outer.q_limit_switches.is_empty() {
            println!(
                "\n{} bus(es) switched PV -> PQ on their reactive limit: {:?}",
                outer.q_limit_switches.len(),
                outer.q_limit_switches
            );
        }
        if let Some(s) = outer.slack.as_ref() {
            let total: f64 = s.shift.iter().sum();
            println!(
                "\nslack distribution: {:.3} MW moved over {} pass(es), converged = {}",
                total * net.s_base_va / 1e6,
                s.outer_iterations,
                s.converged
            );
            for (island, why) in &s.undistributed {
                println!("  island {island} left on a single slack: {why}");
            }
        }
        if let Some(a) = outer.area.as_ref() {
            println!(
                "\narea interchange over {} pass(es), converged = {}:",
                a.outer_iterations, a.converged
            );
            for (i, name) in area_names.iter().enumerate() {
                let dependent = a.dependent == Some(i);
                println!(
                    "  {:<20} {:>10.3} MW exported, {:>8.3} MW off schedule{}",
                    name,
                    a.interchange.get(i).copied().unwrap_or(0.0) * net.s_base_va / 1e6,
                    a.residual.get(i).copied().unwrap_or(0.0) * net.s_base_va / 1e6,
                    if dependent { "  (dependent: takes the residual)" } else { "" }
                );
            }
            for (area, why) in &a.unbalanced {
                match area_names.get(*area) {
                    Some(name) => println!("  {name}: {why}"),
                    None => println!("  {why}"),
                }
            }
        }
        if !outer.taps.is_empty() {
            println!("\ntap controllers:");
            for c in &outer.taps {
                let moved = c.final_position - c.initial_position;
                println!(
                    "  {:<40} {:>4} -> {:>4} ({moved:+})  {:?}",
                    c.id, c.initial_position, c.final_position, c.outcome
                );
            }
            let settled = outer
                .taps
                .iter()
                .filter(|c| c.outcome == ControllerOutcome::InDeadband)
                .count();
            println!("  {settled} of {} inside their deadbands", outer.taps.len());
        }
    }

    // The five buses furthest from nominal, which is where a reader looks
    // first and what the labels were loaded for.
    let mut extremes: Vec<(usize, f64)> = report
        .buses
        .iter()
        .enumerate()
        .map(|(i, b)| (i, b.voltage_mag))
        .filter(|(_, v)| *v > 0.0)
        .collect();
    extremes.sort_by(|a, b| (b.1 - 1.0).abs().total_cmp(&(a.1 - 1.0).abs()));
    if !extremes.is_empty() {
        println!("\nfurthest from nominal:");
        for (i, v) in extremes.iter().take(5) {
            let label = net.labels.get(*i).map(String::as_str).unwrap_or("");
            println!("  {label:<24} {v:.5} pu");
        }
    }

    match report.stats.status {
        SolveStatus::Converged => Ok(()),
        other => Err(format!("power flow did not converge: {other:?}")),
    }
}

#[cfg(not(any(feature = "ucte", feature = "iidm", feature = "cgmes")))]
fn run_solve(_path: &str, _flags: &[String]) -> Result<(), String> {
    Err("solve needs an importer feature (`cgmes`, `ucte` or `iidm`); \
         rebuild with `--features cgmes`"
        .to_string())
}
