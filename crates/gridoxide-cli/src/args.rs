//! `gridox`'s arguments, as clap parses them.
//!
//! What a subcommand accepts is declared here and nowhere else: an unknown
//! flag, a flag of another mode, a missing value or one that does not parse is
//! a usage error (exit 2) before anything runs. Checks that need the network
//! (an index in range) or explain a modelling choice stay with the command.
//!
//! A subcommand whose feature this build lacks still parses, and its command
//! says which feature it needs.

use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "gridox",
    version,
    about = "gridoxide's command line: power flow, state estimation, short circuit, OPF, \
             security analysis and remedial actions",
    arg_required_else_help = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Solve(SolveArgs),
    Estimate(EstimateArgs),
    Dc(DcArgs),
    Sensitivity(SensitivityArgs),
    ShortCircuit(ShortCircuitArgs),
    Qv(QvArgs),
    Continuation(ContinuationArgs),
    #[command(subcommand)]
    Dynamics(DynamicsCommand),
    Opf(OpfArgs),
    Security(SecurityArgs),
    Rao(RaoArgs),
    CgmesSecurity(CgmesSecurityArgs),
    Switches(SwitchesArgs),
}

/// Run an AC power flow, optionally with the outer loops that make a control a
/// control rather than a constant.
///
/// On-load tap changers holding a voltage or a flow, generators respecting
/// their reactive limits, and the system imbalance shared across generators
/// instead of dumped on one slack. All compose; without any of them this is an
/// ordinary Newton solve.
///
/// The network may be CGMES (a directory, or the profile .xml files), UCTE-DEF
/// (.uct), IIDM (.xiidm) or MATPOWER (.m). Needs the matching importer feature.
#[derive(Args)]
pub struct SolveArgs {
    /// The network: a CGMES directory or profile, .uct, .xiidm or .m
    pub network: String,
    /// Let on-load tap changers hold their voltage or flow
    #[arg(long)]
    pub control_taps: bool,
    /// Switch a generator to fixed Q at its reactive limit
    #[arg(long)]
    pub enforce_q_limits: bool,
    /// Attribute each voltage-controlled bus's reactive power to the machines
    /// holding it, and name any machine at its own limit while the bus is not.
    /// CGMES only: no other format states per-machine reactive capability
    #[arg(long)]
    pub dispatch: bool,
    /// Share the system imbalance across generators instead of the slack
    #[arg(long)]
    pub distribute_slack: bool,
    /// Hold each control area's net export at what the file says it agreed
    /// to. Generalizes --distribute-slack. CGMES supplies areas and schedule;
    /// UCTE areas from its ##Z country codes, with every target zero
    #[arg(long)]
    pub area_interchange: bool,
    /// Move a machine that regulates a bus other than its own onto that
    /// machine; without it the reactive power appears at the bus being held.
    /// It changes answers, which is why it is opt-in. CGMES only
    #[arg(long)]
    pub control_remote_voltage: bool,
    /// Newton mismatch tolerance, per-unit
    #[arg(long, value_name = "T", default_value = "1e-8")]
    pub tol: f64,
    /// Newton iterations per solve
    #[arg(long, value_name = "N", default_value_t = 30)]
    pub max_iter: usize,
    /// Outer-loop rounds
    #[arg(long, value_name = "N", default_value_t = 40)]
    pub max_outer: usize,
    /// Tap steps a changer may move per round
    #[arg(long, value_name = "N", default_value_t = 3)]
    pub max_tap_shift: usize,
}

/// Run state estimation over a PGM JSON document containing voltage, power or
/// current sensors.
///
/// Newton-Raphson by default; --iterative-linear selects the faster, less exact
/// linearized method.
#[derive(Args)]
pub struct EstimateArgs {
    /// A PGM JSON document with sensors
    pub path: String,
    /// Use the linearized method instead of Newton-Raphson
    #[arg(long)]
    pub iterative_linear: bool,
    /// Estimate in the phase domain: three buses per node, each asym sensor
    /// reading its own phase rather than a balanced total. What makes an
    /// unbalanced network estimable rather than approximable. Refuses, by
    /// name, the components the three-phase model does not carry
    #[arg(long)]
    pub asymmetric: bool,
    /// Drop the constraint holding each node with no load or generator at zero
    /// injection: it then injects whatever its sensors say, and one without
    /// sensors is unobservable. A source's node and a three-winding star point
    /// keep theirs, which gridoxide's model of them needs
    #[arg(long)]
    pub no_zero_injection: bool,
    /// Make each source's node the slack, with no impedance behind it, instead
    /// of power-grid-model's virtual bus behind the source impedance
    /// (symmetric only)
    #[arg(long)]
    pub ideal_sources: bool,
}

/// Run a DC (Bθ) power flow.
///
/// Prints bus angles, branch flows and per-island slack pickup. Bus and branch
/// arguments are gridoxide's own 0-based indices, as the tables print them.
/// Shunts are left out, as the DC model has none.
#[derive(Args)]
pub struct DcArgs {
    /// The network: PGM JSON (.json), .m, .uct, .xiidm or CGMES
    pub path: String,
    /// Use b = x/(r²+x²) instead of the default b = 1/x
    #[arg(long)]
    pub ignore_g: bool,
    /// Also print the PTDF column of this bus
    #[arg(long, value_name = "BUS")]
    pub ptdf: Option<usize>,
    /// Also print the LODF column of this branch
    #[arg(long, value_name = "BRANCH")]
    pub lodf: Option<usize>,
}

/// Solve an AC power flow and differentiate it.
///
/// --dp/--dq/--dk/--dalpha each pick one variable and print how every branch
/// flow and bus voltage responds to it. --watch picks one branch instead and
/// prints the reverse: what every injection and every tap would do to its flow.
/// At least one is needed. Branches are lines first, then transformers.
#[derive(Args)]
#[command(group(
    ArgGroup::new("question")
        .required(true)
        .multiple(true)
        .args(["dp", "dq", "dk", "dalpha", "watch"])
))]
pub struct SensitivityArgs {
    /// The network: PGM JSON (.json), .m, .uct, .xiidm or CGMES
    pub path: String,
    /// Active injection at this bus
    #[arg(long, value_name = "BUS")]
    pub dp: Option<usize>,
    /// Reactive injection at this bus
    #[arg(long, value_name = "BUS")]
    pub dq: Option<usize>,
    /// Ratio of this transformer branch
    #[arg(long, value_name = "BRANCH")]
    pub dk: Option<usize>,
    /// Phase-shift angle of this transformer branch
    #[arg(long, value_name = "BRANCH")]
    pub dalpha: Option<usize>,
    /// What would move this branch's active flow
    #[arg(long, value_name = "BRANCH")]
    pub watch: Option<usize>,
    /// The branch end whose flow is meant
    #[arg(long, value_enum, default_value_t = Terminal::From)]
    pub terminal: Terminal,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Terminal {
    From,
    To,
}

/// Run an IEC 60909 short-circuit calculation over a PGM JSON document
/// containing `fault` entries.
///
/// Prints per-node voltages, per-fault currents and each source's contribution.
#[derive(Args)]
pub struct ShortCircuitArgs {
    /// A PGM JSON document with faults
    pub path: String,
    /// The voltage factor c: max for the largest current, min for the
    /// sensitivity study
    #[arg(long, value_enum, default_value_t = Scaling::Max)]
    pub scaling: Scaling,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Scaling {
    #[value(alias = "maximum")]
    Max,
    #[value(alias = "minimum")]
    Min,
}

/// Trace a bus's Q-V curve, for its reactive margin.
///
/// How much reactive support holding the bus at a given voltage takes, and how
/// much margin there is before no setpoint is reachable. The minimum of the curve is the reactive margin, in MVAr. Complements
/// `continuation`, which asks how much further the whole system can be loaded;
/// the two measure related but different things and are known not to agree on
/// which bus is weakest.
#[derive(Args)]
pub struct QvArgs {
    /// The network: PGM JSON (.json), .m, .uct, .xiidm or CGMES
    pub path: String,
    /// Trace this bus and print its curve
    #[arg(long, value_name = "BUS", conflicts_with = "weakest")]
    pub bus: Option<usize>,
    /// Rank buses by margin and print the n weakest
    #[arg(long, value_name = "N")]
    pub weakest: Option<usize>,
    /// Top of the sweep, per-unit
    #[arg(long, value_name = "V", default_value_t = 1.10)]
    pub v_max: f64,
    /// Bottom of the sweep, per-unit
    #[arg(long, value_name = "V", default_value_t = 0.40)]
    pub v_min: f64,
    /// Sweep step, per-unit
    #[arg(long, value_name = "S", default_value_t = 0.01)]
    pub step: f64,
    /// Report the sampled minimum instead of interpolating it
    #[arg(long)]
    pub no_refine: bool,
    /// A PGM document's system base, MVA (default 100); other formats state
    /// their own
    #[arg(long, value_name = "M")]
    pub base_mva: Option<f64>,
    /// Print the curve's points
    #[arg(long)]
    pub curve: bool,
}

/// Trace the P-V curve to the point of voltage collapse.
///
/// How much further the network can be loaded, which bus gives way first, and
/// where each generator runs out of reactive capability on the way. The answer
/// is a property of the direction loading grows in: the default grows every
/// net consumer at constant power factor and leaves the slack to pick it up.
#[derive(Args)]
pub struct ContinuationArgs {
    /// The network: PGM JSON (.json), .m, .uct, .xiidm or CGMES
    pub path: String,
    /// Stop at this loading
    #[arg(long, value_name = "LAMBDA", conflicts_with = "lower_branch")]
    pub target_lambda: Option<f64>,
    /// Walk past the nose and back down the lower branch
    #[arg(long)]
    pub lower_branch: bool,
    /// Continuation step
    #[arg(long, value_name = "S", default_value_t = 0.1)]
    pub step: f64,
    /// Most steps to take
    #[arg(long, value_name = "N", default_value_t = 200)]
    pub max_steps: usize,
    /// Respect reactive limits. Usually what decides where the nose is; the
    /// lambda each machine saturates at is located, not rounded to a step
    #[arg(long)]
    pub enforce_q_limits: bool,
    /// The curve's parametrization (default: the library's)
    #[arg(long, value_enum)]
    pub parametrization: Option<Parametrization>,
    /// Share the pickup over these buses instead of the slack
    #[arg(long, value_name = "I,J,K", value_delimiter = ',')]
    pub pickup: Option<Vec<usize>>,
    /// Stress only these buses
    #[arg(long, value_name = "I,J,K", value_delimiter = ',')]
    pub buses: Option<Vec<usize>>,
    /// A PGM document's system base, MVA (default 100); other formats state
    /// their own
    #[arg(long, value_name = "M")]
    pub base_mva: Option<f64>,
    /// Corrector tolerance
    #[arg(long, value_name = "T", default_value = "1e-8")]
    pub tol: f64,
    /// Corrector iterations per step
    #[arg(long, value_name = "N", default_value_t = 30)]
    pub max_iter: usize,
    /// Print every point of the walk
    #[arg(long)]
    pub curve: bool,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Parametrization {
    Local,
    Natural,
    Arclength,
}

/// RMS dynamics over a gridoxide JSON document with a `dynamics` section.
#[derive(Subcommand)]
pub enum DynamicsCommand {
    Simulate(SimulateArgs),
    Modes(ModesArgs),
}

/// The document and how to read it, for both dynamics commands.
#[derive(Args)]
pub struct DynamicsDocument {
    /// A gridoxide JSON document with a `dynamics` section
    pub path: String,
    /// Carry rotor speed on the machines' speed-voltage terms and write their
    /// swing equations in torque, as Dynawo and Sauer & Pai do. Overrides the
    /// document. Worth 0.6% of terminal power at a 0.9% speed deviation
    #[arg(long, conflicts_with = "no_speed_voltages")]
    pub speed_voltages: bool,
    /// The omega ≈ 1 assumption the phasor formulation and the closed-form
    /// gates rest on. Overrides the document
    #[arg(long)]
    pub no_speed_voltages: bool,
}

/// Run an RMS (phasor-domain) simulation: what the network does over time
/// after a disturbance.
///
/// The document carries the machines, their exciters, governors and
/// stabilizers, and the faults, trips and load steps to apply. Everything
/// starts from a solved power flow, and every device is initialized so that
/// nothing moves until the first event does. Prints how each machine fared
/// and what the voltages did.
#[derive(Args)]
pub struct SimulateArgs {
    #[command(flatten)]
    pub document: DynamicsDocument,
    /// End time, s
    #[arg(long, value_name = "T", default_value_t = 10.0)]
    pub stop: f64,
    /// Time step, s
    #[arg(long, value_name = "H", default_value_t = 0.005)]
    pub step: f64,
    /// Newton tolerance per step
    #[arg(long, value_name = "T", default_value = "1e-9")]
    pub tol: f64,
    /// Backward-Euler steps after each event, damping the switch
    #[arg(long, value_name = "N", default_value_t = 2)]
    pub damping: usize,
    /// The network solver
    #[arg(long, value_enum, default_value_t = Backend::KluNative)]
    pub backend: Backend,
    /// Write the whole trajectory to this CSV file
    #[arg(long, value_name = "OUT")]
    pub csv: Option<String>,
    /// Write only the columns whose names contain this
    #[arg(long, value_name = "SUBSTRING", requires = "csv")]
    pub observe: Option<String>,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Backend {
    Scalar,
    KluNative,
}

/// Report the least-damped modes of the linearized system: which oscillations
/// exist, how fast each decays, and which machines take part.
///
/// Answers what no single run can, and is the only reliable way to find a
/// negatively damped mode. Below two thousand states this computes every mode;
/// past that, a sparse method computes only those nearest a point, named with
/// --freq (and --damping-ratio) or --near. Naming one selects the sparse method
/// at any size, which is how to ask about one band of a large system.
#[derive(Args)]
pub struct ModesArgs {
    #[command(flatten)]
    pub document: DynamicsDocument,
    /// How many modes to report
    #[arg(long, value_name = "N")]
    pub count: usize,
    /// Aim the sparse method at this frequency, Hz
    #[arg(long, value_name = "HZ", conflicts_with = "near")]
    pub freq: Option<f64>,
    /// How far off the axis to aim, as a damping ratio (default 0.05)
    #[arg(long, value_name = "ZETA", requires = "freq")]
    pub damping_ratio: Option<f64>,
    /// Aim the sparse method at this point of the complex plane
    #[arg(long, value_name = "RE,IM", value_parser = parse_complex, allow_hyphen_values = true)]
    pub near: Option<num_complex::Complex<f64>>,
    /// For the least-damped mode, how far it moves per unit of each machine
    /// parameter the equilibrium does not depend on
    #[arg(long)]
    pub sensitivity: bool,
}

fn parse_complex(raw: &str) -> Result<num_complex::Complex<f64>, String> {
    let (re, im) = raw
        .split_once(',')
        .ok_or_else(|| format!("expected <re>,<im>, got {raw:?}"))?;
    let part = |s: &str| {
        s.trim()
            .parse::<f64>()
            .map_err(|_| format!("{s:?} is not a number"))
    };
    Ok(num_complex::Complex::new(part(re)?, part(im)?))
}

/// Run an optimal power flow over a PGM JSON document: least-cost dispatch
/// subject to generator limits and branch ratings.
///
/// Prints the dispatch, locational marginal prices and whatever binds. Costs
/// and limits come from a companion document, by default <network>.opf.json,
/// the pair `gridoxide-matpower` writes. DC unless --ac. Needs the `opf`
/// feature.
#[derive(Args)]
pub struct OpfArgs {
    /// A PGM JSON document
    pub network: String,
    /// The companion OPF document
    #[arg(long, value_name = "OPF.JSON")]
    pub data: Option<String>,
    /// Solve the full AC problem: voltages, reactive power and losses,
    /// generator P and Q together. Nonconvex, so the answer is a local optimum
    #[arg(long)]
    pub ac: bool,
    /// Forbid load shedding (DC)
    #[arg(long, conflicts_with = "ac")]
    pub no_shedding: bool,
    /// Price of shed load, $/MWh (DC)
    #[arg(long, value_name = "PRICE", conflicts_with = "ac")]
    pub shed_price: Option<f64>,
    /// Use b = 1/x instead of b = x/(r²+x²), the opposite default from `dc`,
    /// on purpose: each matches its own field's reference tools (DC)
    #[arg(long, conflicts_with = "ac")]
    pub ignore_r: bool,
    /// Solve with HiGHS instead of the built-in interior-point method; needs
    /// the `opf-highs` feature (DC)
    #[arg(long, conflicts_with = "ac")]
    pub highs: bool,
    /// Drop branch ratings (AC)
    #[arg(long, requires = "ac")]
    pub no_limits: bool,
    /// Cap the interior-point iterations (AC)
    #[arg(long, value_name = "N", requires = "ac")]
    pub max_iter: Option<usize>,
}

/// Assess a network against a CRAC: which critical elements are overloaded,
/// in which state, and by how much.
///
/// The network may be UCTE-DEF (.uct), IIDM (.xiidm) or CGMES; the CRAC
/// OpenRAO JSON or gridoxide's own <network>.rao.json. Exits 0 when every
/// margin is non-negative, 1 when any is not, 2 when it could not tell.
#[derive(Args)]
pub struct SecurityArgs {
    /// The network
    pub network: String,
    /// The CRAC
    #[arg(long, value_name = "CRAC.JSON")]
    pub crac: String,
    /// Print JSON
    #[arg(long)]
    pub json: bool,
}

/// Optimize remedial actions against a CRAC, and report what to do in each
/// state.
///
/// Searches over the network actions the CRAC permits, re-optimizing the range
/// actions at every candidate. Exits 0 when every perimeter ends secure, 1 when one does not, 2 when it
/// could not tell.
#[derive(Args)]
pub struct RaoArgs {
    /// The network: .uct, .xiidm or CGMES
    pub network: String,
    /// The CRAC
    #[arg(long, value_name = "CRAC.JSON")]
    pub crac: String,
    /// OpenRAO's RaoParameters document: objective (MIN_COST included), flow
    /// model, MNECs, second preventive and search thresholds
    #[arg(long, value_name = "P.JSON")]
    pub parameters: Option<String>,
    /// How many network actions may be stacked (default 2); wins over
    /// --parameters
    #[arg(long, value_name = "N")]
    pub depth: Option<usize>,
    /// Score and give the final verdict on AC flows
    #[arg(long)]
    pub validate_ac: bool,
    /// Print JSON
    #[arg(long)]
    pub json: bool,
}

/// Security analysis of a CGMES model over the contingencies of its CO files.
///
/// The base case and each contingency, current limits monitored, optionally
/// written as a SAR. Exits 0 when no limit is violated, 1 when one is, 2 when it could not tell.
/// Needs the `cgmes` feature.
#[derive(Args)]
pub struct CgmesSecurityArgs {
    /// The model's profiles
    #[arg(required = true, value_name = "PROFILES")]
    pub profiles: Vec<String>,
    /// The CO profiles
    #[arg(long, value_name = "CO", num_args = 1.., required = true)]
    pub contingencies: Vec<String>,
    /// How machines share the imbalance: capability (by maxOperatingP, the
    /// default), reference or equal
    #[arg(long, value_name = "SHARING")]
    pub slack: Option<String>,
    /// Write the results as a SAR
    #[arg(
        long,
        value_name = "OUT.XML",
        requires_all = ["publisher", "version_of", "rights_holder", "generated_at"]
    )]
    pub sar: Option<String>,
    /// The SAR's publisher
    #[arg(long, value_name = "IRI", requires = "sar")]
    pub publisher: Option<String>,
    /// What the SAR is a version of
    #[arg(long, value_name = "IRI", requires = "sar")]
    pub version_of: Option<String>,
    /// The SAR's rights holder
    #[arg(long, value_name = "NAME", requires = "sar")]
    pub rights_holder: Option<String>,
    /// When the SAR was generated
    #[arg(long, value_name = "TIME", requires = "sar")]
    pub generated_at: Option<String>,
}

/// List a CGMES model's switching devices, read from EQ+SSH rather than merged
/// away at import.
///
/// Needs the `cgmes` feature.
#[derive(Args)]
pub struct SwitchesArgs {
    /// The model's profiles
    #[arg(required = true, value_name = "PROFILE.XML")]
    pub profiles: Vec<String>,
    /// Which switches survive as elements
    #[arg(long, value_enum, default_value_t = Retain::BusbarAdjacent)]
    pub retain: Retain,
    /// Open this switch (label or index) before solving
    #[arg(long, value_name = "MRID")]
    pub open: Option<String>,
    /// Run a power flow and report each switch's own flow
    #[arg(long)]
    pub solve: bool,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Retain {
    #[value(name = "none")]
    MergeAll,
    #[value(name = "busbar_adjacent")]
    BusbarAdjacent,
    All,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_definitions_are_consistent() {
        Cli::command().debug_assert();
    }
}
