pub mod ac_sensitivity;
pub mod batch;
pub mod bde;
pub mod block_sparse;
pub mod branch_flow;
#[cfg(feature = "capi")]
pub mod capi;
#[cfg(feature = "cgmes")]
pub mod cgmes;
pub mod constrained;
pub mod continuation;
pub mod dc;
pub mod dispatch;
#[cfg(feature = "dynamics")]
pub mod dynamics;
#[cfg(feature = "iidm")]
pub mod iidm;
pub mod injection_hessian;
pub mod jacobian;
pub mod json;
pub mod klu_native;
pub mod linear;
pub mod lu;
pub mod measurement;
pub mod network;
#[cfg(feature = "opf")]
pub mod opf;
pub mod outerloop;
pub mod pgm;
#[cfg(feature = "python")]
mod python;
pub mod qv;
#[cfg(feature = "rao")]
pub mod rao;
pub mod ratings;
pub mod roots;
pub mod scenario;
pub mod se;
pub mod shortcircuit;
pub mod solver;
pub mod sparse;
#[cfg(feature = "klu")]
pub mod sparse_klu;
#[cfg(feature = "pardiso")]
pub mod sparse_pardiso;
pub mod switches;
pub mod topology;
pub mod types;
#[cfg(feature = "ucte")]
pub mod ucte;

use json::NetworkData;
use linear::{
    btheta::{DcIslandStatus, DcSolution, dc_power_flow},
    impedance::{LinearIslandStatus, LinearReport, linear_power_flow},
};
use network::{ShuntAdm, YBus, build_ybus, linear_initial_guess, stamp_shunts};
use solver::{IslandReport, JacobianBackend, PersistentSolver, SolveStats, SolveStatus};
use types::{Bus, Line, Transformer};

/// Which method [`crate::run_power_flow`] uses.
///
/// The two linear methods are genuinely different algorithms, not two
/// spellings of one — see `docs/src/powerflow/dc.md` and
/// `linear_impedance.md`. Both are direct solves, so neither consults
/// [`PowerFlowOptions::tol`] or `max_iter`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PowerFlowMethod {
    /// The full nonlinear solve. Everything else in this crate's history.
    #[default]
    NewtonRaphson,
    /// Real Bθ: active power only, `|V| = 1`, resistance discarded. One
    /// factorization, no iterations. See
    /// [`linear::btheta::dc_power_flow`].
    Dc,
    /// Complex constant-admittance: every load replaced by the admittance
    /// drawing its rated power at `|V| = 1`. Mathematically
    /// power-grid-model's `CalculationMethod.linear`. See
    /// [`linear::impedance::linear_power_flow`].
    LinearImpedance,
}

/// Where [`PowerFlowMethod::NewtonRaphson`] starts from. Ignored by the
/// direct methods, which have no starting point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PowerFlowInit {
    /// `|V| = 1`, `θ = 0`, untouched.
    Flat,
    /// One constant-admittance solve, via `network::linear_initial_guess`.
    ///
    /// The default because it is what this crate has always done — see
    /// [`crate::run_power_flow_analysis_from_ybus`], which calls it
    /// unconditionally — and because it is power-grid-model's only
    /// robustness mechanism, needed where weak sources meet large transformer
    /// phase shifts and flat-start Newton diverges.
    #[default]
    LinearImpedance,
    /// One DC solve, seeding angles and **only** angles — bus types,
    /// injections and magnitudes are all restored afterwards, since
    /// `linear::btheta::dc_power_flow` is a solver rather than an
    /// initializer and would otherwise pre-empt Newton's own classification
    /// pass. DC has no magnitude information to offer in any case, beyond the
    /// flat `|V| = 1` it assumes. This is powsybl's
    /// `DcValueVoltageInitializer`.
    ///
    /// Opt-in rather than default only because `LinearImpedance` is the
    /// long-standing behavior, not because it is weaker. Measured on the
    /// committed PGM fixtures
    /// (`tests/power_flow_methods_test.rs::both_warm_starts_rescue_a_case_flat_start_cannot_solve`):
    /// on `distribution-case`, where a flat start does *not* converge inside
    /// 20 iterations, DC init converges in 5 against `LinearImpedance`'s 4.
    /// Both rescue the case; neither dominates.
    ///
    /// The one thing DC init cannot do is supply voltage magnitudes — it has
    /// none to give beyond the flat 1 p.u. it assumes — so on networks whose
    /// magnitudes stray far from nominal, `LinearImpedance` starts with
    /// strictly more information.
    Dc,
}

/// Everything [`crate::run_power_flow`] needs to know.
///
/// The pre-existing entry points — [`solver::newton_raphson`],
/// [`solver::newton_raphson_with_backend`], [`solver::PersistentSolver::solve`],
/// the outer-loop driver — keep their positional parameters
/// and are unaffected by this type. It exists so that adding a fifth or sixth
/// knob does not mean a fifth or sixth positional argument, following
/// `se::nr::SeOptions`.
#[derive(Clone, Debug)]
pub struct PowerFlowOptions {
    /// Convergence tolerance on the maximum P/Q mismatch. Newton only.
    pub tol: f64,
    /// Newton only.
    pub max_iter: usize,
    /// Which sparse-LU backend solves the Jacobian. Newton only: the direct
    /// methods each own their factorization type.
    pub backend: JacobianBackend,
    pub method: PowerFlowMethod,
    pub init: PowerFlowInit,
    /// Adds [`crate::outerloop::ReactiveLimits`] to the outer-loop list.
    pub enforce_q_limits: bool,
    /// Adds [`crate::outerloop::DistributedSlack`] to the outer-loop list,
    /// innermost, with these participation weights.
    ///
    /// Mutually exclusive with [`area_interchange`](Self::area_interchange),
    /// which subsumes it — configuring both would move the same schedules
    /// twice, and `run_power_flow` refuses rather than doing so.
    pub distribute_slack: Option<crate::outerloop::SlackDistribution>,
    /// Adds [`crate::outerloop::AreaInterchange`] in distributed slack's place,
    /// holding each control area's net export at its scheduled value.
    pub area_interchange: Option<crate::outerloop::AreaDefinition>,
    /// Adds [`crate::outerloop::PhaseControl`] and
    /// [`crate::outerloop::TransformerVoltageControl`], outermost. Needs an
    /// importer that retained tap tables and regulating controls; with none,
    /// both loops are stable on their first check and cost one call each.
    pub control_taps: bool,
    /// Adds [`crate::outerloop::RemoteVoltageControl`] to the loop list, for
    /// machines that regulate a bus other than their own.
    ///
    /// Off by default, and the default is *wrong* rather than merely
    /// conservative: without it a remote machine's reactive power appears at the
    /// bus it holds instead of at the machine, so the reactive flow between them
    /// is missing. It is opt-in because turning it on changes answers, and
    /// because it needs network data only CGMES supplies — see
    /// [`crate::RemoteControlData`].
    pub control_remote_voltage: bool,
    /// Positions a tap controller may move in one outer pass. The sensitivity
    /// is exact only at the converged state, so a large step overshoots;
    /// capping trades passes for stability.
    pub tap_max_shift: i32,
    /// Cap on total outer-loop re-solves across every configured loop.
    pub max_outer_iter: usize,
    /// Susceptance choice and transformer-ratio handling for
    /// [`PowerFlowMethod::Dc`] and for [`PowerFlowInit::Dc`].
    pub dc: crate::linear::DcOptions,
}

impl Default for PowerFlowOptions {
    fn default() -> Self {
        Self {
            // The values `run_power_flow_analysis_from_ybus` has always
            // hard-coded, so the default options reproduce today's behavior
            // exactly.
            tol: 1e-6,
            max_iter: 20,
            backend: JacobianBackend::Scalar,
            method: PowerFlowMethod::default(),
            init: PowerFlowInit::default(),
            enforce_q_limits: false,
            distribute_slack: None,
            area_interchange: None,
            control_taps: false,
            control_remote_voltage: false,
            tap_max_shift: 3,
            max_outer_iter: 10,
            dc: crate::linear::DcOptions::default(),
        }
    }
}

/// Result of a full power-flow analysis: the solved (or placeholder, for
/// unreferenced islands) buses, plus a per-connected-component breakdown of
/// how each one was resolved — see [`solver::IslandReport`]/
/// [`solver::IslandStatus`].
#[derive(Debug)]
pub struct PowerFlowReport {
    pub buses: Vec<Bus>,
    /// Newton-Raphson's per-component breakdown. **Empty** for the two direct
    /// methods, which report through [`dc`](Self::dc) and
    /// [`linear`](Self::linear) instead.
    ///
    /// This is not laziness. `IslandStatus::MaxIterationsReached` cannot
    /// happen without iterations, and `IslandStatus::AmbiguousReferenceBus`
    /// means "over-determined, the answer may satisfy neither reference" —
    /// which is true of AC and false of DC, where two references leave the
    /// system perfectly well posed. Mapping onto this vocabulary would state
    /// things about a DC solve that are not so.
    pub islands: Vec<IslandReport>,
    /// Iteration count and per-iteration convergence trace. The solver
    /// itself prints nothing (see [`solver::SolveStats`]); `src/main.rs`
    /// reconstructs the progress output from this.
    ///
    /// For the direct methods this carries only a status — `Converged` or
    /// `Singular` — with an empty history, since they take no iterations.
    pub stats: SolveStats,
    /// Populated only by [`PowerFlowMethod::Dc`]: per-branch active flows,
    /// per-island slack pickup, and the solve's residual.
    pub dc: Option<DcSolution>,
    /// Populated only by [`PowerFlowMethod::LinearImpedance`].
    pub linear: Option<LinearReport>,
    /// What the outer loops did, when any were configured. `None` for a plain
    /// Newton solve and for the two direct methods.
    pub outer: Option<OuterLoopOutcome>,
}

/// The tap tables and regulating controls a solve may act on.
///
/// Network data, so it travels in the signature rather than in
/// [`PowerFlowOptions`]. A caller with no tap control passes
/// [`TapData::none`], which is what makes
/// [`PowerFlowOptions::control_taps`] a no-op rather than a lie.
#[derive(Clone, Copy, Debug, Default)]
pub struct TapData<'a> {
    /// Parallel to the `transformers` slice.
    pub changers: &'a [Option<types::TapChanger>],
    pub regulation: &'a [outerloop::TapRegulation],
}

impl TapData<'_> {
    /// No tap tables and no controls: what every importer that does not retain
    /// them supplies, and what a caller uninterested in tap control passes.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.changers.is_empty() || self.regulation.is_empty()
    }
}

/// The regulating machines a solve may act on.
///
/// Network data, so it travels in the signature rather than in
/// [`PowerFlowOptions`] — the same place and for the same reason as
/// [`TapData`]. A caller with none passes [`RemoteControlData::none`], which is
/// what makes [`PowerFlowOptions::control_remote_voltage`] a no-op
/// rather than a lie.
#[derive(Clone, Copy, Debug, Default)]
pub struct RemoteControlData<'a> {
    pub machines: &'a [types::RegulatingMachine],
}

impl RemoteControlData<'_> {
    /// No machines: what every importer but CGMES supplies.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.machines.iter().all(|m| m.at_bus == m.controls_bus)
    }
}

/// The outer loops' side of a solve, gathered so a caller need not downcast
/// [`outerloop::OuterLoop`] trait objects to read their reports.
#[derive(Clone, Debug, Default)]
pub struct OuterLoopOutcome {
    /// Per-loop iteration counts and final statuses.
    pub report: Option<outerloop::OuterLoopReport>,
    /// Buses switched `PV → PQ`, in switch order. Empty unless
    /// [`PowerFlowOptions::enforce_q_limits`].
    pub q_limit_switches: Vec<usize>,
    /// Populated when [`PowerFlowOptions::distribute_slack`] was set.
    pub slack: Option<outerloop::SlackDistributionReport>,
    /// Populated when [`PowerFlowOptions::area_interchange`] was set.
    pub area: Option<outerloop::AreaInterchangeReport>,
    /// Populated when [`PowerFlowOptions::control_taps`] was set:
    /// the voltage controllers, then the phase controllers.
    pub taps: Vec<outerloop::ControllerReport>,
    /// Populated when [`PowerFlowOptions::control_remote_voltage`] was
    /// set: what each remotely-regulating machine reached, and how.
    pub remote: Vec<outerloop::RemoteControlReport>,
    /// Buses this run re-typed to move a control onto the machine holding it,
    /// as `(bus, from, to)`. Worth reporting: the network solved is not quite
    /// the one handed over.
    pub retyped: Vec<(usize, types::BusType, types::BusType)>,
    /// The transformers as the loops left them — tap positions moved. Empty
    /// unless a tap loop ran, since nothing else changes a transformer.
    pub transformers: Vec<Transformer>,
    /// The tap changers as the loops left them, parallel to `transformers`.
    pub changers: Vec<Option<types::TapChanger>>,
}

impl PowerFlowReport {
    /// The shape every method-specific constructor starts from.
    fn empty(buses: Vec<Bus>, status: SolveStatus) -> Self {
        Self {
            buses,
            islands: Vec::new(),
            stats: SolveStats {
                status,
                mismatch_history: Vec::new(),
            },
            dc: None,
            linear: None,
            outer: None,
        }
    }
}

pub fn run_power_flow_analysis(network_data: NetworkData) -> PowerFlowReport {
    let ybus = build_ybus(network_data.buses.len(), &network_data.lines, &[]);
    run_power_flow_analysis_from_ybus(network_data.buses, ybus)
}

/// Runs a power flow by the method named in `opts`.
///
/// Takes branch lists rather than a finished Y-bus because
/// [`PowerFlowMethod::Dc`] is formulated per branch — it needs each branch's
/// own `x` and `tap`, which a Y-bus has already summed away — and never
/// builds one at all. The other two methods assemble their Y-bus here, from
/// these same lists plus `shunts`.
///
/// [`run_power_flow_analysis`] and [`run_power_flow_analysis_from_ybus`] are
/// unchanged and remain the shortest path to an ordinary Newton solve;
/// `PowerFlowOptions::default()` here reproduces exactly what they do.
pub fn run_power_flow(
    buses: Vec<Bus>,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    taps: TapData<'_>,
    opts: PowerFlowOptions,
) -> PowerFlowReport {
    run_power_flow_with_remote(
        buses,
        lines,
        transformers,
        shunts,
        taps,
        RemoteControlData::none(),
        opts,
    )
}

/// [`run_power_flow`] with the regulating machines a remote-voltage control
/// needs. The two-argument form is the shorter path for every caller that has
/// none, which is every importer but CGMES.
pub fn run_power_flow_with_remote(
    mut buses: Vec<Bus>,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    taps: TapData<'_>,
    remote: RemoteControlData<'_>,
    opts: PowerFlowOptions,
) -> PowerFlowReport {
    match opts.method {
        PowerFlowMethod::Dc => {
            let solution = dc_power_flow(&mut buses, lines, transformers, opts.dc);
            let status = if solution
                .islands
                .iter()
                .any(|i| i.status == DcIslandStatus::Singular)
            {
                SolveStatus::Singular
            } else {
                SolveStatus::Converged
            };
            let mut report = PowerFlowReport::empty(buses, status);
            report.dc = Some(solution);
            report
        }
        PowerFlowMethod::LinearImpedance => {
            let ybus = finished_ybus(buses.len(), lines, transformers, shunts);
            let solution = linear_power_flow(&mut buses, &ybus);
            let status = if solution
                .islands
                .iter()
                .any(|i| i.status == LinearIslandStatus::Singular)
            {
                SolveStatus::Singular
            } else {
                SolveStatus::Converged
            };
            let mut report = PowerFlowReport::empty(buses, status);
            report.linear = Some(solution);
            report
        }
        PowerFlowMethod::NewtonRaphson => {
            let ybus = finished_ybus(buses.len(), lines, transformers, shunts);
            match opts.init {
                PowerFlowInit::Flat => {}
                PowerFlowInit::LinearImpedance => linear_initial_guess(&mut buses, &ybus),
                // Seed angles and *only* angles.
                //
                // `dc_power_flow` is a solver, not an initializer: it also
                // normalizes PQ magnitudes to its own |V| = 1 assumption, and
                // it runs `network::mark_unreferenced_islands`, which rewrites
                // a sourceless island's buses to `Slack` with zero injection.
                // Both are right for a DC solve and wrong here — Newton has
                // its own classification pass, and letting the initializer
                // pre-empt it changes the island statuses that pass reports
                // (a sourceless island comes back `AmbiguousReferenceBus`, or
                // for a single bus, silently `Converged`). See
                // `network::linear_initial_guess`, which states the same
                // invariant for the other initializer.
                //
                // So everything but `voltage_ang` is restored afterwards.
                PowerFlowInit::Dc => {
                    let saved: Vec<(types::BusType, f64, f64, f64)> = buses
                        .iter()
                        .map(|b| (b.bus_type, b.p_spec, b.q_spec, b.voltage_mag))
                        .collect();
                    dc_power_flow(&mut buses, lines, transformers, opts.dc);
                    for (bus, (bus_type, p_spec, q_spec, voltage_mag)) in
                        buses.iter_mut().zip(saved)
                    {
                        bus.bus_type = bus_type;
                        bus.p_spec = p_spec;
                        bus.q_spec = q_spec;
                        bus.voltage_mag = voltage_mag;
                    }
                }
            }

            newton_with_loops(buses, lines, transformers, shunts, taps, remote, ybus, opts)
        }
    }
}

/// The Newton branch of [`run_power_flow`]: assemble the configured outer
/// loops, run them to a fixed point, and gather their reports.
///
/// With no loop configured this is one ordinary
/// [`PersistentSolver::solve_with_stats`] call and the report's `outer` field
/// stays `None` — the path every existing caller takes, unchanged in what it
/// computes.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn newton_with_loops(
    mut buses: Vec<Bus>,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
    taps: TapData<'_>,
    remote: RemoteControlData<'_>,
    mut ybus: network::YBusSparse,
    opts: PowerFlowOptions,
) -> PowerFlowReport {
    let wants_taps = opts.control_taps && !taps.is_empty();
    let wants_remote = opts.control_remote_voltage && !remote.is_empty();
    if !opts.enforce_q_limits
        && opts.distribute_slack.is_none()
        && opts.area_interchange.is_none()
        && !wants_taps
        && !wants_remote
    {
        let mut solver = PersistentSolver::new(opts.backend);
        let (islands, stats) = solver.solve_with_stats(&mut buses, &ybus, opts.tol, opts.max_iter);
        return PowerFlowReport {
            buses,
            islands,
            stats,
            dc: None,
            linear: None,
            outer: None,
        };
    }

    // Both would move the same schedules, so this is a caller mistake rather
    // than a combination to resolve. Refusing beats picking one silently.
    assert!(
        !(opts.distribute_slack.is_some() && opts.area_interchange.is_some()),
        "distribute_slack and area_interchange cannot both be set: area interchange subsumes \
         distributed slack (one area with a zero target *is* distributed slack), so running both \
         would dispatch the same generators twice"
    );

    // Owned copies, because a tap loop mutates them and the caller handed over
    // shared slices. Returned on the report so the moved positions are not
    // lost — a tap position is an answer here, not an input.
    let mut transformers = transformers.to_vec();
    let mut changers = taps.changers.to_vec();

    let mut slack = opts
        .distribute_slack
        .clone()
        .map(outerloop::DistributedSlack::new);
    let mut area = opts
        .area_interchange
        .clone()
        .map(outerloop::AreaInterchange::new);
    let mut qlim = opts.enforce_q_limits.then(outerloop::ReactiveLimits::new);
    let mut remote_loop =
        wants_remote.then(|| outerloop::RemoteVoltageControl::new(remote.machines));
    let mut phase =
        wants_taps.then(|| outerloop::PhaseControl::new().max_tap_shift(opts.tap_max_shift));
    let mut voltage = wants_taps
        .then(|| outerloop::TransformerVoltageControl::new().max_tap_shift(opts.tap_max_shift));

    let (islands, report) = {
        let mut ctx = outerloop::SolveContext::new(&mut buses, &mut ybus)
            .with_branches(lines, &mut transformers, shunts)
            .with_taps(&mut changers, taps.regulation);

        // Innermost first: the active-power balance, then reactive limits,
        // then the tap controls. Area interchange stands in distributed
        // slack's place rather than beside it — it generalizes it, and running
        // both would move the same schedules twice.
        let mut list: Vec<&mut dyn outerloop::OuterLoop> = Vec::new();
        // Innermost of all: a remote controller decides *where* the reactive
        // power is produced, which every loop after it reads. Running it inside
        // reactive limits also means a machine that saturates is seen by
        // `ReactiveLimits` as the `PV` bus it now is, rather than as a bus it
        // was never connected to.
        if let Some(l) = remote_loop.as_mut() {
            list.push(l);
        }
        if let Some(l) = area.as_mut() {
            list.push(l);
        }
        if let Some(l) = slack.as_mut() {
            list.push(l);
        }
        if let Some(l) = qlim.as_mut() {
            list.push(l);
        }
        if let Some(l) = phase.as_mut() {
            list.push(l);
        }
        if let Some(l) = voltage.as_mut() {
            list.push(l);
        }
        outerloop::solve_with_loops(
            &mut ctx,
            opts.tol,
            opts.max_iter,
            opts.backend,
            &mut list,
            opts.max_outer_iter,
        )
    };

    let mut outcome = OuterLoopOutcome {
        report: Some(report),
        q_limit_switches: qlim
            .as_ref()
            .map(|l| l.switches().to_vec())
            .unwrap_or_default(),
        slack: slack.map(outerloop::DistributedSlack::into_report),
        area: area.map(outerloop::AreaInterchange::into_report),
        taps: Vec::new(),
        remote: remote_loop
            .as_ref()
            .map(|l| l.report_against(&buses))
            .unwrap_or_default(),
        retyped: remote_loop
            .as_ref()
            .map(|l| l.retyped().to_vec())
            .unwrap_or_default(),
        transformers: Vec::new(),
        changers: Vec::new(),
    };
    if let Some(l) = voltage.as_ref() {
        outcome
            .taps
            .extend(l.report(taps.regulation, &changers).controllers);
    }
    if let Some(l) = phase.as_ref() {
        outcome
            .taps
            .extend(l.report(taps.regulation, &changers).controllers);
    }
    if wants_taps {
        outcome.transformers = transformers;
        outcome.changers = changers;
    }

    // The outer loops re-solve through `PersistentSolver::solve`, which does
    // not hand back `SolveStats`; the status is reconstructed from the island
    // reports, which is the same verdict by a different route.
    let status = if islands.iter().all(|i| {
        !matches!(
            i.status,
            solver::IslandStatus::Singular | solver::IslandStatus::MaxIterationsReached
        )
    }) {
        SolveStatus::Converged
    } else if islands
        .iter()
        .any(|i| i.status == solver::IslandStatus::Singular)
    {
        SolveStatus::Singular
    } else {
        SolveStatus::MaxIterationsReached
    };

    PowerFlowReport {
        buses,
        islands,
        stats: SolveStats {
            status,
            mismatch_history: Vec::new(),
        },
        dc: None,
        linear: None,
        outer: Some(outcome),
    }
}

fn finished_ybus(
    n: usize,
    lines: &[Line],
    transformers: &[Transformer],
    shunts: &[ShuntAdm],
) -> network::YBusSparse {
    let mut ybus = build_ybus(n, lines, transformers);
    stamp_shunts(&mut ybus, shunts);
    ybus.finish()
}

/// Runs a power flow analysis given a pre-built Y-bus matrix.
/// Intended for the 3-phase case too, where `buses` is a 3N-element vector
/// and `ybus` is the 3N×3N phase-domain admittance matrix from
/// `build_ybus_3ph` — a physical bus's 3 phase-rows always share the same
/// `BusType`, so the island partitioning `PersistentSolver::solve` does
/// internally is safe for that case too (worst case, in a network with
/// perfectly balanced zero/positive-sequence impedances, phases may
/// partition into separate components rather than staying grouped by
/// physical bus; each still carries its own valid reference, so this
/// doesn't change correctness, only granularity).
///
/// Every disconnected component of `ybus` is solved in this same call (not
/// just the largest one) — see [`solver::PersistentSolver::solve`], the one
/// canonical solve entry point every public function in this crate
/// (including this one) ultimately goes through. `PowerFlowReport::islands`
/// gives the resulting per-component breakdown.
pub fn run_power_flow_analysis_from_ybus(mut buses: Vec<Bus>, ybus: YBus) -> PowerFlowReport {
    let ybus = ybus.finish();
    linear_initial_guess(&mut buses, &ybus);
    let mut solver = PersistentSolver::new(JacobianBackend::Scalar);
    let (islands, stats) = solver.solve_with_stats(&mut buses, &ybus, 1e-6, 20);
    PowerFlowReport {
        buses,
        islands,
        stats,
        dc: None,
        linear: None,
        outer: None,
    }
}
