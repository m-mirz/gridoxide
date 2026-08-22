//! The power-flow handle.
//!
//! Follows the rule [`python`](crate::python) sets out: a handle exists because
//! a factorization survives between calls. Construct once per topology, solve
//! as many times as you like — the symbolic factorization inside
//! [`PersistentSolver`] is computed on the first solve and reused by every one
//! after it.
//!
//! Every solve restarts from a pristine copy of the buses, so calling one after
//! another is idempotent rather than cumulative, and a failed solve cannot
//! poison the next.

use std::ffi::{c_char, c_int, CStr};

use super::types::{
    GridoxideBackend, GridoxideBus, GridoxideLine, GridoxideOptions, GridoxideShunt,
    GridoxideTransformer,
};
use super::{guard, invalid, island_status_code, set_error, write_slice, GridoxideIslandStatus, Status};
use crate::branch_flow::{branch_params, bus_voltages, terminal_flow, BranchParams, Terminal};
use crate::network::{build_ybus, stamp_shunts, ShuntAdm, YBusSparse};
use crate::solver::{
    IslandReport, IslandStatus, JacobianBackend, PersistentSolver, SolveStats,
};
use crate::types::{Bus, Line, Transformer};

/// An opaque power-flow model. **Single-thread-affine** — see the module docs
/// on [`capi`](super).
///
/// In C this is `gridoxide_powerflow`, reached only through a pointer.
pub struct GridoxidePowerFlow {
    template: Vec<Bus>,
    buses: Vec<Bus>,
    ybus: YBusSparse,
    branches: Vec<BranchParams>,
    solver: PersistentSolver,
    options: GridoxideOptions,
    /// `None` until the first solve, so accessors can say "solve first" rather
    /// than serving a flat start as if it were an answer.
    stats: Option<SolveStats>,
    islands: Vec<IslandReport>,
    /// Per-bus schedule shift from the last distributed-slack solve, empty
    /// otherwise.
    slack_shift: Vec<f64>,
}

fn backend_of(backend: GridoxideBackend) -> Result<JacobianBackend, String> {
    match backend {
        GridoxideBackend::Scalar => Ok(JacobianBackend::Scalar),
        GridoxideBackend::Block => Ok(JacobianBackend::Block),
        GridoxideBackend::KluNative => Ok(JacobianBackend::KluNative),
        #[cfg(feature = "klu")]
        GridoxideBackend::Klu => Ok(JacobianBackend::Klu),
        #[cfg(not(feature = "klu"))]
        GridoxideBackend::Klu => Err(
            "backend KLU needs gridoxide built with the `klu` feature".to_string(),
        ),
        #[cfg(feature = "pardiso")]
        GridoxideBackend::Pardiso => Ok(JacobianBackend::Pardiso),
        #[cfg(not(feature = "pardiso"))]
        GridoxideBackend::Pardiso => Err(
            "backend PARDISO needs gridoxide built with the `pardiso` feature".to_string(),
        ),
    }
}

impl GridoxidePowerFlow {
    fn assemble(
        buses: Vec<Bus>,
        lines: Vec<Line>,
        transformers: Vec<Transformer>,
        shunts: Vec<ShuntAdm>,
        options: GridoxideOptions,
    ) -> Result<Self, String> {
        let backend = backend_of(options.backend)?;
        let mut ybus = build_ybus(buses.len(), &lines, &transformers);
        stamp_shunts(&mut ybus, &shunts);
        let ybus = ybus.finish();
        let branches = branch_params(&lines, &transformers);
        Ok(Self {
            buses: buses.clone(),
            template: buses,
            ybus,
            branches,
            solver: PersistentSolver::new(backend),
            options,
            stats: None,
            islands: Vec::new(),
            slack_shift: Vec::new(),
        })
    }

    /// Maps the outcome of a solve onto a status code.
    ///
    /// A non-converged solve is an *error* here, matching how the Python
    /// binding raises on a single solve — but the voltages stay readable, so a
    /// caller who wants to inspect where it got stuck still can. That is the
    /// reason this does not clear the state on failure.
    fn finish(&mut self, islands: Vec<IslandReport>, stats: SolveStats) -> Status {
        self.islands = islands;
        self.stats = Some(stats);

        let worst = self.islands.iter().map(|i| i.status).find(|s| {
            matches!(s, IslandStatus::Singular | IslandStatus::MaxIterationsReached)
        });
        match worst {
            Some(IslandStatus::Singular) => {
                set_error("the Jacobian was singular; the returned state is not a solution");
                Status::Singular
            }
            Some(IslandStatus::MaxIterationsReached) => {
                set_error(format!(
                    "power flow did not converge within {} iterations (largest mismatch {:.3e})",
                    self.options.max_iterations,
                    self.stats.as_ref().map(|s| s.final_mismatch()).unwrap_or(f64::NAN)
                ));
                Status::NotConverged
            }
            // `NoReferenceBus` and `AmbiguousReferenceBus` are deliberately not
            // failures: a de-energized island is a normal feature of a real
            // network, reported per island rather than fatal to the whole call.
            _ => Status::Ok,
        }
    }
}

/// Reads a handle pointer, or reports why it cannot.
///
/// # Safety
///
/// `handle` must be null or a pointer returned by one of the constructors and
/// not yet freed.
unsafe fn handle_ref<'a>(handle: *const GridoxidePowerFlow) -> Result<&'a GridoxidePowerFlow, Status> {
    if handle.is_null() {
        return Err(invalid("handle is null"));
    }
    // SAFETY: non-null and valid by contract.
    Ok(unsafe { &*handle })
}

/// # Safety
///
/// As [`handle_ref`], and no other reference to the handle may be live.
unsafe fn handle_mut<'a>(handle: *mut GridoxidePowerFlow) -> Result<&'a mut GridoxidePowerFlow, Status> {
    if handle.is_null() {
        return Err(invalid("handle is null"));
    }
    // SAFETY: non-null and valid by contract; single-thread-affine, so no
    // other reference can be live on another thread.
    Ok(unsafe { &mut *handle })
}

fn options_or_default(options: *const GridoxideOptions) -> GridoxideOptions {
    if options.is_null() {
        GridoxideOptions::default()
    } else {
        // SAFETY: non-null and readable by contract.
        unsafe { *options }
    }
}

fn build_from_pgm(text: &str, options: GridoxideOptions) -> Result<GridoxidePowerFlow, (Status, String)> {
    let input: crate::pgm::PgmInput = serde_json::from_str(text)
        .map_err(|e| (Status::Parse, format!("parsing PGM JSON: {e}")))?;
    // Shunts must be taken before the conversion consumes the document.
    let id_to_idx = crate::pgm::node_id_to_idx(&input);
    let shunts = crate::pgm::pgm_shunts_1ph(&input, &id_to_idx, options.s_base_va);
    let (buses, lines, transformers) =
        crate::pgm::pgm_to_buses_and_branches(input, options.s_base_va, options.frequency_hz);
    GridoxidePowerFlow::assemble(buses, lines, transformers, shunts, options)
        .map_err(|e| (Status::InvalidArgument, e))
}

/// Builds a model from a power-grid-model JSON file.
///
/// # Safety
///
/// `path` must be a NUL-terminated string; `options` may be null (defaults are
/// used); `out` must be a writable pointer. On success `*out` owns a handle the
/// caller must release with [`gridoxide_powerflow_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_from_pgm_file(
    path: *const c_char,
    options: *const GridoxideOptions,
    out: *mut *mut GridoxidePowerFlow,
) -> Status {
    guard(|| {
        if out.is_null() {
            return invalid("out pointer is null");
        }
        if path.is_null() {
            return invalid("path is null");
        }
        // SAFETY: non-null and NUL-terminated by contract.
        let path = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(p) => p,
            Err(_) => return invalid("path is not valid UTF-8"),
        };
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                set_error(format!("reading {path}: {e}"));
                return Status::Io;
            }
        };
        match build_from_pgm(&text, options_or_default(options)) {
            Ok(model) => {
                // SAFETY: `out` is non-null and writable by contract.
                unsafe { *out = Box::into_raw(Box::new(model)) };
                Status::Ok
            }
            Err((status, message)) => {
                set_error(message);
                status
            }
        }
    })
}

/// Builds a model from a power-grid-model JSON document held in memory.
///
/// `len` is the byte length, so the document need not be NUL-terminated.
///
/// # Safety
///
/// `json` must point to `len` readable bytes; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_from_pgm_string(
    json: *const c_char,
    len: usize,
    options: *const GridoxideOptions,
    out: *mut *mut GridoxidePowerFlow,
) -> Status {
    guard(|| {
        if out.is_null() {
            return invalid("out pointer is null");
        }
        if json.is_null() {
            return invalid("json pointer is null");
        }
        // SAFETY: `len` readable bytes by contract.
        let bytes = unsafe { std::slice::from_raw_parts(json as *const u8, len) };
        let text = match std::str::from_utf8(bytes) {
            Ok(t) => t,
            Err(_) => return invalid("json is not valid UTF-8"),
        };
        match build_from_pgm(text, options_or_default(options)) {
            Ok(model) => {
                // SAFETY: non-null and writable by contract.
                unsafe { *out = Box::into_raw(Box::new(model)) };
                Status::Ok
            }
            Err((status, message)) => {
                set_error(message);
                status
            }
        }
    })
}

/// Builds a model from arrays the caller already holds, in per-unit.
///
/// Any array may be null if its count is zero. Bus indices in `lines`,
/// `transformers` and `shunts` refer to positions in `buses`.
///
/// Note that ZIP (voltage-dependent) load terms cannot be expressed here — see
/// [`GridoxideBus`]. Use a PGM document if the network has them.
///
/// # Safety
///
/// Each pointer must point to at least its stated count of elements, and `out`
/// must be writable.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gridoxide_powerflow_from_arrays(
    buses: *const GridoxideBus,
    n_buses: usize,
    lines: *const GridoxideLine,
    n_lines: usize,
    transformers: *const GridoxideTransformer,
    n_transformers: usize,
    shunts: *const GridoxideShunt,
    n_shunts: usize,
    options: *const GridoxideOptions,
    out: *mut *mut GridoxidePowerFlow,
) -> Status {
    guard(|| {
        if out.is_null() {
            return invalid("out pointer is null");
        }
        if n_buses == 0 {
            return invalid("a network needs at least one bus");
        }
        if buses.is_null() {
            return invalid("bus array is null but n_buses is nonzero");
        }
        if (n_lines > 0 && lines.is_null())
            || (n_transformers > 0 && transformers.is_null())
            || (n_shunts > 0 && shunts.is_null())
        {
            return invalid("an array is null but its count is nonzero");
        }

        // SAFETY: each pointer covers its stated count by contract; the
        // zero-count cases were just checked, and `from_raw_parts` with a null
        // pointer and length 0 is not permitted, hence the explicit empties.
        let bus_slice = unsafe { std::slice::from_raw_parts(buses, n_buses) };
        let line_slice = if n_lines == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(lines, n_lines) }
        };
        let tx_slice = if n_transformers == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(transformers, n_transformers) }
        };
        let shunt_slice = if n_shunts == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(shunts, n_shunts) }
        };

        // Bounds-check every reference before assembly. `build_ybus` would
        // otherwise index out of range and panic — caught by `guard`, but
        // reported as an internal bug rather than as the caller's typo.
        for (k, line) in line_slice.iter().enumerate() {
            if line.from >= n_buses || line.to >= n_buses {
                return invalid(format!(
                    "line {k} joins buses {} and {}, but there are only {n_buses}",
                    line.from, line.to
                ));
            }
        }
        for (k, tx) in tx_slice.iter().enumerate() {
            if tx.from >= n_buses || tx.to >= n_buses {
                return invalid(format!(
                    "transformer {k} joins buses {} and {}, but there are only {n_buses}",
                    tx.from, tx.to
                ));
            }
        }
        for (k, shunt) in shunt_slice.iter().enumerate() {
            if shunt.at >= n_buses {
                return invalid(format!(
                    "shunt {k} sits at bus {}, but there are only {n_buses}",
                    shunt.at
                ));
            }
        }

        let model = GridoxidePowerFlow::assemble(
            bus_slice.iter().enumerate().map(|(i, b)| b.to_bus(i)).collect(),
            line_slice.iter().map(|l| (*l).into()).collect(),
            tx_slice.iter().map(|t| (*t).into()).collect(),
            shunt_slice.iter().map(|s| (*s).into()).collect(),
            options_or_default(options),
        );
        match model {
            Ok(model) => {
                // SAFETY: non-null and writable by contract.
                unsafe { *out = Box::into_raw(Box::new(model)) };
                Status::Ok
            }
            Err(message) => invalid(message),
        }
    })
}

/// Releases a handle. Null is accepted and ignored.
///
/// # Safety
///
/// `handle` must have come from a constructor here and must not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_free(handle: *mut GridoxidePowerFlow) {
    if handle.is_null() {
        return;
    }
    // SAFETY: came from `Box::into_raw` and is dropped exactly once by
    // contract. Wrapped because a `Drop` impl reached from here could panic.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(handle) });
    }));
}

/// Solves an ordinary AC power flow.
///
/// # Safety
///
/// `handle` must be a live handle from a constructor here.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_solve(handle: *mut GridoxidePowerFlow) -> Status {
    guard(|| {
        // SAFETY: checked non-null inside; single-thread-affine by contract.
        let model = match unsafe { handle_mut(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        model.buses.clone_from(&model.template);
        model.slack_shift.clear();
        let (islands, stats) = model.solver.solve_with_stats(
            &mut model.buses,
            &model.ybus,
            model.options.tolerance,
            model.options.max_iterations,
        );
        model.finish(islands, stats)
    })
}

/// Solves with the PV→PQ outer loop, so generators respect their reactive
/// limits.
///
/// # Safety
///
/// As [`gridoxide_powerflow_solve`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_solve_enforcing_q_limits(
    handle: *mut GridoxidePowerFlow,
    max_outer_iterations: usize,
) -> Status {
    guard(|| {
        // SAFETY: as above.
        let model = match unsafe { handle_mut(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        if max_outer_iterations == 0 {
            return invalid("max_outer_iterations must be at least 1");
        }
        model.buses.clone_from(&model.template);
        model.slack_shift.clear();
        let backend = match backend_of(model.options.backend) {
            Ok(b) => b,
            Err(message) => return invalid(message),
        };
        let mut qlim = crate::outerloop::ReactiveLimits::new();
        let islands = {
            let mut ctx =
                crate::outerloop::SolveContext::new(&mut model.buses, &mut model.ybus);
            let mut list: Vec<&mut dyn crate::outerloop::OuterLoop> = vec![&mut qlim];
            crate::outerloop::solve_with_loops(
                &mut ctx,
                model.options.tolerance,
                model.options.max_iterations,
                backend,
                &mut list,
                max_outer_iterations,
            )
            .0
        };
        // The outer loop switches bus types, so the handle's cached
        // factorization no longer matches the network it was built for.
        model.solver.reset();
        let stats = stats_from_islands(&islands);
        model.finish(islands, stats)
    })
}

/// Solves with the system imbalance shared across generators rather than left
/// on one slack bus.
///
/// `factors` is one participation weight per bus, normalized per island; zero
/// means the bus does not participate. Passing null uses an equal share for
/// every `Slack` and `PV` bus.
///
/// After this returns,
/// [`gridoxide_powerflow_slack_shift`](gridoxide_powerflow_slack_shift) reports
/// how far each bus's schedule moved.
///
/// # Safety
///
/// As [`gridoxide_powerflow_solve`]; `factors`, if non-null, must point to
/// `n_factors` readable doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_solve_distributing_slack(
    handle: *mut GridoxidePowerFlow,
    factors: *const f64,
    n_factors: usize,
) -> Status {
    guard(|| {
        // SAFETY: as above.
        let model = match unsafe { handle_mut(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        model.buses.clone_from(&model.template);

        // Checked here rather than left to `newton_raphson_distributing_slack`,
        // which asserts — and an assert crossing this boundary would abort the
        // caller's process. `guard` would catch it, but reporting it as the
        // caller's mistake is far more useful than reporting it as our bug.
        let distribution = if factors.is_null() {
            SlackDistribution::uniform(&model.buses)
        } else {
            if n_factors != model.buses.len() {
                return invalid(format!(
                    "factors holds {n_factors} weights, but this network has {} buses",
                    model.buses.len()
                ));
            }
            // SAFETY: non-null with `n_factors` readable doubles by contract.
            let weights = unsafe { std::slice::from_raw_parts(factors, n_factors) };
            if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return invalid("participation weights must be finite and non-negative");
            }
            SlackDistribution::from_weights(weights.to_vec())
        };

        let backend = match backend_of(model.options.backend) {
            Ok(b) => b,
            Err(message) => return invalid(message),
        };
        let cap = distribution.max_outer_iter;
        let mut slack = crate::outerloop::DistributedSlack::new(distribution);
        let islands = {
            let mut ctx =
                crate::outerloop::SolveContext::new(&mut model.buses, &mut model.ybus);
            let mut list: Vec<&mut dyn crate::outerloop::OuterLoop> = vec![&mut slack];
            crate::outerloop::solve_with_loops(
                &mut ctx,
                model.options.tolerance,
                model.options.max_iterations,
                backend,
                &mut list,
                cap,
            )
            .0
        };
        let report = slack.into_report();
        model.slack_shift = report.shift;
        // The loop leaves its own solver behind; ours never saw those solves.
        model.solver.reset();

        // Reconstruct stats from the islands: the outer-loop driver reports
        // per-island statuses rather than one inner solve's `SolveStats`.
        let stats = stats_from_islands(&islands);
        let status = model.finish(islands, stats);
        if status == Status::Ok && !report.converged {
            set_error(
                "the slack-distribution outer loop hit its iteration cap with the slack still \
                 off its schedule",
            );
            return Status::NotConverged;
        }
        status
    })
}

/// Solves with **both** controls active: reactive limits and distributed slack
/// in one solve.
///
/// This is the capability the two single-control entry points above cannot
/// express between them — before the outer-loop layer existed a caller picked
/// one, and a network that needs both (a real transmission grid usually does)
/// could not be solved correctly at all.
///
/// `enforce_q_limits` is a boolean. `slack_factors` may be null, which means
/// "no distributed slack" when `distribute_slack` is 0 and "an equal share for
/// every `Slack` and `PV` bus" when it is 1. Both controls off is an ordinary
/// solve.
///
/// # Safety
///
/// As [`gridoxide_powerflow_solve`]; `slack_factors`, if non-null, must point
/// to `n_factors` readable doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_solve_with_controls(
    handle: *mut GridoxidePowerFlow,
    enforce_q_limits: i32,
    distribute_slack: i32,
    slack_factors: *const f64,
    n_factors: usize,
    max_outer_iterations: usize,
) -> Status {
    guard(|| {
        // SAFETY: as above.
        let model = match unsafe { handle_mut(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        if max_outer_iterations == 0 {
            return invalid("max_outer_iterations must be at least 1");
        }
        model.buses.clone_from(&model.template);
        model.slack_shift.clear();

        let distribution = if distribute_slack == 0 {
            None
        } else if slack_factors.is_null() {
            Some(crate::outerloop::SlackDistribution::uniform(&model.buses))
        } else {
            if n_factors != model.buses.len() {
                return invalid(format!(
                    "factors holds {n_factors} weights, but this network has {} buses",
                    model.buses.len()
                ));
            }
            // SAFETY: non-null with `n_factors` readable doubles by contract.
            let weights = unsafe { std::slice::from_raw_parts(slack_factors, n_factors) };
            if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return invalid("participation weights must be finite and non-negative");
            }
            Some(crate::outerloop::SlackDistribution::from_weights(weights.to_vec()))
        };

        let backend = match backend_of(model.options.backend) {
            Ok(b) => b,
            Err(message) => return invalid(message),
        };

        let mut slack = distribution.map(crate::outerloop::DistributedSlack::new);
        let mut qlim = (enforce_q_limits != 0).then(crate::outerloop::ReactiveLimits::new);
        let islands = {
            let mut ctx =
                crate::outerloop::SolveContext::new(&mut model.buses, &mut model.ybus);
            let mut list: Vec<&mut dyn crate::outerloop::OuterLoop> = Vec::new();
            if let Some(l) = slack.as_mut() {
                list.push(l);
            }
            if let Some(l) = qlim.as_mut() {
                list.push(l);
            }
            crate::outerloop::solve_with_loops(
                &mut ctx,
                model.options.tolerance,
                model.options.max_iterations,
                backend,
                &mut list,
                max_outer_iterations,
            )
            .0
        };
        if let Some(l) = slack {
            model.slack_shift = l.into_report().shift;
        }
        // A `PV → PQ` switch changes the sparsity pattern, so the handle's
        // cached factorization is stale whenever the Q-limit loop ran.
        model.solver.reset();
        let stats = stats_from_islands(&islands);
        model.finish(islands, stats)
    })
}

/// The verdict of a whole outer-loop run, in the shape the handle stores.
///
/// The driver re-solves through `PersistentSolver::solve`, which does not hand
/// back `SolveStats`; the island statuses carry the same verdict by a different
/// route, and the mismatch history belongs to whichever inner solve ran last
/// rather than to the run.
fn stats_from_islands(islands: &[IslandReport]) -> SolveStats {
    SolveStats::from_loop(
        if islands.iter().any(|i| i.status == IslandStatus::Singular) {
            crate::solver::SolveStatus::Singular
        } else if islands.iter().any(|i| i.status == IslandStatus::MaxIterationsReached) {
            crate::solver::SolveStatus::MaxIterationsReached
        } else {
            crate::solver::SolveStatus::Converged
        },
        Vec::new(),
    )
}

/// Number of buses.
///
/// # Safety
///
/// `handle` must be live, or null (which returns 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_bus_count(handle: *const GridoxidePowerFlow) -> usize {
    // SAFETY: null-checked by `handle_ref`.
    unsafe { handle_ref(handle) }.map(|m| m.buses.len()).unwrap_or(0)
}

/// Number of branches — **lines first, then transformers**, which is the order
/// every per-branch array here uses.
///
/// # Safety
///
/// As [`gridoxide_powerflow_bus_count`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_branch_count(
    handle: *const GridoxidePowerFlow,
) -> usize {
    // SAFETY: null-checked by `handle_ref`.
    unsafe { handle_ref(handle) }.map(|m| m.branches.len()).unwrap_or(0)
}

// These two were briefly a `macro_rules!` — two near-identical accessors are
// exactly what a macro is for. It was wrong here, and the drift test caught it:
// **cbindgen does not expand macros**, so the functions existed in the library
// and were absent from the header. A consumer would have met a symbol that
// links but does not declare. Written out longhand, which costs a dozen lines
// and keeps the header honest.

/// Bus voltage magnitudes, per-unit.
///
/// # Safety
///
/// `out` must point to `len` writable doubles, `len` equal to
/// [`gridoxide_powerflow_bus_count`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_voltage_magnitude(
    handle: *const GridoxidePowerFlow,
    out: *mut f64,
    len: usize,
) -> Status {
    guard(|| {
        // SAFETY: null-checked inside.
        let model = match unsafe { handle_ref(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        let values: Vec<f64> = model.buses.iter().map(|b| b.voltage_mag).collect();
        // SAFETY: contract on `out`/`len`, re-checked by `write_slice`.
        unsafe { write_slice(&values, out, len, "voltage_magnitude") }
    })
}

/// Bus voltage angles, radians.
///
/// # Safety
///
/// As [`gridoxide_powerflow_voltage_magnitude`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_voltage_angle(
    handle: *const GridoxidePowerFlow,
    out: *mut f64,
    len: usize,
) -> Status {
    guard(|| {
        // SAFETY: null-checked inside.
        let model = match unsafe { handle_ref(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        let values: Vec<f64> = model.buses.iter().map(|b| b.voltage_ang).collect();
        // SAFETY: contract on `out`/`len`, re-checked by `write_slice`.
        unsafe { write_slice(&values, out, len, "voltage_angle") }
    })
}

/// Active and reactive power entering each branch at the chosen terminal, in
/// per-unit — `terminal` is 0 for *from*, 1 for *to*.
///
/// Either output may be null if you want only the other. Both arrays are
/// indexed by branch, lines before transformers.
///
/// This has no counterpart in the Python binding, which exposes branch flows
/// for DC only. It is here because a caller who has just solved an AC power
/// flow almost always wants them, and computing them from the voltages by hand
/// means reimplementing the π-model and its tap handling.
///
/// # Safety
///
/// Each non-null output must point to `len` writable doubles, with `len` equal
/// to [`gridoxide_powerflow_branch_count`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_branch_flow(
    handle: *const GridoxidePowerFlow,
    terminal: c_int,
    p_out: *mut f64,
    q_out: *mut f64,
    len: usize,
) -> Status {
    guard(|| {
        // SAFETY: null-checked inside.
        let model = match unsafe { handle_ref(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        if model.stats.is_none() {
            return invalid("solve before reading branch flows");
        }
        let terminal = match terminal {
            0 => Terminal::From,
            1 => Terminal::To,
            other => return invalid(format!("terminal must be 0 (from) or 1 (to), got {other}")),
        };
        if len != model.branches.len() {
            return invalid(format!(
                "branch_flow: buffer holds {len} entries, but this network has {} branches",
                model.branches.len()
            ));
        }

        let v = bus_voltages(&model.buses);
        let (mut p, mut q) = (Vec::with_capacity(len), Vec::with_capacity(len));
        for branch in &model.branches {
            let (bp, bq) = terminal_flow(branch, terminal, &v);
            p.push(bp);
            q.push(bq);
        }
        if !p_out.is_null() {
            // SAFETY: contract on `p_out`/`len`.
            let status = unsafe { write_slice(&p, p_out, len, "branch_flow (P)") };
            if status != Status::Ok {
                return status;
            }
        }
        if !q_out.is_null() {
            // SAFETY: contract on `q_out`/`len`.
            let status = unsafe { write_slice(&q, q_out, len, "branch_flow (Q)") };
            if status != Status::Ok {
                return status;
            }
        }
        Status::Ok
    })
}

/// How far each bus's active schedule moved in the last distributed-slack
/// solve, per-unit.
///
/// Returns [`Status::NoAnswer`] if the last solve did not distribute slack —
/// there is no shift to report, which is not an error.
///
/// # Safety
///
/// `out` must point to `len` writable doubles, `len` equal to
/// [`gridoxide_powerflow_bus_count`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_slack_shift(
    handle: *const GridoxidePowerFlow,
    out: *mut f64,
    len: usize,
) -> Status {
    guard(|| {
        // SAFETY: null-checked inside.
        let model = match unsafe { handle_ref(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        if model.slack_shift.is_empty() {
            set_error("the last solve did not distribute slack, so there is no shift to report");
            return Status::NoAnswer;
        }
        // SAFETY: contract on `out`/`len`.
        unsafe { write_slice(&model.slack_shift, out, len, "slack_shift") }
    })
}

/// Newton iterations taken by the last solve, or 0 if none has run.
///
/// # Safety
///
/// As [`gridoxide_powerflow_bus_count`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_iterations(
    handle: *const GridoxidePowerFlow,
) -> usize {
    // SAFETY: null-checked by `handle_ref`.
    unsafe { handle_ref(handle) }
        .ok()
        .and_then(|m| m.stats.as_ref())
        .map(|s| s.iterations())
        .unwrap_or(0)
}

/// Largest power mismatch at the last solve, or NaN if none has run.
///
/// # Safety
///
/// As [`gridoxide_powerflow_bus_count`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_max_mismatch(
    handle: *const GridoxidePowerFlow,
) -> f64 {
    // SAFETY: null-checked by `handle_ref`.
    unsafe { handle_ref(handle) }
        .ok()
        .and_then(|m| m.stats.as_ref())
        .map(|s| s.final_mismatch())
        .unwrap_or(f64::NAN)
}

/// Number of connected components the last solve found.
///
/// # Safety
///
/// As [`gridoxide_powerflow_bus_count`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_island_count(
    handle: *const GridoxidePowerFlow,
) -> usize {
    // SAFETY: null-checked by `handle_ref`.
    unsafe { handle_ref(handle) }.map(|m| m.islands.len()).unwrap_or(0)
}

/// One island's outcome.
///
/// Worth reading even when the solve returned `GRIDOXIDE_OK`: an island with no
/// source reports `NO_REFERENCE_BUS` and is *not* a failure of the call, so the
/// only way to notice a de-energized part of the network is here.
///
/// # Safety
///
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_island_status(
    handle: *const GridoxidePowerFlow,
    island: usize,
    out: *mut GridoxideIslandStatus,
) -> Status {
    guard(|| {
        // SAFETY: null-checked inside.
        let model = match unsafe { handle_ref(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        if out.is_null() {
            return invalid("out pointer is null");
        }
        let Some(report) = model.islands.get(island) else {
            return invalid(format!(
                "island {island} is out of range; the last solve found {}",
                model.islands.len()
            ));
        };
        // SAFETY: non-null and writable by contract.
        unsafe { *out = island_status_code(report.status) };
        Status::Ok
    })
}

/// How many buses are in one island.
///
/// # Safety
///
/// As [`gridoxide_powerflow_bus_count`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_island_bus_count(
    handle: *const GridoxidePowerFlow,
    island: usize,
) -> usize {
    // SAFETY: null-checked by `handle_ref`.
    unsafe { handle_ref(handle) }
        .ok()
        .and_then(|m| m.islands.get(island))
        .map(|i| i.bus_indices.len())
        .unwrap_or(0)
}

/// The bus indices in one island.
///
/// # Safety
///
/// `out` must point to `len` writable `size_t`s, `len` equal to
/// [`gridoxide_powerflow_island_bus_count`] for the same island.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gridoxide_powerflow_island_buses(
    handle: *const GridoxidePowerFlow,
    island: usize,
    out: *mut usize,
    len: usize,
) -> Status {
    guard(|| {
        // SAFETY: null-checked inside.
        let model = match unsafe { handle_ref(handle) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        if out.is_null() {
            return invalid("out pointer is null");
        }
        let Some(report) = model.islands.get(island) else {
            return invalid(format!(
                "island {island} is out of range; the last solve found {}",
                model.islands.len()
            ));
        };
        if len != report.bus_indices.len() {
            return invalid(format!(
                "island_buses: buffer holds {len} entries, but island {island} has {}",
                report.bus_indices.len()
            ));
        }
        // SAFETY: `len` writable elements by contract, length just checked.
        unsafe { std::ptr::copy_nonoverlapping(report.bus_indices.as_ptr(), out, len) };
        Status::Ok
    })
}
