use super::jacobian::JacobianPattern;
use super::network::{
    Classified, Verdict, YBusSparse, classify, connected_components, effective_injection,
    mark_unreferenced_islands, power_injections,
};
use gridoxide_sparse::block_sparse::{BlockLu, BlockMatrix, BlockSymbolic};
// Defined beside the backends it abstracts over, so that they need nothing
// from this module; re-exported here, where the Newton loop is generic over it.
use super::types::{Bus, BusType};
use gridoxide_sparse::lu::{LuBackend, SparseLu};
pub use gridoxide_sparse::sparse::LinearSolver;

/// Selects which sparse-LU backend `newton_raphson_with_backend` uses to
/// solve the Jacobian system each iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JacobianBackend {
    /// The default, production path: scalar sparse triplets solved via
    /// `faer` (`sparse::RealSparseSystem`). Used by `newton_raphson`.
    Scalar,
    /// Experimental, opt-in: groups each bus's own (angle, magnitude)
    /// unknowns into one 2×2 block, mirroring power-grid-model's
    /// block-per-bus matrix structure (`block_sparse::BlockLu`). Every
    /// non-slack bus gets a block — including `PV` buses, which have no
    /// real Q-mismatch equation. Rather than varying block size per bus
    /// (which `block_sparse::BlockLu` doesn't support), a `PV` bus's second
    /// row is replaced with a dummy `ΔVmag_i = 0` equation (coefficients
    /// `[0, 1]`, target `0`) that pins its voltage-magnitude update to zero
    /// every iteration — mathematically equivalent to the scalar backend's
    /// actual dimension reduction, since that row is fully decoupled from
    /// every other unknown. See `build_jacobian_blocks`.
    Block,
    /// Experimental, opt-in, only compiled with the `klu` Cargo feature:
    /// the same scalar Jacobian as `Scalar` (one `jacobian::JacobianPattern`
    /// serves both), solved via the **vendored SuiteSparse KLU C library**,
    /// linked over FFI (`sparse_klu::KluRealSystem`) instead of `faer`. Needs
    /// a C compiler and `libclang` at build time. See
    /// `docs/src/solvers/backends.md` and
    /// `crates/gridoxide-sparse/vendor/suitesparse/PROVENANCE.md`. For the
    /// pure-Rust reimplementation of the same algorithm, see `KluNative`.
    #[cfg(feature = "klu")]
    Klu,
    /// Experimental: the same scalar Jacobian as `Scalar`, solved via a
    /// from-scratch Rust port of the same KLU algorithm `Klu` links over FFI
    /// — no C toolchain needed, **always compiled** (unlike `Klu`, no
    /// feature gate). See `klu_native`'s module doc comment for scope and
    /// what's ported vs. simplified (row scaling is ported but not yet
    /// wired in; everything else — BTF, AMD, partial pivoting, Eisenstat-Liu
    /// pruning, refactor — is a faithful translation).
    KluNative,
    /// Experimental, opt-in, only compiled with the `pardiso` Cargo feature:
    /// the same scalar Jacobian as `Scalar`, solved via **Intel oneMKL's
    /// PARDISO** sparse direct solver, linked dynamically against a
    /// locally-installed oneMKL (`sparse_pardiso::PardisoRealSystem`) —
    /// needs `MKLROOT` set at build time (see
    /// `docs/src/solvers/backends.md`). Unlike `Klu`, nothing is vendored:
    /// MKL is
    /// proprietary, so this only links a system install rather than
    /// compiling any bundled source. Not built or tested in CI (no MKL on
    /// CI runners) — local/manual-verification-only.
    #[cfg(feature = "pardiso")]
    Pardiso,
}

impl JacobianBackend {
    /// The sparse LU this backend factors with. `Block` has no general sparse
    /// LU of its own, so wherever a Jacobian is not power flow's block-per-bus
    /// shape it falls back to `faer`, as `Scalar` does.
    pub fn lu_backend(self) -> LuBackend {
        match self {
            JacobianBackend::Scalar | JacobianBackend::Block => LuBackend::Faer,
            JacobianBackend::KluNative => LuBackend::KluNative,
            #[cfg(feature = "klu")]
            JacobianBackend::Klu => LuBackend::Klu,
            #[cfg(feature = "pardiso")]
            JacobianBackend::Pardiso => LuBackend::Pardiso,
        }
    }
}

/// How a Newton-Raphson solve terminated, plus the per-iteration convergence
/// trace it used to print to stdout.
///
/// The solve loop is silent: at batch scale (`crate::batch::BatchSolver`)
/// per-iteration `println!` would serialize every worker on the stdout lock
/// and dominate the measurement. Callers that want the old progress output
/// reconstruct it from [`mismatch_history`](Self::mismatch_history) — see
/// `src/main.rs`, which reproduces it verbatim.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveStats {
    pub status: SolveStatus,
    /// Max |mismatch| measured at the top of each iteration, in order. The
    /// last element is the value the convergence check accepted (for
    /// `Converged`) or rejected. Length is the iteration count — see
    /// [`iterations`](Self::iterations).
    pub mismatch_history: Vec<f64>,
}

impl SolveStats {
    pub fn from_loop(status: SolveStatus, mismatch_history: Vec<f64>) -> Self {
        Self {
            status,
            mismatch_history,
        }
    }

    /// Iterations actually run. Derived from `mismatch_history` rather than
    /// stored separately so the two can't drift apart.
    pub fn iterations(&self) -> usize {
        self.mismatch_history.len()
    }

    /// Max |mismatch| at the final iteration, or `0.0` if none ran.
    pub fn final_mismatch(&self) -> f64 {
        self.mismatch_history.last().copied().unwrap_or(0.0)
    }
}

/// Outcome of a single Newton-Raphson solve — see [`SolveStats`], which
/// carries this alongside the convergence trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveStatus {
    /// Converged within `max_iter` iterations; `buses` holds the solution.
    Converged,
    /// Ran `max_iter` iterations without reaching `tol`; `buses` holds
    /// whatever the last iteration produced, not a converged solution.
    MaxIterationsReached,
    /// The Jacobian was singular at some iteration; `buses` holds whatever
    /// partial state existed at that point, not a valid solution.
    Singular,
}

/// Per-connected-component outcome of a multi-island power-flow solve (see
/// `network::connected_components`/`classify`) — one shared Newton-Raphson
/// call across every solvable component's unknowns at once (the Y-bus has
/// no coupling between disconnected components, so this is mathematically
/// equivalent, iteration for iteration, to solving each independently), with
/// per-component status recovered afterward rather than tracked live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IslandStatus {
    /// This component's own mismatch is below `tol` in the final `buses`
    /// state, regardless of the overall multi-component solve's status.
    Converged,
    /// The overall solve ran out of iterations, and this component's own
    /// mismatch is still above `tol`.
    MaxIterationsReached,
    /// The overall solve hit a singular Jacobian, and this component's own
    /// mismatch is still above `tol`. Best-effort, not precise: every
    /// backend solves one combined sparse factorization and detects
    /// singularity via a finiteness check on the *result* vector (see
    /// `sparse.rs`), not a specific failing pivot/row — there is no way to
    /// attribute a singular failure to one particular component in general,
    /// so every still-unconverged component is marked `Singular` when the
    /// overall result is `Singular`, whether or not it was the actual cause.
    Singular,
    /// This component had no `Slack` bus at all — every member bus was
    /// pinned to a fixed `V = 0`, `P = Q = 0` placeholder
    /// (`network::mark_unreferenced_islands`) instead of being solved; its
    /// bus values are not a power-flow result, just the marker value.
    /// There is no principled way to fabricate a reference bus for a
    /// genuinely sourceless island, so this is never attempted.
    NoReferenceBus,
    /// This component had more than one `Slack` bus. Left in the shared
    /// solve unmodified (its non-slack buses are still live unknowns): two
    /// independently-fixed slack voltages in one connected component is
    /// physically over-determined, not necessarily numerically singular, so
    /// Newton-Raphson can converge to a result that satisfies neither
    /// slack's true power balance. This verdict is decided once, before the
    /// solve, and is never overwritten by a numerically-convergent-looking
    /// post-hoc mismatch check.
    AmbiguousReferenceBus,
}

/// One connected component's power-flow outcome — see `IslandStatus`.
#[derive(Clone, Debug)]
pub struct IslandReport {
    pub bus_indices: Vec<usize>,
    /// The `Slack` bus(es) originally found in this component: empty for
    /// `NoReferenceBus`, exactly one for a normally-solved island,
    /// two-or-more for `AmbiguousReferenceBus`.
    pub slack_indices: Vec<usize>,
    pub status: IslandStatus,
}

/// Computes each still-undecided (`Solvable`-verdict) component's own P/Q
/// mismatch against `tol`, using the same `power_injections`/
/// `effective_injection` primitives — restricted to that component's own
/// bus indices — that `newton_raphson_scalar_cached`'s own convergence
/// check uses. Safe to call even after an overall `Singular` result: every
/// backend returns `Singular` *before* applying that iteration's update (see
/// e.g. `newton_raphson_scalar_cached`/`newton_raphson_block_cached` below),
/// so `buses` always holds the last fully-applied, fully-finite iterate.
fn resolve_pending(
    pending: Vec<(Vec<usize>, Vec<usize>)>,
    buses: &[Bus],
    ybus: &YBusSparse,
    overall: SolveStatus,
    tol: f64,
) -> Vec<IslandReport> {
    let (p_calc, q_calc) = power_injections(buses, ybus);
    pending
        .into_iter()
        .map(|(bus_indices, slack_indices)| {
            let mut max_mis = 0.0f64;
            for &i in &bus_indices {
                if buses[i].bus_type == BusType::Slack {
                    continue;
                }
                let (p_eff, q_eff) = effective_injection(&buses[i]);
                max_mis = max_mis.max((p_eff - p_calc[i]).abs());
                if buses[i].bus_type == BusType::PQ {
                    max_mis = max_mis.max((q_eff - q_calc[i]).abs());
                }
            }
            let status = if max_mis < tol {
                IslandStatus::Converged
            } else if overall == SolveStatus::Singular {
                IslandStatus::Singular
            } else {
                IslandStatus::MaxIterationsReached
            };
            IslandReport {
                bus_indices,
                slack_indices,
                status,
            }
        })
        .collect()
}

/// Turns each component's classify-time `Verdict` into a final
/// `IslandReport`, in the same order `classify` produced them: `NoReferenceBus`/
/// `AmbiguousReferenceBus` verdicts become their matching `IslandStatus`
/// immediately (never touched by the post-hoc mismatch check — see
/// `IslandStatus::AmbiguousReferenceBus`'s doc comment for why), while
/// `Solvable` components are resolved via `resolve_pending` against the
/// `buses`/`ybus` state left by the shared solve.
pub(crate) fn finish_island_reports(
    classified: Vec<Classified>,
    buses: &[Bus],
    ybus: &YBusSparse,
    overall: SolveStatus,
    tol: f64,
) -> Vec<IslandReport> {
    let mut reports: Vec<Option<IslandReport>> = Vec::with_capacity(classified.len());
    let mut pending: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
    for (pos, c) in classified.into_iter().enumerate() {
        match c.verdict {
            Verdict::NoReferenceBus => reports.push(Some(IslandReport {
                bus_indices: c.bus_indices,
                slack_indices: c.slack_indices,
                status: IslandStatus::NoReferenceBus,
            })),
            Verdict::AmbiguousReferenceBus => reports.push(Some(IslandReport {
                bus_indices: c.bus_indices,
                slack_indices: c.slack_indices,
                status: IslandStatus::AmbiguousReferenceBus,
            })),
            Verdict::Solvable => {
                pending.push((pos, c.bus_indices, c.slack_indices));
                reports.push(None);
            }
        }
    }

    let (positions, pending): (Vec<usize>, Vec<(Vec<usize>, Vec<usize>)>) = pending
        .into_iter()
        .map(|(pos, bus_indices, slack_indices)| (pos, (bus_indices, slack_indices)))
        .unzip();
    let resolved = resolve_pending(pending, buses, ybus, overall, tol);
    for (pos, report) in positions.into_iter().zip(resolved) {
        reports[pos] = Some(report);
    }

    reports
        .into_iter()
        .map(|r| r.expect("every component classified exactly once"))
        .collect()
}

pub fn newton_raphson(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    tol: f64,
    max_iter: usize,
) -> Vec<IslandReport> {
    newton_raphson_with_backend(buses, ybus, tol, max_iter, JacobianBackend::Scalar)
}

/// Partitions `buses` into connected components first (`network::
/// connected_components`/`classify`/`mark_unreferenced_islands` — see
/// `docs/src/powerflow/multi_island.md`), so a disconnected, sourceless
/// region can never make the *whole* Jacobian singular, then solves every
/// component in one shared Newton-Raphson call and returns each one's own
/// outcome. This
/// is gridoxide's one canonical way to run a solve — [`PersistentSolver::solve`]
/// and [`crate::outerloop::solve_with_loops`] share this exact same
/// partitioning step, so behavior is identical regardless of which entry
/// point a caller uses.
pub fn newton_raphson_with_backend(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    tol: f64,
    max_iter: usize,
    backend: JacobianBackend,
) -> Vec<IslandReport> {
    let components = connected_components(ybus);
    let classified = classify(buses, &components);
    mark_unreferenced_islands(buses, &classified);

    let stats = match backend {
        JacobianBackend::Block => {
            newton_raphson_block_cached(buses, ybus, tol, max_iter, &mut None)
        }
        _ => newton_raphson_cached(
            buses,
            ybus,
            tol,
            max_iter,
            &mut SparseLu::new(backend.lu_backend()),
            &mut None,
        ),
    };

    finish_island_reports(classified, buses, ybus, stats.status, tol)
}

/// A Newton-Raphson solver that reuses its symbolic factorization (fill-
/// reducing ordering, elimination-graph structure — the expensive,
/// topology-only part of a sparse LU factorization) across repeated
/// [`solve`](Self::solve) calls, not just across the iterations *within*
/// one call the way [`newton_raphson_with_backend`] already does.
///
/// This matters for any workload that solves the *same* topology
/// repeatedly — a time series, a batch of scenarios, contingency analysis —
/// where re-deriving the ordering from scratch on every call is pure
/// overhead. Measured on a 9,241-bus case, reusing analysis across repeated
/// solves cut per-solve time by ~45% (COLAMD/AMD/BTF ordering was ~1/3 of
/// total solve time when redone from scratch every call).
///
/// ```
/// # use gridoxide_core::solver::{PersistentSolver, JacobianBackend};
/// # use gridoxide_core::network::YBusSparse;
/// # use gridoxide_core::types::Bus;
/// # fn example(mut buses: Vec<Bus>, ybus: &YBusSparse) {
/// let mut solver = PersistentSolver::new(JacobianBackend::Scalar);
/// for _ in 0..10 {
///     // ... update buses' p_spec/q_spec for the next scenario ...
///     solver.solve(&mut buses, ybus, 1e-6, 20);
/// }
/// # }
/// ```
///
/// Call [`reset`](Self::reset) (or construct a new `PersistentSolver`) if
/// the topology itself changes between solves — a different set of lines,
/// transformers, or bus types — since the cached ordering would otherwise
/// silently apply to the wrong sparsity pattern. Changing only bus
/// *values* (`p_spec`, `q_spec`, `voltage_mag`/`voltage_ang` initial guess)
/// between calls is exactly the case this is designed for and needs no
/// reset.
pub struct PersistentSolver {
    backend: JacobianBackend,
    /// Backend-independent, so one cache serves whichever backend is
    /// selected.
    ///
    /// Its *sparsity pattern* depends only on topology and bus types — but the
    /// per-nonzero recipe does not: [`JacobianPattern`]'s entries carry the
    /// Y-bus admittance `y` they were analyzed against, and `fill` reads it
    /// every iteration. So this is valid only while the Y-bus **values** are
    /// unchanged too, which is automatic for an ordinary batch (every scenario
    /// shares one Y-bus) and is not automatic for a contingency sweep (each
    /// outage has its own). See [`invalidate_admittances`](Self::invalidate_admittances).
    jacobian: Option<JacobianPattern>,
    /// The Y-bus's connected components, which depend on its structure alone
    /// — the same thing the symbolic factorization assumes fixed until
    /// [`reset`](Self::reset), and kept by
    /// [`invalidate_admittances`](Self::invalidate_admittances) for the same
    /// reason. Finding them cost 4% of a warm solve on case9241pegase.
    components: Option<Vec<Vec<usize>>>,
    /// The general sparse LU, for every backend but `Block`.
    lu: SparseLu,
    block: Option<BlockSymbolic>,
}

impl PersistentSolver {
    pub fn new(backend: JacobianBackend) -> Self {
        Self {
            backend,
            jacobian: None,
            components: None,
            lu: SparseLu::new(backend.lu_backend()),
            block: None,
        }
    }

    /// Runs Newton-Raphson to convergence (or `max_iter`), reusing cached
    /// symbolic factorization from a previous `solve()` call on this same
    /// `PersistentSolver` when available. See the type-level doc comment
    /// for when a cached factorization stays valid.
    ///
    /// Like [`newton_raphson_with_backend`], partitions `buses` into
    /// connected components first (`network::connected_components`/
    /// `classify`/`mark_unreferenced_islands`) so a disconnected, sourceless
    /// region can't make the whole Jacobian singular, and returns each
    /// component's own outcome rather than one flat status.
    pub fn solve(
        &mut self,
        buses: &mut [Bus],
        ybus: &YBusSparse,
        tol: f64,
        max_iter: usize,
    ) -> Vec<IslandReport> {
        self.solve_with_stats(buses, ybus, tol, max_iter).0
    }

    /// Exactly [`solve`](Self::solve), additionally returning the
    /// [`SolveStats`] the solve produced — iteration count and the
    /// per-iteration convergence trace the loop used to print to stdout.
    /// `solve` is a wrapper that discards them.
    pub fn solve_with_stats(
        &mut self,
        buses: &mut [Bus],
        ybus: &YBusSparse,
        tol: f64,
        max_iter: usize,
    ) -> (Vec<IslandReport>, SolveStats) {
        // Recomputed if the bus count changed, the one structural change
        // that is cheap to notice.
        if self
            .components
            .as_ref()
            .is_none_or(|c| c.iter().map(Vec::len).sum::<usize>() != buses.len())
        {
            self.components = Some(connected_components(ybus));
        }
        let components = self.components.as_ref().expect("just populated above");
        let classified = classify(buses, components);
        mark_unreferenced_islands(buses, &classified);

        let stats = match self.backend {
            JacobianBackend::Block => {
                newton_raphson_block_cached(buses, ybus, tol, max_iter, &mut self.block)
            }
            _ => {
                newton_raphson_cached(buses, ybus, tol, max_iter, &mut self.lu, &mut self.jacobian)
            }
        };

        let reports = finish_island_reports(classified, buses, ybus, stats.status, tol);
        (reports, stats)
    }

    /// Discards any cached symbolic factorization. Call this before the
    /// next `solve()` if the topology (not just bus values) has changed
    /// since the last call.
    /// Invalidates the cached Jacobian recipe while **keeping** every
    /// backend's symbolic factorization.
    ///
    /// The right call when the Y-bus's *values* changed but its sparsity
    /// pattern did not — which is exactly a branch contingency built by
    /// [`network::build_ybus_with_outages`](crate::network::build_ybus_with_outages),
    /// where the outaged branch keeps its structural entries at zero. The
    /// symbolic factorization depends only on the `(row, col)` pairs
    /// `JacobianPattern::to_triplets` emits, and those are unchanged, so it
    /// stays valid; the entries' cached admittances do not, and must be
    /// re-analyzed.
    ///
    /// Re-analysis is O(nonzeros) against the symbolic factorization's
    /// fill-reducing ordering and elimination-tree work, which is the
    /// expensive half — measured at ~45% of solve time on a 9,241-bus case.
    /// Keeping that is what makes an N-1 sweep cheaper than a sequence of
    /// independent solves.
    ///
    /// Use [`reset`](Self::reset) instead when the pattern itself changed:
    /// a different bus-type assignment, or a topology whose branches were
    /// genuinely dropped rather than zeroed.
    pub fn invalidate_admittances(&mut self) {
        self.jacobian = None;
    }

    pub fn reset(&mut self) {
        self.jacobian = None;
        self.components = None;
        self.lu.reset();
        self.block = None;
    }
}

/// The one Newton-Raphson loop, for every general sparse-LU backend: the
/// [`SparseLu`] carries the choice. There were once four copies of this
/// function, one per backend, and then one generic over [`LinearSolver`]
/// with a `match` at every call site. `newton_raphson_block_cached` stays separate: it has a genuinely
/// different matrix type and a `[f64; 2]`-shaped mismatch vector.
///
/// The cached symbolic factorization lives in a caller-supplied `lu`
/// rather than a function-local variable — what lets `PersistentSolver`
/// reuse it across repeated `solve()` calls on unchanged topology, not just
/// across the iterations within a single call. `newton_raphson_with_backend`
/// calls this directly too, with a function-local, throwaway cache, since it
/// needs the returned status to build its own `IslandReport`s.
///
/// Returns [`SolveStats`] rather than printing progress. Per-iteration
/// `println!` from inside this loop would serialize every worker of a
/// [`crate::batch::BatchSolver`] run on the stdout lock; `src/main.rs`
/// reproduces the old console output from `SolveStats::mismatch_history`
/// instead.
fn newton_raphson_cached(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    tol: f64,
    max_iter: usize,
    lu: &mut SparseLu,
    pattern: &mut Option<JacobianPattern>,
) -> SolveStats {
    // Every non-slack bus carries an angle unknown; PQ buses additionally
    // carry a voltage-magnitude unknown.
    let non_slack_idx: Vec<usize> = buses
        .iter()
        .filter(|b| !matches!(b.bus_type, BusType::Slack))
        .map(|b| b.idx)
        .collect();
    let pq_idx: Vec<usize> = buses
        .iter()
        .filter(|b| matches!(b.bus_type, BusType::PQ))
        .map(|b| b.idx)
        .collect();

    let n_unknowns = non_slack_idx.len() + pq_idx.len();

    // Both the Jacobian's sparsity pattern and the per-nonzero recipe depend
    // only on the topology and the bus-type assignment, so they are derived
    // once here (or reused from a previous solve) rather than per iteration.
    if pattern.is_none() {
        let mut fresh = JacobianPattern::analyze(buses, ybus);
        // A factorization kept across `invalidate_admittances` still takes
        // its values in the order it adopted below.
        if let Some(order) = lu.storage_order() {
            fresh.permute(order);
        }
        *pattern = Some(fresh);
    }
    let pattern = pattern.as_mut().expect("just populated above");

    // Refilled in place every iteration — the allocation happens once.
    let mut values: Vec<f64> = Vec::with_capacity(pattern.len());
    let mut history: Vec<f64> = Vec::with_capacity(max_iter);

    // The Jacobian's sparsity *pattern* is fixed across iterations (same
    // bus topology every time, only numeric values change), so the
    // symbolic factorization (ordering + fill-in) is computed once here and
    // reused via `factor_and_solve` for every iteration's numeric-only
    // refactorization, mirroring PGM's own prefactorization-reuse approach.
    for _ in 0..max_iter {
        // compute injections
        let (p_calc, q_calc) = power_injections(buses, ybus);

        // Build mismatch vector
        let mut mismatch = vec![0.0; n_unknowns];
        let mut mis_idx = 0;
        for &i in &non_slack_idx {
            // P mismatch for PV and PQ buses
            let (p_eff, _) = effective_injection(&buses[i]);
            mismatch[mis_idx] = p_eff - p_calc[i];
            mis_idx += 1;
        }
        for &i in &pq_idx {
            // Q mismatch for PQ buses
            let (_, q_eff) = effective_injection(&buses[i]);
            mismatch[mis_idx] = q_eff - q_calc[i];
            mis_idx += 1;
        }

        let max_mis = mismatch.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        history.push(max_mis);
        if max_mis < tol {
            return SolveStats::from_loop(SolveStatus::Converged, history);
        }

        // Refill the Jacobian's values at their precomputed offsets.
        pattern.fill(buses, &p_calc, &q_calc, &mut values);

        // The backend's symbolic analysis needs the `(row, col)` pairs, but
        // only once per topology — every subsequent iteration hands over
        // nothing but the values.
        if !lu.is_analyzed() {
            if !lu.analyze(n_unknowns, &pattern.to_triplets(&values)) {
                return SolveStats::from_loop(SolveStatus::Singular, history);
            }
            // Where the backend has one layout (both KLUs), fill straight into
            // it from here on rather than have it copy every iteration.
            if let Some(order) = lu.adopt_storage_order() {
                pattern.permute(&order);
                values = order.iter().map(|&i| values[i]).collect();
            }
        }
        let Some(dx) = lu.factor_and_solve_values(&values, &mismatch) else {
            return SolveStats::from_loop(SolveStatus::Singular, history);
        };

        // Update state
        let mut dx_idx = 0;
        for &i in &non_slack_idx {
            // update voltage angles
            buses[i].voltage_ang += dx[dx_idx];
            dx_idx += 1;
        }
        for &i in &pq_idx {
            // update voltage magnitudes
            buses[i].voltage_mag += dx[dx_idx];
            dx_idx += 1;
        }
    }

    SolveStats::from_loop(SolveStatus::MaxIterationsReached, history)
}

/// Experimental block-per-bus Newton-Raphson, backed by
/// `block_sparse::BlockLu`. See `JacobianBackend::Block`'s doc comment for
/// scope. Like `newton_raphson_scalar_cached`'s `RealSparseSystem` reuse,
/// the `colamd` ordering and elimination-graph reachability
/// (`BlockSymbolic`) are computed once and reused via `BlockLu::refactor`
/// for cheap numeric-only refactorization on every subsequent iteration.
/// See `newton_raphson_scalar_cached` — same caller-supplied-cache pattern,
/// for the `Block` backend.
fn newton_raphson_block_cached(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    tol: f64,
    max_iter: usize,
    symbolic: &mut Option<BlockSymbolic>,
) -> SolveStats {
    let non_slack_idx: Vec<usize> = buses
        .iter()
        .filter(|b| !matches!(b.bus_type, BusType::Slack))
        .map(|b| b.idx)
        .collect();

    let mut block_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in non_slack_idx.iter().enumerate() {
        block_pos[i] = Some(pos);
    }

    let mut history: Vec<f64> = Vec::with_capacity(max_iter);

    for _ in 0..max_iter {
        let (p_calc, q_calc) = power_injections(buses, ybus);

        // PV buses get a dummy `ΔVmag = 0` target in the mismatch vector's
        // second slot instead of a real Q mismatch — see
        // `JacobianBackend::Block`'s doc comment. It's always exactly 0, so
        // it never affects the `max_mis` convergence check below (matching
        // the scalar backend, which has no Q-mismatch entry for PV buses at
        // all).
        let mismatch: Vec<[f64; 2]> = non_slack_idx
            .iter()
            .map(|&i| {
                let (p_eff, q_eff) = effective_injection(&buses[i]);
                let q_mismatch = match buses[i].bus_type {
                    BusType::PV => 0.0,
                    _ => q_eff - q_calc[i],
                };
                [p_eff - p_calc[i], q_mismatch]
            })
            .collect();

        let max_mis = mismatch
            .iter()
            .flatten()
            .fold(0.0f64, |a, &b| a.max(b.abs()));
        history.push(max_mis);
        if max_mis < tol {
            return SolveStats::from_loop(SolveStatus::Converged, history);
        }

        let blocks =
            build_jacobian_blocks(buses, ybus, &non_slack_idx, &block_pos, &p_calc, &q_calc)
                .finish();
        if symbolic.is_none() {
            *symbolic = Some(BlockSymbolic::analyze(&blocks));
        }
        let Some(lu) = BlockLu::refactor(symbolic.as_ref().unwrap(), &blocks) else {
            return SolveStats::from_loop(SolveStatus::Singular, history);
        };
        let dx = lu.solve(&mismatch);

        for (pos, &i) in non_slack_idx.iter().enumerate() {
            buses[i].voltage_ang += dx[pos][0];
            buses[i].voltage_mag += dx[pos][1];
        }
    }

    SolveStats::from_loop(SolveStatus::MaxIterationsReached, history)
}

/// Assembles the same H/N/M/L formulas as the scalar Jacobian
/// (`jacobian::JacobianPattern`), but as one 2×2 block per (bus, bus) pair —
/// `[[H, N], [M, L]]` on the diagonal, `[[H_ik, N_ik], [M_ik, L_ik]]`
/// off-diagonal — instead of four separate scalar triplets.
///
/// A `PV` row bus has no real Q-mismatch equation, so its block's second row
/// is replaced with the dummy `ΔVmag_i = 0` equation instead: `[0, 1]` on
/// the diagonal block (pins this bus's own magnitude update to zero) and
/// `[0, 0]` on every off-diagonal block in that row (so the dummy equation
/// stays fully decoupled from every other bus's unknowns — required for it
/// to pin exactly `0`, not something entangled with the rest of the solve).
/// The H/N formulas themselves (first row) are unaffected — they're already
/// correct for any non-slack bus, PV or PQ, matching
/// the scalar Jacobian's H/N block.
fn build_jacobian_blocks(
    buses: &[Bus],
    ybus: &YBusSparse,
    non_slack_idx: &[usize],
    block_pos: &[Option<usize>],
    p_calc: &[f64],
    q_calc: &[f64],
) -> BlockMatrix {
    let vm: Vec<f64> = buses.iter().map(|b| b.voltage_mag).collect();
    let va: Vec<f64> = buses.iter().map(|b| b.voltage_ang).collect();
    let mut blocks = BlockMatrix::new(non_slack_idx.len());

    for (row, &i) in non_slack_idx.iter().enumerate() {
        let is_pv = matches!(buses[i].bus_type, BusType::PV);
        for &(k, y_ik) in ybus.row(i) {
            if k == i {
                let h_ii = -q_calc[i] - vm[i].powi(2) * y_ik.im;
                let n_ii = p_calc[i] / vm[i] + vm[i] * y_ik.re;
                let (m_ii, l_ii) = if is_pv {
                    (0.0, 1.0)
                } else {
                    (
                        p_calc[i] - vm[i].powi(2) * y_ik.re,
                        q_calc[i] / vm[i] - vm[i] * y_ik.im,
                    )
                };
                blocks.add(row, row, [[h_ii, n_ii], [m_ii, l_ii]]);
                continue;
            }
            let Some(col) = block_pos[k] else { continue };
            let angle_ik = va[i] - va[k];
            let h_ik = vm[i] * vm[k] * (y_ik.re * angle_ik.sin() - y_ik.im * angle_ik.cos());
            let n_ik = vm[i] * (y_ik.re * angle_ik.cos() + y_ik.im * angle_ik.sin());
            let (m_ik, l_ik) = if is_pv {
                (0.0, 0.0)
            } else {
                (
                    -vm[i] * vm[k] * (y_ik.re * angle_ik.cos() + y_ik.im * angle_ik.sin()),
                    vm[i] * (y_ik.re * angle_ik.sin() - y_ik.im * angle_ik.cos()),
                )
            };
            blocks.add(row, col, [[h_ik, n_ik], [m_ik, l_ik]]);
        }
    }

    blocks
}
