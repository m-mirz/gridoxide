use super::types::{Bus, BusType};
use super::network::{effective_injection, power_injections, YBusSparse};
use super::sparse::RealSparseSystem;
use super::block_sparse::{BlockLu, BlockMatrix, BlockSymbolic};
#[cfg(feature = "klu")]
use super::sparse_klu::KluRealSystem;
use super::klu_native::KluNativeSystem;

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
    /// the same scalar Jacobian as `Scalar` (reuses `build_jacobian_triplets`
    /// unchanged), solved via the **vendored SuiteSparse KLU C library**,
    /// linked over FFI (`sparse_klu::KluRealSystem`) instead of `faer`. Needs
    /// a C compiler and `libclang` at build time. See the README's "Sparse
    /// solver" section and `vendor/suitesparse/PROVENANCE.md`. For the
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
}

/// Outcome of a single Newton-Raphson solve — returned by
/// [`PersistentSolver::solve`] (the one-shot `newton_raphson`/
/// `newton_raphson_with_backend` stay `()`-returning, printing status to
/// stdout instead, to avoid changing their long-established signature).
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

pub fn newton_raphson(buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize) {
    newton_raphson_with_backend(buses, ybus, tol, max_iter, JacobianBackend::Scalar);
}

pub fn newton_raphson_with_backend(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    tol: f64,
    max_iter: usize,
    backend: JacobianBackend,
) {
    match backend {
        JacobianBackend::Scalar => newton_raphson_scalar(buses, ybus, tol, max_iter),
        JacobianBackend::Block => newton_raphson_block(buses, ybus, tol, max_iter),
        #[cfg(feature = "klu")]
        JacobianBackend::Klu => newton_raphson_klu(buses, ybus, tol, max_iter),
        JacobianBackend::KluNative => newton_raphson_native_klu(buses, ybus, tol, max_iter),
    }
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
/// # use gridoxide::solver::{PersistentSolver, JacobianBackend};
/// # use gridoxide::network::YBusSparse;
/// # use gridoxide::types::Bus;
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
    scalar: Option<RealSparseSystem>,
    block: Option<BlockSymbolic>,
    #[cfg(feature = "klu")]
    klu: Option<KluRealSystem>,
    klu_native: Option<KluNativeSystem>,
}

impl PersistentSolver {
    pub fn new(backend: JacobianBackend) -> Self {
        Self {
            backend,
            scalar: None,
            block: None,
            #[cfg(feature = "klu")]
            klu: None,
            klu_native: None,
        }
    }

    /// Runs Newton-Raphson to convergence (or `max_iter`), reusing cached
    /// symbolic factorization from a previous `solve()` call on this same
    /// `PersistentSolver` when available. See the type-level doc comment
    /// for when a cached factorization stays valid.
    pub fn solve(&mut self, buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize) -> SolveStatus {
        match self.backend {
            JacobianBackend::Scalar => newton_raphson_scalar_cached(buses, ybus, tol, max_iter, &mut self.scalar),
            JacobianBackend::Block => newton_raphson_block_cached(buses, ybus, tol, max_iter, &mut self.block),
            #[cfg(feature = "klu")]
            JacobianBackend::Klu => newton_raphson_klu_cached(buses, ybus, tol, max_iter, &mut self.klu),
            JacobianBackend::KluNative => {
                newton_raphson_native_klu_cached(buses, ybus, tol, max_iter, &mut self.klu_native)
            }
        }
    }

    /// Discards any cached symbolic factorization. Call this before the
    /// next `solve()` if the topology (not just bus values) has changed
    /// since the last call.
    pub fn reset(&mut self) {
        self.scalar = None;
        self.block = None;
        #[cfg(feature = "klu")]
        {
            self.klu = None;
        }
        self.klu_native = None;
    }
}

/// Runs Newton-Raphson to convergence like [`newton_raphson_with_backend`],
/// but additionally enforces each `PV` bus's `q_min`/`q_max` — the one gap
/// every reference power-flow tool this project benchmarks against either
/// has, half-has, or explicitly disclaims not having (see
/// `references/FEATURE_COMPARISON.md`). Plain `newton_raphson`/
/// `PersistentSolver::solve` ignore `q_min`/`q_max` entirely, matching every
/// existing test/benchmark's behavior unchanged; this is a separate, opt-in
/// entry point.
///
/// Implements the standard "PV→PQ switching" heuristic (the same algorithm
/// MATPOWER's `runpf` uses under `enforce_q_lims`): solve with every `PV`
/// bus free, then check each one's actual computed Q against its limits. A
/// bus that violates one is switched to `PQ` with `q_spec` pinned at the
/// violated limit (so it now targets exactly that Q, letting its voltage
/// float instead of holding `u_ref`) and the whole system is re-solved —
/// repeated until an outer pass finds no new violations, or `max_outer_iter`
/// is exhausted. Switching is one-directional (a bus switched to `PQ` here
/// never switches back to `PV` within the same call) — deliberately simple,
/// avoiding the oscillation a bidirectional scheme would need real
/// anti-oscillation logic to prevent, matching MATPOWER's own default
/// behavior.
///
/// Bus voltages are *not* reset between outer passes — each re-solve starts
/// from the previous pass's converged state, which is normally very close
/// to the next equilibrium (only one bus's type changed), so this typically
/// converges in very few extra Newton iterations per outer pass.
///
/// Every outer pass invalidates and rebuilds the cached factorization
/// (`PersistentSolver::reset`), since switching a bus from `PV` to `PQ`
/// changes `n_unknowns` itself for the `Scalar`/`Klu` backends (the `Block`
/// backend's per-bus block count doesn't change, only that bus's block
/// values, but it's reset here too for simplicity and to stay
/// backend-agnostic).
///
/// **Scope note**: a `PV` bus's `q_min`/`q_max` bound its *net* reactive
/// injection, matching how `Bus.q_spec` is itself already a net value
/// aggregated from every load/gen at that node (see `PgmVoltageRegulator`'s
/// doc comment) — pinning `q_spec` to the violated limit exactly achieves
/// that limit only if the bus carries no voltage-dependent `zip_terms` (no
/// co-located ZIP-model load); the common case for a PV/generator bus.
pub fn newton_raphson_enforcing_q_limits(
    buses: &mut [Bus],
    ybus: &YBusSparse,
    tol: f64,
    max_iter: usize,
    backend: JacobianBackend,
    max_outer_iter: usize,
) -> SolveStatus {
    let mut solver = PersistentSolver::new(backend);
    for _ in 0..max_outer_iter {
        let status = solver.solve(buses, ybus, tol, max_iter);
        if status != SolveStatus::Converged {
            return status;
        }

        let (_, q_calc) = power_injections(buses, ybus);
        let mut switched = false;
        for b in buses.iter_mut() {
            if b.bus_type != BusType::PV {
                continue;
            }
            let q = q_calc[b.idx];
            if q < b.q_min {
                println!("bus {}: Q={:.6} below q_min={:.6}, switching PV -> PQ", b.idx, q, b.q_min);
                b.bus_type = BusType::PQ;
                b.q_spec = b.q_min;
                switched = true;
            } else if q > b.q_max {
                println!("bus {}: Q={:.6} above q_max={:.6}, switching PV -> PQ", b.idx, q, b.q_max);
                b.bus_type = BusType::PQ;
                b.q_spec = b.q_max;
                switched = true;
            }
        }
        if !switched {
            return SolveStatus::Converged;
        }
        solver.reset();
    }

    println!("Q-limit enforcement did not stabilize within {} outer iterations", max_outer_iter);
    SolveStatus::MaxIterationsReached
}

fn newton_raphson_scalar(buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize) {
    let mut sparse_system: Option<RealSparseSystem> = None;
    newton_raphson_scalar_cached(buses, ybus, tol, max_iter, &mut sparse_system);
}

/// Identical to `newton_raphson_scalar`, except the cached symbolic
/// factorization lives in a caller-supplied `sparse_system` rather than a
/// function-local variable (what lets `PersistentSolver` reuse it across
/// repeated `solve()` calls on unchanged topology, not just across the
/// iterations within a single call), and convergence status is returned
/// rather than only printed.
fn newton_raphson_scalar_cached(
    buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize,
    sparse_system: &mut Option<RealSparseSystem>,
) -> SolveStatus {
    // Identify PV and PQ indices (exclude slack)
    let mut pv_idx: Vec<usize> = Vec::new();
    let mut pq_idx: Vec<usize> = Vec::new();
    for b in buses.iter() {
        match b.bus_type {
            BusType::Slack => (),
            BusType::PV => pv_idx.push(b.idx),
            BusType::PQ => pq_idx.push(b.idx),
        }
    }

    let non_slack_idx: Vec<usize> = buses
        .iter()
        .filter(|b| !matches!(b.bus_type, BusType::Slack))
        .map(|b| b.idx)
        .collect();

    let n_angle = non_slack_idx.len();
    let n_vmag = pq_idx.len();
    let n_unknowns = n_angle + n_vmag;

    // Physical bus index -> position within non_slack_idx / pq_idx, for
    // O(1) lookup while walking each bus's actual Y-bus neighbors (as
    // opposed to the full non_slack_idx × non_slack_idx / pq_idx × pq_idx
    // cross product the dense implementation used).
    let mut non_slack_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in non_slack_idx.iter().enumerate() {
        non_slack_pos[i] = Some(pos);
    }
    let mut pq_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in pq_idx.iter().enumerate() {
        pq_pos[i] = Some(pos);
    }
    let triplet_capacity = jacobian_triplet_capacity(ybus, &non_slack_idx, &pq_idx);

    // The Jacobian's sparsity *pattern* is fixed across iterations (same
    // bus topology every time, only numeric values change), so the
    // symbolic factorization (ordering + fill-in) is computed once here and
    // reused via `factor_and_solve` for every iteration's numeric-only
    // refactorization, mirroring PGM's own prefactorization-reuse approach.
    for iter in 0..max_iter {
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
        println!("iter {}: max mismatch = {:.6e}", iter + 1, max_mis);
        if max_mis < tol {
            println!("Converged in {} iterations", iter + 1);
            return SolveStatus::Converged;
        }

        // Build Jacobian (sparse triplets)
        let triplets = build_jacobian_triplets(
            buses, ybus, &non_slack_idx, &pq_idx, &non_slack_pos, &pq_pos, n_angle, &p_calc, &q_calc,
            triplet_capacity,
        );

        if sparse_system.is_none() {
            *sparse_system = RealSparseSystem::new(n_unknowns, &triplets);
        }
        let Some(system) = sparse_system.as_ref() else {
            println!("Jacobian is singular. Failed to solve.");
            return SolveStatus::Singular;
        };
        let dx = match system.factor_and_solve(&triplets, &mismatch) {
            Some(sol) => sol,
            None => {
                println!("Jacobian is singular. Failed to solve.");
                return SolveStatus::Singular;
            }
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

    println!("Failed to converge in {} iterations", max_iter);
    SolveStatus::MaxIterationsReached
}

/// Identical to `newton_raphson_scalar` except the sparse solve is backed by
/// `sparse_klu::KluRealSystem` instead of `sparse::RealSparseSystem` — reuses
/// `build_jacobian_triplets` unchanged, since `Klu` solves the same scalar
/// Jacobian shape as `Scalar`, only the solver library differs.
#[cfg(feature = "klu")]
fn newton_raphson_klu(buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize) {
    let mut sparse_system: Option<KluRealSystem> = None;
    newton_raphson_klu_cached(buses, ybus, tol, max_iter, &mut sparse_system);
}

/// See `newton_raphson_scalar_cached` — same caller-supplied-cache pattern,
/// for the `Klu` backend.
#[cfg(feature = "klu")]
fn newton_raphson_klu_cached(
    buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize,
    sparse_system: &mut Option<KluRealSystem>,
) -> SolveStatus {
    let mut pv_idx: Vec<usize> = Vec::new();
    let mut pq_idx: Vec<usize> = Vec::new();
    for b in buses.iter() {
        match b.bus_type {
            BusType::Slack => (),
            BusType::PV => pv_idx.push(b.idx),
            BusType::PQ => pq_idx.push(b.idx),
        }
    }

    let non_slack_idx: Vec<usize> = buses
        .iter()
        .filter(|b| !matches!(b.bus_type, BusType::Slack))
        .map(|b| b.idx)
        .collect();

    let n_angle = non_slack_idx.len();
    let n_vmag = pq_idx.len();
    let n_unknowns = n_angle + n_vmag;

    let mut non_slack_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in non_slack_idx.iter().enumerate() {
        non_slack_pos[i] = Some(pos);
    }
    let mut pq_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in pq_idx.iter().enumerate() {
        pq_pos[i] = Some(pos);
    }
    let triplet_capacity = jacobian_triplet_capacity(ybus, &non_slack_idx, &pq_idx);

    for iter in 0..max_iter {
        let (p_calc, q_calc) = power_injections(buses, ybus);

        let mut mismatch = vec![0.0; n_unknowns];
        let mut mis_idx = 0;
        for &i in &non_slack_idx {
            let (p_eff, _) = effective_injection(&buses[i]);
            mismatch[mis_idx] = p_eff - p_calc[i];
            mis_idx += 1;
        }
        for &i in &pq_idx {
            let (_, q_eff) = effective_injection(&buses[i]);
            mismatch[mis_idx] = q_eff - q_calc[i];
            mis_idx += 1;
        }

        let max_mis = mismatch.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        println!("iter {}: max mismatch = {:.6e}", iter + 1, max_mis);
        if max_mis < tol {
            println!("Converged in {} iterations", iter + 1);
            return SolveStatus::Converged;
        }

        let triplets = build_jacobian_triplets(
            buses, ybus, &non_slack_idx, &pq_idx, &non_slack_pos, &pq_pos, n_angle, &p_calc, &q_calc,
            triplet_capacity,
        );

        if sparse_system.is_none() {
            *sparse_system = KluRealSystem::new(n_unknowns, &triplets);
        }
        let Some(system) = sparse_system.as_mut() else {
            println!("Jacobian is singular. Failed to solve.");
            return SolveStatus::Singular;
        };
        let dx = match system.factor_and_solve(&triplets, &mismatch) {
            Some(sol) => sol,
            None => {
                println!("Jacobian is singular. Failed to solve.");
                return SolveStatus::Singular;
            }
        };

        let mut dx_idx = 0;
        for &i in &non_slack_idx {
            buses[i].voltage_ang += dx[dx_idx];
            dx_idx += 1;
        }
        for &i in &pq_idx {
            buses[i].voltage_mag += dx[dx_idx];
            dx_idx += 1;
        }
    }

    println!("Failed to converge in {} iterations", max_iter);
    SolveStatus::MaxIterationsReached
}

/// Identical to `newton_raphson_klu`, except the sparse solve is backed by
/// `klu_native::KluNativeSystem` — the pure-Rust port of the same KLU
/// algorithm — instead of the FFI-linked vendored C. Reuses
/// `build_jacobian_triplets` unchanged, since `KluNative` solves the same
/// scalar Jacobian shape as `Scalar`/`Klu`, only the solver implementation
/// differs.
fn newton_raphson_native_klu(buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize) {
    let mut sparse_system: Option<KluNativeSystem> = None;
    newton_raphson_native_klu_cached(buses, ybus, tol, max_iter, &mut sparse_system);
}

/// See `newton_raphson_scalar_cached` — same caller-supplied-cache pattern,
/// for the `KluNative` backend.
fn newton_raphson_native_klu_cached(
    buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize,
    sparse_system: &mut Option<KluNativeSystem>,
) -> SolveStatus {
    let mut pv_idx: Vec<usize> = Vec::new();
    let mut pq_idx: Vec<usize> = Vec::new();
    for b in buses.iter() {
        match b.bus_type {
            BusType::Slack => (),
            BusType::PV => pv_idx.push(b.idx),
            BusType::PQ => pq_idx.push(b.idx),
        }
    }

    let non_slack_idx: Vec<usize> = buses
        .iter()
        .filter(|b| !matches!(b.bus_type, BusType::Slack))
        .map(|b| b.idx)
        .collect();

    let n_angle = non_slack_idx.len();
    let n_vmag = pq_idx.len();
    let n_unknowns = n_angle + n_vmag;

    let mut non_slack_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in non_slack_idx.iter().enumerate() {
        non_slack_pos[i] = Some(pos);
    }
    let mut pq_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in pq_idx.iter().enumerate() {
        pq_pos[i] = Some(pos);
    }
    let triplet_capacity = jacobian_triplet_capacity(ybus, &non_slack_idx, &pq_idx);

    for iter in 0..max_iter {
        let (p_calc, q_calc) = power_injections(buses, ybus);

        let mut mismatch = vec![0.0; n_unknowns];
        let mut mis_idx = 0;
        for &i in &non_slack_idx {
            let (p_eff, _) = effective_injection(&buses[i]);
            mismatch[mis_idx] = p_eff - p_calc[i];
            mis_idx += 1;
        }
        for &i in &pq_idx {
            let (_, q_eff) = effective_injection(&buses[i]);
            mismatch[mis_idx] = q_eff - q_calc[i];
            mis_idx += 1;
        }

        let max_mis = mismatch.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        println!("iter {}: max mismatch = {:.6e}", iter + 1, max_mis);
        if max_mis < tol {
            println!("Converged in {} iterations", iter + 1);
            return SolveStatus::Converged;
        }

        let triplets = build_jacobian_triplets(
            buses, ybus, &non_slack_idx, &pq_idx, &non_slack_pos, &pq_pos, n_angle, &p_calc, &q_calc,
            triplet_capacity,
        );

        if sparse_system.is_none() {
            *sparse_system = KluNativeSystem::new(n_unknowns, &triplets);
        }
        let Some(system) = sparse_system.as_mut() else {
            println!("Jacobian is singular. Failed to solve.");
            return SolveStatus::Singular;
        };
        let dx = match system.factor_and_solve(&triplets, &mismatch) {
            Some(sol) => sol,
            None => {
                println!("Jacobian is singular. Failed to solve.");
                return SolveStatus::Singular;
            }
        };

        let mut dx_idx = 0;
        for &i in &non_slack_idx {
            buses[i].voltage_ang += dx[dx_idx];
            dx_idx += 1;
        }
        for &i in &pq_idx {
            buses[i].voltage_mag += dx[dx_idx];
            dx_idx += 1;
        }
    }

    println!("Failed to converge in {} iterations", max_iter);
    SolveStatus::MaxIterationsReached
}

/// Upper bound on `build_jacobian_triplets`' output length for a given
/// topology — computed once per solve (not per iteration) and reused via
/// `Vec::with_capacity`, since the sparsity pattern, and hence this bound,
/// never changes across a solve's iterations while `build_jacobian_triplets`
/// itself starts a fresh `Vec` from scratch every iteration. Each Y-bus
/// neighbor contributes at most 2 triplets to the H/N block (one per
/// non-slack row) and at most 2 to the M/L block (one per PQ row).
fn jacobian_triplet_capacity(ybus: &YBusSparse, non_slack_idx: &[usize], pq_idx: &[usize]) -> usize {
    let non_slack_degree: usize = non_slack_idx.iter().map(|&i| ybus.row(i).len()).sum();
    let pq_degree: usize = pq_idx.iter().map(|&i| ybus.row(i).len()).sum();
    2 * non_slack_degree + 2 * pq_degree
}

/// Assembles the Newton-Raphson Jacobian's nonzero entries as `(row, col,
/// value)` triplets, walking only each unknown bus's actual Y-bus neighbors
/// (`ybus.row(i)`) instead of the full cross product of unknown-bus indices.
/// Every topological neighbor is always included regardless of its computed
/// value (even if it happens to evaluate to exactly zero at some iteration),
/// so the resulting sparsity *pattern* is identical across calls — required
/// for `RealSparseSystem`'s cached symbolic factorization to stay valid.
///
/// Block structure (H/N/M/L), matching the original dense Jacobian:
/// ```text
/// J = [ H  N ]   H = dP/d_ang, N = dP/d_vmag
///     [ M  L ]   M = dQ/d_ang, L = dQ/d_vmag
/// ```
#[allow(clippy::too_many_arguments)]
fn build_jacobian_triplets(
    buses: &[Bus],
    ybus: &YBusSparse,
    non_slack_idx: &[usize],
    pq_idx: &[usize],
    non_slack_pos: &[Option<usize>],
    pq_pos: &[Option<usize>],
    n_angle: usize,
    p_calc: &[f64],
    q_calc: &[f64],
    triplet_capacity: usize,
) -> Vec<(usize, usize, f64)> {
    let vm: Vec<f64> = buses.iter().map(|b| b.voltage_mag).collect();
    let va: Vec<f64> = buses.iter().map(|b| b.voltage_ang).collect();
    let mut triplets = Vec::with_capacity(triplet_capacity);

    // H and N blocks: rows = non_slack_idx (P-mismatch equations).
    for (row_idx, &i) in non_slack_idx.iter().enumerate() {
        for &(k, y_ik) in ybus.row(i) {
            if k == i {
                // H_ii = -Q_i - V_i^2 * B_ii
                triplets.push((row_idx, row_idx, -q_calc[i] - vm[i].powi(2) * y_ik.im));
                // N_ii = P_i/V_i + V_i * G_ii (only if i has a magnitude unknown)
                if let Some(col_idx) = pq_pos[i] {
                    triplets.push((row_idx, n_angle + col_idx, p_calc[i] / vm[i] + vm[i] * y_ik.re));
                }
                continue;
            }
            let angle_ik = va[i] - va[k];
            if let Some(col_idx) = non_slack_pos[k] {
                // H_ik = V_i * V_k * (G_ik * sin(d_ik) - B_ik * cos(d_ik))
                let h_ik = vm[i] * vm[k] * (y_ik.re * angle_ik.sin() - y_ik.im * angle_ik.cos());
                triplets.push((row_idx, col_idx, h_ik));
            }
            if let Some(col_idx) = pq_pos[k] {
                // N_ik = V_i * (G_ik * cos(d_ik) + B_ik * sin(d_ik))
                let n_ik = vm[i] * (y_ik.re * angle_ik.cos() + y_ik.im * angle_ik.sin());
                triplets.push((row_idx, n_angle + col_idx, n_ik));
            }
        }
    }

    // M and L blocks: rows = pq_idx (Q-mismatch equations).
    for (row_idx, &i) in pq_idx.iter().enumerate() {
        for &(k, y_ik) in ybus.row(i) {
            if k == i {
                // M_ii = P_i - V_i^2 * G_ii; column is i's position among
                // *all* non-slack buses, not just PQ ones (may differ from
                // row_idx once PV buses exist).
                let col_idx = non_slack_pos[i].expect("a PQ bus is always non-slack");
                triplets.push((n_angle + row_idx, col_idx, p_calc[i] - vm[i].powi(2) * y_ik.re));
                // L_ii = Q_i/V_i - V_i * B_ii
                triplets.push((n_angle + row_idx, n_angle + row_idx, q_calc[i] / vm[i] - vm[i] * y_ik.im));
                continue;
            }
            let angle_ik = va[i] - va[k];
            if let Some(col_idx) = non_slack_pos[k] {
                // M_ik = -V_i * V_k * (G_ik * cos(d_ik) + B_ik * sin(d_ik))
                let m_ik = -vm[i] * vm[k] * (y_ik.re * angle_ik.cos() + y_ik.im * angle_ik.sin());
                triplets.push((n_angle + row_idx, col_idx, m_ik));
            }
            if let Some(col_idx) = pq_pos[k] {
                // L_ik = V_i * (G_ik * sin(d_ik) - B_ik * cos(d_ik))
                let l_ik = vm[i] * (y_ik.re * angle_ik.sin() - y_ik.im * angle_ik.cos());
                triplets.push((n_angle + row_idx, n_angle + col_idx, l_ik));
            }
        }
    }

    triplets
}

/// Experimental block-per-bus Newton-Raphson, backed by
/// `block_sparse::BlockLu`. See `JacobianBackend::Block`'s doc comment for
/// scope. Like `newton_raphson_scalar`'s `RealSparseSystem` reuse, the
/// `colamd` ordering and elimination-graph reachability (`BlockSymbolic`)
/// are computed once and reused via `BlockLu::refactor` for cheap
/// numeric-only refactorization on every subsequent iteration.
fn newton_raphson_block(buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize) {
    let mut symbolic: Option<BlockSymbolic> = None;
    newton_raphson_block_cached(buses, ybus, tol, max_iter, &mut symbolic);
}

/// See `newton_raphson_scalar_cached` — same caller-supplied-cache pattern,
/// for the `Block` backend.
fn newton_raphson_block_cached(
    buses: &mut [Bus], ybus: &YBusSparse, tol: f64, max_iter: usize,
    symbolic: &mut Option<BlockSymbolic>,
) -> SolveStatus {
    let non_slack_idx: Vec<usize> = buses
        .iter()
        .filter(|b| !matches!(b.bus_type, BusType::Slack))
        .map(|b| b.idx)
        .collect();

    let mut block_pos: Vec<Option<usize>> = vec![None; buses.len()];
    for (pos, &i) in non_slack_idx.iter().enumerate() {
        block_pos[i] = Some(pos);
    }

    for iter in 0..max_iter {
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

        let max_mis = mismatch.iter().flatten().fold(0.0f64, |a, &b| a.max(b.abs()));
        println!("iter {}: max mismatch = {:.6e}", iter + 1, max_mis);
        if max_mis < tol {
            println!("Converged in {} iterations", iter + 1);
            return SolveStatus::Converged;
        }

        let blocks = build_jacobian_blocks(buses, ybus, &non_slack_idx, &block_pos, &p_calc, &q_calc).finish();
        if symbolic.is_none() {
            *symbolic = Some(BlockSymbolic::analyze(&blocks));
        }
        let Some(lu) = BlockLu::refactor(symbolic.as_ref().unwrap(), &blocks) else {
            println!("Jacobian is singular. Failed to solve.");
            return SolveStatus::Singular;
        };
        let dx = lu.solve(&mismatch);

        for (pos, &i) in non_slack_idx.iter().enumerate() {
            buses[i].voltage_ang += dx[pos][0];
            buses[i].voltage_mag += dx[pos][1];
        }
    }

    println!("Failed to converge in {} iterations", max_iter);
    SolveStatus::MaxIterationsReached
}

/// Assembles the same H/N/M/L formulas as `build_jacobian_triplets`, but as
/// one 2×2 block per (bus, bus) pair — `[[H, N], [M, L]]` on the diagonal,
/// `[[H_ik, N_ik], [M_ik, L_ik]]` off-diagonal — instead of four separate
/// scalar triplets.
///
/// A `PV` row bus has no real Q-mismatch equation, so its block's second row
/// is replaced with the dummy `ΔVmag_i = 0` equation instead: `[0, 1]` on
/// the diagonal block (pins this bus's own magnitude update to zero) and
/// `[0, 0]` on every off-diagonal block in that row (so the dummy equation
/// stays fully decoupled from every other bus's unknowns — required for it
/// to pin exactly `0`, not something entangled with the rest of the solve).
/// The H/N formulas themselves (first row) are unaffected — they're already
/// correct for any non-slack bus, PV or PQ, matching
/// `build_jacobian_triplets`'s scalar H/N block.
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
                    (p_calc[i] - vm[i].powi(2) * y_ik.re, q_calc[i] / vm[i] - vm[i] * y_ik.im)
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
