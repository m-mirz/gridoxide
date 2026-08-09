//! The DC approximation to AC power flow: `B θ = P`.
//!
//! Three assumptions turn the AC power-flow equations into one linear system:
//! every voltage magnitude is 1 p.u., every branch resistance is zero, and
//! every angle difference is small enough that `sin δ ≈ δ` and `cos δ ≈ 1`.
//! What survives is exact, direct and unconditionally solvable — no initial
//! guess, no iteration, no convergence to fail. What is lost is every
//! reactive quantity and every voltage magnitude, so DC answers exactly one
//! question: how does active power divide between the available paths.
//!
//! # The sign conventions, derived from gridoxide's own AC code
//!
//! Rather than copy MATPOWER's formulas and hope the conventions line up, the
//! ones below are derived from [`network::branch_calc_param`], which is this
//! crate's authority on branch orientation. With `tap = k·e^{jα}` that
//! function produces `yft = −y_s/conj(tap)` and `yff = (y_s + y_sh/2)/k²`, so
//! for `y_s = g + j·b_s`, `y_sh = 0`, `|V| = 1` and `δ = θ_f − θ_t − α`:
//!
//! \\[ P_{from} = \frac{g}{k^2} - \frac{1}{k}\left[g\cos\delta + b_s\sin\delta\right] \\]
//!
//! Dropping `g` and linearizing leaves the two formulas this module implements:
//!
//! \\[ P_{from} = b\,(\theta_{from} - \theta_{to} - \alpha), \qquad P_{to} = -P_{from} \\]
//! \\[ b = \frac{-b_s}{k} = \frac{x}{k\,(r^2 + x^2)} \\]
//!
//! This agrees with MATPOWER `makeBdc` exactly (`b = 1./(x.*tap)`,
//! `Pfinj = b .* (-shift·π/180)`). It appears to disagree with
//! powsybl-open-loadflow, which writes `p1 = b·(θ1 − θ2 + a1)` — but powsybl's
//! `a1` is the negation of gridoxide's `tap.arg()`, so the two agree on the
//! physics. Do not "fix" the sign here to match powsybl's source without
//! following that definition through.
//!
//! The phase shift never enters the matrix: it is a constant, so it moves to
//! the right-hand side and `B` stays symmetric even on a network full of
//! phase-shifting transformers.
//!
//! \\[ B\,\theta = P + \varphi, \qquad
//!    \varphi_i = \sum_{from_k = i} b_k \alpha_k - \sum_{to_k = i} b_k \alpha_k \\]
//!
//! (MATPOWER stores `−φ` and calls it `Pbusinj`.) The symmetry of `B` is
//! relied on by [`sensitivity`](super::sensitivity), which solves against
//! `Bᵀ` without ever forming it.
//!
//! # What this module deliberately does not do
//!
//! Shunt admittance is ignored entirely, both `b` and `g`, matching
//! powsybl-open-loadflow (whose `DcEquationSystemCreator` has no shunt term).
//! MATPOWER and pandapower additionally fold `Gs` in as a constant real load
//! at `|V| = 1`; gridoxide does not, so on data where `g_shunt ≠ 0` its slack
//! pickup differs from theirs by exactly `Σ g_ii`. Covering one of gridoxide's
//! three shunt sources (`Line::g_shunt`, `Transformer::y_shunt`, and the
//! separate `network::ShuntAdm` list this function is not even handed) and not
//! the others would be worse than covering none.
//!
//! Bus type is ignored beyond `Slack`: `PV` and `PQ` are treated identically,
//! because the distinction is entirely about reactive power.

use num_complex::Complex;

use crate::network::{classify, mark_unreferenced_islands, Classified, Verdict};
use crate::sparse::RealFactorization;
use crate::topology::{clamp_branch_impedance, ideal_connection_z, union_all, IDEAL_CONNECTION_Y};
use crate::types::{Bus, BusType, Line, Transformer};

use super::{DcApproximation, DcOptions};

/// One series branch reduced to the two numbers the DC model needs.
///
/// Branches with no series path at all — a half-open self-loop line, a
/// transformer with an open terminal — produce no `DcBranch`: they contribute
/// nothing to `B` and carry no DC flow. Their flat indices show up in
/// [`DcSolution::ignored_branches`] instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DcBranch {
    /// Flat branch index — lines first, then transformers. This is the
    /// crate-wide branch index space that [`branch_flow::branch_params`] and
    /// `pgm::PgmNetwork::branch_idx` also use, so a `DcBranch` and a
    /// `BranchParams` with the same `index` are the same physical branch.
    ///
    /// [`branch_flow::branch_params`]: crate::branch_flow::branch_params
    pub index: usize,
    pub from: usize,
    pub to: usize,
    /// Susceptance entering `B`, already divided by the off-nominal ratio
    /// `k`. Positive for every ordinary inductive branch.
    pub b: f64,
    /// Phase shift α = `tap.arg()`, radians. Always 0 for a [`Line`].
    pub shift: f64,
}

/// How one connected component's DC solve turned out.
///
/// Deliberately *not* [`solver::IslandStatus`](crate::solver::IslandStatus).
/// A direct solve has no iterations, so that enum's `MaxIterationsReached`
/// is unrepresentable here — and two of its cases genuinely come out better
/// in DC, which a shared type could not express:
///
/// - `IslandStatus::Singular` documents that it cannot attribute a singular
///   factorization to one component, because the AC path solves every
///   component in one shared system. DC factorizes each island separately
///   (the LU of a block-diagonal matrix *is* the sum of the block LUs, so
///   this costs nothing), and so can name the island that failed.
/// - `IslandStatus::AmbiguousReferenceBus` exists because two independently
///   fixed complex voltages over-determine an AC island. DC fixes only
///   angles, so the same island stays perfectly well posed —
///   see [`SolvedMultipleReferences`](Self::SolvedMultipleReferences).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DcIslandStatus {
    /// Reduced `B` factorized; this island's angles satisfy its equations.
    Solved,
    /// Two or more `Slack` buses, and solved anyway. Every non-slack bus's
    /// injection is still met exactly — only the *split* of slack pickup
    /// among the references is arbitrary, since DC pins their angles
    /// independently and lets each supply whatever its own incident branches
    /// demand. The per-island `slack_pickup` total is still meaningful; the
    /// individual references' shares are not.
    SolvedMultipleReferences,
    /// The reduced `B` was singular, so this island's angles were left
    /// untouched and its branch flows are zero. Attributable to exactly this
    /// island, unlike the AC path's best-effort equivalent.
    Singular,
    /// No `Slack` bus, so there is no angle reference and nothing to solve
    /// against. Pinned to `θ = 0` by
    /// [`network::mark_unreferenced_islands`](crate::network), which never
    /// fabricates a reference.
    NoReferenceBus,
}

/// One connected component's DC outcome.
#[derive(Clone, Debug)]
pub struct DcIslandReport {
    pub bus_indices: Vec<usize>,
    /// The `Slack` bus(es) found in this component *before*
    /// `mark_unreferenced_islands` ran — empty for
    /// [`NoReferenceBus`](DcIslandStatus::NoReferenceBus).
    pub slack_indices: Vec<usize>,
    pub status: DcIslandStatus,
    /// Total active power this island's reference bus(es) supply, per-unit.
    /// Equal and opposite to the sum of every other bus's injection, since
    /// DC is lossless — which is what
    /// `slack_pickup_balances_injections_exactly` asserts.
    pub slack_pickup: f64,
}

/// The result of a DC solve. Bus angles are written back into the `buses`
/// slice; everything that has nowhere to live on a [`Bus`] is here.
#[derive(Clone, Debug)]
pub struct DcSolution {
    pub islands: Vec<DcIslandReport>,
    /// Active power entering each branch at its `from` terminal, per-unit,
    /// indexed by the **flat** branch index (lines then transformers) — never
    /// by position in [`dc_branches`]'s output. Branches that carry no DC
    /// flow hold `0.0`.
    ///
    /// Sign convention matches
    /// [`branch_flow::terminal_flow`](crate::branch_flow::terminal_flow):
    /// positive means power entering the branch at `from`. DC is lossless, so
    /// the `to`-terminal flow is always exactly the negation.
    pub branch_p: Vec<f64>,
    /// `max |Σ(flows out of i) − P_i|` over every solved non-slack bus.
    ///
    /// A factorization-quality check, **not** a convergence check — there is
    /// nothing to converge. On a healthy network this is round-off, ~1e-15.
    /// A large value means `B` was ill-conditioned, which on gridoxide data
    /// most often means clamped zero-impedance branches (see
    /// [`topology::ZERO_IMPEDANCE_THRESHOLD`](crate::topology::ZERO_IMPEDANCE_THRESHOLD)).
    pub max_residual: f64,
    /// Flat indices of branches with a negative susceptance — series
    /// capacitors. Legitimate and kept, but they make `B` indefinite, so they
    /// are reported rather than passed over in silence.
    pub negative_reactance_branches: Vec<usize>,
    /// Flat indices of branches that contributed nothing: self-loops
    /// (gridoxide's half-open representation), transformers with an open
    /// terminal, and any branch whose susceptance came out non-finite (a
    /// purely resistive branch under
    /// [`DcApproximation::IgnoreR`](super::DcApproximation::IgnoreR)).
    pub ignored_branches: Vec<usize>,
}

/// Active-power injection under the DC assumption.
///
/// Every [`ZipKind`](crate::types::ZipKind) collapses to its own `s_const` at
/// `|V| = 1`, so this is `network::effective_injection(bus).0` evaluated
/// there. It is written out rather than delegated precisely because
/// `effective_injection` reads the bus's *current* `voltage_mag` — which for
/// a DC solve is whatever the caller happened to leave in the struct, and is
/// not part of the DC model at all.
fn dc_injection(bus: &Bus) -> f64 {
    bus.p_spec + bus.zip_terms.iter().map(|z| z.s_const.re).sum::<f64>()
}

/// The branch susceptance entering `B`, before the transformer ratio.
///
/// Each approximation clamps against
/// [`topology::ZERO_IMPEDANCE_THRESHOLD`](crate::topology::ZERO_IMPEDANCE_THRESHOLD)
/// on *the impedance it actually models*, which is not the same quantity for
/// the two of them:
///
/// - [`IgnoreR`](DcApproximation::IgnoreR) asserts `r = 0`, so a branch's
///   impedance in its world is `|x|` alone. A line with `x = 0` and any `r`
///   at all — power-grid-model's `link/dummy-test` fixture contains exactly
///   one, at `r = 10 Ω, x = 0` — is therefore a *short circuit* as far as
///   this approximation is concerned, and gets the crate's standard
///   zero-impedance treatment rather than an infinite susceptance. Without
///   this, `1/x` overflows, the branch has to be dropped, and dropping it
///   silently splits the network into a live island and a sourceless one.
/// - [`IgnoreG`](DcApproximation::IgnoreG) keeps `r`, so it clamps on the
///   full `|z|` and that same line comes out at `b = 0`. That is not a bug
///   but the correct answer for the model: with `x = 0` the exact AC flow
///   `g/k² − g·cos δ` is second order in `δ`, so a purely resistive branch
///   genuinely transmits no angle-driven active power. The two
///   approximations disagree about such a branch by construction, and a
///   network held together by one will island under `IgnoreG`.
fn susceptance(r: f64, x: f64, approximation: DcApproximation) -> f64 {
    match approximation {
        DcApproximation::IgnoreR => 1.0 / clamp_branch_impedance(0.0, x).1,
        DcApproximation::IgnoreG => {
            let (r, x) = clamp_branch_impedance(r, x);
            x / (r * r + x * x)
        }
    }
}

/// Reduces every branch to its DC parameters, in flat branch-index order.
///
/// See [`DcBranch::index`] for the index contract and the module doc for the
/// susceptance and phase-shift conventions.
pub fn dc_branches(
    lines: &[Line],
    transformers: &[Transformer],
    opts: DcOptions,
) -> Vec<DcBranch> {
    let mut out = Vec::with_capacity(lines.len() + transformers.len());

    for (i, ln) in lines.iter().enumerate() {
        // A self-loop is gridoxide's half-open branch: pure shunt, no series
        // path, so nothing for DC to carry. See `branch_flow::line_params`.
        if ln.from == ln.to {
            continue;
        }
        let b = susceptance(ln.r, ln.x, opts.approximation);
        if !b.is_finite() {
            continue;
        }
        out.push(DcBranch { index: i, from: ln.from, to: ln.to, b, shift: 0.0 });
    }

    for (j, t) in transformers.iter().enumerate() {
        let index = lines.len() + j;
        if t.from == t.to {
            continue;
        }
        // `branch_calc_param` zeroes `yft`/`ytf` when either terminal is
        // open, so such a transformer contributes only its shunt to the
        // Y-bus — and shunts are outside the DC model entirely.
        if t.from_status == 0 || t.to_status == 0 {
            continue;
        }

        let k = t.tap.norm();
        let mut b = if t.y_series == IDEAL_CONNECTION_Y {
            // A PGM `link` (`pgm::LINK_Y`) carries `2e5 + j2e5`, whose
            // *positive* imaginary part is inherited from power-grid-model's
            // own `1e8 + j1e8` and is a regularization choice, not a claim
            // that the element is capacitive. Inverting it gives
            // `x = −2.5e-6`, and a negative susceptance here would flip this
            // branch's coupling sign and destroy B's positive-definiteness.
            // AC never noticed, because only `|y|` and the link's own
            // reported Q depend on that sign — which is what
            // `tests/link_test.rs` pins against PGM's fixtures.
            //
            // Match on the constant, never on `x < 0` in general: a genuinely
            // negative reactance is a series capacitor and must pass through.
            1.0 / ideal_connection_z()
        } else {
            let z = Complex::new(1.0, 0.0) / t.y_series;
            susceptance(z.re, z.im, opts.approximation)
        };
        if opts.use_transformer_ratio && k.is_finite() && k > 0.0 {
            b /= k;
        }
        if !b.is_finite() {
            continue;
        }
        out.push(DcBranch { index, from: t.from, to: t.to, b, shift: t.tap.arg() });
    }

    out
}

/// Partitions buses into connected components using only the DC branch list.
///
/// Returns the same shape as
/// [`network::connected_components`](crate::network::connected_components):
/// members sorted ascending, components ordered by lowest member. It does not
/// call that function, which needs a `&YBusSparse` — building a full complex
/// Y-bus purely to partition a branch-list-based solver would be the tail
/// wagging the dog, and [`topology::UnionFind`](crate::topology::UnionFind)
/// already exists for exactly this.
///
/// The two partitions can genuinely differ, in one case: a **half-open
/// transformer** joins its two ends under `connected_components`, because
/// `network::stamp_transformers` inserts the structural `(from, to)` entry
/// even when `branch_calc_param` has zeroed its value. Here it does not,
/// because a branch with no series path cannot carry DC flow. Where they
/// disagree, this is the physically correct partition for DC.
fn dc_components(n: usize, branches: &[DcBranch]) -> Vec<Vec<usize>> {
    let mut uf = union_all(n, branches.iter().map(|b| (b.from, b.to)));
    let mut comp_of_root = vec![usize::MAX; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    // Ascending `i` gives both halves of the contract for free: members land
    // in each component in ascending order, and a component is created the
    // first time its lowest-indexed member is seen.
    for i in 0..n {
        let root = uf.find(i);
        if comp_of_root[root] == usize::MAX {
            comp_of_root[root] = components.len();
            components.push(Vec::new());
        }
        components[comp_of_root[root]].push(i);
    }
    components
}

/// One island partition, shared by the solver and by
/// [`sensitivity::DcSensitivity`](super::sensitivity::DcSensitivity) so the
/// two cannot drift apart in how they group buses or assign branches.
///
/// Deliberately does not mutate `buses`: the sensitivity factors are computed
/// from a network the caller still owns, and only the solver has a reason to
/// pin sourceless islands. Callers that want that must invoke
/// [`network::mark_unreferenced_islands`](crate::network) themselves.
pub(crate) struct Partition {
    pub(crate) classified: Vec<Classified>,
    /// Per island, the positions into the `branches` slice whose endpoints lie
    /// in it. A branch's endpoints are always in the same island by
    /// construction, so indexing by `from` assigns it unambiguously.
    pub(crate) island_branches: Vec<Vec<usize>>,
}

pub(crate) fn partition(buses: &[Bus], branches: &[DcBranch]) -> Partition {
    let n = buses.len();
    let components = dc_components(n, branches);
    let mut island_of = vec![usize::MAX; n];
    for (ci, members) in components.iter().enumerate() {
        for &i in members {
            island_of[i] = ci;
        }
    }
    let mut island_branches: Vec<Vec<usize>> = vec![Vec::new(); components.len()];
    for (pos, br) in branches.iter().enumerate() {
        island_branches[island_of[br.from]].push(pos);
    }
    Partition { classified: classify(buses, &components), island_branches }
}

/// Stamps one island's reduced susceptance matrix `B_NN`.
///
/// `reduced_pos` maps a bus index to its row in the reduced system, or
/// [`NOT_UNKNOWN`] for a reference bus. Reference buses contribute to the
/// diagonal of their unknown neighbours — a branch to a fixed angle still
/// stiffens the bus it lands on — but produce no off-diagonal entry, since
/// their angle is not a variable.
///
/// The result is symmetric even on a network full of phase shifters: the
/// shift is constant and lives entirely on the right-hand side. Both callers
/// depend on that, and `sensitivity` depends on it twice over, since it
/// solves against `Bᵀ` without ever forming it.
pub(crate) fn reduced_b_triplets(
    branches: &[DcBranch],
    positions: &[usize],
    reduced_pos: &[usize],
) -> Vec<(usize, usize, f64)> {
    let mut triplets = Vec::with_capacity(positions.len() * 4);
    for &pos in positions {
        let br = &branches[pos];
        let (rf, rt) = (reduced_pos[br.from], reduced_pos[br.to]);
        if rf != NOT_UNKNOWN {
            triplets.push((rf, rf, br.b));
            if rt != NOT_UNKNOWN {
                triplets.push((rf, rt, -br.b));
            }
        }
        if rt != NOT_UNKNOWN {
            triplets.push((rt, rt, br.b));
            if rf != NOT_UNKNOWN {
                triplets.push((rt, rf, -br.b));
            }
        }
    }
    triplets
}

/// Marker in a `reduced_pos` map for a bus that is a reference, and so has no
/// row in the reduced system.
pub(crate) const NOT_UNKNOWN: usize = usize::MAX;

/// Solves the DC power flow, writing bus angles back into `buses`.
///
/// Each connected component is factorized and solved independently, so one
/// island's singular `B` cannot spoil another's answer.
///
/// `voltage_ang` is overwritten on every solved non-slack bus. `voltage_mag`
/// is set to 1 p.u. on solved **`PQ`** buses only — the DC model's own
/// assumption, applied where nothing else specifies a magnitude. `Slack` and
/// `PV` buses keep theirs: those are setpoints, inputs rather than results,
/// and DC neither uses nor computes them. `Slack` buses likewise keep their
/// angle, which is the reference this island is solved against and need not
/// be zero.
///
/// Sourceless islands are handled by
/// [`network::mark_unreferenced_islands`](crate::network), exactly as the AC
/// path handles them: pinned to a placeholder, never given a fabricated
/// reference bus.
pub fn dc_power_flow(
    buses: &mut [Bus],
    lines: &[Line],
    transformers: &[Transformer],
    opts: DcOptions,
) -> DcSolution {
    let n = buses.len();
    let n_branches = lines.len() + transformers.len();
    let branches = dc_branches(lines, transformers, opts);

    let Partition { classified, island_branches } = partition(buses, &branches);
    mark_unreferenced_islands(buses, &classified);

    let mut contributing = vec![false; n_branches];
    let mut negative_reactance_branches = Vec::new();
    for br in &branches {
        contributing[br.index] = true;
        if br.b < 0.0 {
            negative_reactance_branches.push(br.index);
        }
    }
    let ignored_branches: Vec<usize> =
        (0..n_branches).filter(|&i| !contributing[i]).collect();

    let mut branch_p = vec![0.0; n_branches];
    let mut max_residual: f64 = 0.0;
    let mut islands = Vec::with_capacity(classified.len());
    // Reused across islands so the reduced-index map costs one allocation
    // rather than one per component; only each island's own members are read,
    // and each writes its own entries before reading them.
    let mut reduced_pos = vec![NOT_UNKNOWN; n];

    for (ci, c) in classified.iter().enumerate() {
        if matches!(c.verdict, Verdict::NoReferenceBus) {
            islands.push(DcIslandReport {
                bus_indices: c.bus_indices.clone(),
                slack_indices: c.slack_indices.clone(),
                status: DcIslandStatus::NoReferenceBus,
                slack_pickup: 0.0,
            });
            continue;
        }

        let mut unknown: Vec<usize> = Vec::new();
        for &i in &c.bus_indices {
            if buses[i].bus_type != BusType::Slack {
                reduced_pos[i] = unknown.len();
                unknown.push(i);
            }
        }
        let m = unknown.len();

        let mut solved = true;
        if m > 0 {
            let triplets = reduced_b_triplets(&branches, &island_branches[ci], &reduced_pos);

            let mut rhs = vec![0.0; m];
            for (r, &i) in unknown.iter().enumerate() {
                rhs[r] = dc_injection(&buses[i]);
            }
            for &pos in &island_branches[ci] {
                let br = &branches[pos];
                let (rf, rt) = (reduced_pos[br.from], reduced_pos[br.to]);
                // φ: the phase shift is constant, so it lives on the RHS and
                // `B` stays symmetric. See the module doc.
                let shift_p = br.b * br.shift;
                if rf != NOT_UNKNOWN {
                    rhs[rf] += shift_p;
                    // A reference's known angle also moves to the RHS:
                    // subtracting `B_ij θ_j = −b θ_j` adds `+b θ_j`.
                    if rt == NOT_UNKNOWN {
                        rhs[rf] += br.b * buses[br.to].voltage_ang;
                    }
                }
                if rt != NOT_UNKNOWN {
                    rhs[rt] -= shift_p;
                    if rf == NOT_UNKNOWN {
                        rhs[rt] += br.b * buses[br.from].voltage_ang;
                    }
                }
            }

            match RealFactorization::new(m, &triplets).and_then(|sys| sys.solve(&rhs)) {
                Some(theta) => {
                    for (r, &i) in unknown.iter().enumerate() {
                        buses[i].voltage_ang = theta[r];
                        // Only `PQ` magnitudes are normalized to the DC model's
                        // own |V| = 1 assumption. A `PV` bus's `voltage_mag` is
                        // its *setpoint* — an input the AC solver holds fixed
                        // (`jacobian` pins ΔVmag = 0 there and never updates
                        // it), not a quantity DC computed. Overwriting it would
                        // silently change the answer of any later AC solve, and
                        // DC has nothing to say about it either way.
                        if buses[i].bus_type == BusType::PQ {
                            buses[i].voltage_mag = 1.0;
                        }
                    }
                }
                None => solved = false,
            }
        }

        if !solved {
            islands.push(DcIslandReport {
                bus_indices: c.bus_indices.clone(),
                slack_indices: c.slack_indices.clone(),
                status: DcIslandStatus::Singular,
                slack_pickup: 0.0,
            });
            continue;
        }

        // Flows, and from them the two derived quantities. Deriving the slack
        // pickup and the residual from the *flows* rather than from `B θ`
        // makes them an end-to-end check: a sign error in φ that happened to
        // be consistent between assembly and solve would still show up here,
        // because these formulas go through `br.shift` a second time.
        let mut net_out = vec![0.0; n];
        for &pos in &island_branches[ci] {
            let br = &branches[pos];
            let p = br.b
                * (buses[br.from].voltage_ang - buses[br.to].voltage_ang - br.shift);
            branch_p[br.index] = p;
            net_out[br.from] += p;
            net_out[br.to] -= p;
        }

        for &i in &c.bus_indices {
            if buses[i].bus_type != BusType::Slack {
                max_residual = max_residual.max((net_out[i] - dc_injection(&buses[i])).abs());
            }
        }
        let slack_pickup: f64 = c.slack_indices.iter().map(|&s| net_out[s]).sum();

        let status = if c.slack_indices.len() > 1 {
            DcIslandStatus::SolvedMultipleReferences
        } else {
            DcIslandStatus::Solved
        };
        islands.push(DcIslandReport {
            bus_indices: c.bus_indices.clone(),
            slack_indices: c.slack_indices.clone(),
            status,
            slack_pickup,
        });
    }

    DcSolution {
        islands,
        branch_p,
        max_residual,
        negative_reactance_branches,
        ignored_branches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bus(idx: usize, bus_type: BusType, p_spec: f64) -> Bus {
        Bus {
            idx,
            bus_type,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec,
            q_spec: 0.0,
            q_min: 0.0,
            q_max: 0.0,
            u_rated: 0.0,
            zip_terms: Vec::new(),
        }
    }

    fn line(from: usize, to: usize, x: f64) -> Line {
        Line { from, to, r: 0.0, x, b_shunt: 0.0, g_shunt: 0.0 }
    }

    /// Three buses in a line, slack at 0. With no meshing the flows are fixed
    /// by continuity alone, so the angles follow in closed form:
    ///
    ///   bus 2 draws 0.5, bus 1 draws 0.3 → branch 1→2 carries 0.5,
    ///   branch 0→1 carries 0.8.
    ///   θ1 = −0.8·x01,  θ2 = θ1 − 0.5·x12.
    #[test]
    fn radial_three_bus_matches_closed_form() {
        let mut buses = vec![
            bus(0, BusType::Slack, 0.0),
            bus(1, BusType::PQ, -0.3),
            bus(2, BusType::PQ, -0.5),
        ];
        let lines = vec![line(0, 1, 0.1), line(1, 2, 0.2)];
        let sol = dc_power_flow(&mut buses, &lines, &[], DcOptions::default());

        assert_eq!(sol.islands.len(), 1);
        assert_eq!(sol.islands[0].status, DcIslandStatus::Solved);

        let theta1 = -0.8 * 0.1;
        let theta2 = theta1 - 0.5 * 0.2;
        assert!((buses[1].voltage_ang - theta1).abs() < 1e-12, "{}", buses[1].voltage_ang);
        assert!((buses[2].voltage_ang - theta2).abs() < 1e-12, "{}", buses[2].voltage_ang);
        assert!((sol.branch_p[0] - 0.8).abs() < 1e-12, "{}", sol.branch_p[0]);
        assert!((sol.branch_p[1] - 0.5).abs() < 1e-12, "{}", sol.branch_p[1]);
        assert!((sol.islands[0].slack_pickup - 0.8).abs() < 1e-12);
        assert!(sol.max_residual < 1e-12, "{}", sol.max_residual);
    }

    /// A mesh divides flow in inverse proportion to reactance. Two parallel
    /// paths from slack to bus 2, one of reactance 0.2 and one of 0.4, must
    /// split a 0.6 load as 0.4 / 0.2.
    #[test]
    fn parallel_paths_split_inversely_with_reactance() {
        let mut buses = vec![
            bus(0, BusType::Slack, 0.0),
            bus(1, BusType::PQ, 0.0),
            bus(2, BusType::PQ, -0.6),
        ];
        // Path A: 0 → 2 directly, x = 0.2. Path B: 0 → 1 → 2, x = 0.2 + 0.2.
        let lines = vec![line(0, 2, 0.2), line(0, 1, 0.2), line(1, 2, 0.2)];
        let sol = dc_power_flow(&mut buses, &lines, &[], DcOptions::default());

        assert!((sol.branch_p[0] - 0.4).abs() < 1e-12, "{}", sol.branch_p[0]);
        assert!((sol.branch_p[1] - 0.2).abs() < 1e-12, "{}", sol.branch_p[1]);
        assert!((sol.branch_p[2] - 0.2).abs() < 1e-12, "{}", sol.branch_p[2]);
        assert!(sol.max_residual < 1e-12, "{}", sol.max_residual);
    }

    fn pst(from: usize, to: usize, x: f64, alpha: f64) -> Transformer {
        Transformer {
            from,
            to,
            from_status: 1,
            to_status: 1,
            y_series: Complex::new(1.0, 0.0) / Complex::new(0.0, x),
            y_shunt: Complex::new(0.0, 0.0),
            tap: Complex::from_polar(1.0, alpha),
        }
    }

    /// A phase shifter on a *radial* branch carries no flow — there is
    /// nowhere for power to go — but it does displace the far bus's angle by
    /// exactly `−α`. That displacement is the cleanest available check that
    /// φ enters the right-hand side with the right sign.
    #[test]
    fn radial_phase_shifter_displaces_the_angle_and_carries_nothing() {
        let alpha = 0.05_f64;
        let mut buses = vec![bus(0, BusType::Slack, 0.0), bus(1, BusType::PQ, 0.0)];
        let sol = dc_power_flow(&mut buses, &[], &[pst(0, 1, 0.25, alpha)], DcOptions::default());

        assert!((buses[1].voltage_ang - (-alpha)).abs() < 1e-12, "{}", buses[1].voltage_ang);
        assert!(sol.branch_p[0].abs() < 1e-12, "{}", sol.branch_p[0]);
        assert!(sol.islands[0].slack_pickup.abs() < 1e-12);
    }

    /// A phase shifter in a *loop* drives circulating power with no load
    /// anywhere — which is the entire reason to install one, and the
    /// tightest check on the φ convention because the answer depends on its
    /// sign, not merely its magnitude.
    ///
    /// With a plain line (`b_l`) parallel to a PST (`b_t`, shift α) between
    /// the same two buses and zero injection everywhere, bus 1's balance
    /// gives `θ₁ = −b_t·α/(b_l + b_t)`, so the loop carries
    /// `±b_l·b_t·α/(b_l + b_t)` — negative through the PST, positive through
    /// the line, and nothing through the slack.
    #[test]
    fn phase_shifter_in_a_loop_drives_circulating_flow() {
        let alpha = 0.05_f64;
        let (x_line, x_pst) = (0.2_f64, 0.25_f64);
        let (b_l, b_t) = (1.0 / x_line, 1.0 / x_pst);

        let mut buses = vec![bus(0, BusType::Slack, 0.0), bus(1, BusType::PQ, 0.0)];
        let lines = vec![line(0, 1, x_line)];
        let sol =
            dc_power_flow(&mut buses, &lines, &[pst(0, 1, x_pst, alpha)], DcOptions::default());

        let circulating = b_l * b_t * alpha / (b_l + b_t);
        assert!((sol.branch_p[0] - circulating).abs() < 1e-12, "line: {}", sol.branch_p[0]);
        assert!((sol.branch_p[1] + circulating).abs() < 1e-12, "pst: {}", sol.branch_p[1]);
        assert!(
            (buses[1].voltage_ang - (-b_t * alpha / (b_l + b_t))).abs() < 1e-12,
            "{}",
            buses[1].voltage_ang
        );
        // No load and no losses, so the reference supplies nothing.
        assert!(sol.islands[0].slack_pickup.abs() < 1e-12, "{}", sol.islands[0].slack_pickup);
        assert!(sol.max_residual < 1e-12);
    }

    /// The off-nominal ratio divides `b` once, not twice — the `1/k` vs
    /// `1/k²` distinction, checked against a hand-computed value.
    #[test]
    fn transformer_ratio_divides_susceptance_once() {
        let x = 0.1_f64;
        let k = 1.05_f64;
        let t = Transformer {
            from: 0,
            to: 1,
            from_status: 1,
            to_status: 1,
            y_series: Complex::new(1.0, 0.0) / Complex::new(0.0, x),
            y_shunt: Complex::new(0.0, 0.0),
            tap: Complex::from_polar(k, 0.0),
        };
        let got = dc_branches(&[], std::slice::from_ref(&t), DcOptions::default());
        assert!((got[0].b - 1.0 / (x * k)).abs() < 1e-12, "{}", got[0].b);

        let no_ratio = DcOptions { use_transformer_ratio: false, ..Default::default() };
        let got = dc_branches(&[], std::slice::from_ref(&t), no_ratio);
        assert!((got[0].b - 1.0 / x).abs() < 1e-12, "{}", got[0].b);
    }

    /// `IgnoreG` keeps resistance in the denominator, and equals
    /// `−Im(y_series)` exactly. On a branch with `r = x` the two
    /// approximations differ by a factor of two, which is the point of
    /// offering the choice.
    #[test]
    fn ignore_g_uses_the_true_series_susceptance() {
        let ln = Line { from: 0, to: 1, r: 0.1, x: 0.1, b_shunt: 0.0, g_shunt: 0.0 };
        let opts = DcOptions { approximation: DcApproximation::IgnoreG, ..Default::default() };
        let got = dc_branches(std::slice::from_ref(&ln), &[], opts);
        let y_series = Complex::new(1.0, 0.0) / Complex::new(ln.r, ln.x);
        assert!((got[0].b - (-y_series.im)).abs() < 1e-12, "{}", got[0].b);
        assert!((got[0].b - 5.0).abs() < 1e-12, "{}", got[0].b);

        let got = dc_branches(std::slice::from_ref(&ln), &[], DcOptions::default());
        assert!((got[0].b - 10.0).abs() < 1e-12, "{}", got[0].b);
    }

    /// Branches with no series path contribute nothing and are reported.
    #[test]
    fn self_loops_and_open_terminals_are_ignored() {
        let lines = vec![line(0, 1, 0.1), line(1, 1, 0.2)];
        let transformers = vec![Transformer {
            from: 0,
            to: 1,
            from_status: 1,
            to_status: 0,
            y_series: Complex::new(1.0, 0.0) / Complex::new(0.0, 0.1),
            y_shunt: Complex::new(0.0, 0.0),
            tap: Complex::new(1.0, 0.0),
        }];
        let got = dc_branches(&lines, &transformers, DcOptions::default());
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].index, 0);

        let mut buses = vec![bus(0, BusType::Slack, 0.0), bus(1, BusType::PQ, -0.4)];
        let sol = dc_power_flow(&mut buses, &lines, &transformers, DcOptions::default());
        assert_eq!(sol.ignored_branches, vec![1, 2]);
        assert_eq!(sol.branch_p[1], 0.0);
        assert_eq!(sol.branch_p[2], 0.0);
    }

    /// A component with no slack gets no fabricated reference, matching the
    /// AC path — and its neighbours still solve.
    #[test]
    fn sourceless_island_is_reported_not_invented() {
        let mut buses = vec![
            bus(0, BusType::Slack, 0.0),
            bus(1, BusType::PQ, -0.4),
            bus(2, BusType::PQ, -0.1),
            bus(3, BusType::PQ, 0.1),
        ];
        let lines = vec![line(0, 1, 0.1), line(2, 3, 0.2)];
        let sol = dc_power_flow(&mut buses, &lines, &[], DcOptions::default());

        assert_eq!(sol.islands.len(), 2);
        assert_eq!(sol.islands[0].status, DcIslandStatus::Solved);
        assert_eq!(sol.islands[1].status, DcIslandStatus::NoReferenceBus);
        assert_eq!(sol.islands[1].bus_indices, vec![2, 3]);
        assert_eq!(buses[2].voltage_ang, 0.0);
        assert_eq!(sol.branch_p[1], 0.0);
    }

    /// Two slacks in one component is over-determined for AC but well posed
    /// for DC: the load's injection is still met exactly, and only the split
    /// of pickup between the two references is arbitrary.
    #[test]
    fn two_references_in_one_island_still_solve() {
        let mut buses = vec![
            bus(0, BusType::Slack, 0.0),
            bus(1, BusType::PQ, -0.6),
            bus(2, BusType::Slack, 0.0),
        ];
        let lines = vec![line(0, 1, 0.2), line(1, 2, 0.2)];
        let sol = dc_power_flow(&mut buses, &lines, &[], DcOptions::default());

        assert_eq!(sol.islands[0].status, DcIslandStatus::SolvedMultipleReferences);
        assert!(sol.max_residual < 1e-12, "{}", sol.max_residual);
        assert!((sol.islands[0].slack_pickup - 0.6).abs() < 1e-12);
        // Equal reactances and equal reference angles ⇒ an even split here,
        // but that is this fixture's symmetry, not a guarantee of the method.
        assert!((sol.branch_p[0] - 0.3).abs() < 1e-12, "{}", sol.branch_p[0]);
    }

    /// A slack whose angle is not zero is a valid reference, and every other
    /// angle must come out shifted by exactly that offset.
    #[test]
    fn nonzero_slack_angle_offsets_the_whole_island() {
        let offset = 0.2_f64;
        let lines = vec![line(0, 1, 0.1), line(1, 2, 0.2)];

        let mut base = vec![
            bus(0, BusType::Slack, 0.0),
            bus(1, BusType::PQ, -0.3),
            bus(2, BusType::PQ, -0.5),
        ];
        dc_power_flow(&mut base, &lines, &[], DcOptions::default());

        let mut shifted = vec![
            bus(0, BusType::Slack, 0.0),
            bus(1, BusType::PQ, -0.3),
            bus(2, BusType::PQ, -0.5),
        ];
        shifted[0].voltage_ang = offset;
        let sol = dc_power_flow(&mut shifted, &lines, &[], DcOptions::default());

        for i in 1..3 {
            assert!(
                (shifted[i].voltage_ang - (base[i].voltage_ang + offset)).abs() < 1e-12,
                "bus {i}: {} vs {}",
                shifted[i].voltage_ang,
                base[i].voltage_ang
            );
        }
        assert!(sol.max_residual < 1e-12);
    }

    /// A PGM `link` inverts to a *negative* reactance; DC must not let that
    /// through as a negative susceptance. See the guard in `dc_branches`.
    #[test]
    fn link_admittance_yields_a_positive_susceptance() {
        let t = Transformer {
            from: 0,
            to: 1,
            from_status: 1,
            to_status: 1,
            y_series: IDEAL_CONNECTION_Y,
            y_shunt: Complex::new(0.0, 0.0),
            tap: Complex::new(1.0, 0.0),
        };
        // The hazard, spelled out: the naive route really does go negative.
        let z = Complex::new(1.0, 0.0) / IDEAL_CONNECTION_Y;
        assert!(z.im < 0.0, "fixture assumption broken: link reactance is {}", z.im);

        let got = dc_branches(&[], std::slice::from_ref(&t), DcOptions::default());
        assert!(got[0].b > 0.0, "link susceptance came out {}", got[0].b);
        assert!((got[0].b - 1.0 / ideal_connection_z()).abs() < 1e-6, "{}", got[0].b);
    }

    /// A series capacitor is legitimate and must pass through with its sign
    /// intact — the link guard must not be a blanket `x < 0` clamp.
    #[test]
    fn series_capacitor_keeps_its_negative_susceptance() {
        let lines = vec![line(0, 1, 0.2), line(0, 1, -0.1)];
        let mut buses = vec![bus(0, BusType::Slack, 0.0), bus(1, BusType::PQ, -0.5)];
        let sol = dc_power_flow(&mut buses, &lines, &[], DcOptions::default());

        let got = dc_branches(&lines, &[], DcOptions::default());
        assert!((got[1].b - (-10.0)).abs() < 1e-12, "{}", got[1].b);
        assert_eq!(sol.negative_reactance_branches, vec![1]);
    }

    /// A `PV` bus's `voltage_mag` is a setpoint the AC solver holds fixed, not
    /// a result. DC must leave it alone: overwriting it with the model's
    /// |V| = 1 assumption silently changes the answer of any later AC solve
    /// seeded from this state, and reports `Converged` while doing it.
    #[test]
    fn a_pv_bus_keeps_its_voltage_setpoint() {
        let mut buses = vec![
            bus(0, BusType::Slack, 0.0),
            bus(1, BusType::PV, 0.2),
            bus(2, BusType::PQ, -0.5),
        ];
        buses[0].voltage_mag = 1.06;
        buses[1].voltage_mag = 1.04;
        let lines = vec![line(0, 1, 0.1), line(1, 2, 0.2), line(0, 2, 0.3)];

        dc_power_flow(&mut buses, &lines, &[], DcOptions::default());

        assert_eq!(buses[0].voltage_mag, 1.06, "slack setpoint was overwritten");
        assert_eq!(buses[1].voltage_mag, 1.04, "PV setpoint was overwritten");
        // The PQ bus has no specified magnitude, so it takes the DC assumption.
        assert_eq!(buses[2].voltage_mag, 1.0);
        // A PV bus is still a solved unknown in *angle*, which is the whole
        // point of including it.
        assert!(buses[1].voltage_ang != 0.0);
    }

    /// Components must come back in the same order and shape
    /// `network::connected_components` promises: members ascending,
    /// components ordered by lowest member.
    #[test]
    fn components_match_the_documented_ordering_contract() {
        // Deliberately wire the higher-numbered buses together first.
        let branches = vec![
            DcBranch { index: 0, from: 3, to: 1, b: 1.0, shift: 0.0 },
            DcBranch { index: 1, from: 4, to: 2, b: 1.0, shift: 0.0 },
        ];
        assert_eq!(dc_components(5, &branches), vec![vec![0], vec![1, 3], vec![2, 4]]);
    }
}
