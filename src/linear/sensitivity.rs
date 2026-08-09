//! DC sensitivity factors: PTDF and LODF.
//!
//! Because the DC model is exactly linear, the derivative of a branch flow
//! with respect to a bus injection is a constant — not a local slope that
//! drifts as the operating point moves, but the true, global answer. That is
//! what makes these factors worth precomputing, and it is why they exist for
//! DC and not for AC.
//!
//! **PTDF** (power transfer distribution factors) answers *if I inject one
//! more unit at bus j and let the reference absorb it, how much of it shows up
//! on branch k?* **LODF** (line outage distribution factors) answers *if
//! branch l trips, what fraction of the power it was carrying lands on branch
//! k?* — the workhorse of contingency screening, since it turns "re-solve the
//! network once per outage" into one matrix column.
//!
//! # The formulation
//!
//! With `N` the non-reference buses of one island, `B_NN` its reduced
//! susceptance matrix, and `Bf` the branch-flow matrix
//! (`Bf[k, from_k] = +b_k`, `Bf[k, to_k] = −b_k`):
//!
//! \\[ \mathrm{PTDF}[:, j] = B_f\,\theta^{(j)}, \qquad B_{NN}\theta^{(j)} = e_j \\]
//! \\[ \mathrm{PTDF}[k, :] = x, \qquad B_{NN}\,x = B_f[k, N]^\mathsf{T} \\]
//!
//! The second line uses `B_NN = B_NNᵀ` — true even with phase shifters, since
//! the shift lives on the right-hand side — so a *row* costs one solve rather
//! than one per bus.
//!
//! \\[ \mathrm{LODF}[:, l] = \frac{\mathrm{PTDF}[:, from_l] - \mathrm{PTDF}[:, to_l]}{1 - d_l},
//!    \qquad d_l = \mathrm{PTDF}[l, from_l] - \mathrm{PTDF}[l, to_l] \\]
//!
//! The numerator is one solve against `e_from − e_to`, not two, and it yields
//! `d_l` as a by-product — so a LODF column costs exactly what a PTDF column
//! does.
//!
//! # Why columns and rows, not matrices
//!
//! The dense factors are enormous. On `case9241pegase` (9,241 buses, 16,049
//! branches) a full PTDF is 1.19 GB and a full LODF 2.06 GB. Both are offered
//! ([`DcSensitivity::ptdf_dense`], [`DcSensitivity::lodf_dense`]) because small
//! networks and test code genuinely want them, but the column and row accessors
//! are the API to reach for, and each costs a single triangular solve against a
//! factorization computed once in [`DcSensitivity::new`].
//!
//! # Scope
//!
//! Distributed slack is not supported: each island has one reference, or
//! several whose pickup split is arbitrary. lightsim2grid carries the same
//! limitation, as a `TODO PTDF: distributed slack` in the equivalent place.

use crate::sparse::RealFactorization;
use crate::types::{Bus, BusType};

use super::btheta::{partition, reduced_b_triplets, DcBranch, Partition, NOT_UNKNOWN};
use crate::network::Verdict;

/// Below this, `1 − d_l` is treated as zero and the branch is called radial.
///
/// `d_l → 1` exactly when branch `l` is a bridge: removing it disconnects the
/// network, so its power has nowhere to redistribute to and no finite
/// redistribution factor exists. This is a structural fact, not a numerical
/// accident, which is why the accessors return `None` rather than a large
/// number.
const LODF_BRIDGE_TOL: f64 = 1e-9;

/// A minimal row-major dense matrix.
///
/// Deliberately not `faer::Mat`: `sparse.rs`'s module doc commits to keeping
/// the linear-algebra backend out of gridoxide's public signatures, and a
/// dense return type is exactly where that would leak.
#[derive(Clone, Debug)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Row-major, `rows * cols` entries.
    pub data: Vec<f64>,
}

impl DenseMatrix {
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    pub fn row(&self, row: usize) -> &[f64] {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }
}

/// One island's factorized reduced system, plus the bookkeeping to map
/// between global bus indices and its own rows.
struct Island {
    factorization: RealFactorization,
    /// Reduced row -> global bus index.
    unknown: Vec<usize>,
    /// Positions into `DcSensitivity::branches` for this island's branches.
    branch_positions: Vec<usize>,
}

/// Precomputed DC sensitivity factors for one network.
///
/// Holds one factorization per island, computed once. Every accessor below is
/// a solve against those, never a refactorization — which is the entire reason
/// [`sparse::RealFactorization`](crate::sparse::RealFactorization) exists
/// rather than reusing the Newton path's
/// [`solver::LinearSolver`](crate::solver::LinearSolver).
pub struct DcSensitivity {
    n_buses: usize,
    /// Total flat branch count (lines + transformers), the index space every
    /// returned vector uses.
    n_branches: usize,
    branches: Vec<DcBranch>,
    islands: Vec<Island>,
    /// Bus -> index into `islands`, or `NOT_UNKNOWN` when its island has no
    /// factorization — either because it has no reference bus, or because it
    /// has no unknowns to solve for. [`referenced`](Self::referenced) tells
    /// those two apart.
    island_of: Vec<usize>,
    /// Bus -> whether its island has at least one `Slack` bus. Tracked
    /// separately from `island_of` so that a lone reference bus — an island
    /// with a reference but nothing to solve — is not mistaken for an
    /// unreferenced one, which is the distinction `ptdf_column`'s `None`
    /// documents.
    referenced: Vec<bool>,
    /// Bus -> row within its island's reduced system, or `NOT_UNKNOWN` for a
    /// reference bus.
    reduced_pos: Vec<usize>,
    /// Flat branch index -> position in `branches`, or `NOT_UNKNOWN` for a
    /// branch that carries no DC flow.
    ///
    /// A map rather than a linear scan because `ptdf_row`, `lodf_column` and
    /// `is_radial` are each called once per branch by the dense builders and
    /// by `examples/bench_dc.rs`'s sweeps; scanning would make those O(m²) —
    /// ~2.6e8 comparisons at case9241pegase, in a module whose whole purpose
    /// is not paying costs of that shape.
    branch_position: Vec<usize>,
}

impl DcSensitivity {
    /// Factorizes every referenced island's reduced `B`.
    ///
    /// `branches` must come from
    /// [`dc_branches`](super::btheta::dc_branches) over the same network, and
    /// `n_branches` must be its flat branch count (`lines.len() +
    /// transformers.len()`) — the same index space
    /// [`DcSolution::branch_p`](super::btheta::DcSolution::branch_p) uses.
    ///
    /// Returns `None` if any referenced island's `B` is singular. Islands with
    /// no reference bus are not an error: they simply have no sensitivities,
    /// and every accessor returns `None` for their buses and branches.
    pub fn new(buses: &[Bus], branches: &[DcBranch], n_branches: usize) -> Option<Self> {
        let n = buses.len();
        let Partition { classified, island_branches } = partition(buses, branches);

        let mut islands = Vec::new();
        let mut island_of = vec![NOT_UNKNOWN; n];
        let mut referenced = vec![false; n];
        let mut reduced_pos = vec![NOT_UNKNOWN; n];

        let mut branch_position = vec![NOT_UNKNOWN; n_branches];
        for (pos, br) in branches.iter().enumerate() {
            branch_position[br.index] = pos;
        }

        for (ci, c) in classified.iter().enumerate() {
            if matches!(c.verdict, Verdict::NoReferenceBus) {
                continue;
            }
            for &i in &c.bus_indices {
                referenced[i] = true;
            }

            let mut unknown = Vec::new();
            for &i in &c.bus_indices {
                if buses[i].bus_type != BusType::Slack {
                    reduced_pos[i] = unknown.len();
                    unknown.push(i);
                }
            }
            let m = unknown.len();
            if m == 0 {
                // Every bus here is a reference, so there is nothing to solve
                // and nothing that could respond. Such an island gets no
                // `Island` entry, but `referenced` still marks it, so its
                // buses return an all-zero column rather than `None`.
                continue;
            }

            let triplets = reduced_b_triplets(branches, &island_branches[ci], &reduced_pos);
            let factorization = RealFactorization::new(m, &triplets)?;

            let idx = islands.len();
            for &i in &c.bus_indices {
                island_of[i] = idx;
            }
            islands.push(Island {
                factorization,
                unknown,
                branch_positions: island_branches[ci].clone(),
            });
        }

        Some(Self {
            n_buses: n,
            n_branches,
            branches: branches.to_vec(),
            islands,
            island_of,
            referenced,
            reduced_pos,
            branch_position,
        })
    }

    pub fn n_buses(&self) -> usize {
        self.n_buses
    }

    pub fn n_branches(&self) -> usize {
        self.n_branches
    }

    /// Turns one island's solved reduced angles into per-branch flow responses
    /// over the full flat branch index space.
    ///
    /// The phase shift is deliberately absent: these are *responses* to a
    /// change, and a constant term contributes nothing to a derivative.
    fn flows_from_angles(&self, island: &Island, theta: &[f64]) -> Vec<f64> {
        let angle = |bus: usize| -> f64 {
            match self.reduced_pos[bus] {
                NOT_UNKNOWN => 0.0,
                r => theta[r],
            }
        };
        let mut out = vec![0.0; self.n_branches];
        for &pos in &island.branch_positions {
            let br = &self.branches[pos];
            out[br.index] = br.b * (angle(br.from) - angle(br.to));
        }
        out
    }

    /// Branch-flow response to an arbitrary injection pattern — the primitive
    /// the other accessors are special cases of.
    ///
    /// `injections` is per-bus and per-unit; within each island whatever does
    /// not sum to zero is absorbed at that island's reference. Returns per-
    /// branch flow *changes*, in the flat branch index space.
    ///
    /// Costs one solve per island that the pattern actually touches.
    pub fn transfer_factors(&self, injections: &[f64]) -> Option<Vec<f64>> {
        self.response(injections).map(|(_, flows)| flows)
    }

    /// Both halves of the response to an injection pattern: the change in
    /// every bus angle and the change in every branch flow.
    ///
    /// Returns `(d_angle, d_flow)`, indexed by bus and by flat branch index.
    /// This is what [`linear::batch`](super::batch) builds a whole batch on:
    /// because DC is exactly linear, a scenario's answer is the base solve
    /// plus this response to the scenario's injection *delta* — one solve
    /// against a factorization computed once, rather than a fresh solve per
    /// scenario.
    ///
    /// Islands the pattern does not touch are skipped without a solve, and
    /// contribute nothing.
    pub fn response(&self, injections: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        if injections.len() != self.n_buses {
            return None;
        }
        let mut d_angle = vec![0.0; self.n_buses];
        let mut d_flow = vec![0.0; self.n_branches];
        for island in &self.islands {
            let rhs: Vec<f64> = island.unknown.iter().map(|&i| injections[i]).collect();
            if rhs.iter().all(|v| *v == 0.0) {
                continue;
            }
            let theta = island.factorization.solve(&rhs)?;
            for (dst, src) in d_flow.iter_mut().zip(self.flows_from_angles(island, &theta)) {
                *dst += src;
            }
            for (r, &bus) in island.unknown.iter().enumerate() {
                d_angle[bus] += theta[r];
            }
        }
        Some((d_angle, d_flow))
    }

    /// Branch flows after `branch` trips, given the flows before it did.
    ///
    /// This is the N-1 screening primitive: `f + LODF[:, l]·f[l]`, one solve
    /// against the cached factorization and no re-solve of the network. The
    /// outaged branch itself comes back at exactly zero.
    ///
    /// `None` if `branch` is radial — removing it islands the network, so its
    /// power has nowhere to redistribute to and no post-outage flow exists —
    /// or if it is out of range or in an unreferenced island. Use
    /// [`is_radial`](Self::is_radial) to tell those apart in advance.
    ///
    /// `base_flows` must be indexed by flat branch index, as
    /// [`DcSolution::branch_p`](super::btheta::DcSolution::branch_p) is.
    pub fn outage_flows(&self, base_flows: &[f64], branch: usize) -> Option<Vec<f64>> {
        if base_flows.len() != self.n_branches {
            return None;
        }
        let column = self.lodf_column(branch)?;
        let lost = base_flows[branch];
        let mut out: Vec<f64> = base_flows
            .iter()
            .zip(&column)
            .map(|(f, factor)| f + factor * lost)
            .collect();
        // `column[branch]` is exactly -1, so this is already zero up to
        // round-off; setting it makes that exact, since "the branch that
        // tripped carries nothing" is a fact rather than a computed value.
        out[branch] = 0.0;
        Some(out)
    }

    /// `∂P_branch/∂P_bus` over every branch, for injection at one bus.
    ///
    /// `None` if `bus` is out of range or sits in an island with no reference.
    /// A reference bus's own column is all zeros — injecting at the slack and
    /// letting the slack absorb it moves nothing.
    pub fn ptdf_column(&self, bus: usize) -> Option<Vec<f64>> {
        // Out of range, or an island with no reference: no factors exist.
        if !self.referenced.get(bus).copied().unwrap_or(false) {
            return None;
        }
        let r = self.reduced_pos[bus];
        // A reference bus, or a bus in an island with nothing to solve for.
        // Injecting at the slack and letting the slack absorb it moves nothing.
        if r == NOT_UNKNOWN || self.island_of[bus] == NOT_UNKNOWN {
            return Some(vec![0.0; self.n_branches]);
        }
        let island = &self.islands[self.island_of[bus]];
        let mut rhs = vec![0.0; island.unknown.len()];
        rhs[r] = 1.0;
        let theta = island.factorization.solve(&rhs)?;
        Some(self.flows_from_angles(island, &theta))
    }

    /// Resolves a flat branch index to its `DcBranch` and its island's
    /// factorization, or `None` if it carries no DC flow or has no
    /// factorization to solve against.
    fn branch_and_island(&self, branch: usize) -> Option<(&DcBranch, &Island)> {
        let pos = *self.branch_position.get(branch)?;
        if pos == NOT_UNKNOWN {
            return None;
        }
        let br = &self.branches[pos];
        let island_idx = self.island_of[br.from];
        if island_idx == NOT_UNKNOWN {
            return None;
        }
        Some((br, &self.islands[island_idx]))
    }

    /// `∂P_branch/∂P_bus` over every bus, for one branch.
    ///
    /// One solve, not `n_buses` of them, because `B_NN` is symmetric.
    /// `None` if `branch` is out of range, carries no DC flow, or sits in an
    /// island with no reference.
    pub fn ptdf_row(&self, branch: usize) -> Option<Vec<f64>> {
        let (br, island) = self.branch_and_island(branch)?;

        let mut rhs = vec![0.0; island.unknown.len()];
        if self.reduced_pos[br.from] != NOT_UNKNOWN {
            rhs[self.reduced_pos[br.from]] += br.b;
        }
        if self.reduced_pos[br.to] != NOT_UNKNOWN {
            rhs[self.reduced_pos[br.to]] -= br.b;
        }
        let x = island.factorization.solve(&rhs)?;

        let mut out = vec![0.0; self.n_buses];
        for (r, &bus) in island.unknown.iter().enumerate() {
            out[bus] = x[r];
        }
        Some(out)
    }

    /// Solves `e_from − e_to` for `branch`, returning the resulting per-branch
    /// flows and `d_l`. Shared by [`lodf_column`](Self::lodf_column) and
    /// [`is_radial`](Self::is_radial).
    fn outage_response(&self, branch: usize) -> Option<(Vec<f64>, f64)> {
        let (br, island) = self.branch_and_island(branch)?;

        let mut rhs = vec![0.0; island.unknown.len()];
        if self.reduced_pos[br.from] != NOT_UNKNOWN {
            rhs[self.reduced_pos[br.from]] += 1.0;
        }
        if self.reduced_pos[br.to] != NOT_UNKNOWN {
            rhs[self.reduced_pos[br.to]] -= 1.0;
        }
        let theta = island.factorization.solve(&rhs)?;
        let flows = self.flows_from_angles(island, &theta);
        let d = flows[branch];
        Some((flows, d))
    }

    /// The fraction of `branch`'s pre-outage flow that lands on each other
    /// branch when it trips.
    ///
    /// `None` if `branch` is radial (see [`is_radial`](Self::is_radial)), out
    /// of range, or in an unreferenced island. The outaged branch's own entry
    /// is `−1`: it loses all of its flow, by definition.
    pub fn lodf_column(&self, branch: usize) -> Option<Vec<f64>> {
        let (flows, d) = self.outage_response(branch)?;
        if (1.0 - d).abs() < LODF_BRIDGE_TOL {
            return None;
        }
        let scale = 1.0 / (1.0 - d);
        let mut out: Vec<f64> = flows.iter().map(|f| f * scale).collect();
        out[branch] = -1.0;
        Some(out)
    }

    /// Whether removing `branch` would disconnect the network, leaving its
    /// flow nowhere to redistribute to and its LODF column undefined.
    ///
    /// `true` for a branch that is out of range or in an unreferenced island:
    /// there are no redistribution factors for it either way.
    pub fn is_radial(&self, branch: usize) -> bool {
        match self.outage_response(branch) {
            Some((_, d)) => (1.0 - d).abs() < LODF_BRIDGE_TOL,
            None => true,
        }
    }

    /// The full `n_branches × n_buses` PTDF.
    ///
    /// Allocates `n_branches · n_buses · 8` bytes — **1.19 GB** on
    /// `case9241pegase`. Prefer [`ptdf_column`](Self::ptdf_column) or
    /// [`ptdf_row`](Self::ptdf_row) unless the whole matrix is genuinely
    /// wanted.
    pub fn ptdf_dense(&self) -> Option<DenseMatrix> {
        let mut data = vec![0.0; self.n_branches * self.n_buses];
        for bus in 0..self.n_buses {
            let Some(column) = self.ptdf_column(bus) else { continue };
            for (k, v) in column.into_iter().enumerate() {
                data[k * self.n_buses + bus] = v;
            }
        }
        Some(DenseMatrix { rows: self.n_branches, cols: self.n_buses, data })
    }

    /// The full `n_branches × n_branches` LODF.
    ///
    /// Allocates `n_branches² · 8` bytes — **2.06 GB** on `case9241pegase`.
    /// Radial branches' columns are left at zero, since they have no defined
    /// factors; use [`is_radial`](Self::is_radial) to tell those apart from
    /// branches that genuinely redistribute nothing.
    pub fn lodf_dense(&self) -> Option<DenseMatrix> {
        let mut data = vec![0.0; self.n_branches * self.n_branches];
        for l in 0..self.n_branches {
            let Some(column) = self.lodf_column(l) else { continue };
            for (k, v) in column.into_iter().enumerate() {
                data[k * self.n_branches + l] = v;
            }
        }
        Some(DenseMatrix { rows: self.n_branches, cols: self.n_branches, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::{dc_branches, DcOptions};
    use crate::types::Line;

    fn bus(idx: usize, bus_type: BusType) -> Bus {
        Bus {
            idx,
            bus_type,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.0,
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

    /// Three buses in a triangle, all reactances equal, slack at 0. Injecting
    /// at bus 1 and absorbing at bus 0 splits 2:1 between the direct path
    /// (0–1) and the long way round (1–2–0), by symmetry — the textbook
    /// answer, and one that needs no solver to know.
    fn triangle() -> (Vec<Bus>, Vec<Line>) {
        (
            vec![bus(0, BusType::Slack), bus(1, BusType::PQ), bus(2, BusType::PQ)],
            vec![line(0, 1, 0.1), line(1, 2, 0.1), line(0, 2, 0.1)],
        )
    }

    fn sensitivity(buses: &[Bus], lines: &[Line]) -> DcSensitivity {
        let branches = dc_branches(lines, &[], DcOptions::default());
        DcSensitivity::new(buses, &branches, lines.len()).unwrap()
    }

    #[test]
    fn ptdf_column_matches_the_symmetric_two_thirds_split() {
        let (buses, lines) = triangle();
        let s = sensitivity(&buses, &lines);
        let col = s.ptdf_column(1).unwrap();

        // Signs follow each branch's own from→to orientation, so power moving
        // toward the slack reads negative on a branch that points away from it:
        //   branch 0 is 0→1, carrying 2/3 the "wrong" way   → −2/3
        //   branch 1 is 1→2, carrying 1/3 the right way     → +1/3
        //   branch 2 is 0→2, carrying that same 1/3 back    → −1/3
        assert!((col[0] + 2.0 / 3.0).abs() < 1e-12, "{col:?}");
        assert!((col[1] - 1.0 / 3.0).abs() < 1e-12, "{col:?}");
        assert!((col[2] + 1.0 / 3.0).abs() < 1e-12, "{col:?}");
    }

    /// The reference's own column is zero, and `ptdf_row` must reproduce
    /// `ptdf_column` transposed — the check that the symmetry shortcut in
    /// `ptdf_row` is sound.
    #[test]
    fn ptdf_row_is_the_transpose_of_the_columns() {
        let (buses, lines) = triangle();
        let s = sensitivity(&buses, &lines);

        assert!(s.ptdf_column(0).unwrap().iter().all(|v| *v == 0.0));

        let columns: Vec<Vec<f64>> = (0..3).map(|b| s.ptdf_column(b).unwrap()).collect();
        for branch in 0..3 {
            let row = s.ptdf_row(branch).unwrap();
            for bus in 0..3 {
                assert!(
                    (row[bus] - columns[bus][branch]).abs() < 1e-12,
                    "branch {branch} bus {bus}: row {} vs column {}",
                    row[bus],
                    columns[bus][branch]
                );
            }
        }
    }

    /// Outaging one edge of a symmetric triangle pushes all of its flow onto
    /// the remaining path, so both surviving branches take exactly 1.
    #[test]
    fn lodf_of_a_symmetric_triangle_is_unity() {
        let (buses, lines) = triangle();
        let s = sensitivity(&buses, &lines);
        let col = s.lodf_column(0).unwrap();

        assert_eq!(col[0], -1.0);
        assert!((col[1].abs() - 1.0).abs() < 1e-12, "{col:?}");
        assert!((col[2].abs() - 1.0).abs() < 1e-12, "{col:?}");
        assert!(!s.is_radial(0));
    }

    /// A bridge has no redistribution factors, because removing it islands
    /// the network rather than rerouting anything.
    #[test]
    fn a_radial_branch_has_no_lodf_column() {
        let buses = vec![bus(0, BusType::Slack), bus(1, BusType::PQ), bus(2, BusType::PQ)];
        let lines = vec![line(0, 1, 0.1), line(1, 2, 0.2)];
        let s = sensitivity(&buses, &lines);

        for branch in 0..2 {
            assert!(s.is_radial(branch), "branch {branch} should be radial");
            assert!(s.lodf_column(branch).is_none(), "branch {branch} should have no LODF");
        }
        // PTDF is still perfectly well defined on a radial network.
        let col = s.ptdf_column(2).unwrap();
        assert!((col[0] + 1.0).abs() < 1e-12 && (col[1] + 1.0).abs() < 1e-12, "{col:?}");
    }

    /// `transfer_factors` must agree with a weighted sum of PTDF columns,
    /// which is the linearity the whole module rests on.
    #[test]
    fn transfer_factors_are_a_linear_combination_of_ptdf_columns() {
        let (buses, lines) = triangle();
        let s = sensitivity(&buses, &lines);

        // Inject 0.3 at bus 1 and 0.7 at bus 2; the reference absorbs both.
        let got = s.transfer_factors(&[0.0, 0.3, 0.7]).unwrap();
        let c1 = s.ptdf_column(1).unwrap();
        let c2 = s.ptdf_column(2).unwrap();
        for k in 0..3 {
            let expected = 0.3 * c1[k] + 0.7 * c2[k];
            assert!((got[k] - expected).abs() < 1e-12, "branch {k}: {} vs {expected}", got[k]);
        }
        assert!(s.transfer_factors(&[0.0, 0.0]).is_none(), "length mismatch must be rejected");
    }

    #[test]
    fn dense_matrices_agree_with_the_accessors() {
        let (buses, lines) = triangle();
        let s = sensitivity(&buses, &lines);

        let ptdf = s.ptdf_dense().unwrap();
        assert_eq!((ptdf.rows, ptdf.cols), (3, 3));
        for bus in 0..3 {
            let col = s.ptdf_column(bus).unwrap();
            for branch in 0..3 {
                assert_eq!(ptdf.get(branch, bus), col[branch]);
            }
        }

        let lodf = s.lodf_dense().unwrap();
        for l in 0..3 {
            let col = s.lodf_column(l).unwrap();
            for k in 0..3 {
                assert_eq!(lodf.get(k, l), col[k]);
            }
        }
    }

    /// Buses in an island with no reference have no sensitivities, and say so
    /// rather than returning zeros that look like an answer.
    #[test]
    fn an_unreferenced_island_has_no_factors() {
        let buses = vec![
            bus(0, BusType::Slack),
            bus(1, BusType::PQ),
            bus(2, BusType::PQ),
            bus(3, BusType::PQ),
        ];
        let lines = vec![line(0, 1, 0.1), line(2, 3, 0.2)];
        let s = sensitivity(&buses, &lines);

        assert!(s.ptdf_column(1).is_some());
        assert!(s.ptdf_column(2).is_none());
        assert!(s.ptdf_column(3).is_none());
        assert!(s.ptdf_row(1).is_none(), "the unreferenced island's branch has no row");
        assert!(s.is_radial(1));
    }

    /// `None` from `ptdf_column` means "no reference bus", per its own doc.
    /// A lone reference bus is a different thing — it has a reference and
    /// simply nothing that could respond — so it must return zeros. Confusing
    /// the two would make `None` useless for detecting unreferenced islands.
    #[test]
    fn a_lone_reference_bus_has_a_zero_column_not_a_missing_one() {
        let buses = vec![bus(0, BusType::Slack), bus(1, BusType::Slack), bus(2, BusType::PQ)];
        // Bus 0 is an island unto itself; buses 1 and 2 are the other island.
        let lines = vec![line(1, 2, 0.1)];
        let s = sensitivity(&buses, &lines);

        let column = s.ptdf_column(0).expect("a referenced island must not report None");
        assert!(column.iter().all(|v| *v == 0.0), "{column:?}");
        assert!(s.ptdf_column(2).is_some());
        // Out of range still means None.
        assert!(s.ptdf_column(3).is_none());
    }

    /// Branch lookup must be a map, not a scan: the dense builders call these
    /// once per branch, so a scan would make them quadratic. Checked by
    /// behavior — every valid index resolves, and gaps left by branches that
    /// carry no DC flow resolve to `None` rather than to a neighbour.
    #[test]
    fn branch_lookup_handles_gaps_in_the_flat_index() {
        let buses = vec![bus(0, BusType::Slack), bus(1, BusType::PQ), bus(2, BusType::PQ)];
        // Branch 1 is a self-loop: gridoxide's half-open representation, which
        // `dc_branches` skips, leaving a hole in the flat index space.
        let lines = vec![line(0, 1, 0.1), line(1, 1, 0.2), line(1, 2, 0.3), line(0, 2, 0.4)];
        let s = sensitivity(&buses, &lines);

        assert!(s.ptdf_row(0).is_some());
        assert!(s.ptdf_row(1).is_none(), "the skipped self-loop must not resolve");
        assert!(s.ptdf_row(2).is_some());
        assert!(s.ptdf_row(3).is_some());
        assert!(s.ptdf_row(4).is_none(), "out of range must not resolve");
        assert!(s.is_radial(1), "a branch carrying no DC flow has no factors");
    }
}
