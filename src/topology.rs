//! Topological reduction: collapsing buses tied together by zero-impedance
//! connections.
//!
//! This is approach 1 of the three surveyed in
//! `docs/src/powerflow/zero_impedance_branches.md` — merge before you
//! formulate, so the admittance matrix never sees an infinite entry. It lives
//! here rather than in an importer because both importers can reach a network
//! needing it, and because the graph pass is the same regardless of what named
//! the connection.
//!
//! # When merging is the right answer
//!
//! Merging is exact and free — no new equations, no new unknowns, no numerical
//! stiffness — but it *destroys identity*: once two buses are one, nothing
//! downstream can report a flow through the connection or an injection at
//! either end separately.
//!
//! So the policy is about identity, not about impedance:
//!
//! - A **CGMES closed switch** has no identity in the bus/branch view — that
//!   view is *defined* as the merged one, which is what powsybl's
//!   `CalculatedBusImpl` produces too. Merge.
//! - A **PGM `link`** does have identity: power-grid-model's output schema
//!   carries a `link` record with its own flows, and its fixtures assert them.
//!   Merging deletes the branch those numbers describe, so a link stays a
//!   branch.
//!
//! There is a second, empirical reason the CGMES side merges. Stamping its
//! switches as large-admittance branches was tried and *diverged*: the AC
//! Newton-Raphson solve failed on FullGrid with 20-odd such branches active at
//! once. That is the conditioning cost approach 2 carries, met in practice.

/// The admittance gridoxide gives any element it treats as an ideal
/// connection, declared or detected: `2e5 + j2e5` per-unit.
///
/// Three orders below power-grid-model's own `1e8 + j1e8` for a `link`, and
/// chosen by measurement rather than derivation, because the two calculation
/// types pull in opposite directions and their windows barely overlap:
///
/// - **Power flow** wants it large. The drop across the connection is
///   `ΔV = I/y`, and the `dummy-test` fixture checks node voltages and the
///   link's own current at 1e-5 relative. At `1e5` both land exactly on that
///   boundary and fail; from `2e5` up they pass.
/// - **State estimation** wants it small. `G = HᵀWH` *squares* the admittance,
///   so power-grid-model's `1e8` becomes `1e16` in the gain matrix and the
///   `node-injection-*` fixtures come back singular. They stay singular at
///   `1e6`.
///
/// | `y` | power flow | state estimation |
/// |---|---|---|
/// | `1e8` (power-grid-model's) | pass | **singular** |
/// | `1e6` | pass | **singular** |
/// | **`2e5`** | **pass** | **converge** |
/// | `1e5` | **fail**, at tolerance | converge |
///
/// The narrowness of that window is the argument for treating this as a
/// regularization parameter with a measured value rather than a physical
/// constant: a network far outside these fixtures' power scale may need it
/// re-measured, and if no value serves both, the equality-constrained
/// formulation (`docs/src/powerflow/zero_impedance_branches.md`, approach 3) is
/// the exit — it imposes `V_i = V_j` exactly, with no large number anywhere.
pub const IDEAL_CONNECTION_Y: num_complex::Complex<f64> = num_complex::Complex::new(2e5, 2e5);

/// The impedance corresponding to [`IDEAL_CONNECTION_Y`]: `3.54e-6` p.u.
///
/// What a branch caught by [`ZERO_IMPEDANCE_THRESHOLD`] is raised *to*.
fn ideal_connection_z() -> f64 {
    1.0 / (IDEAL_CONNECTION_Y.re.powi(2) + IDEAL_CONNECTION_Y.im.powi(2)).sqrt()
}

/// Below this impedance, per-unit, a branch is treated as an ideal connection
/// rather than as an ordinary one.
///
/// `Y = 1/Z` is unbounded as `Z -> 0`, and a network can reach that by accident
/// — a jumper modelled as a very short line rather than as a declared switch or
/// link. power-grid-model's own `ill-conditioned-by-line-meshed` fixture
/// contains exactly that, a line at `7.07e-9` p.u., and is named for the
/// consequence.
///
/// **Detection and treatment are separate numbers, deliberately.** This
/// threshold only decides *whether* a branch is an ideal connection; what it is
/// clamped to is [`IDEAL_CONNECTION_Y`], the same admittance a declared link
/// gets. Tying the two together would force a single value to satisfy two
/// incompatible constraints:
///
/// - it must sit below every *legitimate* branch, or real data gets mangled —
///   the smallest in the committed fixtures are `1.0e-6` (PGM) and `2.92e-6`
///   (CGMES);
/// - yet clamping only to that value leaves `|Y|` as high as `1e7`, which is
///   inside the range measured as singular for state estimation.
///
/// Separating them satisfies both: detection at `1e-7` touches nothing real,
/// and anything it does catch is raised to a link's stiffness — which is what
/// such a branch *is*, an undeclared link.
///
/// powsybl-open-loadflow's equivalent (`lowImpedanceThreshold`) is `1e-8`. Its
/// default treatment is the equality-constrained formulation rather than a
/// clamp, so it never has to reconcile these two roles in one number.
pub const ZERO_IMPEDANCE_THRESHOLD: f64 = 1e-7;

/// Raises a branch below [`ZERO_IMPEDANCE_THRESHOLD`] to the impedance of
/// [`IDEAL_CONNECTION_Y`], preserving the R/X ratio so it keeps its character.
///
/// A branch at exactly zero has no ratio to preserve and becomes purely
/// reactive, matching how a zero-impedance connection behaves in practice.
///
/// This is powsybl's `REPLACE_BY_MIN_IMPEDANCE_LINE` treatment
/// (`SimplePiModel::setMinZ`). Its *default* is the equality-constrained
/// alternative instead; gridoxide clamps because it has no augmented-system
/// machinery on the power-flow side — see
/// `docs/src/powerflow/zero_impedance_branches.md`.
pub fn clamp_branch_impedance(r: f64, x: f64) -> (f64, f64) {
    let z = (r * r + x * x).sqrt();
    if z >= ZERO_IMPEDANCE_THRESHOLD || !z.is_finite() {
        return (r, x);
    }
    let target = ideal_connection_z();
    if z == 0.0 {
        return (0.0, target);
    }
    let scale = target / z;
    (r * scale, x * scale)
}

/// Minimal path-compressing union-find.
///
/// Deliberately without union-by-rank: the inputs here are at most a few
/// hundred buses, and path compression alone already flattens them.
pub struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind { parent: (0..n).collect() }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    pub fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Collapses each union-find group into a single index.
///
/// Returns `(remap, n_merged)`, where `remap[old] = new`. Group representatives
/// are chosen by whichever member the root lands on — arbitrary but
/// deterministic, and the resulting order is stable across runs because it
/// follows ascending original index.
///
/// The caller keeps ownership of what a bus *is*: this returns only the index
/// mapping, so an importer can carry across whichever fields the representative
/// should keep. That is safe for the usual case because a zero-impedance
/// connection ties buses at the same nominal voltage, so `u_rated` and friends
/// agree across a group by construction.
pub fn merge_groups(n: usize, uf: &mut UnionFind) -> (Vec<usize>, usize) {
    let mut remap = vec![usize::MAX; n];
    let mut next = 0;
    for i in 0..n {
        let root = uf.find(i);
        if remap[root] == usize::MAX {
            remap[root] = next;
            next += 1;
        }
        remap[i] = remap[root];
    }
    (remap, next)
}

/// Builds the union-find for a set of index pairs to be tied together.
///
/// A convenience for callers that already have their pairs in hand; importers
/// with per-element conditions to check (open switches, out-of-service
/// equipment) will usually drive [`UnionFind`] directly instead.
pub fn union_all(n: usize, pairs: impl IntoIterator<Item = (usize, usize)>) -> UnionFind {
    let mut uf = UnionFind::new(n);
    for (a, b) in pairs {
        uf.union(a, b);
    }
    uf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untied_indices_are_left_alone() {
        let mut uf = UnionFind::new(3);
        let (remap, n) = merge_groups(3, &mut uf);
        assert_eq!(remap, vec![0, 1, 2]);
        assert_eq!(n, 3);
    }

    #[test]
    fn a_tied_pair_becomes_one_index() {
        let mut uf = union_all(3, [(0, 2)]);
        let (remap, n) = merge_groups(3, &mut uf);
        assert_eq!(n, 2);
        assert_eq!(remap[0], remap[2], "tied indices must land together");
        assert_ne!(remap[0], remap[1]);
    }

    /// Transitivity: a chain collapses to one index, not to a pair of groups.
    #[test]
    fn a_chain_collapses_entirely() {
        let mut uf = union_all(4, [(0, 1), (1, 2), (2, 3)]);
        let (remap, n) = merge_groups(4, &mut uf);
        assert_eq!(n, 1);
        assert!(remap.iter().all(|&r| r == remap[0]));
    }

    /// A loop of ties is not a problem for merging — it is only a problem for
    /// the equality-constrained approach, which needs a spanning tree to avoid
    /// linearly dependent constraints. Worth pinning, since it is one of the
    /// concrete reasons this codebase prefers merging where it can.
    #[test]
    fn a_loop_merges_without_special_handling() {
        let mut uf = union_all(3, [(0, 1), (1, 2), (2, 0)]);
        let (remap, n) = merge_groups(3, &mut uf);
        assert_eq!(n, 1);
        assert!(remap.iter().all(|&r| r == remap[0]));
    }

    #[test]
    fn an_ordinary_impedance_is_left_alone() {
        let (r, x) = clamp_branch_impedance(0.05, 0.2);
        assert_eq!((r, x), (0.05, 0.2));
    }

    /// Clamping preserves the R/X ratio, so a clamped branch keeps its
    /// character rather than becoming arbitrarily resistive or reactive.
    #[test]
    fn clamping_preserves_the_r_over_x_ratio() {
        let (r, x) = clamp_branch_impedance(1e-10, 2e-10);
        let z = (r * r + x * x).sqrt();
        assert!((z - ideal_connection_z()).abs() < 1e-18, "z = {z}");
        assert!((x / r - 2.0).abs() < 1e-9, "ratio not preserved: {}", x / r);
    }

    /// The fixture case: power-grid-model's `ill-conditioned-by-line-meshed`
    /// carries a line at 7.07e-9 p.u., which would otherwise put an admittance
    /// of 1.4e8 into the Y-bus.
    #[test]
    fn the_ill_conditioned_fixture_value_is_caught() {
        let (r, x) = clamp_branch_impedance(5e-9, 5e-9);
        let y = 1.0 / (r * r + x * x).sqrt();
        let ideal = (IDEAL_CONNECTION_Y.re.powi(2) + IDEAL_CONNECTION_Y.im.powi(2)).sqrt();
        assert!(
            (y - ideal).abs() < 1.0,
            "a caught branch should land at a link's stiffness, not merely under the \
             detection threshold: {y:.3e} vs {ideal:.3e}"
        );
    }

    /// A branch at exactly zero has no ratio to preserve.
    #[test]
    fn exactly_zero_becomes_purely_reactive() {
        assert_eq!(clamp_branch_impedance(0.0, 0.0), (0.0, ideal_connection_z()));
    }

    /// The smallest legitimate branch in the committed fixtures is 1.0e-6 p.u.
    /// and must pass through untouched, or the threshold is disturbing real
    /// data rather than protecting against pathological data.
    #[test]
    fn the_smallest_real_fixture_branch_is_untouched() {
        let (r, x) = clamp_branch_impedance(0.0, 1.0e-6);
        assert_eq!((r, x), (0.0, 1.0e-6));
    }

    /// The remap is dense and ascending, so it can index straight into a
    /// freshly built vector without a second pass.
    #[test]
    fn the_remap_is_dense_and_ordered() {
        let mut uf = union_all(5, [(1, 3)]);
        let (remap, n) = merge_groups(5, &mut uf);
        assert_eq!(n, 4);
        assert_eq!(remap, vec![0, 1, 2, 1, 3]);
    }
}
