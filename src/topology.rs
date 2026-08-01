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
