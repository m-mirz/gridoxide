//! Deriving an electrical bus view from a node-breaker topology.
//!
//! One rule, with one clause more than the importer used to have:
//!
//! > Union two nodes iff the switch between them is **closed**, **in service**,
//! > and **not retained**.
//!
//! The first two clauses are what `cgmes::merge_closed_switches` already did.
//! The third is the whole feature: a retained switch survives into the solver
//! as an element with an identity, a flow, and a state that can be changed
//! without re-importing.
//!
//! [`RetentionPolicy::MergeAll`] retains nothing, which makes the third clause
//! vacuous and reproduces the historical behavior exactly. It is the default
//! everywhere, and every CGMES fixture asserts against it.

use std::collections::HashSet;

use super::model::{BusIdx, NodeBreakerTopology, NodeIdx, SwitchIdx, SwitchKind};
use super::reduction::{merge_groups, UnionFind};

/// Which switches survive into the bus view as elements rather than being
/// merged away.
///
/// This is the knob that keeps node-breaker support from being a performance
/// regression. On SmallGrid, [`MergeAll`](Self::MergeAll) gives 167 buses;
/// [`RetainAll`](Self::RetainAll) gives 1,366 nodes plus 1,266 retained edges,
/// which for an AC solve is roughly a 16x larger system answering the same
/// question. Nobody wants that by default, and most people never want it at
/// all — they want the twenty switches they intend to operate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum RetentionPolicy {
    /// Merge every conducting switch. The bus-branch view, and gridoxide's
    /// historical and default behavior.
    #[default]
    MergeAll,
    /// Retain every operable switch — the full node-breaker view. A
    /// [`Junction`](SwitchKind::Junction) is still merged, since it has no
    /// position to retain.
    RetainAll,
    /// Retain only these kinds. `RetainKinds(vec![Breaker])` is the usual way
    /// to get a bus-breaker view on data whose disconnectors are noise.
    RetainKinds(Vec<SwitchKind>),
    /// Retain operable switches with a busbar section on at least one side —
    /// the classic bus-breaker view.
    ///
    /// Retains nothing when the source has no busbar concept, which is the
    /// honest answer rather than a silent fallback to something else.
    RetainAdjacentToBusbar,
    /// Retain exactly these, and merge everything else. What a contingency
    /// campaign uses: name the switches you intend to operate and pay for
    /// nothing more.
    Explicit(HashSet<SwitchIdx>),
}

impl RetentionPolicy {
    /// Whether `switch` survives into the bus view.
    ///
    /// A non-operable switch (a junction) is never retained under any policy:
    /// there is no state to give it, so retaining it would add an unknown and
    /// a constraint that could never do anything.
    fn retains(&self, idx: SwitchIdx, topo: &NodeBreakerTopology) -> bool {
        let switch = &topo.switches[idx.0];
        if !switch.kind.is_operable() {
            return false;
        }
        match self {
            RetentionPolicy::MergeAll => false,
            RetentionPolicy::RetainAll => true,
            RetentionPolicy::RetainKinds(kinds) => kinds.contains(&switch.kind),
            RetentionPolicy::RetainAdjacentToBusbar => {
                switch.nodes.iter().any(|n| topo.busbars.contains(n))
            }
            RetentionPolicy::Explicit(set) => set.contains(&idx),
        }
    }
}

/// A switch that survived into the bus view, with its endpoints resolved to
/// buses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetainedSwitch {
    pub switch: SwitchIdx,
    /// The buses its two nodes landed on.
    ///
    /// These can be **equal**: a retained switch whose endpoints are tied
    /// together by some *other* conducting path merges into one bus anyway. Its
    /// state then cannot affect connectivity, and a solver formulation must
    /// treat it as a self-loop rather than as a constraint between two
    /// distinct buses. See [`BusView::is_degenerate`].
    pub buses: [BusIdx; 2],
    pub open: bool,
}

/// The mapping from nodes to electrical buses, plus what survived.
#[derive(Clone, Debug)]
pub struct BusView {
    /// Node -> bus, indexed by [`NodeIdx`].
    bus_of: Vec<BusIdx>,
    /// Bus -> its nodes, in ascending node order. The inverse of `bus_of`,
    /// which reporting needs — "which connectivity nodes is this bus?" is the
    /// question a substation operator actually asks.
    nodes_of: Vec<Vec<NodeIdx>>,
    /// Bus -> the union-find root of its group.
    ///
    /// Arbitrary but deterministic, and *not* generally the lowest member.
    /// It exists because a merger has to adopt one member's attributes and
    /// this is the choice `cgmes::merge_closed_switches` has always made — see
    /// [`BusView::representative`].
    representative: Vec<NodeIdx>,
    retained: Vec<RetainedSwitch>,
}

impl BusView {
    pub fn n_buses(&self) -> usize {
        self.nodes_of.len()
    }

    pub fn n_nodes(&self) -> usize {
        self.bus_of.len()
    }

    /// The bus a node landed on.
    pub fn bus_of(&self, node: NodeIdx) -> BusIdx {
        self.bus_of[node.0]
    }

    /// The nodes that merged into a bus, ascending.
    pub fn nodes_of(&self, bus: BusIdx) -> &[NodeIdx] {
        &self.nodes_of[bus.0]
    }

    /// One node standing for the whole group, for a caller that must collapse
    /// per-node records into one per-bus record and has to pick whose.
    ///
    /// **Arbitrary but deterministic**, and deliberately *not* documented as
    /// the lowest member — it is the union-find root, which depends on the
    /// order switches appear in [`NodeBreakerTopology::switches`]. Callers that
    /// need a stable, order-independent choice should use `nodes_of(bus)[0]`
    /// instead and accept that it may differ from this.
    ///
    /// The distinction is not hypothetical. `cgmes::merge_closed_switches`
    /// clones this node's `Bus` record — `u_rated`, `bus_type` — into the
    /// merged bus, and its own comment records that the root was chosen over
    /// the first-encountered index to make that extraction "provably
    /// behaviour-preserving rather than merely equivalent-looking". Exposing
    /// the same choice here is what keeps rerouting through this layer
    /// bit-identical. The fields that matter do agree across a group in
    /// practice, since a conducting switch ties nodes at one nominal voltage —
    /// but "in practice" is not the same as "provably".
    pub fn representative(&self, bus: BusIdx) -> NodeIdx {
        self.representative[bus.0]
    }

    pub fn retained(&self) -> &[RetainedSwitch] {
        &self.retained
    }

    /// The raw node -> bus map, for callers that need to remap a parallel
    /// array in one pass.
    pub fn bus_of_slice(&self) -> &[BusIdx] {
        &self.bus_of
    }

    /// Whether a retained switch's two endpoints landed on the same bus, so
    /// that its position cannot affect connectivity.
    ///
    /// Happens when some other conducting path already ties the two ends —
    /// a bus-coupler in a ring, or a switch retained in parallel with a merged
    /// one. Worth knowing about rather than discovering as a singular
    /// constraint row later.
    pub fn is_degenerate(&self, retained: &RetainedSwitch) -> bool {
        retained.buses[0] == retained.buses[1]
    }
}

/// Derives the bus view under `policy`.
///
/// # Determinism, and why the switch order matters
///
/// Nodes are unioned in [`NodeBreakerTopology::switches`] order, and
/// [`merge_groups`] then numbers the groups by lowest member. Union-find's root
/// — and therefore which node a caller may treat as a merged group's
/// representative — depends on that order. This is not an incidental detail:
/// `cgmes::merge_closed_switches` clones the *root* node's bus record into the
/// merged bus, so a different order would produce a different (still correct,
/// but different) `u_rated`, and the fixture tests compare exact values.
/// Preserving the historical class-by-class order is what makes rerouting the
/// importer through here bit-identical.
pub fn bus_view(topo: &NodeBreakerTopology, policy: &RetentionPolicy) -> BusView {
    let mut uf = UnionFind::new(topo.n_nodes);
    let mut retained_idx: Vec<SwitchIdx> = Vec::new();

    for (i, switch) in topo.switches.iter().enumerate() {
        let idx = SwitchIdx(i);
        if policy.retains(idx, topo) {
            retained_idx.push(idx);
            continue;
        }
        if switch.conducting() {
            uf.union(switch.nodes[0].0, switch.nodes[1].0);
        }
    }

    let (remap, n_buses) = merge_groups(topo.n_nodes, &mut uf);
    let bus_of: Vec<BusIdx> = remap.into_iter().map(BusIdx).collect();

    let mut nodes_of: Vec<Vec<NodeIdx>> = vec![Vec::new(); n_buses];
    // `merge_groups` has already path-compressed every node, so `find` is a
    // stable lookup here and every member of a group yields the same root.
    let mut representative: Vec<NodeIdx> = vec![NodeIdx(0); n_buses];
    for (node, bus) in bus_of.iter().enumerate() {
        nodes_of[bus.0].push(NodeIdx(node));
        representative[bus.0] = NodeIdx(uf.find(node));
    }

    let retained = retained_idx
        .into_iter()
        .map(|idx| {
            let switch = &topo.switches[idx.0];
            RetainedSwitch {
                switch: idx,
                buses: [bus_of[switch.nodes[0].0], bus_of[switch.nodes[1].0]],
                open: switch.open,
            }
        })
        .collect();

    BusView { bus_of, nodes_of, representative, retained }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::model::Switch;

    fn sw(kind: SwitchKind, a: usize, b: usize, open: bool) -> Switch {
        Switch { kind, nodes: [NodeIdx(a), NodeIdx(b)], open, in_service: true }
    }

    /// 0 —[closed]— 1 —[open]— 2, plus an isolated node 3.
    fn fixture() -> NodeBreakerTopology {
        let mut topo = NodeBreakerTopology::new(4);
        topo.add_switch(sw(SwitchKind::Breaker, 0, 1, false));
        topo.add_switch(sw(SwitchKind::Disconnector, 1, 2, true));
        topo
    }

    #[test]
    fn merge_all_collapses_only_conducting_switches() {
        let topo = fixture();
        let view = bus_view(&topo, &RetentionPolicy::MergeAll);

        assert_eq!(view.n_nodes(), 4);
        assert_eq!(view.n_buses(), 3, "0 and 1 merge; 2 and 3 stay their own");
        assert_eq!(view.bus_of(NodeIdx(0)), view.bus_of(NodeIdx(1)));
        assert_ne!(view.bus_of(NodeIdx(1)), view.bus_of(NodeIdx(2)));
        assert!(view.retained().is_empty());
        assert_eq!(view.nodes_of(view.bus_of(NodeIdx(0))), &[NodeIdx(0), NodeIdx(1)]);
    }

    /// The defining property of retention: a retained switch does *not* merge
    /// its ends, even when closed.
    #[test]
    fn retain_all_keeps_every_operable_switch_and_merges_nothing() {
        let topo = fixture();
        let view = bus_view(&topo, &RetentionPolicy::RetainAll);

        assert_eq!(view.n_buses(), 4, "nothing merged");
        assert_eq!(view.retained().len(), 2);
        let closed = view.retained().iter().find(|r| !r.open).unwrap();
        assert_eq!(closed.buses, [BusIdx(0), BusIdx(1)]);
        assert!(!view.is_degenerate(closed));
    }

    #[test]
    fn retain_kinds_selects_by_class() {
        let topo = fixture();
        let view = bus_view(&topo, &RetentionPolicy::RetainKinds(vec![SwitchKind::Breaker]));

        // The breaker is retained (so 0 and 1 stay apart); the disconnector is
        // not retained but is open, so it merges nothing either.
        assert_eq!(view.n_buses(), 4);
        assert_eq!(view.retained().len(), 1);
        assert_eq!(view.retained()[0].switch, SwitchIdx(0));
    }

    #[test]
    fn explicit_retains_exactly_what_it_names() {
        let topo = fixture();
        let policy = RetentionPolicy::Explicit(HashSet::from([SwitchIdx(1)]));
        let view = bus_view(&topo, &policy);

        assert_eq!(view.retained().len(), 1);
        assert_eq!(view.retained()[0].switch, SwitchIdx(1));
        // Switch 0 was not retained and is closed, so it still merges.
        assert_eq!(view.bus_of(NodeIdx(0)), view.bus_of(NodeIdx(1)));
    }

    #[test]
    fn retain_adjacent_to_busbar_needs_busbars_and_says_so_when_absent() {
        let mut topo = fixture();
        assert!(
            bus_view(&topo, &RetentionPolicy::RetainAdjacentToBusbar).retained().is_empty(),
            "with no busbars declared, this policy retains nothing"
        );

        topo.busbars.push(NodeIdx(2));
        let view = bus_view(&topo, &RetentionPolicy::RetainAdjacentToBusbar);
        assert_eq!(view.retained().len(), 1, "only the switch touching node 2");
        assert_eq!(view.retained()[0].switch, SwitchIdx(1));
    }

    /// A junction has no position, so no policy retains it — retaining one
    /// would add an unknown and a constraint that could never do anything.
    #[test]
    fn a_junction_is_never_retained() {
        let mut topo = NodeBreakerTopology::new(2);
        topo.add_switch(sw(SwitchKind::Junction, 0, 1, false));
        topo.busbars.push(NodeIdx(0));

        for policy in [
            RetentionPolicy::RetainAll,
            RetentionPolicy::RetainKinds(vec![SwitchKind::Junction]),
            RetentionPolicy::RetainAdjacentToBusbar,
            RetentionPolicy::Explicit(HashSet::from([SwitchIdx(0)])),
        ] {
            let view = bus_view(&topo, &policy);
            assert!(view.retained().is_empty(), "{policy:?} retained a junction");
            assert_eq!(view.n_buses(), 1, "{policy:?} failed to merge a junction");
        }
    }

    /// An out-of-service switch conducts nothing whatever its position.
    #[test]
    fn an_out_of_service_switch_merges_nothing() {
        let mut topo = NodeBreakerTopology::new(2);
        topo.add_switch(Switch {
            kind: SwitchKind::Breaker,
            nodes: [NodeIdx(0), NodeIdx(1)],
            open: false,
            in_service: false,
        });
        assert_eq!(bus_view(&topo, &RetentionPolicy::MergeAll).n_buses(), 2);
    }

    /// A retained switch in parallel with a merged path lands both ends on one
    /// bus. Its position then cannot change connectivity, and a formulation
    /// has to know that rather than emit a constraint between a bus and itself.
    #[test]
    fn a_retained_switch_parallel_to_a_merged_path_is_degenerate() {
        let mut topo = NodeBreakerTopology::new(2);
        topo.add_switch(sw(SwitchKind::Breaker, 0, 1, false)); // retained
        topo.add_switch(sw(SwitchKind::Junction, 0, 1, false)); // always merged

        let view = bus_view(&topo, &RetentionPolicy::RetainAll);
        assert_eq!(view.n_buses(), 1);
        assert_eq!(view.retained().len(), 1);
        assert!(view.is_degenerate(&view.retained()[0]));
    }

    /// The representative is a member of its own group and is consistent for
    /// every member — the property `cgmes::merge_closed_switches` relies on
    /// when it clones that node's bus record.
    #[test]
    fn the_representative_is_a_consistent_member_of_its_group() {
        let mut topo = NodeBreakerTopology::new(5);
        topo.add_switch(sw(SwitchKind::Breaker, 3, 4, false));
        topo.add_switch(sw(SwitchKind::Breaker, 1, 2, false));
        topo.add_switch(sw(SwitchKind::Breaker, 2, 3, false));

        let view = bus_view(&topo, &RetentionPolicy::MergeAll);
        for bus in 0..view.n_buses() {
            let rep = view.representative(BusIdx(bus));
            assert!(
                view.nodes_of(BusIdx(bus)).contains(&rep),
                "bus {bus}'s representative {rep:?} is not one of its nodes"
            );
            for &node in view.nodes_of(BusIdx(bus)) {
                assert_eq!(view.representative(view.bus_of(node)), rep);
            }
        }
    }

    /// Bus numbering follows lowest member, and `nodes_of` inverts `bus_of`
    /// exactly — both are contracts callers rely on.
    #[test]
    fn bus_numbering_and_the_inverse_map_are_consistent() {
        let mut topo = NodeBreakerTopology::new(5);
        // Wire the higher nodes together first, to prove numbering does not
        // follow switch order.
        topo.add_switch(sw(SwitchKind::Breaker, 3, 4, false));
        topo.add_switch(sw(SwitchKind::Breaker, 1, 2, false));

        let view = bus_view(&topo, &RetentionPolicy::MergeAll);
        assert_eq!(view.n_buses(), 3);
        assert_eq!(view.bus_of(NodeIdx(0)), BusIdx(0));
        assert_eq!(view.bus_of(NodeIdx(1)), BusIdx(1));
        assert_eq!(view.bus_of(NodeIdx(3)), BusIdx(2));

        for bus in 0..view.n_buses() {
            for &node in view.nodes_of(BusIdx(bus)) {
                assert_eq!(view.bus_of(node), BusIdx(bus));
            }
        }
        let total: usize = (0..view.n_buses()).map(|b| view.nodes_of(BusIdx(b)).len()).sum();
        assert_eq!(total, view.n_nodes(), "every node lands on exactly one bus");
    }
}
