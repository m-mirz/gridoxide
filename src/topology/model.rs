//! The node-breaker description: connectivity nodes, and the switches between
//! them.
//!
//! This is what an importer produces *before* deciding what an electrical bus
//! is. Turning it into buses is [`bus_view`](fn@super::bus_view)'s job, and the
//! two are separate precisely so that the decision can be made per solve
//! rather than baked in at import — which is what gridoxide did until now, and
//! why switching state was frozen the moment a file was read.

/// A connectivity node: a point where equipment terminals meet, before any
/// merging.
///
/// Distinct from [`BusIdx`] in the type system on purpose — see this module's
/// parent doc for the bug that motivates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIdx(pub usize);

/// A resolved electrical bus: one or more nodes merged together, and what the
/// solver actually indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BusIdx(pub usize);

/// Position of a switch in [`NodeBreakerTopology::switches`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SwitchIdx(pub usize);

/// The switching-device classes gridoxide distinguishes.
///
/// These are the nine CIM switch classes `cgmes::merge_closed_switches`
/// already reads, plus [`Junction`](Self::Junction). The distinction is not
/// cosmetic: a retention policy that keeps breakers but merges disconnectors
/// is the standard way to get a bus-breaker view, and that needs the kind to
/// survive import.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SwitchKind {
    /// CIM `Switch` — the base class, used directly when nothing more
    /// specific applies.
    Generic,
    Breaker,
    Disconnector,
    LoadBreakSwitch,
    DisconnectingCircuitBreaker,
    GroundDisconnector,
    Jumper,
    Cut,
    Fuse,
    /// CIM `Junction`: "a point where one or more conducting equipment are
    /// connected with zero impedance".
    ///
    /// Not switchable at all — CIM gives it no `open` state — so it is a
    /// permanent tie rather than a device with a position. It is represented
    /// here anyway so that a node-breaker view can show *why* two nodes are
    /// one, and [`Switch::open`] is always false for it.
    Junction,
}

impl SwitchKind {
    /// Whether this kind has an operable position at all.
    ///
    /// False only for [`Junction`](Self::Junction). A retention policy has no
    /// reason to retain something that cannot be operated, and a switching
    /// campaign has no reason to enumerate it.
    pub fn is_operable(self) -> bool {
        !matches!(self, SwitchKind::Junction)
    }
}

/// One switching device, between exactly two nodes.
///
/// Deliberately has no `retained` field. Retention is a property of the
/// *question being asked* — see [`RetentionPolicy`](super::RetentionPolicy) —
/// not of the device, and giving a switch a stored flag as well would mean two
/// mechanisms deciding one thing. If a source model turns out to carry its own
/// retained flag (powsybl's node-breaker importer has the concept), the policy
/// gains a variant that consults it rather than the struct gaining a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Switch {
    pub kind: SwitchKind,
    pub nodes: [NodeIdx; 2],
    /// Position. Always false for [`SwitchKind::Junction`], which has none.
    pub open: bool,
    /// Whether the device is in service at all. An out-of-service switch
    /// conducts nothing regardless of position.
    pub in_service: bool,
}

impl Switch {
    /// Whether this device currently ties its two nodes together: in service,
    /// and closed.
    pub fn conducting(&self) -> bool {
        self.in_service && !self.open
    }
}

/// A node-breaker network: how many connectivity nodes there are, and what
/// switches connect them.
///
/// # Why nodes are a count rather than records
///
/// There is nothing to *put* in a node record yet. The only producer today is
/// `cgmes::merge_closed_switches`, for which a "node" is a pre-merge
/// `TopologicalNode` that the importer already holds as a `Bus`; a parallel
/// `Vec<Node>` would duplicate that and immediately go stale. When the
/// node-breaker CGMES reader lands and nodes gain identity of their own —
/// mRID, voltage level, substation, bay containment — this becomes
/// `nodes: Vec<Node>` and `n_nodes` becomes `nodes.len()`. That is a small,
/// mechanical change, and making it later costs less than carrying an empty
/// struct now.
#[derive(Clone, Debug, Default)]
pub struct NodeBreakerTopology {
    /// Number of connectivity nodes. Every [`NodeIdx`] must be below this.
    pub n_nodes: usize,
    /// Every switching device, **in a fixed order that callers may depend on**.
    ///
    /// [`bus_view`](super::bus_view::bus_view) unions in this order, and
    /// union-find's root — and therefore which node represents a merged group
    /// — depends on it. `cgmes::merge_closed_switches` preserves its historical
    /// class-by-class order for exactly this reason, so that rerouting through
    /// this layer is bit-identical.
    pub switches: Vec<Switch>,
    /// Nodes that are busbar sections, for
    /// [`RetentionPolicy::RetainAdjacentToBusbar`](super::RetentionPolicy).
    /// Empty when the source has no busbar concept, in which case that policy
    /// retains nothing.
    pub busbars: Vec<NodeIdx>,
}

impl NodeBreakerTopology {
    /// A topology with `n_nodes` nodes and no switches — every node is its own
    /// bus under any policy.
    pub fn new(n_nodes: usize) -> Self {
        Self { n_nodes, switches: Vec::new(), busbars: Vec::new() }
    }

    /// Appends a switch, returning its index.
    pub fn add_switch(&mut self, switch: Switch) -> SwitchIdx {
        debug_assert!(switch.nodes[0].0 < self.n_nodes && switch.nodes[1].0 < self.n_nodes);
        self.switches.push(switch);
        SwitchIdx(self.switches.len() - 1)
    }

    /// How many switches are in service and closed — the ones that actually
    /// tie nodes together right now.
    pub fn conducting_count(&self) -> usize {
        self.switches.iter().filter(|s| s.conducting()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn switch(kind: SwitchKind, a: usize, b: usize, open: bool, in_service: bool) -> Switch {
        Switch { kind, nodes: [NodeIdx(a), NodeIdx(b)], open, in_service }
    }

    #[test]
    fn conducting_requires_both_in_service_and_closed() {
        assert!(switch(SwitchKind::Breaker, 0, 1, false, true).conducting());
        assert!(!switch(SwitchKind::Breaker, 0, 1, true, true).conducting());
        assert!(!switch(SwitchKind::Breaker, 0, 1, false, false).conducting());
        assert!(!switch(SwitchKind::Breaker, 0, 1, true, false).conducting());
    }

    /// A junction is a permanent tie, not a device with a position — CIM gives
    /// it no `open` at all.
    #[test]
    fn only_a_junction_is_inoperable() {
        assert!(!SwitchKind::Junction.is_operable());
        for kind in [
            SwitchKind::Generic,
            SwitchKind::Breaker,
            SwitchKind::Disconnector,
            SwitchKind::LoadBreakSwitch,
            SwitchKind::DisconnectingCircuitBreaker,
            SwitchKind::GroundDisconnector,
            SwitchKind::Jumper,
            SwitchKind::Cut,
            SwitchKind::Fuse,
        ] {
            assert!(kind.is_operable(), "{kind:?}");
        }
    }

    #[test]
    fn add_switch_returns_a_usable_index_and_counts_conducting() {
        let mut topo = NodeBreakerTopology::new(4);
        let a = topo.add_switch(switch(SwitchKind::Breaker, 0, 1, false, true));
        let b = topo.add_switch(switch(SwitchKind::Disconnector, 1, 2, true, true));
        assert_eq!((a, b), (SwitchIdx(0), SwitchIdx(1)));
        assert_eq!(topo.conducting_count(), 1);
    }
}
