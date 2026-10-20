//! A network, a starting state and a measurement set that agree with each
//! other — what an estimate is run on.
//!
//! The estimator itself is domain-agnostic — [`Target`](crate::measurement::Target),
//! [`StateLayout`](super::jacobian::StateLayout), the Jacobian, the constraints,
//! both methods and the batch solver are the same code whether a bus is a node
//! or a node's phase. [`SeCase`] carries which of the two it is, and the
//! document's node ids, so a report can name its rows.
//!
//! Building one from a PGM document is
//! gridoxide's `pgm::se`'s job.

use std::collections::HashMap;

use crate::measurement::Measurement;
use crate::types::Bus;

use super::SeNetwork;

/// A network, a starting state and a measurement set, assembled together.
///
/// The three have to agree about what a bus index means, which is the whole
/// reason they are built in one place rather than three.
pub struct SeCase {
    pub network: SeNetwork,
    /// The buses, in the layout the network is indexed by. **Not** an estimate:
    /// a flat start, which [`linear_start`](super::nr::linear_start) improves on
    /// before the first iteration.
    pub buses: Vec<Bus>,
    pub measurements: Vec<Measurement>,
    /// Document node id to *node* index. In the phase domain a node owns three
    /// buses; use [`bus_of`](Self::bus_of) rather than this directly.
    pub node_idx: HashMap<u64, usize>,
    /// 1 for the symmetric domain, 3 for the phase domain.
    ///
    /// The one thing a caller downstream of this genuinely needs to know, and it
    /// is here so that reporting can say `node 7 phase b` rather than `bus 22`.
    pub phases: usize,
}

impl SeCase {
    /// The bus index carrying one phase of one document node.
    ///
    /// `phase` is ignored in the symmetric domain, where a node is one bus.
    pub fn bus_of(&self, node: u64, phase: usize) -> Option<usize> {
        let idx = *self.node_idx.get(&node)?;
        Some(self.phases * idx + if self.phases == 1 { 0 } else { phase })
    }

    /// The document's node ids, in index order — what a report iterates to name
    /// its rows.
    pub fn nodes(&self) -> Vec<(u64, usize)> {
        let mut out: Vec<(u64, usize)> = self.node_idx.iter().map(|(&id, &i)| (id, i)).collect();
        out.sort_unstable();
        out
    }
}

/// The phase labels, in the order the index arithmetic uses.
pub const PHASE_NAMES: [&str; 3] = ["a", "b", "c"];
