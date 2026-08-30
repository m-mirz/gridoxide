//! Standing an estimate up from a document, in either domain.
//!
//! The estimator itself is domain-agnostic — [`Target`](crate::measurement::Target),
//! [`StateLayout`](super::jacobian::StateLayout), the Jacobian, the constraints,
//! both methods and the batch solver are the same code whether a bus is a node
//! or a node's phase. What is *not* the same is the assembly in front of it:
//! reaching the phase domain took eight calls in a particular order, against
//! four converters and three stamping functions, and getting one of them wrong
//! produces a model that estimates something plausible and slightly untrue.
//!
//! That assembly is why the phase-domain estimator existed for a while without
//! being reachable from anything but a test. [`SeCase`] is the single door, and
//! it opens onto both rooms.
//!
//! ```no_run
//! # use gridoxide::se::case::SeCase;
//! # use gridoxide::pgm::PgmInput;
//! # fn f(input: &PgmInput) -> Result<(), Box<dyn std::error::Error>> {
//! let case = SeCase::from_pgm_3ph(input, 1e6, 50.0)?;
//! # Ok(()) }
//! ```

use std::collections::HashMap;

use crate::measurement::{Measurement, MeasurementError};
use crate::network::{
    build_ybus, build_ybus_3ph, stamp_shunts, stamp_shunts_3ph, stamp_transformers_3ph,
};
use crate::pgm::{PgmInput, Unsupported3Ph};
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

/// Why a document could not be turned into an estimate.
#[derive(Clone, Debug, PartialEq)]
pub enum SeCaseError {
    /// The phase domain does not model one of the document's components.
    Unsupported(Unsupported3Ph),
    /// A sensor could not be resolved against the network.
    Measurement(MeasurementError),
    /// The document has no sensor this estimator can use.
    ///
    /// Reported rather than estimated: with no measurements the gain matrix is
    /// identically zero, and "singular" would be a true statement about the
    /// wrong problem.
    NoSensors,
}

impl std::fmt::Display for SeCaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeCaseError::Unsupported(e) => write!(f, "{e}"),
            SeCaseError::Measurement(e) => write!(f, "{e}"),
            SeCaseError::NoSensors => {
                write!(f, "the document contains no usable sensors, so there is nothing to estimate")
            }
        }
    }
}

impl std::error::Error for SeCaseError {}

impl From<Unsupported3Ph> for SeCaseError {
    fn from(e: Unsupported3Ph) -> Self {
        SeCaseError::Unsupported(e)
    }
}

impl From<MeasurementError> for SeCaseError {
    fn from(e: MeasurementError) -> Self {
        SeCaseError::Measurement(e)
    }
}

impl SeCase {
    /// The symmetric case: one bus per node, sensors read as balanced totals.
    pub fn from_pgm(input: &PgmInput, s_base_va: f64, f_nom: f64) -> Result<Self, SeCaseError> {
        let id_to_idx = crate::pgm::node_id_to_idx(input);
        let shunts = crate::pgm::pgm_shunts_1ph(input, &id_to_idx, s_base_va);
        let net = crate::pgm::pgm_to_network(input.clone(), s_base_va, f_nom);
        let measurements = crate::measurement::measurements_from_pgm(input, &net, s_base_va)?;

        let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
        stamp_shunts(&mut ybus, &shunts);
        let buses = net.buses.clone();
        let node_idx = net.node_idx.clone();
        let network = SeNetwork::new(&net, ybus.finish(), &shunts);

        Self::finish(network, buses, measurements, node_idx, 1)
    }

    /// The phase domain: three buses per node, `3·node + phase`, and sensors
    /// that read their own phase rather than a balanced total.
    ///
    /// This is what makes an unbalanced network estimable rather than
    /// approximable. The reduction to a symmetric problem is exact only for a
    /// balanced one, and on a distribution feeder the unbalance *is* the
    /// question.
    pub fn from_pgm_3ph(input: &PgmInput, s_base_va: f64, f_nom: f64) -> Result<Self, SeCaseError> {
        // Refuses, by name, the components the three-phase conversion does not
        // model — before anything is built, so a document that cannot be
        // estimated says so rather than being estimated wrongly.
        let maps = crate::pgm::pgm_3ph_maps(input)?;

        let id_to_idx = crate::pgm::node_id_to_idx(input);
        let transformers = crate::pgm::pgm_transformers_3ph(input, &id_to_idx, s_base_va);
        let shunts = crate::pgm::pgm_shunts_3ph(input, &id_to_idx, s_base_va);
        let (buses, lines, _) = crate::pgm::pgm_to_3ph_network(input.clone(), s_base_va, f_nom);

        let mut ybus = build_ybus_3ph(buses.len() / 3, &lines);
        stamp_transformers_3ph(&mut ybus, &transformers);
        stamp_shunts_3ph(&mut ybus, &shunts);
        let network = SeNetwork::from_3ph(
            ybus.finish(),
            &lines,
            &transformers,
            &shunts,
            &maps.source_branch_idx,
            &maps.zero_injection,
        );

        let u_rated = |bus: usize| buses[bus].u_rated;
        let measurements =
            crate::measurement::measurements_from_pgm_3ph(input, &maps, s_base_va, &u_rated)?;

        Self::finish(network, buses, measurements, maps.node_idx.clone(), 3)
    }

    fn finish(
        network: SeNetwork,
        buses: Vec<Bus>,
        measurements: Vec<Measurement>,
        node_idx: HashMap<u64, usize>,
        phases: usize,
    ) -> Result<Self, SeCaseError> {
        if measurements.is_empty() {
            return Err(SeCaseError::NoSensors);
        }
        Ok(Self { network, buses, measurements, node_idx, phases })
    }

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
