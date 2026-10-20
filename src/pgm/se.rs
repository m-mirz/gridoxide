//! Standing a state estimate up from a PGM document, in either domain.
//!
//! The estimator itself is domain-agnostic — [`Target`](crate::measurement::Target),
//! [`StateLayout`](crate::se::jacobian::StateLayout), the Jacobian, the constraints,
//! both methods and the batch solver are the same code whether a bus is a node
//! or a node's phase. What is *not* the same is the assembly in front of it:
//! reaching the phase domain took eight calls in a particular order, against
//! four converters and three stamping functions, and getting one of them wrong
//! produces a model that estimates something plausible and slightly untrue.
//!
//! That assembly is why the phase-domain estimator existed for a while without
//! being reachable from anything but a test. [`se_case_from_pgm_3ph`] and [`se_case_from_pgm`] are the single door, and
//! they open onto both rooms.
//!
//! ```no_run
//! # use gridoxide::pgm::se::se_case_from_pgm_3ph;
//! # use gridoxide::pgm::PgmInput;
//! # fn f(input: &PgmInput) -> Result<(), Box<dyn std::error::Error>> {
//! let case = se_case_from_pgm_3ph(input, 1e6, 50.0)?;
//! # Ok(()) }
//! ```

use std::collections::HashMap;

use crate::measurement::Measurement;
use crate::network::{
    build_ybus, build_ybus_3ph, stamp_shunts, stamp_shunts_3ph, stamp_transformers_3ph,
};
use crate::se::SeNetwork;
use crate::se::case::SeCase;
use crate::types::Bus;

use super::PgmInput;
use super::measurement::MeasurementError;

/// Why a document could not be turned into an estimate.
#[derive(Clone, Debug, PartialEq)]
pub enum SeCaseError {
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
            SeCaseError::Measurement(e) => write!(f, "{e}"),
            SeCaseError::NoSensors => {
                write!(
                    f,
                    "the document contains no usable sensors, so there is nothing to estimate"
                )
            }
        }
    }
}

impl std::error::Error for SeCaseError {}

impl From<MeasurementError> for SeCaseError {
    fn from(e: MeasurementError) -> Self {
        SeCaseError::Measurement(e)
    }
}

/// The symmetric case: one bus per node, sensors read as balanced totals.
pub fn se_case_from_pgm(
    input: &PgmInput,
    s_base_va: f64,
    f_nom: f64,
) -> Result<SeCase, SeCaseError> {
    let id_to_idx = super::node_id_to_idx(input);
    let shunts = super::pgm_shunts_1ph(input, &id_to_idx, s_base_va);
    let net = super::pgm_to_network(input.clone(), s_base_va, f_nom);
    let measurements = super::measurement::measurements_from_pgm(input, &net, s_base_va)?;

    let mut ybus = build_ybus(net.buses.len(), &net.lines, &net.transformers);
    stamp_shunts(&mut ybus, &shunts);
    let buses = net.buses.clone();
    let node_idx = net.node_idx.clone();
    let network = net.se_network(ybus.finish(), &shunts);

    finish(network, buses, measurements, node_idx, 1)
}

/// The phase domain: three buses per node, `3·node + phase`, and sensors
/// that read their own phase rather than a balanced total.
///
/// This is what makes an unbalanced network estimable rather than
/// approximable. The reduction to a symmetric problem is exact only for a
/// balanced one, and on a distribution feeder the unbalance *is* the
/// question.
pub fn se_case_from_pgm_3ph(
    input: &PgmInput,
    s_base_va: f64,
    f_nom: f64,
) -> Result<SeCase, SeCaseError> {
    // The three-phase conversion models everything the symmetric one does
    // — links, three-winding transformers and all — so this cannot fail.
    // It returned a `Result` while that was untrue.
    let maps = super::pgm_3ph_maps(input);

    let id_to_idx = super::node_id_to_idx(input);
    let transformers = super::pgm_transformers_3ph(input, &id_to_idx, s_base_va);
    let shunts = super::pgm_shunts_3ph(input, &id_to_idx, s_base_va);
    let (buses, lines, _) = super::pgm_to_3ph_network(input.clone(), s_base_va, f_nom);

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
        super::measurement::measurements_from_pgm_3ph(input, &maps, s_base_va, &u_rated)?;

    finish(network, buses, measurements, maps.node_idx.clone(), 3)
}

fn finish(
    network: SeNetwork,
    buses: Vec<Bus>,
    measurements: Vec<Measurement>,
    node_idx: HashMap<u64, usize>,
    phases: usize,
) -> Result<SeCase, SeCaseError> {
    if measurements.is_empty() {
        return Err(SeCaseError::NoSensors);
    }
    Ok(SeCase {
        network,
        buses,
        measurements,
        node_idx,
        phases,
    })
}
