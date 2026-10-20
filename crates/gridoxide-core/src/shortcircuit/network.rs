//! The network a short-circuit calculation solves, independent of the
//! document it came from. gridoxide's `pgm::shortcircuit`
//! builds one from a PGM document.

use std::collections::HashMap;

use num_complex::Complex;

use crate::network::ShuntAdm3Ph;
use crate::types::{Line3Ph, Transformer3PhSeq};

/// One active `source`, reduced to the Norton equivalent a short-circuit
/// calculation needs: an internal admittance at its node, behind a fixed EMF.
#[derive(Clone, Copy, Debug)]
pub struct ScSource {
    /// The document's own `source` id.
    pub id: u64,
    /// Physical node index the source sits on.
    pub node: usize,
    /// Positive- (= negative-) sequence internal admittance, per unit.
    pub y1: Complex<f64>,
    /// Zero-sequence internal admittance, per unit.
    pub y0: Complex<f64>,
    /// `u_ref_angle`, the only part of the source's voltage specification that
    /// survives into a short-circuit calculation — see
    /// gridoxide's `pgm::shortcircuit::pgm_to_3ph_sc_network`.
    pub u_ref_angle: f64,
    /// Rated line-to-line voltage of the node this source sits on, which is
    /// what selects its IEC 60909 voltage-scaling factor `c`.
    pub u_rated: f64,
}

/// The network a short-circuit calculation solves: the passive circuit plus
/// its sources as Norton equivalents.
pub struct ScNetwork3Ph {
    /// Physical node count. Bus `k`'s phases are rows `3k`, `3k+1`, `3k+2`.
    pub n_nodes: usize,
    pub lines: Vec<Line3Ph>,
    pub transformers: Vec<Transformer3PhSeq>,
    pub shunts: Vec<ShuntAdm3Ph>,
    pub sources: Vec<ScSource>,
    /// The PGM id behind each branch, in the flat order the report uses: the
    /// lines that yield a branch, then the transformers, then the links. A
    /// fully open line or link yields no branch and has no entry, which is why
    /// this cannot be derived from document position.
    pub branch_ids: Vec<u64>,
    /// Whether each branch's `[from, to]` terminal is closed, parallel to
    /// `branch_ids`. A line with one live terminal has already collapsed to a
    /// self-loop on that terminal's node, which carries its current at `from`
    /// either way; this says which PGM terminal that current belongs to.
    pub branch_closed: Vec<[bool; 2]>,
    /// The PGM id of each entry of `shunts`.
    pub shunt_ids: Vec<u64>,
    /// Node id → physical node index.
    pub node_idx: HashMap<u64, usize>,
    /// Per physical node, its rated line-to-line voltage — needed to convert
    /// per-unit result voltages back to volts.
    pub u_rated: Vec<f64>,
    /// The three-phase base power the per-unit quantities are referred to.
    /// Carried alongside so results can be converted back to volts and amperes
    /// without the caller having to remember what it passed in.
    pub s_base_va: f64,
}
