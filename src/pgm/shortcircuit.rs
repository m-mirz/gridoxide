//! Short-circuit calculation from a PGM document: the network
//! [`shortcircuit`](crate::shortcircuit) solves, built from the document, and
//! the document's `fault` entries resolved against it.

use std::collections::HashMap;

use num_complex::Complex;

use crate::shortcircuit::{
    Fault, FaultAdmittance, FaultPhase, FaultType, ScNetwork3Ph, ScSource, ShortCircuitError,
    ShortCircuitOptions, ShortCircuitReport, short_circuit,
};

use crate::network::source_impedance_pu_seq;

use super::{PgmInput, node_id_to_idx, pgm_lines_3ph, pgm_shunts_3ph, pgm_transformers_3ph};

/// Builds the three-phase network a short-circuit calculation solves.
///
/// Differs from [`pgm_to_3ph_network`](super::pgm_to_3ph_network) in exactly
/// two ways, both required by IEC 60909:
///
/// 1. **No loads and no generation.** The standard deliberately isolates the
///    short-circuit current from the pre-fault dispatch, so every `sym_load`,
///    `asym_load`, `sym_gen` and `asym_gen` is ignored. Shunt admittances are
///    *kept* — power-grid-model's own documentation calls that exception out
///    explicitly, and it is a passive element rather than a dispatch decision.
/// 2. **Sources become Norton equivalents, not virtual slack buses.**
///    `pgm_to_3ph_network` appends a virtual slack bus and an impedance branch
///    per source, which is right for a power flow whose unknowns include the
///    source's own power. Here the source's internal admittance is folded onto
///    its node's own diagonal and its EMF becomes a current injection, which
///    is what power-grid-model's `add_sources` does and what keeps the result
///    indexed by physical node.
///
/// The EMF itself is *not* built here — it depends on the voltage-scaling
/// choice, which is a per-calculation option rather than a property of the
/// network, so [`ScSource`] carries the angle and the rated voltage and
/// `shortcircuit` combines them with `c`.
pub fn pgm_to_3ph_sc_network(input: &PgmInput, s_base_va: f64, freq_hz: f64) -> ScNetwork3Ph {
    let node_idx = node_id_to_idx(input);
    let id_to_u_rated: HashMap<u64, f64> =
        input.data.node.iter().map(|n| (n.id, n.u_rated)).collect();

    let lines = pgm_lines_3ph(input, &node_idx, s_base_va, freq_hz);
    let transformers = pgm_transformers_3ph(input, &node_idx, s_base_va);
    let shunts = pgm_shunts_3ph(input, &node_idx, s_base_va);

    // The ids behind those, replaying the rules the conversions above apply —
    // the same approach `pgm_3ph_maps` takes, and checked against the lengths
    // below so the two cannot drift apart silently.
    let mut branch_ids = Vec::new();
    let mut branch_closed = Vec::new();
    for ln in &input.data.line {
        if !matches!((ln.from_status, ln.to_status), (1, 1) | (1, 0) | (0, 1)) {
            continue;
        }
        branch_ids.push(ln.id);
        branch_closed.push([ln.from_status != 0, ln.to_status != 0]);
    }
    assert_eq!(
        branch_ids.len(),
        lines.len(),
        "line ids out of step with pgm_lines_3ph"
    );
    for t in &input.data.transformer {
        branch_ids.push(t.id);
        branch_closed.push([t.from_status != 0, t.to_status != 0]);
    }
    let shunt_ids: Vec<u64> = input
        .data
        .shunt
        .iter()
        .filter(|s| s.status != 0)
        .map(|s| s.id)
        .collect();
    assert_eq!(
        shunt_ids.len(),
        shunts.len(),
        "shunt ids out of step with pgm_shunts_3ph"
    );

    // Links, which `pgm_transformers_3ph` appends after the transformers with
    // the same both-open rule, so this path and the estimator cannot disagree
    // about what a link is. Only their ids are recorded here.
    for lk in &input.data.link {
        if lk.from_status == 0 && lk.to_status == 0 {
            continue;
        }
        branch_ids.push(lk.id);
        branch_closed.push([lk.from_status != 0, lk.to_status != 0]);
    }
    assert_eq!(
        branch_ids.len(),
        lines.len() + transformers.len(),
        "transformer and link ids out of step with pgm_transformers_3ph",
    );

    let sources = input
        .data
        .source
        .iter()
        .filter(|s| s.status != 0)
        .map(|src| {
            let (r1, x1, r0, x0) =
                source_impedance_pu_seq(src.sk, src.rx_ratio, src.z01_ratio, s_base_va);
            let one = Complex::new(1.0, 0.0);
            ScSource {
                id: src.id,
                node: node_idx[&src.node],
                y1: one / Complex::new(r1, x1),
                y0: one / Complex::new(r0, x0),
                u_ref_angle: src.u_ref_angle,
                u_rated: id_to_u_rated[&src.node],
            }
        })
        .collect();

    let mut u_rated = vec![0.0; input.data.node.len()];
    for n in &input.data.node {
        u_rated[node_idx[&n.id]] = n.u_rated;
    }

    ScNetwork3Ph {
        n_nodes: input.data.node.len(),
        lines,
        transformers,
        shunts,
        sources,
        branch_ids,
        branch_closed,
        shunt_ids,
        node_idx,
        u_rated,
        s_base_va,
    }
}

/// Resolves a document's `fault` entries against the network they apply to.
///
/// Inactive faults are dropped. `fault_phase` left at its default sentinel is
/// resolved through [`FaultType::default_phase`], and a zero `r_f`/`x_f` pair
/// becomes [`FaultAdmittance::Bolted`] rather than a zero admittance — see
/// [`PgmFault`](super::PgmFault) for why that reading is the right one.
///
/// The fault admittance is per-unit against the **faulted node's** own base,
/// `z_base = u_rated² / s_base`, which is why this needs the network and not
/// just the document.
pub fn faults_from_pgm(
    input: &PgmInput,
    net: &ScNetwork3Ph,
) -> Result<Vec<Fault>, ShortCircuitError> {
    let mut out = Vec::new();
    for f in input.data.fault.iter().filter(|f| f.status != 0) {
        let bus = *net
            .node_idx
            .get(&f.fault_object)
            .ok_or(ShortCircuitError::UnknownFaultObject(f.fault_object))?;
        let fault_type = FaultType::from_code(f.fault_type)?;
        let fault_phase =
            FaultPhase::from_code(f.fault_phase)?.unwrap_or_else(|| fault_type.default_phase());

        let y_fault = if f.r_f == 0.0 && f.x_f == 0.0 {
            FaultAdmittance::Bolted
        } else {
            let u_rated = net.u_rated[bus];
            let z_base = u_rated * u_rated / net.s_base_va;
            FaultAdmittance::Finite(z_base / Complex::new(f.r_f, f.x_f))
        };

        out.push(Fault {
            id: f.id,
            bus,
            fault_type,
            fault_phase,
            y_fault,
        });
    }
    Ok(out)
}

/// Runs a short-circuit calculation straight from a PGM-format document.
///
/// The shortest path from a file to an answer; [`short_circuit`] is the entry
/// point for callers that already hold a network.
pub fn short_circuit_from_pgm(
    input: &PgmInput,
    s_base_va: f64,
    freq_hz: f64,
    opts: ShortCircuitOptions,
) -> Result<(ScNetwork3Ph, ShortCircuitReport), ShortCircuitError> {
    let net = pgm_to_3ph_sc_network(input, s_base_va, freq_hz);
    let faults = faults_from_pgm(input, &net)?;
    let report = short_circuit(&net, &faults, opts)?;
    Ok((net, report))
}
