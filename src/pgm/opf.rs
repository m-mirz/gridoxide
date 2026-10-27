//! OPF from a PGM document and its companion OPF document: the networks
//! [`opf::dc`](crate::opf::dc) and [`opf::ac`](crate::opf::ac) optimize,
//! assembled from the two.

use std::collections::HashMap;

use crate::linear::btheta::DcBranch;
use crate::opf::OpfError;
use crate::opf::ac::{AcGenerator, AcOpfNetwork, AcOpfOptions};
use crate::opf::dc::{DcGenerator, DcLoad, DcOpfNetwork};
use crate::opf::model::OpfData;
use crate::types::{Line, Transformer};

/// Builds the network from a PGM document and its companion OPF document.
///
/// Three things need care, and each is a place a naive assembly goes
/// wrong:
///
/// - **The per-unit base must be the OPF document's.** `pgm_to_network`
///   takes `s_base_va`, and passing anything other than
///   `base_mva · 1e6` would put the network's per-unit powers on a
///   different base from the cost curves', which is invisible until the
///   dispatch comes out scaled.
/// - **The synthesized source branches and buses are dropped.** Converting
///   a PGM document appends a virtual slack bus per `source`, joined by an
///   impedance branch. That is right for a power flow, where the source's
///   output is an unknown; here the source's generator is a decision
///   variable listed in the OPF document, so the virtual pair would double
///   it. The physical end of a source branch is the reference bus.
/// - **Branch limits are keyed by PGM component id, not by flat index.**
///   `PgmNetwork::branch_idx` is the translation.
///
/// `approximation` picks how branch susceptance is formed, and it is not a
/// cosmetic choice — see [`DcOpfOptions`](crate::opf::dc::DcOpfOptions) for
/// the measured effect on the published baseline. [`DcApproximation::IgnoreG`] is what OPF should
/// normally use.
///
/// [`DcApproximation::IgnoreG`]: crate::linear::DcApproximation::IgnoreG
pub fn dc_opf_network_from_pgm(
    input: super::PgmInput,
    opf: &OpfData,
    freq_hz: f64,
    approximation: crate::linear::DcApproximation,
) -> Result<DcOpfNetwork, OpfError> {
    let base_mva = opf.base_mva;
    let s_base_va = base_mva * 1e6;

    let loads_raw: Vec<(u64, f64)> = input
        .data
        .sym_load
        .iter()
        .filter(|l| l.status != 0)
        .map(|l| (l.node, l.p_specified))
        .collect();
    let n_physical = input.data.node.len();

    let net = super::pgm_to_network(input, s_base_va, freq_hz);
    let branches: Vec<DcBranch> = crate::linear::btheta::dc_branches(
        &net.lines,
        &net.transformers,
        crate::linear::DcOptions {
            approximation,
            ..crate::linear::DcOptions::default()
        },
    )
    .into_iter()
    // A branch touching a virtual bus is a synthesized source connection.
    .filter(|b| b.from < n_physical && b.to < n_physical)
    .collect();

    // The reference is the physical bus a source attaches to. Falling back
    // to bus 0 keeps a source-less document solvable rather than erroring
    // on something the OPF itself does not need.
    let reference = net
        .source_branch_idx
        .values()
        .filter_map(|&flat| net.lines.get(flat))
        .map(|line| line.to.min(line.from))
        .find(|&bus| bus < n_physical)
        .unwrap_or(0);

    let mut limits = HashMap::new();
    for limit in &opf.branch_limit {
        let Some(&flat) = net.branch_idx.get(&limit.id) else {
            continue;
        };
        limits.insert(flat, limit.rate_pu(base_mva));
    }

    let mut generators = Vec::with_capacity(opf.generator.len());
    for g in &opf.generator {
        let Some(&bus) = net.node_idx.get(&g.node) else {
            return Err(OpfError::Backend(format!(
                "generator {} sits on node {}, which is not in the network document",
                g.index, g.node
            )));
        };
        let (p_min, p_max) = g.p_limits_pu(base_mva);
        generators.push(DcGenerator {
            index: g.index,
            bus,
            p_min,
            p_max,
            cost: g.cost.as_ref().map(|c| c.to_per_unit(base_mva)),
        });
    }

    let mut loads = Vec::new();
    for (node, p_specified) in loads_raw {
        let Some(&bus) = net.node_idx.get(&node) else {
            continue;
        };
        loads.push(DcLoad {
            bus,
            p: p_specified / s_base_va,
        });
    }

    Ok(DcOpfNetwork {
        n_buses: n_physical,
        reference,
        branches,
        limits,
        generators,
        loads,
        base_mva,
    })
}

/// Builds from a PGM document and its companion OPF document.
///
/// The same three traps [`dc_opf_network_from_pgm`]
/// documents apply unchanged — the per-unit base must be the OPF
/// document's, synthesized source buses must be dropped, and branch limits
/// are keyed by component id — so they are not repeated here.
pub fn ac_opf_network_from_pgm(
    input: super::PgmInput,
    opf: &OpfData,
    freq_hz: f64,
    options: &AcOpfOptions,
) -> Result<AcOpfNetwork, OpfError> {
    let base_mva = opf.base_mva;
    let s_base_va = base_mva * 1e6;

    let loads_raw: Vec<(u64, f64, f64)> = input
        .data
        .sym_load
        .iter()
        .filter(|l| l.status != 0)
        .map(|l| (l.node, l.p_specified, l.q_specified))
        .collect();
    let n_physical = input.data.node.len();

    // Bus shunts, captured before `pgm_to_network` consumes the document.
    //
    // Dropping these is not a rounding error. A shunt capacitor supplies
    // reactive power for free; without it the generators have to, and the
    // dispatch gets more expensive. On the pglib fixtures exactly the
    // three cases carrying shunts were the three that disagreed with the
    // published objectives, and the two with none matched to 0.001%.
    let id_to_idx = super::node_id_to_idx(&input);
    let shunts_raw: Vec<(usize, num_complex::Complex<f64>)> =
        super::pgm_shunts_1ph(&input, &id_to_idx, s_base_va)
            .into_iter()
            .filter(|s| s.at < n_physical)
            .map(|s| (s.at, s.y))
            .collect();

    let net = super::pgm_to_network(input, s_base_va, freq_hz);

    let lines: Vec<Line> = net
        .lines
        .iter()
        .enumerate()
        .filter(|(flat, l)| {
            l.from < n_physical
                && l.to < n_physical
                && !net.source_branch_idx.values().any(|v| v == flat)
        })
        .map(|(_, l)| l.clone())
        .collect();
    let transformers: Vec<Transformer> = net
        .transformers
        .iter()
        .filter(|t| t.from < n_physical && t.to < n_physical)
        .cloned()
        .collect();

    let reference = net
        .source_branch_idx
        .values()
        .filter_map(|&flat| net.lines.get(flat))
        .map(|line| line.to.min(line.from))
        .find(|&bus| bus < n_physical)
        .unwrap_or(0);

    let mut p_load = vec![0.0; n_physical];
    let mut q_load = vec![0.0; n_physical];
    for (node, p, q) in loads_raw {
        if let Some(&bus) = net.node_idx.get(&node) {
            if bus < n_physical {
                p_load[bus] += p / s_base_va;
                q_load[bus] += q / s_base_va;
            }
        }
    }

    let mut generators = Vec::with_capacity(opf.generator.len());
    for g in &opf.generator {
        let Some(&bus) = net.node_idx.get(&g.node) else {
            return Err(OpfError::Backend(format!(
                "generator {} sits on node {}, which is not in the network document",
                g.index, g.node
            )));
        };
        let (p_min, p_max) = g.p_limits_pu(base_mva);
        let (q_min, q_max) = g.q_limits_pu(base_mva);
        generators.push(AcGenerator {
            index: g.index,
            bus,
            p_min,
            p_max,
            q_min,
            q_max,
            cost: g.cost.as_ref().map(|c| c.to_per_unit(base_mva)),
        });
    }

    // Branch limits are keyed on the *filtered* branch order, which is
    // lines-then-transformers over the physical branches only.
    let mut limits = HashMap::new();
    for limit in &opf.branch_limit {
        if let Some(&flat) = net.branch_idx.get(&limit.id) {
            if flat < lines.len() + transformers.len() {
                limits.insert(flat, limit.rate_pu(base_mva));
            }
        }
    }

    // Per-bus limits where the document supplies them, the option
    // defaults where it does not. A document written before `bus_voltage`
    // existed carries none, and falling back is right there — but falling
    // back *per bus* would be wrong, since a partial list would silently
    // mix two conventions. The list is all-or-nothing by construction:
    // `build_opf_data` emits an entry for every bus or none at all.
    let mut v_min = vec![options.default_v_min; n_physical];
    let mut v_max = vec![options.default_v_max; n_physical];
    for entry in &opf.bus_voltage {
        if let Some(&bus) = net.node_idx.get(&entry.node) {
            if bus < n_physical {
                v_min[bus] = entry.v_min;
                v_max[bus] = entry.v_max;
            }
        }
    }

    Ok(AcOpfNetwork {
        n_buses: n_physical,
        reference,
        lines,
        transformers,
        generators,
        p_load,
        q_load,
        v_min,
        v_max,
        limits,
        shunts: shunts_raw,
        base_mva,
        v_start: None,
        angle_limits: Vec::new(),
    })
}
