use std::collections::HashMap;
use serde::Deserialize;
use super::types::{Bus, BusType, Line};
use super::json::NetworkData;

// ── Input structs ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct PgmInput {
    pub data: PgmData,
}

#[derive(Deserialize)]
pub struct PgmData {
    pub node: Vec<PgmNode>,
    pub line: Vec<PgmLine>,
    pub source: Vec<PgmSource>,
    pub sym_load: Vec<PgmSymLoad>,
}

#[derive(Deserialize)]
pub struct PgmNode {
    pub id: u64,
    pub u_rated: f64,
}

#[derive(Deserialize)]
pub struct PgmLine {
    pub id: u64,
    pub from_node: u64,
    pub to_node: u64,
    pub from_status: u8,
    pub to_status: u8,
    pub r1: f64,
    pub x1: f64,
    pub c1: f64,
}

#[derive(Deserialize)]
pub struct PgmSource {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    pub u_ref: f64,
    pub sk: f64,
    pub rx_ratio: f64,
}

#[derive(Deserialize)]
pub struct PgmSymLoad {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    pub p_specified: f64,
    pub q_specified: f64,
}

// ── Output structs (used by integration tests) ────────────────────────────────

#[derive(Deserialize)]
pub struct PgmOutput {
    pub data: PgmOutputData,
}

#[derive(Deserialize)]
pub struct PgmOutputData {
    pub node: Vec<PgmNodeOutput>,
}

#[derive(Deserialize)]
pub struct PgmNodeOutput {
    pub id: u64,
    pub u_pu: f64,
    pub u_angle: f64,
}

// ── Public helpers ────────────────────────────────────────────────────────────

/// Returns a stable node-ID → 0-based-index map (sorted by node ID).
pub fn node_id_to_idx(input: &PgmInput) -> HashMap<u64, usize> {
    let mut ids: Vec<u64> = input.data.node.iter().map(|n| n.id).collect();
    ids.sort_unstable();
    ids.into_iter().enumerate().map(|(idx, id)| (id, idx)).collect()
}

/// Converts a PGM input document to the gridoxide `NetworkData` type.
///
/// `s_base_va` — chosen system base power in VA (e.g. 1e6 for 1 MVA).
/// `freq_hz`   — grid frequency in Hz (e.g. 50.0 or 60.0).
///
/// Unit-conversion rules:
///   Z_base     = u_rated² / s_base_va
///   r_pu       = r1 / Z_base
///   x_pu       = x1 / Z_base
///   b_shunt_pu = (2π · freq · c1) · Z_base
///   p_pu       = p_specified / s_base_va   (load → negative injection)
///   q_pu       = q_specified / s_base_va
///
/// Lines with from_status=0 or to_status=0 are excluded (open terminals).
///
/// Each active source is modelled as a virtual Slack bus (EMF = u_ref) behind a
/// series source impedance z_s (R_s, X_s, b_shunt=0), where:
///   |z_s_pu| = u_ref² · s_base_va / sk
///   X_s      = |z_s_pu| / sqrt(rx_ratio² + 1)
///   R_s      = rx_ratio · X_s
/// The virtual bus is appended after all PGM nodes; the source terminal node
/// itself becomes a plain PQ bus.
pub fn pgm_to_network_data(input: PgmInput, s_base_va: f64, freq_hz: f64) -> NetworkData {
    let id_to_idx = node_id_to_idx(&input);
    let id_to_u_rated: HashMap<u64, f64> = input.data.node.iter()
        .map(|n| (n.id, n.u_rated))
        .collect();

    // Accumulate per-node net injection from active loads (load = negative injection).
    let mut p_inj: HashMap<u64, f64> = HashMap::new();
    let mut q_inj: HashMap<u64, f64> = HashMap::new();
    for load in &input.data.sym_load {
        if load.status == 0 { continue; }
        *p_inj.entry(load.node).or_insert(0.0) -= load.p_specified / s_base_va;
        *q_inj.entry(load.node).or_insert(0.0) -= load.q_specified / s_base_va;
    }

    // All PGM nodes are PQ buses — sources are modelled via virtual Slack buses below.
    let n_nodes = input.data.node.len();
    let mut opt_buses = vec![None::<Bus>; n_nodes];
    let mut sorted_ids: Vec<u64> = input.data.node.iter().map(|n| n.id).collect();
    sorted_ids.sort_unstable();

    for id in &sorted_ids {
        let idx = id_to_idx[id];
        opt_buses[idx] = Some(Bus {
            idx,
            bus_type: BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: *p_inj.get(id).unwrap_or(&0.0),
            q_spec: *q_inj.get(id).unwrap_or(&0.0),
            q_min: -f64::INFINITY,
            q_max: f64::INFINITY,
        });
    }
    let mut buses: Vec<Bus> = opt_buses.into_iter().map(|b| b.unwrap()).collect();

    // Build line list. PGM's c1 is the *total* shunt capacitance; each π-end
    // carries c1/2. build_ybus halves b_shunt, so passing ω·c1·Z_base gives
    // ω·c1·Z_base/2 per end, matching PGM's y_shunt/2.
    // Half-open lines: the connected-end shunt (c1/2) plus the far-end shunt
    // seen through the series impedance sum to ≈ ω·c1·Z_base, so we model
    // the half-open contribution as a self-loop with that susceptance.
    let omega = 2.0 * std::f64::consts::PI * freq_hz;
    let mut lines = Vec::new();
    for ln in &input.data.line {
        match (ln.from_status, ln.to_status) {
            (1, 1) => {
                let u_rated = id_to_u_rated[&ln.from_node];
                let z_base = u_rated * u_rated / s_base_va;
                lines.push(Line {
                    from: id_to_idx[&ln.from_node],
                    to: id_to_idx[&ln.to_node],
                    r: ln.r1 / z_base,
                    x: ln.x1 / z_base,
                    b_shunt: omega * ln.c1 * z_base,
                });
            }
            (1, 0) => {
                let u_rated = id_to_u_rated[&ln.from_node];
                let z_base = u_rated * u_rated / s_base_va;
                let idx = id_to_idx[&ln.from_node];
                lines.push(Line { from: idx, to: idx, r: 0.0, x: 0.0,
                    b_shunt: omega * ln.c1 * z_base });
            }
            (0, 1) => {
                let u_rated = id_to_u_rated[&ln.to_node];
                let z_base = u_rated * u_rated / s_base_va;
                let idx = id_to_idx[&ln.to_node];
                lines.push(Line { from: idx, to: idx, r: 0.0, x: 0.0,
                    b_shunt: omega * ln.c1 * z_base });
            }
            _ => {} // both open — nothing connected
        }
    }

    // Virtual Slack bus + source-impedance branch for each active source.
    for (i, src) in input.data.source.iter().filter(|s| s.status != 0).enumerate() {
        let virtual_idx = n_nodes + i;
        let z_s_pu = src.u_ref * src.u_ref * s_base_va / src.sk;
        let x_s = z_s_pu / (src.rx_ratio * src.rx_ratio + 1.0_f64).sqrt();
        let r_s = src.rx_ratio * x_s;
        buses.push(Bus {
            idx: virtual_idx,
            bus_type: BusType::Slack,
            voltage_mag: src.u_ref,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: -f64::INFINITY,
            q_max: f64::INFINITY,
        });
        lines.push(Line {
            from: virtual_idx,
            to: id_to_idx[&src.node],
            r: r_s,
            x: x_s,
            b_shunt: 0.0,
        });
    }

    NetworkData { buses, lines }
}
