use std::collections::HashMap;
use serde::Deserialize;
use super::types::{Bus, BusType, Line, Line3Ph};
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
    #[serde(default)]
    pub sym_load: Vec<PgmSymLoad>,
    #[serde(default)]
    pub asym_load: Vec<PgmAsymLoad>,
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
    pub r0: f64,
    pub x0: f64,
    pub c0: f64,
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

#[derive(Deserialize)]
pub struct PgmAsymLoad {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    pub p_specified: [f64; 3],
    pub q_specified: [f64; 3],
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

#[derive(Deserialize)]
pub struct PgmAsymOutput {
    pub data: PgmAsymOutputData,
}

#[derive(Deserialize)]
pub struct PgmAsymOutputData {
    pub node: Vec<PgmNodeAsymOutput>,
}

#[derive(Deserialize)]
pub struct PgmNodeAsymOutput {
    pub id: u64,
    pub u_pu: [f64; 3],
    pub u_angle: [f64; 3],
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

/// Converts a PGM input document (with `asym_load`) into a 3N-bus expanded
/// representation suitable for three-phase power flow.
///
/// Physical node `k` (sorted by PGM node ID) maps to buses at indices
/// `3k`, `3k+1`, `3k+2` for phases a, b, c respectively.  Each active
/// source adds one virtual Slack node appended after all physical nodes,
/// whose three phase buses carry fixed voltages `u_ref ∠ 0°`, `u_ref ∠ -120°`,
/// `u_ref ∠ +120°`.
///
/// Returns `(buses, lines_3ph, id_to_physical_idx)`.  Pass `buses.len() / 3`
/// as `n` to `build_ybus_3ph`.
pub fn pgm_to_3ph_network(
    input: PgmInput,
    s_base_va: f64,
    freq_hz: f64,
) -> (Vec<Bus>, Vec<Line3Ph>, HashMap<u64, usize>) {
    let id_to_idx = node_id_to_idx(&input);
    let id_to_u_rated: HashMap<u64, f64> =
        input.data.node.iter().map(|n| (n.id, n.u_rated)).collect();

    let n_nodes = input.data.node.len();
    let two_pi_f = 2.0 * std::f64::consts::PI * freq_hz;
    let phase_ang = [0.0_f64, -2.0 * std::f64::consts::PI / 3.0, 2.0 * std::f64::consts::PI / 3.0];

    // Per-node, per-phase net injection [phase_a, phase_b, phase_c] in p.u.
    let mut p_inj: HashMap<u64, [f64; 3]> = HashMap::new();
    let mut q_inj: HashMap<u64, [f64; 3]> = HashMap::new();
    // The phase-domain power flow equations naturally work in units of S_base/3
    // (per-phase base), so per-phase Watts must be divided by s_base_va/3.
    let s_base_1ph = s_base_va / 3.0;
    for load in &input.data.asym_load {
        if load.status == 0 { continue; }
        let pe = p_inj.entry(load.node).or_insert([0.0; 3]);
        let qe = q_inj.entry(load.node).or_insert([0.0; 3]);
        for ph in 0..3 {
            pe[ph] -= load.p_specified[ph] / s_base_1ph;
            qe[ph] -= load.q_specified[ph] / s_base_1ph;
        }
    }
    // sym_load p_specified is 3-phase total; each phase gets 1/3 of total,
    // and P_1ph_pu = (P_total/3) / (s_base/3) = P_total / s_base.
    for load in &input.data.sym_load {
        if load.status == 0 { continue; }
        let pe = p_inj.entry(load.node).or_insert([0.0; 3]);
        let qe = q_inj.entry(load.node).or_insert([0.0; 3]);
        let p_ph = load.p_specified / s_base_va;
        let q_ph = load.q_specified / s_base_va;
        for ph in 0..3 {
            pe[ph] -= p_ph;
            qe[ph] -= q_ph;
        }
    }

    // Build 3N buses: physical node k → buses 3k, 3k+1, 3k+2.
    let mut sorted_ids: Vec<u64> = input.data.node.iter().map(|n| n.id).collect();
    sorted_ids.sort_unstable();

    let mut buses: Vec<Bus> = Vec::with_capacity(3 * n_nodes);
    for _ in 0..3 * n_nodes {
        buses.push(Bus {
            idx: 0,
            bus_type: BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: -f64::INFINITY,
            q_max: f64::INFINITY,
        });
    }
    for id in &sorted_ids {
        let phys = id_to_idx[id];
        let p_arr = p_inj.get(id).copied().unwrap_or([0.0; 3]);
        let q_arr = q_inj.get(id).copied().unwrap_or([0.0; 3]);
        for ph in 0..3 {
            let bus_idx = 3 * phys + ph;
            buses[bus_idx] = Bus {
                idx: bus_idx,
                bus_type: BusType::PQ,
                voltage_mag: 1.0,
                voltage_ang: phase_ang[ph],
                p_spec: p_arr[ph],
                q_spec: q_arr[ph],
                q_min: -f64::INFINITY,
                q_max: f64::INFINITY,
            };
        }
    }

    // Build Line3Ph list.
    let mut lines: Vec<Line3Ph> = Vec::new();
    for ln in &input.data.line {
        let u_rated_from = id_to_u_rated[&ln.from_node];
        let z_base = u_rated_from * u_rated_from / s_base_va;
        let b1 = two_pi_f * ln.c1 * z_base;
        let b0 = two_pi_f * ln.c0 * z_base;
        let r1_pu = ln.r1 / z_base;
        let x1_pu = ln.x1 / z_base;
        let r0_pu = ln.r0 / z_base;
        let x0_pu = ln.x0 / z_base;
        let from_phys = id_to_idx[&ln.from_node];
        let to_phys = id_to_idx[&ln.to_node];

        match (ln.from_status, ln.to_status) {
            (1, 1) => {
                lines.push(Line3Ph {
                    from: from_phys,
                    to: to_phys,
                    r1: r1_pu, x1: x1_pu, b1,
                    r0: r0_pu, x0: x0_pu, b0,
                });
            }
            (1, 0) => {
                lines.push(Line3Ph {
                    from: from_phys, to: from_phys,
                    r1: 0.0, x1: 0.0, b1,
                    r0: 0.0, x0: 0.0, b0,
                });
            }
            (0, 1) => {
                lines.push(Line3Ph {
                    from: to_phys, to: to_phys,
                    r1: 0.0, x1: 0.0, b1,
                    r0: 0.0, x0: 0.0, b0,
                });
            }
            _ => {}
        }
    }

    // Virtual Slack buses + source-impedance lines for each active source.
    for (i, src) in input.data.source.iter().filter(|s| s.status != 0).enumerate() {
        let virtual_phys = n_nodes + i;
        let z_s_pu = src.u_ref * src.u_ref * s_base_va / src.sk;
        let x_s = z_s_pu / (src.rx_ratio * src.rx_ratio + 1.0_f64).sqrt();
        let r_s = src.rx_ratio * x_s;

        for ph in 0..3 {
            let bus_idx = 3 * virtual_phys + ph;
            buses.push(Bus {
                idx: bus_idx,
                bus_type: BusType::Slack,
                voltage_mag: src.u_ref,
                voltage_ang: phase_ang[ph],
                p_spec: 0.0,
                q_spec: 0.0,
                q_min: -f64::INFINITY,
                q_max: f64::INFINITY,
            });
        }

        // Source impedance: symmetric (r0=r1, x0=x1) → diagonal 3×3 block.
        lines.push(Line3Ph {
            from: virtual_phys,
            to: id_to_idx[&src.node],
            r1: r_s, x1: x_s, b1: 0.0,
            r0: r_s, x0: x_s, b0: 0.0,
        });
    }

    (buses, lines, id_to_idx)
}
