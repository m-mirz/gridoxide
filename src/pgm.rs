use std::collections::{HashMap, HashSet};
use serde::Deserialize;
use super::types::{Bus, BusType, Line, Line3Ph, Transformer, Transformer3PhSeq, ZipKind, ZipTerm};
use super::network::{
    source_impedance_pu, source_impedance_pu_seq, transformer_tap, transformer_admittances,
    transformer_admittances_ex, transformer_seq_params, tap_ratio_from_voltages, three_winding_star_params,
    ShuntAdm, ShuntAdm3Ph,
};
use nalgebra::Complex;

// ── Input structs ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct PgmInput {
    pub data: PgmData,
}

#[derive(Deserialize)]
pub struct PgmData {
    pub node: Vec<PgmNode>,
    #[serde(default)]
    pub line: Vec<PgmLine>,
    pub source: Vec<PgmSource>,
    #[serde(default)]
    pub sym_load: Vec<PgmSymLoad>,
    #[serde(default)]
    pub asym_load: Vec<PgmAsymLoad>,
    #[serde(default)]
    pub sym_gen: Vec<PgmSymGen>,
    #[serde(default)]
    pub asym_gen: Vec<PgmAsymGen>,
    #[serde(default)]
    pub shunt: Vec<PgmShunt>,
    #[serde(default)]
    pub transformer: Vec<PgmTransformer>,
    #[serde(default)]
    pub three_winding_transformer: Vec<PgmThreeWindingTransformer>,
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

fn one() -> f64 { 1.0 }

#[derive(Deserialize)]
pub struct PgmSource {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    pub u_ref: f64,
    pub sk: f64,
    pub rx_ratio: f64,
    #[serde(default = "one")]
    pub z01_ratio: f64,
}

/// PGM's ZIP-model type code: 0 = constant power, 1 = constant impedance
/// (S ∝ |V|²), 2 = constant current (S ∝ |V|).
fn default_load_type() -> u8 { 0 }

#[derive(Deserialize)]
pub struct PgmSymLoad {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    #[serde(rename = "type", default = "default_load_type")]
    pub load_type: u8,
    pub p_specified: f64,
    pub q_specified: f64,
}

#[derive(Deserialize)]
pub struct PgmAsymLoad {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    #[serde(rename = "type", default = "default_load_type")]
    pub load_type: u8,
    pub p_specified: [f64; 3],
    pub q_specified: [f64; 3],
}

#[derive(Deserialize)]
pub struct PgmSymGen {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    #[serde(rename = "type", default = "default_load_type")]
    pub load_type: u8,
    pub p_specified: f64,
    pub q_specified: f64,
}

#[derive(Deserialize)]
pub struct PgmAsymGen {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    #[serde(rename = "type", default = "default_load_type")]
    pub load_type: u8,
    pub p_specified: [f64; 3],
    pub q_specified: [f64; 3],
}

#[derive(Deserialize)]
pub struct PgmShunt {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    pub g1: f64,
    pub b1: f64,
    pub g0: f64,
    pub b0: f64,
}

#[derive(Deserialize)]
pub struct PgmThreeWindingTransformer {
    pub id: u64,
    pub node_1: u64,
    pub node_2: u64,
    pub node_3: u64,
    pub status_1: u8,
    pub status_2: u8,
    pub status_3: u8,
    pub u1: f64,
    pub u2: f64,
    pub u3: f64,
    pub sn_1: f64,
    pub sn_2: f64,
    pub sn_3: f64,
    pub uk_12: f64,
    pub uk_13: f64,
    pub uk_23: f64,
    pub pk_12: f64,
    pub pk_13: f64,
    pub pk_23: f64,
    pub i0: f64,
    pub p0: f64,
    pub clock_12: i32,
    pub clock_13: i32,
    pub tap_side: u8,
    pub tap_pos: i32,
    pub tap_min: i32,
    pub tap_max: i32,
    pub tap_nom: i32,
    pub tap_size: f64,
}

#[derive(Deserialize, Clone)]
pub struct PgmTransformer {
    pub id: u64,
    pub from_node: u64,
    pub to_node: u64,
    pub from_status: u8,
    pub to_status: u8,
    pub u1: f64,
    pub u2: f64,
    pub sn: f64,
    pub uk: f64,
    pub pk: f64,
    pub i0: f64,
    pub p0: f64,
    #[serde(default)]
    pub winding_from: u8,
    #[serde(default)]
    pub winding_to: u8,
    pub clock: i32,
    pub tap_side: u8,
    pub tap_pos: i32,
    pub tap_nom: i32,
    pub tap_size: f64,
}

// ── Output structs (used by integration tests) ────────────────────────────────

#[derive(Deserialize)]
pub struct PgmOutput<N> {
    pub data: PgmOutputData<N>,
}

#[derive(Deserialize)]
pub struct PgmOutputData<N> {
    pub node: Vec<N>,
}

/// A batch output document: one `PgmOutputData<N>` per scenario, in scenario order.
#[derive(Deserialize)]
pub struct PgmBatchOutput<N> {
    pub data: Vec<PgmOutputData<N>>,
}

#[derive(Deserialize)]
pub struct PgmNodeOutput {
    pub id: u64,
    pub u_pu: f64,
    pub u_angle: f64,
    #[serde(default)]
    pub u: f64,
}

#[derive(Deserialize)]
pub struct PgmNodeAsymOutput {
    pub id: u64,
    pub u_pu: [f64; 3],
    pub u_angle: [f64; 3],
}

// ── Public helpers ────────────────────────────────────────────────────────────

/// Returns the set of node IDs reachable from an active source via fully-closed
/// (`from_status == 1 && to_status == 1`) lines/transformers. Nodes outside this
/// set are "de-energized" (PGM's term for a node with no path to any source) and
/// are reported with zero voltage rather than solved as an ordinary PQ bus.
fn energized_node_ids(input: &PgmInput) -> HashSet<u64> {
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for ln in &input.data.line {
        if ln.from_status == 1 && ln.to_status == 1 {
            adj.entry(ln.from_node).or_default().push(ln.to_node);
            adj.entry(ln.to_node).or_default().push(ln.from_node);
        }
    }
    for t in &input.data.transformer {
        if t.from_status == 1 && t.to_status == 1 {
            adj.entry(t.from_node).or_default().push(t.to_node);
            adj.entry(t.to_node).or_default().push(t.from_node);
        }
    }
    for t in &input.data.three_winding_transformer {
        let sides = [(t.node_1, t.status_1), (t.node_2, t.status_2), (t.node_3, t.status_3)];
        for i in 0..3 {
            for j in (i + 1)..3 {
                let (ni, si) = sides[i];
                let (nj, sj) = sides[j];
                if si == 1 && sj == 1 {
                    adj.entry(ni).or_default().push(nj);
                    adj.entry(nj).or_default().push(ni);
                }
            }
        }
    }

    let mut visited: HashSet<u64> = HashSet::new();
    let mut stack: Vec<u64> = input.data.source.iter()
        .filter(|s| s.status != 0)
        .map(|s| s.node)
        .collect();
    while let Some(n) = stack.pop() {
        if visited.insert(n) {
            if let Some(neighbors) = adj.get(&n) {
                stack.extend(neighbors.iter().filter(|nb| !visited.contains(nb)));
            }
        }
    }
    visited
}

/// Returns a stable node-ID → 0-based-index map (sorted by node ID).
pub fn node_id_to_idx(input: &PgmInput) -> HashMap<u64, usize> {
    let mut ids: Vec<u64> = input.data.node.iter().map(|n| n.id).collect();
    ids.sort_unstable();
    ids.into_iter().enumerate().map(|(idx, id)| (id, idx)).collect()
}

/// Converts active `shunt` entries to per-unit self-admittances, for stamping
/// into a 1-phase Y-bus via `network::stamp_shunts`. Call before the owning
/// `PgmInput` is consumed by `pgm_to_buses_and_branches`.
pub fn pgm_shunts_1ph(
    input: &PgmInput,
    id_to_idx: &HashMap<u64, usize>,
    s_base_va: f64,
) -> Vec<ShuntAdm> {
    let id_to_u_rated: HashMap<u64, f64> = input.data.node.iter().map(|n| (n.id, n.u_rated)).collect();
    input.data.shunt.iter()
        .filter(|s| s.status != 0)
        .map(|s| {
            let u_rated = id_to_u_rated[&s.node];
            let base_y = s_base_va / (u_rated * u_rated);
            ShuntAdm { at: id_to_idx[&s.node], y: Complex::new(s.g1, s.b1) / base_y }
        })
        .collect()
}

/// Converts active `shunt` entries to per-unit sequence self-admittances, for
/// stamping into a 3-phase Y-bus via `network::stamp_shunts_3ph`. `at` is the
/// physical node index (i.e. `id_to_idx` from `pgm_to_3ph_network`, not the
/// 3N-bus index — `stamp_shunts_3ph` multiplies by 3 internally).
pub fn pgm_shunts_3ph(
    input: &PgmInput,
    id_to_idx: &HashMap<u64, usize>,
    s_base_va: f64,
) -> Vec<ShuntAdm3Ph> {
    let id_to_u_rated: HashMap<u64, f64> = input.data.node.iter().map(|n| (n.id, n.u_rated)).collect();
    input.data.shunt.iter()
        .filter(|s| s.status != 0)
        .map(|s| {
            let u_rated = id_to_u_rated[&s.node];
            let base_y = s_base_va / (u_rated * u_rated);
            ShuntAdm3Ph {
                at: id_to_idx[&s.node],
                y1: Complex::new(s.g1, s.b1) / base_y,
                y0: Complex::new(s.g0, s.b0) / base_y,
            }
        })
        .collect()
}

/// Converts active `transformer` entries to sequence-domain branch admittance
/// parameters, for stamping into a 3-phase Y-bus via
/// `network::stamp_transformers_3ph`. Call before the owning `PgmInput` is
/// consumed by `pgm_to_3ph_network`. Only Dyn (`winding_from`=delta,
/// `winding_to`=wye_n) and YNyn (`winding_from`=`winding_to`=wye_n)
/// transformers are supported — see `network::transformer_seq_params`.
pub fn pgm_transformers_3ph(
    input: &PgmInput,
    id_to_idx: &HashMap<u64, usize>,
    s_base_va: f64,
) -> Vec<Transformer3PhSeq> {
    input.data.transformer.iter()
        .map(|t| {
            let tap = transformer_tap(t.u1, t.u2, t.tap_side, t.tap_pos, t.tap_nom, t.tap_size, t.clock);
            let (y_series, y_shunt) = transformer_admittances(t.u2, t.sn, t.uk, t.pk, t.i0, t.p0, s_base_va);
            let (y0, y1, y2) = transformer_seq_params(
                y_series, y_shunt, tap, t.from_status, t.to_status,
                t.winding_from, t.winding_to, t.sn, t.uk, s_base_va, t.clock,
            );
            Transformer3PhSeq {
                from: id_to_idx[&t.from_node],
                to: id_to_idx[&t.to_node],
                y0, y1, y2,
            }
        })
        .collect()
}

/// Converts a PGM input document to per-unit buses, lines, and transformers.
///
/// `s_base_va` — system base power in VA (e.g. 1e6 for 1 MVA).
/// `freq_hz`   — grid frequency in Hz (e.g. 50.0 or 60.0).
///
/// All PGM nodes become PQ buses. Each active source is modelled as a virtual
/// Slack bus appended after the physical nodes, connected via a source-impedance
/// `Line`. Transformers are returned as `Transformer` values in per-unit on the
/// system base; stamp them into a Y-bus with `network::stamp_transformers`.
pub fn pgm_to_buses_and_branches(
    input: PgmInput,
    s_base_va: f64,
    freq_hz: f64,
) -> (Vec<Bus>, Vec<Line>, Vec<Transformer>) {
    let id_to_idx = node_id_to_idx(&input);
    let id_to_u_rated: HashMap<u64, f64> = input.data.node.iter()
        .map(|n| (n.id, n.u_rated))
        .collect();

    // Accumulate per-node net injection from active loads (load = negative
    // injection) and gens (generation = positive injection). Constant-power
    // (type 0) entries fold into the flat p_inj/q_inj maps; constant-current/
    // -impedance (type 1/2) entries become voltage-dependent ZIP terms on Bus.
    let mut p_inj: HashMap<u64, f64> = HashMap::new();
    let mut q_inj: HashMap<u64, f64> = HashMap::new();
    let mut zip_map: HashMap<u64, Vec<ZipTerm>> = HashMap::new();
    let mut accumulate = |node: u64, load_type: u8, p: f64, q: f64, sign: f64| {
        let s = Complex::new(p, q) * sign / s_base_va;
        match load_type {
            1 => zip_map.entry(node).or_default().push(ZipTerm { s_const: s, kind: ZipKind::ConstImpedance }),
            2 => zip_map.entry(node).or_default().push(ZipTerm { s_const: s, kind: ZipKind::ConstCurrent }),
            _ => {
                *p_inj.entry(node).or_insert(0.0) += s.re;
                *q_inj.entry(node).or_insert(0.0) += s.im;
            }
        }
    };
    for load in &input.data.sym_load {
        if load.status == 0 { continue; }
        accumulate(load.node, load.load_type, load.p_specified, load.q_specified, -1.0);
    }
    for sgen in &input.data.sym_gen {
        if sgen.status == 0 { continue; }
        accumulate(sgen.node, sgen.load_type, sgen.p_specified, sgen.q_specified, 1.0);
    }
    // Asymmetric (3-phase) loads/gens are folded into their total three-phase
    // power for the symmetric (positive-sequence) equivalent.
    for load in &input.data.asym_load {
        if load.status == 0 { continue; }
        let p: f64 = load.p_specified.iter().sum();
        let q: f64 = load.q_specified.iter().sum();
        accumulate(load.node, load.load_type, p, q, -1.0);
    }
    for agen in &input.data.asym_gen {
        if agen.status == 0 { continue; }
        let p: f64 = agen.p_specified.iter().sum();
        let q: f64 = agen.q_specified.iter().sum();
        accumulate(agen.node, agen.load_type, p, q, 1.0);
    }

    // Physical PQ buses. Nodes with no path to any active source are
    // "de-energized" — reported at zero voltage and excluded from the NR solve
    // by modelling them as a fixed (Slack-like) bus at V=0.
    let energized = energized_node_ids(&input);
    let n_nodes = input.data.node.len();
    let mut sorted_ids: Vec<u64> = input.data.node.iter().map(|n| n.id).collect();
    sorted_ids.sort_unstable();
    let mut opt_buses = vec![None::<Bus>; n_nodes];
    for id in &sorted_ids {
        let idx = id_to_idx[id];
        let is_energized = energized.contains(id);
        opt_buses[idx] = Some(Bus {
            idx,
            bus_type: if is_energized { BusType::PQ } else { BusType::Slack },
            voltage_mag: if is_energized { 1.0 } else { 0.0 },
            voltage_ang: 0.0,
            p_spec: *p_inj.get(id).unwrap_or(&0.0),
            q_spec: *q_inj.get(id).unwrap_or(&0.0),
            q_min: -f64::INFINITY,
            q_max: f64::INFINITY,
            u_rated: id_to_u_rated[id],
            zip_terms: zip_map.remove(id).unwrap_or_default(),
        });
    }
    let mut buses: Vec<Bus> = opt_buses.into_iter().map(|b| b.unwrap()).collect();

    // Lines. PGM's c1 is the *total* shunt capacitance; build_ybus splits b_shunt/2
    // per end, matching PGM's y_shunt/2. Half-open cases become self-loop shunts.
    let omega = 2.0 * std::f64::consts::PI * freq_hz;
    let mut lines: Vec<Line> = Vec::new();
    for ln in &input.data.line {
        match (ln.from_status, ln.to_status) {
            (1, 1) => {
                let z_base = id_to_u_rated[&ln.from_node].powi(2) / s_base_va;
                lines.push(Line {
                    from: id_to_idx[&ln.from_node],
                    to: id_to_idx[&ln.to_node],
                    r: ln.r1 / z_base,
                    x: ln.x1 / z_base,
                    b_shunt: omega * ln.c1 * z_base,
                });
            }
            (1, 0) => {
                let z_base = id_to_u_rated[&ln.from_node].powi(2) / s_base_va;
                let idx = id_to_idx[&ln.from_node];
                lines.push(Line { from: idx, to: idx, r: 0.0, x: 0.0,
                    b_shunt: omega * ln.c1 * z_base });
            }
            (0, 1) => {
                let z_base = id_to_u_rated[&ln.to_node].powi(2) / s_base_va;
                let idx = id_to_idx[&ln.to_node];
                lines.push(Line { from: idx, to: idx, r: 0.0, x: 0.0,
                    b_shunt: omega * ln.c1 * z_base });
            }
            _ => {}
        }
    }

    // Transformers — convert physical-unit PGM parameters to system pu.
    let mut transformers: Vec<Transformer> = Vec::new();
    for t in &input.data.transformer {
        let tap = transformer_tap(t.u1, t.u2, t.tap_side, t.tap_pos, t.tap_nom, t.tap_size, t.clock);
        let (y_series, y_shunt) = transformer_admittances(t.u2, t.sn, t.uk, t.pk, t.i0, t.p0, s_base_va);
        transformers.push(Transformer {
            from: id_to_idx[&t.from_node],
            to: id_to_idx[&t.to_node],
            from_status: t.from_status,
            to_status: t.to_status,
            y_series,
            y_shunt,
            tap,
        });
    }

    // Three-winding transformers — modelled as three virtual 2-winding
    // `Transformer` legs from each physical node to a synthesized internal
    // star bus (PGM's star-equivalent model: T1/T2/T3 all connect to a
    // shared dummy node whose per-unit base tracks side 1's rated voltage).
    // Star buses are appended right after the physical nodes and before
    // source virtual-slack buses, so the source loop's index offset below
    // must account for them via `n_3wdg`.
    for (i, t) in input.data.three_winding_transformer.iter().enumerate() {
        let star_idx = n_nodes + i;
        let u1_rated = id_to_u_rated[&t.node_1];
        let u2_rated = id_to_u_rated[&t.node_2];
        let u3_rated = id_to_u_rated[&t.node_3];

        // Tap adjustment applies to exactly one of u1/u2/u3, matching `tap_side`.
        let tap_direction = if t.tap_max > t.tap_min { 1.0 } else { -1.0 };
        let delta = tap_direction * (t.tap_pos - t.tap_nom) as f64 * t.tap_size;
        let (u1_local, u2_local, u3_local) = match t.tap_side {
            0 => (t.u1 + delta, t.u2, t.u3),
            1 => (t.u1, t.u2 + delta, t.u3),
            _ => (t.u1, t.u2, t.u3 + delta),
        };

        let ((uk_t1, uk_t2, uk_t3), (pk_t1, pk_t2, pk_t3)) = three_winding_star_params(
            t.sn_1, t.sn_2, t.sn_3, t.uk_12, t.uk_13, t.uk_23, t.pk_12, t.pk_13, t.pk_23,
        );

        buses.push(Bus {
            idx: star_idx,
            bus_type: BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: -f64::INFINITY,
            q_max: f64::INFINITY,
            u_rated: u1_rated,
            zip_terms: Vec::new(),
        });

        // All three legs' "to"-side nameplate/base is the star node, whose
        // nameplate voltage is u1_local (tracks side 1's tap) and whose
        // per-unit base is pinned to u1_rated (physical, fixed) — this is
        // why `transformer_admittances_ex` always takes (u1_local, u1_rated)
        // regardless of which leg is being built.
        let t1_tap = tap_ratio_from_voltages(u1_local * u1_rated, u1_local * u1_rated, 0);
        let (t1_series, t1_shunt) =
            transformer_admittances_ex(u1_local, u1_rated, t.sn_1, uk_t1, pk_t1, t.i0, t.p0, s_base_va);
        transformers.push(Transformer {
            from: id_to_idx[&t.node_1], to: star_idx,
            from_status: t.status_1, to_status: 1,
            y_series: t1_series, y_shunt: t1_shunt, tap: t1_tap,
        });

        let t2_tap = tap_ratio_from_voltages(u2_local * u1_rated, u1_local * u2_rated, 12 - t.clock_12);
        let (t2_series, t2_shunt) =
            transformer_admittances_ex(u1_local, u1_rated, t.sn_2, uk_t2, pk_t2, 0.0, 0.0, s_base_va);
        transformers.push(Transformer {
            from: id_to_idx[&t.node_2], to: star_idx,
            from_status: t.status_2, to_status: 1,
            y_series: t2_series, y_shunt: t2_shunt, tap: t2_tap,
        });

        let t3_tap = tap_ratio_from_voltages(u3_local * u1_rated, u1_local * u3_rated, 12 - t.clock_13);
        let (t3_series, t3_shunt) =
            transformer_admittances_ex(u1_local, u1_rated, t.sn_3, uk_t3, pk_t3, 0.0, 0.0, s_base_va);
        transformers.push(Transformer {
            from: id_to_idx[&t.node_3], to: star_idx,
            from_status: t.status_3, to_status: 1,
            y_series: t3_series, y_shunt: t3_shunt, tap: t3_tap,
        });
    }
    let n_3wdg = input.data.three_winding_transformer.len();

    // Virtual Slack bus + source-impedance Line for each active source.
    for (i, src) in input.data.source.iter().filter(|s| s.status != 0).enumerate() {
        let virtual_idx = n_nodes + n_3wdg + i;
        let (r_s, x_s) = source_impedance_pu(src.sk, src.rx_ratio, s_base_va);
        buses.push(Bus {
            idx: virtual_idx,
            bus_type: BusType::Slack,
            voltage_mag: src.u_ref,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: -f64::INFINITY,
            q_max: f64::INFINITY,
            u_rated: id_to_u_rated[&src.node],
            zip_terms: Vec::new(),
        });
        lines.push(Line { from: virtual_idx, to: id_to_idx[&src.node], r: r_s, x: x_s, b_shunt: 0.0 });
    }

    (buses, lines, transformers)
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

    // Per-node, per-phase net injection [phase_a, phase_b, phase_c] in p.u.,
    // plus per-phase ZIP terms for constant-current/-impedance loads and gens.
    // The phase-domain power flow equations naturally work in units of S_base/3
    // (per-phase base), so per-phase Watts must be divided by s_base_va/3.
    let mut p_inj: HashMap<u64, [f64; 3]> = HashMap::new();
    let mut q_inj: HashMap<u64, [f64; 3]> = HashMap::new();
    let mut zip_map: HashMap<u64, [Vec<ZipTerm>; 3]> = HashMap::new();
    let s_base_1ph = s_base_va / 3.0;
    let mut accumulate3 = |node: u64, load_type: u8, p: [f64; 3], q: [f64; 3], sign: f64, base: f64| {
        match load_type {
            1 | 2 => {
                let kind = if load_type == 1 { ZipKind::ConstImpedance } else { ZipKind::ConstCurrent };
                let entry = zip_map.entry(node).or_insert_with(|| [Vec::new(), Vec::new(), Vec::new()]);
                for ph in 0..3 {
                    entry[ph].push(ZipTerm { s_const: Complex::new(p[ph], q[ph]) * sign / base, kind });
                }
            }
            _ => {
                let pe = p_inj.entry(node).or_insert([0.0; 3]);
                let qe = q_inj.entry(node).or_insert([0.0; 3]);
                for ph in 0..3 {
                    pe[ph] += sign * p[ph] / base;
                    qe[ph] += sign * q[ph] / base;
                }
            }
        }
    };
    for load in &input.data.asym_load {
        if load.status == 0 { continue; }
        accumulate3(load.node, load.load_type, load.p_specified, load.q_specified, -1.0, s_base_1ph);
    }
    for agen in &input.data.asym_gen {
        if agen.status == 0 { continue; }
        accumulate3(agen.node, agen.load_type, agen.p_specified, agen.q_specified, 1.0, s_base_1ph);
    }
    // sym_load/sym_gen p_specified is 3-phase total; each phase gets 1/3 of
    // total, and P_1ph_pu = (P_total/3) / (s_base/3) = P_total / s_base.
    for load in &input.data.sym_load {
        if load.status == 0 { continue; }
        let p = [load.p_specified; 3];
        let q = [load.q_specified; 3];
        accumulate3(load.node, load.load_type, p, q, -1.0, s_base_va);
    }
    for sgen in &input.data.sym_gen {
        if sgen.status == 0 { continue; }
        let p = [sgen.p_specified; 3];
        let q = [sgen.q_specified; 3];
        accumulate3(sgen.node, sgen.load_type, p, q, 1.0, s_base_va);
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
            u_rated: 0.0,
            zip_terms: Vec::new(),
        });
    }
    let energized = energized_node_ids(&input);
    for id in &sorted_ids {
        let phys = id_to_idx[id];
        let is_energized = energized.contains(id);
        let p_arr = p_inj.get(id).copied().unwrap_or([0.0; 3]);
        let q_arr = q_inj.get(id).copied().unwrap_or([0.0; 3]);
        let mut zip_arr = zip_map.remove(id).unwrap_or_default();
        for ph in 0..3 {
            let bus_idx = 3 * phys + ph;
            buses[bus_idx] = Bus {
                idx: bus_idx,
                bus_type: if is_energized { BusType::PQ } else { BusType::Slack },
                voltage_mag: if is_energized { 1.0 } else { 0.0 },
                voltage_ang: if is_energized { phase_ang[ph] } else { 0.0 },
                p_spec: p_arr[ph],
                q_spec: q_arr[ph],
                q_min: -f64::INFINITY,
                q_max: f64::INFINITY,
                u_rated: id_to_u_rated[id],
                zip_terms: std::mem::take(&mut zip_arr[ph]),
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
        let (r1_s, x1_s, r0_s, x0_s) =
            source_impedance_pu_seq(src.sk, src.rx_ratio, src.z01_ratio, s_base_va);

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
                u_rated: id_to_u_rated[&src.node],
                zip_terms: Vec::new(),
            });
        }

        lines.push(Line3Ph {
            from: virtual_phys,
            to: id_to_idx[&src.node],
            r1: r1_s, x1: x1_s, b1: 0.0,
            r0: r0_s, x0: x0_s, b0: 0.0,
        });
    }

    (buses, lines, id_to_idx)
}
