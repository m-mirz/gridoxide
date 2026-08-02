use std::collections::HashMap;
use serde::Deserialize;
use super::branch_flow::Terminal;
use super::types::{Bus, BusType, Line, Line3Ph, Transformer, Transformer3PhSeq, ZipKind, ZipTerm};
use super::network::{
    half_open_branch_shunt,
    source_impedance_pu, source_impedance_pu_seq, transformer_tap, transformer_admittances,
    transformer_admittances_ex, transformer_seq_params, tap_ratio_from_voltages, three_winding_star_params,
    ShuntAdm, ShuntAdm3Ph,
};
use num_complex::Complex;

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
    #[serde(default)]
    pub voltage_regulator: Vec<PgmVoltageRegulator>,
    #[serde(default)]
    pub link: Vec<PgmLink>,
    #[serde(default)]
    pub sym_voltage_sensor: Vec<PgmSymVoltageSensor>,
    #[serde(default)]
    pub sym_power_sensor: Vec<PgmSymPowerSensor>,
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
    /// Positive-sequence shunt loss factor (tan δ). PGM forms the shunt
    /// admittance as `2πf·c1·(tan1 + j)` (`line.hpp`), so this is what gives a
    /// line's shunt a conductive part. Defaulted because not every fixture
    /// specifies it, and PGM treats an absent loss factor as zero.
    #[serde(default)]
    pub tan1: f64,
    /// Zero-sequence parameters, needed only for asymmetric calculations —
    /// power-grid-model marks them optional and many of its own symmetric
    /// fixtures omit them entirely.
    ///
    /// Defaulted to NaN rather than to zero or to the positive-sequence value:
    /// `r0 = x0 = 0` would make the zero-sequence admittance infinite, and
    /// `r0 = r1` is a modelling assumption that is wrong for real lines. NaN
    /// keeps the symmetric path (which never reads these) working while making
    /// any asymmetric use of an incomplete fixture loudly wrong instead of
    /// quietly plausible.
    #[serde(default = "nan")]
    pub r0: f64,
    #[serde(default = "nan")]
    pub x0: f64,
    #[serde(default = "nan")]
    pub c0: f64,
    /// Zero-sequence shunt loss factor. Parsed for completeness; the
    /// three-phase conversion (`pgm_to_3ph_network`) still models zero-sequence
    /// shunts as purely susceptive, since `Line3Ph` carries no conductance.
    #[serde(default)]
    pub tan0: f64,
}

/// power-grid-model's own `link` admittance, `1e8 + j1e8` per-unit — recorded
/// for reference, but **not** what gridoxide uses. See [`LINK_Y`].
///
/// Derived in its `common.hpp:83` as `1e6 / (base_power_3p / 10e3 / 10e3)` —
/// "1e6 siemens in a 10 kV network", evaluated once against a 1 MVA base and
/// frozen. It is a *per-unit* constant: the `10e3` is not a live voltage, so it
/// does not track the network's voltage level, and re-scaling it by voltage
/// would double-count what per-unit already removed. powsybl-open-loadflow
/// reaches the same order independently, clamping `|Z|` at `1e-8` p.u.
/// (`LfNetworkParameters.java:39`).
pub const PGM_LINK_Y: Complex<f64> = Complex::new(1e8, 1e8);

/// The admittance gridoxide stamps for a `link`.
///
/// An alias for [`topology::IDEAL_CONNECTION_Y`](crate::topology::IDEAL_CONNECTION_Y),
/// which documents the measurement behind the value and is shared with the
/// branches detected as ideal by impedance rather than declared as such — a
/// link and an undeclared jumper get the same treatment because they are the
/// same thing.
pub use crate::topology::IDEAL_CONNECTION_Y as LINK_Y;

fn one() -> f64 { 1.0 }
fn nan() -> f64 { f64::NAN }
/// power-grid-model's documented `source.sk` default (`components.md`).
fn default_sk() -> f64 { 1e10 }
/// power-grid-model's documented `source.rx_ratio` default (`components.md`).
fn default_rx_ratio() -> f64 { 0.1 }
fn nan3() -> [f64; 3] { [f64::NAN; 3] }

/// Deserializes a number that power-grid-model may also write as the string
/// `"inf"`.
///
/// Its state-estimation fixtures use this for a measurement that is present but
/// carries no information (`inf-measurement-with-injection` and friends): an
/// infinite standard deviation is a zero weight, so the row exists structurally
/// and contributes nothing. JSON has no infinity literal, hence the string.
fn de_f64_or_inf<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{Error, Unexpected};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Num(f64),
        Str(String),
    }

    match Option::<Raw>::deserialize(d)? {
        None => Ok(f64::NAN),
        Some(Raw::Num(v)) => Ok(v),
        Some(Raw::Str(s)) => match s.trim() {
            "inf" | "+inf" | "Infinity" => Ok(f64::INFINITY),
            "-inf" | "-Infinity" => Ok(f64::NEG_INFINITY),
            "nan" | "NaN" => Ok(f64::NAN),
            other => Err(D::Error::invalid_value(
                Unexpected::Str(other),
                &"a number or the string \"inf\"",
            )),
        },
    }
}

#[derive(Deserialize)]
pub struct PgmSource {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    /// Reference voltage, per-unit. Power-grid-model marks this required *only
    /// for power flow* — a state-estimation input has no reason to assert the
    /// source voltage, since that is precisely what is being estimated — so it
    /// falls back to 1.0 p.u. here.
    #[serde(default = "one")]
    pub u_ref: f64,
    /// Short-circuit power, VA. PGM's documented default is 1e10.
    #[serde(default = "default_sk")]
    pub sk: f64,
    /// R-to-X ratio. PGM's documented default is 0.1.
    #[serde(default = "default_rx_ratio")]
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
    /// Specified power. power-grid-model marks this required *only for power
    /// flow*: a state-estimation input deliberately leaves it unset, because
    /// the appliance's power is what the estimator solves for. NaN when
    /// absent, and `pgm_to_network` contributes nothing to the bus injection
    /// for a non-finite value.
    #[serde(default = "nan")]
    pub p_specified: f64,
    #[serde(default = "nan")]
    pub q_specified: f64,
}

#[derive(Deserialize)]
pub struct PgmAsymLoad {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    #[serde(rename = "type", default = "default_load_type")]
    pub load_type: u8,
    /// Per-phase specified power; see `PgmSymLoad::p_specified` for why this
    /// is optional.
    #[serde(default = "nan3")]
    pub p_specified: [f64; 3],
    #[serde(default = "nan3")]
    pub q_specified: [f64; 3],
}

#[derive(Deserialize)]
pub struct PgmSymGen {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    #[serde(rename = "type", default = "default_load_type")]
    pub load_type: u8,
    /// Specified power. power-grid-model marks this required *only for power
    /// flow*: a state-estimation input deliberately leaves it unset, because
    /// the appliance's power is what the estimator solves for. NaN when
    /// absent, and `pgm_to_network` contributes nothing to the bus injection
    /// for a non-finite value.
    #[serde(default = "nan")]
    pub p_specified: f64,
    #[serde(default = "nan")]
    pub q_specified: f64,
}

/// PV-bus control: an active `voltage_regulator` pins its `regulated_object`
/// (a `sym_gen` id) bus's voltage magnitude to `u_ref`, letting Q float —
/// mirrors PGM's own `VoltageRegulator` component (`regulated_object` ==
/// `generator_id` in PGM's C++ `VoltageRegulator::calc_param()`).
///
/// `q_min`/`q_max` (VAr, PGM's own field names, optional — `NaN` if
/// omitted, matching PGM's own "unset" convention for this component) bound
/// the *bus's net* reactive injection, not any one generator's own gross
/// terminal output, since `Bus` (like PGM's own aggregated `q_specified`)
/// only tracks one netted P/Q per node — a real simplification versus a
/// bus with a PV generator *and* a significant co-located load, but exactly
/// consistent with how this parser already aggregates every other P/Q
/// quantity per bus. See `solver::newton_raphson_enforcing_q_limits` for
/// where these get enforced (PV→PQ switching); plain `newton_raphson`
/// ignores them entirely, same as before.
#[derive(Deserialize)]
pub struct PgmVoltageRegulator {
    pub id: u64,
    pub regulated_object: u64,
    pub status: u8,
    pub u_ref: f64,
    #[serde(default = "nan")]
    pub q_min: f64,
    #[serde(default = "nan")]
    pub q_max: f64,
}

#[derive(Deserialize)]
pub struct PgmAsymGen {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    #[serde(rename = "type", default = "default_load_type")]
    pub load_type: u8,
    /// Per-phase specified power; see `PgmSymLoad::p_specified` for why this
    /// is optional.
    #[serde(default = "nan3")]
    pub p_specified: [f64; 3],
    #[serde(default = "nan3")]
    pub q_specified: [f64; 3],
}

#[derive(Deserialize)]
pub struct PgmShunt {
    pub id: u64,
    pub node: u64,
    pub status: u8,
    pub g1: f64,
    pub b1: f64,
    /// Zero-sequence admittance, needed only for asymmetric calculations and
    /// optional in power-grid-model. NaN when absent, for the same reason
    /// `PgmLine`'s zero-sequence fields are.
    #[serde(default = "nan")]
    pub g0: f64,
    #[serde(default = "nan")]
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
    pub tap_min: i32,
    pub tap_max: i32,
    pub tap_nom: i32,
    pub tap_size: f64,
}

/// PGM's `sym_voltage_sensor`: a voltage measurement at a node.
///
/// Both `u_measured` and `u_angle_measured` are optional in practice — PGM's
/// own fixtures omit one or the other, and PGM reads an absent value as NaN
/// rather than as zero. They are defaulted to NaN here for the same reason, and
/// `measurement` skips any reading that isn't finite.
///
/// An angle-carrying voltage sensor is a phasor (PMU) measurement; one without
/// is an ordinary magnitude-only SCADA measurement.
#[derive(Deserialize)]
pub struct PgmSymVoltageSensor {
    pub id: u64,
    /// The measured `node`'s id.
    pub measured_object: u64,
    #[serde(default = "nan")]
    pub u_measured: f64,
    #[serde(default = "nan")]
    pub u_angle_measured: f64,
    /// Standard deviation of the magnitude error, in volts.
    #[serde(default = "nan")]
    pub u_sigma: f64,
    #[serde(default = "nan")]
    pub u_angle_sigma: f64,
}

/// PGM's `sym_power_sensor`: an active/reactive power measurement at one
/// terminal, of either a branch or an appliance.
///
/// `measured_terminal_type` is PGM's `MeasuredTerminalType`: 0 `branch_from`,
/// 1 `branch_to`, 2 `source`, 3 `shunt`, 4 `load`, 5 `generator`, 6/7/8
/// `branch3_1`/`_2`/`_3`, 9 `node`. It decides both what `measured_object`
/// refers to and the sign convention of the reading — see
/// [`measurement`](crate::measurement) for how each is mapped.
///
/// `power_sigma` applies to both components; `p_sigma`/`q_sigma` override it
/// per component when present (only two of power-grid-model's own fixtures use
/// them, but they are the more specific form).
#[derive(Deserialize)]
pub struct PgmSymPowerSensor {
    pub id: u64,
    pub measured_object: u64,
    pub measured_terminal_type: u8,
    #[serde(default = "nan", deserialize_with = "de_f64_or_inf")]
    pub p_measured: f64,
    #[serde(default = "nan", deserialize_with = "de_f64_or_inf")]
    pub q_measured: f64,
    #[serde(default = "nan", deserialize_with = "de_f64_or_inf")]
    pub power_sigma: f64,
    #[serde(default = "nan", deserialize_with = "de_f64_or_inf")]
    pub p_sigma: f64,
    #[serde(default = "nan", deserialize_with = "de_f64_or_inf")]
    pub q_sigma: f64,
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

/// PGM's per-branch output record, as found in `sym_output.json`'s `line`,
/// `transformer` and `link` arrays.
///
/// Powers are in W/var and currents in A — physical units, not per-unit, so
/// comparing against gridoxide's [`branch_flow`](crate::branch_flow) results
/// means scaling those by `s_base_va`. Only the flow fields are modelled;
/// `loading`/`i_from`/`s_from` are derived quantities gridoxide does not
/// compute.
#[derive(Debug, Deserialize)]
pub struct PgmLineOutput {
    pub id: u64,
    #[serde(default)]
    pub energized: u8,
    pub p_from: f64,
    pub q_from: f64,
    pub p_to: f64,
    pub q_to: f64,
}

#[derive(Debug, Deserialize)]
pub struct PgmNodeAsymOutput {
    pub id: u64,
    pub u_pu: [f64; 3],
    pub u_angle: [f64; 3],
}

// ── Public helpers ────────────────────────────────────────────────────────────

/// power-grid-model's `link`: an ideal connection between two nodes, carrying
/// no attributes beyond its endpoints and their statuses.
///
/// Modelled as a branch rather than by merging its endpoints, because a `link`
/// has an identity in power-grid-model's output model — the schema carries a
/// `link` record with its own `p`/`q`/`i` flows, and its fixtures assert them.
/// Merging would delete the branch those numbers describe. See
/// [`crate::topology`] for the policy and
/// `docs/src/powerflow/zero_impedance_branches.md` for the alternatives.
#[derive(Deserialize)]
pub struct PgmLink {
    pub id: u64,
    pub from_node: u64,
    pub to_node: u64,
    pub from_status: u8,
    pub to_status: u8,
}

/// A line's total per-unit shunt admittance, `2πf·c1·(tan1 + j)·z_base`.
///
/// Matches power-grid-model's `line.hpp`:
/// `y1_shunt = 2π·f·c1/base_y·(tan1 + 1i)`. The conductive part comes entirely
/// from the loss factor, so it is zero for the many fixtures that leave `tan1`
/// unset — but not for the state-estimation fixtures, which specify it.
fn line_y_shunt(ln: &PgmLine, omega: f64, z_base: f64) -> Complex<f64> {
    Complex::new(ln.tan1, 1.0) * (omega * ln.c1 * z_base)
}

/// Builds the `Line` representing a branch with exactly one terminal connected.
///
/// PGM does not simply drop the series impedance here: an open-ended branch
/// still presents the near-end shunt half *in parallel with* the series
/// impedance feeding the far-end shunt half, i.e.
/// `y_sh/2 + 1/(1/y_s + 2/y_sh)` — [`half_open_branch_shunt`], the same
/// function [`branch_calc_param`] already applies to half-open transformers.
///
/// Modelling this as the bare shunt instead (which gridoxide did until the
/// branch-flow work) omits the small conductive path through the series
/// resistance: `tests/branch_flow_test.rs`'s `line` fixture has two half-open
/// lines, each carrying 0.68 W in PGM's own expected output, and their absence
/// showed up as a 1.36 W deficit on the healthy line feeding them.
///
/// The result is a self-loop `Line`, which `build_ybus` stamps as a pure
/// diagonal admittance. That representation loses which terminal was the
/// connected one — see [`branch_flow::line_params`](crate::branch_flow::line_params).
fn half_open_line(ln: &PgmLine, omega: f64, z_base: f64, idx: usize) -> Line {
    let y_shunt = line_y_shunt(ln, omega, z_base);
    let (r, x) = crate::topology::clamp_branch_impedance(ln.r1 / z_base, ln.x1 / z_base);
    let y_series = Complex::new(1.0, 0.0) / Complex::new(r, x);
    let y_eq = half_open_branch_shunt(y_series, y_shunt);
    Line { from: idx, to: idx, r: 0.0, x: 0.0, b_shunt: y_eq.im, g_shunt: y_eq.re }
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
            let tap = transformer_tap(t.u1, t.u2, t.tap_side, t.tap_pos, t.tap_min, t.tap_max, t.tap_nom, t.tap_size, t.clock);
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
/// A converted PGM network together with the object-ID maps needed to address
/// its pieces by their original PGM ids.
///
/// [`pgm_to_buses_and_branches`] returns only the three vectors, which is all
/// power flow ever needed: it addresses buses positionally and never has to name
/// a branch. Measurements do — a `sym_power_sensor` carries
/// `measured_object: 4`, a PGM object id — and the mapping from that id to a
/// branch index is *not* recoverable by arithmetic on input positions:
///
/// - a line with both terminals open is dropped entirely (no branch at all),
/// - each active `source` appends a virtual slack bus and an extra branch,
/// - each `three_winding_transformer` expands into three branches plus a star
///   bus,
///
/// so input position and branch index diverge as soon as any of those appear.
/// Hence these maps are recorded during conversion rather than reconstructed.
///
/// All branch indices are *flat*: lines first, then transformers, matching the
/// order [`branch_flow::branch_params`](crate::branch_flow::branch_params)
/// assembles them in.
#[derive(Clone, Debug)]
pub struct PgmNetwork {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub transformers: Vec<Transformer>,
    /// Node id → bus index. Covers physical nodes only; star buses and virtual
    /// slack buses have no PGM id of their own.
    pub node_idx: HashMap<u64, usize>,
    /// `line`/`transformer` id → flat branch index. A line with both terminals
    /// open is absent, since it produces no branch.
    pub branch_idx: HashMap<u64, usize>,
    /// `three_winding_transformer` id → its three flat branch indices, in
    /// side-1/2/3 order.
    pub three_winding_branch_idx: HashMap<u64, [usize; 3]>,
    /// `source` id → the flat branch index of the virtual source-impedance
    /// branch gridoxide synthesizes for it. Inactive sources are absent.
    pub source_branch_idx: HashMap<u64, usize>,
    /// Appliance id (`sym_load`, `asym_load`, `sym_gen`, `asym_gen`, `shunt`,
    /// `source`) → the bus it is attached to. Includes inactive appliances: a
    /// sensor may reference one, and reporting a zero flow for a disconnected
    /// appliance is better than failing to resolve the id at all.
    pub appliance_bus: HashMap<u64, usize>,
    /// Buses whose net injection is identically zero, as a per-bus flag.
    ///
    /// A bus with no load and no generator injects nothing into the network,
    /// and that is *structural knowledge*, not a measurement — it holds exactly,
    /// with no uncertainty. State estimation can use it as a hard constraint
    /// rather than as a very-high-weight pseudo-measurement, which is what
    /// `se::constraints` does.
    ///
    /// Sources and shunts do not disqualify a bus here, because gridoxide
    /// models both structurally: a source's power arrives through its
    /// synthesized branch and a shunt sits on the Y-bus diagonal, so neither
    /// appears in `network::power_injections` at that bus. The virtual slack
    /// buses themselves *are* excluded — that is precisely where the source's
    /// unknown power enters the network.
    ///
    /// Note this is read off the *input document*, not off `Bus::p_spec`. A
    /// state-estimation document leaves `p_specified` unset, so an unmeasured
    /// load looks like zero injection in the converted network while being
    /// nothing of the sort.
    pub zero_injection: Vec<bool>,
    /// Branch ids that collapsed to a self-loop because exactly one terminal
    /// was connected, mapped to *which* PGM terminal is the live one.
    ///
    /// gridoxide represents such a branch as a single diagonal admittance
    /// (`from == to`), which by construction cannot distinguish its two ends,
    /// so the distinction is kept here instead. Use [`resolve_terminal`] rather
    /// than reading this directly.
    ///
    /// [`resolve_terminal`]: PgmNetwork::resolve_terminal
    pub half_open_terminal: HashMap<u64, Terminal>,
}

impl PgmNetwork {
    /// Maps a PGM branch id and the terminal a sensor measures to the branch
    /// index and terminal to evaluate with
    /// [`branch_flow::terminal_flow`](crate::branch_flow::terminal_flow).
    ///
    /// Returns `None` in the two cases where there is no flow to compute:
    ///
    /// - the branch has both terminals open, so no branch exists at all;
    /// - the requested terminal is the *open* end of a half-open branch, whose
    ///   flow is identically zero (which is what PGM reports for it too).
    ///
    /// For the live end of a half-open branch the returned terminal is always
    /// [`Terminal::From`], since the whole equivalent admittance sits there
    /// regardless of which PGM end was connected.
    pub fn resolve_terminal(&self, id: u64, terminal: Terminal) -> Option<(usize, Terminal)> {
        let &branch = self.branch_idx.get(&id)?;
        match self.half_open_terminal.get(&id) {
            Some(&live) if live == terminal => Some((branch, Terminal::From)),
            Some(_) => None,
            None => Some((branch, terminal)),
        }
    }
}

/// Converts a PGM input document into gridoxide's own network types, keeping
/// the object-ID maps — see [`PgmNetwork`] for why those cannot be derived
/// afterwards.
///
/// [`pgm_to_buses_and_branches`] is the same conversion with the maps dropped.
pub fn pgm_to_network(
    input: PgmInput,
    s_base_va: f64,
    freq_hz: f64,
) -> PgmNetwork {
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
        // An appliance with no specified power contributes nothing. PGM marks
        // `p_specified`/`q_specified` required only for power flow, and a
        // state-estimation input leaves them unset precisely because the
        // appliance's power is an unknown — letting NaN through here would
        // poison the bus injection and, from there, the whole Y-bus solve.
        if !p.is_finite() && !q.is_finite() {
            return;
        }
        let p = if p.is_finite() { p } else { 0.0 };
        let q = if q.is_finite() { q } else { 0.0 };
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

    // Physical PQ buses. Nodes with no path to any active source (PGM calls
    // these "de-energized") aren't special-cased here — every active source
    // already becomes its own real `Slack` bus wired in via a genuine Line
    // (below), so `network::connected_components`/`classify` downstream
    // (run inside the solver entry points themselves) correctly discovers
    // such a node's component has no reference bus and pins it to a V=0
    // placeholder without this function needing to duplicate that graph walk.
    let n_nodes = input.data.node.len();
    let mut sorted_ids: Vec<u64> = input.data.node.iter().map(|n| n.id).collect();
    sorted_ids.sort_unstable();
    let mut opt_buses = vec![None::<Bus>; n_nodes];
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
            u_rated: id_to_u_rated[id],
            zip_terms: zip_map.remove(id).unwrap_or_default(),
        });
    }
    let mut buses: Vec<Bus> = opt_buses.into_iter().map(|b| b.unwrap()).collect();

    // PV buses: an active `voltage_regulator` pins its regulated `sym_gen`'s
    // bus voltage magnitude to `u_ref`, letting Q float (see `PgmVoltageRegulator`).
    let sym_gen_node: HashMap<u64, u64> = input.data.sym_gen.iter().map(|g| (g.id, g.node)).collect();
    for vr in &input.data.voltage_regulator {
        if vr.status == 0 { continue; }
        let Some(&node) = sym_gen_node.get(&vr.regulated_object) else { continue };
        let idx = id_to_idx[&node];
        if buses[idx].bus_type == BusType::PQ {
            buses[idx].bus_type = BusType::PV;
            buses[idx].voltage_mag = vr.u_ref;
            buses[idx].q_min = if vr.q_min.is_nan() { -f64::INFINITY } else { vr.q_min / s_base_va };
            buses[idx].q_max = if vr.q_max.is_nan() { f64::INFINITY } else { vr.q_max / s_base_va };
        }
    }

    // Lines. PGM's c1 is the *total* shunt capacitance; build_ybus splits b_shunt/2
    // per end, matching PGM's y_shunt/2. Half-open cases become self-loop shunts.
    let omega = 2.0 * std::f64::consts::PI * freq_hz;
    let mut lines: Vec<Line> = Vec::new();
    // Recorded as we go; see `PgmNetwork` for why these can't be rebuilt later.
    let mut branch_idx: HashMap<u64, usize> = HashMap::new();
    let mut three_winding_branch_idx: HashMap<u64, [usize; 3]> = HashMap::new();
    let mut source_branch_idx: HashMap<u64, usize> = HashMap::new();
    let mut half_open_terminal: HashMap<u64, Terminal> = HashMap::new();
    for ln in &input.data.line {
        match (ln.from_status, ln.to_status) {
            (1, 1) => {
                let z_base = id_to_u_rated[&ln.from_node].powi(2) / s_base_va;
                let y_shunt = line_y_shunt(ln, omega, z_base);
                // A line short enough to be a jumper would otherwise put an
                // unbounded admittance into the Y-bus; see
                // `topology::ZERO_IMPEDANCE_THRESHOLD`.
                let (r, x) = crate::topology::clamp_branch_impedance(
                    ln.r1 / z_base,
                    ln.x1 / z_base,
                );
                branch_idx.insert(ln.id, lines.len());
                lines.push(Line {
                    from: id_to_idx[&ln.from_node],
                    to: id_to_idx[&ln.to_node],
                    r,
                    x,
                    b_shunt: y_shunt.im,
                    g_shunt: y_shunt.re,
                });
            }
            (1, 0) => {
                let z_base = id_to_u_rated[&ln.from_node].powi(2) / s_base_va;
                let idx = id_to_idx[&ln.from_node];
                branch_idx.insert(ln.id, lines.len());
                half_open_terminal.insert(ln.id, Terminal::From);
                lines.push(half_open_line(ln, omega, z_base, idx));
            }
            (0, 1) => {
                let z_base = id_to_u_rated[&ln.to_node].powi(2) / s_base_va;
                let idx = id_to_idx[&ln.to_node];
                branch_idx.insert(ln.id, lines.len());
                half_open_terminal.insert(ln.id, Terminal::To);
                lines.push(half_open_line(ln, omega, z_base, idx));
            }
            // Both terminals open: no branch is created, so this line
            // deliberately gets no `branch_idx` entry.
            _ => {}
        }
    }

    // Transformers — convert physical-unit PGM parameters to system pu.
    let mut transformers: Vec<Transformer> = Vec::new();
    // Positions *within* `transformers`. They can't be turned into flat branch
    // indices yet: the source loop below still appends to `lines`, so the
    // offset (`lines.len()`) isn't final until the very end of this function.
    let mut transformer_pos: HashMap<u64, usize> = HashMap::new();
    let mut three_winding_pos: HashMap<u64, [usize; 3]> = HashMap::new();
    for t in &input.data.transformer {
        let tap = transformer_tap(t.u1, t.u2, t.tap_side, t.tap_pos, t.tap_min, t.tap_max, t.tap_nom, t.tap_size, t.clock);
        let (y_series, y_shunt) = transformer_admittances(t.u2, t.sn, t.uk, t.pk, t.i0, t.p0, s_base_va);
        transformer_pos.insert(t.id, transformers.len());
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

    // Links: ideal connections, stamped as branches rather than merged. See
    // `PgmLink` for why, and `link_admittance` for the value.
    for ln in &input.data.link {
        if ln.from_status == 0 && ln.to_status == 0 {
            continue;
        }
        transformer_pos.insert(ln.id, transformers.len());
        transformers.push(Transformer {
            from: id_to_idx[&ln.from_node],
            to: id_to_idx[&ln.to_node],
            from_status: ln.from_status,
            to_status: ln.to_status,
            y_series: LINK_Y,
            y_shunt: Complex::new(0.0, 0.0),
            tap: Complex::new(1.0, 0.0),
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
        three_winding_pos.insert(
            t.id,
            [transformers.len(), transformers.len() + 1, transformers.len() + 2],
        );

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
        source_branch_idx.insert(src.id, lines.len());
        lines.push(Line { from: virtual_idx, to: id_to_idx[&src.node], r: r_s, x: x_s, b_shunt: 0.0, g_shunt: 0.0 });
    }

    // `lines` is final now, so transformer positions can become flat indices.
    let n_lines = lines.len();
    branch_idx.extend(transformer_pos.into_iter().map(|(id, pos)| (id, n_lines + pos)));
    three_winding_branch_idx.extend(
        three_winding_pos
            .into_iter()
            .map(|(id, legs)| (id, legs.map(|pos| n_lines + pos))),
    );

    // Appliance → bus. Built in one pass at the end rather than inside the
    // accumulation closure above, which only sees active constant-power
    // appliances; a sensor may reference an inactive or ZIP-modelled one.
    let mut appliance_bus: HashMap<u64, usize> = HashMap::new();
    for (id, node) in input.data.sym_load.iter().map(|a| (a.id, a.node))
        .chain(input.data.asym_load.iter().map(|a| (a.id, a.node)))
        .chain(input.data.sym_gen.iter().map(|a| (a.id, a.node)))
        .chain(input.data.asym_gen.iter().map(|a| (a.id, a.node)))
        .chain(input.data.shunt.iter().map(|a| (a.id, a.node)))
        .chain(input.data.source.iter().map(|a| (a.id, a.node)))
    {
        if let Some(&idx) = id_to_idx.get(&node) {
            appliance_bus.insert(id, idx);
        }
    }

    // Zero-injection buses: everything except the ones carrying an active load
    // or generator, and except the virtual slack buses where source power
    // enters. Star buses of three-winding transformers qualify, which is the
    // textbook example of the constraint being worth having.
    let mut zero_injection = vec![true; buses.len()];
    for (node, status) in input.data.sym_load.iter().map(|a| (a.node, a.status))
        .chain(input.data.asym_load.iter().map(|a| (a.node, a.status)))
        .chain(input.data.sym_gen.iter().map(|a| (a.node, a.status)))
        .chain(input.data.asym_gen.iter().map(|a| (a.node, a.status)))
    {
        if status != 0 {
            if let Some(&idx) = id_to_idx.get(&node) {
                zero_injection[idx] = false;
            }
        }
    }
    for &branch in source_branch_idx.values() {
        zero_injection[lines[branch].from] = false;
    }

    PgmNetwork {
        buses,
        lines,
        transformers,
        zero_injection,
        node_idx: id_to_idx,
        branch_idx,
        three_winding_branch_idx,
        source_branch_idx,
        appliance_bus,
        half_open_terminal,
    }
}

/// The [`pgm_to_network`] conversion with the object-ID maps dropped — the
/// shape power flow has always used, kept so its many call sites stay
/// unchanged.
pub fn pgm_to_buses_and_branches(
    input: PgmInput,
    s_base_va: f64,
    freq_hz: f64,
) -> (Vec<Bus>, Vec<Line>, Vec<Transformer>) {
    let net = pgm_to_network(input, s_base_va, freq_hz);
    (net.buses, net.lines, net.transformers)
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
    for id in &sorted_ids {
        let phys = id_to_idx[id];
        let p_arr = p_inj.get(id).copied().unwrap_or([0.0; 3]);
        let q_arr = q_inj.get(id).copied().unwrap_or([0.0; 3]);
        let mut zip_arr = zip_map.remove(id).unwrap_or_default();
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
