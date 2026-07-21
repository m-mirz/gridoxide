use serde::{Deserialize, Serialize};
use num_complex::Complex;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BusType {
    Slack,
    PV,
    PQ,
}

/// A voltage-dependent (ZIP-model) power term: constant power, constant
/// current (S ∝ |V|), or constant impedance (S ∝ |V|²), evaluated at |V|=1.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ZipKind {
    ConstPower,
    ConstCurrent,
    ConstImpedance,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ZipTerm {
    pub s_const: Complex<f64>,
    pub kind: ZipKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bus {
    pub idx: usize,          // index in arrays (0-based)
    pub bus_type: BusType,
    pub voltage_mag: f64,    // Vm (p.u.)
    pub voltage_ang: f64,    // Va (rad)
    pub p_spec: f64,         // P specified (generation - load) in p.u., constant-power part
    pub q_spec: f64,         // Q specified (generation - load) in p.u., constant-power part
    pub q_min: f64,          // PV bus reactive limits, enforced only by
    pub q_max: f64,          // solver::newton_raphson_enforcing_q_limits
    #[serde(default)]
    pub u_rated: f64,        // rated line-to-line voltage in V (0 = not set)
    /// Additional voltage-dependent (constant-current/-impedance) injection terms,
    /// summed on top of `p_spec`/`q_spec` at the current voltage estimate.
    #[serde(default)]
    pub zip_terms: Vec<ZipTerm>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Line {
    pub from: usize,
    pub to: usize,
    pub r: f64,
    pub x: f64,
    pub b_shunt: f64, // total line charging
    #[serde(default)]
    pub g_shunt: f64, // total shunt conductance (CGMES ACLineSegment.gch; usually 0)
}

/// Two-winding transformer parameters in per-unit (system base, to-side voltage base).
///
/// `tap` = k · exp(j · clock · π/6) where k is the off-nominal voltage-magnitude ratio
/// and the argument encodes the vector-group phase shift. `tap.norm()` gives k.
#[derive(Clone, Debug)]
pub struct Transformer {
    pub from: usize,
    pub to: usize,
    pub from_status: u8,
    pub to_status: u8,
    pub y_series: Complex<f64>,
    pub y_shunt: Complex<f64>,
    pub tap: Complex<f64>,
}

/// Three-phase line parameters in per-unit.
/// Positive- and zero-sequence values are stored separately;
/// `build_ybus_3ph` converts them to the phase-domain 3×3 admittance matrix.
/// `b1`/`b0` are the *total* shunt susceptances (ω·c·Z_base); the π-model
/// splits them equally to both terminals, analogous to `Line::b_shunt`.
/// Sequence-domain (0, 1, 2) branch admittance parameters for an asymmetric
/// (3-phase) transformer branch. Each `[Complex<f64>; 4]` holds the four
/// branch entries `[yff, yft, ytf, ytt]` for that sequence; `network::
/// stamp_transformers_3ph` converts them to phase-domain 3×3 blocks via the
/// Fortescue transform and stamps them into the Y-bus.
#[derive(Clone, Debug)]
pub struct Transformer3PhSeq {
    pub from: usize,
    pub to: usize,
    pub y0: [Complex<f64>; 4],
    pub y1: [Complex<f64>; 4],
    pub y2: [Complex<f64>; 4],
}

#[derive(Clone, Debug)]
pub struct Line3Ph {
    pub from: usize, // physical node index
    pub to: usize,
    pub r1: f64,
    pub x1: f64,
    pub b1: f64, // total positive-sequence shunt susceptance (p.u.)
    pub r0: f64,
    pub x0: f64,
    pub b0: f64, // total zero-sequence shunt susceptance (p.u.)
}
