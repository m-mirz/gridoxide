use serde::{Deserialize, Serialize};
use nalgebra::Complex;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BusType {
    Slack,
    PV,
    PQ,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bus {
    pub idx: usize,          // index in arrays (0-based)
    pub bus_type: BusType,
    pub voltage_mag: f64,    // Vm (p.u.)
    pub voltage_ang: f64,    // Va (rad)
    pub p_spec: f64,         // P specified (generation - load) in p.u.
    pub q_spec: f64,         // Q specified (generation - load) in p.u.
    pub q_min: f64,          // reactive limits (for PV handling, optional)
    pub q_max: f64,
    #[serde(default)]
    pub u_rated: f64,        // rated line-to-line voltage in V (0 = not set)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Line {
    pub from: usize,
    pub to: usize,
    pub r: f64,
    pub x: f64,
    pub b_shunt: f64, // total line charging
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
