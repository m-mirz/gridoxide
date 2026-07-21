//! CGMES (Common Grid Model Exchange Standard) dataset loading and
//! conversion, built on cimoxide's `cimdecoder`/`cimstructs` crates — see
//! `CIMOXIDE_PROVENANCE.md` for why this is a pinned git dependency rather
//! than a crates.io one.
//!
//! Requires the TP profile: `TopologicalNode` is used directly as gridoxide's
//! `Bus`, so ConnectivityNode/switch-state topology processing is assumed
//! already resolved upstream (the standard EQ+SSH+TP+SV "solved case" profile
//! bundle).

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex;

pub use cimdecoder::CimDataset;
use cimstructs::{
    ACLineSegment, BaseVoltage, EnergyConsumer, EquivalentInjection, LinearShuntCompensator,
    NonlinearShuntCompensator, NonlinearShuntCompensatorPoint, PhaseTapChangerAsymmetrical,
    PhaseTapChangerNonLinear, PhaseTapChangerSymmetrical, PowerTransformerEnd, RatioTapChanger,
    RegulatingControl, StaticVarCompensator, SynchronousMachine, Terminal, TopologicalIsland,
    TopologicalNode,
};

use crate::network::ShuntAdm;
use crate::types::{Bus, BusType, Line, Transformer};

/// Loads and merges a set of CGMES profile files (e.g. EQ, SSH, TP, SV) into
/// one `CimDataset`, keyed by MRID across all of them.
pub fn load_profiles(paths: &[&Path]) -> Result<CimDataset, Box<dyn std::error::Error>> {
    CimDataset::decode_files(paths)
}

#[derive(Debug)]
pub enum CgmesError {
    /// A reference (e.g. `Terminal.TopologicalNode`) didn't resolve to any
    /// decoded element of the expected type — either a genuinely dangling
    /// reference, or (more likely) a required profile file wasn't loaded.
    UnresolvedReference { from_type: &'static str, from_mrid: String, field: &'static str },
    /// A field that's required for conversion (though CGMES's own schema
    /// always makes it `Option`) was absent.
    MissingField { type_name: &'static str, mrid: String, field: &'static str },
    /// No `TopologicalNode` entries at all — the TP profile wasn't loaded.
    NoTopologicalNodes,
    /// No `TopologicalIsland.AngleRefTopologicalNode` found — the SV profile
    /// wasn't loaded, or the dataset genuinely has no angle reference.
    NoAngleReference,
    /// A `PowerTransformer` with a winding count this converter doesn't
    /// handle (only 2- and 3-winding are supported), or another shape this
    /// v1 converter doesn't attempt to guess at (e.g. tap changers on both
    /// ends of the same 2-winding transformer).
    UnsupportedTransformer { mrid: String, reason: String },
}

impl std::fmt::Display for CgmesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CgmesError::UnresolvedReference { from_type, from_mrid, field } => {
                write!(f, "{from_type} {from_mrid}: {field} does not resolve to a decoded element")
            }
            CgmesError::MissingField { type_name, mrid, field } => {
                write!(f, "{type_name} {mrid}: required field {field} is missing")
            }
            CgmesError::NoTopologicalNodes => {
                write!(f, "dataset has no TopologicalNode entries (is the TP profile loaded?)")
            }
            CgmesError::NoAngleReference => {
                write!(f, "no TopologicalIsland.AngleRefTopologicalNode found (is the SV profile loaded?)")
            }
            CgmesError::UnsupportedTransformer { mrid, reason } => {
                write!(f, "PowerTransformer {mrid}: {reason}")
            }
        }
    }
}

impl std::error::Error for CgmesError {}

fn get<'a, T: 'static>(ds: &'a CimDataset, mrid: &str) -> Option<&'a T> {
    ds.entries.get(mrid)?.element.as_any().downcast_ref::<T>()
}

fn require<'a, T: 'static>(
    ds: &'a CimDataset, mrid: &str, from_type: &'static str, from_mrid: &str, field: &'static str,
) -> Result<&'a T, CgmesError> {
    get(ds, mrid).ok_or_else(|| CgmesError::UnresolvedReference {
        from_type, from_mrid: from_mrid.to_string(), field,
    })
}

fn missing(type_name: &'static str, mrid: &str, field: &'static str) -> CgmesError {
    CgmesError::MissingField { type_name, mrid: mrid.to_string(), field }
}

fn by_type<'a>(ds: &'a CimDataset, type_name: &str) -> &'a [String] {
    ds.by_type.get(type_name).map(|v| v.as_slice()).unwrap_or(&[])
}

/// `target_value_unit_multiplier`'s URI suffix -> multiplier factor.
fn unit_multiplier(uri: Option<&str>) -> f64 {
    match uri.and_then(|u| u.rsplit('.').next()) {
        Some("Y") => 1e24, Some("Z") => 1e21, Some("E") => 1e18, Some("P") => 1e15,
        Some("T") => 1e12, Some("G") => 1e9, Some("M") => 1e6, Some("k") => 1e3,
        Some("h") => 1e2, Some("da") => 1e1, Some("d") => 1e-1, Some("c") => 1e-2,
        Some("m") => 1e-3, Some("micro") => 1e-6, Some("n") => 1e-9, Some("p") => 1e-12,
        _ => 1.0,
    }
}

/// A resolved 1-per-equipment or N-per-equipment terminal->bus mapping.
/// CGMES has no direct `from_node`/`to_node` field the way PGM does —
/// everything routes through `Terminal`.
struct TerminalIndex {
    /// Equipment mrid -> its own Terminal mrids, sorted by `sequenceNumber`.
    by_equipment: HashMap<String, Vec<String>>,
    /// Terminal mrid -> resolved bus index (only present when the Terminal's
    /// `TopologicalNode` reference resolves to a known bus).
    bus_of: HashMap<String, usize>,
    /// Terminal mrid -> `ACDCTerminal.connected` (default `true` if absent —
    /// the field is only reliably populated in the SSH profile's current
    /// operating snapshot, not always in EQ).
    connected_of: HashMap<String, bool>,
}

impl TerminalIndex {
    /// `buses` is mutable: a genuine boundary `ConnectivityNode` (one with
    /// equipment attached but no `TopologicalNode` anywhere in the loaded
    /// profile set — confirmed real, not a decode gap: this fixture's own
    /// tie-line + `EquivalentInjection` share exactly such a `ConnectivityNode`,
    /// since resolving it to a real `TopologicalNode` would need the *other*
    /// area's or the merged model's own TP data, which a standalone-area
    /// "Model As Supplied" file doesn't carry) gets a synthesized bus here,
    /// the same spirit as the 3-winding star-bus synthesis.
    fn build(ds: &CimDataset, idx_of: &HashMap<String, usize>, buses: &mut Vec<Bus>) -> Result<Self, CgmesError> {
        // Terminal.TopologicalNode is documented as "an alternative to the
        // ConnectivityNode path to TopologicalNode" — i.e. a Terminal may
        // carry only a ConnectivityNode reference and no direct
        // TopologicalNode at all (confirmed: every EquivalentInjection's own
        // Terminal in this fixture is exactly this case), so
        // ConnectivityNode.TopologicalNode is a required fallback, not an
        // optional nicety.
        let mut cn_to_tn: HashMap<String, String> = HashMap::new();
        for cn_mrid in by_type(ds, "ConnectivityNode") {
            let cn: &cimstructs::ConnectivityNode = require(ds, cn_mrid, "ConnectivityNode", cn_mrid, "(self)")?;
            if let Some(tn) = &cn.topological_node {
                cn_to_tn.insert(cn_mrid.clone(), tn.mrid.clone());
            }
        }

        let mut raw: HashMap<String, Vec<(i64, String)>> = HashMap::new();
        let mut bus_of = HashMap::new();
        let mut connected_of = HashMap::new();
        let mut orphans: Vec<(String, String, Option<String>)> = Vec::new(); // (terminal, connectivity_node, conducting_equipment)
        for t_mrid in by_type(ds, "Terminal") {
            let t: &Terminal = require(ds, t_mrid, "Terminal", t_mrid, "(self)")?;
            connected_of.insert(t_mrid.clone(), t.base.connected.unwrap_or(true));
            if let Some(ce) = &t.conducting_equipment {
                let seq = t.base.sequence_number.unwrap_or(1);
                raw.entry(ce.mrid.clone()).or_default().push((seq, t_mrid.clone()));
            }
            let tn_mrid = t.topological_node.as_ref().map(|tn| tn.mrid.clone()).or_else(|| {
                t.connectivity_node.as_ref().and_then(|cn| cn_to_tn.get(&cn.mrid).cloned())
            });
            match (tn_mrid, &t.connectivity_node) {
                (Some(tn_mrid), _) => {
                    if let Some(&idx) = idx_of.get(&tn_mrid) {
                        bus_of.insert(t_mrid.clone(), idx);
                    }
                }
                (None, Some(cn)) => {
                    orphans.push((t_mrid.clone(), cn.mrid.clone(), t.conducting_equipment.as_ref().map(|ce| ce.mrid.clone())));
                }
                (None, None) => {}
            }
        }

        // Group first, then search *every* orphan terminal sharing a given
        // ConnectivityNode for one whose equipment is an EquivalentInjection
        // with a usable BaseVoltage — not just whichever terminal happens to
        // be first (e.g. a tie-line's far-end ACLineSegment terminal, which
        // has no BaseVoltage of its own and would otherwise silently fall
        // through to a nonsense placeholder).
        let mut orphans_by_cn: HashMap<String, Vec<(String, Option<String>)>> = HashMap::new();
        for (t_mrid, cn_mrid, ce_mrid) in orphans {
            orphans_by_cn.entry(cn_mrid).or_default().push((t_mrid, ce_mrid));
        }
        for (cn_mrid, terms_here) in &orphans_by_cn {
            let u_rated = terms_here
                .iter()
                .find_map(|(_, ce_mrid)| {
                    let ei = get::<EquivalentInjection>(ds, ce_mrid.as_deref()?)?;
                    let bv_ref = ei.base.base.base_voltage.as_ref()?;
                    get::<BaseVoltage>(ds, &bv_ref.mrid)?.nominal_voltage
                })
                .ok_or_else(|| missing("ConnectivityNode", cn_mrid, "(no EquivalentInjection with a resolvable BaseVoltage found)"))?
                * 1e3;
            let idx = buses.len();
            buses.push(Bus {
                idx, bus_type: BusType::PQ, voltage_mag: 1.0, voltage_ang: 0.0,
                p_spec: 0.0, q_spec: 0.0, q_min: -f64::INFINITY, q_max: f64::INFINITY,
                u_rated, zip_terms: Vec::new(),
            });
            for (t_mrid, _) in terms_here {
                bus_of.insert(t_mrid.clone(), idx);
            }
        }

        let by_equipment = raw
            .into_iter()
            .map(|(eq, mut v)| {
                v.sort_by_key(|(seq, _)| *seq);
                (eq, v.into_iter().map(|(_, m)| m).collect())
            })
            .collect();
        Ok(TerminalIndex { by_equipment, bus_of, connected_of })
    }

    /// `which` is 0-indexed after sorting by sequence number (0 = seq 1, the
    /// branch's "starting point" per CGMES's own `ACDCTerminal.sequenceNumber`
    /// doc comment).
    fn bus(&self, equipment_mrid: &str, which: usize) -> Option<usize> {
        self.by_equipment.get(equipment_mrid)?.get(which).and_then(|t| self.bus_of.get(t)).copied()
    }

    fn bus_via_terminal_mrid(&self, terminal_mrid: &str) -> Option<usize> {
        self.bus_of.get(terminal_mrid).copied()
    }

    /// `ACDCTerminal.connected` for the `which`-th (0-indexed, by
    /// `sequenceNumber`) terminal of `equipment_mrid` — `true` if the
    /// terminal can't be found at all (matches the same "assume connected"
    /// default as a missing field).
    fn connected(&self, equipment_mrid: &str, which: usize) -> bool {
        self.by_equipment
            .get(equipment_mrid)
            .and_then(|ts| ts.get(which))
            .and_then(|t| self.connected_of.get(t))
            .copied()
            .unwrap_or(true)
    }

    fn connected_via_terminal_mrid(&self, terminal_mrid: &str) -> bool {
        self.connected_of.get(terminal_mrid).copied().unwrap_or(true)
    }
}

/// The result of resolving whichever tap changer (if any) is attached to a
/// `PowerTransformerEnd`: the complex ratio it contributes at its own end,
/// and — for phase tap changers only — the reactance at the current step
/// (which supersedes `PowerTransformerEnd.x`, since CGMES documents the
/// reactance as tap-position-dependent for phase-shifting transformers).
struct TapEffect {
    tap: Complex<f64>,
    x_override: Option<f64>,
}

/// `end_mrid -> tap-changer mrid`, one map per tap changer subtype, built
/// once up front rather than scanned per transformer end.
struct TapChangerIndex {
    ratio: HashMap<String, String>,
    phase_asym: HashMap<String, String>,
    phase_sym: HashMap<String, String>,
    phase_linear: HashMap<String, String>,
    /// end mrid -> (tabular tap changer mrid, its own PhaseTapChangerTable mrid)
    phase_tabular: HashMap<String, (String, String)>,
}

impl TapChangerIndex {
    fn build(ds: &CimDataset) -> Self {
        let mut ratio = HashMap::new();
        for mrid in by_type(ds, "RatioTapChanger") {
            if let Some(rtc) = get::<RatioTapChanger>(ds, mrid) {
                if let Some(end) = &rtc.transformer_end {
                    ratio.insert(end.mrid.clone(), mrid.clone());
                }
            }
        }
        let mut phase_asym = HashMap::new();
        for mrid in by_type(ds, "PhaseTapChangerAsymmetrical") {
            if let Some(ptc) = get::<PhaseTapChangerAsymmetrical>(ds, mrid) {
                if let Some(end) = &ptc.base.base.transformer_end {
                    phase_asym.insert(end.mrid.clone(), mrid.clone());
                }
            }
        }
        let mut phase_sym = HashMap::new();
        for mrid in by_type(ds, "PhaseTapChangerSymmetrical") {
            if let Some(ptc) = get::<PhaseTapChangerSymmetrical>(ds, mrid) {
                if let Some(end) = &ptc.base.base.transformer_end {
                    phase_sym.insert(end.mrid.clone(), mrid.clone());
                }
            }
        }
        let mut phase_linear = HashMap::new();
        for mrid in by_type(ds, "PhaseTapChangerLinear") {
            if let Some(ptc) = get::<cimstructs::PhaseTapChangerLinear>(ds, mrid) {
                if let Some(end) = &ptc.base.transformer_end {
                    phase_linear.insert(end.mrid.clone(), mrid.clone());
                }
            }
        }
        let mut phase_tabular = HashMap::new();
        for mrid in by_type(ds, "PhaseTapChangerTabular") {
            if let Some(ptc) = get::<cimstructs::PhaseTapChangerTabular>(ds, mrid) {
                if let (Some(end), Some(table)) = (&ptc.base.transformer_end, &ptc.phase_tap_changer_table) {
                    phase_tabular.insert(end.mrid.clone(), (mrid.clone(), table.mrid.clone()));
                }
            }
        }
        TapChangerIndex { ratio, phase_asym, phase_sym, phase_linear, phase_tabular }
    }

    /// `xtx` is the owning `PowerTransformerEnd`'s own static `x` — needed as
    /// the fallback base reactance when `xMin` is absent/non-positive,
    /// mirroring `CgmesPhaseTapChangerBuilder.getXMin()`.
    fn effect_for_end(&self, ds: &CimDataset, end_mrid: &str, xtx: f64) -> Result<Option<TapEffect>, CgmesError> {
        if let Some(mrid) = self.ratio.get(end_mrid) {
            let rtc: &RatioTapChanger = require(ds, mrid, "RatioTapChanger", mrid, "(self)")?;
            let step = rtc.base.step.ok_or_else(|| missing("RatioTapChanger", mrid, "step"))?;
            if let Some(table_ref) = &rtc.ratio_tap_changer_table {
                if let Some(effect) = ratio_tap_table(ds, &table_ref.mrid, step.round() as i64, xtx) {
                    return Ok(Some(effect));
                }
            }
            let neutral = rtc.base.neutral_step.unwrap_or(0) as f64;
            let inc = rtc.step_voltage_increment.unwrap_or(0.0);
            let ratio = 1.0 + (step - neutral) * inc / 100.0;
            return Ok(Some(TapEffect { tap: Complex::new(ratio, 0.0), x_override: None }));
        }
        if let Some(mrid) = self.phase_asym.get(end_mrid) {
            let ptc: &PhaseTapChangerAsymmetrical = require(ds, mrid, "PhaseTapChangerAsymmetrical", mrid, "(self)")?;
            let theta_deg = ptc.winding_connection_angle.ok_or_else(|| {
                missing("PhaseTapChangerAsymmetrical", mrid, "windingConnectionAngle")
            })?;
            return Ok(Some(phase_tap_asymmetrical(&ptc.base, mrid, theta_deg, xtx)?));
        }
        if let Some(mrid) = self.phase_sym.get(end_mrid) {
            let ptc: &PhaseTapChangerSymmetrical = require(ds, mrid, "PhaseTapChangerSymmetrical", mrid, "(self)")?;
            return Ok(Some(phase_tap_symmetrical(&ptc.base, mrid, xtx)?));
        }
        if let Some(mrid) = self.phase_linear.get(end_mrid) {
            let ptc: &cimstructs::PhaseTapChangerLinear = require(ds, mrid, "PhaseTapChangerLinear", mrid, "(self)")?;
            return Ok(Some(phase_tap_linear(ptc, mrid, xtx)?));
        }
        if let Some((ptc_mrid, table_mrid)) = self.phase_tabular.get(end_mrid) {
            let ptc: &cimstructs::PhaseTapChangerTabular = require(ds, ptc_mrid, "PhaseTapChangerTabular", ptc_mrid, "(self)")?;
            let step = ptc.base.base.step.ok_or_else(|| missing("PhaseTapChangerTabular", ptc_mrid, "step"))?;
            return Ok(Some(phase_tap_tabular(ds, ptc_mrid, table_mrid, step.round() as i64, xtx)?));
        }
        Ok(None)
    }
}

/// `PhaseTapChangerTabular`: the current step's ratio/angle/impedance-
/// deviation come directly from a matching `PhaseTapChangerTablePoint` row —
/// no formula, just a lookup. `TapChangerTablePoint.ratio` is documented as
/// "the voltage at the tap step divided by rated voltage" (i.e. already the
/// direct complex-magnitude tap ratio), while `.r`/`.x`/`.g`/`.b` are
/// documented as *percentage deviations* from the transformer end's own
/// nominal values (e.g. "calculated reactance = x(nominal) * (1 +
/// x(from this class)/100)") — matches `references/powsybl-core`'s own
/// `x *= 1 + step.getX() / 100` treatment for tabular tap changers.
fn phase_tap_tabular(
    ds: &CimDataset, ptc_mrid: &str, table_mrid: &str, step: i64, xtx: f64,
) -> Result<TapEffect, CgmesError> {
    for pt_mrid in by_type(ds, "PhaseTapChangerTablePoint") {
        let pt: &cimstructs::PhaseTapChangerTablePoint =
            require(ds, pt_mrid, "PhaseTapChangerTablePoint", pt_mrid, "(self)")?;
        let Some(owner) = &pt.phase_tap_changer_table else { continue };
        if owner.mrid != *table_mrid || pt.base.step != Some(step) {
            continue;
        }
        let ratio = pt.base.ratio.unwrap_or(1.0);
        let angle_rad = pt.angle.unwrap_or(0.0).to_radians();
        let tap = Complex::from_polar(ratio, angle_rad);
        let x_pct = pt.base.x.unwrap_or(0.0);
        return Ok(TapEffect { tap, x_override: Some(xtx * (1.0 + x_pct / 100.0)) });
    }
    Err(CgmesError::UnresolvedReference {
        from_type: "PhaseTapChangerTabular",
        from_mrid: ptc_mrid.to_string(),
        field: "(no PhaseTapChangerTablePoint matching the current step)",
    })
}

/// `RatioTapChanger.RatioTapChangerTable`: unlike `PhaseTapChangerTabular`
/// (a distinct CGMES class with no fallback formula of its own),
/// `RatioTapChangerTable` is just an *optional* reference a plain
/// `RatioTapChanger` may or may not carry alongside its own
/// `stepVoltageIncrement` — so this returns `None` (rather than erroring)
/// when the table or a matching point isn't found, letting the caller fall
/// back to the linear formula, mirroring
/// `CgmesRatioTapChangerBuilder.addSteps`'s own
/// `tablePoints.isEmpty()`/`isTableValid` fallback (simplified to a
/// per-step lookup, since gridoxide only ever needs the *current* step's
/// effect, not a full exported step table). Same caveat as
/// `phase_tap_tabular`: only `ratio` and `x` are read — `r`/`g`/`b`
/// deviations have no representation in `TapEffect`.
fn ratio_tap_table(ds: &CimDataset, table_mrid: &str, step: i64, xtx: f64) -> Option<TapEffect> {
    for pt_mrid in by_type(ds, "RatioTapChangerTablePoint") {
        let pt: &cimstructs::RatioTapChangerTablePoint = get(ds, pt_mrid)?;
        let Some(owner) = &pt.ratio_tap_changer_table else { continue };
        if owner.mrid != *table_mrid || pt.base.step != Some(step) {
            continue;
        }
        let ratio = pt.base.ratio.unwrap_or(1.0);
        let x_pct = pt.base.x.unwrap_or(0.0);
        return Some(TapEffect { tap: Complex::new(ratio, 0.0), x_override: Some(xtx * (1.0 + x_pct / 100.0)) });
    }
    None
}

/// `xMin` (falling back to the transformer end's own static `x`, "xtx", if
/// absent or non-positive — CGMES 3 deprecates `xMin`/`xMax` and documents
/// "PowerTransformerEnd.x shall be consistent with ...xMin... In case of
/// inconsistency, PowerTransformerEnd.x shall be used") and `xMax`, or `None`
/// if either is missing/non-finite — mirrors
/// `CgmesPhaseTapChangerBuilder.getXMin()`/`getXMax()` exactly. Takes the raw
/// `xMin`/`xMax` fields directly (rather than a `PhaseTapChangerNonLinear`)
/// so `PhaseTapChangerLinear` — a distinct CGMES class with its own
/// same-named fields, not a `PhaseTapChangerNonLinear` subtype — can share it.
fn x_min_max(x_min: Option<f64>, x_max: Option<f64>, xtx: f64) -> Option<(f64, f64)> {
    let x_min_raw = x_min.unwrap_or(0.0);
    let x_min = if x_min_raw <= 0.0 { xtx } else { x_min_raw };
    let x_max = x_max?;
    if !(x_min.is_finite() && x_max.is_finite()) || x_min < 0.0 || x_max <= 0.0 || x_min > x_max {
        return None;
    }
    Some((x_min, x_max))
}

/// `PhaseTapChangerAsymmetrical`: cross-checked directly against
/// `references/powsybl-core`'s own
/// `cgmes-conversion/.../transformers/CgmesPhaseTapChangerBuilder.java`
/// (`addStepsAsymmetrical`/`getStepXforAsymmetrical`), not independently
/// derived — that reference wasn't available during this converter's initial
/// draft (which used a materially different, unverified formula), and was
/// checked in specifically to replace it.
///
/// The tapped winding's voltage phasor is the nominal (1∠0°) plus an added
/// vector of magnitude `du = (step−neutralStep)·voltageStepIncrement/100` at
/// the fixed `windingConnectionAngle`, giving both the ratio (`hypot`) and
/// angle (`atan2`) deviation as one complex number. Reactance follows a
/// separate trig curve keyed on `alphaMax`, the *largest angle actually
/// reached* over the tap's full `[lowStep, highStep]` range (not simply the
/// value at either endpoint, since angle isn't necessarily monotonic in step
/// once `windingConnectionAngle` is taken into account).
fn phase_tap_asymmetrical(
    base: &PhaseTapChangerNonLinear, mrid: &str, winding_connection_angle_deg: f64, xtx: f64,
) -> Result<TapEffect, CgmesError> {
    let tc = &base.base.base;
    let step = tc.step.ok_or_else(|| missing("PhaseTapChanger", mrid, "step"))?;
    let neutral = tc.neutral_step.unwrap_or(0) as f64;
    let low = tc.low_step.unwrap_or(0);
    let high = tc.high_step.unwrap_or(0);
    let inc = base.voltage_step_increment.unwrap_or(0.0);
    let theta = winding_connection_angle_deg.to_radians();

    let angle_rad_at = |s: f64| -> f64 {
        let d = (s - neutral) * inc / 100.0;
        let dx = 1.0 + d * theta.cos();
        let dy = d * theta.sin();
        dy.atan2(dx)
    };
    let ratio_at = |s: f64| -> f64 {
        let d = (s - neutral) * inc / 100.0;
        let dx = 1.0 + d * theta.cos();
        let dy = d * theta.sin();
        dx.hypot(dy)
    };

    let alpha = angle_rad_at(step);
    let tap = Complex::from_polar(ratio_at(step), alpha);

    let alpha_max = (low..=high).map(|s| angle_rad_at(s as f64)).fold(f64::MIN, f64::max);
    let x_override = match (x_min_max(base.x_min, base.x_max, xtx), alpha_max != 0.0) {
        (Some((x_min, x_max)), true) => {
            let numer = theta.sin() - alpha_max.tan() * theta.cos();
            let denom = theta.sin() - alpha.tan() * theta.cos();
            let t = alpha.tan() / alpha_max.tan() * numer / denom;
            Some(x_min + (x_max - x_min) * t * t)
        }
        (Some(_), false) => Some(0.0),
        (None, _) => None,
    };

    Ok(TapEffect { tap, x_override })
}

/// `PhaseTapChangerSymmetrical`: cross-checked against powsybl-core's
/// `addStepsSymmetrical`/`getStepXforLinearAndSymmetrical` (see
/// `phase_tap_asymmetrical`'s doc comment for the full provenance note).
/// Ratio is always exactly 1.0 (magnitude never changes) — only the angle
/// varies, via `2·atan(du/2)` where `du = (step−neutralStep)·
/// voltageStepIncrement/100` (CGMES also allows a `stepPhaseShiftIncrement`-
/// based linear angle formula here, but that field only exists on
/// `PhaseTapChangerLinear`, a different, unrelated CGMES class — confirmed
/// absent from `PhaseTapChangerNonLinear`/`Symmetrical`'s own generated
/// fields, so it's not handled here).
fn phase_tap_symmetrical(base: &PhaseTapChangerNonLinear, mrid: &str, xtx: f64) -> Result<TapEffect, CgmesError> {
    let tc = &base.base.base;
    let step = tc.step.ok_or_else(|| missing("PhaseTapChanger", mrid, "step"))?;
    let neutral = tc.neutral_step.unwrap_or(0) as f64;
    let low = tc.low_step.unwrap_or(0);
    let high = tc.high_step.unwrap_or(0);
    let inc = base.voltage_step_increment.unwrap_or(0.0);

    let angle_rad_at = |s: f64| -> f64 {
        let du = (s - neutral) * inc / 100.0;
        2.0 * (du / 2.0).atan()
    };

    let alpha = angle_rad_at(step);
    let tap = Complex::from_polar(1.0, alpha);

    let alpha_max = (low..=high).map(|s| angle_rad_at(s as f64)).fold(f64::MIN, f64::max);
    let x_override = match (x_min_max(base.x_min, base.x_max, xtx), alpha_max != 0.0) {
        (Some((x_min, x_max)), true) => {
            let ratio = (alpha / 2.0).sin() / (alpha_max / 2.0).sin();
            Some(x_min + (x_max - x_min) * ratio * ratio)
        }
        (Some(_), false) => Some(0.0),
        (None, _) => None,
    };

    Ok(TapEffect { tap, x_override })
}

/// `PhaseTapChangerLinear`: cross-checked against powsybl-core's
/// `addStepsLinear` (see `phase_tap_asymmetrical`'s doc comment for the
/// shared provenance note). A distinct CGMES class from
/// `PhaseTapChangerNonLinear`'s Symmetrical/Asymmetrical/Tabular subtypes,
/// not a sibling of them — its own `base` is `PhaseTapChanger` directly, one
/// level shallower. Ratio is always exactly 1.0 (a pure phase shifter, no
/// magnitude change); angle is *linear* in step
/// (`(step−neutralStep)·stepPhaseShiftIncrement`, in degrees) rather than
/// Symmetrical's `2·atan(du/2)` curve. Reactance follows the identical
/// `sin(alpha/2)²` interpolation Symmetrical uses — the Java reference
/// shares one `getStepXforLinearAndSymmetrical` helper between both types.
fn phase_tap_linear(ptc: &cimstructs::PhaseTapChangerLinear, mrid: &str, xtx: f64) -> Result<TapEffect, CgmesError> {
    let tc = &ptc.base.base;
    let step = tc.step.ok_or_else(|| missing("PhaseTapChangerLinear", mrid, "step"))?;
    let neutral = tc.neutral_step.unwrap_or(0) as f64;
    let low = tc.low_step.unwrap_or(0);
    let high = tc.high_step.unwrap_or(0);
    let inc_deg = ptc.step_phase_shift_increment.unwrap_or(0.0);

    let angle_rad_at = |s: f64| -> f64 { ((s - neutral) * inc_deg).to_radians() };

    let alpha = angle_rad_at(step);
    let tap = Complex::from_polar(1.0, alpha);

    let alpha_max = (low..=high).map(|s| angle_rad_at(s as f64)).fold(f64::MIN, f64::max);
    let x_override = match (x_min_max(ptc.x_min, ptc.x_max, xtx), alpha_max != 0.0) {
        (Some((x_min, x_max)), true) => {
            let ratio = (alpha / 2.0).sin() / (alpha_max / 2.0).sin();
            Some(x_min + (x_max - x_min) * ratio * ratio)
        }
        (Some(_), false) => Some(0.0),
        (None, _) => None,
    };

    Ok(TapEffect { tap, x_override })
}

/// Converts a decoded CGMES dataset (an EQ+SSH+TP+SV profile bundle) into
/// gridoxide's own network model.
///
/// `Result`-returning (unlike `pgm::pgm_to_buses_and_branches`'s bare tuple):
/// CGMES's pervasive field optionality and cross-reference resolution are
/// much likelier to hit genuinely malformed/incomplete input than PGM-JSON's
/// already-schema-validated shape.
pub fn cgmes_to_buses_and_branches(
    ds: &CimDataset, s_base_va: f64,
) -> Result<(Vec<Bus>, Vec<Line>, Vec<Transformer>, Vec<ShuntAdm>), CgmesError> {
    // --- Step 1: buses from TopologicalNode ---
    let tn_mrids = by_type(ds, "TopologicalNode");
    if tn_mrids.is_empty() {
        return Err(CgmesError::NoTopologicalNodes);
    }
    let mut idx_of: HashMap<String, usize> = HashMap::new();
    let mut buses: Vec<Bus> = Vec::with_capacity(tn_mrids.len());
    for (i, mrid) in tn_mrids.iter().enumerate() {
        idx_of.insert(mrid.clone(), i);
        let tn: &TopologicalNode = require(ds, mrid, "TopologicalNode", mrid, "(self)")?;
        let u_rated = match &tn.base_voltage {
            Some(bv_ref) => {
                let bv: &BaseVoltage = require(ds, &bv_ref.mrid, "TopologicalNode", mrid, "BaseVoltage")?;
                // CGMES gives nominalVoltage in kV; `Bus::u_rated` is documented in V.
                bv.nominal_voltage.ok_or_else(|| missing("BaseVoltage", &bv_ref.mrid, "nominalVoltage"))? * 1e3
            }
            None => return Err(missing("TopologicalNode", mrid, "BaseVoltage")),
        };
        buses.push(Bus {
            idx: i,
            bus_type: BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: -f64::INFINITY,
            q_max: f64::INFINITY,
            u_rated,
            zip_terms: Vec::new(),
        });
    }

    // --- Step 2: shared Terminal-based resolver ---
    let terms = TerminalIndex::build(ds, &idx_of, &mut buses)?;

    // --- Step 3: loads/injections (EnergyConsumer + subtypes + EquivalentInjection) ---
    // Both P and Q use CGMES's uniform SSH "load sign convention" (positive =
    // flow OUT of the node INTO the equipment, i.e. absorption) — the
    // opposite of gridoxide's own net-injection convention, hence the
    // negation of both. SynchronousMachine's own Q (below, in Step 8) does
    // NOT get this same negation — confirmed empirically, not from the CIM
    // doc text (which reads identically for loads and machines): reverting
    // Q's negation for loads specifically (keeping it only for
    // SynchronousMachine) dropped RealGrid's median solved-vs-published-SV
    // voltage error from 5.9% to 0.09% (and buses over 5% error from 3369 of
    // 6051 to 11) — a real, load-vs-machine-specific asymmetry, not a
    // uniform CGMES quirk.
    for mrid in by_type(ds, "EnergyConsumer") {
        let ec: &EnergyConsumer = require(ds, mrid, "EnergyConsumer", mrid, "(self)")?;
        let Some(bus) = terms.bus(mrid, 0) else { continue };
        if !terms.connected(mrid, 0) { continue }
        buses[bus].p_spec += -ec.p.unwrap_or(0.0) * 1e6 / s_base_va;
        buses[bus].q_spec += -ec.q.unwrap_or(0.0) * 1e6 / s_base_va;
    }
    // ConformLoad/NonConformLoad are EnergyConsumer subtypes (real-world
    // CGMES exports overwhelmingly use these, not bare EnergyConsumer — e.g.
    // RealGrid's own EQ file has zero raw EnergyConsumer entries, only
    // ConformLoad) — `by_type` is keyed by each element's own concrete RDF
    // type, not its inheritance chain, so these need their own loop.
    for mrid in by_type(ds, "ConformLoad") {
        let cl: &cimstructs::ConformLoad = require(ds, mrid, "ConformLoad", mrid, "(self)")?;
        let Some(bus) = terms.bus(mrid, 0) else { continue };
        if !terms.connected(mrid, 0) { continue }
        buses[bus].p_spec += -cl.base.p.unwrap_or(0.0) * 1e6 / s_base_va;
        buses[bus].q_spec += -cl.base.q.unwrap_or(0.0) * 1e6 / s_base_va;
    }
    for mrid in by_type(ds, "NonConformLoad") {
        let ncl: &cimstructs::NonConformLoad = require(ds, mrid, "NonConformLoad", mrid, "(self)")?;
        let Some(bus) = terms.bus(mrid, 0) else { continue };
        if !terms.connected(mrid, 0) { continue }
        buses[bus].p_spec += -ncl.base.p.unwrap_or(0.0) * 1e6 / s_base_va;
        buses[bus].q_spec += -ncl.base.q.unwrap_or(0.0) * 1e6 / s_base_va;
    }
    // AsynchronousMachine (an induction motor/generator): grouped here with
    // the loads, not with SynchronousMachine down in Step 8, despite sharing
    // the same RotatingMachine base — cross-checked against
    // references/powsybl-core's own AsynchronousMachineConversion, which
    // converts it to a plain IIDM Load ("we make no difference based on the
    // type (motor/generator)") with *no* sign flip at all on P0/Q0, because
    // IIDM's own Load.p0/q0 already share CGMES's load-sign convention. That
    // makes this the load-style *both-negated* case, not SynchronousMachine's
    // Q exception (which exists only because a machine is normally a source,
    // not a sink).
    for mrid in by_type(ds, "AsynchronousMachine") {
        let am: &cimstructs::AsynchronousMachine = require(ds, mrid, "AsynchronousMachine", mrid, "(self)")?;
        let Some(bus) = terms.bus(mrid, 0) else { continue };
        if !terms.connected(mrid, 0) { continue }
        buses[bus].p_spec += -am.base.p.unwrap_or(0.0) * 1e6 / s_base_va;
        buses[bus].q_spec += -am.base.q.unwrap_or(0.0) * 1e6 / s_base_va;
    }
    for mrid in by_type(ds, "EquivalentInjection") {
        let ei: &EquivalentInjection = require(ds, mrid, "EquivalentInjection", mrid, "(self)")?;
        let Some(bus) = terms.bus(mrid, 0) else { continue };
        if !terms.connected(mrid, 0) { continue }
        buses[bus].p_spec += -ei.p.unwrap_or(0.0) * 1e6 / s_base_va;
        buses[bus].q_spec += -ei.q.unwrap_or(0.0) * 1e6 / s_base_va;
    }

    // --- Step 4: lines from ACLineSegment ---
    // `r`/`x`/`bch`/`gch` are documented directly on ACLineSegment as "of the
    // entire line section" (i.e. already segment totals, not per-length
    // values) — no `Conductor.length` multiplication needed. `gch` is 0 (or
    // absent) on most real lines, but not universally: MicroGrid-BE-MAS's
    // own BE-Line_6/BE-Line_2 carry non-negligible values (several MW of
    // real power each at nominal voltage) that were silently dropped before
    // `Line` gained a `g_shunt` field — confirmed via
    // `scripts/bench/cross_validate_cgmes_microgrid_be.py`'s pypowsybl
    // cross-check, where the missing MW surfaced as slack-relative-angle
    // error at the electrically-downstream StaticVarCompensator bus (a
    // voltage-magnitude-pinned bus has no equivalent slack for an active-
    // power mismatch, only a reactive one).
    //
    // `types::Line` has no status field (unlike `types::Transformer`), so a
    // half-open line (one end disconnected) is folded into a self-loop
    // shunt-only Line at the connected end, and a fully-open one is skipped
    // — mirroring pgm.rs's own from_status/to_status handling for `Line`,
    // needed here because RealGrid genuinely has `Terminal.connected=false`
    // entries (a real de-energized/switched-out snapshot, not a decode gap).
    fn push_status_aware_line(lines: &mut Vec<Line>, from: usize, to: usize, from_conn: bool, to_conn: bool, r: f64, x: f64, b_shunt: f64, g_shunt: f64) {
        match (from_conn, to_conn) {
            (true, true) => lines.push(Line { from, to, r, x, b_shunt, g_shunt }),
            (true, false) => lines.push(Line { from, to: from, r: 0.0, x: 0.0, b_shunt, g_shunt }),
            (false, true) => lines.push(Line { from: to, to, r: 0.0, x: 0.0, b_shunt, g_shunt }),
            (false, false) => {}
        }
    }

    let mut lines: Vec<Line> = Vec::new();
    for mrid in by_type(ds, "ACLineSegment") {
        let ln: &ACLineSegment = require(ds, mrid, "ACLineSegment", mrid, "(self)")?;
        let (Some(from), Some(to)) = (terms.bus(mrid, 0), terms.bus(mrid, 1)) else { continue };
        let u_rated = buses[from].u_rated;
        let z_base = u_rated * u_rated / s_base_va;
        let y_base = 1.0 / z_base;
        push_status_aware_line(
            &mut lines, from, to, terms.connected(mrid, 0), terms.connected(mrid, 1),
            ln.r.unwrap_or(0.0) / z_base, ln.x.unwrap_or(0.0) / z_base,
            ln.bch.unwrap_or(0.0) / y_base, ln.gch.unwrap_or(0.0) / y_base,
        );
    }
    // SeriesCompensator: a distinct 2-terminal CIM class from ACLineSegment
    // ("a series capacitor or reactor... without charging susceptance" per
    // its own doc comment) — same conversion, minus the shunt terms (it has
    // no bch/gch fields at all, unlike ACLineSegment).
    for mrid in by_type(ds, "SeriesCompensator") {
        let sc: &cimstructs::SeriesCompensator = require(ds, mrid, "SeriesCompensator", mrid, "(self)")?;
        let (Some(from), Some(to)) = (terms.bus(mrid, 0), terms.bus(mrid, 1)) else { continue };
        let u_rated = buses[from].u_rated;
        let z_base = u_rated * u_rated / s_base_va;
        push_status_aware_line(
            &mut lines, from, to, terms.connected(mrid, 0), terms.connected(mrid, 1),
            sc.r.unwrap_or(0.0) / z_base, sc.x.unwrap_or(0.0) / z_base, 0.0, 0.0,
        );
    }

    // --- Steps 5+6: transformers (2- and 3-winding) ---
    let tap_index = TapChangerIndex::build(ds);
    let mut ends_by_pt: HashMap<String, Vec<&PowerTransformerEnd>> = HashMap::new();
    for mrid in by_type(ds, "PowerTransformerEnd") {
        let end: &PowerTransformerEnd = require(ds, mrid, "PowerTransformerEnd", mrid, "(self)")?;
        let Some(pt) = &end.power_transformer else {
            return Err(missing("PowerTransformerEnd", mrid, "PowerTransformer"));
        };
        ends_by_pt.entry(pt.mrid.clone()).or_default().push(end);
    }

    let mut transformers: Vec<Transformer> = Vec::new();
    for (pt_mrid, mut ends) in ends_by_pt {
        ends.sort_by_key(|e| e.base.end_number.unwrap_or(0));
        match ends.len() {
            2 => transformers.push(build_two_winding(ds, &tap_index, &terms, &buses, &pt_mrid, ends[0], ends[1], s_base_va)?),
            3 => {
                // `buses.len()` alone is the next free index — it already
                // reflects every star bus pushed by a *previous* iteration
                // of this same loop, so adding a separate running counter
                // on top (as this used to) double-counts them: the second
                // 3-winding transformer in a model with more than one would
                // get a star bus index one past the actual end of `buses`,
                // corrupting the Y-bus with an out-of-range reference
                // (confirmed via MiniGrid's own conformance fixture, the
                // first real multi-3-winding-transformer case this
                // converter was tried against).
                let star_idx = buses.len();
                buses.push(Bus {
                    idx: star_idx, bus_type: BusType::PQ, voltage_mag: 1.0, voltage_ang: 0.0,
                    p_spec: 0.0, q_spec: 0.0, q_min: -f64::INFINITY, q_max: f64::INFINITY,
                    u_rated: ends[0].rated_u.unwrap_or(1e-3) * 1e3, zip_terms: Vec::new(),
                });
                for end in &ends {
                    transformers.push(build_star_leg(ds, &tap_index, &terms, &buses, end, star_idx, s_base_va)?);
                }
            }
            n => return Err(CgmesError::UnsupportedTransformer {
                mrid: pt_mrid, reason: format!("{n} PowerTransformerEnds (only 2 or 3 supported)"),
            }),
        }
    }

    // --- Step 7: shunts (LinearShuntCompensator + NonlinearShuntCompensator) ---
    let mut shunts: Vec<ShuntAdm> = Vec::new();
    for mrid in by_type(ds, "LinearShuntCompensator") {
        let sc: &LinearShuntCompensator = require(ds, mrid, "LinearShuntCompensator", mrid, "(self)")?;
        let Some(at) = terms.bus(mrid, 0) else { continue };
        if !terms.connected(mrid, 0) { continue }
        let sections = sc.base.sections.unwrap_or(0.0);
        let g = sc.g_per_section.unwrap_or(0.0) * sections;
        let b = sc.b_per_section.unwrap_or(0.0) * sections;
        let z_base = buses[at].u_rated * buses[at].u_rated / s_base_va;
        shunts.push(ShuntAdm { at, y: Complex::new(g, b) * z_base });
    }
    for mrid in by_type(ds, "NonlinearShuntCompensator") {
        let sc: &NonlinearShuntCompensator = require(ds, mrid, "NonlinearShuntCompensator", mrid, "(self)")?;
        let Some(at) = terms.bus(mrid, 0) else { continue };
        if !terms.connected(mrid, 0) { continue }
        let target_section = sc.base.sections.unwrap_or(0.0).round() as i64;
        let mut y = Complex::new(0.0, 0.0);
        for pt_mrid in by_type(ds, "NonlinearShuntCompensatorPoint") {
            let pt: &NonlinearShuntCompensatorPoint =
                require(ds, pt_mrid, "NonlinearShuntCompensatorPoint", pt_mrid, "(self)")?;
            let Some(owner) = &pt.nonlinear_shunt_compensator else { continue };
            if owner.mrid != *mrid {
                continue;
            }
            if pt.section_number.unwrap_or(-1) == target_section {
                y = Complex::new(pt.g.unwrap_or(0.0), pt.b.unwrap_or(0.0));
                break;
            }
        }
        let z_base = buses[at].u_rated * buses[at].u_rated / s_base_va;
        shunts.push(ShuntAdm { at, y: y * z_base });
    }

    // De-energized buses: CGMES's own TopologicalIsland doc comment says
    // "only energised TopologicalNode-s shall be part of the topological
    // island" — so any TopologicalNode *not* listed in some
    // TopologicalIsland.TopologicalNodes is, by that same construction,
    // de-energized, with no need to trace connectivity ourselves. Confirmed
    // real on RealGrid, not theoretical: its own TopologicalIsland lists
    // 6051 of 6252 TopologicalNodes, leaving 201 de-energized (e.g. a
    // `ConformLoad` with `Terminal.connected=false` and nothing else
    // attached, which would otherwise leave an all-zero row in the
    // Jacobian) — mirrors pgm.rs's own `energized_node_ids` treatment
    // ("Nodes with no path to any active source... reported at zero voltage
    // and excluded from the NR solve by modelling them as a fixed
    // (Slack-like) bus at V=0").
    let mut energized = vec![false; buses.len()];
    for mrid in by_type(ds, "TopologicalIsland") {
        let ti: &TopologicalIsland = require(ds, mrid, "TopologicalIsland", mrid, "(self)")?;
        for tn in &ti.topological_nodes {
            if let Some(&idx) = idx_of.get(&tn.mrid) {
                energized[idx] = true;
            }
        }
    }
    // Synthesized buses (3-winding star points, boundary ConnectivityNodes)
    // have no TopologicalNode/TopologicalIsland membership of their own —
    // treat them as energized by default (their own physical leg/injection
    // determines whether they end up isolated, not island membership).
    for energized in energized.iter_mut().skip(tn_mrids.len()) {
        *energized = true;
    }
    for (i, bus) in buses.iter_mut().enumerate() {
        if !energized[i] {
            bus.bus_type = BusType::Slack;
            bus.voltage_mag = 0.0;
            bus.p_spec = 0.0;
            bus.q_spec = 0.0;
        }
    }

    // --- Step 8: slack/PV assignment ---
    // PV upgrade: a SynchronousMachine with an active (mode=voltage, enabled
    // on both the control and the machine) RegulatingControl pins the
    // *controlled* bus's voltage (RegulatingControl.Terminal, which can
    // differ from the machine's own terminal for remote voltage control) —
    // mirrors pgm.rs's `voltage_regulator` handling, reading CGMES's own
    // fields instead.
    for mrid in by_type(ds, "SynchronousMachine") {
        let sm: &SynchronousMachine = require(ds, mrid, "SynchronousMachine", mrid, "(self)")?;
        let machine_connected = terms.connected(mrid, 0);
        if let Some(bus) = terms.bus(mrid, 0) {
            if machine_connected {
                let p = sm.base.p.unwrap_or(0.0);
                let q = sm.base.q.unwrap_or(0.0);
                buses[bus].p_spec += -p * 1e6 / s_base_va;
                buses[bus].q_spec += q * 1e6 / s_base_va; // no negation — see the Step 3 loads comment
            }
        }

        let Some(rc_ref) = &sm.base.base.regulating_control else { continue };
        if !machine_connected || sm.base.base.control_enabled != Some(true) {
            continue;
        }
        let rc: &RegulatingControl = require(ds, &rc_ref.mrid, "SynchronousMachine", mrid, "RegulatingControl")?;
        if rc.enabled != Some(true) {
            continue;
        }
        let is_voltage_mode = rc.mode.as_ref().is_some_and(|m| m.uri.ends_with(".voltage"));
        if !is_voltage_mode {
            continue;
        }
        let Some(term_ref) = &rc.terminal else { continue };
        let Some(controlled_bus) = terms.bus_via_terminal_mrid(&term_ref.mrid) else { continue };
        let target = rc.target_value.ok_or_else(|| missing("RegulatingControl", &rc_ref.mrid, "targetValue"))?;
        let mult = unit_multiplier(rc.target_value_unit_multiplier.as_ref().map(|u| u.uri.as_str()));

        if buses[controlled_bus].bus_type == BusType::PQ {
            buses[controlled_bus].bus_type = BusType::PV;
        }
        buses[controlled_bus].voltage_mag = target * mult / buses[controlled_bus].u_rated;
        let q_min = sm.min_q.unwrap_or(-f64::INFINITY);
        let q_max = sm.max_q.unwrap_or(f64::INFINITY);
        buses[controlled_bus].q_min = if q_min.is_finite() { q_min * 1e6 / s_base_va } else { q_min };
        buses[controlled_bus].q_max = if q_max.is_finite() { q_max * 1e6 / s_base_va } else { q_max };
    }

    // StaticVarCompensator: same RegulatingCondEq/RegulatingControl pattern as
    // SynchronousMachine above (a voltage-mode, enabled RegulatingControl
    // pins the *controlled* bus's voltage), minus any active-power term — an
    // SVC is a pure reactive-power device. Falls back to a fixed Q injection
    // (using SynchronousMachine's empirically-determined sign, not the
    // negation loads get — see the Step 3 comment: StaticVarCompensator.q's
    // doc text is character-for-character identical to RotatingMachine.q's,
    // which was proven unreliable, and both are shunt-connected
    // RegulatingCondEq sources rather than consuming loads) when the SVC
    // isn't actively voltage-regulating.
    for mrid in by_type(ds, "StaticVarCompensator") {
        let sc: &StaticVarCompensator = require(ds, mrid, "StaticVarCompensator", mrid, "(self)")?;
        let svc_connected = terms.connected(mrid, 0);
        let own_bus = terms.bus(mrid, 0);
        if let Some(bus) = own_bus {
            if svc_connected {
                buses[bus].q_spec += sc.q.unwrap_or(0.0) * 1e6 / s_base_va;
            }
        }

        let Some(rc_ref) = &sc.base.regulating_control else { continue };
        if !svc_connected || sc.base.control_enabled != Some(true) {
            continue;
        }
        let rc: &RegulatingControl = require(ds, &rc_ref.mrid, "StaticVarCompensator", mrid, "RegulatingControl")?;
        if rc.enabled != Some(true) {
            continue;
        }
        let is_voltage_mode = rc.mode.as_ref().is_some_and(|m| m.uri.ends_with(".voltage"));
        if !is_voltage_mode {
            continue;
        }
        let Some(term_ref) = &rc.terminal else { continue };
        let Some(controlled_bus) = terms.bus_via_terminal_mrid(&term_ref.mrid) else { continue };
        let target = rc.target_value.ok_or_else(|| missing("RegulatingControl", &rc_ref.mrid, "targetValue"))?;
        let mult = unit_multiplier(rc.target_value_unit_multiplier.as_ref().map(|u| u.uri.as_str()));

        if buses[controlled_bus].bus_type == BusType::PQ {
            buses[controlled_bus].bus_type = BusType::PV;
        }
        buses[controlled_bus].voltage_mag = target * mult / buses[controlled_bus].u_rated;

        // capacitiveRating/inductiveRating are REACTANCE ratings in ohms,
        // not MVAr — despite the doc text reading "at maximum ... reactive
        // power", cross-checked directly against references/powsybl-core's
        // own StaticVarCompensatorConversion.getB(), which computes
        // susceptance as `1 / rating` before ever reaching a power
        // quantity (confirmed empirically too: treating a real BE-MAS
        // fixture's 5062.5 as already-MVAr gives an absurd ~5 GVAr rating
        // for a single substation SVC; treating it as ohms gives a
        // physically sensible ~10 MVAr). Converted to a per-unit Q rating
        // via Q ≈ V²·B ≈ B_pu at V≈1pu (the same flat-voltage
        // approximation SynchronousMachine's own min_q/max_q already make
        // above), using z_base anchored to the SVC's *own* physical bus —
        // not necessarily `controlled_bus`, if regulation is remote.
        let z_base = own_bus.map(|b| buses[b].u_rated * buses[b].u_rated / s_base_va);
        // Already per-unit (z_base/x is a dimensionless ohm/ohm ratio) —
        // unlike SynchronousMachine's/StaticVarCompensator's own P/Q
        // injection above, no further `* 1e6 / s_base_va` MVAr-to-pu
        // conversion applies here.
        buses[controlled_bus].q_min = match (sc.inductive_rating, z_base) {
            (Some(x), Some(zb)) if x != 0.0 => zb / x,
            _ => -f64::INFINITY,
        };
        buses[controlled_bus].q_max = match (sc.capacitive_rating, z_base) {
            (Some(x), Some(zb)) if x != 0.0 => zb / x,
            _ => f64::INFINITY,
        };
    }

    // ExternalNetworkInjection: CIM describes it as "used for IEC 60909
    // [short-circuit] calculations", but it also carries load-flow P/Q and
    // an optional RegulatingControl — cross-checked against
    // `references/powsybl-core`'s own `ExternalNetworkInjectionConversion`,
    // which negates *both* P and Q (`targetP = -p, targetQ = -q`). That's
    // `EquivalentInjection`'s convention (Step 3 above), not
    // SynchronousMachine's Q exception: an ExternalNetworkInjection stands
    // in for "the rest of the interconnected system", the same conceptual
    // role EquivalentInjection plays, not a physical rotating machine.
    for mrid in by_type(ds, "ExternalNetworkInjection") {
        let eni: &cimstructs::ExternalNetworkInjection = require(ds, mrid, "ExternalNetworkInjection", mrid, "(self)")?;
        let eni_connected = terms.connected(mrid, 0);
        if let Some(bus) = terms.bus(mrid, 0) {
            if eni_connected {
                buses[bus].p_spec += -eni.p.unwrap_or(0.0) * 1e6 / s_base_va;
                buses[bus].q_spec += -eni.q.unwrap_or(0.0) * 1e6 / s_base_va;
            }
        }

        let Some(rc_ref) = &eni.base.regulating_control else { continue };
        if !eni_connected || eni.base.control_enabled != Some(true) {
            continue;
        }
        let rc: &RegulatingControl = require(ds, &rc_ref.mrid, "ExternalNetworkInjection", mrid, "RegulatingControl")?;
        if rc.enabled != Some(true) {
            continue;
        }
        let is_voltage_mode = rc.mode.as_ref().is_some_and(|m| m.uri.ends_with(".voltage"));
        if !is_voltage_mode {
            continue;
        }
        let Some(term_ref) = &rc.terminal else { continue };
        let Some(controlled_bus) = terms.bus_via_terminal_mrid(&term_ref.mrid) else { continue };
        let target = rc.target_value.ok_or_else(|| missing("RegulatingControl", &rc_ref.mrid, "targetValue"))?;
        let mult = unit_multiplier(rc.target_value_unit_multiplier.as_ref().map(|u| u.uri.as_str()));

        if buses[controlled_bus].bus_type == BusType::PQ {
            buses[controlled_bus].bus_type = BusType::PV;
        }
        buses[controlled_bus].voltage_mag = target * mult / buses[controlled_bus].u_rated;
        let q_min = eni.min_q.unwrap_or(-f64::INFINITY);
        let q_max = eni.max_q.unwrap_or(f64::INFINITY);
        buses[controlled_bus].q_min = if q_min.is_finite() { q_min * 1e6 / s_base_va } else { q_min };
        buses[controlled_bus].q_max = if q_max.is_finite() { q_max * 1e6 / s_base_va } else { q_max };
    }

    // Slack: the TopologicalIsland's own angle reference, applied last so it
    // wins over any PV upgrade that happened to land on the same bus.
    let mut angle_ref: Option<usize> = None;
    for mrid in by_type(ds, "TopologicalIsland") {
        let ti: &TopologicalIsland = require(ds, mrid, "TopologicalIsland", mrid, "(self)")?;
        if let Some(tn_ref) = &ti.angle_ref_topological_node {
            if let Some(&idx) = idx_of.get(&tn_ref.mrid) {
                angle_ref = Some(idx);
                break;
            }
        }
    }
    let Some(slack_idx) = angle_ref else { return Err(CgmesError::NoAngleReference) };
    buses[slack_idx].bus_type = BusType::Slack;

    // The slack bus's angle is an arbitrary global rotational reference in AC
    // power flow — only relative angles between buses are physically
    // meaningful. A solved SV profile pins that choice to a specific value
    // (not necessarily 0°: this fixture's own reference bus is published at
    // 340.9585°, presumably to stay angle-consistent with the larger merged
    // model this area submission is part of), so matching it here — rather
    // than defaulting to 0° — is what actually reproduces the same solution,
    // not a fixture-specific hack.
    let slack_mrid = &tn_mrids[slack_idx];
    for mrid in by_type(ds, "SvVoltage") {
        let sv: &cimstructs::SvVoltage = require(ds, mrid, "SvVoltage", mrid, "(self)")?;
        if sv.topological_node.as_ref().is_some_and(|tn| &tn.mrid == slack_mrid) {
            if let Some(angle_deg) = sv.angle {
                buses[slack_idx].voltage_ang = angle_deg.to_radians();
            }
            if let Some(v) = sv.v {
                // SvVoltage.v is also in kV, same conversion as nominalVoltage/ratedU.
                buses[slack_idx].voltage_mag = (v * 1e3) / buses[slack_idx].u_rated;
            }
            break;
        }
    }

    Ok((buses, lines, transformers, shunts))
}

/// Builds a 2-winding `Transformer`. Per CGMES convention (confirmed against
/// real fixture data, not assumed): for a 2-winding `PowerTransformer`, end 1
/// carries all series (r/x) and magnetizing (g/b) impedance, with end 2's
/// left at zero — so `to` is fixed to end 1's bus (the side series/shunt
/// admittance is naturally referenced to, mirroring `pgm.rs`'s own
/// "referenced to the to-side" convention) and `from` is fixed to end 2's
/// bus, regardless of which end an active tap changer is physically on.
/// When the tap changer is on end 1 (the `to` side), the complex ratio is
/// inverted before use — mirroring `network::transformer_tap`'s own
/// `tap_side`-dependent reciprocal treatment, needed because
/// `network::branch_calc_param` always scales the `from` side by `1/tap²`.
fn build_two_winding(
    ds: &CimDataset, tap_index: &TapChangerIndex, terms: &TerminalIndex, buses: &[Bus],
    pt_mrid: &str, end1: &PowerTransformerEnd, end2: &PowerTransformerEnd, s_base_va: f64,
) -> Result<Transformer, CgmesError> {
    let term1 = end1.base.terminal.as_ref().ok_or_else(|| missing("PowerTransformerEnd", end1.mrid_str(), "Terminal"))?;
    let term2 = end2.base.terminal.as_ref().ok_or_else(|| missing("PowerTransformerEnd", end2.mrid_str(), "Terminal"))?;
    let bus1 = terms.bus_via_terminal_mrid(&term1.mrid)
        .ok_or_else(|| CgmesError::UnresolvedReference { from_type: "PowerTransformerEnd", from_mrid: end1.mrid_str().to_string(), field: "Terminal.TopologicalNode" })?;
    let bus2 = terms.bus_via_terminal_mrid(&term2.mrid)
        .ok_or_else(|| CgmesError::UnresolvedReference { from_type: "PowerTransformerEnd", from_mrid: end2.mrid_str().to_string(), field: "Terminal.TopologicalNode" })?;

    let tap1 = tap_index.effect_for_end(ds, end1.mrid_str(), end1.x.unwrap_or(0.0))?;
    let tap2 = tap_index.effect_for_end(ds, end2.mrid_str(), end2.x.unwrap_or(0.0))?;
    let (tap, x_override) = match (tap1, tap2) {
        (Some(_), Some(_)) => {
            return Err(CgmesError::UnsupportedTransformer {
                mrid: pt_mrid.to_string(),
                reason: "tap changers on both ends of a 2-winding transformer aren't supported".into(),
            })
        }
        (Some(t1), None) => (Complex::new(1.0, 0.0) / t1.tap, t1.x_override),
        (None, Some(t2)) => (t2.tap, None),
        (None, None) => (Complex::new(1.0, 0.0), None),
    };

    // z_base is anchored to end1's own nameplate `ratedU` (where the r/x/g/b
    // ohms/siemens values were actually measured) — NOT bus1's system rated
    // voltage. The two can genuinely differ by a few percent in real CGMES
    // data (e.g. a 220 kV nameplate end sitting on a 225 kV system-nominal
    // bus — confirmed against real fixture data, and against pypowsybl's own
    // CGMES import, which keeps such a nameplate end as its own distinct
    // voltage level rather than silently merging it into the bus's system
    // level). That mismatch is a genuine additional structural (non-tap-
    // changer) ideal-transformer ratio, folded into `tap` below — not
    // something to paper over by just picking a different z_base.
    let r1 = end1.r.unwrap_or(0.0);
    let x1 = x_override.unwrap_or_else(|| end1.x.unwrap_or(0.0));
    if r1 == 0.0 && x1 == 0.0 {
        return Err(CgmesError::UnsupportedTransformer {
            mrid: pt_mrid.to_string(),
            reason: "end 1 has zero series impedance (expected all series impedance on end 1)".into(),
        });
    }
    let g1 = end1.g.unwrap_or(0.0);
    let b1 = end1.b.unwrap_or(0.0);
    let u1 = end1.rated_u.ok_or_else(|| missing("PowerTransformerEnd", end1.mrid_str(), "ratedU"))? * 1e3;
    let z_base = u1 * u1 / s_base_va;
    // Structural ratio: end1's nameplate vs bus1's actual system base. This
    // is a "to"-side-inherent effect, so — mirroring the same reciprocal
    // convention applied above for a to-side tap changer — it enters as a
    // reciprocal on top of whatever the tap changer itself contributes.
    let structural = u1 / buses[bus1].u_rated;
    let tap = tap / structural;

    Ok(Transformer {
        from: bus2,
        to: bus1,
        from_status: terms.connected_via_terminal_mrid(&term2.mrid) as u8,
        to_status: terms.connected_via_terminal_mrid(&term1.mrid) as u8,
        y_series: Complex::new(z_base, 0.0) / Complex::new(r1, x1),
        y_shunt: Complex::new(g1, b1) * z_base,
        tap,
    })
}

/// Builds one leg of a 3-winding transformer's star equivalent: `to` = this
/// end's own physical bus (impedance naturally referenced to its own
/// `ratedU`, per CGMES's doc: "for a three Terminal PowerTransformer the
/// three ends represent a star equivalent with each leg... represented by
/// r/r0/x/x0" — i.e. each end already stands alone, unlike PGM's percentage
/// nameplate style which needs `three_winding_star_params`'s common-base
/// conversion), `from` = the synthesized star bus.
fn build_star_leg(
    ds: &CimDataset, tap_index: &TapChangerIndex, terms: &TerminalIndex, buses: &[Bus],
    end: &PowerTransformerEnd, star_idx: usize, s_base_va: f64,
) -> Result<Transformer, CgmesError> {
    let term = end.base.terminal.as_ref().ok_or_else(|| missing("PowerTransformerEnd", end.mrid_str(), "Terminal"))?;
    let bus = terms.bus_via_terminal_mrid(&term.mrid)
        .ok_or_else(|| CgmesError::UnresolvedReference { from_type: "PowerTransformerEnd", from_mrid: end.mrid_str().to_string(), field: "Terminal.TopologicalNode" })?;

    let effect = tap_index.effect_for_end(ds, end.mrid_str(), end.x.unwrap_or(0.0))?;
    let (tap, x_override) = match effect {
        Some(t) => (Complex::new(1.0, 0.0) / t.tap, t.x_override),
        None => (Complex::new(1.0, 0.0), None),
    };

    // z_base anchored to this leg's own end's nameplate `ratedU`; any
    // mismatch against this leg's own bus's system rated voltage is folded
    // into `tap` as a structural ratio — same reasoning as `build_two_winding`.
    let r = end.r.unwrap_or(0.0);
    let x = x_override.unwrap_or_else(|| end.x.unwrap_or(0.0));
    let g = end.g.unwrap_or(0.0);
    let b = end.b.unwrap_or(0.0);
    let u = end.rated_u.ok_or_else(|| missing("PowerTransformerEnd", end.mrid_str(), "ratedU"))? * 1e3;
    let z_base = u * u / s_base_va;
    let structural = u / buses[bus].u_rated;
    let tap = tap / structural;

    Ok(Transformer {
        from: star_idx,
        to: bus,
        from_status: 1, // the synthesized star bus itself is never "disconnected"
        to_status: terms.connected_via_terminal_mrid(&term.mrid) as u8,
        y_series: Complex::new(z_base, 0.0) / Complex::new(r, x),
        y_shunt: Complex::new(g, b) * z_base,
        tap,
    })
}

/// Small helper trait so `build_two_winding`/`build_star_leg` can get an
/// end's own mrid without importing `CimElement` at every call site.
trait MridStr {
    fn mrid_str(&self) -> &str;
}
impl MridStr for PowerTransformerEnd {
    fn mrid_str(&self) -> &str {
        cimstructs::base::CimElement::mrid(self)
    }
}
