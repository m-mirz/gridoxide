//! UCTE-DEF network import.
//!
//! UCTE-DEF is the fixed-column text format the continental-European TSOs used
//! for exchanging network snapshots before CGMES, and it is still what most
//! published remedial-action and capacity-calculation test material is written
//! in. `docs/src/reference/resources.md` lists it as "potential fallback path
//! if CGMES export isn't available"; this module promotes that note to an
//! importer, for a reason that has nothing to do with fallback:
//!
//! **UCTE carries the two things gridoxide's network model was missing.** A
//! `##L` record ends in a current rating and a `##R` record *is* a tap table:
//!
//! ```text
//! ##L
//! BBE1AA1  BBE2AA1  1 0 0.0000 10.000 0.000000   5000
//! //                                             ^^^^ permanent rating, amperes
//! ##R
//! BBE2AA1  BBE3AA1  1                    -0.68 90.00 16  0        SYMM
//! //                                     ^^^^^ ^^^^^ ^^ ^^        ^^^^
//! //                                     du%   theta n  n'        kind
//! ```
//!
//! so the limits of [`crate::ratings`] and the tap changers of
//! [`crate::types::TapChanger`] arrive populated rather than empty.
//!
//! # What is read
//!
//! `##C` (skipped), `##N` with its `##Z<country>` sub-headers, `##L`, `##T` and
//! `##R`. Across the 176 UCTE fixtures in the vendored OpenRAO checkout those
//! five are the only record types that ever appear — there is no `##TT`, no
//! `##E`, no `##DD` — so anything else is skipped with a note rather than
//! treated as an error.
//!
//! # Conventions worth stating, because getting them wrong is silent
//!
//! - **Generation is negative.** UCTE writes generation as a negative number in
//!   a field named "active power generation". The net injection at a node is
//!   therefore `(−generation) − load`, and the reactive limit fields arrive
//!   pre-swapped by the same negation.
//! - **Nominal voltage comes from the node code, not the record.** Character 7
//!   of an 8-character node code is a voltage-level class (`1` = 380 kV,
//!   `2` = 220 kV, …). The `voltage` field in the record is the *reference*
//!   value a PU node regulates to — commonly 400 kV on a 380 kV node, which is
//!   a per-unit set-point of 1.0526 and not a different base.
//! - **Transformer impedances are referred to side 2**, and its `##R` tap
//!   changer acts on side 1.
//! - **Files are Latin-1**, and columns are counted in bytes. Decoding as UTF-8
//!   would let one accented character in a node name shift every field after
//!   it, so this module slices bytes and decodes per byte.

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex;

use crate::network::ShuntAdm;
use crate::ratings::BranchLimits;
use crate::types::{Bus, BusType, Line, TapChanger, Transformer};

/// Default system base. UCTE states everything in MW/MVar and names no base of
/// its own, so one has to be chosen; 100 MVA is the universal convention.
pub const DEFAULT_BASE_MVA: f64 = 100.0;

#[derive(Debug)]
pub enum UcteError {
    Io(std::io::Error),
    /// A record referred to a node that no `##N` record defines.
    UnknownNode { record: String, node: String },
    /// A node code whose voltage-level character is not one of `0`–`9`.
    BadVoltageLevel { node: String, code: char },
    /// A line too short to hold even the fields that are mandatory.
    TruncatedRecord { block: char, line: usize },
    /// No node at all, so there is nothing to solve.
    NoNodes,
    /// A data record appeared before any `##` block header, so there is no way
    /// to know what it describes. The reference implementation rejects the
    /// same shape with "a node must be defined in a ##Z context".
    DataOutsideBlock { line: usize },
}

impl std::fmt::Display for UcteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UcteError::Io(e) => write!(f, "reading UCTE file: {e}"),
            UcteError::UnknownNode { record, node } => {
                write!(f, "record `{record}` refers to undefined node `{node}`")
            }
            UcteError::BadVoltageLevel { node, code } => {
                write!(f, "node `{node}` has unknown voltage-level code `{code}`")
            }
            UcteError::TruncatedRecord { block, line } => {
                write!(f, "##{block} record on line {line} is too short")
            }
            UcteError::NoNodes => write!(f, "file defines no nodes"),
            UcteError::DataOutsideBlock { line } => {
                write!(f, "line {line} carries data before any `##` block header")
            }
        }
    }
}

impl std::error::Error for UcteError {}

impl From<std::io::Error> for UcteError {
    fn from(e: std::io::Error) -> Self {
        UcteError::Io(e)
    }
}

/// How to pick the slack bus when the file does not say.
///
/// UCTE has a node type for it — `3`, "U and θ constant" — but **none of the
/// 176 vendored fixtures uses it**, because the tools that wrote them let the
/// load flow choose. gridoxide's solver needs one, so this is a decision the
/// importer has to make and had better make visibly.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum SlackPolicy {
    /// The regulating (`PU`) node with the largest generation, falling back to
    /// the largest generator of any type, then to node 0. Deterministic: ties
    /// break on the node code, so re-importing the same file picks the same
    /// bus.
    #[default]
    LargestGeneration,
    /// A named node code. Errors as [`UcteError::UnknownNode`] if absent.
    Node(String),
}

#[derive(Clone, Debug)]
pub struct UcteOptions {
    pub base_mva: f64,
    pub slack: SlackPolicy,
    /// Substitutions applied to the nominal voltage a node code implies, as
    /// `(from_volts, to_volts)`.
    ///
    /// UCTE's voltage-level classes are *classes*, and some processes operate
    /// them at a different figure: the CORE capacity-calculation convention
    /// runs the 380 kV class at 400 kV and the 220 kV class at 225. That is not
    /// cosmetic — the nominal voltage is the per-unit base, so a 5% change in it
    /// moves every susceptance by 11% and every ampere-to-MW conversion by 5%.
    ///
    /// Applied to the *class* nominal before anything is per-unitised, which is
    /// why it belongs here rather than in a caller's post-processing: rescaling
    /// an already-converted network correctly means touching impedances, shunts
    /// and tap ratios in three different directions.
    ///
    /// Empty by default, so the classes are used as UCTE defines them.
    pub nominal_voltages: Vec<(f64, f64)>,
}

impl UcteOptions {
    /// The CORE capacity-calculation convention: 380 kV class at 400, 220 at
    /// 225.
    pub fn core_capacity_calculation() -> Self {
        Self {
            nominal_voltages: vec![(380_000.0, 400_000.0), (220_000.0, 225_000.0)],
            ..Default::default()
        }
    }
}

impl Default for UcteOptions {
    fn default() -> Self {
        Self {
            base_mva: DEFAULT_BASE_MVA,
            slack: SlackPolicy::default(),
            nominal_voltages: Vec::new(),
        }
    }
}

/// A parsed UCTE file, in gridoxide's own types.
#[derive(Debug)]
pub struct UcteImport {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub transformers: Vec<Transformer>,
    pub shunts: Vec<ShuntAdm>,
    /// Bus index → the 8-character node code that produced it.
    pub node_codes: Vec<String>,
    /// ISO country code per bus, parallel to `buses`, from the file's
    /// `##Z<cc>` sub-headers. `None` where the file did not say.
    ///
    /// The only network fact `rao::search`'s "skip actions far from the most
    /// limiting element" filter needs, and one no other importer here supplies
    /// yet.
    pub bus_countries: Vec<Option<String>>,
    /// Node code → bus index.
    pub node_index: HashMap<String, usize>,
    /// Flat branch index (lines first, then transformers — the crate-wide
    /// convention `branch_flow::branch_params` defines) → the 19-character
    /// UCTE element id.
    pub branch_ids: Vec<String>,
    /// Flat branch index → operating limits. Same indexing as `branch_ids`.
    pub limits: Vec<BranchLimits>,
    /// Parallel to `transformers`: the tap changer, where the file gave one.
    pub tap_changers: Vec<Option<TapChanger>>,
    /// Bus index of the slack, and how it was chosen.
    pub slack: usize,
    /// Per-node active generation limits in per-unit, `(p_min, p_max)`, from
    /// the permissible-generation fields. Kept because a redispatch range
    /// action needs them and nothing in [`Bus`] has anywhere to put them.
    pub p_limits: Vec<Option<(f64, f64)>>,
    pub base_mva: f64,
    /// Everything the importer decided rather than read: skipped blocks,
    /// out-of-service branches, the slack choice. The equivalent of the
    /// reference implementation's `CracCreationContext` report — a silently
    /// dropped element is how an importer produces a plausible wrong answer.
    pub notes: Vec<String>,
    /// Element ids of branches left out because the file marked them out of
    /// operation (status 7, 8 or 9).
    /// Flat branch indices the file marks out of operation (status 7, 8 or 9).
    ///
    /// These branches are **kept in the model**, with their real impedance, and
    /// listed here as open. Dropping them would be simpler and would make a
    /// whole class of remedial action inexpressible: an automaton that *closes*
    /// a standby circuit is one of the commonest there is, and it cannot be
    /// applied to a branch the importer discarded. See
    /// [`as_switched`](Self::as_switched).
    pub initially_open: Vec<usize>,
    /// Element ids of those branches, parallel to `initially_open`.
    pub out_of_service: Vec<String>,
}

impl UcteImport {
    /// Working copies of the branch arrays with every initially-open branch
    /// made non-conducting.
    ///
    /// [`lines`](Self::lines) and [`transformers`](Self::transformers) hold
    /// **every** branch the file describes, open ones included, with their real
    /// impedance — that is what makes a "close this circuit" remedial action
    /// expressible at all. It also means handing those arrays straight to a
    /// power flow energises circuits the file says are out of service, so
    /// anything that just wants to solve the network as given should solve
    /// these instead.
    ///
    /// A consumer that tracks its own open set — the remedial-action layer does
    /// — should ignore this and pass [`initially_open`](Self::initially_open)
    /// through its own bookkeeping, so that closing a branch is a removal from
    /// that set rather than a second copy of the arrays.
    pub fn as_switched(&self) -> (Vec<Line>, Vec<Transformer>) {
        let mut lines = self.lines.clone();
        let mut transformers = self.transformers.clone();
        for &branch in &self.initially_open {
            if branch < lines.len() {
                lines[branch].r = crate::topology::reduction::OPEN_BRANCH_Z;
                lines[branch].x = crate::topology::reduction::OPEN_BRANCH_Z;
                lines[branch].b_shunt = 0.0;
                lines[branch].g_shunt = 0.0;
            } else if let Some(t) = transformers.get_mut(branch - lines.len()) {
                t.from_status = 0;
                t.to_status = 0;
            }
        }
        (lines, transformers)
    }

    pub fn s_base_va(&self) -> f64 {
        self.base_mva * 1e6
    }

    /// Number of branches, lines and transformers together.
    pub fn n_branches(&self) -> usize {
        self.lines.len() + self.transformers.len()
    }

    /// Flat branch index of transformer `i`.
    pub fn transformer_branch(&self, i: usize) -> usize {
        self.lines.len() + i
    }
}

// ---------------------------------------------------------------------------
// Fixed-column record access
// ---------------------------------------------------------------------------

/// One physical line, sliced by byte position.
///
/// Every accessor tolerates a short record: UCTE writers routinely omit
/// trailing fields rather than padding them, so an out-of-range slice is
/// "absent", not an error.
struct Record<'a> {
    raw: &'a [u8],
}

impl<'a> Record<'a> {
    fn new(raw: &'a [u8]) -> Self {
        Self { raw }
    }

    fn slice(&self, from: usize, to: usize) -> &'a [u8] {
        let lo = from.min(self.raw.len());
        let hi = to.min(self.raw.len());
        if lo >= hi { &[] } else { &self.raw[lo..hi] }
    }

    /// Latin-1 decode, keeping every byte as one character so that column
    /// arithmetic stays byte arithmetic.
    fn text(&self, from: usize, to: usize) -> String {
        self.slice(from, to).iter().map(|&b| b as char).collect()
    }

    fn trimmed(&self, from: usize, to: usize) -> String {
        self.text(from, to).trim().to_string()
    }

    fn number(&self, from: usize, to: usize) -> Option<f64> {
        let s = self.trimmed(from, to);
        if s.is_empty() { None } else { s.parse::<f64>().ok() }
    }

    fn integer(&self, from: usize, to: usize) -> Option<i32> {
        let s = self.trimmed(from, to);
        if s.is_empty() { None } else { s.parse::<i32>().ok() }
    }

    fn char_at(&self, i: usize) -> Option<char> {
        self.raw.get(i).map(|&b| b as char)
    }

    fn digit_at(&self, i: usize) -> Option<i32> {
        self.char_at(i).and_then(|c| c.to_digit(10)).map(|d| d as i32)
    }

    fn is_blank(&self) -> bool {
        self.raw.iter().all(|b| b.is_ascii_whitespace())
    }
}

/// Nominal voltage in volts for a UCTE voltage-level class character.
///
/// The ordinal ordering is the format's, not a sorted one — `8` and `9` were
/// appended when 330 kV and 500 kV networks joined.
fn voltage_level_v(code: char) -> Option<f64> {
    let kv = match code {
        '0' => 750.0,
        '1' => 380.0,
        '2' => 220.0,
        '3' => 150.0,
        '4' => 120.0,
        '5' => 110.0,
        '6' => 70.0,
        '7' => 27.0,
        '8' => 330.0,
        '9' => 500.0,
        _ => return None,
    };
    Some(kv * 1000.0)
}

// ---------------------------------------------------------------------------
// Raw records
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RawNode {
    code: String,
    /// The ISO code from the `##Z<cc>` sub-header this node appeared under.
    country: Option<String>,
    type_code: i32,
    voltage_reference: Option<f64>,
    active_load: Option<f64>,
    reactive_load: Option<f64>,
    active_generation: Option<f64>,
    reactive_generation: Option<f64>,
    min_p: Option<f64>,
    max_p: Option<f64>,
    min_q: Option<f64>,
    max_q: Option<f64>,
}

impl RawNode {
    /// UCTE writes generation negative; this is the physically-signed value.
    fn generation_mw(&self) -> f64 {
        -self.active_generation.unwrap_or(0.0)
    }

    fn generation_mvar(&self) -> f64 {
        -self.reactive_generation.unwrap_or(0.0)
    }

    fn regulates_voltage(&self) -> bool {
        // 2 = P and U constant, 3 = U and θ constant.
        self.type_code == 2 || self.type_code == 3
    }
}

#[derive(Clone, Debug)]
struct RawBranch {
    id: String,
    node1: String,
    node2: String,
    status: i32,
    r: f64,
    x: f64,
    b: f64,
    current_limit: Option<f64>,
    /// Transformer-only fields.
    transformer: Option<RawTransformerFields>,
}

#[derive(Clone, Debug)]
struct RawTransformerFields {
    rated_v1: f64,
    rated_v2: f64,
    g: f64,
}

#[derive(Clone, Debug, Default)]
struct RawRegulation {
    /// `##R` columns 20–38: voltage (ratio) regulation.
    ratio: Option<RatioRegulation>,
    /// `##R` columns 39–68: angle (phase-shifter) regulation.
    angle: Option<AngleRegulation>,
}

#[derive(Clone, Copy, Debug)]
struct RatioRegulation {
    du_percent: f64,
    n: i32,
    position: i32,
}

#[derive(Clone, Copy, Debug)]
struct AngleRegulation {
    du_percent: f64,
    theta_deg: f64,
    n: i32,
    position: i32,
    symmetrical: bool,
}

/// A branch is in service unless the file says otherwise. 7 is an open busbar
/// coupler, 8 a real element out of operation, 9 an equivalent one.
fn status_in_service(status: i32) -> bool {
    !matches!(status, 7 | 8 | 9)
}

/// Statuses 2 and 7 mark a busbar coupler — a switch, not a branch with
/// impedance.
fn status_is_coupler(status: i32) -> bool {
    matches!(status, 2 | 7)
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Read and convert a UCTE-DEF file.
pub fn read(path: impl AsRef<Path>) -> Result<UcteImport, UcteError> {
    read_with(path, &UcteOptions::default())
}

pub fn read_with(path: impl AsRef<Path>, options: &UcteOptions) -> Result<UcteImport, UcteError> {
    let bytes = std::fs::read(path)?;
    parse_with(&bytes, options)
}

/// Convert an in-memory UCTE-DEF document.
pub fn parse(bytes: &[u8]) -> Result<UcteImport, UcteError> {
    parse_with(bytes, &UcteOptions::default())
}

pub fn parse_with(bytes: &[u8], options: &UcteOptions) -> Result<UcteImport, UcteError> {
    let mut notes = Vec::new();
    let (nodes, branches, regulations) = parse_records(bytes, &mut notes)?;
    convert(nodes, branches, regulations, options, notes)
}

/// Which block a physical line belongs to. `None` for a block this importer
/// skips wholesale.
#[derive(Clone, Copy, PartialEq)]
enum Block {
    /// Before the first `##` header. Distinct from `Skip`, which is a block we
    /// deliberately ignore: data here is malformed, not unsupported.
    None,
    Nodes,
    Lines,
    Transformers,
    Regulations,
    Skip,
}

fn parse_records(
    bytes: &[u8],
    notes: &mut Vec<String>,
) -> Result<(Vec<RawNode>, Vec<RawBranch>, HashMap<String, RawRegulation>), UcteError> {
    let mut nodes = Vec::new();
    let mut branches = Vec::new();
    let mut regulations: HashMap<String, RawRegulation> = HashMap::new();
    let mut block = Block::None;
    // The country in force, from the most recent `##Z<cc>` sub-header.
    let mut country: Option<String> = None;
    let mut skipped: Vec<String> = Vec::new();

    for (n, raw) in bytes.split(|&b| b == b'\n').enumerate() {
        let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
        let record = Record::new(raw);
        if raw.starts_with(b"##") {
            // Longest names first: `##TT` must not be read as `##T`.
            let tag = record.text(2, raw.len());
            let tag = tag.trim_end();
            block = if tag.starts_with("TT") {
                if !skipped.contains(&"TT".to_string()) {
                    skipped.push("TT".to_string());
                }
                Block::Skip
            } else if tag.starts_with('N') || tag.starts_with('Z') {
                // `##Z<cc>` is a country sub-header *inside* the node block. It
                // carries no record of its own, so both open the same block —
                // but the two letters after the `Z` say which country every
                // node until the next such header belongs to, and that is the
                // only place a UCTE file states it.
                if let Some(code) = tag.strip_prefix('Z') {
                    let code = code.trim();
                    country = (!code.is_empty()).then(|| code.to_string());
                }
                Block::Nodes
            } else if tag.starts_with('L') {
                Block::Lines
            } else if tag.starts_with('T') {
                Block::Transformers
            } else if tag.starts_with('R') {
                Block::Regulations
            } else if tag.starts_with('C') {
                Block::Skip
            } else {
                let name = tag.split_whitespace().next().unwrap_or("").to_string();
                if !name.is_empty() && !skipped.contains(&name) {
                    skipped.push(name);
                }
                Block::Skip
            };
            continue;
        }
        if record.is_blank() {
            continue;
        }
        match block {
            Block::Nodes => {
                let mut node = parse_node(&record, n + 1)?;
                node.country = country.clone();
                nodes.push(node);
            }
            Block::Lines => branches.push(parse_line(&record, n + 1)?),
            Block::Transformers => branches.push(parse_transformer(&record, n + 1)?),
            Block::Regulations => {
                let (id, reg) = parse_regulation(&record, n + 1)?;
                regulations.insert(id, reg);
            }
            Block::Skip => {}
            Block::None => return Err(UcteError::DataOutsideBlock { line: n + 1 }),
        }
    }

    for name in skipped {
        notes.push(format!("skipped unsupported `##{name}` block"));
    }
    if nodes.is_empty() {
        return Err(UcteError::NoNodes);
    }
    Ok((nodes, branches, regulations))
}

fn parse_node(record: &Record<'_>, line: usize) -> Result<RawNode, UcteError> {
    if record.raw.len() < 25 {
        return Err(UcteError::TruncatedRecord { block: 'N', line });
    }
    Ok(RawNode {
        // Filled in by the caller, which is the only place that knows which
        // `##Z<cc>` sub-header this record fell under.
        country: None,
        code: record.text(0, 8),
        type_code: record.digit_at(24).unwrap_or(0),
        voltage_reference: record.number(26, 32),
        active_load: record.number(33, 40),
        reactive_load: record.number(41, 48),
        active_generation: record.number(49, 56),
        reactive_generation: record.number(57, 64),
        min_p: record.number(65, 72),
        max_p: record.number(73, 80),
        min_q: record.number(81, 88),
        max_q: record.number(89, 96),
    })
}

fn element_id(record: &Record<'_>) -> (String, String, String) {
    (record.text(0, 19), record.text(0, 8), record.text(9, 17))
}

fn parse_line(record: &Record<'_>, line: usize) -> Result<RawBranch, UcteError> {
    if record.raw.len() < 21 {
        return Err(UcteError::TruncatedRecord { block: 'L', line });
    }
    let (id, node1, node2) = element_id(record);
    Ok(RawBranch {
        id,
        node1,
        node2,
        status: record.digit_at(20).unwrap_or(0),
        r: record.number(22, 28).unwrap_or(0.0),
        x: record.number(29, 35).unwrap_or(0.0),
        // Susceptance is stated in microsiemens.
        b: record.number(36, 44).unwrap_or(0.0) * 1e-6,
        current_limit: record.number(45, 51),
        transformer: None,
    })
}

fn parse_transformer(record: &Record<'_>, line: usize) -> Result<RawBranch, UcteError> {
    if record.raw.len() < 21 {
        return Err(UcteError::TruncatedRecord { block: 'T', line });
    }
    let (id, node1, node2) = element_id(record);
    Ok(RawBranch {
        id,
        node1,
        node2,
        status: record.digit_at(20).unwrap_or(0),
        r: record.number(40, 46).unwrap_or(0.0),
        x: record.number(47, 53).unwrap_or(0.0),
        b: record.number(54, 62).unwrap_or(0.0) * 1e-6,
        current_limit: record.number(70, 76),
        transformer: Some(RawTransformerFields {
            rated_v1: record.number(22, 27).unwrap_or(0.0),
            rated_v2: record.number(28, 33).unwrap_or(0.0),
            g: record.number(63, 69).unwrap_or(0.0) * 1e-6,
        }),
    })
}

fn parse_regulation(record: &Record<'_>, line: usize) -> Result<(String, RawRegulation), UcteError> {
    if record.raw.len() < 19 {
        return Err(UcteError::TruncatedRecord { block: 'R', line });
    }
    let id = record.text(0, 19);
    let mut reg = RawRegulation::default();

    let du = record.number(20, 25);
    let n = record.integer(26, 28);
    let np = record.integer(29, 32);
    if du.is_some() || n.is_some() || np.is_some() {
        reg.ratio = Some(RatioRegulation {
            du_percent: du.unwrap_or(0.0),
            n: n.unwrap_or(0),
            position: np.unwrap_or(0),
        });
    }

    let adu = record.number(39, 44);
    let theta = record.number(45, 50);
    let an = record.integer(51, 53);
    let anp = record.integer(54, 57);
    let kind = record.trimmed(64, 68);
    if adu.is_some() || theta.is_some() || an.is_some() || anp.is_some() || !kind.is_empty() {
        reg.angle = Some(AngleRegulation {
            du_percent: adu.unwrap_or(0.0),
            theta_deg: theta.unwrap_or(0.0),
            n: an.unwrap_or(0),
            position: anp.unwrap_or(0),
            // ASYM is the other spelling; anything unrecognised is treated as
            // symmetrical, which is what every vendored fixture but one uses.
            symmetrical: !kind.eq_ignore_ascii_case("ASYM"),
        });
    }

    let _ = line;
    Ok((id, reg))
}

// ---------------------------------------------------------------------------
// Tap mathematics
// ---------------------------------------------------------------------------

/// The complex multiplier a ratio (voltage) regulation applies to the side-1
/// winding voltage at tap `position`.
///
/// UCTE states the step as a percentage of the winding voltage, so position
/// `n` raises side 1 by `n·δu%`. A higher side-1 winding voltage means more
/// side-1 volts are needed per side-2 volt, which is an *increase* in the
/// from-side tap ratio — the direction is easy to invert and is pinned by
/// `a_ratio_tap_raises_the_from_side_ratio` below.
fn ratio_multiplier(reg: &RatioRegulation, position: i32) -> Complex<f64> {
    Complex::new(1.0 + position as f64 * reg.du_percent / 100.0, 0.0)
}

/// The complex multiplier an angle (phase-shifter) regulation applies at tap
/// `position`.
///
/// The regulation injects a voltage of magnitude `n·δu%` at angle `θ` relative
/// to the winding voltage, giving a boost phasor `1 + dx + j·dy`. For an
/// **asymmetrical** changer that phasor is the whole answer: it changes both
/// the magnitude and the angle of the ratio.
///
/// A **symmetrical** changer splits the injection half on each side, so the
/// magnitude cancels and only the angle survives — `|ratio| = 1` and the shift
/// is twice the half-angle. That is the case essentially every real
/// phase-shifter fixture uses, and the reason a PST can move flow without
/// moving voltage.
fn angle_multiplier(reg: &AngleRegulation, position: i32) -> Complex<f64> {
    let du = position as f64 * reg.du_percent / 100.0;
    let theta = reg.theta_deg.to_radians();
    let dx = du * theta.cos();
    let dy = du * theta.sin();
    if reg.symmetrical {
        let alpha = 2.0 * (dy / 2.0).atan2(1.0 + dx);
        Complex::from_polar(1.0, alpha)
    } else {
        Complex::new(1.0 + dx, dy)
    }
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

fn convert(
    nodes: Vec<RawNode>,
    branches: Vec<RawBranch>,
    regulations: HashMap<String, RawRegulation>,
    options: &UcteOptions,
    mut notes: Vec<String>,
) -> Result<UcteImport, UcteError> {
    let s_base_va = options.base_mva * 1e6;

    let mut buses = Vec::with_capacity(nodes.len());
    let mut node_codes = Vec::with_capacity(nodes.len());
    let mut bus_countries = Vec::with_capacity(nodes.len());
    let mut node_index = HashMap::with_capacity(nodes.len());
    let mut p_limits = Vec::with_capacity(nodes.len());
    let mut x_nodes = 0usize;

    for (idx, node) in nodes.iter().enumerate() {
        let level = node.code.chars().nth(6).unwrap_or(' ');
        let class = voltage_level_v(level)
            .ok_or_else(|| UcteError::BadVoltageLevel { node: node.code.clone(), code: level })?;
        let u_rated = options
            .nominal_voltages
            .iter()
            .find(|(from, _)| (class - from).abs() < 1.0)
            .map(|(_, to)| *to)
            .unwrap_or(class);
        if node.code.starts_with('X') {
            x_nodes += 1;
        }

        let bus_type = match node.type_code {
            3 => BusType::Slack,
            2 => BusType::PV,
            _ => BusType::PQ,
        };
        let voltage_mag = match (node.regulates_voltage(), node.voltage_reference) {
            (true, Some(v)) if v > 0.0 => v * 1000.0 / u_rated,
            _ => 1.0,
        };
        // Net injection: generation (sign-corrected) minus load.
        let p_mw = node.generation_mw() - node.active_load.unwrap_or(0.0);
        let q_mvar = node.generation_mvar() - node.reactive_load.unwrap_or(0.0);
        // The permissible-generation fields are negated too, which swaps which
        // of the pair is the lower bound.
        let (q_min, q_max) = match (node.min_q, node.max_q) {
            (Some(lo), Some(hi)) => (-lo * 1e6 / s_base_va, -hi * 1e6 / s_base_va),
            _ => (f64::NEG_INFINITY, f64::INFINITY),
        };

        buses.push(Bus {
            idx,
            bus_type,
            voltage_mag,
            voltage_ang: 0.0,
            p_spec: p_mw * 1e6 / s_base_va,
            q_spec: q_mvar * 1e6 / s_base_va,
            q_min,
            q_max,
            u_rated,
            zip_terms: Vec::new(),
        });
        p_limits.push(match (node.min_p, node.max_p) {
            (Some(lo), Some(hi)) => Some((-lo * 1e6 / s_base_va, -hi * 1e6 / s_base_va)),
            _ => None,
        });
        node_index.insert(node.code.clone(), idx);
        node_codes.push(node.code.clone());
        bus_countries.push(node.country.clone());
    }

    if !options.nominal_voltages.is_empty() {
        notes.push(format!(
            "nominal voltages substituted: {}",
            options
                .nominal_voltages
                .iter()
                .map(|(a, b)| format!("{:.0}->{:.0} kV", a / 1000.0, b / 1000.0))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if x_nodes > 0 {
        notes.push(format!(
            "{x_nodes} X-node(s) kept as ordinary buses; boundary-line semantics are not modelled"
        ));
    }

    // Branches, split into the crate-wide flat order: lines then transformers.
    let mut lines = Vec::new();
    let mut line_ids = Vec::new();
    let mut line_limits = Vec::new();
    let mut transformers = Vec::new();
    let mut transformer_ids = Vec::new();
    let mut transformer_limits = Vec::new();
    let mut tap_changers = Vec::new();
    let mut out_of_service = Vec::new();
    let mut initially_open: Vec<usize> = Vec::new();
    let mut initially_open_transformers: Vec<usize> = Vec::new();
    let mut couplers = 0usize;

    for branch in &branches {
        let node1 = *node_index.get(&branch.node1).ok_or_else(|| UcteError::UnknownNode {
            record: branch.id.clone(),
            node: branch.node1.clone(),
        })?;
        let node2 = *node_index.get(&branch.node2).ok_or_else(|| UcteError::UnknownNode {
            record: branch.id.clone(),
            node: branch.node2.clone(),
        })?;
        // Lines keep the file's order. **Transformers are reversed**: UCTE
        // refers a transformer's impedance to its node-1 side and puts the tap
        // changer on node 2, whereas gridoxide's `branch_calc_param` — the
        // MATPOWER convention — wants the series admittance at `to` and the
        // complex ratio at `from`. Mapping `from = node2`, `to = node1` lines
        // the two up exactly, and is the same swap the reference importer makes
        // (`UcteImporter.createTransformers` sets `VoltageLevel1` from
        // `ucteVoltageLevel2`). Getting this wrong is invisible on a 400/400
        // phase shifter and a 3% error on a 400/225 unit.
        let (from, to) = match branch.transformer {
            None => (node1, node2),
            Some(_) => (node2, node1),
        };

        let in_service = status_in_service(branch.status);
        if !in_service {
            out_of_service.push(branch.id.clone());
        }
        if status_is_coupler(branch.status) {
            couplers += 1;
        }

        let limits = match branch.current_limit {
            Some(a) if a > 0.0 => BranchLimits::permanent(a),
            _ => BranchLimits::default(),
        };

        match &branch.transformer {
            None => {
                let z_base = buses[from].u_rated * buses[from].u_rated / s_base_va;
                let (r, x) = crate::topology::reduction::clamp_branch_impedance(
                    branch.r / z_base,
                    branch.x / z_base,
                );
                lines.push(Line {
                    from,
                    to,
                    r,
                    x,
                    b_shunt: branch.b * z_base,
                    g_shunt: 0.0,
                });
                if !in_service {
                    initially_open.push(lines.len() - 1);
                }
                line_ids.push(branch.id.clone());
                line_limits.push(limits);
            }
            Some(t) => {
                // Impedances are referred to side 2, so they per-unit on the
                // to-bus base.
                let z_base = buses[to].u_rated * buses[to].u_rated / s_base_va;
                let (r, x) = crate::topology::reduction::clamp_branch_impedance(
                    branch.r / z_base,
                    branch.x / z_base,
                );
                let y_series = Complex::new(1.0, 0.0) / Complex::new(r, x);
                let y_shunt = Complex::new(t.g * z_base, branch.b * z_base);

                // Nominal from-side ratio, expressed on the two buses' own
                // per-unit bases: `k = (V1r/Vn1) / (V2r/Vn2)`.
                // `from` is UCTE node 2 and `to` is node 1, so the rated
                // voltages pair up the other way round.
                let nominal = if t.rated_v1 > 0.0 && t.rated_v2 > 0.0 {
                    (t.rated_v2 * 1000.0 / buses[from].u_rated)
                        / (t.rated_v1 * 1000.0 / buses[to].u_rated)
                } else {
                    1.0
                };

                let regulation = regulations.get(&branch.id);
                let (tap, changer) = build_tap(nominal, regulation);
                transformers.push(Transformer {
                    from,
                    to,
                    from_status: 1,
                    to_status: 1,
                    y_series,
                    y_shunt,
                    tap,
                });
                transformer_ids.push(branch.id.clone());
                transformer_limits.push(limits);
                tap_changers.push(changer);
                if !in_service {
                    // Recorded against the *flat* index, which for a
                    // transformer is offset by the line count — resolved below,
                    // once that count is final.
                    initially_open_transformers.push(transformers.len() - 1);
                }
            }
        }
    }

    if couplers > 0 {
        notes.push(format!(
            "{couplers} closed busbar coupler(s) imported as near-zero-impedance branches"
        ));
    }
    for i in initially_open_transformers {
        initially_open.push(lines.len() + i);
    }
    initially_open.sort_unstable();
    if !out_of_service.is_empty() {
        notes.push(format!(
            "{} branch(es) present but open (out of operation)",
            out_of_service.len()
        ));
    }
    let unmatched = regulations.len().saturating_sub(
        transformer_ids.iter().filter(|id| regulations.contains_key(*id)).count(),
    );
    if unmatched > 0 {
        notes.push(format!("{unmatched} `##R` record(s) matched no in-service transformer"));
    }

    let mut branch_ids = line_ids;
    branch_ids.extend(transformer_ids);
    let mut limits = line_limits;
    limits.extend(transformer_limits);

    let slack = choose_slack(&nodes, &node_index, &options.slack, &mut notes)?;
    buses[slack].bus_type = BusType::Slack;

    Ok(UcteImport {
        buses,
        lines,
        transformers,
        shunts: Vec::new(),
        node_codes,
        bus_countries,
        node_index,
        branch_ids,
        limits,
        tap_changers,
        slack,
        p_limits,
        base_mva: options.base_mva,
        notes,
        initially_open,
        out_of_service,
    })
}

/// Build the current tap value and, when the file gave a regulation, the whole
/// tap table.
///
/// When a transformer has *both* regulations the table is built over the angle
/// positions with the ratio held at its current position. That is an
/// approximation, and it is the same one the reference implementation makes,
/// for the same reason: the exact answer is a two-dimensional table indexed by
/// both tap numbers, and no downstream consumer models a transformer with two
/// independent changers. One vendored fixture out of 176 is affected.
fn build_tap(nominal: f64, regulation: Option<&RawRegulation>) -> (Complex<f64>, Option<TapChanger>) {
    let base = Complex::new(nominal, 0.0);
    let Some(reg) = regulation else { return (base, None) };

    let ratio_at = |pos: i32| reg.ratio.map(|r| ratio_multiplier(&r, pos)).unwrap_or(Complex::new(1.0, 0.0));
    let angle_at = |pos: i32| reg.angle.map(|a| angle_multiplier(&a, pos)).unwrap_or(Complex::new(1.0, 0.0));

    // The angle regulation owns the tap table when present, since that is what
    // a phase-shifter remedial action moves.
    let (n, position, use_angle) = match (&reg.angle, &reg.ratio) {
        (Some(a), _) => (a.n, a.position, true),
        (None, Some(r)) => (r.n, r.position, false),
        (None, None) => return (base, None),
    };
    if n <= 0 {
        let tap = base * ratio_at(reg.ratio.map(|r| r.position).unwrap_or(0))
            * angle_at(reg.angle.map(|a| a.position).unwrap_or(0));
        return (tap, None);
    }

    let held_ratio = ratio_at(reg.ratio.map(|r| r.position).unwrap_or(0));
    let held_angle = angle_at(reg.angle.map(|a| a.position).unwrap_or(0));
    let steps: Vec<Complex<f64>> = (-n..=n)
        .map(|i| {
            if use_angle {
                base * held_ratio * angle_at(i)
            } else {
                base * ratio_at(i) * held_angle
            }
        })
        .collect();

    let position = position.clamp(-n, n);
    let changer = TapChanger { low: -n, position, neutral: 0, steps };
    let tap = changer.current().unwrap_or(base);
    (tap, Some(changer))
}

fn choose_slack(
    nodes: &[RawNode],
    node_index: &HashMap<String, usize>,
    policy: &SlackPolicy,
    notes: &mut Vec<String>,
) -> Result<usize, UcteError> {
    if let SlackPolicy::Node(code) = policy {
        return node_index.get(code).copied().ok_or_else(|| UcteError::UnknownNode {
            record: "slack policy".to_string(),
            node: code.clone(),
        });
    }

    // The file's own answer, if it gave one.
    if let Some(i) = nodes.iter().position(|n| n.type_code == 3) {
        return Ok(i);
    }

    // Otherwise the largest generator, preferring one that regulates voltage.
    // `total_cmp` on the negated generation plus the node code makes the choice
    // independent of hash order, so two imports of one file agree.
    let pick = |filter: &dyn Fn(&RawNode) -> bool| -> Option<usize> {
        nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| filter(n) && n.generation_mw() > 0.0)
            .max_by(|(_, a), (_, b)| {
                a.generation_mw().total_cmp(&b.generation_mw()).then_with(|| b.code.cmp(&a.code))
            })
            .map(|(i, _)| i)
    };

    let chosen = pick(&|n: &RawNode| n.regulates_voltage())
        .or_else(|| pick(&|_: &RawNode| true))
        .unwrap_or(0);
    notes.push(format!(
        "no `U and theta constant` node in file; slack chosen as `{}` (largest generation)",
        nodes[chosen].code
    ));
    Ok(chosen)
}
