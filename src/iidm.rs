//! PowSyBl IIDM (`.xiidm`) network import.
//!
//! IIDM is the native serialization of powsybl-core's network model, and the
//! second of the two formats the published remedial-action test material comes
//! in — the other being [UCTE-DEF](crate::ucte).
//!
//! # Version tolerance is the design constraint
//!
//! The 51 IIDM fixtures in the vendored OpenRAO checkout span **eleven schema
//! versions**, `1_0` through `1_16`, and the model moved underneath them. The
//! clearest example is limits, which appear in two entirely different shapes:
//!
//! ```xml
//! <!-- older -->                     <!-- newer -->
//! <iidm:currentLimits1               <iidm:operationalLimitsGroup1 id="DEFAULT">
//!     permanentLimit="721.7"/>           <iidm:currentLimits permanentLimit="5000.0"/>
//!                                    </iidm:operationalLimitsGroup1>
//! ```
//!
//! So this importer accepts **any** `1_*` namespace, understands both
//! spellings, and **skips elements it does not recognise** instead of failing.
//! That is not laxity: a parser pinned to one version reads a fifth of the
//! available fixtures today and breaks against the next powsybl release. What
//! it does not do is skip *silently* — every unknown element type is counted
//! and named in [`IidmImport::notes`].
//!
//! # One topology, two spellings
//!
//! IIDM voltage levels come in two kinds and this importer folds them into one
//! representation. A `busBreakerTopology` names its buses; a
//! `nodeBreakerTopology` numbers its nodes and connects them with switches.
//! Both become nodes in a single [`NodeBreakerTopology`], and the bus view is
//! then whatever [`bus_view`] makes of it — which for a bus-breaker file with
//! no switches is exactly the buses the file declared, and for a node-breaker
//! file is the connected components of its closed switches.
//!
//! Folding them together is what lets the *same* downstream code serve both,
//! and it is why a topological remedial action will be able to act on an IIDM
//! network at all: the switches survive import as first-class objects rather
//! than being resolved away.

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex;
use quick_xml::events::Event;
use quick_xml::Reader;

use crate::network::ShuntAdm;
use crate::ratings::{BranchLimits, TemporaryLimit};
use crate::topology::bus_view::{bus_view, BusView, RetentionPolicy};
use crate::topology::model::{NodeBreakerTopology, NodeIdx, Switch, SwitchKind};
use crate::types::{Bus, BusType, Line, TapChanger, Transformer};

/// Default system base. IIDM states powers in MW/MVar and names no base.
pub const DEFAULT_BASE_MVA: f64 = 100.0;

#[derive(Debug)]
pub enum IidmError {
    Io(std::io::Error),
    Xml(String),
    /// An element referred to a voltage level or bus the file never declared.
    UnknownReference { element: String, reference: String },
    /// The document contained no `voltageLevel` at all.
    NoVoltageLevels,
}

impl std::fmt::Display for IidmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IidmError::Io(e) => write!(f, "reading IIDM file: {e}"),
            IidmError::Xml(e) => write!(f, "malformed IIDM XML: {e}"),
            IidmError::UnknownReference { element, reference } => {
                write!(f, "`{element}` refers to undeclared `{reference}`")
            }
            IidmError::NoVoltageLevels => write!(f, "document declares no voltage levels"),
        }
    }
}

impl std::error::Error for IidmError {}

impl From<std::io::Error> for IidmError {
    fn from(e: std::io::Error) -> Self {
        IidmError::Io(e)
    }
}

impl From<quick_xml::Error> for IidmError {
    fn from(e: quick_xml::Error) -> Self {
        IidmError::Xml(e.to_string())
    }
}

/// How to pick the slack bus.
///
/// IIDM has no slack attribute of its own — powsybl carries it as a
/// `slackTerminal` *extension*, which most files omit — so this is a decision
/// the importer makes, exactly as it does for [UCTE](crate::ucte::SlackPolicy).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum SlackPolicy {
    /// The bus with the largest generation, preferring one whose generator
    /// regulates voltage. Ties break on the bus id, so two imports agree.
    #[default]
    LargestGeneration,
    /// A named bus id.
    Bus(String),
}

#[derive(Clone, Debug)]
pub struct IidmOptions {
    pub base_mva: f64,
    pub slack: SlackPolicy,
    /// Which switches survive into the bus view. The default merges every
    /// closed switch, reproducing powsybl's own calculated bus view; retaining
    /// them instead is what a topological remedial action needs.
    pub retention: RetentionPolicy,
}

impl Default for IidmOptions {
    fn default() -> Self {
        Self {
            base_mva: DEFAULT_BASE_MVA,
            slack: SlackPolicy::default(),
            retention: RetentionPolicy::MergeAll,
        }
    }
}

/// A parsed IIDM document, in gridoxide's own types.
/// The `<iidm:area>` elements a file declares, resolved onto bus indices.
///
/// IIDM is the only one of the three importers that states area **membership**
/// directly: `<voltageLevelRef>` lists what is inside, where CGMES states only
/// the boundary and leaves membership to be derived, and UCTE states a country
/// code per node and no schedule at all.
#[derive(Clone, Debug, Default)]
pub struct IidmAreas {
    /// Area index per bus, indexed by [`Bus::idx`]; `None` for a bus no area
    /// claims. Feeds
    /// [`AreaDefinition::of_bus`](crate::outerloop::AreaDefinition::of_bus).
    pub of_bus: Vec<Option<usize>>,
    /// Scheduled net **export** per area, per-unit.
    ///
    /// IIDM states `interchangeTarget` in load sign convention — "negative is
    /// export, positive is import", per `Area.java`'s own javadoc — and
    /// [`AreaDefinition::targets`](crate::outerloop::AreaDefinition::targets)
    /// is an export, so this is its negation. The same flip CGMES's
    /// `netInterchange` needs, for the same reason.
    pub targets: Vec<f64>,
    /// `(id, name)` per area, in declaration order.
    pub ids: Vec<(String, String)>,
    pub report: IidmAreaReport,
}

/// What an area import could not use, counted rather than dropped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IidmAreaReport {
    /// `ControlArea`s converted.
    pub areas: usize,
    /// Areas of some other `areaType` — a `BiddingZone`, say. Recognised and
    /// skipped: they partition the network for a different purpose.
    pub other_types: usize,
    /// Areas stating no `interchangeTarget`; their target is left at zero,
    /// which asks the area to serve its own load rather than being a schedule
    /// read from the file.
    pub without_target: Vec<usize>,
    /// `<voltageLevelRef>`s naming a voltage level this file does not define.
    pub unknown_voltage_levels: usize,
    /// `<areaBoundary>` elements seen and not read — gridoxide derives the
    /// boundary from membership, so a stated one is redundant.
    pub boundaries_ignored: usize,
    /// Buses whose nodes are claimed by more than one area. First wins.
    pub contested_buses: usize,
    /// Buses no area claims.
    pub unassigned_buses: usize,
}

#[derive(Debug)]
pub struct IidmImport {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub transformers: Vec<Transformer>,
    pub shunts: Vec<ShuntAdm>,
    /// Bus index → a readable label. For a bus-breaker file this is the file's
    /// own bus id; for a node-breaker file it names the voltage level and the
    /// nodes that merged.
    pub bus_labels: Vec<String>,
    /// Flat branch index (lines then transformers) → the element's IIDM id.
    pub branch_ids: Vec<String>,
    /// Flat branch index → limits, side 1 and side 2. IIDM states them per
    /// terminal and a transformer's two windings genuinely differ, so they are
    /// kept apart.
    pub limits: Vec<[BranchLimits; 2]>,
    /// Parallel to `transformers`.
    pub tap_changers: Vec<Option<TapChanger>>,
    /// The control areas the file declares, if any.
    pub areas: IidmAreas,
    /// The regulating controls the file's `regulating`/`targetV`/
    /// `regulationValue` attributes declare, resolved onto this import's own
    /// indices.
    ///
    /// A `phaseTapChanger` in `CURRENT_LIMITER` mode is skipped: it holds a
    /// current rather than a power, which no outer loop here models. A
    /// `FIXED_TAP` one is skipped because that is what it means.
    pub regulation: Vec<crate::outerloop::TapRegulation>,
    /// The switch graph, retained. This is what makes a topological remedial
    /// action expressible on an IIDM network.
    pub topology: NodeBreakerTopology,
    pub view: BusView,
    /// Parallel to `topology.switches`.
    pub switch_ids: Vec<String>,
    pub slack: usize,
    pub base_mva: f64,
    /// The `1_x` schema version the document declared, if it declared one.
    pub version: Option<String>,
    pub notes: Vec<String>,
    /// Elements omitted because a terminal was disconnected.
    pub disconnected: Vec<String>,
    /// Every generator and load, by its own IIDM id, with the bus it sits on
    /// and the injection the file states.
    ///
    /// The importer folds each of these into its bus's net `p_spec`/`q_spec`,
    /// which is all a power flow needs and is not reversible afterwards.
    /// Anything that has to address an *individual* machine needs the
    /// correspondence back — a Dynawo dynamic model attaches to a generator by
    /// `staticId`, and a dynamic study needs each machine's own terminal power
    /// rather than its bus's total.
    pub injections: Vec<IidmInjection>,
}

/// One generator or load, addressable by the id its file gave it.
#[derive(Clone, Debug, PartialEq)]
pub struct IidmInjection {
    pub id: String,
    pub bus: usize,
    /// A generator rather than a load. Note that a load with negative `p` is
    /// still a load: the distinction is the element type the file used, not the
    /// sign of the power.
    pub generator: bool,
    /// Active injection, per unit on the network base, **generation positive**
    /// — the same convention `Bus::p_spec` uses.
    pub p: f64,
    pub q: f64,
}

impl IidmImport {
    pub fn s_base_va(&self) -> f64 {
        self.base_mva * 1e6
    }

    pub fn n_branches(&self) -> usize {
        self.lines.len() + self.transformers.len()
    }

    pub fn transformer_branch(&self, i: usize) -> usize {
        self.lines.len() + i
    }
}

// ---------------------------------------------------------------------------
// Raw records
// ---------------------------------------------------------------------------

/// Where a terminal attaches: a voltage level plus a key that is a bus id in a
/// bus-breaker level and a node number in a node-breaker one.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NodeKey {
    voltage_level: String,
    key: String,
}

#[derive(Clone, Debug)]
struct RawVoltageLevel {
    id: String,
    nominal_v: f64,
}

#[derive(Clone, Debug)]
struct RawSwitch {
    id: String,
    a: NodeKey,
    b: NodeKey,
    kind: SwitchKind,
    open: bool,
}

#[derive(Clone, Debug, Default)]
struct RawLimits {
    permanent: Option<f64>,
    temporary: Vec<TemporaryLimit>,
}

#[derive(Clone, Debug)]
struct RawBranch {
    id: String,
    end1: Option<NodeKey>,
    end2: Option<NodeKey>,
    r: f64,
    x: f64,
    g1: f64,
    b1: f64,
    g2: f64,
    b2: f64,
    limits: [RawLimits; 2],
    /// `Some` for a two-winding transformer.
    transformer: Option<RawTransformerFields>,
}

#[derive(Clone, Debug, Default)]
struct RawTransformerFields {
    rated_u1: f64,
    rated_u2: f64,
    ratio: Option<RawTapChanger>,
    phase: Option<RawTapChanger>,
}

#[derive(Clone, Debug, Default)]
struct RawTapChanger {
    low: i32,
    position: i32,
    /// `(rho, alpha_deg)` per step, from `low` upwards.
    steps: Vec<(f64, f64)>,
    /// `regulating`: whether this changer is holding anything at all.
    regulating: bool,
    /// `regulationMode`, for a phase tap changer: `CURRENT_LIMITER`,
    /// `ACTIVE_POWER_CONTROL` or `FIXED_TAP`. A ratio tap changer has no such
    /// attribute and holds a voltage by construction.
    mode: Option<String>,
    /// `targetV` (kV) for a ratio changer, `regulationValue` (MW) for a phase
    /// one.
    target: Option<f64>,
    /// `targetDeadband`, in the target's own unit.
    deadband: Option<f64>,
}

#[derive(Clone, Debug)]
struct RawInjection {
    at: NodeKey,
    /// The element's own IIDM id, retained so an individual machine stays
    /// addressable after its power has been folded into its bus.
    id: String,
    /// Whether this is a generator rather than a load.
    generator: bool,
    /// Net active injection in MW, generation positive.
    p: f64,
    q: f64,
    /// `Some(target_kv)` when this injection regulates its bus voltage.
    regulates: Option<f64>,
    /// Generation in MW, used only for the slack heuristic.
    generation: f64,
    q_min: f64,
    q_max: f64,
}

#[derive(Clone, Debug)]
struct RawShunt {
    at: NodeKey,
    /// Susceptance and conductance in siemens, already multiplied by the
    /// section count.
    g: f64,
    b: f64,
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

/// The attributes of one element, decoded once.
struct Attrs(HashMap<String, String>);

impl Attrs {
    fn of(e: &quick_xml::events::BytesStart<'_>) -> Result<Self, IidmError> {
        let mut map = HashMap::new();
        for a in e.attributes() {
            let a = a.map_err(|e| IidmError::Xml(e.to_string()))?;
            let key = String::from_utf8_lossy(a.key.local_name().as_ref()).into_owned();
            // Decoded by hand rather than through `decode_and_unescape_value`,
            // which needs a `Decoder`. `Decoder`'s shape depends on whether
            // quick-xml's `encoding` feature is on, and cimdecoder turns it on
            // — so constructing one compiles under `--features iidm` and fails
            // under `--features iidm,cgmes`. Feature unification makes that a
            // real configuration, not a hypothetical one.
            let raw = String::from_utf8_lossy(&a.value);
            let value = quick_xml::escape::unescape(&raw)
                .map_err(|e| IidmError::Xml(e.to_string()))?
                .into_owned();
            map.insert(key, value);
        }
        Ok(Self(map))
    }

    fn text(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(|s| s.as_str())
    }

    fn string(&self, key: &str) -> Option<String> {
        self.0.get(key).cloned()
    }

    fn number(&self, key: &str) -> Option<f64> {
        self.0.get(key).and_then(|s| s.parse::<f64>().ok())
    }

    fn or_zero(&self, key: &str) -> f64 {
        self.number(key).unwrap_or(0.0)
    }

    fn integer(&self, key: &str) -> Option<i32> {
        self.0.get(key).and_then(|s| s.parse::<i32>().ok())
    }

    fn flag(&self, key: &str) -> Option<bool> {
        self.0.get(key).and_then(|s| s.parse::<bool>().ok())
    }

    /// The node this terminal attaches to, for suffix `""`, `"1"` or `"2"`.
    ///
    /// Returns `None` when the terminal is **disconnected**: IIDM marks that by
    /// giving `connectableBus` without `bus`, which is a real state and not a
    /// missing field.
    fn terminal(&self, suffix: &str, fallback_level: Option<&str>) -> Option<NodeKey> {
        let level = self
            .string(&format!("voltageLevelId{suffix}"))
            .or_else(|| fallback_level.map(|s| s.to_string()))?;
        if let Some(node) = self.text(&format!("node{suffix}")) {
            return Some(NodeKey { voltage_level: level, key: format!("#{node}") });
        }
        let bus = self.text(&format!("bus{suffix}"))?;
        Some(NodeKey { voltage_level: level, key: bus.to_string() })
    }
}

/// The boundary node two half-lines meet at.
///
/// `pairingKey` (newer) and `ucteXnodeCode` (older) are the same idea under two
/// names: the X-node both halves of a tie line reference. Falling back to the
/// element's own id keeps an *unpaired* dangling line working — it simply gets
/// a boundary of its own, which is what an unpaired dangling line is.
///
/// The key is placed in a voltage level of its own so that it can never collide
/// with a real bus id.
fn boundary_key(attrs: &Attrs, id: &str) -> NodeKey {
    let key = attrs
        .string("pairingKey")
        .or_else(|| attrs.string("ucteXnodeCode"))
        .unwrap_or_else(|| id.to_string());
    NodeKey { voltage_level: BOUNDARY_LEVEL.to_string(), key }
}

/// Voltage-level id reserved for boundary nodes.
const BOUNDARY_LEVEL: &str = "\u{0}boundary";

fn switch_kind(name: Option<&str>) -> SwitchKind {
    match name.unwrap_or("") {
        "BREAKER" => SwitchKind::Breaker,
        "DISCONNECTOR" => SwitchKind::Disconnector,
        "LOAD_BREAK_SWITCH" => SwitchKind::LoadBreakSwitch,
        _ => SwitchKind::Generic,
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

pub fn read(path: impl AsRef<Path>) -> Result<IidmImport, IidmError> {
    read_with(path, &IidmOptions::default())
}

pub fn read_with(path: impl AsRef<Path>, options: &IidmOptions) -> Result<IidmImport, IidmError> {
    let text = std::fs::read_to_string(path)?;
    parse_with(&text, options)
}

pub fn parse(text: &str) -> Result<IidmImport, IidmError> {
    parse_with(text, &IidmOptions::default())
}

/// Everything the streaming pass collects, before indices exist.
/// An `<iidm:area>`, as the file states it.
#[derive(Clone, Debug, Default)]
struct RawArea {
    id: String,
    name: String,
    /// `ControlArea`, `BiddingZone`, or anything else a file cares to define.
    /// Only `ControlArea` becomes a control area here.
    area_type: String,
    /// `interchangeTarget`, MW, in IIDM's **load** sign convention: negative is
    /// export, positive is import.
    target_mw: Option<f64>,
    /// The voltage levels this area contains. IIDM states membership
    /// *directly*, unlike CGMES, which states only the boundary.
    voltage_levels: Vec<String>,
    /// `<areaBoundary>` children, counted rather than read: gridoxide derives
    /// the boundary from membership instead, so a stated one is redundant here.
    boundaries: usize,
}

#[derive(Default)]
struct Collected {
    version: Option<String>,
    levels: Vec<RawVoltageLevel>,
    areas: Vec<RawArea>,
    /// The `<area>` currently open, so its nested refs attach. Kept here rather
    /// than threaded through `open`/`close`, which already carry five pieces of
    /// parser state.
    open_area: Option<RawArea>,
    /// Declared nodes, in declaration order, so bus labels are stable.
    nodes: Vec<NodeKey>,
    switches: Vec<RawSwitch>,
    branches: Vec<RawBranch>,
    injections: Vec<RawInjection>,
    shunts: Vec<RawShunt>,
    unknown: HashMap<String, usize>,
}

impl Collected {
    fn declare(&mut self, key: &NodeKey) {
        if !self.nodes.contains(key) {
            self.nodes.push(key.clone());
        }
    }
}

pub fn parse_with(text: &str, options: &IidmOptions) -> Result<IidmImport, IidmError> {
    let mut collected = collect(text)?;
    if collected.levels.is_empty() {
        return Err(IidmError::NoVoltageLevels);
    }
    convert(&mut collected, options)
}

fn collect(text: &str) -> Result<Collected, IidmError> {
    let mut reader = Reader::from_str(text);
    reader.config_mut().trim_text(true);

    let mut c = Collected::default();
    let mut buf = Vec::new();
    // The voltage level currently open, so equipment nested inside it can omit
    // `voltageLevelId`.
    let mut level: Option<String> = None;
    // The branch currently open, so nested limits and tap changers attach.
    let mut branch: Option<RawBranch> = None;
    // Which side a nested `operationalLimitsGroupN` / `currentLimitsN` applies
    // to, and which tap changer is being filled.
    let mut limit_side: Option<usize> = None;
    let mut tap_target: Option<bool> = None; // true = phase, false = ratio

    loop {
        let event = reader.read_event_into(&mut buf);
        let (start, empty) = match &event {
            Ok(Event::Start(e)) => (Some(e.clone()), false),
            Ok(Event::Empty(e)) => (Some(e.clone()), true),
            Ok(Event::End(e)) => {
                let name = String::from_utf8_lossy(e.local_name().as_ref()).into_owned();
                close(&name, &mut level, &mut branch, &mut limit_side, &mut tap_target, &mut c);
                buf.clear();
                continue;
            }
            Ok(Event::Eof) => break,
            Ok(_) => {
                buf.clear();
                continue;
            }
            Err(e) => return Err(IidmError::Xml(e.to_string())),
        };
        let e = start.expect("start or empty");
        let name = String::from_utf8_lossy(e.local_name().as_ref()).into_owned();
        let attrs = Attrs::of(&e)?;
        open(
            &name, &attrs, empty, &mut level, &mut branch, &mut limit_side, &mut tap_target, &mut c,
        );
        if empty {
            close(&name, &mut level, &mut branch, &mut limit_side, &mut tap_target, &mut c);
        }
        buf.clear();
    }
    Ok(c)
}

/// The side a suffixed element name refers to: `…1` → 0, `…2` → 1, bare → both.
fn suffix_side(name: &str) -> Option<usize> {
    match name.chars().last() {
        Some('1') => Some(0),
        Some('2') => Some(1),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn open(
    name: &str,
    attrs: &Attrs,
    _empty: bool,
    level: &mut Option<String>,
    branch: &mut Option<RawBranch>,
    limit_side: &mut Option<usize>,
    tap_target: &mut Option<bool>,
    c: &mut Collected,
) {
    match name {
        "network" => {
            // The namespace carries the version: ".../schema/iidm/1_14". A
            // document also declares one namespace per *extension*, several of
            // which are versioned the same way (".../ext/xnode/1_0"), so the
            // core one has to be selected rather than whichever the attribute
            // map happens to yield first.
            let mut versions: Vec<String> = attrs
                .0
                .values()
                .filter(|v| !v.contains("/ext/"))
                .filter_map(|v| v.rsplit_once("/schema/iidm/").map(|(_, x)| x.to_string()))
                .filter(|v| v.starts_with('1'))
                .collect();
            versions.sort();
            c.version = versions.into_iter().next();
        }
        "area" => {
            c.open_area = Some(RawArea {
                id: attrs.string("id").unwrap_or_default(),
                name: attrs.string("name").unwrap_or_default(),
                area_type: attrs.string("areaType").unwrap_or_default(),
                target_mw: attrs.number("interchangeTarget"),
                ..RawArea::default()
            });
        }
        "voltageLevelRef" => {
            if let (Some(a), Some(id)) = (c.open_area.as_mut(), attrs.string("id")) {
                a.voltage_levels.push(id);
            }
        }
        "areaBoundary" => {
            if let Some(a) = c.open_area.as_mut() {
                a.boundaries += 1;
            }
        }
        "voltageLevel" => {
            let id = attrs.string("id").unwrap_or_default();
            c.levels.push(RawVoltageLevel {
                id: id.clone(),
                nominal_v: attrs.number("nominalV").unwrap_or(1.0) * 1000.0,
            });
            *level = Some(id);
        }
        // A declared bus (bus-breaker) or busbar section (node-breaker). Both
        // simply introduce a node.
        "bus" | "busbarSection" => {
            let Some(vl) = level.clone() else { return };
            // Inside `nodeBreakerTopology`, `<bus nodes="0,1,2">` is powsybl's
            // *calculated* view rather than a declaration. Its nodes are
            // introduced by the switches and equipment that mention them, so
            // there is nothing to declare here — and treating it as a bus would
            // invent one that does not exist in the topology.
            if attrs.text("nodes").is_some() {
                return;
            }
            let key = match attrs.text("node") {
                Some(n) => format!("#{n}"),
                None => match attrs.string("id") {
                    Some(id) => id,
                    None => return,
                },
            };
            c.declare(&NodeKey { voltage_level: vl, key });
        }
        "switch" => {
            let Some(vl) = level.clone() else { return };
            let (Some(a), Some(b)) = (attrs.terminal("1", Some(&vl)), attrs.terminal("2", Some(&vl)))
            else {
                return;
            };
            c.declare(&a);
            c.declare(&b);
            c.switches.push(RawSwitch {
                id: attrs.string("id").unwrap_or_default(),
                a,
                b,
                kind: switch_kind(attrs.text("kind")),
                open: attrs.flag("open").unwrap_or(false),
            });
        }
        "line" | "twoWindingsTransformer" => {
            let end1 = attrs.terminal("1", None);
            let end2 = attrs.terminal("2", None);
            if let Some(k) = &end1 {
                c.declare(k);
            }
            if let Some(k) = &end2 {
                c.declare(k);
            }
            let transformer = (name == "twoWindingsTransformer").then(|| RawTransformerFields {
                rated_u1: attrs.or_zero("ratedU1") * 1000.0,
                rated_u2: attrs.or_zero("ratedU2") * 1000.0,
                ..Default::default()
            });
            // A transformer states one shunt (`g`, `b`) at side 2; a line states
            // one per side.
            let (g1, b1, g2, b2) = if transformer.is_some() {
                (0.0, 0.0, attrs.or_zero("g"), attrs.or_zero("b"))
            } else {
                (attrs.or_zero("g1"), attrs.or_zero("b1"), attrs.or_zero("g2"), attrs.or_zero("b2"))
            };
            *branch = Some(RawBranch {
                id: attrs.string("id").unwrap_or_default(),
                end1,
                end2,
                r: attrs.or_zero("r"),
                x: attrs.or_zero("x"),
                g1,
                b1,
                g2,
                b2,
                limits: [RawLimits::default(), RawLimits::default()],
                transformer,
            });
        }
        "generator" | "vscConverterStation" => {
            let Some(at) = attrs.terminal("", level.as_deref()) else { return };
            c.declare(&at);
            let p = attrs.number("targetP").unwrap_or(0.0);
            let regulating = attrs.flag("voltageRegulatorOn").unwrap_or(false);
            c.injections.push(RawInjection {
                at,
                id: attrs.string("id").unwrap_or_default(),
                generator: true,
                p,
                q: attrs.number("targetQ").or(attrs.number("reactivePowerSetpoint")).unwrap_or(0.0),
                regulates: regulating
                    .then(|| attrs.number("targetV").or(attrs.number("voltageSetpoint")))
                    .flatten(),
                generation: p,
                q_min: f64::NEG_INFINITY,
                q_max: f64::INFINITY,
            });
        }
        "load" => {
            let Some(at) = attrs.terminal("", level.as_deref()) else { return };
            c.declare(&at);
            c.injections.push(RawInjection {
                at,
                id: attrs.string("id").unwrap_or_default(),
                generator: false,
                p: -attrs.or_zero("p0"),
                q: -attrs.or_zero("q0"),
                regulates: None,
                generation: 0.0,
                q_min: f64::NEG_INFINITY,
                q_max: f64::INFINITY,
            });
        }
        // A dangling line is half of a tie line: it runs from a real bus to a
        // *boundary* node, which is exactly what UCTE calls an X-node. Keeping
        // the boundary as a real bus — rather than collapsing the half-line
        // into an injection — is what makes the two importers agree
        // structurally on the same network, and it is what lets the two halves
        // of a tie line find each other: they share a `pairingKey`, so they
        // land on the same boundary bus with no pairing logic at all.
        "danglingLine" | "boundaryLine" => {
            let Some(at) = attrs.terminal("", level.as_deref()) else { return };
            let id = attrs.string("id").unwrap_or_default();
            let boundary = boundary_key(attrs, &id);
            c.declare(&at);
            c.declare(&boundary);
            // Left open, like a line, so nested `currentLimits` attach; the
            // `close` handler commits it.
            *branch = Some(RawBranch {
                id: id.clone(),
                end1: Some(at),
                end2: Some(boundary.clone()),
                r: attrs.or_zero("r"),
                x: attrs.or_zero("x"),
                // A dangling line states one shunt for the whole half-line.
                g1: attrs.or_zero("g"),
                b1: attrs.or_zero("b"),
                g2: 0.0,
                b2: 0.0,
                limits: [RawLimits::default(), RawLimits::default()],
                transformer: None,
            });
            // The boundary carries whatever the file says is consumed or
            // generated there.
            c.injections.push(RawInjection {
                at: boundary,
                // A dangling line's boundary injection belongs to the line, not
                // to a machine, so it is not a generator for addressing
                // purposes however much it generates.
                id: id.clone(),
                generator: false,
                p: attrs.number("generationTargetP").unwrap_or(0.0) - attrs.or_zero("p0"),
                q: attrs.number("generationTargetQ").unwrap_or(0.0) - attrs.or_zero("q0"),
                regulates: None,
                generation: attrs.number("generationTargetP").unwrap_or(0.0),
                q_min: f64::NEG_INFINITY,
                q_max: f64::INFINITY,
            });
        }
        // The older, inline form: both halves as `_1`/`_2` attributes on the
        // tie line itself. The newer form pairs two `danglingLine` elements by
        // key and needs nothing here.
        "tieLine" if attrs.text("danglingLineId1").is_none() => {
            let id = attrs.string("id").unwrap_or_default();
            let boundary = boundary_key(attrs, &id);
            for half in ["1", "2"] {
                let Some(at) = attrs.terminal(half, None) else { continue };
                c.declare(&at);
                c.declare(&boundary);
                c.branches.push(RawBranch {
                    id: attrs.string(&format!("id_{half}")).unwrap_or_else(|| format!("{id}_{half}")),
                    end1: Some(at),
                    end2: Some(boundary.clone()),
                    r: attrs.or_zero(&format!("r_{half}")),
                    x: attrs.or_zero(&format!("x_{half}")),
                    g1: attrs.or_zero(&format!("g1_{half}")),
                    b1: attrs.or_zero(&format!("b1_{half}")),
                    g2: attrs.or_zero(&format!("g2_{half}")),
                    b2: attrs.or_zero(&format!("b2_{half}")),
                    limits: [RawLimits::default(), RawLimits::default()],
                    transformer: None,
                });
            }
        }
        "tieLine" => {}
        "shunt" => {
            let Some(at) = attrs.terminal("", level.as_deref()) else { return };
            c.declare(&at);
            c.shunts.push(RawShunt { at, g: 0.0, b: 0.0 });
        }
        "shuntLinearModel" => {
            // Applies to the shunt currently being read.
            if let Some(last) = c.shunts.last_mut() {
                last.g += attrs.or_zero("gPerSection");
                last.b += attrs.or_zero("bPerSection");
            }
        }
        "ratioTapChanger" | "phaseTapChanger" => {
            let phase = name == "phaseTapChanger";
            *tap_target = Some(phase);
            if let Some(b) = branch.as_mut() {
                if let Some(t) = b.transformer.as_mut() {
                    let changer = RawTapChanger {
                        low: attrs.integer("lowTapPosition").unwrap_or(0),
                        position: attrs.integer("tapPosition").unwrap_or(0),
                        steps: Vec::new(),
                        regulating: attrs.flag("regulating").unwrap_or(false),
                        mode: attrs.text("regulationMode").map(str::to_string),
                        // A ratio changer states `targetV`; a phase one states
                        // `regulationValue`. Neither file uses the other's
                        // name, so reading both here needs no branch.
                        target: attrs.number("targetV").or(attrs.number("regulationValue")),
                        deadband: attrs.number("targetDeadband"),
                    };
                    if phase {
                        t.phase = Some(changer);
                    } else {
                        t.ratio = Some(changer);
                    }
                }
            }
        }
        "step" => {
            let Some(phase) = *tap_target else { return };
            let Some(b) = branch.as_mut() else { return };
            let Some(t) = b.transformer.as_mut() else { return };
            let target = if phase { t.phase.as_mut() } else { t.ratio.as_mut() };
            if let Some(changer) = target {
                changer
                    .steps
                    .push((attrs.number("rho").unwrap_or(1.0), attrs.number("alpha").unwrap_or(0.0)));
            }
        }
        _ if name.starts_with("operationalLimitsGroup") => {
            // A bare group (no `1`/`2`) belongs to a one-terminal element — a
            // boundary line. Two-terminal elements always spell the side out,
            // so defaulting to side 0 cannot mis-assign one.
            *limit_side = suffix_side(name).or(Some(0));
        }
        _ if name.starts_with("currentLimits") => {
            // `currentLimits1`/`currentLimits2` name their own side; a bare
            // `currentLimits` inherits the enclosing group's.
            if let Some(side) = suffix_side(name) {
                *limit_side = Some(side);
            }
            let Some(side) = *limit_side else { return };
            if let Some(b) = branch.as_mut() {
                b.limits[side].permanent = attrs.number("permanentLimit");
            }
        }
        "temporaryLimit" => {
            let Some(side) = *limit_side else { return };
            let Some(value) = attrs.number("value") else { return };
            if let Some(b) = branch.as_mut() {
                b.limits[side].temporary.push(TemporaryLimit {
                    acceptable_duration_s: attrs.number("acceptableDuration"),
                    value_a: value,
                });
            }
        }
        // Structural or presentational elements with nothing to contribute.
        "substation" | "busBreakerTopology" | "nodeBreakerTopology"
        | "minMaxReactiveLimits" | "reactiveCapabilityCurve" | "point" | "property"
        | "extension" | "terminalRef" => {}
        other => {
            *c.unknown.entry(other.to_string()).or_insert(0) += 1;
        }
    }
}

fn close(
    name: &str,
    level: &mut Option<String>,
    branch: &mut Option<RawBranch>,
    limit_side: &mut Option<usize>,
    tap_target: &mut Option<bool>,
    c: &mut Collected,
) {
    match name {
        "area" => {
            if let Some(a) = c.open_area.take() {
                c.areas.push(a);
            }
        }
        "voltageLevel" => *level = None,
        "line" | "twoWindingsTransformer" | "danglingLine" | "boundaryLine" => {
            if let Some(b) = branch.take() {
                c.branches.push(b);
            }
            *limit_side = None;
            *tap_target = None;
        }
        "ratioTapChanger" | "phaseTapChanger" => *tap_target = None,
        _ if name.starts_with("operationalLimitsGroup") => *limit_side = None,
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

fn convert(c: &mut Collected, options: &IidmOptions) -> Result<IidmImport, IidmError> {
    let s_base_va = options.base_mva * 1e6;
    let nominal: HashMap<&str, f64> =
        c.levels.iter().map(|l| (l.id.as_str(), l.nominal_v)).collect();

    // Every declared node becomes a topology node; the bus view then decides
    // which of them are one bus.
    let node_of: HashMap<&NodeKey, usize> =
        c.nodes.iter().enumerate().map(|(i, k)| (k, i)).collect();
    let mut topology = NodeBreakerTopology::new(c.nodes.len());
    let mut switch_ids = Vec::new();
    for s in &c.switches {
        let (Some(&a), Some(&b)) = (node_of.get(&s.a), node_of.get(&s.b)) else { continue };
        topology.add_switch(Switch {
            kind: s.kind,
            nodes: [NodeIdx(a), NodeIdx(b)],
            open: s.open,
            in_service: true,
        });
        switch_ids.push(s.id.clone());
    }
    let view = bus_view(&topology, &options.retention);

    // A boundary node belongs to no declared voltage level, so it has no
    // `nominalV` of its own. It adopts the one at the other end of the
    // half-line that reaches it — which is what a boundary node *is*: the far
    // end of a line, at that line's voltage. Without this the per-unit base
    // would silently default to 1 V and every quantity through the boundary
    // would be nonsense.
    let mut boundary_nominal: HashMap<&str, f64> = HashMap::new();
    for raw in &c.branches {
        let (Some(k1), Some(k2)) = (&raw.end1, &raw.end2) else { continue };
        for (boundary, real) in [(k1, k2), (k2, k1)] {
            if boundary.voltage_level != BOUNDARY_LEVEL {
                continue;
            }
            if let Some(&v) = nominal.get(real.voltage_level.as_str()) {
                boundary_nominal.entry(boundary.key.as_str()).or_insert(v);
            }
        }
    }

    // Areas. IIDM lists the voltage levels inside each, so membership needs no
    // derivation — unlike CGMES, which states only the boundary. A bus can
    // merge nodes from more than one voltage level through a switch, so the
    // claim is per node and the first area to claim any of a bus's nodes wins.
    let mut areas = IidmAreas::default();
    let mut area_of_level: HashMap<&str, usize> = HashMap::new();
    let known_levels: std::collections::HashSet<&str> =
        c.levels.iter().map(|l| l.id.as_str()).collect();
    for raw in &c.areas {
        areas.report.boundaries_ignored += raw.boundaries;
        if raw.area_type != "ControlArea" {
            areas.report.other_types += 1;
            continue;
        }
        let a = areas.ids.len();
        areas.ids.push((raw.id.clone(), raw.name.clone()));
        match raw.target_mw {
            // Negated: IIDM states an import, `AreaDefinition` wants an export.
            Some(mw) => areas.targets.push(-mw * 1e6 / s_base_va),
            None => {
                areas.report.without_target.push(a);
                areas.targets.push(0.0);
            }
        }
        for vl in &raw.voltage_levels {
            if !known_levels.contains(vl.as_str()) {
                areas.report.unknown_voltage_levels += 1;
                continue;
            }
            area_of_level.entry(vl.as_str()).or_insert(a);
        }
    }
    areas.report.areas = areas.ids.len();
    // Filled per bus below, then counted once the loop has run.

    // One gridoxide bus per bus-view bus.
    let n_buses = view.n_buses();
    let mut buses: Vec<Bus> = Vec::with_capacity(n_buses);
    let mut bus_labels = Vec::with_capacity(n_buses);
    for b in 0..n_buses {
        let nodes = view.nodes_of(crate::topology::model::BusIdx(b));
        let first = nodes.first().map(|n| &c.nodes[n.0]);
        let u_rated = first
            .and_then(|k| {
                if k.voltage_level == BOUNDARY_LEVEL {
                    boundary_nominal.get(k.key.as_str()).copied()
                } else {
                    nominal.get(k.voltage_level.as_str()).copied()
                }
            })
            .unwrap_or(1.0);
        let label = match first {
            Some(k) if k.voltage_level == BOUNDARY_LEVEL => k.key.clone(),
            Some(k) if nodes.len() == 1 => k.key.clone(),
            Some(k) => format!("{}#{}", k.voltage_level, b),
            None => format!("bus{b}"),
        };
        // Which area, if any, claims this bus. A bus merging nodes from two
        // areas is a contradiction in the file's own membership; first wins,
        // and it is counted.
        let mut claim: Option<usize> = None;
        for n in nodes {
            let Some(&a) = area_of_level.get(c.nodes[n.0].voltage_level.as_str()) else { continue };
            match claim {
                None => claim = Some(a),
                Some(existing) if existing != a => areas.report.contested_buses += 1,
                Some(_) => {}
            }
        }
        areas.of_bus.push(claim);

        buses.push(Bus {
            idx: b,
            bus_type: BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            q_min: f64::NEG_INFINITY,
            q_max: f64::INFINITY,
            u_rated,
            zip_terms: Vec::new(),
        });
        bus_labels.push(label);
    }

    let bus_of = |key: &NodeKey| -> Option<usize> {
        node_of.get(key).map(|&n| view.bus_of(NodeIdx(n)).0)
    };

    // Injections.
    let mut generation = vec![0.0f64; n_buses];
    let mut regulating = vec![false; n_buses];
    let mut injections = Vec::with_capacity(c.injections.len());
    for inj in &c.injections {
        let Some(b) = bus_of(&inj.at) else { continue };
        let (p_pu, q_pu) = (inj.p * 1e6 / s_base_va, inj.q * 1e6 / s_base_va);
        injections.push(IidmInjection {
            id: inj.id.clone(),
            bus: b,
            generator: inj.generator,
            p: p_pu,
            q: q_pu,
        });
        buses[b].p_spec += p_pu;
        buses[b].q_spec += q_pu;
        generation[b] += inj.generation;
        if let Some(target_kv) = inj.regulates {
            if target_kv > 0.0 && buses[b].u_rated > 0.0 {
                buses[b].bus_type = BusType::PV;
                buses[b].voltage_mag = target_kv * 1000.0 / buses[b].u_rated;
                regulating[b] = true;
            }
        }
        if inj.q_min.is_finite() {
            buses[b].q_min = inj.q_min * 1e6 / s_base_va;
        }
        if inj.q_max.is_finite() {
            buses[b].q_max = inj.q_max * 1e6 / s_base_va;
        }
    }

    let mut shunts = Vec::new();
    for sh in &c.shunts {
        let Some(b) = bus_of(&sh.at) else { continue };
        let z_base = buses[b].u_rated * buses[b].u_rated / s_base_va;
        shunts.push(ShuntAdm { at: b, y: Complex::new(sh.g * z_base, sh.b * z_base) });
    }

    // Branches.
    let mut lines = Vec::new();
    let mut line_ids = Vec::new();
    let mut line_limits = Vec::new();
    let mut transformers = Vec::new();
    let mut transformer_ids = Vec::new();
    let mut transformer_limits = Vec::new();
    let mut tap_changers = Vec::new();
    let mut tap_regulation: Vec<crate::outerloop::TapRegulation> = Vec::new();
    let mut pending_power_regulation: Vec<(usize, usize, f64, f64, String)> = Vec::new();
    // Phase changers regulating a current rather than a power. Counted so the
    // absence is visible in `notes` rather than assumed.
    let mut unsupported_phase_mode = 0usize;
    let mut disconnected = Vec::new();
    let mut asymmetric = 0usize;

    for raw in &c.branches {
        let (Some(k1), Some(k2)) = (&raw.end1, &raw.end2) else {
            disconnected.push(raw.id.clone());
            continue;
        };
        let (Some(from), Some(to)) = (bus_of(k1), bus_of(k2)) else {
            disconnected.push(raw.id.clone());
            continue;
        };
        let limits = [to_limits(&raw.limits[0]), to_limits(&raw.limits[1])];

        match &raw.transformer {
            None => {
                // Series impedance is stated in ohms at the side-2 voltage; on
                // a line both sides share a base, so either serves.
                let z_base = buses[to].u_rated * buses[to].u_rated / s_base_va;
                let (r, x) = crate::topology::reduction::clamp_branch_impedance(
                    raw.r / z_base,
                    raw.x / z_base,
                );
                if (raw.g1 - raw.g2).abs() > 1e-12 || (raw.b1 - raw.b2).abs() > 1e-12 {
                    asymmetric += 1;
                }
                lines.push(Line {
                    from,
                    to,
                    r,
                    x,
                    // `Line` carries one total shunt that the pi-model splits
                    // equally, so an asymmetric pair is summed. Every line in
                    // every vendored fixture is symmetric, and `asymmetric`
                    // counts any that are not rather than hiding the loss.
                    b_shunt: (raw.b1 + raw.b2) * z_base,
                    g_shunt: (raw.g1 + raw.g2) * z_base,
                });
                line_ids.push(raw.id.clone());
                line_limits.push(limits);
            }
            Some(t) => {
                let z_base = buses[to].u_rated * buses[to].u_rated / s_base_va;
                let (r, x) = crate::topology::reduction::clamp_branch_impedance(
                    raw.r / z_base,
                    raw.x / z_base,
                );
                let y_series = Complex::new(1.0, 0.0) / Complex::new(r, x);
                let y_shunt = Complex::new(raw.g2 * z_base, raw.b2 * z_base);
                let (tap, changer) = build_tap(t, buses[from].u_rated, buses[to].u_rated);
                // A ratio changer holds the voltage at the side it regulates —
                // taken as the `to` bus, matching `build_tap`'s own convention
                // that `rho` acts on side 1. `regulationTerminal` can name a
                // different one, which this does not follow yet.
                if let Some(reg) = t.ratio.as_ref().filter(|r| r.regulating) {
                    if let Some(kv) = reg.target {
                        tap_regulation.push(crate::outerloop::TapRegulation {
                            transformer: transformers.len(),
                            controlled_bus: to,
                            mode: crate::outerloop::RegulationMode::Voltage,
                            target: kv * 1000.0 / buses[to].u_rated,
                            deadband: reg.deadband.unwrap_or(0.0) * 1000.0 / buses[to].u_rated,
                            enabled: true,
                            id: raw.id.clone(),
                        });
                    }
                }
                if let Some(reg) = t.phase.as_ref().filter(|r| r.regulating) {
                    let active = reg.mode.as_deref() == Some("ACTIVE_POWER_CONTROL");
                    if let (true, Some(mw)) = (active, reg.target) {
                        pending_power_regulation.push((
                            transformers.len(),
                            to,
                            mw,
                            reg.deadband.unwrap_or(0.0),
                            raw.id.clone(),
                        ));
                    } else if !active {
                        unsupported_phase_mode += 1;
                    }
                }
                transformers.push(Transformer {
                    from,
                    to,
                    from_status: 1,
                    to_status: 1,
                    y_series,
                    y_shunt,
                    tap,
                });
                transformer_ids.push(raw.id.clone());
                transformer_limits.push(limits);
                tap_changers.push(changer);
            }
        }
    }

    // `lines` is final now, so an angle regulation's flat branch index is.
    for (transformer, to, mw, deadband, id) in pending_power_regulation {
        tap_regulation.push(crate::outerloop::TapRegulation {
            transformer,
            controlled_bus: to,
            mode: crate::outerloop::RegulationMode::ActivePower {
                branch: lines.len() + transformer,
                terminal: crate::branch_flow::Terminal::To,
            },
            target: mw / options.base_mva,
            deadband: deadband / options.base_mva,
            enabled: true,
            id,
        });
    }

    areas.report.unassigned_buses = areas.of_bus.iter().filter(|a| a.is_none()).count();

    let mut branch_ids = line_ids;
    branch_ids.extend(transformer_ids);
    let mut limits = line_limits;
    limits.extend(transformer_limits);

    // Notes.
    let mut notes = Vec::new();
    if !disconnected.is_empty() {
        notes.push(format!("{} branch(es) omitted as disconnected", disconnected.len()));
    }
    if unsupported_phase_mode > 0 {
        notes.push(format!(
            "{unsupported_phase_mode} regulating phase tap changer(s) hold a current rather than \
             an active power; no outer loop models that, so their controls are dropped"
        ));
    }
    if asymmetric > 0 {
        notes.push(format!(
            "{asymmetric} line(s) declare asymmetric shunts; summed into one pi-model term"
        ));
    }
    let mut unknown: Vec<(&String, &usize)> = c.unknown.iter().collect();
    unknown.sort();
    for (name, count) in unknown {
        notes.push(format!("skipped {count} `{name}` element(s)"));
    }

    let slack = choose_slack(&buses, &bus_labels, &generation, &regulating, &options.slack, &mut notes)?;
    buses[slack].bus_type = BusType::Slack;

    Ok(IidmImport {
        buses,
        lines,
        transformers,
        shunts,
        bus_labels,
        branch_ids,
        limits,
        tap_changers,
        areas,
        regulation: tap_regulation,
        topology,
        view,
        switch_ids,
        slack,
        base_mva: options.base_mva,
        version: c.version.clone(),
        notes,
        disconnected,
        injections,
    })
}

fn to_limits(raw: &RawLimits) -> BranchLimits {
    BranchLimits { patl_a: raw.permanent, tatl: raw.temporary.clone() }
}

/// The from-side complex tap, and the tap table when the file gave one.
///
/// IIDM states a transformer's ratio as `rho` and its shift as `alpha`, both
/// relative to a *rated* ratio carried separately in `ratedU1`/`ratedU2`. The
/// nominal part is the same expression UCTE needs — the rated ratio measured
/// against the two buses' own per-unit bases — and `rho` multiplies it.
///
/// `alpha` is stated in degrees and is negated here: IIDM defines it on side 1
/// with the opposite sign convention to the MATPOWER-style complex tap
/// `network::branch_calc_param` expects.
fn build_tap(
    t: &RawTransformerFields,
    u_from: f64,
    u_to: f64,
) -> (Complex<f64>, Option<TapChanger>) {
    let nominal = if t.rated_u1 > 0.0 && t.rated_u2 > 0.0 && u_from > 0.0 && u_to > 0.0 {
        (t.rated_u1 / u_from) / (t.rated_u2 / u_to)
    } else {
        1.0
    };
    let base = Complex::new(nominal, 0.0);

    let step_of = |changer: &Option<RawTapChanger>, at: Option<i32>| -> Complex<f64> {
        let Some(ch) = changer else { return Complex::new(1.0, 0.0) };
        let position = at.unwrap_or(ch.position);
        let index = position - ch.low;
        if index < 0 {
            return Complex::new(1.0, 0.0);
        }
        match ch.steps.get(index as usize) {
            Some(&(rho, alpha)) => Complex::from_polar(rho, -alpha.to_radians()),
            None => Complex::new(1.0, 0.0),
        }
    };

    let held_ratio = step_of(&t.ratio, None);
    let held_phase = step_of(&t.phase, None);

    // The phase changer owns the tap table when present, since that is what a
    // phase-shifter remedial action moves. With both present the other is held
    // at its current position — the same approximation the reference importer
    // makes, and for the same reason: the exact answer is a two-dimensional
    // table and nothing downstream models a transformer with two changers.
    let (source, phase) = match (&t.phase, &t.ratio) {
        (Some(p), _) => (p, true),
        (None, Some(r)) => (r, false),
        (None, None) => return (base * held_ratio * held_phase, None),
    };
    if source.steps.is_empty() {
        return (base * held_ratio * held_phase, None);
    }

    let steps: Vec<Complex<f64>> = (0..source.steps.len())
        .map(|i| {
            let at = Some(source.low + i as i32);
            if phase {
                base * held_ratio * step_of(&t.phase, at)
            } else {
                base * step_of(&t.ratio, at) * held_phase
            }
        })
        .collect();

    let changer = TapChanger {
        low: source.low,
        position: source.position.clamp(source.low, source.low + steps.len() as i32 - 1),
        neutral: source.low + steps.len() as i32 / 2,
        steps,
        // IIDM states per-step `rho`/`alpha` alongside `r`/`x`/`g`/`b` ratios;
        // only the first two are read today, so the reactance is treated as
        // constant across the range. Correct for every vendored fixture, which
        // leave the impedance ratios at their defaults.
        series: None,
    };
    let tap = changer.current().unwrap_or(base);
    (tap, Some(changer))
}

fn choose_slack(
    buses: &[Bus],
    labels: &[String],
    generation: &[f64],
    regulating: &[bool],
    policy: &SlackPolicy,
    notes: &mut Vec<String>,
) -> Result<usize, IidmError> {
    if let SlackPolicy::Bus(id) = policy {
        return labels
            .iter()
            .position(|l| l == id)
            .ok_or_else(|| IidmError::UnknownReference {
                element: "slack policy".to_string(),
                reference: id.clone(),
            });
    }
    let pick = |want_regulating: bool| -> Option<usize> {
        (0..buses.len())
            .filter(|&i| regulating[i] == want_regulating && generation[i] > 0.0)
            .max_by(|&a, &b| {
                generation[a].total_cmp(&generation[b]).then_with(|| labels[b].cmp(&labels[a]))
            })
    };
    let chosen = pick(true).or_else(|| pick(false)).unwrap_or(0);
    notes.push(format!("slack chosen as `{}` (largest generation)", labels[chosen]));
    Ok(chosen)
}
