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
}

#[derive(Clone, Debug)]
struct RawInjection {
    at: NodeKey,
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
#[derive(Default)]
struct Collected {
    version: Option<String>,
    levels: Vec<RawVoltageLevel>,
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
        | "extension" | "voltageLevelRef" | "terminalRef" | "area" | "areaBoundary" => {}
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
    for inj in &c.injections {
        let Some(b) = bus_of(&inj.at) else { continue };
        buses[b].p_spec += inj.p * 1e6 / s_base_va;
        buses[b].q_spec += inj.q * 1e6 / s_base_va;
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

    let mut branch_ids = line_ids;
    branch_ids.extend(transformer_ids);
    let mut limits = line_limits;
    limits.extend(transformer_limits);

    // Notes.
    let mut notes = Vec::new();
    if !disconnected.is_empty() {
        notes.push(format!("{} branch(es) omitted as disconnected", disconnected.len()));
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
        topology,
        view,
        switch_ids,
        slack,
        base_mva: options.base_mva,
        version: c.version.clone(),
        notes,
        disconnected,
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
