//! The CRAC domain model — Contingency List, Remedial Actions and Additional
//! Constraints.
//!
//! A CRAC is the answer to "what am I allowed to do?". It is not derived from
//! the network and it cannot be: it records what a TSO has *agreed* is
//! permissible, with owners, costs, and rules about when each action may be
//! used. That is the difference between an OPF, which optimizes over whatever
//! the physics admits, and a remedial action optimization, which optimizes over
//! a curated list.
//!
//! # Three things this model gets from the reference rather than inventing
//!
//! **[`Instant`] is data, not an enum.** The obvious Rust instinct — a
//! four-variant enum — is wrong, and expensively so. A CRAC may declare
//! *several* curative instants (`curative1`, `curative2`, …) optimized in
//! sequence, and the "latest state at or before this instant at which this
//! device is controllable" chain is load-bearing throughout the optimizer. The
//! reference implementation's own porting notes flag retrofitting this as
//! painful, so it is built in from the start. [`InstantKind`] is the enum; the
//! instant itself is a named, ordered record.
//!
//! **There is no `UsageMethod`.** Older descriptions of this model have an
//! available/forced/unavailable enum on each usage rule. The current reference
//! has removed it: availability is a predicate evaluated against the network
//! state at optimization time, and the available-versus-forced distinction is
//! implicit in the instant kind — everything at [`InstantKind::Auto`] is forced
//! and simulated, everything else is offered to the optimizer. Reading the
//! older spelling is a compatibility concern for the importer
//! ([`crate::rao::crac_json`]), not a modelling one.
//!
//! **Network elements are strings here.** A CRAC names branches, switches and
//! generators by the id its source network uses, and this model keeps them that
//! way. Resolving them to gridoxide's own indices needs a network, is fallible
//! in an interesting way, and therefore gets [its own step](Crac::resolve)
//! with its own report — a silently mis-resolved element produces a plausible
//! wrong answer, which is the worst kind.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The kind of moment an [`Instant`] represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum InstantKind {
    /// Before any contingency — the base case.
    Preventive,
    /// Immediately after a contingency, too soon for any action.
    Outage,
    /// After automatic devices have acted. Actions here are **forced and
    /// simulated**, never optimized: a protection scheme fires whether or not
    /// it helps.
    Auto,
    /// Late enough for a human to act. A CRAC may declare several, optimized in
    /// chronological order.
    Curative,
}

/// A named moment in the chronology following a contingency.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Instant {
    pub id: String,
    pub kind: InstantKind,
}

/// A contingency: the simultaneous loss of one or more network elements.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contingency {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// Ids of the elements that trip together. More than one is an N-k
    /// contingency, which the model allows because CRACs contain them.
    pub elements: Vec<String>,
}

/// A moment in a particular future: an instant, and the contingency (if any)
/// that led to it.
///
/// `contingency: None` at a preventive instant is the base case.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct State {
    /// Index into [`Crac::instants`].
    pub instant: usize,
    /// Index into [`Crac::contingencies`].
    pub contingency: Option<usize>,
}

impl State {
    pub fn preventive(instant: usize) -> Self {
        Self { instant, contingency: None }
    }

    pub fn is_preventive(&self) -> bool {
        self.contingency.is_none()
    }
}

/// Which end of a branch a threshold applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    One,
    Two,
    /// The source did not say. Taken to mean both ends.
    Both,
}

/// The unit a threshold is expressed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    Ampere,
    Megawatt,
    /// A fraction of the branch's own `iMax`. **Despite the name this is a
    /// fraction, not a percentage** — `1.0` means 100%, as the reference's own
    /// `ThresholdAdder` javadoc states. Reading it as a percentage makes every
    /// such threshold a hundred times too tight.
    PercentImax,
    Degree,
    Kilovolt,
}

/// One limit on a monitored quantity.
///
/// `min` and `max` are both optional and at least one is present: a CNEC may be
/// limited in one direction only, which is common for a flow whose reverse
/// direction is unconstrained.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Threshold {
    pub unit: Unit,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    pub side: Side,
}

/// A Critical Network Element and Contingency: a branch flow that matters, in a
/// particular state, against a particular limit.
///
/// The two flags are not the same question and both are load-bearing:
///
/// - `optimized` — its margin enters the objective, so the optimizer will spend
///   remedial actions to improve it.
/// - `monitored` — its margin must not *get worse*, enforced as a penalized
///   soft constraint rather than as something to maximize. An MNEC.
///
/// A CNEC can be both, either, or neither.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlowCnec {
    pub id: String,
    /// The branch, by the id its network uses.
    pub network_element: String,
    pub state: State,
    pub thresholds: Vec<Threshold>,
    /// Subtracted from every threshold before use — the operator's own safety
    /// margin against model error.
    #[serde(default)]
    pub reliability_margin: f64,
    #[serde(default = "yes")]
    pub optimized: bool,
    #[serde(default)]
    pub monitored: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    /// Rated current per side, where the source gave it. Needed to interpret a
    /// [`Unit::PercentImax`] threshold at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub i_max: Option<[Option<f64>; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nominal_v: Option<[Option<f64>; 2]>,
}

fn yes() -> bool {
    true
}

/// One indivisible change to the network.
///
/// The vocabulary is powsybl-core's `action-api`, reduced to what a CRAC
/// actually contains across the vendored corpus.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ElementaryAction {
    /// Open or close a switch.
    Switch { element: String, open: bool },
    /// Connect or disconnect a branch's terminals — the same idea for an
    /// element that has no switch of its own.
    TerminalsConnection { element: String, connected: bool },
    /// Move a phase shifter to a fixed tap.
    PstTapPosition { element: String, tap: i32 },
    /// Set a generator's active power target, in MW.
    GeneratorSetpoint { element: String, p: f64 },
    /// Set a load's active power target, in MW.
    LoadSetpoint { element: String, p: f64 },
    /// Move a shunt compensator to a fixed section count.
    ShuntSection { element: String, section: i32 },
    /// Open one switch and close another as a single act. Kept as one action
    /// rather than two because splitting it lets an optimizer choose the half
    /// that islands a substation.
    SwitchPair { open: String, close: String },
}

impl RangeActionKind {
    /// The angle a phase shifter reaches at `tap`, if the table has it.
    pub fn angle_at(&self, tap: i32) -> Option<f64> {
        match self {
            RangeActionKind::Pst { tap_to_angle, .. } => tap_to_angle
                .binary_search_by_key(&tap, |(t, _)| *t)
                .ok()
                .map(|i| tap_to_angle[i].1),
            _ => None,
        }
    }

    /// The tap whose angle is closest to `angle_deg` — the rounding step every
    /// continuous relaxation needs before anyone can act on it.
    pub fn nearest_tap(&self, angle_deg: f64) -> Option<i32> {
        match self {
            RangeActionKind::Pst { tap_to_angle, .. } => tap_to_angle
                .iter()
                .min_by(|a, b| (a.1 - angle_deg).abs().total_cmp(&(b.1 - angle_deg).abs()))
                .map(|(t, _)| *t),
            _ => None,
        }
    }
}

impl ElementaryAction {
    /// Every network element this action touches.
    pub fn elements(&self) -> Vec<&str> {
        match self {
            ElementaryAction::Switch { element, .. }
            | ElementaryAction::TerminalsConnection { element, .. }
            | ElementaryAction::PstTapPosition { element, .. }
            | ElementaryAction::GeneratorSetpoint { element, .. }
            | ElementaryAction::LoadSetpoint { element, .. }
            | ElementaryAction::ShuntSection { element, .. } => vec![element],
            ElementaryAction::SwitchPair { open, close } => vec![open, close],
        }
    }
}

/// When a remedial action may be used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UsageRule {
    /// Available in every state of this instant.
    OnInstant { instant: usize },
    /// Available only in this exact state.
    OnContingencyState { state: State },
    /// Available at this instant *if* the named CNEC is constrained — a
    /// condition on the flow result, not on the topology, so it can only be
    /// evaluated during optimization.
    OnConstraint { instant: usize, cnec: String },
    /// Available at this instant if any CNEC in the country is constrained.
    OnFlowConstraintInCountry {
        instant: usize,
        country: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        contingency: Option<usize>,
    },
}

impl UsageRule {
    pub fn instant(&self) -> usize {
        match self {
            UsageRule::OnInstant { instant }
            | UsageRule::OnConstraint { instant, .. }
            | UsageRule::OnFlowConstraintInCountry { instant, .. } => *instant,
            UsageRule::OnContingencyState { state } => state.instant,
        }
    }

    /// Whether this rule *could* make its action available in `state`.
    ///
    /// "Could", not "does": [`UsageRule::OnConstraint`] and
    /// [`UsageRule::OnFlowConstraintInCountry`] additionally require a CNEC to
    /// be constrained, which depends on a flow result this model does not have.
    /// So this is the topological half of the test, and the optimizer applies
    /// the rest.
    pub fn covers(&self, state: &State) -> bool {
        match self {
            UsageRule::OnInstant { instant } => state.instant == *instant,
            UsageRule::OnContingencyState { state: s } => s == state,
            UsageRule::OnConstraint { instant, .. } => state.instant == *instant,
            UsageRule::OnFlowConstraintInCountry { instant, contingency, .. } => {
                state.instant == *instant
                    && contingency.is_none_or(|c| state.contingency == Some(c))
            }
        }
    }
}

/// A remedial action that is either applied or not — no degree of freedom.
///
/// A network action is a *set* of elementary actions applied together. That is
/// what makes "split this busbar" one decision rather than six.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetworkAction {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    /// Seconds until this action takes effect. Only meaningful at an auto
    /// instant, where it orders the simulation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<i64>,
    /// Cost of using it at all, for a cost-minimizing objective.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_cost: Option<f64>,
    pub elementary: Vec<ElementaryAction>,
    pub usage_rules: Vec<UsageRule>,
}

/// How a range action's bounds are interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RangeKind {
    /// Bounds are absolute set-points.
    Absolute,
    /// Bounds are relative to the set-point in the network as imported.
    RelativeToInitialNetwork,
    /// Bounds are relative to wherever the previous instant left it — which is
    /// what chains a curative range action to the preventive one before it.
    RelativeToPreviousInstant,
    /// Multi-timestamp only, and not modelled by the optimizer.
    RelativeToPreviousTimeStep,
}

/// One bound pair on a range action. A range action carries several, which are
/// intersected.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Range {
    pub kind: RangeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
}

/// What a range action actually moves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RangeActionKind {
    /// A phase shifter. Its set-point is an **angle** and its decision variable
    /// is a **tap**; the map between them is nonlinear and is carried here
    /// because it cannot be reconstructed from a step size.
    Pst {
        element: String,
        initial_tap: i32,
        /// Tap position and its angle in degrees, ascending by tap.
        ///
        /// A sorted vector rather than a map for two reasons. JSON object keys
        /// are strings, so an `i32`-keyed map serializes but will not
        /// deserialize — the native companion document has to round-trip.
        /// And a map's iteration order is unspecified, which would make two
        /// runs over one CRAC disagree on tie-breaks.
        tap_to_angle: Vec<(i32, f64)>,
    },
    /// A redispatch: one scalar in MW, distributed over generators and loads by
    /// keys that sum to the shift.
    Injection { distribution: Vec<(String, f64)> },
    /// An HVDC set-point in MW.
    Hvdc { element: String },
    /// A cross-border exchange adjustment. Carried so a CRAC round-trips, but
    /// the optimizer does not model it — it has no network sensitivity, which
    /// is why the reference implementation leaves it out of the LP too.
    CounterTrade { exporting: String, importing: String },
}

/// A remedial action with a continuous degree of freedom.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RangeAction {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_cost: Option<f64>,
    /// Actions sharing a group id must move together — aligned phase shifters
    /// on parallel circuits, typically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    pub kind: RangeActionKind,
    pub ranges: Vec<Range>,
    pub usage_rules: Vec<UsageRule>,
}

/// Caps on how many remedial actions may be used in one state.
///
/// These are what stop an optimizer returning a mathematically optimal answer
/// no control room would carry out.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RaUsageLimits {
    pub instant: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_ra: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tso: Option<usize>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub max_topo_per_tso: HashMap<String, usize>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub max_pst_per_tso: HashMap<String, usize>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub max_ra_per_tso: HashMap<String, usize>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub max_elementary_actions_per_tso: HashMap<String, usize>,
}

/// A complete contingency and remedial-action definition.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Crac {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// In chronological order. An [`Instant`]'s index *is* its order.
    pub instants: Vec<Instant>,
    pub contingencies: Vec<Contingency>,
    pub flow_cnecs: Vec<FlowCnec>,
    pub network_actions: Vec<NetworkAction>,
    pub range_actions: Vec<RangeAction>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub usage_limits: Vec<RaUsageLimits>,
    /// Angle and voltage CNECs are **counted, not modelled**. The reference
    /// optimizer does not put them in its LP either — it checks them in a
    /// separate monitoring pass after the fact — so carrying them as a count
    /// keeps the boundary honest without pretending to optimize them.
    #[serde(default)]
    pub angle_cnecs: usize,
    #[serde(default)]
    pub voltage_cnecs: usize,
}

impl Crac {
    pub fn instant(&self, id: &str) -> Option<usize> {
        self.instants.iter().position(|i| i.id == id)
    }

    pub fn contingency(&self, id: &str) -> Option<usize> {
        self.contingencies.iter().position(|c| c.id == id)
    }

    /// The one preventive instant, if the CRAC declares one.
    pub fn preventive_instant(&self) -> Option<usize> {
        self.instants.iter().position(|i| i.kind == InstantKind::Preventive)
    }

    /// Curative instants in chronological order.
    pub fn curative_instants(&self) -> Vec<usize> {
        self.instants
            .iter()
            .enumerate()
            .filter(|(_, i)| i.kind == InstantKind::Curative)
            .map(|(i, _)| i)
            .collect()
    }

    /// Every state any CNEC is defined at, deduplicated and ordered by instant.
    ///
    /// This is the set of perimeters an optimization has to visit, and deriving
    /// it here rather than asking the caller to assemble it keeps the two from
    /// disagreeing.
    pub fn states(&self) -> Vec<State> {
        let mut states: Vec<State> = Vec::new();
        for cnec in &self.flow_cnecs {
            if !states.contains(&cnec.state) {
                states.push(cnec.state.clone());
            }
        }
        states.sort_by_key(|s| (s.instant, s.contingency));
        states
    }

    /// Every network element id the CRAC refers to, deduplicated.
    ///
    /// The input to resolution against a real network, and the thing to report
    /// on when resolution fails.
    pub fn network_elements(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = Vec::new();
        for c in &self.contingencies {
            ids.extend(c.elements.iter().map(String::as_str));
        }
        for c in &self.flow_cnecs {
            ids.push(&c.network_element);
        }
        for a in &self.network_actions {
            ids.extend(a.elementary.iter().flat_map(ElementaryAction::elements));
        }
        for a in &self.range_actions {
            match &a.kind {
                RangeActionKind::Pst { element, .. } | RangeActionKind::Hvdc { element } => {
                    ids.push(element)
                }
                RangeActionKind::Injection { distribution } => {
                    ids.extend(distribution.iter().map(|(e, _)| e.as_str()))
                }
                RangeActionKind::CounterTrade { .. } => {}
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Remedial actions available in `state`, by the topological half of their
    /// usage rules.
    pub fn network_actions_for(&self, state: &State) -> Vec<&NetworkAction> {
        self.network_actions
            .iter()
            .filter(|a| a.usage_rules.iter().any(|r| r.covers(state)))
            .collect()
    }

    pub fn range_actions_for(&self, state: &State) -> Vec<&RangeAction> {
        self.range_actions
            .iter()
            .filter(|a| a.usage_rules.iter().any(|r| r.covers(state)))
            .collect()
    }
}

/// gridoxide's own on-disk CRAC: a companion document beside the network file.
///
/// The pattern [`opf::model::OpfData`](crate::opf::model::OpfData) established,
/// for the reason its module docs give — inventing fields inside someone else's
/// format is how a converter becomes a liability. A `<network>.rao.json` sits
/// next to `<network>.uct` or `<network>.xiidm` and names elements by the ids
/// that network uses.
///
/// Unlike [`crate::rao::crac_json`], which reads 24 versions of somebody else's
/// format and is deliberately forgiving, this one is strict: it is gridoxide's
/// own format, so an unreadable field is a bug rather than a compatibility
/// question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RaoDocument {
    pub version: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub crac: Crac,
}

/// The `type` field a `<network>.rao.json` must carry, mirroring
/// `"opf_input"`.
pub const DOCUMENT_TYPE: &str = "rao_input";
pub const DOCUMENT_VERSION: &str = "1.0";

impl RaoDocument {
    pub fn new(crac: Crac) -> Self {
        Self {
            version: DOCUMENT_VERSION.to_string(),
            kind: DOCUMENT_TYPE.to_string(),
            crac,
        }
    }
}

impl Crac {
    /// Serialize as a companion document.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&RaoDocument::new(self.clone()))
    }

    /// Read a companion document.
    ///
    /// The `type` field is checked rather than assumed: pointing this at an
    /// `<network>.opf.json` by mistake would otherwise deserialize into an
    /// empty CRAC and report success, which is a security analysis that finds
    /// nothing wrong because it was asked nothing.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        let doc: RaoDocument = serde_json::from_str(text)?;
        if doc.kind != DOCUMENT_TYPE {
            return Err(serde::de::Error::custom(format!(
                "expected a `{DOCUMENT_TYPE}` document, found `{}`",
                doc.kind
            )));
        }
        Ok(doc.crac)
    }
}
