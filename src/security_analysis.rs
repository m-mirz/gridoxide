//! A security analysis of a CGMES model: its base case and each contingency of a CO dataset
//! solved, each terminal's current held against its `CurrentLimit`s, and what is found
//! written as a `SecurityAnalysisResult` dataset. See `docs/src/cgmes/security_analysis.md`.
//!
//! Choices the SAR profile leaves open:
//!
//! - Results are the violations, a violating terminal's per limit exceeded, plus the base
//!   case of each terminal a contingency violates. A contingency's own equipment is not
//!   monitored: it carries nothing.
//! - A TATL allows more than the PATL for a while, so one that does not is not a limit, and
//!   [`current_limits`] leaves it out.
//! - A contingency that does not converge has no result, as the profile has no place for it.
//!   [`SecurityAnalysis::stalled`] lists them.

use std::collections::{HashMap, HashSet};
use std::error::Error;

use cimmodel::Element;
use cimmodel::convert::{dataset_to_json, dataset_to_xml_for_profile};

use crate::branch_flow::{Terminal, branch_params, bus_voltages, terminal_flow};
use crate::cgmes::{
    CgmesTopologyMode, CimDataset, cgmes_resolve_dc_converters_with, cgmes_to_network_with,
    out_of_service,
};
use crate::outerloop::SlackDistribution;
use crate::solver::{IslandReport, IslandStatus};
use crate::types::{Bus, Line, Transformer};
use crate::{PowerFlowOptions, TapData, run_power_flow};

const SAR_TEMPLATE: &str = include_str!("security_analysis_sar_template.xml");
const TEMPLATE_HEADER: &str = "urn:uuid:00000000-0000-0000-0000-000000000000";
const TEMPLATE_CONTINGENCY_RESULT: &str = "_00000000-0000-0000-0000-000000000002";
const TEMPLATE_BASE_RESULT: &str = "_00000000-0000-0000-0000-000000000003";

/// Who takes the difference between what a state schedules and what it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImbalanceSharing {
    /// The machines in service, by `GeneratingUnit.maxOperatingP`. Generator limits do not
    /// stop a share: [`SecurityAnalysis::past_limits`] lists where one goes past them.
    Capability,
    /// The angle reference the model declares.
    Reference,
    /// Every generator bus alike.
    Equal,
}

impl ImbalanceSharing {
    pub fn name(self) -> &'static str {
        match self {
            Self::Capability => "capability",
            Self::Reference => "reference",
            Self::Equal => "equal",
        }
    }
}

impl std::str::FromStr for ImbalanceSharing {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "capability" => Ok(Self::Capability),
            "reference" => Ok(Self::Reference),
            "equal" => Ok(Self::Equal),
            other => Err(format!(
                "unknown imbalance sharing '{other}', expected 'capability', 'reference' or 'equal'"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SecurityOptions {
    pub sharing: ImbalanceSharing,
    pub s_base_va: f64,
    pub tol: f64,
    pub max_iter: usize,
    /// Off by default: a model whose voltage targets its reactive limits cannot all hold
    /// stalls with them, in the base case and so everywhere.
    pub enforce_q_limits: bool,
    /// Where the buses come from: the TP's `TopologicalNode`s (the default), or the switches as
    /// the SSH sets them, so that a switch opened or closed in the data is a different network.
    pub topology: CgmesTopologyMode,
}

impl Default for SecurityOptions {
    fn default() -> Self {
        Self {
            sharing: ImbalanceSharing::Capability,
            s_base_va: 100e6,
            tol: 1e-6,
            max_iter: 20,
            enforce_q_limits: false,
            topology: CgmesTopologyMode::BusBranchFromTp,
        }
    }
}

/// One contingency of a CO dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Contingency {
    pub mrid: String,
    pub name: String,
    /// Its `ContingencyEquipment.Equipment`, each `outOfService`.
    pub equipment: Vec<String>,
}

/// Every contingency in `ds`, sorted by mRID. An element with a `contingentStatus` other than
/// `outOfService` is an error: it would need equipment put back, which this does not do.
pub fn read_contingencies(ds: &CimDataset) -> Result<Vec<Contingency>, String> {
    let mut equipment: HashMap<&str, Vec<String>> = HashMap::new();
    for mrid in ds
        .by_type
        .get("nc:ContingencyEquipment")
        .into_iter()
        .flatten()
    {
        let element = &ds.entries[mrid];
        let status = element
            .get_ref("ContingencyEquipment.contingentStatus")
            .or_else(|| element.get_str("ContingencyEquipment.contingentStatus"))
            .unwrap_or("");
        if !status.ends_with("outOfService") {
            return Err(format!(
                "{mrid}: contingentStatus '{status}' is not outOfService"
            ));
        }
        if let (Some(contingency), Some(target)) = (
            element.get_ref("ContingencyElement.Contingency"),
            element.get_ref("ContingencyEquipment.Equipment"),
        ) {
            equipment
                .entry(contingency)
                .or_default()
                .push(target.to_string());
        }
    }
    let mut found = Vec::new();
    for class in [
        "nc:OrdinaryContingency",
        "nc:ExceptionalContingency",
        "nc:OutOfRangeContingency",
    ] {
        for mrid in ds.by_type.get(class).into_iter().flatten() {
            let mut lost = equipment.remove(mrid.as_str()).unwrap_or_default();
            lost.sort();
            found.push(Contingency {
                mrid: mrid.clone(),
                name: ds.entries[mrid]
                    .get_str("IdentifiedObject.name")
                    .unwrap_or_default()
                    .to_string(),
                equipment: lost,
            });
        }
    }
    found.sort_by(|a, b| a.mrid.cmp(&b.mrid));
    Ok(found)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitKind {
    Patl,
    Tatl,
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurrentLimit {
    pub mrid: String,
    pub amps: f64,
    pub kind: LimitKind,
}

/// Each terminal's `CurrentLimit`s, by terminal mRID: the SSH's `value` where it gives one,
/// else the EQ's `normalValue`. A limit of zero, which any current would violate, is no limit;
/// nor is a TATL at or below the terminal's PATL, which would forbid for minutes what is
/// allowed for good.
pub fn current_limits(ds: &CimDataset) -> HashMap<String, Vec<CurrentLimit>> {
    let mut limits: HashMap<String, Vec<CurrentLimit>> = HashMap::new();
    for mrid in ds.by_type.get("CurrentLimit").into_iter().flatten() {
        let element = &ds.entries[mrid];
        let amps = element
            .get_f64("CurrentLimit.value")
            .or_else(|| element.get_f64("CurrentLimit.normalValue"))
            .unwrap_or(0.0);
        let terminal = element
            .get_ref("OperationalLimit.OperationalLimitSet")
            .and_then(|set| ds.entries.get(set))
            .and_then(|set| set.get_ref("OperationalLimitSet.Terminal"));
        let kind = element
            .get_ref("OperationalLimit.OperationalLimitType")
            .and_then(|kind| ds.entries.get(kind))
            .and_then(|kind| {
                kind.get_ref("OperationalLimitType.kind")
                    .or_else(|| kind.get_str("OperationalLimitType.kind"))
            })
            .map_or(LimitKind::Other, |kind| {
                if kind.ends_with("patl") {
                    LimitKind::Patl
                } else if kind.ends_with("tatl") {
                    LimitKind::Tatl
                } else {
                    LimitKind::Other
                }
            });
        if let Some(terminal) = terminal.filter(|_| amps > 0.0) {
            limits
                .entry(terminal.to_string())
                .or_default()
                .push(CurrentLimit {
                    mrid: mrid.clone(),
                    amps,
                    kind,
                });
        }
    }
    limits.retain(|_, found| {
        let patl = found
            .iter()
            .filter(|l| l.kind == LimitKind::Patl)
            .map(|l| l.amps)
            .fold(0.0, f64::max);
        found.retain(|l| l.kind != LimitKind::Tatl || l.amps > patl);
        !found.is_empty()
    });
    limits
}

/// The flow at a branch terminal: into the branch, CIM's load sign.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalFlow {
    pub p_mw: f64,
    pub q_mvar: f64,
    pub amps: f64,
}

/// The flow at each terminal of `terminal_branch` (`CgmesNetwork::terminal_branch`), by its
/// mRID. A dead bus's terminal has zero.
pub fn terminal_flows(
    terminal_branch: &HashMap<String, (usize, Terminal)>,
    buses: &[Bus],
    lines: &[Line],
    transformers: &[Transformer],
    s_base_va: f64,
) -> HashMap<String, TerminalFlow> {
    let voltages = bus_voltages(buses);
    let branches = branch_params(lines, transformers);
    terminal_branch
        .iter()
        .filter_map(|(mrid, &(index, side))| {
            let branch = branches.get(index)?;
            let (near, far) = match side {
                Terminal::From => (branch.from, branch.to),
                Terminal::To => (branch.to, branch.from),
            };
            let dead = buses[near].voltage_mag == 0.0 || buses[far].voltage_mag == 0.0;
            let (p, q) = if dead {
                (0.0, 0.0)
            } else {
                terminal_flow(branch, side, &voltages)
            };
            let volts = buses[near].voltage_mag * buses[near].u_rated;
            let amps = if volts > 0.0 {
                p.hypot(q) * s_base_va / (3f64.sqrt() * volts)
            } else {
                0.0
            };
            Some((
                mrid.clone(),
                TerminalFlow {
                    p_mw: p * s_base_va / 1e6,
                    q_mvar: q * s_base_va / 1e6,
                    amps,
                },
            ))
        })
        .collect()
}

/// A limit a terminal exceeds in a state.
#[derive(Clone, Debug, PartialEq)]
pub struct Violation {
    /// `None` in the base case.
    pub contingency: Option<String>,
    pub terminal: String,
    pub limit: CurrentLimit,
    pub flow: TerminalFlow,
}

impl Violation {
    /// As the profile defines `PowerFlowResult.value`: 1100 A on 1000 A is 110 %.
    pub fn percent(&self) -> f64 {
        100.0 * self.flow.amps / self.limit.amps
    }
}

/// The limits `flows` exceed.
pub fn monitor(
    flows: &HashMap<String, TerminalFlow>,
    limits: &HashMap<String, Vec<CurrentLimit>>,
    contingency: Option<&str>,
) -> Vec<Violation> {
    let mut found = Vec::new();
    for (terminal, flow) in flows {
        for limit in limits.get(terminal).into_iter().flatten() {
            if flow.amps > limit.amps {
                found.push(Violation {
                    contingency: contingency.map(str::to_string),
                    terminal: terminal.clone(),
                    limit: limit.clone(),
                    flow: *flow,
                });
            }
        }
    }
    found
}

/// A `SynchronousMachine` in service and connected, with its unit's limits.
#[derive(Clone, Debug)]
pub struct Machine {
    pub mrid: String,
    pub name: String,
    node: String,
    /// Its terminal: its bus, whichever the topology.
    terminal: String,
    /// MW generated, as the SSH schedules it.
    pub scheduled: f64,
    pub minimum: f64,
    pub maximum: f64,
}

/// The machines that can take a share of an imbalance.
pub fn machines(ds: &CimDataset) -> Vec<Machine> {
    let mut terminals: HashMap<&str, &Element> = HashMap::new();
    let mut terminal_of: HashMap<&str, &str> = HashMap::new();
    for mrid in ds.by_type.get("Terminal").into_iter().flatten() {
        let terminal = &ds.entries[mrid];
        if let Some(equipment) = terminal.get_ref("Terminal.ConductingEquipment") {
            terminals.insert(equipment, terminal);
            terminal_of.insert(equipment, mrid.as_str());
        }
    }
    let mut found = Vec::new();
    for mrid in ds.by_type.get("SynchronousMachine").into_iter().flatten() {
        let machine = &ds.entries[mrid];
        let Some(terminal) = terminals.get(mrid.as_str()) else {
            continue;
        };
        if machine.get_bool("Equipment.inService") == Some(false)
            || terminal.get_bool("ACDCTerminal.connected") == Some(false)
        {
            continue;
        }
        let node = terminal.get_ref("Terminal.TopologicalNode").or_else(|| {
            terminal
                .get_ref("Terminal.ConnectivityNode")
                .and_then(|node| ds.entries.get(node))
                .and_then(|node| node.get_ref("ConnectivityNode.TopologicalNode"))
        });
        let Some(node) = node else { continue };
        let terminal_mrid = terminal_of.get(mrid.as_str()).copied().unwrap_or_default();
        let unit = machine
            .get_ref("RotatingMachine.GeneratingUnit")
            .and_then(|unit| ds.entries.get(unit));
        let of_unit = |key| {
            unit.and_then(|unit: &Element| unit.get_f64(key))
                .unwrap_or(0.0)
        };
        found.push(Machine {
            mrid: mrid.clone(),
            name: machine
                .get_str("IdentifiedObject.name")
                .unwrap_or(mrid)
                .to_string(),
            node: node.to_string(),
            terminal: terminal_mrid.to_string(),
            scheduled: -machine.get_f64("RotatingMachine.p").unwrap_or(0.0),
            minimum: of_unit("GeneratingUnit.minOperatingP"),
            maximum: of_unit("GeneratingUnit.maxOperatingP"),
        });
    }
    found.sort_by(|a, b| a.mrid.cmp(&b.mrid));
    found
}

/// Per bus, the `maxOperatingP` of its machines not in `out`.
impl Machine {
    /// Its bus in `bus_index`, by its terminal (as [`CgmesNetwork::terminal_bus`] keys them),
    /// else by its `TopologicalNode`.
    ///
    /// [`CgmesNetwork::terminal_bus`]: crate::cgmes::CgmesNetwork::terminal_bus
    fn bus<'a>(&self, bus_index: &'a HashMap<String, usize>) -> Option<&'a usize> {
        bus_index
            .get(&self.terminal)
            .or_else(|| bus_index.get(&self.node))
    }
}

pub fn capability_weights(
    fleet: &[Machine],
    bus_index: &HashMap<String, usize>,
    n_buses: usize,
    out: &HashSet<&str>,
) -> Vec<f64> {
    let mut weights = vec![0.0; n_buses];
    for machine in fleet {
        if let Some(&bus) = machine.bus(bus_index)
            && !out.contains(machine.mrid.as_str())
        {
            weights[bus] += machine.maximum;
        }
    }
    weights
}

/// A machine a share moved past a limit.
#[derive(Clone, Debug, PartialEq)]
pub struct PastLimit {
    pub contingency: Option<String>,
    pub machine: String,
    pub mw: f64,
    pub limit: f64,
}

/// Where capability sharing moved a machine past a limit: each machine takes its bus's `shift`
/// (per unit) in proportion to its `maxOperatingP`.
pub fn outside_limits(
    fleet: &[Machine],
    bus_index: &HashMap<String, usize>,
    weights: &[f64],
    shift: &[f64],
    s_base_va: f64,
    out: &HashSet<&str>,
    contingency: Option<&str>,
) -> Vec<PastLimit> {
    let mut found = Vec::new();
    for machine in fleet {
        let Some(&bus) = machine.bus(bus_index) else {
            continue;
        };
        if out.contains(machine.mrid.as_str()) || weights[bus] <= 0.0 {
            continue;
        }
        let mw = machine.scheduled
            + shift.get(bus).copied().unwrap_or(0.0) * s_base_va / 1e6 * machine.maximum
                / weights[bus];
        let limit = if mw > machine.maximum {
            machine.maximum
        } else if mw < machine.minimum {
            machine.minimum
        } else {
            continue;
        };
        found.push(PastLimit {
            contingency: contingency.map(str::to_string),
            machine: machine.name.clone(),
            mw,
            limit,
        });
    }
    found
}

/// One state of a CGMES model solved, as [`run`] solves its base case and each contingency:
/// HVDC converters resolved, the imbalance shared as [`SecurityOptions::sharing`] says, and
/// reactive limits held if [`SecurityOptions::enforce_q_limits`].
#[derive(Clone, Debug)]
pub struct State {
    pub buses: Vec<Bus>,
    /// Each connected component's outcome. A component with no angle reference, or more than
    /// one, is reported here and not solved as the others are.
    pub islands: Vec<IslandReport>,
    /// Each `TopologicalNode`'s bus, by its mRID. From the switches, a node an open switch
    /// splits names one of its buses: [`State::bus_of_terminal`] tells them apart.
    pub bus_of: HashMap<String, usize>,
    /// Each terminal's bus, by its mRID.
    pub bus_of_terminal: HashMap<String, usize>,
    /// Each terminal's flow, by its mRID; empty unless every component converged.
    pub flows: HashMap<String, TerminalFlow>,
    /// The machines capability sharing takes past their limits.
    pub past_limits: Vec<PastLimit>,
    /// The buses reactive limits took off voltage control, in the order they were.
    pub q_limit_switches: Vec<usize>,
    /// Per bus, how far sharing the imbalance moved its active schedule, in MW; empty when
    /// the reference took it all.
    pub shift_mw: Vec<f64>,
}

impl State {
    /// Whether every component converged or was left as one that cannot be solved: none
    /// stopped short or met a singular Jacobian.
    pub fn converged(&self) -> bool {
        !self.islands.iter().any(|island| {
            matches!(
                island.status,
                IslandStatus::MaxIterationsReached | IslandStatus::Singular
            )
        })
    }
}

/// Solves `ds` as it stands: its base case, as [`run`] solves it. See [`State`].
pub fn solve(ds: &CimDataset, options: &SecurityOptions) -> Result<State, String> {
    solve_with(ds, &machines(ds), &HashSet::new(), None, options)
}

/// One state of `ds`, with `out` the equipment a contingency takes out.
fn solve_with(
    ds: &CimDataset,
    fleet: &[Machine],
    out: &HashSet<&str>,
    contingency: Option<&str>,
    options: &SecurityOptions,
) -> Result<State, String> {
    let net = cgmes_to_network_with(ds, options.s_base_va, options.topology)
        .map_err(|e| e.to_string())?;
    let mut buses = net.buses;
    cgmes_resolve_dc_converters_with(ds, &mut buses, options.s_base_va, options.topology)
        .map_err(|e| e.to_string())?;
    // Machines by their terminals, which a bus split by an open switch keeps apart.
    let bus_index = net.terminal_bus;
    let weights = capability_weights(fleet, &bus_index, buses.len(), out);
    let distribute_slack = match options.sharing {
        ImbalanceSharing::Capability => Some(SlackDistribution::from_weights(weights.clone())),
        ImbalanceSharing::Equal => Some(SlackDistribution::uniform(&buses)),
        ImbalanceSharing::Reference => None,
    };
    let report = run_power_flow(
        buses,
        &net.lines,
        &net.transformers,
        &net.shunts,
        TapData {
            changers: &net.tap_changers,
            regulation: &net.regulation,
        },
        PowerFlowOptions {
            enforce_q_limits: options.enforce_q_limits,
            distribute_slack,
            tol: options.tol,
            max_iter: options.max_iter,
            ..Default::default()
        },
    );
    let slack = report.outer.as_ref().and_then(|o| o.slack.as_ref());
    let mut state = State {
        flows: HashMap::new(),
        past_limits: match (options.sharing, slack) {
            (ImbalanceSharing::Capability, Some(slack)) => outside_limits(
                fleet,
                &bus_index,
                &weights,
                &slack.shift,
                options.s_base_va,
                out,
                contingency,
            ),
            _ => Vec::new(),
        },
        q_limit_switches: report
            .outer
            .as_ref()
            .map_or_else(Vec::new, |o| o.q_limit_switches.clone()),
        shift_mw: slack.map_or_else(Vec::new, |s| {
            s.shift
                .iter()
                .map(|p| p * options.s_base_va / 1e6)
                .collect()
        }),
        buses: report.buses,
        islands: report.islands,
        bus_of: net.node_bus,
        bus_of_terminal: bus_index,
    };
    if state.converged() {
        state.flows = terminal_flows(
            &net.terminal_branch,
            &state.buses,
            &net.lines,
            &net.transformers,
            options.s_base_va,
        );
    }
    Ok(state)
}

/// One state of `ds`, or `None` if a component did not converge.
fn solve_state(
    ds: &CimDataset,
    fleet: &[Machine],
    out: &HashSet<&str>,
    contingency: Option<&str>,
    options: &SecurityOptions,
) -> Result<Option<State>, String> {
    let state = solve_with(ds, fleet, out, contingency, options)?;
    Ok(state.converged().then_some(state))
}

/// The result of [`run`].
#[derive(Clone, Debug)]
pub struct SecurityAnalysis {
    pub sharing: ImbalanceSharing,
    /// `md:Model.scenarioTime` of the model's state.
    pub scenario_time: String,
    /// The base case's flow at each terminal.
    pub base: HashMap<String, TerminalFlow>,
    /// Sorted by contingency (the base case's first), terminal and limit.
    pub violations: Vec<Violation>,
    pub solved: usize,
    /// The contingencies that did not converge.
    pub stalled: Vec<String>,
    /// The contingencies naming equipment the model lacks: another IGM's, say.
    pub absent: Vec<String>,
    pub past_limits: Vec<PastLimit>,
    /// The contingency datasets' identifiers, which a SAR requires.
    pub requires: Vec<String>,
}

/// The model's scenario: its state's, as files of an IGM may carry their own.
fn scenario_time(ds: &CimDataset) -> Result<String, String> {
    let headers = ds.by_type.get("FullModel").into_iter().flatten();
    let mut times = headers.filter_map(|mrid| {
        let header = &ds.entries[mrid];
        let profiles = header.get("Model.profile")?;
        let state = match profiles {
            cimmodel::FieldValue::Text(p) | cimmodel::FieldValue::Resource(p) => {
                p.contains("StateVariables")
            }
            cimmodel::FieldValue::TextList(p) | cimmodel::FieldValue::ResourceList(p) => {
                p.iter().any(|p| p.contains("StateVariables"))
            }
        };
        state.then(|| header.get_str("Model.scenarioTime").map(str::to_string))?
    });
    let time = times.next().ok_or("the model has no state (SV) header")?;
    Ok(time)
}

/// Solves `grid`'s base case and each contingency of `contingency_data` (a decoded CO
/// dataset), and holds each terminal's current against its limits. `grid` is as it was
/// after: an outage is applied to the elements for its solve and put back.
pub fn run(
    grid: &mut CimDataset,
    contingency_data: &CimDataset,
    options: &SecurityOptions,
) -> Result<SecurityAnalysis, String> {
    let limits = current_limits(grid);
    let fleet = machines(grid);
    let scenario_time = scenario_time(grid)?;
    let none = HashSet::new();
    let base = solve_state(grid, &fleet, &none, None, options)?
        .ok_or("the base case does not converge")?;
    let mut analysis = SecurityAnalysis {
        sharing: options.sharing,
        scenario_time,
        violations: monitor(&base.flows, &limits, None),
        past_limits: base.past_limits,
        base: base.flows,
        solved: 0,
        stalled: Vec::new(),
        absent: Vec::new(),
        requires: {
            let mut datasets: Vec<String> = contingency_data
                .by_type
                .get("nc:Dataset")
                .into_iter()
                .flatten()
                .cloned()
                .collect();
            datasets.sort();
            datasets
        },
    };
    let mut terminals_of: HashMap<String, Vec<String>> = HashMap::new();
    for mrid in grid.by_type.get("Terminal").into_iter().flatten() {
        if let Some(equipment) = grid.entries[mrid].get_ref("Terminal.ConductingEquipment") {
            terminals_of
                .entry(equipment.to_string())
                .or_default()
                .push(mrid.clone());
        }
    }
    for contingency in read_contingencies(contingency_data)? {
        if contingency
            .equipment
            .iter()
            .any(|mrid| !grid.entries.contains_key(mrid))
        {
            analysis.absent.push(contingency.mrid);
            continue;
        }
        let mut originals = Vec::with_capacity(contingency.equipment.len());
        for mrid in &contingency.equipment {
            let element = grid.entries.get_mut(mrid).expect("checked above");
            let out = out_of_service(element);
            originals.push((mrid.clone(), std::mem::replace(element, out)));
        }
        let out: HashSet<&str> = contingency.equipment.iter().map(String::as_str).collect();
        let state = solve_state(grid, &fleet, &out, Some(&contingency.mrid), options);
        for (mrid, original) in originals.into_iter().rev() {
            grid.entries.insert(mrid, original);
        }
        let Some(mut state) = state? else {
            analysis.stalled.push(contingency.mrid);
            continue;
        };
        analysis.solved += 1;
        let tripped: HashSet<&String> = contingency
            .equipment
            .iter()
            .filter_map(|mrid| terminals_of.get(mrid))
            .flatten()
            .collect();
        state
            .flows
            .retain(|terminal, _| !tripped.contains(terminal));
        analysis
            .violations
            .extend(monitor(&state.flows, &limits, Some(&contingency.mrid)));
        analysis.past_limits.extend(state.past_limits);
    }
    analysis.violations.sort_by(|a, b| {
        (a.contingency.as_deref(), &a.terminal, &a.limit.mrid).cmp(&(
            b.contingency.as_deref(),
            &b.terminal,
            &b.limit.mrid,
        ))
    });
    analysis.stalled.sort();
    Ok(analysis)
}

/// What a SAR's `dcat:Dataset` header says that the analysis cannot.
#[derive(Clone, Debug)]
pub struct SarHeader {
    pub title: String,
    pub description: String,
    /// A party in the reference data.
    pub publisher: String,
    /// The model collection the dataset is a version of.
    pub is_version_of: String,
    pub rights_holder: String,
    /// `prov:generatedAtTime`.
    pub generated_at: String,
}

/// A name-based identifier, in the form of a UUID: the same name gives the same one, so the same
/// analysis gives the same dataset. FNV-1a over 128 bits; nothing relies on its strength.
fn identifier(name: &str) -> String {
    let mut hash: u128 = 0x6c62272e07bb014262b821756295c58d;
    for byte in name.bytes() {
        hash ^= u128::from(byte);
        hash = hash.wrapping_mul(0x0000000001000000000000000000013b);
    }
    let hash = (hash & !(0xf_u128 << 76)) | (0x8_u128 << 76);
    let hash = (hash & !(0x3_u128 << 62)) | (0x2_u128 << 62);
    let hex = format!("{hash:032x}");
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn number(value: f64) -> serde_json::Value {
    serde_json::Value::String(format!("{value:.4}"))
}

impl SecurityAnalysis {
    /// The analysis as a dataset of the `SecurityAnalysisResult` profile.
    pub fn to_sar(&self, header: &SarHeader) -> Result<String, Box<dyn Error>> {
        use serde_json::{Value, json};

        let mut dataset = CimDataset::decode_str(SAR_TEMPLATE)?;
        let template = dataset_to_json(&dataset);
        let header_class = dataset.entries[TEMPLATE_HEADER].class();
        let contingency_class = dataset.entries[TEMPLATE_CONTINGENCY_RESULT].class();
        let base_class = dataset.entries[TEMPLATE_BASE_RESULT].class();

        let id = identifier(&format!(
            "dataset/{}/{}/{}",
            header.title, self.scenario_time, header.generated_at
        ));
        let mut own = template[TEMPLATE_HEADER].clone();
        let map = own.as_object_mut().expect("an element's JSON is an object");
        for (key, value) in [
            ("id", json!(format!("urn:uuid:{id}"))),
            ("identifier", json!(id)),
            ("title", json!(header.title)),
            ("description", json!(header.description)),
            (
                "versionNotes",
                json!(format!(
                    "Violations of current limits, and the base case of the terminals they are \
                     on. Imbalance: {}.",
                    self.sharing.name()
                )),
            ),
            ("publisher", json!(header.publisher)),
            ("isVersionOf", json!(header.is_version_of)),
            ("rightsHolder", json!(header.rights_holder)),
            ("generatedAtTime", json!(header.generated_at)),
            ("startDate", json!(self.scenario_time)),
            ("endDate", json!(self.scenario_time)),
            ("requires", json!(self.requires)),
        ] {
            map.insert(key.to_string(), value);
        }
        dataset.set(
            format!("urn:uuid:{id}"),
            dataset.element_from_json(header_class, &own)?,
        );

        let mut add = |contingency: Option<&str>,
                       terminal: &str,
                       limit: Option<&str>,
                       mut fields: Value|
         -> Result<(), Box<dyn Error>> {
            let key = format!(
                "_{}",
                identifier(&format!(
                    "{}/{terminal}/{}",
                    contingency.unwrap_or(""),
                    limit.unwrap_or("")
                ))
            );
            let map = fields.as_object_mut().expect("built as an object");
            map.insert("id".to_string(), json!(key));
            map.insert("PowerFlowResult.ACDCTerminal".to_string(), json!(terminal));
            map.insert(
                "PowerFlowResult.atTime".to_string(),
                json!(self.scenario_time),
            );
            if let Some(contingency) = contingency {
                map.insert(
                    "ContingencyPowerFlowResult.Contingency".to_string(),
                    json!(contingency),
                );
            }
            if let Some(limit) = limit {
                map.insert("PowerFlowResult.OperationalLimit".to_string(), json!(limit));
            }
            let class = if contingency.is_some() {
                contingency_class
            } else {
                base_class
            };
            let element = dataset.element_from_json(class, &fields)?;
            dataset.set(key, element);
            Ok(())
        };

        let mut violated_in_base = HashSet::new();
        let mut on_terminals = std::collections::BTreeSet::new();
        for violation in &self.violations {
            on_terminals.insert(violation.terminal.as_str());
            if violation.contingency.is_none() {
                violated_in_base.insert(violation.terminal.as_str());
            }
            add(
                violation.contingency.as_deref(),
                &violation.terminal,
                Some(&violation.limit.mrid),
                json!({
                    "_type": if violation.contingency.is_some() {
                        "nc:ContingencyPowerFlowResult"
                    } else {
                        "nc:BaseCasePowerFlowResult"
                    },
                    "PowerFlowResult.isViolation": "true",
                    "PowerFlowResult.value": number(violation.percent()),
                    "PowerFlowResult.absoluteValue": number(violation.flow.amps),
                    "PowerFlowResult.valueA": number(violation.flow.amps),
                    "PowerFlowResult.valueW": number(violation.flow.p_mw),
                    "PowerFlowResult.valueVAR": number(violation.flow.q_mvar),
                }),
            )?;
        }
        for terminal in on_terminals {
            let Some(flow) = self.base.get(terminal) else {
                continue;
            };
            if violated_in_base.contains(terminal) {
                continue;
            }
            add(
                None,
                terminal,
                None,
                json!({
                    "_type": "nc:BaseCasePowerFlowResult",
                    "PowerFlowResult.isViolation": "false",
                    "PowerFlowResult.valueA": number(flow.amps),
                    "PowerFlowResult.valueW": number(flow.p_mw),
                    "PowerFlowResult.valueVAR": number(flow.q_mvar),
                }),
            )?;
        }

        for mrid in [
            TEMPLATE_HEADER,
            TEMPLATE_CONTINGENCY_RESULT,
            TEMPLATE_BASE_RESULT,
        ] {
            dataset.remove(mrid);
        }
        dataset_to_xml_for_profile(&dataset, "SAR")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flow(amps: f64) -> TerminalFlow {
        TerminalFlow {
            p_mw: 1.0,
            q_mvar: 2.0,
            amps,
        }
    }

    fn limits() -> HashMap<String, Vec<CurrentLimit>> {
        HashMap::from([(
            "t".to_string(),
            vec![
                CurrentLimit {
                    mrid: "p".to_string(),
                    amps: 1000.0,
                    kind: LimitKind::Patl,
                },
                CurrentLimit {
                    mrid: "h".to_string(),
                    amps: 1500.0,
                    kind: LimitKind::Tatl,
                },
            ],
        )])
    }

    #[test]
    fn a_limit_is_exceeded_only_above_it() {
        let at = HashMap::from([("t".to_string(), flow(1000.0))]);
        assert!(monitor(&at, &limits(), None).is_empty());
        let over = HashMap::from([("t".to_string(), flow(1100.0))]);
        let found = monitor(&over, &limits(), Some("c"));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].limit.mrid, "p");
        assert_eq!(found[0].contingency.as_deref(), Some("c"));
        assert!((found[0].percent() - 110.0).abs() < 1e-9);
        let both = HashMap::from([("t".to_string(), flow(1600.0))]);
        assert_eq!(monitor(&both, &limits(), None).len(), 2);
        let elsewhere = HashMap::from([("other".to_string(), flow(1e9))]);
        assert!(monitor(&elsewhere, &limits(), None).is_empty());
    }

    fn fleet() -> Vec<Machine> {
        vec![
            Machine {
                mrid: "s".to_string(),
                name: "small".to_string(),
                node: "n".to_string(),
                terminal: String::new(),
                scheduled: 50.0,
                minimum: 40.0,
                maximum: 100.0,
            },
            Machine {
                mrid: "b".to_string(),
                name: "big".to_string(),
                node: "n".to_string(),
                terminal: String::new(),
                scheduled: 850.0,
                minimum: 0.0,
                maximum: 900.0,
            },
        ]
    }

    #[test]
    fn a_tripped_machine_has_no_share() {
        let index = HashMap::from([("n".to_string(), 0)]);
        assert_eq!(
            capability_weights(&fleet(), &index, 1, &HashSet::new()),
            [1000.0]
        );
        assert_eq!(
            capability_weights(&fleet(), &index, 1, &HashSet::from(["b"])),
            [100.0]
        );
    }

    /// A bus of 100 MW and 900 MW machines, each taking the shift by its `maxOperatingP`: 1 p.u.
    /// is 10 MW and 90 MW, the big one's 940 past its 900; -2 p.u. leaves the small one 30 under
    /// its 40.
    #[test]
    fn a_share_past_a_limit_is_noticed() {
        let index = HashMap::from([("n".to_string(), 0)]);
        let weights = [1000.0];
        let past = |shift: f64| -> Vec<String> {
            outside_limits(
                &fleet(),
                &index,
                &weights,
                &[shift],
                100e6,
                &HashSet::new(),
                None,
            )
            .into_iter()
            .map(|p| p.machine)
            .collect()
        };
        assert!(past(0.0).is_empty());
        assert!(past(0.2).is_empty());
        assert_eq!(past(1.0), ["big"]);
        assert_eq!(past(-2.0), ["small"]);
    }

    #[test]
    fn an_identifier_is_of_its_name_and_in_a_uuids_form() {
        let id = identifier("a");
        assert_eq!(id, identifier("a"));
        assert_ne!(id, identifier("b"));
        assert_eq!(id.len(), 36);
        assert_eq!(id.as_bytes()[14], b'8');
    }

    #[test]
    fn the_template_decodes_to_what_to_sar_removes() {
        let dataset = CimDataset::decode_str(SAR_TEMPLATE).expect("the template decodes");
        for mrid in [
            TEMPLATE_HEADER,
            TEMPLATE_CONTINGENCY_RESULT,
            TEMPLATE_BASE_RESULT,
        ] {
            assert!(dataset.entries.contains_key(mrid), "{mrid}");
        }
    }

    #[test]
    fn an_unknown_sharing_is_refused() {
        assert_eq!("capability".parse(), Ok(ImbalanceSharing::Capability));
        assert!("nonsense".parse::<ImbalanceSharing>().is_err());
    }
}
