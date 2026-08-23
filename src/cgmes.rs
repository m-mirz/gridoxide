//! CGMES (Common Grid Model Exchange Standard) dataset loading and
//! conversion, built on cimoxide's `cimdecoder`/`cimstructs` crates — see
//! `docs/src/reference/provenance.md` for why this is a pinned git
//! dependency rather than a crates.io one.
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
    ACDCConverterDCTerminal, ACLineSegment, BaseVoltage, CsConverter, CurrentLimit, DCBreaker,
    DCDisconnector,
    DCGround, DCLineSegment, DCSeriesDevice, DCShunt, DCSwitch, DCTerminal, EnergyConsumer,
    EquivalentInjection, LinearShuntCompensator, NonlinearShuntCompensator,
    NonlinearShuntCompensatorPoint, OperationalLimitSet, OperationalLimitType,
    PhaseTapChangerAsymmetrical, PhaseTapChangerNonLinear,
    PhaseTapChangerSymmetrical, PowerElectronicsConnection, PowerTransformerEnd, RatioTapChanger,
    RegulatingControl, StaticVarCompensator, SynchronousMachine, Terminal, TopologicalIsland,
    TopologicalNode, VsConverter,
};

use crate::dc::{injected_currents, solve_dc_network, DcBus, DcBusRole, DcLine, DcSolveStatus};
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
    /// A `VsConverter`/`CsConverter` whose `pPccControl` mode isn't one of
    /// the ones `cgmes_resolve_dc_converters` handles (`udc`/`dcVoltage`,
    /// `pPcc`/`activePower`, `dcCurrent`) — e.g. a droop or phase-control
    /// mode. An honest, explicit limitation rather than a silently wrong
    /// power flow, mirroring `UnsupportedTransformer` above.
    UnsupportedConverterControl { mrid: String, mode: String },
    /// `dc::solve_dc_network` didn't converge while resolving a converter's
    /// `pPcc`/`activePower` target through its loss curve.
    DcNetworkDidNotConverge,
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
            CgmesError::UnsupportedConverterControl { mrid, mode } => {
                write!(f, "ACDCConverter {mrid}: unsupported pPccControl mode {mode:?}")
            }
            CgmesError::DcNetworkDidNotConverge => {
                write!(f, "DC network solve did not converge")
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

/// Every CGMES class that puts power into or takes power out of a bus.
///
/// This is deliberately *wider* than the set [`convert_equipment`] actually
/// converts, and the asymmetry is the safe direction: the list is used to decide
/// which buses inject **nothing**, and a class named here that gridoxide ignores
/// merely leaves a bus unconstrained. A class *missing* here would do the
/// opposite — assert as exact fact that a real appliance's bus injects zero.
const INJECTING_CLASSES: &[&str] = &[
    "EnergyConsumer",
    "ConformLoad",
    "NonConformLoad",
    "StationSupply",
    "EnergySource",
    "SynchronousMachine",
    "AsynchronousMachine",
    "RotatingMachine",
    "EquivalentInjection",
    "ExternalNetworkInjection",
    "PowerElectronicsConnection",
    "StaticVarCompensator",
    "LinearShuntCompensator",
    "NonlinearShuntCompensator",
    "ShuntCompensator",
    "GroundingImpedance",
    "PetersenCoil",
    "EarthFaultCompensator",
    "Ground",
    "VsConverter",
    "CsConverter",
    "ACDCConverter",
];

/// Per bus, whether *no* injecting equipment terminates on it.
///
/// This is the network property state estimation turns into a hard equality
/// constraint (`P = Q = 0` exactly, no sensor and no noise — see
/// [`se::constraints`](crate::se::constraints)), so it has to be read from the
/// model's structure rather than from the snapshot's numbers. A load whose SSH
/// `p`/`q` happen to be zero this hour is not a bus that injects nothing, and
/// constraining it would bias every estimate around it.
///
/// It matters most in the node-breaker view, and that is the point of computing
/// it at all: the internal nodes of a bay carry switches and nothing else, so
/// almost every bus the finer topology adds is exactly this case. A bus-branch
/// model hides them by merging them away.
fn zero_injection_flags(ds: &CimDataset, terms: &TerminalIndex, n_buses: usize) -> Vec<bool> {
    let mut zero = vec![true; n_buses];
    for class in INJECTING_CLASSES {
        for mrid in by_type(ds, class) {
            let Some(terminals) = terms.by_equipment.get(mrid) else {
                continue;
            };
            for t in terminals {
                // A *disconnected* terminal injects nothing, and CGMES says so
                // in SSH — but this deliberately ignores that. `connected` is a
                // snapshot flag like `p` and `q`; the estimator's constraint
                // asserts a property of the network, and "there is a generator
                // here, currently open" is not the same claim as "nothing can
                // inject here".
                if let Some(&bus) = terms.bus_of.get(t) {
                    if bus < n_buses {
                        zero[bus] = false;
                    }
                }
            }
        }
    }
    zero
}

/// Merges buses tied together by a *closed*, in-service switch (`Breaker`,
/// `Switch`, `Disconnector`, `LoadBreakSwitch`, `DisconnectingCircuitBreaker`,
/// `GroundDisconnector`, `Jumper`, `Cut`, `Fuse`) into one bus each, before
/// any equipment loop below ever calls `terms.bus(...)` — approach #1 from
/// `docs/src/powerflow/zero_impedance_branches.md` ("topological
/// reduction... the most direct fix"), not approach #2 (an
/// extreme-admittance branch): this file's
/// own top doc comment already commits to `TopologicalNode` *being* the
/// fully-resolved bus everywhere downstream, so stamping switches as
/// near-zero-impedance `Line`s instead would fight that assumption — and,
/// confirmed empirically on FullGrid, is numerically unstable (the AC
/// Newton-Raphson solve diverged with 20+ such branches active at once,
/// exactly the conditioning cost `zero_impedance_branches.md` warns
/// large-admittance regularization carries). Nothing downstream needs the
/// two original terminals to stay numerically distinct (no per-side flow
/// reporting), so the merge has no real downside here.
///
/// This converter's original assumption — that CGMES's TP profile always
/// pre-merges a closed switch's two ends into one `TopologicalNode`, making
/// switches topologically invisible — held for MiniGrid/MicroGrid-BE/
/// RealGrid, but is FALSE for FullGrid specifically: its own plain `Switch`
/// instance is `open=false` in SSH yet resolves to two distinct
/// `TopologicalNode`s in TP. Real exporters don't universally do this
/// reduction, so gridoxide does it itself when needed.
/// Returns the merged buses plus the pre-merge -> post-merge index remap
/// (so callers can fix up any *other* pre-merge-indexed mapping they hold —
/// `cgmes_to_buses_and_branches`'s own `idx_of` in particular).
fn merge_closed_switches(ds: &CimDataset, buses: Vec<Bus>, terms: &mut TerminalIndex) -> Result<(Vec<Bus>, Vec<usize>), CgmesError> {
    let topo = switch_topology(ds, buses.len(), terms)?;
    let view = crate::topology::bus_view(&topo, &crate::topology::RetentionPolicy::MergeAll);

    let remap: Vec<usize> = view.bus_of_slice().iter().map(|b| b.0).collect();
    let mut merged: Vec<Bus> = Vec::with_capacity(view.n_buses());
    for bus in 0..view.n_buses() {
        // Clone the group's representative — the union-find root, not the
        // lowest member. The two coincide in most groups and the fields that
        // matter (`u_rated`, `bus_type`) agree across a group anyway, since a
        // closed switch ties nodes at one nominal voltage — but keeping the
        // original choice makes this extraction provably behaviour-preserving
        // rather than merely equivalent-looking. See
        // `topology::BusView::representative`.
        let mut b = buses[view.representative(crate::topology::model::BusIdx(bus)).0].clone();
        b.idx = bus;
        merged.push(b);
    }

    for v in terms.bus_of.values_mut() {
        *v = remap[*v];
    }
    Ok((merged, remap))
}

/// Reads every switching device into a [`NodeBreakerTopology`], with each
/// `TopologicalNode` as a node.
///
/// **The order of `topo.switches` is load-bearing**, not incidental. Union-find
/// picks a group's root by union order, `merge_closed_switches` clones that
/// root's bus record, and the CGMES fixtures compare exact values — so this
/// preserves the class-by-class order the function used before it was extracted
/// (`Switch`, `Breaker`, `LoadBreakSwitch`, `DisconnectingCircuitBreaker`,
/// `Disconnector`, `GroundDisconnector`, `Jumper`, `Cut`, `Fuse`, `Junction`),
/// and within each class whatever order `by_type` yields.
///
/// A switch whose two terminals do not both resolve to a bus is skipped
/// entirely rather than recorded with a placeholder, which is what the
/// pre-extraction code did by falling through its `if let`. Recording it would
/// mean inventing a node for it.
///
/// Nodes here are topological nodes, so this is not yet the *connectivity*-node
/// graph a full node-breaker view needs — `TopologicalNode` is already a
/// partial merge performed by the exporter. Reading `ConnectivityNode` from EQ
/// instead is a separate piece of work; this function is what lets the rest of
/// the pipeline stop caring which one it got.
fn switch_topology(
    ds: &CimDataset,
    n_nodes: usize,
    terms: &TerminalIndex,
) -> Result<crate::topology::NodeBreakerTopology, CgmesError> {
    use crate::topology::model::{NodeIdx, Switch as TopoSwitch, SwitchKind};

    let mut topo = crate::topology::NodeBreakerTopology::new(n_nodes);

    let add = |topo: &mut crate::topology::NodeBreakerTopology,
                   mrid: &str,
                   kind: SwitchKind,
                   in_service: bool,
                   open: bool| {
        if let (Some(a), Some(b)) = (terms.bus(mrid, 0), terms.bus(mrid, 1)) {
            topo.add_switch(TopoSwitch {
                kind,
                nodes: [NodeIdx(a), NodeIdx(b)],
                open,
                in_service,
            });
        }
    };

    for mrid in by_type(ds, "Switch") {
        let sw: &cimstructs::Switch = require(ds, mrid, "Switch", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::Generic, equipment_in_service(sw.base.base.in_service, sw.base.base.normally_in_service), sw.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Breaker") {
        let br: &cimstructs::Breaker = require(ds, mrid, "Breaker", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::Breaker, equipment_in_service(br.base.base.base.base.in_service, br.base.base.base.base.normally_in_service), br.base.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "LoadBreakSwitch") {
        let lbs: &cimstructs::LoadBreakSwitch = require(ds, mrid, "LoadBreakSwitch", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::LoadBreakSwitch, equipment_in_service(lbs.base.base.base.base.in_service, lbs.base.base.base.base.normally_in_service), lbs.base.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "DisconnectingCircuitBreaker") {
        let dcb: &cimstructs::DisconnectingCircuitBreaker = require(ds, mrid, "DisconnectingCircuitBreaker", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::DisconnectingCircuitBreaker, equipment_in_service(dcb.base.base.base.base.base.in_service, dcb.base.base.base.base.base.normally_in_service), dcb.base.base.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Disconnector") {
        let d: &cimstructs::Disconnector = require(ds, mrid, "Disconnector", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::Disconnector, equipment_in_service(d.base.base.base.in_service, d.base.base.base.normally_in_service), d.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "GroundDisconnector") {
        let g: &cimstructs::GroundDisconnector = require(ds, mrid, "GroundDisconnector", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::GroundDisconnector, equipment_in_service(g.base.base.base.in_service, g.base.base.base.normally_in_service), g.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Jumper") {
        let j: &cimstructs::Jumper = require(ds, mrid, "Jumper", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::Jumper, equipment_in_service(j.base.base.base.in_service, j.base.base.base.normally_in_service), j.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Cut") {
        let c: &cimstructs::Cut = require(ds, mrid, "Cut", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::Cut, equipment_in_service(c.base.base.base.in_service, c.base.base.base.normally_in_service), c.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Fuse") {
        let f: &cimstructs::Fuse = require(ds, mrid, "Fuse", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::Fuse, equipment_in_service(f.base.base.base.in_service, f.base.base.base.normally_in_service), f.base.open.unwrap_or(false));
    }
    // Junction: CIM's own doc text calls it "a point where one or more
    // conducting equipment are connected with zero impedance" — always a
    // permanent zero-impedance tie, no `open`/switchable state at all
    // (unlike every other class in this function), so it's merged
    // unconditionally whenever in-service. `SwitchKind::Junction` carries
    // that: `is_operable` is false for it, so no retention policy can keep it.
    for mrid in by_type(ds, "Junction") {
        let j: &cimstructs::Junction = require(ds, mrid, "Junction", mrid, "(self)")?;
        add(&mut topo, mrid, SwitchKind::Junction, equipment_in_service(j.base.base.base.in_service, j.base.base.base.normally_in_service), false);
    }

    Ok(topo)
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

/// Every position of one tap changer, before the owning transformer folds in
/// its structural ratio and its side convention.
///
/// Raw because a CGMES tap changer's own ratio is not what
/// [`crate::types::TapChanger::steps`] holds: the builder divides by the
/// nameplate-vs-bus structural ratio and, when the changer sits on the side
/// that ends up as `to`, inverts. Composing here would need the bus voltages,
/// which this index does not have.
struct RawTapTable {
    low: i32,
    position: i32,
    neutral: i32,
    /// One per position, from `low` upwards.
    effects: Vec<TapEffect>,
}

impl RawTapTable {
    /// Folds this table into the [`crate::types::TapChanger`] the owning
    /// transformer needs: `compose` applies the side convention and the
    /// structural ratio, `series` turns a per-step reactance override into the
    /// per-step series admittance.
    fn compose(
        &self,
        compose: impl Fn(Complex<f64>) -> Complex<f64>,
        series: impl Fn(Option<f64>) -> Complex<f64>,
    ) -> crate::types::TapChanger {
        let admittances: Vec<Complex<f64>> =
            self.effects.iter().map(|e| series(e.x_override)).collect();
        // Only carried when it actually varies: a constant vector would make
        // every `set_position` write a value the branch already has, and would
        // hide the distinction the field exists to record.
        let varies = admittances.windows(2).any(|w| w[0] != w[1]);
        crate::types::TapChanger {
            low: self.low,
            position: self.position,
            neutral: self.neutral,
            steps: self.effects.iter().map(|e| compose(e.tap)).collect(),
            series: varies.then_some(admittances),
        }
    }
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
            return Ok(Some(phase_tap_asymmetrical(&ptc.base, mrid, theta_deg, xtx, None)?));
        }
        if let Some(mrid) = self.phase_sym.get(end_mrid) {
            let ptc: &PhaseTapChangerSymmetrical = require(ds, mrid, "PhaseTapChangerSymmetrical", mrid, "(self)")?;
            return Ok(Some(phase_tap_symmetrical(&ptc.base, mrid, xtx, None)?));
        }
        if let Some(mrid) = self.phase_linear.get(end_mrid) {
            let ptc: &cimstructs::PhaseTapChangerLinear = require(ds, mrid, "PhaseTapChangerLinear", mrid, "(self)")?;
            return Ok(Some(phase_tap_linear(ptc, mrid, xtx, None)?));
        }
        if let Some((ptc_mrid, table_mrid)) = self.phase_tabular.get(end_mrid) {
            let ptc: &cimstructs::PhaseTapChangerTabular = require(ds, ptc_mrid, "PhaseTapChangerTabular", ptc_mrid, "(self)")?;
            let step = ptc.base.base.step.ok_or_else(|| missing("PhaseTapChangerTabular", ptc_mrid, "step"))?;
            return Ok(Some(phase_tap_tabular(ds, ptc_mrid, table_mrid, step.round() as i64, xtx)?));
        }
        Ok(None)
    }

    /// Whether any tap-changer subtype sits on this end.
    fn has_changer(&self, end_mrid: &str) -> bool {
        self.ratio.contains_key(end_mrid)
            || self.phase_asym.contains_key(end_mrid)
            || self.phase_sym.contains_key(end_mrid)
            || self.phase_linear.contains_key(end_mrid)
            || self.phase_tabular.contains_key(end_mrid)
    }

    /// The `TapChanger` base of whichever subtype sits on this end — where
    /// `TapChangerControl` and `controlEnabled` live, regardless of subtype.
    fn control_of(&self, ds: &CimDataset, end_mrid: &str) -> Option<(Option<String>, bool)> {
        let read = |tc: &cimstructs::TapChanger| {
            (tc.tap_changer_control.as_ref().map(|r| r.mrid.clone()), tc.control_enabled == Some(true))
        };
        if let Some(m) = self.ratio.get(end_mrid) {
            return get::<RatioTapChanger>(ds, m).map(|t| read(&t.base));
        }
        if let Some(m) = self.phase_asym.get(end_mrid) {
            return get::<PhaseTapChangerAsymmetrical>(ds, m).map(|t| read(&t.base.base.base));
        }
        if let Some(m) = self.phase_sym.get(end_mrid) {
            return get::<PhaseTapChangerSymmetrical>(ds, m).map(|t| read(&t.base.base.base));
        }
        if let Some(m) = self.phase_linear.get(end_mrid) {
            return get::<cimstructs::PhaseTapChangerLinear>(ds, m).map(|t| read(&t.base.base));
        }
        if let Some((m, _)) = self.phase_tabular.get(end_mrid) {
            return get::<cimstructs::PhaseTapChangerTabular>(ds, m).map(|t| read(&t.base.base));
        }
        None
    }

    /// Every position of the tap changer on `end_mrid`, not just the one the
    /// SSH profile currently names.
    ///
    /// [`effect_for_end`](Self::effect_for_end) evaluates one step and throws
    /// the rest away, which is all a fixed-tap power flow ever wanted. Anything
    /// that *moves* a tap needs the discarded half back: the map from position
    /// to ratio and angle is nonlinear, and for a table-driven changer it is
    /// not even regular, so it cannot be reconstructed from the current value
    /// plus a step size. The other two importers (`src/ucte.rs`, `src/iidm.rs`)
    /// already retain it; this closes the asymmetry.
    ///
    /// Shares every formula with `effect_for_end` rather than reimplementing
    /// them — the helpers take the step to evaluate at, and this loops
    /// `lowStep..=highStep`. The current step therefore reads back from the
    /// table bit-for-bit identical to what the single-step path returns, which
    /// is the property `tests/cgmes_tap_table_test.rs` asserts.
    ///
    /// A `PhaseTapChangerTabular` or table-driven `RatioTapChanger` whose table
    /// has no row for some position in range yields no changer at all rather
    /// than a table with holes: a position that cannot be evaluated is one a
    /// control must never select.
    fn steps_for_end(
        &self,
        ds: &CimDataset,
        end_mrid: &str,
        xtx: f64,
    ) -> Result<Option<RawTapTable>, CgmesError> {
        // `(low, high, neutral, current)` plus a per-step evaluator, chosen by
        // whichever subtype owns this end.
        let build = |tc: &cimstructs::TapChanger,
                     eval: &dyn Fn(f64) -> Option<TapEffect>|
         -> Option<RawTapTable> {
            let low = tc.low_step? as i32;
            let high = tc.high_step? as i32;
            let neutral = tc.neutral_step.unwrap_or(0) as i32;
            let position = tc.step.map(|s| s.round() as i32).unwrap_or(neutral);
            if high < low {
                return None;
            }
            let mut effects = Vec::with_capacity((high - low + 1) as usize);
            for s in low..=high {
                effects.push(eval(s as f64)?);
            }
            Some(RawTapTable { low, position, neutral, effects })
        };

        if let Some(mrid) = self.ratio.get(end_mrid) {
            let rtc: &RatioTapChanger = require(ds, mrid, "RatioTapChanger", mrid, "(self)")?;
            let tc = &rtc.base;
            let table = rtc.ratio_tap_changer_table.as_ref().map(|t| t.mrid.clone());
            let neutral = tc.neutral_step.unwrap_or(0) as f64;
            let inc = rtc.step_voltage_increment.unwrap_or(0.0);
            let eval = |s: f64| -> Option<TapEffect> {
                if let Some(t) = table.as_deref() {
                    if let Some(effect) = ratio_tap_table(ds, t, s.round() as i64, xtx) {
                        return Some(effect);
                    }
                }
                Some(TapEffect {
                    tap: Complex::new(1.0 + (s - neutral) * inc / 100.0, 0.0),
                    x_override: None,
                })
            };
            return Ok(build(tc, &eval));
        }
        if let Some(mrid) = self.phase_asym.get(end_mrid) {
            let ptc: &PhaseTapChangerAsymmetrical =
                require(ds, mrid, "PhaseTapChangerAsymmetrical", mrid, "(self)")?;
            let Some(theta_deg) = ptc.winding_connection_angle else { return Ok(None) };
            let eval =
                |s: f64| phase_tap_asymmetrical(&ptc.base, mrid, theta_deg, xtx, Some(s)).ok();
            return Ok(build(&ptc.base.base.base, &eval));
        }
        if let Some(mrid) = self.phase_sym.get(end_mrid) {
            let ptc: &PhaseTapChangerSymmetrical =
                require(ds, mrid, "PhaseTapChangerSymmetrical", mrid, "(self)")?;
            let eval = |s: f64| phase_tap_symmetrical(&ptc.base, mrid, xtx, Some(s)).ok();
            return Ok(build(&ptc.base.base.base, &eval));
        }
        if let Some(mrid) = self.phase_linear.get(end_mrid) {
            let ptc: &cimstructs::PhaseTapChangerLinear =
                require(ds, mrid, "PhaseTapChangerLinear", mrid, "(self)")?;
            let eval = |s: f64| phase_tap_linear(ptc, mrid, xtx, Some(s)).ok();
            return Ok(build(&ptc.base.base, &eval));
        }
        if let Some((ptc_mrid, table_mrid)) = self.phase_tabular.get(end_mrid) {
            let ptc: &cimstructs::PhaseTapChangerTabular =
                require(ds, ptc_mrid, "PhaseTapChangerTabular", ptc_mrid, "(self)")?;
            let eval =
                |s: f64| phase_tap_tabular(ds, ptc_mrid, table_mrid, s.round() as i64, xtx).ok();
            return Ok(build(&ptc.base.base, &eval));
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
    at: Option<f64>,
) -> Result<TapEffect, CgmesError> {
    let tc = &base.base.base;
    let step = match at {
        Some(s) => s,
        None => tc.step.ok_or_else(|| missing("PhaseTapChanger", mrid, "step"))?,
    };
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
fn phase_tap_symmetrical(
    base: &PhaseTapChangerNonLinear, mrid: &str, xtx: f64, at: Option<f64>,
) -> Result<TapEffect, CgmesError> {
    let tc = &base.base.base;
    let step = match at {
        Some(s) => s,
        None => tc.step.ok_or_else(|| missing("PhaseTapChanger", mrid, "step"))?,
    };
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
fn phase_tap_linear(
    ptc: &cimstructs::PhaseTapChangerLinear, mrid: &str, xtx: f64, at: Option<f64>,
) -> Result<TapEffect, CgmesError> {
    let tc = &ptc.base.base;
    let step = match at {
        Some(s) => s,
        None => tc.step.ok_or_else(|| missing("PhaseTapChangerLinear", mrid, "step"))?,
    };
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
/// Steps 1+2+2.5: bus skeleton from `TopologicalNode`, the shared
/// `Terminal`-based resolver, and the closed-switch topological merge —
/// shared by `cgmes_to_buses_and_branches` itself, `cgmes_resolve_dc_converters`
/// (to resolve a converter's AC `pcc_terminal`/own terminal to a bus index),
/// and `cgmes_topological_node_bus_index` (the public mrid -> bus-index
/// lookup other callers, including test code, need — bus indices are *not*
/// 1:1 with `by_type(ds, "TopologicalNode")`'s own order once Step 2.5
/// merges anything, confirmed a real, live bug on SmallGrid: naively
/// resolving `tn_mrids.iter().position(...)` against the *returned* `buses`
/// silently indexes the wrong bus, or panics outright once enough merging
/// shrinks `buses.len()` below the stale position).
///
/// Returns the post-merge `buses` plus `idx_of`, already remapped from
/// pre-merge to post-merge indices (unlike `merge_closed_switches`'s own
/// return value, which is a raw remap table, not a finished mrid map).
fn build_ac_bus_skeleton(ds: &CimDataset) -> Result<(Vec<Bus>, HashMap<String, usize>, TerminalIndex), CgmesError> {
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
    let mut terms = TerminalIndex::build(ds, &idx_of, &mut buses)?;

    // --- Step 2.5: merge buses tied together by a closed switch ---
    let (buses, switch_merge_remap) = merge_closed_switches(ds, buses, &mut terms)?;
    // `idx_of` is keyed by TopologicalNode mrid -> pre-merge index; every
    // later use of it needs the same post-merge index `terms`/`buses` now use.
    for v in idx_of.values_mut() {
        *v = switch_merge_remap[*v];
    }

    Ok((buses, idx_of, terms))
}

/// The final (post-Step-2.5-merge) bus index for every `TopologicalNode` in
/// `ds`, keyed by its own mrid. Needed by any caller that wants to look up a
/// specific bus by mrid against `cgmes_to_buses_and_branches`'s own returned
/// `buses` — see `build_ac_bus_skeleton`'s doc comment for why naively using
/// `by_type(ds, "TopologicalNode")`'s own list position is unsafe once any
/// closed-switch merging happens.
pub fn cgmes_topological_node_bus_index(ds: &CimDataset) -> Result<HashMap<String, usize>, CgmesError> {
    let (_, idx_of, _) = build_ac_bus_skeleton(ds)?;
    Ok(idx_of)
}

pub fn cgmes_to_buses_and_branches(
    ds: &CimDataset, s_base_va: f64,
) -> Result<(Vec<Bus>, Vec<Line>, Vec<Transformer>, Vec<ShuntAdm>), CgmesError> {
    let net = cgmes_to_network(ds, s_base_va)?;
    Ok((net.buses, net.lines, net.transformers, net.shunts))
}

/// A converted CGMES network, tap tables included.
///
/// [`cgmes_to_buses_and_branches`] is this minus
/// [`tap_changers`](Self::tap_changers) — the four things a fixed-tap power
/// flow needs, and the shape most of this crate's tests want. Anything that
/// *moves* a tap wants the fifth, and a five-element tuple is past the point
/// where positional returns help anyone.
#[derive(Clone, Debug)]
pub struct CgmesNetwork {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub transformers: Vec<Transformer>,
    pub shunts: Vec<ShuntAdm>,
    /// Parallel to [`transformers`](Self::transformers): every position of the
    /// tap changer on that transformer, or `None` where it has none.
    ///
    /// Composed exactly as the current position was, so reading
    /// [`TapChanger::position`](crate::types::TapChanger::position) back out
    /// reproduces [`Transformer::tap`](crate::types::Transformer::tap) bit for
    /// bit — the property `tests/cgmes_tap_table_test.rs` asserts, and the one
    /// that makes the table trustworthy as a *replacement* for the single-step
    /// path rather than a second opinion about it.
    pub tap_changers: Vec<Option<crate::types::TapChanger>>,
    /// The regulating controls acting on those changers, resolved onto
    /// gridoxide's own bus and branch indices. Only enabled, modelled controls
    /// appear; [`tap_report`](Self::tap_report) accounts for the rest.
    pub regulation: Vec<crate::outerloop::TapRegulation>,
    /// What the regulation import could not use.
    pub tap_report: TapRegulationReport,
    /// What generator-side voltage control did: how many buses are held, how
    /// many by more than one machine, and any target two controllers disagreed
    /// on.
    pub voltage_control: VoltageControlReport,
    /// `Terminal` mRID → the flat branch index it became and which side of it,
    /// for every two-terminal branch this conversion produced.
    ///
    /// Flat means lines first, then transformers — the crate-wide convention
    /// `branch_flow::branch_params` defines. Only the conversion knows which
    /// branches survived and in what order, which is why this is returned
    /// rather than reconstructed; [`cgmes_control_areas`] turns a `TieFlow`'s
    /// terminal into a boundary branch through it.
    pub terminal_branch: HashMap<String, (usize, crate::branch_flow::Terminal)>,
}

/// [`cgmes_to_buses_and_branches`], keeping the tap tables.
pub fn cgmes_to_network(ds: &CimDataset, s_base_va: f64) -> Result<CgmesNetwork, CgmesError> {
    let skeleton = build_ac_bus_skeleton(ds)?;
    convert_equipment(ds, s_base_va, skeleton)
}

/// Converts a CGMES dataset's equipment onto an already-built bus skeleton.
///
/// Split out from [`cgmes_to_buses_and_branches`] so the node-breaker path
/// ([`cgmes_node_breaker_to_buses_and_branches`]) can reuse every equipment
/// loop unchanged. Nothing below this point cares how a bus came to exist —
/// it all resolves through `terms.bus(...)` — which is exactly why the two
/// skeletons are interchangeable.
fn convert_equipment(
    ds: &CimDataset,
    s_base_va: f64,
    skeleton: (Vec<Bus>, HashMap<String, usize>, TerminalIndex),
) -> Result<CgmesNetwork, CgmesError> {
    let (mut buses, idx_of, terms) = skeleton;

    // Shared by all four regulating-machine loops below, which run in two
    // separate steps: `PowerElectronicsConnection` in step 3, then
    // `SynchronousMachine`, `StaticVarCompensator` and
    // `ExternalNetworkInjection` in step 8. A bus can be held by machines of
    // different kinds, so the accumulation has to span them.
    let mut voltage_control = VoltageControl::new(buses.len());

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
    // PowerElectronicsConnection: a renewable/inverter-based generation
    // source (wind/solar/battery via power electronics rather than a
    // rotating machine). Its `p`/`q` doc text is character-for-character
    // EquivalentInjection's ("Load sign convention... positive sign means
    // flow out from a node"), not SynchronousMachine's — so both get
    // negated, EquivalentInjection-style, not the machine's Q exception.
    for mrid in by_type(ds, "PowerElectronicsConnection") {
        let pec: &PowerElectronicsConnection = require(ds, mrid, "PowerElectronicsConnection", mrid, "(self)")?;
        let pec_connected = terms.connected(mrid, 0);
        if let Some(bus) = terms.bus(mrid, 0) {
            if pec_connected {
                buses[bus].p_spec += -pec.p.unwrap_or(0.0) * 1e6 / s_base_va;
                buses[bus].q_spec += -pec.q.unwrap_or(0.0) * 1e6 / s_base_va;
            }
        }

        let Some(rc_ref) = &pec.base.regulating_control else { continue };
        if !pec_connected || pec.base.control_enabled != Some(true) {
            continue;
        }
        let rc: &RegulatingControl = require(ds, &rc_ref.mrid, "PowerElectronicsConnection", mrid, "RegulatingControl")?;
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

        let q_min = pec.min_q.unwrap_or(-f64::INFINITY);
        let q_max = pec.max_q.unwrap_or(f64::INFINITY);
        let target_pu = target * mult / buses[controlled_bus].u_rated;
        voltage_control.regulate(
            &mut buses,
            controlled_bus,
            target_pu,
            if q_min.is_finite() { q_min * 1e6 / s_base_va } else { q_min },
            if q_max.is_finite() { q_max * 1e6 / s_base_va } else { q_max },
            mrid,
        );
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
    /// Returns the index of the pushed line, or `None` when both ends are
    /// disconnected and nothing was pushed. The index is what lets a caller
    /// map a `Terminal` onto a branch — see `terminal_branch` below.
    fn push_status_aware_line(lines: &mut Vec<Line>, from: usize, to: usize, from_conn: bool, to_conn: bool, r: f64, x: f64, b_shunt: f64, g_shunt: f64) -> Option<usize> {
        match (from_conn, to_conn) {
            (true, true) => {
                // A jumper exported as a very short `ACLineSegment` would put an
                // unbounded admittance into the Y-bus; see
                // `topology::ZERO_IMPEDANCE_THRESHOLD`. Measured across every committed
                // CGMES fixture the smallest branch is 2.92e-6 p.u., some 30x
                // above the threshold, so this changes nothing modelled today
                // and exists for exports that are less well behaved.
                let (r, x) = crate::topology::clamp_branch_impedance(r, x);
                lines.push(Line { from, to, r, x, b_shunt, g_shunt });
                Some(lines.len() - 1)
            }
            // A half-open line becomes a shunt-only self-loop at the connected
            // end. It carries no through flow, so it can never be an area
            // boundary and is deliberately not mapped.
            (true, false) => {
                lines.push(Line { from, to: from, r: 0.0, x: 0.0, b_shunt, g_shunt });
                None
            }
            (false, true) => {
                lines.push(Line { from: to, to, r: 0.0, x: 0.0, b_shunt, g_shunt });
                None
            }
            (false, false) => None,
        }
    }

    let mut lines: Vec<Line> = Vec::new();
    // Terminal mRID -> (branch index within `lines`, which side it became).
    // Built here because only the conversion knows which branches survived and
    // in what order; `cgmes_control_areas` needs it to turn a `TieFlow`'s
    // terminal into a boundary branch.
    let mut line_terminal: HashMap<String, (usize, crate::branch_flow::Terminal)> = HashMap::new();
    let record_line = |terms: &TerminalIndex,
                           line_terminal: &mut HashMap<String, (usize, crate::branch_flow::Terminal)>,
                           mrid: &str,
                           pushed: Option<usize>| {
        let Some(idx) = pushed else { return };
        let Some(ts) = terms.by_equipment.get(mrid) else { return };
        for (which, side) in
            [(0, crate::branch_flow::Terminal::From), (1, crate::branch_flow::Terminal::To)]
        {
            if let Some(t) = ts.get(which) {
                line_terminal.insert(t.clone(), (idx, side));
            }
        }
    };

    for mrid in by_type(ds, "ACLineSegment") {
        let ln: &ACLineSegment = require(ds, mrid, "ACLineSegment", mrid, "(self)")?;
        let (Some(from), Some(to)) = (terms.bus(mrid, 0), terms.bus(mrid, 1)) else { continue };
        let u_rated = buses[from].u_rated;
        let z_base = u_rated * u_rated / s_base_va;
        let y_base = 1.0 / z_base;
        let pushed = push_status_aware_line(
            &mut lines, from, to, terms.connected(mrid, 0), terms.connected(mrid, 1),
            ln.r.unwrap_or(0.0) / z_base, ln.x.unwrap_or(0.0) / z_base,
            ln.bch.unwrap_or(0.0) / y_base, ln.gch.unwrap_or(0.0) / y_base,
        );
        record_line(&terms, &mut line_terminal, mrid, pushed);
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
        let pushed = push_status_aware_line(
            &mut lines, from, to, terms.connected(mrid, 0), terms.connected(mrid, 1),
            sc.r.unwrap_or(0.0) / z_base, sc.x.unwrap_or(0.0) / z_base, 0.0, 0.0,
        );
        record_line(&terms, &mut line_terminal, mrid, pushed);
    }
    // EquivalentBranch: a simplified series-impedance stand-in for a
    // reduced/boundary part of the network (an `EquivalentNetwork`
    // container) — same shape as ACLineSegment, using the primary `r`/`x`
    // (not the `r21`/`x21`/`negative*`/`zero*` directional variants, which
    // FullGrid's own instance leaves equal to `r`/`x` anyway).
    for mrid in by_type(ds, "EquivalentBranch") {
        let eb: &cimstructs::EquivalentBranch = require(ds, mrid, "EquivalentBranch", mrid, "(self)")?;
        let (Some(from), Some(to)) = (terms.bus(mrid, 0), terms.bus(mrid, 1)) else { continue };
        let u_rated = buses[from].u_rated;
        let z_base = u_rated * u_rated / s_base_va;
        let pushed = push_status_aware_line(
            &mut lines, from, to, terms.connected(mrid, 0), terms.connected(mrid, 1),
            eb.r.unwrap_or(0.0) / z_base, eb.x.unwrap_or(0.0) / z_base, 0.0, 0.0,
        );
        record_line(&terms, &mut line_terminal, mrid, pushed);
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
    let mut tap_changers: Vec<Option<crate::types::TapChanger>> = Vec::new();
    // Parallel to `transformers`: the `PowerTransformerEnd` whose tap changer
    // produced the table, and each end's terminal with the side it became.
    // Both are needed to resolve a `TapChangerControl` — one to find the
    // transformer, the other to find the branch flow an `activePower` control
    // regulates — and both are known here rather than inside the builders.
    let mut changer_end: Vec<Option<String>> = Vec::new();
    let mut terminal_sides: Vec<Vec<(String, crate::branch_flow::Terminal)>> = Vec::new();
    // Sorted by mRID before conversion. `HashMap` iteration order is
    // randomized per process, so without this the transformer list — and every
    // flat branch index derived from it — comes out in a different order on
    // every run of the same program against the same file. Nothing asserted on
    // a transformer index, so it never surfaced as a failure; it would have
    // surfaced as an irreproducible `TapRegulation.branch`, and as a RAO
    // decision that names a different element each run.
    let mut ends_by_pt: Vec<(String, Vec<&PowerTransformerEnd>)> = ends_by_pt.into_iter().collect();
    ends_by_pt.sort_by(|a, b| a.0.cmp(&b.0));
    for (pt_mrid, mut ends) in ends_by_pt {
        ends.sort_by_key(|e| e.base.end_number.unwrap_or(0));
        match ends.len() {
            2 => {
                let (t, c) = build_two_winding(ds, &tap_index, &terms, &buses, &pt_mrid, ends[0], ends[1], s_base_va)?;
                transformers.push(t);
                tap_changers.push(c);
                changer_end.push(
                    [ends[0], ends[1]]
                        .iter()
                        .find(|e| tap_index.has_changer(e.mrid_str()))
                        .map(|e| e.mrid_str().to_string()),
                );
                // `build_two_winding` fixes `to` to end 1's bus and `from` to
                // end 2's, whichever end the changer is physically on.
                terminal_sides.push(
                    [(ends[0], crate::branch_flow::Terminal::To), (ends[1], crate::branch_flow::Terminal::From)]
                        .iter()
                        .filter_map(|(e, side)| {
                            e.base.terminal.as_ref().map(|t| (t.mrid.clone(), *side))
                        })
                        .collect(),
                );
            }
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
                    let (t, c) = build_star_leg(ds, &tap_index, &terms, &buses, end, star_idx, s_base_va)?;
                    transformers.push(t);
                    tap_changers.push(c);
                    changer_end.push(
                        tap_index.has_changer(end.mrid_str()).then(|| end.mrid_str().to_string()),
                    );
                    // A star leg's `to` is this end's own bus; `from` is the
                    // synthesized star point, which no terminal names.
                    terminal_sides.push(
                        end.base
                            .terminal
                            .as_ref()
                            .map(|t| vec![(t.mrid.clone(), crate::branch_flow::Terminal::To)])
                            .unwrap_or_default(),
                    );
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
    // Identified by NOT being any `idx_of` value (i.e. no real
    // `TopologicalNode` maps to that bus index) rather than by position
    // (`buses.len()..tn_mrids.len()` used to be exactly the synthesized
    // range, back when every real `TopologicalNode` bus kept its own
    // distinct index — no longer true once Step 2.5 merges some of them
    // together, which can leave a synthesized bus's index anywhere).
    let tn_backed_indices: std::collections::HashSet<usize> = idx_of.values().copied().collect();
    for (i, energized) in energized.iter_mut().enumerate() {
        if !tn_backed_indices.contains(&i) {
            *energized = true;
        }
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
        // `Equipment.inService` is independent of terminal connectivity, and a
        // machine can be out of service while its terminals stay connected and
        // its `RegulatingCondEq.controlEnabled` stays `true` — Svedala's
        // `_f4cde1f4` is exactly that (`inService=false`, `controlEnabled=true`,
        // `p=q=0`, a 21 kV target still recorded on its RegulatingControl).
        // A machine that isn't in service can't hold a voltage setpoint, so
        // honoring `controlEnabled` alone pinned that bus to 21.000 kV against
        // the fixture's own published 20.134 kV — Svedala's single worst bus.
        // The same applies to its P/Q injection, hence gating both.
        let machine_in_service = equipment_in_service(
            sm.base.base.base.base.base.in_service, sm.base.base.base.base.base.normally_in_service);
        let machine_connected = terms.connected(mrid, 0) && machine_in_service;
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

        let q_min = sm.min_q.unwrap_or(-f64::INFINITY);
        let q_max = sm.max_q.unwrap_or(f64::INFINITY);
        let target_pu = target * mult / buses[controlled_bus].u_rated;
        voltage_control.regulate(
            &mut buses,
            controlled_bus,
            target_pu,
            if q_min.is_finite() { q_min * 1e6 / s_base_va } else { q_min },
            if q_max.is_finite() { q_max * 1e6 / s_base_va } else { q_max },
            mrid,
        );
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
        let q_min = match (sc.inductive_rating, z_base) {
            (Some(x), Some(zb)) if x != 0.0 => zb / x,
            _ => -f64::INFINITY,
        };
        let q_max = match (sc.capacitive_rating, z_base) {
            (Some(x), Some(zb)) if x != 0.0 => zb / x,
            _ => f64::INFINITY,
        };
        let target_pu = target * mult / buses[controlled_bus].u_rated;
        voltage_control.regulate(
            &mut buses,
            controlled_bus,
            target_pu,
            q_min,
            q_max,
            mrid,
        );
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

        let q_min = eni.min_q.unwrap_or(-f64::INFINITY);
        let q_max = eni.max_q.unwrap_or(f64::INFINITY);
        let target_pu = target * mult / buses[controlled_bus].u_rated;
        voltage_control.regulate(
            &mut buses,
            controlled_bus,
            target_pu,
            if q_min.is_finite() { q_min * 1e6 / s_base_va } else { q_min },
            if q_max.is_finite() { q_max * 1e6 / s_base_va } else { q_max },
            mrid,
        );
    }

    // Slack: each TopologicalIsland's own angle reference, applied last so it
    // wins over any PV upgrade that happened to land on the same bus. CGMES
    // explicitly supports more than one TopologicalIsland in a single
    // submitted model (e.g. genuinely separate synchronous areas), each with
    // its own AngleRefTopologicalNode — so every one is marked here, not
    // just the first found. (Previously this `break`d after the first
    // resolvable reference, silently discarding any other island's own
    // reference bus — a real bug, though one that happened to not affect any
    // fixture validated so far, since none of them declare more than one
    // island. Any island that ends up with no Slack bus at all — malformed
    // data, not this fixture set — falls through to
    // `network::mark_unreferenced_islands`'s generic handling downstream.)
    let mut slack_indices: Vec<usize> = Vec::new();
    // Paired with each slack bus's own *original* AngleRefTopologicalNode
    // mrid (not re-derived from `tn_mrids[idx]` below — `idx` is a
    // post-Step-2.5 (post-switch-merge) index, and `tn_mrids` is still the
    // pre-merge list in its original order, so indexing it with a post-merge
    // idx picks out an unrelated TopologicalNode whenever any merging
    // happened at all. A real bug, caught on FullGrid: it fed some other
    // bus's `SvVoltage.v` into the slack's `voltage_mag`, producing a
    // nonsensical ~10-20x-scale starting voltage — and since which
    // TopologicalNode landed at that numeric index depended on de-
    // duplicated `HashMap`-iteration-order effects elsewhere, WHICH bus
    // ended up corrupted varied from run to run.)
    let mut slack_angle_ref_mrid: Vec<String> = Vec::new();
    for mrid in by_type(ds, "TopologicalIsland") {
        let ti: &TopologicalIsland = require(ds, mrid, "TopologicalIsland", mrid, "(self)")?;
        if let Some(tn_ref) = &ti.angle_ref_topological_node {
            if let Some(&idx) = idx_of.get(&tn_ref.mrid) {
                buses[idx].bus_type = BusType::Slack;
                slack_indices.push(idx);
                slack_angle_ref_mrid.push(tn_ref.mrid.clone());
            }
        }
    }
    if slack_indices.is_empty() {
        return Err(CgmesError::NoAngleReference);
    }

    // Each slack bus's angle is an arbitrary global rotational reference in
    // AC power flow — only relative angles between buses are physically
    // meaningful. A solved SV profile pins that choice to a specific value
    // (not necessarily 0°: this fixture's own reference bus is published at
    // 340.9585°, presumably to stay angle-consistent with the larger merged
    // model this area submission is part of), so matching it here for every
    // slack bus — rather than defaulting to 0° — is what actually reproduces
    // the same solution, not a fixture-specific hack.
    for (&slack_idx, slack_mrid) in slack_indices.iter().zip(slack_angle_ref_mrid.iter()) {
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
    }

    // Built before the struct takes ownership of `lines`.
    let terminal_branch = {
        // Transformer terminals are offset onto the flat index; line ones
        // already are, being first.
        let mut map = line_terminal;
        for (i, sides) in terminal_sides.iter().enumerate() {
            for (terminal, side) in sides {
                map.insert(terminal.clone(), (lines.len() + i, *side));
            }
        }
        map
    };

    let (regulation, tap_report) = read_tap_regulation(
        ds,
        &tap_index,
        &terms,
        &buses,
        &lines,
        &tap_changers,
        &changer_end,
        &terminal_sides,
        s_base_va,
    );

    Ok(CgmesNetwork {
        buses,
        lines,
        transformers,
        shunts,
        tap_changers,
        regulation,
        tap_report,
        voltage_control: voltage_control.finish(),
        terminal_branch,
    })
}

/// A CGMES `ControlArea` set, resolved onto gridoxide's own indices.
#[derive(Clone, Debug)]
pub struct ControlAreaImport {
    /// Area index per bus, indexed by [`Bus::idx`]; `None` for a bus no area
    /// claims. Feeds [`AreaDefinition::of_bus`](crate::outerloop::AreaDefinition::of_bus).
    pub of_bus: Vec<Option<usize>>,
    /// Scheduled net **export** per area, per-unit.
    ///
    /// CGMES states `ControlArea.netInterchange` as an *import* — "positive
    /// sign means flow in to the area" — and
    /// [`AreaDefinition::targets`](crate::outerloop::AreaDefinition::targets)
    /// is an export, so this is its negation. Getting that backwards would
    /// dispatch every area exactly the wrong way while converging perfectly
    /// happily, which is why it is stated here rather than left to a reader.
    pub targets: Vec<f64>,
    /// `pTolerance`, per-unit, per area. `None` where the file gave none.
    pub tolerances: Vec<Option<f64>>,
    /// `(mRID, name)` per area, in the order the indices above use.
    pub ids: Vec<(String, String)>,
    pub report: ControlAreaReport,
}

/// What a `ControlArea` import could not use, counted rather than dropped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ControlAreaReport {
    /// Areas converted.
    pub areas: usize,
    /// `TieFlow` objects read.
    pub tie_flows: usize,
    /// Tie flows whose terminal named no branch this conversion produced —
    /// an open line, or equipment the converter skipped.
    pub unresolved_tie_flows: usize,
    /// Areas with no usable tie flow, so no measurable boundary. Their
    /// position cannot be measured *or* controlled.
    pub without_boundary: Vec<usize>,
    /// Areas whose SSH stated no `netInterchange`; their target is left at
    /// zero, which is a guess rather than a schedule.
    pub without_target: Vec<usize>,
    /// Buses reached from more than one area's seeds once the tie branches are
    /// cut. A contradiction in the file's own boundary; first area wins.
    pub contested_buses: usize,
    /// Buses no area claims.
    pub unassigned_buses: usize,
}

/// Reads every `ControlArea` in the dataset onto gridoxide's own bus indices.
///
/// # How membership is derived, since CGMES does not state it
///
/// CGMES defines an area by its **boundary**, not its contents: a `TieFlow`
/// names one `Terminal` per boundary point, and `ControlArea` has no list of
/// the buses inside. [`AreaInterchange`](crate::outerloop::AreaInterchange)
/// needs both — the boundary to measure the position, and the membership to
/// know whose generators to dispatch.
///
/// So membership is derived, exactly rather than heuristically: take the
/// branches the tie flows name, **cut them**, and compute the connected
/// components of what remains. Each tie flow's own terminal sits on a bus
/// inside its area, so the component holding that bus *is* that area. A
/// component reached from two areas' seeds is a contradiction in the file's own
/// boundary and is reported through
/// [`contested_buses`](ControlAreaReport::contested_buses) rather than
/// arbitrated silently.
///
/// The alternative — flooding outward from the seeds and letting areas claim
/// buses by proximity — was rejected: it happens to work on a network whose
/// areas are far apart and fails quietly on one where they are not, which is
/// the worst combination.
pub fn cgmes_control_areas(
    ds: &CimDataset,
    net: &CgmesNetwork,
    s_base_va: f64,
) -> Result<ControlAreaImport, CgmesError> {
    let n = net.buses.len();
    let mut report = ControlAreaReport::default();

    // Areas, in mRID order so the indices are stable across runs — the same
    // reason the transformer list is sorted.
    let mut area_mrids: Vec<String> = by_type(ds, "ControlArea").to_vec();
    area_mrids.sort();
    let mut ids = Vec::new();
    let mut targets = Vec::new();
    let mut tolerances = Vec::new();
    for mrid in &area_mrids {
        let ca: &cimstructs::ControlArea = require(ds, mrid, "ControlArea", mrid, "(self)")?;
        ids.push((mrid.clone(), ca.base.base.name.clone()));
        match ca.net_interchange {
            // Negated: CGMES states an import, `AreaDefinition` wants an export.
            Some(mw) => targets.push(-mw * 1e6 / s_base_va),
            None => {
                report.without_target.push(targets.len());
                targets.push(0.0);
            }
        }
        tolerances.push(ca.p_tolerance.map(|mw| mw * 1e6 / s_base_va));
    }
    report.areas = ids.len();
    let index_of = |mrid: &str| area_mrids.iter().position(|m| m == mrid);

    // Tie flows: the boundary branches, and the seed buses inside each area.
    let params = crate::branch_flow::branch_params(&net.lines, &net.transformers);
    let mut cut = vec![false; params.len()];
    let mut seeds: Vec<Vec<usize>> = vec![Vec::new(); ids.len()];
    for mrid in by_type(ds, "TieFlow") {
        let tf: &cimstructs::TieFlow = require(ds, mrid, "TieFlow", mrid, "(self)")?;
        report.tie_flows += 1;
        let (Some(area_ref), Some(term_ref)) = (&tf.control_area, &tf.terminal) else {
            report.unresolved_tie_flows += 1;
            continue;
        };
        let Some(a) = index_of(&area_ref.mrid) else {
            report.unresolved_tie_flows += 1;
            continue;
        };
        let Some(&(branch, side)) = net.terminal_branch.get(&term_ref.mrid) else {
            report.unresolved_tie_flows += 1;
            continue;
        };
        cut[branch] = true;
        // **The far end, not this one.** A `TieFlow` names the terminal at the
        // *boundary*, which in a merged model is the X-node the two areas'
        // lines meet at — `TN_Border_AL11` carries both `NL-Line_1`'s and
        // `BE-Line_3`'s tie flow. What belongs to the area is the *equipment*;
        // the node is shared. Seeding from the named end put every area's seed
        // on the same border node, which showed up as five contested buses and
        // one area owning nothing at all.
        seeds[a].push(match side {
            crate::branch_flow::Terminal::From => params[branch].to,
            crate::branch_flow::Terminal::To => params[branch].from,
        });
    }

    // Components of the network with the tie branches removed.
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, bp) in params.iter().enumerate() {
        if cut[i] || bp.from == bp.to {
            continue;
        }
        adjacency[bp.from].push(bp.to);
        adjacency[bp.to].push(bp.from);
    }
    let mut component = vec![usize::MAX; n];
    let mut n_components = 0;
    for start in 0..n {
        if component[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        component[start] = n_components;
        while let Some(b) = stack.pop() {
            for &next in &adjacency[b] {
                if component[next] == usize::MAX {
                    component[next] = n_components;
                    stack.push(next);
                }
            }
        }
        n_components += 1;
    }

    // Each component takes the area of whichever seed it holds.
    let mut area_of_component: Vec<Option<usize>> = vec![None; n_components];
    for (a, buses) in seeds.iter().enumerate() {
        if buses.is_empty() {
            report.without_boundary.push(a);
            continue;
        }
        for &b in buses {
            let c = component[b];
            match area_of_component[c] {
                None => area_of_component[c] = Some(a),
                Some(existing) if existing != a => report.contested_buses += 1,
                Some(_) => {}
            }
        }
    }
    let of_bus: Vec<Option<usize>> = (0..n).map(|b| area_of_component[component[b]]).collect();
    report.unassigned_buses = of_bus.iter().filter(|a| a.is_none()).count();

    Ok(ControlAreaImport { of_bus, targets, tolerances, ids, report })
}

/// Two regulating machines asked one bus to hold different voltages.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetConflict {
    pub bus: usize,
    /// The target already written at that bus, per-unit.
    pub existing: f64,
    /// What this controller asked for, per-unit.
    pub proposed: f64,
    /// The mRID of the controller that disagreed.
    pub id: String,
}

/// What generator-side voltage control did to the bus list.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VoltageControlReport {
    /// Buses held by at least one regulating machine.
    pub regulated_buses: usize,
    /// Of those, how many are held by more than one — the case whose reactive
    /// capability has to be summed rather than overwritten.
    pub shared_buses: usize,
    /// Controllers whose target disagreed with one already written at the same
    /// bus. Empty on every vendored fixture; reported rather than resolved,
    /// because neither answer is right — see
    /// [`VoltageControl::regulate`].
    pub target_conflicts: Vec<TargetConflict>,
}

/// Accumulates what the regulating machines at each bus jointly hold.
///
/// # Why this exists
///
/// `SynchronousMachine`, `StaticVarCompensator`, `PowerElectronicsConnection`
/// and `ExternalNetworkInjection` each carry a `RegulatingControl`, and each
/// used to write the controlled bus's reactive limits with `=` while writing
/// its injections with `+=` — the same loop body, opposite conventions. Two
/// machines on one bus therefore contributed both their reactive power to the
/// injection but only the **last one's capability** to the limits, so the bus
/// ran with understated headroom and
/// [`ReactiveLimits`](crate::outerloop::ReactiveLimits) clamped it early.
///
/// Not rare: 62 of RealGrid's 417 voltage-regulated nodes are held by more
/// than one machine, 2 of FullGrid's 3, and 1 of MicroGrid-Type1's 5. It went
/// unnoticed because Q-limit enforcement was opt-in and library-only until the
/// outer-loop layer exposed it.
struct VoltageControl {
    /// Whether a bus has had any contribution yet. The first assigns (the
    /// bus starts at ±∞, which is "no limit" rather than "zero capability"),
    /// every later one adds.
    seen: Vec<bool>,
    /// The target already written at each bus, and by whom.
    target: Vec<Option<(f64, String)>>,
    controllers: Vec<usize>,
    conflicts: Vec<TargetConflict>,
}

impl VoltageControl {
    fn new(n: usize) -> Self {
        Self {
            seen: vec![false; n],
            target: vec![None; n],
            controllers: vec![0; n],
            conflicts: Vec::new(),
        }
    }

    /// One machine holding `bus` at `target_pu` with reactive capability
    /// `(q_min, q_max)`, already in per-unit.
    ///
    /// Limits **sum** across machines. Infinity propagates, which is the right
    /// reading: one machine with no stated limit makes the bus's joint
    /// capability unlimited.
    ///
    /// The target does **not** sum, and the last writer still wins. That is
    /// deliberately unchanged: no vendored fixture has two controllers
    /// disagreeing about a target — checked across MicroGrid, SmallGrid,
    /// Svedala, FullGrid and RealGrid — so any resolution rule would be
    /// untested, and picking one here would make this a behaviour change
    /// rather than the limits-only fix it is. A disagreement is recorded in
    /// [`VoltageControlReport::target_conflicts`] instead. Doing it properly
    /// means reactive dispatch inside the Newton system, which is a different
    /// job.
    fn regulate(
        &mut self,
        buses: &mut [Bus],
        bus: usize,
        target_pu: f64,
        q_min: f64,
        q_max: f64,
        id: &str,
    ) {
        if buses[bus].bus_type == BusType::PQ {
            buses[bus].bus_type = BusType::PV;
        }
        if let Some((existing, _)) = &self.target[bus] {
            if (*existing - target_pu).abs() > 1e-9 {
                self.conflicts.push(TargetConflict {
                    bus,
                    existing: *existing,
                    proposed: target_pu,
                    id: id.to_string(),
                });
            }
        }
        self.target[bus] = Some((target_pu, id.to_string()));
        buses[bus].voltage_mag = target_pu;

        if self.seen[bus] {
            buses[bus].q_min += q_min;
            buses[bus].q_max += q_max;
        } else {
            buses[bus].q_min = q_min;
            buses[bus].q_max = q_max;
            self.seen[bus] = true;
        }
        self.controllers[bus] += 1;
    }

    fn finish(self) -> VoltageControlReport {
        VoltageControlReport {
            regulated_buses: self.controllers.iter().filter(|c| **c > 0).count(),
            shared_buses: self.controllers.iter().filter(|c| **c > 1).count(),
            target_conflicts: self.conflicts,
        }
    }
}

/// What a tap-regulation import could not use, counted rather than dropped.
///
/// Follows [`LimitImportReport`]'s precedent, and for the same reason: a
/// silently dropped control is how an importer produces a plausible wrong
/// answer. A network solved with a control gridoxide never read looks exactly
/// like one solved correctly, only at the wrong voltage.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TapRegulationReport {
    /// Controls converted.
    pub converted: usize,
    /// Present but switched off, on the control or on the tap changer. Data,
    /// not an error — `PowerFlow`'s single control is one.
    pub disabled: usize,
    /// `RegulatingControl.Terminal` named no bus this dataset defines, or the
    /// changer sat on an end no converted transformer owns.
    pub unattached: usize,
    /// No `targetValue`, so there is nothing to regulate towards.
    pub without_target: usize,
    /// A mode gridoxide does not model — `reactivePower`, `currentFlow`,
    /// `admittance`, `temperature`, `powerFactor`. Recognised and counted, in
    /// the same spirit as [`LimitImportReport::skipped`].
    pub unsupported_mode: usize,
    /// An `activePower` control whose regulated terminal resolves to no branch
    /// this converter produced. Distinguished from `unattached` because the
    /// control itself is well-formed; it is the flow that cannot be located.
    pub unresolved_flow: usize,
}

/// Reads every `TapChangerControl` in the dataset onto gridoxide's own indices.
///
/// A separate pass over the dataset returning a side table plus a report, which
/// is [`cgmes_operational_limits`]'s shape and for the same reasons: the
/// conversion tuple is already at its limit, and what could not be resolved
/// deserves counting.
///
/// # Units
///
/// [`TapRegulation`](crate::outerloop::TapRegulation) is per-unit throughout,
/// so the conversion happens here where the bases are known: a `voltage`
/// target divides by the controlled bus's own `u_rated`, an `activePower`
/// target by `s_base_va`. `targetDeadband` follows its target's unit, and CGMES
/// states it as a **full width** — the controller is satisfied within half of
/// it either side, which is what the outer loop assumes.
#[allow(clippy::too_many_arguments)]
fn read_tap_regulation(
    ds: &CimDataset,
    tap_index: &TapChangerIndex,
    terms: &TerminalIndex,
    buses: &[Bus],
    lines: &[Line],
    tap_changers: &[Option<crate::types::TapChanger>],
    changer_end: &[Option<String>],
    terminal_sides: &[Vec<(String, crate::branch_flow::Terminal)>],
    s_base_va: f64,
) -> (Vec<crate::outerloop::TapRegulation>, TapRegulationReport) {
    use crate::outerloop::{RegulationMode, TapRegulation};

    let mut out = Vec::new();
    let mut report = TapRegulationReport::default();

    // Terminal mRID -> (flat branch index, side), for the `activePower` case.
    // Lines first, then transformers, matching `branch_flow::branch_params`.
    let mut flow_of: HashMap<&str, (usize, crate::branch_flow::Terminal)> = HashMap::new();
    for (i, sides) in terminal_sides.iter().enumerate() {
        for (terminal, side) in sides {
            flow_of.insert(terminal.as_str(), (lines.len() + i, *side));
        }
    }

    for (i, changer) in tap_changers.iter().enumerate() {
        if changer.is_none() {
            continue;
        }
        let Some(end_mrid) = changer_end[i].as_deref() else { continue };
        let Some((control_mrid, control_enabled)) = tap_index.control_of(ds, end_mrid) else {
            continue;
        };
        // A changer with no `TapChangerControl` at all has a fixed position by
        // construction. FullGrid's `BE_TR2_HVDC2` is one, and it is why the
        // published SV moves a tap this importer must not.
        let Some(control_mrid) = control_mrid else { continue };
        let Some(rc) = get::<cimstructs::TapChangerControl>(ds, &control_mrid) else {
            report.unattached += 1;
            continue;
        };
        let rc = &rc.base;

        // Both switches must be on: the control itself, and the changer's own
        // `controlEnabled`. Either off means the position is an input.
        let enabled = control_enabled && rc.enabled == Some(true);
        if !enabled {
            report.disabled += 1;
            continue;
        }

        let Some(term_ref) = &rc.terminal else {
            report.unattached += 1;
            continue;
        };
        let Some(controlled_bus) = terms.bus_via_terminal_mrid(&term_ref.mrid) else {
            report.unattached += 1;
            continue;
        };
        let Some(target) = rc.target_value else {
            report.without_target += 1;
            continue;
        };
        let mult = unit_multiplier(rc.target_value_unit_multiplier.as_ref().map(|u| u.uri.as_str()));
        let deadband = rc.target_deadband.unwrap_or(0.0);

        let uri = rc.mode.as_ref().map(|m| m.uri.as_str()).unwrap_or("");
        let (mode, target, deadband) = if uri.ends_with(".voltage") {
            let base = buses[controlled_bus].u_rated;
            (RegulationMode::Voltage, target * mult / base, deadband * mult / base)
        } else if uri.ends_with(".activePower") {
            let Some(&(branch, terminal)) = flow_of.get(term_ref.mrid.as_str()) else {
                report.unresolved_flow += 1;
                continue;
            };
            // `mult` already carries MW → W, exactly as it carries kV → V for
            // a voltage target: powsybl's own SSH export writes "M" here and
            // "k" there, one multiplier covering both `targetValue` and
            // `targetDeadband`. Multiplying by 1e6 again on top — which an
            // earlier draft did — put FullGrid's -65 MW target at -650000 pu.
            (
                RegulationMode::ActivePower { branch, terminal },
                target * mult / s_base_va,
                deadband * mult / s_base_va,
            )
        } else {
            report.unsupported_mode += 1;
            continue;
        };

        report.converted += 1;
        out.push(TapRegulation {
            transformer: i,
            controlled_bus,
            mode,
            target,
            deadband,
            enabled: true,
            id: control_mrid,
        });
    }

    (out, report)
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
) -> Result<(Transformer, Option<crate::types::TapChanger>), CgmesError> {
    let term1 = end1.base.terminal.as_ref().ok_or_else(|| missing("PowerTransformerEnd", end1.mrid_str(), "Terminal"))?;
    let term2 = end2.base.terminal.as_ref().ok_or_else(|| missing("PowerTransformerEnd", end2.mrid_str(), "Terminal"))?;
    let bus1 = terms.bus_via_terminal_mrid(&term1.mrid)
        .ok_or_else(|| CgmesError::UnresolvedReference { from_type: "PowerTransformerEnd", from_mrid: end1.mrid_str().to_string(), field: "Terminal.TopologicalNode" })?;
    let bus2 = terms.bus_via_terminal_mrid(&term2.mrid)
        .ok_or_else(|| CgmesError::UnresolvedReference { from_type: "PowerTransformerEnd", from_mrid: end2.mrid_str().to_string(), field: "Terminal.TopologicalNode" })?;

    let tap1 = tap_index.effect_for_end(ds, end1.mrid_str(), end1.x.unwrap_or(0.0))?;
    let tap2 = tap_index.effect_for_end(ds, end2.mrid_str(), end2.x.unwrap_or(0.0))?;
    // Which end holds the changer decides how a position composes into
    // `Transformer::tap`, so it is recorded here alongside the current step's
    // effect rather than re-derived below.
    let on_end1 = tap1.is_some();
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
    // Structural ratio: each end's own nameplate `ratedU` against the system
    // base voltage of the bus it actually sits on. Both ends contribute, and
    // for the same reason — a nameplate that differs from its bus's system
    // nominal *is* an off-nominal ideal-transformer ratio, independent of any
    // tap changer.
    //
    // Deriving it, with `from` = bus2 and `to` = bus1: at no load the physical
    // ratio across the device is `V_from / V_to = u2 / u1`, and `tap` is that
    // ratio expressed per-unit, so
    //
    //   tap = (V_from / base2) / (V_to / base1) = (u2 / base2) / (u1 / base1)
    //
    // i.e. exactly the reciprocal of `structural` below. Only the `u1 / base1`
    // half used to be applied here, which silently dropped end2's own
    // contribution whenever its nameplate differed from its bus's base — no
    // effect on a fixture where every `ratedU` equals its bus's `nominalVoltage`
    // (all of this project's hand-authored ones), but pervasive on real data:
    // 666 of RealGrid's 1,509 two-winding transformers and 4 of Svedala's 53,
    // by up to 7.2%, which showed up directly as a ~7% solved-voltage error on
    // the affected buses.
    let structural_1 = u1 / buses[bus1].u_rated;
    // A missing end2 `ratedU` means "no nameplate of its own to disagree with",
    // i.e. a unity contribution — not an error, since end2 carries no impedance.
    let structural_2 = end2.rated_u.map_or(1.0, |u2| (u2 * 1e3) / buses[bus2].u_rated);
    let structural = structural_1 / structural_2;
    let tap = tap / structural;

    // The whole table, composed exactly as the current step above was, so that
    // reading position `step` back out of it reproduces `tap` bit for bit.
    let end_mrid = if on_end1 { end1.mrid_str() } else { end2.mrid_str() };
    let xtx = if on_end1 { end1.x.unwrap_or(0.0) } else { end2.x.unwrap_or(0.0) };
    let changer = tap_index.steps_for_end(ds, end_mrid, xtx)?.map(|raw| {
        raw.compose(
            |t| if on_end1 { Complex::new(1.0, 0.0) / t / structural } else { t / structural },
            |x| {
                // An override on end 2 is dropped by the match above, which is
                // this branch's own convention: only end 1 carries series
                // impedance, so only a changer there can move it.
                let x = if on_end1 { x.unwrap_or_else(|| end1.x.unwrap_or(0.0)) } else { x1 };
                Complex::new(z_base, 0.0) / Complex::new(r1, x)
            },
        )
    });

    Ok((
        Transformer {
            from: bus2,
            to: bus1,
            from_status: terms.connected_via_terminal_mrid(&term2.mrid) as u8,
            to_status: terms.connected_via_terminal_mrid(&term1.mrid) as u8,
            y_series: Complex::new(z_base, 0.0) / Complex::new(r1, x1),
            y_shunt: Complex::new(g1, b1) * z_base,
            tap,
        },
        changer,
    ))
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
) -> Result<(Transformer, Option<crate::types::TapChanger>), CgmesError> {
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

    let changer = tap_index.steps_for_end(ds, end.mrid_str(), end.x.unwrap_or(0.0))?.map(|raw| {
        raw.compose(
            |t| Complex::new(1.0, 0.0) / t / structural,
            |xo| {
                let x = xo.unwrap_or_else(|| end.x.unwrap_or(0.0));
                Complex::new(z_base, 0.0) / Complex::new(r, x)
            },
        )
    });

    Ok((
        Transformer {
            from: star_idx,
            to: bus,
            from_status: 1, // the synthesized star bus itself is never "disconnected"
            to_status: terms.connected_via_terminal_mrid(&term.mrid) as u8,
            y_series: Complex::new(z_base, 0.0) / Complex::new(r, x),
            y_shunt: Complex::new(g, b) * z_base,
            tap,
        },
        changer,
    ))
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

// ============================================================================
// HVDC (VsConverter/CsConverter + DC network) support
// ============================================================================
//
// Unlike every other Step above, DC resolution isn't folded into
// `cgmes_to_buses_and_branches` itself — it runs as a separate pass,
// `cgmes_resolve_dc_converters`, called after it, mutating the AC `buses` it
// returned in place. This is deliberate, not a layering shortcut: every
// converter control mode FullGrid actually uses (`udc`/`dcVoltage` DC-voltage
// slack, `dcCurrent` fixed DC current, `pPcc`/`activePower` fixed AC-side
// power) has its DC-side target either fully static (straight from the SSH
// profile) or a direct result of solving the DC network — none of them make
// a converter's DC-side behavior depend on the AC network's own solved
// state. So the whole DC network can be solved once, standalone, before the
// AC Newton-Raphson solve ever runs, rather than needing a generic outer
// AC<->DC coupling loop that repeatedly re-solves both sides. (A control mode
// that genuinely coupled the two — e.g. `pPccAndUdcDroop` — would need one;
// FullGrid doesn't use any, so `UnsupportedConverterControl` covers that gap
// honestly instead of silently guessing.)
//
// Once the DC network is solved, every converter's final AC-side power is
// recovered by one identity, valid for every role (slack or follower) and
// every direction (rectifying or inverting) via simple energy conservation:
//
//   Pac_absorbed = P_dc_injected + loss(Idc)
//
// where `Pac_absorbed` is in CIM's own "load sign convention" (positive =
// power flowing OUT of the AC node INTO the converter), `P_dc_injected` is
// the power the converter pushes out of its own DC terminal into the DC
// network (`V_dc * Idc`, `dc::injected_currents`' own sign convention), and
// `loss` is always >= 0 regardless of direction (it depends on `|Idc|`/
// `Idc^2`). No rectifier/inverter branch is needed anywhere in this code.

/// Ideal-switch resistance stamped for closed DC switches/breakers/
/// disconnectors, which carry no resistance field in CIM at all — small
/// relative to FullGrid's real `DCLineSegment` resistance (2.5 Ω) so it's
/// numerically negligible without ill-conditioning the small (<20-bus) dense
/// Newton solve in `dc::solve_dc_network`. Mirrors the same "ideal switch as
/// a tiny resistance" approach
/// `docs/src/powerflow/zero_impedance_branches.md` documents for AC.
const DC_SWITCH_R: f64 = 1e-4;

/// Absolute mismatch tolerance for `dc::solve_dc_network` calls below (MW/kA
/// scale, not `newton_raphson`'s p.u.-scale `1e-6`/`1e-9`). Looser than the
/// dc.rs unit tests' own `1e-9`, deliberately: those synthetic networks have
/// no `DC_SWITCH_R`-scale branches, so they're well-conditioned enough for
/// `1e-9` to be reachable in double precision. A real CGMES DC network mixes
/// `DC_SWITCH_R` (1e-4 Ω, G≈10,000) with real line resistances (2.5 Ω here,
/// G≈0.4) — a ~25,000:1 conductance ratio that amplifies rounding error
/// enough through Gaussian elimination that `1e-9` is empirically
/// unreachable on FullGrid (confirmed: it ran to `max_iter` without
/// technically converging, even though the solved voltages were already
/// correct to 6 decimal places by iteration 3 at this looser tolerance).
const DC_SOLVE_TOL: f64 = 1e-6;

fn equipment_in_service(in_service: Option<bool>, normally_in_service: Option<bool>) -> bool {
    in_service.or(normally_in_service).unwrap_or(true)
}

/// `pole_loss_p = idleLoss + switchingLoss*|Idc_pu| + resistiveLoss*Idc_pu^2`,
/// per `ACDCConverter.poleLossP`'s own doc text. `Idc_pu` normalizes `idc_amps`
/// by a base current `I_base = baseS*1000/ratedUdc` (MVA*1000/kV = A) —
/// cross-validated against FullGrid's own data, not assumed: the Inverter
/// CsConverter's `baseS=334.6`/`ratedUdc=167.3` gives `I_base≈2000.0`,
/// matching its own explicit `CsConverter.ratedIdc=2000` almost exactly.
/// Treating `idc_amps` as already-per-unit (skipping this normalization)
/// gives an absurd ~50,000 MW "loss" at FullGrid's real Idc values (hundreds
/// of amps) — confirming the coefficients are meant to be applied to a
/// per-unit, not raw-Amp, current.
fn converter_loss_mw(idle: f64, switching: f64, resistive: f64, base_s: f64, rated_udc: f64, idc_amps: f64) -> f64 {
    let i_base = base_s * 1000.0 / rated_udc;
    let idc_pu = if i_base > 0.0 { idc_amps / i_base } else { 0.0 };
    idle + switching * idc_pu.abs() + resistive * idc_pu * idc_pu
}

/// What a converter's `pPccControl` mode fixes, before it's translated into
/// a `dc::DcBusRole` (which needs unit conversion — kV/kA vs. CIM's kV/A —
/// and, for `FixedAc`, the loss-curve self-consistency loop below).
#[derive(Clone, Copy)]
enum ConverterRole {
    /// `udc`/`dcVoltage`: DC voltage fixed, in kV (`ACDCConverter.targetUdc`).
    UdcSlack(f64),
    /// `dcCurrent`: DC current fixed, in A, already signed by
    /// `CsConverter.operatingMode` (positive = injecting into the DC
    /// network, i.e. rectifying).
    FixedIdc(f64),
    /// `pPcc`/`activePower`: AC-side power fixed, in MW, CIM load-sign
    /// convention (`ACDCConverter.targetPpcc`).
    FixedAc(f64),
}

fn classify_vs_converter(vc: &VsConverter, mrid: &str) -> Result<ConverterRole, CgmesError> {
    let suffix = vc.p_pcc_control.as_ref().and_then(|u| u.uri.rsplit('.').next());
    match suffix {
        Some("udc") => Ok(ConverterRole::UdcSlack(vc.base.target_udc.ok_or_else(|| missing("VsConverter", mrid, "targetUdc"))?)),
        Some("pPcc") => Ok(ConverterRole::FixedAc(vc.base.target_ppcc.ok_or_else(|| missing("VsConverter", mrid, "targetPpcc"))?)),
        other => Err(CgmesError::UnsupportedConverterControl { mrid: mrid.to_string(), mode: other.unwrap_or("(none)").to_string() }),
    }
}

fn classify_cs_converter(cc: &CsConverter, mrid: &str) -> Result<ConverterRole, CgmesError> {
    let suffix = cc.p_pcc_control.as_ref().and_then(|u| u.uri.rsplit('.').next());
    match suffix {
        Some("dcVoltage") => Ok(ConverterRole::UdcSlack(cc.base.target_udc.ok_or_else(|| missing("CsConverter", mrid, "targetUdc"))?)),
        Some("dcCurrent") => {
            let idc = cc.target_idc.ok_or_else(|| missing("CsConverter", mrid, "targetIdc"))?;
            let is_rectifier = cc.operating_mode.as_ref().is_some_and(|m| m.uri.ends_with(".rectifier"));
            Ok(ConverterRole::FixedIdc(if is_rectifier { idc } else { -idc }))
        }
        Some("activePower") => Ok(ConverterRole::FixedAc(cc.base.target_ppcc.ok_or_else(|| missing("CsConverter", mrid, "targetPpcc"))?)),
        other => Err(CgmesError::UnsupportedConverterControl { mrid: mrid.to_string(), mode: other.unwrap_or("(none)").to_string() }),
    }
}

struct ConverterInfo {
    ac_bus: usize,
    dc_bus: usize,
    role: ConverterRole,
    idle_loss: f64,
    switching_loss: f64,
    resistive_loss: f64,
    base_s: f64,
    rated_udc: f64,
    q_mw: f64,
}

/// Equipment mrid -> its own (sorted-by-sequence) plain `DCTerminal` mrids,
/// each resolved to a `dc::DcBus` index — the DC-side analogue of
/// `TerminalIndex`, scoped to plain `DCTerminal` (lines/switches/ground/
/// shunt) only. Converters use `ACDCConverterDCTerminal` instead, resolved
/// separately below since they additionally need `polarity`.
struct DcTerminalIndex {
    by_equipment: HashMap<String, Vec<String>>,
    bus_of: HashMap<String, usize>,
}

impl DcTerminalIndex {
    fn build(ds: &CimDataset, dc_idx_of: &HashMap<String, usize>) -> Self {
        let mut raw: HashMap<String, Vec<(i64, String)>> = HashMap::new();
        let mut bus_of = HashMap::new();
        for t_mrid in by_type(ds, "DCTerminal") {
            let Some(t) = get::<DCTerminal>(ds, t_mrid) else { continue };
            if let Some(ce) = &t.dc_conducting_equipment {
                let seq = t.base.base.sequence_number.unwrap_or(1);
                raw.entry(ce.mrid.clone()).or_default().push((seq, t_mrid.clone()));
            }
            // Direct `DCTopologicalNode` reference, merged in from the TP
            // profile onto the same terminal mrid — confirmed present on
            // every DCTerminal instance in FullGrid's TP file, but a
            // `DCNode`-mediated fallback isn't added here since it's never
            // exercised; keeping this as direct-only mirrors what's actually
            // used rather than speculatively guessing at an untested path.
            if let Some(tn) = &t.base.dc_topological_node {
                if let Some(&idx) = dc_idx_of.get(&tn.mrid) {
                    bus_of.insert(t_mrid.clone(), idx);
                }
            }
        }
        let by_equipment = raw.into_iter().map(|(eq, mut v)| {
            v.sort_by_key(|(seq, _)| *seq);
            (eq, v.into_iter().map(|(_, m)| m).collect())
        }).collect();
        DcTerminalIndex { by_equipment, bus_of }
    }

    /// The two DC buses a two-terminal piece of DC equipment connects, in
    /// terminal-sequence order (order doesn't matter for a plain resistor).
    fn line(&self, equipment_mrid: &str) -> Option<(usize, usize)> {
        let ts = self.by_equipment.get(equipment_mrid)?;
        if ts.len() < 2 {
            return None;
        }
        Some((*self.bus_of.get(&ts[0])?, *self.bus_of.get(&ts[1])?))
    }

    /// The single DC bus a one-terminal piece of DC equipment (DCGround,
    /// DCShunt) connects to.
    fn single_bus(&self, equipment_mrid: &str) -> Option<usize> {
        let ts = self.by_equipment.get(equipment_mrid)?;
        self.bus_of.get(ts.first()?).copied()
    }
}

/// The outcome of resolving a dataset's HVDC equipment: every
/// `DCTopologicalNode`'s solved voltage (kV) and the underlying DC network
/// solve status. `dc_bus_mrids[i]`/`voltages_kv[i]` are index-aligned.
///
/// The whole dataset's DC equipment is solved as one combined graph rather
/// than split per-link: `dc::solve_dc_network`'s own connected-components
/// handling already solves every electrically independent HVDC link (and
/// isolates a dead/disconnected subgraph, like FullGrid's spare switchyard
/// branch) correctly in a single call — exactly as AC's own multi-island
/// support solves every component in one shared Newton-Raphson call. So
/// there's one `DcResolution` for the whole dataset, not one per link.
pub struct DcResolution {
    pub dc_bus_mrids: Vec<String>,
    pub voltages_kv: Vec<f64>,
    pub status: DcSolveStatus,
}

/// Resolves every `VsConverter`/`CsConverter` in `ds` into fixed `p_spec`/
/// `q_spec` contributions on `buses` (the AC buses `cgmes_to_buses_and_branches`
/// already returned — this is meant to run immediately after it, over that
/// same `buses`, before it's passed to the AC power-flow solve). Returns
/// `None` if the dataset has no `DCTopologicalNode`s at all (no HVDC
/// equipment, or the TP profile hasn't been loaded for it).
///
/// `q_spec` is taken directly from each converter's static SSH `q` (CIM's
/// own "starting value for a steady state solution in the case a simplified
/// power flow model is used" language) rather than solved dynamically — a
/// deliberate, documented scope cut: implementing `qPccControl`/droop as a
/// genuine control mode is a comparably-sized second project, and FullGrid's
/// own converters have static, non-dynamic Q targets, so this doesn't cost
/// accuracy in the fixture used to validate this function.
pub fn cgmes_resolve_dc_converters(
    ds: &CimDataset, buses: &mut [Bus], s_base_va: f64,
) -> Result<Option<DcResolution>, CgmesError> {
    let dctn_mrids = by_type(ds, "DCTopologicalNode");
    if dctn_mrids.is_empty() {
        return Ok(None);
    }

    // Rebuilds exactly the same Steps 1+2+2.5 `cgmes_to_buses_and_branches`
    // itself runs (bus skeleton, `TerminalIndex`, closed-switch merge) via
    // the shared `build_ac_bus_skeleton` helper, so a converter's
    // `PccTerminal`/own terminal resolves to the same AC bus index that call
    // already produced — deterministic given the same `ds`, since every step
    // involved (`by_type` ordering, `TerminalIndex::build`, the union-find
    // merge) depends only on `ds`'s own contents, not on anything from that
    // other call's own local state. Only `ac_terms` (for
    // `bus_via_terminal_mrid`/`bus`) is kept; the rebuilt bus skeleton and
    // `idx_of` are discarded — this function only ever writes into the
    // caller's own already-merged `buses`.
    let (_, _, ac_terms) = build_ac_bus_skeleton(ds)?;

    let dc_idx_of: HashMap<String, usize> = dctn_mrids.iter().enumerate()
        .map(|(i, mrid)| (mrid.clone(), i)).collect();
    let mut dc_buses: Vec<DcBus> = (0..dctn_mrids.len()).map(|idx| DcBus {
        idx, role: DcBusRole::Passive, udc_fixed: 0.0, voltage: 0.0, shunt_g: 0.0,
    }).collect();

    let terms = DcTerminalIndex::build(ds, &dc_idx_of);

    // --- DC branches ---
    let mut dc_lines: Vec<DcLine> = Vec::new();
    for mrid in by_type(ds, "DCLineSegment") {
        let seg: &DCLineSegment = require(ds, mrid, "DCLineSegment", mrid, "(self)")?;
        if !equipment_in_service(seg.base.base.in_service, seg.base.base.normally_in_service) { continue }
        let Some((a, b)) = terms.line(mrid) else { continue };
        let r = seg.resistance.ok_or_else(|| missing("DCLineSegment", mrid, "resistance"))?;
        dc_lines.push(DcLine { from: a, to: b, r });
    }
    for mrid in by_type(ds, "DCSeriesDevice") {
        let dev: &DCSeriesDevice = require(ds, mrid, "DCSeriesDevice", mrid, "(self)")?;
        if !equipment_in_service(dev.base.base.in_service, dev.base.base.normally_in_service) { continue }
        let Some((a, b)) = terms.line(mrid) else { continue };
        let r = dev.resistance.ok_or_else(|| missing("DCSeriesDevice", mrid, "resistance"))?;
        dc_lines.push(DcLine { from: a, to: b, r });
    }
    for mrid in by_type(ds, "DCBreaker") {
        let br: &DCBreaker = require(ds, mrid, "DCBreaker", mrid, "(self)")?;
        if !equipment_in_service(br.base.base.base.in_service, br.base.base.base.normally_in_service) { continue }
        let Some((a, b)) = terms.line(mrid) else { continue };
        dc_lines.push(DcLine { from: a, to: b, r: DC_SWITCH_R });
    }
    for mrid in by_type(ds, "DCDisconnector") {
        let dc: &DCDisconnector = require(ds, mrid, "DCDisconnector", mrid, "(self)")?;
        if !equipment_in_service(dc.base.base.base.in_service, dc.base.base.base.normally_in_service) { continue }
        let Some((a, b)) = terms.line(mrid) else { continue };
        dc_lines.push(DcLine { from: a, to: b, r: DC_SWITCH_R });
    }
    for mrid in by_type(ds, "DCSwitch") {
        let sw: &DCSwitch = require(ds, mrid, "DCSwitch", mrid, "(self)")?;
        if !equipment_in_service(sw.base.base.in_service, sw.base.base.normally_in_service) { continue }
        let Some((a, b)) = terms.line(mrid) else { continue };
        dc_lines.push(DcLine { from: a, to: b, r: DC_SWITCH_R });
    }

    // --- DCGround (fixes its bus at 0 kV) / DCShunt (adds shunt conductance) ---
    // `DCBusbar`/`DCChopper` get no code at all: a busbar is `Passive` by
    // default (the role every DcBus starts with), and CIM defines no
    // steady-state resistance for a chopper (a transient overvoltage-
    // protection device) — both documented limitations, harmless for
    // FullGrid since the branch feeding its own spare busbar/chopper is
    // already out of service and gets isolated by dead-subgraph detection.
    for mrid in by_type(ds, "DCGround") {
        let g: &DCGround = require(ds, mrid, "DCGround", mrid, "(self)")?;
        if !equipment_in_service(g.base.base.in_service, g.base.base.normally_in_service) { continue }
        let Some(bus) = terms.single_bus(mrid) else { continue };
        // `DCGround.r` (a real grounding resistance) is 0 in every FullGrid
        // instance; a nonzero value isn't modeled as a resistor-to-earth
        // here (there's no separate "earth" bus in this graph) — a known
        // simplification, not silently wrong for this fixture specifically.
        dc_buses[bus].role = DcBusRole::Ground;
        dc_buses[bus].udc_fixed = 0.0;
    }
    for mrid in by_type(ds, "DCShunt") {
        let sh: &DCShunt = require(ds, mrid, "DCShunt", mrid, "(self)")?;
        if !equipment_in_service(sh.base.base.in_service, sh.base.base.normally_in_service) { continue }
        let Some(bus) = terms.single_bus(mrid) else { continue };
        if let Some(r) = sh.resistance {
            if r != 0.0 {
                dc_buses[bus].shunt_g += 1.0 / r;
            }
        }
    }

    // --- Converters: resolve each to its own positive-pole DC bus ---
    let mut positive_pole_of: HashMap<String, usize> = HashMap::new();
    for t_mrid in by_type(ds, "ACDCConverterDCTerminal") {
        let Some(t) = get::<ACDCConverterDCTerminal>(ds, t_mrid) else { continue };
        let is_positive = t.polarity.as_ref().is_some_and(|p| p.uri.ends_with(".positive"));
        if !is_positive { continue }
        let (Some(ce), Some(tn)) = (&t.dc_conducting_equipment, &t.base.dc_topological_node) else { continue };
        if let Some(&bus) = dc_idx_of.get(&tn.mrid) {
            positive_pole_of.insert(ce.mrid.clone(), bus);
        }
    }

    // `ACDCConverter.PccTerminal` is optional in CIM and, confirmed on
    // MicroGrid-Type2-HVDC-MAS, genuinely absent from some real exports —
    // the point of common coupling then defaults to the converter's own
    // regular (AC-side) `Terminal`, sequence 1, exactly like every other
    // `ConductingEquipment` in this file resolves its own connection point.
    let mut converters: Vec<ConverterInfo> = Vec::new();
    for mrid in by_type(ds, "VsConverter") {
        let vc: &VsConverter = require(ds, mrid, "VsConverter", mrid, "(self)")?;
        let Some(&dc_bus) = positive_pole_of.get(mrid) else { continue };
        let Some(ac_bus) = vc.base.pcc_terminal.as_ref()
            .and_then(|r| ac_terms.bus_via_terminal_mrid(&r.mrid))
            .or_else(|| ac_terms.bus(mrid, 0)) else { continue };
        converters.push(ConverterInfo {
            ac_bus, dc_bus,
            role: classify_vs_converter(vc, mrid)?,
            idle_loss: vc.base.idle_loss.unwrap_or(0.0),
            switching_loss: vc.base.switching_loss.unwrap_or(0.0),
            resistive_loss: vc.base.resistive_loss.unwrap_or(0.0),
            base_s: vc.base.base_s.ok_or_else(|| missing("VsConverter", mrid, "baseS"))?,
            rated_udc: vc.base.rated_udc.ok_or_else(|| missing("VsConverter", mrid, "ratedUdc"))?,
            q_mw: vc.base.q.unwrap_or(0.0),
        });
    }
    for mrid in by_type(ds, "CsConverter") {
        let cc: &CsConverter = require(ds, mrid, "CsConverter", mrid, "(self)")?;
        let Some(&dc_bus) = positive_pole_of.get(mrid) else { continue };
        let Some(ac_bus) = cc.base.pcc_terminal.as_ref()
            .and_then(|r| ac_terms.bus_via_terminal_mrid(&r.mrid))
            .or_else(|| ac_terms.bus(mrid, 0)) else { continue };
        converters.push(ConverterInfo {
            ac_bus, dc_bus,
            role: classify_cs_converter(cc, mrid)?,
            idle_loss: cc.base.idle_loss.unwrap_or(0.0),
            switching_loss: cc.base.switching_loss.unwrap_or(0.0),
            resistive_loss: cc.base.resistive_loss.unwrap_or(0.0),
            base_s: cc.base.base_s.ok_or_else(|| missing("CsConverter", mrid, "baseS"))?,
            rated_udc: cc.base.rated_udc.ok_or_else(|| missing("CsConverter", mrid, "ratedUdc"))?,
            q_mw: cc.base.q.unwrap_or(0.0),
        });
    }

    // --- Apply UdcSlack/FixedIdc roles directly: both are already DC-native
    // targets straight from SSH, no translation needed. ---
    for c in &converters {
        match c.role {
            ConverterRole::UdcSlack(udc_kv) => {
                dc_buses[c.dc_bus].role = DcBusRole::UdcSlack;
                dc_buses[c.dc_bus].udc_fixed = udc_kv;
            }
            // A/1000 -> kA, matching dc::solve_dc_network's implied units
            // (kV buses, Ω lines => kA currents, MW powers).
            ConverterRole::FixedIdc(idc_amps) => {
                dc_buses[c.dc_bus].role = DcBusRole::FixedIdc { idc_spec: idc_amps / 1000.0 };
            }
            ConverterRole::FixedAc(_) => {} // resolved below
        }
    }

    // --- FixedAc (AC-side-target) followers: self-consistently translate
    // the static AC target into a DC-side power via the loss curve. This
    // loop is entirely self-contained (only ever calls solve_dc_network,
    // never touches the AC solver) because the AC target is already a known
    // static SSH value, not something derived from AC network state — the
    // only unknown is the converter's own Idc, needed for the loss term,
    // which the DC solve itself produces. ---
    for c in &converters {
        let ConverterRole::FixedAc(pac_absorbed_target) = c.role else { continue };
        let mut idc_amps = 0.0;
        for _ in 0..20 {
            let loss = converter_loss_mw(c.idle_loss, c.switching_loss, c.resistive_loss, c.base_s, c.rated_udc, idc_amps);
            dc_buses[c.dc_bus].role = DcBusRole::FixedP { p_spec: pac_absorbed_target - loss };
            let status = solve_dc_network(&mut dc_buses, &dc_lines, DC_SOLVE_TOL, 100);
            if !status.converged {
                return Err(CgmesError::DcNetworkDidNotConverge);
            }
            let currents = injected_currents(&dc_buses, &dc_lines);
            let idc_new = currents[c.dc_bus].abs() * 1000.0; // kA -> A
            let converged = (idc_new - idc_amps).abs() < 1e-6;
            idc_amps = idc_new;
            if converged {
                break;
            }
        }
    }

    // --- Final solve (also the only solve needed if there were no FixedAc
    // followers at all) and universal AC-side power recovery. ---
    let status = solve_dc_network(&mut dc_buses, &dc_lines, DC_SOLVE_TOL, 100);
    let currents = injected_currents(&dc_buses, &dc_lines);
    for c in &converters {
        let p_dc_injected = dc_buses[c.dc_bus].voltage * currents[c.dc_bus];
        let idc_amps = currents[c.dc_bus].abs() * 1000.0;
        let loss = converter_loss_mw(c.idle_loss, c.switching_loss, c.resistive_loss, c.base_s, c.rated_udc, idc_amps);
        let pac_absorbed = p_dc_injected + loss;
        buses[c.ac_bus].p_spec += -pac_absorbed * 1e6 / s_base_va;
        buses[c.ac_bus].q_spec += -c.q_mw * 1e6 / s_base_va;
    }

    Ok(Some(DcResolution {
        dc_bus_mrids: dctn_mrids.to_vec(),
        voltages_kv: dc_buses.iter().map(|b| b.voltage).collect(),
        status,
    }))
}

/// Where a bus view's node set comes from.
///
/// CGMES carries connectivity at two levels, and which one is available
/// depends on how the model was exported rather than on anything intrinsic.
/// `ConnectivityNode` (EQ) is the substation-level truth — every point where
/// terminals meet. `TopologicalNode` (TP) is a *partial reduction of it* that
/// the exporter already performed, usually but not always collapsing closed
/// switches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CgmesTopologyMode {
    /// Node-breaker when EQ carries `ConnectivityNode`s, otherwise the TP path.
    ///
    /// Not yet the default anywhere — nothing in the solver pipeline consumes a
    /// node-breaker view, so selecting one automatically would only change
    /// which code path produced an identical answer. It becomes meaningful when
    /// a retention policy can keep a switch.
    Auto,
    /// `TopologicalNode` *is* the bus. gridoxide's historical path, and still
    /// what every importer entry point uses.
    #[default]
    BusBranchFromTp,
    /// `ConnectivityNode`s are the nodes and switches are edges between them.
    NodeBreakerFromEq,
}

/// The node-breaker graph read out of EQ+SSH, with the identity needed to map
/// back to the source model.
///
/// Both mrid vectors are indexed by the corresponding newtype's `.0`, so
/// `node_mrids[node.0]` is the `ConnectivityNode` a [`NodeIdx`] came from and
/// `switch_mrids[switch.0]` is the switching device a [`SwitchIdx`] came from.
///
/// [`NodeIdx`]: crate::topology::NodeIdx
/// [`SwitchIdx`]: crate::topology::SwitchIdx
#[derive(Clone, Debug)]
pub struct CgmesNodeBreaker {
    pub topology: crate::topology::NodeBreakerTopology,
    pub node_mrids: Vec<String>,
    pub switch_mrids: Vec<String>,
    /// `ConnectivityNode` mrid -> the `TopologicalNode` mrid the exporter
    /// assigned it, where it assigned one.
    ///
    /// This is the exporter's own answer to the same question
    /// [`bus_view`](crate::topology::bus_view) computes, which makes it free
    /// ground truth — see `tests/cgmes_node_breaker_test.rs`. Boundary nodes
    /// legitimately have none.
    pub tn_of_node: Vec<Option<String>>,
}

/// Reads the node-breaker graph: `ConnectivityNode`s as nodes, the nine CIM
/// switch classes plus `Junction` as edges, `BusbarSection`s as busbars.
///
/// Independent of [`build_ac_bus_skeleton`]'s `TopologicalNode` path, and
/// deliberately so — the point of this function is to derive connectivity
/// *without* relying on the reduction an exporter may or may not have
/// performed. It needs EQ (for `ConnectivityNode` and equipment) and SSH (for
/// switch positions); it does not need TP at all.
///
/// Switch order matches `switch_topology`'s and, through it, the historical
/// order of `merge_closed_switches` — see
/// [`NodeBreakerTopology::switches`](crate::topology::NodeBreakerTopology::switches)
/// for why that is load-bearing.
pub fn cgmes_node_breaker_topology(ds: &CimDataset) -> Result<CgmesNodeBreaker, CgmesError> {
    use crate::topology::model::{NodeIdx, Switch as TopoSwitch, SwitchKind};

    // Nodes, in `by_type` order.
    let cn_mrids = by_type(ds, "ConnectivityNode");
    let mut node_of_cn: HashMap<&str, usize> = HashMap::with_capacity(cn_mrids.len());
    let mut node_mrids: Vec<String> = Vec::with_capacity(cn_mrids.len());
    let mut tn_of_node: Vec<Option<String>> = Vec::with_capacity(cn_mrids.len());
    for mrid in cn_mrids {
        let cn: &cimstructs::ConnectivityNode =
            require(ds, mrid, "ConnectivityNode", mrid, "(self)")?;
        node_of_cn.insert(mrid.as_str(), node_mrids.len());
        node_mrids.push(mrid.clone());
        tn_of_node.push(cn.topological_node.as_ref().map(|r| r.mrid.clone()));
    }

    // Terminal -> node, and equipment -> its terminals ordered by
    // `sequenceNumber`. A terminal with no `ConnectivityNode` reference is not
    // an error: a bus-branch export has none at all, which is exactly the case
    // `CgmesTopologyMode::Auto` distinguishes.
    let mut node_of_terminal: HashMap<&str, usize> = HashMap::new();
    let mut terminals_of_equipment: HashMap<&str, Vec<(i64, &str)>> = HashMap::new();
    for t_mrid in by_type(ds, "Terminal") {
        let t: &Terminal = require(ds, t_mrid, "Terminal", t_mrid, "(self)")?;
        if let Some(cn) = &t.connectivity_node {
            if let Some(&node) = node_of_cn.get(cn.mrid.as_str()) {
                node_of_terminal.insert(t_mrid.as_str(), node);
            }
        }
        if let Some(ce) = &t.conducting_equipment {
            terminals_of_equipment
                .entry(ce.mrid.as_str())
                .or_default()
                .push((t.base.sequence_number.unwrap_or(1), t_mrid.as_str()));
        }
    }
    for list in terminals_of_equipment.values_mut() {
        list.sort();
    }

    let mut topology = crate::topology::NodeBreakerTopology::new(node_mrids.len());
    let mut switch_mrids: Vec<String> = Vec::new();

    let add = |topology: &mut crate::topology::NodeBreakerTopology,
                   switch_mrids: &mut Vec<String>,
                   mrid: &String,
                   kind: SwitchKind,
                   in_service: bool,
                   open: bool| {
        let Some(terms) = terminals_of_equipment.get(mrid.as_str()) else { return };
        if terms.len() < 2 {
            return;
        }
        let (Some(&a), Some(&b)) = (
            node_of_terminal.get(terms[0].1),
            node_of_terminal.get(terms[1].1),
        ) else {
            return;
        };
        topology.add_switch(TopoSwitch {
            kind,
            nodes: [NodeIdx(a), NodeIdx(b)],
            open,
            in_service,
        });
        switch_mrids.push(mrid.clone());
    };

    for mrid in by_type(ds, "Switch") {
        let sw: &cimstructs::Switch = require(ds, mrid, "Switch", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::Generic, equipment_in_service(sw.base.base.in_service, sw.base.base.normally_in_service), sw.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Breaker") {
        let br: &cimstructs::Breaker = require(ds, mrid, "Breaker", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::Breaker, equipment_in_service(br.base.base.base.base.in_service, br.base.base.base.base.normally_in_service), br.base.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "LoadBreakSwitch") {
        let lbs: &cimstructs::LoadBreakSwitch = require(ds, mrid, "LoadBreakSwitch", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::LoadBreakSwitch, equipment_in_service(lbs.base.base.base.base.in_service, lbs.base.base.base.base.normally_in_service), lbs.base.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "DisconnectingCircuitBreaker") {
        let dcb: &cimstructs::DisconnectingCircuitBreaker = require(ds, mrid, "DisconnectingCircuitBreaker", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::DisconnectingCircuitBreaker, equipment_in_service(dcb.base.base.base.base.base.in_service, dcb.base.base.base.base.base.normally_in_service), dcb.base.base.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Disconnector") {
        let d: &cimstructs::Disconnector = require(ds, mrid, "Disconnector", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::Disconnector, equipment_in_service(d.base.base.base.in_service, d.base.base.base.normally_in_service), d.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "GroundDisconnector") {
        let g: &cimstructs::GroundDisconnector = require(ds, mrid, "GroundDisconnector", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::GroundDisconnector, equipment_in_service(g.base.base.base.in_service, g.base.base.base.normally_in_service), g.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Jumper") {
        let j: &cimstructs::Jumper = require(ds, mrid, "Jumper", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::Jumper, equipment_in_service(j.base.base.base.in_service, j.base.base.base.normally_in_service), j.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Cut") {
        let c: &cimstructs::Cut = require(ds, mrid, "Cut", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::Cut, equipment_in_service(c.base.base.base.in_service, c.base.base.base.normally_in_service), c.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Fuse") {
        let f: &cimstructs::Fuse = require(ds, mrid, "Fuse", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::Fuse, equipment_in_service(f.base.base.base.in_service, f.base.base.base.normally_in_service), f.base.open.unwrap_or(false));
    }
    for mrid in by_type(ds, "Junction") {
        let j: &cimstructs::Junction = require(ds, mrid, "Junction", mrid, "(self)")?;
        add(&mut topology, &mut switch_mrids, mrid, SwitchKind::Junction, equipment_in_service(j.base.base.base.in_service, j.base.base.base.normally_in_service), false);
    }

    // Busbars, for `RetentionPolicy::RetainAdjacentToBusbar`. A busbar section
    // is single-terminal equipment, so it contributes the node its one terminal
    // sits on.
    for mrid in by_type(ds, "BusbarSection") {
        let Some(terms) = terminals_of_equipment.get(mrid.as_str()) else { continue };
        for &(_, t) in terms {
            if let Some(&node) = node_of_terminal.get(t) {
                topology.busbars.push(NodeIdx(node));
            }
        }
    }
    topology.busbars.sort_unstable();
    topology.busbars.dedup();

    Ok(CgmesNodeBreaker { topology, node_mrids, switch_mrids, tn_of_node })
}

/// A `ConnectivityNode`'s nominal voltage in volts, resolved through its
/// container.
///
/// `ConnectivityNode` has no `BaseVoltage` of its own — a `TopologicalNode`
/// does, which is why the bus-branch path never needed this. The chain is
/// `ConnectivityNodeContainer`, which is a `VoltageLevel` directly or a `Bay`
/// that names one.
///
/// `None` where the container is neither — most often a CGMES `Line`
/// container holding a boundary node, which genuinely has no voltage level.
/// Measured on the four node-breaker configurations: 0 nodes with no container
/// at all, and 2/5/3/0 whose container is something else.
fn connectivity_node_voltage(ds: &CimDataset, cn_mrid: &str) -> Option<f64> {
    let cn: &cimstructs::ConnectivityNode = get(ds, cn_mrid)?;
    let container = cn.connectivity_node_container.as_ref()?;

    let vl_mrid = match get::<cimstructs::VoltageLevel>(ds, &container.mrid) {
        Some(_) => container.mrid.clone(),
        None => {
            let bay: &cimstructs::Bay = get(ds, &container.mrid)?;
            bay.voltage_level.as_ref()?.mrid.clone()
        }
    };
    let vl: &cimstructs::VoltageLevel = get(ds, &vl_mrid)?;
    let bv: &BaseVoltage = get(ds, &vl.base_voltage.as_ref()?.mrid)?;
    // CGMES gives nominalVoltage in kV; `Bus::u_rated` is documented in V.
    Some(bv.nominal_voltage? * 1e3)
}

/// Builds the bus skeleton from a node-breaker bus view rather than from
/// `TopologicalNode`s.
///
/// Produces exactly what [`build_ac_bus_skeleton`] does — buses, a
/// `TopologicalNode`-keyed index, and a terminal resolver — so every equipment
/// loop in [`convert_equipment`] works against it unchanged. The difference is
/// only in what a bus *is*: a group of connectivity nodes under `policy`,
/// rather than a topological node.
///
/// `idx_of` is still keyed by `TopologicalNode` mrid, populated through
/// `CgmesNodeBreaker::tn_of_node`, so the downstream code that resolves an
/// angle reference or an HVDC converter by TN keeps working. It is empty when
/// TP was not loaded, which those paths already tolerate.
fn build_node_breaker_skeleton(
    ds: &CimDataset,
    policy: &crate::topology::RetentionPolicy,
) -> Result<
    (Vec<Bus>, HashMap<String, usize>, TerminalIndex, CgmesNodeBreaker, crate::topology::BusView),
    CgmesError,
> {
    use crate::topology::model::{BusIdx, NodeIdx};

    let nb = cgmes_node_breaker_topology(ds)?;
    if nb.topology.n_nodes == 0 {
        return Err(CgmesError::NoTopologicalNodes);
    }
    let view = crate::topology::bus_view(&nb.topology, policy);

    // One bus per view bus. `u_rated` comes from the first member node that
    // resolves one; a bus whose members are all boundary nodes keeps 0.0,
    // which `Bus::u_rated` documents as "not set" — the same thing the
    // bus-branch path produces for a synthesized boundary bus.
    let mut buses: Vec<Bus> = (0..view.n_buses())
        .map(|b| {
            let u_rated = view
                .nodes_of(BusIdx(b))
                .iter()
                .find_map(|n| connectivity_node_voltage(ds, &nb.node_mrids[n.0]))
                .unwrap_or(0.0);
            Bus {
                idx: b,
                bus_type: BusType::PQ,
                voltage_mag: 1.0,
                voltage_ang: 0.0,
                p_spec: 0.0,
                q_spec: 0.0,
                q_min: -f64::INFINITY,
                q_max: f64::INFINITY,
                u_rated,
                zip_terms: Vec::new(),
            }
        })
        .collect();

    let mut node_of_cn: HashMap<&str, usize> = HashMap::with_capacity(nb.node_mrids.len());
    for (i, mrid) in nb.node_mrids.iter().enumerate() {
        node_of_cn.insert(mrid.as_str(), i);
    }

    // The terminal resolver, built the same way as the bus-branch one except
    // that a terminal resolves through its `ConnectivityNode` rather than its
    // `TopologicalNode`.
    let mut raw: HashMap<String, Vec<(i64, String)>> = HashMap::new();
    let mut bus_of: HashMap<String, usize> = HashMap::new();
    let mut connected_of: HashMap<String, bool> = HashMap::new();
    let mut unresolved: Vec<String> = Vec::new();
    for t_mrid in by_type(ds, "Terminal") {
        let t: &Terminal = require(ds, t_mrid, "Terminal", t_mrid, "(self)")?;
        connected_of.insert(t_mrid.clone(), t.base.connected.unwrap_or(true));
        if let Some(ce) = &t.conducting_equipment {
            let seq = t.base.sequence_number.unwrap_or(1);
            raw.entry(ce.mrid.clone()).or_default().push((seq, t_mrid.clone()));
        }
        match t.connectivity_node.as_ref().and_then(|cn| node_of_cn.get(cn.mrid.as_str())) {
            Some(&node) => {
                bus_of.insert(t_mrid.clone(), view.bus_of(NodeIdx(node)).0);
            }
            None => unresolved.push(t_mrid.clone()),
        }
    }

    // A terminal with no ConnectivityNode gets a bus of its own, mirroring the
    // bus-branch path's boundary-node synthesis. It has no voltage level to
    // read, so `u_rated` stays unset.
    for t_mrid in unresolved {
        let idx = buses.len();
        buses.push(Bus {
            idx,
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
        bus_of.insert(t_mrid, idx);
    }

    let by_equipment = raw
        .into_iter()
        .map(|(eq, mut v)| {
            v.sort_by_key(|(seq, _)| *seq);
            (eq, v.into_iter().map(|(_, m)| m).collect())
        })
        .collect();

    // TopologicalNode -> bus, for the angle-reference and HVDC paths.
    let mut idx_of: HashMap<String, usize> = HashMap::new();
    for (node, tn) in nb.tn_of_node.iter().enumerate() {
        if let Some(tn) = tn {
            idx_of.insert(tn.clone(), view.bus_of(NodeIdx(node)).0);
        }
    }

    Ok((buses, idx_of, TerminalIndex { by_equipment, bus_of, connected_of }, nb, view))
}

/// Converts a CGMES dataset into a **node-breaker** network: buses from
/// connectivity nodes under `policy`, plus the branches that give retained
/// switches identity under `treatment`.
///
/// The returned transformer list is the model's own transformers followed by
/// one branch per non-degenerate retained switch, so a switch's flat branch
/// index is `lines.len() + model_transformers + n` for the *n*th entry of
/// [`BusView::retained`](crate::topology::BusView::retained) that survived. The
/// returned [`BusView`](crate::topology::BusView) is what maps those back.
///
/// With [`RetentionPolicy::MergeAll`](crate::topology::RetentionPolicy::MergeAll)
/// this is an ordinary bus-branch conversion that happens to have derived its
/// buses from EQ rather than TP — useful on a dataset with no TP profile at
/// all, which [`cgmes_to_buses_and_branches`] cannot read.
pub fn cgmes_node_breaker_to_buses_and_branches(
    ds: &CimDataset,
    s_base_va: f64,
    policy: &crate::topology::RetentionPolicy,
    treatment: crate::switches::SwitchTreatment,
) -> Result<crate::switches::NodeBreakerNetwork, CgmesError> {
    let (buses, idx_of, terms, nb, view) = build_node_breaker_skeleton(ds, policy)?;
    let mut zero_injection = zero_injection_flags(ds, &terms, buses.len());
    let converted = convert_equipment(ds, s_base_va, (buses, idx_of, terms))?;
    // The node-breaker path appends one branch per retained switch after the
    // model's own transformers, so the tap tables — which are parallel to the
    // model's transformers only — are dropped here rather than handed on
    // misaligned. Tap control over a node-breaker view is not wired up.
    let CgmesNetwork { buses, lines, transformers, shunts, .. } = converted;
    // Conversion appends buses of its own — a three-winding transformer's star
    // point, chiefly. Nothing terminates on those by construction, which is
    // what `true` says: they are the textbook zero-injection bus.
    zero_injection.resize(buses.len(), true);

    if treatment == crate::switches::SwitchTreatment::Merge && !view.retained().is_empty() {
        return Err(CgmesError::UnsupportedTransformer {
            mrid: "(retention policy)".to_string(),
            reason: "SwitchTreatment::Merge cannot represent a retained switch; use \
                     RetentionPolicy::MergeAll or SwitchTreatment::Regularize"
                .to_string(),
        });
    }

    Ok(crate::switches::NodeBreakerNetwork::new(
        buses,
        lines,
        transformers,
        shunts,
        view,
        nb.topology,
        nb.switch_mrids,
        zero_injection,
        treatment,
    ))
}

// ---------------------------------------------------------------------------
// Operational limits
// ---------------------------------------------------------------------------

/// The operating limits CGMES declares for one piece of equipment, indexed by
/// terminal.
///
/// `terminals[i]` corresponds to the same `i` that [`TerminalIndex::bus`] uses
/// — 0 is `sequenceNumber` 1, the branch's starting point — so a caller that
/// already knows which end of a branch it is looking at can index straight in.
/// Equipment whose limits are attached to the *equipment* rather than to a
/// terminal (CGMES permits both) has them repeated on every terminal, which is
/// the honest reading of a limit that names no side.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EquipmentLimits {
    pub terminals: Vec<crate::ratings::BranchLimits>,
}

impl EquipmentLimits {
    /// The limits of terminal `which`, or an empty set if that terminal has
    /// none.
    pub fn terminal(&self, which: usize) -> crate::ratings::BranchLimits {
        self.terminals.get(which).cloned().unwrap_or_default()
    }

    /// The tightest permanent rating over every terminal — the single number to
    /// use when a caller has one branch and no side in mind.
    pub fn tightest_patl(&self) -> Option<f64> {
        self.terminals.iter().filter_map(|t| t.patl_a).fold(None, |acc, v| {
            Some(match acc {
                Some(a) => f64::min(a, v),
                None => v,
            })
        })
    }
}

/// Read every `OperationalLimit` in the dataset, keyed by the mRID of the
/// equipment it applies to.
///
/// **This is the only route by which a CGMES network acquires ratings.** Until
/// this existed, `cgmes_to_buses_and_branches` produced `Line`s and
/// `Transformer`s with impedances and nothing else, so the question "is this
/// network secure" could not be asked of a CGMES input at all — even though
/// every conformity fixture in the tree carries the answer (768 `CurrentLimit`
/// objects in SmallGrid, 33,262 in RealGrid).
///
/// # What is read, and what is skipped
///
/// `CurrentLimit` only. `ActivePowerLimit`, `ApparentPowerLimit` and
/// `VoltageLimit` are recognised and **counted** rather than converted:
/// [`ratings::BranchLimits`](crate::ratings::BranchLimits) is expressed in
/// amperes, and silently folding an MVA limit into an ampere field would need a
/// voltage this function does not have. They are reported through the returned
/// `skipped` count so their absence is visible rather than assumed.
///
/// A limit is classified permanent or temporary from its
/// `OperationalLimitType`: `isInfiniteDuration`, or a `kind` of `patl`, makes it
/// the PATL; anything with an `acceptableDuration` becomes a
/// [`TemporaryLimit`](crate::ratings::TemporaryLimit). A type that says neither
/// is treated as permanent, since an unqualified limit that always applies is
/// what a bare `OperationalLimit` means.
///
/// `value` is preferred over `normalValue`, falling back to it — the conformity
/// fixtures populate `normalValue` and leave `value` absent, so a reader that
/// only looked at `value` would find every limit empty and report success.
pub fn cgmes_operational_limits(
    ds: &CimDataset,
) -> Result<(HashMap<String, EquipmentLimits>, LimitImportReport), CgmesError> {
    use crate::ratings::{BranchLimits, TemporaryLimit};

    // Terminal mRID -> (equipment mRID, 0-based position after sequence sort).
    let mut position_of: HashMap<String, (String, usize)> = HashMap::new();
    let mut by_equipment: HashMap<String, Vec<(i64, String)>> = HashMap::new();
    for mrid in by_type(ds, "Terminal") {
        let Some(t) = get::<Terminal>(ds, mrid) else { continue };
        let Some(ce) = t.conducting_equipment.as_ref() else { continue };
        by_equipment
            .entry(ce.mrid.clone())
            .or_default()
            .push((t.base.sequence_number.unwrap_or(1), mrid.clone()));
    }
    let mut terminal_count: HashMap<String, usize> = HashMap::new();
    for (equipment, mut terms) in by_equipment {
        terms.sort_by_key(|(seq, _)| *seq);
        terminal_count.insert(equipment.clone(), terms.len());
        for (i, (_, t_mrid)) in terms.into_iter().enumerate() {
            position_of.insert(t_mrid, (equipment.clone(), i));
        }
    }

    // Limit-set mRID -> which terminals it covers. A set attached to the
    // equipment rather than a terminal covers all of them.
    let mut set_targets: HashMap<String, (String, Vec<usize>)> = HashMap::new();
    for mrid in by_type(ds, "OperationalLimitSet") {
        let Some(set) = get::<OperationalLimitSet>(ds, mrid) else { continue };
        if let Some(t) = set.terminal.as_ref() {
            if let Some((equipment, which)) = position_of.get(&t.mrid) {
                set_targets.insert(mrid.clone(), (equipment.clone(), vec![*which]));
                continue;
            }
        }
        if let Some(e) = set.equipment.as_ref() {
            let n = terminal_count.get(&e.mrid).copied().unwrap_or(0);
            if n > 0 {
                set_targets.insert(mrid.clone(), (e.mrid.clone(), (0..n).collect()));
            }
        }
    }

    let mut limits: HashMap<String, EquipmentLimits> = HashMap::new();
    let mut report = LimitImportReport::default();

    for mrid in by_type(ds, "CurrentLimit") {
        let Some(limit) = get::<CurrentLimit>(ds, mrid) else { continue };
        let Some(value) = limit.value.or(limit.normal_value) else {
            report.without_value += 1;
            continue;
        };
        let Some(set_ref) = limit.base.operational_limit_set.as_ref() else {
            report.unattached += 1;
            continue;
        };
        let Some((equipment, which)) = set_targets.get(&set_ref.mrid) else {
            report.unattached += 1;
            continue;
        };

        let (duration, permanent) = match limit.base.operational_limit_type.as_ref() {
            Some(t) => match get::<OperationalLimitType>(ds, &t.mrid) {
                Some(kind) => {
                    let patl = kind.is_infinite_duration.unwrap_or(false)
                        || kind
                            .kind
                            .as_ref()
                            .is_some_and(|k| k.mrid.rsplit('.').next() == Some("patl"));
                    (kind.acceptable_duration, patl || kind.acceptable_duration.is_none())
                }
                None => (None, true),
            },
            None => (None, true),
        };

        let entry = limits.entry(equipment.clone()).or_default();
        let needed = which.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        if entry.terminals.len() < needed {
            entry.terminals.resize(needed, BranchLimits::default());
        }
        for &i in which {
            let slot = &mut entry.terminals[i];
            if permanent {
                // Several PATLs on one terminal is malformed but does occur;
                // honouring all of them means keeping the tightest.
                slot.patl_a = Some(match slot.patl_a {
                    Some(existing) => existing.min(value),
                    None => value,
                });
            } else {
                slot.tatl.push(TemporaryLimit {
                    acceptable_duration_s: duration,
                    value_a: value,
                });
            }
        }
        report.current_limits += 1;
    }

    for name in ["ActivePowerLimit", "ApparentPowerLimit", "VoltageLimit"] {
        report.skipped += by_type(ds, name).len();
    }

    Ok((limits, report))
}

/// What [`cgmes_operational_limits`] did and did not convert.
///
/// Returned rather than logged because a rating that quietly failed to import
/// reads downstream as "unlimited", which is the most dangerous possible
/// default: a security analysis on a network with no limits reports everything
/// secure.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LimitImportReport {
    /// `CurrentLimit` objects converted.
    pub current_limits: usize,
    /// Limits whose `OperationalLimitSet` named no terminal or equipment this
    /// dataset defines.
    pub unattached: usize,
    /// Limits carrying neither `value` nor `normalValue`.
    pub without_value: usize,
    /// Power and voltage limits, recognised but not expressible in amperes.
    pub skipped: usize,
}

#[cfg(test)]
mod voltage_control_tests {
    use super::*;

    fn bus(idx: usize) -> Bus {
        Bus {
            idx,
            bus_type: BusType::PQ,
            voltage_mag: 1.0,
            voltage_ang: 0.0,
            p_spec: 0.0,
            q_spec: 0.0,
            // What `build_ac_bus_skeleton` starts every bus at: no limit, as
            // opposed to no capability. It is why the first contribution has
            // to assign rather than add.
            q_min: -f64::INFINITY,
            q_max: f64::INFINITY,
            u_rated: 400e3,
            zip_terms: Vec::new(),
        }
    }

    /// Two machines on one bus contribute both their capabilities, which is
    /// what the injections beside them have always done. Before this, the
    /// second `=` threw the first machine's limits away.
    #[test]
    fn two_machines_on_one_bus_sum_their_capability() {
        let mut buses = vec![bus(0), bus(1)];
        let mut vc = VoltageControl::new(2);
        vc.regulate(&mut buses, 1, 1.02, -2.0, 3.0, "a");
        assert_eq!(buses[1].bus_type, BusType::PV);
        assert_eq!((buses[1].q_min, buses[1].q_max), (-2.0, 3.0), "the first assigns");

        vc.regulate(&mut buses, 1, 1.02, -1.5, 4.0, "b");
        assert_eq!((buses[1].q_min, buses[1].q_max), (-3.5, 7.0), "the second adds");

        let report = vc.finish();
        assert_eq!(report.regulated_buses, 1);
        assert_eq!(report.shared_buses, 1);
        assert!(report.target_conflicts.is_empty());
    }

    /// One machine with no stated limit makes the bus's joint capability
    /// unlimited, which is the right reading of an absent `minQ`/`maxQ` — and
    /// the reason the accumulation must not treat ±∞ as a number to be
    /// replaced.
    #[test]
    fn an_unlimited_machine_makes_the_joint_capability_unlimited() {
        let mut buses = vec![bus(0)];
        let mut vc = VoltageControl::new(1);
        vc.regulate(&mut buses, 0, 1.0, -2.0, 3.0, "a");
        vc.regulate(&mut buses, 0, 1.0, -f64::INFINITY, f64::INFINITY, "b");
        assert!(buses[0].q_min.is_infinite() && buses[0].q_min < 0.0);
        assert!(buses[0].q_max.is_infinite() && buses[0].q_max > 0.0);
        assert!(!buses[0].q_min.is_nan() && !buses[0].q_max.is_nan());
    }

    /// A single machine is unaffected: it assigns, exactly as it always did.
    /// This is the case every existing fixture test depends on.
    #[test]
    fn one_machine_is_left_exactly_as_it_was() {
        let mut buses = vec![bus(0)];
        let mut vc = VoltageControl::new(1);
        vc.regulate(&mut buses, 0, 1.05, -1.0, 1.0, "only");
        assert_eq!((buses[0].q_min, buses[0].q_max), (-1.0, 1.0));
        assert_eq!(buses[0].voltage_mag, 1.05);
        let report = vc.finish();
        assert_eq!(report.shared_buses, 0);
    }

    /// A target two controllers disagree on is **reported, not resolved**. The
    /// last writer still wins, which is what it always did — no vendored
    /// fixture has a disagreement, so any resolution rule would be untested,
    /// and choosing one here would make this a behaviour change rather than
    /// the limits-only fix it is.
    #[test]
    fn a_disagreeing_target_is_reported_and_the_last_still_wins() {
        let mut buses = vec![bus(0)];
        let mut vc = VoltageControl::new(1);
        vc.regulate(&mut buses, 0, 1.02, -1.0, 1.0, "first");
        vc.regulate(&mut buses, 0, 1.06, -1.0, 1.0, "second");

        assert_eq!(buses[0].voltage_mag, 1.06, "last writer still wins");
        let report = vc.finish();
        assert_eq!(report.target_conflicts.len(), 1);
        let c = &report.target_conflicts[0];
        assert_eq!((c.bus, c.existing, c.proposed, c.id.as_str()), (0, 1.02, 1.06, "second"));
        // And the limits summed regardless — a disagreement about the target
        // says nothing about the machines' capability.
        assert_eq!((buses[0].q_min, buses[0].q_max), (-2.0, 2.0));
    }

    /// Agreement within floating-point noise is agreement. Two exports of one
    /// set-point routinely differ in the last digit.
    #[test]
    fn a_negligible_difference_is_not_a_conflict() {
        let mut buses = vec![bus(0)];
        let mut vc = VoltageControl::new(1);
        vc.regulate(&mut buses, 0, 1.02, -1.0, 1.0, "first");
        vc.regulate(&mut buses, 0, 1.02 + 1e-12, -1.0, 1.0, "second");
        assert!(vc.finish().target_conflicts.is_empty());
    }
}
