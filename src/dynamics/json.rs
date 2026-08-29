//! gridoxide's own JSON format, extended with a `dynamics` section.
//!
//! This is the **internal representation every other reader targets**. The
//! PSS/E and Dynawo readers produce these types; nothing downstream of here
//! knows which format a case arrived in.
//!
//! ```json
//! {
//!   "buses": [ … ], "lines": [ … ],
//!   "dynamics": {
//!     "s_base": 100.0, "f_nom": 50.0,
//!     "fixed_buses": [1],
//!     "units": [
//!       { "id": "G1", "bus": 0,
//!         "machine": { "model": "gen_transient", "h": 5.0, "d": 0.0, "ra": 0.003,
//!                      "xd": 1.8, "xq": 1.7, "xdp": 0.3, "xqp": 0.55,
//!                      "td0p": 8.0, "tq0p": 0.4, "mbase": 100.0 },
//!         "avr": { "model": "sexs", "k": 200.0, "ta": 0.1, "tb": 1.0, "te": 0.05 },
//!         "gov": { "model": "tgov1", "r": 0.05, "t1": 0.5, "t2": 1.0, "t3": 5.0, "dt": 0.0 } }
//!     ],
//!     "events": [
//!       { "kind": "bus_fault", "t": 1.0, "bus": 0 },
//!       { "kind": "clear_fault", "t": 1.1, "bus": 0 }
//!     ]
//!   }
//! }
//! ```
//!
//! The model parameter blocks *are* the parameter structs — [`GenTransientParams`]
//! and friends derive `Deserialize` — so the file format cannot drift away from
//! what the models actually take. A renamed field is a compile error rather
//! than a silently-defaulted zero.
//!
//! # The device/load split, and why it is derived here
//!
//! `plans/RMS_PLAN.md` §11 records the one hazard the equilibrium gate cannot
//! catch: if a caller states a machine's terminal power wrongly, the result is
//! *self-consistent* — the machine initializes to an equilibrium at the power
//! it was told, the remainder is absorbed into the bus's admittance, and what
//! comes out is a different machine swinging plausibly.
//!
//! A file removes that hazard for the ordinary case, because the file says
//! where everything is. A device may state its own `p`/`q`, and if it does not,
//! it takes whatever the bus's solved injection has left after the devices that
//! did. One device omitting it at a bus is the common case and is unambiguous.
//! **Two** devices omitting it at the same bus is refused, by name, rather than
//! split by a guess — see [`DynamicsError::AmbiguousSplit`].

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex;
use serde::{Deserialize, Serialize};

use crate::json::NetworkData;
use crate::network::{build_ybus, power_injections};
use crate::solver::SolveStatus;
use crate::types::Bus;

use super::events::{Event, EventKind, Relay};
use super::init::{build, BuildError, DeviceSpec, SystemSpec};
use super::models::avr::{Sexs, SexsParams, VrProportional};
use super::models::gov::{GoverProportional, Tgov1, Tgov1Params};
use super::models::load::ZipLoad;
use super::models::machine::{
    GenCls, GenClsParams, GenRound, GenRoundParams, GenSalient, GenSalientParams, GenTransient,
    GenTransientParams, Machine,
};
use super::models::pss::{Stab1, Stab1Params};
use super::models::{Control, DynamicModel, GeneratingUnit, InitError, Limits};
use super::DynamicSystem;

/// A network document that also carries dynamic data.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DynamicsDocument {
    /// `buses` and `lines`, exactly as an ordinary network document states
    /// them — flattened, so one file serves both.
    #[serde(flatten)]
    pub network: NetworkData,
    pub dynamics: DynamicsData,
}

fn default_s_base() -> f64 {
    100.0
}

fn default_f_nom() -> f64 {
    50.0
}

fn default_cutoff() -> f64 {
    0.5
}

/// The `dynamics` section.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DynamicsData {
    /// The base the network's per-unit values are on, in the same unit as each
    /// machine's `mbase`. Only their *ratio* matters, so any consistent unit
    /// works.
    #[serde(default = "default_s_base")]
    pub s_base: f64,
    /// Nominal frequency, Hz. The one place a physical time unit enters.
    #[serde(default = "default_f_nom")]
    pub f_nom: f64,
    /// Carry the rotor speed on the machines' speed-voltage terms, and write
    /// their swing equations in torque rather than power.
    ///
    /// Off by default, because the `ω ≈ 1` approximation is what makes the
    /// phasor formulation coherent and is what every closed-form gate in this
    /// crate is derived from. On, gridoxide matches Dynawo's and Sauer & Pai's
    /// form instead — worth 0.6% of terminal power at a 0.9% speed deviation.
    /// See `src/dynamics/models/machine.rs`.
    #[serde(default)]
    pub speed_voltages: bool,
    #[serde(default)]
    pub units: Vec<UnitSpec>,
    #[serde(default)]
    pub loads: Vec<LoadSpec>,
    /// Buses held at constant voltage for the whole run. See
    /// [`SystemSpec::fixed_buses`].
    #[serde(default)]
    pub fixed_buses: Vec<usize>,
    #[serde(default)]
    pub events: Vec<EventSpec>,
    /// Protection relays. Unlike an event these carry no time: when they act is
    /// found, not stated. See [`Relay`](super::events::Relay).
    #[serde(default)]
    pub relays: Vec<super::events::Relay>,
}

/// One generating unit: a machine, and whichever controls it carries.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UnitSpec {
    pub id: String,
    pub bus: usize,
    /// Terminal injection at the operating point. Omit both to take whatever
    /// the bus's solved injection has left.
    #[serde(default)]
    pub p: Option<f64>,
    #[serde(default)]
    pub q: Option<f64>,
    pub machine: MachineSpec,
    #[serde(default)]
    pub avr: Option<AvrSpec>,
    #[serde(default)]
    pub gov: Option<GovSpec>,
    #[serde(default)]
    pub pss: Option<PssSpec>,
}

/// A voltage-dependent load.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoadSpec {
    pub id: String,
    pub bus: usize,
    #[serde(default)]
    pub p: Option<f64>,
    #[serde(default)]
    pub q: Option<f64>,
    /// Constant-impedance, constant-current and constant-power fractions, in
    /// that order. Must sum to one.
    pub zip: [f64; 3],
    #[serde(default = "default_cutoff")]
    pub cutoff: f64,
}

/// The machine models, tagged by `"model"`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum MachineSpec {
    GenCls(GenClsParams),
    GenTransient(GenTransientParams),
    GenRound(GenRoundParams),
    /// Fifth order: a salient-pole machine, with one `q`-axis damper.
    GenSalient(GenSalientParams),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum AvrSpec {
    Sexs(SexsParams),
    /// A pure gain, with no dynamics — what Dynawo's `VRProportional` is.
    VrProportional {
        k: f64,
        #[serde(default)]
        limits: Limits,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum GovSpec {
    Tgov1(Tgov1Params),
    /// A pure gain, `K = 1/R`, on the **network** base — what Dynawo's
    /// `GoverProportional` is.
    GoverProportional {
        k: f64,
        #[serde(default)]
        limits: Limits,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum PssSpec {
    Stab1(Stab1Params),
}

/// A scheduled disturbance, tagged by `"kind"`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventSpec {
    /// A shunt fault. `y` is `[g, b]` per unit; omitted, it is a bolted fault.
    BusFault {
        t: f64,
        bus: usize,
        #[serde(default)]
        y: Option<[f64; 2]>,
    },
    ClearFault {
        t: f64,
        bus: usize,
    },
    BranchTrip {
        t: f64,
        branch: usize,
    },
    BranchClose {
        t: f64,
        branch: usize,
    },
    /// `unit` indexes `dynamics.units`, in the order the file lists them.
    UnitTrip {
        t: f64,
        unit: usize,
    },
    UnitClose {
        t: f64,
        unit: usize,
    },
    /// A step in a bus's load, stated as a change in **injection** — so a load
    /// increase is negative.
    LoadStep {
        t: f64,
        bus: usize,
        dp: f64,
        #[serde(default)]
        dq: f64,
    },
}

impl EventSpec {
    pub fn time(&self) -> f64 {
        match *self {
            EventSpec::BusFault { t, .. }
            | EventSpec::ClearFault { t, .. }
            | EventSpec::BranchTrip { t, .. }
            | EventSpec::BranchClose { t, .. }
            | EventSpec::UnitTrip { t, .. }
            | EventSpec::UnitClose { t, .. }
            | EventSpec::LoadStep { t, .. } => t,
        }
    }
}

impl From<EventSpec> for Event {
    fn from(spec: EventSpec) -> Self {
        let t = spec.time();
        let kind = match spec {
            EventSpec::BusFault { bus, y, .. } => match y {
                Some([g, b]) => EventKind::BusFault { bus, y: Complex::new(g, b) },
                None => return Event::bolted_fault(t, bus),
            },
            EventSpec::ClearFault { bus, .. } => EventKind::ClearFault { bus },
            EventSpec::BranchTrip { branch, .. } => EventKind::BranchTrip { branch },
            EventSpec::BranchClose { branch, .. } => EventKind::BranchClose { branch },
            EventSpec::UnitTrip { unit, .. } => EventKind::UnitTrip { unit },
            EventSpec::UnitClose { unit, .. } => EventKind::UnitClose { unit },
            EventSpec::LoadStep { bus, dp, dq, .. } => {
                EventKind::LoadStep { bus, ds: Complex::new(dp, dq) }
            }
        };
        Event::new(t, kind)
    }
}

/// Why a document could not be turned into a system.
#[derive(Clone, Debug, PartialEq)]
pub enum DynamicsError {
    Io(String),
    Parse(String),
    /// The base-case power flow did not converge, so there is no operating
    /// point to initialize from. Reported rather than pressed on with: every
    /// device's state is derived from the solved voltages, so an unconverged
    /// solve produces a plausible-looking equilibrium of the wrong network.
    PowerFlow(SolveStatus),
    BusOutOfRange { id: String, bus: usize, n_bus: usize },
    /// Two or more devices at one bus each left their terminal power unstated,
    /// so the bus's injection could be split between them in any proportion.
    /// Refused by name rather than guessed — see the module doc.
    AmbiguousSplit { bus: usize, devices: Vec<String> },
    Model { id: String, source: InitError },
    Build(BuildError),
}

impl std::fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynamicsError::Io(message) => write!(f, "{message}"),
            DynamicsError::Parse(message) => write!(f, "{message}"),
            DynamicsError::PowerFlow(status) => {
                write!(f, "the base-case power flow did not converge ({status:?})")
            }
            DynamicsError::BusOutOfRange { id, bus, n_bus } => {
                write!(f, "{id} is at bus {bus}, but the network has {n_bus} buses")
            }
            DynamicsError::AmbiguousSplit { bus, devices } => write!(
                f,
                "bus {bus} has {} devices with no stated terminal power ({}); \
                 state p and q on all but one",
                devices.len(),
                devices.join(", ")
            ),
            DynamicsError::Model { id, source } => write!(f, "{id}: {source}"),
            DynamicsError::Build(source) => write!(f, "{source}"),
        }
    }
}

impl std::error::Error for DynamicsError {}

impl From<BuildError> for DynamicsError {
    fn from(source: BuildError) -> Self {
        DynamicsError::Build(source)
    }
}

/// Reads a document from a file.
pub fn read(path: impl AsRef<Path>) -> Result<DynamicsDocument, DynamicsError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|e| DynamicsError::Io(format!("{}: {e}", path.display())))?;
    parse(&text)
}

/// Parses a document from text.
pub fn parse(text: &str) -> Result<DynamicsDocument, DynamicsError> {
    serde_json::from_str(text).map_err(|e| DynamicsError::Parse(e.to_string()))
}

impl MachineSpec {
    fn build(
        self,
        s_base: f64,
        f_nom: f64,
        speed_voltages: bool,
    ) -> Result<Box<dyn Machine>, InitError> {
        Ok(match self {
            MachineSpec::GenCls(p) => {
                Box::new(GenCls::new(p, s_base, f_nom)?.with_speed_voltages(speed_voltages))
            }
            MachineSpec::GenTransient(p) => {
                Box::new(GenTransient::new(p, s_base, f_nom)?.with_speed_voltages(speed_voltages))
            }
            MachineSpec::GenRound(p) => {
                Box::new(GenRound::new(p, s_base, f_nom)?.with_speed_voltages(speed_voltages))
            }
            MachineSpec::GenSalient(p) => {
                Box::new(GenSalient::new(p, s_base, f_nom)?.with_speed_voltages(speed_voltages))
            }
        })
    }
}

impl AvrSpec {
    fn build(self) -> Result<Box<dyn Control>, InitError> {
        match self {
            AvrSpec::Sexs(p) => Ok(Box::new(Sexs::new(p)?)),
            AvrSpec::VrProportional { k, limits } => {
                Ok(Box::new(VrProportional::limited(k, limits)?))
            }
        }
    }
}

impl GovSpec {
    fn build(self) -> Result<Box<dyn Control>, InitError> {
        match self {
            GovSpec::Tgov1(p) => Ok(Box::new(Tgov1::new(p)?)),
            GovSpec::GoverProportional { k, limits } => {
                Ok(Box::new(GoverProportional::limited(k, limits)?))
            }
        }
    }
}

impl PssSpec {
    fn build(self) -> Result<Box<dyn Control>, InitError> {
        match self {
            PssSpec::Stab1(p) => Ok(Box::new(Stab1::new(p)?)),
        }
    }
}

/// A device's place and declared terminal power, before the split is resolved.
struct Placed {
    id: String,
    bus: usize,
    stated: Option<Complex<f64>>,
}

impl UnitSpec {
    /// Builds this unit's model. Exposed so a reader that assembles a case its
    /// own way — the Dynawo one, which gets its network from IIDM — does not
    /// have to reimplement the composition.
    pub fn into_model(
        self,
        s_base: f64,
        f_nom: f64,
    ) -> Result<Box<dyn DynamicModel>, InitError> {
        Ok(Box::new(GeneratingUnit::new(
            self.machine.build(s_base, f_nom, false)?,
            self.avr.map(|a| a.build()).transpose()?,
            self.gov.map(|g| g.build()).transpose()?,
            self.pss.map(|p| p.build()).transpose()?,
        )))
    }
}

impl DynamicsDocument {
    /// Solves the base-case power flow and assembles the initialized system.
    ///
    /// Returns the system together with the event schedule the file states, so
    /// a caller has everything a run needs from one call.
    pub fn build(&self) -> Result<(DynamicSystem, Vec<Event>), DynamicsError> {
        self.build_with_relays().map(|(system, events, _)| (system, events))
    }

    /// As [`build`](Self::build), additionally returning the relays the
    /// document declares.
    pub fn build_with_relays(
        &self,
    ) -> Result<(DynamicSystem, Vec<Event>, Vec<Relay>), DynamicsError> {
        let report = crate::run_power_flow_analysis(self.network.clone());
        if report.stats.status != SolveStatus::Converged {
            return Err(DynamicsError::PowerFlow(report.stats.status));
        }
        let buses = report.buses;
        let n_bus = buses.len();

        let placed: Vec<Placed> = self
            .dynamics
            .units
            .iter()
            .map(|u| Placed {
                id: u.id.clone(),
                bus: u.bus,
                stated: complex_or_none(u.p, u.q),
            })
            .chain(self.dynamics.loads.iter().map(|l| Placed {
                id: l.id.clone(),
                bus: l.bus,
                stated: complex_or_none(l.p, l.q),
            }))
            .collect();

        for device in &placed {
            if device.bus >= n_bus {
                return Err(DynamicsError::BusOutOfRange {
                    id: device.id.clone(),
                    bus: device.bus,
                    n_bus,
                });
            }
        }

        let resolved = resolve_split(&placed, &buses, &self.network)?;

        let (s_base, f_nom) = (self.dynamics.s_base, self.dynamics.f_nom);
        let mut devices = Vec::with_capacity(placed.len());
        for (i, spec) in self.dynamics.units.iter().enumerate() {
            let model = GeneratingUnit::new(
                spec.machine.build(s_base, f_nom, self.dynamics.speed_voltages).map_err(
                    |source| DynamicsError::Model {
                        id: spec.id.clone(),
                        source,
                    },
                )?,
                option_build(spec.avr, |a| a.build(), &spec.id)?,
                option_build(spec.gov, |g| g.build(), &spec.id)?,
                option_build(spec.pss, |p| p.build(), &spec.id)?,
            );
            devices.push(DeviceSpec {
                id: spec.id.clone(),
                bus: spec.bus,
                s: resolved[i],
                model: Box::new(model),
            });
        }
        for (j, spec) in self.dynamics.loads.iter().enumerate() {
            let load = ZipLoad::new(spec.zip[0], spec.zip[1], spec.zip[2], spec.cutoff)
                .map_err(|source| DynamicsError::Model { id: spec.id.clone(), source })?;
            devices.push(DeviceSpec {
                id: spec.id.clone(),
                bus: spec.bus,
                s: resolved[self.dynamics.units.len() + j],
                model: Box::new(load) as Box<dyn DynamicModel>,
            });
        }

        let system = build(SystemSpec {
            buses: &buses,
            lines: &self.network.lines,
            transformers: &[],
            shunts: &[],
            devices,
            fixed_buses: self.dynamics.fixed_buses.clone(),
        })?;

        let events = self.dynamics.events.iter().copied().map(Event::from).collect();
        Ok((system, events, self.dynamics.relays.clone()))
    }
}

fn complex_or_none(p: Option<f64>, q: Option<f64>) -> Option<Complex<f64>> {
    match (p, q) {
        (None, None) => None,
        (p, q) => Some(Complex::new(p.unwrap_or(0.0), q.unwrap_or(0.0))),
    }
}

fn option_build<T: Copy, F>(
    spec: Option<T>,
    make: F,
    id: &str,
) -> Result<Option<Box<dyn Control>>, DynamicsError>
where
    F: Fn(T) -> Result<Box<dyn Control>, InitError>,
{
    match spec {
        None => Ok(None),
        Some(s) => make(s)
            .map(Some)
            .map_err(|source| DynamicsError::Model { id: id.to_string(), source }),
    }
}

/// Gives every device a terminal injection, deriving the ones the file left
/// unstated from what its bus has left over.
fn resolve_split(
    placed: &[Placed],
    buses: &[Bus],
    network: &NetworkData,
) -> Result<Vec<Complex<f64>>, DynamicsError> {
    let ybus = build_ybus(buses.len(), &network.lines, &[]).finish();
    let (p_calc, q_calc) = power_injections(buses, &ybus);

    // Per bus: what the solve put there, less everything already spoken for.
    let mut remaining: HashMap<usize, Complex<f64>> = HashMap::new();
    let mut unstated: HashMap<usize, Vec<usize>> = HashMap::new();
    for (k, device) in placed.iter().enumerate() {
        let entry = remaining
            .entry(device.bus)
            .or_insert_with(|| Complex::new(p_calc[device.bus], q_calc[device.bus]));
        match device.stated {
            Some(s) => *entry -= s,
            None => unstated.entry(device.bus).or_default().push(k),
        }
    }

    for (&bus, indices) in &unstated {
        if indices.len() > 1 {
            return Err(DynamicsError::AmbiguousSplit {
                bus,
                devices: indices.iter().map(|&k| placed[k].id.clone()).collect(),
            });
        }
    }

    Ok(placed
        .iter()
        .enumerate()
        .map(|(k, device)| match device.stated {
            Some(s) => s,
            None => {
                let _ = k;
                remaining[&device.bus]
            }
        })
        .collect())
}
