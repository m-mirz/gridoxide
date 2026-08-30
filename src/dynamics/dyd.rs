//! Dynawo `.dyd` / `.par` / `.crv` files.
//!
//! Dynawo is MPL-2.0, RTE-maintained, ships validated cases, and **reads
//! IIDM** — which `src/iidm.rs` already imports. That combination is why
//! `docs/src/reference/feature_comparison.md` named it the route for this whole
//! feature: the corpus and the methodology come close to free.
//!
//! # The three files
//!
//! - **`.dyd`** — the architecture. A list of `blackBoxModel` elements, each
//!   naming a Modelica `lib` to instantiate, a `parId` into a `.par` file, and
//!   optionally a `staticId` tying it to an element of the IIDM network.
//! - **`.par`** — named parameter sets. Keyword-addressed, not positional,
//!   which is what makes this format pleasant to read and unpleasant to guess
//!   at: a misremembered name is a silently absent parameter.
//! - **`.crv`** — which variables to record.
//!
//! Because the addressing is by name, the fixtures under
//! `tests/data/dynamics/dynawo/` are copied **verbatim** from Dynawo's own
//! repository rather than written here. The parameter names this reader looks
//! for are the ones a real case uses, and the gate proves it.
//!
//! # What maps exactly, and what does not
//!
//! `GeneratorSynchronousFourWindings*` is a field winding, one `d`-axis damper
//! and two `q`-axis dampers — six electrical-plus-rotor states, which is
//! exactly [`GenRound`](super::models::GenRound). Every parameter it needs is
//! read straight from the file by name, including `generator_SNom`, which is
//! the machine's MVA base.
//!
//! `...ProportionalRegulations` carries a purely proportional voltage regulator
//! and a purely proportional governor. Those are not approximations of
//! [`Sexs`](super::models::Sexs) and [`Tgov1`](super::models::Tgov1) — they are
//! different devices — so this library grew
//! [`VrProportional`](super::models::VrProportional) and
//! [`GoverProportional`](super::models::GoverProportional) to match them
//! exactly, rather than inventing time constants a Dynawo file never stated.
//!
//! `GeneratorSynchronousThreeWindings*` is a fifth-order salient-pole machine
//! and maps onto [`GenSalient`](super::models::GenSalient). Its parameter set
//! carries no `XpqPu` and no `Tpq0` — a salient rotor has no `q`-axis transient
//! to have a time constant for — so the reader keys on the library name rather
//! than on which parameters happen to be present.
//!
//! # The one conversion that is inferred
//!
//! `governor_KGover` is a gain on the machine's own `governor_PNom` base, while
//! [`GoverProportional`] wants one on the network base, so the reader applies
//! `k = KGover · PNom / s_base`. Every other quantity is read as stated or is
//! converted by a model that documents its own rule. This one rests on a
//! reading of Dynawo's base convention rather than on anything in the file, and
//! it is flagged here so phase 5's comparison knows where to look first if the
//! frequencies disagree.
//!
//! # Limits are carried through
//!
//! `voltageRegulator_EfdMinPu`/`MaxPu` and `governor_PMin`/`PMax` both reach
//! the models. The field-voltage pair goes through unchanged because
//! gridoxide's initialization reproduces Dynawo's `efdPu` exactly — the two
//! agree on what a per-unit field voltage is, which is what makes carrying a
//! ceiling stated in that base sound. The power pair is stated in MW and
//! divides by the network base.
//!
//! # Saturation, again
//!
//! Dynawo states saturation as `generator_md`, `mq`, `nd`, `nq` — an
//! exponential characteristic, and *not* PSS/E's `S(1.0)`/`S(1.2)` pair. Two
//! incompatible representations of the same physics is precisely the
//! divergence `plans/RMS_PLAN.md` §7 predicted, and precisely why §13 declined
//! to pick one before a reference was running. Nonzero values are reported.

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex;
use quick_xml::events::Event as XmlEvent;
use quick_xml::Reader;

use super::init::{build, DeviceSpec, SystemSpec};
use super::json::{AvrSpec, GovSpec, MachineSpec, UnitSpec};
use super::models::machine::{GenRoundParams, GenSalientParams};
use super::models::Limits;

/// One `blackBoxModel` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct BlackBox {
    pub id: String,
    /// The Modelica library to instantiate, e.g.
    /// `GeneratorSynchronousFourWindingsProportionalRegulations`.
    pub lib: String,
    pub par_file: Option<String>,
    pub par_id: Option<String>,
    /// The IIDM element this attaches to, where there is one.
    pub static_id: Option<String>,
}

/// A parsed `.dyd`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DydDocument {
    pub models: Vec<BlackBox>,
}

/// A parsed `.par`: named sets of named values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParDocument {
    pub sets: HashMap<String, HashMap<String, String>>,
}

impl ParDocument {
    /// A parameter as a number, or `None` if absent or not numeric.
    pub fn number(&self, set: &str, name: &str) -> Option<f64> {
        self.sets.get(set)?.get(name)?.parse().ok()
    }
}

/// One requested curve from a `.crv`.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub model: String,
    pub variable: String,
}

/// Why a Dynawo file could not be read.
#[derive(Clone, Debug, PartialEq)]
pub enum DydError {
    Io(String),
    Xml(String),
    /// A model with no `parId`, so its parameters cannot be found.
    NoParameterSet { id: String },
    /// A `parId` naming a set the `.par` does not contain.
    UnknownParameterSet { id: String, par_id: String },
    /// A parameter the model needs, absent from its set. Named, because in a
    /// keyword-addressed format a missing name is the failure mode.
    MissingParameter { id: String, set: String, name: String },
    /// A `staticId` with no corresponding bus index.
    UnknownStaticId { id: String, static_id: String },
    /// The IIDM half of the case could not be read.
    Iidm(String),
    /// The base-case power flow did not converge, so there is no operating
    /// point to initialize from.
    PowerFlow(crate::solver::SolveStatus),
    /// Two dynamic models attached to generators on the same bus. Their share
    /// of that bus's solved reactive power cannot be recovered from the file —
    /// only the total is a fact — so it is refused rather than guessed.
    SharedBus { bus: usize, models: Vec<String> },
    /// The assembled system could not be initialized.
    Build(String),
}

impl std::fmt::Display for DydError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DydError::Io(message) | DydError::Xml(message) => write!(f, "{message}"),
            DydError::NoParameterSet { id } => write!(f, "{id} names no parId"),
            DydError::UnknownParameterSet { id, par_id } => {
                write!(f, "{id} names parameter set `{par_id}`, which the .par does not define")
            }
            DydError::MissingParameter { id, set, name } => {
                write!(f, "{id}: parameter set `{set}` has no `{name}`")
            }
            DydError::UnknownStaticId { id, static_id } => {
                write!(f, "{id} attaches to static element `{static_id}`, which has no bus")
            }
            DydError::Iidm(message) => write!(f, "{message}"),
            DydError::PowerFlow(status) => {
                write!(f, "the base-case power flow did not converge ({status:?})")
            }
            DydError::SharedBus { bus, models } => write!(
                f,
                "bus {bus} carries {} dynamic machines ({}); their share of its solved \
                 reactive power is not in the file, only the total is",
                models.len(),
                models.join(", ")
            ),
            DydError::Build(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for DydError {}

/// Something read but not acted on.
#[derive(Clone, Debug, PartialEq)]
pub enum DydWarning {
    /// A `lib` this reader does not implement.
    UnsupportedLib { id: String, lib: String },
    /// A nonzero exponential saturation characteristic, which no model here
    /// uses.
    SaturationIgnored { id: String, md: f64, mq: f64, nd: f64, nq: f64 },
    /// A limit this reader carries through but that the model does not use.
    /// Kept for anything that gains a limit the library still cannot represent.
    LimitsIgnored { id: String, detail: String },
}

impl std::fmt::Display for DydWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DydWarning::UnsupportedLib { id, lib } => {
                write!(f, "{id}: `{lib}` is not implemented; skipped")
            }
            DydWarning::SaturationIgnored { id, md, mq, nd, nq } => write!(
                f,
                "{id}: exponential saturation (md = {md}, mq = {mq}, nd = {nd}, nq = {nq}) \
                 is not modelled and was dropped"
            ),
            DydWarning::LimitsIgnored { id, detail } => {
                write!(f, "{id}: {detail} is not modelled and was dropped")
            }
        }
    }
}

fn attributes(
    e: &quick_xml::events::BytesStart<'_>,
) -> Result<HashMap<String, String>, DydError> {
    let mut out = HashMap::new();
    for attribute in e.attributes() {
        let attribute = attribute.map_err(|e| DydError::Xml(e.to_string()))?;
        // Dynawo namespaces its own elements (`dyn:`) but not its attributes;
        // stripping any prefix anyway costs nothing and survives a file that
        // does.
        let key = String::from_utf8_lossy(attribute.key.as_ref()).to_string();
        let key = key.rsplit(':').next().unwrap_or(&key).to_string();
        // Decoded by hand rather than through `unescape_value`, which
        // quick-xml compiles only when its `encoding` feature is *off* —
        // `decode_and_unescape_value` replaces it and wants a `Decoder` whose
        // shape depends on that same feature. `cimdecoder` turns `encoding`
        // on, so feature unification makes this file compile under
        // `--features dynamics,iidm` and fail under `--features
        // dynamics,iidm,cgmes`: a configuration CI builds and no single-feature
        // step can see. `src/iidm.rs`'s `Attrs::of` carries the same note and
        // the same workaround, which is the other half of this reader.
        let raw = String::from_utf8_lossy(&attribute.value);
        let value = quick_xml::escape::unescape(&raw)
            .map_err(|e| DydError::Xml(e.to_string()))?
            .into_owned();
        out.insert(key, value);
    }
    Ok(out)
}

/// Reads a `.dyd`.
pub fn read_dyd(path: impl AsRef<Path>) -> Result<DydDocument, DydError> {
    parse_dyd(&slurp(path)?)
}

/// Reads a `.par`.
pub fn read_par(path: impl AsRef<Path>) -> Result<ParDocument, DydError> {
    parse_par(&slurp(path)?)
}

/// Reads a `.crv`.
pub fn read_crv(path: impl AsRef<Path>) -> Result<Vec<Curve>, DydError> {
    parse_crv(&slurp(path)?)
}

fn slurp(path: impl AsRef<Path>) -> Result<String, DydError> {
    let path = path.as_ref();
    std::fs::read_to_string(path).map_err(|e| DydError::Io(format!("{}: {e}", path.display())))
}

/// Local name of an element, with any namespace prefix removed. Dynawo
/// namespaces its `.dyd` elements as `dyn:` and its `.par` and `.crv` elements
/// not at all, so every match here goes through this rather than through the
/// raw name.
fn local_name(name: quick_xml::name::QName<'_>) -> String {
    let raw = String::from_utf8_lossy(name.as_ref()).to_string();
    raw.rsplit(':').next().unwrap_or(&raw).to_string()
}

pub fn parse_dyd(text: &str) -> Result<DydDocument, DydError> {
    let mut reader = Reader::from_str(text);
    let mut models = Vec::new();
    loop {
        match reader.read_event() {
            Ok(XmlEvent::Empty(e)) | Ok(XmlEvent::Start(e)) => {
                if local_name(e.name()) != "blackBoxModel" {
                    continue;
                }
                let a = attributes(&e)?;
                models.push(BlackBox {
                    id: a.get("id").cloned().unwrap_or_default(),
                    lib: a.get("lib").cloned().unwrap_or_default(),
                    par_file: a.get("parFile").cloned(),
                    par_id: a.get("parId").cloned(),
                    static_id: a.get("staticId").cloned(),
                });
            }
            Ok(XmlEvent::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(DydError::Xml(e.to_string())),
        }
    }
    Ok(DydDocument { models })
}

pub fn parse_par(text: &str) -> Result<ParDocument, DydError> {
    let mut reader = Reader::from_str(text);
    let mut sets: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut current: Option<String> = None;
    loop {
        match reader.read_event() {
            Ok(XmlEvent::Start(e)) if local_name(e.name()) == "set" => {
                current = attributes(&e)?.get("id").cloned();
                if let Some(id) = &current {
                    sets.entry(id.clone()).or_default();
                }
            }
            Ok(XmlEvent::End(e)) if local_name(e.name()) == "set" => current = None,
            Ok(XmlEvent::Empty(e)) | Ok(XmlEvent::Start(e)) if local_name(e.name()) == "par" => {
                let a = attributes(&e)?;
                if let (Some(set), Some(name), Some(value)) =
                    (current.as_ref(), a.get("name"), a.get("value"))
                {
                    sets.entry(set.clone()).or_default().insert(name.clone(), value.clone());
                }
            }
            Ok(XmlEvent::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(DydError::Xml(e.to_string())),
        }
    }
    Ok(ParDocument { sets })
}

pub fn parse_crv(text: &str) -> Result<Vec<Curve>, DydError> {
    let mut reader = Reader::from_str(text);
    let mut curves = Vec::new();
    loop {
        match reader.read_event() {
            Ok(XmlEvent::Empty(e)) | Ok(XmlEvent::Start(e)) if local_name(e.name()) == "curve" => {
                let a = attributes(&e)?;
                curves.push(Curve {
                    model: a.get("model").cloned().unwrap_or_default(),
                    variable: a.get("variable").cloned().unwrap_or_default(),
                });
            }
            Ok(XmlEvent::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(DydError::Xml(e.to_string())),
        }
    }
    Ok(curves)
}

/// Which Dynawo generator libraries this reader understands.
fn is_four_windings(lib: &str) -> bool {
    lib.starts_with("GeneratorSynchronousFourWindings")
}

fn is_three_windings(lib: &str) -> bool {
    lib.starts_with("GeneratorSynchronousThreeWindings")
}

fn is_generator(lib: &str) -> bool {
    lib.starts_with("GeneratorSynchronous")
}

fn has_proportional_regulations(lib: &str) -> bool {
    lib.ends_with("ProportionalRegulations")
}

fn need(
    par: &ParDocument,
    id: &str,
    set: &str,
    name: &str,
) -> Result<f64, DydError> {
    par.number(set, name).ok_or_else(|| DydError::MissingParameter {
        id: id.to_string(),
        set: set.to_string(),
        name: name.to_string(),
    })
}

/// Turns a `.dyd`/`.par` pair into [`UnitSpec`]s.
///
/// `bus_of` maps a `staticId` — or, for a standalone case that has none, the
/// model's own `id` — to a bus index. That correspondence is what the IIDM half
/// of a coupled case carries, and this reader does not attempt to invent it.
/// `s_base` is the network's own base, needed for the one inferred conversion
/// the module doc describes.
pub fn to_units(
    dyd: &DydDocument,
    par: &ParDocument,
    bus_of: &HashMap<String, usize>,
    s_base: f64,
) -> Result<(Vec<UnitSpec>, Vec<DydWarning>), DydError> {
    let mut warnings = Vec::new();
    let mut units = Vec::new();

    for model in &dyd.models {
        if !is_generator(&model.lib) {
            continue;
        }
        if !is_four_windings(&model.lib) && !is_three_windings(&model.lib) {
            warnings.push(DydWarning::UnsupportedLib {
                id: model.id.clone(),
                lib: model.lib.clone(),
            });
            continue;
        }
        let set = model
            .par_id
            .clone()
            .ok_or_else(|| DydError::NoParameterSet { id: model.id.clone() })?;
        if !par.sets.contains_key(&set) {
            return Err(DydError::UnknownParameterSet {
                id: model.id.clone(),
                par_id: set,
            });
        }
        // A model attached to an IIDM network names the element it sits on; a
        // standalone case — DynaSwing's own examples are all standalone — has
        // no static half to attach to and is identified by its own id. Falling
        // back rather than refusing is what lets one reader serve both.
        let static_id = model.static_id.clone().unwrap_or_else(|| model.id.clone());
        let &bus = bus_of.get(&static_id).ok_or_else(|| DydError::UnknownStaticId {
            id: model.id.clone(),
            static_id: static_id.clone(),
        })?;

        let g = |name: &str| need(par, &model.id, &set, name);
        // Dynawo states the machine's rating as its apparent-power nominal, and
        // puts every reactance on that base — the same convention every model
        // here already expects.
        //
        // A three-windings set carries no `XpqPu` and no `Tpq0`, because a
        // salient-pole rotor has no q-axis transient to have a time constant
        // for. That absence *is* the model, so the reader keys on the library
        // name rather than on which parameters happen to be present.
        let machine = if is_three_windings(&model.lib) {
            MachineSpec::GenSalient(GenSalientParams {
                h: g("generator_H")?,
                d: g("generator_DPu")?,
                ra: g("generator_RaPu")?,
                xd: g("generator_XdPu")?,
                xq: g("generator_XqPu")?,
                xdp: g("generator_XpdPu")?,
                xdpp: g("generator_XppdPu")?,
                xqpp: g("generator_XppqPu")?,
                xl: g("generator_XlPu")?,
                td0p: g("generator_Tpd0")?,
                td0pp: g("generator_Tppd0")?,
                tq0pp: g("generator_Tppq0")?,
                mbase: g("generator_SNom")?,
            })
        } else {
            MachineSpec::GenRound(GenRoundParams {
                h: g("generator_H")?,
                d: g("generator_DPu")?,
                ra: g("generator_RaPu")?,
                xd: g("generator_XdPu")?,
                xq: g("generator_XqPu")?,
                xdp: g("generator_XpdPu")?,
                xqp: g("generator_XpqPu")?,
                xdpp: g("generator_XppdPu")?,
                xqpp: g("generator_XppqPu")?,
                xl: g("generator_XlPu")?,
                td0p: g("generator_Tpd0")?,
                tq0p: g("generator_Tpq0")?,
                td0pp: g("generator_Tppd0")?,
                tq0pp: g("generator_Tppq0")?,
                mbase: g("generator_SNom")?,
            })
        };

        let (md, mq, nd, nq) = (
            par.number(&set, "generator_md").unwrap_or(0.0),
            par.number(&set, "generator_mq").unwrap_or(0.0),
            par.number(&set, "generator_nd").unwrap_or(0.0),
            par.number(&set, "generator_nq").unwrap_or(0.0),
        );
        if md != 0.0 || mq != 0.0 {
            warnings.push(DydWarning::SaturationIgnored {
                id: model.id.clone(),
                md,
                mq,
                nd,
                nq,
            });
        }

        let (mut avr, mut gov) = (None, None);
        if has_proportional_regulations(&model.lib) {
            // The field-voltage ceiling is carried straight through: gridoxide's
            // own initialization reproduces Dynawo's `efdPu` exactly, so the two
            // agree on what a per-unit field voltage is. See
            // `tests/dynamics_reference_test.rs`.
            avr = Some(AvrSpec::VrProportional {
                k: g("voltageRegulator_Gain")?,
                limits: Limits {
                    min: par.number(&set, "voltageRegulator_EfdMinPu"),
                    max: par.number(&set, "voltageRegulator_EfdMaxPu"),
                },
            });

            // The one inferred conversion — see the module doc. KGover is a
            // gain on the machine's own PNom, and GoverProportional wants one
            // on the network base. The power limits are stated in MW, so they
            // divide by the network base directly.
            let k_gover = g("governor_KGover")?;
            let p_nom = g("governor_PNom")?;
            gov = Some(GovSpec::GoverProportional {
                k: k_gover * p_nom / s_base,
                limits: Limits {
                    min: par.number(&set, "governor_PMin").map(|p| p / s_base),
                    max: par.number(&set, "governor_PMax").map(|p| p / s_base),
                },
            });
        }

        units.push(UnitSpec {
            id: model.id.clone(),
            bus,
            p: None,
            q: None,
            machine,
            avr,
            gov,
            pss: None,
        });
    }

    Ok((units, warnings))
}

/// Reads a whole Dynawo case — the IIDM network and the `.dyd`/`.par` pair —
/// and returns a system ready to integrate.
///
/// This is what the two halves were always for. `src/iidm.rs` reads the static
/// network, [`to_units`] reads the dynamic models, and the `staticId` on each
/// `blackBoxModel` is the correspondence between them.
///
/// # How each machine's terminal power is recovered, exactly
///
/// A dynamic study needs each machine's *own* terminal power, and a power flow
/// produces only its bus's total. For a Dynawo case that is not a guess: the
/// IIDM states every load's `p0`/`q0`, and those are precisely what went into
/// the bus's specification, so
///
/// ```text
/// machine's injection = bus's solved injection − the loads the file states there
/// ```
///
/// is exact for both active and reactive power, at a `PV` bus as much as a `PQ`
/// one. The one case it cannot resolve is **two machines on one bus**: their
/// share of the bus's solved reactive power is genuinely not in the file, only
/// the total is, so that is refused by name rather than split by a guess. It
/// is the same position `json::resolve_split` takes, reached from the other
/// direction.
///
/// # What is not carried over
///
/// Tap changers, whose dynamics this library does not model, and saturation.
/// A case that depends on either is known in advance to diverge — which is a
/// better position than discovering it during a comparison.
#[cfg(feature = "iidm")]
pub fn load_case(
    iidm_path: impl AsRef<Path>,
    dyd_path: impl AsRef<Path>,
    par_path: impl AsRef<Path>,
    f_nom: f64,
) -> Result<(super::DynamicSystem, Vec<DydWarning>), DydError> {
    use crate::network::{build_ybus, power_injections, stamp_shunts};
    use crate::solver::{PowerFlowOptions, SolveStatus};

    let network = crate::iidm::read(iidm_path).map_err(|e| DydError::Iidm(e.to_string()))?;
    let dyd = read_dyd(dyd_path)?;
    let par = read_par(par_path)?;

    // A dynamic model attaches to a *generator*, not to a bus, so the map is
    // the one the importer now retains rather than a bus lookup.
    let bus_of: HashMap<String, usize> = network
        .injections
        .iter()
        .filter(|i| i.generator)
        .map(|i| (i.id.clone(), i.bus))
        .collect();
    let (units, mut warnings) = to_units(&dyd, &par, &bus_of, network.base_mva)?;

    // A model attached to a static element that is not a generator is a
    // dynamic model of something this library treats as static — the case's
    // tap-changing loads, most often. Their static data is still used, so the
    // network is right; what is lost is their dynamics, and that is worth
    // saying rather than leaving to be noticed.
    for model in dyd.models.iter().filter(|m| m.static_id.is_some() && !is_generator(&m.lib)) {
        warnings.push(DydWarning::UnsupportedLib {
            id: model.id.clone(),
            lib: model.lib.clone(),
        });
    }

    let report = crate::run_power_flow(
        network.buses.clone(),
        &network.lines,
        &network.transformers,
        &network.shunts,
        crate::TapData::none(),
        PowerFlowOptions::default(),
    );
    if report.stats.status != SolveStatus::Converged {
        return Err(DydError::PowerFlow(report.stats.status));
    }
    let buses = report.buses;

    let mut ybus = build_ybus(buses.len(), &network.lines, &network.transformers);
    stamp_shunts(&mut ybus, &network.shunts);
    let (p_calc, q_calc) = power_injections(&buses, &ybus.finish());

    // Loads are stated, so they come off exactly; whatever is left at a bus is
    // its machine's.
    let mut machine_share: HashMap<usize, Complex<f64>> = HashMap::new();
    for unit in &units {
        machine_share
            .entry(unit.bus)
            .or_insert_with(|| Complex::new(p_calc[unit.bus], q_calc[unit.bus]));
    }
    for injection in network.injections.iter().filter(|i| !i.generator) {
        if let Some(share) = machine_share.get_mut(&injection.bus) {
            *share -= Complex::new(injection.p, injection.q);
        }
    }

    let mut per_bus: HashMap<usize, Vec<String>> = HashMap::new();
    for unit in &units {
        per_bus.entry(unit.bus).or_default().push(unit.id.clone());
    }
    for (&bus, models) in &per_bus {
        if models.len() > 1 {
            return Err(DydError::SharedBus { bus, models: models.clone() });
        }
    }

    let mut devices = Vec::with_capacity(units.len());
    for unit in units {
        let model = unit
            .clone()
            .into_model(network.base_mva, f_nom)
            .map_err(|e| DydError::Build(e.to_string()))?;
        devices.push(DeviceSpec {
            id: unit.id.clone(),
            bus: unit.bus,
            s: machine_share[&unit.bus],
            model,
        });
    }

    let system = build(SystemSpec {
        buses: &buses,
        lines: &network.lines,
        transformers: &network.transformers,
        shunts: &network.shunts,
        devices,
        fixed_buses: Vec::new(),
    })
    .map_err(|e| DydError::Build(e.to_string()))?;

    Ok((system, warnings))
}
