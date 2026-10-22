//! A network read from any of the importers, and an AC power flow over it.
//!
//! What `gridoxide solve` and the `gridmcp` MCP server share: the format is
//! told from the path ([`detect`]), the file read into one shape
//! ([`load`]) whichever importer produced it, and solved with the outer loops
//! the settings ask for ([`Network::solve`]).

use std::path::Path;

use crate::types::{Bus, Line, RegulatingMachine, TapChanger, Transformer};
use crate::{PowerFlowOptions, PowerFlowReport, RemoteControlData, TapData};

/// The file formats [`load`] reads, each behind its importer's feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Cgmes,
    Ucte,
    Iidm,
    Matpower,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Format::Cgmes => "CGMES",
            Format::Ucte => "UCTE-DEF",
            Format::Iidm => "IIDM",
            Format::Matpower => "MATPOWER",
        }
    }
}

/// The format `path` is in, as [`load`] would read it: by extension, and a
/// directory holding `.xml` files as a CGMES profile set. `None` for anything
/// else, and for a format this build has no importer for.
///
/// A single `.xml` file is not taken for CGMES here, though [`load`] reads
/// one as its directory's set: a directory has no other reading, a lone
/// `.xml` file has many.
pub fn detect(path: &Path) -> Option<Format> {
    if path.is_dir() {
        let has_xml = std::fs::read_dir(path).ok()?.flatten().any(|e| {
            let p = e.path();
            p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("xml"))
        });
        return (cfg!(feature = "cgmes") && has_xml).then_some(Format::Cgmes);
    }
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "m" if cfg!(feature = "matpower") => Some(Format::Matpower),
        "uct" | "ucte" if cfg!(feature = "ucte") => Some(Format::Ucte),
        "xiidm" if cfg!(feature = "iidm") => Some(Format::Iidm),
        _ => None,
    }
}

/// A network loaded for a solve, whichever importer produced it.
///
/// Deliberately not `SecurityNetwork`: that one is gated behind the `rao`
/// feature and carries CRAC-shaped fields (countries, initially-open branches)
/// a power flow has no use for, and it cannot load CGMES — which is the only
/// format whose fixtures declare tap controls.
pub struct Network {
    pub format: Format,
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub transformers: Vec<Transformer>,
    pub shunts: Vec<crate::network::ShuntAdm>,
    pub tap_changers: Vec<Option<TapChanger>>,
    pub regulation: Vec<crate::outerloop::TapRegulation>,
    /// Area index per bus, and what each area is called. Empty when the format
    /// says nothing about areas.
    pub areas: (Vec<Option<usize>>, Vec<String>),
    /// Scheduled net export per area, per-unit. Empty unless the format states
    /// one — only CGMES does.
    pub area_targets: Vec<f64>,
    /// Per bus: what to call it. The file's own name for it — MATPOWER bus
    /// number, UCTE node code, IIDM bus id, CGMES `TopologicalNode` name (its
    /// mRID where it has none).
    pub labels: Vec<String>,
    /// Per bus: the file's own identifier, where it differs from the label —
    /// the CGMES mRID, since names need not be unique. Otherwise the label.
    pub bus_ids: Vec<String>,
    /// Per branch, lines first and then transformers (the order
    /// [`branch_params`](crate::branch_flow::branch_params) uses): what to call
    /// it. The file's own id — CGMES the equipment's name, else its mRID.
    /// MATPOWER has none, so there it is `from-to` in bus numbers, with `#k`
    /// on the k-th parallel branch.
    pub branch_labels: Vec<String>,
    /// Per branch: the file's own identifier, as [`bus_ids`](Self::bus_ids) is
    /// per bus.
    pub branch_ids: Vec<String>,
    pub notes: Vec<String>,
    pub s_base_va: f64,
    /// The regulating machines the document declared, where it declares any.
    /// Only CGMES states per-machine reactive capability; the other importers
    /// leave this empty and `--dispatch` then has nothing to attribute.
    pub machines: Vec<RegulatingMachine>,
    /// Per-bus reactive injection that is not a regulating machine's.
    pub nonregulating_q: Vec<f64>,
}

/// The outer loops and tolerances of a solve — `gridoxide solve`'s flags.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveSettings {
    pub control_taps: bool,
    pub enforce_q_limits: bool,
    pub distribute_slack: bool,
    pub area_interchange: bool,
    pub control_remote_voltage: bool,
    pub tol: f64,
    pub max_iter: usize,
    pub max_outer: usize,
    pub max_tap_shift: i32,
}

impl Default for SolveSettings {
    fn default() -> Self {
        Self {
            control_taps: false,
            enforce_q_limits: false,
            distribute_slack: false,
            area_interchange: false,
            control_remote_voltage: false,
            tol: 1e-8,
            max_iter: 30,
            max_outer: 40,
            max_tap_shift: 3,
        }
    }
}

impl SolveSettings {
    /// Settings no network can honour.
    pub fn check(&self) -> Result<(), String> {
        if self.distribute_slack && self.area_interchange {
            return Err(
                "--distribute-slack and --area-interchange cannot both be given: area \
                        interchange subsumes distributed slack (one area with a zero target *is* \
                        distributed slack)"
                    .to_string(),
            );
        }
        Ok(())
    }
}

impl Network {
    /// An AC power flow with the outer loops `settings` ask for. An error only
    /// for settings the network cannot honour; a solve that does not converge
    /// is a report saying so.
    pub fn solve(&self, settings: &SolveSettings) -> Result<PowerFlowReport, String> {
        settings.check()?;
        let distribution = settings
            .distribute_slack
            .then(|| crate::outerloop::SlackDistribution::uniform(&self.buses));
        let (of_bus, area_names) = self.areas.clone();
        let area_definition = if settings.area_interchange {
            if area_names.is_empty() {
                return Err(
                    "--area-interchange needs an area assignment, and this network states none. \
                     CGMES supplies one through ControlArea/TieFlow and UCTE through its ##Z \
                     country codes"
                        .to_string(),
                );
            }
            // Weighted by generation rather than by bus type: a net position is met
            // by redispatch, and a generator held at a fixed active set-point — a
            // `PQ` bus here — is exactly the machine an operator redispatches. The
            // `uniform` policy distributed slack uses would leave such an area with
            // no participant at all.
            let mut d = crate::outerloop::AreaDefinition::by_generation(
                &self.buses,
                of_bus,
                area_names.len(),
            );
            if !self.area_targets.is_empty() {
                d.targets = self.area_targets.clone();
            }
            d.max_outer_iter = settings.max_outer;
            Some(d)
        } else {
            None
        };
        Ok(crate::run_power_flow_with_remote(
            self.buses.clone(),
            &self.lines,
            &self.transformers,
            &self.shunts,
            TapData {
                changers: &self.tap_changers,
                regulation: &self.regulation,
            },
            RemoteControlData {
                machines: &self.machines,
            },
            PowerFlowOptions {
                control_taps: settings.control_taps,
                control_remote_voltage: settings.control_remote_voltage,
                enforce_q_limits: settings.enforce_q_limits,
                distribute_slack: distribution,
                area_interchange: area_definition,
                tap_max_shift: settings.max_tap_shift,
                max_outer_iter: settings.max_outer,
                tol: settings.tol,
                max_iter: settings.max_iter,
                ..Default::default()
            },
        ))
    }
}

/// `from-to` in MATPOWER bus numbers per branch, `#k` on the k-th parallel one.
#[cfg(feature = "matpower")]
fn matpower_branch_labels(
    lines: &[Line],
    transformers: &[Transformer],
    numbers: &[u64],
) -> Vec<String> {
    let mut seen = std::collections::HashMap::new();
    let ends = lines
        .iter()
        .map(|l| (l.from, l.to))
        .chain(transformers.iter().map(|t| (t.from, t.to)));
    ends.map(|(f, t)| {
        let base = format!("{}-{}", numbers[f], numbers[t]);
        let k = seen.entry(base.clone()).or_insert(0usize);
        *k += 1;
        if *k == 1 { base } else { format!("{base}#{k}") }
    })
    .collect()
}

/// What to call the CGMES object `mrid`: its `IdentifiedObject.name`, else
/// the mRID. A three-winding transformer's star point, which no document
/// names, is its transformer's name with ` star`.
#[cfg(feature = "cgmes")]
fn cgmes_name(ds: &cimmodel::CimDataset, mrid: &str) -> String {
    let (object, suffix) = match mrid.strip_suffix("_star") {
        Some(transformer) => (transformer, " star"),
        None => (mrid, ""),
    };
    match ds
        .entries
        .get(object)
        .and_then(|e| e.get_str("IdentifiedObject.name"))
        .map(str::trim)
        .filter(|n| !n.is_empty())
    {
        Some(name) => format!("{name}{suffix}"),
        None => mrid.to_string(),
    }
}

/// Reads the network at `path`: a `.m`, `.uct`, `.xiidm`, or a directory of
/// CGMES profiles (or one of them, standing for its directory).
pub fn load(path: &str) -> Result<Network, String> {
    // Only the extension-dispatched importers look at it; a cgmes-only
    // build reaches the directory branch without ever asking.
    #[cfg(any(feature = "ucte", feature = "iidm", feature = "matpower"))]
    let lower = path.to_ascii_lowercase();

    #[cfg(feature = "matpower")]
    if lower.ends_with(".m") {
        let n = crate::matpower::read(path).map_err(|e| e.to_string())?;
        let branch_labels = matpower_branch_labels(&n.lines, &n.transformers, &n.bus_numbers);
        return Ok(Network {
            format: Format::Matpower,
            branch_ids: branch_labels.clone(),
            branch_labels,
            bus_ids: n.bus_numbers.iter().map(u64::to_string).collect(),
            buses: n.buses,
            lines: n.lines,
            transformers: n.transformers,
            shunts: n.shunts,
            // MATPOWER's power flow has no tap control, and its `area` column
            // states no schedule to hold.
            tap_changers: Vec::new(),
            regulation: Vec::new(),
            areas: (Vec::new(), Vec::new()),
            area_targets: Vec::new(),
            labels: n.bus_numbers.iter().map(u64::to_string).collect(),
            notes: n.notes,
            s_base_va: n.base_mva * 1e6,
            machines: Vec::new(),
            nonregulating_q: Vec::new(),
        });
    }

    #[cfg(feature = "ucte")]
    if lower.ends_with(".uct") || lower.ends_with(".ucte") {
        let n = crate::ucte::read(path).map_err(|e| e.to_string())?;
        let areas = n.country_areas();
        return Ok(Network {
            buses: n.buses,
            lines: n.lines,
            transformers: n.transformers,
            shunts: n.shunts,
            tap_changers: n.tap_changers,
            regulation: n.regulation,
            areas,
            // UCTE states no scheduled net position anywhere; `##Z<cc>` gives
            // the membership and nothing else.
            area_targets: Vec::new(),
            bus_ids: n.node_codes.clone(),
            labels: n.node_codes,
            branch_labels: n.branch_ids.clone(),
            branch_ids: n.branch_ids,
            format: Format::Ucte,
            notes: n.notes,
            s_base_va: n.base_mva * 1e6,
            // UCTE-DEF and IIDM state no per-machine reactive capability, so
            // there is nothing to attribute and `--dispatch` says so rather
            // than printing an empty table.
            machines: Vec::new(),
            nonregulating_q: Vec::new(),
        });
    }

    #[cfg(feature = "iidm")]
    if lower.ends_with(".xiidm") {
        let n = crate::iidm::read(path).map_err(|e| e.to_string())?;
        let mut notes = n.notes.clone();
        if n.areas.report.areas > 0 || n.areas.report.other_types > 0 {
            let r = &n.areas.report;
            notes.push(format!(
                "{} control area(s): {} of another areaType, {} without a schedule, \
                 {} unknown voltage level(s), {} contested bus(es), {} bus(es) in no area",
                r.areas,
                r.other_types,
                r.without_target.len(),
                r.unknown_voltage_levels,
                r.contested_buses,
                r.unassigned_buses
            ));
        }
        return Ok(Network {
            buses: n.buses,
            lines: n.lines,
            transformers: n.transformers,
            shunts: n.shunts,
            areas: (
                n.areas.of_bus.clone(),
                n.areas
                    .ids
                    .iter()
                    .map(|(id, name)| {
                        if name.trim().is_empty() {
                            id.clone()
                        } else {
                            name.trim().to_string()
                        }
                    })
                    .collect(),
            ),
            area_targets: n.areas.targets.clone(),
            tap_changers: n.tap_changers,
            regulation: n.regulation,
            bus_ids: n.bus_labels.clone(),
            labels: n.bus_labels,
            branch_labels: n.branch_ids.clone(),
            branch_ids: n.branch_ids,
            format: Format::Iidm,
            notes,
            s_base_va: n.base_mva * 1e6,
            // See the UCTE branch above.
            machines: Vec::new(),
            nonregulating_q: Vec::new(),
        });
    }

    #[cfg(feature = "cgmes")]
    {
        // A directory of profiles, or one profile file whose siblings are the
        // rest of the set — which is how every conformity configuration is
        // laid out, and how a user actually has them on disk.
        let dir = std::path::Path::new(path);
        let dir = if dir.is_dir() {
            dir.to_path_buf()
        } else {
            dir.parent().unwrap_or(dir).to_path_buf()
        };
        let mut profiles: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| format!("reading {}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "xml"))
            .collect();
        profiles.sort();
        if profiles.is_empty() {
            return Err(format!("no CGMES profile .xml files in {}", dir.display()));
        }
        let refs: Vec<&std::path::Path> = profiles.iter().map(|p| p.as_path()).collect();
        let ds = crate::cgmes::load_profiles(&refs).map_err(|e| e.to_string())?;
        let s_base_va = 100e6;
        let net = crate::cgmes::cgmes_to_network(&ds, s_base_va).map_err(|e| e.to_string())?;
        let mut notes = Vec::new();
        let r = &net.tap_report;
        notes.push(format!(
            "{} CGMES profile file(s); tap controls: {} read, {} disabled, {} unattached, \
             {} without a target, {} in an unmodelled mode, {} with an unresolvable flow",
            profiles.len(),
            r.converted,
            r.disabled,
            r.unattached,
            r.without_target,
            r.unsupported_mode,
            r.unresolved_flow
        ));
        let areas =
            crate::cgmes::cgmes_control_areas(&ds, &net, s_base_va).map_err(|e| e.to_string())?;
        if areas.report.areas > 0 {
            notes.push(format!(
                "{} control area(s), {} tie flow(s): {} unresolved, {} without a boundary, \
                 {} without a schedule, {} contested bus(es), {} bus(es) in no area",
                areas.report.areas,
                areas.report.tie_flows,
                areas.report.unresolved_tie_flows,
                areas.report.without_boundary.len(),
                areas.report.without_target.len(),
                areas.report.contested_buses,
                areas.report.unassigned_buses
            ));
        }
        let labels = net.bus_ids.iter().map(|id| cgmes_name(&ds, id)).collect();
        let branch_labels = net
            .branch_ids
            .iter()
            .map(|id| cgmes_name(&ds, id))
            .collect();
        let bus_ids = net.bus_ids;
        let branch_ids = net.branch_ids;
        let machines = net.voltage_control.machines;
        let nonregulating_q = net.voltage_control.nonregulating_q;
        return Ok(Network {
            areas: (
                areas.of_bus,
                areas.ids.into_iter().map(|(_, n)| n).collect(),
            ),
            area_targets: areas.targets,
            buses: net.buses,
            lines: net.lines,
            transformers: net.transformers,
            shunts: net.shunts,
            tap_changers: net.tap_changers,
            regulation: net.regulation,
            labels,
            bus_ids,
            branch_labels,
            branch_ids,
            format: Format::Cgmes,
            notes,
            s_base_va,
            machines,
            nonregulating_q,
        });
    }

    #[allow(unreachable_code)]
    Err(format!(
        "cannot tell what `{path}` is; expected a .uct, .xiidm, .m, or a directory of CGMES profiles"
    ))
}
