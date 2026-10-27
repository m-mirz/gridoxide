//! MATPOWER case import (`.m`).
//!
//! MATPOWER's case format is a MATLAB function that assigns literal matrices
//! to the fields of a struct `mpc`: `mpc.baseMVA`, `mpc.bus`, `mpc.gen` and
//! `mpc.branch` are what a power flow needs, and the only fields read here.
//! Before this module, gridoxide reached MATPOWER only through
//! `python/gridoxide/matpower.py`, which converts a case into PGM JSON. That
//! path cannot carry a MATPOWER case exactly: PGM has no slack bus, only a
//! `source` behind a short-circuit impedance, so the reference bus ends up
//! behind an impedance it does not have, off its voltage setpoint by however
//! much current it carries; and PGM's transformer `clock` holds a phase shift
//! only in multiples of 30°. Read directly, neither loss happens.
//!
//! # The problem MATPOWER solves, and so this importer
//!
//! The rules are `runpf`'s, from MATPOWER's `bustypes`, `ext2int` and
//! `makeSbus`/`makeYbus`:
//!
//! - **Isolated buses** (type 4) take no part: they, every branch touching
//!   one and every generator at one are left out of the solve. The bus stays
//!   in the bus list, as an island with no reference, which the solver
//!   reports at zero voltage.
//! - **Out-of-service** branches (`BR_STATUS = 0`) and generators
//!   (`GEN_STATUS <= 0`) are dropped.
//! - **Bus types**: a reference bus (3) or PV bus (2) keeps its type only with
//!   an online generator; without one it is solved as PQ. If no reference bus
//!   remains, the first PV bus becomes the reference, as `bustypes` does.
//! - **Injections**: every online generator adds `Pg + jQg` at its bus and
//!   every bus subtracts `Pd + jQd`. A generator on a PQ bus is therefore a
//!   fixed injection; on a PV or reference bus its `Qg` is overwritten by the
//!   solve anyway.
//! - **Setpoints**: a PV or reference bus holds the `Vg` of its generators —
//!   the last online one in file order, as `V0(gbus) = gen(on, VG)` assigns —
//!   and never the bus's own `Vm`, which MATPOWER reads only as a starting
//!   value. The reference bus holds its `Va` as the angle reference.
//! - **Branches** are MATPOWER's π-model: series `1/(r + jx)`, total charging
//!   `b` split over both ends, and on the from side an ideal transformer of
//!   complex ratio `τ·e^{jθ}` (`RATIO`, 0 meaning 1, and `ANGLE` in degrees).
//!   That is exactly [`network::branch_calc_param`](crate::network::branch_calc_param)'s
//!   two-winding model, so a branch with a ratio or a shift becomes a
//!   [`Transformer`] and the rest a [`Line`].
//! - **Shunts**: `Gs + jBs` (MW and MVAr consumed at 1 p.u.) become a shunt
//!   admittance on the bus.
//!
//! Reactive limits (`Qmax`, `Qmin`) are carried on each bus, summed over its
//! online generators, and enforced only if a caller asks for it.
//!
//! # Optimal power flow
//!
//! [`ac_opf_network`] reads the same case as MATPOWER's `runopf` does: the
//! columns above plus `mpc.gencost`, bus voltage limits (`VMAX`, `VMIN`),
//! generator output limits (`PMAX`, `PMIN`, `QMAX`, `QMIN`), `RATE_A` as an
//! apparent-power limit at both ends (0 unlimited) and the branch
//! angle-difference limits `ANGMIN`/`ANGMAX`. See its documentation.
//!
//! Every value is per unit on the case's own `baseMVA`, so a mismatch
//! tolerance means the same thing it does to MATPOWER. Starting values are a
//! flat start: 1 p.u. and zero angle, except where the case fixes a setpoint.
//!
//! # What is refused
//!
//! A case file is MATLAB code, and some cases finish with code that edits the
//! matrices they just wrote — `case33bw` converts its loads from kW and its
//! impedances from ohms that way. Reading the literals and ignoring the code
//! would solve a different network without a word, so an assignment to
//! `mpc.bus`, `mpc.gen`, `mpc.branch` or `mpc.baseMVA` that is not a literal
//! is an error. So is a version-1 case, whose layout differs, and a binary
//! `.mat` file.

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex;

use crate::network::ShuntAdm;
use crate::types::{Bus, BusType, Line, Transformer};

// Column indices, from MATPOWER's `idx_bus`, `idx_gen` and `idx_brch`.
const BUS_I: usize = 0;
const BUS_TYPE: usize = 1;
const PD: usize = 2;
const QD: usize = 3;
const GS: usize = 4;
const BS: usize = 5;
const VA: usize = 8;
const BASE_KV: usize = 9;
const BUS_COLUMNS: usize = 10;
#[cfg(feature = "opf")]
const VMAX: usize = 11;
#[cfg(feature = "opf")]
const VMIN: usize = 12;

const GEN_BUS: usize = 0;
const PG: usize = 1;
const QG: usize = 2;
const QMAX: usize = 3;
const QMIN: usize = 4;
const VG: usize = 5;
const GEN_STATUS: usize = 7;
const GEN_COLUMNS: usize = 8;
#[cfg(feature = "opf")]
const PMAX: usize = 8;
#[cfg(feature = "opf")]
const PMIN: usize = 9;

const F_BUS: usize = 0;
const T_BUS: usize = 1;
const BR_R: usize = 2;
const BR_X: usize = 3;
const BR_B: usize = 4;
const TAP: usize = 8;
const SHIFT: usize = 9;
const BR_STATUS: usize = 10;
const BRANCH_COLUMNS: usize = 11;
#[cfg(feature = "opf")]
const RATE_A: usize = 5;
#[cfg(feature = "opf")]
const ANGMIN: usize = 11;
#[cfg(feature = "opf")]
const ANGMAX: usize = 12;

// From MATPOWER's `idx_cost`.
#[cfg(feature = "opf")]
const MODEL: usize = 0;
#[cfg(feature = "opf")]
const NCOST: usize = 3;
#[cfg(feature = "opf")]
const COST: usize = 4;
#[cfg(feature = "opf")]
const PW_LINEAR: f64 = 1.0;
#[cfg(feature = "opf")]
const POLYNOMIAL: f64 = 2.0;

const PQ: f64 = 1.0;
const PV: f64 = 2.0;
const REF: f64 = 3.0;
const NONE: f64 = 4.0;

#[derive(Debug)]
pub enum MatpowerError {
    Io(std::io::Error),
    /// Text that is not what a MATPOWER case can contain at that point.
    Parse {
        line: usize,
        message: String,
    },
    /// One of the four fields a power flow needs was never assigned.
    Missing(&'static str),
    /// A row with fewer columns than the format requires.
    ShortRow {
        matrix: &'static str,
        row: usize,
        columns: usize,
        required: usize,
    },
    /// A generator or branch naming a bus the bus matrix does not define.
    UnknownBus {
        matrix: &'static str,
        row: usize,
        bus: f64,
    },
    /// Something the file says that this importer will not read rather than
    /// read wrongly — see the module documentation.
    Unsupported(String),
    /// A measurement [`se_case`] cannot place on the case.
    Measurement(String),
}

impl std::fmt::Display for MatpowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatpowerError::Io(e) => write!(f, "{e}"),
            MatpowerError::Parse { line, message } => write!(f, "line {line}: {message}"),
            MatpowerError::Missing(field) => write!(f, "the case never assigns mpc.{field}"),
            MatpowerError::ShortRow {
                matrix,
                row,
                columns,
                required,
            } => write!(
                f,
                "mpc.{matrix} row {row} has {columns} columns, at least {required} are required"
            ),
            MatpowerError::UnknownBus { matrix, row, bus } => {
                write!(
                    f,
                    "mpc.{matrix} row {row} names bus {bus}, which mpc.bus does not define"
                )
            }
            MatpowerError::Unsupported(what) => write!(f, "{what}"),
            MatpowerError::Measurement(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for MatpowerError {}

impl From<std::io::Error> for MatpowerError {
    fn from(e: std::io::Error) -> Self {
        MatpowerError::Io(e)
    }
}

/// The matrices of a case, as written. Rows keep every column the file gave.
#[derive(Clone, Debug, Default)]
pub struct MatpowerCase {
    pub base_mva: f64,
    pub bus: Vec<Vec<f64>>,
    pub gens: Vec<Vec<f64>>,
    pub branch: Vec<Vec<f64>>,
    /// `mpc.gencost`, if the case has one; only an OPF reads it.
    pub gencost: Option<Vec<Vec<f64>>>,
}

/// A case converted for the solver, per unit on [`base_mva`](Self::base_mva).
#[derive(Clone, Debug)]
pub struct MatpowerNetwork {
    /// One per row of `mpc.bus`, in file order, isolated buses included.
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub transformers: Vec<Transformer>,
    pub shunts: Vec<ShuntAdm>,
    pub base_mva: f64,
    /// Bus index → MATPOWER bus number (`BUS_I`).
    pub bus_numbers: Vec<u64>,
    /// Flat branch index (lines first, then transformers) → 0-based row of
    /// `mpc.branch`. Out-of-service branches have no index.
    pub branch_rows: Vec<usize>,
    /// Bus index → per-unit active generation of its online generators, kept
    /// apart from the net injection [`Bus::p_spec`] folds it into.
    pub generation: Vec<f64>,
    /// What the conversion decided on the case's behalf.
    pub notes: Vec<String>,
}

/// Reads a MATPOWER `.m` case into the solver's model.
pub fn read(path: impl AsRef<Path>) -> Result<MatpowerNetwork, MatpowerError> {
    let path = path.as_ref();
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("mat"))
    {
        return Err(MatpowerError::Unsupported(format!(
            "{} is a binary .mat file; only the .m text format is read",
            path.display()
        )));
    }
    let text = std::fs::read_to_string(path)?;
    to_network(&parse(&text)?)
}

/// Parses the text of a MATPOWER `.m` case.
pub fn parse(text: &str) -> Result<MatpowerCase, MatpowerError> {
    let code = strip_comments(text);
    let mut case = MatpowerCase::default();
    let (mut base_mva, mut bus, mut gens, mut branch, mut gencost) = (None, None, None, None, None);

    let bytes = code.as_bytes();
    let mut i = 0;
    while let Some(found) = code[i..].find("mpc.") {
        let start = i + found;
        i = start + 4;
        // Only where a statement begins: `xmpc.` or `a = mpc.bus(…)` is not
        // an assignment to the case.
        let before = code[..start].trim_end_matches([' ', '\t']);
        if !(before.is_empty() || before.ends_with(['\n', ';', ','])) {
            continue;
        }
        let name_end = code[i..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(code.len(), |n| i + n);
        let name = &code[i..name_end];
        let line = line_of(&code, start);
        let mut j = skip_blanks(bytes, name_end);
        let read = matches!(
            name,
            "bus" | "gen" | "branch" | "gencost" | "baseMVA" | "version"
        );
        if j >= bytes.len() || bytes[j] != b'=' || bytes.get(j + 1) == Some(&b'=') {
            if read {
                return Err(MatpowerError::Unsupported(format!(
                    "line {line}: mpc.{name} is modified by MATLAB code, which this importer \
                     does not run; use a case that states the matrices as plain data"
                )));
            }
            continue;
        }
        j = skip_blanks(bytes, j + 1);
        match name {
            "bus" | "gen" | "branch" | "gencost" => {
                if bytes.get(j) != Some(&b'[') {
                    return Err(MatpowerError::Unsupported(format!(
                        "line {line}: mpc.{name} is assigned an expression, not a literal \
                         matrix, and this importer does not run MATLAB code"
                    )));
                }
                let close =
                    code[j..]
                        .find(']')
                        .map(|n| j + n)
                        .ok_or_else(|| MatpowerError::Parse {
                            line,
                            message: format!("mpc.{name} has no closing ']'"),
                        })?;
                let rows = parse_matrix(&code[j + 1..close], line_of(&code, j))?;
                i = close + 1;
                match name {
                    "bus" => bus = Some(rows),
                    "gen" => gens = Some(rows),
                    "gencost" => gencost = Some(rows),
                    _ => branch = Some(rows),
                }
            }
            "baseMVA" => {
                let end = statement_end(&code, j);
                let value = code[j..end].trim();
                base_mva = Some(value.parse::<f64>().map_err(|_| {
                    MatpowerError::Unsupported(format!(
                        "line {line}: mpc.baseMVA = {value} is not a number"
                    ))
                })?);
                i = end;
            }
            "version" => {
                let end = statement_end(&code, j);
                let value = code[j..end].trim().trim_matches(['\'', '"']);
                if value != "2" {
                    return Err(MatpowerError::Unsupported(format!(
                        "line {line}: MATPOWER case format version {value}; only version 2 is read"
                    )));
                }
                i = end;
            }
            _ => {}
        }
    }

    case.base_mva = base_mva.ok_or(MatpowerError::Missing("baseMVA"))?;
    case.bus = bus.ok_or(MatpowerError::Missing("bus"))?;
    case.gens = gens.ok_or(MatpowerError::Missing("gen"))?;
    case.branch = branch.ok_or(MatpowerError::Missing("branch"))?;
    case.gencost = gencost;
    if case.base_mva.is_nan() || case.base_mva <= 0.0 {
        return Err(MatpowerError::Unsupported(format!(
            "mpc.baseMVA = {} is not positive",
            case.base_mva
        )));
    }
    check_columns("bus", &case.bus, BUS_COLUMNS)?;
    check_columns("gen", &case.gens, GEN_COLUMNS)?;
    check_columns("branch", &case.branch, BRANCH_COLUMNS)?;
    Ok(case)
}

/// Converts parsed matrices into the solver's model; see the module
/// documentation for the rules.
pub fn to_network(case: &MatpowerCase) -> Result<MatpowerNetwork, MatpowerError> {
    let base = case.base_mva;
    let n = case.bus.len();
    let mut index: HashMap<u64, usize> = HashMap::with_capacity(n);
    let mut bus_numbers = Vec::with_capacity(n);
    for (row, b) in case.bus.iter().enumerate() {
        let number = bus_number(b[BUS_I]).ok_or_else(|| MatpowerError::Parse {
            line: 0,
            message: format!("mpc.bus row {} has bus number {}", row + 1, b[BUS_I]),
        })?;
        if index.insert(number, row).is_some() {
            return Err(MatpowerError::Unsupported(format!(
                "mpc.bus defines bus {number} twice"
            )));
        }
        bus_numbers.push(number);
    }
    let resolve = |matrix: &'static str, row: usize, value: f64| {
        bus_number(value)
            .and_then(|b| index.get(&b).copied())
            .ok_or(MatpowerError::UnknownBus {
                matrix,
                row: row + 1,
                bus: value,
            })
    };
    let isolated: Vec<bool> = case.bus.iter().map(|b| b[BUS_TYPE] == NONE).collect();

    // Online generators, per bus.
    let mut gen_p = vec![0.0; n];
    let mut gen_q = vec![0.0; n];
    let mut q_max = vec![0.0; n];
    let mut q_min = vec![0.0; n];
    let mut vg: Vec<Option<f64>> = vec![None; n];
    for (row, g) in case.gens.iter().enumerate() {
        let at = resolve("gen", row, g[GEN_BUS])?;
        if g[GEN_STATUS] <= 0.0 || isolated[at] {
            continue;
        }
        gen_p[at] += g[PG];
        gen_q[at] += g[QG];
        q_max[at] += g[QMAX];
        q_min[at] += g[QMIN];
        vg[at] = Some(g[VG]);
    }

    let mut notes = Vec::new();
    let mut types: Vec<BusType> = case
        .bus
        .iter()
        .zip(&vg)
        .map(|(b, vg)| match (b[BUS_TYPE], vg) {
            (t, Some(_)) if t == REF => BusType::Slack,
            (t, Some(_)) if t == PV => BusType::PV,
            _ => BusType::PQ,
        })
        .collect();
    let demoted = case
        .bus
        .iter()
        .zip(&vg)
        .filter(|(b, vg)| (b[BUS_TYPE] == PV || b[BUS_TYPE] == REF) && vg.is_none())
        .count();
    if demoted > 0 {
        notes.push(format!(
            "{demoted} PV or reference bus(es) without an online generator solved as PQ"
        ));
    }
    if !types.contains(&BusType::Slack)
        && let Some(first_pv) = types.iter().position(|t| *t == BusType::PV)
    {
        types[first_pv] = BusType::Slack;
        notes.push(format!(
            "no reference bus with an online generator; bus {} (the first PV bus) is the reference",
            bus_numbers[first_pv]
        ));
    }
    let unknown_types = case
        .bus
        .iter()
        .filter(|b| ![PQ, PV, REF, NONE].contains(&b[BUS_TYPE]))
        .count();
    if unknown_types > 0 {
        notes.push(format!(
            "{unknown_types} bus(es) of an unknown type solved as PQ"
        ));
    }

    let mut buses = Vec::with_capacity(n);
    let mut shunts = Vec::new();
    let mut generation = vec![0.0; n];
    for (i, b) in case.bus.iter().enumerate() {
        let bus_type = types[i];
        let regulated = bus_type != BusType::PQ;
        let (p, q) = if isolated[i] {
            (0.0, 0.0)
        } else {
            ((gen_p[i] - b[PD]) / base, (gen_q[i] - b[QD]) / base)
        };
        generation[i] = gen_p[i].max(0.0) / base;
        buses.push(Bus {
            idx: i,
            bus_type,
            voltage_mag: if regulated { vg[i].unwrap_or(1.0) } else { 1.0 },
            voltage_ang: if bus_type == BusType::Slack {
                b[VA].to_radians()
            } else {
                0.0
            },
            p_spec: p,
            q_spec: q,
            q_min: if regulated {
                q_min[i] / base
            } else {
                f64::NEG_INFINITY
            },
            q_max: if regulated {
                q_max[i] / base
            } else {
                f64::INFINITY
            },
            u_rated: b[BASE_KV] * 1e3,
            zip_terms: Vec::new(),
        });
        if !isolated[i] && (b[GS] != 0.0 || b[BS] != 0.0) {
            shunts.push(ShuntAdm {
                at: i,
                y: Complex::new(b[GS], b[BS]) / base,
            });
        }
    }

    let mut lines = Vec::new();
    let mut transformers = Vec::new();
    let mut line_rows = Vec::new();
    let mut transformer_rows = Vec::new();
    for (row, br) in case.branch.iter().enumerate() {
        let from = resolve("branch", row, br[F_BUS])?;
        let to = resolve("branch", row, br[T_BUS])?;
        if br[BR_STATUS] == 0.0 || isolated[from] || isolated[to] {
            continue;
        }
        let (r, x, b) = (br[BR_R], br[BR_X], br[BR_B]);
        if br[TAP] == 0.0 && br[SHIFT] == 0.0 {
            lines.push(Line {
                from,
                to,
                r,
                x,
                b_shunt: b,
                g_shunt: 0.0,
            });
            line_rows.push(row);
        } else {
            let ratio = if br[TAP] == 0.0 { 1.0 } else { br[TAP] };
            transformers.push(Transformer {
                from,
                to,
                from_status: 1,
                to_status: 1,
                y_series: Complex::new(1.0, 0.0) / Complex::new(r, x),
                y_shunt: Complex::new(0.0, b),
                tap: Complex::from_polar(ratio, br[SHIFT].to_radians()),
            });
            transformer_rows.push(row);
        }
    }
    let isolated_count = isolated.iter().filter(|&&i| i).count();
    if isolated_count > 0 {
        notes.push(format!(
            "{isolated_count} isolated bus(es) left out of the solve, with their branches and generators"
        ));
    }

    let mut branch_rows = line_rows;
    branch_rows.extend(transformer_rows);
    Ok(MatpowerNetwork {
        buses,
        lines,
        transformers,
        shunts,
        base_mva: base,
        bus_numbers,
        branch_rows,
        generation,
        notes,
    })
}

/// A bus number as MATPOWER writes it: a positive whole number.
fn bus_number(value: f64) -> Option<u64> {
    (value >= 1.0 && value.fract() == 0.0 && value < u64::MAX as f64).then_some(value as u64)
}

fn check_columns(
    matrix: &'static str,
    rows: &[Vec<f64>],
    required: usize,
) -> Result<(), MatpowerError> {
    match rows.iter().position(|r| r.len() < required) {
        Some(row) => Err(MatpowerError::ShortRow {
            matrix,
            row: row + 1,
            columns: rows[row].len(),
            required,
        }),
        None => Ok(()),
    }
}

/// The text with every `%` comment removed and every `...` continuation
/// joined, line structure otherwise kept so positions still map to lines.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let mut quoted = false;
        let mut end = line.len();
        for (k, c) in line.char_indices() {
            match c {
                '\'' => quoted = !quoted,
                '%' if !quoted => {
                    end = k;
                    break;
                }
                _ => {}
            }
        }
        let code = &line[..end];
        match code.find("...") {
            // A continuation: the next line carries on this one, and the rest
            // of this one is a comment. A vertical tab stands in for the line
            // break — blank to the tokenizer, still a line to `line_of`.
            Some(dots) => {
                out.push_str(&code[..dots]);
                out.push(CONTINUED);
            }
            None => {
                out.push_str(code);
                out.push('\n');
            }
        }
    }
    out
}

/// The line break of a `...` continuation, after [`strip_comments`].
const CONTINUED: char = '\x0b';

/// Rows of a matrix literal: separated by `;` or a newline, columns by
/// blanks or commas.
fn parse_matrix(body: &str, first_line: usize) -> Result<Vec<Vec<f64>>, MatpowerError> {
    let mut rows = Vec::new();
    let mut line = first_line;
    for (k, chunk) in body.split('\n').enumerate() {
        if k > 0 {
            line += 1;
        }
        let continued = chunk.matches(CONTINUED).count();
        for row in chunk.split(';') {
            let values = row
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|t| !t.is_empty())
                .map(|t| {
                    t.parse::<f64>().map_err(|_| MatpowerError::Parse {
                        line,
                        message: format!("`{t}` is not a number"),
                    })
                })
                .collect::<Result<Vec<f64>, _>>()?;
            if !values.is_empty() {
                rows.push(values);
            }
        }
        line += continued;
    }
    Ok(rows)
}

fn skip_blanks(bytes: &[u8], mut j: usize) -> usize {
    while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
        j += 1;
    }
    j
}

/// Where the statement starting at `j` ends: its `;` or the end of its line.
fn statement_end(code: &str, j: usize) -> usize {
    code[j..].find([';', '\n']).map_or(code.len(), |n| j + n)
}

fn line_of(code: &str, pos: usize) -> usize {
    code[..pos].matches(['\n', CONTINUED]).count() + 1
}

/// What a [`CaseMeasurement`] measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasuredQuantity {
    /// |V| at a bus, per unit.
    VoltageMagnitude,
    /// A bus's injection, MW and MVAr, positive into the network: its online
    /// generators' output less its load, MATPOWER's `Sbus`. Not its shunt,
    /// which is part of the network, as in `makeYbus`.
    ActiveInjection,
    ReactiveInjection,
    /// A branch's flow at its from end, MW and MVAr, positive out of the
    /// from bus into the branch, as MATPOWER's `PF` and `QF`.
    ActiveFlowFrom,
    ReactiveFlowFrom,
    /// The same at its to end (`PT`, `QT`), positive out of the to bus.
    ActiveFlowTo,
    ReactiveFlowTo,
}

/// One measurement on a MATPOWER case, in the case's own terms: the bus by
/// its number (`BUS_I`), the branch by its 0-based row in `mpc.branch`, and
/// the value and its standard deviation in MATPOWER's units (p.u. for |V|,
/// MW and MVAr for powers).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaseMeasurement {
    pub quantity: MeasuredQuantity,
    /// Bus number for |V| and injections, branch row for flows.
    pub at: u64,
    pub value: f64,
    pub sigma: f64,
}

/// A state-estimation case on a MATPOWER network, read by [`read`]: the
/// network exactly as `runpf` sees it, with no converter in between, and
/// `measurements` placed on it.
///
/// The case's reference bus is the angle reference and its magnitude is
/// estimated like any other bus's; bus types mean nothing else to an
/// estimator. No bus is held at zero injection: a `.m` says what a bus
/// injects in one snapshot, not that it can inject nothing, so a bus without
/// load or generator injects whatever its measurements say. Node ids of the
/// result are bus numbers.
///
/// A measurement on an isolated bus, or on a branch out of service or
/// touching one, is refused: nothing in the solved network carries it.
pub fn se_case(
    net: &MatpowerNetwork,
    measurements: &[CaseMeasurement],
) -> Result<crate::se::case::SeCase, MatpowerError> {
    use crate::branch_flow::Terminal;
    use crate::measurement::{Measurement, MeasurementKind, Target};

    let n = net.buses.len();
    let mut ybus = crate::network::build_ybus(n, &net.lines, &net.transformers);
    crate::network::stamp_shunts(&mut ybus, &net.shunts);
    let network = crate::se::SeNetwork::from_bus_network(
        &net.buses,
        ybus.finish(),
        &net.lines,
        &net.transformers,
        &net.shunts,
        vec![false; n],
    );

    let node_idx: HashMap<u64, usize> = net
        .bus_numbers
        .iter()
        .enumerate()
        .map(|(i, &number)| (number, i))
        .collect();
    let branch_of_row: HashMap<u64, usize> = net
        .branch_rows
        .iter()
        .enumerate()
        .map(|(branch, &row)| (row as u64, branch))
        .collect();
    let refuse = |what: String| Err(MatpowerError::Measurement(what));

    let mut rows = Vec::with_capacity(measurements.len());
    for m in measurements {
        if !(m.sigma.is_finite() && m.sigma > 0.0 && m.value.is_finite()) {
            return refuse(format!(
                "{:?} at {} has value {} and sigma {}; a measurement needs a finite value \
                 and a positive sigma",
                m.quantity, m.at, m.value, m.sigma
            ));
        }
        let bus = || match node_idx.get(&m.at) {
            Some(&i) if network.energized[i] => Ok(i),
            Some(_) => Err(MatpowerError::Measurement(format!(
                "{:?} at bus {}: the bus is isolated",
                m.quantity, m.at
            ))),
            None => Err(MatpowerError::Measurement(format!(
                "{:?} at bus {}, which mpc.bus does not define",
                m.quantity, m.at
            ))),
        };
        let branch = |terminal| match branch_of_row.get(&m.at) {
            Some(&branch) => Ok(Target::BranchTerminal { branch, terminal }),
            None => Err(MatpowerError::Measurement(format!(
                "{:?} on mpc.branch row {}: no such branch in service",
                m.quantity, m.at
            ))),
        };
        use MeasuredQuantity as Q;
        let (kind, target, scale) = match m.quantity {
            Q::VoltageMagnitude => (MeasurementKind::VoltageMagnitude, Target::Bus(bus()?), 1.0),
            Q::ActiveInjection => (
                MeasurementKind::ActivePower,
                Target::Bus(bus()?),
                net.base_mva,
            ),
            Q::ReactiveInjection => (
                MeasurementKind::ReactivePower,
                Target::Bus(bus()?),
                net.base_mva,
            ),
            Q::ActiveFlowFrom => (
                MeasurementKind::ActivePower,
                branch(Terminal::From)?,
                net.base_mva,
            ),
            Q::ReactiveFlowFrom => (
                MeasurementKind::ReactivePower,
                branch(Terminal::From)?,
                net.base_mva,
            ),
            Q::ActiveFlowTo => (
                MeasurementKind::ActivePower,
                branch(Terminal::To)?,
                net.base_mva,
            ),
            Q::ReactiveFlowTo => (
                MeasurementKind::ReactivePower,
                branch(Terminal::To)?,
                net.base_mva,
            ),
        };
        rows.push(Measurement {
            kind,
            target,
            value: m.value / scale,
            sigma: m.sigma / scale,
        });
    }
    if rows.is_empty() {
        return refuse("no measurements: there is nothing to estimate from".into());
    }

    Ok(crate::se::case::SeCase {
        network,
        buses: net.buses.clone(),
        measurements: rows,
        node_idx,
        phases: 1,
    })
}

/// MATPOWER's AC-OPF on a case, ready for [`AcOpf::build`](crate::opf::ac::AcOpf::build).
#[cfg(feature = "opf")]
#[derive(Clone, Debug)]
pub struct MatpowerOpf {
    /// Isolated buses are left out, so its bus indices are not
    /// [`MatpowerNetwork`]'s: [`bus_numbers`](Self::bus_numbers) maps them.
    pub network: crate::opf::ac::AcOpfNetwork,
    /// OPF bus index → MATPOWER bus number (`BUS_I`).
    pub bus_numbers: Vec<u64>,
    /// The reference bus's `VA`, radians. The OPF holds the reference at
    /// zero; adding this gives the angles MATPOWER reports.
    pub reference_angle: f64,
    /// What the conversion decided on the case's behalf.
    pub notes: Vec<String>,
}

/// The AC-OPF `runopf` solves on `case`, in polar form.
///
/// The network is [`to_network`]'s, so out-of-service branches and
/// generators, isolated buses and the reference bus follow the power-flow
/// rules; isolated buses are then dropped, as MATPOWER's `ext2int` does. On
/// top of it:
///
/// - **Cost**: `mpc.gencost`, one row per generator row. Polynomial (model
///   2) up to quadratic, or piecewise linear (model 1). Rows beyond the
///   generators, MATPOWER's reactive-power costs, are refused, as is a
///   polynomial of higher degree: the AC-OPF's Hessian carries only a
///   quadratic term.
/// - **Generators** keep their row in `mpc.gen` as
///   [`AcGenerator::index`](crate::opf::ac::AcGenerator::index), bounded by
///   `PMIN`..`PMAX` and `QMIN`..`QMAX`.
/// - **Voltage magnitudes** within `VMIN`..`VMAX`.
/// - **Branch limits**: `RATE_A` as apparent power at both ends; 0 is
///   unlimited.
/// - **Angle differences**: `ANGMIN <= θ_f − θ_t <= ANGMAX`, a side ignored
///   when it is 0 or at ±360 degrees or beyond, as `makeAang` reads it.
/// - **The reference** angle is held at zero: one reference bus, which an
///   OPF of a case with several islands would need more of, so such a case
///   is refused.
///
/// The start is [`AcOpf`](crate::opf::ac::AcOpf)'s own: flat voltages, the
/// dispatch spread over the generators' ranges to meet the load.
#[cfg(feature = "opf")]
pub fn ac_opf_network(case: &MatpowerCase) -> Result<MatpowerOpf, MatpowerError> {
    use crate::opf::ac::{AcGenerator, AcOpfNetwork, AngleLimit};

    check_columns("bus", &case.bus, VMIN + 1)?;
    check_columns("gen", &case.gens, PMIN + 1)?;
    check_columns("branch", &case.branch, ANGMAX + 1)?;
    let gencost = case
        .gencost
        .as_ref()
        .ok_or(MatpowerError::Missing("gencost"))?;
    if gencost.len() != case.gens.len() {
        return Err(MatpowerError::Unsupported(format!(
            "mpc.gencost has {} rows for {} generators; only active-power costs, one row \
             per generator, are read",
            gencost.len(),
            case.gens.len()
        )));
    }
    check_columns("gencost", gencost, COST)?;

    let net = to_network(case)?;
    let base = net.base_mva;
    let mut notes = net.notes.clone();

    // Energized buses only, renumbered in file order.
    let mut opf_index = vec![None; net.buses.len()];
    let mut bus_numbers = Vec::new();
    for (i, b) in case.bus.iter().enumerate() {
        if b[BUS_TYPE] != NONE {
            opf_index[i] = Some(bus_numbers.len());
            bus_numbers.push(net.bus_numbers[i]);
        }
    }
    let at = |i: usize| opf_index[i].expect("to_network drops what touches an isolated bus");
    let n = bus_numbers.len();

    let references: Vec<usize> = net
        .buses
        .iter()
        .enumerate()
        .filter(|(_, b)| b.bus_type == BusType::Slack)
        .map(|(i, _)| i)
        .collect();
    let reference = match references[..] {
        [one] => one,
        [] => {
            return Err(MatpowerError::Unsupported(
                "the case has no reference bus with an online generator".into(),
            ));
        }
        _ => {
            return Err(MatpowerError::Unsupported(format!(
                "the case has {} reference buses; an OPF here holds one angle reference",
                references.len()
            )));
        }
    };

    let mut p_load = vec![0.0; n];
    let mut q_load = vec![0.0; n];
    let mut v_min = vec![0.0; n];
    let mut v_max = vec![0.0; n];
    for (i, b) in case.bus.iter().enumerate() {
        if let Some(k) = opf_index[i] {
            p_load[k] = b[PD] / base;
            q_load[k] = b[QD] / base;
            v_min[k] = b[VMIN];
            v_max[k] = b[VMAX];
        }
    }

    let index: HashMap<u64, usize> = net
        .bus_numbers
        .iter()
        .enumerate()
        .map(|(i, &number)| (number, i))
        .collect();
    let mut generators = Vec::new();
    for (row, g) in case.gens.iter().enumerate() {
        let bus = bus_number(g[GEN_BUS])
            .and_then(|number| index.get(&number).copied())
            .ok_or(MatpowerError::UnknownBus {
                matrix: "gen",
                row: row + 1,
                bus: g[GEN_BUS],
            })?;
        let Some(bus) = opf_index[bus].filter(|_| g[GEN_STATUS] > 0.0) else {
            continue;
        };
        generators.push(AcGenerator {
            index: row,
            bus,
            p_min: g[PMIN] / base,
            p_max: g[PMAX] / base,
            q_min: g[QMIN] / base,
            q_max: g[QMAX] / base,
            cost: Some(cost_curve(&gencost[row], row)?.to_per_unit(base)),
        });
    }

    let lines: Vec<Line> = net
        .lines
        .iter()
        .map(|l| Line {
            from: at(l.from),
            to: at(l.to),
            ..l.clone()
        })
        .collect();
    let transformers: Vec<Transformer> = net
        .transformers
        .iter()
        .map(|t| Transformer {
            from: at(t.from),
            to: at(t.to),
            ..t.clone()
        })
        .collect();

    let mut limits = HashMap::new();
    let mut angle_limits = Vec::new();
    for (flat, &row) in net.branch_rows.iter().enumerate() {
        let br = &case.branch[row];
        if br[RATE_A] > 0.0 {
            limits.insert(flat, Some(br[RATE_A] / base));
        }
        let min = if br[ANGMIN] != 0.0 && br[ANGMIN] > -360.0 {
            br[ANGMIN].to_radians()
        } else {
            f64::NEG_INFINITY
        };
        let max = if br[ANGMAX] != 0.0 && br[ANGMAX] < 360.0 {
            br[ANGMAX].to_radians()
        } else {
            f64::INFINITY
        };
        if min.is_finite() || max.is_finite() {
            let (from, to) = if flat < lines.len() {
                (lines[flat].from, lines[flat].to)
            } else {
                let t = &transformers[flat - lines.len()];
                (t.from, t.to)
            };
            angle_limits.push(AngleLimit { from, to, min, max });
        }
    }
    if generators.is_empty() {
        notes.push("no online generator".into());
    }

    Ok(MatpowerOpf {
        network: AcOpfNetwork {
            n_buses: n,
            reference: at(reference),
            lines,
            transformers,
            generators,
            p_load,
            q_load,
            v_min,
            v_max,
            limits,
            shunts: net.shunts.iter().map(|s| (at(s.at), s.y)).collect(),
            base_mva: base,
            v_start: None,
            angle_limits,
        },
        bus_numbers,
        reference_angle: case.bus[reference][VA].to_radians(),
        notes,
    })
}

/// A `gencost` row as a cost in MW, ascending coefficients.
#[cfg(feature = "opf")]
fn cost_curve(row: &[f64], gen_row: usize) -> Result<crate::opf::model::CostCurve, MatpowerError> {
    use crate::opf::model::CostCurve;

    let count = row[NCOST];
    let values = |n: usize| {
        row.get(COST..COST + n).ok_or(MatpowerError::ShortRow {
            matrix: "gencost",
            row: gen_row + 1,
            columns: row.len(),
            required: COST + n,
        })
    };
    if !(count >= 0.0 && count.fract() == 0.0) {
        return Err(MatpowerError::Unsupported(format!(
            "mpc.gencost row {}: NCOST = {count}",
            gen_row + 1
        )));
    }
    let count = count as usize;
    match row[MODEL] {
        m if m == POLYNOMIAL => {
            // Highest degree first in the file.
            let mut coefficients: Vec<f64> = values(count)?.iter().rev().copied().collect();
            while coefficients.len() > 3 && coefficients.last() == Some(&0.0) {
                coefficients.pop();
            }
            if coefficients.len() > 3 {
                return Err(MatpowerError::Unsupported(format!(
                    "mpc.gencost row {}: a polynomial cost of degree {}; only up to \
                     quadratic is read",
                    gen_row + 1,
                    coefficients.len() - 1
                )));
            }
            Ok(CostCurve::Polynomial { coefficients })
        }
        m if m == PW_LINEAR => Ok(CostCurve::PiecewiseLinear {
            points: values(2 * count)?
                .chunks_exact(2)
                .map(|p| (p[0], p[1]))
                .collect(),
        }),
        m => Err(MatpowerError::Unsupported(format!(
            "mpc.gencost row {}: cost model {m}; only 1 (piecewise linear) and 2 \
             (polynomial) exist",
            gen_row + 1
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CASE: &str = "function mpc = tiny
%% a three-bus case
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [ % bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
\t1\t3\t0\t0\t0\t0\t1\t1.02\t5\t230\t1\t1.1\t0.9;
\t2\t2\t20\t10\t0\t5\t1\t1\t0\t230\t1\t1.1\t0.9;
\t3\t1\t50, 20, 1, 0, 1, 1, 0, 230, 1, 1.1, 0.9
\t4\t4\t10\t0\t0\t0\t1\t1\t0\t230\t1\t1.1\t0.9;
];
mpc.gen = [
\t1\t0\t0\t300\t-300\t1.04\t100\t1;
\t2\t40\t5\t100\t-50\t1.01\t100\t1;
\t3\t10\t2\t9\t-9\t1.05\t100\t1;
\t4\t10\t2\t9\t-9\t1.05\t100\t1;
\t2\t99\t0\t9\t-9\t1.00\t100\t0;
];
mpc.branch = [
\t1\t2\t0.01\t0.1\t0.02\t0\t0\t0\t0\t0\t1;
\t2\t3\t0.01\t0.1\t0\t0\t0\t0\t0.98\t-3\t1;
\t1\t3\t0.01\t0.1\t0\t0\t0\t0\t0\t0\t0;
\t3\t4\t0.01\t0.1\t0\t0\t0\t0\t0\t0\t1;
];
";

    #[test]
    fn reads_the_matrices_as_written() {
        let case = parse(CASE).unwrap();
        assert_eq!(case.base_mva, 100.0);
        assert_eq!(case.bus.len(), 4);
        assert_eq!(case.bus[2][..4], [3.0, 1.0, 50.0, 20.0]);
        assert_eq!(case.gens.len(), 5);
        assert_eq!(case.branch.len(), 4);
    }

    #[test]
    fn converts_by_runpfs_rules() {
        let net = to_network(&parse(CASE).unwrap()).unwrap();
        assert_eq!(net.bus_numbers, [1, 2, 3, 4]);
        let [b1, b2, b3, b4] = &net.buses[..] else {
            panic!()
        };

        // Setpoints from the generators' Vg, never the bus's Vm; the
        // reference keeps its Va.
        assert_eq!(b1.bus_type, BusType::Slack);
        assert_eq!(b1.voltage_mag, 1.04);
        assert!((b1.voltage_ang - 5f64.to_radians()).abs() < 1e-15);
        assert_eq!(b2.bus_type, BusType::PV);
        assert_eq!(b2.voltage_mag, 1.01);
        assert_eq!(b2.voltage_ang, 0.0);
        // The offline generator at bus 2 adds nothing.
        assert!((b2.p_spec - (40.0 - 20.0) / 100.0).abs() < 1e-15);

        // A generator on a PQ bus is a fixed injection, Q included.
        assert_eq!(b3.bus_type, BusType::PQ);
        assert_eq!(b3.voltage_mag, 1.0);
        assert!((b3.p_spec - (10.0 - 50.0) / 100.0).abs() < 1e-15);
        assert!((b3.q_spec - (2.0 - 20.0) / 100.0).abs() < 1e-15);
        assert_eq!(b3.q_max, f64::INFINITY);

        // The isolated bus keeps its place and nothing else.
        assert_eq!((b4.p_spec, b4.q_spec), (0.0, 0.0));

        // Gs/Bs per unit, isolated buses excepted.
        assert_eq!(net.shunts.len(), 2);
        assert_eq!(net.shunts[0].y, Complex::new(0.0, 0.05));
        assert_eq!(net.shunts[1].y, Complex::new(0.01, 0.0));

        // Out of service and touching an isolated bus: dropped. The tapped,
        // shifted branch is a transformer with its ratio on the from side.
        assert_eq!(net.lines.len(), 1);
        assert_eq!(net.lines[0].b_shunt, 0.02);
        assert_eq!(net.transformers.len(), 1);
        let t = &net.transformers[0];
        assert!((t.tap.norm() - 0.98).abs() < 1e-15);
        assert!((t.tap.arg() - (-3f64).to_radians()).abs() < 1e-15);
        assert_eq!(net.branch_rows, [0, 1]);
    }

    #[test]
    fn a_reference_bus_without_a_generator_hands_over_to_the_first_pv_bus() {
        let text = CASE.replace(
            "\t1\t0\t0\t300\t-300\t1.04\t100\t1;",
            "\t1\t0\t0\t300\t-300\t1.04\t100\t0;",
        );
        let net = to_network(&parse(&text).unwrap()).unwrap();
        assert_eq!(net.buses[0].bus_type, BusType::PQ);
        assert_eq!(net.buses[1].bus_type, BusType::Slack);
        assert!(net.notes.iter().any(|n| n.contains("first PV bus")));
    }

    #[test]
    fn code_that_edits_the_matrices_is_refused() {
        let text = format!("{CASE}\nmpc.bus(:, [PD, QD]) = mpc.bus(:, [PD, QD]) / 1e3;\n");
        let err = parse(&text).unwrap_err().to_string();
        assert!(err.contains("mpc.bus is modified by MATLAB code"), "{err}");
    }

    #[test]
    fn a_continued_row_and_a_commented_bracket_are_read() {
        let text = CASE.replace(
            "\t2\t3\t0.01\t0.1\t0\t0\t0\t0\t0.98\t-3\t1;",
            "\t2\t3\t0.01\t0.1\t0\t0 ... ]\n\t0\t0\t0.98\t-3\t1; % ]",
        );
        let case = parse(&text).unwrap();
        assert_eq!(case.branch[1][TAP], 0.98);
    }

    #[test]
    fn version_one_and_mat_files_are_refused() {
        let text = CASE.replace("mpc.version = '2';", "mpc.version = '1';");
        assert!(parse(&text).unwrap_err().to_string().contains("version 1"));
        let err = read("case.mat").unwrap_err().to_string();
        assert!(err.contains(".mat"), "{err}");
    }
}
