//! PSS/E `.dyr` dynamic data records.
//!
//! The `.dyr` is the format most published dynamic cases ship in, and it is
//! what ANDES reads — so the same file can drive both gridoxide and the
//! reference, which removes an entire class of "the two tools were given
//! different data" disagreement before phase 5 starts.
//!
//! # The format
//!
//! Free-format records, whitespace- or comma-separated, each terminated by a
//! `/`:
//!
//! ```text
//!      1 'GENROU' 1   8.0  0.03  0.4  0.05  5.0  0.0
//!        1.8 1.7 0.30 0.55 0.22 0.15 0.0 0.0  /
//!      1 'SEXS'   1   0.1  1.0  200.0  0.05  -5.0  5.0 /
//!      1 'TGOV1'  1   0.05 0.5  1.0  0.0  1.0  5.0  0.0 /
//! ```
//!
//! A record may span any number of lines; only the `/` ends it. The first field
//! is a **bus number**, the second a quoted model name, the third a quoted
//! machine identifier, and the rest are positional parameters.
//!
//! # What a `.dyr` cannot say
//!
//! Two things every machine model here needs are **not in the file**: the
//! machine's MVA rating and its armature resistance. In PSS/E both live in the
//! `.raw` alongside the network, as `MBASE` and `ZSORCE`. gridoxide has no
//! `.raw` importer, so [`to_units`] takes them from the caller rather than
//! inventing them.
//!
//! The same is true of the bus numbering. A `.dyr` names buses by the `.raw`'s
//! numbers, which no format gridoxide *does* read shares, so the correspondence
//! to bus indices is also the caller's to supply. This is a real limitation of
//! reading half a pair of files, and it is stated rather than papered over.
//!
//! # Limits are carried through
//!
//! `SEXS`'s `EMIN`/`EMAX` and `TGOV1`'s `VMIN`/`VMAX` reach the models, which
//! enforce them as non-windup limits — see [`avr`](super::models::avr). Note
//! that `TGOV1` states them **VMAX first**; reading them in field order gives a
//! governor whose valve is limited upside-down and which therefore never moves.
//!
//! # Saturation is read and discarded
//!
//! `GENROU` carries `S(1.0)` and `S(1.2)`, the saturation characteristic. No
//! model here implements saturation — see `plans/RMS_PLAN.md` §13 for why that
//! choice waits for a reference to compare against — so those two parameters
//! are parsed, checked, and dropped. A **nonzero** value produces a
//! [`DyrWarning::SaturationIgnored`], because ignoring it changes answers and a
//! silent drop would be the kind of omission that only shows up as a small
//! unexplained disagreement later.

use std::collections::HashMap;
use std::path::Path;

use super::json::{AvrSpec, GovSpec, MachineSpec, UnitSpec};
use super::models::avr::SexsParams;
use super::models::gov::Tgov1Params;
use super::models::machine::{GenClsParams, GenRoundParams};
use super::models::Limits;

/// One record, tokenized but not yet interpreted.
#[derive(Clone, Debug, PartialEq)]
pub struct DyrRecord {
    pub bus: i64,
    /// Upper-cased, quotes stripped.
    pub model: String,
    /// The machine identifier, quotes stripped and trimmed. Usually `"1"`.
    pub id: String,
    pub params: Vec<f64>,
    /// 1-based line the record started on, for error messages.
    pub line: usize,
}

/// A parsed `.dyr`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DyrDocument {
    pub records: Vec<DyrRecord>,
}

/// Why a `.dyr` could not be read or interpreted.
#[derive(Clone, Debug, PartialEq)]
pub enum DyrError {
    Io(String),
    /// A record that does not begin `bus 'MODEL' id`.
    Malformed { line: usize, detail: String },
    /// A model this reader knows, given the wrong number of parameters.
    WrongArity { line: usize, model: String, expected: usize, found: usize },
    /// A parameter that must be positive was not, or an ordering the model
    /// requires does not hold. Caught here rather than at construction so the
    /// message can name the line.
    BadParameter { line: usize, model: String, detail: String },
    /// A machine record for which the caller supplied no `MBASE`/`ZSORCE`.
    MissingMachineData { bus: i64, id: String },
    /// A bus number with no corresponding index.
    UnknownBus { bus: i64 },
    /// A control record whose machine has no `GENCLS`/`GENROU` record.
    OrphanedControl { bus: i64, id: String, model: String },
}

impl std::fmt::Display for DyrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DyrError::Io(message) => write!(f, "{message}"),
            DyrError::Malformed { line, detail } => write!(f, "line {line}: {detail}"),
            DyrError::WrongArity { line, model, expected, found } => write!(
                f,
                "line {line}: {model} takes {expected} parameters, found {found}"
            ),
            DyrError::BadParameter { line, model, detail } => {
                write!(f, "line {line}: {model}: {detail}")
            }
            DyrError::MissingMachineData { bus, id } => write!(
                f,
                "machine {id} at bus {bus} has no MBASE/ZSORCE supplied; a .dyr does not \
                 carry them (they are the .raw's MBASE and ZSORCE)"
            ),
            DyrError::UnknownBus { bus } => {
                write!(f, "bus number {bus} has no corresponding bus index")
            }
            DyrError::OrphanedControl { bus, id, model } => write!(
                f,
                "{model} at bus {bus} machine {id} has no GENCLS or GENROU record to attach to"
            ),
        }
    }
}

impl std::error::Error for DyrError {}

/// Something read but not acted on.
#[derive(Clone, Debug, PartialEq)]
pub enum DyrWarning {
    /// A model this reader does not implement. Skipped, and named.
    UnsupportedModel { bus: i64, id: String, model: String, line: usize },
    /// A nonzero saturation characteristic, which no model here uses.
    SaturationIgnored { bus: i64, id: String, s10: f64, s12: f64 },
    /// A limit this reader carries through but that the model does not use.
    /// Retained for anything that gains a limit the library still cannot
    /// represent — a valve *rate* limit, say.
    LimitsIgnored { bus: i64, id: String, model: String },
}

impl std::fmt::Display for DyrWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DyrWarning::UnsupportedModel { bus, id, model, line } => {
                write!(f, "line {line}: {model} at bus {bus} machine {id} is not implemented; skipped")
            }
            DyrWarning::SaturationIgnored { bus, id, s10, s12 } => write!(
                f,
                "bus {bus} machine {id}: saturation S(1.0) = {s10}, S(1.2) = {s12} is not \
                 modelled and was dropped"
            ),
            DyrWarning::LimitsIgnored { bus, id, model } => write!(
                f,
                "bus {bus} machine {id}: {model}'s output limits are not modelled and were dropped"
            ),
        }
    }
}

/// What a `.dyr` cannot state, supplied by the caller.
///
/// In PSS/E both live in the `.raw`: `MBASE` is the machine's MVA rating and
/// `ZSORCE` its source impedance, whose real part is the armature resistance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MachineData {
    pub mbase: f64,
    pub ra: f64,
}

/// Reads a `.dyr` from a file.
pub fn read(path: impl AsRef<Path>) -> Result<DyrDocument, DyrError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|e| DyrError::Io(format!("{}: {e}", path.display())))?;
    parse(&text)
}

/// Splits the text into records and tokenizes each.
///
/// Deliberately tolerant about layout — a record may span lines and separators
/// may be spaces or commas — and strict about shape, since a record that does
/// not start `bus 'MODEL' id` is not something to guess at.
pub fn parse(text: &str) -> Result<DyrDocument, DyrError> {
    let mut records = Vec::new();
    let mut tokens: Vec<String> = Vec::new();
    let mut start_line = 1usize;
    let mut fresh = true;

    for (index, raw) in text.lines().enumerate() {
        let line_no = index + 1;
        // A line whose first non-blank character is `/` is a comment — but only
        // when no record is open. Mid-record, that same `/` is the terminator,
        // and files do put it on a line of its own. The two readings are
        // distinguished by whether anything is pending, which is the only
        // information available and is what other readers of this format use.
        if fresh && raw.trim_start().starts_with('/') {
            continue;
        }
        let mut rest = raw;
        loop {
            let (chunk, terminated, remainder) = split_at_terminator(rest);
            if fresh && !chunk.trim().is_empty() {
                start_line = line_no;
                fresh = false;
            }
            tokens.extend(tokenize(chunk));
            if !terminated {
                break;
            }
            if !tokens.is_empty() {
                records.push(build_record(&tokens, start_line)?);
                tokens.clear();
            }
            fresh = true;
            rest = remainder;
            if rest.is_empty() {
                break;
            }
        }
    }
    // A trailing record with no `/` is accepted: some files end without one,
    // and refusing would reject a file every other tool reads.
    if !tokens.is_empty() {
        records.push(build_record(&tokens, start_line)?);
    }
    Ok(DyrDocument { records })
}

/// Splits at the first `/` outside a quoted string.
fn split_at_terminator(line: &str) -> (&str, bool, &str) {
    let mut in_quote = false;
    for (i, c) in line.char_indices() {
        match c {
            '\'' => in_quote = !in_quote,
            '/' if !in_quote => return (&line[..i], true, &line[i + 1..]),
            _ => {}
        }
    }
    (line, false, "")
}

fn tokenize(chunk: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    for c in chunk.chars() {
        match c {
            '\'' => {
                in_quote = !in_quote;
                if !in_quote {
                    out.push(std::mem::take(&mut current));
                }
            }
            _ if in_quote => current.push(c),
            c if c.is_whitespace() || c == ',' => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn build_record(tokens: &[String], line: usize) -> Result<DyrRecord, DyrError> {
    if tokens.len() < 3 {
        return Err(DyrError::Malformed {
            line,
            detail: format!("expected `bus 'MODEL' id …`, found {} field(s)", tokens.len()),
        });
    }
    let bus = tokens[0].parse::<i64>().map_err(|_| DyrError::Malformed {
        line,
        detail: format!("`{}` is not a bus number", tokens[0]),
    })?;
    let mut params = Vec::with_capacity(tokens.len() - 3);
    for token in &tokens[3..] {
        params.push(token.parse::<f64>().map_err(|_| DyrError::Malformed {
            line,
            detail: format!("`{token}` is not a number"),
        })?);
    }
    Ok(DyrRecord {
        bus,
        model: tokens[1].trim().to_ascii_uppercase(),
        id: tokens[2].trim().to_string(),
        params,
        line,
    })
}

fn arity(record: &DyrRecord, expected: usize) -> Result<(), DyrError> {
    if record.params.len() != expected {
        return Err(DyrError::WrongArity {
            line: record.line,
            model: record.model.clone(),
            expected,
            found: record.params.len(),
        });
    }
    Ok(())
}

impl DyrRecord {
    /// Whether this record is a machine rather than a control.
    pub fn is_machine(&self) -> bool {
        matches!(self.model.as_str(), "GENCLS" | "GENROU")
    }

    /// `GENCLS`: `H D`. The reactance and resistance come from `data`, since
    /// the file does not carry them.
    fn as_gencls(&self, data: MachineData, xdp: f64) -> Result<MachineSpec, DyrError> {
        arity(self, 2)?;
        Ok(MachineSpec::GenCls(GenClsParams {
            h: self.params[0],
            d: self.params[1],
            ra: data.ra,
            xdp,
            mbase: data.mbase,
        }))
    }

    /// `GENROU`: `T'do T''do T'qo T''qo H D Xd Xq X'd X'q X''d Xl S(1.0) S(1.2)`.
    ///
    /// `X''q` is not in the record: `GENROU` is a round-rotor model and takes
    /// the two subtransient reactances equal, which is the assumption that
    /// makes it "round".
    fn as_genrou(
        &self,
        data: MachineData,
        warnings: &mut Vec<DyrWarning>,
    ) -> Result<MachineSpec, DyrError> {
        arity(self, 14)?;
        let p = &self.params;
        let (s10, s12) = (p[12], p[13]);
        if s10 != 0.0 || s12 != 0.0 {
            warnings.push(DyrWarning::SaturationIgnored {
                bus: self.bus,
                id: self.id.clone(),
                s10,
                s12,
            });
        }
        let xdpp = p[10];
        let params = GenRoundParams {
            td0p: p[0],
            td0pp: p[1],
            tq0p: p[2],
            tq0pp: p[3],
            h: p[4],
            d: p[5],
            xd: p[6],
            xq: p[7],
            xdp: p[8],
            xqp: p[9],
            xdpp,
            xqpp: xdpp,
            xl: p[11],
            ra: data.ra,
            mbase: data.mbase,
        };
        // The model's own constructor enforces `x_l < x'' < x' < x`; checking
        // here as well is what lets the message name the line.
        for (name, lo, hi) in [
            ("Xl < X''d", params.xl, params.xdpp),
            ("X''d < X'd", params.xdpp, params.xdp),
            ("X'd < Xd", params.xdp, params.xd),
            ("X''q < X'q", params.xqpp, params.xqp),
            ("X'q < Xq", params.xqp, params.xq),
        ] {
            if hi <= lo {
                return Err(DyrError::BadParameter {
                    line: self.line,
                    model: self.model.clone(),
                    detail: format!("{name} does not hold ({lo} vs {hi})"),
                });
            }
        }
        Ok(MachineSpec::GenRound(params))
    }

    /// `SEXS`: `TA/TB TB K TE EMIN EMAX`. The first field is a **ratio**, not a
    /// time constant — a detail that silently produces a very fast exciter if
    /// missed.
    fn as_sexs(&self, _warnings: &mut Vec<DyrWarning>) -> Result<AvrSpec, DyrError> {
        arity(self, 6)?;
        let p = &self.params;
        Ok(AvrSpec::Sexs(SexsParams {
            k: p[2],
            ta: p[0] * p[1],
            tb: p[1],
            te: p[3],
            limits: Limits { min: Some(p[4]), max: Some(p[5]) },
        }))
    }

    /// `TGOV1`: `R T1 VMAX VMIN T2 T3 Dt`.
    fn as_tgov1(&self, _warnings: &mut Vec<DyrWarning>) -> Result<GovSpec, DyrError> {
        arity(self, 7)?;
        let p = &self.params;
        Ok(GovSpec::Tgov1(Tgov1Params {
            r: p[0],
            t1: p[1],
            t2: p[4],
            t3: p[5],
            dt: p[6],
            limits: Limits { min: Some(p[3]), max: Some(p[2]) },
            // PSS/E's TGOV1 states no rate limit.
            rate: Limits::NONE,
        }))
    }
}

/// Assembles [`UnitSpec`]s from a `.dyr`, given the two things the file cannot
/// state: which bus index each bus number is, and each machine's `MBASE` and
/// `ZSORCE`.
///
/// `xdp` for a `GENCLS` is also the caller's, for the same reason — the
/// classical model's reactance is `ZSORCE`'s imaginary part, which lives in the
/// `.raw`.
///
/// Records naming a model this reader does not implement are skipped and
/// reported, not refused: a real `.dyr` carries relays, wind models and
/// station controllers alongside the machines, and rejecting the file because
/// of one of them would be useless.
pub fn to_units(
    doc: &DyrDocument,
    bus_index: &HashMap<i64, usize>,
    machine_data: &HashMap<(i64, String), (MachineData, f64)>,
) -> Result<(Vec<UnitSpec>, Vec<DyrWarning>), DyrError> {
    let mut warnings = Vec::new();
    let mut units: Vec<UnitSpec> = Vec::new();
    let mut index_of: HashMap<(i64, String), usize> = HashMap::new();

    // Machines first, so a control always has something to attach to whatever
    // order the file lists them in.
    for record in doc.records.iter().filter(|r| r.is_machine()) {
        let key = (record.bus, record.id.clone());
        let &bus = bus_index.get(&record.bus).ok_or(DyrError::UnknownBus { bus: record.bus })?;
        let &(data, xdp) = machine_data.get(&key).ok_or_else(|| DyrError::MissingMachineData {
            bus: record.bus,
            id: record.id.clone(),
        })?;
        let machine = match record.model.as_str() {
            "GENCLS" => record.as_gencls(data, xdp)?,
            "GENROU" => record.as_genrou(data, &mut warnings)?,
            _ => unreachable!("is_machine covers exactly these"),
        };
        index_of.insert(key, units.len());
        units.push(UnitSpec {
            id: format!("{}_{}", record.bus, record.id),
            bus,
            p: None,
            q: None,
            machine,
            avr: None,
            gov: None,
            pss: None,
        });
    }

    for record in doc.records.iter().filter(|r| !r.is_machine()) {
        let key = (record.bus, record.id.clone());
        // Resolved before the model is interpreted, so an orphan is reported as
        // an orphan rather than as whatever its parameters happen to be wrong
        // about.
        let owner = || {
            index_of.get(&key).copied().ok_or_else(|| DyrError::OrphanedControl {
                bus: record.bus,
                id: record.id.clone(),
                model: record.model.clone(),
            })
        };
        match record.model.as_str() {
            "SEXS" => {
                let i = owner()?;
                units[i].avr = Some(record.as_sexs(&mut warnings)?);
            }
            "TGOV1" => {
                let i = owner()?;
                units[i].gov = Some(record.as_tgov1(&mut warnings)?);
            }
            other => warnings.push(DyrWarning::UnsupportedModel {
                bus: record.bus,
                id: record.id.clone(),
                model: other.to_string(),
                line: record.line,
            }),
        }
    }

    Ok((units, warnings))
}
